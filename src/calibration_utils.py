"""Shared helpers: config loading, paths, atomic YAML/CSV writes, transforms.

Importing this module has no side effects beyond reading nothing. It never
opens hardware.

ROTATION CONVENTIONS USED THROUGHOUT THIS PACKAGE
-------------------------------------------------
A UR TCP pose is [x, y, z, rx, ry, rz] where (rx, ry, rz) is a ROTATION
VECTOR (axis-angle): the axis is the unit vector rx,ry,rz / ||rx,ry,rz|| and
the angle is ||rx,ry,rz|| radians. It is NOT roll/pitch/yaw and must never be
fed to an Euler-angle routine. Use rotvec_to_matrix() / matrix_to_rotvec().

OpenCV's rvec is the same axis-angle convention, so UR rotation vectors and
cv2.Rodrigues interoperate directly.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import yaml

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

PACKAGE_ROOT = Path(__file__).resolve().parent.parent

# Directories can be redirected by environment variable. This lets you keep a
# second dataset (or an archived one) beside the live one, and lets the test
# suite exercise the real scripts without touching real calibration data.
CONFIG_DIR = Path(os.environ.get("CAMERA_CALIBRATION_CONFIG_DIR",
                                 PACKAGE_ROOT / "config"))
DATA_DIR = Path(os.environ.get("CAMERA_CALIBRATION_DATA_DIR",
                               PACKAGE_ROOT / "data"))
LOG_DIR = Path(os.environ.get("CAMERA_CALIBRATION_LOG_DIR",
                              PACKAGE_ROOT / "logs"))

CAMERAS_CONFIG = CONFIG_DIR / "cameras.yaml"
CALIBRATION_CONFIG = CONFIG_DIR / "calibration.yaml"
SAFETY_CONFIG = CONFIG_DIR / "safety.yaml"

VALID_CAMERA_NAMES = ("camera_1", "camera_2", "camera_3")

# UR joint names, in controller order.
JOINT_NAMES = ("base", "shoulder", "elbow", "wrist_1", "wrist_2", "wrist_3")
JOINT_LABELS = ("Joint 1 = Base", "Joint 2 = Shoulder", "Joint 3 = Elbow",
                "Joint 4 = Wrist 1", "Joint 5 = Wrist 2", "Joint 6 = Wrist 3")

#: Calibration movement priority, most-preferred first (section I).
#: Distal joints give orientation diversity with the least whole-arm travel.
JOINT_PRIORITY = ("wrist_3", "wrist_2", "wrist_1", "elbow", "shoulder", "base")


class ConfigError(RuntimeError):
    """Raised when configuration is missing, malformed, or unverified."""


class CalibrationError(RuntimeError):
    """Raised when a calibration step cannot produce a trustworthy result."""


# --------------------------------------------------------------------------
# Config loading
# --------------------------------------------------------------------------

def load_yaml(path: Path) -> dict:
    """Load a YAML file into a dict, with a useful error if it is unusable."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Configuration file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    if data is None:
        raise ConfigError(f"{path}: file is empty")
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping")
    return data


def load_cameras_config(path: Path | None = None) -> dict:
    """Load cameras.yaml and validate its shape (not its serial validity)."""
    config = load_yaml(path or CAMERAS_CONFIG)
    cameras = config.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise ConfigError("cameras.yaml: missing or empty 'cameras' mapping")
    for name, entry in cameras.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"cameras.yaml: '{name}' must be a mapping")
        for key in ("backend", "serial"):
            if key not in entry:
                raise ConfigError(f"cameras.yaml: '{name}' is missing '{key}'")
        if entry["backend"] not in ("realsense", "v4l2"):
            raise ConfigError(
                f"cameras.yaml: '{name}' has unknown backend "
                f"{entry['backend']!r} (expected 'realsense' or 'v4l2')")
    config.setdefault("capture", {})
    return config


def load_calibration_config(path: Path | None = None) -> dict:
    """Load calibration.yaml and validate the board geometry numerically."""
    config = load_yaml(path or CALIBRATION_CONFIG)
    grid = config.get("apriltag_grid")
    if not isinstance(grid, dict):
        raise ConfigError("calibration.yaml: missing 'apriltag_grid'")
    for key in ("tag_family", "rows", "columns", "tag_size_m", "tag_spacing_m"):
        if key not in grid:
            raise ConfigError(f"calibration.yaml: apriltag_grid missing '{key}'")
    for key in ("rows", "columns"):
        value = grid[key]
        if not isinstance(value, int) or value < 1:
            raise ConfigError(
                f"calibration.yaml: apriltag_grid.{key} must be a positive "
                f"integer, got {value!r}")
    if not isinstance(grid["tag_size_m"], (int, float)) or grid["tag_size_m"] <= 0:
        raise ConfigError("calibration.yaml: tag_size_m must be > 0 (metres)")
    if not isinstance(grid["tag_spacing_m"], (int, float)) or grid["tag_spacing_m"] < 0:
        raise ConfigError("calibration.yaml: tag_spacing_m must be >= 0 (metres)")
    if grid["tag_size_m"] > 1.0:
        raise ConfigError(
            f"calibration.yaml: tag_size_m={grid['tag_size_m']} looks like "
            f"millimetres. This value must be in METRES.")
    return config


def load_safety_config(path: Path | None = None) -> dict:
    """Load safety.yaml and validate the fields motion code depends on."""
    config = load_yaml(path or SAFETY_CONFIG)
    for section in ("connection", "motion", "workspace", "joint_limits", "jog"):
        if section not in config:
            raise ConfigError(f"safety.yaml: missing '{section}' section")
    limits = config["joint_limits"]
    for joint in JOINT_NAMES:
        entry = limits.get(joint)
        if not isinstance(entry, dict) or "min" not in entry or "max" not in entry:
            raise ConfigError(f"safety.yaml: joint_limits.{joint} needs min and max")
        if entry["min"] >= entry["max"]:
            raise ConfigError(f"safety.yaml: joint_limits.{joint}: min >= max")
    return config


def require_board_verified(calibration_config: Mapping[str, Any]) -> None:
    """Refuse to emit metric results while the board geometry is unconfirmed.

    A wrong tag_size_m scales every translation in the final hand-eye result
    without inflating reprojection error, so it cannot be caught downstream.
    The only defence is a human confirming the measurement.
    """
    grid = calibration_config["apriltag_grid"]
    if not grid.get("verified_by_user", False):
        raise ConfigError(
            "Board geometry is not confirmed.\n"
            "  Measure your printed board, set the real values in\n"
            f"  {CALIBRATION_CONFIG}\n"
            "  under 'apriltag_grid', then set 'verified_by_user: true'.\n"
            "  Current values: "
            f"{grid['rows']}x{grid['columns']} tags, "
            f"tag_size_m={grid['tag_size_m']}, "
            f"tag_spacing_m={grid['tag_spacing_m']}")


def resolve_camera(name: str, cameras_config: Mapping[str, Any]) -> dict:
    """Return the config entry for a logical camera name, with capture defaults."""
    cameras = cameras_config["cameras"]
    if name not in cameras:
        raise ConfigError(
            f"Unknown camera {name!r}. Defined in cameras.yaml: "
            f"{', '.join(sorted(cameras))}")
    entry = dict(cameras[name])
    entry.setdefault("name", name)
    entry["capture"] = dict(cameras_config.get("capture") or {})
    return entry


HANDEYE_MODES = ("eye_to_hand", "eye_in_hand")


def resolve_handeye_mode(camera_config: Mapping[str, Any],
                         calibration_config: Mapping[str, Any]) -> str:
    """Hand-eye mode for one camera: per-camera override, else the default.

    This rig mixes both modes (wrist D405 is eye-in-hand, the two fixed D435s
    are eye-to-hand), so the mode is a property of the CAMERA, not of the
    project. Getting it wrong yields a numerically plausible, geometrically
    meaningless transform, so it is never inferred.
    """
    mode = camera_config.get("handeye_mode")
    if mode is None:
        mode = (calibration_config.get("handeye") or {}).get("mode")
    if mode is None:
        raise ConfigError(
            f"No hand-eye mode for camera {camera_config.get('name', '?')!r}. "
            f"Set 'handeye_mode' on the camera in cameras.yaml, or "
            f"'handeye.mode' in calibration.yaml. Expected one of: "
            f"{', '.join(HANDEYE_MODES)}")
    mode = str(mode).strip().lower()
    if mode not in HANDEYE_MODES:
        raise ConfigError(
            f"Invalid hand-eye mode {mode!r} for camera "
            f"{camera_config.get('name', '?')!r}. Expected one of: "
            f"{', '.join(HANDEYE_MODES)}")
    return mode


def require_handeye_verified(calibration_config: Mapping[str, Any]) -> None:
    """Refuse to solve while the eye-in-hand / eye-to-hand setup is unconfirmed."""
    handeye = calibration_config.get("handeye") or {}
    if not handeye.get("verified_by_user", False):
        raise ConfigError(
            "Hand-eye mounting is not confirmed.\n"
            "  Check, for each camera, that its 'handeye_mode' in cameras.yaml\n"
            "  matches the physical rig:\n"
            "    eye_in_hand -> camera ON the robot, board FIXED on the table\n"
            "    eye_to_hand -> camera FIXED in the cell, board ON the robot\n"
            f"  Then set 'handeye.verified_by_user: true' in {CALIBRATION_CONFIG}")


def is_placeholder_serial(serial: Any) -> bool:
    """True if a serial is still an unfilled template value."""
    if serial is None:
        return True
    text = str(serial).strip()
    if not text:
        return True
    upper = text.upper()
    return "REPLACE" in upper or upper in {"SERIAL_NUMBER", "XXXXXXXX", "TODO", "NONE"}


# --------------------------------------------------------------------------
# Data layout: every camera owns a disjoint subtree. Nothing is shared.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraPaths:
    """Filesystem layout for one camera. Camera subtrees never overlap."""
    camera: str
    root: Path

    @property
    def intrinsics(self) -> Path:
        return self.root / "intrinsics"

    @property
    def intrinsics_images(self) -> Path:
        return self.intrinsics / "images"

    @property
    def intrinsics_observations(self) -> Path:
        return self.intrinsics / "observations"

    @property
    def intrinsics_result(self) -> Path:
        return self.intrinsics / "result.yaml"

    @property
    def handeye(self) -> Path:
        return self.root / "handeye"

    @property
    def handeye_images(self) -> Path:
        return self.handeye / "images"

    @property
    def handeye_observations(self) -> Path:
        return self.handeye / "observations"

    @property
    def handeye_waypoints(self) -> Path:
        return self.handeye / "waypoints"

    @property
    def waypoints_file(self) -> Path:
        return self.handeye_waypoints / "waypoints.yaml"

    @property
    def initial_center(self) -> Path:
        return self.handeye_waypoints / "initial_center.yaml"

    @property
    def selection(self) -> Path:
        return self.handeye / "selection"

    @property
    def waypoint_scores(self) -> Path:
        return self.selection / "waypoint_scores.csv"

    @property
    def preliminary_result(self) -> Path:
        return self.handeye / "preliminary_result_all_30.yaml"

    @property
    def final_result(self) -> Path:
        return self.handeye / "final_result_best_20.yaml"

    @property
    def verification(self) -> Path:
        return self.handeye / "verification"

    @property
    def sessions(self) -> Path:
        return self.handeye / "sessions"

    def all_dirs(self) -> tuple[Path, ...]:
        return (self.intrinsics_images, self.intrinsics_observations,
                self.handeye_images, self.handeye_observations,
                self.handeye_waypoints, self.selection, self.verification)

    def ensure(self) -> None:
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)


def camera_paths(camera: str, data_root: Path | None = None) -> CameraPaths:
    """Build the path set for one camera, rejecting anything path-unsafe."""
    if not camera or "/" in camera or "\\" in camera or camera.startswith("."):
        raise ConfigError(f"Invalid camera name for a path: {camera!r}")
    root = Path(data_root or DATA_DIR) / camera
    return CameraPaths(camera=camera, root=root)


# --------------------------------------------------------------------------
# Atomic writes. A half-written observation is worse than a missing one.
# --------------------------------------------------------------------------

def _atomic_write_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent,
        prefix=f".{path.name}.", suffix=".tmp", delete=False)
    try:
        with handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def save_yaml(path: Path, data: Mapping[str, Any], header: str | None = None) -> None:
    """Write YAML atomically, converting NumPy types to plain Python first."""
    body = yaml.safe_dump(to_builtin(data), sort_keys=False, default_flow_style=False,
                          allow_unicode=True, width=100)
    if header:
        prefix = "".join(f"# {line}\n" for line in header.splitlines())
        body = prefix + body
    _atomic_write_text(Path(path), body)


def save_csv(path: Path, rows: Sequence[Mapping[str, Any]],
             fieldnames: Sequence[str] | None = None) -> None:
    """Write a CSV atomically."""
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    import io
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    for row in rows:
        writer.writerow({key: to_builtin(row.get(key)) for key in fieldnames})
    _atomic_write_text(Path(path), buffer.getvalue())


def to_builtin(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays and Paths to YAML-safe types."""
    if isinstance(value, np.ndarray):
        return [to_builtin(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if not math.isfinite(number) else number
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def timestamp_utc() -> str:
    """ISO-8601 UTC timestamp, used in every saved record."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def timestamp_slug() -> str:
    """Filesystem-safe local timestamp, used for session folder names."""
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


# --------------------------------------------------------------------------
# Rigid transforms
# --------------------------------------------------------------------------

def rotvec_to_matrix(rotvec: Sequence[float]) -> np.ndarray:
    """Axis-angle rotation vector (UR rx,ry,rz / OpenCV rvec) -> 3x3 matrix."""
    import cv2
    vector = np.asarray(rotvec, dtype=np.float64).reshape(3, 1)
    matrix, _ = cv2.Rodrigues(vector)
    return matrix


def matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> axis-angle rotation vector."""
    import cv2
    vector, _ = cv2.Rodrigues(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    return vector.reshape(3)


def pose_to_matrix(pose: Sequence[float]) -> np.ndarray:
    """UR TCP pose [x,y,z,rx,ry,rz] -> 4x4 homogeneous transform.

    (rx, ry, rz) is a ROTATION VECTOR, not roll/pitch/yaw.
    """
    pose = np.asarray(pose, dtype=np.float64).reshape(6)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotvec_to_matrix(pose[3:6])
    transform[:3, 3] = pose[0:3]
    return transform


def matrix_to_pose(transform: np.ndarray) -> np.ndarray:
    """4x4 homogeneous transform -> UR TCP pose [x,y,z,rx,ry,rz]."""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    pose = np.zeros(6, dtype=np.float64)
    pose[0:3] = transform[:3, 3]
    pose[3:6] = matrix_to_rotvec(transform[:3, :3])
    return pose


def make_transform(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    """Assemble a 4x4 transform from a 3x3 rotation and a 3-vector."""
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Inverse of a 4x4 rigid transform (transpose-based, not a full inverse)."""
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ transform[:3, 3]
    return inverse


def rotation_angle_deg(rotation: np.ndarray) -> float:
    """Magnitude of a 3x3 rotation, in degrees, numerically safe near 0 and pi."""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    cosine = (trace - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def transform_difference(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """(translation distance in metres, rotation angle in degrees) between poses."""
    a = np.asarray(a, dtype=np.float64).reshape(4, 4)
    b = np.asarray(b, dtype=np.float64).reshape(4, 4)
    translation = float(np.linalg.norm(a[:3, 3] - b[:3, 3]))
    rotation = rotation_angle_deg(a[:3, :3].T @ b[:3, :3])
    return translation, rotation


def pose_difference(pose_a: Sequence[float], pose_b: Sequence[float]) -> tuple[float, float]:
    """(translation in metres, rotation in degrees) between two UR TCP poses."""
    return transform_difference(pose_to_matrix(pose_a), pose_to_matrix(pose_b))


def is_rotation_matrix(matrix: np.ndarray, tolerance: float = 1e-6) -> bool:
    """True if `matrix` is a proper orthonormal rotation (det == +1)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3):
        return False
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=tolerance):
        return False
    return abs(float(np.linalg.det(matrix)) - 1.0) < tolerance


def rotvec_to_rpy_deg(rotvec: Sequence[float]) -> np.ndarray:
    """Axis-angle -> intrinsic XYZ roll/pitch/yaw in degrees, FOR DISPLAY ONLY.

    Provided so the UI can show something human-readable. Never use the result
    for calibration maths -- Euler angles are ambiguous and gimbal-locked.
    """
    matrix = rotvec_to_matrix(rotvec)
    sy = math.hypot(matrix[0, 0], matrix[1, 0])
    if sy < 1e-9:                                    # gimbal lock
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = 0.0
    else:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        pitch = math.atan2(-matrix[2, 0], sy)
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    return np.degrees([roll, pitch, yaw])


# --------------------------------------------------------------------------
# Small numeric helpers
# --------------------------------------------------------------------------

def normalize_min_max(values: Sequence[float]) -> np.ndarray:
    """Scale to [0, 1]. An all-equal input maps to all zeros, not NaN.

    Used so that pixels, millimetres and degrees can be combined into one
    score without adding incompatible units (section X1).
    """
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    low, high = float(finite.min()), float(finite.max())
    if high - low < 1e-12:
        return np.zeros_like(array)
    scaled = (array - low) / (high - low)
    return np.where(np.isfinite(scaled), scaled, 1.0)


def median_absolute_deviation(values: Sequence[float]) -> tuple[float, float]:
    """(median, MAD scaled to be a std-dev estimate) for robust outlier tests."""
    array = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if array.size == 0:
        return float("nan"), float("nan")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median))) * 1.4826
    return median, mad


def error_statistics(errors: Iterable[float]) -> dict:
    """mean / median / rms / max / count for a set of scalar errors."""
    array = np.asarray([e for e in errors if np.isfinite(e)], dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None,
                "rms": None, "max": None, "min": None, "std": None}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "rms": float(np.sqrt(np.mean(array ** 2))),
        "max": float(np.max(array)),
        "min": float(np.min(array)),
        "std": float(np.std(array)),
    }


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------

def setup_logging(name: str, camera: str | None = None,
                  console_level: int = logging.INFO,
                  log_dir: Path | None = None) -> logging.Logger:
    """Console + rotating-per-run file logging.

    The console stays quiet (INFO and up, terse format) because these tools run
    alongside a live preview; the file keeps full DEBUG detail for diagnosis.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    console = logging.StreamHandler()
    console.setLevel(console_level)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console)

    directory = Path(log_dir or LOG_DIR)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        suffix = f"_{camera}" if camera else ""
        path = directory / f"{name}{suffix}_{timestamp_slug()}.log"
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
        logger.addHandler(file_handler)
        logger.debug("Log file: %s", path)
    except OSError as exc:                       # logging must never be fatal
        logger.warning("File logging disabled (%s)", exc)
    return logger


def describe_environment() -> dict:
    """Versions of everything that can change a numeric result."""
    import platform
    import sys
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    for module_name, key in (("cv2", "opencv"), ("scipy", "scipy")):
        try:
            module = __import__(module_name)
            info[key] = getattr(module, "__version__", "unknown")
        except ImportError:
            info[key] = "not installed"
    return info


def json_safe(data: Any) -> str:
    """Compact JSON for log lines."""
    return json.dumps(to_builtin(data), separators=(",", ":"), default=str)
