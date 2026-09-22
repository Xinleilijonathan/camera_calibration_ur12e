"""Per-camera intrinsic calibration: observation storage, diversity, solving.

Intrinsics are NEVER shared between cameras. Two D435s off the same production
line still differ in lens centring and distortion by more than the error we are
trying to measure, so each camera's parameters are solved from its own images
and written only into its own directory.

No hardware access here -- this module works on stored observations.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from calibration_utils import (CalibrationError, error_statistics, load_yaml,
                               save_yaml, timestamp_utc, to_builtin)

LOGGER = logging.getLogger(__name__)

#: Distortion models -> (cv2 flags, number of coefficients).
DISTORTION_MODELS = {
    "standard": (0, 5),                                    # k1 k2 p1 p2 k3
    "rational": (cv2.CALIB_RATIONAL_MODEL, 8),             # + k4 k5 k6
    "thin_prism": (cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL, 12),
}


@dataclass
class IntrinsicObservation:
    """One accepted view of the board, with everything needed to re-solve."""
    index: int
    image_name: str
    object_points: np.ndarray            # (M, 3) float64, board frame
    image_points: np.ndarray             # (M, 2) float64, pixels
    tag_ids: list[int]
    image_size: tuple[int, int]
    timestamp: str = field(default_factory=timestamp_utc)
    detection: dict = field(default_factory=dict)
    cell: tuple[int, int] | None = None  # which image cell the board centre hit
    tilt_deg: float | None = None
    distance_m: float | None = None
    area_fraction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "image": self.image_name,
            "timestamp": self.timestamp,
            "image_size": list(self.image_size),
            "tag_ids": self.tag_ids,
            "point_count": int(len(self.object_points)),
            "object_points": self.object_points.tolist(),
            "image_points": self.image_points.tolist(),
            "cell": list(self.cell) if self.cell else None,
            "tilt_deg": self.tilt_deg,
            "distance_m": self.distance_m,
            "area_fraction": self.area_fraction,
            "detection": self.detection,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "IntrinsicObservation":
        return cls(
            index=int(data["index"]),
            image_name=str(data["image"]),
            object_points=np.asarray(data["object_points"], dtype=np.float64).reshape(-1, 3),
            image_points=np.asarray(data["image_points"], dtype=np.float64).reshape(-1, 2),
            tag_ids=[int(i) for i in data.get("tag_ids", [])],
            image_size=tuple(int(v) for v in data["image_size"]),
            timestamp=str(data.get("timestamp", "")),
            detection=dict(data.get("detection") or {}),
            cell=tuple(data["cell"]) if data.get("cell") else None,
            tilt_deg=data.get("tilt_deg"),
            distance_m=data.get("distance_m"),
            area_fraction=float(data.get("area_fraction", 0.0)),
        )


class IntrinsicDiversityTracker:
    """Enforces that the observation set actually spans the imaging conditions.

    Thirty views of the board in the same place, at the same angle, at the same
    distance will produce a confident, precise and wrong calibration: with no
    perspective variety, focal length and board distance trade off against each
    other almost freely, and the distortion coefficients are unconstrained
    wherever the board never went. This tracker is what stops that happening.
    """

    def __init__(self, config: Mapping[str, Any], image_size: tuple[int, int]):
        diversity = dict(config.get("diversity") or {})
        self.cells = int(diversity.get("image_cells", 3))
        self.minimum_cells = int(diversity.get("minimum_cells_covered", 6))
        self.minimum_tilt_deg = float(diversity.get("minimum_tilt_deg", 20.0))
        self.minimum_distinct_tilts = int(diversity.get("minimum_distinct_tilts", 6))
        self.minimum_scale_ratio = float(diversity.get("minimum_scale_ratio", 1.6))
        self.minimum_new_translation_m = float(
            diversity.get("minimum_new_translation_m", 0.02))
        self.minimum_new_rotation_deg = float(
            diversity.get("minimum_new_rotation_deg", 5.0))
        self.image_size = image_size
        self.observations: list[IntrinsicObservation] = []

    def cell_for(self, center_px: Sequence[float]) -> tuple[int, int]:
        """Which cell of the NxN image grid the board centre falls in."""
        width, height = self.image_size
        column = min(self.cells - 1, max(0, int(center_px[0] / max(1, width) * self.cells)))
        row = min(self.cells - 1, max(0, int(center_px[1] / max(1, height) * self.cells)))
        return row, column

    def covered_cells(self) -> set:
        return {obs.cell for obs in self.observations if obs.cell is not None}

    def is_novel(self, detection) -> tuple[bool, str]:
        """Is this view different enough from every stored one to be worth saving?"""
        if not self.observations:
            return True, "first observation"
        if detection.tvec is None:
            return True, "no pose available; accepting on detection quality alone"
        from calibration_utils import rotation_angle_deg
        current_rotation, _ = cv2.Rodrigues(detection.rvec)
        for obs in self.observations:
            stored = obs.detection.get("rvec"), obs.detection.get("tvec")
            if stored[0] is None or stored[1] is None:
                continue
            translation = float(np.linalg.norm(
                detection.tvec - np.asarray(stored[1], dtype=np.float64)))
            stored_rotation, _ = cv2.Rodrigues(
                np.asarray(stored[0], dtype=np.float64).reshape(3, 1))
            rotation = rotation_angle_deg(current_rotation.T @ stored_rotation)
            # Either kind of novelty is enough; demanding both would reject
            # genuinely useful views (a pure rotation in place, for instance).
            if (translation < self.minimum_new_translation_m
                    and rotation < self.minimum_new_rotation_deg):
                return False, (f"too similar to observation {obs.index:02d} "
                               f"({translation * 1000:.0f} mm, {rotation:.1f} deg apart)")
        return True, "novel view"

    def add(self, observation: IntrinsicObservation) -> None:
        self.observations.append(observation)

    def report(self) -> dict:
        """What the set currently covers, and what it still lacks."""
        cells = self.covered_cells()
        tilts = [o.tilt_deg for o in self.observations if o.tilt_deg is not None]
        areas = [o.area_fraction for o in self.observations if o.area_fraction > 0]
        scale_ratio = (max(areas) / min(areas)) if len(areas) >= 2 and min(areas) > 0 else 1.0
        high_tilts = sum(1 for t in tilts if t >= self.minimum_tilt_deg)

        missing = []
        if len(cells) < self.minimum_cells:
            missing.append(
                f"board centre has visited {len(cells)}/{self.cells ** 2} image "
                f"cells (need {self.minimum_cells}) -- move it into the corners")
        if high_tilts < self.minimum_distinct_tilts:
            missing.append(
                f"only {high_tilts} views tilted past {self.minimum_tilt_deg:.0f} deg "
                f"(need {self.minimum_distinct_tilts}) -- tilt the board more")
        if scale_ratio < self.minimum_scale_ratio:
            missing.append(
                f"near/far size ratio is {scale_ratio:.2f} (need "
                f"{self.minimum_scale_ratio:.2f}) -- take some closer and some "
                f"further away")
        return {
            "count": len(self.observations),
            "cells_covered": len(cells),
            "cells_total": self.cells ** 2,
            "cell_list": sorted(cells),
            "tilted_views": high_tilts,
            "max_tilt_deg": max(tilts) if tilts else 0.0,
            "scale_ratio": scale_ratio,
            "missing": missing,
            "satisfied": not missing,
        }

    def coverage_grid(self) -> np.ndarray:
        """Per-cell observation counts, for the on-screen coverage widget."""
        grid = np.zeros((self.cells, self.cells), dtype=np.int32)
        for obs in self.observations:
            if obs.cell:
                grid[obs.cell[0], obs.cell[1]] += 1
        return grid


def solve_intrinsics(observations: Sequence[IntrinsicObservation],
                     image_size: tuple[int, int],
                     config: Mapping[str, Any]) -> dict:
    """Run cv2.calibrateCamera and return a full result record.

    Also computes per-observation and per-point reprojection statistics, which
    are what tell you whether one bad view is dragging the whole solve.
    """
    if len(observations) < 3:
        raise CalibrationError(
            f"Need at least 3 observations to solve intrinsics, have {len(observations)}")

    minimum = int(config.get("minimum_observations", 20))
    if len(observations) < minimum:
        LOGGER.warning("Only %d observations; configuration asks for %d",
                       len(observations), minimum)

    model = str(config.get("distortion_model", "standard")).lower()
    if model not in DISTORTION_MODELS:
        raise CalibrationError(
            f"Unknown distortion_model {model!r}. Supported: "
            f"{', '.join(sorted(DISTORTION_MODELS))}")
    flags, coefficient_count = DISTORTION_MODELS[model]
    if config.get("fix_aspect_ratio", False):
        flags |= cv2.CALIB_FIX_ASPECT_RATIO
    if config.get("zero_tangential_distortion", False):
        flags |= cv2.CALIB_ZERO_TANGENT_DIST

    object_points = [o.object_points.astype(np.float32).reshape(-1, 1, 3)
                     for o in observations]
    image_points = [o.image_points.astype(np.float32).reshape(-1, 1, 2)
                    for o in observations]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs, _, _, per_view = \
        cv2.calibrateCameraExtended(
            object_points, image_points, image_size, None, None,
            flags=flags, criteria=criteria)

    all_errors: list[float] = []
    per_observation = []
    for index, observation in enumerate(observations):
        projected, _ = cv2.projectPoints(
            observation.object_points, rvecs[index], tvecs[index],
            camera_matrix, dist_coeffs)
        residuals = np.linalg.norm(
            projected.reshape(-1, 2) - observation.image_points, axis=1)
        all_errors.extend(residuals.tolist())
        per_observation.append({
            "index": observation.index,
            "image": observation.image_name,
            "points": int(residuals.size),
            "mean_px": float(np.mean(residuals)),
            "rms_px": float(np.sqrt(np.mean(residuals ** 2))),
            "max_px": float(np.max(residuals)),
            "opencv_per_view_error": float(per_view[index][0]),
            "tilt_deg": observation.tilt_deg,
            "distance_m": observation.distance_m,
        })

    statistics = error_statistics(all_errors)
    fx, fy = float(camera_matrix[0, 0]), float(camera_matrix[1, 1])
    cx, cy = float(camera_matrix[0, 2]), float(camera_matrix[1, 2])
    coefficients = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)[:coefficient_count]

    width, height = image_size
    result = {
        "timestamp": timestamp_utc(),
        "image_width": int(width),
        "image_height": int(height),
        "observation_count": len(observations),
        "distortion_model": model,
        "camera_matrix": camera_matrix.tolist(),
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "distortion_coefficients": coefficients.tolist(),
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": statistics["mean"],
        "median_reprojection_error_px": statistics["median"],
        "max_reprojection_error_px": statistics["max"],
        "std_reprojection_error_px": statistics["std"],
        "total_points": statistics["count"],
        "per_observation": per_observation,
        # Sanity indicators, not calibration outputs.
        "field_of_view_deg": {
            "horizontal": 2 * math.degrees(math.atan(width / (2 * fx))),
            "vertical": 2 * math.degrees(math.atan(height / (2 * fy))),
        },
        "principal_point_offset_px": {
            "x": cx - width / 2.0,
            "y": cy - height / 2.0,
        },
        "aspect_ratio": fy / fx if fx else None,
    }
    result["warnings"] = intrinsic_warnings(result, config)
    return result


def intrinsic_warnings(result: Mapping[str, Any],
                       config: Mapping[str, Any]) -> list[str]:
    """Plausibility checks that a low RMS alone would not catch."""
    warnings: list[str] = []
    limit = float(config.get("maximum_reprojection_error", 1.0))
    if result["rms_reprojection_error_px"] > limit:
        warnings.append(
            f"RMS reprojection error {result['rms_reprojection_error_px']:.3f} px "
            f"exceeds the configured limit of {limit:.2f} px")

    aspect = result.get("aspect_ratio")
    if aspect and not 0.95 <= aspect <= 1.05:
        warnings.append(
            f"fy/fx = {aspect:.4f}. Square pixels should give ~1.0; a large "
            f"deviation usually means too little perspective variety in the set")

    offset = result["principal_point_offset_px"]
    width, height = result["image_width"], result["image_height"]
    if abs(offset["x"]) > 0.15 * width or abs(offset["y"]) > 0.15 * height:
        warnings.append(
            f"Principal point is {offset['x']:.0f}, {offset['y']:.0f} px from the "
            f"image centre, which is unusually far. Suspect poor coverage near "
            f"the image corners")

    worst = max(result["per_observation"], key=lambda o: o["rms_px"], default=None)
    if worst and result["mean_reprojection_error_px"]:
        if worst["rms_px"] > 3 * result["mean_reprojection_error_px"]:
            warnings.append(
                f"Observation {worst['index']:02d} ({worst['image']}) has "
                f"{worst['rms_px']:.3f} px RMS, far above the set mean. Consider "
                f"removing it and re-solving")
    return warnings


def save_intrinsics(path: Path, result: Mapping[str, Any],
                    camera_name: str, camera_serial: str,
                    board: Mapping[str, Any],
                    environment: Mapping[str, Any] | None = None) -> None:
    """Write result.yaml with the provenance needed to reproduce it."""
    record = dict(result)
    record["camera_name"] = camera_name
    record["camera_serial"] = camera_serial
    record["board"] = to_builtin(board)
    if environment:
        record["environment"] = to_builtin(environment)
    save_yaml(path, record, header=(
        f"Intrinsic calibration for {camera_name} (serial {camera_serial}).\n"
        f"These parameters belong to THIS camera only and must never be\n"
        f"reused for another camera, even one of the same model.\n"
        f"Valid only at {result['image_width']}x{result['image_height']}."))


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load result.yaml, validating the parts downstream code relies on."""
    path = Path(path)
    if not path.is_file():
        raise CalibrationError(
            f"No intrinsics at {path}.\n"
            f"  Run collect_intrinsics.py then solve_intrinsics.py for this camera "
            f"first. Hand-eye calibration cannot proceed without them.")
    data = load_yaml(path)
    try:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(
            data["distortion_coefficients"], dtype=np.float64).reshape(1, -1)
    except (KeyError, ValueError) as exc:
        raise CalibrationError(f"{path}: malformed intrinsics ({exc})") from exc
    if not np.all(np.isfinite(camera_matrix)) or camera_matrix[0, 0] <= 0:
        raise CalibrationError(f"{path}: implausible camera matrix")
    return camera_matrix, dist_coeffs, data


def check_resolution_match(intrinsics: Mapping[str, Any],
                           image_size: tuple[int, int]) -> None:
    """Intrinsics are resolution-specific; using them at another size is wrong.

    fx, fy, cx, cy are all in pixels. Applying 1280x720 intrinsics to a 640x480
    image scales every projection by a factor of two and the error is not
    obvious from the numbers.
    """
    expected = (int(intrinsics.get("image_width", 0)),
                int(intrinsics.get("image_height", 0)))
    if expected != tuple(int(v) for v in image_size):
        raise CalibrationError(
            f"Resolution mismatch: intrinsics were solved at "
            f"{expected[0]}x{expected[1]} but the camera is delivering "
            f"{image_size[0]}x{image_size[1]}.\n"
            f"  Intrinsics are in pixels and do not transfer between "
            f"resolutions. Either set the camera back to "
            f"{expected[0]}x{expected[1]} in cameras.yaml, or recalibrate.")
