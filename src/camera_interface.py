"""Camera enumeration and capture, addressed by SERIAL NUMBER, never by index.

Two backends:
  * realsense : Intel RealSense via pyrealsense2 (optional dependency)
  * v4l2      : any UVC webcam via OpenCV, identified through udev

Why not /dev/videoN: the kernel assigns those indices in probe order, so they
are reshuffled by a reboot or a replug. Calibrating camera_2 with camera_3's
pixels is silent and unrecoverable, so every open() resolves a configured
serial to whatever device node currently carries it, and fails loudly if that
serial is absent.

Importing this module does not open any camera.
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from calibration_utils import ConfigError, is_placeholder_serial

LOGGER = logging.getLogger(__name__)

V4L_SYSFS = Path("/sys/class/video4linux")


class CameraError(RuntimeError):
    """Raised when a camera cannot be enumerated, opened, or read."""


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------

@dataclass
class CameraInfo:
    """One discovered physical camera (or one usable node of one)."""
    backend: str
    name: str
    serial: str
    device: str                         # /dev/videoN, or the RealSense USB port
    vendor_id: str = ""
    product_id: str = ""
    usb_path: str = ""                  # stable topological path, as a fallback
    product_line: str = ""              # RealSense: D400, L500, ...
    firmware: str = ""
    modes: list[dict] = field(default_factory=list)
    has_color: bool = True
    usable: bool = True
    note: str = ""

    def best_mode(self) -> dict | None:
        """Highest-resolution mode, preferring higher FPS on a tie."""
        if not self.modes:
            return None
        return max(self.modes, key=lambda m: (m["width"] * m["height"], m["fps"]))


def _udev_properties(device: str) -> dict:
    """Read udev properties for a device node. Returns {} if udevadm is absent."""
    try:
        output = subprocess.run(
            ["udevadm", "info", "--query=property", f"--name={device}"],
            capture_output=True, text=True, timeout=5, check=False).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("udevadm failed for %s: %s", device, exc)
        return {}
    properties = {}
    for line in output.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            properties[key] = value
    return properties


_FORMAT_RE = re.compile(r"\[\d+\]:\s+'(\w+)'")
_SIZE_RE = re.compile(r"Size:\s+\w+\s+(\d+)x(\d+)")
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")


def _v4l2_modes(device: str) -> list[dict]:
    """Parse `v4l2-ctl --list-formats-ext` into a list of capture modes."""
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", device, "--list-formats-ext"],
            capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        LOGGER.debug("v4l2-ctl failed for %s: %s", device, exc)
        return []

    modes: list[dict] = []
    fourcc = ""
    width = height = 0
    for line in result.stdout.splitlines():
        format_match = _FORMAT_RE.search(line)
        if format_match:
            fourcc = format_match.group(1)
            continue
        size_match = _SIZE_RE.search(line)
        if size_match:
            width, height = int(size_match.group(1)), int(size_match.group(2))
            continue
        fps_match = _FPS_RE.search(line)
        if fps_match and width and height:
            modes.append({"format": fourcc, "width": width,
                          "height": height, "fps": float(fps_match.group(1))})
    return modes


def enumerate_v4l2_cameras() -> list[CameraInfo]:
    """Discover UVC cameras through sysfs + udev.

    A single physical webcam commonly exposes several /dev/videoN nodes, only
    one of which actually streams video (the others carry metadata). Nodes
    with no enumerable capture format are reported with usable=False rather
    than hidden, so a missing camera is distinguishable from a filtered one.
    """
    if not V4L_SYSFS.is_dir():
        return []

    cameras: list[CameraInfo] = []
    for node in sorted(V4L_SYSFS.iterdir(), key=lambda p: p.name):
        device = f"/dev/{node.name}"
        if not Path(device).exists():
            continue

        properties = _udev_properties(device)
        capabilities = properties.get("ID_V4L_CAPABILITIES", "")
        if capabilities and ":capture:" not in capabilities:
            continue

        try:
            product = (node / "name").read_text(encoding="utf-8").strip()
        except OSError:
            product = properties.get("ID_MODEL", "unknown")

        serial = (properties.get("ID_SERIAL_SHORT")
                  or properties.get("ID_USB_SERIAL_SHORT") or "")
        vendor_id = properties.get("ID_VENDOR_ID", "")
        modes = _v4l2_modes(device)

        info = CameraInfo(
            backend="v4l2",
            name=product,
            serial=serial,
            device=device,
            vendor_id=vendor_id,
            product_id=properties.get("ID_MODEL_ID", ""),
            usb_path=properties.get("ID_PATH", ""),
            modes=modes,
            has_color=bool(modes),
            usable=bool(modes),
        )
        if not modes:
            info.note = "no capture formats (metadata node) - not usable for calibration"
        elif not serial:
            info.note = ("device reports NO serial number; identify it by "
                         "usb_path instead, and never replug it into a "
                         "different port")
        if vendor_id == "8086":
            info.note = (info.note + " | " if info.note else "") + \
                "Intel device: prefer the 'realsense' backend for this camera"
        cameras.append(info)
    return cameras


def enumerate_realsense_cameras() -> list[CameraInfo]:
    """Discover Intel RealSense devices. Returns [] if pyrealsense2 is absent."""
    try:
        import pyrealsense2 as rs
    except ImportError:
        LOGGER.debug("pyrealsense2 not installed; skipping RealSense enumeration")
        return []

    cameras: list[CameraInfo] = []
    try:
        context = rs.context()
        devices = context.query_devices()
    except Exception as exc:                     # librealsense raises broadly
        LOGGER.warning("RealSense enumeration failed: %s", exc)
        return []

    for device in devices:
        def get(key, default=""):
            try:
                return device.get_info(key)
            except Exception:
                return default

        modes, has_color = [], False
        try:
            for sensor in device.query_sensors():
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() != rs.stream.color:
                        continue
                    has_color = True
                    video = profile.as_video_stream_profile()
                    modes.append({"format": str(profile.format()),
                                  "width": video.width(),
                                  "height": video.height(),
                                  "fps": float(profile.fps())})
        except Exception as exc:
            LOGGER.warning("Could not list RealSense streams: %s", exc)

        # De-duplicate and keep only 8-bit colour formats useful for detection.
        unique = {(m["width"], m["height"], m["fps"], m["format"]): m for m in modes}
        cameras.append(CameraInfo(
            backend="realsense",
            name=get(rs.camera_info.name, "RealSense"),
            serial=get(rs.camera_info.serial_number),
            device=get(rs.camera_info.physical_port),
            product_line=get(rs.camera_info.product_line),
            firmware=get(rs.camera_info.firmware_version),
            usb_path=get(rs.camera_info.physical_port),
            modes=sorted(unique.values(),
                         key=lambda m: (-m["width"] * m["height"], -m["fps"])),
            has_color=has_color,
            usable=has_color,
            note="" if has_color else "no colour stream - unusable for calibration",
        ))
    return cameras


# A v4l2 node belonging to a RealSense identifies itself in its udev name.
_REALSENSE_NAME_HINT = "realsense"


def enumerate_cameras() -> list[CameraInfo]:
    """All cameras from all available backends, without double-reporting.

    A RealSense exposes four to six /dev/videoN nodes -- colour, depth, two
    infrared, metadata -- and the v4l2 pass sees every one of them, with no
    serial number, because the serial lives behind the RealSense SDK. Listing
    those alongside the SDK's own entry turns three cameras into twelve and
    trips the duplicate-serial warning on a rig that is in fact correct.

    So when the SDK has already reported a device, its v4l2 shadows are
    dropped. If the SDK found nothing -- pyrealsense2 missing, or a permissions
    problem -- the v4l2 nodes are kept, because then they are the only
    evidence the camera exists at all, and a confusing list beats a silently
    empty one.
    """
    realsense = enumerate_realsense_cameras()
    v4l2 = enumerate_v4l2_cameras()
    if not realsense:
        return v4l2
    return realsense + [c for c in v4l2
                        if _REALSENSE_NAME_HINT not in c.name.lower()]


def find_device_for_serial(serial: str, backend: str) -> CameraInfo:
    """Resolve a configured serial to the device node that currently holds it."""
    if is_placeholder_serial(serial):
        raise ConfigError(
            f"Camera serial is still the placeholder {serial!r}. "
            "Run 'scripts/list_cameras.py --emit-yaml' and fill in cameras.yaml.")

    # Enumerate the requested backend directly rather than filtering the
    # combined list: enumerate_cameras() hides the v4l2 shadows of RealSense
    # devices, which would make a RealSense deliberately configured as
    # backend: v4l2 unresolvable.
    if backend == "realsense":
        discovered = enumerate_realsense_cameras()
    elif backend == "v4l2":
        discovered = enumerate_v4l2_cameras()
    else:
        discovered = enumerate_cameras()
    candidates = [c for c in discovered if c.backend == backend and c.usable]
    matches = [c for c in candidates if c.serial and c.serial == str(serial)]
    if not matches:
        # Fall back to the stable USB topological path for cameras with no serial.
        matches = [c for c in candidates if c.usb_path and c.usb_path == str(serial)]
    if not matches:
        seen = ", ".join(f"{c.serial or c.usb_path or '?'} ({c.name})"
                         for c in candidates) or "none"
        raise CameraError(
            f"No {backend} camera with serial {serial!r} is connected.\n"
            f"  Currently connected {backend} cameras: {seen}\n"
            f"  Check the cable, then re-run scripts/list_cameras.py.")
    if len(matches) > 1:
        # Several nodes of one physical camera: take the lowest-numbered.
        matches.sort(key=lambda c: c.device)
        LOGGER.debug("Serial %s matched %d nodes; using %s",
                     serial, len(matches), matches[0].device)
    return matches[0]


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

@dataclass
class Frame:
    """One captured colour frame."""
    image: np.ndarray                  # BGR, uint8
    timestamp: float                   # time.time() at capture
    index: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height


class BaseCamera:
    """Common camera behaviour. Subclasses implement _open/_read/_close.

    Use as a context manager so the device is always released, including on
    an exception -- a camera left open blocks the next run.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.name = str(config.get("name", "camera"))
        self.serial = str(config.get("serial", ""))
        self.requested_width = int(config.get("width", 1280))
        self.requested_height = int(config.get("height", 720))
        self.requested_fps = float(config.get("fps", 30))
        capture = config.get("capture") or {}
        self.warmup_frames = int(capture.get("warmup_frames", 10))
        self.flush_frames = int(capture.get("flush_frames", 5))
        self.read_timeout_s = float(capture.get("read_timeout_s", 2.0))
        self.disable_auto_exposure = bool(capture.get("disable_auto_exposure", False))
        self.realsense_options = dict(capture.get("realsense") or {})
        self.info: CameraInfo | None = None
        self._frame_index = 0
        self._opened = False
        self._actual_resolution: tuple[int, int] | None = None

    # -- subclass hooks ----------------------------------------------------
    def _open(self) -> None:
        raise NotImplementedError

    def _read(self) -> np.ndarray | None:
        raise NotImplementedError

    def _close(self) -> None:
        raise NotImplementedError

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "BaseCamera":
        if self._opened:
            return self
        self._open()
        self._opened = True
        try:
            for _ in range(max(0, self.warmup_frames)):
                self.read(flush=0)
        except CameraError:
            self.close()
            raise
        LOGGER.info("Opened %s (serial=%s) at %dx%d",
                    self.name, self.serial, *self.resolution)
        return self

    def read(self, flush: int | None = None) -> Frame:
        """Grab a frame, optionally discarding buffered stale frames first.

        Buffered frames are the classic source of a calibration image that
        shows where the robot *was*, paired with the joint angles of where it
        *is*. Always flush before a capture that will be saved.
        """
        if not self._opened:
            raise CameraError(f"{self.name}: camera is not open")
        discard = self.flush_frames if flush is None else flush
        for _ in range(max(0, discard)):
            self._read_with_timeout()
        image = self._read_with_timeout()
        self._frame_index += 1
        self._actual_resolution = (int(image.shape[1]), int(image.shape[0]))
        return Frame(image=image, timestamp=time.time(), index=self._frame_index)

    def _read_with_timeout(self) -> np.ndarray:
        deadline = time.monotonic() + self.read_timeout_s
        while time.monotonic() < deadline:
            image = self._read()
            if image is not None and image.size:
                return image
            time.sleep(0.005)
        raise CameraError(
            f"{self.name}: no frame within {self.read_timeout_s:.1f} s. "
            "The camera may have been unplugged or claimed by another process.")

    def close(self) -> None:
        if not self._opened:
            return
        try:
            self._close()
        except Exception as exc:                 # never mask the real error
            LOGGER.warning("%s: error while closing: %s", self.name, exc)
        finally:
            self._opened = False
            LOGGER.info("Closed %s", self.name)

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def resolution(self) -> tuple[int, int]:
        if self._actual_resolution:
            return self._actual_resolution
        return self.requested_width, self.requested_height

    def describe(self) -> dict:
        """Provenance recorded alongside every observation."""
        return {
            "camera_name": self.name,
            "camera_serial": self.serial,
            "backend": self.config.get("backend", "unknown"),
            "device": self.info.device if self.info else "",
            "model": self.info.name if self.info else "",
            "requested_resolution": [self.requested_width, self.requested_height],
            "actual_resolution": list(self.resolution),
            "requested_fps": self.requested_fps,
        }

    def __enter__(self) -> "BaseCamera":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()


class V4L2Camera(BaseCamera):
    """UVC webcam through OpenCV's V4L2 backend."""

    def __init__(self, config):
        super().__init__(config)
        self._capture: cv2.VideoCapture | None = None
        self.fourcc = str(config.get("fourcc", "MJPG") or "").upper()

    def _open(self) -> None:
        self.info = find_device_for_serial(self.serial, "v4l2")
        capture = cv2.VideoCapture(self.info.device, cv2.CAP_V4L2)
        if not capture.isOpened():
            raise CameraError(
                f"{self.name}: cannot open {self.info.device}. "
                "Another process may hold it, or you may lack 'video' group access.")

        # Order matters: FOURCC before size, or the driver may refuse the size.
        if self.fourcc and len(self.fourcc) == 4:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.requested_height)
        capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
        # A 1-frame buffer keeps read() close to real time. Not all drivers obey.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.disable_auto_exposure:
            capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)   # 1 = manual on V4L2

        self._capture = capture
        actual = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if actual != (self.requested_width, self.requested_height):
            LOGGER.warning(
                "%s: requested %dx%d but the driver gave %dx%d. Intrinsics are "
                "resolution-specific -- update cameras.yaml to the real value.",
                self.name, self.requested_width, self.requested_height, *actual)

    def _read(self):
        if self._capture is None:
            return None
        ok, image = self._capture.read()
        return image if ok else None

    def _close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class RealSenseCamera(BaseCamera):
    """Intel RealSense colour stream through pyrealsense2."""

    def __init__(self, config):
        super().__init__(config)
        self._pipeline = None
        self._rs = None

    def _open(self) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise CameraError(
                "pyrealsense2 is not installed, so the 'realsense' backend is "
                "unavailable.\n"
                "  Either install it (pip install pyrealsense2) or build the "
                "Python bindings from your existing ~/librealsense source tree, "
                "or switch this camera to backend: v4l2 in cameras.yaml."
            ) from exc
        self._rs = rs

        if is_placeholder_serial(self.serial):
            raise ConfigError(
                f"{self.name}: serial is still a placeholder. "
                "Run scripts/list_cameras.py --emit-yaml.")

        self.info = find_device_for_serial(self.serial, "realsense")
        config = rs.config()
        config.enable_device(str(self.serial))     # bind to THIS serial only
        config.enable_stream(rs.stream.color, self.requested_width,
                             self.requested_height, rs.format.bgr8,
                             int(self.requested_fps))
        pipeline = rs.pipeline()
        try:
            profile = pipeline.start(config)
        except Exception as exc:
            raise CameraError(
                f"{self.name}: RealSense {self.serial} would not start at "
                f"{self.requested_width}x{self.requested_height}@"
                f"{int(self.requested_fps)}. Run scripts/list_cameras.py to see "
                f"the modes this device actually supports. ({exc})") from exc
        self._pipeline = pipeline

        self._configure_sensors(profile, rs)

    def _configure_sensors(self, profile, rs) -> None:
        """Apply the options that affect corner sharpness.

        The IR dot projector exists for depth and is useless here -- we only
        use the colour stream. On the D405, whose colour and depth come from
        the same imager, the projected dots land on the printed tags and
        measurably degrade corner localisation, so it is off by default.
        """
        device = profile.get_device()

        def set_option(sensor, option, value, label):
            try:
                if sensor.supports(option):
                    sensor.set_option(option, value)
                    LOGGER.debug("%s: %s = %s", self.name, label, value)
            except Exception as exc:
                LOGGER.warning("%s: could not set %s: %s", self.name, label, exc)

        if self.realsense_options.get("disable_emitter", True):
            try:
                for sensor in device.query_sensors():
                    set_option(sensor, rs.option.emitter_enabled, 0, "emitter_enabled")
            except Exception as exc:
                LOGGER.debug("%s: emitter control unavailable: %s", self.name, exc)

        try:
            color_sensor = device.first_color_sensor()
        except Exception as exc:
            LOGGER.debug("%s: no colour sensor handle: %s", self.name, exc)
            return

        if self.disable_auto_exposure:
            set_option(color_sensor, rs.option.enable_auto_exposure, 0,
                       "enable_auto_exposure")
        elif not self.realsense_options.get("auto_exposure_priority", False):
            # Auto-exposure priority lets the sensor drop frame rate to gather
            # light, which lengthens exposure and blurs a moving board.
            set_option(color_sensor, rs.option.auto_exposure_priority, 0,
                       "auto_exposure_priority")

    def _read(self):
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(
                timeout_ms=int(self.read_timeout_s * 1000))
        except Exception:
            return None
        color = frames.get_color_frame()
        if not color:
            return None
        # Copy: the librealsense frame buffer is recycled once this frame goes
        # out of scope, which would corrupt an image we are about to save.
        return np.array(np.asanyarray(color.get_data()), copy=True)

    def _close(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None


BACKENDS = {"v4l2": V4L2Camera, "realsense": RealSenseCamera}


def open_camera(config: Mapping[str, Any]) -> BaseCamera:
    """Construct and open the camera described by a cameras.yaml entry."""
    backend = str(config.get("backend", "")).lower()
    if backend not in BACKENDS:
        raise ConfigError(
            f"Unknown camera backend {backend!r}. Expected one of: "
            f"{', '.join(sorted(BACKENDS))}")
    return BACKENDS[backend](config).open()


def format_camera_table(cameras: Sequence[CameraInfo]) -> str:
    """Human-readable enumeration table for list_cameras.py."""
    if not cameras:
        return "No cameras detected."
    lines = []
    for index, camera in enumerate(cameras, start=1):
        best = camera.best_mode()
        lines.append(f"[{index}] {camera.name}")
        lines.append(f"      backend        : {camera.backend}")
        lines.append(f"      serial         : {camera.serial or '(none reported)'}")
        lines.append(f"      device         : {camera.device}")
        if camera.product_line:
            lines.append(f"      product line   : {camera.product_line}")
        if camera.firmware:
            lines.append(f"      firmware       : {camera.firmware}")
        if camera.vendor_id:
            lines.append(f"      usb id         : {camera.vendor_id}:{camera.product_id}")
        if camera.usb_path:
            lines.append(f"      usb path       : {camera.usb_path}")
        lines.append(f"      colour stream  : {'yes' if camera.has_color else 'NO'}")
        if best:
            lines.append(f"      best mode      : {best['width']}x{best['height']} "
                         f"@ {best['fps']:g} fps ({best['format']})")
            lines.append(f"      modes          : {len(camera.modes)} "
                         f"(use --verbose to list them)")
        else:
            lines.append("      modes          : none enumerable")
        lines.append(f"      usable         : {'YES' if camera.usable else 'no'}")
        if camera.note:
            lines.append(f"      note           : {camera.note}")
        lines.append("")
    return "\n".join(lines)
