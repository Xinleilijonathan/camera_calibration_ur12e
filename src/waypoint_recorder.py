"""Records one waypoint: image + ACTUAL robot state, atomically or not at all.

Implements the procedure in section P. The ordering matters and is not
negotiable:

  1. verify AprilTag detection
  2. verify enough tags / corners
  3. verify the board is not clipped
  4. verify the robot is stationary
  5. wait for mechanical and camera settling
  6. capture the image
  7. read the ACTUAL robot state
  8. save atomically

Steps 4-7 are the ones that decide whether the calibration is any good. A
waypoint pairs an image with a robot pose, and the pairing is only true if the
arm was genuinely still when the shutter opened. An image taken while the arm
was still ringing, or paired with a commanded target rather than the measured
position, injects an error that no amount of downstream maths can remove --
and it will not show up as a large reprojection error either, because each
individual board detection remains perfectly sharp and self-consistent.

If ANY step fails, nothing is written and the waypoint count does not advance.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from calibration_utils import (JOINT_NAMES, describe_environment, load_yaml,
                               pose_to_matrix, save_yaml, timestamp_utc)

LOGGER = logging.getLogger(__name__)


class RecordingRejected(RuntimeError):
    """A record attempt failed a check. Nothing was written."""


@dataclass
class WaypointRecord:
    """Everything saved for one waypoint (section S)."""
    number: int
    camera_name: str
    camera_serial: str
    timestamp: str
    image_name: str
    actual_q: np.ndarray
    actual_tcp: np.ndarray
    actual_qd: np.ndarray
    detection: dict
    tag_ids: list[int]
    tag_corners: np.ndarray
    object_points: np.ndarray
    image_points: np.ndarray
    board_pose_camera: dict | None = None
    joint_delta_from_center: dict = field(default_factory=dict)
    tcp_translation_delta_from_center_mm: float = 0.0
    tcp_rotation_delta_from_center_deg: float = 0.0
    dominant_joint_change: str = ""
    intrinsics_file: str = ""
    intrinsics_timestamp: str = ""
    settle_time_s: float = 0.0
    robot_mode: int | None = None
    safety_mode: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"waypoint_{self.number:03d}"

    def to_dict(self) -> dict:
        """The on-disk observation record.

        Joints and TCP are stored both as raw vectors and as named fields:
        the vectors are what the solver reads, the named fields are what a
        human reads six months later.
        """
        return {
            "camera_name": self.camera_name,
            "camera_serial": self.camera_serial,
            "waypoint_number": self.number,
            "waypoint_name": self.name,
            "timestamp": self.timestamp,
            "image": self.image_name,

            "actual_joints": {
                name: float(value)
                for name, value in zip(JOINT_NAMES, self.actual_q)},
            "actual_joints_rad": self.actual_q.tolist(),
            "actual_joints_deg": np.degrees(self.actual_q).tolist(),
            "actual_joint_velocity_rad_s": self.actual_qd.tolist(),

            # UR TCP pose: rx, ry, rz form a ROTATION VECTOR (axis-angle).
            # They are NOT roll/pitch/yaw. See calibration_utils.
            "actual_tcp": {
                "x": float(self.actual_tcp[0]), "y": float(self.actual_tcp[1]),
                "z": float(self.actual_tcp[2]), "rx": float(self.actual_tcp[3]),
                "ry": float(self.actual_tcp[4]), "rz": float(self.actual_tcp[5]),
                "rotation_representation": "axis-angle rotation vector (NOT rpy)",
            },
            "actual_tcp_vector": self.actual_tcp.tolist(),

            "joint_delta_from_center": self.joint_delta_from_center,
            "tcp_translation_delta_from_center_mm":
                self.tcp_translation_delta_from_center_mm,
            "tcp_rotation_delta_from_center_deg":
                self.tcp_rotation_delta_from_center_deg,
            "dominant_joint_change": self.dominant_joint_change,

            "detected_tag_ids": self.tag_ids,
            "number_of_tags": len(self.tag_ids),
            "number_of_corners": int(self.tag_corners.reshape(-1, 2).shape[0]),
            "detected_tag_corners": self.tag_corners.tolist(),
            "object_points": self.object_points.tolist(),
            "image_points": self.image_points.tolist(),
            "board_pose_camera": self.board_pose_camera,
            "detection_quality": self.detection,

            "intrinsics_file": self.intrinsics_file,
            "intrinsics_timestamp": self.intrinsics_timestamp,
            "settle_time_s": self.settle_time_s,
            "robot_mode": self.robot_mode,
            "safety_mode": self.safety_mode,
            **self.extra,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "WaypointRecord":
        board_pose = data.get("board_pose_camera")
        return cls(
            number=int(data["waypoint_number"]),
            camera_name=str(data.get("camera_name", "")),
            camera_serial=str(data.get("camera_serial", "")),
            timestamp=str(data.get("timestamp", "")),
            image_name=str(data.get("image", "")),
            actual_q=np.asarray(data["actual_joints_rad"], dtype=np.float64).reshape(6),
            actual_tcp=np.asarray(data["actual_tcp_vector"], dtype=np.float64).reshape(6),
            actual_qd=np.asarray(
                data.get("actual_joint_velocity_rad_s", [0.0] * 6),
                dtype=np.float64).reshape(6),
            detection=dict(data.get("detection_quality") or {}),
            tag_ids=[int(i) for i in data.get("detected_tag_ids", [])],
            tag_corners=np.asarray(
                data.get("detected_tag_corners", []), dtype=np.float64),
            object_points=np.asarray(
                data.get("object_points", []), dtype=np.float64).reshape(-1, 3),
            image_points=np.asarray(
                data.get("image_points", []), dtype=np.float64).reshape(-1, 2),
            board_pose_camera=dict(board_pose) if board_pose else None,
            joint_delta_from_center=dict(data.get("joint_delta_from_center") or {}),
            tcp_translation_delta_from_center_mm=float(
                data.get("tcp_translation_delta_from_center_mm", 0.0)),
            tcp_rotation_delta_from_center_deg=float(
                data.get("tcp_rotation_delta_from_center_deg", 0.0)),
            dominant_joint_change=str(data.get("dominant_joint_change", "")),
            intrinsics_file=str(data.get("intrinsics_file", "")),
            intrinsics_timestamp=str(data.get("intrinsics_timestamp", "")),
            settle_time_s=float(data.get("settle_time_s", 0.0)),
            robot_mode=data.get("robot_mode"),
            safety_mode=data.get("safety_mode"),
        )

    @property
    def board_transform(self) -> np.ndarray | None:
        """Board pose in the camera frame, as a 4x4."""
        if not self.board_pose_camera:
            return None
        rvec = np.asarray(self.board_pose_camera["rvec"], dtype=np.float64)
        tvec = np.asarray(self.board_pose_camera["tvec"], dtype=np.float64)
        transform = np.eye(4)
        transform[:3, :3], _ = cv2.Rodrigues(rvec.reshape(3, 1))
        transform[:3, 3] = tvec
        return transform

    @property
    def tcp_transform(self) -> np.ndarray:
        """TCP (flange) pose in the robot base frame, as a 4x4."""
        return pose_to_matrix(self.actual_tcp)


class WaypointRecorder:
    """Runs the section-P procedure and owns the on-disk waypoint set."""

    def __init__(self, paths, camera, detector, robot, config: Mapping[str, Any],
                 camera_matrix, dist_coeffs, intrinsics_info: Mapping[str, Any]):
        self.paths = paths
        self.camera = camera
        self.detector = detector
        self.robot = robot
        collection = dict(config.get("waypoint_collection") or {})
        self.settle_time_s = float(collection.get("settle_time_s", 0.5))
        self.stationary_threshold = float(
            collection.get("stationary_velocity_threshold", 0.005))
        self.stationary_samples = int(
            collection.get("stationary_consecutive_samples", 5))
        self.stationary_interval = float(
            collection.get("stationary_sample_interval_s", 0.02))
        self.stationary_timeout = float(collection.get("stationary_timeout_s", 5.0))
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.intrinsics_info = dict(intrinsics_info or {})
        self.records: list[WaypointRecord] = []

    # -- the procedure -----------------------------------------------------

    def record(self, number: int) -> WaypointRecord:
        """Attempt one waypoint. Raises RecordingRejected; writes nothing on failure."""
        started = time.time()

        # STEPS 1-3: is the board usable at all? Checked BEFORE waiting for the
        # robot, so an obviously bad frame fails fast.
        preview = self._detect(self.camera.read(flush=0).image)
        if not preview.valid:
            raise RecordingRejected("; ".join(preview.reasons) or "board not valid")

        # STEP 4: the robot must genuinely be stationary.
        self._verify_stationary()

        # STEP 5: settle. Even at zero commanded velocity the arm is still
        # ringing mechanically, and the camera's auto-exposure is still
        # converging from the last lighting change.
        time.sleep(self.settle_time_s)

        # STEP 6: capture, with a flush, so the saved image cannot be a
        # buffered frame from before the arm stopped.
        frame = self.camera.read()

        # STEP 7: read ACTUAL state, immediately after the capture, and confirm
        # nothing moved during it.
        state = self.robot.read_state()
        if not state.connected:
            raise RecordingRejected("lost the robot connection during capture")
        if state.max_joint_speed > self.stationary_threshold:
            raise RecordingRejected(
                f"robot moved during capture "
                f"({state.max_joint_speed:.4f} rad/s)")

        # Re-detect on the frame that will actually be saved. The preview frame
        # is not the saved frame, so its validity does not transfer.
        detection = self._detect(frame.image)
        if not detection.valid:
            raise RecordingRejected(
                "post-capture re-check failed: " + "; ".join(detection.reasons))
        if not detection.has_pose:
            raise RecordingRejected("no board pose from the captured frame")

        # STEP 8: build and write the record atomically.
        record = self._build(number, frame, detection, state)
        self._write(record, frame)
        LOGGER.info("Recorded %s in %.2f s (%d tags, PnP %.3f px)",
                    record.name, time.time() - started,
                    len(record.tag_ids),
                    detection.pnp_reprojection_px or float("nan"))
        self.records.append(record)
        return record

    def _detect(self, image):
        return self.detector.process(image, self.camera_matrix, self.dist_coeffs,
                                     require_pose=True)

    def _verify_stationary(self) -> None:
        """Require several consecutive quiet velocity samples (section Q).

        One quiet sample proves nothing: the arm passes through zero velocity
        at every direction reversal, and RTDE velocity samples are noisy.
        """
        quiet = 0
        deadline = time.monotonic() + self.stationary_timeout
        peak = float("inf")
        while time.monotonic() < deadline:
            state = self.robot.read_state()
            if not state.connected:
                raise RecordingRejected("robot connection lost while settling")
            peak = state.max_joint_speed
            quiet = quiet + 1 if peak <= self.stationary_threshold else 0
            if quiet >= self.stationary_samples:
                return
            time.sleep(self.stationary_interval)
        raise RecordingRejected(
            f"robot not stationary: peak joint speed {peak:.4f} rad/s exceeds "
            f"{self.stationary_threshold:.4f} rad/s")

    def _build(self, number, frame, detection, state) -> WaypointRecord:
        deltas = self.robot.deltas_from_center(state)
        board_pose = {
            "rvec": detection.rvec.tolist(),
            "tvec": detection.tvec.tolist(),
            "distance_m": detection.distance_m,
            "tilt_deg": detection.tilt_deg,
            "pnp_reprojection_px": detection.pnp_reprojection_px,
            "pnp_max_reprojection_px": detection.pnp_max_reprojection_px,
            "frame": ("board frame expressed in the camera frame; rvec is "
                      "axis-angle"),
        }
        return WaypointRecord(
            number=number,
            camera_name=self.camera.name,
            camera_serial=self.camera.serial,
            timestamp=timestamp_utc(),
            image_name=f"waypoint_{number:03d}.png",
            actual_q=state.q.copy(),
            actual_tcp=state.tcp.copy(),
            actual_qd=state.qd.copy(),
            detection=detection.summary(),
            tag_ids=[int(i) for i in detection.ids.ravel()],
            tag_corners=detection.corners.copy(),
            object_points=detection.object_points.copy(),
            image_points=detection.image_points.copy(),
            board_pose_camera=board_pose,
            joint_delta_from_center={
                name: float(value) for name, value
                in zip(JOINT_NAMES, deltas.get("joint_delta_deg", [0.0] * 6))},
            tcp_translation_delta_from_center_mm=float(
                deltas.get("tcp_translation_delta_mm", 0.0)),
            tcp_rotation_delta_from_center_deg=float(
                deltas.get("tcp_rotation_delta_deg", 0.0)),
            dominant_joint_change=str(deltas.get("dominant_joint_change", "")),
            settle_time_s=self.settle_time_s,
            intrinsics_file=str(self.paths.intrinsics_result),
            intrinsics_timestamp=str(self.intrinsics_info.get("timestamp", "")),
            robot_mode=state.robot_mode,
            safety_mode=state.safety_mode,
            extra={"camera": self.camera.describe(),
                   "image_size": [frame.width, frame.height]},
        )

    def _write(self, record: WaypointRecord, frame) -> None:
        """Image first, then metadata; metadata is the commit point.

        If the process dies between the two, the orphan image is harmless --
        every loader keys off the metadata. The reverse order could leave
        metadata pointing at an image that does not exist.
        """
        image_path = self.paths.handeye_images / record.image_name
        if not cv2.imwrite(str(image_path), frame.image):
            raise RecordingRejected(f"could not write image {image_path}")
        try:
            save_yaml(self.paths.handeye_observations / f"{record.name}.yaml",
                      record.to_dict(),
                      header=(f"Waypoint {record.number} for {record.camera_name} "
                              f"(serial {record.camera_serial}).\n"
                              f"actual_tcp rx/ry/rz is an AXIS-ANGLE ROTATION "
                              f"VECTOR, not roll/pitch/yaw."))
        except Exception:
            # Do not leave an image with no metadata: it would look like a
            # recorded waypoint to a human but be invisible to every loader.
            image_path.unlink(missing_ok=True)
            raise

    # -- set management ----------------------------------------------------

    def undo_last(self) -> str | None:
        """Remove the most recent waypoint's image and metadata together."""
        if not self.records:
            return None
        record = self.records.pop()
        for path in (self.paths.handeye_observations / f"{record.name}.yaml",
                     self.paths.handeye_images / record.image_name):
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                LOGGER.warning("Could not remove %s: %s", path, exc)
        LOGGER.info("Undid %s", record.name)
        return record.name

    def save_master(self, center_q, center_tcp, extra: Mapping[str, Any] | None = None):
        """Write waypoints.yaml, the master list of everything recorded."""
        data = {
            "camera_name": self.camera.name,
            "camera_serial": self.camera.serial,
            "timestamp": timestamp_utc(),
            "count": len(self.records),
            "calibration_center_q": (None if center_q is None
                                     else np.asarray(center_q).tolist()),
            "calibration_center_tcp": (None if center_tcp is None
                                       else np.asarray(center_tcp).tolist()),
            "intrinsics_file": str(self.paths.intrinsics_result),
            "waypoints": [{
                "name": record.name,
                "number": record.number,
                "timestamp": record.timestamp,
                "actual_joints_rad": record.actual_q.tolist(),
                "actual_joints_deg": np.degrees(record.actual_q).tolist(),
                "actual_tcp": record.actual_tcp.tolist(),
                "image": f"../images/{record.image_name}",
                "observation": f"../observations/{record.name}.yaml",
                "tags": len(record.tag_ids),
                "pnp_reprojection_px": (record.board_pose_camera or {}).get(
                    "pnp_reprojection_px"),
            } for record in self.records],
            **(dict(extra) if extra else {}),
        }
        save_yaml(self.paths.waypoints_file, data, header=(
            f"Master waypoint list for {self.camera.name}. "
            f"All poses are ACTUAL measured robot state."))
        return self.paths.waypoints_file


def save_initial_center(paths, camera, state, envelope=None) -> Path:
    """Write initial_center.yaml (section AO)."""
    data = {
        "camera": camera.name,
        "camera_serial": camera.serial,
        "timestamp": timestamp_utc(),
        "actual_q_rad": state.q.tolist(),
        "actual_q_deg": np.degrees(state.q).tolist(),
        "actual_q_named": {name: float(value)
                           for name, value in zip(JOINT_NAMES, state.q)},
        "actual_tcp": state.tcp.tolist(),
        "actual_tcp_named": {
            "x": float(state.tcp[0]), "y": float(state.tcp[1]),
            "z": float(state.tcp[2]), "rx": float(state.tcp[3]),
            "ry": float(state.tcp[4]), "rz": float(state.tcp[5]),
            "rotation_representation": "axis-angle rotation vector (NOT rpy)",
        },
        "robot_mode": state.robot_mode,
        "safety_mode": state.safety_mode,
        "environment": describe_environment(),
    }
    if envelope is not None:
        data["safety"] = envelope.describe()
    save_yaml(paths.initial_center, data, header=(
        "Calibration centre for this session. Every recorded waypoint's\n"
        "joint_delta_from_center is measured against this pose."))
    return paths.initial_center


def load_waypoints(paths, logger=None) -> list[WaypointRecord]:
    """Load every stored waypoint observation for one camera, in order."""
    records = []
    for path in sorted(paths.handeye_observations.glob("waypoint_*.yaml")):
        try:
            records.append(WaypointRecord.from_dict(load_yaml(path)))
        except Exception as exc:
            message = f"Could not read {path}: {exc}"
            if logger:
                logger.error(message)
            raise RuntimeError(message) from exc
    records.sort(key=lambda r: r.number)
    return records
