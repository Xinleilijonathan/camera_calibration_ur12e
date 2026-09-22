"""Shared startup for the interactive scripts: connect, check, confirm, arm.

Encodes the section-AI startup contract:

  * connect READ-ONLY and display state first
  * nothing can move until the operator types an explicit confirmation
  * the calibration centre is captured from ACTUAL measured state
  * any failure tears everything down and stops the robot
"""
from __future__ import annotations

import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from apriltag_detector import build_detector
from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               load_calibration_config, load_cameras_config,
                               resolve_camera, resolve_handeye_mode,
                               timestamp_slug)
from camera_interface import open_camera
from intrinsic_calibration import check_resolution_match, load_intrinsics
from jog_controller import JogState
from pose_diversity import PoseDiversityAnalyzer
from robot_interface import RobotInterface
from safety import SafetyEnvelope, SafetyError, load_envelope

LOGGER = logging.getLogger(__name__)

CONFIRMATION_PHRASE = "MOVE"


@dataclass
class SessionContext:
    """Everything an interactive script needs, all validated."""
    camera_name: str
    camera_config: dict
    cameras_config: dict
    calibration_config: dict
    envelope: SafetyEnvelope
    paths: Any
    detector: Any
    handeye_mode: str
    camera: Any = None
    robot: RobotInterface | None = None
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    intrinsics: dict | None = None
    jog_state: JogState | None = None
    analyzer: PoseDiversityAnalyzer | None = None

    def close(self) -> None:
        """Stop the robot, then release the camera. Order matters."""
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception as exc:
                LOGGER.error("Robot teardown: %s", exc)
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception as exc:
                LOGGER.error("Camera teardown: %s", exc)


def load_context(camera_name: str, require_intrinsics: bool = True) -> SessionContext:
    """Load and validate all configuration. Touches no hardware."""
    cameras_config = load_cameras_config()
    calibration_config = load_calibration_config()
    camera_config = resolve_camera(camera_name, cameras_config)
    envelope = load_envelope()
    paths = camera_paths(camera_name)
    paths.ensure()
    detector = build_detector(calibration_config)
    handeye_mode = resolve_handeye_mode(camera_config, calibration_config)

    context = SessionContext(
        camera_name=camera_name, camera_config=camera_config,
        cameras_config=cameras_config, calibration_config=calibration_config,
        envelope=envelope, paths=paths, detector=detector,
        handeye_mode=handeye_mode)

    if require_intrinsics:
        camera_matrix, dist_coeffs, intrinsics = load_intrinsics(paths.intrinsics_result)
        context.camera_matrix = camera_matrix
        context.dist_coeffs = dist_coeffs
        context.intrinsics = intrinsics

    context.jog_state = JogState(envelope.config)
    context.analyzer = PoseDiversityAnalyzer(calibration_config)
    return context


def print_header(context: SessionContext, title: str) -> None:
    """The startup block. Read it before touching anything."""
    print("=" * 74)
    print(f"{title} -- {context.camera_name}")
    print("=" * 74)
    print(f"Camera      : {context.camera_config.get('model', '?')} "
          f"serial {context.camera_config.get('serial')}")
    print(f"Mounting    : {context.handeye_mode}")
    if context.handeye_mode == "eye_in_hand":
        print("              camera ON the robot; board must be FIXED on the table")
    else:
        print("              camera FIXED in the cell; board must be ON the robot")
    print(f"Board       : {context.detector.spec.describe()}")
    print(f"Intrinsics  : {'loaded' if context.camera_matrix is not None else 'NOT LOADED'}")
    print()
    print("SAFETY")
    for line in context.envelope.summary_text().splitlines():
        print(f"  {line}")
    print()


def open_camera_checked(context: SessionContext) -> Any:
    """Open the camera and verify it matches the solved intrinsics."""
    context.camera = open_camera(context.camera_config)
    if context.intrinsics is not None:
        check_resolution_match(context.intrinsics, context.camera.resolution)
    return context.camera


def connect_robot_readonly(context: SessionContext) -> RobotInterface:
    """Connect read-only and print the live state. Cannot move anything."""
    robot = RobotInterface(context.envelope)
    robot.connect()
    context.robot = robot

    state = robot.read_state()
    print("ROBOT STATE (read-only)")
    print(f"  mode          : {state.mode_text()}")
    print(f"  joints (deg)  : "
          + "  ".join(f"{v:8.3f}" for v in np.degrees(state.q)))
    print(f"  TCP (m, rad)  : "
          + "  ".join(f"{v:8.4f}" for v in state.tcp))
    print(f"  max |qd|      : {state.max_joint_speed:.5f} rad/s")
    print(f"  protective    : {state.protective_stopped}")
    print(f"  emergency     : {state.emergency_stopped}")
    print()
    return robot


def confirm_and_arm(context: SessionContext, require_motion: bool) -> bool:
    """Obtain typed confirmation, then enable motion. Returns whether armed.

    A typed word, not a keypress: it is impossible to arm a robot by leaning
    on the keyboard, and the phrase appears in the log as evidence that a
    human agreed.
    """
    robot = context.robot
    if not require_motion:
        print("Running READ-ONLY: no motion commands will be sent.")
        print("Move the arm by hand with the pendant in freedrive if you wish.")
        print()
        return False

    if not context.envelope.allow_motion:
        print("MOTION IS DISABLED in config/safety.yaml "
              "(connection.allow_motion: false).")
        print("Continuing read-only. Move the arm with the pendant, or enable")
        print("motion in the config once you have verified the limits.")
        print()
        return False

    unverified = context.envelope.unverified_sections()
    if unverified:
        print("REFUSING TO ARM. These must be verified for THIS cell first:")
        for section in unverified:
            print(f"  * {section}")
        print()
        print("Continuing read-only.")
        print()
        return False

    print("-" * 74)
    print("ABOUT TO ENABLE ROBOT MOTION")
    print("-" * 74)
    print(f"  Target       : {context.envelope.robot_ip}"
          f"{'  (URSim)' if context.envelope.is_simulator else '  *** PHYSICAL ROBOT ***'}")
    print(f"  Joint speed  : {context.envelope.motion.joint_speed:.3f} rad/s")
    print(f"  Max step     : "
          f"{np.degrees(context.envelope.motion.max_joint_step):.1f} deg per command")
    print()
    print("  Check now: the area is clear, the e-stop is within reach, and the")
    print("  board is mounted as the mounting mode above describes.")
    print()
    answer = input(f'  Type {CONFIRMATION_PHRASE} to enable motion, '
                   f'anything else to stay read-only: ').strip()
    if answer != CONFIRMATION_PHRASE:
        print("  Not armed. Continuing read-only.")
        print()
        return False

    try:
        robot.enable_motion(confirm=True)
    except SafetyError as exc:
        print(f"  COULD NOT ARM: {exc}")
        print("  Continuing read-only.")
        print()
        return False
    print("  MOTION ENABLED.")
    print()
    return True


def capture_center(context: SessionContext):
    """Record the session's calibration centre from ACTUAL state."""
    robot = context.robot
    q, tcp = robot.capture_calibration_center()
    print("CALIBRATION CENTRE captured (all deltas are measured from here):")
    print("  joints (deg) : " + "  ".join(f"{v:8.3f}" for v in np.degrees(q)))
    print("  TCP          : " + "  ".join(f"{v:8.4f}" for v in tcp))
    print()
    return q, tcp


def prepare_output(paths, force: bool, resume: bool, logger) -> str:
    """Never silently overwrite an existing collection (section AP)."""
    existing = sorted(paths.handeye_observations.glob("waypoint_*.yaml"))
    if not existing:
        return "new"
    print()
    print(f"WARNING: {paths.handeye_observations} already holds "
          f"{len(existing)} waypoint(s).")
    if resume:
        choice = "r"
    elif force:
        choice = "a"
        print("--force: archiving the old set rather than deleting it.")
    else:
        print("  [a] archive the old set to a timestamped session folder")
        print("  [r] resume, adding to the existing set")
        print("  [q] quit")
        choice = input("Choose [a/r/q]: ").strip().lower()

    if choice == "q":
        raise SystemExit("Cancelled; nothing was changed.")
    if choice == "r":
        print(f"Resuming with {len(existing)} existing waypoint(s).")
        return "resume"
    if choice != "a":
        raise SystemExit("Unrecognised choice; nothing was changed.")

    archive = paths.sessions / timestamp_slug()
    (archive / "observations").mkdir(parents=True, exist_ok=True)
    (archive / "images").mkdir(parents=True, exist_ok=True)
    for path in existing:
        shutil.move(str(path), str(archive / "observations" / path.name))
    for image in sorted(paths.handeye_images.glob("waypoint_*.png")):
        shutil.move(str(image), str(archive / "images" / image.name))
    for name in ("waypoints.yaml", "initial_center.yaml"):
        source = paths.handeye_waypoints / name
        if source.is_file():
            shutil.move(str(source), str(archive / name))
    print(f"Archived to {archive}")
    logger.info("Archived previous waypoint set to %s", archive)
    return "archived"
