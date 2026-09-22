"""UR12e connection via ur_rtde. Read-only by default; motion is interlocked.

THE ONLY FILE IN THIS PROJECT THAT SENDS ROBOT MOTION COMMANDS.

Design rules, in priority order:

1. Connecting never moves anything. `connect()` opens RTDEReceiveInterface
   only. RTDEControlInterface -- the object that can command motion -- is not
   even constructed unless `enable_motion()` is called explicitly, after the
   safety envelope and the live robot state have both been checked.

2. Every motion command passes through SafetyEnvelope first: joint limits,
   deviation from the session's calibration centre, workspace box, and
   per-command step size.

3. Motion is discrete and blocking: moveJ / moveL at low speed, one nudge per
   command. There is no streaming controller and no servo loop, so there is no
   state that can run away if this process stalls or dies.

4. Stopping is unconditional. `stop()` and `disconnect()` work from any state
   and are called from `finally` blocks.

This reuses the approach proven in ~/leader_arm (same library, same state
reads, same stop-in-finally discipline) but is a separate implementation,
because calibration needs discrete low-speed jogging rather than 500 Hz
servoJ streaming.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from calibration_utils import (JOINT_NAMES, matrix_to_pose, pose_to_matrix,
                               transform_difference)
from safety import (ROBOT_MODE_NAMES, SAFETY_MODE_NAMES, SafetyEnvelope,
                    SafetyError, _validate_vector)

LOGGER = logging.getLogger(__name__)


@dataclass
class RobotState:
    """One snapshot of ACTUAL measured robot state. Never a commanded target."""
    connected: bool
    q: np.ndarray | None = None                  # actual joint positions, rad
    qd: np.ndarray | None = None                 # actual joint velocities, rad/s
    tcp: np.ndarray | None = None                # actual TCP [x,y,z,rx,ry,rz]
    tcp_speed: np.ndarray | None = None
    robot_mode: int | None = None
    safety_mode: int | None = None
    protective_stopped: bool = False
    emergency_stopped: bool = False
    controller_timestamp: float | None = None
    host_timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict:
        """Plain dict for SafetyEnvelope.check_robot_state and for saving."""
        return {
            "connected": self.connected,
            "q": None if self.q is None else self.q.tolist(),
            "qd": None if self.qd is None else self.qd.tolist(),
            "tcp": None if self.tcp is None else self.tcp.tolist(),
            "robot_mode": self.robot_mode,
            "safety_mode": self.safety_mode,
            "protective_stopped": self.protective_stopped,
            "emergency_stopped": self.emergency_stopped,
            "controller_timestamp": self.controller_timestamp,
            "host_timestamp": self.host_timestamp,
        }

    @property
    def max_joint_speed(self) -> float:
        """Peak absolute joint velocity, rad/s."""
        if self.qd is None:
            return float("inf")
        return float(np.max(np.abs(self.qd)))

    def mode_text(self) -> str:
        return (f"{ROBOT_MODE_NAMES.get(self.robot_mode, self.robot_mode)} / "
                f"{SAFETY_MODE_NAMES.get(self.safety_mode, self.safety_mode)}")

    def joints_deg(self) -> np.ndarray:
        return np.degrees(self.q) if self.q is not None else np.zeros(6)


class RobotInterface:
    """RTDE connection with a hard separation between reading and commanding."""

    def __init__(self, envelope: SafetyEnvelope):
        self.envelope = envelope
        self._receive = None
        self._control = None
        self._motion_enabled = False
        self.calibration_center_q: np.ndarray | None = None
        self.calibration_center_tcp: np.ndarray | None = None
        self._last_command_time = 0.0

    # -- connection --------------------------------------------------------

    def connect(self) -> "RobotInterface":
        """Open a READ-ONLY RTDE connection. Cannot move the robot.

        RTDEControlInterface is deliberately not constructed here: it is the
        object capable of commanding motion, and it also uploads a control
        script to the controller. Reading state must not require either.
        """
        self.envelope.check_connection_allowed()
        try:
            from rtde_receive import RTDEReceiveInterface
        except ImportError as exc:
            raise SafetyError(
                "ur_rtde is not installed. pip install ur_rtde") from exc

        LOGGER.info("Connecting (read-only) to %s", self.envelope.robot_ip)
        try:
            self._receive = RTDEReceiveInterface(
                self.envelope.robot_ip, self.envelope.rtde_frequency)
        except Exception as exc:
            raise SafetyError(
                f"Could not open RTDE receive to {self.envelope.robot_ip}: {exc}\n"
                f"  Is URSim running and the arm powered on? Check "
                f"http://localhost:6080") from exc
        if not self._receive.isConnected():
            raise SafetyError(f"RTDE receive did not connect to {self.envelope.robot_ip}")
        LOGGER.info("Read-only connection established to %s", self.envelope.robot_ip)
        return self

    @property
    def is_connected(self) -> bool:
        try:
            return self._receive is not None and self._receive.isConnected()
        except Exception:
            return False

    @property
    def motion_enabled(self) -> bool:
        return self._motion_enabled and self._control is not None

    # -- reading -----------------------------------------------------------

    def read_state(self) -> RobotState:
        """Snapshot of ACTUAL measured state. Never raises on a dead link."""
        if self._receive is None:
            return RobotState(connected=False)
        try:
            if not self._receive.isConnected():
                return RobotState(connected=False)
            return RobotState(
                connected=True,
                q=np.asarray(self._receive.getActualQ(), dtype=np.float64),
                qd=np.asarray(self._receive.getActualQd(), dtype=np.float64),
                tcp=np.asarray(self._receive.getActualTCPPose(), dtype=np.float64),
                tcp_speed=np.asarray(self._receive.getActualTCPSpeed(), dtype=np.float64),
                robot_mode=int(self._receive.getRobotMode()),
                safety_mode=int(self._receive.getSafetyMode()),
                protective_stopped=bool(self._receive.isProtectiveStopped()),
                emergency_stopped=bool(self._receive.isEmergencyStopped()),
                controller_timestamp=float(self._receive.getTimestamp()),
            )
        except Exception as exc:
            LOGGER.error("RTDE read failed: %s", exc)
            return RobotState(connected=False)

    def require_state(self, require_stationary: bool = False) -> RobotState:
        """Read state and raise SafetyError if it forbids operation."""
        state = self.read_state()
        self.envelope.check_robot_state(state.as_dict(),
                                        require_stationary=require_stationary)
        return state

    def wait_until_stationary(self, threshold: float | None = None,
                              consecutive: int = 5, interval_s: float = 0.02,
                              timeout_s: float = 5.0) -> RobotState:
        """Block until joint velocity stays below `threshold` for N samples.

        A single sample below threshold is not enough: the arm passes through
        zero velocity at every direction reversal, and RTDE samples are noisy.
        Requiring consecutive quiet samples is what makes this meaningful.
        """
        collection = self.envelope.config.get("waypoint_collection", {})
        if threshold is None:
            threshold = float(collection.get("stationary_velocity_threshold", 0.005))
        quiet = 0
        deadline = time.monotonic() + timeout_s
        last_peak = float("inf")
        state = self.read_state()
        while time.monotonic() < deadline:
            state = self.read_state()
            if not state.connected:
                raise SafetyError("Lost the robot connection while waiting for it to settle")
            last_peak = state.max_joint_speed
            quiet = quiet + 1 if last_peak <= threshold else 0
            if quiet >= consecutive:
                return state
            time.sleep(interval_s)
        raise SafetyError(
            f"Robot did not settle within {timeout_s:.1f} s "
            f"(peak joint speed {last_peak:.4f} rad/s > {threshold:.4f}). "
            f"Is something still driving it, or is the threshold too tight?")

    # -- calibration centre ------------------------------------------------

    def capture_calibration_center(self) -> tuple[np.ndarray, np.ndarray]:
        """Record the session's reference pose from ACTUAL measured state.

        Every deviation limit and every displayed delta is relative to this.
        """
        state = self.require_state(require_stationary=True)
        self.calibration_center_q = state.q.copy()
        self.calibration_center_tcp = state.tcp.copy()
        LOGGER.info("Calibration centre q (deg): %s",
                    np.round(np.degrees(state.q), 3).tolist())
        LOGGER.info("Calibration centre TCP: %s", np.round(state.tcp, 5).tolist())
        return self.calibration_center_q, self.calibration_center_tcp

    def deltas_from_center(self, state: RobotState) -> dict:
        """Per-joint and TCP deviation from the calibration centre."""
        if self.calibration_center_q is None or state.q is None:
            return {}
        joint_delta = state.q - self.calibration_center_q
        translation, rotation = transform_difference(
            pose_to_matrix(self.calibration_center_tcp), pose_to_matrix(state.tcp))
        return {
            "joint_delta_rad": joint_delta.tolist(),
            "joint_delta_deg": np.degrees(joint_delta).tolist(),
            "tcp_translation_delta_mm": translation * 1000.0,
            "tcp_rotation_delta_deg": rotation,
            "dominant_joint_change": JOINT_NAMES[int(np.argmax(np.abs(joint_delta)))],
        }

    # -- enabling motion ---------------------------------------------------

    def enable_motion(self, confirm: bool = False) -> None:
        """Construct RTDEControlInterface. Nothing can move before this.

        Requires, in order: the config interlock, a healthy stationary robot,
        and an explicit `confirm=True` from a caller that has obtained typed
        user consent.
        """
        self.envelope.check_motion_allowed()
        if not confirm:
            raise SafetyError(
                "enable_motion() requires confirm=True. The caller must obtain "
                "explicit user confirmation before motion becomes possible.")
        if self._motion_enabled:
            return
        if not self.is_connected:
            raise SafetyError("Connect read-only and verify state before enabling motion")

        self.require_state(require_stationary=True)
        for section in self.envelope.unverified_sections():
            raise SafetyError(
                f"Refusing to enable motion: {section} is still false. Verify the "
                f"real limits for this cell and set the flag deliberately.")

        try:
            from rtde_control import RTDEControlInterface
        except ImportError as exc:
            raise SafetyError("ur_rtde is not installed") from exc

        LOGGER.warning("ENABLING MOTION on %s", self.envelope.robot_ip)
        try:
            self._control = RTDEControlInterface(
                self.envelope.robot_ip, self.envelope.rtde_frequency)
        except Exception as exc:
            raise SafetyError(f"Could not open RTDE control: {exc}") from exc
        if not self._control.isConnected():
            raise SafetyError("RTDE control did not connect")

        self._motion_enabled = True
        LOGGER.warning("Motion ENABLED. Joint speed %.3f rad/s, accel %.3f rad/s^2",
                       self.envelope.motion.joint_speed,
                       self.envelope.motion.joint_acceleration)

    # -- motion ------------------------------------------------------------

    def move_joints(self, target: Sequence[float],
                    check_step: bool = True,
                    speed: float | None = None,
                    acceleration: float | None = None) -> RobotState:
        """Blocking moveJ to a validated absolute joint target.

        moveJ, not servoJ/speedJ: it is a single bounded motion with a defined
        end state. If this process dies mid-move the controller simply
        finishes it and stops, with nothing left streaming.
        """
        self._require_motion()
        state = self.require_state()
        target = self.envelope.check_joint_target(
            target, current=state.q, center=self.calibration_center_q,
            check_step=check_step)

        speed = self.envelope.motion.joint_speed if speed is None else speed
        acceleration = (self.envelope.motion.joint_acceleration
                        if acceleration is None else acceleration)
        if speed > self.envelope.motion.joint_speed:
            raise SafetyError(
                f"Requested joint speed {speed} exceeds the configured "
                f"{self.envelope.motion.joint_speed}")

        LOGGER.debug("moveJ -> %s deg (speed %.3f)",
                     np.round(np.degrees(target), 3).tolist(), speed)
        try:
            ok = self._control.moveJ(target.tolist(), speed, acceleration)
        except Exception as exc:
            self.stop()
            raise SafetyError(f"moveJ failed, robot stopped: {exc}") from exc
        if ok is False:
            self.stop()
            raise SafetyError("Controller rejected moveJ; robot stopped")
        self._last_command_time = time.time()
        return self.read_state()

    def move_pose(self, target: Sequence[float],
                  check_step: bool = True,
                  speed: float | None = None,
                  acceleration: float | None = None) -> RobotState:
        """Blocking moveL to a validated absolute TCP pose.

        The resulting JOINT configuration is also checked, so a Cartesian jog
        cannot sneak past the joint limits or the centre leash via IK.
        """
        self._require_motion()
        state = self.require_state()
        target = self.envelope.check_pose_target(
            target, current=state.tcp, check_step=check_step)

        joints = self.inverse_kinematics(target, seed=state.q)
        if joints is not None:
            self.envelope.joints.check(joints)
            if self.calibration_center_q is not None:
                self.envelope.joints.check_deviation(joints, self.calibration_center_q)
        else:
            LOGGER.warning("No IK solution available to pre-check the Cartesian target")

        speed = self.envelope.motion.tcp_speed if speed is None else speed
        acceleration = (self.envelope.motion.tcp_acceleration
                        if acceleration is None else acceleration)
        LOGGER.debug("moveL -> %s", np.round(target, 5).tolist())
        try:
            ok = self._control.moveL(target.tolist(), speed, acceleration)
        except Exception as exc:
            self.stop()
            raise SafetyError(f"moveL failed, robot stopped: {exc}") from exc
        if ok is False:
            self.stop()
            raise SafetyError("Controller rejected moveL; robot stopped")
        self._last_command_time = time.time()
        return self.read_state()

    def jog_joint(self, joint: str | int, delta_rad: float) -> RobotState:
        """Move ONE named joint by a relative amount from its actual position."""
        index = (JOINT_NAMES.index(joint) if isinstance(joint, str) else int(joint))
        if not 0 <= index < 6:
            raise SafetyError(f"Invalid joint {joint!r}")
        state = self.require_state()
        target = state.q.copy()
        target[index] += float(delta_rad)
        return self.move_joints(target)

    def jog_tcp(self, translation: Sequence[float] = (0, 0, 0),
                rotation_rad: Sequence[float] = (0, 0, 0),
                frame: str = "base") -> RobotState:
        """Nudge the TCP. `frame` is 'base' or 'tool'.

        Rotations compose as matrices, never by adding rotation vectors --
        rotation vectors are not additive, and adding them produces a
        silently wrong orientation.
        """
        state = self.require_state()
        current = pose_to_matrix(state.tcp)
        delta = np.eye(4)
        delta[:3, :3] = _rotation_from_xyz(rotation_rad)
        delta[:3, 3] = np.asarray(translation, dtype=np.float64)

        if frame == "tool":
            target = current @ delta
        elif frame == "base":
            target = np.eye(4)
            target[:3, :3] = delta[:3, :3] @ current[:3, :3]
            target[:3, 3] = current[:3, 3] + delta[:3, 3]
        else:
            raise SafetyError(f"Unknown jog frame {frame!r}; expected 'base' or 'tool'")
        return self.move_pose(matrix_to_pose(target))

    def inverse_kinematics(self, pose: Sequence[float],
                           seed: Sequence[float] | None = None):
        """IK via the controller, or None if unavailable."""
        if self._control is None:
            return None
        try:
            pose = _validate_vector(pose, "IK pose", size=6)
            if seed is not None:
                result = self._control.getInverseKinematics(
                    pose.tolist(), np.asarray(seed, dtype=np.float64).tolist())
            else:
                result = self._control.getInverseKinematics(pose.tolist())
            if result is None or len(result) != 6:
                return None
            solution = np.asarray(result, dtype=np.float64)
            return solution if np.all(np.isfinite(solution)) else None
        except Exception as exc:
            LOGGER.debug("IK unavailable: %s", exc)
            return None

    # -- stopping ----------------------------------------------------------

    def stop(self) -> None:
        """Stop commanded motion. Safe to call from any state, never raises.

        Called from every error path and every `finally`. A stop that throws
        is a stop that did not happen, so every failure is logged and the next
        stop method is still attempted.
        """
        if self._control is None:
            return
        for name in ("stopJ", "stopL", "servoStop"):
            method = getattr(self._control, name, None)
            if method is None:
                continue
            try:
                method(self.envelope.motion.joint_acceleration)
            except TypeError:
                try:
                    method()
                except Exception as exc:
                    LOGGER.error("Could not confirm %s: %s", name, exc)
            except Exception as exc:
                LOGGER.error("Could not confirm %s: %s", name, exc)
        LOGGER.info("Stop commands issued")

    def emergency_software_stop(self) -> None:
        """ESC handler: stop motion and tear down the control interface.

        This is a SOFTWARE stop of commands this program issued. It is not a
        substitute for the physical emergency stop, which remains the only
        thing that removes drive power.
        """
        LOGGER.warning("SOFTWARE EMERGENCY STOP requested")
        self.stop()
        if self._control is not None:
            try:
                self._control.stopScript()
            except Exception as exc:
                LOGGER.error("stopScript failed: %s", exc)
        self._motion_enabled = False

    def disconnect(self) -> None:
        """Stop motion, then close both interfaces. Never raises."""
        try:
            self.stop()
        finally:
            for name, interface in (("control", self._control), ("receive", self._receive)):
                if interface is None:
                    continue
                try:
                    if name == "control":
                        interface.stopScript()
                except Exception as exc:
                    LOGGER.error("stopScript on disconnect failed: %s", exc)
                try:
                    interface.disconnect()
                except Exception as exc:
                    LOGGER.error("Disconnecting %s failed: %s", name, exc)
            self._control = None
            self._receive = None
            self._motion_enabled = False
            LOGGER.info("Disconnected")

    def _require_motion(self) -> None:
        self.envelope.check_motion_allowed()
        if not self.motion_enabled:
            raise SafetyError(
                "Motion has not been enabled. Call enable_motion(confirm=True) "
                "after the user has explicitly confirmed.")

    def __enter__(self) -> "RobotInterface":
        return self.connect()

    def __exit__(self, *exc_info) -> None:
        self.disconnect()


def _rotation_from_xyz(angles: Sequence[float]) -> np.ndarray:
    """Small intrinsic X-then-Y-then-Z rotation, as a matrix.

    Used only to turn jog key presses into a rotation increment. Composing
    matrices is required: rotation vectors cannot be added.
    """
    rx, ry, rz = (float(a) for a in angles)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx
