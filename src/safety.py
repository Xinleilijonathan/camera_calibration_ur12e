"""Motion safety envelope: pure checks, no hardware access, no SDK imports.

Everything here is a predicate over numbers. It opens nothing, connects to
nothing and commands nothing, so it can be exercised exhaustively by tests
without a robot present. `robot_interface` calls into it before every command.

WHAT THIS IS NOT
----------------
These are SOFTWARE COMMAND LIMITS for this application. They are not a
substitute for, and must never be used to work around, the UR controller's own
safety configuration: safety planes, joint limits, tool/payload setup, reduced
mode, protective stop and emergency stop. Those are configured on the teach
pendant and remain the real protection. This module only makes this program
refuse to send a command it considers out of bounds.
"""
from __future__ import annotations

import ipaddress
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from calibration_utils import JOINT_NAMES, ConfigError, pose_to_matrix

# UR robot_mode values (from the RTDE spec).
ROBOT_MODE_RUNNING = 7
# UR safety_mode values.
SAFETY_MODE_NORMAL = 1
SAFETY_MODE_REDUCED = 2
SAFETY_MODE_NAMES = {
    0: "UNKNOWN", 1: "NORMAL", 2: "REDUCED", 3: "PROTECTIVE_STOP",
    4: "RECOVERY", 5: "SAFEGUARD_STOP", 6: "SYSTEM_EMERGENCY_STOP",
    7: "ROBOT_EMERGENCY_STOP", 8: "VIOLATION", 9: "FAULT",
}
ROBOT_MODE_NAMES = {
    -1: "NO_CONTROLLER", 0: "DISCONNECTED", 1: "CONFIRM_SAFETY",
    2: "BOOTING", 3: "POWER_OFF", 4: "POWER_ON", 5: "IDLE",
    6: "BACKDRIVE", 7: "RUNNING", 8: "UPDATING_FIRMWARE",
}


class SafetyError(RuntimeError):
    """Raised when a command or robot state violates the configured envelope.

    Always treat this as a stop condition. Never catch it to retry a slightly
    smaller motion.
    """


@dataclass(frozen=True)
class WorkspaceLimits:
    """Axis-aligned Cartesian box in the robot base frame, metres."""
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    min_radius_from_base_m: float = 0.0
    verified_by_user: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "WorkspaceLimits":
        try:
            limits = cls(
                x_min=float(config["x_min"]), x_max=float(config["x_max"]),
                y_min=float(config["y_min"]), y_max=float(config["y_max"]),
                z_min=float(config["z_min"]), z_max=float(config["z_max"]),
                min_radius_from_base_m=float(config.get("min_radius_from_base_m", 0.0)),
                verified_by_user=bool(config.get("verified_by_user", False)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"safety.yaml: invalid workspace block ({exc})") from exc
        for low, high, axis in ((limits.x_min, limits.x_max, "x"),
                                (limits.y_min, limits.y_max, "y"),
                                (limits.z_min, limits.z_max, "z")):
            if low >= high:
                raise ConfigError(f"safety.yaml: workspace {axis}_min >= {axis}_max")
        return limits

    def check(self, pose: Sequence[float]) -> None:
        """Raise SafetyError if a TCP pose lies outside the box."""
        x, y, z = (float(v) for v in pose[:3])
        if not all(math.isfinite(v) for v in (x, y, z)):
            raise SafetyError("TCP target contains a non-finite coordinate")
        for value, low, high, axis in ((x, self.x_min, self.x_max, "X"),
                                       (y, self.y_min, self.y_max, "Y"),
                                       (z, self.z_min, self.z_max, "Z")):
            if not low <= value <= high:
                raise SafetyError(
                    f"TCP {axis} = {value * 1000:.1f} mm is outside the configured "
                    f"workspace [{low * 1000:.0f}, {high * 1000:.0f}] mm")
        radius = math.hypot(x, y)
        if radius < self.min_radius_from_base_m:
            raise SafetyError(
                f"TCP is {radius * 1000:.0f} mm from the base Z axis, inside the "
                f"{self.min_radius_from_base_m * 1000:.0f} mm keep-out where the arm "
                f"can fold onto itself")

    def contains(self, pose: Sequence[float]) -> bool:
        try:
            self.check(pose)
        except SafetyError:
            return False
        return True


@dataclass(frozen=True)
class JointLimits:
    """Absolute joint limits plus a per-joint leash to the session centre."""
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]
    max_deviation_from_center: tuple[float, ...]
    verified_by_user: bool = False

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "JointLimits":
        minimum, maximum, deviation = [], [], []
        deviation_config = config.get("max_deviation_from_center") or {}
        for joint in JOINT_NAMES:
            entry = config.get(joint)
            if not isinstance(entry, dict):
                raise ConfigError(f"safety.yaml: joint_limits.{joint} is missing")
            low, high = float(entry["min"]), float(entry["max"])
            if low >= high:
                raise ConfigError(f"safety.yaml: joint_limits.{joint}: min >= max")
            minimum.append(low)
            maximum.append(high)
            deviation.append(float(deviation_config.get(joint, math.inf)))
        return cls(tuple(minimum), tuple(maximum), tuple(deviation),
                   bool(config.get("verified_by_user", False)))

    def check(self, joints: Sequence[float]) -> None:
        """Raise SafetyError if any joint target is out of range."""
        joints = _validate_vector(joints, "joint target")
        for index, (value, low, high) in enumerate(
                zip(joints, self.minimum, self.maximum)):
            if not low <= value <= high:
                raise SafetyError(
                    f"{JOINT_NAMES[index]} target {math.degrees(value):.2f} deg is "
                    f"outside the configured limit "
                    f"[{math.degrees(low):.1f}, {math.degrees(high):.1f}] deg")

    def check_deviation(self, joints: Sequence[float],
                        center: Sequence[float]) -> None:
        """Raise SafetyError if any joint strays too far from the centre pose.

        This is the strongest practical guard during jogging: it bounds the
        whole session to a small neighbourhood of a pose you chose and checked
        by eye, regardless of how many individual small steps are taken.
        """
        joints = _validate_vector(joints, "joint target")
        center = _validate_vector(center, "calibration centre")
        for index, (value, origin, allowed) in enumerate(
                zip(joints, center, self.max_deviation_from_center)):
            delta = abs(value - origin)
            if delta > allowed:
                raise SafetyError(
                    f"{JOINT_NAMES[index]} would be {math.degrees(delta):.2f} deg "
                    f"from the calibration centre, past the "
                    f"{math.degrees(allowed):.1f} deg limit for that joint")

    def deviations(self, joints: Sequence[float],
                   center: Sequence[float]) -> np.ndarray:
        """Signed per-joint deviation from the centre pose, radians."""
        return (np.asarray(joints, dtype=np.float64)
                - np.asarray(center, dtype=np.float64))

    def headroom(self, joints: Sequence[float],
                 center: Sequence[float]) -> np.ndarray:
        """Remaining travel per joint before the centre leash bites, radians."""
        return np.array([allowed - abs(value - origin)
                         for value, origin, allowed
                         in zip(joints, center, self.max_deviation_from_center)],
                        dtype=np.float64)


@dataclass(frozen=True)
class MotionLimits:
    """Speeds, accelerations and maximum single-command step sizes."""
    joint_speed: float
    joint_acceleration: float
    tcp_speed: float
    tcp_acceleration: float
    max_joint_step: float
    max_tcp_translation_step: float
    max_tcp_rotation_step: float
    max_replay_joint_delta: float
    watchdog_timeout_s: float
    deadman_max_sample_age_s: float

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "MotionLimits":
        try:
            limits = cls(
                joint_speed=float(config["joint_speed_rad_s"]),
                joint_acceleration=float(config["joint_acceleration_rad_s2"]),
                tcp_speed=float(config["tcp_speed_m_s"]),
                tcp_acceleration=float(config["tcp_acceleration_m_s2"]),
                max_joint_step=float(config["max_joint_step_rad"]),
                max_tcp_translation_step=float(config["max_tcp_translation_step_m"]),
                max_tcp_rotation_step=float(config["max_tcp_rotation_step_rad"]),
                max_replay_joint_delta=float(config["max_replay_joint_delta_rad"]),
                watchdog_timeout_s=float(config.get("watchdog_timeout_s", 0.5)),
                deadman_max_sample_age_s=float(
                    config.get("deadman_max_sample_age_s", 0.15)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"safety.yaml: invalid motion block ({exc})") from exc
        for name, value in (("joint_speed_rad_s", limits.joint_speed),
                            ("joint_acceleration_rad_s2", limits.joint_acceleration),
                            ("tcp_speed_m_s", limits.tcp_speed),
                            ("tcp_acceleration_m_s2", limits.tcp_acceleration)):
            if value <= 0:
                raise ConfigError(f"safety.yaml: motion.{name} must be > 0")
        # A calibration jog is a nudge. Anything faster is a configuration
        # mistake, and the consequence of the mistake is a moving robot arm.
        if limits.joint_speed > 0.5:
            raise ConfigError(
                f"safety.yaml: motion.joint_speed_rad_s = {limits.joint_speed} is "
                f"too fast for calibration jogging (hard cap 0.5 rad/s)")
        if limits.tcp_speed > 0.25:
            raise ConfigError(
                f"safety.yaml: motion.tcp_speed_m_s = {limits.tcp_speed} is too "
                f"fast for calibration jogging (hard cap 0.25 m/s)")
        return limits

    def check_joint_step(self, current: Sequence[float],
                         target: Sequence[float]) -> None:
        """Raise SafetyError if a single joint command moves too far at once."""
        current = _validate_vector(current, "current joints")
        target = _validate_vector(target, "joint target")
        deltas = np.abs(target - current)
        index = int(np.argmax(deltas))
        if deltas[index] > self.max_joint_step:
            raise SafetyError(
                f"Single step would move {JOINT_NAMES[index]} by "
                f"{math.degrees(deltas[index]):.2f} deg, past the "
                f"{math.degrees(self.max_joint_step):.1f} deg per-command limit")

    def check_pose_step(self, current: Sequence[float],
                        target: Sequence[float]) -> None:
        """Raise SafetyError if a single Cartesian command moves too far."""
        from calibration_utils import transform_difference
        current = _validate_vector(current, "current TCP pose", size=6)
        target = _validate_vector(target, "TCP target", size=6)
        translation, rotation = transform_difference(
            pose_to_matrix(current), pose_to_matrix(target))
        if translation > self.max_tcp_translation_step:
            raise SafetyError(
                f"Single step would translate the TCP by {translation * 1000:.1f} mm, "
                f"past the {self.max_tcp_translation_step * 1000:.0f} mm limit")
        if math.radians(rotation) > self.max_tcp_rotation_step:
            raise SafetyError(
                f"Single step would rotate the TCP by {rotation:.2f} deg, past the "
                f"{math.degrees(self.max_tcp_rotation_step):.1f} deg limit")


def _validate_vector(values: Sequence[float], label: str, size: int = 6) -> np.ndarray:
    """Reject wrong-length, None, NaN or infinite vectors before they reach the robot."""
    if values is None:
        raise SafetyError(f"{label} is missing")
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise SafetyError(f"{label} must have {size} values, got {array.size}")
    if not np.all(np.isfinite(array)):
        raise SafetyError(f"{label} contains a non-finite value: {array.tolist()}")
    return array


class SafetyEnvelope:
    """All configured limits, plus the two motion interlocks.

    The interlocks are config settings rather than command-line flags on
    purpose: no command typed by accident can enable robot motion.
    """

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        connection = config.get("connection") or {}
        self.robot_ip = str(connection.get("robot_ip", "127.0.0.1"))
        self.rtde_frequency = float(connection.get("rtde_frequency", 125.0))
        self.connect_timeout_s = float(connection.get("connect_timeout_s", 5.0))
        self.allow_physical_robot = bool(connection.get("allow_physical_robot", False))
        self.allow_motion = bool(connection.get("allow_motion", False))

        self.motion = MotionLimits.from_config(config.get("motion") or {})
        self.workspace = WorkspaceLimits.from_config(config.get("workspace") or {})
        self.joints = JointLimits.from_config(config.get("joint_limits") or {})
        self.preflight = dict(config.get("preflight") or {})
        self.jog = dict(config.get("jog") or {})
        self.gamepad = dict(config.get("gamepad") or {})

    # -- connection gating -------------------------------------------------

    def check_connection_allowed(self) -> None:
        """Refuse a non-loopback address unless physical operation is enabled.

        Mirrors the interlock in ~/leader_arm: the physical UR12e has not had
        its safety limits, collision checking, tool/payload configuration or
        e-stop interlock commissioned, so pointing software at it must be a
        deliberate, documented act rather than an edit to one IP string.
        """
        try:
            address = ipaddress.ip_address(self.robot_ip)
        except ValueError:
            # A hostname. Cannot prove it is local, so treat it as physical.
            if not self.allow_physical_robot:
                raise SafetyError(
                    f"robot_ip {self.robot_ip!r} is a hostname, which cannot be "
                    f"verified as local URSim. Set connection.allow_physical_robot: "
                    f"true in safety.yaml only after the robot is commissioned.")
            return
        if not address.is_loopback and not self.allow_physical_robot:
            raise SafetyError(
                f"robot_ip {self.robot_ip} is not local URSim, and "
                f"connection.allow_physical_robot is false.\n"
                f"  This interlock exists because the physical UR12e has not had "
                f"its safety limits, collision checks, tool/payload configuration "
                f"and e-stop interlock commissioned.\n"
                f"  Prove the workflow in URSim first. Then, deliberately, set "
                f"allow_physical_robot: true in config/safety.yaml.")

    def check_motion_allowed(self) -> None:
        """Refuse any motion command while the master interlock is off."""
        if not self.allow_motion:
            raise SafetyError(
                "Motion is disabled. Every motion command is refused while "
                "connection.allow_motion is false in config/safety.yaml.\n"
                "  Read-only state polling still works. Enable motion only after "
                "you have verified the workspace box, the joint limits and the "
                "calibration centre pose for this session.")

    @property
    def is_simulator(self) -> bool:
        try:
            return ipaddress.ip_address(self.robot_ip).is_loopback
        except ValueError:
            return False

    # -- state gating ------------------------------------------------------

    def check_robot_state(self, state: Mapping[str, Any],
                          require_stationary: bool = False) -> None:
        """Raise SafetyError if the reported robot state forbids motion."""
        if not state.get("connected", False):
            raise SafetyError("Robot is not connected")
        if state.get("protective_stopped"):
            raise SafetyError(
                "Robot is in PROTECTIVE STOP. Clear it on the teach pendant. "
                "Never bypass it.")
        if state.get("emergency_stopped"):
            raise SafetyError("Robot is in EMERGENCY STOP. Release it physically.")

        mode = state.get("robot_mode")
        if self.preflight.get("require_robot_mode_running", True):
            if mode != ROBOT_MODE_RUNNING:
                raise SafetyError(
                    f"Robot mode is {ROBOT_MODE_NAMES.get(mode, mode)}, not RUNNING. "
                    f"Power on and release the brakes first.")

        safety_mode = state.get("safety_mode")
        if self.preflight.get("require_safety_mode_normal", True):
            if safety_mode not in (SAFETY_MODE_NORMAL, SAFETY_MODE_REDUCED):
                raise SafetyError(
                    f"Safety mode is "
                    f"{SAFETY_MODE_NAMES.get(safety_mode, safety_mode)}, "
                    f"not NORMAL or REDUCED")

        if require_stationary:
            velocities = state.get("qd")
            if velocities is None:
                raise SafetyError("Cannot verify the robot is stationary: no qd")
            peak = float(np.max(np.abs(np.asarray(velocities, dtype=np.float64))))
            threshold = float(self.config.get("waypoint_collection", {}).get(
                "stationary_velocity_threshold", 0.005))
            if peak > threshold:
                raise SafetyError(
                    f"Robot is still moving: peak joint speed {peak:.4f} rad/s "
                    f"exceeds {threshold:.4f} rad/s")

    # -- command gating ----------------------------------------------------

    def check_joint_target(self, target: Sequence[float],
                           current: Sequence[float] | None = None,
                           center: Sequence[float] | None = None,
                           check_step: bool = True) -> np.ndarray:
        """Full validation of a joint target. Returns it as a clean array."""
        self.check_motion_allowed()
        target = _validate_vector(target, "joint target")
        self.joints.check(target)
        if center is not None:
            self.joints.check_deviation(target, center)
        if current is not None and check_step:
            self.motion.check_joint_step(current, target)
        return target

    def check_pose_target(self, target: Sequence[float],
                          current: Sequence[float] | None = None,
                          check_step: bool = True) -> np.ndarray:
        """Full validation of a Cartesian target. Returns it as a clean array."""
        self.check_motion_allowed()
        target = _validate_vector(target, "TCP target", size=6)
        self.workspace.check(target)
        if current is not None and check_step:
            self.motion.check_pose_step(current, target)
        return target

    # -- reporting ---------------------------------------------------------

    def unverified_sections(self) -> list[str]:
        """Which human-confirmation flags are still unset."""
        missing = []
        if not self.workspace.verified_by_user:
            missing.append("safety.yaml -> workspace.verified_by_user")
        if not self.joints.verified_by_user:
            missing.append("safety.yaml -> joint_limits.verified_by_user")
        return missing

    def describe(self) -> dict:
        """Safety configuration recorded in every session's metadata."""
        return {
            "robot_ip": self.robot_ip,
            "is_simulator": self.is_simulator,
            "allow_physical_robot": self.allow_physical_robot,
            "allow_motion": self.allow_motion,
            "joint_speed_rad_s": self.motion.joint_speed,
            "joint_acceleration_rad_s2": self.motion.joint_acceleration,
            "tcp_speed_m_s": self.motion.tcp_speed,
            "tcp_acceleration_m_s2": self.motion.tcp_acceleration,
            "max_joint_step_deg": math.degrees(self.motion.max_joint_step),
            "workspace_m": {
                "x": [self.workspace.x_min, self.workspace.x_max],
                "y": [self.workspace.y_min, self.workspace.y_max],
                "z": [self.workspace.z_min, self.workspace.z_max],
            },
            "workspace_verified": self.workspace.verified_by_user,
            "joint_limits_verified": self.joints.verified_by_user,
        }

    def summary_text(self) -> str:
        """Block printed at startup, before anything can move."""
        lines = [
            f"Robot IP           : {self.robot_ip}"
            f"{'  (URSim, local)' if self.is_simulator else '  (NOT loopback)'}",
            f"Physical robot     : {'ALLOWED' if self.allow_physical_robot else 'BLOCKED'}",
            f"Motion commands    : {'ENABLED' if self.allow_motion else 'DISABLED'}",
            f"Joint speed / accel: {self.motion.joint_speed:.3f} rad/s / "
            f"{self.motion.joint_acceleration:.3f} rad/s^2",
            f"TCP speed / accel  : {self.motion.tcp_speed:.3f} m/s / "
            f"{self.motion.tcp_acceleration:.3f} m/s^2",
            f"Max step per cmd   : {math.degrees(self.motion.max_joint_step):.1f} deg "
            f"joint, {self.motion.max_tcp_translation_step * 1000:.0f} mm TCP",
            f"Workspace box (mm) : "
            f"X[{self.workspace.x_min * 1000:.0f}, {self.workspace.x_max * 1000:.0f}] "
            f"Y[{self.workspace.y_min * 1000:.0f}, {self.workspace.y_max * 1000:.0f}] "
            f"Z[{self.workspace.z_min * 1000:.0f}, {self.workspace.z_max * 1000:.0f}]",
        ]
        for section in self.unverified_sections():
            lines.append(f"UNVERIFIED         : {section}")
        return "\n".join(lines)


def load_envelope(config: Mapping[str, Any] | None = None) -> SafetyEnvelope:
    """Build the envelope from safety.yaml (plus calibration.yaml thresholds)."""
    from calibration_utils import load_calibration_config, load_safety_config
    safety = dict(config) if config is not None else load_safety_config()
    if "waypoint_collection" not in safety:
        try:
            safety["waypoint_collection"] = load_calibration_config().get(
                "waypoint_collection", {})
        except ConfigError:
            safety["waypoint_collection"] = {}
    return SafetyEnvelope(safety)
