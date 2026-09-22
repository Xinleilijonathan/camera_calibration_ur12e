"""Safety envelope: limits, interlocks, and state gating.

These are the checks that stand between a typo and a moving robot arm, so
they are tested exhaustively. Nothing here touches hardware.
"""
import ipaddress
import math

import numpy as np
import pytest
import yaml

from calibration_utils import ConfigError, JOINT_NAMES, SAFETY_CONFIG
from safety import (JointLimits, MotionLimits, ROBOT_MODE_RUNNING,
                    SAFETY_MODE_NORMAL, SafetyEnvelope, SafetyError,
                    WorkspaceLimits, _validate_vector, load_envelope)


def base_config():
    config = yaml.safe_load(SAFETY_CONFIG.read_text())
    config["waypoint_collection"] = {"stationary_velocity_threshold": 0.005}
    return config


def healthy_state(**overrides):
    state = {"connected": True, "robot_mode": ROBOT_MODE_RUNNING,
             "safety_mode": SAFETY_MODE_NORMAL, "protective_stopped": False,
             "emergency_stopped": False, "qd": [0.0] * 6, "q": [0.0] * 6}
    state.update(overrides)
    return state


class TestShippedDefaults:
    """The repository must ship locked down."""

    def test_motion_is_disabled(self):
        assert load_envelope().allow_motion is False

    def test_physical_robot_is_blocked(self):
        assert load_envelope().allow_physical_robot is False

    def test_target_is_local_ursim(self):
        assert ipaddress.ip_address(load_envelope().robot_ip).is_loopback

    def test_limits_are_unverified_so_motion_cannot_be_armed(self):
        assert len(load_envelope().unverified_sections()) == 2


class TestConnectionInterlock:
    def test_loopback_is_allowed_without_the_physical_flag(self):
        config = base_config()
        config["connection"]["robot_ip"] = "127.0.0.1"
        SafetyEnvelope(config).check_connection_allowed()

    def test_physical_ip_is_refused_while_the_flag_is_false(self):
        config = base_config()
        config["connection"]["robot_ip"] = "192.168.137.23"
        with pytest.raises(SafetyError, match="not local URSim"):
            SafetyEnvelope(config).check_connection_allowed()

    def test_physical_ip_is_allowed_once_the_flag_is_set(self):
        config = base_config()
        config["connection"]["robot_ip"] = "192.168.137.23"
        config["connection"]["allow_physical_robot"] = True
        SafetyEnvelope(config).check_connection_allowed()

    def test_hostname_is_treated_as_physical(self):
        """A hostname cannot be proven local, so it must not slip through."""
        config = base_config()
        config["connection"]["robot_ip"] = "ur12e.local"
        with pytest.raises(SafetyError, match="hostname"):
            SafetyEnvelope(config).check_connection_allowed()

    def test_refusal_explains_why(self):
        config = base_config()
        config["connection"]["robot_ip"] = "10.0.0.5"
        with pytest.raises(SafetyError, match="commissioned"):
            SafetyEnvelope(config).check_connection_allowed()


class TestMotionInterlock:
    def test_motion_is_refused_while_disabled(self):
        with pytest.raises(SafetyError, match="Motion is disabled"):
            SafetyEnvelope(base_config()).check_motion_allowed()

    def test_motion_is_allowed_once_enabled(self):
        config = base_config()
        config["connection"]["allow_motion"] = True
        SafetyEnvelope(config).check_motion_allowed()

    def test_joint_target_is_refused_while_motion_is_disabled(self):
        """The interlock is checked before any geometry."""
        with pytest.raises(SafetyError, match="Motion is disabled"):
            SafetyEnvelope(base_config()).check_joint_target([0.0] * 6)


class TestWorkspaceLimits:
    def _limits(self):
        return WorkspaceLimits(-0.6, 0.6, -0.6, 0.6, 0.1, 0.9, 0.20)

    def test_pose_inside_the_box_passes(self):
        self._limits().check([0.4, 0.1, 0.3, 0, 0, 0])

    @pytest.mark.parametrize("pose,axis", [
        ([0.7, 0.0, 0.3, 0, 0, 0], "X"),
        ([-0.7, 0.0, 0.3, 0, 0, 0], "X"),
        ([0.3, 0.8, 0.3, 0, 0, 0], "Y"),
        ([0.3, 0.0, 0.05, 0, 0, 0], "Z"),
        ([0.3, 0.0, 1.5, 0, 0, 0], "Z"),
    ])
    def test_pose_outside_the_box_is_refused(self, pose, axis):
        with pytest.raises(SafetyError, match=f"TCP {axis}"):
            self._limits().check(pose)

    def test_keep_out_cylinder_around_the_base(self):
        with pytest.raises(SafetyError, match="keep-out"):
            self._limits().check([0.05, 0.05, 0.4, 0, 0, 0])

    def test_non_finite_coordinate_is_refused(self):
        with pytest.raises(SafetyError, match="non-finite"):
            self._limits().check([float("nan"), 0, 0.3, 0, 0, 0])

    def test_contains_does_not_raise(self):
        assert self._limits().contains([0.4, 0.1, 0.3, 0, 0, 0])
        assert not self._limits().contains([9.0, 0.1, 0.3, 0, 0, 0])

    def test_inverted_limits_are_refused_at_load(self):
        with pytest.raises(ConfigError, match="x_min >= x_max"):
            WorkspaceLimits.from_config({"x_min": 1.0, "x_max": -1.0,
                                         "y_min": -1, "y_max": 1,
                                         "z_min": 0, "z_max": 1})


class TestJointLimits:
    def _limits(self):
        return JointLimits(
            minimum=tuple([-3.14] * 6), maximum=tuple([3.14] * 6),
            max_deviation_from_center=(0.1745, 0.1745, 0.2618,
                                       0.5236, 0.5236, 0.7854))

    def test_in_range_passes(self):
        self._limits().check([0.0] * 6)

    def test_out_of_range_names_the_joint(self):
        target = [0.0] * 6
        target[2] = 4.0
        with pytest.raises(SafetyError, match="elbow"):
            self._limits().check(target)

    def test_wrong_length_is_refused(self):
        with pytest.raises(SafetyError, match="must have 6 values"):
            self._limits().check([0.0] * 5)

    def test_nan_is_refused(self):
        with pytest.raises(SafetyError, match="non-finite"):
            self._limits().check([0.0, float("nan"), 0, 0, 0, 0])

    def test_none_is_refused(self):
        with pytest.raises(SafetyError, match="is missing"):
            self._limits().check(None)

    def test_deviation_from_centre_is_enforced_per_joint(self):
        centre = [0.0] * 6
        target = [0.0] * 6
        target[0] = math.radians(15)          # base limit is 10 deg
        with pytest.raises(SafetyError, match="base"):
            self._limits().check_deviation(target, centre)

    def test_wrists_are_allowed_more_travel_than_the_base(self):
        """Section I: distal joints are the preferred ones to move."""
        limits = self._limits()
        centre = [0.0] * 6
        wrist = [0.0] * 6
        wrist[5] = math.radians(40)
        limits.check_deviation(wrist, centre)       # fine for wrist 3
        base = [0.0] * 6
        base[0] = math.radians(40)
        with pytest.raises(SafetyError):
            limits.check_deviation(base, centre)

    def test_headroom_reports_remaining_travel(self):
        limits = self._limits()
        current = [0.0] * 6
        current[5] = math.radians(20)
        headroom = limits.headroom(current, [0.0] * 6)
        # The configured wrist_3 leash is 0.7854 rad, a rounded pi/4, so this
        # compares against the configured value rather than exact degrees.
        assert headroom[5] == pytest.approx(0.7854 - math.radians(20), abs=1e-9)

    def test_shipped_limits_load(self):
        limits = JointLimits.from_config(base_config()["joint_limits"])
        assert len(limits.minimum) == len(JOINT_NAMES)


class TestMotionLimits:
    def _limits(self):
        return MotionLimits.from_config(base_config()["motion"])

    def test_small_joint_step_passes(self):
        current = [0.0] * 6
        target = [0.0] * 6
        target[5] = math.radians(3)
        self._limits().check_joint_step(current, target)

    def test_large_joint_step_is_refused_and_names_the_joint(self):
        current = [0.0] * 6
        target = [0.0] * 6
        target[1] = math.radians(45)
        with pytest.raises(SafetyError, match="shoulder"):
            self._limits().check_joint_step(current, target)

    def test_large_translation_step_is_refused(self):
        with pytest.raises(SafetyError, match="translate"):
            self._limits().check_pose_step([0.4, 0, 0.3, 0, 0, 0],
                                           [0.9, 0, 0.3, 0, 0, 0])

    def test_large_rotation_step_is_refused(self):
        with pytest.raises(SafetyError, match="rotate"):
            self._limits().check_pose_step(
                [0.4, 0, 0.3, 0, 0, 0], [0.4, 0, 0.3, 0, 0, math.radians(45)])

    def test_excessive_configured_speed_is_refused(self):
        config = base_config()
        config["motion"]["joint_speed_rad_s"] = 2.0
        with pytest.raises(ConfigError, match="too fast"):
            MotionLimits.from_config(config["motion"])

    def test_excessive_configured_tcp_speed_is_refused(self):
        config = base_config()
        config["motion"]["tcp_speed_m_s"] = 1.0
        with pytest.raises(ConfigError, match="too fast"):
            MotionLimits.from_config(config["motion"])

    def test_zero_speed_is_refused(self):
        config = base_config()
        config["motion"]["joint_speed_rad_s"] = 0.0
        with pytest.raises(ConfigError, match="must be > 0"):
            MotionLimits.from_config(config["motion"])


class TestRobotStateGating:
    def _envelope(self):
        config = base_config()
        config["connection"]["allow_motion"] = True
        return SafetyEnvelope(config)

    def test_healthy_state_passes(self):
        self._envelope().check_robot_state(healthy_state())

    def test_disconnected_is_refused(self):
        with pytest.raises(SafetyError, match="not connected"):
            self._envelope().check_robot_state(healthy_state(connected=False))

    def test_protective_stop_is_refused_and_never_bypassed(self):
        with pytest.raises(SafetyError, match="PROTECTIVE STOP"):
            self._envelope().check_robot_state(
                healthy_state(protective_stopped=True))

    def test_emergency_stop_is_refused(self):
        with pytest.raises(SafetyError, match="EMERGENCY STOP"):
            self._envelope().check_robot_state(
                healthy_state(emergency_stopped=True))

    def test_non_running_mode_is_refused(self):
        with pytest.raises(SafetyError, match="POWER_OFF"):
            self._envelope().check_robot_state(healthy_state(robot_mode=3))

    def test_abnormal_safety_mode_is_refused(self):
        with pytest.raises(SafetyError, match="VIOLATION"):
            self._envelope().check_robot_state(healthy_state(safety_mode=8))

    def test_reduced_mode_is_accepted(self):
        self._envelope().check_robot_state(healthy_state(safety_mode=2))

    def test_moving_robot_is_refused_when_stationary_is_required(self):
        with pytest.raises(SafetyError, match="still moving"):
            self._envelope().check_robot_state(
                healthy_state(qd=[0.0, 0.0, 0.2, 0.0, 0.0, 0.0]),
                require_stationary=True)

    def test_stationary_robot_passes(self):
        self._envelope().check_robot_state(
            healthy_state(qd=[0.001] * 6), require_stationary=True)

    def test_missing_velocity_is_refused_when_stationary_is_required(self):
        with pytest.raises(SafetyError, match="no qd"):
            self._envelope().check_robot_state(healthy_state(qd=None),
                                               require_stationary=True)


class TestCombinedTargetChecks:
    def _envelope(self):
        config = base_config()
        config["connection"]["allow_motion"] = True
        return SafetyEnvelope(config)

    def test_joint_target_checks_limits_step_and_centre_together(self):
        envelope = self._envelope()
        centre = [0.0] * 6
        current = [0.0] * 6
        good = [0.0] * 6
        good[5] = math.radians(3)
        envelope.check_joint_target(good, current=current, center=centre)

        far = [0.0] * 6
        far[5] = math.radians(60)             # inside limits, outside the leash
        with pytest.raises(SafetyError, match="calibration centre"):
            envelope.check_joint_target(far, current=far, center=centre,
                                        check_step=False)

    def test_pose_target_checks_the_workspace(self):
        envelope = self._envelope()
        with pytest.raises(SafetyError, match="workspace"):
            envelope.check_pose_target([2.0, 0, 0.3, 0, 0, 0], check_step=False)

    def test_returned_target_is_a_clean_array(self):
        envelope = self._envelope()
        target = envelope.check_joint_target([0.0] * 6, check_step=False)
        assert isinstance(target, np.ndarray) and target.shape == (6,)


class TestReporting:
    def test_summary_names_the_simulator(self):
        assert "URSim" in SafetyEnvelope(base_config()).summary_text()

    def test_summary_flags_a_non_loopback_target(self):
        config = base_config()
        config["connection"]["robot_ip"] = "192.168.1.5"
        assert "NOT loopback" in SafetyEnvelope(config).summary_text()

    def test_summary_lists_unverified_sections(self):
        assert "UNVERIFIED" in SafetyEnvelope(base_config()).summary_text()

    def test_describe_is_yaml_safe(self):
        from calibration_utils import to_builtin
        yaml.safe_dump(to_builtin(SafetyEnvelope(base_config()).describe()))

    def test_verified_config_reports_no_unverified_sections(self):
        config = base_config()
        config["workspace"]["verified_by_user"] = True
        config["joint_limits"]["verified_by_user"] = True
        assert SafetyEnvelope(config).unverified_sections() == []


class TestVectorValidation:
    def test_accepts_a_good_vector(self):
        assert _validate_vector([1, 2, 3, 4, 5, 6], "test").shape == (6,)

    def test_rejects_infinity(self):
        with pytest.raises(SafetyError, match="non-finite"):
            _validate_vector([1, 2, 3, 4, 5, float("inf")], "test")
