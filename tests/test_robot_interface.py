"""RobotInterface behaviour, with a fake RTDE. No real robot is ever contacted.

The properties under test are the ones that keep an operator safe:
connecting cannot move anything, motion needs explicit confirmation, every
command is validated, and stopping works from any state.
"""
import math
import sys
import types

import numpy as np
import pytest
import yaml

from calibration_utils import SAFETY_CONFIG, pose_to_matrix
from robot_interface import RobotInterface, RobotState, _rotation_from_xyz
from safety import SafetyEnvelope, SafetyError


class FakeReceive:
    """Stands in for RTDEReceiveInterface."""

    def __init__(self, q=None, tcp=None, qd=None, mode=7, safety=1):
        self.q = list(q or [0.0, -1.2, 1.0, -1.3, -1.57, 0.0])
        self.tcp = list(tcp or [0.40, 0.05, 0.30, 0.1, -0.2, 0.05])
        self.qd = list(qd or [0.0] * 6)
        self.mode = mode
        self.safety = safety
        self.protective = False
        self.emergency = False
        self.connected = True
        self.disconnected = False
        self.timestamp = 0.0

    def isConnected(self): return self.connected
    def getActualQ(self): return list(self.q)
    def getActualQd(self): return list(self.qd)
    def getActualTCPPose(self): return list(self.tcp)
    def getActualTCPSpeed(self): return [0.0] * 6
    def getRobotMode(self): return self.mode
    def getSafetyMode(self): return self.safety
    def isProtectiveStopped(self): return self.protective
    def isEmergencyStopped(self): return self.emergency
    def getTimestamp(self):
        self.timestamp += 0.008
        return self.timestamp
    def disconnect(self): self.disconnected = True


class FakeControl:
    """Stands in for RTDEControlInterface, recording every command."""

    def __init__(self, receive):
        self.receive = receive
        self.moveJ_calls = []
        self.moveL_calls = []
        self.stops = []
        self.script_stopped = False
        self.connected = True
        self.reject = False

    def isConnected(self): return self.connected

    def moveJ(self, q, speed, acceleration):
        self.moveJ_calls.append((list(q), speed, acceleration))
        if self.reject:
            return False
        self.receive.q = list(q)
        return True

    def moveL(self, pose, speed, acceleration):
        self.moveL_calls.append((list(pose), speed, acceleration))
        if self.reject:
            return False
        self.receive.tcp = list(pose)
        return True

    def stopJ(self, a=None): self.stops.append("stopJ")
    def stopL(self, a=None): self.stops.append("stopL")
    def servoStop(self, a=None): self.stops.append("servoStop")
    def stopScript(self): self.script_stopped = True
    def getInverseKinematics(self, pose, seed=None): return list(self.receive.q)
    def disconnect(self): self.connected = False


@pytest.fixture
def envelope():
    config = yaml.safe_load(SAFETY_CONFIG.read_text())
    config["connection"]["allow_motion"] = True
    config["workspace"]["verified_by_user"] = True
    config["joint_limits"]["verified_by_user"] = True
    config["waypoint_collection"] = {"stationary_velocity_threshold": 0.005,
                                     "stationary_consecutive_samples": 2,
                                     "stationary_sample_interval_s": 0.001,
                                     "stationary_timeout_s": 0.5}
    return SafetyEnvelope(config)


@pytest.fixture
def fake_rtde(monkeypatch):
    """Install fake rtde_receive / rtde_control modules."""
    receive = FakeReceive()
    control = FakeControl(receive)

    receive_module = types.ModuleType("rtde_receive")
    receive_module.RTDEReceiveInterface = lambda ip, frequency=125.0: receive
    control_module = types.ModuleType("rtde_control")
    control_module.RTDEControlInterface = lambda ip, frequency=125.0: control
    monkeypatch.setitem(sys.modules, "rtde_receive", receive_module)
    monkeypatch.setitem(sys.modules, "rtde_control", control_module)
    return receive, control


class TestConnecting:
    def test_connect_is_read_only(self, envelope, fake_rtde):
        """Connecting must not construct the interface that can move the arm."""
        receive, control = fake_rtde
        robot = RobotInterface(envelope).connect()
        assert robot.is_connected
        assert robot.motion_enabled is False
        assert robot._control is None
        assert control.moveJ_calls == []

    def test_reading_state_never_moves_anything(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = RobotInterface(envelope).connect()
        for _ in range(5):
            robot.read_state()
        assert control.moveJ_calls == [] and control.moveL_calls == []

    def test_connect_refuses_a_physical_ip_while_blocked(self, envelope, fake_rtde):
        envelope.robot_ip = "192.168.137.23"
        envelope.allow_physical_robot = False
        with pytest.raises(SafetyError, match="not local URSim"):
            RobotInterface(envelope).connect()

    def test_read_state_on_a_dead_link_reports_disconnected(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        robot = RobotInterface(envelope).connect()
        receive.connected = False
        assert robot.read_state().connected is False

    def test_read_state_survives_an_rtde_exception(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        robot = RobotInterface(envelope).connect()

        def explode():
            raise RuntimeError("link down")
        receive.getActualQ = explode
        assert robot.read_state().connected is False


class TestEnablingMotion:
    def test_confirm_is_required(self, envelope, fake_rtde):
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="confirm=True"):
            robot.enable_motion(confirm=False)
        assert not robot.motion_enabled

    def test_enable_works_with_confirmation(self, envelope, fake_rtde):
        robot = RobotInterface(envelope).connect()
        robot.enable_motion(confirm=True)
        assert robot.motion_enabled

    def test_enable_is_refused_while_the_config_interlock_is_off(
            self, envelope, fake_rtde):
        envelope.allow_motion = False
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="Motion is disabled"):
            robot.enable_motion(confirm=True)

    def test_enable_is_refused_while_limits_are_unverified(self, envelope, fake_rtde):
        envelope.config["workspace"]["verified_by_user"] = False
        envelope.workspace = type(envelope.workspace)(
            **{**envelope.workspace.__dict__, "verified_by_user": False})
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="verified_by_user"):
            robot.enable_motion(confirm=True)

    def test_enable_is_refused_while_the_robot_is_moving(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        receive.qd = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="still moving"):
            robot.enable_motion(confirm=True)

    def test_enable_is_refused_in_protective_stop(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        receive.protective = True
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="PROTECTIVE STOP"):
            robot.enable_motion(confirm=True)

    def test_motion_before_enabling_is_refused(self, envelope, fake_rtde):
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="has not been enabled"):
            robot.move_joints([0.0] * 6)


class TestMoving:
    def _armed(self, envelope, fake_rtde):
        robot = RobotInterface(envelope).connect()
        robot.enable_motion(confirm=True)
        robot.capture_calibration_center()
        return robot

    def test_small_joint_move_is_sent_at_the_configured_speed(
            self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        target = list(receive.q)
        target[5] += math.radians(2)
        robot.move_joints(target)
        assert len(control.moveJ_calls) == 1
        sent, speed, acceleration = control.moveJ_calls[0]
        assert speed == envelope.motion.joint_speed
        assert acceleration == envelope.motion.joint_acceleration
        assert np.allclose(sent, target)

    def test_oversized_step_is_refused_before_anything_is_sent(
            self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        target = list(receive.q)
        # wrist_1 has a 30 deg centre leash and a 10 deg per-command cap, so
        # 20 deg violates only the step cap.
        target[3] += math.radians(20)
        with pytest.raises(SafetyError, match="Single step"):
            robot.move_joints(target)
        assert control.moveJ_calls == []

    def test_target_beyond_the_centre_leash_is_refused(self, envelope, fake_rtde):
        """The leash bounds the whole session, not just one command.

        Checked with check_step=False so this exercises the leash alone: many
        small legal steps must still not be able to walk the arm away from the
        pose the operator verified.
        """
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        target = list(receive.q)
        target[3] += math.radians(35)          # wrist_1 leash is 30 deg
        with pytest.raises(SafetyError, match="calibration centre"):
            robot.move_joints(target, check_step=False)
        assert control.moveJ_calls == []

    def test_many_small_steps_cannot_escape_the_leash(self, envelope, fake_rtde):
        """Each step is legal; the cumulative result must still be refused."""
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        with pytest.raises(SafetyError, match="calibration centre"):
            for _ in range(20):               # 20 x 5 deg = 100 deg on wrist_1
                robot.jog_joint("wrist_1", math.radians(5))
        moved = abs(receive.q[3] - robot.calibration_center_q[3])
        assert math.degrees(moved) <= 30.0 + 1e-6

    def test_requesting_a_faster_speed_is_refused(self, envelope, fake_rtde):
        robot = self._armed(envelope, fake_rtde)
        with pytest.raises(SafetyError, match="exceeds the configured"):
            robot.move_joints(robot.calibration_center_q, speed=5.0)

    def test_a_rejected_command_stops_the_robot(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        control.reject = True
        target = list(receive.q)
        target[5] += math.radians(1)
        with pytest.raises(SafetyError, match="rejected moveJ"):
            robot.move_joints(target)
        assert control.stops, "a rejected command must trigger a stop"

    def test_jog_joint_moves_only_that_joint(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        before = list(receive.q)
        robot.jog_joint("wrist_3", math.radians(2))
        sent = control.moveJ_calls[0][0]
        for index in range(5):
            assert sent[index] == pytest.approx(before[index])
        assert sent[5] == pytest.approx(before[5] + math.radians(2))

    def test_jog_joint_accepts_a_name_or_an_index(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        robot.jog_joint(5, math.radians(1))
        assert len(control.moveJ_calls) == 1

    def test_unknown_joint_is_refused(self, envelope, fake_rtde):
        robot = self._armed(envelope, fake_rtde)
        with pytest.raises((SafetyError, ValueError)):
            robot.jog_joint("elbow_2", 0.01)

    def test_cartesian_jog_translates_in_the_base_frame(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = self._armed(envelope, fake_rtde)
        before = list(receive.tcp)
        robot.jog_tcp(translation=(0.005, 0, 0), frame="base")
        sent = control.moveL_calls[0][0]
        assert sent[0] == pytest.approx(before[0] + 0.005)
        assert sent[1] == pytest.approx(before[1])

    def test_cartesian_jog_outside_the_workspace_is_refused(
            self, envelope, fake_rtde):
        receive, control = fake_rtde
        receive.tcp = [0.595, 0.0, 0.30, 0.0, 0.0, 0.0]
        robot = self._armed(envelope, fake_rtde)
        with pytest.raises(SafetyError, match="workspace"):
            robot.jog_tcp(translation=(0.01, 0, 0))
        assert control.moveL_calls == []

    def test_unknown_jog_frame_is_refused(self, envelope, fake_rtde):
        robot = self._armed(envelope, fake_rtde)
        with pytest.raises(SafetyError, match="Unknown jog frame"):
            robot.jog_tcp(translation=(0.001, 0, 0), frame="galactic")

    def test_rotation_composes_as_matrices_not_by_adding_rotvecs(self):
        """Adding rotation vectors is wrong; the helper must multiply."""
        combined = _rotation_from_xyz([0.3, 0.4, 0.5])
        from calibration_utils import is_rotation_matrix, matrix_to_rotvec
        assert is_rotation_matrix(combined)
        assert not np.allclose(matrix_to_rotvec(combined), [0.3, 0.4, 0.5])


class TestCalibrationCentre:
    def test_centre_is_captured_from_actual_state(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        robot = RobotInterface(envelope).connect()
        q, tcp = robot.capture_calibration_center()
        assert np.allclose(q, receive.q)
        assert np.allclose(tcp, receive.tcp)

    def test_centre_capture_requires_a_stationary_robot(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        receive.qd = [0.3] * 6
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="still moving"):
            robot.capture_calibration_center()

    def test_deltas_report_per_joint_and_tcp_change(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        robot = RobotInterface(envelope).connect()
        robot.capture_calibration_center()
        receive.q[5] += math.radians(7)
        receive.tcp[0] += 0.030
        deltas = robot.deltas_from_center(robot.read_state())
        assert deltas["joint_delta_deg"][5] == pytest.approx(7.0, abs=1e-6)
        assert deltas["tcp_translation_delta_mm"] == pytest.approx(30.0, abs=0.1)
        assert deltas["dominant_joint_change"] == "wrist_3"

    def test_deltas_are_empty_before_a_centre_exists(self, envelope, fake_rtde):
        robot = RobotInterface(envelope).connect()
        assert robot.deltas_from_center(robot.read_state()) == {}


class TestSettling:
    def test_waits_for_consecutive_quiet_samples(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        robot = RobotInterface(envelope).connect()
        robot.wait_until_stationary()

    def test_times_out_on_a_robot_that_keeps_moving(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        receive.qd = [0.5] * 6
        robot = RobotInterface(envelope).connect()
        with pytest.raises(SafetyError, match="did not settle"):
            robot.wait_until_stationary()

    def test_lost_connection_while_settling_is_reported(self, envelope, fake_rtde):
        receive, _ = fake_rtde
        robot = RobotInterface(envelope).connect()
        receive.connected = False              # link drops AFTER connecting
        with pytest.raises(SafetyError, match="Lost the robot connection"):
            robot.wait_until_stationary()


class TestStopping:
    def test_stop_issues_every_stop_command(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = RobotInterface(envelope).connect()
        robot.enable_motion(confirm=True)
        robot.stop()
        assert set(control.stops) >= {"stopJ", "stopL", "servoStop"}

    def test_stop_is_safe_before_motion_is_enabled(self, envelope, fake_rtde):
        RobotInterface(envelope).connect().stop()      # must not raise

    def test_stop_never_raises_even_if_the_sdk_throws(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = RobotInterface(envelope).connect()
        robot.enable_motion(confirm=True)

        def explode(*args):
            raise RuntimeError("controller gone")
        control.stopJ = explode
        control.stopL = explode
        control.servoStop = explode
        robot.stop()          # a stop that throws is a stop that did not happen

    def test_emergency_stop_tears_down_motion(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = RobotInterface(envelope).connect()
        robot.enable_motion(confirm=True)
        robot.emergency_software_stop()
        assert control.script_stopped
        assert not robot.motion_enabled

    def test_disconnect_stops_first_then_closes(self, envelope, fake_rtde):
        receive, control = fake_rtde
        robot = RobotInterface(envelope).connect()
        robot.enable_motion(confirm=True)
        robot.disconnect()
        assert control.stops
        assert control.script_stopped
        assert receive.disconnected
        assert not robot.is_connected

    def test_disconnect_is_safe_to_call_twice(self, envelope, fake_rtde):
        robot = RobotInterface(envelope).connect()
        robot.disconnect()
        robot.disconnect()

    def test_context_manager_disconnects_on_exception(self, envelope, fake_rtde):
        receive, control = fake_rtde
        with pytest.raises(RuntimeError):
            with RobotInterface(envelope) as robot:
                robot.enable_motion(confirm=True)
                raise RuntimeError("boom")
        assert receive.disconnected


class TestRobotState:
    def test_max_joint_speed(self):
        state = RobotState(connected=True, qd=np.array([0.0, 0.1, -0.3, 0, 0, 0]))
        assert state.max_joint_speed == pytest.approx(0.3)

    def test_missing_velocity_reads_as_infinite_not_zero(self):
        """A missing velocity must never look like 'stationary'."""
        assert RobotState(connected=True).max_joint_speed == math.inf

    def test_as_dict_is_yaml_safe(self):
        from calibration_utils import to_builtin
        state = RobotState(connected=True, q=np.zeros(6), qd=np.zeros(6),
                           tcp=np.zeros(6), robot_mode=7, safety_mode=1)
        yaml.safe_dump(to_builtin(state.as_dict()))

    def test_mode_text_is_human_readable(self):
        state = RobotState(connected=True, robot_mode=7, safety_mode=1)
        assert state.mode_text() == "RUNNING / NORMAL"
