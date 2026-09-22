"""Jog state, command construction, and the keyboard/gamepad input layer."""
import math

import numpy as np
import pytest
import yaml

from calibration_utils import JOINT_NAMES, JOINT_PRIORITY, SAFETY_CONFIG
from input_devices import (ACTION_HELP, ACTION_QUIT, ACTION_RECORD,
                           ACTION_STOP, ACTION_TOGGLE_SPACE, ACTION_UNDO,
                           GamepadInput, KeyboardInput)
from jog_controller import (CARTESIAN_SPACE, JOINT_SPACE, JogState, apply_jog)
from safety import SafetyError


def config():
    return yaml.safe_load(SAFETY_CONFIG.read_text())


def state():
    return JogState(config())


class TestJogState:
    def test_defaults_to_joint_mode_on_the_most_preferred_joint(self):
        subject = state()
        assert subject.space == JOINT_SPACE
        assert subject.selected_joint == JOINT_PRIORITY[0] == "wrist_3"

    def test_toggle_space(self):
        subject = state()
        assert subject.toggle_space() == CARTESIAN_SPACE
        assert subject.toggle_space() == JOINT_SPACE

    def test_step_modes_cycle_and_clamp(self):
        subject = state()
        subject.set_step_mode("fine")
        assert subject.cycle_step_mode(-1) == "fine"      # clamped at the bottom
        assert subject.cycle_step_mode(1) == "normal"
        assert subject.cycle_step_mode(1) == "coarse"
        assert subject.cycle_step_mode(1) == "coarse"     # clamped at the top

    def test_distal_joints_get_larger_steps_than_the_base(self):
        """Section I1: wrists are the cheap joints to move, the base is not."""
        subject = state()
        subject.set_step_mode("coarse")
        assert subject.joint_step_deg("wrist_3") > subject.joint_step_deg("base")
        assert subject.joint_step_deg("elbow") >= subject.joint_step_deg("shoulder")

    def test_no_configured_step_is_huge(self):
        """Section I1: never +-30/45/90 degree jumps."""
        subject = state()
        for mode in ("fine", "normal", "coarse"):
            subject.set_step_mode(mode)
            for joint in JOINT_NAMES:
                assert subject.joint_step_deg(joint) <= 5.0, (
                    f"{joint} {mode} step is {subject.joint_step_deg(joint)} deg")

    def test_selecting_a_joint_by_name_and_index(self):
        subject = state()
        assert subject.select_joint("elbow") == "elbow"
        assert subject.select_joint(0) == "base"

    def test_invalid_joint_is_refused(self):
        with pytest.raises(SafetyError):
            state().select_joint("knee")
        with pytest.raises(SafetyError):
            state().select_joint(9)

    def test_joint_command_direction_and_label(self):
        subject = state()
        subject.select_joint("wrist_2")
        subject.set_step_mode("normal")
        command = subject.joint_command(+1)
        assert command.joint == "wrist_2"
        assert command.joint_delta_rad > 0
        assert "Wrist 2" in command.label
        assert subject.joint_command(-1).joint_delta_rad < 0

    def test_analog_command_scales_with_magnitude(self):
        """A gentle stick must give a gentle nudge, never full speed."""
        subject = state()
        full = subject.analog_command((1.0, 0, 0), (0, 0, 0))
        half = subject.analog_command((0.5, 0, 0), (0, 0, 0))
        assert half.translation_m[0] == pytest.approx(full.translation_m[0] / 2)

    def test_empty_command_is_detected(self):
        assert state().analog_command((0, 0, 0), (0, 0, 0)).is_empty

    def test_status_lines_always_name_what_will_move(self):
        """Section G2: it must never be ambiguous which joint moves."""
        subject = state()
        subject.select_joint("elbow")
        joined = " ".join(subject.status_lines())
        assert "Elbow" in joined and "deg" in joined

    def test_describe_is_yaml_safe(self):
        from calibration_utils import to_builtin
        yaml.safe_dump(to_builtin(state().describe()))


class TestKeyboardInput:
    def _device(self):
        subject = state()
        return KeyboardInput(subject), subject

    def test_escape_requests_an_emergency_stop(self):
        device, _ = self._device()
        assert device.poll(27).action == ACTION_STOP

    def test_enter_records(self):
        device, _ = self._device()
        assert device.poll(13).action == ACTION_RECORD
        assert device.poll(10).action == ACTION_RECORD

    def test_q_quits_and_u_undoes(self):
        device, _ = self._device()
        assert device.poll(ord("q")).action == ACTION_QUIT
        assert device.poll(ord("u")).action == ACTION_UNDO
        assert device.poll(ord("h")).action == ACTION_HELP

    def test_no_key_produces_no_command(self):
        device, _ = self._device()
        event = device.poll(-1)
        assert event.jog is None and event.action == ""

    def test_number_keys_select_joints_in_controller_order(self):
        device, jog = self._device()
        for index, key in enumerate("123456"):
            device.poll(ord(key))
            assert jog.selected_joint == JOINT_NAMES[index]

    def test_brackets_jog_the_selected_joint(self):
        device, jog = self._device()
        jog.select_joint("wrist_1")
        assert device.poll(ord("]")).jog.joint_delta_rad > 0
        assert device.poll(ord("[")).jog.joint_delta_rad < 0

    def test_tab_switches_space(self):
        device, jog = self._device()
        assert device.poll(9).action == ACTION_TOGGLE_SPACE
        assert jog.space == CARTESIAN_SPACE

    @pytest.mark.parametrize("key,axis,sign", [
        ("w", 0, +1), ("s", 0, -1), ("a", 1, +1),
        ("d", 1, -1), ("r", 2, +1), ("f", 2, -1)])
    def test_cartesian_translation_keys(self, key, axis, sign):
        device, jog = self._device()
        jog.space = CARTESIAN_SPACE
        command = device.poll(ord(key)).jog
        assert command is not None
        assert math.copysign(1, command.translation_m[axis]) == sign
        assert all(abs(command.translation_m[other]) < 1e-12
                   for other in range(3) if other != axis)

    def test_qe_are_yaw_in_cartesian_mode(self):
        device, jog = self._device()
        jog.space = CARTESIAN_SPACE
        assert device.poll(ord("e")).jog.rotation_rad[2] > 0
        # 'q' is quit even in cartesian mode; that is deliberate and documented.
        assert device.poll(ord("q")).action == ACTION_QUIT

    def test_arrow_keys_work_in_both_gtk_and_qt_encodings(self):
        device, jog = self._device()
        jog.space = CARTESIAN_SPACE
        for code in (65362, 2490368):
            assert device.poll(code).jog.rotation_rad[1] > 0

    def test_step_size_keys(self):
        device, jog = self._device()
        device.poll(ord("-"))
        assert jog.step_mode == "fine"
        device.poll(ord("="))
        assert jog.step_mode == "normal"

    def test_p_holds_motion_and_blocks_jogging(self):
        device, jog = self._device()
        device.poll(ord("p"))
        assert device.motion_permitted is False
        event = device.poll(ord("]"))
        assert event.jog is None
        assert "HELD" in event.message
        device.poll(ord("p"))
        assert device.poll(ord("]")).jog is not None

    def test_help_lines_mention_escape(self):
        device, _ = self._device()
        assert any("ESC" in line for line in device.help_lines())


class FakeJoystick:
    def __init__(self, axes=None, buttons=None, hats=None):
        self.axes = list(axes or [0.0] * 6)
        self.buttons = list(buttons or [False] * 8)
        self.hats = list(hats or [(0, 0)])

    def get_name(self): return "Fake Xbox Pad"
    def get_numaxes(self): return len(self.axes)
    def get_numbuttons(self): return len(self.buttons)
    def get_numhats(self): return len(self.hats)
    def get_axis(self, i): return self.axes[i]
    def get_button(self, i): return bool(self.buttons[i])
    def get_hat(self, i): return self.hats[i]
    def init(self): pass


class FakePygame:
    class event:
        @staticmethod
        def pump(): pass


def gamepad(axes=None, buttons=None, hats=None):
    jog = JogState(config())
    device = GamepadInput(jog, config()["gamepad"])
    device.joystick = FakeJoystick(axes, buttons, hats)
    device._pygame = FakePygame()
    device.connected = True
    return device, jog


class TestGamepadDeadman:
    def test_no_motion_without_the_deadman(self):
        """The single most important gamepad property."""
        device, _ = gamepad(axes=[1.0, 0, -1, 0, 0, -1])
        event = device.poll()
        assert event.jog is None
        assert event.motion_permitted is False
        assert "deadman" in event.message.lower()

    def test_motion_is_permitted_while_the_deadman_is_held(self):
        buttons = [False] * 8
        buttons[4] = True                    # LB
        device, jog = gamepad(axes=[1.0, 0, -1, 0, 0, -1], buttons=buttons)
        jog.space = CARTESIAN_SPACE
        event = device.poll()
        assert event.motion_permitted is True
        assert event.jog is not None

    def test_releasing_the_deadman_stops_motion(self):
        buttons = [False] * 8
        buttons[4] = True
        device, jog = gamepad(axes=[1.0, 0, -1, 0, 0, -1], buttons=buttons)
        jog.space = CARTESIAN_SPACE
        device.poll()
        device.joystick.buttons[4] = False
        device._last_command = 0.0
        assert device.poll().jog is None

    def test_disconnected_pad_blocks_motion(self):
        device, _ = gamepad()
        device.connected = False
        event = device.poll()
        assert event.motion_permitted is False
        assert "disconnect" in event.message.lower()

    def test_escape_still_works_from_the_window(self):
        device, _ = gamepad()
        assert device.poll(27).action == ACTION_STOP


class TestGamepadMapping:
    def _held(self, **kwargs):
        buttons = kwargs.pop("buttons", [False] * 8)
        buttons[4] = True
        return gamepad(buttons=buttons, **kwargs)

    def test_deadzone_suppresses_stick_noise(self):
        device, jog = self._held(axes=[0.05, 0.05, -1, 0, 0, -1])
        jog.space = CARTESIAN_SPACE
        assert device.poll().jog is None

    def test_response_curve_gives_fine_control_near_centre(self):
        device, _ = gamepad()
        assert abs(device._curve(0.5)) < 0.5
        assert device._curve(1.0) == pytest.approx(1.0)
        assert device._curve(0.05) == 0.0

    def test_curve_preserves_sign(self):
        device, _ = gamepad()
        assert device._curve(-0.8) < 0

    def test_a_button_records(self):
        buttons = [False] * 8
        buttons[0] = True
        device, _ = gamepad(buttons=buttons)
        assert device.poll().action == ACTION_RECORD

    def test_button_actions_do_not_repeat_while_held(self):
        buttons = [False] * 8
        buttons[0] = True
        device, _ = gamepad(buttons=buttons)
        assert device.poll().action == ACTION_RECORD
        assert device.poll().action != ACTION_RECORD      # rising edge only

    def test_start_quits(self):
        buttons = [False] * 8
        buttons[7] = True
        device, _ = gamepad(buttons=buttons)
        assert device.poll().action == ACTION_QUIT

    def test_x_selects_fine_mode(self):
        buttons = [False] * 8
        buttons[2] = True
        device, jog = gamepad(buttons=buttons)
        device.poll()
        assert jog.step_mode == "fine"

    def test_triggers_drive_z(self):
        device, jog = self._held(axes=[0, 0, -1.0, 0, 0, 1.0])
        jog.space = CARTESIAN_SPACE
        command = device.poll().jog
        assert command.translation_m[2] > 0        # RT pressed -> +Z

    def test_rate_limiting_caps_command_frequency(self):
        device, jog = self._held(axes=[1.0, 0, -1, 0, 0, -1])
        jog.space = CARTESIAN_SPACE
        assert device.poll().jog is not None
        assert device.poll().jog is None          # rate limited

    def test_joint_mode_moves_the_selected_joint(self):
        device, jog = self._held(axes=[0.9, 0, -1, 0, 0, -1])
        jog.space = JOINT_SPACE
        jog.select_joint("wrist_2")
        command = device.poll().jog
        assert command.joint == "wrist_2"
        assert command.joint_delta_rad > 0

    def test_dpad_walks_the_joint_priority_order(self):
        buttons = [False] * 8
        buttons[4] = True
        device, jog = gamepad(buttons=buttons, hats=[(0, -1)])
        jog.space = JOINT_SPACE
        jog.select_joint("wrist_3")
        device.poll()
        assert jog.selected_joint == "wrist_2"

    def test_help_lines_lead_with_the_deadman(self):
        device, _ = gamepad()
        assert "DEADMAN" in device.help_lines()[0]


class TestApplyJog:
    class FakeRobot:
        def __init__(self):
            self.calls = []

        def read_state(self): return "state"

        def jog_joint(self, joint, delta):
            self.calls.append(("joint", joint, delta))
            return "moved"

        def jog_tcp(self, translation, rotation_rad, frame):
            self.calls.append(("tcp", translation, rotation_rad, frame))
            return "moved"

    def test_joint_command_is_routed_to_jog_joint(self):
        robot = self.FakeRobot()
        subject = state()
        apply_jog(robot, subject.joint_command(+1))
        assert robot.calls[0][0] == "joint"

    def test_cartesian_command_is_routed_to_jog_tcp(self):
        robot = self.FakeRobot()
        subject = state()
        apply_jog(robot, subject.cartesian_command(translation_axis=0,
                                                   translation_direction=1))
        assert robot.calls[0][0] == "tcp"

    def test_empty_command_sends_nothing(self):
        robot = self.FakeRobot()
        apply_jog(robot, state().analog_command((0, 0, 0), (0, 0, 0)))
        assert robot.calls == []
