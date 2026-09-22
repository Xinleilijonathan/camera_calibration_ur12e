"""Operator input: keyboard (via the OpenCV window) and gamepad (via pygame).

Both devices produce the same `InputEvent`, so the collection session does not
care which one you are holding.

SAFETY PROPERTIES
-----------------
Keyboard: keys are read from the preview window, so they only register while
that window has focus. Clicking away stops input reaching the robot.

Gamepad: a DEADMAN button must be held continuously. Motion is permitted only
while it is down AND a fresh sample arrived recently -- so an unplugged or
frozen controller reads as "released", not as "still held". Stick magnitude
scales the step through a response curve, so there is no path from a flick to
a full-speed move.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from jog_controller import CARTESIAN_SPACE, JOINT_SPACE, JogCommand, JogState

LOGGER = logging.getLogger(__name__)

# Actions the session understands, independent of the device that produced them.
ACTION_NONE = ""
ACTION_RECORD = "record"
ACTION_UNDO = "undo"
ACTION_QUIT = "quit"
ACTION_STOP = "emergency_stop"
ACTION_HELP = "help"
ACTION_TOGGLE_SPACE = "toggle_space"
ACTION_TOGGLE_FRAME = "toggle_frame"
ACTION_TOGGLE_MOTION = "toggle_motion"
ACTION_CANCEL = "cancel"


@dataclass
class InputEvent:
    """One poll's worth of operator intent."""
    action: str = ACTION_NONE
    jog: JogCommand | None = None
    motion_permitted: bool = True
    message: str = ""
    notes: dict = field(default_factory=dict)


# Arrow-key codes from cv2.waitKeyEx. GTK and Qt builds report different
# values, so both sets are accepted rather than guessed at.
ARROW_LEFT = {65361, 2424832, 81}
ARROW_UP = {65362, 2490368, 82}
ARROW_RIGHT = {65363, 2555904, 83}
ARROW_DOWN = {65364, 2621440, 84}


class KeyboardInput:
    """Keyboard jog, read from the OpenCV preview window.

    Mapping (section G1/G2):
      Cartesian  W/S +/-X   A/D +/-Y   R/F +/-Z
                 arrows pitch/roll     Q/E yaw
      Joint      1..6 select joint     [ / ] decrease / increase
      Both       TAB toggle space      - / = step size
                 ENTER record          U undo
                 ESC emergency stop    H help
    """

    name = "keyboard"
    requires_deadman = False

    def __init__(self, state: JogState, config: Mapping[str, Any] | None = None):
        self.state = state
        self.motion_permitted = True

    def poll(self, key: int) -> InputEvent:
        """Translate one key code into an event. `key` is from cv2.waitKeyEx."""
        if key in (-1, 255):
            return InputEvent(motion_permitted=self.motion_permitted)

        # --- global actions ---
        if key == 27:                                   # ESC
            return InputEvent(action=ACTION_STOP,
                              message="ESC: stopping commanded motion")
        if key in (ord("q"), ord("Q")):
            return InputEvent(action=ACTION_QUIT)
        if key in (13, 10):                             # ENTER
            return InputEvent(action=ACTION_RECORD)
        if key in (ord("u"), ord("U")):
            return InputEvent(action=ACTION_UNDO)
        if key in (ord("h"), ord("H")):
            return InputEvent(action=ACTION_HELP)
        if key == 9:                                    # TAB
            return InputEvent(action=ACTION_TOGGLE_SPACE,
                              message=f"jog space -> {self.state.toggle_space()}")
        if key in (ord("g"), ord("G")):
            return InputEvent(action=ACTION_TOGGLE_FRAME,
                              message=f"frame -> {self.state.toggle_frame()}")
        if key in (ord("p"), ord("P")):
            self.motion_permitted = not self.motion_permitted
            return InputEvent(
                action=ACTION_TOGGLE_MOTION,
                motion_permitted=self.motion_permitted,
                message=("motion ARMED" if self.motion_permitted
                         else "motion HELD -- keys will not move the robot"))

        # --- step size ---
        if key in (ord("-"), ord("_")):
            return InputEvent(message=f"step -> {self.state.cycle_step_mode(-1)}",
                              motion_permitted=self.motion_permitted)
        if key in (ord("="), ord("+")):
            return InputEvent(message=f"step -> {self.state.cycle_step_mode(1)}",
                              motion_permitted=self.motion_permitted)

        # --- joint selection: 1..6 ---
        if ord("1") <= key <= ord("6"):
            joint = self.state.select_joint(key - ord("1"))
            self.state.space = JOINT_SPACE
            return InputEvent(message=f"selected {joint}",
                              motion_permitted=self.motion_permitted)

        if not self.motion_permitted:
            return InputEvent(motion_permitted=False,
                              message="motion is HELD; press P to arm")

        # --- joint jog ---
        if key == ord("["):
            return self._jog(self.state.joint_command(-1))
        if key == ord("]"):
            return self._jog(self.state.joint_command(+1))

        # --- cartesian jog ---
        if self.state.space == CARTESIAN_SPACE:
            translation = {ord("w"): (0, +1), ord("s"): (0, -1),
                           ord("a"): (1, +1), ord("d"): (1, -1),
                           ord("r"): (2, +1), ord("f"): (2, -1)}
            lowered = key | 0x20 if ord("A") <= key <= ord("Z") else key
            if lowered in translation:
                axis, direction = translation[lowered]
                return self._jog(self.state.cartesian_command(
                    translation_axis=axis, translation_direction=direction))
            if lowered == ord("e"):
                return self._jog(self.state.cartesian_command(
                    rotation_axis=2, rotation_direction=+1))
            if lowered == ord("q"):
                return self._jog(self.state.cartesian_command(
                    rotation_axis=2, rotation_direction=-1))
            if key in ARROW_UP:
                return self._jog(self.state.cartesian_command(
                    rotation_axis=1, rotation_direction=+1))
            if key in ARROW_DOWN:
                return self._jog(self.state.cartesian_command(
                    rotation_axis=1, rotation_direction=-1))
            if key in ARROW_LEFT:
                return self._jog(self.state.cartesian_command(
                    rotation_axis=0, rotation_direction=-1))
            if key in ARROW_RIGHT:
                return self._jog(self.state.cartesian_command(
                    rotation_axis=0, rotation_direction=+1))
        else:
            # In joint mode the arrows step the selected joint too, so the
            # hands-on-keyboard case does not require reaching for brackets.
            if key in ARROW_RIGHT or key in ARROW_UP:
                return self._jog(self.state.joint_command(+1))
            if key in ARROW_LEFT or key in ARROW_DOWN:
                return self._jog(self.state.joint_command(-1))

        return InputEvent(motion_permitted=self.motion_permitted)

    def _jog(self, command: JogCommand) -> InputEvent:
        return InputEvent(jog=command, motion_permitted=self.motion_permitted,
                          message=command.label)

    def help_lines(self) -> list[str]:
        return [
            "JOINT MODE   1-6 select joint   [ / ] move it   arrows also move it",
            "CARTESIAN    W/S +-X   A/D +-Y   R/F +-Z",
            "             arrows pitch/roll  Q/E yaw   G base/tool frame",
            "BOTH         TAB switch mode    - / = step size",
            "             ENTER record       U undo     P arm/hold motion",
            "             H help             Q quit     ESC EMERGENCY STOP",
        ]

    def close(self) -> None:
        pass


class GamepadInput:
    """Xbox-style controller through pygame, with a mandatory deadman."""

    name = "gamepad"
    requires_deadman = True

    def __init__(self, state: JogState, config: Mapping[str, Any] | None = None):
        self.state = state
        config = dict(config or {})
        self.deadzone = float(config.get("deadzone", 0.15))
        self.exponent = float(config.get("response_curve_exponent", 2.0))
        self.max_rate_hz = float(config.get("max_command_rate_hz", 5.0))
        self.require_deadman = bool(config.get("require_deadman", True))
        self.mapping = dict(config.get("mapping") or {})
        self.max_sample_age_s = float(config.get("deadman_max_sample_age_s", 0.15))

        self._last_command = 0.0
        self._last_sample = 0.0
        self._previous_buttons: dict[int, bool] = {}
        self.motion_permitted = False
        self.joystick = None
        self._pygame = None
        self.connected = False

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> "GamepadInput":
        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "pygame is not installed, so the gamepad cannot be used. "
                "pip install pygame, or use the keyboard script instead.") from exc
        self._pygame = pygame
        # No video surface: this process already owns an OpenCV window.
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise RuntimeError(
                "No gamepad detected.\n"
                "  Plug in the controller and check it appears as /dev/input/js0.\n"
                "  Bluetooth pads sometimes need a button press to wake up.")
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.connected = True
        LOGGER.info("Gamepad: %s (%d axes, %d buttons, %d hats)",
                    self.joystick.get_name(), self.joystick.get_numaxes(),
                    self.joystick.get_numbuttons(), self.joystick.get_numhats())
        return self

    def close(self) -> None:
        if self._pygame is not None:
            try:
                self._pygame.joystick.quit()
                self._pygame.quit()
            except Exception as exc:
                LOGGER.debug("pygame shutdown: %s", exc)
        self.connected = False

    def describe(self) -> str:
        if not self.joystick:
            return "no gamepad"
        return (f"{self.joystick.get_name()} "
                f"({self.joystick.get_numaxes()} axes, "
                f"{self.joystick.get_numbuttons()} buttons)")

    # -- helpers -----------------------------------------------------------

    def _button(self, key: str, default: int) -> bool:
        index = int(self.mapping.get(key, default))
        if not self.joystick or not 0 <= index < self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(index))

    def _pressed(self, key: str, default: int) -> bool:
        """Rising edge, so holding a button does not repeat the action."""
        index = int(self.mapping.get(key, default))
        now = self._button(key, default)
        was = self._previous_buttons.get(index, False)
        self._previous_buttons[index] = now
        return now and not was

    def _axis(self, key: str, default: int) -> float:
        index = int(self.mapping.get(key, default))
        if not self.joystick or not 0 <= index < self.joystick.get_numaxes():
            return 0.0
        return float(self.joystick.get_axis(index))

    def _curve(self, value: float) -> float:
        """Deadzone, then a power curve for fine control near centre."""
        magnitude = abs(value)
        if magnitude < self.deadzone:
            return 0.0
        scaled = (magnitude - self.deadzone) / (1.0 - self.deadzone)
        return math.copysign(scaled ** self.exponent, value)

    def _trigger(self, key: str, default: int) -> float:
        """Normalise a trigger to [0, 1] regardless of its resting convention."""
        raw = self._axis(key, default)
        if self.mapping.get("triggers_rest_at_minus_one", True):
            raw = (raw + 1.0) / 2.0
        return 0.0 if raw < self.deadzone else raw

    # -- polling -----------------------------------------------------------

    def poll(self, key: int = -1) -> InputEvent:
        """Read the controller once. `key` lets ESC still work from the window."""
        if key == 27:
            return InputEvent(action=ACTION_STOP, motion_permitted=False,
                              message="ESC: stopping commanded motion")
        if not self.connected or self.joystick is None:
            return InputEvent(motion_permitted=False,
                              message="gamepad disconnected -- motion blocked")

        try:
            self._pygame.event.pump()
            self._last_sample = time.monotonic()
        except Exception as exc:
            self.connected = False
            self.motion_permitted = False
            return InputEvent(motion_permitted=False,
                              message=f"gamepad read failed: {exc}")

        # Deadman. A stale sample counts as released, so a frozen or unplugged
        # controller can never leave motion enabled.
        held = self._button("deadman_button", 4)
        fresh = (time.monotonic() - self._last_sample) <= self.max_sample_age_s
        self.motion_permitted = bool(held and fresh) or not self.require_deadman

        if self._pressed("exit_button", 7):
            return InputEvent(action=ACTION_QUIT, motion_permitted=False)
        if self._pressed("record_button", 0):
            return InputEvent(action=ACTION_RECORD,
                              motion_permitted=self.motion_permitted)
        if self._pressed("cancel_button", 1):
            return InputEvent(action=ACTION_CANCEL,
                              motion_permitted=self.motion_permitted)
        if self._pressed("fine_button", 2):
            return InputEvent(message=f"step -> {self.state.set_step_mode('fine')}",
                              motion_permitted=self.motion_permitted)
        if self._pressed("coarse_button", 3):
            mode = "coarse" if self.state.step_mode == "normal" else "normal"
            return InputEvent(message=f"step -> {self.state.set_step_mode(mode)}",
                              motion_permitted=self.motion_permitted)
        if self._pressed("toggle_space_button", 6):
            return InputEvent(action=ACTION_TOGGLE_SPACE,
                              motion_permitted=self.motion_permitted,
                              message=f"jog space -> {self.state.toggle_space()}")

        if not self.motion_permitted:
            return InputEvent(motion_permitted=False,
                              message="HOLD LB (deadman) to move the robot")

        # Rate limit: one bounded moveJ/moveL at a time, never a stream.
        now = time.monotonic()
        if now - self._last_command < 1.0 / max(0.1, self.max_rate_hz):
            return InputEvent(motion_permitted=True)

        command = (self._joint_jog() if self.state.space == JOINT_SPACE
                   else self._cartesian_jog())
        if command is None or command.is_empty:
            return InputEvent(motion_permitted=True)
        self._last_command = now
        return InputEvent(jog=command, motion_permitted=True, message=command.label)

    def _cartesian_jog(self) -> JogCommand | None:
        x = self._curve(self._axis("axis_translate_x", 0))
        y = self._curve(self._axis("axis_translate_y", 1))
        if self.mapping.get("invert_translate_y", True):
            y = -y
        z = self._trigger("axis_trigger_right", 5) - self._trigger("axis_trigger_left", 2)
        roll = self._curve(self._axis("axis_rotate_roll", 3))
        pitch = self._curve(self._axis("axis_rotate_pitch", 4))
        if self.mapping.get("invert_rotate_pitch", True):
            pitch = -pitch
        yaw = 0.0
        if self.joystick.get_numhats() > 0:
            hat_x, _ = self.joystick.get_hat(0)
            yaw = float(hat_x)
        if not any(abs(v) > 1e-6 for v in (x, y, z, roll, pitch, yaw)):
            return None
        return self.state.analog_command((x, y, z), (roll, pitch, yaw))

    def _joint_jog(self) -> JogCommand | None:
        """Left stick horizontal moves the selected joint; D-pad selects it."""
        if self.joystick.get_numhats() > 0:
            _, hat_y = self.joystick.get_hat(0)
            if hat_y:
                from calibration_utils import JOINT_PRIORITY
                order = list(JOINT_PRIORITY)
                index = order.index(self.state.selected_joint)
                self.state.select_joint(
                    order[max(0, min(len(order) - 1, index - int(hat_y)))])
        value = self._curve(self._axis("axis_translate_x", 0))
        if abs(value) < 1e-6:
            return None
        return self.state.joint_command(
            1 if value > 0 else -1, scale=min(1.0, abs(value)))

    def help_lines(self) -> list[str]:
        return [
            "LB           DEADMAN -- hold it or nothing moves",
            "Left stick   joint mode: move selected joint",
            "             cartesian mode: X/Y translation",
            "Triggers     Z translation      Right stick: roll/pitch",
            "D-pad        up/down select joint, left/right yaw",
            "A record     B cancel     X fine     Y normal/coarse",
            "BACK switch mode          START quit      ESC EMERGENCY STOP",
        ]

    def probe_lines(self) -> list[str]:
        """Live axis/button state, for correcting the mapping."""
        if not self.joystick:
            return ["no gamepad"]
        self._pygame.event.pump()
        axes = [f"{i}:{self.joystick.get_axis(i):+.2f}"
                for i in range(self.joystick.get_numaxes())]
        buttons = [str(i) for i in range(self.joystick.get_numbuttons())
                   if self.joystick.get_button(i)]
        hats = [str(self.joystick.get_hat(i))
                for i in range(self.joystick.get_numhats())]
        return [
            f"axes    {'  '.join(axes)}",
            f"pressed {' '.join(buttons) if buttons else '(none)'}",
            f"hats    {' '.join(hats) if hats else '(none)'}",
        ]
