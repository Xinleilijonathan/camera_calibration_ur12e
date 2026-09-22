"""Jog state and command construction. Decides WHAT to move, never moves it.

This module builds a validated jog request from an operator input event. The
actual command is issued by robot_interface, which re-checks everything
against the safety envelope. Nothing here talks to hardware.

Two jog spaces:

  JOINT      one named joint at a time, with per-joint step sizes. This is the
             mode that matters for calibration: distal joints give orientation
             diversity for very little whole-arm travel, and moving exactly one
             joint keeps the motion predictable and easy to undo.

  CARTESIAN  TCP translation and rotation in the base or tool frame, for
             framing the board.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from calibration_utils import JOINT_NAMES, JOINT_PRIORITY
from pose_diversity import JOINT_DISPLAY
from safety import SafetyError

LOGGER = logging.getLogger(__name__)

JOINT_SPACE = "joint"
CARTESIAN_SPACE = "cartesian"
STEP_MODES = ("fine", "normal", "coarse")


@dataclass
class JogCommand:
    """One requested nudge. Advisory until robot_interface validates it."""
    space: str
    joint: str | None = None
    joint_delta_rad: float = 0.0
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    frame: str = "base"
    label: str = ""

    @property
    def is_empty(self) -> bool:
        return (abs(self.joint_delta_rad) < 1e-12
                and not any(abs(v) > 1e-12 for v in self.translation_m)
                and not any(abs(v) > 1e-12 for v in self.rotation_rad))


class JogState:
    """Current jog space, step size and selected joint.

    Deliberately explicit: the display always names the joint that will move
    and by how much, because an ambiguous jog on a real arm is how people get
    surprised (section G2).
    """

    def __init__(self, config: Mapping[str, Any]):
        jog = dict(config.get("jog") or {})
        self.cartesian_steps = dict(jog.get("cartesian") or {})
        self.joint_steps = dict(jog.get("joint_deg") or {})
        self.space = str(jog.get("default_jog_space", JOINT_SPACE))
        if self.space not in (JOINT_SPACE, CARTESIAN_SPACE):
            self.space = JOINT_SPACE
        self.step_mode = str(jog.get("default_mode", "normal"))
        if self.step_mode not in STEP_MODES:
            self.step_mode = "normal"
        # Start on the most-preferred calibration joint, wrist 3.
        self.selected_joint = JOINT_PRIORITY[0]
        self.frame = "base"

    # -- mode switching ----------------------------------------------------

    def toggle_space(self) -> str:
        self.space = (CARTESIAN_SPACE if self.space == JOINT_SPACE else JOINT_SPACE)
        return self.space

    def set_step_mode(self, mode: str) -> str:
        if mode in STEP_MODES:
            self.step_mode = mode
        return self.step_mode

    def cycle_step_mode(self, direction: int) -> str:
        index = STEP_MODES.index(self.step_mode)
        self.step_mode = STEP_MODES[max(0, min(len(STEP_MODES) - 1, index + direction))]
        return self.step_mode

    def select_joint(self, joint: str | int) -> str:
        if isinstance(joint, int):
            if not 0 <= joint < len(JOINT_NAMES):
                raise SafetyError(f"No such joint index {joint}")
            joint = JOINT_NAMES[joint]
        if joint not in JOINT_NAMES:
            raise SafetyError(f"No such joint {joint!r}")
        self.selected_joint = joint
        return joint

    def toggle_frame(self) -> str:
        self.frame = "tool" if self.frame == "base" else "base"
        return self.frame

    # -- step sizes --------------------------------------------------------

    def joint_step_rad(self, joint: str | None = None) -> float:
        """Step for a joint, in radians, from the configured degree table."""
        joint = joint or self.selected_joint
        entry = self.joint_steps.get(joint) or {}
        degrees = float(entry.get(self.step_mode, {"fine": 0.5, "normal": 2.0,
                                                   "coarse": 5.0}[self.step_mode]))
        return math.radians(degrees)

    def joint_step_deg(self, joint: str | None = None) -> float:
        return math.degrees(self.joint_step_rad(joint))

    def translation_step_m(self) -> float:
        table = self.cartesian_steps.get("translation_m") or {}
        return float(table.get(self.step_mode,
                               {"fine": 0.001, "normal": 0.005,
                                "coarse": 0.010}[self.step_mode]))

    def rotation_step_rad(self) -> float:
        table = self.cartesian_steps.get("rotation_deg") or {}
        return math.radians(float(table.get(
            self.step_mode, {"fine": 1.0, "normal": 3.0, "coarse": 5.0}[self.step_mode])))

    # -- command construction ---------------------------------------------

    def joint_command(self, direction: int,
                      joint: str | None = None,
                      scale: float = 1.0) -> JogCommand:
        """Move the selected joint by one step in `direction` (+1 / -1)."""
        joint = joint or self.selected_joint
        delta = self.joint_step_rad(joint) * float(direction) * float(scale)
        return JogCommand(
            space=JOINT_SPACE, joint=joint, joint_delta_rad=delta,
            label=f"{JOINT_DISPLAY[joint]} {math.degrees(delta):+.2f} deg")

    def cartesian_command(self, translation_axis: int = -1,
                          translation_direction: int = 0,
                          rotation_axis: int = -1,
                          rotation_direction: int = 0,
                          scale: float = 1.0) -> JogCommand:
        """Move the TCP one step along/about one axis."""
        translation = [0.0, 0.0, 0.0]
        rotation = [0.0, 0.0, 0.0]
        label_parts = []
        if 0 <= translation_axis < 3 and translation_direction:
            step = self.translation_step_m() * translation_direction * scale
            translation[translation_axis] = step
            label_parts.append(f"{'XYZ'[translation_axis]} {step * 1000:+.1f} mm")
        if 0 <= rotation_axis < 3 and rotation_direction:
            step = self.rotation_step_rad() * rotation_direction * scale
            rotation[rotation_axis] = step
            label_parts.append(
                f"{['roll', 'pitch', 'yaw'][rotation_axis]} "
                f"{math.degrees(step):+.2f} deg")
        return JogCommand(
            space=CARTESIAN_SPACE, translation_m=tuple(translation),
            rotation_rad=tuple(rotation), frame=self.frame,
            label=" ".join(label_parts))

    def analog_command(self, translation: Sequence[float],
                       rotation: Sequence[float]) -> JogCommand:
        """Build a jog from continuous stick values already in [-1, 1].

        Magnitudes scale the configured step, so a gentle stick gives a gentle
        nudge. There is no path from a stick flick to a full-speed move.
        """
        translation_step = self.translation_step_m()
        rotation_step = self.rotation_step_rad()
        translation_vector = tuple(float(v) * translation_step for v in translation)
        rotation_vector = tuple(float(v) * rotation_step for v in rotation)
        label = (f"dXYZ {translation_vector[0] * 1000:+.1f},"
                 f"{translation_vector[1] * 1000:+.1f},"
                 f"{translation_vector[2] * 1000:+.1f} mm")
        return JogCommand(
            space=CARTESIAN_SPACE, translation_m=translation_vector,
            rotation_rad=rotation_vector, frame=self.frame, label=label)

    # -- reporting ---------------------------------------------------------

    def describe(self) -> dict:
        return {
            "space": self.space,
            "step_mode": self.step_mode,
            "selected_joint": self.selected_joint,
            "selected_joint_label": JOINT_DISPLAY[self.selected_joint],
            "joint_step_deg": self.joint_step_deg(),
            "translation_step_mm": self.translation_step_m() * 1000.0,
            "rotation_step_deg": math.degrees(self.rotation_step_rad()),
            "frame": self.frame,
        }

    def status_lines(self) -> list[str]:
        """Compact HUD lines describing exactly what the next key press does."""
        if self.space == JOINT_SPACE:
            return [
                f"JOINT JOG   step {self.step_mode.upper()}",
                f"SELECTED: {JOINT_DISPLAY[self.selected_joint]}",
                f"Each press moves it {self.joint_step_deg():+.2f} deg",
            ]
        return [
            f"CARTESIAN JOG ({self.frame})   step {self.step_mode.upper()}",
            f"Translate {self.translation_step_m() * 1000:.1f} mm per press",
            f"Rotate {math.degrees(self.rotation_step_rad()):.1f} deg per press",
        ]


def apply_jog(robot, command: JogCommand):
    """Send a jog through robot_interface, which re-validates everything.

    Returns the post-move RobotState. Raises SafetyError if the envelope
    rejects the move; the caller reports it and carries on without moving.
    """
    if command.is_empty:
        return robot.read_state()
    if command.space == JOINT_SPACE:
        return robot.jog_joint(command.joint, command.joint_delta_rad)
    return robot.jog_tcp(translation=command.translation_m,
                         rotation_rad=command.rotation_rad,
                         frame=command.frame)
