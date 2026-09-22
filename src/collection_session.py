"""The interactive session loop: live board preview + jogging + recording.

Shared by keyboard_teleop_calibration.py, gamepad_teleop_calibration.py and
collect_waypoints.py, so all three behave identically and there is exactly one
place where a record decision is made.

NOTHING HERE MOVES THE ROBOT ON ITS OWN. Every motion originates from an
operator input event, is bounded to one configured step, and is validated by
the safety envelope before it is sent.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

import ui_overlay as ui
from calibration_utils import JOINT_NAMES, timestamp_utc
from input_devices import (ACTION_CANCEL, ACTION_HELP, ACTION_QUIT,
                           ACTION_RECORD, ACTION_STOP, ACTION_TOGGLE_FRAME,
                           ACTION_TOGGLE_MOTION, ACTION_TOGGLE_SPACE,
                           ACTION_UNDO)
from jog_controller import apply_jog
from pose_diversity import JOINT_DISPLAY, PoseDiversityAnalyzer
from safety import SafetyError
from waypoint_recorder import RecordingRejected

LOGGER = logging.getLogger(__name__)


@dataclass
class SessionResult:
    recorded: int
    stopped_reason: str
    exit_code: int


class CollectionSession:
    """Runs the live loop. Recording is optional, so the teleop scripts can
    use the same loop purely for framing the board."""

    def __init__(self, camera, detector, robot, device, jog_state, analyzer,
                 recorder, config: Mapping[str, Any], camera_matrix, dist_coeffs,
                 target_count: int, allow_recording: bool = True,
                 logger=None):
        self.camera = camera
        self.detector = detector
        self.robot = robot
        self.device = device
        self.jog_state = jog_state
        self.analyzer: PoseDiversityAnalyzer = analyzer
        self.recorder = recorder
        self.config = config
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.target_count = target_count
        self.allow_recording = allow_recording
        self.logger = logger or LOGGER

        self.message = ""
        self.message_until = 0.0
        self.message_color = ui.WHITE
        self.show_help = False
        self.frame_rate = ui.FrameRate()
        self.recorded = 0
        self.rejected_attempts = 0

    # -- messaging ---------------------------------------------------------

    def notify(self, message: str, color=ui.WHITE, seconds: float = 2.5) -> None:
        self.message = message
        self.message_color = color
        self.message_until = time.time() + seconds
        if message:
            self.logger.info("%s", message)

    # -- main loop ---------------------------------------------------------

    def run(self, window: str) -> SessionResult:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        reason = "finished"
        exit_code = 0
        try:
            while True:
                if self.recorded >= self.target_count and self.allow_recording:
                    self.notify(f"TARGET REACHED: {self.recorded} waypoints",
                                ui.GREEN, 4.0)

                try:
                    frame = self.camera.read(flush=0)
                except Exception as exc:
                    # A camera failure must never leave the robot moving.
                    self.logger.error("Camera failure: %s", exc)
                    self._safe_stop()
                    return SessionResult(self.recorded, f"camera failure: {exc}", 1)

                detection = self.detector.process(
                    frame.image, self.camera_matrix, self.dist_coeffs,
                    require_pose=True)
                state = self.robot.read_state()
                if not state.connected:
                    self.logger.error("Lost the robot connection")
                    self._safe_stop()
                    return SessionResult(self.recorded, "robot connection lost", 1)

                evaluation = None
                if state.q is not None and state.tcp is not None:
                    evaluation = self.analyzer.evaluate_candidate(state.q, state.tcp)

                canvas = self._render(frame, detection, state, evaluation)
                cv2.imshow(window, ui.fit_to_screen(canvas))
                key = cv2.waitKeyEx(1)

                event = self.device.poll(key)
                if event.message:
                    self.notify(event.message, ui.GREY, 1.5)

                if event.action == ACTION_STOP:
                    self._safe_stop()
                    reason = "operator pressed ESC"
                    exit_code = 0
                    break
                if event.action == ACTION_QUIT:
                    reason = "operator quit"
                    break
                if event.action == ACTION_HELP:
                    self.show_help = not self.show_help
                elif event.action == ACTION_UNDO:
                    self._undo()
                elif event.action == ACTION_CANCEL:
                    self.notify("cancelled", ui.GREY, 1.0)
                elif event.action == ACTION_RECORD:
                    self._attempt_record(detection, state, evaluation)
                elif event.action in (ACTION_TOGGLE_SPACE, ACTION_TOGGLE_FRAME,
                                      ACTION_TOGGLE_MOTION):
                    pass                     # state already updated by the device

                if event.jog is not None and event.motion_permitted:
                    self._apply_jog(event)

        except KeyboardInterrupt:
            self._safe_stop()
            reason = "interrupted"
        finally:
            cv2.destroyWindow(window)
        return SessionResult(self.recorded, reason, exit_code)

    # -- actions -----------------------------------------------------------

    def _apply_jog(self, event) -> None:
        if not self.robot.motion_enabled:
            self.notify("motion is not enabled -- this session is read-only",
                        ui.AMBER, 2.0)
            return
        try:
            apply_jog(self.robot, event.jog)
        except SafetyError as exc:
            # Refusing a move is normal and expected at the envelope edge.
            self.notify(f"BLOCKED: {exc}", ui.RED, 3.5)
            self.logger.warning("Jog rejected: %s", exc)

    def _attempt_record(self, detection, state, evaluation) -> None:
        if not self.allow_recording or self.recorder is None:
            self.notify("this session does not record waypoints", ui.AMBER, 2.0)
            return
        if not detection.valid:
            self.notify("WAYPOINT NOT RECORDABLE: "
                        + (detection.reasons[0] if detection.reasons else "invalid"),
                        ui.RED, 3.0)
            self.rejected_attempts += 1
            return

        number = self.recorded + 1
        try:
            record = self.recorder.record(number)
        except RecordingRejected as exc:
            self.rejected_attempts += 1
            self.notify(f"NOT RECORDED: {exc}", ui.RED, 3.5)
            self.logger.info("Record attempt %d rejected: %s", number, exc)
            return
        except Exception as exc:
            self.rejected_attempts += 1
            self.notify(f"RECORD FAILED: {exc}", ui.RED, 4.0)
            self.logger.exception("Unexpected failure recording waypoint %d", number)
            return

        self.recorded += 1
        self.analyzer.add(record.number, record.actual_q, record.actual_tcp)
        note = ""
        if evaluation and evaluation.status != "GOOD":
            note = f"  ({evaluation.status.lower()})"
        self.notify(f"RECORDED {record.name}  "
                    f"{self.recorded}/{self.target_count}{note}", ui.GREEN, 2.0)

    def _undo(self) -> None:
        if self.recorder is None or not self.recorder.records:
            self.notify("nothing to undo", ui.GREY, 1.5)
            return
        name = self.recorder.undo_last()
        self.analyzer.remove_last()
        self.recorded = max(0, self.recorded - 1)
        self.notify(f"REMOVED {name}", ui.AMBER, 2.0)

    def _safe_stop(self) -> None:
        try:
            self.robot.emergency_software_stop()
        except Exception as exc:
            self.logger.error("Stop failed: %s", exc)

    # -- rendering ---------------------------------------------------------

    def _render(self, frame, detection, state, evaluation) -> np.ndarray:
        canvas = self.detector.annotate(frame.image, detection,
                                        self.camera_matrix, self.dist_coeffs)
        fps = self.frame_rate.tick(time.time())

        ui.border_warning(canvas, detection.border_margin_px,
                          self.detector.min_border_margin)
        if detection.center_px:
            ui.draw_reticle(canvas, detection.center_px)

        recommendation = self.analyzer.recommend()
        lines = self._status_lines(detection, state, evaluation, recommendation, fps)
        ui.panel(canvas, lines, width=470)

        if self.allow_recording:
            ui.progress_bar(canvas, self.recorded, self.target_count,
                            (10, canvas.shape[0] - 46), width=280,
                            color=ui.GREEN if self.recorded >= self.target_count
                            else ui.BLUE,
                            label=f"WAYPOINTS {self.recorded} / {self.target_count}")

        self._draw_deltas(canvas, state)

        if detection.tags_detected and not detection.valid:
            ui.banner(canvas, "WAYPOINT NOT RECORDABLE", ui.RED, 0.75)
        elif detection.border_margin_px < self.detector.min_border_margin * 2:
            ui.banner(canvas, "APRILTAG GRID VISIBILITY DECREASING", ui.AMBER, 0.62)

        if self.device.requires_deadman and not self.device.motion_permitted:
            ui.banner(canvas, "DEADMAN RELEASED -- ROBOT WILL NOT MOVE",
                      ui.AMBER, 0.62, y=canvas.shape[0] - 90)

        if time.time() < self.message_until and self.message:
            ui.banner(canvas, self.message, self.message_color, 0.7,
                      y=canvas.shape[0] - 120)

        ui.footer(canvas,
                  "ENTER record   U undo   H help   ESC EMERGENCY STOP",
                  f"{self.device.name}   {'MOTION ON' if self.robot.motion_enabled else 'READ-ONLY'}")
        if self.show_help:
            self._draw_help(canvas)
        return canvas

    def _status_lines(self, detection, state, evaluation, recommendation,
                      fps) -> list:
        camera_line = f"CAMERA {self.camera.name}  [{self.camera.serial}]"
        lines = [
            (camera_line, ui.WHITE, 0.55),
            (f"{frame_size(detection)}   {fps:4.1f} fps"
             f"   {state.mode_text()}", ui.GREY, 0.45),
            ("", ui.WHITE, 0.25),
            ("APRILTAG: VALID" if detection.valid else "APRILTAG: INVALID",
             ui.GREEN if detection.valid else ui.RED, 0.75),
            (f"Tags {detection.tags_detected}/{self.detector.spec.tag_count}"
             f"   Corners {detection.corners_detected}"
             f"   Margin {detection.border_margin_px:.0f} px", ui.WHITE, 0.46),
        ]
        if detection.has_pose:
            lines.append((f"Board {detection.distance_m * 1000:.0f} mm"
                          f"   tilt {detection.tilt_deg:.1f} deg"
                          f"   PnP {detection.pnp_reprojection_px:.3f} px",
                          ui.GREY, 0.44))
        for reason in detection.reasons[:2]:
            lines.append((f"! {reason}", ui.RED, 0.44))

        if evaluation is not None:
            lines += [
                ("", ui.WHITE, 0.25),
                ("NEW POSE QUALITY", ui.WHITE, 0.5),
                (f"  nearest translation {fmt_inf(evaluation.nearest_translation_mm)} mm",
                 ui.GREY, 0.44),
                (f"  nearest rotation    {fmt_inf(evaluation.nearest_rotation_deg)} deg",
                 ui.GREY, 0.44),
                (f"  {evaluation.status}: {evaluation.reason}",
                 ui.color_for(evaluation.status), 0.46),
            ]
            for suggestion in evaluation.suggestions[:2]:
                lines.append((f"  -> {suggestion}", ui.AMBER, 0.43))

        diversity = recommendation["diversity"]
        lines += [
            ("", ui.WHITE, 0.25),
            (f"STAGE: {recommendation['stage']}", ui.WHITE, 0.5),
            ("  " + "  ".join(
                f"{axis.upper()}:{diversity['translation'][axis]['rating'][:4]}"
                for axis in "xyz") + "  (translation)", ui.GREY, 0.44),
            ("  " + "  ".join(
                f"R{index + 1}:{diversity['rotation'][f'axis_{index + 1}']['rating'][:4]}"
                for index in range(3)) + "  (rotation)", ui.GREY, 0.44),
        ]
        for chunk in wrap(recommendation["message"], 54)[:3]:
            lines.append((f"  {chunk}", ui.AMBER, 0.43))
        lines.append((f"  BASE MOVEMENT: {recommendation['base_movement']}",
                      ui.GREY, 0.44))

        lines.append(("", ui.WHITE, 0.25))
        for line in self.jog_state.status_lines():
            lines.append((line, ui.BLUE, 0.48))
        return lines

    def _draw_deltas(self, canvas, state) -> None:
        """Per-joint deviation from the calibration centre (section I2)."""
        deltas = self.robot.deltas_from_center(state)
        if not deltas:
            return
        lines = [("DELTA FROM CALIBRATION CENTRE", ui.WHITE, 0.5)]
        for name, value in zip(JOINT_NAMES, deltas["joint_delta_deg"]):
            headroom = self.robot.envelope.joints.max_deviation_from_center[
                JOINT_NAMES.index(name)]
            limit_deg = np.degrees(headroom)
            fraction = abs(value) / limit_deg if np.isfinite(limit_deg) else 0.0
            color = (ui.RED if fraction > 0.9 else
                     ui.AMBER if fraction > 0.7 else ui.GREY)
            marker = " <<" if name == self.jog_state.selected_joint else ""
            lines.append((f"  {JOINT_DISPLAY[name]:<14} {value:+7.2f} deg"
                          f"  / {limit_deg:5.1f}{marker}", color, 0.45))
        lines += [
            ("", ui.WHITE, 0.2),
            (f"  TCP translation {deltas['tcp_translation_delta_mm']:7.1f} mm",
             ui.GREY, 0.46),
            (f"  TCP rotation    {deltas['tcp_rotation_delta_deg']:7.2f} deg",
             ui.GREY, 0.46),
        ]
        ui.panel(canvas, lines, origin=(canvas.shape[1] - 440, 10), width=430)

    def _draw_help(self, canvas) -> None:
        lines = [("CONTROLS", ui.WHITE, 0.6)]
        lines += [(line, ui.GREY, 0.45) for line in self.device.help_lines()]
        ui.panel(canvas, lines,
                 origin=(20, canvas.shape[0] - 40 - 26 * (len(lines) + 1)),
                 width=560, alpha=0.8)


def frame_size(detection) -> str:
    return f"{detection.image_size[0]} x {detection.image_size[1]}"


def fmt_inf(value: float) -> str:
    return "  inf" if not np.isfinite(value) else f"{value:6.1f}"


def wrap(message: str, width: int) -> list[str]:
    words, lines, current = message.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines
