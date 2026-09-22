"""Shared on-screen HUD drawing for the live preview and collection tools.

Pure rendering over a NumPy image. Opens no camera, commands no robot.

The overlay exists to answer one question at a glance, while your hands are on
the controller and your eyes are on the arm: MAY I RECORD RIGHT NOW? Everything
else is secondary and drawn smaller.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX

# BGR palette. Kept deliberately small and high-contrast for a live video feed.
WHITE = (255, 255, 255)
GREY = (170, 170, 170)
GREEN = (80, 220, 80)
RED = (60, 60, 240)
AMBER = (0, 190, 255)
BLUE = (255, 200, 0)
MAGENTA = (255, 0, 255)

#: Status -> (colour, label). Used for the big VALID/INVALID banner.
STATUS_COLORS = {
    "GOOD": GREEN, "VALID": GREEN, "OK": GREEN, "SELECTED": GREEN,
    "MEDIUM": AMBER, "MARGINAL": AMBER, "WARN": AMBER, "LOW": AMBER,
    "BAD": RED, "INVALID": RED, "BLOCKED": RED, "REJECTED": RED,
}


def color_for(status: str) -> tuple:
    return STATUS_COLORS.get(str(status).upper(), WHITE)


def text(canvas: np.ndarray, message: str, origin: tuple[int, int],
         color=WHITE, scale: float = 0.55, thickness: int = 1,
         shadow: bool = True) -> None:
    """Draw text with a dark outline so it stays readable over any image."""
    if shadow:
        cv2.putText(canvas, message, origin, FONT, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
    cv2.putText(canvas, message, origin, FONT, scale, color, thickness, cv2.LINE_AA)


def panel(canvas: np.ndarray, lines: Sequence[tuple],
          origin: tuple[int, int] = (10, 10), width: int = 430,
          alpha: float = 0.55) -> np.ndarray:
    """Translucent text panel. Each line is (text, colour, scale).

    A blank text entry renders as vertical space, which is how the collection
    HUD separates its sections without drawing rules.
    """
    x, y = origin
    height = sum(max(14, int(24 * line[2] / 0.55)) for line in lines) + 18
    height = min(height, canvas.shape[0] - y - 4)
    width = min(width, canvas.shape[1] - x - 4)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (60, 60, 60), 1)

    cursor = y + 6
    for message, color, scale in lines:
        cursor += max(14, int(24 * scale / 0.55))
        if cursor > y + height - 2:
            break
        if message:
            text(canvas, message, (x + 10, cursor), color, scale, shadow=False)
    return canvas


def banner(canvas: np.ndarray, message: str, color, scale: float = 1.1,
           y: int | None = None) -> None:
    """Large centred banner, for states that must not be missed."""
    (tw, th), _ = cv2.getTextSize(message, FONT, scale, 3)
    x = (canvas.shape[1] - tw) // 2
    y = th + 24 if y is None else y
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x - 18, y - th - 14), (x + tw + 18, y + 16), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
    text(canvas, message, (x, y), color, scale, 3)


def progress_bar(canvas: np.ndarray, value: int, total: int,
                 origin: tuple[int, int], width: int = 300, height: int = 22,
                 color=GREEN, label: str | None = None) -> None:
    """Horizontal progress bar, e.g. WAYPOINTS 17 / 30."""
    x, y = origin
    total = max(1, total)
    filled = int(width * min(1.0, value / total))
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + height), color, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (140, 140, 140), 1)
    text(canvas, label or f"{value} / {total}",
         (x + width + 12, y + height - 5), WHITE, 0.6, 2)


def coverage_grid(canvas: np.ndarray, grid: np.ndarray,
                  origin: tuple[int, int], cell_px: int = 34) -> None:
    """Draw the image-cell coverage map used during intrinsic collection.

    Green = covered, dark = still empty. Tells you at a glance which corner of
    the frame the board has not visited yet.
    """
    x, y = origin
    rows, columns = grid.shape
    for row in range(rows):
        for column in range(columns):
            count = int(grid[row, column])
            left, top = x + column * cell_px, y + row * cell_px
            color = (30, 30, 30) if count == 0 else (
                40, min(255, 90 + 45 * count), 40)
            cv2.rectangle(canvas, (left, top),
                          (left + cell_px - 3, top + cell_px - 3), color, -1)
            cv2.rectangle(canvas, (left, top),
                          (left + cell_px - 3, top + cell_px - 3), (90, 90, 90), 1)
            if count:
                text(canvas, str(count), (left + cell_px // 2 - 5,
                                          top + cell_px // 2 + 5),
                     WHITE, 0.45, 1, shadow=False)


def draw_reticle(canvas: np.ndarray, center: Sequence[float],
                 color=BLUE, size: int = 18) -> None:
    """Crosshair marking the detected board centre."""
    cx, cy = int(round(center[0])), int(round(center[1]))
    cv2.line(canvas, (cx - size, cy), (cx + size, cy), color, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - size), (cx, cy + size), color, 1, cv2.LINE_AA)
    cv2.circle(canvas, (cx, cy), 4, color, 1, cv2.LINE_AA)


def border_warning(canvas: np.ndarray, margin_px: float, threshold: float) -> None:
    """Tint the frame edge when the board approaches it (section O)."""
    if margin_px >= threshold * 3:
        return
    severity = max(0.0, min(1.0, 1.0 - margin_px / max(1.0, threshold * 3)))
    thickness = int(4 + 10 * severity)
    color = AMBER if margin_px >= threshold else RED
    height, width = canvas.shape[:2]
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), color, thickness)


def footer(canvas: np.ndarray, left: str, right: str = "") -> None:
    """Key-binding reminder along the bottom of the frame."""
    height = canvas.shape[0]
    if left:
        text(canvas, left, (10, height - 12), WHITE, 0.52, 1)
    if right:
        (tw, _), _ = cv2.getTextSize(right, FONT, 0.52, 1)
        text(canvas, right, (canvas.shape[1] - tw - 10, height - 12), GREY, 0.52, 1)


def fit_to_screen(canvas: np.ndarray, max_width: int = 1600,
                  max_height: int = 900) -> np.ndarray:
    """Shrink a frame that would not fit on screen. Never enlarges."""
    height, width = canvas.shape[:2]
    scale = min(1.0, max_width / width, max_height / height)
    if scale >= 1.0:
        return canvas
    return cv2.resize(canvas, (int(width * scale), int(height * scale)),
                      interpolation=cv2.INTER_AREA)


class FrameRate:
    """Rolling frame-rate estimate for the HUD."""

    def __init__(self, window: int = 30):
        self.window = window
        self._times: list[float] = []

    def tick(self, now: float) -> float:
        self._times.append(now)
        if len(self._times) > self.window:
            self._times = self._times[-self.window:]
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0
