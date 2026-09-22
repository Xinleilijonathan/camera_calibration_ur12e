"""Live pose-diversity analysis and next-move guidance.

Pure geometry over recorded poses. Opens nothing, commands nothing. Every
output is ADVICE for the operator -- nothing here ever moves the robot.

THE TENSION THIS MODULE MANAGES
-------------------------------
Hand-eye calibration is only well conditioned when the observations span
several rotation axes and a real translation range. But a big sweep of the arm
is slow, risky and can lose sight of the board. So the guidance prefers the
smallest motion that fixes the weakest axis, and prefers distal joints, which
buy orientation diversity for very little whole-arm travel:

    wrist_3 -> wrist_2 -> wrist_1 -> elbow -> shoulder -> base

"Minimal movement" is NOT "thirty identical poses". A set with no spread will
produce a hand-eye solve that is numerically confident and geometrically
meaningless, so the diversity ratings below are what stop you stopping early.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from calibration_utils import (JOINT_NAMES, JOINT_PRIORITY, pose_to_matrix,
                               rotation_angle_deg, transform_difference)

LOGGER = logging.getLogger(__name__)

#: Human labels for the HUD.
JOINT_DISPLAY = {
    "base": "Base (J1)", "shoulder": "Shoulder (J2)", "elbow": "Elbow (J3)",
    "wrist_1": "Wrist 1 (J4)", "wrist_2": "Wrist 2 (J5)", "wrist_3": "Wrist 3 (J6)",
}

GOOD, MEDIUM, LOW = "GOOD", "MEDIUM", "LOW"


@dataclass
class RecordedPose:
    """One waypoint's kinematic state, as ACTUALLY measured."""
    index: int
    q: np.ndarray                        # actual joint positions, rad
    tcp: np.ndarray                      # actual TCP [x,y,z,rx,ry,rz]

    @property
    def matrix(self) -> np.ndarray:
        return pose_to_matrix(self.tcp)


@dataclass
class CandidateEvaluation:
    """How useful the pose the robot is in right now would be."""
    is_first: bool
    nearest_index: int | None
    nearest_translation_mm: float
    nearest_rotation_deg: float
    nearest_joint_deg: float
    status: str                          # GOOD / MARGINAL / TOO SIMILAR
    reason: str
    suggestions: list[str] = field(default_factory=list)

    @property
    def is_useful(self) -> bool:
        """Recording is never blocked on diversity -- this is advice only."""
        return self.status != "TOO SIMILAR"


class PoseDiversityAnalyzer:
    """Rates the recorded set and advises what to change next."""

    def __init__(self, config: Mapping[str, Any]):
        collection = dict(config.get("waypoint_collection") or config)
        self.target_count = int(collection.get("target_count", 30))
        self.min_translation_mm = float(
            collection.get("minimum_translation_diversity_mm", 15.0))
        self.min_rotation_deg = float(
            collection.get("minimum_rotation_diversity_deg", 3.0))
        self.marginal_translation_mm = float(
            collection.get("marginal_translation_diversity_mm", 25.0))
        self.marginal_rotation_deg = float(
            collection.get("marginal_rotation_diversity_deg", 5.0))

        rating = dict(collection.get("diversity_rating") or {})
        self.translation_good_mm = float(rating.get("translation_good_mm", 60.0))
        self.translation_medium_mm = float(rating.get("translation_medium_mm", 25.0))
        self.rotation_good_deg = float(rating.get("rotation_good_deg", 20.0))
        self.rotation_medium_deg = float(rating.get("rotation_medium_deg", 8.0))

        self.stages = list(collection.get("stages") or [])
        self.poses: list[RecordedPose] = []

    # -- set management ----------------------------------------------------

    def add(self, index: int, q: Sequence[float], tcp: Sequence[float]) -> None:
        self.poses.append(RecordedPose(
            index=index,
            q=np.asarray(q, dtype=np.float64).reshape(6),
            tcp=np.asarray(tcp, dtype=np.float64).reshape(6)))

    def remove_last(self) -> RecordedPose | None:
        return self.poses.pop() if self.poses else None

    @property
    def count(self) -> int:
        return len(self.poses)

    # -- candidate evaluation (section M) ----------------------------------

    def evaluate_candidate(self, q: Sequence[float],
                           tcp: Sequence[float]) -> CandidateEvaluation:
        """Compare the current pose against every recorded one."""
        q = np.asarray(q, dtype=np.float64).reshape(6)
        tcp = np.asarray(tcp, dtype=np.float64).reshape(6)

        if not self.poses:
            return CandidateEvaluation(
                is_first=True, nearest_index=None,
                nearest_translation_mm=float("inf"),
                nearest_rotation_deg=float("inf"),
                nearest_joint_deg=float("inf"),
                status=GOOD, reason="first waypoint of the session")

        candidate = pose_to_matrix(tcp)
        best_combined = None
        nearest_index = None
        nearest_translation = float("inf")
        nearest_rotation = float("inf")
        nearest_joint = float("inf")

        for pose in self.poses:
            translation, rotation = transform_difference(candidate, pose.matrix)
            translation_mm = translation * 1000.0
            joint_deg = float(np.max(np.abs(np.degrees(q - pose.q))))
            # "Nearest" must blend both units, or a pure rotation would look
            # infinitely close in translation and be judged a duplicate.
            combined = (translation_mm / max(1e-6, self.min_translation_mm)
                        + rotation / max(1e-6, self.min_rotation_deg))
            if best_combined is None or combined < best_combined:
                best_combined = combined
                nearest_index = pose.index
                nearest_translation = translation_mm
                nearest_rotation = rotation
                nearest_joint = joint_deg

        # EITHER kind of novelty qualifies. Requiring both would reject a pure
        # in-place wrist rotation, which is one of the most valuable poses
        # available and the cheapest to reach.
        if (nearest_translation >= self.marginal_translation_mm
                or nearest_rotation >= self.marginal_rotation_deg):
            status, reason = GOOD, "clearly different from every recorded pose"
        elif (nearest_translation >= self.min_translation_mm
                or nearest_rotation >= self.min_rotation_deg):
            status = "MARGINAL"
            reason = (f"only just different from waypoint "
                      f"{nearest_index:02d}")
        else:
            status = "TOO SIMILAR"
            reason = (f"too similar to waypoint {nearest_index:02d} "
                      f"({nearest_translation:.0f} mm, {nearest_rotation:.1f} deg)")

        evaluation = CandidateEvaluation(
            is_first=False, nearest_index=nearest_index,
            nearest_translation_mm=nearest_translation,
            nearest_rotation_deg=nearest_rotation,
            nearest_joint_deg=nearest_joint,
            status=status, reason=reason)
        if status != GOOD:
            evaluation.suggestions = self._separation_suggestions(
                nearest_translation, nearest_rotation)
        return evaluation

    def _separation_suggestions(self, translation_mm: float,
                                rotation_deg: float) -> list[str]:
        """Cheapest way to make a near-duplicate pose distinct."""
        suggestions = []
        rotation_gap = self.marginal_rotation_deg - rotation_deg
        translation_gap = self.marginal_translation_mm - translation_mm
        if rotation_gap > 0:
            suggestions.append(
                f"turn Wrist 2 or Wrist 3 another {max(3.0, rotation_gap):.0f} deg")
        if translation_gap > 0:
            suggestions.append(
                f"shift the TCP another {max(10.0, translation_gap):.0f} mm "
                f"(small Elbow change is the cheapest way)")
        return suggestions

    # -- set-level diversity (sections L, N) -------------------------------

    def axis_diversity(self) -> dict:
        """Per-axis spread of the recorded set, rated GOOD / MEDIUM / LOW.

        Translation axes are the base-frame X, Y, Z ranges. Rotation axes are
        the spread of the rotation-vector components, which is a direct proxy
        for how many distinct rotation axes the set excites -- exactly the
        thing hand-eye calibration needs and the thing a wrist-only set lacks.
        """
        if len(self.poses) < 2:
            empty = {"span": 0.0, "rating": LOW}
            return {"translation": {axis: dict(empty) for axis in "xyz"},
                    "rotation": {f"axis_{i + 1}": dict(empty) for i in range(3)},
                    "overall": LOW}

        translations = np.array([p.tcp[:3] for p in self.poses])
        rotations = np.array([p.tcp[3:6] for p in self.poses])

        result = {"translation": {}, "rotation": {}}
        for index, axis in enumerate("xyz"):
            # np.ptp(), not array.ptp(): the method was removed in NumPy 2.0.
            span_mm = float(np.ptp(translations[:, index]) * 1000.0)
            result["translation"][axis] = {
                "span": span_mm,
                "rating": self._rate(span_mm, self.translation_good_mm,
                                     self.translation_medium_mm),
            }
        for index in range(3):
            span_deg = float(math.degrees(np.ptp(rotations[:, index])))
            result["rotation"][f"axis_{index + 1}"] = {
                "span": span_deg,
                "rating": self._rate(span_deg, self.rotation_good_deg,
                                     self.rotation_medium_deg),
            }

        ratings = ([v["rating"] for v in result["translation"].values()]
                   + [v["rating"] for v in result["rotation"].values()])
        if all(r == GOOD for r in ratings):
            result["overall"] = GOOD
        elif any(r == LOW for r in ratings):
            result["overall"] = LOW
        else:
            result["overall"] = MEDIUM
        return result

    @staticmethod
    def _rate(value: float, good: float, medium: float) -> str:
        if value >= good:
            return GOOD
        if value >= medium:
            return MEDIUM
        return LOW

    def weakest_axes(self, limit: int = 2) -> list[str]:
        """Which axes most need attention, worst first."""
        diversity = self.axis_diversity()
        scored = []
        for axis, entry in diversity["translation"].items():
            scored.append((entry["span"] / max(1e-6, self.translation_good_mm),
                           f"{axis.upper()} translation"))
        for axis, entry in diversity["rotation"].items():
            scored.append((entry["span"] / max(1e-6, self.rotation_good_deg),
                           f"rotation {axis.replace('_', ' ')}"))
        scored.sort()
        return [name for _, name in scored[:limit]]

    # -- staging and recommendation (sections K, N, AJ) --------------------

    def current_stage(self) -> dict:
        """The guidance stage for the next waypoint. Guidance, not a rule."""
        next_index = self.count + 1
        for stage in self.stages:
            if next_index <= int(stage.get("up_to_waypoint", 0)):
                return dict(stage)
        if self.stages:
            return dict(self.stages[-1])
        return {"name": "FREE", "preferred_joints": list(JOINT_PRIORITY),
                "note": "no stages configured"}

    def recommend(self) -> dict:
        """What to change next: the smallest move that fixes the weakest axis.

        Preference order is distal-first, but it is overridden when the set is
        short of TRANSLATION: wrists rotate the tool without moving it far, so
        a wrist-only set saturates in rotation and starves in translation. That
        is the classic way a 'minimal movement' session ends up unusable.
        """
        stage = self.current_stage()
        preferred = [j for j in stage.get("preferred_joints", JOINT_PRIORITY)
                     if j in JOINT_NAMES]
        preferred.sort(key=lambda j: JOINT_PRIORITY.index(j))

        diversity = self.axis_diversity()
        translation_ratings = [v["rating"] for v in diversity["translation"].values()]
        rotation_ratings = [v["rating"] for v in diversity["rotation"].values()]
        translation_weak = translation_ratings.count(LOW)
        rotation_weak = rotation_ratings.count(LOW)

        if self.count < 2:
            joints = [j for j in preferred if j.startswith("wrist")] or preferred
            message = ("Start with wrist motion: maximum orientation change for "
                       "minimum whole-arm travel.")
        elif translation_weak > rotation_weak:
            # Need to MOVE the tool, not just turn it.
            translation_joints = [j for j in preferred
                                  if j in ("elbow", "shoulder", "base")]
            joints = translation_joints or preferred
            weakest = self.weakest_axes(1)
            message = (f"Translation is the weak axis ({weakest[0] if weakest else '?'}). "
                       f"Wrist motion cannot fix this -- it rotates the tool without "
                       f"moving it. Use a small Elbow change"
                       + (", or Shoulder if the elbow is not enough."
                          if "shoulder" in translation_joints else "."))
        elif rotation_weak:
            joints = [j for j in preferred if j.startswith("wrist")] or preferred
            message = "Rotation spread is thin. Turn Wrist 2 / Wrist 3 further."
        else:
            joints = preferred
            message = ("Coverage looks balanced. Keep making small varied changes, "
                       "preferring the wrists.")

        base_needed = ("base" in joints and translation_weak
                       and diversity["overall"] == LOW)
        return {
            "stage": stage.get("name", "FREE"),
            "stage_note": stage.get("note", ""),
            "joints": joints,
            "primary_joint": joints[0] if joints else "wrist_3",
            "message": message,
            "base_movement": "CONSIDER" if base_needed else "NOT NECESSARY",
            "weakest_axes": self.weakest_axes(),
            "diversity": diversity,
        }

    # -- travel accounting (section AJ) ------------------------------------

    def total_travel(self) -> dict:
        """Cumulative joint and TCP travel across the recorded sequence."""
        if len(self.poses) < 2:
            return {"joint_deg": 0.0, "tcp_mm": 0.0, "rotation_deg": 0.0}
        joint_total = tcp_total = rotation_total = 0.0
        for previous, current in zip(self.poses, self.poses[1:]):
            joint_total += float(np.sum(np.abs(np.degrees(current.q - previous.q))))
            translation, rotation = transform_difference(previous.matrix, current.matrix)
            tcp_total += translation * 1000.0
            rotation_total += rotation
        return {"joint_deg": joint_total, "tcp_mm": tcp_total,
                "rotation_deg": rotation_total}

    def summary(self) -> dict:
        """Everything the HUD and the saved session record need."""
        diversity = self.axis_diversity()
        return {
            "count": self.count,
            "target": self.target_count,
            "diversity": diversity,
            "overall": diversity["overall"],
            "weakest_axes": self.weakest_axes(),
            "travel": self.total_travel(),
            "stage": self.current_stage().get("name", "FREE"),
        }


def joint_space_distance_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """Largest single-joint difference between two configurations, degrees."""
    return float(np.max(np.abs(np.degrees(
        np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))))


def pairwise_separations(poses: Sequence[RecordedPose]) -> dict:
    """Min / mean / max separation over all pairs -- a set-quality summary."""
    if len(poses) < 2:
        return {"translation_mm": {}, "rotation_deg": {}}
    translations, rotations = [], []
    for index, first in enumerate(poses):
        for second in poses[index + 1:]:
            translation, rotation = transform_difference(first.matrix, second.matrix)
            translations.append(translation * 1000.0)
            rotations.append(rotation)
    return {
        "translation_mm": {"min": min(translations), "mean": float(np.mean(translations)),
                           "max": max(translations)},
        "rotation_deg": {"min": min(rotations), "mean": float(np.mean(rotations)),
                         "max": max(rotations)},
    }
