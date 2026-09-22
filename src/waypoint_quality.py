"""Per-waypoint quality metrics, outlier flagging, and best-N selection.

Pure analysis over stored waypoints. Opens nothing, commands nothing, and
NEVER deletes an observation -- rejection is a label, not a delete (section U).

WHY SELECTION IS NOT "SORT BY ERROR, TAKE THE FIRST N"
-----------------------------------------------------
Reprojection error is lowest where the geometry is easiest: board square-on,
close, centred, arm barely moved. Sorting by it therefore selects a cluster of
near-identical poses. Those fit beautifully and generalise badly, because the
hand-eye problem is only well conditioned when rotations span several axes.

So selection here is two-stage: reject genuinely bad observations on absolute
quality grounds, then walk the survivors best-first, accepting each only if it
adds geometry the set does not already have. The result is the lowest-error
set that still spans the workspace.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from calibration_utils import (median_absolute_deviation, normalize_min_max,
                               pose_to_matrix, transform_difference)

LOGGER = logging.getLogger(__name__)

#: CSV column order (section Y).
SCORE_COLUMNS = [
    "waypoint_id", "waypoint_number",
    "reprojection_mean_px", "reprojection_rms_px", "reprojection_max_px",
    "pnp_reprojection_px",
    "translation_residual_mm", "rotation_residual_deg",
    "tags_detected", "corners_detected", "visible_fraction",
    "border_margin_px", "clipped_fraction", "sharpness", "board_distance_m",
    "board_tilt_deg",
    "translation_diversity_mm", "rotation_diversity_deg", "joint_diversity_deg",
    "normalized_reprojection", "normalized_translation", "normalized_rotation",
    "combined_score", "outlier", "selected", "rejection_reason",
]


@dataclass
class WaypointScore:
    """All metrics for one waypoint, plus its verdict."""
    number: int
    name: str
    reprojection_mean_px: float = float("nan")
    reprojection_rms_px: float = float("nan")
    reprojection_max_px: float = float("nan")
    pnp_reprojection_px: float = float("nan")
    translation_residual_mm: float = float("nan")
    rotation_residual_deg: float = float("nan")
    tags_detected: int = 0
    corners_detected: int = 0
    visible_fraction: float = 0.0
    border_margin_px: float = float("nan")
    clipped_fraction: float = 0.0
    sharpness: float = 0.0
    board_distance_m: float = float("nan")
    board_tilt_deg: float = float("nan")
    translation_diversity_mm: float = float("nan")
    rotation_diversity_deg: float = float("nan")
    joint_diversity_deg: float = float("nan")
    normalized_reprojection: float = 0.0
    normalized_translation: float = 0.0
    normalized_rotation: float = 0.0
    combined_score: float = float("nan")
    outlier: bool = False
    selected: bool = False
    rejection_reason: str = ""

    def as_row(self) -> dict:
        row = {
            "waypoint_id": self.name,
            "waypoint_number": self.number,
            "outlier": int(self.outlier),
            "selected": int(self.selected),
            "rejection_reason": self.rejection_reason,
        }
        for column in SCORE_COLUMNS:
            if column in row:
                continue
            value = getattr(self, column, None)
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            elif isinstance(value, float):
                value = round(value, 5)
            row[column] = value
        return row


def compute_scores(records, preliminary: Mapping[str, Any]) -> list[WaypointScore]:
    """Assemble every metric for every waypoint (section W).

    `preliminary` is the ALL-N hand-eye result, which supplies the per-waypoint
    hand-eye residuals and chain reprojection errors.
    """
    per_waypoint = {int(entry["number"]): entry
                    for entry in preliminary.get("per_waypoint", [])}
    scores = []
    for record in records:
        entry = per_waypoint.get(record.number, {})
        detection = record.detection or {}
        board = record.board_pose_camera or {}
        score = WaypointScore(
            number=record.number,
            name=record.name,
            reprojection_mean_px=_float(entry.get("mean_px")),
            reprojection_rms_px=_float(entry.get("rms_px")),
            reprojection_max_px=_float(entry.get("max_px")),
            pnp_reprojection_px=_float(board.get("pnp_reprojection_px")),
            translation_residual_mm=_float(entry.get("translation_residual_mm")),
            rotation_residual_deg=_float(entry.get("rotation_residual_deg")),
            tags_detected=int(detection.get("tags_detected", len(record.tag_ids))),
            corners_detected=int(detection.get("corners_detected", 0)),
            visible_fraction=float(detection.get("visible_fraction", 0.0) or 0.0),
            border_margin_px=_float(detection.get("border_margin_px")),
            clipped_fraction=float(detection.get("clipped_fraction", 0.0) or 0.0),
            sharpness=float(detection.get("sharpness", 0.0) or 0.0),
            board_distance_m=_float(board.get("distance_m")),
            board_tilt_deg=_float(board.get("tilt_deg")),
        )
        scores.append(score)

    _add_diversity(records, scores)
    return scores


def _float(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return number if math.isfinite(number) else float("nan")


def _add_diversity(records, scores: Sequence[WaypointScore]) -> None:
    """Nearest-neighbour separation for each waypoint (section W part 4)."""
    matrices = [pose_to_matrix(record.actual_tcp) for record in records]
    joints = [np.asarray(record.actual_q, dtype=np.float64) for record in records]
    for index, score in enumerate(scores):
        translations, rotations, joint_deltas = [], [], []
        for other in range(len(records)):
            if other == index:
                continue
            translation, rotation = transform_difference(
                matrices[index], matrices[other])
            translations.append(translation * 1000.0)
            rotations.append(rotation)
            joint_deltas.append(float(np.max(np.abs(
                np.degrees(joints[index] - joints[other])))))
        if translations:
            score.translation_diversity_mm = min(translations)
            score.rotation_diversity_deg = min(rotations)
            score.joint_diversity_deg = min(joint_deltas)


def flag_outliers(scores: Sequence[WaypointScore],
                  config: Mapping[str, Any]) -> list[WaypointScore]:
    """Mark obviously bad observations WITH A REASON (section X2).

    Nothing is deleted. A flagged observation keeps its image and metadata
    forever and simply becomes ineligible for the final calibration set.
    """
    limits = dict(config.get("outlier_limits") or {})
    max_reprojection = float(limits.get("max_reprojection_rms_px", math.inf))
    max_translation = float(limits.get("max_translation_residual_mm", math.inf))
    max_rotation = float(limits.get("max_rotation_residual_deg", math.inf))
    min_tags = int(limits.get("min_tags", 0))
    min_corners = int(limits.get("min_corners", 0))
    min_border = float(limits.get("min_border_margin_px", 0.0))
    max_pnp = float(limits.get("max_pnp_reprojection_px", math.inf))
    mad_multiplier = float(limits.get("robust_mad_multiplier", 0.0))

    for score in scores:
        reasons = []
        if not math.isfinite(score.reprojection_rms_px):
            reasons.append("no reprojection error available")
        elif score.reprojection_rms_px > max_reprojection:
            reasons.append(
                f"reprojection RMS {score.reprojection_rms_px:.2f} px "
                f"> {max_reprojection:.2f}")
        if (math.isfinite(score.translation_residual_mm)
                and score.translation_residual_mm > max_translation):
            reasons.append(
                f"hand-eye translation residual "
                f"{score.translation_residual_mm:.1f} mm > {max_translation:.1f}")
        if (math.isfinite(score.rotation_residual_deg)
                and score.rotation_residual_deg > max_rotation):
            reasons.append(
                f"hand-eye rotation residual {score.rotation_residual_deg:.2f} deg "
                f"> {max_rotation:.2f}")
        if score.tags_detected < min_tags:
            reasons.append(f"only {score.tags_detected} tags (need {min_tags})")
        if score.corners_detected < min_corners:
            reasons.append(
                f"only {score.corners_detected} corners (need {min_corners})")
        if math.isfinite(score.border_margin_px) and score.border_margin_px < min_border:
            reasons.append(
                f"board {score.border_margin_px:.0f} px from the image edge")
        if score.clipped_fraction > 0:
            reasons.append(
                f"{score.clipped_fraction * 100:.0f}% of the board is off-frame")
        if math.isfinite(score.pnp_reprojection_px) and score.pnp_reprojection_px > max_pnp:
            reasons.append(
                f"PnP residual {score.pnp_reprojection_px:.2f} px > {max_pnp:.2f}")
        if reasons:
            score.outlier = True
            score.rejection_reason = "; ".join(reasons)

    # Robust statistical outliers, relative to THIS set rather than to an
    # absolute limit. Catches a set that is uniformly worse than the limits
    # assume, where absolute thresholds would flag nothing.
    if mad_multiplier > 0:
        values = [s.reprojection_rms_px for s in scores
                  if not s.outlier and math.isfinite(s.reprojection_rms_px)]
        median, scale = median_absolute_deviation(values)
        if not math.isfinite(scale) or scale <= 1e-9:
            # MAD collapses to zero when more than half the set shares a value,
            # which would silently switch this check off exactly when one
            # member stands out most clearly. Fall back to the mean absolute
            # deviation, scaled by its own normality constant.
            finite = [v for v in values if math.isfinite(v)]
            if len(finite) >= 3:
                scale = float(np.mean(np.abs(np.asarray(finite) - median))) * 1.2533
        if math.isfinite(scale) and scale > 1e-9:
            threshold = median + mad_multiplier * scale
            for score in scores:
                if score.outlier or not math.isfinite(score.reprojection_rms_px):
                    continue
                if score.reprojection_rms_px > threshold:
                    score.outlier = True
                    score.rejection_reason = (
                        f"reprojection RMS {score.reprojection_rms_px:.2f} px is "
                        f"{mad_multiplier:.1f} MAD above the set median "
                        f"({median:.2f} px)")
        else:
            LOGGER.debug("Robust outlier check skipped: the set has no spread")
    return list(scores)


def compute_combined_scores(scores: Sequence[WaypointScore],
                            config: Mapping[str, Any]) -> list[WaypointScore]:
    """Normalise each metric to [0,1] across the set, then weight and sum.

    Normalisation is not cosmetic. Pixels, millimetres and degrees have no
    common scale, so adding them raw would let whichever happens to have the
    largest numeric range silently dominate the ranking (section X1).
    """
    weights = dict(config.get("weights") or {})
    weight_reprojection = float(weights.get("reprojection_rms", 0.5))
    weight_translation = float(weights.get("handeye_translation", 0.25))
    weight_rotation = float(weights.get("handeye_rotation", 0.25))
    total = weight_reprojection + weight_translation + weight_rotation
    if total <= 0:
        raise ValueError("selection.weights must sum to more than zero")

    candidates = [s for s in scores if not s.outlier] or list(scores)
    normalized_reprojection = normalize_min_max(
        [s.reprojection_rms_px for s in candidates])
    normalized_translation = normalize_min_max(
        [s.translation_residual_mm for s in candidates])
    normalized_rotation = normalize_min_max(
        [s.rotation_residual_deg for s in candidates])

    for index, score in enumerate(candidates):
        score.normalized_reprojection = float(normalized_reprojection[index])
        score.normalized_translation = float(normalized_translation[index])
        score.normalized_rotation = float(normalized_rotation[index])
        score.combined_score = float(
            (weight_reprojection * score.normalized_reprojection
             + weight_translation * score.normalized_translation
             + weight_rotation * score.normalized_rotation) / total)

    for score in scores:
        if score.outlier and not math.isfinite(score.combined_score):
            score.combined_score = float("inf")   # ranks last, never selected
    return list(scores)


def select_best(scores: Sequence[WaypointScore], records, count: int,
                config: Mapping[str, Any]) -> dict:
    """Greedy low-error, diversity-preserving selection (sections X3, X4).

    Walks candidates best-score-first and accepts one only if it is far enough
    from EVERYTHING already selected in translation OR rotation. Either alone
    suffices: 5 mm with 8 degrees is valuable, and so is 40 mm with 1 degree.
    Demanding both would reject a pure in-place wrist rotation, which is the
    cheapest useful pose there is.
    """
    thresholds = dict(config.get("diversity_thresholds") or {})
    base_translation = float(
        thresholds.get("minimum_translation_separation_mm", 15.0))
    base_rotation = float(thresholds.get("minimum_rotation_separation_deg", 3.0))
    relax_factor = float(config.get("relax_factor", 0.7))
    relax_min = float(config.get("relax_min_factor", 0.2))

    by_number = {record.number: record for record in records}
    candidates = sorted((s for s in scores if not s.outlier),
                        key=lambda s: (s.combined_score, s.number))

    if not candidates:
        return {"selected": [], "rejected": [s.number for s in scores],
                "factor_used": 0.0,
                "note": "every observation was flagged as an outlier"}

    factor = 1.0
    chosen: list[WaypointScore] = []
    while True:
        chosen = _greedy_pass(candidates, by_number, count,
                              base_translation * factor, base_rotation * factor)
        if len(chosen) >= count or factor <= relax_min:
            break
        factor *= relax_factor
        LOGGER.info("Only %d of %d selected; relaxing diversity thresholds to "
                    "%.0f%% and retrying", len(chosen), count, factor * 100)

    if len(chosen) < count:
        # Diversity could not be satisfied even relaxed. Top up by pure score
        # so the calibration still gets the requested number, and say so.
        remaining = [s for s in candidates if s not in chosen]
        shortfall = count - len(chosen)
        chosen.extend(remaining[:shortfall])
        note = (f"diversity thresholds could not be met even at "
                f"{factor * 100:.0f}% strength; the last {min(shortfall, len(remaining))} "
                f"were added on score alone. The pose set lacks geometric spread.")
    else:
        note = (f"diversity thresholds satisfied at {factor * 100:.0f}% of the "
                f"configured separation")

    chosen_numbers = {s.number for s in chosen}
    for score in scores:
        if score.number in chosen_numbers:
            score.selected = True
        elif not score.rejection_reason:
            score.rejection_reason = "redundant pose: too close to a selected waypoint"

    return {
        "selected": sorted(chosen_numbers),
        "rejected": sorted(s.number for s in scores if s.number not in chosen_numbers),
        "factor_used": factor,
        "translation_threshold_mm": base_translation * factor,
        "rotation_threshold_deg": base_rotation * factor,
        "note": note,
    }


def _greedy_pass(candidates, by_number, count, translation_threshold,
                 rotation_threshold) -> list[WaypointScore]:
    """One greedy sweep at a given diversity strength."""
    chosen: list[WaypointScore] = []
    chosen_matrices: list[np.ndarray] = []
    for candidate in candidates:
        if len(chosen) >= count:
            break
        record = by_number.get(candidate.number)
        if record is None:
            continue
        matrix = pose_to_matrix(record.actual_tcp)
        if not _adds_diversity(matrix, chosen_matrices,
                               translation_threshold, rotation_threshold):
            continue
        chosen.append(candidate)
        chosen_matrices.append(matrix)
    return chosen


def _adds_diversity(matrix: np.ndarray, existing: Sequence[np.ndarray],
                    translation_threshold: float,
                    rotation_threshold: float) -> bool:
    """True if `matrix` is far enough from EVERY selected pose.

    Far enough means EITHER threshold is met -- see select_best for why
    demanding both would throw away the most useful poses.
    """
    for other in existing:
        translation, rotation = transform_difference(matrix, other)
        separated = (translation * 1000.0 >= translation_threshold
                     or rotation >= rotation_threshold)
        if not separated:
            return False
    return True


def format_table(scores: Sequence[WaypointScore], limit: int | None = None) -> str:
    """Ranking table for the terminal (section Y)."""
    ordered = sorted(scores, key=lambda s: (
        math.inf if not math.isfinite(s.combined_score) else s.combined_score,
        s.number))
    if limit:
        ordered = ordered[:limit]

    lines = [
        f"{'ID':>4}  {'Reproj RMS':>11}  {'Trans Res':>10}  {'Rot Res':>9}  "
        f"{'Score':>7}  {'Status':<9}  Reason",
        "-" * 100,
    ]
    for score in ordered:
        status = ("SELECTED" if score.selected
                  else "OUTLIER" if score.outlier else "REJECTED")
        combined = ("    -  " if not math.isfinite(score.combined_score)
                    else f"{score.combined_score:7.3f}")
        lines.append(
            f"{score.number:>4}  "
            f"{_fmt(score.reprojection_rms_px, 8, 3, 'px')}  "
            f"{_fmt(score.translation_residual_mm, 7, 2, 'mm')}  "
            f"{_fmt(score.rotation_residual_deg, 5, 3, 'deg')}  "
            f"{combined}  {status:<9}  {score.rejection_reason[:40]}")
    return "\n".join(lines)


def _fmt(value: float, width: int, precision: int, unit: str) -> str:
    if not math.isfinite(value):
        return f"{'-':>{width}} {unit}"
    return f"{value:>{width}.{precision}f} {unit}"


def selection_summary(scores: Sequence[WaypointScore]) -> dict:
    """Counts and reasons, for the saved selection files."""
    selected = [s for s in scores if s.selected]
    rejected = [s for s in scores if not s.selected]
    return {
        "total": len(scores),
        "selected_count": len(selected),
        "rejected_count": len(rejected),
        "outlier_count": sum(1 for s in scores if s.outlier),
        "selected_waypoints": [s.name for s in sorted(selected, key=lambda s: s.number)],
        "rejected_waypoints": {
            s.name: {"reason": s.rejection_reason or "not selected",
                     "outlier": bool(s.outlier),
                     "combined_score": (None if not math.isfinite(s.combined_score)
                                        else round(s.combined_score, 5))}
            for s in sorted(rejected, key=lambda s: s.number)},
    }
