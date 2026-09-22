"""Scoring, outlier flagging, and diversity-preserving selection."""
import math

import numpy as np
import pytest

from calibration_utils import pose_to_matrix, transform_difference
from waypoint_quality import (SCORE_COLUMNS, WaypointScore, _adds_diversity,
                              compute_combined_scores, compute_scores,
                              flag_outliers, format_table, select_best,
                              selection_summary)


class FakeRecord:
    def __init__(self, number, tcp, q=None, detection=None, pnp=0.2):
        self.number = number
        self.name = f"waypoint_{number:03d}"
        self.actual_tcp = np.asarray(tcp, dtype=np.float64)
        self.actual_q = np.asarray(q if q is not None else [0.0] * 6, dtype=np.float64)
        self.tag_ids = list(range(20))
        self.detection = detection or {
            "tags_detected": 20, "corners_detected": 80, "visible_fraction": 1.0,
            "border_margin_px": 60.0, "clipped_fraction": 0.0, "sharpness": 900.0}
        self.board_pose_camera = {"pnp_reprojection_px": pnp, "distance_m": 0.5,
                                  "tilt_deg": 15.0}


def make_set(count=30, spread=True):
    """Waypoints that are genuinely spread out, with synthetic error metrics."""
    records, per_waypoint = [], []
    for index in range(1, count + 1):
        if spread:
            tcp = [0.4 + 0.02 * index, 0.1 + 0.015 * (index % 7),
                   0.3 + 0.01 * (index % 5),
                   0.1 * (index % 4), 0.08 * (index % 3), 0.05 * index]
        else:
            tcp = [0.4 + 1e-5 * index, 0.1, 0.3, 0.0, 0.0, 1e-5 * index]
        records.append(FakeRecord(index, tcp, q=[0.01 * index] * 6))
        per_waypoint.append({
            "number": index,
            "mean_px": 0.2 + 0.01 * index,
            "rms_px": 0.25 + 0.012 * index,
            "max_px": 0.6 + 0.02 * index,
            "translation_residual_mm": 0.5 + 0.08 * index,
            "rotation_residual_deg": 0.05 + 0.006 * index,
        })
    return records, {"per_waypoint": per_waypoint}


SELECTION_CONFIG = {
    "weights": {"reprojection_rms": 0.5, "handeye_translation": 0.25,
                "handeye_rotation": 0.25},
    "outlier_limits": {"max_reprojection_rms_px": 2.0,
                       "max_translation_residual_mm": 15.0,
                       "max_rotation_residual_deg": 3.0,
                       "min_tags": 12, "min_corners": 48,
                       "min_border_margin_px": 8.0,
                       "max_pnp_reprojection_px": 2.0,
                       "robust_mad_multiplier": 3.5},
    "diversity_thresholds": {"minimum_translation_separation_mm": 15.0,
                             "minimum_rotation_separation_deg": 3.0},
    "relax_factor": 0.7, "relax_min_factor": 0.2,
}


class TestComputeScores:
    def test_every_metric_is_populated(self):
        records, preliminary = make_set(10)
        scores = compute_scores(records, preliminary)
        assert len(scores) == 10
        first = scores[0]
        assert first.reprojection_rms_px > 0
        assert first.translation_residual_mm > 0
        assert first.tags_detected == 20
        assert first.corners_detected == 80
        assert math.isfinite(first.translation_diversity_mm)
        assert math.isfinite(first.rotation_diversity_deg)
        assert math.isfinite(first.joint_diversity_deg)

    def test_diversity_is_nearest_neighbour_not_mean(self):
        records = [FakeRecord(1, [0.4, 0, 0.3, 0, 0, 0]),
                   FakeRecord(2, [0.41, 0, 0.3, 0, 0, 0]),
                   FakeRecord(3, [0.9, 0, 0.3, 0, 0, 0])]
        scores = compute_scores(records, {"per_waypoint": []})
        # Waypoint 1's nearest is 2, at 10 mm -- not the average over 2 and 3.
        assert scores[0].translation_diversity_mm == pytest.approx(10.0, abs=0.1)

    def test_missing_preliminary_entry_leaves_nan_not_zero(self):
        """A missing metric must not masquerade as a perfect score."""
        records, _ = make_set(3)
        scores = compute_scores(records, {"per_waypoint": []})
        assert not math.isfinite(scores[0].reprojection_rms_px)

    def test_row_covers_every_csv_column(self):
        records, preliminary = make_set(3)
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        row = scores[0].as_row()
        for column in SCORE_COLUMNS:
            assert column in row, f"missing CSV column {column}"


class TestOutlierFlagging:
    def test_clean_set_has_no_outliers(self):
        records, preliminary = make_set(20)
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert not any(s.outlier for s in scores)

    def test_high_reprojection_is_flagged_with_a_reason(self):
        records, preliminary = make_set(20)
        preliminary["per_waypoint"][4]["rms_px"] = 9.0
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        flagged = scores[4]
        assert flagged.outlier
        assert "reprojection RMS" in flagged.rejection_reason

    def test_high_translation_residual_is_flagged(self):
        records, preliminary = make_set(20)
        preliminary["per_waypoint"][2]["translation_residual_mm"] = 40.0
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert scores[2].outlier
        assert "translation residual" in scores[2].rejection_reason

    def test_too_few_tags_is_flagged(self):
        records, preliminary = make_set(20)
        records[6].detection["tags_detected"] = 5
        records[6].detection["corners_detected"] = 20
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert scores[6].outlier
        assert "tags" in scores[6].rejection_reason

    def test_clipped_board_is_flagged(self):
        records, preliminary = make_set(20)
        records[3].detection["clipped_fraction"] = 0.25
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert scores[3].outlier
        assert "off-frame" in scores[3].rejection_reason

    def test_mad_catches_a_relative_outlier_within_absolute_limits(self):
        """A set that is uniformly good except one member well out."""
        records, preliminary = make_set(20)
        rng = np.random.default_rng(1)
        for entry in preliminary["per_waypoint"]:
            entry["rms_px"] = 0.30 + rng.normal(0, 0.01)
        preliminary["per_waypoint"][9]["rms_px"] = 1.4    # under the 2.0 limit
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert scores[9].outlier
        assert "MAD" in scores[9].rejection_reason

    def test_robust_check_still_works_when_mad_collapses_to_zero(self):
        """More than half the set sharing one value drives MAD to exactly 0.

        Without a fallback that would silently disable the robust check at the
        moment the odd one out is most obvious.
        """
        records, preliminary = make_set(20)
        for entry in preliminary["per_waypoint"]:
            entry["rms_px"] = 0.30
        preliminary["per_waypoint"][9]["rms_px"] = 1.4
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert scores[9].outlier, "zero MAD disabled the robust outlier check"

    def test_a_genuinely_uniform_set_flags_nothing(self):
        records, preliminary = make_set(20)
        for entry in preliminary["per_waypoint"]:
            entry["rms_px"] = 0.30
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert not any(s.outlier for s in scores)

    def test_outliers_are_never_deleted(self):
        """Section U: rejection is a label, never a delete."""
        records, preliminary = make_set(20)
        preliminary["per_waypoint"][1]["rms_px"] = 50.0
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        assert len(scores) == 20
        assert scores[1] in scores


class TestCombinedScore:
    def test_units_are_normalised_before_being_summed(self):
        """Section X1: pixels, mm and degrees must never be added raw."""
        records, preliminary = make_set(20)
        # Make translation residuals numerically enormous but perfectly ranked.
        for index, entry in enumerate(preliminary["per_waypoint"]):
            entry["translation_residual_mm"] = 1000.0 + index
            entry["rms_px"] = 1.0 - index * 0.04      # best is LAST
            entry["rotation_residual_deg"] = 0.1
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        # Reprojection has weight 0.5 vs translation 0.25, so despite the huge
        # raw millimetre values the last waypoint must still win.
        best = min(scores, key=lambda s: s.combined_score)
        assert best.number == 20

    def test_all_normalised_terms_are_in_unit_range(self):
        records, preliminary = make_set(20)
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        for score in scores:
            for value in (score.normalized_reprojection,
                          score.normalized_translation,
                          score.normalized_rotation):
                assert 0.0 <= value <= 1.0
            assert 0.0 <= score.combined_score <= 1.0

    def test_lower_score_is_better(self):
        records, preliminary = make_set(20)
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        ordered = sorted(scores, key=lambda s: s.combined_score)
        assert ordered[0].reprojection_rms_px < ordered[-1].reprojection_rms_px

    def test_outliers_sort_last(self):
        records, preliminary = make_set(20)
        preliminary["per_waypoint"][0]["rms_px"] = 99.0
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        assert scores[0].combined_score == math.inf

    def test_zero_weights_are_refused(self):
        records, preliminary = make_set(5)
        scores = flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG)
        with pytest.raises(ValueError, match="sum to more than zero"):
            compute_combined_scores(scores, {"weights": {
                "reprojection_rms": 0.0, "handeye_translation": 0.0,
                "handeye_rotation": 0.0}})


class TestDiversityHelper:
    def test_first_pose_always_adds_diversity(self):
        assert _adds_diversity(pose_to_matrix([0, 0, 0, 0, 0, 0]), [], 15.0, 3.0)

    def test_translation_alone_is_enough(self):
        a = pose_to_matrix([0, 0, 0, 0, 0, 0])
        b = pose_to_matrix([0.05, 0, 0, 0, 0, 0])     # 50 mm, 0 deg
        assert _adds_diversity(b, [a], 15.0, 3.0)

    def test_rotation_alone_is_enough(self):
        a = pose_to_matrix([0, 0, 0, 0, 0, 0])
        b = pose_to_matrix([0, 0, 0, 0, 0, math.radians(10)])   # 0 mm, 10 deg
        assert _adds_diversity(b, [a], 15.0, 3.0)

    def test_neither_is_rejected(self):
        a = pose_to_matrix([0, 0, 0, 0, 0, 0])
        b = pose_to_matrix([0.002, 0, 0, 0, 0, math.radians(0.5)])
        assert not _adds_diversity(b, [a], 15.0, 3.0)

    def test_must_be_far_from_every_selected_pose_not_just_one(self):
        far = pose_to_matrix([1.0, 0, 0, 0, 0, 0])
        near = pose_to_matrix([0.001, 0, 0, 0, 0, 0])
        candidate = pose_to_matrix([0, 0, 0, 0, 0, 0])
        assert not _adds_diversity(candidate, [far, near], 15.0, 3.0)


class TestSelection:
    def _prepare(self, records, preliminary):
        scores = compute_scores(records, preliminary)
        scores = flag_outliers(scores, SELECTION_CONFIG)
        return compute_combined_scores(scores, SELECTION_CONFIG)

    def test_selects_exactly_the_requested_count(self):
        records, preliminary = make_set(30)
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        assert len(result["selected"]) == 20
        assert len(result["rejected"]) == 10

    def test_selected_poses_are_mutually_diverse(self):
        """The whole point: 20 near-identical poses must not be the answer."""
        records, preliminary = make_set(30)
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        chosen = [r for r in records if r.number in set(result["selected"])]
        threshold_mm = result["translation_threshold_mm"]
        threshold_deg = result["rotation_threshold_deg"]
        for index, first in enumerate(chosen):
            for second in chosen[index + 1:]:
                translation, rotation = transform_difference(
                    pose_to_matrix(first.actual_tcp), pose_to_matrix(second.actual_tcp))
                assert (translation * 1000 >= threshold_mm - 1e-6
                        or rotation >= threshold_deg - 1e-6), (
                    f"waypoints {first.number} and {second.number} are duplicates")

    def test_does_not_simply_take_the_20_lowest_errors(self):
        """Section AV: diversity must actually change the outcome."""
        records, preliminary = make_set(30)
        # Make the 10 LOWEST-error waypoints nearly identical in pose.
        for index in range(10):
            records[index].actual_tcp = np.array(
                [0.40 + 1e-5 * index, 0.10, 0.30, 0.0, 0.0, 1e-5 * index])
            preliminary["per_waypoint"][index]["rms_px"] = 0.10
            preliminary["per_waypoint"][index]["translation_residual_mm"] = 0.2
            preliminary["per_waypoint"][index]["rotation_residual_deg"] = 0.02
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        cluster = set(range(1, 11))
        chosen_from_cluster = cluster & set(result["selected"])
        assert len(chosen_from_cluster) <= 3, (
            f"selection took {len(chosen_from_cluster)} of the duplicate cluster "
            f"despite their lower error")

    def test_outliers_are_never_selected(self):
        records, preliminary = make_set(30)
        preliminary["per_waypoint"][5]["rms_px"] = 99.0
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        assert 6 not in result["selected"]

    def test_relaxes_thresholds_when_poses_are_too_close(self):
        records, preliminary = make_set(30, spread=False)
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        assert len(result["selected"]) == 20
        assert result["factor_used"] < 1.0
        assert "lacks geometric spread" in result["note"] or result["factor_used"] < 1.0

    def test_every_rejected_waypoint_has_a_reason(self):
        """Section Z: rejections must be explained, never silent."""
        records, preliminary = make_set(30)
        scores = self._prepare(records, preliminary)
        select_best(scores, records, 20, SELECTION_CONFIG)
        for score in scores:
            if not score.selected:
                assert score.rejection_reason, f"{score.name} rejected with no reason"

    def test_all_outliers_is_handled_gracefully(self):
        records, preliminary = make_set(5)
        for entry in preliminary["per_waypoint"]:
            entry["rms_px"] = 99.0
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        assert result["selected"] == []
        assert "outlier" in result["note"]

    def test_requesting_more_than_available_returns_what_exists(self):
        records, preliminary = make_set(8)
        scores = self._prepare(records, preliminary)
        result = select_best(scores, records, 20, SELECTION_CONFIG)
        assert len(result["selected"]) <= 8


class TestReporting:
    def test_summary_partitions_the_set(self):
        records, preliminary = make_set(30)
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        select_best(scores, records, 20, SELECTION_CONFIG)
        summary = selection_summary(scores)
        assert summary["total"] == 30
        assert summary["selected_count"] == 20
        assert summary["rejected_count"] == 10
        assert len(summary["selected_waypoints"]) == 20
        assert all(entry["reason"]
                   for entry in summary["rejected_waypoints"].values())

    def test_table_renders(self):
        records, preliminary = make_set(30)
        scores = compute_combined_scores(
            flag_outliers(compute_scores(records, preliminary), SELECTION_CONFIG),
            SELECTION_CONFIG)
        select_best(scores, records, 20, SELECTION_CONFIG)
        table = format_table(scores)
        assert "SELECTED" in table and "Reproj RMS" in table
        assert len(table.splitlines()) == 32          # header + rule + 30

    def test_table_handles_missing_metrics(self):
        scores = [WaypointScore(number=1, name="waypoint_001")]
        assert "-" in format_table(scores)
