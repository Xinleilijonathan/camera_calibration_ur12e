"""Pose diversity evaluation and next-move guidance."""
import math

import numpy as np
import pytest

from calibration_utils import JOINT_PRIORITY, pose_to_matrix
from pose_diversity import (GOOD, LOW, MEDIUM, PoseDiversityAnalyzer,
                            joint_space_distance_deg, pairwise_separations)

CONFIG = {"waypoint_collection": {
    "target_count": 30,
    "minimum_translation_diversity_mm": 15.0,
    "minimum_rotation_diversity_deg": 3.0,
    "marginal_translation_diversity_mm": 25.0,
    "marginal_rotation_diversity_deg": 5.0,
    "diversity_rating": {"translation_good_mm": 60.0, "translation_medium_mm": 25.0,
                         "rotation_good_deg": 20.0, "rotation_medium_deg": 8.0},
    "stages": [
        {"name": "WRIST ONLY", "up_to_waypoint": 10,
         "preferred_joints": ["wrist_3", "wrist_2", "wrist_1"]},
        {"name": "WRIST + SMALL ELBOW", "up_to_waypoint": 18,
         "preferred_joints": ["wrist_3", "wrist_2", "wrist_1", "elbow"]},
        {"name": "WRIST + ELBOW + SMALL SHOULDER", "up_to_waypoint": 25,
         "preferred_joints": ["wrist_3", "wrist_2", "wrist_1", "elbow", "shoulder"]},
        {"name": "FILL MISSING DIVERSITY", "up_to_waypoint": 30,
         "preferred_joints": list(JOINT_PRIORITY)},
    ]}}


def analyzer():
    return PoseDiversityAnalyzer(CONFIG)


class TestCandidateEvaluation:
    def test_first_pose_is_always_good(self):
        evaluation = analyzer().evaluate_candidate([0.0] * 6, [0.4, 0, 0.3, 0, 0, 0])
        assert evaluation.is_first and evaluation.status == GOOD

    def test_distant_pose_is_good(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate([0.1] * 6, [0.5, 0.05, 0.35, 0, 0, 0.3])
        assert evaluation.status == GOOD

    def test_near_duplicate_is_flagged_with_the_waypoint_number(self):
        subject = analyzer()
        subject.add(8, [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate(
            [0.0] * 6, [0.4005, 0.0, 0.3, 0, 0, 0.002])
        assert evaluation.status == "TOO SIMILAR"
        assert "08" in evaluation.reason
        assert evaluation.suggestions

    def test_rotation_alone_makes_a_pose_useful(self):
        """Section X4: either kind of novelty suffices."""
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate(
            [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, math.radians(12)])
        assert evaluation.status == GOOD

    def test_translation_alone_makes_a_pose_useful(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate([0.0] * 6, [0.45, 0.0, 0.3, 0, 0, 0])
        assert evaluation.status == GOOD

    def test_marginal_pose_is_labelled_marginal(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate([0.0] * 6, [0.418, 0.0, 0.3, 0, 0, 0])
        assert evaluation.status == "MARGINAL"

    def test_nearest_is_found_across_the_whole_set_not_just_the_last(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.40, 0.0, 0.3, 0, 0, 0])
        subject.add(2, [0.0] * 6, [0.90, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate([0.0] * 6, [0.405, 0.0, 0.3, 0, 0, 0])
        assert evaluation.nearest_index == 1

    def test_evaluation_never_blocks_recording(self):
        """Diversity is advice; only board validity blocks a recording."""
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        evaluation = subject.evaluate_candidate([0.0] * 6, [0.4, 0.0, 0.3, 0, 0, 0])
        assert evaluation.status == "TOO SIMILAR"
        assert evaluation.is_useful is False        # advisory flag only


class TestAxisDiversity:
    def test_empty_set_rates_low(self):
        assert analyzer().axis_diversity()["overall"] == LOW

    def test_wide_spread_rates_good(self):
        subject = analyzer()
        for index in range(10):
            subject.add(index, [0.0] * 6,
                        [0.3 + 0.01 * index, -0.05 + 0.012 * index,
                         0.25 + 0.009 * index, 0.05 * index, 0.04 * index,
                         0.06 * index])
        diversity = subject.axis_diversity()
        assert diversity["translation"]["x"]["rating"] == GOOD
        assert diversity["overall"] in (GOOD, MEDIUM)

    def test_wrist_only_set_is_rich_in_rotation_and_poor_in_translation(self):
        """The exact failure mode the guidance exists to prevent."""
        subject = analyzer()
        for index in range(12):
            subject.add(index, [0.0] * 6,
                        [0.40, 0.0, 0.30, 0.0, 0.0, math.radians(4 * index)])
        diversity = subject.axis_diversity()
        assert diversity["translation"]["x"]["rating"] == LOW
        assert diversity["rotation"]["axis_3"]["rating"] == GOOD

    def test_weakest_axes_are_reported_worst_first(self):
        subject = analyzer()
        for index in range(12):
            subject.add(index, [0.0] * 6,
                        [0.40 + 0.02 * index, 0.0, 0.30, 0.0, 0.0, 0.0])
        weakest = subject.weakest_axes(2)
        assert all("translation" in axis or "rotation" in axis for axis in weakest)
        assert "X translation" not in weakest       # X is the one that IS varied


class TestStaging:
    def test_first_stage_prefers_wrists_only(self):
        stage = analyzer().current_stage()
        assert stage["name"] == "WRIST ONLY"
        assert set(stage["preferred_joints"]) == {"wrist_3", "wrist_2", "wrist_1"}

    def test_stage_advances_with_the_count(self):
        subject = analyzer()
        for index in range(12):
            subject.add(index, [0.0] * 6, [0.4, 0, 0.3, 0, 0, 0.01 * index])
        assert subject.current_stage()["name"] == "WRIST + SMALL ELBOW"

    def test_final_stage_permits_the_base(self):
        subject = analyzer()
        for index in range(26):
            subject.add(index, [0.0] * 6, [0.4, 0, 0.3, 0, 0, 0.01 * index])
        assert "base" in subject.current_stage()["preferred_joints"]

    def test_stage_is_clamped_past_the_target(self):
        subject = analyzer()
        for index in range(60):
            subject.add(index, [0.0] * 6, [0.4, 0, 0.3, 0, 0, 0.01 * index])
        assert subject.current_stage()["name"] == "FILL MISSING DIVERSITY"


class TestRecommendation:
    def test_starts_with_wrist_guidance(self):
        recommendation = analyzer().recommend()
        assert recommendation["primary_joint"].startswith("wrist")
        assert "wrist" in recommendation["message"].lower()

    def test_recommends_proximal_joints_when_translation_is_starved(self):
        """Wrist-only sets saturate in rotation; the advice must switch."""
        subject = analyzer()
        for index in range(12):
            subject.add(index, [0.0] * 6,
                        [0.40, 0.0, 0.30, 0.0, 0.0, math.radians(5 * index)])
        recommendation = subject.recommend()
        assert recommendation["primary_joint"] in ("elbow", "shoulder", "base")
        assert "translation" in recommendation["message"].lower()

    def test_base_is_not_recommended_while_other_joints_suffice(self):
        subject = analyzer()
        for index in range(6):
            subject.add(index, [0.0] * 6,
                        [0.4 + 0.02 * index, 0.01 * index, 0.3, 0.05 * index,
                         0.04 * index, 0.05 * index])
        assert subject.recommend()["base_movement"] == "NOT NECESSARY"

    def test_recommended_joints_respect_the_priority_order(self):
        subject = analyzer()
        for index in range(20):
            subject.add(index, [0.0] * 6,
                        [0.4 + 0.01 * index, 0, 0.3, 0, 0, 0.02 * index])
        joints = subject.recommend()["joints"]
        indices = [JOINT_PRIORITY.index(joint) for joint in joints]
        assert indices == sorted(indices), "guidance must stay distal-first"


class TestTravelAccounting:
    def test_no_travel_for_a_single_pose(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0, 0.3, 0, 0, 0])
        assert subject.total_travel()["tcp_mm"] == 0.0

    def test_travel_accumulates_along_the_sequence(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.40, 0, 0.3, 0, 0, 0])
        subject.add(2, [0.0] * 6, [0.45, 0, 0.3, 0, 0, 0])
        subject.add(3, [0.0] * 6, [0.50, 0, 0.3, 0, 0, 0])
        assert subject.total_travel()["tcp_mm"] == pytest.approx(100.0, abs=0.1)

    def test_joint_travel_sums_absolute_changes(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0, 0.3, 0, 0, 0])
        subject.add(2, [math.radians(5)] + [0.0] * 5, [0.4, 0, 0.3, 0, 0, 0])
        assert subject.total_travel()["joint_deg"] == pytest.approx(5.0, abs=1e-6)


class TestSetManagement:
    def test_remove_last_undoes_an_addition(self):
        subject = analyzer()
        subject.add(1, [0.0] * 6, [0.4, 0, 0.3, 0, 0, 0])
        assert subject.remove_last().index == 1
        assert subject.count == 0

    def test_remove_on_an_empty_set_is_safe(self):
        assert analyzer().remove_last() is None

    def test_summary_is_yaml_safe(self):
        import yaml
        from calibration_utils import to_builtin
        subject = analyzer()
        for index in range(5):
            subject.add(index, [0.0] * 6, [0.4 + 0.01 * index, 0, 0.3, 0, 0, 0])
        yaml.safe_dump(to_builtin(subject.summary()))


class TestHelpers:
    def test_joint_distance_is_the_largest_single_joint_change(self):
        distance = joint_space_distance_deg(
            [0, 0, 0, 0, 0, 0], [math.radians(1), 0, 0, 0, 0, math.radians(7)])
        assert distance == pytest.approx(7.0, abs=1e-9)

    def test_pairwise_separations(self):
        from pose_diversity import RecordedPose
        poses = [RecordedPose(1, np.zeros(6), np.array([0.4, 0, 0.3, 0, 0, 0])),
                 RecordedPose(2, np.zeros(6), np.array([0.45, 0, 0.3, 0, 0, 0]))]
        result = pairwise_separations(poses)
        assert result["translation_mm"]["min"] == pytest.approx(50.0, abs=0.1)

    def test_pairwise_separations_needs_two_poses(self):
        assert pairwise_separations([])["translation_mm"] == {}
