"""Hand-eye solving for both mountings, validated against KNOWN transforms.

Synthetic waypoints are generated from a chosen ground-truth X, then the
solver must recover it. This is the test that catches an inverted transform
chain -- the failure mode that produces a clean-looking, completely wrong
calibration.
"""
import math

import cv2
import numpy as np
import pytest

from apriltag_detector import AprilGridDetector, GridSpec
from calibration_utils import (CalibrationError, invert_transform,
                               make_transform, pose_to_matrix, matrix_to_pose,
                               rotation_angle_deg, transform_difference)
from handeye_calibration import (EYE_IN_HAND, EYE_TO_HAND, METHODS,
                                 average_transform, calibrate, compare,
                                 constant_transforms, cross_check_methods,
                                 handeye_warnings, reprojection_errors,
                                 residuals, solve_handeye, validate)

K = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
D = np.zeros((1, 5))


def transform(rotvec, translation):
    rotation, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64).reshape(3, 1))
    return make_transform(rotation, translation)


# Ground truth for each mounting.
TRUE_X_EYE_IN_HAND = transform([0.02, -0.05, 1.5707], [0.035, -0.052, 0.081])
TRUE_X_EYE_TO_HAND = transform([2.1, 0.15, 0.30], [0.85, -0.42, 0.63])
TRUE_BOARD_IN_BASE = transform([0.02, 0.01, 0.4], [0.45, 0.10, 0.12])
TRUE_BOARD_ON_FLANGE = transform([0.05, 3.10, 0.02], [0.012, 0.030, 0.145])


def flange_poses(count=24, seed=7):
    """Robot poses with rotation about several distinct axes.

    Hand-eye is only observable when the rotations are not all about one axis,
    so the fixture deliberately spans three.
    """
    rng = np.random.default_rng(seed)
    poses = []
    for index in range(count):
        rotvec = np.array([
            0.9 + 0.30 * math.sin(index * 0.7) + rng.normal(0, 0.05),
            0.4 * math.cos(index * 0.9) + rng.normal(0, 0.05),
            0.25 * math.sin(index * 1.3) + rng.normal(0, 0.05)])
        translation = np.array([
            0.42 + 0.06 * math.cos(index * 0.5),
            -0.05 + 0.08 * math.sin(index * 0.6),
            0.38 + 0.05 * math.sin(index * 0.35)])
        poses.append(transform(rotvec, translation))
    return poses


def board_views(robot_transforms, X, mode, board_constant, noise=0.0, seed=3):
    """The board-in-camera transforms a perfect sensor would report."""
    rng = np.random.default_rng(seed)
    views = []
    for robot in robot_transforms:
        if mode == EYE_IN_HAND:
            view = invert_transform(X) @ invert_transform(robot) @ board_constant
        else:
            view = invert_transform(X) @ robot @ board_constant
        if noise:
            view = view @ transform(rng.normal(0, noise, 3),
                                    rng.normal(0, noise * 0.01, 3))
        views.append(view)
    return views


class FakeRecord:
    """Minimal stand-in for WaypointRecord, for solver-level tests."""

    def __init__(self, number, robot, board, object_points=None, image_points=None):
        self.number = number
        self._robot = robot
        self._board = board
        self.object_points = (np.zeros((0, 3)) if object_points is None
                              else object_points)
        self.image_points = (np.zeros((0, 2)) if image_points is None
                             else image_points)

    @property
    def tcp_transform(self):
        return self._robot

    @property
    def board_transform(self):
        return self._board


def make_records(mode, count=24, noise=0.0):
    X = TRUE_X_EYE_IN_HAND if mode == EYE_IN_HAND else TRUE_X_EYE_TO_HAND
    constant = TRUE_BOARD_IN_BASE if mode == EYE_IN_HAND else TRUE_BOARD_ON_FLANGE
    robots = flange_poses(count)
    boards = board_views(robots, X, mode, constant, noise=noise)

    spec = GridSpec("tag36h11", 4, 5, 0.030, 0.009)
    detector = AprilGridDetector(spec, {"minimum_tags_required": 1})
    points = np.asarray(detector.board.getObjPoints(),
                        dtype=np.float64).reshape(-1, 3)
    points = points - np.array([spec.width_m / 2, spec.height_m / 2, 0.0])

    records = []
    for index, (robot, board) in enumerate(zip(robots, boards), start=1):
        rvec, _ = cv2.Rodrigues(board[:3, :3])
        projected, _ = cv2.projectPoints(points, rvec, board[:3, 3], K, D)
        records.append(FakeRecord(index, robot, board, points,
                                  projected.reshape(-1, 2)))
    return records, X, constant


class TestEyeInHand:
    """Camera on the wrist (camera_1, the D405), board fixed on the table."""

    def test_recovers_the_known_transform(self):
        records, X, _ = make_records(EYE_IN_HAND)
        solved = solve_handeye([r.tcp_transform for r in records],
                               [r.board_transform for r in records],
                               EYE_IN_HAND, "park")
        translation, rotation = transform_difference(solved, X)
        assert translation * 1000 < 0.5, f"{translation * 1000:.3f} mm off"
        assert rotation < 0.05, f"{rotation:.4f} deg off"

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_every_method_recovers_it(self, method):
        records, X, _ = make_records(EYE_IN_HAND)
        solved = solve_handeye([r.tcp_transform for r in records],
                               [r.board_transform for r in records],
                               EYE_IN_HAND, method)
        translation, rotation = transform_difference(solved, X)
        assert translation * 1000 < 2.0, f"{method}: {translation * 1000:.3f} mm"
        assert rotation < 0.2, f"{method}: {rotation:.4f} deg"

    def test_residuals_are_near_zero_on_clean_data(self):
        records, X, _ = make_records(EYE_IN_HAND)
        result = residuals([r.tcp_transform for r in records],
                           [r.board_transform for r in records], X, EYE_IN_HAND)
        assert result["translation_stats"]["max"] < 0.05     # mm
        assert result["rotation_stats"]["max"] < 0.01        # deg

    def test_reference_transform_is_the_true_board_pose(self):
        records, X, constant = make_records(EYE_IN_HAND)
        result = residuals([r.tcp_transform for r in records],
                           [r.board_transform for r in records], X, EYE_IN_HAND)
        translation, rotation = transform_difference(
            result["reference_transform"], constant)
        assert translation * 1000 < 0.1
        assert rotation < 0.01


class TestEyeToHand:
    """Camera fixed at the side (camera_2/3, the D435s), board on the robot."""

    def test_recovers_the_known_transform(self):
        records, X, _ = make_records(EYE_TO_HAND)
        solved = solve_handeye([r.tcp_transform for r in records],
                               [r.board_transform for r in records],
                               EYE_TO_HAND, "park")
        translation, rotation = transform_difference(solved, X)
        assert translation * 1000 < 0.5, f"{translation * 1000:.3f} mm off"
        assert rotation < 0.05, f"{rotation:.4f} deg off"

    @pytest.mark.parametrize("method", sorted(METHODS))
    def test_every_method_recovers_it(self, method):
        records, X, _ = make_records(EYE_TO_HAND)
        solved = solve_handeye([r.tcp_transform for r in records],
                               [r.board_transform for r in records],
                               EYE_TO_HAND, method)
        translation, rotation = transform_difference(solved, X)
        assert translation * 1000 < 2.0, f"{method}: {translation * 1000:.3f} mm"
        assert rotation < 0.2, f"{method}: {rotation:.4f} deg"

    def test_reference_transform_is_the_board_on_the_flange(self):
        records, X, constant = make_records(EYE_TO_HAND)
        result = residuals([r.tcp_transform for r in records],
                           [r.board_transform for r in records], X, EYE_TO_HAND)
        translation, _ = transform_difference(result["reference_transform"], constant)
        assert translation * 1000 < 0.1


class TestModeMatters:
    """The two modes are genuinely different maths, not a labelling detail."""

    def test_solving_eye_to_hand_data_as_eye_in_hand_gives_the_wrong_answer(self):
        records, X, _ = make_records(EYE_TO_HAND)
        wrong = solve_handeye([r.tcp_transform for r in records],
                              [r.board_transform for r in records],
                              EYE_IN_HAND, "park")
        translation, _ = transform_difference(wrong, X)
        assert translation * 1000 > 50, (
            "using the wrong mode must NOT accidentally produce the right answer")

    def test_wrong_mode_inflates_the_residuals(self):
        """The wrong mode is detectable: its residuals blow up."""
        records, _, _ = make_records(EYE_TO_HAND)
        robots = [r.tcp_transform for r in records]
        boards = [r.board_transform for r in records]
        wrong = solve_handeye(robots, boards, EYE_IN_HAND, "park")
        result = residuals(robots, boards, wrong, EYE_IN_HAND)
        assert result["translation_stats"]["mean"] > 5.0

    def test_unknown_mode_is_refused(self):
        records, _, _ = make_records(EYE_IN_HAND)
        with pytest.raises(CalibrationError, match="Unknown hand-eye mode"):
            solve_handeye([r.tcp_transform for r in records],
                          [r.board_transform for r in records], "sideways", "park")


class TestSolverGuards:
    def test_too_few_observations_is_refused(self):
        with pytest.raises(CalibrationError, match="at least 3"):
            solve_handeye([np.eye(4)], [np.eye(4)], EYE_IN_HAND, "park")

    def test_mismatched_input_lengths_are_refused(self):
        with pytest.raises(CalibrationError, match="Mismatched"):
            solve_handeye([np.eye(4)] * 4, [np.eye(4)] * 3, EYE_IN_HAND, "park")

    def test_unknown_method_is_refused(self):
        records, _, _ = make_records(EYE_IN_HAND)
        with pytest.raises(CalibrationError, match="Unknown hand-eye method"):
            solve_handeye([r.tcp_transform for r in records],
                          [r.board_transform for r in records], EYE_IN_HAND, "magic")


class TestAverageTransform:
    def test_averaging_identical_transforms_is_a_no_op(self):
        base = transform([0.1, 0.2, 0.3], [1, 2, 3])
        assert np.allclose(average_transform([base] * 5), base, atol=1e-12)

    def test_result_is_a_proper_rotation(self):
        from calibration_utils import is_rotation_matrix
        transforms = [transform(np.random.default_rng(i).normal(0, 0.3, 3),
                                np.random.default_rng(i).normal(0, 0.05, 3))
                      for i in range(8)]
        assert is_rotation_matrix(average_transform(transforms)[:3, :3])

    def test_translation_uses_the_median_so_one_outlier_cannot_drag_it(self):
        good = [transform([0, 0, 0], [1.0, 0, 0]) for _ in range(9)]
        outlier = transform([0, 0, 0], [100.0, 0, 0])
        averaged = average_transform(good + [outlier])
        assert averaged[0, 3] == pytest.approx(1.0, abs=0.01)


class TestFullCalibrate:
    def test_calibrate_produces_a_complete_record(self):
        records, X, _ = make_records(EYE_IN_HAND)
        result = calibrate(records, EYE_IN_HAND, "park", K, D, label="test")
        for key in ("transform", "rotation_matrix", "rotation_vector",
                    "translation_m", "translation_mm", "mode", "method",
                    "observation_count", "waypoint_numbers",
                    "translation_residual_mm", "rotation_residual_deg",
                    "reprojection_px", "per_waypoint", "timestamp",
                    "transform_meaning", "cross_check", "warnings"):
            assert key in result, f"missing {key}"
        assert result["observation_count"] == len(records)

    def test_chain_reprojection_is_sub_pixel_on_clean_data(self):
        records, _, _ = make_records(EYE_IN_HAND)
        result = calibrate(records, EYE_IN_HAND, "park", K, D)
        assert result["reprojection_px"]["mean"] < 0.5

    def test_chain_reprojection_catches_a_wrong_transform(self):
        """The metric that can actually fail, unlike a PnP residual."""
        records, X, _ = make_records(EYE_IN_HAND)
        broken = X.copy()
        broken[0, 3] += 0.05           # 50 mm error
        result = reprojection_errors(records, broken, EYE_IN_HAND, K, D)
        assert result["overall"]["mean"] > 5.0

    def test_cross_check_agrees_on_clean_data(self):
        records, _, _ = make_records(EYE_IN_HAND)
        check = cross_check_methods([r.tcp_transform for r in records],
                                    [r.board_transform for r in records],
                                    EYE_IN_HAND)
        assert len(check["solutions"]) == len(METHODS)
        assert check["spread"]["max_translation_difference_mm"] < 3.0

    def test_noise_degrades_but_does_not_break_the_solve(self):
        records, X, _ = make_records(EYE_IN_HAND, noise=0.004)
        result = calibrate(records, EYE_IN_HAND, "park", K, D)
        solved = np.asarray(result["transform"])
        translation, _ = transform_difference(solved, X)
        assert translation * 1000 < 25.0

    def test_records_without_a_board_pose_are_skipped(self):
        records, _, _ = make_records(EYE_IN_HAND)
        records[0]._board = None
        result = calibrate(records, EYE_IN_HAND, "park", K, D)
        assert result["observation_count"] == len(records) - 1
        assert 1 not in result["waypoint_numbers"]


class TestWarnings:
    def test_implausible_wrist_camera_offset_is_flagged(self):
        result = {"mode": EYE_IN_HAND, "translation_m": [0.9, 0.0, 0.0],
                  "cross_check": {}, "translation_residual_mm": {"max": 1.0},
                  "reprojection_px": {"mean": 0.3}}
        assert any("implausibly far" in w for w in handeye_warnings(result))

    def test_method_disagreement_is_flagged(self):
        result = {"mode": EYE_IN_HAND, "translation_m": [0.05, 0, 0.08],
                  "cross_check": {"spread": {"max_translation_difference_mm": 45.0,
                                             "max_rotation_difference_deg": 0.1}},
                  "translation_residual_mm": {"max": 1.0},
                  "reprojection_px": {"mean": 0.3}}
        assert any("disagree" in w for w in handeye_warnings(result))

    def test_zero_translation_is_flagged(self):
        result = {"mode": EYE_IN_HAND, "translation_m": [0.0, 0.0, 0.0],
                  "cross_check": {}, "translation_residual_mm": {"max": 0.1},
                  "reprojection_px": {"mean": 0.1}}
        assert any("essentially zero" in w for w in handeye_warnings(result))

    def test_clean_result_produces_no_warnings(self):
        records, _, _ = make_records(EYE_IN_HAND)
        result = calibrate(records, EYE_IN_HAND, "park", K, D)
        assert result["warnings"] == [], result["warnings"]


class TestValidation:
    def test_holdout_uses_the_reference_from_the_fit(self):
        records, X, _ = make_records(EYE_IN_HAND, count=30)
        fit, held = records[:20], records[20:]
        result = calibrate(fit, EYE_IN_HAND, "park", K, D)
        reference = np.asarray(result["reference_constant_transform"])
        validation = validate(held, np.asarray(result["transform"]),
                              EYE_IN_HAND, K, D, reference=reference)
        assert validation["count"] == len(held)
        assert "true hold-out" in validation["note"]
        assert validation["reprojection_px"]["mean"] < 0.5

    def test_without_a_reference_the_note_says_so(self):
        records, X, _ = make_records(EYE_IN_HAND, count=30)
        validation = validate(records[20:], X, EYE_IN_HAND, K, D)
        assert "internal consistency only" in validation["note"]

    def test_empty_holdout_is_handled(self):
        assert validate([], TRUE_X_EYE_IN_HAND, EYE_IN_HAND, K, D)["count"] == 0

    def test_validation_detects_a_bad_transform(self):
        records, X, _ = make_records(EYE_IN_HAND, count=30)
        broken = X.copy()
        broken[1, 3] += 0.04
        result = calibrate(records[:20], EYE_IN_HAND, "park", K, D)
        validation = validate(records[20:], broken, EYE_IN_HAND, K, D,
                              reference=np.asarray(result["reference_constant_transform"]))
        assert validation["reprojection_px"]["mean"] > 5.0


class TestCompare:
    def _result(self, count, reprojection, translation, rotation, X=None):
        return {"observation_count": count,
                "transform": (X if X is not None else TRUE_X_EYE_IN_HAND).tolist(),
                "reprojection_px": {"mean": reprojection, "median": reprojection,
                                    "max": reprojection * 2},
                "translation_residual_mm": {"mean": translation},
                "rotation_residual_deg": {"mean": rotation}}

    def test_reports_all_30_as_better_when_it_holds_out_better(self):
        """Section AD: never automatically claim best-20 wins."""
        comparison = compare(
            self._result(30, 0.4, 1.0, 0.1), self._result(20, 0.3, 0.8, 0.08),
            all_validation={"reprojection_px": {"mean": 0.45}},
            best_validation={"reprojection_px": {"mean": 0.90}})
        assert comparison["recommendation"] == "ALL-30"
        assert any("ALL-30 is better" in r for r in comparison["reasons"])

    def test_reports_best_20_when_it_holds_out_better(self):
        comparison = compare(
            self._result(30, 0.6, 1.4, 0.2), self._result(20, 0.3, 0.8, 0.08),
            all_validation={"reprojection_px": {"mean": 0.80}},
            best_validation={"reprojection_px": {"mean": 0.40}})
        assert comparison["recommendation"] == "BEST-20"
        assert any("BEST-20 is better" in r for r in comparison["reasons"])

    def test_warns_when_there_is_no_holdout_evidence(self):
        comparison = compare(self._result(30, 0.5, 1.0, 0.1),
                             self._result(20, 0.3, 0.8, 0.08))
        assert any("weak evidence" in r for r in comparison["reasons"])

    def test_flags_large_disagreement_between_the_two_solutions(self):
        other = TRUE_X_EYE_IN_HAND.copy()
        other[0, 3] += 0.03
        comparison = compare(self._result(30, 0.5, 1.0, 0.1),
                             self._result(20, 0.3, 0.8, 0.08, X=other))
        assert any("differ by" in r for r in comparison["reasons"])
