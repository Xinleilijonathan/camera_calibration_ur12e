"""Intrinsic solving, diversity tracking, and the resolution guard.

The headline test generates views from a KNOWN camera matrix and checks the
solver recovers it. A calibration test that only asserts "RMS is small" would
pass on a confidently wrong answer.
"""
import math

import cv2
import numpy as np
import pytest

from apriltag_detector import AprilGridDetector, GridSpec
from calibration_utils import CalibrationError
from intrinsic_calibration import (DISTORTION_MODELS, IntrinsicDiversityTracker,
                                   IntrinsicObservation, check_resolution_match,
                                   intrinsic_warnings, load_intrinsics,
                                   save_intrinsics, solve_intrinsics)

TRUE_K = np.array([[910.0, 0.0, 646.0],
                   [0.0, 905.0, 357.0],
                   [0.0, 0.0, 1.0]])
TRUE_D = np.array([[0.09, -0.16, 0.001, -0.002, 0.03]])
IMAGE_SIZE = (1280, 720)


@pytest.fixture(scope="module")
def board_spec():
    return GridSpec("tag36h11", 4, 5, 0.030, 0.009)


@pytest.fixture(scope="module")
def synthetic_observations(board_spec):
    """Project the board from many poses through KNOWN intrinsics.

    Points are generated analytically rather than rendered and re-detected, so
    the test measures the SOLVER, not the detector.
    """
    detector = AprilGridDetector(board_spec, {"minimum_tags_required": 1})
    object_points = np.asarray(detector.board.getObjPoints(),
                               dtype=np.float64).reshape(-1, 3)
    centred = object_points - np.array(
        [board_spec.width_m / 2, board_spec.height_m / 2, 0.0])

    rng = np.random.default_rng(20260914)
    observations = []
    index = 1
    # Deliberately spread across tilt, distance and image position.
    for tilt_x in (-0.45, -0.2, 0.0, 0.2, 0.45):
        for tilt_y in (-0.4, 0.0, 0.4):
            for distance in (0.38, 0.55):
                rvec = np.array([tilt_x, tilt_y, rng.uniform(-0.3, 0.3)])
                offset = rng.uniform(-0.09, 0.09, size=2)
                tvec = np.array([offset[0], offset[1], distance])
                projected, _ = cv2.projectPoints(centred, rvec, tvec, TRUE_K, TRUE_D)
                image_points = projected.reshape(-1, 2)
                if (image_points[:, 0].min() < 5
                        or image_points[:, 0].max() > IMAGE_SIZE[0] - 5
                        or image_points[:, 1].min() < 5
                        or image_points[:, 1].max() > IMAGE_SIZE[1] - 5):
                    continue
                # Sub-pixel detector noise, so the test is not unrealistically clean.
                image_points = image_points + rng.normal(0, 0.05, image_points.shape)
                observations.append(IntrinsicObservation(
                    index=index, image_name=f"observation_{index:03d}.png",
                    object_points=centred.copy(), image_points=image_points,
                    tag_ids=list(range(board_spec.tag_count)),
                    image_size=IMAGE_SIZE,
                    tilt_deg=math.degrees(math.hypot(tilt_x, tilt_y)),
                    distance_m=float(distance), area_fraction=0.1))
                index += 1
    assert len(observations) >= 20, f"fixture produced only {len(observations)}"
    return observations


class TestSolveRecoversKnownIntrinsics:
    def test_focal_length_within_one_percent(self, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE,
                                  {"distortion_model": "standard"})
        assert result["fx"] == pytest.approx(TRUE_K[0, 0], rel=0.01)
        assert result["fy"] == pytest.approx(TRUE_K[1, 1], rel=0.01)

    def test_principal_point_within_a_few_pixels(self, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE,
                                  {"distortion_model": "standard"})
        assert result["cx"] == pytest.approx(TRUE_K[0, 2], abs=8.0)
        assert result["cy"] == pytest.approx(TRUE_K[1, 2], abs=8.0)

    def test_radial_distortion_is_recovered(self, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE,
                                  {"distortion_model": "standard"})
        coefficients = result["distortion_coefficients"]
        assert coefficients[0] == pytest.approx(TRUE_D[0, 0], abs=0.03)
        assert coefficients[1] == pytest.approx(TRUE_D[0, 1], abs=0.08)

    def test_residual_is_at_the_noise_floor(self, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE,
                                  {"distortion_model": "standard"})
        # Injected noise was sigma = 0.05 px per axis.
        assert result["rms_reprojection_error_px"] < 0.15

    def test_result_record_is_complete(self, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE, {})
        for key in ("fx", "fy", "cx", "cy", "camera_matrix",
                    "distortion_coefficients", "image_width", "image_height",
                    "observation_count", "rms_reprojection_error_px",
                    "mean_reprojection_error_px", "median_reprojection_error_px",
                    "max_reprojection_error_px", "per_observation", "timestamp"):
            assert key in result, f"missing {key}"
        assert result["observation_count"] == len(synthetic_observations)
        assert len(result["per_observation"]) == len(synthetic_observations)

    def test_per_observation_errors_are_reported(self, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE, {})
        for entry in result["per_observation"]:
            assert entry["rms_px"] >= 0
            assert entry["max_px"] >= entry["mean_px"]

    @pytest.mark.parametrize("model", sorted(DISTORTION_MODELS))
    def test_every_distortion_model_solves(self, synthetic_observations, model):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE,
                                  {"distortion_model": model})
        assert np.isfinite(result["rms_reprojection_error_px"])
        assert len(result["distortion_coefficients"]) == DISTORTION_MODELS[model][1]


class TestSolveGuards:
    def test_too_few_observations_is_refused(self):
        with pytest.raises(CalibrationError, match="at least 3"):
            solve_intrinsics([], IMAGE_SIZE, {})

    def test_unknown_distortion_model_is_refused(self, synthetic_observations):
        with pytest.raises(CalibrationError, match="Unknown distortion_model"):
            solve_intrinsics(synthetic_observations, IMAGE_SIZE,
                             {"distortion_model": "quantum"})

    def test_high_rms_produces_a_warning(self):
        result = {"rms_reprojection_error_px": 2.5, "aspect_ratio": 1.0,
                  "principal_point_offset_px": {"x": 0, "y": 0},
                  "image_width": 1280, "image_height": 720,
                  "mean_reprojection_error_px": 2.0, "per_observation": []}
        warnings = intrinsic_warnings(result, {"maximum_reprojection_error": 1.0})
        assert any("exceeds the configured limit" in w for w in warnings)

    def test_implausible_aspect_ratio_produces_a_warning(self):
        result = {"rms_reprojection_error_px": 0.2, "aspect_ratio": 1.35,
                  "principal_point_offset_px": {"x": 0, "y": 0},
                  "image_width": 1280, "image_height": 720,
                  "mean_reprojection_error_px": 0.2, "per_observation": []}
        assert any("fy/fx" in w for w in intrinsic_warnings(result, {}))

    def test_far_principal_point_produces_a_warning(self):
        result = {"rms_reprojection_error_px": 0.2, "aspect_ratio": 1.0,
                  "principal_point_offset_px": {"x": 400, "y": 0},
                  "image_width": 1280, "image_height": 720,
                  "mean_reprojection_error_px": 0.2, "per_observation": []}
        assert any("Principal point" in w for w in intrinsic_warnings(result, {}))

    def test_outlier_observation_produces_a_warning(self):
        result = {"rms_reprojection_error_px": 0.2, "aspect_ratio": 1.0,
                  "principal_point_offset_px": {"x": 0, "y": 0},
                  "image_width": 1280, "image_height": 720,
                  "mean_reprojection_error_px": 0.2,
                  "per_observation": [{"index": 7, "image": "a.png", "rms_px": 1.5}]}
        assert any("far above the set mean" in w for w in intrinsic_warnings(result, {}))


class TestResolutionGuard:
    """Intrinsics are in pixels; using them at another resolution is silently wrong."""

    def test_matching_resolution_passes(self):
        check_resolution_match({"image_width": 1280, "image_height": 720}, (1280, 720))

    def test_mismatched_resolution_is_refused(self):
        with pytest.raises(CalibrationError, match="Resolution mismatch"):
            check_resolution_match({"image_width": 1280, "image_height": 720}, (640, 480))


class TestPersistence:
    def test_save_then_load_roundtrip(self, tmp_path, synthetic_observations):
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE, {})
        path = tmp_path / "result.yaml"
        save_intrinsics(path, result, "camera_1", "SERIAL123",
                        {"tag_family": "tag36h11", "rows": 4, "columns": 5})
        K, D, data = load_intrinsics(path)
        assert K[0, 0] == pytest.approx(result["fx"])
        assert K[1, 2] == pytest.approx(result["cy"])
        assert D.shape[1] == len(result["distortion_coefficients"])
        assert data["camera_serial"] == "SERIAL123"

    def test_saved_file_names_the_owning_camera(self, tmp_path, synthetic_observations):
        """Guards section AK: intrinsics must never be reused across cameras."""
        result = solve_intrinsics(synthetic_observations, IMAGE_SIZE, {})
        path = tmp_path / "result.yaml"
        save_intrinsics(path, result, "camera_2", "SERIAL999", {})
        header = path.read_text()
        assert "camera_2" in header and "SERIAL999" in header
        assert "never be" in header and "another camera" in header

    def test_missing_file_gives_actionable_guidance(self, tmp_path):
        with pytest.raises(CalibrationError, match="collect_intrinsics"):
            load_intrinsics(tmp_path / "absent.yaml")

    def test_malformed_file_is_refused(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("camera_matrix: [[1,2],[3,4]]\ndistortion_coefficients: [0]\n")
        with pytest.raises(CalibrationError, match="malformed"):
            load_intrinsics(path)

    def test_observation_dict_roundtrip(self, synthetic_observations):
        original = synthetic_observations[0]
        restored = IntrinsicObservation.from_dict(original.to_dict())
        assert restored.index == original.index
        assert np.allclose(restored.object_points, original.object_points)
        assert np.allclose(restored.image_points, original.image_points)
        assert restored.image_size == original.image_size


class TestDiversityTracker:
    def _tracker(self, **overrides):
        config = {"diversity": {"image_cells": 3, "minimum_cells_covered": 6,
                                "minimum_tilt_deg": 20.0, "minimum_distinct_tilts": 6,
                                "minimum_scale_ratio": 1.6,
                                "minimum_new_translation_m": 0.02,
                                "minimum_new_rotation_deg": 5.0, **overrides}}
        return IntrinsicDiversityTracker(config, IMAGE_SIZE)

    def test_cell_mapping_covers_the_frame(self):
        tracker = self._tracker()
        assert tracker.cell_for((10, 10)) == (0, 0)
        assert tracker.cell_for((1270, 710)) == (2, 2)
        assert tracker.cell_for((640, 360)) == (1, 1)

    def test_out_of_frame_centre_is_clamped(self):
        tracker = self._tracker()
        assert tracker.cell_for((-50, -50)) == (0, 0)
        assert tracker.cell_for((99999, 99999)) == (2, 2)

    def test_first_observation_is_always_novel(self):
        detection = type("D", (), {"tvec": np.zeros(3), "rvec": np.zeros(3)})()
        novel, _ = self._tracker().is_novel(detection)
        assert novel

    def test_nearly_identical_view_is_rejected(self):
        tracker = self._tracker()
        tracker.add(IntrinsicObservation(
            index=1, image_name="a.png", object_points=np.zeros((4, 3)),
            image_points=np.zeros((4, 2)), tag_ids=[], image_size=IMAGE_SIZE,
            detection={"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.5]}))
        detection = type("D", (), {"tvec": np.array([0.003, 0.0, 0.5]),
                                   "rvec": np.array([0.0, 0.0, 0.01])})()
        novel, reason = tracker.is_novel(detection)
        assert not novel and "too similar" in reason

    def test_translation_alone_makes_a_view_novel(self):
        """Section X4 logic: EITHER kind of novelty suffices, never both."""
        tracker = self._tracker()
        tracker.add(IntrinsicObservation(
            index=1, image_name="a.png", object_points=np.zeros((4, 3)),
            image_points=np.zeros((4, 2)), tag_ids=[], image_size=IMAGE_SIZE,
            detection={"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.5]}))
        detection = type("D", (), {"tvec": np.array([0.10, 0.0, 0.5]),
                                   "rvec": np.array([0.0, 0.0, 0.0])})()
        assert tracker.is_novel(detection)[0]

    def test_rotation_alone_makes_a_view_novel(self):
        tracker = self._tracker()
        tracker.add(IntrinsicObservation(
            index=1, image_name="a.png", object_points=np.zeros((4, 3)),
            image_points=np.zeros((4, 2)), tag_ids=[], image_size=IMAGE_SIZE,
            detection={"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.5]}))
        detection = type("D", (), {"tvec": np.array([0.0, 0.0, 0.5]),
                                   "rvec": np.array([0.0, 0.0, 0.5])})()
        assert tracker.is_novel(detection)[0]

    def test_report_names_what_is_missing(self):
        tracker = self._tracker()
        report = tracker.report()
        assert not report["satisfied"]
        assert any("image cells" in m for m in report["missing"])

    def test_report_is_satisfied_once_covered(self):
        tracker = self._tracker()
        for row in range(3):
            for column in range(3):
                tracker.add(IntrinsicObservation(
                    index=row * 3 + column, image_name="x.png",
                    object_points=np.zeros((4, 3)), image_points=np.zeros((4, 2)),
                    tag_ids=[], image_size=IMAGE_SIZE, cell=(row, column),
                    tilt_deg=30.0,
                    area_fraction=0.05 if (row + column) % 2 else 0.15))
        report = tracker.report()
        assert report["satisfied"], report["missing"]
        assert report["cells_covered"] == 9

    def test_coverage_grid_counts_cells(self):
        tracker = self._tracker()
        for _ in range(3):
            tracker.add(IntrinsicObservation(
                index=1, image_name="x.png", object_points=np.zeros((4, 3)),
                image_points=np.zeros((4, 2)), tag_ids=[], image_size=IMAGE_SIZE,
                cell=(1, 1)))
        grid = tracker.coverage_grid()
        assert grid[1, 1] == 3 and grid.sum() == 3
