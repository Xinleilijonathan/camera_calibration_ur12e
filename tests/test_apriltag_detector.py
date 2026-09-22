"""AprilTag grid detection and board pose recovery."""
import math

import cv2
import numpy as np
import pytest

from apriltag_detector import AprilGridDetector, GridSpec, build_detector
from calibration_utils import ConfigError


class TestGridSpec:
    def test_geometry_matches_hand_computation(self, spec):
        assert spec.tag_count == 20
        assert spec.corner_count == 80
        assert spec.pitch_m == pytest.approx(0.039)
        # 5 tags wide: 5*30mm + 4*9mm = 186mm
        assert spec.width_m == pytest.approx(0.186)
        # 4 tags tall: 4*30mm + 3*9mm = 147mm
        assert spec.height_m == pytest.approx(0.147)

    def test_from_config_accepts_the_shipped_config_shape(self):
        built = GridSpec.from_config({"apriltag_grid": {
            "tag_family": "tag36h11", "rows": 6, "columns": 6,
            "tag_size_m": 0.03, "tag_spacing_m": 0.009}})
        assert built.tag_count == 36

    def test_unknown_family_is_rejected(self):
        with pytest.raises(ConfigError, match="Unsupported tag_family"):
            GridSpec.from_config({"apriltag_grid": {
                "tag_family": "tag99h42", "rows": 2, "columns": 2,
                "tag_size_m": 0.03, "tag_spacing_m": 0.009}})


class TestDetectorConstruction:
    def test_impossible_minimum_tag_count_is_rejected(self, spec):
        with pytest.raises(ConfigError, match="can never produce a valid detection"):
            AprilGridDetector(spec, {"minimum_tags_required": spec.tag_count + 1})

    def test_unknown_refinement_is_rejected(self, spec):
        with pytest.raises(ConfigError, match="Unknown corner_refinement"):
            AprilGridDetector(spec, {"corner_refinement": "magic"})

    def test_board_ids_follow_first_tag_id(self):
        spec = GridSpec("tag36h11", 2, 2, 0.03, 0.009, first_tag_id=40)
        detector = AprilGridDetector(spec, {"minimum_tags_required": 1})
        assert sorted(detector.board.getIds().ravel().tolist()) == [40, 41, 42, 43]


class TestDetection:
    def test_blank_image_detects_nothing_and_is_invalid(self, detector):
        blank = np.full((480, 640, 3), 128, dtype=np.uint8)
        detection = detector.process(blank)
        assert detection.tags_detected == 0
        assert not detection.valid
        assert "no tags detected" in detection.reasons[0]

    def test_empty_image_raises(self, detector):
        with pytest.raises(ValueError):
            detector.detect(np.empty((0, 0, 3), dtype=np.uint8))

    def test_detects_every_tag_in_a_frontal_view(self, detector, synthetic_view):
        image, K, D = synthetic_view([0, 0, 0], [0, 0, 0.45])
        detection = detector.process(image, K, D, require_pose=True)
        assert detection.tags_detected == detector.spec.tag_count
        assert detection.corners_detected == detector.spec.corner_count
        assert detection.valid, detection.reasons

    def test_recovers_a_known_board_pose(self, detector, synthetic_view):
        rvec = np.array([0.20, -0.15, 0.05])
        tvec = np.array([0.03, -0.02, 0.50])
        image, K, D = synthetic_view(rvec, tvec)
        detection = detector.process(image, K, D, require_pose=True)
        assert detection.valid, detection.reasons
        # Translation within 2 mm and rotation within 1 degree of ground truth.
        assert np.linalg.norm(detection.tvec - tvec) < 2e-3
        recovered, _ = cv2.Rodrigues(detection.rvec)
        truth, _ = cv2.Rodrigues(rvec)
        angle = math.degrees(math.acos(
            max(-1.0, min(1.0, (np.trace(recovered.T @ truth) - 1) / 2))))
        assert angle < 1.0

    def test_pnp_residual_is_sub_pixel_on_clean_synthetic_data(
            self, detector, synthetic_view):
        image, K, D = synthetic_view([0.1, 0.1, 0.0], [0, 0, 0.45])
        detection = detector.process(image, K, D, require_pose=True)
        assert detection.pnp_reprojection_px < 0.5

    def test_tilt_and_distance_are_reported(self, detector, synthetic_view):
        image, K, D = synthetic_view([0.5, 0.0, 0.0], [0, 0, 0.60])
        detection = detector.process(image, K, D, require_pose=True)
        assert detection.distance_m == pytest.approx(0.60, abs=5e-3)
        assert detection.tilt_deg == pytest.approx(math.degrees(0.5), abs=2.0)

    def test_foreign_tags_are_ignored(self, detector, synthetic_view):
        """A stray tag not on the board must not enter the pose solve."""
        image, K, D = synthetic_view([0, 0, 0], [0, 0, 0.45])
        stray = cv2.aruco.generateImageMarker(
            detector.dictionary, detector.spec.first_tag_id + detector.spec.tag_count + 5, 90)
        image[10:100, 10:100] = cv2.cvtColor(stray, cv2.COLOR_GRAY2BGR)
        detection = detector.process(image, K, D, require_pose=True)
        assert detection.tags_detected == detector.spec.tag_count
        assert all(int(i) < detector.spec.tag_count for i in detection.ids)

    def test_missing_ids_lists_undetected_tags(self, detector, synthetic_view):
        image, K, D = synthetic_view([0, 0, 0], [0, 0, 0.45])
        detection = detector.process(image, K, D)
        assert detector.missing_ids(detection) == []


class TestValidation:
    def test_far_away_board_fails_the_area_check(self, detector, synthetic_view):
        image, K, D = synthetic_view([0, 0, 0], [0, 0, 3.0])
        detection = detector.process(image, K, D, require_pose=True)
        assert not detection.valid
        assert any("too small" in r or "tags" in r for r in detection.reasons)

    def test_clipped_board_fails_the_border_check(self, detector, synthetic_view):
        # tvec positions the board ORIGIN (top-left tag corner), not its
        # centre, so -0.36 m puts the left half of the board off the image.
        image, K, D = synthetic_view([0, 0, 0], [-0.36, 0, 0.45])
        detection = detector.process(image, K, D, require_pose=True)
        assert not detection.valid
        assert any("clipped" in r for r in detection.reasons), detection.reasons
        # The surviving tags are nowhere near the image edge, which is exactly
        # why border_margin_px alone cannot catch this.
        assert detection.border_margin_px > detector.min_border_margin
        assert detection.clipped_fraction > 0.2
        assert detection.missing_tag_count > 0

    def test_fully_visible_board_is_not_reported_as_clipped(
            self, detector, synthetic_view):
        image, K, D = synthetic_view([0.1, 0.05, 0], [-0.09, -0.07, 0.45])
        detection = detector.process(image, K, D, require_pose=True)
        assert detection.clipped_fraction == 0.0
        assert detection.visible_fraction == 1.0
        assert detection.valid, detection.reasons

    def test_blurred_board_fails_the_sharpness_check(self, detector, synthetic_view):
        image, K, D = synthetic_view([0, 0, 0], [0, 0, 0.45])
        blurred = cv2.GaussianBlur(image, (21, 21), 8)
        detection = detector.detect(blurred)
        detection = detector.validate(detection)
        assert detection.sharpness < detector.min_sharpness or not detection.valid

    def test_require_pose_rejects_a_detection_without_intrinsics(
            self, detector, synthetic_view):
        image, _, _ = synthetic_view([0, 0, 0], [0, 0, 0.45])
        detection = detector.process(image, None, None, require_pose=True)
        assert not detection.valid
        assert any("no board pose" in r for r in detection.reasons)

    def test_summary_is_yaml_safe(self, detector, synthetic_view):
        import yaml
        from calibration_utils import to_builtin
        image, K, D = synthetic_view([0.1, 0.1, 0], [0, 0, 0.45])
        detection = detector.process(image, K, D, require_pose=True)
        yaml.safe_dump(to_builtin(detection.summary()))


class TestAnnotation:
    def test_annotate_returns_a_same_size_copy_and_does_not_mutate(
            self, detector, synthetic_view):
        image, K, D = synthetic_view([0, 0, 0], [0, 0, 0.45])
        original = image.copy()
        canvas = detector.annotate(image, detector.process(image, K, D), K, D)
        assert canvas.shape == image.shape
        assert np.array_equal(image, original)

    def test_annotate_handles_an_empty_detection(self, detector):
        blank = np.full((240, 320, 3), 128, dtype=np.uint8)
        canvas = detector.annotate(blank, detector.process(blank))
        assert canvas.shape == blank.shape
