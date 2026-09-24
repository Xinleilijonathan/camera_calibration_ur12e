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


class TestPrintableBoardMatchesObjectPoints:
    """The printed board and the object points must agree by construction.

    This is the invariant that a real board violated in September 2026: a
    board whose tag IDs ran right-to-left along each row instead of
    left-to-right. Every tag decoded, the grid looked perfectly regular, the
    tag images themselves matched the dictionary, and nothing warned -- but
    each tag was matched to the wrong 3D point, and the intrinsic solve
    returned fx=2097 against a true 656 with a 32 px RMS. Re-solving the same
    captures with the IDs mirrored reproduced the factory intrinsics to
    within 1%.

    So: whatever make_board.py emits must round-trip through the detector and
    land on the object points the solver will use.
    """

    def test_rendered_board_detects_completely(self, detector):
        import cv2
        image = detector.render_board(4000.0, margin_px=80)
        detection = detector.detect(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        found = sorted(int(i) for i in detection.ids.ravel())
        expected = list(range(detector.spec.first_tag_id,
                              detector.spec.first_tag_id + detector.spec.tag_count))
        assert found == expected, "rendered board does not show every configured tag"

    def test_rendered_board_lands_on_the_object_points(self, detector):
        """A mirrored or transposed layout still detects; only this catches it."""
        import cv2
        image = detector.render_board(4000.0, margin_px=80)
        detection = detector.detect(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        ids = detection.ids.ravel().tolist()

        board_points = np.asarray(detector.board.getObjPoints(),
                                  dtype=np.float64).reshape(-1, 4, 3)
        board_ids = np.asarray(detector.board.getIds()).ravel()
        lookup = {int(i): board_points[k] for k, i in enumerate(board_ids)}

        object_xy = np.concatenate([lookup[int(t)][:, :2] for t in ids], axis=0)
        image_xy = detection.corners.reshape(-1, 2).astype(np.float64)

        # The board is planar and the render is orthographic-ish, so a correct
        # correspondence fits a homography to well under a pixel. A mirrored
        # layout lands one tag pitch out -- tens of pixels.
        homography, _ = cv2.findHomography(object_xy, image_xy, 0)
        projected = cv2.perspectiveTransform(
            object_xy.reshape(-1, 1, 2), homography).reshape(-1, 2)
        residual = np.linalg.norm(projected - image_xy, axis=1).mean()
        assert residual < 2.0, (
            f"board layout disagrees with getObjPoints(): {residual:.1f} px")

    def test_tag_ids_increase_left_to_right_along_a_row(self, detector):
        """Pin the convention the printed board has to follow."""
        board_points = np.asarray(detector.board.getObjPoints(),
                                  dtype=np.float64).reshape(-1, 4, 3)
        board_ids = np.asarray(detector.board.getIds()).ravel()
        centres = {int(i): board_points[k].mean(axis=0)
                   for k, i in enumerate(board_ids)}
        first = detector.spec.first_tag_id
        # Consecutive IDs inside one row step along +X; the row itself is flat in Y.
        if detector.spec.columns >= 2:
            a, b = centres[first], centres[first + 1]
            assert b[0] > a[0], "consecutive IDs must run left to right"
            assert abs(b[1] - a[1]) < 1e-9, "consecutive IDs must stay in one row"
        # The tag one full row on steps along +Y.
        if detector.spec.rows >= 2:
            c = centres[first + detector.spec.columns]
            assert c[1] > centres[first][1], "the next row must be at larger Y"

    def test_the_check_above_actually_catches_a_mirrored_board(self, detector):
        """Prove the residual check discriminates, rather than always passing.

        render_board() and getObjPoints() both come from the same GridBoard,
        so on its own the test above cannot fail. Build the board that burned
        us -- IDs reversed along each row -- render THAT, and confirm the
        residual explodes. Without this, the check is decoration.
        """
        import cv2
        spec = detector.spec
        rows, cols = spec.rows, spec.columns
        mirrored_ids = np.array(
            [spec.first_tag_id + (r * cols) + (cols - 1 - c)
             for r in range(rows) for c in range(cols)], dtype=np.int32)
        mirrored = cv2.aruco.GridBoard(
            (cols, rows), spec.tag_size_m, spec.tag_spacing_m,
            detector.dictionary, mirrored_ids)
        image = mirrored.generateImage(
            (int(spec.width_m * 4000) + 160, int(spec.height_m * 4000) + 160),
            marginSize=80)

        detection = detector.detect(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        ids = detection.ids.ravel().tolist()
        assert len(ids) == spec.tag_count, "the mirrored board still detects fully"

        board_points = np.asarray(detector.board.getObjPoints(),
                                  dtype=np.float64).reshape(-1, 4, 3)
        board_ids = np.asarray(detector.board.getIds()).ravel()
        lookup = {int(i): board_points[k] for k, i in enumerate(board_ids)}
        object_xy = np.concatenate([lookup[int(t)][:, :2] for t in ids], axis=0)
        image_xy = detection.corners.reshape(-1, 2).astype(np.float64)

        homography, _ = cv2.findHomography(object_xy, image_xy, 0)
        projected = cv2.perspectiveTransform(
            object_xy.reshape(-1, 1, 2), homography).reshape(-1, 2)
        residual = np.linalg.norm(projected - image_xy, axis=1).mean()
        assert residual > 20.0, (
            f"a mirrored board should be obvious, but fitted to {residual:.1f} px")
