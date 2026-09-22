"""Shared test fixtures. No test in this suite opens a camera or a robot."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from apriltag_detector import AprilGridDetector, GridSpec  # noqa: E402


@pytest.fixture
def spec():
    return GridSpec(family="tag36h11", rows=4, columns=5,
                    tag_size_m=0.030, tag_spacing_m=0.009, first_tag_id=0)


@pytest.fixture
def detector(spec):
    return AprilGridDetector(spec, {
        "corner_refinement": "apriltag",
        "minimum_tags_required": 12,
        "minimum_corners_required": 48,
        "minimum_border_margin_px": 10.0,
        "minimum_board_area_fraction": 0.02,
        "maximum_board_area_fraction": 0.90,
        "minimum_sharpness": 10.0,
        "maximum_clipped_fraction": 0.0,
        "maximum_pnp_reprojection_px": 1.5,
    })


@pytest.fixture
def synthetic_view(detector):
    """Render the board, then warp it into a camera view with known intrinsics.

    Returns a callable (rvec, tvec) -> (image, K, D) so tests can place the
    board at an exact known pose and check what the detector recovers.
    """
    import cv2

    width, height = 1280, 720
    K = np.array([[900.0, 0.0, width / 2 - 8.0],
                  [0.0, 905.0, height / 2 + 5.0],
                  [0.0, 0.0, 1.0]])
    D = np.zeros((1, 5))
    spec = detector.spec
    pixels_per_metre = 6000.0
    board_image = detector.render_board(pixels_per_metre, margin_px=60)
    board_h, board_w = board_image.shape[:2]
    margin_m = 60.0 / pixels_per_metre

    # Board-frame coordinates of the rendered image's four corners.
    # cv2.aruco.GridBoard puts the origin at the top-left tag's top-left
    # corner with +X right and +Y DOWN, i.e. the same orientation as the
    # rendered image, so this mapping is a direct scale-and-offset.
    src = np.float32([[0, 0], [board_w, 0], [board_w, board_h], [0, board_h]])
    object_corners = np.float32([
        [-margin_m, -margin_m, 0.0],
        [spec.width_m + margin_m, -margin_m, 0.0],
        [spec.width_m + margin_m, spec.height_m + margin_m, 0.0],
        [-margin_m, spec.height_m + margin_m, 0.0],
    ])

    def render(rvec, tvec):
        projected, _ = cv2.projectPoints(
            object_corners, np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            np.asarray(tvec, dtype=np.float64).reshape(3, 1), K, D)
        transform = cv2.getPerspectiveTransform(src, projected.reshape(-1, 2).astype(np.float32))
        scene = cv2.warpPerspective(
            board_image, transform, (width, height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=128)
        return cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR), K, D

    return render
