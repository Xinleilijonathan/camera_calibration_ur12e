"""AprilTag grid detection, board pose estimation, and preview annotation.

Uses OpenCV's ArUco module with the AprilTag dictionaries. That is the only
AprilTag implementation that is both installed and working in this venv:
  * ros-jazzy's `apriltag` .so is compiled against NumPy 1.x and fails to
    import under the NumPy 2.2 in .venv;
  * pupil-apriltags is not installed.
OpenCV additionally gives CORNER_REFINE_APRILTAG sub-pixel refinement, which
is what actually determines calibration accuracy.

BOARD FRAME
-----------
Object points come from cv2.aruco.GridBoard. Its origin is the top-left
corner of the first tag, with +X to the right along a row, +Y DOWN along a
column, and +Z = X x Y pointing out of the printed face away from the viewer.
(Verified against getObjPoints(): tag 0 spans (0,0) to (s,s), and the last tag
ends at (board_width, board_height).) Every pose called a "board pose" in this
package is that frame expressed in the camera frame.

Importing this module does not open a camera.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from calibration_utils import ConfigError

LOGGER = logging.getLogger(__name__)

#: Supported families -> OpenCV predefined dictionary id.
TAG_FAMILIES = {
    "tag36h11": cv2.aruco.DICT_APRILTAG_36h11,
    "tag36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "tag25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "tag16h5": cv2.aruco.DICT_APRILTAG_16h5,
}

CORNER_REFINEMENT = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
    "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG,
}

# BGR colours for the overlay.
COLOR_OK = (80, 220, 80)
COLOR_BAD = (60, 60, 240)
COLOR_WARN = (0, 190, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_CORNER = (255, 200, 0)


@dataclass
class GridSpec:
    """Physical geometry of the printed AprilTag grid board."""
    family: str
    rows: int
    columns: int
    tag_size_m: float
    tag_spacing_m: float
    first_tag_id: int = 0

    @property
    def tag_count(self) -> int:
        return self.rows * self.columns

    @property
    def corner_count(self) -> int:
        return self.tag_count * 4

    @property
    def pitch_m(self) -> float:
        """Centre-to-centre distance between neighbouring tags."""
        return self.tag_size_m + self.tag_spacing_m

    @property
    def width_m(self) -> float:
        return self.columns * self.tag_size_m + (self.columns - 1) * self.tag_spacing_m

    @property
    def height_m(self) -> float:
        return self.rows * self.tag_size_m + (self.rows - 1) * self.tag_spacing_m

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "GridSpec":
        """Build from the `apriltag_grid` block of calibration.yaml."""
        grid = config.get("apriltag_grid", config)
        family = str(grid["tag_family"]).lower().replace("_", "")
        if family not in TAG_FAMILIES:
            raise ConfigError(
                f"Unsupported tag_family {grid['tag_family']!r}. "
                f"Supported: {', '.join(sorted(TAG_FAMILIES))}")
        return cls(
            family=family,
            rows=int(grid["rows"]),
            columns=int(grid["columns"]),
            tag_size_m=float(grid["tag_size_m"]),
            tag_spacing_m=float(grid["tag_spacing_m"]),
            first_tag_id=int(grid.get("first_tag_id", 0)),
        )

    def describe(self) -> str:
        return (f"{self.family} {self.rows}x{self.columns} "
                f"({self.tag_count} tags), tag {self.tag_size_m * 1000:.1f} mm, "
                f"gap {self.tag_spacing_m * 1000:.1f} mm, board "
                f"{self.width_m * 1000:.0f}x{self.height_m * 1000:.0f} mm")


@dataclass
class Detection:
    """Result of detecting the grid in one frame."""
    ids: np.ndarray                              # (N,) int, detected tag IDs
    corners: np.ndarray                          # (N, 4, 2) float32, image corners
    image_size: tuple[int, int]                  # (width, height)
    tags_detected: int = 0
    corners_detected: int = 0
    border_margin_px: float = float("inf")       # min corner distance to an edge
    visible_fraction: float = 0.0                # detected tags / tags on board
    missing_tag_count: int = 0
    clipped_fraction: float = 0.0                # board outline outside the image
    board_area_fraction: float = 0.0
    sharpness: float = 0.0                       # Laplacian variance in the board
    center_px: tuple[float, float] | None = None
    # Filled in by estimate_pose():
    rvec: np.ndarray | None = None
    tvec: np.ndarray | None = None
    pnp_reprojection_px: float | None = None
    pnp_max_reprojection_px: float | None = None
    object_points: np.ndarray | None = None      # (M, 3) matched board points
    image_points: np.ndarray | None = None       # (M, 2) matched image points
    # Filled in by validate():
    valid: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def has_pose(self) -> bool:
        return self.rvec is not None and self.tvec is not None

    @property
    def distance_m(self) -> float | None:
        """Board distance from the camera, metres."""
        return float(np.linalg.norm(self.tvec)) if self.tvec is not None else None

    @property
    def tilt_deg(self) -> float | None:
        """Angle between the board normal and the camera optical axis."""
        if self.rvec is None:
            return None
        rotation, _ = cv2.Rodrigues(self.rvec)
        normal = rotation[:, 2]                  # board +Z in camera frame
        cosine = abs(float(normal[2]))
        return math.degrees(math.acos(max(0.0, min(1.0, cosine))))

    def summary(self) -> dict:
        """Detection-quality block stored in every observation file."""
        return {
            "tags_detected": int(self.tags_detected),
            "corners_detected": int(self.corners_detected),
            "detected_tag_ids": [int(i) for i in self.ids.ravel()] if self.ids.size else [],
            "border_margin_px": round(float(self.border_margin_px), 3)
                                if math.isfinite(self.border_margin_px) else None,
            "visible_fraction": round(float(self.visible_fraction), 4),
            "missing_tag_count": int(self.missing_tag_count),
            "clipped_fraction": round(float(self.clipped_fraction), 4),
            "board_area_fraction": round(float(self.board_area_fraction), 5),
            "sharpness": round(float(self.sharpness), 2),
            "board_center_px": [round(v, 2) for v in self.center_px] if self.center_px else None,
            "pnp_reprojection_px": round(self.pnp_reprojection_px, 4)
                                   if self.pnp_reprojection_px is not None else None,
            "pnp_max_reprojection_px": round(self.pnp_max_reprojection_px, 4)
                                       if self.pnp_max_reprojection_px is not None else None,
            "board_distance_m": round(self.distance_m, 4) if self.distance_m else None,
            "board_tilt_deg": round(self.tilt_deg, 2) if self.tilt_deg is not None else None,
            "valid": bool(self.valid),
            "reasons": list(self.reasons),
        }


class AprilGridDetector:
    """Detects the configured AprilTag grid and rates the detection quality."""

    def __init__(self, spec: GridSpec, detection_config: Mapping[str, Any] | None = None):
        self.spec = spec
        config = dict(detection_config or {})
        self.min_tags = int(config.get("minimum_tags_required", 1))
        self.min_corners = int(config.get("minimum_corners_required", 4))
        self.min_border_margin = float(config.get("minimum_border_margin_px", 0.0))
        self.min_area_fraction = float(config.get("minimum_board_area_fraction", 0.0))
        self.max_area_fraction = float(config.get("maximum_board_area_fraction", 1.0))
        self.min_sharpness = float(config.get("minimum_sharpness", 0.0))
        self.max_clipped_fraction = float(config.get("maximum_clipped_fraction", 1.0))
        self.max_pnp_reprojection = float(config.get("maximum_pnp_reprojection_px", 1e9))

        if self.min_tags > spec.tag_count:
            raise ConfigError(
                f"minimum_tags_required ({self.min_tags}) exceeds the number of "
                f"tags on the configured board ({spec.tag_count}). This board "
                f"can never produce a valid detection.")

        self.dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILIES[spec.family])
        ids = np.arange(spec.first_tag_id,
                        spec.first_tag_id + spec.tag_count, dtype=np.int32)
        # GridBoard size is (markersX, markersY) = (columns, rows).
        self.board = cv2.aruco.GridBoard(
            (spec.columns, spec.rows), spec.tag_size_m, spec.tag_spacing_m,
            self.dictionary, ids)

        parameters = cv2.aruco.DetectorParameters()
        refinement = str(config.get("corner_refinement", "apriltag")).lower()
        if refinement not in CORNER_REFINEMENT:
            raise ConfigError(
                f"Unknown corner_refinement {refinement!r}. "
                f"Supported: {', '.join(sorted(CORNER_REFINEMENT))}")
        parameters.cornerRefinementMethod = CORNER_REFINEMENT[refinement]
        parameters.cornerRefinementWinSize = int(config.get("corner_refine_win_size", 5))
        parameters.cornerRefinementMaxIterations = int(
            config.get("corner_refine_max_iterations", 30))
        parameters.cornerRefinementMinAccuracy = float(
            config.get("corner_refine_min_accuracy", 0.05))
        self.parameters = parameters
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, parameters)
        self.refinement = refinement

    # -- detection ---------------------------------------------------------

    def detect(self, image: np.ndarray) -> Detection:
        """Detect the grid in a BGR or grayscale image."""
        if image is None or image.size == 0:
            raise ValueError("detect() received an empty image")
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            detection = Detection(ids=np.empty(0, dtype=np.int32),
                                  corners=np.empty((0, 4, 2), dtype=np.float32),
                                  image_size=(width, height))
            detection.reasons.append("no tags detected")
            return detection

        ids = np.asarray(ids, dtype=np.int32).reshape(-1)
        corner_array = np.asarray(corners, dtype=np.float32).reshape(-1, 4, 2)

        # Tags that are not part of the configured board must not be used: a
        # stray tag elsewhere in the scene would corrupt the board pose.
        board_ids = set(self.board.getIds().ravel().tolist())
        keep = np.array([int(i) in board_ids for i in ids], dtype=bool)
        if not keep.all():
            LOGGER.debug("Ignoring %d tag(s) not on the configured board",
                         int((~keep).sum()))
        ids, corner_array = ids[keep], corner_array[keep]

        detection = Detection(ids=ids, corners=corner_array, image_size=(width, height))
        detection.tags_detected = int(ids.size)
        detection.corners_detected = int(ids.size * 4)
        detection.missing_tag_count = int(self.spec.tag_count - ids.size)
        detection.visible_fraction = float(ids.size) / float(self.spec.tag_count)
        if ids.size == 0:
            detection.reasons.append("no tags belonging to the configured board")
            return detection

        flat = corner_array.reshape(-1, 2)
        detection.border_margin_px = float(min(
            flat[:, 0].min(), flat[:, 1].min(),
            (width - 1) - flat[:, 0].max(), (height - 1) - flat[:, 1].max()))
        detection.center_px = (float(flat[:, 0].mean()), float(flat[:, 1].mean()))

        hull = cv2.convexHull(flat.astype(np.float32))
        detection.board_area_fraction = float(
            cv2.contourArea(hull) / float(width * height))
        detection.sharpness = self._sharpness(gray, hull)
        return detection

    @staticmethod
    def _sharpness(gray: np.ndarray, hull: np.ndarray) -> float:
        """Laplacian variance inside the board's convex hull (blur detector).

        Measured only inside the board so that a busy background cannot make a
        blurry board look sharp.
        """
        x, y, w, h = cv2.boundingRect(hull.astype(np.int32))
        x, y = max(0, x), max(0, y)
        w = min(w, gray.shape[1] - x)
        h = min(h, gray.shape[0] - y)
        if w < 8 or h < 8:
            return 0.0
        region = gray[y:y + h, x:x + w]
        return float(cv2.Laplacian(region, cv2.CV_64F).var())

    # -- pose --------------------------------------------------------------

    def estimate_pose(self, detection: Detection,
                      camera_matrix: np.ndarray,
                      dist_coeffs: np.ndarray) -> Detection:
        """Solve the board pose by PnP and record its reprojection residual.

        Mutates and returns `detection`. Requires intrinsics; without them the
        detection is still usable for intrinsic collection, just pose-free.
        """
        if detection.tags_detected == 0:
            return detection
        object_points, image_points = self.board.matchImagePoints(
            [c.reshape(1, 4, 2) for c in detection.corners],
            detection.ids.reshape(-1, 1))
        if object_points is None or len(object_points) < 4:
            detection.reasons.append("too few matched points for PnP")
            return detection

        object_points = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
        dist_coeffs = np.asarray(dist_coeffs, dtype=np.float64).reshape(1, -1)

        # IPPE is the right initialiser for a planar target; LM then refines it.
        ok, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            ok, rvec, tvec = cv2.solvePnP(
                object_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            detection.reasons.append("PnP failed to converge")
            return detection

        rvec, tvec = cv2.solvePnPRefineLM(
            object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec)

        projected, _ = cv2.projectPoints(object_points, rvec, tvec,
                                         camera_matrix, dist_coeffs)
        residuals = np.linalg.norm(
            projected.reshape(-1, 2) - image_points, axis=1)

        detection.rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
        detection.tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
        detection.pnp_reprojection_px = float(np.sqrt(np.mean(residuals ** 2)))
        detection.pnp_max_reprojection_px = float(np.max(residuals))
        detection.object_points = object_points
        detection.image_points = image_points
        detection.clipped_fraction = self._clipped_fraction(
            detection, camera_matrix, dist_coeffs)
        return detection

    def _clipped_fraction(self, detection: Detection,
                          camera_matrix: np.ndarray,
                          dist_coeffs: np.ndarray) -> float:
        """Fraction of the FULL board that falls outside the image.

        Measuring the margin of detected corners cannot see a clipped board:
        tags that leave the frame are simply not detected, so the survivors
        can sit comfortably inside the image while half the board is gone.
        Once a pose is known, projecting every board corner -- including the
        ones that were not detected -- shows directly how much is off-frame.
        """
        if not detection.has_pose:
            return 0.0
        all_points = np.asarray(self.board.getObjPoints(),
                                dtype=np.float64).reshape(-1, 3)
        projected, _ = cv2.projectPoints(
            all_points, detection.rvec, detection.tvec,
            camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        width, height = detection.image_size
        inside = ((projected[:, 0] >= 0) & (projected[:, 0] <= width - 1) &
                  (projected[:, 1] >= 0) & (projected[:, 1] <= height - 1))
        return float(1.0 - inside.mean())

    def board_transform(self, detection: Detection) -> np.ndarray | None:
        """Board pose as a 4x4 transform in the camera frame."""
        if not detection.has_pose:
            return None
        from calibration_utils import make_transform
        rotation, _ = cv2.Rodrigues(detection.rvec)
        return make_transform(rotation, detection.tvec)

    # -- validation --------------------------------------------------------

    def validate(self, detection: Detection, require_pose: bool = False) -> Detection:
        """Apply the configured acceptance thresholds. Mutates `detection`."""
        # Keep any reason recorded during detect()/estimate_pose(); those
        # describe failures that the threshold checks below cannot see.
        reasons: list[str] = [r for r in detection.reasons
                              if r not in ("no tags detected",)]
        if detection.tags_detected == 0:
            reasons.append("no tags detected")
        elif detection.tags_detected < self.min_tags:
            reasons.append(
                f"only {detection.tags_detected} tags (need {self.min_tags})")
        if detection.tags_detected and detection.corners_detected < self.min_corners:
            reasons.append(
                f"only {detection.corners_detected} corners (need {self.min_corners})")
        if detection.tags_detected and detection.border_margin_px < self.min_border_margin:
            reasons.append(
                f"board clipped: {detection.border_margin_px:.0f} px from the edge "
                f"(need {self.min_border_margin:.0f})")
        if detection.tags_detected and detection.board_area_fraction < self.min_area_fraction:
            reasons.append(
                f"board too small/far: {detection.board_area_fraction * 100:.1f}% "
                f"of the image (need {self.min_area_fraction * 100:.1f}%)")
        if detection.board_area_fraction > self.max_area_fraction:
            reasons.append(
                f"board too close: {detection.board_area_fraction * 100:.0f}% "
                f"of the image")
        if detection.tags_detected and detection.sharpness < self.min_sharpness:
            reasons.append(
                f"blurry: sharpness {detection.sharpness:.0f} "
                f"(need {self.min_sharpness:.0f})")
        if detection.clipped_fraction > self.max_clipped_fraction:
            reasons.append(
                f"board clipped: {detection.clipped_fraction * 100:.0f}% of it is "
                f"outside the image (limit {self.max_clipped_fraction * 100:.0f}%)")
        if require_pose:
            if not detection.has_pose:
                reasons.append("no board pose (intrinsics missing or PnP failed)")
            elif detection.pnp_reprojection_px > self.max_pnp_reprojection:
                reasons.append(
                    f"poor PnP fit: {detection.pnp_reprojection_px:.2f} px "
                    f"(limit {self.max_pnp_reprojection:.2f})")

        detection.reasons = reasons
        detection.valid = not reasons
        return detection

    def process(self, image: np.ndarray,
                camera_matrix: np.ndarray | None = None,
                dist_coeffs: np.ndarray | None = None,
                require_pose: bool = False) -> Detection:
        """detect -> estimate_pose (if intrinsics given) -> validate."""
        detection = self.detect(image)
        if camera_matrix is not None and detection.tags_detected:
            self.estimate_pose(detection, camera_matrix, dist_coeffs
                               if dist_coeffs is not None else np.zeros(5))
        return self.validate(detection, require_pose=require_pose)

    # -- rendering ---------------------------------------------------------

    def annotate(self, image: np.ndarray, detection: Detection,
                 camera_matrix: np.ndarray | None = None,
                 dist_coeffs: np.ndarray | None = None,
                 draw_axes: bool = True) -> np.ndarray:
        """Draw tag outlines, IDs, corners and an axis triad onto a copy."""
        canvas = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if detection.tags_detected == 0:
            return canvas

        outline = COLOR_OK if detection.valid else COLOR_WARN
        for tag_id, quad in zip(detection.ids, detection.corners):
            points = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], True, outline, 2, cv2.LINE_AA)
            for corner in quad:
                cv2.circle(canvas, (int(round(corner[0])), int(round(corner[1]))),
                           3, COLOR_CORNER, -1, cv2.LINE_AA)
            # Mark corner 0 so the tag's orientation is visible at a glance.
            cv2.circle(canvas, tuple(np.int32(np.round(quad[0]))), 5,
                       (255, 0, 255), 1, cv2.LINE_AA)
            centre = quad.mean(axis=0)
            label = str(int(tag_id))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            origin = (int(centre[0] - tw / 2), int(centre[1] + th / 2))
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, COLOR_TEXT, 1, cv2.LINE_AA)

        if draw_axes and detection.has_pose and camera_matrix is not None:
            try:
                cv2.drawFrameAxes(
                    canvas, np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3),
                    np.asarray(dist_coeffs if dist_coeffs is not None else np.zeros(5),
                               dtype=np.float64).reshape(1, -1),
                    detection.rvec, detection.tvec,
                    self.spec.tag_size_m * 2.0, 2)
            except cv2.error as exc:
                LOGGER.debug("drawFrameAxes failed: %s", exc)
        return canvas

    def missing_ids(self, detection: Detection) -> list[int]:
        """Board tag IDs that were not detected in this frame."""
        found = set(int(i) for i in detection.ids.ravel())
        return [int(i) for i in self.board.getIds().ravel() if int(i) not in found]

    def render_board(self, pixels_per_metre: float = 4000.0,
                     margin_px: int = 40) -> np.ndarray:
        """Render the configured board to an image.

        Useful for printing a board that exactly matches the config, and for
        generating synthetic test inputs.
        """
        width = int(round(self.spec.width_m * pixels_per_metre))
        height = int(round(self.spec.height_m * pixels_per_metre))
        return self.board.generateImage((width + 2 * margin_px,
                                         height + 2 * margin_px),
                                        marginSize=margin_px)


def build_detector(calibration_config: Mapping[str, Any]) -> AprilGridDetector:
    """Construct a detector straight from a loaded calibration.yaml."""
    return AprilGridDetector(GridSpec.from_config(calibration_config),
                             calibration_config.get("detection", {}))
