#!/usr/bin/env python3
"""Live AprilTag grid preview for one camera. NEVER moves the robot.

    python scripts/preview_apriltag.py --camera camera_1

Shows tag outlines, IDs, corners, tag count, VALID/INVALID status, image
resolution and -- when intrinsics exist -- the board pose.

This script contains no robot code at all. It cannot move anything.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from apriltag_detector import (COLOR_BAD, COLOR_OK, COLOR_TEXT, COLOR_WARN,
                               build_detector)
from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               load_calibration_config, load_cameras_config,
                               resolve_camera, setup_logging)
from camera_interface import CameraError, open_camera

FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_intrinsics(camera: str):
    """Load result.yaml for a camera if it has been solved. Returns (K, D, info)."""
    path = camera_paths(camera).intrinsics_result
    if not path.is_file():
        return None, None, None
    from calibration_utils import load_yaml
    data = load_yaml(path)
    try:
        K = np.asarray(data["camera_matrix"], dtype=np.float64).reshape(3, 3)
        D = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(1, -1)
    except (KeyError, ValueError) as exc:
        raise CalibrationError(f"{path}: malformed intrinsics ({exc})") from exc
    return K, D, data


def draw_panel(canvas, lines, origin=(10, 10), width=430, alpha=0.55):
    """Draw a translucent text panel. Each line is (text, colour, scale)."""
    x, y = origin
    height = sum(int(26 * line[2] / 0.55) for line in lines) + 16
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cursor = y + 8
    for text, color, scale in lines:
        cursor += int(22 * scale / 0.55)
        cv2.putText(canvas, text, (x + 10, cursor), FONT, scale, color, 1, cv2.LINE_AA)
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Live AprilTag grid preview (read-only, never moves the robot).")
    parser.add_argument("--camera", required=True,
                        help="logical camera name from cameras.yaml, e.g. camera_1")
    parser.add_argument("--no-intrinsics", action="store_true",
                        help="ignore any solved intrinsics and skip board pose")
    parser.add_argument("--save-frame", metavar="PATH",
                        help="write the current frame to PATH when you press S")
    args = parser.parse_args(argv)

    logger = setup_logging("preview_apriltag", args.camera)

    try:
        cameras_config = load_cameras_config()
        calibration_config = load_calibration_config()
        camera_config = resolve_camera(args.camera, cameras_config)
        detector = build_detector(calibration_config)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    grid = calibration_config["apriltag_grid"]
    board_verified = bool(grid.get("verified_by_user", False))

    K = D = None
    intrinsics_info = None
    if not args.no_intrinsics:
        try:
            K, D, intrinsics_info = load_intrinsics(args.camera)
        except CalibrationError as exc:
            logger.warning("%s", exc)

    print("=" * 74)
    print(f"APRILTAG PREVIEW -- {args.camera}")
    print("=" * 74)
    print(f"Board    : {detector.spec.describe()}")
    print(f"Refine   : {detector.refinement}")
    print(f"Intrinsics: {'loaded' if K is not None else 'NOT AVAILABLE (no board pose)'}")
    if not board_verified:
        print()
        print("WARNING: apriltag_grid.verified_by_user is false. Detection works,")
        print("         but every metric number shown is only as correct as the")
        print("         board dimensions in config/calibration.yaml.")
    print()
    print("Keys:  Q or ESC = quit    S = save frame    H = toggle help")
    print()

    camera = None
    show_help = True
    frame_times: list[float] = []
    try:
        camera = open_camera(camera_config)
        window = f"AprilTag preview -- {args.camera}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            loop_start = time.time()
            try:
                frame = camera.read(flush=0)
            except CameraError as exc:
                logger.error("Camera read failed: %s", exc)
                print(f"\nCAMERA FAILURE: {exc}", file=sys.stderr)
                return 1

            detection = detector.process(frame.image, K, D, require_pose=False)
            canvas = detector.annotate(frame.image, detection, K, D)

            frame_times.append(loop_start)
            frame_times = frame_times[-30:]
            fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                   if len(frame_times) > 1 else 0.0)

            status_color = COLOR_OK if detection.valid else COLOR_BAD
            lines = [
                (f"CAMERA: {args.camera}   [{camera.serial}]", COLOR_TEXT, 0.55),
                (f"{frame.width} x {frame.height}   {fps:4.1f} fps", COLOR_TEXT, 0.5),
                ("VALID" if detection.valid else "INVALID", status_color, 0.95),
                (f"Tags: {detection.tags_detected} / {detector.spec.tag_count}"
                 f"    Corners: {detection.corners_detected}", COLOR_TEXT, 0.55),
            ]
            if detection.tags_detected:
                margin_color = (COLOR_WARN
                                if detection.border_margin_px < detector.min_border_margin * 2
                                else COLOR_TEXT)
                lines.append((f"Border margin: {detection.border_margin_px:6.1f} px",
                              margin_color, 0.5))
                lines.append((f"Board area   : {detection.board_area_fraction * 100:5.1f} %",
                              COLOR_TEXT, 0.5))
                lines.append((f"Sharpness    : {detection.sharpness:6.0f}"
                              f"  (min {detector.min_sharpness:.0f})",
                              COLOR_WARN if detection.sharpness < detector.min_sharpness
                              else COLOR_TEXT, 0.5))
            if detection.has_pose:
                x, y, z = detection.tvec
                lines.append((f"Board pose   : x{x * 1000:7.1f} y{y * 1000:7.1f} "
                              f"z{z * 1000:7.1f} mm", COLOR_TEXT, 0.5))
                lines.append((f"Distance {detection.distance_m * 1000:6.0f} mm"
                              f"   Tilt {detection.tilt_deg:5.1f} deg",
                              COLOR_TEXT, 0.5))
                lines.append((f"PnP residual : {detection.pnp_reprojection_px:6.3f} px "
                              f"(max {detection.pnp_max_reprojection_px:.3f})",
                              COLOR_TEXT, 0.5))
            elif K is None:
                lines.append(("No intrinsics -> no board pose.", COLOR_WARN, 0.5))
                lines.append(("Run collect_intrinsics.py then solve_intrinsics.py.",
                              COLOR_WARN, 0.45))
            for reason in detection.reasons[:3]:
                lines.append((f"! {reason}", COLOR_BAD, 0.5))

            draw_panel(canvas, lines)

            if not board_verified:
                cv2.putText(canvas, "BOARD GEOMETRY UNVERIFIED",
                            (10, canvas.shape[0] - 14), FONT, 0.6, COLOR_WARN, 2,
                            cv2.LINE_AA)
            if show_help:
                cv2.putText(canvas, "Q/ESC quit   S save   H help",
                            (canvas.shape[1] - 330, canvas.shape[0] - 14),
                            FONT, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)

            cv2.imshow(window, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help
            if key == ord("s"):
                target = Path(args.save_frame or
                              f"preview_{args.camera}_{int(time.time())}.png")
                cv2.imwrite(str(target), frame.image)
                cv2.imwrite(str(target.with_name(target.stem + "_annotated.png")), canvas)
                print(f"Saved {target} (+ annotated copy)")

    except (CameraError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
