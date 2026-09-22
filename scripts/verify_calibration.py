#!/usr/bin/env python3
"""Verify a finished calibration, numerically and visually. Never moves the robot.

    python scripts/verify_calibration.py --camera camera_1
    python scripts/verify_calibration.py --camera camera_1 --live

Offline mode replays every stored waypoint: it predicts where the board must
appear given the robot pose and the solved transform, projects the board points
into the saved image, and measures the distance to the corners that were
actually detected. Overlays are written to handeye/verification/.

--live opens the camera and does the same thing against the current frame and
the current robot pose. It reads the robot; it never commands it.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401
import _analysis
import ui_overlay as ui

from calibration_utils import (CalibrationError, ConfigError, error_statistics,
                               invert_transform, load_yaml, matrix_to_rotvec,
                               save_yaml, setup_logging, timestamp_utc)
from handeye_calibration import EYE_IN_HAND


def predicted_board_in_camera(transform, robot_transform, reference, mode):
    """Where the board MUST be in the camera, per the calibration chain."""
    if mode == EYE_IN_HAND:
        return invert_transform(transform) @ invert_transform(robot_transform) @ reference
    return invert_transform(transform) @ robot_transform @ reference


def overlay(image, object_points, measured, predicted, label_lines):
    """Draw measured corners (cyan) against predicted ones (magenta)."""
    canvas = image.copy()
    for point in measured:
        cv2.circle(canvas, tuple(np.int32(np.round(point))), 4, ui.BLUE, 1, cv2.LINE_AA)
    for point in predicted:
        cv2.drawMarker(canvas, tuple(np.int32(np.round(point))), ui.MAGENTA,
                       cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
    for start, end in zip(measured, predicted):
        cv2.line(canvas, tuple(np.int32(np.round(start))),
                 tuple(np.int32(np.round(end))), (60, 200, 255), 1, cv2.LINE_AA)
    ui.panel(canvas, label_lines, width=430)
    ui.footer(canvas, "cyan circle = detected    magenta cross = predicted by "
                      "the calibration")
    return canvas


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a hand-eye calibration. Reads the robot; never "
                    "commands it.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--result", default=None,
                        help="result file to verify (default: final_result_best_20.yaml)")
    parser.add_argument("--live", action="store_true",
                        help="verify against the live camera and current robot pose")
    parser.add_argument("--no-images", action="store_true",
                        help="skip writing the per-waypoint overlay images")
    args = parser.parse_args(argv)

    logger = setup_logging("verify_calibration", args.camera)
    try:
        context = _analysis.load_for_analysis(args.camera, logger=logger)
    except (ConfigError, CalibrationError) as exc:
        return _analysis.fail(exc)

    result_path = (context.paths.handeye / args.result if args.result
                   else context.paths.final_result)
    if not result_path.is_file():
        print(f"ERROR: no calibration at {result_path}\n"
              f"  Run: python scripts/solve_handeye.py --camera {args.camera}",
              file=sys.stderr)
        return 1

    result = load_yaml(result_path)
    transform = np.asarray(result["transform"], dtype=np.float64)
    reference = np.asarray(result["reference_constant_transform"], dtype=np.float64)
    mode = str(result.get("handeye_mode", result.get("mode")))
    fitted = set(int(n) for n in result.get("selected_numbers", []))

    _analysis.print_context(context, "CALIBRATION VERIFICATION")
    print(f"Result      : {result_path.name}")
    print(f"Fitted on   : {len(fitted)} waypoints")
    print(f"{result.get('transform_meaning', '')}")
    print()

    if args.live:
        return verify_live(context, transform, reference, mode, result_path)

    context.paths.verification.mkdir(parents=True, exist_ok=True)
    print(f"{'ID':>4}  {'set':<8}  {'mean px':>9}  {'rms px':>9}  {'max px':>9}  {'pts':>5}")
    print("-" * 60)

    rows, fit_errors, holdout_errors = [], [], []
    for record in context.records:
        if record.board_transform is None or record.object_points.size == 0:
            continue
        predicted = predicted_board_in_camera(
            transform, record.tcp_transform, reference, mode)
        projected, _ = cv2.projectPoints(
            record.object_points,
            matrix_to_rotvec(predicted[:3, :3]).reshape(3, 1),
            predicted[:3, 3].reshape(3, 1),
            context.camera_matrix, context.dist_coeffs)
        projected = projected.reshape(-1, 2)
        errors = np.linalg.norm(projected - record.image_points, axis=1)

        in_fit = record.number in fitted
        (fit_errors if in_fit else holdout_errors).extend(errors.tolist())
        rows.append({
            "waypoint": record.name, "number": record.number,
            "in_fit": in_fit, "points": int(errors.size),
            "mean_px": float(np.mean(errors)), "rms_px": float(np.sqrt(np.mean(errors ** 2))),
            "max_px": float(np.max(errors))})
        print(f"{record.number:>4}  {'fit' if in_fit else 'HELD-OUT':<8}  "
              f"{np.mean(errors):>9.4f}  {np.sqrt(np.mean(errors ** 2)):>9.4f}  "
              f"{np.max(errors):>9.4f}  {errors.size:>5}")

        if not args.no_images:
            image_path = context.paths.handeye_images / record.image_name
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Could not read %s", image_path)
                continue
            canvas = overlay(image, record.object_points, record.image_points,
                             projected, [
                                 (f"{record.name}  "
                                  f"[{'fit' if in_fit else 'HELD-OUT'}]", ui.WHITE, 0.6),
                                 (f"mean {np.mean(errors):.3f} px", ui.WHITE, 0.5),
                                 (f"max  {np.max(errors):.3f} px", ui.GREY, 0.5),
                                 (f"{record.detection.get('tags_detected', '?')} tags",
                                  ui.GREY, 0.45)])
            cv2.imwrite(str(context.paths.verification / f"{record.name}_verify.png"),
                        canvas)

    if not rows:
        print("No verifiable waypoints.", file=sys.stderr)
        return 1

    limits = context.calibration_config.get("verification", {})
    limit_px = float(limits.get("maximum_validation_reprojection_px", 2.0))
    fit_stats = error_statistics(fit_errors)
    holdout_stats = error_statistics(holdout_errors)
    all_stats = error_statistics(fit_errors + holdout_errors)

    print()
    print("-" * 60)
    for name, stats in (("FITTED", fit_stats), ("HELD-OUT", holdout_stats),
                        ("ALL", all_stats)):
        if not stats["count"]:
            continue
        print(f"{name:<9} mean {stats['mean']:7.4f}  median {stats['median']:7.4f}  "
              f"max {stats['max']:7.4f} px   ({stats['count']} points)")
    print("-" * 60)
    print()

    status = 0
    if holdout_stats["count"]:
        if holdout_stats["mean"] > limit_px:
            print(f"FAIL: held-out mean {holdout_stats['mean']:.3f} px exceeds "
                  f"the configured limit of {limit_px:.2f} px.")
            status = 1
        else:
            print(f"PASS: held-out mean {holdout_stats['mean']:.3f} px is within "
                  f"the {limit_px:.2f} px limit.")
        if fit_stats["count"] and holdout_stats["mean"] > 2.5 * fit_stats["mean"]:
            print(f"WARNING: held-out error is {holdout_stats['mean'] / fit_stats['mean']:.1f}x "
                  f"the fitted error. The calibration is overfitted to the "
                  f"selected waypoints.")
    else:
        print("No held-out waypoints, so this check only measures self-consistency.")

    report = {
        "camera_name": context.camera_name,
        "camera_serial": str(context.camera_config.get("serial", "")),
        "timestamp": timestamp_utc(),
        "result_file": str(result_path),
        "handeye_mode": mode,
        "limit_px": limit_px,
        "fitted": fit_stats, "holdout": holdout_stats, "all": all_stats,
        "per_waypoint": rows,
        "passed": status == 0,
    }
    report_path = context.paths.verification / "verification_report.yaml"
    save_yaml(report_path, report, header=(
        f"Verification of {result_path.name} for {context.camera_name}.\n"
        f"Board points projected through the calibration chain and compared "
        f"with the corners actually detected."))
    print()
    print(f"Saved: {report_path}")
    if not args.no_images:
        print(f"Overlays: {context.paths.verification}")
    return status


def verify_live(context, transform, reference, mode, result_path) -> int:
    """Live check against the current frame and the current robot pose."""
    from camera_interface import open_camera
    from robot_interface import RobotInterface
    from safety import SafetyError, load_envelope

    print("LIVE VERIFICATION. This reads the robot but never commands it.")
    print("Move the arm by hand (pendant freedrive) and watch the overlay.")
    print("Q or ESC to quit.")
    print()

    envelope = load_envelope()
    camera = robot = None
    try:
        camera = open_camera(context.camera_config)
        robot = RobotInterface(envelope).connect()
        window = f"Live verification -- {context.camera_name}"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            frame = camera.read(flush=0)
            state = robot.read_state()
            detection = context.detector.process(
                frame.image, context.camera_matrix, context.dist_coeffs,
                require_pose=True)
            canvas = context.detector.annotate(
                frame.image, detection, context.camera_matrix, context.dist_coeffs)

            lines = [(f"LIVE VERIFICATION -- {context.camera_name}", ui.WHITE, 0.6),
                     (f"{result_path.name}", ui.GREY, 0.45)]
            if not state.connected:
                lines.append(("robot not connected", ui.RED, 0.55))
            elif not detection.valid:
                lines.append(("board not valid: "
                              + (detection.reasons[0] if detection.reasons else ""),
                              ui.RED, 0.5))
            else:
                predicted = predicted_board_in_camera(
                    transform, state.tcp_transform if hasattr(state, "tcp_transform")
                    else __import__("calibration_utils").pose_to_matrix(state.tcp),
                    reference, mode)
                projected, _ = cv2.projectPoints(
                    detection.object_points,
                    matrix_to_rotvec(predicted[:3, :3]).reshape(3, 1),
                    predicted[:3, 3].reshape(3, 1),
                    context.camera_matrix, context.dist_coeffs)
                projected = projected.reshape(-1, 2)
                errors = np.linalg.norm(projected - detection.image_points, axis=1)
                for point in projected:
                    cv2.drawMarker(canvas, tuple(np.int32(np.round(point))),
                                   ui.MAGENTA, cv2.MARKER_CROSS, 8, 1, cv2.LINE_AA)
                mean = float(np.mean(errors))
                lines += [
                    (f"mean error {mean:7.3f} px", ui.color_for(
                        "GOOD" if mean < 2 else "MEDIUM" if mean < 5 else "BAD"), 0.7),
                    (f"max  error {np.max(errors):7.3f} px", ui.GREY, 0.5),
                    (f"{detection.tags_detected} tags", ui.GREY, 0.45),
                ]
            ui.panel(canvas, lines, width=420)
            ui.footer(canvas, "magenta cross = predicted by the calibration",
                      "Q/ESC quit")
            cv2.imshow(window, ui.fit_to_screen(canvas))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        return 0
    except (SafetyError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        if robot is not None:
            robot.disconnect()
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
