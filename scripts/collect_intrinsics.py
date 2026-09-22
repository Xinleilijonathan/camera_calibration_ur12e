#!/usr/bin/env python3
"""Collect intrinsic calibration observations for ONE camera. No robot code.

    python scripts/collect_intrinsics.py --camera camera_1

You hold the printed AprilTag board and move it by hand. The preview shows
whether the current view is acceptable and what kind of variety the set still
needs. SPACE records an observation; only valid, genuinely novel views count.

This script never touches the robot. For camera_1 (the wrist-mounted D405) you
can either hold the board and move it, or leave the board still and move the
arm by hand with the pendant in freedrive -- this program does not care and
does not command the arm either way.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401

import ui_overlay as ui
from apriltag_detector import build_detector
from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               describe_environment, load_calibration_config,
                               load_cameras_config, resolve_camera, save_yaml,
                               setup_logging, timestamp_slug, timestamp_utc)
from camera_interface import CameraError, open_camera
from intrinsic_calibration import (IntrinsicDiversityTracker,
                                   IntrinsicObservation)


def confirm_overwrite(directory: Path, force: bool, logger) -> None:
    """Never silently destroy an existing collection (section AP)."""
    existing = sorted(directory.glob("*.yaml"))
    if not existing:
        return
    print()
    print(f"WARNING: {directory} already holds {len(existing)} observation(s).")
    if force:
        print("--force given: archiving the old set rather than deleting it.")
        answer = "a"
    else:
        print("  [a] archive the old set to a timestamped folder and start fresh")
        print("  [r] resume, adding to the existing set")
        print("  [q] quit")
        answer = input("Choose [a/r/q]: ").strip().lower()

    if answer == "q":
        raise SystemExit("Cancelled; nothing was changed.")
    if answer == "a":
        archive = directory.parent / "sessions" / timestamp_slug()
        archive.mkdir(parents=True, exist_ok=True)
        for path in existing:
            shutil.move(str(path), str(archive / path.name))
        images = directory.parent / "images"
        if images.is_dir():
            archive_images = archive / "images"
            archive_images.mkdir(exist_ok=True)
            for image in images.glob("*.png"):
                shutil.move(str(image), str(archive_images / image.name))
        print(f"Archived to {archive}")
        logger.info("Archived previous intrinsic set to %s", archive)
    elif answer != "r":
        raise SystemExit("Unrecognised choice; nothing was changed.")


def load_existing(paths, logger) -> list[IntrinsicObservation]:
    """Reload a previous partial session so collection can be resumed."""
    observations = []
    from calibration_utils import load_yaml
    for path in sorted(paths.intrinsics_observations.glob("observation_*.yaml")):
        try:
            observations.append(IntrinsicObservation.from_dict(load_yaml(path)))
        except Exception as exc:
            logger.warning("Ignoring unreadable observation %s: %s", path, exc)
    if observations:
        print(f"Resuming with {len(observations)} existing observation(s).")
    return observations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect intrinsic calibration views for one camera "
                    "(camera only; never moves the robot).")
    parser.add_argument("--camera", required=True,
                        help="logical camera name, e.g. camera_1")
    parser.add_argument("--target-count", type=int, default=None,
                        help="override intrinsics.target_observations")
    parser.add_argument("--force", action="store_true",
                        help="archive any existing set without asking")
    parser.add_argument("--allow-similar", action="store_true",
                        help="permit recording views the diversity check rejects")
    args = parser.parse_args(argv)

    logger = setup_logging("collect_intrinsics", args.camera)

    try:
        cameras_config = load_cameras_config()
        calibration_config = load_calibration_config()
        camera_config = resolve_camera(args.camera, cameras_config)
        detector = build_detector(calibration_config)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    intrinsics_config = calibration_config.get("intrinsics", {})
    target = args.target_count or int(intrinsics_config.get("target_observations", 30))
    minimum = int(intrinsics_config.get("minimum_observations", 20))
    maximum = int(intrinsics_config.get("maximum_observations", 40))

    paths = camera_paths(args.camera)
    paths.ensure()

    print("=" * 74)
    print(f"INTRINSIC COLLECTION -- {args.camera}")
    print("=" * 74)
    print(f"Board   : {detector.spec.describe()}")
    print(f"Target  : {target} observations (minimum {minimum}, maximum {maximum})")
    print(f"Output  : {paths.intrinsics}")
    print()
    print("This script does not move the robot and contains no robot code.")
    print()
    print("Vary the board deliberately:")
    print("  * move it into all four corners of the frame, not just the centre")
    print("  * tilt it well past 20 degrees, in several directions")
    print("  * take some views close and some far away")
    print("Thirty near-identical views produce a confident, precise, wrong result.")
    print()
    print("Keys:  SPACE record   U undo last   Q/ESC finish   H help")
    print()

    try:
        confirm_overwrite(paths.intrinsics_observations, args.force, logger)
    except SystemExit as exc:
        print(exc)
        return 1

    camera = None
    try:
        camera = open_camera(camera_config)
    except (CameraError, ConfigError) as exc:
        print(f"CAMERA ERROR: {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1

    width, height = camera.resolution
    tracker = IntrinsicDiversityTracker(intrinsics_config, (width, height))
    for observation in load_existing(paths, logger):
        tracker.add(observation)
    next_index = max((o.index for o in tracker.observations), default=0) + 1

    window = f"Intrinsic collection -- {args.camera}"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    frame_rate = ui.FrameRate()
    message = ""
    message_until = 0.0
    show_help = False
    status = 0

    try:
        while True:
            now = time.time()
            try:
                frame = camera.read(flush=0)
            except CameraError as exc:
                print(f"\nCAMERA FAILURE: {exc}", file=sys.stderr)
                logger.error("Camera failure: %s", exc)
                status = 1
                break

            if (frame.width, frame.height) != (width, height):
                width, height = frame.width, frame.height
                tracker.image_size = (width, height)

            # No intrinsics yet, so no metric pose. A guess is enough to rate
            # tilt and distance for guidance: a pinhole at the nominal focal
            # length, which is honest about being provisional and is never
            # written to disk or used in the solve.
            guess = np.array([[float(width), 0, width / 2.0],
                              [0, float(width), height / 2.0],
                              [0, 0, 1.0]])
            detection = detector.process(frame.image, guess, np.zeros(5),
                                         require_pose=False)
            canvas = detector.annotate(frame.image, detection, guess, np.zeros(5))

            novel, novelty_reason = (True, "")
            if detection.valid and detection.has_pose:
                novel, novelty_reason = tracker.is_novel(detection)

            recordable = detection.valid and (novel or args.allow_similar)
            report = tracker.report()
            fps = frame_rate.tick(now)

            lines = [
                (f"CAMERA {args.camera}  [{camera.serial}]", ui.WHITE, 0.55),
                (f"{width} x {height}   {fps:4.1f} fps", ui.GREY, 0.45),
                ("", ui.WHITE, 0.3),
                ("RECORDABLE" if recordable else "NOT RECORDABLE",
                 ui.GREEN if recordable else ui.RED, 0.8),
                (f"Tags {detection.tags_detected}/{detector.spec.tag_count}"
                 f"   Corners {detection.corners_detected}", ui.WHITE, 0.5),
            ]
            if detection.tags_detected:
                lines.append((f"Sharpness {detection.sharpness:5.0f}"
                              f"  (min {detector.min_sharpness:.0f})",
                              ui.AMBER if detection.sharpness < detector.min_sharpness
                              else ui.GREY, 0.45))
                if detection.tilt_deg is not None:
                    lines.append((f"Tilt {detection.tilt_deg:4.1f} deg"
                                  f"   Area {detection.board_area_fraction * 100:4.1f}%",
                                  ui.GREY, 0.45))
            for reason in detection.reasons[:2]:
                lines.append((f"! {reason}", ui.RED, 0.45))
            if detection.valid and not novel:
                lines.append((f"! {novelty_reason}", ui.AMBER, 0.45))

            lines += [
                ("", ui.WHITE, 0.3),
                (f"COLLECTED {len(tracker.observations)} / {target}", ui.WHITE, 0.6),
                (f"Image cells {report['cells_covered']}/{report['cells_total']}"
                 f"   Tilted {report['tilted_views']}"
                 f"   Scale x{report['scale_ratio']:.2f}", ui.GREY, 0.45),
            ]
            if report["missing"]:
                lines.append(("STILL NEEDED:", ui.AMBER, 0.5))
                for item in report["missing"][:3]:
                    for chunk in _wrap(item, 52):
                        lines.append((f"  {chunk}", ui.AMBER, 0.42))
            else:
                lines.append(("Diversity requirements satisfied.", ui.GREEN, 0.5))

            ui.panel(canvas, lines, width=470)
            if detection.center_px:
                ui.draw_reticle(canvas, detection.center_px)
            ui.coverage_grid(canvas, tracker.coverage_grid(),
                             (canvas.shape[1] - 130, 20))

            if now < message_until and message:
                ui.banner(canvas, message, ui.GREEN if "SAVED" in message else ui.AMBER,
                          0.85, canvas.shape[0] - 70)
            ui.footer(canvas, "SPACE record   U undo   Q/ESC finish   H help",
                      f"{len(tracker.observations)} saved")
            if show_help:
                _draw_help(canvas)

            cv2.imshow(window, ui.fit_to_screen(canvas))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break
            if key == ord("h"):
                show_help = not show_help
            elif key == ord("u"):
                message = _undo(tracker, paths, logger)
                message_until = now + 2.0
            elif key == ord(" "):
                if not recordable:
                    message = "CANNOT RECORD: " + (
                        detection.reasons[0] if detection.reasons else novelty_reason)
                    message_until = now + 2.5
                    logger.info("Record refused: %s", message)
                elif len(tracker.observations) >= maximum:
                    message = f"MAXIMUM {maximum} REACHED"
                    message_until = now + 2.5
                else:
                    saved = _record(camera, detector, tracker, paths,
                                    next_index, logger)
                    if saved:
                        next_index += 1
                        message = f"SAVED {saved}"
                    else:
                        message = "CAPTURE REJECTED ON RE-CHECK"
                    message_until = now + 1.5

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if camera is not None:
            camera.close()
        cv2.destroyAllWindows()

    count = len(tracker.observations)
    print()
    print("=" * 74)
    print(f"Collected {count} observation(s) for {args.camera}")
    report = tracker.report()
    for item in report["missing"]:
        print(f"  STILL MISSING: {item}")
    if count < minimum:
        print(f"  WARNING: below the configured minimum of {minimum}.")
        status = status or 1
    if count and report["satisfied"] and count >= minimum:
        print(f"\nNext: python scripts/solve_intrinsics.py --camera {args.camera}")
    print("=" * 74)
    return status


def _wrap(message: str, width: int) -> list[str]:
    """Wrap a guidance string to fit the HUD panel."""
    words, lines, current = message.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return lines[:2]


def _record(camera, detector, tracker, paths, index, logger) -> str | None:
    """Flush, re-capture, re-validate, then save image and metadata atomically.

    The frame is grabbed again with a flush rather than reusing the previewed
    one: the preview runs without flushing to stay responsive, so its frame can
    be a buffered image from before the board stopped moving.
    """
    try:
        frame = camera.read()
    except CameraError as exc:
        logger.error("Capture failed: %s", exc)
        return None

    guess = np.array([[float(frame.width), 0, frame.width / 2.0],
                      [0, float(frame.width), frame.height / 2.0],
                      [0, 0, 1.0]])
    detection = detector.process(frame.image, guess, np.zeros(5), require_pose=False)
    if not detection.valid:
        logger.info("Re-check after flush rejected the frame: %s", detection.reasons)
        return None

    object_points, image_points = detector.board.matchImagePoints(
        [c.reshape(1, 4, 2) for c in detection.corners],
        detection.ids.reshape(-1, 1))
    if object_points is None or len(object_points) < 4:
        logger.info("Could not match board points; not recording")
        return None

    name = f"observation_{index:03d}"
    image_name = f"{name}.png"
    image_path = paths.intrinsics_images / image_name
    if not cv2.imwrite(str(image_path), frame.image):
        logger.error("Could not write %s", image_path)
        return None

    observation = IntrinsicObservation(
        index=index,
        image_name=image_name,
        object_points=np.asarray(object_points, dtype=np.float64).reshape(-1, 3),
        image_points=np.asarray(image_points, dtype=np.float64).reshape(-1, 2),
        tag_ids=[int(i) for i in detection.ids.ravel()],
        image_size=(frame.width, frame.height),
        timestamp=timestamp_utc(),
        detection={**detection.summary(),
                   "rvec": detection.rvec.tolist() if detection.has_pose else None,
                   "tvec": detection.tvec.tolist() if detection.has_pose else None,
                   "pose_is_provisional": True},
        cell=tracker.cell_for(detection.center_px) if detection.center_px else None,
        tilt_deg=detection.tilt_deg,
        distance_m=detection.distance_m,
        area_fraction=detection.board_area_fraction,
    )
    record = observation.to_dict()
    record["camera_name"] = camera.name
    record["camera_serial"] = camera.serial
    record["camera"] = camera.describe()
    record["environment"] = describe_environment()
    save_yaml(paths.intrinsics_observations / f"{name}.yaml", record, header=(
        f"Intrinsic observation {index} for {camera.name} "
        f"(serial {camera.serial}).\n"
        f"rvec/tvec here are PROVISIONAL, from a nominal focal-length guess "
        f"used only\nfor live guidance. They are not used by the solver."))

    tracker.add(observation)
    logger.info("Saved %s (%d points, tilt %.1f deg)", name,
                len(observation.object_points), observation.tilt_deg or 0.0)
    return name


def _undo(tracker, paths, logger) -> str:
    """Remove the most recent observation, image and metadata together."""
    if not tracker.observations:
        return "NOTHING TO UNDO"
    observation = tracker.observations.pop()
    name = f"observation_{observation.index:03d}"
    for path in (paths.intrinsics_observations / f"{name}.yaml",
                 paths.intrinsics_images / observation.image_name):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove %s: %s", path, exc)
    logger.info("Undid %s", name)
    return f"REMOVED {name}"


def _draw_help(canvas) -> None:
    lines = [
        ("KEYS", ui.WHITE, 0.6),
        ("SPACE  record the current view", ui.GREY, 0.5),
        ("U      undo the last recording", ui.GREY, 0.5),
        ("H      toggle this help", ui.GREY, 0.5),
        ("Q/ESC  finish and exit", ui.GREY, 0.5),
        ("", ui.WHITE, 0.3),
        ("The green grid, top right, is where the board", ui.GREY, 0.45),
        ("centre has been. Fill every cell.", ui.GREY, 0.45),
    ]
    ui.panel(canvas, lines, origin=(canvas.shape[1] - 430, canvas.shape[0] - 250),
             width=420, alpha=0.75)


if __name__ == "__main__":
    sys.exit(main())
