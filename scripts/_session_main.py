"""Shared entry point for the three interactive scripts.

keyboard_teleop_calibration.py, gamepad_teleop_calibration.py and
collect_waypoints.py differ only in their default input device and whether
recording is on, so the whole flow lives here exactly once.
"""
from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, ConfigError, setup_logging)
from camera_interface import CameraError
from collection_session import CollectionSession
from input_devices import GamepadInput, KeyboardInput
from safety import SafetyError
from session_setup import (capture_center, confirm_and_arm,
                           connect_robot_readonly, load_context,
                           open_camera_checked, prepare_output, print_header)
from waypoint_recorder import (WaypointRecorder, load_waypoints,
                               save_initial_center)


def add_common_arguments(parser) -> None:
    parser.add_argument("--camera", required=True,
                        help="logical camera name, e.g. camera_1")
    parser.add_argument("--target-count", type=int, default=None,
                        help="number of valid waypoints to collect")
    parser.add_argument("--input", choices=["keyboard", "gamepad"], default=None,
                        help="override the input device")
    parser.add_argument("--read-only", action="store_true",
                        help="never send motion commands; move the arm by hand")
    parser.add_argument("--force", action="store_true",
                        help="archive any existing waypoint set without asking")
    parser.add_argument("--resume", action="store_true",
                        help="add to the existing waypoint set")


def run(args, title: str, default_device: str, allow_recording: bool) -> int:
    logger = setup_logging(title.lower().replace(" ", "_"), args.camera)
    device_name = args.input or default_device

    try:
        context = load_context(args.camera, require_intrinsics=True)
    except (ConfigError, CalibrationError) as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    collection = context.calibration_config.get("waypoint_collection", {})
    target = args.target_count or int(collection.get("target_count", 30))

    print_header(context, title)

    if allow_recording:
        try:
            prepare_output(context.paths, args.force, args.resume, logger)
        except SystemExit as exc:
            print(exc)
            return 1

    device = None
    try:
        # Camera first: a camera failure must never reach a robot that is armed.
        open_camera_checked(context)
        print(f"Camera open: {context.camera.resolution[0]}x"
              f"{context.camera.resolution[1]}")
        print()

        connect_robot_readonly(context)

        armed = confirm_and_arm(context, require_motion=not args.read_only)
        capture_center(context)

        if device_name == "gamepad":
            device = GamepadInput(context.jog_state,
                                  context.envelope.gamepad).open()
            print(f"Gamepad: {device.describe()}")
            print("HOLD LB (deadman) for any motion. Release it to stop.")
        else:
            device = KeyboardInput(context.jog_state)
            print("Keyboard input. Keys register only while the preview window "
                  "has focus.")
        print()
        for line in device.help_lines():
            print(f"  {line}")
        print()

        recorder = None
        if allow_recording:
            recorder = WaypointRecorder(
                context.paths, context.camera, context.detector, context.robot,
                context.calibration_config, context.camera_matrix,
                context.dist_coeffs, context.intrinsics)
            if args.resume:
                for record in load_waypoints(context.paths, logger):
                    recorder.records.append(record)
                    context.analyzer.add(record.number, record.actual_q,
                                         record.actual_tcp)
            save_initial_center(context.paths, context.camera,
                                context.robot.read_state(), context.envelope)

        session = CollectionSession(
            camera=context.camera, detector=context.detector,
            robot=context.robot, device=device, jog_state=context.jog_state,
            analyzer=context.analyzer, recorder=recorder,
            config=context.calibration_config,
            camera_matrix=context.camera_matrix, dist_coeffs=context.dist_coeffs,
            target_count=target, allow_recording=allow_recording, logger=logger)
        if recorder is not None:
            session.recorded = len(recorder.records)

        result = session.run(f"{title} -- {args.camera}")

        if recorder is not None:
            recorder.save_master(context.robot.calibration_center_q,
                                 context.robot.calibration_center_tcp,
                                 extra={"armed": armed,
                                        "input_device": device_name,
                                        "handeye_mode": context.handeye_mode})
        print()
        print("=" * 74)
        print(f"Session ended: {result.stopped_reason}")
        if allow_recording:
            print(f"Recorded {result.recorded} / {target} valid waypoints")
            print(f"Rejected attempts: {session.rejected_attempts}")
            summary = context.analyzer.summary()
            print(f"Overall pose diversity: {summary['overall']}")
            if summary["weakest_axes"]:
                print(f"Weakest axes: {', '.join(summary['weakest_axes'])}")
            travel = summary["travel"]
            print(f"Total travel: {travel['joint_deg']:.1f} deg of joint motion, "
                  f"{travel['tcp_mm']:.0f} mm of TCP motion")
            print()
            print(f"Waypoints : {context.paths.waypoints_file}")
            if result.recorded >= target:
                print()
                print(f"Next: python scripts/analyze_waypoints.py "
                      f"--camera {args.camera}")
            else:
                print()
                print(f"Re-run with --resume to add the remaining "
                      f"{max(0, target - result.recorded)}.")
        print("=" * 74)
        return result.exit_code

    except (SafetyError, CameraError, CalibrationError, ConfigError) as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    except RuntimeError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 0
    finally:
        if device is not None:
            device.close()
        context.close()
