#!/usr/bin/env python3
"""Keyboard jog of the UR12e while watching the AprilTag board.

    python scripts/keyboard_teleop_calibration.py --camera camera_1

Full waypoint recording is available here too, so this and collect_waypoints.py
are the same tool with different defaults. Motion requires typed confirmation
at startup AND connection.allow_motion: true in config/safety.yaml.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
import _session_main as session_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for calibration, with live "
                    "AprilTag preview.")
    session_main.add_common_arguments(parser)
    parser.add_argument("--no-record", action="store_true",
                        help="framing only; disable waypoint recording")
    args = parser.parse_args(argv)
    return session_main.run(args, "KEYBOARD TELEOP", "keyboard",
                            allow_recording=not args.no_record)


if __name__ == "__main__":
    sys.exit(main())
