#!/usr/bin/env python3
"""Collect 30 VALID hand-eye waypoints for ONE camera.

    python scripts/collect_waypoints.py --camera camera_1 --target-count 30
    python scripts/collect_waypoints.py --camera camera_1 --input gamepad

You move the robot. The program watches the board, tells you whether the
current pose is worth recording, and suggests what to change next. It never
generates or executes robot poses by itself.

Only valid observations count toward the target. Every recorded waypoint stores
the robot's ACTUAL measured joints and TCP, captured after the arm has been
verified stationary and allowed to settle.
"""
import argparse
import sys

import _bootstrap  # noqa: F401
import _session_main as session_main


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Guided manual collection of hand-eye calibration waypoints.")
    session_main.add_common_arguments(parser)
    args = parser.parse_args(argv)
    if args.input is None:
        args.input = "keyboard"
    return session_main.run(args, "HAND-EYE COLLECTION", args.input,
                            allow_recording=True)


if __name__ == "__main__":
    sys.exit(main())
