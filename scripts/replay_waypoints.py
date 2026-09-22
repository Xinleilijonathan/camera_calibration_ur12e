#!/usr/bin/env python3
"""Replay recorded waypoints. DRY RUN BY DEFAULT.

    python scripts/replay_waypoints.py --camera camera_1 --dry-run
    python scripts/replay_waypoints.py --camera camera_1 --enable-motion

Without --enable-motion this prints what WOULD happen and sends nothing. Real
motion additionally requires connection.allow_motion: true in safety.yaml, the
verified-limits flags, and a typed confirmation -- three independent gates.

Replay uses the SAVED JOINT CONFIGURATIONS with moveJ, not fresh IK from the
saved TCP pose. The recorded joints are the configuration the arm was actually
in; re-solving IK can return a different branch that reaches the same TCP pose
through a completely different arm posture.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

import _bootstrap  # noqa: F401
import _analysis

from calibration_utils import (JOINT_NAMES, CalibrationError, ConfigError,
                               setup_logging, transform_difference)
from robot_interface import RobotInterface
from safety import SafetyError, load_envelope
from session_setup import CONFIRMATION_PHRASE


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay recorded waypoints. Dry run unless --enable-motion.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="explicit dry run (this is the default anyway)")
    parser.add_argument("--enable-motion", action="store_true",
                        help="ACTUALLY MOVE THE ROBOT (requires confirmation)")
    parser.add_argument("--selection", default=None,
                        help="replay only the selected set, e.g. best20")
    parser.add_argument("--only", nargs="*", type=int, default=None,
                        metavar="N", help="replay only these waypoint numbers")
    parser.add_argument("--pause", type=float, default=1.0,
                        help="seconds to hold at each waypoint")
    args = parser.parse_args(argv)

    logger = setup_logging("replay_waypoints", args.camera)
    # Dry run unless motion is explicitly requested. --dry-run is accepted for
    # clarity but changes nothing: it is already the default.
    execute = bool(args.enable_motion) and not args.dry_run

    try:
        context = _analysis.load_for_analysis(args.camera, require_verified=False,
                                              logger=logger)
    except (ConfigError, CalibrationError) as exc:
        return _analysis.fail(exc)

    records = context.records
    if args.selection:
        from calibration_utils import load_yaml
        candidates = sorted(context.paths.selection.glob("selected_*.yaml"))
        if not candidates:
            print(f"ERROR: no selection file in {context.paths.selection}",
                  file=sys.stderr)
            return 1
        numbers = set(int(n) for n in load_yaml(candidates[0]).get("selected_numbers", []))
        records = [r for r in records if r.number in numbers]
    if args.only:
        records = [r for r in records if r.number in set(args.only)]
    if not records:
        print("ERROR: no waypoints match the requested filter", file=sys.stderr)
        return 1

    envelope = load_envelope()
    print("=" * 78)
    print(f"WAYPOINT REPLAY -- {args.camera}")
    print("=" * 78)
    print(f"Mode        : {'*** LIVE MOTION ***' if execute else 'DRY RUN (nothing is sent)'}")
    print(f"Waypoints   : {len(records)}")
    print(f"Target      : {envelope.robot_ip}"
          f"{'  (URSim)' if envelope.is_simulator else '  *** PHYSICAL ROBOT ***'}")
    print(f"moveJ speed : {envelope.motion.joint_speed:.3f} rad/s, "
          f"accel {envelope.motion.joint_acceleration:.3f} rad/s^2")
    print()

    if execute:
        # Refuse on POLICY before touching hardware. Whether motion is allowed
        # does not depend on whether a robot happens to be plugged in, and the
        # operator should get the real reason, not a connection error.
        if not envelope.allow_motion:
            print("MOTION IS DISABLED in config/safety.yaml "
                  "(connection.allow_motion: false).", file=sys.stderr)
            print("Nothing was sent and no connection was attempted.",
                  file=sys.stderr)
            return 1
        unverified = envelope.unverified_sections()
        if unverified:
            print("REFUSING TO MOVE. Verify these for this cell first:",
                  file=sys.stderr)
            for section in unverified:
                print(f"  * {section}", file=sys.stderr)
            return 1

    robot = None
    try:
        # A DRY RUN MUST NOT REQUIRE A ROBOT. Reviewing the plan is exactly the
        # thing you want to do with the arm powered off, so the connection is
        # attempted but never required, and its absence is not an error.
        start_q = None
        if execute:
            robot = RobotInterface(envelope).connect()
            state = robot.read_state()
            start_q = state.q
            print("Current robot state:")
            print("  joints (deg): "
                  + "  ".join(f"{v:8.3f}" for v in np.degrees(state.q)))
            print("  TCP         : " + "  ".join(f"{v:8.4f}" for v in state.tcp))
            print()
        else:
            try:
                robot = RobotInterface(envelope).connect()
                state = robot.read_state()
                if state.connected and state.q is not None:
                    start_q = state.q
                    print("Current robot state (read-only):")
                    print("  joints (deg): "
                          + "  ".join(f"{v:8.3f}" for v in np.degrees(state.q)))
                    print()
            except SafetyError as exc:
                logger.info("No robot available for the dry run: %s", exc)
            if start_q is None:
                print("No robot connected. Planning from the FIRST recorded")
                print("waypoint instead, so the first transition is not shown.")
                print()
                start_q = records[0].actual_q

        plan = build_plan(records, start_q, envelope)
        print_plan(plan, envelope)

        if not execute:
            print()
            print("DRY RUN: nothing was sent to the robot.")
            print("To move for real, add --enable-motion (and read the warnings "
                  "above first).")
            return 0

        if plan["oversized"]:
            print()
            print(f"REFUSING TO REPLAY: {len(plan['oversized'])} transition(s) "
                  f"exceed the "
                  f"{np.degrees(envelope.motion.max_replay_joint_delta):.0f} deg "
                  f"per-move limit.", file=sys.stderr)
            print("Move the arm near the first waypoint by hand, or replay a "
                  "subset with --only.", file=sys.stderr)
            return 1

        print()
        print("-" * 78)
        print("ABOUT TO MOVE THE ROBOT THROUGH "
              f"{len(records)} RECORDED CONFIGURATIONS")
        print("-" * 78)
        print("  The area must be clear and the e-stop within reach.")
        answer = input(f"  Type {CONFIRMATION_PHRASE} to proceed: ").strip()
        if answer != CONFIRMATION_PHRASE:
            print("  Cancelled. Nothing was sent.")
            return 0

        robot.enable_motion(confirm=True)
        # The replay leaves the recorded neighbourhood by design, so the
        # per-session centre leash does not apply here. Absolute joint limits,
        # the workspace box and the per-move size cap all still do.
        robot.calibration_center_q = None

        for index, step in enumerate(plan["steps"], start=1):
            record = step["record"]
            print(f"[{index}/{len(plan['steps'])}] {record.name}  "
                  f"max joint delta {np.degrees(step['max_delta']):.2f} deg ...",
                  end="", flush=True)
            robot.move_joints(record.actual_q, check_step=False)
            robot.wait_until_stationary()
            reached = robot.read_state()
            error = np.degrees(np.max(np.abs(reached.q - record.actual_q)))
            translation, rotation = transform_difference(
                record.tcp_transform, __import__("calibration_utils").pose_to_matrix(
                    reached.tcp))
            print(f" reached (joint error {error:.3f} deg, "
                  f"TCP {translation * 1000:.2f} mm)")
            logger.info("Replayed %s: joint error %.4f deg, TCP error %.3f mm",
                        record.name, error, translation * 1000)
            time.sleep(max(0.0, args.pause))

        print()
        print("Replay complete.")
        return 0

    except SafetyError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the robot.")
        if robot is not None:
            robot.emergency_software_stop()
        return 0
    finally:
        if robot is not None:
            robot.disconnect()


def build_plan(records, current_q, envelope) -> dict:
    """Work out every transition and flag the large ones BEFORE moving."""
    steps = []
    oversized = []
    previous = np.asarray(current_q, dtype=np.float64)
    for record in records:
        deltas = np.asarray(record.actual_q, dtype=np.float64) - previous
        max_delta = float(np.max(np.abs(deltas)))
        step = {"record": record, "deltas": deltas, "max_delta": max_delta,
                "from": previous.copy()}
        steps.append(step)
        if max_delta > envelope.motion.max_replay_joint_delta:
            oversized.append(step)
        previous = np.asarray(record.actual_q, dtype=np.float64)
    return {"steps": steps, "oversized": oversized}


def print_plan(plan, envelope) -> None:
    limit = envelope.motion.max_replay_joint_delta
    print(f"{'ID':>4}  {'joint targets (deg)':<58}  {'max move':>9}  flag")
    print("-" * 92)
    for step in plan["steps"]:
        record = step["record"]
        joints = "  ".join(f"{v:7.2f}" for v in np.degrees(record.actual_q))
        flag = "TOO LARGE" if step["max_delta"] > limit else ""
        print(f"{record.number:>4}  {joints:<58}  "
              f"{np.degrees(step['max_delta']):>8.2f}  {flag}")
    print("-" * 92)
    print()
    print("Recorded TCP poses (x y z in mm, rx ry rz axis-angle in rad):")
    for step in plan["steps"][:5]:
        tcp = step["record"].actual_tcp
        print(f"  {step['record'].number:>4}  "
              f"{tcp[0] * 1000:8.2f} {tcp[1] * 1000:8.2f} {tcp[2] * 1000:8.2f}   "
              f"{tcp[3]:+.4f} {tcp[4]:+.4f} {tcp[5]:+.4f}")
    if len(plan["steps"]) > 5:
        print(f"  ... and {len(plan['steps']) - 5} more")
    if plan["oversized"]:
        print()
        print(f"WARNING: {len(plan['oversized'])} transition(s) exceed the "
              f"{np.degrees(limit):.0f} deg per-move limit:")
        for step in plan["oversized"]:
            worst = int(np.argmax(np.abs(step["deltas"])))
            print(f"  -> {step['record'].name}: {JOINT_NAMES[worst]} moves "
                  f"{np.degrees(step['deltas'][worst]):+.1f} deg")


if __name__ == "__main__":
    sys.exit(main())
