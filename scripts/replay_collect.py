#!/usr/bin/env python3
"""Replay recorded waypoints AND record a fresh observation at each. DRY RUN BY DEFAULT.

    python scripts/replay_collect.py --camera camera_3
    python scripts/replay_collect.py --camera camera_3 --enable-motion
    python scripts/replay_collect.py --camera camera_3 --enable-motion --resume \
        --targets data/camera_3/handeye/waypoints/planned_extra.yaml

This is the automated version of collect_waypoints.py for re-calibrating a
camera that moved: the arm revisits the SAVED JOINT CONFIGURATIONS of an
earlier collection with moveJ, and at each one the normal WaypointRecorder
procedure (stationary check, settle, flushed capture, re-detection) writes a
new observation. The new set is written where collect_waypoints.py writes, so
check_board_rigid.py and solve_handeye.py run on it unchanged.

Transitions larger than motion.max_replay_joint_delta_rad are split into
equal joint-space segments that each respect the limit; the limit itself is
not raised. Joint-space interpolation between two recorded configurations
stays on the same IK branch, but nothing here checks the swept volume for
collisions: watch the arm with the e-stop in hand.

Motion requires the same three gates as replay_waypoints.py:
connection.allow_motion in safety.yaml, the verified-limits flags, and a typed
confirmation. A waypoint whose board is not usable (clipped, too few tags) is
skipped and reported, never retried in a loop.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, ConfigError, JOINT_NAMES, load_yaml,
                               setup_logging)
from safety import SafetyError
from session_setup import (CONFIRMATION_PHRASE, load_context, open_camera_checked,
                           prepare_output, print_header)
from waypoint_recorder import (RecordingRejected, WaypointRecord, WaypointRecorder,
                               load_waypoints, save_initial_center)


def split_transition(start, end, max_delta: float) -> list[np.ndarray]:
    """Joint targets from `start` to `end`, no segment larger than `max_delta`.

    The last target is exactly `end`. A zero-length move returns [end].
    """
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    if max_delta <= 0:
        raise ValueError("max_delta must be positive")
    largest = float(np.max(np.abs(end - start)))
    count = max(1, math.ceil(largest / max_delta - 1e-9))
    # The final target is `end` itself, not start + 1.0 * (end - start), which
    # can differ from the recorded configuration in the last bit.
    return [start + (end - start) * (i / count) for i in range(1, count)] + [end.copy()]


def build_segments(records, start_q, max_delta: float) -> list[dict]:
    """One entry per waypoint: its segment targets and the largest joint move."""
    plan, previous = [], np.asarray(start_q, dtype=np.float64)
    for record in records:
        target = np.asarray(record.actual_q, dtype=np.float64)
        deltas = target - previous
        plan.append({"record": record,
                     "targets": split_transition(previous, target, max_delta),
                     "max_delta": float(np.max(np.abs(deltas))),
                     "worst_joint": JOINT_NAMES[int(np.argmax(np.abs(deltas)))]})
        previous = target
    return plan


@dataclass
class PlannedTarget:
    """A joint target from plan_extra_waypoints.py; quacks like a WaypointRecord here."""
    number: int
    actual_q: np.ndarray

    @property
    def name(self) -> str:
        return f"planned_{self.number:03d}"


def load_targets(path: str) -> list[PlannedTarget]:
    entries = load_yaml(Path(path).expanduser()).get("targets") or []
    return [PlannedTarget(number, np.asarray(entry["joints_rad"], dtype=np.float64))
            for number, entry in enumerate(entries, start=1)]


def load_source(context, source: str | None, logger) -> list[WaypointRecord]:
    """Waypoints to revisit: the current set, or an archived observations folder."""
    if source is None:
        return load_waypoints(context.paths, logger)
    folder = Path(source).expanduser()
    if (folder / "observations").is_dir():
        folder = folder / "observations"
    records = [WaypointRecord.from_dict(load_yaml(path))
               for path in sorted(folder.glob("waypoint_*.yaml"))]
    return sorted(records, key=lambda r: r.number)


def print_plan(plan, max_delta: float) -> None:
    moves = sum(len(step["targets"]) for step in plan)
    print(f"{'ID':>4}  {'joint targets (deg)':<58}  {'max move':>9}  segments")
    print("-" * 92)
    for step in plan:
        record = step["record"]
        joints = "  ".join(f"{v:7.2f}" for v in np.degrees(record.actual_q))
        note = (f"{len(step['targets'])}  ({step['worst_joint']})"
                if len(step["targets"]) > 1 else "1")
        print(f"{record.number:>4}  {joints:<58}  "
              f"{np.degrees(step['max_delta']):>8.2f}  {note}")
    print("-" * 92)
    print(f"{len(plan)} waypoints, {moves} moveJ commands, every segment "
          f"<= {np.degrees(max_delta):.0f} deg")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay waypoints and record a new observation at each. "
                    "Dry run unless --enable-motion.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--source", default=None,
                        help="folder of waypoint_*.yaml to revisit (default: the "
                             "current set, which is archived before recording)")
    parser.add_argument("--targets", default=None,
                        help="planned_extra.yaml from plan_extra_waypoints.py "
                             "(visit these instead of recorded waypoints)")
    parser.add_argument("--resume", action="store_true",
                        help="add to the existing set instead of archiving it")
    parser.add_argument("--only", nargs="*", type=int, default=None, metavar="N",
                        help="revisit only these waypoint numbers")
    parser.add_argument("--enable-motion", action="store_true",
                        help="ACTUALLY MOVE THE ROBOT (requires confirmation)")
    parser.add_argument("--force", action="store_true",
                        help="archive an existing set without asking")
    args = parser.parse_args(argv)

    logger = setup_logging("replay_collect", args.camera)
    try:
        context = load_context(args.camera, require_intrinsics=True)
        if args.targets and args.source:
            print("ERROR: --targets and --source are exclusive", file=sys.stderr)
            return 2
        records = (load_targets(args.targets) if args.targets
                   else load_source(context, args.source, logger))
    except (ConfigError, CalibrationError, RuntimeError) as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2
    if args.only:
        records = [r for r in records if r.number in set(args.only)]
    if not records:
        print("ERROR: no waypoints to revisit", file=sys.stderr)
        return 1

    envelope = context.envelope
    max_delta = envelope.motion.max_replay_joint_delta
    print_header(context, "REPLAY AND COLLECT")
    print(f"Mode        : {'*** LIVE MOTION ***' if args.enable_motion else 'DRY RUN (nothing is sent)'}")
    print(f"Target      : {envelope.robot_ip}"
          f"{'  (URSim)' if envelope.is_simulator else '  *** PHYSICAL ROBOT ***'}")
    print(f"moveJ speed : {envelope.motion.joint_speed:.3f} rad/s, "
          f"accel {envelope.motion.joint_acceleration:.3f} rad/s^2")
    print()

    if not args.enable_motion:
        plan = build_segments(records, records[0].actual_q, max_delta)
        print("Planning from the FIRST recorded waypoint; the move from the "
              "arm's current pose is planned at run time.")
        print()
        print_plan(plan, max_delta)
        print()
        print("DRY RUN: nothing was sent to the robot and nothing was written.")
        return 0

    # Refuse on POLICY before touching any hardware.
    if not envelope.allow_motion:
        print("MOTION IS DISABLED in config/safety.yaml (connection.allow_motion: "
              "false). Nothing was sent.", file=sys.stderr)
        return 1
    unverified = envelope.unverified_sections()
    if unverified:
        print("REFUSING TO MOVE. Verify these for this cell first:", file=sys.stderr)
        for section in unverified:
            print(f"  * {section}", file=sys.stderr)
        return 1

    from robot_interface import RobotInterface

    try:
        open_camera_checked(context)
        context.robot = RobotInterface(envelope).connect()
        state = context.robot.require_state(require_stationary=True)
        plan = build_segments(records, state.q, max_delta)
        print_plan(plan, max_delta)
        print()
        try:
            mode = prepare_output(context.paths, args.force, args.resume, logger)
        except SystemExit as exc:
            print(exc)
            return 1
        if mode == "archived" and args.source is None and args.targets is None:
            print("The waypoints being revisited are in that archive; pass it as "
                  "--source to run again.")
        print()
        print("-" * 78)
        print(f"ABOUT TO MOVE THE ROBOT THROUGH {len(plan)} RECORDED CONFIGURATIONS")
        print("-" * 78)
        print("  The area must be clear and the e-stop within reach.")
        answer = input(f"  Type {CONFIRMATION_PHRASE} to proceed: ").strip()
        if answer != CONFIRMATION_PHRASE:
            print("  Cancelled. Nothing was sent.")
            return 0

        robot = context.robot
        robot.enable_motion(confirm=True)
        center_q, center_tcp = robot.capture_calibration_center()
        # The replay leaves the recorded neighbourhood by design, so the
        # per-session centre leash does not apply. Joint limits, the workspace
        # box and the per-segment size cap still do.
        robot.calibration_center_q = None
        save_initial_center(context.paths, context.camera, robot.read_state(), envelope)

        recorder = WaypointRecorder(
            context.paths, context.camera, context.detector, robot,
            context.calibration_config, context.camera_matrix,
            context.dist_coeffs, context.intrinsics)
        if mode == "resume":
            # Keep the existing waypoints in waypoints.yaml and continue numbering.
            recorder.records.extend(load_waypoints(context.paths, logger))
        skipped = []
        for index, step in enumerate(plan, start=1):
            record = step["record"]
            print(f"[{index}/{len(plan)}] {record.name} "
                  f"({len(step['targets'])} segment(s)) ...", end="", flush=True)
            for target in step["targets"]:
                robot.move_joints(target, check_step=False)
                robot.wait_until_stationary()
            try:
                new = recorder.record(recorder.next_number())
                print(f" recorded {new.name}, {len(new.tag_ids)} tags")
            except RecordingRejected as exc:
                skipped.append((record.name, str(exc)))
                print(f" SKIPPED: {exc}")
                logger.info("Skipped %s: %s", record.name, exc)

        recorder.save_master(center_q, center_tcp,
                             extra={"armed": True, "input_device": "replay_collect",
                                    "handeye_mode": context.handeye_mode})
        print()
        print(f"Recorded {len(recorder.records)} of {len(plan)}; skipped {len(skipped)}.")
        for name, reason in skipped:
            print(f"  {name}: {reason}")
        print(f"Waypoints : {context.paths.waypoints_file}")
        print(f"Next: python scripts/check_board_rigid.py --camera {args.camera}")
        return 0

    except SafetyError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        logger.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted; stopping the robot.")
        if context.robot is not None:
            context.robot.emergency_software_stop()
        return 0
    finally:
        context.close()


if __name__ == "__main__":
    sys.exit(main())
