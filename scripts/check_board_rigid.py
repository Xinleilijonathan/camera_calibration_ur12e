#!/usr/bin/env python3
"""Check that the board has not moved relative to its mount, mid-collection.

    python scripts/check_board_rigid.py --camera camera_2

Run this after every batch of waypoints. A board that works loose produces a
dataset that looks perfectly healthy per-frame -- every PnP fit is crisp,
every frame is sharp, nothing is rejected -- and is nonetheless unusable,
because hand-eye assumes the board-to-flange transform is a constant. The
failure only surfaces at solve time, after all 30 poses are in the bag.

HOW IT WORKS
    Write X for the camera pose and Y for the board-to-mount transform. Both
    are constant if the board is fixed, and for every waypoint:

        Y(i) = inv(T_robot(i)) @ X @ T_board(i)

    So: solve X once from the batch, then recover Y(i) for each waypoint and
    look at whether it holds still. A board that slips shows up directly, in
    millimetres, at the waypoint where it happened.

    X is biased when the board has moved, but that does not matter here --
    the bias is common to every Y(i), and what betrays a slip is the SPREAD
    of Y(i), not its absolute value.

A NOTE ON WHAT WAS TRIED FIRST
    The tempting check is calibration-free: a rigid setup makes the robot
    motion and the observed board motion conjugate, hence equal in rotation
    angle and screw pitch, with no need to solve anything. That check is real
    but it is BLIND TO THE COMMON FAILURE. If the board slides without
    turning, the discrepancy is a pure translation, whose rotation angle is
    zero -- so a 10 mm slip registers as perfectly rigid. Measured on the
    camera_2 set that a slip actually ruined, it returned "looks rigid".
    Hence the solve-based check below, which caught the same slip at 10 mm.

WHAT A FAILURE LOOKS LIKE
    Y(i) sits still, then steps. The script reports the per-waypoint position
    of the board on its mount, scans for the step, and names the waypoint
    where it happened.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               invert_transform, rotation_angle_deg,
                               setup_logging)

# A STEP is the reliable signal, and these two thresholds are deliberately
# decoupled -- tying them together makes the step test inherit the scatter
# tolerance and miss small slips.
#
# Calibrated against synthetic sets built from these real robot poses -- a
# known-rigid board plus PnP noise, and known slips injected -- over 8 noise
# seeds each. Measured hit rate at the 1.5 multiple below:
#
#   rigid, 2-3 mm noise, 30 or 10 waypoints ..... 0/8 false alarms
#   rigid, 2 mm noise, 15 waypoints ............. 1/8 false alarms
#   slip 10 mm, 10 or 15 waypoints .............. 8/8 caught
#   slip  5 mm, 30 waypoints .................... 7/8 caught
#   slip  5 mm, 10 waypoints .................... 4/8 caught
#
# So: a ~10 mm slip is caught reliably even in a small batch; a 5 mm slip in a
# 10-waypoint batch is a coin flip. This is a safety net, not a guarantee --
# the fix for a loose board is a better mount, not a better test.
#
# The multiple is 1.5 rather than 2.0 deliberately. At 2.0 the false-alarm
# rate went to zero but detection collapsed (10 mm slip in 10 waypoints fell
# to 3/8), and the costs are asymmetric: a false alarm costs one re-seat and
# re-check, a miss costs a whole 30-waypoint session.
STEP_MINIMUM_MM = 4.0
STEP_SCATTER_MULTIPLE = 1.5
# Absolute wander is the backstop, not the main test. It has to sit clear of
# PnP noise at ~1 m: 3 mm of pure noise already produces 8.4 mm of p90 offset,
# so a tighter tolerance cries wolf on a perfectly good board.
DEFAULT_TOLERANCE_MM = 10.0
MINIMUM_WAYPOINTS = 8
MINIMUM_ROTATION_DEG = 5.0   # pairs barely rotating carry no information
# A split has to leave enough pairs on BOTH sides to mean anything. Allowing a
# 3-waypoint group lets small-sample noise win the contrast: on a set whose
# board demonstrably moved at 7|8, an unconstrained scan picked 3|4, because
# three waypoints have only three pairs between them and a low median by luck.
MINIMUM_GROUP = 5


def board_on_mount(records, camera_matrix, dist_coeffs, mode):
    """Recover the board-to-mount transform Y(i) implied by each waypoint."""
    import handeye_calibration as hc
    result = hc.calibrate(list(records), mode, "park", camera_matrix,
                          dist_coeffs, cross_check=False, label="rigidity")
    X = np.asarray(result["transform"], dtype=np.float64)
    transforms = [invert_transform(r.tcp_transform) @ X @ r.board_transform
                  for r in records]
    return X, transforms


def find_step(positions: np.ndarray) -> tuple[int, float, float] | None:
    """Largest step in the board's position on its mount.

    Returns (index, step_mm, within_mm): the split, how far the board moved
    across it, and the typical scatter inside the two blocks.
    """
    n = len(positions)
    if n < 2 * MINIMUM_GROUP:
        return None
    best = None
    for k in range(MINIMUM_GROUP, n - MINIMUM_GROUP + 1):
        before, after = positions[:k], positions[k:]
        step = float(np.linalg.norm(np.median(after, axis=0)
                                    - np.median(before, axis=0)))
        within = float(np.mean([
            np.median(np.linalg.norm(before - np.median(before, axis=0), axis=1)),
            np.median(np.linalg.norm(after - np.median(after, axis=0), axis=1))]))
        if best is None or step > best[1]:
            best = (k, step, within)
    return best


def is_step(size: float, within: float) -> bool:
    """A step counts only if it beats both the floor and the local scatter."""
    return size > max(STEP_MINIMUM_MM, STEP_SCATTER_MULTIPLE * within)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check the board has not moved relative to its mount.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--tolerance-mm", type=float,
                        default=DEFAULT_TOLERANCE_MM,
                        help=f"backstop on absolute wander (default "
                             f"{DEFAULT_TOLERANCE_MM}); the step test is the "
                             f"primary signal and is not affected by this")
    args = parser.parse_args(argv)

    logger = setup_logging("check_board_rigid", args.camera)
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from _analysis import load_for_analysis  # noqa: E402

    try:
        ctx = load_for_analysis(args.camera, logger=logger)
    except (CalibrationError, ConfigError, SystemExit) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    records = sorted([r for r in ctx.records if r.board_transform is not None],
                     key=lambda r: r.number)
    if len(records) < MINIMUM_WAYPOINTS:
        print(f"ERROR: need at least {MINIMUM_WAYPOINTS} waypoints with a board "
              f"pose to judge rigidity, have {len(records)}.", file=sys.stderr)
        return 1

    numbers = [r.number for r in records]
    X, mounts = board_on_mount(records, ctx.camera_matrix, ctx.dist_coeffs,
                               ctx.handeye_mode)
    positions = np.array([m[:3, 3] for m in mounts]) * 1000.0
    centre = np.median(positions, axis=0)
    offsets = np.linalg.norm(positions - centre, axis=1)

    reference = mounts[int(np.argmin(offsets))][:3, :3]
    angles = np.array([rotation_angle_deg(reference.T @ m[:3, :3])
                       for m in mounts])

    print("=" * 74)
    print(f"BOARD RIGIDITY CHECK -- {args.camera}")
    print("=" * 74)
    print(f"Waypoints : {len(records)}  (numbers {numbers[0]}..{numbers[-1]})")
    print(f"Mounting  : {ctx.handeye_mode}")
    print()
    print("Where the board sits on its mount, recovered from each waypoint.")
    print("This is a fixed bolt-down, so it should not move at all.")
    print()
    print(f"  {'wp':>4}  {'x':>8} {'y':>8} {'z':>8}   {'off-centre':>10}  "
          f"{'tilt':>7}")
    previous = None
    for index, number in enumerate(numbers):
        jump = ""
        if previous is not None:
            step = float(np.linalg.norm(positions[index] - previous))
            if step > STEP_MINIMUM_MM:
                jump = f"   <-- moved {step:.1f} mm"
        previous = positions[index]
        print(f"  {number:>4}  {positions[index, 0]:>8.1f} "
              f"{positions[index, 1]:>8.1f} {positions[index, 2]:>8.1f}   "
              f"{offsets[index]:>10.1f}  {angles[index]:>6.2f}d{jump}")

    spread = float(np.percentile(offsets, 90))
    print()
    print("-" * 74)
    print(f"  scatter about the median : {np.median(offsets):.2f} mm "
          f"(p90 {spread:.2f}, max {offsets.max():.2f})")
    print(f"  orientation scatter      : {np.median(angles):.2f} deg "
          f"(max {angles.max():.2f})")

    step = find_step(positions)
    stepped = False
    if step is not None:
        k, size, within = step
        print(f"  largest step             : {size:.2f} mm between waypoint "
              f"{numbers[k - 1]} and {numbers[k]}")
        print(f"  scatter inside the blocks: {within:.2f} mm")
        stepped = is_step(size, within)

    over = spread > args.tolerance_mm
    print()
    print("=" * 74)
    if not (over or stepped):
        print("VERDICT: BOARD IS HOLDING STILL")
        print()
        print(f"  The board stays within {spread:.2f} mm of one position on its "
              f"mount across")
        print(f"  all {len(records)} waypoints, inside the "
              f"{args.tolerance_mm:.1f} mm tolerance, with no step. That is")
        print("  consistent with measurement noise rather than movement.")
        print("  Carry on collecting.")
        print("=" * 74)
        logger.info("Board rigid for %s (p90 %.2f mm)", args.camera, spread)
        return 0

    print("VERDICT: THE BOARD HAS MOVED")
    print()
    if stepped:
        k, size, within = step
        print(f"  It sat still, then shifted {size:.1f} mm between waypoint "
              f"{numbers[k - 1]} and")
        print(f"  {numbers[k]} -- against only {within:.2f} mm of scatter "
              f"within each block. That is")
        print("  a mechanical slip, not noise.")
    if over:
        print(f"  It wanders {spread:.1f} mm about its median position "
              f"(tolerance {args.tolerance_mm:.1f} mm).")
    print()
    print("  Per-frame quality cannot see this: every PnP fit stays crisp and")
    print("  no frame is rejected, yet hand-eye assumes this transform is")
    print("  constant and will solve to nonsense. Waypoints on BOTH sides of")
    print("  a slip are suspect, because which side is 'right' is unknowable.")
    print()
    print("  Secure the board, then archive this set and start again:")
    print(f"    python scripts/collect_waypoints.py --camera {args.camera} "
          f"--target-count 30 --read-only --force")
    print("=" * 74)
    logger.error("Board moved for %s (p90 %.2f mm, step=%s)",
                 args.camera, spread, stepped)
    return 1


if __name__ == "__main__":
    sys.exit(main())
