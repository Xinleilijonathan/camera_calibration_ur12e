#!/usr/bin/env python3
"""Plan extra eye-to-hand waypoints for a camera from its current calibration. Moves nothing.

    python scripts/plan_extra_waypoints.py --camera camera_3 --count 10

Samples joint configurations around the existing waypoints (same IK branch),
predicts where the board lands in the image through the CURRENT solved
calibration (T_base_camera and T_flange_board), and keeps candidates that are
fully visible with margin, larger in the image than the current set, tilted,
and different in orientation from every waypoint already collected. Every
candidate and every moveJ path between them (split at the replay step limit)
is checked against the joint limits, the workspace box, a minimum TCP height,
the whole clamped sheet (not just the printed grid) against the arm links and
the base plane,
and a keep-out sphere around the camera.

Forward kinematics uses the nominal UR12e (UR10e-arm) DH parameters; it is
validated against the recorded TCP poses before anything is planned and the
script refuses to continue if they disagree by more than 2 mm.

Writes handeye/waypoints/planned_extra.yaml for replay_collect.py --targets.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401
import _analysis

from calibration_utils import (CalibrationError, ConfigError, load_yaml, pose_to_matrix,
                               rotation_angle_deg, save_yaml, setup_logging, timestamp_utc)
from replay_collect import split_transition
from safety import load_envelope

# Nominal UR12e DH (same arm as the UR10e): base -> tool0.
DH_D = np.array([0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655])
DH_A = np.array([0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
FK_TOLERANCE_M = 0.002

IMAGE_MARGIN_PX = 30.0          # detector needs 12
MIN_AREA_FRACTION = 0.04        # detector needs 0.02
MAX_TILT_DEG = 50.0
MIN_NEW_ROTATION_DEG = 8.0
MIN_NEW_TRANSLATION_MM = 40.0
MIN_TCP_Z_M = 0.33
CAMERA_KEEPOUT_M = 0.40         # no joint origin or TCP closer than this
SAMPLE_SPREAD_DEG = np.array([20.0, 15.0, 20.0, 35.0, 35.0, 60.0])

# The physical sheet, not just the printed grid: landscape US-letter-like sheet
# clamped by the Hand-E at the middle of its long top edge, 30 mm deep.
PAPER_WIDTH_M = 0.2794          # across the clamp line
PAPER_TOP_M = 0.130             # top edge, along the tool axis from the flange
PAPER_BOTTOM_M = 0.345          # bottom edge (top + ~210 mm sheet + margin)
MIN_PAPER_Z_M = 0.08            # lowest sheet point above the base plane
# Arm links as capsules between DH joint origins (start frame, end frame, radius).
LINK_CAPSULES = ((0, 1, 0.075), (1, 2, 0.065), (2, 3, 0.060), (3, 4, 0.060))
LINK_CLEARANCE_M = 0.08            # the collected set never came closer than 95 mm


def joint_frames(q) -> list[np.ndarray]:
    """Base -> each joint frame, for one configuration. The last is tool0."""
    frames, transform = [], np.eye(4)
    for i in range(6):
        ct, st = np.cos(q[i]), np.sin(q[i])
        ca, sa = np.cos(DH_ALPHA[i]), np.sin(DH_ALPHA[i])
        transform = transform @ np.array([
            [ct, -st * ca, st * sa, DH_A[i] * ct],
            [st, ct * ca, -ct * sa, DH_A[i] * st],
            [0.0, sa, ca, DH_D[i]],
            [0.0, 0.0, 0.0, 1.0]])
        frames.append(transform)
    return frames


def paper_points_tcp(tool_board, step: float = 0.02) -> np.ndarray:
    """4xN homogeneous points covering the sheet, in the TCP frame.

    The solved T_tcp_board puts the tool axis in the sheet plane (the sheet is
    clamped between the fingers), so the sheet is laid out along that axis.
    """
    board_tool = np.linalg.inv(tool_board)
    origin, axis = board_tool[:3, 3], board_tool[:3, 2]
    along = np.array([axis[0], axis[1], 0.0])
    along /= np.linalg.norm(along)
    across = np.cross([0.0, 0.0, 1.0], along)
    foot = np.array([origin[0], origin[1], 0.0])     # the flange, projected onto the sheet
    rows = np.arange(PAPER_TOP_M, PAPER_BOTTOM_M + 1e-9, step)
    cols = np.arange(-PAPER_WIDTH_M / 2, PAPER_WIDTH_M / 2 + 1e-9, step)
    board = np.array([foot + along * r + across * c for r in rows for c in cols])
    return tool_board @ np.c_[board, np.ones(len(board))].T


def clearances(q, tool_tcp, paper_tcp, camera_position) -> dict:
    """Sheet height, sheet-to-arm and anything-to-camera distances at one pose."""
    frames = joint_frames(q)
    tcp = frames[-1] @ tool_tcp
    paper = (tcp @ paper_tcp)[:3].T
    arm = min(segment_distance(paper, frames[i][:3, 3], frames[j][:3, 3]) - radius
              for i, j, radius in LINK_CAPSULES)
    origins = np.array([f[:3, 3] for f in frames[1:]] + [tcp[:3, 3]])
    camera = float(min(np.min(np.linalg.norm(origins - camera_position, axis=1)),
                       np.min(np.linalg.norm(paper - camera_position, axis=1))))
    return {"tcp": tcp, "paper_z": float(paper[:, 2].min()), "arm": float(arm),
            "camera": camera}


def clear(c: dict) -> bool:
    return (c["tcp"][2, 3] >= MIN_TCP_Z_M and c["paper_z"] >= MIN_PAPER_Z_M
            and c["arm"] >= LINK_CLEARANCE_M and c["camera"] >= CAMERA_KEEPOUT_M)


def segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Smallest distance from any of `points` (N,3) to segment ab."""
    ab = b - a
    t = np.clip(((points - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return float(np.min(np.linalg.norm(points - (a + t[:, None] * ab), axis=1)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Plan extra waypoints. Moves nothing.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--result", default=None,
                        help="calibration to plan with (default: newest final_result_all_*.yaml)")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--samples", type=int, default=40000)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--start-deg", type=float, nargs=6, default=None,
                        help="joint pose the arm is at now (default: last collected waypoint)")
    args = parser.parse_args(argv)
    setup_logging("plan_extra_waypoints", args.camera)

    try:
        context = _analysis.load_for_analysis(args.camera, require_verified=False)
    except (ConfigError, CalibrationError) as exc:
        return _analysis.fail(exc)
    if context.handeye_mode == "eye_in_hand":
        print("ERROR: this planner is for eye-to-hand cameras.", file=sys.stderr)
        return 1
    if args.result:
        result_path = context.paths.handeye / args.result
    else:
        candidates = sorted(context.paths.handeye.glob("final_result_all_*.yaml"),
                            key=lambda p: p.stat().st_mtime)
        if not candidates:
            print("ERROR: no final_result_all_*.yaml to plan with", file=sys.stderr)
            return 1
        result_path = candidates[-1]
    result = load_yaml(result_path)
    base_camera = np.asarray(result["transform"], dtype=np.float64)
    tool_board = np.asarray(result["reference_constant_transform"], dtype=np.float64)
    camera_base = np.linalg.inv(base_camera)
    camera_position = base_camera[:3, 3]

    records = [r for r in context.records if r.board_transform is not None]
    best_record = max(records, key=lambda r: len(r.object_points))
    board_points = np.asarray(best_record.object_points, dtype=np.float64)

    # FK check against measured data, and the tool0 -> TCP offset.
    tool_tcp = np.linalg.inv(joint_frames(records[0].actual_q)[-1]) @ records[0].tcp_transform
    fk_error = max(np.linalg.norm((joint_frames(r.actual_q)[-1] @ tool_tcp)[:3, 3]
                                  - r.tcp_transform[:3, 3]) for r in records)
    print(f"Planning with : {result_path.name}")
    print(f"FK check      : max {fk_error * 1000:.2f} mm against {len(records)} recorded TCP poses")
    if fk_error > FK_TOLERANCE_M:
        print("ERROR: forward kinematics does not match this robot; refusing to plan.",
              file=sys.stderr)
        return 1

    envelope = load_envelope()
    width = int(context.intrinsics["image_width"])
    height = int(context.intrinsics["image_height"])

    # The side of the board the camera sees, from the collected set.
    facing = np.sign(np.median([r.board_transform[2, 2] for r in records]))

    paper_tcp = paper_points_tcp(tool_board)

    def evaluate(q):
        c = clearances(q, tool_tcp, paper_tcp, camera_position)
        if not clear(c):
            return None
        tcp = c["tcp"]
        board = camera_base @ tcp @ tool_board
        if np.sign(board[2, 2]) != facing or board[2, 3] <= 0:
            return None
        tilt = np.degrees(np.arccos(min(1.0, abs(board[2, 2]))))
        if tilt > MAX_TILT_DEG:
            return None
        rvec, _ = cv2.Rodrigues(board[:3, :3])
        pixels, _ = cv2.projectPoints(board_points, rvec, board[:3, 3],
                                      context.camera_matrix, context.dist_coeffs)
        pixels = pixels.reshape(-1, 2)
        if (pixels.min(0) < IMAGE_MARGIN_PX).any() or \
                pixels[:, 0].max() > width - IMAGE_MARGIN_PX or \
                pixels[:, 1].max() > height - IMAGE_MARGIN_PX:
            return None
        area = cv2.contourArea(cv2.convexHull(pixels.astype(np.float32))) / (width * height)
        if area < MIN_AREA_FRACTION:
            return None
        return {"tcp": tcp, "tilt": tilt, "area": area, "arm": c["arm"], "paper_z": c["paper_z"],
                "distance": float(np.linalg.norm(board[:3, 3]))}

    def safe(q) -> bool:
        try:
            envelope.joints.check(q)
        except Exception:
            return False
        tcp = joint_frames(q)[-1] @ tool_tcp
        rotvec, _ = cv2.Rodrigues(tcp[:3, :3])
        return envelope.workspace.contains(list(tcp[:3, 3]) + list(rotvec.ravel()))

    def path_ok(start, end) -> bool:
        for target in split_transition(start, end, envelope.motion.max_replay_joint_delta):
            for s in np.linspace(0.0, 1.0, 30):
                q = start + (target - start) * s
                if not safe(q) or not clear(clearances(q, tool_tcp, paper_tcp, camera_position)):
                    return False
            start = target
        return True

    def order(start_q, targets):
        """Nearest-neighbour then 2-opt on the largest joint move, from start_q."""
        cost = lambda a, b: float(np.max(np.abs(a - b)))
        remaining, route, current = list(range(len(targets))), [], start_q
        while remaining:
            nxt = min(remaining, key=lambda i: cost(current, targets[i][0]))
            route.append(nxt)
            remaining.remove(nxt)
            current = targets[nxt][0]
        total = lambda r: sum(cost(a, b) for a, b in zip(
            [start_q] + [targets[i][0] for i in r[:-1]], [targets[i][0] for i in r]))
        improved = True
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 1, len(route)):
                    candidate = route[:i] + route[i:j + 1][::-1] + route[j + 1:]
                    if total(candidate) < total(route) - 1e-9:
                        route, improved = candidate, True
        return [targets[i] for i in route]

    rng = np.random.default_rng(args.seed)
    seeds = np.array([r.actual_q for r in records])
    existing_tcp = [r.tcp_transform for r in records]
    existing_area = float(np.median([evaluate(r.actual_q)["area"] for r in records
                                     if evaluate(r.actual_q)]))
    pool = []
    for _ in range(args.samples):
        q = seeds[rng.integers(len(seeds))] + np.radians(
            rng.uniform(-1.0, 1.0, 6) * SAMPLE_SPREAD_DEG)
        if not safe(q):
            continue
        info = evaluate(q)
        if info is not None:
            pool.append((q, info))
    print(f"Candidates    : {len(pool)} of {args.samples} samples pass visibility and safety")
    print(f"Board size    : collected set median {existing_area * 100:.1f}% of the image")

    def novelty(tcp, references):
        rotation = min(rotation_angle_deg(tcp[:3, :3].T @ ref[:3, :3]) for ref in references)
        translation = min(np.linalg.norm(tcp[:3, 3] - ref[:3, 3]) for ref in references) * 1000
        return rotation, translation

    start_q = (np.radians(args.start_deg) if args.start_deg
               else np.asarray(records[-1].actual_q, dtype=np.float64))
    rejected_legs = 0
    chosen, references = [], list(existing_tcp)
    while len(chosen) < args.count and pool:
        best = None
        for index, (q, info) in enumerate(pool):
            rotation, translation = novelty(info["tcp"], references)
            if rotation < MIN_NEW_ROTATION_DEG or translation < MIN_NEW_TRANSLATION_MM:
                continue
            # Favour orientation novelty, then a bigger board, then tilt.
            score = rotation + 100.0 * info["area"] + 0.2 * info["tilt"]
            if best is None or score > best[0]:
                best = (score, index)
        if best is None:
            break
        q, info = pool.pop(best[1])
        trial = order(start_q, chosen + [(q, info)])
        legs = zip([start_q] + [t[0] for t in trial[:-1]], [t[0] for t in trial])
        if all(path_ok(a, b) for a, b in legs):
            chosen = trial
            references.append(info["tcp"])
        else:
            rejected_legs += 1

    if not chosen:
        print("ERROR: no candidate passed every check.", file=sys.stderr)
        return 1
    print(f"Route         : {rejected_legs} candidate(s) dropped because a path "
          f"failed a check; largest single move "
          f"{max(np.degrees(np.max(np.abs(b - a))) for a, b in zip([start_q] + [t[0] for t in chosen[:-1]], [t[0] for t in chosen])):.1f} deg")

    print()
    print(f"{'#':>3}  {'joints (deg)':<56}  {'dist':>6}  {'area':>6}  {'tilt':>6}  {'new rot':>7}  {'arm gap':>7}")
    entries = []
    for number, (q, info) in enumerate(chosen, start=1):
        rotation, _ = novelty(info["tcp"], existing_tcp)
        print(f"{number:>3}  {'  '.join(f'{v:7.2f}' for v in np.degrees(q)):<56}  "
              f"{info['distance'] * 1000:5.0f}mm  {info['area'] * 100:5.1f}%  "
              f"{info['tilt']:5.1f}d  {rotation:6.1f}d  {info['arm'] * 1000:5.0f}mm")
        rotvec, _ = cv2.Rodrigues(info["tcp"][:3, :3])
        entries.append({"joints_rad": [float(v) for v in q],
                        "joints_deg": [float(v) for v in np.degrees(q)],
                        "predicted_tcp": [float(v) for v in info["tcp"][:3, 3]] +
                                         [float(v) for v in rotvec.ravel()],
                        "predicted_board_distance_mm": info["distance"] * 1000,
                        "predicted_board_area_fraction": info["area"],
                        "predicted_tilt_deg": info["tilt"],
                        "predicted_sheet_to_arm_mm": info["arm"] * 1000,
                        "predicted_sheet_min_z_mm": info["paper_z"] * 1000})
    output = context.paths.handeye_waypoints / "planned_extra.yaml"
    save_yaml(output, {"camera_name": context.camera_name, "timestamp": timestamp_utc(),
                       "planned_with": str(result_path), "fk_check_max_mm": fk_error * 1000,
                       "start_from_deg": [float(v) for v in np.degrees(start_q)],
                       "targets": entries},
              header="Planned extra waypoints. Predictions only; the robot was not moved.")
    print()
    print(f"{len(chosen)} targets, paths checked from {np.round(np.degrees(start_q), 2).tolist()} deg onward.")
    print(f"Saved: {output}")
    print(f"Next : python scripts/replay_collect.py --camera {args.camera} "
          f"--targets {output} --resume")
    return 0


if __name__ == "__main__":
    sys.exit(main())
