#!/usr/bin/env python3
"""Recompute stored hand-eye board poses with the current intrinsics. Never touches the robot or a camera.

    python scripts/reestimate_board_poses.py --camera camera_3

Each observation stores the board pose that PnP gave at capture time, with
the intrinsics of that moment, and solve_handeye.py uses those stored poses.
After the intrinsics change, every stored pose is stale. This re-runs the
detector's own PnP (IPPE initialisation, LM refinement) on the stored matched
object/image points, so nothing is re-detected and no image is needed.

The observations folder is copied to handeye/sessions/<timestamp>_pre_reestimate/
before anything is rewritten.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import camera_paths, load_yaml, save_yaml, setup_logging, timestamp_slug


def solve_pose(object_points, image_points, matrix, coeffs):
    """Same PnP as AprilGridDetector.estimate_pose."""
    ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, coeffs,
                                  flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv2.solvePnP(object_points, image_points, matrix, coeffs,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, matrix, coeffs, rvec, tvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, matrix, coeffs)
    residuals = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
    rotation, _ = cv2.Rodrigues(rvec)
    return {"rvec": rvec.ravel(), "tvec": tvec.ravel(),
            "rms": float(np.sqrt(np.mean(residuals ** 2))), "max": float(np.max(residuals)),
            "tilt": math.degrees(math.acos(max(0.0, min(1.0, abs(float(rotation[2, 2]))))))}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Recompute stored board poses.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)
    logger = setup_logging("reestimate_board_poses", args.camera)

    paths = camera_paths(args.camera)
    intrinsics = load_yaml(paths.intrinsics_result)
    matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    coeffs = np.asarray(intrinsics["distortion_coefficients"], dtype=np.float64).reshape(1, -1)
    files = sorted(paths.handeye_observations.glob("waypoint_*.yaml"))
    if not files:
        print(f"ERROR: no observations in {paths.handeye_observations}", file=sys.stderr)
        return 1

    size = [int(intrinsics["image_width"]), int(intrinsics["image_height"])]
    updates, shifts = [], []
    for path in files:
        data = load_yaml(path)
        if data.get("image_size") and [int(v) for v in data["image_size"]] != size:
            print(f"ERROR: {path.name} was captured at {data['image_size']}, the intrinsics are "
                  f"for {size}; pixel coordinates do not transfer. Nothing written.",
                  file=sys.stderr)
            return 1
        pose = data.get("board_pose_camera") or {}
        if not data.get("object_points") or not data.get("image_points"):
            continue
        new = solve_pose(np.asarray(data["object_points"], dtype=np.float64).reshape(-1, 3),
                         np.asarray(data["image_points"], dtype=np.float64).reshape(-1, 2),
                         matrix, coeffs)
        if new is None:
            print(f"ERROR: PnP failed for {path.name}; nothing written.", file=sys.stderr)
            return 1
        old_t = np.asarray(pose.get("tvec", new["tvec"]), dtype=np.float64).ravel()
        shifts.append(float(np.linalg.norm(new["tvec"] - old_t)) * 1000)
        updates.append((path, data, new))

    print(f"Intrinsics   : {paths.intrinsics_result} ({intrinsics.get('source')}, {intrinsics.get('timestamp')})")
    print(f"Observations : {len(updates)}")
    print(f"Board shift  : mean {np.mean(shifts):.2f} mm, max {np.max(shifts):.2f} mm in the camera frame")
    print(f"PnP RMS      : mean {np.mean([u[2]['rms'] for u in updates]):.3f} px")
    if args.dry_run:
        print("DRY RUN: nothing written.")
        return 0

    archive = paths.sessions / f"{timestamp_slug()}_pre_reestimate"
    shutil.copytree(paths.handeye_observations, archive / "observations")
    for path, data, new in updates:
        pose = dict(data.get("board_pose_camera") or {})
        pose.update({"rvec": new["rvec"].tolist(), "tvec": new["tvec"].tolist(),
                     "distance_m": float(np.linalg.norm(new["tvec"])), "tilt_deg": new["tilt"],
                     "pnp_reprojection_px": new["rms"], "pnp_max_reprojection_px": new["max"]})
        data["board_pose_camera"] = pose
        data["intrinsics_file"] = str(paths.intrinsics_result)
        data["intrinsics_timestamp"] = intrinsics.get("timestamp")
        data["board_pose_reestimated"] = {"from": str(archive / "observations" / path.name),
                                          "intrinsics_source": intrinsics.get("source")}
        save_yaml(path, data, header=(
            f"Hand-eye observation {data.get('waypoint_name', path.stem)} for {args.camera}; "
            f"board pose re-estimated with the current intrinsics."))
    logger.info("Re-estimated %d board poses; originals in %s", len(updates), archive)
    print(f"Archived     : {archive / 'observations'}")
    print(f"Next         : python scripts/solve_handeye.py --camera {args.camera} --selection all")
    return 0


if __name__ == "__main__":
    sys.exit(main())
