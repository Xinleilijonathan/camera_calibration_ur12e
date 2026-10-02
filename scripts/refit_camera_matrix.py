#!/usr/bin/env python3
"""Refit a camera matrix with the distortion held fixed. Never touches the robot or a camera.

    python scripts/refit_camera_matrix.py --camera camera_3 \
        --distortion-from data/camera_3/intrinsics/sessions/2026-09-29_145536/result.yaml
    python scripts/refit_camera_matrix.py --camera camera_3 --derive \
        --distortion-from data/camera_3/intrinsics/sessions/2026-09-29_145536/result.yaml

A camera matrix and its distortion are fitted together and are correlated:
pairing the matrix of one solve with the distortion of another (a "blend")
leaves the pair inconsistent. When the current resolution cannot constrain the
distortion on its own, the consistent alternative is to take the distortion
from the solve that can, hold it fixed, and refit fx, fy, cx and cy on the
stored observations of the current resolution.

With --derive nothing is fitted: a 1280x720 solve is mapped to 640x480 through
the RealSense colour crop/scale model, which keeps the matrix and distortion
the consistent pair they were solved as.

The current result.yaml is copied to intrinsics/sessions/<timestamp>/ first.
Board poses in already-collected hand-eye observations were computed with the
old intrinsics: run reestimate_board_poses.py afterwards.
"""
from __future__ import annotations

import argparse
import shutil
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (camera_paths, load_yaml, save_yaml, setup_logging,
                               timestamp_slug, timestamp_utc)


DERIVE_REASON = (
    "RealSense colour 640x480 is the centred 1440x1080 crop of the 1920x1080 sensor "
    "image scaled by 4/9, and 1280x720 is the full image scaled by 2/3; the factory "
    "intrinsics of these units satisfy this exactly. Mapping the 720p solve keeps its "
    "camera matrix and distortion as the consistent pair they were fitted as, with the "
    "full field of view constraining the distortion")


def derive_640x480(matrix_720: np.ndarray) -> np.ndarray:
    """Map a 1280x720 RealSense colour camera matrix to 640x480 (crop + scale)."""
    fx, fy, cx, cy = matrix_720[0, 0], matrix_720[1, 1], matrix_720[0, 2], matrix_720[1, 2]
    return np.array([[fx * 2 / 3, 0.0, (cx * 1.5 - 240.0) * 4 / 9],
                     [0.0, fy * 2 / 3, cy * 2 / 3],
                     [0.0, 0.0, 1.0]])


def pose_only_errors(object_points, image_points, matrix, coeffs) -> np.ndarray:
    """Per-point reprojection error with only the board pose fitted."""
    op = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    ip = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    _, rvec, tvec = cv2.solvePnP(op, ip, matrix, coeffs, flags=cv2.SOLVEPNP_IPPE)
    rvec, tvec = cv2.solvePnPRefineLM(op, ip, matrix, coeffs, rvec, tvec)
    return np.linalg.norm(cv2.projectPoints(op, rvec, tvec, matrix, coeffs)[0].reshape(-1, 2) - ip,
                          axis=1)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Refit fx, fy, cx, cy with fixed distortion.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--distortion-from", required=True,
                        help="result.yaml whose distortion is held fixed (with --derive, "
                             "whose camera matrix and distortion are mapped)")
    parser.add_argument("--derive", action="store_true",
                        help="do not fit: map the source's camera matrix to this resolution "
                             "with the RealSense colour crop/scale model (1280x720 -> 640x480)")
    parser.add_argument("--dry-run", action="store_true", help="report without writing")
    args = parser.parse_args(argv)
    logger = setup_logging("refit_camera_matrix", args.camera)

    paths = camera_paths(args.camera)
    current = load_yaml(paths.intrinsics_result)
    source = load_yaml(args.distortion_from)
    size = (int(current["image_width"]), int(current["image_height"]))
    distortion = np.asarray(source["distortion_coefficients"], dtype=np.float64).ravel()

    observations = [load_yaml(p) for p in sorted(paths.intrinsics_observations.glob("observation_*.yaml"))]
    observations = [o for o in observations
                    if tuple(o.get("image_size", size)) == size and o.get("image_points")]
    if len(observations) < 10:
        print(f"ERROR: only {len(observations)} intrinsic observations at {size}", file=sys.stderr)
        return 1
    object_points = [np.asarray(o["object_points"], dtype=np.float32) for o in observations]
    image_points = [np.asarray(o["image_points"], dtype=np.float32) for o in observations]

    guess = np.asarray(current["camera_matrix"], dtype=np.float64)
    if args.derive:
        source_size = (int(source["image_width"]), int(source["image_height"]))
        if source_size != (1280, 720) or size != (640, 480):
            print(f"ERROR: --derive maps 1280x720 to 640x480 only, not {source_size} to {size}",
                  file=sys.stderr)
            return 1
        matrix = derive_640x480(np.asarray(source["camera_matrix"], dtype=np.float64))
        coeffs = distortion.reshape(1, -1)
        errors = np.concatenate([pose_only_errors(op, ip, matrix, coeffs)
                                 for op, ip in zip(object_points, image_points)])
        rms = float(np.sqrt(np.mean(errors ** 2)))
    else:
        flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2
                 | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_TANGENT_DIST)
        rms, matrix, coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, size, guess.copy(), distortion.copy(), flags=flags)
        errors = np.concatenate([
            np.linalg.norm(cv2.projectPoints(op, rv, tv, matrix, coeffs)[0].reshape(-1, 2)
                           - ip.reshape(-1, 2), axis=1)
            for op, ip, rv, tv in zip(object_points, image_points, rvecs, tvecs)])

    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    print(f"Observations : {len(observations)} at {size[0]}x{size[1]}, {errors.size} points")
    print(f"Method       : {'derived from ' + args.distortion_from if args.derive else 'refit, distortion fixed from ' + args.distortion_from}")
    print(f"Before       : fx {guess[0, 0]:.2f} fy {guess[1, 1]:.2f} cx {guess[0, 2]:.2f} cy {guess[1, 2]:.2f}")
    print(f"After        : fx {fx:.2f} fy {fy:.2f} cx {cx:.2f} cy {cy:.2f}")
    print(f"RMS          : {rms:.4f} px (was {current.get('rms_reprojection_error_px', float('nan')):.4f})")
    if args.dry_run:
        print("DRY RUN: nothing written.")
        return 0

    archive = paths.intrinsics / "sessions" / timestamp_slug()
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(paths.intrinsics_result, archive / "result.yaml")

    result = dict(current)
    for stale in ("blend", "refit", "derived"):
        result.pop(stale, None)
    result.update({
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": coeffs.ravel().tolist(),
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "rms_reprojection_error_px": float(rms),
        "mean_reprojection_error_px": float(np.mean(errors)),
        "median_reprojection_error_px": float(np.median(errors)),
        "max_reprojection_error_px": float(np.max(errors)),
        "total_points": int(errors.size),
        "observation_count": len(observations),
        "field_of_view_deg": {
            "horizontal": float(np.degrees(2 * np.arctan(size[0] / (2 * fx)))),
            "vertical": float(np.degrees(2 * np.arctan(size[1] / (2 * fy))))},
        "principal_point_offset_px": {"x": float(cx - size[0] / 2), "y": float(cy - size[1] / 2)},
        "aspect_ratio": float(fy / fx),
        "timestamp": timestamp_utc(),
        "source": "derived_from_720p" if args.derive else "refit_fixed_distortion",
        ("derived" if args.derive else "refit"): {
            "distortion_from": {"file": str(args.distortion_from),
                                "resolution": [int(source.get("image_width", 0)),
                                               int(source.get("image_height", 0))],
                                "timestamp": source.get("timestamp")},
            "observations_from": str(paths.intrinsics_observations),
            "replaces": str(archive / "result.yaml"),
            "reason": (DERIVE_REASON if args.derive else
                       "the previous blend paired a camera matrix solved together with "
                       "its own distortion with another solve's distortion; refitting the "
                       "matrix with that distortion held fixed makes the pair consistent"),
        },
    })
    save_yaml(paths.intrinsics_result, result, header=(
        f"Intrinsics for {args.camera} (serial {current.get('camera_serial')}).\n"
        + (f"Camera matrix and distortion derived from the 1280x720 solve; see `derived`."
           if args.derive else
           f"Camera matrix refitted at {size[0]}x{size[1]} with the distortion held fixed;\n"
           f"see the `refit` block.")))
    logger.info("Refitted %s camera matrix; previous result archived to %s", args.camera, archive)
    print(f"Archived     : {archive / 'result.yaml'}")
    print(f"Saved        : {paths.intrinsics_result}")
    print(f"Next         : python scripts/reestimate_board_poses.py --camera {args.camera}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
