#!/usr/bin/env python3
"""Solve intrinsics for ONE camera from its collected observations.

    python scripts/solve_intrinsics.py --camera camera_1

Writes data/camera_N/intrinsics/result.yaml. Reads and writes only that
camera's directory, so no camera can ever inherit another's parameters.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               describe_environment, load_calibration_config,
                               load_cameras_config, load_yaml,
                               require_board_verified, resolve_camera,
                               setup_logging)
from intrinsic_calibration import (IntrinsicObservation, save_intrinsics,
                                   solve_intrinsics)


def load_observations(paths, logger) -> list[IntrinsicObservation]:
    observations = []
    for path in sorted(paths.intrinsics_observations.glob("observation_*.yaml")):
        try:
            observations.append(IntrinsicObservation.from_dict(load_yaml(path)))
        except Exception as exc:
            logger.error("Could not read %s: %s", path, exc)
            raise CalibrationError(f"Unreadable observation {path}: {exc}") from exc
    return observations


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Solve camera intrinsics from collected observations.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--exclude", nargs="*", type=int, default=[],
                        metavar="INDEX",
                        help="observation indices to leave out of the solve")
    parser.add_argument("--auto-prune", action="store_true",
                        help="drop observations worse than 3x the mean error and re-solve once")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the result without writing result.yaml")
    args = parser.parse_args(argv)

    logger = setup_logging("solve_intrinsics", args.camera)

    try:
        cameras_config = load_cameras_config()
        calibration_config = load_calibration_config()
        camera_config = resolve_camera(args.camera, cameras_config)
        # Board geometry sets the metric scale of everything downstream.
        require_board_verified(calibration_config)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    paths = camera_paths(args.camera)
    intrinsics_config = calibration_config.get("intrinsics", {})

    try:
        observations = load_observations(paths, logger)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.exclude:
        before = len(observations)
        observations = [o for o in observations if o.index not in set(args.exclude)]
        print(f"Excluded {before - len(observations)} observation(s) by request.")

    if not observations:
        print(f"ERROR: no observations in {paths.intrinsics_observations}\n"
              f"  Run: python scripts/collect_intrinsics.py --camera {args.camera}",
              file=sys.stderr)
        return 1

    sizes = {o.image_size for o in observations}
    if len(sizes) > 1:
        print(f"ERROR: observations were captured at mixed resolutions: {sizes}\n"
              f"  Intrinsics are in pixels and cannot be solved across "
              f"resolutions. Re-collect at a single resolution.", file=sys.stderr)
        return 1
    image_size = observations[0].image_size

    print("=" * 74)
    print(f"INTRINSIC SOLVE -- {args.camera}")
    print("=" * 74)
    print(f"Observations : {len(observations)}")
    print(f"Resolution   : {image_size[0]} x {image_size[1]}")
    print(f"Model        : {intrinsics_config.get('distortion_model', 'standard')}")
    print()

    try:
        result = solve_intrinsics(observations, image_size, intrinsics_config)
    except CalibrationError as exc:
        print(f"CALIBRATION FAILED: {exc}", file=sys.stderr)
        return 1

    if args.auto_prune:
        mean = result["mean_reprojection_error_px"]
        bad = {o["index"] for o in result["per_observation"] if o["rms_px"] > 3 * mean}
        if bad and len(observations) - len(bad) >= 5:
            print(f"Auto-prune: dropping {sorted(bad)} (>3x the mean error) "
                  f"and re-solving.")
            observations = [o for o in observations if o.index not in bad]
            result = solve_intrinsics(observations, image_size, intrinsics_config)
            result["auto_pruned"] = sorted(bad)
        elif bad:
            print(f"Auto-prune: would drop {sorted(bad)}, but too few would "
                  f"remain. Keeping all.")

    print_result(result, args.camera)

    if args.dry_run:
        print("\n--dry-run: result.yaml was NOT written.")
        return 0

    save_intrinsics(paths.intrinsics_result, result,
                    camera_name=args.camera,
                    camera_serial=str(camera_config.get("serial", "")),
                    board=calibration_config["apriltag_grid"],
                    environment=describe_environment())
    print(f"\nSaved: {paths.intrinsics_result}")
    logger.info("Wrote %s (RMS %.4f px)", paths.intrinsics_result,
                result["rms_reprojection_error_px"])

    limit = float(intrinsics_config.get("maximum_reprojection_error", 1.0))
    if result["rms_reprojection_error_px"] > limit:
        print(f"\nRESULT REJECTED BY YOUR OWN LIMIT: RMS "
              f"{result['rms_reprojection_error_px']:.3f} px > {limit:.2f} px")
        print("The file was written so you can inspect it, but do not use it.")
        return 1
    return 0


def print_result(result, camera_name: str) -> None:
    print("-" * 74)
    print(f"fx {result['fx']:10.3f}    fy {result['fy']:10.3f}")
    print(f"cx {result['cx']:10.3f}    cy {result['cy']:10.3f}")
    print()
    print("Distortion:", "  ".join(f"{c: .6f}"
                                   for c in result["distortion_coefficients"]))
    print()
    fov = result["field_of_view_deg"]
    print(f"Field of view      : {fov['horizontal']:.1f} deg H, "
          f"{fov['vertical']:.1f} deg V")
    offset = result["principal_point_offset_px"]
    print(f"Principal offset   : {offset['x']:+.1f}, {offset['y']:+.1f} px "
          f"from image centre")
    print(f"Aspect ratio fy/fx : {result['aspect_ratio']:.5f}")
    print()
    print(f"RMS reprojection   : {result['rms_reprojection_error_px']:.4f} px")
    print(f"Mean reprojection  : {result['mean_reprojection_error_px']:.4f} px")
    print(f"Median reprojection: {result['median_reprojection_error_px']:.4f} px")
    print(f"Max reprojection   : {result['max_reprojection_error_px']:.4f} px")
    print(f"Points used        : {result['total_points']}")
    print()
    print("Per observation (worst 5):")
    print(f"  {'idx':>4}  {'rms px':>8}  {'max px':>8}  {'tilt':>6}  {'dist m':>7}")
    worst = sorted(result["per_observation"], key=lambda o: -o["rms_px"])[:5]
    for observation in worst:
        tilt = observation.get("tilt_deg")
        distance = observation.get("distance_m")
        print(f"  {observation['index']:>4}  {observation['rms_px']:>8.4f}  "
              f"{observation['max_px']:>8.4f}  "
              f"{tilt if tilt is None else f'{tilt:6.1f}'}  "
              f"{distance if distance is None else f'{distance:7.3f}'}")

    if result["warnings"]:
        print()
        print("WARNINGS:")
        for warning in result["warnings"]:
            print(f"  * {warning}")
    print("-" * 74)


if __name__ == "__main__":
    sys.exit(main())
