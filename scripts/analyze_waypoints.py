#!/usr/bin/env python3
"""Preliminary hand-eye solve over ALL waypoints, then score every one.

    python scripts/analyze_waypoints.py --camera camera_1

Writes:
    handeye/preliminary_result_all_30.yaml
    handeye/selection/waypoint_scores.csv

Solves with every waypoint, measures how well each one agrees with that
solution, and ranks them. Nothing is deleted and nothing is selected here --
that is select_best_waypoints.py.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import _analysis

from calibration_utils import (CalibrationError, ConfigError,
                               describe_environment, save_csv, save_yaml,
                               setup_logging)
from handeye_calibration import calibrate
from waypoint_quality import (SCORE_COLUMNS, compute_combined_scores,
                              compute_scores, flag_outliers, format_table)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Preliminary hand-eye solve using ALL waypoints, plus "
                    "per-waypoint quality scoring.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--method", default=None,
                        help="override calibration.yaml handeye.method")
    parser.add_argument("--no-cross-check", action="store_true",
                        help="skip solving with every method")
    args = parser.parse_args(argv)

    logger = setup_logging("analyze_waypoints", args.camera)
    try:
        context = _analysis.load_for_analysis(args.camera, logger=logger)
    except (ConfigError, CalibrationError) as exc:
        return _analysis.fail(exc)

    _analysis.print_context(context, "WAYPOINT ANALYSIS")
    method = args.method or context.method

    print(f"Preliminary hand-eye solve using ALL {len(context.records)} waypoints...")
    try:
        preliminary = calibrate(
            context.records, context.handeye_mode, method,
            context.camera_matrix, context.dist_coeffs,
            cross_check=context.cross_check and not args.no_cross_check,
            label=f"preliminary_all_{len(context.records)}")
    except CalibrationError as exc:
        return _analysis.fail(exc)

    print_result(preliminary)

    scores = compute_scores(context.records, preliminary)
    selection_config = context.calibration_config.get("selection", {})
    scores = flag_outliers(scores, selection_config)
    scores = compute_combined_scores(scores, selection_config)

    print()
    print("PER-WAYPOINT RANKING (lower score is better)")
    print(format_table(scores))

    outliers = [s for s in scores if s.outlier]
    print()
    print(f"Flagged as outliers: {len(outliers)} of {len(scores)} "
          f"(kept on disk, excluded from selection)")
    for score in outliers:
        print(f"  {score.name}: {score.rejection_reason}")

    preliminary_record = dict(preliminary)
    preliminary_record["camera_name"] = context.camera_name
    preliminary_record["camera_serial"] = str(context.camera_config.get("serial", ""))
    preliminary_record["intrinsics_file"] = str(context.paths.intrinsics_result)
    preliminary_record["board"] = context.calibration_config["apriltag_grid"]
    preliminary_record["environment"] = describe_environment()
    save_yaml(context.paths.preliminary_result, preliminary_record, header=(
        f"PRELIMINARY hand-eye calibration for {context.camera_name} using ALL "
        f"{preliminary['observation_count']} waypoints.\n"
        f"This file is never overwritten by the final solve."))

    context.paths.selection.mkdir(parents=True, exist_ok=True)
    save_csv(context.paths.waypoint_scores, [s.as_row() for s in scores],
             SCORE_COLUMNS)

    print()
    print(f"Saved: {context.paths.preliminary_result}")
    print(f"Saved: {context.paths.waypoint_scores}")
    print()
    print(f"Next: python scripts/select_best_waypoints.py --camera "
          f"{args.camera} --count "
          f"{context.calibration_config['waypoint_collection']['final_count']}")
    return 0


def print_result(result) -> None:
    print()
    print("-" * 78)
    print(f"PRELIMINARY RESULT ({result['observation_count']} observations, "
          f"{result['method']}, {result['mode']})")
    print("-" * 78)
    print(f"  {result['transform_meaning']}")
    translation = result["translation_mm"]
    print(f"  translation : {translation[0]:8.2f} {translation[1]:8.2f} "
          f"{translation[2]:8.2f}  mm")
    print(f"  rotation    : {result['rotation_angle_deg']:.3f} deg about "
          + " ".join(f"{v:+.4f}" for v in result["rotation_vector"]))
    print()
    translation_stats = result["translation_residual_mm"]
    rotation_stats = result["rotation_residual_deg"]
    reprojection = result["reprojection_px"]
    print(f"  translation residual : mean {fmt(translation_stats['mean'])} mm   "
          f"median {fmt(translation_stats['median'])}   max {fmt(translation_stats['max'])}")
    print(f"  rotation residual    : mean {fmt(rotation_stats['mean'])} deg  "
          f"median {fmt(rotation_stats['median'])}   max {fmt(rotation_stats['max'])}")
    print(f"  chain reprojection   : mean {fmt(reprojection['mean'])} px   "
          f"median {fmt(reprojection['median'])}   max {fmt(reprojection['max'])}")

    cross_check = result.get("cross_check") or {}
    spread = cross_check.get("spread") or {}
    if spread:
        print()
        print(f"  method cross-check   : the {len(cross_check['solutions'])} solvers "
              f"agree to within {spread['max_translation_difference_mm']:.2f} mm "
              f"and {spread['max_rotation_difference_deg']:.3f} deg")
    for warning in result.get("warnings", []):
        print(f"  WARNING: {warning}")
    print("-" * 78)


def fmt(value) -> str:
    return "   -  " if value is None else f"{value:6.3f}"


if __name__ == "__main__":
    sys.exit(main())
