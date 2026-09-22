#!/usr/bin/env python3
"""Select the best N waypoints: low error AND geometric diversity.

    python scripts/select_best_waypoints.py --camera camera_1 --count 20

Writes:
    handeye/selection/selected_20.yaml
    handeye/selection/rejected_10.yaml
    handeye/selection/waypoint_scores.csv   (updated with the verdicts)

NOTHING IS DELETED. Rejection is a label; every image and every metadata file
stays exactly where it was.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401
import _analysis

from calibration_utils import (CalibrationError, ConfigError, load_yaml,
                               save_csv, save_yaml, setup_logging,
                               timestamp_utc)
from waypoint_quality import (SCORE_COLUMNS, compute_combined_scores,
                              compute_scores, flag_outliers, format_table,
                              select_best, selection_summary)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Select the best N waypoints by combined error and "
                    "geometric diversity.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--count", type=int, default=None,
                        help="how many to select (default: waypoint_collection.final_count)")
    args = parser.parse_args(argv)

    logger = setup_logging("select_best_waypoints", args.camera)
    try:
        context = _analysis.load_for_analysis(args.camera, logger=logger)
    except (ConfigError, CalibrationError) as exc:
        return _analysis.fail(exc)

    if not context.paths.preliminary_result.is_file():
        print(f"ERROR: no preliminary result at {context.paths.preliminary_result}\n"
              f"  Run: python scripts/analyze_waypoints.py --camera {args.camera}",
              file=sys.stderr)
        return 1
    preliminary = load_yaml(context.paths.preliminary_result)

    count = args.count or int(
        context.calibration_config["waypoint_collection"]["final_count"])
    selection_config = context.calibration_config.get("selection", {})

    _analysis.print_context(context, "WAYPOINT SELECTION")
    print(f"Mode        : {selection_config.get('mode', 'best_error_with_diversity')}")
    weights = selection_config.get("weights", {})
    print(f"Weights     : reprojection {weights.get('reprojection_rms')}  "
          f"translation {weights.get('handeye_translation')}  "
          f"rotation {weights.get('handeye_rotation')}")
    print(f"Selecting   : {count} of {len(context.records)}")
    print()

    scores = compute_scores(context.records, preliminary)
    scores = flag_outliers(scores, selection_config)
    scores = compute_combined_scores(scores, selection_config)
    result = select_best(scores, context.records, count, selection_config)

    print(format_table(scores))
    print()
    print(f"Diversity thresholds used: "
          f"{result['translation_threshold_mm']:.1f} mm OR "
          f"{result['rotation_threshold_deg']:.2f} deg "
          f"({result['factor_used'] * 100:.0f}% of configured)")
    print(f"Note: {result['note']}")
    print()

    summary = selection_summary(scores)
    print(f"SELECTED {summary['selected_count']}: "
          f"{', '.join(str(n) for n in result['selected'])}")
    print(f"NOT SELECTED {summary['rejected_count']}: "
          f"{', '.join(str(n) for n in result['rejected'])}")
    print()
    print("Reasons for every waypoint that was not selected:")
    for name, entry in summary["rejected_waypoints"].items():
        marker = "OUTLIER " if entry["outlier"] else "         "
        print(f"  {marker}{name}: {entry['reason']}")

    selected_path = context.paths.selection / f"selected_{summary['selected_count']}.yaml"
    rejected_path = context.paths.selection / f"rejected_{summary['rejected_count']}.yaml"

    save_yaml(selected_path, {
        "camera_name": context.camera_name,
        "camera_serial": str(context.camera_config.get("serial", "")),
        "timestamp": timestamp_utc(),
        "selection_mode": selection_config.get("mode", "best_error_with_diversity"),
        "weights": weights,
        "diversity_threshold_translation_mm": result["translation_threshold_mm"],
        "diversity_threshold_rotation_deg": result["rotation_threshold_deg"],
        "threshold_relaxation_factor": result["factor_used"],
        "note": result["note"],
        "count": summary["selected_count"],
        "selected_waypoints": summary["selected_waypoints"],
        "selected_numbers": result["selected"],
    }, header="Waypoints chosen for the FINAL hand-eye calibration.")

    save_yaml(rejected_path, {
        "camera_name": context.camera_name,
        "timestamp": timestamp_utc(),
        "count": summary["rejected_count"],
        "note": ("These observations remain on disk in full. They are excluded "
                 "from the final fit and used as the hold-out validation set."),
        **{name: entry for name, entry in summary["rejected_waypoints"].items()},
    }, header="Waypoints NOT used in the final fit, each with a reason.")

    save_csv(context.paths.waypoint_scores, [s.as_row() for s in scores],
             SCORE_COLUMNS)

    print()
    print(f"Saved: {selected_path}")
    print(f"Saved: {rejected_path}")
    print(f"Saved: {context.paths.waypoint_scores}")
    print()
    print(f"Next: python scripts/solve_handeye.py --camera {args.camera} "
          f"--selection best{summary['selected_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
