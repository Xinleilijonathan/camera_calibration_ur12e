#!/usr/bin/env python3
"""Final hand-eye calibration, hold-out validation, and an honest comparison.

    python scripts/solve_handeye.py --camera camera_1 --selection best20
    python scripts/solve_handeye.py --camera camera_1 --selection all

Re-solves from scratch on the selected waypoints -- it never reuses the
preliminary result -- then scores that transform on the waypoints that were
held out, and compares ALL-N against BEST-N without assuming selection helped.

Writes handeye/final_result_best_20.yaml. The preliminary result is left
untouched.
"""
from __future__ import annotations

import argparse
import re
import sys

import numpy as np

import _bootstrap  # noqa: F401
import _analysis

from calibration_utils import (CalibrationError, ConfigError,
                               describe_environment, load_yaml, save_yaml,
                               setup_logging)
from handeye_calibration import calibrate, compare, validate


def resolve_selection(context, selection: str, logger):
    """Return (records_to_fit, records_held_out, label)."""
    if selection == "all":
        return context.records, [], f"all_{len(context.records)}"

    match = re.fullmatch(r"best(\d+)", selection)
    expected = int(match.group(1)) if match else None
    candidates = sorted(context.paths.selection.glob("selected_*.yaml"))
    if not candidates:
        raise CalibrationError(
            f"No selection file in {context.paths.selection}.\n"
            f"  Run: python scripts/select_best_waypoints.py --camera "
            f"{context.camera_name}")

    chosen_path = candidates[0]
    if expected is not None:
        preferred = context.paths.selection / f"selected_{expected}.yaml"
        if preferred.is_file():
            chosen_path = preferred
        elif len(candidates) > 1:
            raise CalibrationError(
                f"--selection best{expected} but {preferred.name} does not "
                f"exist. Found: {', '.join(p.name for p in candidates)}")

    data = load_yaml(chosen_path)
    numbers = set(int(n) for n in data.get("selected_numbers", []))
    if not numbers:
        raise CalibrationError(f"{chosen_path} lists no selected waypoints")
    logger.info("Using selection %s (%d waypoints)", chosen_path.name, len(numbers))

    fit = [r for r in context.records if r.number in numbers]
    held = [r for r in context.records if r.number not in numbers]
    missing = numbers - {r.number for r in fit}
    if missing:
        raise CalibrationError(
            f"{chosen_path.name} selects waypoints {sorted(missing)} that no "
            f"longer exist on disk")
    return fit, held, f"best_{len(fit)}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Final hand-eye calibration with hold-out validation.")
    parser.add_argument("--camera", required=True)
    parser.add_argument("--selection", default="best20",
                        help="best20 / bestN / all  (default: best20)")
    parser.add_argument("--method", default=None,
                        help="override calibration.yaml handeye.method")
    parser.add_argument("--dry-run", action="store_true",
                        help="report without writing the result file")
    args = parser.parse_args(argv)

    logger = setup_logging("solve_handeye", args.camera)
    try:
        context = _analysis.load_for_analysis(args.camera, logger=logger)
        fit, held, label = resolve_selection(context, args.selection, logger)
    except (ConfigError, CalibrationError) as exc:
        return _analysis.fail(exc)

    method = args.method or context.method
    _analysis.print_context(context, "FINAL HAND-EYE CALIBRATION")
    print(f"Fitting on : {len(fit)} waypoints ({label})")
    print(f"Held out   : {len(held)} waypoints")
    print()

    try:
        # Solved from scratch. The preliminary result is never reused.
        final = calibrate(fit, context.handeye_mode, method,
                          context.camera_matrix, context.dist_coeffs,
                          cross_check=context.cross_check,
                          label=f"final_{label}")
    except CalibrationError as exc:
        return _analysis.fail(exc)

    print_transform(final, "FINAL RESULT")

    reference = np.asarray(final["reference_constant_transform"], dtype=np.float64)
    transform = np.asarray(final["transform"], dtype=np.float64)

    validation = None
    if held:
        validation = validate(held, transform, context.handeye_mode,
                              context.camera_matrix, context.dist_coeffs,
                              reference=reference)
        print_validation(validation)
    else:
        print("\nNo held-out waypoints: every observation was used for fitting,")
        print("so there is no independent check of this result.")

    comparison = None
    preliminary = None
    if context.paths.preliminary_result.is_file() and held:
        preliminary = load_yaml(context.paths.preliminary_result)
        all_transform = np.asarray(preliminary["transform"], dtype=np.float64)
        all_reference = np.asarray(
            preliminary["reference_constant_transform"], dtype=np.float64)
        # Score ALL-N on the SAME held-out waypoints, even though it was fitted
        # with them. That favours ALL-N, and is stated as such in the output.
        all_validation = validate(held, all_transform, context.handeye_mode,
                                  context.camera_matrix, context.dist_coeffs,
                                  reference=all_reference)
        comparison = compare(preliminary, final, all_validation, validation)
        print_comparison(comparison, len(context.records), len(fit))

    if args.dry_run:
        print("\n--dry-run: nothing was written.")
        return 0

    record = dict(final)
    record["camera_name"] = context.camera_name
    record["camera_serial"] = str(context.camera_config.get("serial", ""))
    record["handeye_mode"] = context.handeye_mode
    record["selection"] = args.selection
    record["selected_waypoints"] = [r.name for r in fit]
    record["selected_numbers"] = [r.number for r in fit]
    record["holdout_waypoints"] = [r.name for r in held]
    record["holdout_numbers"] = [r.number for r in held]
    record["intrinsics_file"] = str(context.paths.intrinsics_result)
    record["intrinsics_rms_px"] = context.intrinsics.get("rms_reprojection_error_px")
    record["board"] = context.calibration_config["apriltag_grid"]
    record["validation"] = validation
    record["comparison_with_all"] = comparison
    record["environment"] = describe_environment()

    output = (context.paths.final_result if args.selection != "all"
              else context.paths.handeye / f"final_result_all_{len(fit)}.yaml")
    save_yaml(output, record, header=(
        f"FINAL hand-eye calibration for {context.camera_name} "
        f"(serial {record['camera_serial']}).\n"
        f"Mounting: {context.handeye_mode}. "
        f"{final['transform_meaning']}.\n"
        f"Fitted on {len(fit)} waypoints; {len(held)} held out for validation."))

    print()
    print(f"Saved: {output}")
    if context.paths.preliminary_result.is_file():
        print(f"Preliminary result left untouched: "
              f"{context.paths.preliminary_result}")
    print_summary(context, final, validation, comparison, preliminary)
    print()
    print(f"Next: python scripts/verify_calibration.py --camera {args.camera}")
    return 0


def print_transform(result, title: str) -> None:
    print("-" * 78)
    print(f"{title} ({result['observation_count']} observations, "
          f"{result['method']}, {result['mode']})")
    print("-" * 78)
    print(f"  {result['transform_meaning']}")
    print()
    for row in np.asarray(result["transform"]):
        print("    [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")
    print()
    translation = result["translation_mm"]
    print(f"  translation : {translation[0]:8.2f} {translation[1]:8.2f} "
          f"{translation[2]:8.2f}  mm")
    print(f"  rotation    : {result['rotation_angle_deg']:.3f} deg about "
          + " ".join(f"{v:+.4f}" for v in result["rotation_vector"]))
    print()
    print(f"  translation residual : mean {fmt(result['translation_residual_mm']['mean'])} mm"
          f"   max {fmt(result['translation_residual_mm']['max'])}")
    print(f"  rotation residual    : mean {fmt(result['rotation_residual_deg']['mean'])} deg"
          f"   max {fmt(result['rotation_residual_deg']['max'])}")
    print(f"  chain reprojection   : mean {fmt(result['reprojection_px']['mean'])} px"
          f"   max {fmt(result['reprojection_px']['max'])}")
    spread = (result.get("cross_check") or {}).get("spread") or {}
    if spread:
        print(f"  solver agreement     : "
              f"{spread['max_translation_difference_mm']:.2f} mm, "
              f"{spread['max_rotation_difference_deg']:.3f} deg")
    for warning in result.get("warnings", []):
        print(f"  WARNING: {warning}")
    print("-" * 78)


def print_validation(validation) -> None:
    print()
    print("-" * 78)
    print(f"HOLD-OUT VALIDATION ({validation['count']} observations "
          f"NOT used in the fit)")
    print("-" * 78)
    print(f"  {validation['note']}")
    print()
    translation = validation["translation_error_mm"]
    rotation = validation["rotation_error_deg"]
    reprojection = validation["reprojection_px"]
    print(f"  translation : mean {fmt(translation['mean'])} mm   "
          f"median {fmt(translation['median'])}   max {fmt(translation['max'])}")
    print(f"  rotation    : mean {fmt(rotation['mean'])} deg  "
          f"median {fmt(rotation['median'])}   max {fmt(rotation['max'])}")
    print(f"  reprojection: mean {fmt(reprojection['mean'])} px   "
          f"median {fmt(reprojection['median'])}   max {fmt(reprojection['max'])}")
    print("-" * 78)


def print_comparison(comparison, total, fitted) -> None:
    print()
    print("-" * 78)
    print(f"ALL-{total} vs BEST-{fitted}")
    print("-" * 78)
    print(f"  {'metric':<34}{'ALL':>14}{'BEST':>14}")
    rows = [
        ("fit reprojection mean (px)", "reprojection_mean_px"),
        ("fit reprojection median (px)", "reprojection_median_px"),
        ("fit reprojection max (px)", "reprojection_max_px"),
        ("fit translation residual (mm)", "translation_residual_mean_mm"),
        ("fit rotation residual (deg)", "rotation_residual_mean_deg"),
    ]
    for label, key in rows:
        print(f"  {label:<34}{fmt(comparison['all'][key]):>14}"
              f"{fmt(comparison['best'][key]):>14}")
    holdout = comparison.get("holdout")
    if holdout:
        print(f"  {'HELD-OUT reprojection mean (px)':<34}"
              f"{fmt(holdout['all_mean_px']):>14}{fmt(holdout['best_mean_px']):>14}")
    difference = comparison["difference_between_solutions"]
    print()
    print(f"  The two solutions differ by "
          f"{difference['translation_mm']:.2f} mm and "
          f"{difference['rotation_deg']:.3f} deg")
    print()
    print(f"  RECOMMENDED: {comparison['recommendation']}")
    for reason in comparison["reasons"]:
        print(f"    * {reason}")
    print("-" * 78)


def print_summary(context, final, validation, comparison, preliminary) -> None:
    print()
    print("=" * 78)
    print(f"{context.camera_name.upper()} CALIBRATION COMPLETE")
    print("=" * 78)
    print()
    print(f"Camera                  : {context.camera_name} "
          f"({context.camera_config.get('model', '?')})")
    print(f"Serial                  : {context.camera_config.get('serial')}")
    print(f"Mounting                : {context.handeye_mode}")
    print(f"Intrinsic observations  : {context.intrinsics.get('observation_count')}")
    print(f"Intrinsic RMS           : "
          f"{context.intrinsics.get('rms_reprojection_error_px', float('nan')):.4f} px")
    print(f"Hand-eye collected      : {len(context.records)}")
    print(f"Used for final fit      : {final['observation_count']}")
    print(f"Held out                : {len(context.records) - final['observation_count']}")
    print()
    if preliminary:
        print(f"ALL-{preliminary['observation_count']} PRELIMINARY")
        print(f"  reprojection mean     : {fmt(preliminary['reprojection_px']['mean'])} px")
        print(f"  translation residual  : {fmt(preliminary['translation_residual_mm']['mean'])} mm")
        print(f"  rotation residual     : {fmt(preliminary['rotation_residual_deg']['mean'])} deg")
        print()
    print(f"BEST-{final['observation_count']} FINAL")
    print(f"  reprojection mean     : {fmt(final['reprojection_px']['mean'])} px")
    print(f"  translation residual  : {fmt(final['translation_residual_mm']['mean'])} mm")
    print(f"  rotation residual     : {fmt(final['rotation_residual_deg']['mean'])} deg")
    print()
    if validation:
        print("HELD-OUT VALIDATION")
        print(f"  mean reprojection     : {fmt(validation['reprojection_px']['mean'])} px")
        print(f"  median reprojection   : {fmt(validation['reprojection_px']['median'])} px")
        print(f"  maximum reprojection  : {fmt(validation['reprojection_px']['max'])} px")
        print(f"  translation mean      : {fmt(validation['translation_error_mm']['mean'])} mm")
        print(f"  rotation mean         : {fmt(validation['rotation_error_deg']['mean'])} deg")
        print()
    print("Final transform:")
    print(f"  {final['transform_meaning']}")
    for row in np.asarray(final["transform"]):
        print("    [" + "  ".join(f"{v:9.5f}" for v in row) + " ]")
    print()
    if comparison:
        print(f"Recommended result      : {comparison['recommendation']}")
    print(f"Saved                   : {context.paths.handeye}")
    print("=" * 78)


def fmt(value) -> str:
    return "     -" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    sys.exit(main())
