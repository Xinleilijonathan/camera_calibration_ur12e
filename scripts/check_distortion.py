#!/usr/bin/env python3
"""Test a camera's factory distortion coefficients against real board views.

    python scripts/check_distortion.py --camera camera_2

Written for the case this rig actually has: the D435IF colour streams report
ALL-ZERO distortion. That is either true (the stream is rectified in the
ASIC) or the field was never populated (and the lens really does bend
straight lines). Both look identical in result.yaml, and only one of them
lets you skip an intrinsic calibration.

The test holds fx, fy, cx, cy fixed at the factory values and refits the
distortion terms against your own board views. If the lens is genuinely
rectified they stay at zero. If it is not, they run away, and the script
tells you how many pixels of error you would have accepted by trusting the
zeros.

WHY IT REFUSES TO ANSWER FROM CENTRE-HEAVY DATA
    Radial distortion is ~zero at the principal point and grows with r^4.
    A set of views that never pushed the board into the image corners cannot
    distinguish k1 = 0 from k1 = 0.1, and would happily report "confirmed".
    So the coverage of the observations is checked first, and a set that
    cannot decide the question is reported as INCONCLUSIVE rather than pass.

WHAT IT SCORES, AND WHY NOT PIXELS
    The verdict is the shift in BOARD POSE between the two distortion
    models, in mm and degrees, against the budget in
    calibration.yaml -> verification. Both pixel proxies mislead here:

      * reprojection RMS understates it, because re-fitting the pose
        absorbs much of a distortion change -- which is exactly the damage.
        The error stays flat while the pose silently goes wrong, and
        hand-eye consumes the pose, not the residual.
      * raw pixel disagreement overstates it, because outside the observed
        region the refitted coefficients are unconstrained and the models
        drift apart by whatever the extrapolation does. On a set with an
        empty top third that reads as several pixels while the fit barely
        moved.

    Both are still printed, because they explain the verdict. Only the
    pose shift decides it.

This is a screening test, not a calibration. Passing it means the factory
numbers are good enough to proceed; it does not produce better ones.
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               load_calibration_config, load_cameras_config,
                               load_yaml, resolve_camera, setup_logging)
from intrinsic_calibration import IntrinsicObservation, load_intrinsics

# A set that never leaves the middle of the frame cannot answer the question.
MINIMUM_RADIUS_FRACTION = 0.75    # furthest corner must reach this much of r_max
MINIMUM_OUTER_FRACTION = 0.05     # and this share of points must sit beyond 0.7 r_max
MINIMUM_VIEWS = 6
# Radial coverage alone is not enough: a set that only ever went left and
# right reaches a high radius while leaving the top and bottom of the frame
# unconstrained, and radial distortion is then pinned on one axis only. Each
# side's outer band must have been visited too.
EDGE_BAND_FRACTION = 0.15         # outer 15% of each side counts as that edge
MINIMUM_EDGE_POINTS = 20          # points needed in each of the four bands


def load_observations(paths) -> list[IntrinsicObservation]:
    observations = []
    for path in sorted(paths.intrinsics_observations.glob("observation_*.yaml")):
        try:
            observations.append(IntrinsicObservation.from_dict(load_yaml(path)))
        except Exception as exc:
            raise CalibrationError(f"Unreadable observation {path}: {exc}") from exc
    return observations


def board_poses(observations, camera_matrix, dist_coeffs):
    """Board pose per view under one distortion model."""
    poses = []
    for observation in observations:
        ok, rvec, tvec = cv2.solvePnP(
            observation.object_points, observation.image_points,
            camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationError(
                f"PnP failed on observation {observation.index}")
        rvec, tvec = cv2.solvePnPRefineLM(
            observation.object_points, observation.image_points,
            camera_matrix, dist_coeffs, rvec, tvec)
        poses.append((rvec, tvec))
    return poses


def pose_impact(observations, camera_matrix, reference, fitted) -> dict:
    """How far the board pose moves when the distortion model changes.

    This, not the pixel disagreement, is what the verdict is built on. A
    change in distortion is partly absorbed by the pose -- which is precisely
    the damage: the reprojection error stays flat while the pose silently
    goes wrong, and hand-eye consumes the pose. Measuring the pose directly
    sidesteps both misleading proxies (reprojection RMS understates it,
    raw pixel disagreement overstates the part that matters).
    """
    a = board_poses(observations, camera_matrix, reference)
    b = board_poses(observations, camera_matrix, fitted)
    translations, rotations = [], []
    for (rvec_a, tvec_a), (rvec_b, tvec_b) in zip(a, b):
        translations.append(float(np.linalg.norm(tvec_a - tvec_b) * 1000.0))
        delta = (cv2.Rodrigues(rvec_a)[0] @ cv2.Rodrigues(rvec_b)[0].T)
        rotations.append(float(np.degrees(
            np.arccos(np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)))))
    return {
        "translation_median_mm": float(np.median(translations)),
        "translation_max_mm": float(np.max(translations)),
        "rotation_median_deg": float(np.median(rotations)),
        "rotation_max_deg": float(np.max(rotations)),
    }


def pose_only_rms(observations, camera_matrix, dist_coeffs) -> float:
    """Reprojection RMS with the intrinsics frozen and only the poses fitted.

    This is the honest way to score a set of intrinsics you did not derive
    from this data: it gives them no freedom to absorb their own error.
    """
    residuals = []
    for observation in observations:
        ok, rvec, tvec = cv2.solvePnP(
            observation.object_points, observation.image_points,
            camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise CalibrationError(
                f"PnP failed on observation {observation.index}")
        rvec, tvec = cv2.solvePnPRefineLM(
            observation.object_points, observation.image_points,
            camera_matrix, dist_coeffs, rvec, tvec)
        projected, _ = cv2.projectPoints(
            observation.object_points, rvec, tvec, camera_matrix, dist_coeffs)
        residuals.extend(np.linalg.norm(
            projected.reshape(-1, 2) - observation.image_points, axis=1))
    return float(np.sqrt(np.mean(np.square(residuals))))


def model_disagreement(camera_matrix, reference, fitted, points) -> dict:
    """How far apart two distortion models put the same pixel, in pixels.

    k1 on its own is not interpretable -- its effect depends on focal length
    and on where the image corners land in normalised coordinates. The pixel
    disagreement is the number worth putting a threshold on, and unlike a
    coefficient comparison it works whether the reference coefficients are
    zero or not: against an all-zero reference it is simply the magnitude of
    the distortion the data found.

    `points` must be the observed corners, NOT a grid over the whole frame.
    Outside the region the board actually visited, the freed coefficients are
    unconstrained and the two models diverge by whatever the extrapolation
    happens to do -- on a set with an empty top third that reads as several
    pixels of "disagreement" while the reprojection error says the fit barely
    moved. Scoring only where there is data keeps the number honest.
    """
    observed = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    a = cv2.undistortPoints(observed, camera_matrix, reference,
                            P=camera_matrix).reshape(-1, 2)
    b = cv2.undistortPoints(observed, camera_matrix, fitted,
                            P=camera_matrix).reshape(-1, 2)
    shift = np.linalg.norm(a - b, axis=1)
    return {"max": float(shift.max()),
            "p99": float(np.percentile(shift, 99)),
            "mean": float(shift.mean())}


def extrapolated_disagreement(camera_matrix, reference, fitted, size) -> float:
    """Same comparison at the four image corners -- reported, never scored.

    Useful to know how far the two models have drifted apart out where the
    board never went, because that is where a later session may put it.
    """
    width, height = size
    corners = np.array([[0.0, 0.0], [width, 0.0], [0.0, height],
                        [width, height]]).reshape(-1, 1, 2)
    a = cv2.undistortPoints(corners, camera_matrix, reference,
                            P=camera_matrix).reshape(-1, 2)
    b = cv2.undistortPoints(corners, camera_matrix, fitted,
                            P=camera_matrix).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).max())


def coverage(observations, camera_matrix, size) -> dict:
    """How much of the image the board corners actually visited."""
    width, height = size
    points = np.vstack([o.image_points for o in observations])
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    radii = np.hypot(points[:, 0] - cx, points[:, 1] - cy)
    corner_radius = max(np.hypot(x - cx, y - cy)
                        for x in (0, width) for y in (0, height))
    band_x, band_y = width * EDGE_BAND_FRACTION, height * EDGE_BAND_FRACTION
    edges = {
        "left": int(np.sum(points[:, 0] < band_x)),
        "right": int(np.sum(points[:, 0] > width - band_x)),
        "top": int(np.sum(points[:, 1] < band_y)),
        "bottom": int(np.sum(points[:, 1] > height - band_y)),
    }
    return {
        "points": int(len(points)),
        "radius_fraction": float(radii.max() / corner_radius),
        "outer_fraction": float(np.mean(radii > 0.7 * corner_radius)),
        "x_min": float(points[:, 0].min()), "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()), "y_max": float(points[:, 1].max()),
        "edges": edges,
        "starved_edges": sorted(name for name, count in edges.items()
                                if count < MINIMUM_EDGE_POINTS),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Check factory distortion coefficients against board views.")
    parser.add_argument("--camera", required=True,
                        help="logical camera name, e.g. camera_2")
    parser.add_argument("--budget-fraction", type=float, default=0.2,
                        help="share of the verification pose budget the "
                             "intrinsics may consume before the reference is "
                             "no longer considered confirmed (default 0.2); "
                             "twice this is the MARGINAL band")
    args = parser.parse_args(argv)

    logger = setup_logging("check_distortion", args.camera)

    try:
        cameras_config = load_cameras_config()
        camera_config = resolve_camera(args.camera, cameras_config)
        calibration_config = load_calibration_config()
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    # Score against the tolerances this project already set for itself, so a
    # pass here means something in the units the rest of the pipeline uses.
    verification = calibration_config.get("verification", {})
    budget_mm = float(verification.get("maximum_validation_translation_mm", 10.0))
    budget_deg = float(verification.get("maximum_validation_rotation_deg", 2.0))
    fraction = max(0.0, min(1.0, args.budget_fraction))
    confirm_mm, confirm_deg = budget_mm * fraction, budget_deg * fraction
    marginal_mm, marginal_deg = budget_mm * 2 * fraction, budget_deg * 2 * fraction

    paths = camera_paths(args.camera)

    try:
        camera_matrix, dist_coeffs, intrinsics = load_intrinsics(
            paths.intrinsics_result)
        observations = load_observations(paths)
    except CalibrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if len(observations) < MINIMUM_VIEWS:
        print(f"ERROR: only {len(observations)} observation(s) in "
              f"{paths.intrinsics_observations}; need at least {MINIMUM_VIEWS}.\n"
              f"  Run: python scripts/collect_intrinsics.py --camera "
              f"{args.camera} --target-count 15", file=sys.stderr)
        return 1

    sizes = {o.image_size for o in observations}
    if len(sizes) > 1:
        print(f"ERROR: observations were captured at mixed resolutions: {sizes}",
              file=sys.stderr)
        return 1
    image_size = observations[0].image_size
    expected = (int(intrinsics.get("image_width", 0)),
                int(intrinsics.get("image_height", 0)))
    if expected != image_size:
        print(f"ERROR: result.yaml is for {expected[0]}x{expected[1]} but the "
              f"observations are {image_size[0]}x{image_size[1]}.",
              file=sys.stderr)
        return 1

    reference = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1)
    source = intrinsics.get("source", "solved")

    print("=" * 74)
    print(f"DISTORTION CHECK -- {args.camera}")
    print("=" * 74)
    print(f"Camera       : {camera_config.get('model', '?')} "
          f"serial {camera_config.get('serial')}")
    print(f"Reference    : {paths.intrinsics_result} (source: {source})")
    print(f"  fx {camera_matrix[0, 0]:.3f}  fy {camera_matrix[1, 1]:.3f}  "
          f"cx {camera_matrix[0, 2]:.3f}  cy {camera_matrix[1, 2]:.3f}")
    print("  dist " + "  ".join(f"{c: .6f}" for c in reference))
    print(f"Observations : {len(observations)} views, "
          f"{sum(len(o.object_points) for o in observations)} points")
    print()

    # ---- Can this data answer the question at all? --------------------
    cover = coverage(observations, camera_matrix, image_size)
    print("-" * 74)
    print("COVERAGE (distortion is only observable away from the centre)")
    print(f"  corners reached {cover['radius_fraction'] * 100:.0f}% of the "
          f"image-corner radius (need >= {MINIMUM_RADIUS_FRACTION * 100:.0f}%)")
    print(f"  {cover['outer_fraction'] * 100:.1f}% of points lie beyond 0.7 r "
          f"(need >= {MINIMUM_OUTER_FRACTION * 100:.0f}%)")
    print(f"  x spanned {cover['x_min']:.0f}..{cover['x_max']:.0f} of "
          f"0..{image_size[0]}")
    print(f"  y spanned {cover['y_min']:.0f}..{cover['y_max']:.0f} of "
          f"0..{image_size[1]}")
    print(f"  points in the outer {EDGE_BAND_FRACTION:.0%} of each side "
          f"(need >= {MINIMUM_EDGE_POINTS} each):")
    print("    " + "   ".join(
        f"{name} {cover['edges'][name]}"
        + ("" if cover["edges"][name] >= MINIMUM_EDGE_POINTS else " <-- starved")
        for name in ("left", "right", "top", "bottom")))
    decisive = (cover["radius_fraction"] >= MINIMUM_RADIUS_FRACTION
                and cover["outer_fraction"] >= MINIMUM_OUTER_FRACTION
                and not cover["starved_edges"])
    print()

    # ---- Refit the distortion; geometry frozen ------------------------
    # k3 is held at the reference value rather than freed: with the board
    # confined to a limited radius, k3 is the term that runs away and buys
    # its fit by distorting the extrapolated corners. Freezing it keeps the
    # comparison about k1/k2, which is where a real lens shows up first.
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS
             | cv2.CALIB_FIX_FOCAL_LENGTH
             | cv2.CALIB_FIX_PRINCIPAL_POINT
             | cv2.CALIB_FIX_K3)
    object_points = [o.object_points.astype(np.float32).reshape(-1, 1, 3)
                     for o in observations]
    image_points = [o.image_points.astype(np.float32).reshape(-1, 1, 2)
                    for o in observations]
    # calibrateCamera* writes the fitted values back into the arrays it is
    # handed, so the reference has to be kept in a copy OpenCV cannot reach --
    # otherwise the comparison below is the fit against itself.
    reference_dist = np.zeros(5, dtype=np.float64)
    reference_dist[:len(reference)] = reference[:5]
    guess = camera_matrix.copy()
    seed = reference_dist.copy()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    _, _, fitted_dist, _, _, deviations, _, _ = \
        cv2.calibrateCameraExtended(
            object_points, image_points, image_size, guess, seed,
            flags=flags, criteria=criteria)
    fitted = np.asarray(fitted_dist, dtype=np.float64).reshape(-1)[:5]
    sigma = np.asarray(deviations, dtype=np.float64).reshape(-1)

    print("-" * 74)
    print("FIT (fx, fy, cx, cy frozen; k3 held at the reference; k1 k2 p1 p2 free)")
    print(f"  {'':4}  {'reference':>12}  {'fitted':>12}  {'1 sigma':>10}")
    for offset, name in enumerate(("k1", "k2", "p1", "p2")):
        print(f"  {name:4}  {reference_dist[offset]:>+12.6f}  "
              f"{fitted[offset]:>+12.6f}  {sigma[4 + offset]:>10.6f}")
    print(f"  {'k3':4}  {reference_dist[4]:>+12.6f}  {fitted[4]:>+12.6f}  "
          f"{'(frozen)':>10}")
    print()

    observed_points = np.vstack([o.image_points for o in observations])
    disagreement = model_disagreement(
        camera_matrix, reference_dist, fitted, observed_points)
    extrapolated = extrapolated_disagreement(
        camera_matrix, reference_dist, fitted, image_size)
    reference_rms = pose_only_rms(observations, camera_matrix, reference_dist)
    fitted_rms = pose_only_rms(observations, camera_matrix, fitted)
    fitted_max = disagreement["max"]

    print("WHAT THAT COSTS IN PIXELS")
    if np.all(np.abs(reference_dist) < 1e-12):
        print("  reference is all-zero, so the disagreement below IS the")
        print("  distortion this data says the lens has.")
    print(f"  reference vs fitted, AT THE OBSERVED CORNERS : "
          f"max {fitted_max:.2f} px, "
          f"p99 {disagreement['p99']:.2f} px, mean {disagreement['mean']:.2f} px")
    print(f"  reprojection RMS, reference coefficients     : "
          f"{reference_rms:.4f} px")
    print(f"  reprojection RMS, fitted coefficients        : "
          f"{fitted_rms:.4f} px")
    print(f"  improvement from refitting distortion        : "
          f"{reference_rms - fitted_rms:+.4f} px")
    impact = pose_impact(observations, camera_matrix, reference_dist, fitted)
    print()
    print("WHAT THAT COSTS IN BOARD POSE  (this is what hand-eye consumes)")
    print(f"  translation shift : median "
          f"{impact['translation_median_mm']:.2f} mm, "
          f"max {impact['translation_max_mm']:.2f} mm")
    print(f"  rotation shift    : median "
          f"{impact['rotation_median_deg']:.3f} deg, "
          f"max {impact['rotation_max_deg']:.3f} deg")
    print(f"  budget            : {budget_mm:.1f} mm / {budget_deg:.1f} deg "
          f"(verification.maximum_validation_*)")
    print(f"  confirmed below   : {confirm_mm:.2f} mm / {confirm_deg:.2f} deg "
          f"({args.budget_fraction:.0%} of budget)")
    print()
    print(f"  (for information, not scored: at the image corners, where the")
    print(f"   board never went, the two models differ by "
          f"{extrapolated:.2f} px. That is")
    print(f"   extrapolation, not evidence -- it says the freed coefficients")
    print(f"   are unconstrained out there, not that the reference is wrong.)")
    print()

    # ---- Verdict -------------------------------------------------------
    print("=" * 74)
    if not decisive:
        print("VERDICT: INCONCLUSIVE")
        print()
        if cover["starved_edges"]:
            print(f"  The board never reached the "
                  f"{', '.join(cover['starved_edges'])} of the frame, so the")
            print("  distortion there is unconstrained and a 'pass' would only")
            print("  describe the middle of the image.")
        else:
            print("  The board never got far enough from the image centre for")
            print("  this data to tell k1 = 0 from k1 != 0.")
        print("  A 'pass' here would mean nothing, so none is given.")
        print()
        print(f"  Re-collect, deliberately pushing the board into all four")
        print(f"  corners of the frame:")
        print(f"    python scripts/collect_intrinsics.py --camera {args.camera} "
              f"--target-count 15")
        print("=" * 74)
        logger.info("Distortion check inconclusive for %s "
                    "(radius %.2f, outer %.3f, starved %s)",
                    args.camera, cover["radius_fraction"],
                    cover["outer_fraction"],
                    cover["starved_edges"] or "none")
        return 3

    within_confirm = (impact["translation_median_mm"] <= confirm_mm
                      and impact["rotation_median_deg"] <= confirm_deg)
    within_marginal = (impact["translation_median_mm"] <= marginal_mm
                       and impact["rotation_median_deg"] <= marginal_deg)

    if within_confirm:
        print("VERDICT: REFERENCE COEFFICIENTS CONFIRMED")
        print()
        print(f"  Refitting the distortion against your own board views moves")
        print(f"  the board pose by {impact['translation_median_mm']:.2f} mm "
              f"/ {impact['rotation_median_deg']:.3f} deg (median),")
        print(f"  within {confirm_mm:.2f} mm / {confirm_deg:.2f} deg. The "
              f"values in result.yaml are")
        print("  good enough to proceed to hand-eye.")
        print("=" * 74)
        logger.info("Distortion check passed for %s (%.2f mm, %.3f deg)",
                    args.camera, impact["translation_median_mm"],
                    impact["rotation_median_deg"])
        return 0

    if within_marginal:
        print("VERDICT: MARGINAL")
        print()
        print(f"  Refitting the distortion moves the board pose by")
        print(f"  {impact['translation_median_mm']:.2f} mm / "
              f"{impact['rotation_median_deg']:.3f} deg (median) -- past "
              f"{confirm_mm:.2f} mm / {confirm_deg:.2f} deg but")
        print(f"  still inside half the verification budget. Usable if your")
        print("  error budget has room; run a full intrinsic calibration")
        print("  if it does not.")
        print("=" * 74)
        logger.warning("Distortion check marginal for %s (%.2f mm, %.3f deg)",
                       args.camera, impact["translation_median_mm"],
                       impact["rotation_median_deg"])
        return 1

    print("VERDICT: REFERENCE COEFFICIENTS REFUTED")
    print()
    print(f"  Refitting the distortion moves the board pose by")
    print(f"  {impact['translation_median_mm']:.2f} mm / "
          f"{impact['rotation_median_deg']:.3f} deg (median), past "
          f"{marginal_mm:.2f} mm / {marginal_deg:.2f} deg.")
    print("  The reference coefficients bias every board pose, and hand-eye")
    print("  consumes exactly those poses.")
    print()
    print("  Run a real intrinsic calibration for this camera:")
    print(f"    python scripts/collect_intrinsics.py --camera {args.camera}")
    print(f"    python scripts/solve_intrinsics.py   --camera {args.camera}")
    print("=" * 74)
    logger.error("Distortion check refuted for %s (%.2f mm, %.3f deg)",
                 args.camera, impact["translation_median_mm"],
                 impact["rotation_median_deg"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
