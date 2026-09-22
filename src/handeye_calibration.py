"""Hand-eye calibration for both mountings, with honest residual reporting.

Pure maths over stored waypoints. Opens nothing, commands nothing.

THE TWO MOUNTINGS
-----------------
This rig uses both, so neither is ever assumed:

  eye_in_hand  (camera_1, D405 on the wrist)
      Camera rides on the flange. Board is FIXED in the world.
      Unknown X = T_flange_camera  (camera pose in the flange frame).
      Invariant: T_base_board = T_base_flange @ X @ T_camera_board
                 is the same for every waypoint.

  eye_to_hand  (camera_2 / camera_3, D435s at the side)
      Camera is FIXED in the world. Board rides on the flange.
      Unknown X = T_base_camera   (camera pose in the robot base frame).
      Invariant: T_flange_board = inv(T_base_flange) @ X @ T_camera_board
                 is the same for every waypoint.

In both cases the residual of a waypoint is how far ITS estimate of that
supposedly-constant transform sits from the set's robust consensus. That is a
physically meaningful number in millimetres and degrees, unlike the algebraic
residual of the underlying AX=XB solve.

WHY cv2.calibrateHandEye IS CALLED WITH DIFFERENT ARGUMENTS PER MODE
--------------------------------------------------------------------
OpenCV's routine always solves "camera relative to the moving frame". For
eye-in-hand the moving frame is the flange, so it is fed flange-in-base
directly. For eye-to-hand the moving frame is the board on the flange, so it
is fed the INVERSE, base-in-flange, and then returns camera-in-base. Getting
this backwards produces a clean-looking, completely wrong transform.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from calibration_utils import (CalibrationError, error_statistics,
                               invert_transform, make_transform,
                               matrix_to_rotvec, rotation_angle_deg,
                               timestamp_utc, transform_difference)

LOGGER = logging.getLogger(__name__)

EYE_IN_HAND = "eye_in_hand"
EYE_TO_HAND = "eye_to_hand"

#: Method name -> OpenCV flag.
METHODS = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


def _decompose(transforms: Sequence[np.ndarray]) -> tuple[list, list]:
    """Split 4x4 transforms into the R/t lists OpenCV expects."""
    rotations = [np.asarray(t, dtype=np.float64)[:3, :3] for t in transforms]
    translations = [np.asarray(t, dtype=np.float64)[:3, 3].reshape(3, 1)
                    for t in transforms]
    return rotations, translations


def solve_handeye(robot_transforms: Sequence[np.ndarray],
                  board_transforms: Sequence[np.ndarray],
                  mode: str,
                  method: str = "park") -> np.ndarray:
    """Solve for X. Returns a 4x4.

    robot_transforms : T_base_flange for each waypoint (from actual TCP pose)
    board_transforms : T_camera_board for each waypoint (from PnP)
    mode             : eye_in_hand -> X = T_flange_camera
                       eye_to_hand -> X = T_base_camera
    """
    if mode not in (EYE_IN_HAND, EYE_TO_HAND):
        raise CalibrationError(
            f"Unknown hand-eye mode {mode!r}. Expected {EYE_IN_HAND} or {EYE_TO_HAND}")
    if method not in METHODS:
        raise CalibrationError(
            f"Unknown hand-eye method {method!r}. Supported: "
            f"{', '.join(sorted(METHODS))}")
    if len(robot_transforms) != len(board_transforms):
        raise CalibrationError(
            f"Mismatched inputs: {len(robot_transforms)} robot poses vs "
            f"{len(board_transforms)} board poses")
    if len(robot_transforms) < 3:
        raise CalibrationError(
            f"Hand-eye calibration needs at least 3 observations, got "
            f"{len(robot_transforms)}. In practice it needs many more, with "
            f"rotation about several distinct axes.")

    if mode == EYE_IN_HAND:
        moving = list(robot_transforms)                       # T_base_flange
    else:
        # The board is what moves, so feed base-relative-to-flange.
        moving = [invert_transform(t) for t in robot_transforms]

    R_moving, t_moving = _decompose(moving)
    R_board, t_board = _decompose(board_transforms)

    try:
        R_x, t_x = cv2.calibrateHandEye(
            R_moving, t_moving, R_board, t_board, method=METHODS[method])
    except cv2.error as exc:
        raise CalibrationError(
            f"cv2.calibrateHandEye failed ({method}): {exc}\n"
            f"  This usually means the poses are degenerate -- too few "
            f"distinct rotation axes, or all rotations about one axis.") from exc

    X = make_transform(np.asarray(R_x, dtype=np.float64),
                       np.asarray(t_x, dtype=np.float64).reshape(3))
    if not np.all(np.isfinite(X)):
        raise CalibrationError(
            f"Hand-eye solve returned a non-finite transform with method "
            f"{method}. The pose set is almost certainly degenerate.")
    return X


def constant_transforms(robot_transforms: Sequence[np.ndarray],
                        board_transforms: Sequence[np.ndarray],
                        X: np.ndarray, mode: str) -> list[np.ndarray]:
    """Per-waypoint estimate of the transform that should be constant.

    eye_in_hand -> T_base_board   (where the fixed board is, per waypoint)
    eye_to_hand -> T_flange_board (where the board sits on the flange)

    Spread in this list IS the calibration error, expressed physically.
    """
    estimates = []
    for robot, board in zip(robot_transforms, board_transforms):
        if mode == EYE_IN_HAND:
            estimates.append(np.asarray(robot) @ X @ np.asarray(board))
        else:
            estimates.append(invert_transform(np.asarray(robot)) @ X @ np.asarray(board))
    return estimates


def average_transform(transforms: Sequence[np.ndarray]) -> np.ndarray:
    """Robust consensus transform.

    Translation uses the MEDIAN, not the mean, so a single bad waypoint cannot
    drag the reference it is then measured against. Rotation is averaged by
    the standard quaternion/SVD projection back onto SO(3) -- averaging
    rotation matrices elementwise does not give a rotation matrix.
    """
    transforms = [np.asarray(t, dtype=np.float64) for t in transforms]
    translation = np.median(np.array([t[:3, 3] for t in transforms]), axis=0)
    rotation_sum = np.zeros((3, 3))
    for transform in transforms:
        rotation_sum += transform[:3, :3]
    U, _, Vt = np.linalg.svd(rotation_sum)
    rotation = U @ Vt
    if np.linalg.det(rotation) < 0:            # reflection, not a rotation
        U[:, -1] *= -1
        rotation = U @ Vt
    return make_transform(rotation, translation)


def residuals(robot_transforms, board_transforms, X, mode) -> dict:
    """Per-waypoint translation (mm) and rotation (deg) residuals."""
    estimates = constant_transforms(robot_transforms, board_transforms, X, mode)
    reference = average_transform(estimates)
    translation_residuals, rotation_residuals = [], []
    for estimate in estimates:
        translation, rotation = transform_difference(estimate, reference)
        translation_residuals.append(translation * 1000.0)
        rotation_residuals.append(rotation)
    return {
        "reference_transform": reference,
        "estimates": estimates,
        "translation_mm": translation_residuals,
        "rotation_deg": rotation_residuals,
        "translation_stats": error_statistics(translation_residuals),
        "rotation_stats": error_statistics(rotation_residuals),
    }


def reprojection_errors(records, X, mode, camera_matrix, dist_coeffs,
                        reference: np.ndarray | None = None) -> dict:
    """Reproject board points through the CALIBRATION CHAIN, not through PnP.

    The per-waypoint PnP residual only says the board was detected
    self-consistently; it is small even when the hand-eye transform is
    nonsense. This instead predicts where the board must be, given the robot
    pose and the candidate X, and measures how far that lands from the corners
    actually detected. It is the metric that can actually fail.
    """
    robot_transforms = [record.tcp_transform for record in records]
    board_transforms = [record.board_transform for record in records]
    if reference is None:
        reference = average_transform(
            constant_transforms(robot_transforms, board_transforms, X, mode))

    per_waypoint, all_errors = [], []
    for record, robot in zip(records, robot_transforms):
        if mode == EYE_IN_HAND:
            # board in camera = inv(X) @ inv(T_base_flange) @ T_base_board
            predicted = invert_transform(X) @ invert_transform(robot) @ reference
        else:
            # board in camera = inv(X) @ T_base_flange @ T_flange_board
            predicted = invert_transform(X) @ robot @ reference

        if record.object_points.size == 0:
            per_waypoint.append({"number": record.number, "mean_px": None,
                                 "rms_px": None, "max_px": None, "points": 0})
            continue

        rvec = matrix_to_rotvec(predicted[:3, :3]).reshape(3, 1)
        tvec = predicted[:3, 3].reshape(3, 1)
        projected, _ = cv2.projectPoints(
            record.object_points, rvec, tvec, camera_matrix, dist_coeffs)
        errors = np.linalg.norm(
            projected.reshape(-1, 2) - record.image_points, axis=1)
        all_errors.extend(errors.tolist())
        per_waypoint.append({
            "number": record.number,
            "points": int(errors.size),
            "mean_px": float(np.mean(errors)),
            "rms_px": float(np.sqrt(np.mean(errors ** 2))),
            "max_px": float(np.max(errors)),
        })
    return {"per_waypoint": per_waypoint, "overall": error_statistics(all_errors)}


def cross_check_methods(robot_transforms, board_transforms, mode) -> dict:
    """Solve with every method and report how far apart the answers are.

    Agreement does not prove correctness, but DISAGREEMENT is a reliable alarm:
    if five well-established algorithms given the same data land centimetres
    apart, the data is degenerate or the mode is wrong.
    """
    solutions, failures = {}, {}
    for name in sorted(METHODS):
        try:
            solutions[name] = solve_handeye(
                robot_transforms, board_transforms, mode, name)
        except CalibrationError as exc:
            failures[name] = str(exc)

    spread = {}
    if len(solutions) >= 2:
        names = sorted(solutions)
        translations, rotations = [], []
        for index, first in enumerate(names):
            for second in names[index + 1:]:
                translation, rotation = transform_difference(
                    solutions[first], solutions[second])
                translations.append(translation * 1000.0)
                rotations.append(rotation)
        spread = {
            "max_translation_difference_mm": max(translations),
            "mean_translation_difference_mm": float(np.mean(translations)),
            "max_rotation_difference_deg": max(rotations),
            "mean_rotation_difference_deg": float(np.mean(rotations)),
        }
    return {
        "solutions": {name: transform.tolist() for name, transform in solutions.items()},
        "translations_mm": {name: (transform[:3, 3] * 1000).tolist()
                            for name, transform in solutions.items()},
        "failures": failures,
        "spread": spread,
    }


def calibrate(records, mode: str, method: str, camera_matrix, dist_coeffs,
              cross_check: bool = True, label: str = "") -> dict:
    """Full hand-eye calibration over a set of waypoint records."""
    usable = [r for r in records if r.board_transform is not None]
    if len(usable) < len(records):
        LOGGER.warning("%d of %d records have no board pose and were skipped",
                       len(records) - len(usable), len(records))
    if len(usable) < 3:
        raise CalibrationError(
            f"Only {len(usable)} usable observation(s); hand-eye needs at least 3 "
            f"(and realistically 10+ with varied rotation axes)")

    robot_transforms = [r.tcp_transform for r in usable]
    board_transforms = [r.board_transform for r in usable]

    X = solve_handeye(robot_transforms, board_transforms, mode, method)
    residual = residuals(robot_transforms, board_transforms, X, mode)
    reprojection = reprojection_errors(
        usable, X, mode, camera_matrix, dist_coeffs,
        reference=residual["reference_transform"])

    rotation_vector = matrix_to_rotvec(X[:3, :3])
    result = {
        "label": label,
        "timestamp": timestamp_utc(),
        "mode": mode,
        "method": method,
        "observation_count": len(usable),
        "waypoint_numbers": [r.number for r in usable],

        "transform": X.tolist(),
        "rotation_matrix": X[:3, :3].tolist(),
        "rotation_vector": rotation_vector.tolist(),
        "rotation_angle_deg": rotation_angle_deg(X[:3, :3]),
        "translation_m": X[:3, 3].tolist(),
        "translation_mm": (X[:3, 3] * 1000).tolist(),
        "transform_meaning": (
            "T_flange_camera: camera pose expressed in the robot flange frame"
            if mode == EYE_IN_HAND else
            "T_base_camera: camera pose expressed in the robot base frame"),

        "reference_constant_transform": residual["reference_transform"].tolist(),
        "reference_meaning": (
            "T_base_board: the fixed board's pose in the robot base frame"
            if mode == EYE_IN_HAND else
            "T_flange_board: the board's pose on the robot flange"),

        "translation_residual_mm": residual["translation_stats"],
        "rotation_residual_deg": residual["rotation_stats"],
        "reprojection_px": reprojection["overall"],
        "per_waypoint": [
            {"number": record.number,
             "translation_residual_mm": residual["translation_mm"][index],
             "rotation_residual_deg": residual["rotation_deg"][index],
             **{key: value for key, value
                in reprojection["per_waypoint"][index].items() if key != "number"}}
            for index, record in enumerate(usable)],
    }
    if cross_check:
        result["cross_check"] = cross_check_methods(
            robot_transforms, board_transforms, mode)
    result["warnings"] = handeye_warnings(result)
    return result


def handeye_warnings(result: Mapping[str, Any]) -> list[str]:
    """Plausibility checks. A small residual alone does not mean correct."""
    warnings: list[str] = []
    mode = result["mode"]
    translation = np.asarray(result["translation_m"], dtype=np.float64)
    distance = float(np.linalg.norm(translation))

    if mode == EYE_IN_HAND and distance > 0.5:
        warnings.append(
            f"Camera sits {distance * 1000:.0f} mm from the flange. For a "
            f"wrist-mounted camera this is implausibly far -- check that "
            f"handeye_mode really is eye_in_hand for this camera")
    if mode == EYE_TO_HAND and distance > 5.0:
        warnings.append(
            f"Camera sits {distance:.2f} m from the robot base, which is "
            f"outside any plausible workcell")
    if distance < 1e-4:
        warnings.append(
            "Solved translation is essentially zero, which almost always means "
            "degenerate input poses")

    spread = (result.get("cross_check") or {}).get("spread") or {}
    if spread.get("max_translation_difference_mm", 0) > 20.0:
        warnings.append(
            f"The five solver methods disagree by up to "
            f"{spread['max_translation_difference_mm']:.1f} mm. That is a strong "
            f"sign of degenerate poses or a wrong eye-in-hand/eye-to-hand mode")
    if spread.get("max_rotation_difference_deg", 0) > 3.0:
        warnings.append(
            f"The five solver methods disagree by up to "
            f"{spread['max_rotation_difference_deg']:.2f} deg in rotation")

    translation_stats = result["translation_residual_mm"]
    if translation_stats.get("max") and translation_stats["max"] > 20.0:
        warnings.append(
            f"Worst waypoint translation residual is "
            f"{translation_stats['max']:.1f} mm; inspect that observation")
    reprojection = result["reprojection_px"]
    if reprojection.get("mean") and reprojection["mean"] > 3.0:
        warnings.append(
            f"Mean chain reprojection error is {reprojection['mean']:.2f} px. The "
            f"hand-eye transform does not explain the observed corners well")
    return warnings


def validate(records, X, mode, camera_matrix, dist_coeffs,
             reference: np.ndarray | None = None) -> dict:
    """Score a hand-eye transform on observations that did NOT fit it.

    The transform is used exactly as given -- nothing here re-fits, re-centres
    or tunes anything against these observations. That is the whole point: a
    hold-out number is only meaningful if the hold-out data never influenced
    the answer (section AB).
    """
    usable = [r for r in records if r.board_transform is not None]
    if not usable:
        return {"count": 0, "note": "no usable hold-out observations"}

    robot_transforms = [r.tcp_transform for r in usable]
    board_transforms = [r.board_transform for r in usable]
    estimates = constant_transforms(robot_transforms, board_transforms, X, mode)

    if reference is None:
        # No reference from the fit: fall back to this set's own consensus,
        # which measures self-consistency rather than agreement with the fit.
        reference = average_transform(estimates)
        note = ("reference derived from the hold-out set itself; measures "
                "internal consistency only")
    else:
        note = "reference carried over from the calibration set (true hold-out)"

    translation_errors, rotation_errors = [], []
    for estimate in estimates:
        translation, rotation = transform_difference(estimate, reference)
        translation_errors.append(translation * 1000.0)
        rotation_errors.append(rotation)

    reprojection = reprojection_errors(
        usable, X, mode, camera_matrix, dist_coeffs, reference=reference)
    return {
        "count": len(usable),
        "waypoint_numbers": [r.number for r in usable],
        "note": note,
        "translation_error_mm": error_statistics(translation_errors),
        "rotation_error_deg": error_statistics(rotation_errors),
        "reprojection_px": reprojection["overall"],
        "per_waypoint": [
            {"number": record.number,
             "translation_error_mm": translation_errors[index],
             "rotation_error_deg": rotation_errors[index],
             "reprojection_rms_px": reprojection["per_waypoint"][index]["rms_px"]}
            for index, record in enumerate(usable)],
    }


def compare(all_result: Mapping[str, Any], best_result: Mapping[str, Any],
            all_validation: Mapping[str, Any] | None = None,
            best_validation: Mapping[str, Any] | None = None) -> dict:
    """Compare ALL-30 against BEST-20 and recommend one -- honestly.

    Selection is NOT assumed to help. Fitting to a hand-picked subset can
    easily look better on the subset and worse everywhere else, so the
    recommendation is driven by hold-out performance when it is available.
    """
    def dig(result, *keys):
        value = result
        for key in keys:
            if not isinstance(value, Mapping):
                return None
            value = value.get(key)
        return value

    comparison = {
        "all": {
            "count": all_result.get("observation_count"),
            "reprojection_mean_px": dig(all_result, "reprojection_px", "mean"),
            "reprojection_median_px": dig(all_result, "reprojection_px", "median"),
            "reprojection_max_px": dig(all_result, "reprojection_px", "max"),
            "translation_residual_mean_mm": dig(all_result, "translation_residual_mm", "mean"),
            "rotation_residual_mean_deg": dig(all_result, "rotation_residual_deg", "mean"),
        },
        "best": {
            "count": best_result.get("observation_count"),
            "reprojection_mean_px": dig(best_result, "reprojection_px", "mean"),
            "reprojection_median_px": dig(best_result, "reprojection_px", "median"),
            "reprojection_max_px": dig(best_result, "reprojection_px", "max"),
            "translation_residual_mean_mm": dig(best_result, "translation_residual_mm", "mean"),
            "rotation_residual_mean_deg": dig(best_result, "rotation_residual_deg", "mean"),
        },
    }

    translation, rotation = transform_difference(
        np.asarray(all_result["transform"], dtype=np.float64),
        np.asarray(best_result["transform"], dtype=np.float64))
    comparison["difference_between_solutions"] = {
        "translation_mm": translation * 1000.0,
        "rotation_deg": rotation,
    }

    reasons: list[str] = []
    recommendation = "BEST-20"

    all_holdout = dig(all_validation or {}, "reprojection_px", "mean")
    best_holdout = dig(best_validation or {}, "reprojection_px", "mean")
    if all_holdout is not None and best_holdout is not None:
        comparison["holdout"] = {"all_mean_px": all_holdout,
                                 "best_mean_px": best_holdout}
        if all_holdout < best_holdout * 0.95:
            recommendation = "ALL-30"
            reasons.append(
                f"On the held-out observations ALL-30 is better "
                f"({all_holdout:.3f} px vs {best_holdout:.3f} px). Selection "
                f"removed useful geometry rather than noise.")
        elif best_holdout < all_holdout * 0.95:
            reasons.append(
                f"On the held-out observations BEST-20 is better "
                f"({best_holdout:.3f} px vs {all_holdout:.3f} px).")
        else:
            reasons.append(
                f"Hold-out performance is effectively tied ({all_holdout:.3f} px "
                f"vs {best_holdout:.3f} px).")
    else:
        reasons.append(
            "No independent hold-out comparison was available, so this "
            "recommendation rests on fit residuals, which favour the smaller "
            "hand-picked set by construction. Treat it as weak evidence.")

    if (comparison["all"]["translation_residual_mean_mm"] is not None
            and comparison["best"]["translation_residual_mean_mm"] is not None):
        if (comparison["best"]["translation_residual_mean_mm"]
                > comparison["all"]["translation_residual_mean_mm"]):
            reasons.append(
                "BEST-20's own fit residual is worse than ALL-30's, which is "
                "unusual and suggests the selection was not removing outliers.")

    if translation * 1000.0 > 10.0 or rotation > 1.0:
        reasons.append(
            f"The two solutions differ by {translation * 1000:.1f} mm and "
            f"{rotation:.2f} deg. With good data they should nearly agree; this "
            f"much disagreement means at least one is being driven by a few "
            f"observations.")

    comparison["recommendation"] = recommendation
    comparison["reasons"] = reasons
    return comparison
