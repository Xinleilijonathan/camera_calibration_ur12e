"""Shared loading for the non-interactive analysis scripts."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, ConfigError, camera_paths,
                               load_calibration_config, load_cameras_config,
                               load_yaml, require_board_verified,
                               require_handeye_verified, resolve_camera,
                               resolve_handeye_mode)
from intrinsic_calibration import load_intrinsics
from waypoint_recorder import load_waypoints


@dataclass
class AnalysisContext:
    camera_name: str
    camera_config: dict
    calibration_config: dict
    paths: Any
    handeye_mode: str
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    intrinsics: dict
    records: list

    @property
    def method(self) -> str:
        return str((self.calibration_config.get("handeye") or {}).get("method", "park"))

    @property
    def cross_check(self) -> bool:
        return bool((self.calibration_config.get("handeye") or {}).get(
            "cross_check_all_methods", True))


def load_for_analysis(camera_name: str, require_verified: bool = True,
                      logger=None) -> AnalysisContext:
    """Load waypoints, intrinsics and config, with all the guards applied."""
    cameras_config = load_cameras_config()
    calibration_config = load_calibration_config()
    camera_config = resolve_camera(camera_name, cameras_config)
    if require_verified:
        # Both scale the result in ways no downstream check can detect.
        require_board_verified(calibration_config)
        require_handeye_verified(calibration_config)
    handeye_mode = resolve_handeye_mode(camera_config, calibration_config)

    paths = camera_paths(camera_name)
    camera_matrix, dist_coeffs, intrinsics = load_intrinsics(paths.intrinsics_result)
    records = load_waypoints(paths, logger)
    if not records:
        raise CalibrationError(
            f"No waypoints in {paths.handeye_observations}.\n"
            f"  Run: python scripts/collect_waypoints.py --camera {camera_name}")

    if intrinsics.get("camera_serial") and camera_config.get("serial"):
        if str(intrinsics["camera_serial"]) != str(camera_config["serial"]):
            raise CalibrationError(
                f"Intrinsics in {paths.intrinsics_result} were solved for serial "
                f"{intrinsics['camera_serial']}, but cameras.yaml says "
                f"{camera_name} is serial {camera_config['serial']}.\n"
                f"  Using one camera's intrinsics for another is never valid.")

    mismatched = [r.number for r in records
                  if r.camera_serial and str(r.camera_serial)
                  != str(camera_config.get("serial", r.camera_serial))]
    if mismatched:
        raise CalibrationError(
            f"Waypoints {mismatched} were recorded with a different camera "
            f"serial than cameras.yaml currently assigns to {camera_name}. "
            f"Refusing to mix cameras.")

    return AnalysisContext(
        camera_name=camera_name, camera_config=camera_config,
        calibration_config=calibration_config, paths=paths,
        handeye_mode=handeye_mode, camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs, intrinsics=intrinsics, records=records)


def print_context(context: AnalysisContext, title: str) -> None:
    print("=" * 78)
    print(f"{title} -- {context.camera_name}")
    print("=" * 78)
    print(f"Serial      : {context.camera_config.get('serial')}")
    print(f"Mounting    : {context.handeye_mode}")
    print(f"Waypoints   : {len(context.records)}")
    print(f"Intrinsics  : {context.paths.intrinsics_result.name} "
          f"(RMS {context.intrinsics.get('rms_reprojection_error_px', float('nan')):.4f} px)")
    print(f"Method      : {context.method}")
    print()


def fail(exc) -> int:
    print(f"ERROR: {exc}", file=sys.stderr)
    return 1
