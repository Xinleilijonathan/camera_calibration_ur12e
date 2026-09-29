#!/usr/bin/env python3
"""Combine a camera matrix from one solve with distortion from another.

    python scripts/blend_intrinsics.py --camera camera_2 \\
        --distortion-from data/camera_2/intrinsics/sessions/<stamp>/result.yaml

There is one situation this exists for. Distortion is only observable away
from the image centre, so a narrow mode can fail to determine it while
determining fx, fy, cx and cy perfectly well. On this rig the 640x480 colour
mode crops 25% horizontally off the 1280x720 one, which halves the share of
corners landing beyond 0.7 of the corner radius, and `check_distortion.py`
returns INCONCLUSIVE for both D435IFs there. The 1280x720 solves reach
further out, and two independent units agreed to about 4%, so their
coefficients are the better-determined description of the same glass.

Distortion coefficients are dimensionless. They act on normalised
coordinates, so a crop or a rescale of the sensor readout does not change
them -- the same lens is the same lens. A camera matrix is in pixels and is
not transferable that way, which is why only the coefficients move here.

The output records where each half came from. Nothing is refitted.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (CalibrationError, camera_paths, load_yaml,
                               save_yaml, setup_logging, timestamp_utc)

MATRIX_KEYS = ("fx", "fy", "cx", "cy")
CARRIED = ("image_width", "image_height", "camera_matrix", "camera_name",
           "camera_serial", "board", "observation_count", "distortion_model",
           "rms_reprojection_error_px", "mean_reprojection_error_px",
           "median_reprojection_error_px", "max_reprojection_error_px",
           "total_points", "field_of_view_deg", "principal_point_offset_px",
           "aspect_ratio")


def summarise(result, path):
    """One line of provenance for a source file."""
    return {
        "file": str(path),
        "resolution": [result.get("image_width"), result.get("image_height")],
        "source": result.get("source", "solved"),
        "observation_count": result.get("observation_count"),
        "rms_reprojection_error_px": result.get("rms_reprojection_error_px"),
        "timestamp": result.get("timestamp"),
    }


def main(argv=None) -> int:
    """Refuse to blend across cameras; everything else is the user's call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--distortion-from", required=True,
                        help="result.yaml whose distortion coefficients to "
                             "adopt, typically a wider-field solve")
    parser.add_argument("--matrix-from", default=None,
                        help="result.yaml whose fx/fy/cx/cy to keep "
                             "(default: the camera's current result)")
    parser.add_argument("--output", default="result_hybrid.yaml",
                        help="file name written beside the current result")
    parser.add_argument("--reason", default=None,
                        help="why this blend is being made; recorded in the "
                             "output so a reader is never left guessing")
    args = parser.parse_args(argv)
    logger = setup_logging("blend_intrinsics")

    paths = camera_paths(args.camera)
    matrix_path = args.matrix_from or paths.intrinsics_result
    matrix = load_yaml(matrix_path)
    distortion = load_yaml(args.distortion_from)

    if str(matrix.get("camera_serial")) != str(distortion.get("camera_serial")):
        raise CalibrationError(
            f"Serial mismatch: the matrix comes from "
            f"{matrix.get('camera_serial')} and the distortion from "
            f"{distortion.get('camera_serial')}. Distortion belongs to one "
            f"piece of glass and never transfers between units.")

    blended = {key: matrix[key] for key in CARRIED if key in matrix}
    blended.update({key: matrix[key] for key in MATRIX_KEYS})
    blended["distortion_coefficients"] = list(
        distortion["distortion_coefficients"])
    blended["timestamp"] = timestamp_utc()
    blended["source"] = "blended"
    blended["blend"] = {
        "camera_matrix_from": summarise(matrix, matrix_path),
        "distortion_from": summarise(distortion, args.distortion_from),
        "reason": args.reason or (
            "distortion is better determined in the wider-field solve; "
            "the coefficients are dimensionless and describe the same lens"),
        "note": "Nothing was refitted. fx, fy, cx and cy are in pixels and "
                "belong to the resolution they were measured at; the "
                "distortion coefficients act on normalised coordinates and "
                "do not.",
    }

    same = np.allclose(matrix["distortion_coefficients"],
                       distortion["distortion_coefficients"], atol=1e-9)
    if same:
        logger.warning("The two sources already share their distortion; "
                       "this blend changes nothing")

    output = paths.intrinsics_result.parent / args.output
    save_yaml(output, blended, header="\n".join([
        f"BLENDED intrinsics for {args.camera} "
        f"(serial {blended.get('camera_serial')}).",
        "Camera matrix and distortion come from different solves; see the",
        "`blend` block below for which, and why. Nothing here was refitted.",
    ]))
    print(f"Wrote {output}")
    print(f"  matrix     {matrix_path}")
    print(f"             {blended['image_width']}x{blended['image_height']}  "
          + "  ".join(f"{k}={blended[k]:.3f}" for k in MATRIX_KEYS))
    print(f"  distortion {args.distortion_from}")
    print("             " + "  ".join(
        f"{v:+.6f}" for v in blended["distortion_coefficients"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except CalibrationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
