#!/usr/bin/env python3
"""Write result.yaml for ONE camera from its FACTORY intrinsics.

    python scripts/factory_intrinsics.py --camera camera_1

Reads the calibration the RealSense module carries in its own flash and
writes it to data/camera_N/intrinsics/result.yaml, in the same shape
solve_intrinsics.py produces, so every downstream script consumes it
unchanged.

THIS IS NOT A CALIBRATION. Nothing here is measured against your board, so
there is no reprojection error to report and this script cannot tell you
whether the numbers are right for your camera. It stamps `source: factory`
into the file so a file produced this way can never be mistaken for a
solved one.

Two things to know before trusting the result:

  * Several RealSense colour streams report ALL-ZERO distortion. That is
    either because the stream is rectified in the ASIC (the zeros are then
    correct) or because the field was never populated (the zeros are then a
    silent lie). This script warns; scripts/check_distortion.py decides.

  * librealsense labels the model `inverse_brown_conrady`, which is not
    OpenCV's convention. For the D405 the coefficients were checked against
    an independent OpenCV solve on 24 board views and matched to 0.74 px
    mean / 5.29 px max; flipping their sign made the fit markedly worse
    (0.898 -> 1.305 px RMS). So they are used here as OpenCV dist_coeffs
    directly. Re-check that on any module whose behaviour you do not know.
"""
from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import (ConfigError, camera_paths, describe_environment,
                               load_cameras_config, resolve_camera, save_yaml,
                               setup_logging, timestamp_slug, timestamp_utc)

# Models whose five coefficients are laid out as OpenCV's k1 k2 p1 p2 k3.
# Anything else (ftheta, kannala_brandt4) is a different parameterisation and
# must not be silently written into a file downstream code reads as OpenCV.
OPENCV_COMPATIBLE_MODELS = {
    "distortion.inverse_brown_conrady",
    "distortion.brown_conrady",
    "distortion.modified_brown_conrady",
    "distortion.none",
}


def query_factory_intrinsics(serial: str, width: int, height: int) -> dict:
    """Read the colour-stream intrinsics for one device, without streaming.

    Profiles carry their intrinsics, so nothing is started and no USB
    bandwidth is used -- which matters on this rig, where camera_2 and
    camera_3 share a USB 2.1 bus and cannot both stream.
    """
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise ConfigError(
            f"pyrealsense2 is not installed ({exc}).\n"
            f"  Factory intrinsics can only be read from the device itself.") from exc

    devices = {d.get_info(rs.camera_info.serial_number): d
               for d in rs.context().query_devices()}
    device = devices.get(serial)
    if device is None:
        raise ConfigError(
            f"No RealSense device with serial {serial} is connected.\n"
            f"  Connected: {sorted(devices) or 'none'}\n"
            f"  Check the cable, then: python scripts/list_cameras.py")

    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            video = profile.as_video_stream_profile()
            if not video or profile.stream_type() != rs.stream.color:
                continue
            if video.width() != width or video.height() != height:
                continue
            intrinsics = video.get_intrinsics()
            return {
                "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
                "cx": float(intrinsics.ppx), "cy": float(intrinsics.ppy),
                "coeffs": [float(c) for c in intrinsics.coeffs],
                "rs_distortion_model": str(intrinsics.model),
                "sensor_name": sensor.get_info(rs.camera_info.name),
                "device_name": device.get_info(rs.camera_info.name),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "librealsense_version": rs.__version__,
            }

    raise ConfigError(
        f"Device {serial} offers no colour stream at {width}x{height}.\n"
        f"  Intrinsics are in pixels and do not transfer between resolutions, "
        f"so this must match cameras.yaml exactly.\n"
        f"  Run: python scripts/list_cameras.py")


def build_record(factory: dict, camera_name: str, serial: str,
                 width: int, height: int) -> dict:
    """Assemble the result.yaml payload and the warnings that belong with it."""
    fx, fy = factory["fx"], factory["fy"]
    cx, cy = factory["cx"], factory["cy"]
    coefficients = factory["coeffs"]

    warnings: list[str] = []
    if all(abs(c) < 1e-12 for c in coefficients):
        warnings.append(
            "Factory distortion coefficients are ALL ZERO. That is only "
            "correct if this colour stream is rectified in hardware. Until "
            "scripts/check_distortion.py confirms it against real board "
            "views, treat this as an untested assumption, not a measurement.")
    if factory["rs_distortion_model"] not in OPENCV_COMPATIBLE_MODELS:
        warnings.append(
            f"librealsense reports distortion model "
            f"{factory['rs_distortion_model']}, which is not laid out as "
            f"OpenCV's k1 k2 p1 p2 k3. These coefficients must NOT be used "
            f"as OpenCV dist_coeffs without conversion.")
    aspect = fy / fx if fx else float("nan")
    if not 0.95 <= aspect <= 1.05:
        warnings.append(
            f"fy/fx = {aspect:.4f}; square pixels should give ~1.0.")

    return {
        "timestamp": timestamp_utc(),
        # Provenance first: this is the field that stops someone reading the
        # numbers below as though a board had been waved at the camera.
        "source": "factory",
        "source_note": (
            "Read from the camera's own flash by scripts/factory_intrinsics.py. "
            "NOT solved from observations. No reprojection error exists for "
            "these values because nothing was measured against a board."),
        "image_width": int(width),
        "image_height": int(height),
        "observation_count": 0,
        "distortion_model": "standard",
        "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "fx": fx, "fy": fy, "cx": cx, "cy": cy,
        "distortion_coefficients": coefficients,
        # Sanity indicators, not calibration outputs. Deliberately no
        # rms_reprojection_error_px key: downstream formats that value as a
        # float and would raise on an explicit null.
        "field_of_view_deg": {
            "horizontal": 2 * math.degrees(math.atan(width / (2 * fx))),
            "vertical": 2 * math.degrees(math.atan(height / (2 * fy))),
        },
        "principal_point_offset_px": {"x": cx - width / 2.0,
                                      "y": cy - height / 2.0},
        "aspect_ratio": aspect,
        "warnings": warnings,
        "camera_name": camera_name,
        "camera_serial": str(serial),
        "factory": {
            "device_name": factory["device_name"],
            "sensor_name": factory["sensor_name"],
            "firmware_version": factory["firmware_version"],
            "librealsense_version": factory["librealsense_version"],
            "rs_distortion_model": factory["rs_distortion_model"],
            "stream": f"color {width}x{height}",
        },
        "environment": describe_environment(),
    }


def archive_existing(path: Path, force: bool, logger) -> bool:
    """Never overwrite a solved result.yaml without saying so. Returns False
    if the user declined."""
    if not path.is_file():
        return True
    print()
    print(f"WARNING: {path} already exists.")
    try:
        from calibration_utils import load_yaml
        existing = load_yaml(path)
        kind = existing.get("source", "solved")
        rms = existing.get("rms_reprojection_error_px")
        print(f"  It is a {kind} calibration"
              + (f" with RMS {rms:.4f} px" if isinstance(rms, (int, float)) else "")
              + f", {existing.get('observation_count', '?')} observation(s).")
    except Exception:
        pass
    if not force:
        print("  [a] archive it alongside the new file and continue")
        print("  [q] quit, change nothing")
        if input("Choose [a/q]: ").strip().lower() != "a":
            print("Cancelled; nothing was changed.")
            return False
    else:
        print("--force given: archiving rather than deleting.")

    archive = path.parent / "sessions" / timestamp_slug()
    archive.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(archive / path.name))
    print(f"Archived previous result to {archive / path.name}")
    logger.info("Archived previous %s to %s", path, archive)
    return True


def print_record(record: dict) -> None:
    print("-" * 74)
    print(f"fx {record['fx']:10.3f}    fy {record['fy']:10.3f}")
    print(f"cx {record['cx']:10.3f}    cy {record['cy']:10.3f}")
    print()
    print("Distortion:", "  ".join(f"{c: .6f}"
                                   for c in record["distortion_coefficients"]))
    print()
    fov = record["field_of_view_deg"]
    print(f"Field of view      : {fov['horizontal']:.1f} deg H, "
          f"{fov['vertical']:.1f} deg V")
    offset = record["principal_point_offset_px"]
    print(f"Principal offset   : {offset['x']:+.1f}, {offset['y']:+.1f} px "
          f"from image centre")
    print(f"Aspect ratio fy/fx : {record['aspect_ratio']:.5f}")
    print()
    factory = record["factory"]
    print(f"Device             : {factory['device_name']} "
          f"fw {factory['firmware_version']}")
    print(f"Sensor / stream    : {factory['sensor_name']} / {factory['stream']}")
    print(f"librealsense model : {factory['rs_distortion_model']} "
          f"(librealsense {factory['librealsense_version']})")
    print("-" * 74)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Write result.yaml from a camera's factory intrinsics.")
    parser.add_argument("--camera", required=True,
                        help="logical camera name, e.g. camera_1")
    parser.add_argument("--force", action="store_true",
                        help="archive any existing result.yaml without asking")
    parser.add_argument("--dry-run", action="store_true",
                        help="report the values without writing result.yaml")
    args = parser.parse_args(argv)

    logger = setup_logging("factory_intrinsics", args.camera)

    try:
        cameras_config = load_cameras_config()
        camera_config = resolve_camera(args.camera, cameras_config)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    backend = str(camera_config.get("backend", "")).lower()
    if backend != "realsense":
        print(f"ERROR: {args.camera} uses backend {backend!r}. Factory "
              f"intrinsics can only be read from a RealSense device.",
              file=sys.stderr)
        return 2

    serial = str(camera_config.get("serial", ""))
    width = int(camera_config["width"])
    height = int(camera_config["height"])

    print("=" * 74)
    print(f"FACTORY INTRINSICS -- {args.camera}")
    print("=" * 74)
    print(f"Camera     : {camera_config.get('model', '?')} serial {serial}")
    print(f"Resolution : {width} x {height}")
    print()
    print("These values are read from the camera, not measured against your")
    print("board. They are a starting point, not a calibration.")
    print()

    try:
        factory = query_factory_intrinsics(serial, width, height)
    except ConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    record = build_record(factory, args.camera, serial, width, height)
    print_record(record)

    if record["warnings"]:
        print()
        print("WARNINGS:")
        for warning in record["warnings"]:
            print(f"  * {warning}")

    if args.dry_run:
        print("\n--dry-run: result.yaml was NOT written.")
        return 0

    paths = camera_paths(args.camera)
    paths.ensure()
    if not archive_existing(paths.intrinsics_result, args.force, logger):
        return 1

    save_yaml(paths.intrinsics_result, record, header=(
        f"FACTORY intrinsics for {args.camera} (serial {serial}).\n"
        f"Read from the camera's own flash -- NOT solved from board\n"
        f"observations. See `source` below before trusting these numbers.\n"
        f"These parameters belong to THIS camera only and must never be\n"
        f"reused for another camera, even one of the same model.\n"
        f"Valid only at {width}x{height}."))
    print(f"\nSaved: {paths.intrinsics_result}")
    logger.info("Wrote factory intrinsics to %s (fx=%.3f fy=%.3f)",
                paths.intrinsics_result, record["fx"], record["fy"])

    if all(abs(c) < 1e-12 for c in record["distortion_coefficients"]):
        print()
        print("NEXT, BEFORE YOU RELY ON THIS:")
        print(f"  python scripts/collect_intrinsics.py --camera {args.camera} "
              f"--target-count 15")
        print(f"  python scripts/check_distortion.py --camera {args.camera}")
        print("  The zero-distortion claim above is untested until you do.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
