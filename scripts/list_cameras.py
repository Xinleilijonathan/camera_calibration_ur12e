#!/usr/bin/env python3
"""Enumerate every connected camera and report how to address it.

    python scripts/list_cameras.py
    python scripts/list_cameras.py --verbose
    python scripts/list_cameras.py --emit-yaml

This script NEVER touches the robot and never opens a stream; it only reads
device metadata.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (sys.path side effect)

from camera_interface import (enumerate_cameras, enumerate_realsense_cameras,
                              format_camera_table)
from calibration_utils import VALID_CAMERA_NAMES


def emit_yaml(cameras) -> str:
    """Generate a cameras.yaml 'cameras:' block from what is plugged in now."""
    usable = [c for c in cameras if c.usable]
    lines = ["cameras:", ""]
    for name, camera in zip(VALID_CAMERA_NAMES, usable):
        best = camera.best_mode() or {"width": 1280, "height": 720,
                                      "fps": 30, "format": "MJPG"}
        identifier = camera.serial or camera.usb_path
        lines += [
            f"  {name}:",
            f"    name: {name}",
            f"    backend: {camera.backend}",
            f"    serial: \"{identifier}\"",
            f"    width: {best['width']}",
            f"    height: {best['height']}",
            f"    fps: {int(best['fps'])}",
        ]
        if camera.backend == "v4l2":
            fourcc = best.get("format", "MJPG")
            lines.append(f"    fourcc: {fourcc if len(fourcc) == 4 else 'MJPG'}")
        lines += [f"    description: \"{camera.name}\"", ""]

    for name in VALID_CAMERA_NAMES[len(usable):]:
        lines += [
            f"  # {name}: NOT DETECTED. Plug it in and re-run this script.",
            "",
        ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Enumerate cameras by serial number (read-only, no robot).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="list every supported mode for every camera")
    parser.add_argument("--emit-yaml", action="store_true",
                        help="print a cameras.yaml block for the detected cameras")
    parser.add_argument("--backend", choices=["all", "realsense", "v4l2"],
                        default="all", help="restrict enumeration to one backend")
    args = parser.parse_args(argv)

    if args.backend == "realsense":
        cameras = enumerate_realsense_cameras()
    elif args.backend == "v4l2":
        from camera_interface import enumerate_v4l2_cameras
        cameras = enumerate_v4l2_cameras()
    else:
        cameras = enumerate_cameras()

    if args.emit_yaml:
        print(emit_yaml(cameras))
        return 0

    print("=" * 74)
    print("CONNECTED CAMERAS")
    print("=" * 74)
    print()
    print(format_camera_table(cameras))

    if args.verbose:
        for index, camera in enumerate(cameras, start=1):
            if not camera.modes:
                continue
            print(f"[{index}] {camera.name} -- {len(camera.modes)} modes:")
            for mode in camera.modes:
                print(f"      {mode['format']:>10}  {mode['width']:>5} x "
                      f"{mode['height']:<5} @ {mode['fps']:g} fps")
            print()

    usable = [c for c in cameras if c.usable]
    print("-" * 74)
    print(f"Usable cameras: {len(usable)}   (this project expects 3)")

    try:
        import pyrealsense2  # noqa: F401
    except ImportError:
        print()
        print("NOTE: pyrealsense2 is not installed, so no Intel RealSense device")
        print("      can be enumerated or opened. If your cameras are RealSense,")
        print("      install the bindings before continuing. UVC webcams work now.")

    missing = [c for c in usable if not c.serial]
    if missing:
        print()
        print("WARNING: these cameras report no serial number:")
        for camera in missing:
            print(f"  - {camera.name} ({camera.device})")
        print("  Use their 'usb path' as the serial in cameras.yaml, and do not")
        print("  move them to a different USB port afterwards.")

    if len(usable) >= 2:
        serials = [c.serial for c in usable if c.serial]
        if len(serials) != len(set(serials)):
            print()
            print("WARNING: duplicate serial numbers detected. Cameras cannot be")
            print("         told apart reliably. Use usb path instead.")

    print()
    print("Next: python scripts/list_cameras.py --emit-yaml")
    print("      then paste the result into config/cameras.yaml")
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())
