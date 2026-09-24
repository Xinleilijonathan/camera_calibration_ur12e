#!/usr/bin/env python3
"""Generate a print-exact, vector PDF of the configured AprilTag board.

    python scripts/make_board_pdf.py --output board.pdf

Vector rather than raster: every cell is a filled rectangle, so there is no
resampling and no DPI to get wrong.

Tag placement comes from the project's own cv2.aruco.GridBoard via
getObjPoints(), never from a reimplementation of the layout. That is the whole
point of this script. A board bought or generated elsewhere may number its
tags by a different convention -- right-to-left within a row, or column-major
-- and the detector will still decode every tag and still report a healthy
tag count, while the 3D points it pairs them with are wrong. The intrinsic
solve then cannot fit the data at all, and the result is not merely imprecise
but meaningless. Printing the board this script emits removes that whole class
of failure, because the board and the detector come from the same object.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from calibration_utils import ConfigError, load_calibration_config
from apriltag_detector import TAG_FAMILIES, build_detector

MM = 72.0 / 25.4                      # PostScript points per millimetre
A4_W, A4_H = 210.0 * MM, 297.0 * MM


def build_pdf(det) -> tuple[bytes, dict]:
    spec = det.spec
    board = det.board
    obj = np.asarray(board.getObjPoints(), dtype=np.float64).reshape(-1, 4, 3)
    ids = np.asarray(board.getIds()).ravel()

    board_w_mm = spec.width_m * 1000.0
    board_h_mm = spec.height_m * 1000.0
    tag_mm = spec.tag_size_m * 1000.0

    # tag36h11 is a 6x6 data grid inside a 1-cell black border => 8x8 cells.
    # tag_size_m is the OUTER edge of that black border, which is where the
    # detector puts the corners, so the cell pitch is tag_size / 8.
    dictionary = cv2.aruco.getPredefinedDictionary(TAG_FAMILIES[spec.family])
    cells = dictionary.markerSize + 2
    cell_mm = tag_mm / cells

    x0 = (A4_W - board_w_mm * MM) / 2.0
    y_top = A4_H - 26.0 * MM
    y0 = y_top - board_h_mm * MM
    if y0 < 55.0 * MM or x0 < 5.0 * MM:
        raise ConfigError(
            f"A board of {board_w_mm:.1f} x {board_h_mm:.1f} mm does not fit on "
            f"A4 with a usable quiet zone. Reduce tag_size_m or the tag count.")

    rects = []
    for k, tag_id in enumerate(ids):
        img = cv2.aruco.generateImageMarker(dictionary, int(tag_id), cells)
        tl = obj[k].min(axis=0)                 # tag top-left in the board frame
        for r in range(cells):
            c = 0
            while c < cells:
                if img[r, c] != 0:              # white: nothing to draw
                    c += 1
                    continue
                run = c                         # merge horizontal runs of black
                while run < cells and img[r, run] == 0:
                    run += 1
                # Board +Y is DOWN; PDF +y is UP.
                bx = tl[0] * 1000.0 + c * cell_mm
                by = tl[1] * 1000.0 + r * cell_mm
                rects.append((x0 + bx * MM,
                              y0 + (board_h_mm - by - cell_mm) * MM,
                              (run - c) * cell_mm * MM, cell_mm * MM))
                c = run

    parts = ["0 0 0 rg"]
    for x, y, w, h in rects:
        parts.append(f"{x:.4f} {y:.4f} {w:.4f} {h:.4f} re f")

    # A 100 mm scale bar, so the print can be checked before anyone relies on it.
    bar_len, bar_y = 100.0 * MM, 34.0 * MM
    bx0 = (A4_W - bar_len) / 2.0
    parts.append("0.7 w")
    parts.append(f"{bx0:.3f} {bar_y:.3f} m {bx0 + bar_len:.3f} {bar_y:.3f} l S")
    for tick in (0.0, 50.0, 100.0):
        tx = bx0 + tick * MM
        parts.append(f"{tx:.3f} {bar_y - 4:.3f} m {tx:.3f} {bar_y + 4:.3f} l S")

    def text(x, y, size, s):
        s = s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
        parts.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({s}) Tj ET")

    text(x0, y_top + 8, 9,
         f"{spec.family}  {spec.rows} rows x {spec.columns} cols  "
         f"({spec.tag_count} tags)   tag {tag_mm:.1f} mm   gap "
         f"{spec.tag_spacing_m * 1000:.1f} mm   board "
         f"{board_w_mm:.1f} x {board_h_mm:.1f} mm")
    text(x0, y0 - 16, 8,
         "Tag 0 is TOP-LEFT. IDs increase LEFT TO RIGHT along each row, then down.")
    text(x0, y0 - 27, 8,
         "Generated from this project's cv2.aruco.GridBoard, so the detector "
         "agrees by construction.")
    text(bx0, bar_y - 16, 8,
         "PRINT AT 100% / ACTUAL SIZE (no 'fit to page'). This bar must measure "
         "exactly 100 mm.")
    text(bx0, bar_y - 26, 8,
         "Then measure a tag's black square with a caliper and put the real "
         "value in calibration.yaml.")

    stream = "\n".join(parts).encode("latin-1")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {A4_W:.4f} {A4_H:.4f}] "
         f"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n"
            f"{xref}\n%%EOF\n").encode()

    info = {"board_w_mm": board_w_mm, "board_h_mm": board_h_mm,
            "tag_mm": tag_mm, "cell_mm": cell_mm, "cells": cells,
            "tag_count": int(spec.tag_count), "rects": len(rects)}
    return bytes(out), info


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a print-exact vector PDF of the configured board.")
    parser.add_argument("--output", "-o", default="apriltag_board.pdf",
                        help="where to write the PDF")
    args = parser.parse_args(argv)

    try:
        det = build_detector(load_calibration_config())
        pdf, info = build_pdf(det)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    dest = Path(args.output)
    dest.write_bytes(pdf)
    print(f"Wrote {dest}  ({len(pdf) / 1024:.1f} kB, {info['rects']} rectangles)")
    print(f"Board  : {info['board_w_mm']:.1f} x {info['board_h_mm']:.1f} mm, "
          f"{info['tag_count']} tags")
    print(f"Tag    : {info['tag_mm']:.1f} mm black square, "
          f"{info['cells']}x{info['cells']} cells of {info['cell_mm']:.3f} mm")
    print()
    print("Print at 100% / actual size. Verify the 100 mm bar with a caliper,")
    print("then measure a real tag and put THAT value in calibration.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
