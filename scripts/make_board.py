#!/usr/bin/env python3
"""Render the configured AprilTag board as a print-ready, exactly-scaled PDF.

    python scripts/make_board.py
    python scripts/make_board.py --paper letter --output ~/board.pdf

The point of this script is that the printed board and the software agree by
construction. cv2.aruco.GridBoard assigns tag IDs left-to-right along each
row; boards from other generators (Kalibr, the upstream AprilTag tools,
various web generators) do not all use that order. A board whose IDs run the
other way still detects perfectly -- every tag decodes, the grid looks
regular, nothing warns -- but each tag is then matched to the wrong 3D point,
and the intrinsic solve returns confident nonsense. Printing the board this
script emits removes that entire class of error.

The PDF carries no scaling of its own: the page is exactly the paper size and
the board is placed at exactly the millimetre dimensions in calibration.yaml.
Print it at 100% / "Actual size" -- NOT "fit to page", which silently rescales
and is the usual reason a measured board disagrees with its own spec sheet.

A 100.00 mm vector bar is printed beside the board. Measure it after printing.
If it is not 100.00 mm the printer rescaled, and every metric result
downstream is wrong by that same factor without any reprojection error to
show for it.
"""
from __future__ import annotations

import argparse
import sys
import zlib
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from apriltag_detector import build_detector
from calibration_utils import ConfigError, load_calibration_config

PT_PER_MM = 72.0 / 25.4
PAPERS_MM = {"a4": (210.0, 297.0), "letter": (215.9, 279.4), "a3": (297.0, 420.0)}
RENDER_DPI = 600.0
CHECK_BAR_MM = 100.0


def _pdf(objects: list[bytes]) -> bytes:
    """Assemble numbered PDF objects into a file with a correct xref table."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return bytes(out)


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def build_pdf(image: np.ndarray, board_w_mm: float, board_h_mm: float,
              paper_w_mm: float, paper_h_mm: float, caption: list[str]) -> bytes:
    height_px, width_px = image.shape[:2]
    page_w, page_h = paper_w_mm * PT_PER_MM, paper_h_mm * PT_PER_MM
    draw_w, draw_h = board_w_mm * PT_PER_MM, board_h_mm * PT_PER_MM

    # Centre horizontally; sit the board high on the page so the caption and
    # the measurement bar have room underneath.
    x = (page_w - draw_w) / 2.0
    y = page_h - draw_h - 18.0 * PT_PER_MM

    if x < 0 or y < 0:
        raise ConfigError(
            f"The board is {board_w_mm:.1f} x {board_h_mm:.1f} mm and does not "
            f"fit on {paper_w_mm:.0f} x {paper_h_mm:.0f} mm with margins. "
            f"Use --paper a3, or reduce rows/columns/tag_size_m.")

    content = [f"q {draw_w:.4f} 0 0 {draw_h:.4f} {x:.4f} {y:.4f} cm /Im0 Do Q"]

    # Caption, then the measurement bar below it.
    text_y = y - 9.0 * PT_PER_MM
    content.append("BT /F1 9 Tf")
    content.append(f"1 0 0 1 {x:.4f} {text_y:.4f} Tm")
    for line in caption:
        content.append(f"({_escape(line)}) Tj 0 -11 Td")
    content.append("ET")

    bar_y = text_y - (11 * len(caption)) - 6.0 * PT_PER_MM
    bar_len = CHECK_BAR_MM * PT_PER_MM
    # Filled rectangles, not stroked lines. A stroke of width w centred on the
    # path extends w/2 past each endpoint, so the ink a caliper actually grips
    # would be CHECK_BAR_MM + w -- 0.2 mm long here. Drawn as a fill, the ink
    # is exactly CHECK_BAR_MM wide, which is the whole point of the bar.
    rule_h = 0.5 * PT_PER_MM
    content.append("0 g")
    content.append(f"{x:.4f} {bar_y:.4f} {bar_len:.4f} {rule_h:.4f} re f")
    tick_w = 0.4 * PT_PER_MM
    for k in range(11):                       # a tick every 10 mm
        # Inset the two end ticks so they cannot widen the measurable ink.
        tx = x + bar_len * k / 10.0
        if k == 0:
            tx_left = tx
        elif k == 10:
            tx_left = tx - tick_w
        else:
            tx_left = tx - tick_w / 2.0
        tall = 5.0 if k % 5 == 0 else 3.0
        content.append(
            f"{tx_left:.4f} {bar_y + rule_h:.4f} {tick_w:.4f} {tall:.4f} re f")
    content.append("BT /F1 8 Tf "
                   f"1 0 0 1 {x:.4f} {bar_y - 11:.4f} Tm "
                   f"(MEASURE ACROSS THE FULL BLACK BAR: exactly {CHECK_BAR_MM:.2f} mm. "
                   f"If not, the printer rescaled - reprint at 100%%.) Tj ET")

    stream = "\n".join(content).encode("latin-1")
    packed = zlib.compress(stream)
    pixels = zlib.compress(np.ascontiguousarray(image, dtype=np.uint8).tobytes())

    return _pdf([
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
         f"/Resources << /XObject << /Im0 5 0 R >> /Font << /F1 6 0 R >> >> "
         f"/Contents 4 0 R >>").encode(),
        (f"<< /Length {len(packed)} /Filter /FlateDecode >>\nstream\n").encode()
        + packed + b"\nendstream",
        (f"<< /Type /XObject /Subtype /Image /Width {width_px} /Height {height_px} "
         f"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /FlateDecode "
         f"/Length {len(pixels)} >>\nstream\n").encode() + pixels + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Render the configured AprilTag board as a print-ready PDF.")
    parser.add_argument("--paper", choices=sorted(PAPERS_MM), default="letter",
                        help="default is US Letter, the paper this lab prints on")
    parser.add_argument("--output", default="apriltag_board.pdf")
    parser.add_argument("--png", action="store_true",
                        help="also write a PNG alongside (for screen checks only; "
                             "printing a PNG rarely preserves scale)")
    args = parser.parse_args(argv)

    try:
        config = load_calibration_config()
        detector = build_detector(config)
    except ConfigError as exc:
        print(f"CONFIGURATION ERROR\n{exc}", file=sys.stderr)
        return 2

    spec = detector.spec
    # margin_px=0 so the rendered image is EXACTLY the board's metric size;
    # the quiet zone comes from the white page around it.
    image = detector.render_board(RENDER_DPI / 0.0254, margin_px=0)

    caption = [
        f"{spec.family}  {spec.rows} rows x {spec.columns} columns "
        f"({spec.tag_count} tags), IDs {spec.first_tag_id}"
        f"-{spec.first_tag_id + spec.tag_count - 1}",
        f"tag {spec.tag_size_m * 1000:.2f} mm   gap {spec.tag_spacing_m * 1000:.2f} mm"
        f"   board {spec.width_m * 1000:.2f} x {spec.height_m * 1000:.2f} mm",
        "Generated by cv2.aruco.GridBoard - IDs increase LEFT TO RIGHT along each row.",
        "Print at 100% / Actual size. Do NOT use 'fit to page'.",
    ]
    paper_w, paper_h = PAPERS_MM[args.paper]

    try:
        pdf = build_pdf(image, spec.width_m * 1000, spec.height_m * 1000,
                        paper_w, paper_h, caption)
    except ConfigError as exc:
        print(f"ERROR\n{exc}", file=sys.stderr)
        return 1

    output = Path(args.output).expanduser()
    output.write_bytes(pdf)

    print("=" * 74)
    print("PRINTABLE APRILTAG BOARD")
    print("=" * 74)
    print(f"Board   : {spec.describe()}")
    print(f"Paper   : {args.paper.upper()} ({paper_w:.0f} x {paper_h:.0f} mm)")
    print(f"Written : {output}  ({len(pdf) / 1024:.0f} kB)")
    if args.png:
        import cv2
        png = output.with_suffix(".png")
        cv2.imwrite(str(png), image)
        print(f"          {png}  (screen check only)")
    print()
    print("NEXT")
    print("  1. Print at 100% / Actual size. Turn OFF 'fit to page' / 'shrink to fit'.")
    print(f"  2. Measure the {CHECK_BAR_MM:.0f} mm bar. Not exact -> reprint, do not proceed.")
    print("  3. Measure a tag's BLACK SQUARE and the white gap with a caliper,")
    print("     and put the measured values in config/calibration.yaml. Even a")
    print("     correct printer is worth verifying: tag_size_m scales the whole")
    print("     hand-eye translation without changing reprojection error.")
    print("  4. Mount it flat. A curved board breaks the planarity the solve assumes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
