#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Generate printable ArUco marker PNGs from a marker config, or a ChArUco board.

Typical use via the ``optical`` CLI::

    optical markers --marker-config software/optical/etc/optical.yaml \\
        --out-dir data/optical/markers_out --side-px 800

    optical markers --charuco --squares-x 7 --squares-y 5 \\
        --square-mm 30 --marker-mm 22 --dictionary DICT_4X4_50 \\
        --out calibrations/charuco_7x5.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from optical.markers.config import (
        generate_charuco_board_png,
        generate_marker_png,
        load_marker_config,
        validate_marker_config,
    )
except ModuleNotFoundError:
    _src_root = Path(__file__).resolve().parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    from optical.markers.config import (
        generate_charuco_board_png,
        generate_marker_png,
        load_marker_config,
        validate_marker_config,
    )


def run_charuco(
    *,
    out: Path,
    squares_x: int = 7,
    squares_y: int = 5,
    square_mm: float = 30.0,
    marker_mm: float = 22.0,
    dictionary: str = "DICT_4X4_50",
    dpi: int = 300,
    margin_mm: float = 5.0,
) -> Path:
    """Render a single printable ChArUco board PNG."""
    out_path = generate_charuco_board_png(
        squares_x=squares_x,
        squares_y=squares_y,
        square_length_mm=square_mm,
        marker_length_mm=marker_mm,
        dictionary_name=dictionary,
        out_path=out,
        dpi=dpi,
        margin_mm=margin_mm,
    )
    print(f"Wrote {out_path}")
    print(
        f"Board: {squares_x}x{squares_y} squares, "
        f"square={square_mm}mm marker={marker_mm}mm "
        f"dict={dictionary} @ {dpi} dpi (print at 100% scale)."
    )
    return Path(out_path)


def run_markers(
    *,
    marker_config: Path,
    out_dir: Path,
    side_px: int = 600,
    ids: list[int] | None = None,
) -> list[Path]:
    """Validate marker config and write one PNG per marker ID."""
    cfg = load_marker_config(marker_config)
    errs = validate_marker_config(cfg)
    if errs:
        raise SystemExit("Invalid marker config:\n- " + "\n- ".join(errs))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ids is None:
        marker_ids: list[int] = []
        for _, (lo, hi) in cfg.id_ranges.items():
            marker_ids.extend(range(lo, hi + 1))
        marker_ids = sorted(set(marker_ids))
    else:
        marker_ids = ids

    written: list[Path] = []
    for mid in marker_ids:
        out_path = out_dir / f"aruco_{cfg.dictionary}_id{mid}.png"
        generate_marker_png(
            dictionary_name=cfg.dictionary,
            marker_id=mid,
            side_px=side_px,
            out_path=out_path,
        )
        print(f"Wrote {out_path}")
        written.append(out_path)
    return written


def main(argv: list[str] | None = None) -> None:
    """Argparse entry for ``python -m optical.markers.generate``."""
    ap = argparse.ArgumentParser(
        description="Generate ArUco marker PNGs (or a ChArUco board) for JTS calibration.",
    )
    ap.add_argument(
        "--charuco",
        action="store_true",
        help="Generate a single ChArUco calibration board instead of per-marker PNGs",
    )
    ap.add_argument(
        "--marker-config",
        help="Path to optical.yaml (uses markers: section)",
    )
    ap.add_argument("--out-dir", help="Output directory for marker PNGs (marker mode)")
    ap.add_argument(
        "--side-px",
        type=int,
        default=600,
        help="Marker image side length in pixels (marker mode)",
    )
    ap.add_argument(
        "--ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional marker IDs; default is all IDs in configured ranges (marker mode)",
    )

    charuco = ap.add_argument_group("charuco board (with --charuco)")
    charuco.add_argument("--out", help="Output PNG path for the ChArUco board")
    charuco.add_argument("--squares-x", type=int, default=7, help="Squares along width")
    charuco.add_argument("--squares-y", type=int, default=5, help="Squares along height")
    charuco.add_argument("--square-mm", type=float, default=30.0, help="Square edge length (mm)")
    charuco.add_argument("--marker-mm", type=float, default=22.0, help="Marker edge length (mm)")
    charuco.add_argument("--dictionary", default="DICT_4X4_50", help="OpenCV ArUco dictionary")
    charuco.add_argument("--dpi", type=int, default=300, help="Print resolution (px per inch)")
    charuco.add_argument("--margin-mm", type=float, default=5.0, help="White border (mm)")

    args = ap.parse_args(argv)

    if args.charuco:
        if not args.out:
            ap.error("--charuco requires --out PATH")
        run_charuco(
            out=Path(args.out),
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_mm=args.square_mm,
            marker_mm=args.marker_mm,
            dictionary=args.dictionary,
            dpi=args.dpi,
            margin_mm=args.margin_mm,
        )
        return

    if not args.marker_config or not args.out_dir:
        ap.error("marker mode requires --marker-config and --out-dir")

    run_markers(
        marker_config=Path(args.marker_config),
        out_dir=Path(args.out_dir),
        side_px=args.side_px,
        ids=args.ids,
    )


if __name__ == "__main__":
    main()
