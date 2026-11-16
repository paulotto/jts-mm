# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""optical markers — printable ArUco / ChArUco PNGs."""

from __future__ import annotations

from pathlib import Path

import click

from optical.markers.generate import run_charuco, run_markers


# Options that only one of the two modes consumes.
_MARKER_ONLY = {"marker_config", "out_dir", "side_px", "ids"}
_CHARUCO_ONLY = {
    "out_path", "squares_x", "squares_y", "square_mm", "marker_mm", "dictionary", "dpi", "margin_mm",
}


@click.command("markers")
@click.option(
    "--charuco",
    is_flag=True,
    help="Generate a single ChArUco calibration board instead of per-marker PNGs",
)
@click.option(
    "--marker-config",
    "marker_config",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="optical.yaml (uses markers: section; default: package etc)",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory for marker PNGs (marker mode)",
)
@click.option(
    "--side-px",
    type=int,
    default=600,
    show_default=True,
    help="Marker image side length in pixels (marker mode)",
)
@click.option(
    "--ids",
    multiple=True,
    type=int,
    help="Optional marker IDs; default is all IDs in configured ranges (repeatable)",
)
@click.option("--out", "out_path", type=click.Path(path_type=Path), default=None, help="ChArUco board PNG path")
@click.option("--squares-x", type=int, default=7, show_default=True, help="Squares along width")
@click.option("--squares-y", type=int, default=5, show_default=True, help="Squares along height")
@click.option("--square-mm", type=float, default=30.0, show_default=True, help="Square edge length (mm)")
@click.option("--marker-mm", type=float, default=22.0, show_default=True, help="Marker edge length (mm)")
@click.option("--dictionary", default="DICT_4X4_50", show_default=True, help="OpenCV ArUco dictionary")
@click.option("--dpi", type=int, default=300, show_default=True, help="Print resolution (px per inch)")
@click.option("--margin-mm", type=float, default=5.0, show_default=True, help="White border (mm)")
@click.pass_context
def markers_cmd(
    ctx: click.Context,
    charuco: bool,
    marker_config: Path | None,
    out_dir: Path | None,
    side_px: int,
    ids: tuple[int, ...],
    out_path: Path | None,
    squares_x: int,
    squares_y: int,
    square_mm: float,
    marker_mm: float,
    dictionary: str,
    dpi: int,
    margin_mm: float,
) -> None:
    """Generate ArUco marker PNGs or a printable ChArUco board."""
    # An option of the other mode would be silently ignored: refuse it instead.
    given = {
        name
        for name in ctx.params
        if ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE
    }
    other = _MARKER_ONLY if charuco else _CHARUCO_ONLY
    stray = sorted(given & other)
    if stray:
        flags = ", ".join("--" + n.replace("_", "-").replace("out-path", "out") for n in stray)
        mode = "--charuco" if charuco else "marker mode"
        raise click.UsageError(f"{flags} not used in {mode}")
    if charuco:
        if out_path is None:
            raise click.UsageError("--charuco requires --out PATH")
        run_charuco(
            out=out_path,
            squares_x=squares_x,
            squares_y=squares_y,
            square_mm=square_mm,
            marker_mm=marker_mm,
            dictionary=dictionary,
            dpi=dpi,
            margin_mm=margin_mm,
        )
        return

    if marker_config is None or out_dir is None:
        from optical.config.loader import default_config_path

        if marker_config is None:
            marker_config = default_config_path()
        if out_dir is None:
            raise click.UsageError("marker mode requires --out-dir")

    run_markers(
        marker_config=marker_config,
        out_dir=out_dir,
        side_px=side_px,
        ids=list(ids) if ids else None,
    )


if __name__ == "__main__":
    markers_cmd.main(standalone_mode=True)
