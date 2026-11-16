# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""optical calibrate — multi-camera checkerboard calibration."""

from __future__ import annotations

from pathlib import Path

import click

from optical.calibration.multicam import run_calibration


@click.command("calibrate")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    help="optical.yaml (uses calibration: section; default: package etc / ~/optical.yaml)",
)
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["intrinsics", "extrinsics", "bundle"], case_sensitive=False),
    help=(
        "intrinsics | extrinsics (writes the bundle only when the quality gates pass) "
        "| bundle (write the bundle from the caches)"
    ),
)
@click.option(
    "--backend",
    default="picamera2",
    type=click.Choice(["picamera2", "v4l2"], case_sensitive=False),
    show_default=True,
    help="Camera capture backend",
)
@click.option("--preview-host", default="0.0.0.0", show_default=True, help="Flask preview bind address")
@click.option("--preview-port", type=int, default=5000, show_default=True, help="Flask preview port")
def calibrate_cmd(
    config_path: Path | None,
    mode: str,
    backend: str,
    preview_host: str,
    preview_port: int,
) -> None:
    """Run intrinsics, extrinsics (bundle if the gates pass), or bundle export from optical.yaml."""
    from optical.config.loader import default_config_path

    cfg = config_path or default_config_path()
    if not Path(cfg).is_file():
        raise click.ClickException(f"optical config not found: {cfg}")
    try:
        run_calibration(
            config_path=Path(cfg),
            mode=mode.lower(),
            backend=backend.lower(),
            preview_host=preview_host,
            preview_port=preview_port,
        )
    except KeyboardInterrupt:
        click.echo("\nAborted (Ctrl+C).", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    calibrate_cmd.main(standalone_mode=True)
