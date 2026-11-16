# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""optical eval / eval-quality — ArUco marker evaluation commands."""

from __future__ import annotations

from pathlib import Path

import click

from optical.evaluation.run_marker_size_eval import run_marker_size_eval
from optical.evaluation.run_quality_size_eval import run_quality_size_eval


@click.command("eval")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(path_type=Path),
    help="optical.yaml (uses eval: section; default: package etc)",
)
@click.option(
    "--backend",
    default="picamera2",
    type=click.Choice(["picamera2", "v4l2"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--preview-port",
    type=int,
    default=5000,
    show_default=True,
    help="MJPEG preview port (0 = off)",
)
@click.option("--preview-host", default="0.0.0.0", show_default=True)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Report JSON path (default: data/optical/evaluation/<timestamp>.json)",
)
@click.option("--skip-motion", is_flag=True, help="Only run the static phase")
@click.option(
    "--resolution",
    default=None,
    help="Capture/eval resolution WxH (e.g. 728x544). Overrides config; scales calibration K.",
)
@click.option(
    "--cameras",
    default=None,
    help="Camera ids (e.g. cam0 or cam0,cam1). 1 camera → PnP only; 2 → triangulation.",
)
@click.option(
    "--n-cameras",
    "n_cameras",
    type=click.Choice(["1", "2"], case_sensitive=False),
    default=None,
    help="Use first N cameras from tracker/calibration (1=PnP, 2=triangulation).",
)
def eval_cmd(
    config_path: Path | None,
    backend: str,
    preview_port: int,
    preview_host: str,
    output_path: Path | None,
    skip_motion: bool,
    resolution: str | None,
    cameras: str | None,
    n_cameras: str | None,
) -> None:
    """ArUco marker size evaluation at fixed distance (live)."""
    from optical.config.loader import default_config_path

    cfg = config_path or default_config_path()
    try:
        run_marker_size_eval(
            config=cfg,
            backend=backend.lower(),
            preview_host=preview_host,
            preview_port=preview_port,
            output=output_path,
            skip_motion=skip_motion,
            resolution=resolution,
            cameras=cameras,
            n_cameras=int(n_cameras) if n_cameras is not None else None,
        )
    except KeyboardInterrupt:
        click.echo("\nAborted (Ctrl+C).", err=True)
        raise SystemExit(130) from None


@click.command("eval-quality")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(path_type=Path),
    help="optical.yaml (uses eval_quality: section; default: package etc)",
)
@click.option(
    "--backend",
    default="picamera2",
    type=click.Choice(["picamera2", "v4l2"], case_sensitive=False),
    show_default=True,
)
@click.option(
    "--preview-port",
    type=int,
    default=5000,
    show_default=True,
    help="MJPEG preview port during live capture (0 = off)",
)
@click.option("--preview-host", default="0.0.0.0", show_default=True)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Report JSON path",
)
@click.option(
    "--frames-dir",
    "frames_dir",
    default=None,
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    help="Replay a prior full-res dump (skips live capture)",
)
@click.option(
    "--save-frames",
    is_flag=True,
    help="Dump the full-res burst for later --frames-dir replay",
)
def eval_quality_cmd(
    config_path: Path | None,
    backend: str,
    preview_port: int,
    preview_host: str,
    output_path: Path | None,
    frames_dir: Path | None,
    save_frames: bool,
) -> None:
    """Marker size × image quality ladder (lores tradeoff)."""
    from optical.config.loader import default_config_path

    cfg = config_path or default_config_path()
    try:
        run_quality_size_eval(
            config=cfg,
            backend=backend.lower(),
            preview_host=preview_host,
            preview_port=preview_port,
            output=output_path,
            frames_dir=frames_dir,
            save_frames=save_frames,
        )
    except KeyboardInterrupt:
        click.echo("\nAborted (Ctrl+C).", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    eval_cmd.main(standalone_mode=True)
