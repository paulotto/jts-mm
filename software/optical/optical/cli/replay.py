# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""optical replay — offline replay of recorded camera frames."""

from __future__ import annotations

from pathlib import Path

import click

from optical.tracking.replay import run_compare, run_replay


@click.command("replay")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    help="optical.yaml (default: ~/optical.yaml or package etc)",
)
@click.option(
    "--frames",
    "frames_dir",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    help="Replay session directory (contains meta.json and frames/)",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Write tracking HDF5 to this path (default: data/optical/tracking/<experiment>.h5)",
)
@click.option(
    "--experiment-name",
    default="replay",
    show_default=True,
    help="Experiment name for HDF5 metadata and default output filename",
)
@click.option(
    "--print-every",
    type=int,
    default=30,
    show_default=True,
    help="Print a debug line every N frames (0 = summary only)",
)
@click.option("--verbose", is_flag=True, help="Include detected marker IDs in printed lines")
@click.option("--max-frames", type=int, default=None, help="Process at most this many frames")
@click.option(
    "--compare-methods",
    is_flag=True,
    help="Run PnP and triangulation on each frame and compare (implies --debug-only)",
)
@click.option(
    "--debug-only",
    is_flag=True,
    help="Run diagnostics only; do not write tracking HDF5",
)
@click.pass_context
def replay_cmd(
    ctx: click.Context,
    config_path: Path | None,
    frames_dir: Path,
    output_path: Path | None,
    experiment_name: str,
    print_every: int,
    verbose: bool,
    max_frames: int | None,
    compare_methods: bool,
    debug_only: bool,
) -> None:
    """Replay recorded camera frames through the jaw tracker."""
    from optical.config.loader import default_config_path

    cfg = config_path or default_config_path()
    if not Path(cfg).is_file():
        raise click.ClickException(f"optical config not found: {cfg}")
    try:
        if compare_methods:
            if output_path is not None:
                raise click.UsageError("--compare-methods cannot be combined with --output")
            if ctx.get_parameter_source("experiment_name") == click.core.ParameterSource.COMMANDLINE:
                raise click.UsageError("--experiment-name names the HDF5; --compare-methods writes none")
            run_compare(
                tracker_config_path=Path(cfg),
                frames_dir=frames_dir,
                print_every=print_every,
                verbose=verbose,
                max_frames=max_frames,
            )
            return

        run_replay(
            tracker_config_path=Path(cfg),
            frames_dir=frames_dir,
            output_path=output_path,
            experiment_name=experiment_name,
            print_every=print_every,
            verbose=verbose,
            max_frames=max_frames,
            debug_only=debug_only,
        )
    except KeyboardInterrupt:
        click.echo("\nAborted (Ctrl+C).", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    replay_cmd.main(standalone_mode=True)
