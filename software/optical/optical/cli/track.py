# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""optical track — live ArUco jaw tracker."""

from __future__ import annotations

from pathlib import Path

import click

from optical.tracking.run import run_live


@click.command("track")
@click.option(
    "--config",
    "config_path",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    help="optical.yaml (default: ~/optical.yaml or software/optical/etc/optical.yaml)",
)
@click.option(
    "--experiment-plan",
    "experiment_plan_path",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    help="Override experiment plan (default: experiment: in optical.yaml)",
)
@click.option(
    "--use-experiment-plan/--no-use-experiment-plan",
    "use_experiment_plan",
    default=None,
    help="Walk motion+landmark phases (default: experiment.use_experiment_plan)",
)
@click.option(
    "--backend",
    default="picamera2",
    type=click.Choice(["picamera2", "v4l2"], case_sensitive=False),
    show_default=True,
    help="Camera backend",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Output HDF5 path (default: data/optical/tracking or persist leaf)",
)
@click.option(
    "--jts-config",
    "jts_config_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Where to write the generated JTS config (default: next to session)",
)
@click.option(
    "--record-frames",
    "record_frames_dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Save synchronized camera PNGs for offline replay (explicit dir)",
)
@click.option(
    "--persist-frames",
    is_flag=True,
    help=(
        "Write session.h5 + frames under paths.recordings_dir "
        "(~/experiments/{folder}/recordings/{name}_{condition}_{stamp}/). "
        "Uses experiment.name / experiment.condition from optical.yaml "
        "(--condition overrides)."
    ),
)
@click.option(
    "--condition",
    default=None,
    help="Override experiment.condition from optical.yaml (e.g. static, near, mid, far)",
)
@click.option("--preview-host", default="0.0.0.0", show_default=True)
@click.option("--preview-port", type=int, default=5000, show_default=True)
@click.option(
    "--hat-port",
    default=None,
    help="Serial port for HAT CDC, used for trigger + sync (default: auto-detect)",
)
@click.option(
    "--trigger-rate",
    "trigger_rate",
    type=click.IntRange(min=1),
    default=None,
    help="Camera trigger rate in Hz (default: mcu.trigger.rate_hz from jtsmm.yaml)",
)
@click.pass_context
def track_cmd(
    ctx: click.Context,
    config_path: Path,
    experiment_plan_path: Path | None,
    use_experiment_plan: bool | None,
    backend: str,
    output_path: Path | None,
    jts_config_path: Path | None,
    record_frames_dir: Path | None,
    persist_frames: bool,
    condition: str | None,
    preview_host: str,
    preview_port: int,
    hat_port: str | None,
    trigger_rate: int | None,
) -> None:
    """Live ArUco jaw tracker (HP/MP/CT poses → HDF5)."""
    from optical.config.loader import default_config_path

    cfg = config_path or default_config_path()
    if not Path(cfg).is_file():
        raise click.ClickException(f"optical config not found: {cfg}")
    if persist_frames and record_frames_dir is not None:
        raise click.UsageError("use either --persist-frames or --record-frames, not both")
    if use_experiment_plan is False and experiment_plan_path is not None:
        raise click.UsageError("--experiment-plan requires --use-experiment-plan")
    if backend.lower() != "picamera2":
        # v4l2 has no HAT: these would be silently ignored.
        for name, flag in (("hat_port", "--hat-port"), ("trigger_rate", "--trigger-rate")):
            if ctx.get_parameter_source(name) == click.core.ParameterSource.COMMANDLINE:
                raise click.UsageError(f"{flag} needs --backend picamera2 (v4l2 has no HAT)")
    try:
        run_live(
            tracker_config_path=Path(cfg),
            experiment_plan_path=experiment_plan_path,
            camera_backend=backend.lower(),
            preview_host=preview_host,
            preview_port=preview_port,
            output_path=output_path,
            jts_config_path=jts_config_path,
            record_frames_dir=record_frames_dir,
            persist_frames=persist_frames,
            condition=condition,
            use_experiment_plan=use_experiment_plan,
            hat_port=hat_port,
            trigger_rate_hz=trigger_rate,
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    except KeyboardInterrupt:
        click.echo("\nAborted (Ctrl+C).", err=True)
        raise SystemExit(130) from None


if __name__ == "__main__":
    track_cmd.main(standalone_mode=True)
