# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ml label — interactive trajectory labeling."""

from __future__ import annotations

from pathlib import Path

import click

from ml.config.loader import describe_source, label_section, load_ml_config
from ml.io.labels import default_label_path


def _parse_invert_axis(value: tuple[str, ...]) -> tuple[bool, bool, bool]:
    flags = {part.strip().lower() for part in value}
    return ("x" in flags, "y" in flags, "z" in flags)


@click.command("label")
@click.argument("session", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="ml.yaml (default: the packaged software/ml/etc/ml.yaml)",
)
@click.option(
    "--jts-output",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JTS jaw_motion.h5 trajectory file (auto-discovered if omitted)",
)
@click.option(
    "--jts-config",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="JTS config JSON for frame alignment (auto-discovered if omitted)",
)
@click.option(
    "--labels",
    "label_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Label sidecar output path (default: <session_stem>_labels.h5)",
)
@click.option("--port", default=None, type=int, help="Override label.port from config")
@click.option(
    "--host",
    default=None,
    help="Override label.host from config (0.0.0.0 = reachable on the network)",
)
@click.option(
    "--local-only",
    is_flag=True,
    help="Bind to 127.0.0.1 instead of config host",
)
@click.option(
    "--velocity/--no-velocity",
    default=None,
    help="Overlay translational velocity (default: label.show_velocity)",
)
@click.option(
    "--invert-axis",
    multiple=True,
    type=click.Choice(["x", "y", "z"], case_sensitive=False),
    help="Flip axis sign (repeatable; overrides label.invert_axis)",
)
def label_cmd(
    session: Path,
    config_path: Path | None,
    jts_output: Path | None,
    jts_config: Path | None,
    label_path: Path | None,
    port: int | None,
    host: str | None,
    local_only: bool,
    velocity: bool | None,
    invert_axis: tuple[str, ...],
) -> None:
    """Label JTS trajectories for a multimodal session."""
    from ml.labeling.app import run_label_app
    from ml.labeling.trajectory import discover_jts_config, resolve_jts_source

    click.echo(describe_source(config_path))
    label_cfg = label_section(load_ml_config(config_path))

    host = "127.0.0.1" if local_only else (host or str(label_cfg.get("host", "0.0.0.0")))
    port = int(port if port is not None else label_cfg.get("port", 8050))
    if velocity is None:
        velocity = bool(label_cfg.get("show_velocity", False))
    if invert_axis:
        flips = _parse_invert_axis(invert_axis)
    else:
        flips = _parse_invert_axis(tuple(label_cfg.get("invert_axis") or ()))

    jts_config = jts_config or discover_jts_config(session)
    jts_source = resolve_jts_source(session, jts_config, jts_output)
    label_path = label_path or default_label_path(session)

    if jts_source:
        if jts_source.parent_group:
            click.echo(f"Using embedded JTS in session: {jts_source.display_path}")
        else:
            click.echo(f"Using JTS output: {jts_source.path}")
    else:
        click.echo("No JTS output found — using /optical CRA-relative fallback trajectory.")

    try:
        out = run_label_app(
            session,
            jts_output=jts_source.path if jts_source else jts_output,
            jts_config=jts_config,
            label_path=label_path,
            port=port,
            host=host,
            show_velocity=velocity,
            invert_axis=flips,
        )
    except ImportError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from exc

    click.echo(f"Labels written to {out}")
