# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm view / jtsmm replay — modern web viewer for EMG + IMU."""

from __future__ import annotations

import webbrowser
from pathlib import Path

import click

from jtsmm.config.catalog import EMG_GAINS


def _run_server(source, host: str, port: int, open_browser: bool) -> None:
    from jtsmm.viewer.server import ViewerServer

    server = ViewerServer(source, host=host, port=port)
    server.print_startup()
    click.echo(f"viewer serving at {server.url} (Ctrl-C to stop)")
    if open_browser:
        try:
            webbrowser.open(server.url)
        except Exception:
            pass
    try:
        server.run()
    except KeyboardInterrupt:
        click.echo("stopping viewer...")


def _viewer_address(
    config_path: Path | None, host: str | None, http_port: int | None, *, mock: bool = False
) -> tuple[str, int]:
    """CLI > YAML (host.viewer.*) > built-in default, for the bind address.

    The YAML used to be read only with an explicit --config, and an explicit
    `--http-port 8770` was indistinguishable from "not given".
    """
    if host is not None and http_port is not None:
        return host, int(http_port)
    from jtsmm.config import load
    from jtsmm.config.registry import viewer_bind_host, viewer_port as viewer_port_from_config

    try:
        resolved = load(config_path)
    except Exception:
        if not mock and config_path is not None:
            raise
        resolved = {}
    return (
        host if host is not None else viewer_bind_host(resolved),
        int(http_port) if http_port is not None else viewer_port_from_config(resolved),
    )


@click.command("view")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", "serial_port", default=None, help="MCU serial port override")
@click.option("--host", default=None, help="Viewer bind host (default: host.viewer.bind_host)")
@click.option("--http-port", type=int, default=None,
              help="Viewer HTTP/WS port (default: host.viewer.port)")
@click.option("--no-open", is_flag=True, help="Do not open a browser")
@click.option("--mock", is_flag=True, help="Use a synthetic source (no hardware)")
@click.option(
    "--bias-drive/--no-bias-drive",
    "bias_drive",
    default=None,
    help="Enable/disable the ADS1299 DRL/bias drive (overrides config)",
)
@click.option(
    "--bias-channels",
    default=None,
    metavar="CH",
    help="CH1–CH6 that drive the DRL average, e.g. '1,4' or '1-3' (default: active channels)",
)
@click.option(
    "--active-channels",
    default=None,
    metavar="CH",
    help="Only these CH1–CH6 stay powered; the rest are powered down, e.g. '4'",
)
@click.option(
    "--leadoff/--no-leadoff",
    "leadoff",
    default=None,
    help=(
        "Enable/disable DC lead-off contact detection (default: on). The contact "
        "LEDs go red for an input stuck at a rail, and for a comparator hit only "
        "when the MCU reports its comparators powered (older firmware left them "
        "off); otherwise they show 'unknown', never a false 'connected'. The DRL "
        "LED is always n/a here: run `jtsmm emg check`."
    ),
)
@click.option(
    "--gain",
    "default_gain",
    type=click.Choice([str(g) for g in EMG_GAINS]),
    default=None,
    help=f"ADS1299 PGA gain for all patient channels, one of {EMG_GAINS} (overrides config)",
)
def view_cmd(
    config_path: Path | None,
    serial_port: str | None,
    host: str | None,
    http_port: int | None,
    no_open: bool,
    mock: bool,
    bias_drive: bool | None,
    bias_channels: str | None,
    active_channels: str | None,
    leadoff: bool | None,
    default_gain: str | None,
) -> None:
    """Standalone live viewer (owns the USB endpoint; not for use while recording)."""
    host, http_port = _viewer_address(config_path, host, http_port, mock=mock)
    try:
        if mock:
            from jtsmm.viewer.sources import MockSource

            source = MockSource()
        else:
            from jtsmm.viewer.sources import LiveSource

            source = LiveSource(
                config_path=config_path,
                port=serial_port,
                bias_drive=bias_drive,
                bias_channels=bias_channels,
                active_channels=active_channels,
                leadoff=leadoff,
                default_gain=int(default_gain) if default_gain is not None else None,
                monitoring=True,
            )
    except Exception as exc:
        raise SystemExit(f"viewer failed to start: {exc}") from exc
    _run_server(source, host, http_port, not no_open)


@click.command("replay")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None,
              help="Config for host.viewer.* defaults")
@click.option("--host", default=None, help="Viewer bind host (default: host.viewer.bind_host)")
@click.option("--http-port", type=int, default=None,
              help="Viewer HTTP/WS port (default: host.viewer.port)")
@click.option("--speed", type=float, default=1.0, help="Playback speed multiplier")
@click.option("--start", "start_s", type=float, default=0.0, help="Start offset (seconds)")
@click.option("--no-open", is_flag=True, help="Do not open a browser")
def replay_cmd(
    path: Path,
    config_path: Path | None,
    host: str | None,
    http_port: int | None,
    speed: float,
    start_s: float,
    no_open: bool,
) -> None:
    """Replay a recorded session.h5 with seek/speed/pause controls."""
    from jtsmm.viewer.sources import ReplaySource, validate_replay_speed

    try:
        speed = validate_replay_speed(speed)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--speed") from exc
    host, http_port = _viewer_address(config_path, host, http_port, mock=True)
    source = ReplaySource(path, speed=speed, start_s=start_s)
    _run_server(source, host, http_port, not no_open)
