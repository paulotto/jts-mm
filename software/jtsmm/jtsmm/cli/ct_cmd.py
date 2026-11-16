# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm ct-calibrate — interactive CT tip pivot calibration only."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import click

from jtsmm.capture.session import resolve_session_config, run_session
from jtsmm.config.loader import default_config_path
from jtsmm.config.registry import (
    record_preflight_enabled,
    session_ct_calibration_n_poses,
    session_ct_calibration_recordings_dir,
    session_ct_tip_calibration,
    session_optical_config,
    session_persist_ct_calibration,
    viewer_bind_host,
    viewer_enabled,
    viewer_port as viewer_port_from_config,
)
from jtsmm.doctor import run_preflight
from jtsmm.doctor.preflight_issues import partition_preflight_issues
from jtsmm.io.session_reproduce import RecordReproduceSpec


@click.command("ct-calibrate")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", default=None, help="Serial port override")
@click.option(
    "--optical-config",
    type=click.Path(path_type=Path),
    default=None,
    help="optical.yaml (default: session.optical_config)",
)
@click.option(
    "--ct-calibration-n-poses",
    type=int,
    default=None,
    help="Pivot poses for OLS tip solve (default: session.ct_calibration_n_poses / 20)",
)
@click.option(
    "--ct-tip-calibration",
    type=click.Path(path_type=Path),
    default=None,
    help="Reusable tip-offset JSON to write (default: session.ct_tip_calibration)",
)
@click.option(
    "--persist-ct-calibration/--no-persist-ct-calibration",
    default=None,
    help=(
        "Save poses + camera frames under ct_calibration_recordings_dir "
        "(default: session.persist_ct_calibration)"
    ),
)
@click.option(
    "--ct-calibration-recordings-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory template for persisted recordings "
        "(default: session.ct_calibration_recordings_dir)"
    ),
)
@click.option("--skip-apply", is_flag=True, help="Do not send config apply to MCU")
@click.option("--skip-preflight", is_flag=True, help="Skip preflight checks")
@click.option("--no-doctor", is_flag=True, help="Skip preflight checks (alias)")
@click.option("--viewer/--no-viewer", default=None, help="Serve live web viewer during calib")
@click.option("--viewer-host", default=None, help="Viewer bind host")
@click.option("--viewer-port", type=int, default=None, help="Viewer HTTP/WS port")
@click.option("--mock", is_flag=True, help="Skip MCU serial; mock cameras only")
def ct_calibrate_cmd(
    config_path: Path | None,
    port: str | None,
    optical_config: Path | None,
    ct_calibration_n_poses: int | None,
    ct_tip_calibration: Path | None,
    persist_ct_calibration: bool | None,
    ct_calibration_recordings_dir: Path | None,
    skip_apply: bool,
    skip_preflight: bool,
    no_doctor: bool,
    viewer: bool | None,
    viewer_host: str | None,
    viewer_port: int | None,
    mock: bool,
) -> None:
    """Run CT tip pivot calibration (cone + SPACE×N) without a motion take.

    Writes ``ct_tip_calibration.json`` into a session folder and, when configured,
    the reusable tip JSON plus optional pose/frame recordings.
    """
    cfg = resolve_session_config(config_path)
    use_viewer = viewer_enabled(cfg) if viewer is None else viewer
    use_viewer_host = viewer_host if viewer_host is not None else viewer_bind_host(cfg)
    use_viewer_port = viewer_port if viewer_port is not None else viewer_port_from_config(cfg)

    if not no_doctor and not skip_preflight and record_preflight_enabled(cfg):
        # Probe the port this run will use: --port wins over hat.serial_port.
        pre_cfg = copy.deepcopy(cfg)
        if port:
            pre_cfg.setdefault("hat", {})["serial_port"] = port
        issues = run_preflight(
            pre_cfg,
            check_serial=not mock,
            check_mcu_selftest=not mock,
        )
        errors, warnings, infos = partition_preflight_issues(issues)
        for w in warnings:
            click.echo(f"warning: {w}", err=True)
        for info in infos:
            click.echo(info)
        if errors:
            for err in errors:
                click.echo(f"error: {err}", err=True)
            raise SystemExit(1)

    src_config = Path(config_path) if config_path is not None else default_config_path()
    eff_n_poses = (
        session_ct_calibration_n_poses(cfg)
        if ct_calibration_n_poses is None
        else int(ct_calibration_n_poses)
    )
    eff_tip = ct_tip_calibration or session_ct_tip_calibration(cfg)
    eff_persist = (
        session_persist_ct_calibration(cfg)
        if persist_ct_calibration is None
        else bool(persist_ct_calibration)
    )
    eff_rec_dir = ct_calibration_recordings_dir or session_ct_calibration_recordings_dir(cfg)
    eff_opt_cfg = optical_config or session_optical_config(cfg)
    if eff_opt_cfg is None and not mock:
        raise click.ClickException(
            "optical.yaml required (session.optical_config or --optical-config)"
        )

    click.echo(
        f"CT tip calibration only: SPACE × {eff_n_poses} "
        f"(persist={'on' if eff_persist else 'off'})"
    )

    reproduce = RecordReproduceSpec(
        argv=list(sys.argv),
        config_path=str(src_config),
        port=port,
        mock=mock,
        skip_apply=skip_apply,
        skip_preflight=skip_preflight or no_doctor,
        no_integrity=True,
        viewer=viewer,
        viewer_host=viewer_host,
        viewer_port=viewer_port,
        emg_check=False,
        optical_track=False,
        optical_config=str(eff_opt_cfg) if eff_opt_cfg is not None else None,
        ct_calibration=True,
        ct_calibration_n_poses=eff_n_poses,
        ct_tip_calibration=str(eff_tip) if eff_tip is not None else None,
        persist_ct_calibration=eff_persist,
        ct_calibration_recordings_dir=str(eff_rec_dir) if eff_rec_dir is not None else None,
        use_experiment_plan=False,
        mvc=False,
        write_jts_config=False,
        ct_calibrate=True,
    )

    try:
        out = run_session(
            config_path,
            port=port,
            skip_mcu_apply=skip_apply,
            mock_only=mock,
            viewer=use_viewer,
            viewer_host=use_viewer_host,
            viewer_port=use_viewer_port,
            optical_track=False,
            optical_config=Path(eff_opt_cfg) if eff_opt_cfg is not None else None,
            use_experiment_plan=False,
            mvc=False,
            write_jts_config=False,
            ct_calibration=True,
            ct_calibration_n_poses=eff_n_poses,
            ct_tip_calibration=Path(eff_tip) if eff_tip is not None else None,
            persist_ct_calibration=eff_persist,
            ct_calibration_recordings_dir=eff_rec_dir,
            ct_calibration_only=True,
            enable_emg=False,
            reproduce=reproduce,
        )
    except Exception as exc:
        click.echo(f"ct-calibrate failed: {exc}", err=True)
        raise SystemExit(1) from exc

    tip_json = out.parent / "ct_tip_calibration.json"
    click.echo(f"Wrote CT tip calibration session -> {out}")
    if tip_json.is_file():
        click.echo(f"Tip JSON -> {tip_json}")


if __name__ == "__main__":
    ct_calibrate_cmd.main(standalone_mode=True)
