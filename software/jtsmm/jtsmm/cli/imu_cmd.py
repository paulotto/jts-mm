# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm imu calibrate — record a mounting take or fit R_mount from a session."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import click

from jtsmm.imu.mounting import (
    MAX_RMS_DEG,
    PHASE_BY_SLOT,
    MountingFitError,
    fit_session_slot,
    read_phase_windows,
    update_mounting_file,
    write_mounting_transform,
)
from jtsmm.imu.persist import (
    IMU_MOUNTING_JSON,
    next_run,
    persist_dir_for,
    read_session_subject_id,
    slot_record_from_error,
    write_imu_mounting_recording,
)

_PLAN = Path(__file__).resolve().parents[2] / "etc" / "imu_mounting.json"


def _fit_session(
    session: Path, which: str, max_rms_deg: float, output_path: Path | None
) -> tuple[int, dict[str, dict[str, Any]]]:
    slots = ("imu_mta", "imu_cra") if which == "both" else (f"imu_{which}",)
    windows = read_phase_windows(session)
    out = output_path or session
    failed = False
    records: dict[str, dict[str, Any]] = {}
    for slot in slots:
        phase_name = PHASE_BY_SLOT[slot]
        t0 = t1 = None
        phase: str | None = None
        if phase_name in windows:
            t0, t1 = windows[phase_name]
            phase = phase_name
        elif any(n.startswith("imu_mount") for n in windows):
            msg = f"{slot}: session has IMU-mount phases but not {phase_name!r}"
            click.echo(f"error: {msg}", err=True)
            records[slot] = slot_record_from_error(slot, msg, phase=phase_name)
            failed = True
            continue
        try:
            result = fit_session_slot(
                session,
                slot,
                t0_us=t0,
                t1_us=t1,
                phase=phase,
                max_rms_deg=max_rms_deg,
                strict=False,
            )
        except MountingFitError as exc:
            click.echo(f"error: {exc}", err=True)
            records[slot] = slot_record_from_error(slot, str(exc), phase=phase)
            failed = True
            continue
        records[slot] = result.to_slot_dict()
        click.echo(result.summary())
        if result.accepted:
            write_mounting_transform(out, result)
        else:
            # Writing a rejected fit would overwrite a previously accepted one
            # in the same file; its numbers still go to the persisted record.
            click.echo(f"{slot}: not written (fit rejected)", err=True)
            failed = True
    if records:
        bits = []
        for slot, rec in records.items():
            if rec.get("axes_3d") is True:
                bits.append(f"{slot}=3/3")
            elif rec.get("axes_3d") is False:
                bits.append(f"{slot}=2/3")
            else:
                bits.append(f"{slot}=n/a")
        click.echo("axis coverage: " + "  ".join(bits))
    if any(rec.get("accepted") for rec in records.values()):
        click.echo(f"wrote /calibration/imu_*/mounting_transform → {out}")
    return (1 if failed else 0, records)


def _update_stable(config_path: Path | None, records: dict[str, dict[str, Any]],
                   source: Path) -> None:
    """Accepted fits go into ``session.imu_mounting_calibration``, which every later
    recording embeds (a rejected fit never replaces a good one)."""
    from jtsmm.capture.session import resolve_session_config
    from jtsmm.config.registry import session_imu_mounting_calibration

    path = session_imu_mounting_calibration(resolve_session_config(config_path))
    if not path:
        return
    done = update_mounting_file(path, records, source)
    if done:
        click.echo(f"updated the stable IMU mounting ({', '.join(done)}) -> {path}")
    else:
        click.echo(f"stable IMU mounting unchanged (no accepted fit) -> {path}")


def _persist_take(
    *,
    session_h5: Path,
    slots: dict[str, dict[str, Any]],
    persist: bool,
    recordings_dir: Path | str | None,
    subject_id: str | None,
    run: int | None,
    optical_config: Path | str | None,
    max_rms_deg: float,
) -> Path | None:
    if not persist:
        return None
    sid = subject_id or read_session_subject_id(session_h5)
    rec_dir = recordings_dir
    run_n = int(run) if run is not None else next_run(rec_dir)
    slug = f"imu_mounting_{run_n}"
    leaf = persist_dir_for(rec_dir, experiment_name=slug)
    write_imu_mounting_recording(
        leaf,
        slots,
        session_h5=session_h5,
        subject_id=sid,
        run=run_n,
        experiment_name=slug,
        source_session=session_h5,
        optical_config_path=optical_config,
        max_rms_deg=max_rms_deg,
    )
    sidecar = session_h5.parent / IMU_MOUNTING_JSON
    if sidecar.resolve() != (leaf / IMU_MOUNTING_JSON).resolve():
        sidecar.write_text((leaf / IMU_MOUNTING_JSON).read_text(encoding="utf-8"), encoding="utf-8")
    click.echo(f"persisted IMU mounting cal → {leaf}")
    return leaf


@click.group("imu")
def imu_group() -> None:
    """IMU mounting calibration (optical + game-RV, no EMG)."""


@imu_group.command("calibrate")
@click.option(
    "--mode",
    type=click.Choice(["record", "fit"]),
    default="record",
    show_default=True,
    help="record: capture figure-eights. fit: Park–Martin on an existing session.h5.",
)
@click.argument("session", required=False, type=click.Path(path_type=Path, exists=True))
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", default=None, help="J_USB1 serial port override")
@click.option(
    "--optical-config",
    type=click.Path(path_type=Path),
    default=None,
    help="optical.yaml (default: session.optical_config)",
)
@click.option("--skip-apply", is_flag=True, help="Skip MCU config apply")
@click.option("--skip-preflight", is_flag=True, help="Skip host pre-flight")
@click.option("--no-doctor", is_flag=True, help="Skip doctor / pre-flight")
@click.option("--no-integrity", is_flag=True, help="Skip post-session HDF5 integrity")
@click.option("--viewer/--no-viewer", default=None, help="Web viewer (default: host.viewer.enabled)")
@click.option("--viewer-host", default=None)
@click.option("--viewer-port", type=int, default=None)
@click.option("--mock", is_flag=True, help="Mock capture (no MCU / cameras)")
@click.option("--fit/--no-fit", default=True, help="After --mode record, also run --mode fit")
@click.option(
    "--imu",
    "which",
    type=click.Choice(["mta", "cra", "both"]),
    default="both",
    help="Which array to fit",
)
@click.option("--max-rms-deg", type=float, default=MAX_RMS_DEG, show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="HDF5 to write /calibration/ into (default: the session file)",
)
@click.option(
    "--persist-imu-calibration/--no-persist-imu-calibration",
    default=None,
    help="Copy session + metrics under imu_calibration_recordings_dir "
    "(default: session.persist_imu_calibration)",
)
@click.option(
    "--imu-calibration-recordings-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Persist leaf base (default: session.imu_calibration_recordings_dir)",
)
@click.option(
    "--run",
    "run_index",
    type=int,
    default=None,
    help="Independent repetition index (default: existing persist leaves + 1). "
    "Not per-subject — one operational cal per remount; extra takes are for §5.2.2.",
)
@click.option(
    "--subject",
    default=None,
    help="Optional note in persist JSON (mounting is not per-subject)",
)
def calibrate_cmd(
    mode: str,
    session: Path | None,
    config_path: Path | None,
    port: str | None,
    optical_config: Path | None,
    skip_apply: bool,
    skip_preflight: bool,
    no_doctor: bool,
    no_integrity: bool,
    viewer: bool | None,
    viewer_host: str | None,
    viewer_port: int | None,
    mock: bool,
    fit: bool,
    which: str,
    max_rms_deg: float,
    output_path: Path | None,
    persist_imu_calibration: bool | None,
    imu_calibration_recordings_dir: Path | None,
    run_index: int | None,
    subject: str | None,
) -> None:
    """IMU–array mounting calibration.

    \b
      jtsmm imu calibrate --mode record
      jtsmm imu calibrate --mode fit SESSION.h5
    """
    persist_opts = _resolve_persist_opts(
        config_path,
        persist_imu_calibration,
        imu_calibration_recordings_dir,
        optical_config,
        subject,
    )
    if mode == "fit":
        if session is None:
            raise click.UsageError("--mode fit requires SESSION.h5")
        code, records = _fit_session(session, which, max_rms_deg, output_path)
        fitted = output_path or session
        _update_stable(config_path, records, fitted)
        _persist_take(
            session_h5=fitted,
            slots=records,
            persist=persist_opts["persist"],
            recordings_dir=persist_opts["recordings_dir"],
            subject_id=persist_opts["subject_id"],
            run=run_index,
            optical_config=persist_opts["optical_config"],
            max_rms_deg=max_rms_deg,
        )
        if code:
            raise SystemExit(code)
        return
    if session is not None:
        raise click.UsageError("--mode record does not take a session path; use --mode fit")
    _record_then_maybe_fit(
        config_path=config_path,
        port=port,
        optical_config=optical_config,
        skip_apply=skip_apply,
        skip_preflight=skip_preflight,
        no_doctor=no_doctor,
        no_integrity=no_integrity,
        viewer=viewer,
        viewer_host=viewer_host,
        viewer_port=viewer_port,
        mock=mock,
        fit=fit,
        which=which,
        max_rms_deg=max_rms_deg,
        output_path=output_path,
        persist=persist_opts["persist"],
        recordings_dir=persist_opts["recordings_dir"],
        subject_id=persist_opts["subject_id"],
        run=run_index,
        persist_optical_config=persist_opts["optical_config"],
    )


def _resolve_persist_opts(
    config_path: Path | None,
    persist_flag: bool | None,
    recordings_dir: Path | None,
    optical_config: Path | None,
    subject: str | None,
) -> dict[str, Any]:
    from jtsmm.capture.session import resolve_session_config
    from jtsmm.config.registry import (
        session_imu_calibration_recordings_dir,
        session_optical_config,
        session_persist_imu_calibration,
        session_subject_id,
    )

    cfg = resolve_session_config(config_path)
    persist = session_persist_imu_calibration(cfg) if persist_flag is None else bool(persist_flag)
    rec_dir = recordings_dir or session_imu_calibration_recordings_dir(cfg)
    opt = optical_config or session_optical_config(cfg)
    sid = subject or session_subject_id(cfg)
    return {
        "persist": persist,
        "recordings_dir": rec_dir,
        "optical_config": str(opt) if opt is not None else None,
        "subject_id": sid,
    }


def _record_then_maybe_fit(
    *,
    config_path: Path | None,
    port: str | None,
    optical_config: Path | None,
    skip_apply: bool,
    skip_preflight: bool,
    no_doctor: bool,
    no_integrity: bool,
    viewer: bool | None,
    viewer_host: str | None,
    viewer_port: int | None,
    mock: bool,
    fit: bool,
    which: str,
    max_rms_deg: float,
    output_path: Path | None,
    persist: bool,
    recordings_dir: Path | str | None,
    subject_id: str | None,
    run: int | None,
    persist_optical_config: str | None,
) -> None:
    from optical.tracking.experiment import ExperimentPlan

    from jtsmm.capture.session import resolve_session_config, run_session
    from jtsmm.config.loader import default_config_path
    from jtsmm.config.registry import (
        record_integrity_enabled,
        record_preflight_enabled,
        session_optical_config,
        viewer_bind_host,
        viewer_enabled,
        viewer_port as viewer_port_from_config,
    )
    from jtsmm.doctor import run_preflight
    from jtsmm.doctor.preflight_issues import partition_preflight_issues
    from jtsmm.io.session_reproduce import RecordReproduceSpec

    if not _PLAN.is_file():
        raise click.ClickException(f"missing experiment plan {_PLAN}")
    plan = ExperimentPlan.from_json(_PLAN)
    cfg = resolve_session_config(config_path)
    opt_cfg = optical_config or session_optical_config(cfg)
    if opt_cfg is None and not mock:
        raise click.ClickException(
            "optical.yaml required (session.optical_config or --optical-config)"
        )

    use_viewer = viewer_enabled(cfg) if viewer is None else viewer
    use_viewer_host = viewer_host if viewer_host is not None else viewer_bind_host(cfg)
    use_viewer_port = viewer_port if viewer_port is not None else viewer_port_from_config(cfg)

    click.echo(
        "IMU mounting cal: optical + IMU only (ADS1299 off). "
        f"Plan '{plan.name}': {len(plan.all_phases)} SPACE phases "
        "(imu_mount_mta handheld figure-eight, then imu_mount_cra whole-head tumble). "
        "Keep tags visible; not nod-only."
    )

    if not no_doctor and not skip_preflight and record_preflight_enabled(cfg):
        # Preflight the config this take will actually use: --port and the
        # effective viewer switch included (it probed the YAML port before).
        import copy

        pre_cfg = copy.deepcopy(cfg)
        if port:
            pre_cfg.setdefault("hat", {})["serial_port"] = port
        pre_cfg.setdefault("host", {}).setdefault("viewer", {})["enabled"] = bool(use_viewer)
        issues = run_preflight(
            pre_cfg,
            check_serial=not mock,
            check_mcu_selftest=not mock,
        )
        errors, warnings, infos = partition_preflight_issues(issues)
        for w in warnings:
            click.echo(f"warning: {w}")
        for info in infos:
            click.echo(info)
        if errors:
            for err in errors:
                click.echo(f"error: {err}", err=True)
            raise SystemExit(1)

    src_config = config_path if config_path is not None else default_config_path()
    reproduce = RecordReproduceSpec(
        argv=list(sys.argv),
        config_path=str(src_config) if src_config is not None else None,
        port=port,
        mock=mock,
        skip_apply=skip_apply,
        skip_preflight=skip_preflight or no_doctor,
        no_integrity=no_integrity,
        viewer=viewer,
        viewer_host=viewer_host,
        viewer_port=viewer_port,
        emg_check=False,
        optical_track=True,
        optical_config=str(opt_cfg) if opt_cfg is not None else None,
        experiment_plan=None,
        use_experiment_plan=False,
        mvc=False,
        write_jts_config=False,
        imu_calibrate=True,
        protocol="imu_mounting",
        persist_imu_calibration=persist,
        imu_calibration_recordings_dir=str(recordings_dir) if recordings_dir else None,
        subject=subject_id,
        ct_calibration=False,
        persist_ct_calibration=False,
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
            optical_track=True,
            optical_config=Path(opt_cfg) if opt_cfg is not None else None,
            # Dedicated imu_mounting.json only — ignore session.experiment / YAML plan.
            experiment_plan=plan,
            use_experiment_plan=False,
            mvc=False,
            write_jts_config=False,
            enable_emg=False,
            ct_calibration=False,
            persist_ct_calibration=False,
            meta={"protocol": "imu_mounting"},
            reproduce=reproduce,
        )
    except Exception as exc:
        click.echo(f"imu calibrate failed: {exc}", err=True)
        raise SystemExit(1) from exc

    click.echo(f"Wrote session -> {out}")

    if not no_integrity and record_integrity_enabled(cfg):
        from jtsmm.integrity import check_session, format_session_summary
        from jtsmm.integrity.summary import SessionIntegritySummary, set_hdf5, write_session_integrity

        report = check_session(out)
        summary = SessionIntegritySummary(session_dir=out.parent)
        set_hdf5(summary, report)
        write_session_integrity(out.parent, summary)
        click.echo(format_session_summary(summary))
        if not report.ok:
            raise SystemExit(2)

    records: dict[str, dict[str, Any]] = {}
    code = 0
    if not fit:
        click.echo("skipping fit (--no-fit); run jtsmm imu calibrate --mode fit on the session")
        for slot in (("imu_mta", "imu_cra") if which == "both" else (f"imu_{which}",)):
            records[slot] = slot_record_from_error(slot, "fit skipped (--no-fit)")
    else:
        # --output applies here too (it was dropped: the fit always wrote into
        # the fresh session file); persist the file that holds /calibration.
        code, records = _fit_session(out, which, max_rms_deg, output_path)
        _update_stable(config_path, records, output_path or out)

    _persist_take(
        session_h5=(output_path or out) if fit else out,
        slots=records,
        persist=persist,
        recordings_dir=recordings_dir,
        subject_id=subject_id,
        run=run,
        optical_config=persist_optical_config or (str(opt_cfg) if opt_cfg else None),
        max_rms_deg=max_rms_deg,
    )
    if code:
        raise SystemExit(code)
