# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""jtsmm record — multimodal capture CLI."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import click

from jtsmm.capture.session import load_experiment_plan, load_mvc_plan, resolve_session_config, run_session
from jtsmm.config.catalog import EMG_GAINS
from jtsmm.config.loader import default_config_path
from jtsmm.config.registry import (
    record_abort_on_error,
    record_emg_check_block_on_contact,
    record_emg_check_enabled,
    record_integrity_enabled,
    record_mcu_apply_enabled,
    record_preflight_enabled,
    session_meta_from_config,
    viewer_bind_host,
    viewer_enabled,
    viewer_port as viewer_port_from_config,
)
from jtsmm.doctor import run_preflight
from jtsmm.doctor.preflight_issues import partition_preflight_issues
from jtsmm.io.session_reproduce import RecordReproduceSpec

# Attrs written unconditionally by the HDF5 writers, plus the two dedicated
# `--subject`/`--protocol` flags below; `--meta` may not clobber any of these
# (custom keys must not collide with the fixed /meta schema, s02 §8.1).
_RESERVED_META_KEYS = frozenset(
    {
        "software_version",
        "schema_version",
        "jts_schema_version",
        "created_unix_utc",
        "recording_date",
        "config_hash",
        "mcu_config_json",
        "host_config_json",
        "firmware_version",
        "session_duration_s",
        "file_hashes_json",
        "subject_id",
        "protocol",
        "emg_check_json",
        "emg_check_passed",
        "mcu_apply_skipped",
        "mcu_apply_skip_reason",
        "mcu_config_dump_json",
        "mcu_config_verify_problems",
        "mcu_config_restored_after_emg_check",
        "capture_duration_s",
        "session_duration_kind",
    }
)


def _check_session_with_config(check_session, path: Path, resolved: dict, source: Path):
    """Grade ``path`` against the config this take actually used (CLI overrides included).

    Without it the checker fell back to whatever `load()` found, ignoring
    `--config X` and every override flag.
    """
    return check_session(path, resolved_config=resolved, config_source=str(source))


def _restore_mcu_config_after_check(cfg: dict, port: str | None, *, apply_off: bool, meta: dict) -> None:
    """Put the recording config (CLI overrides included) back after `emg check`.

    The check leaves the MCU in its last probe config (DRL isolation: CH1-CH6
    powered down, 250 SPS). This used to run only with config apply enabled, so
    with `host.record.mcu_apply.enabled: false` the take was recorded in the
    probe config. The check has already replaced whatever the MCU ran, so there
    is nothing left to preserve: always restore, and say so when apply is off.
    """
    from jtsmm.capture.serial_util import find_hat_port
    from jtsmm.config import apply_mcu, open_hat_serial, stop_mcu_streams

    if apply_off:
        click.echo(
            "note: restoring the recording MCU config after the EMG check although "
            "config apply is off -- the check replaced the MCU's config",
            err=True,
        )
        meta["mcu_config_restored_after_emg_check"] = "true"
    serial_port = find_hat_port(port or (cfg.get("hat") or {}).get("serial_port"))
    with open_hat_serial(str(serial_port)) as ser:
        stop_mcu_streams(ser)
        if not apply_mcu(ser, cfg, persist=False):
            click.echo("error: failed to restore production MCU config after EMG check", err=True)
            raise SystemExit(1)


def _parse_meta(
    pairs: tuple[str, ...], *, subject: str | None = None, protocol: str | None = None
) -> dict[str, str]:
    """Parse repeated ``--meta key=value`` options (plus ``--subject``/``--protocol``) into a dict."""
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise click.BadParameter(f"--meta expects KEY=VALUE, got {pair!r}")
        key, _, value = pair.partition("=")
        key = key.strip()
        if not key:
            raise click.BadParameter(f"--meta key must not be empty, got {pair!r}")
        if key in _RESERVED_META_KEYS:
            raise click.BadParameter(
                f"--meta key {key!r} is reserved; use --subject/--protocol for it"
            )
        out[key] = value
    if subject is not None:
        out["subject_id"] = subject
    if protocol is not None:
        out["protocol"] = protocol
    return out


@click.command("record")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--duration",
    "duration_s",
    type=float,
    default=None,
    help=(
        "Override session.duration_s. Timed captures only: ignored for an "
        "experiment-plan session unless set, in which case it is a safety timeout."
    ),
)
@click.option("--port", default=None, help="Serial port override")
@click.option(
    "--optical-track/--no-optical-track",
    default=None,
    help="ArUco optical tracking into /optical (default: session.optical_track)",
)
@click.option(
    "--optical-config",
    type=click.Path(path_type=Path),
    default=None,
    help="optical.yaml (default: session.optical_config)",
)
@click.option(
    "--ct-calibration/--no-ct-calibration",
    default=None,
    help="CT tip pivot calib (cone + SPACE×N OLS) before record (default: session.ct_calibration)",
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
    help=(
        "Reusable tip-offset JSON (default: session.ct_tip_calibration). "
        "Written when calibrating; loaded when --no-ct-calibration."
    ),
)
@click.option(
    "--persist-ct-calibration/--no-persist-ct-calibration",
    default=None,
    help=(
        "Save CT tip calib poses + camera frames under "
        "session.ct_calibration_recordings_dir (default: session.persist_ct_calibration)"
    ),
)
@click.option(
    "--ct-calibration-recordings-dir",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "Directory template for persisted CT tip calib recordings "
        "(default: session.ct_calibration_recordings_dir; "
        "{experiment_name} is substituted)"
    ),
)
@click.option(
    "--mvc/--no-mvc",
    default=None,
    help=(
        "EMG-only MVC take (mvc_opening + mvc_closing SPACE). Independent of "
        "session.experiment. Implies no optical track, no CT calib, no JTS sidecar. "
        "Cannot combine with --use-experiment-plan / --experiment-plan. "
        "Default: session.mvc."
    ),
)
@click.option(
    "--use-experiment-plan/--no-use-experiment-plan",
    default=None,
    help=(
        "Walk SPACE phases from session.experiment (default: session.use_experiment_plan). "
        "--experiment-plan PATH implies on unless --no-use-experiment-plan."
    ),
)
@click.option(
    "--experiment-plan",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help=(
        "Experiment plan JSON or optical.yaml (motion phases). Overrides "
        "session.experiment. Interactive SPACE marks phase start/end on "
        "t_master_us while recording continues. Landmark calibration is "
        "inserted before motion. MVC is `jtsmm record --mvc`, not a plan name. "
        "Default: session.experiment_plan path, "
        "else session.experiment."
    ),
)
@click.option(
    "--write-jts-config/--no-write-jts-config",
    default=None,
    help="Write JTS config from phase intervals when a plan is walked (default: session.write_jts_config)",
)
@click.option(
    "--jts-config",
    "jts_config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for JTS config (default: <session_dir>/<plan_slug>_jts_config.json)",
)
@click.option("--mock", is_flag=True, help="Skip MCU serial; mock cameras only")
@click.option("--skip-apply", is_flag=True, help="Do not send config apply to MCU")
@click.option("--skip-preflight", is_flag=True, help="Skip preflight checks")
@click.option("--no-doctor", is_flag=True, help="Skip preflight checks (alias)")
@click.option("--no-integrity", is_flag=True, help="Skip post-session integrity check")
@click.option("--viewer/--no-viewer", default=None, help="Serve a live web viewer during recording")
@click.option("--viewer-host", default=None, help="Viewer bind host (default: host.viewer.bind_host or 0.0.0.0)")
@click.option("--viewer-port", type=int, default=None, help="Viewer HTTP/WS port (default: host.viewer.port or 8770)")
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
        "Enable/disable DC lead-off comparator contact detection while recording "
        "(default: on; it's DC-only and doesn't touch the EMG band, see docs/hardware/h03)"
    ),
)
@click.option(
    "--gain",
    "default_gain",
    type=click.Choice([str(g) for g in EMG_GAINS]),
    default=None,
    help=f"ADS1299 PGA gain for all patient channels, one of {EMG_GAINS} (overrides config)",
)
@click.option("--subject", "subject", default=None, help="Subject id/hash, written to /meta/subject_id")
@click.option("--protocol", "protocol", default=None, help="Protocol name, written to /meta/protocol")
@click.option(
    "--meta",
    "meta_kv",
    multiple=True,
    metavar="KEY=VALUE",
    help="Custom metadata written to the HDF5 /meta group, e.g. --meta operator=jd (repeatable)",
)
@click.option(
    "--emg-check/--no-emg-check",
    "emg_check",
    default=True,
    help=(
        "Run 'jtsmm emg check' (lead-off + noise/line + isolated DRL probe, "
        "~15 s) before recording; skipped automatically for --mock. Result "
        "is stored at /meta/emg_check_json + /meta/emg_check_passed (default: on)"
    ),
)
@click.option(
    "--no-abort",
    is_flag=True,
    help=(
        "Do not stop on a quality gate (preflight errors, EMG contact, integrity "
        "verdict) -- report and carry on. Same as "
        "host.record.abort_on_error: false. Operational failures still abort."
    ),
)
@click.option(
    "--force-emg-check",
    is_flag=True,
    help=(
        "Record even when the EMG check finds a hard contact failure "
        "(disconnected / railed channel, or |Z| > 50 kOhm). Noise and line-hum "
        "findings never block regardless of this flag."
    ),
)
def record_cmd(
    config_path: Path | None,
    duration_s: float | None,
    port: str | None,
    optical_track: bool | None,
    optical_config: Path | None,
    ct_calibration: bool | None,
    ct_calibration_n_poses: int | None,
    ct_tip_calibration: Path | None,
    persist_ct_calibration: bool | None,
    ct_calibration_recordings_dir: Path | None,
    mvc: bool | None,
    use_experiment_plan: bool | None,
    experiment_plan: Path | None,
    write_jts_config: bool | None,
    jts_config_path: Path | None,
    mock: bool,
    skip_apply: bool,
    skip_preflight: bool,
    no_doctor: bool,
    no_integrity: bool,
    viewer: bool | None,
    viewer_host: str | None,
    viewer_port: int | None,
    bias_drive: bool | None,
    bias_channels: str | None,
    active_channels: str | None,
    leadoff: bool | None,
    default_gain: str | None,
    subject: str | None,
    protocol: str | None,
    meta_kv: tuple[str, ...],
    emg_check: bool,
    force_emg_check: bool,
    no_abort: bool,
) -> None:
    """Record a multimodal session to HDF5."""
    try:
        cfg_for_viewer = resolve_session_config(
            config_path,
            bias_drive=bias_drive,
            bias_channels=bias_channels,
            active_channels=active_channels,
            leadoff=leadoff,
            default_gain=int(default_gain) if default_gain is not None else None,
        )
    except ValueError as exc:
        # e.g. --no-bias-drive --bias-channels 1, or --active-channels 7
        raise click.BadParameter(str(exc)) from exc
    cli_meta = _parse_meta(meta_kv, subject=subject, protocol=protocol)
    yaml_meta = session_meta_from_config(cfg_for_viewer)
    meta = dict(yaml_meta)
    meta.update(cli_meta)
    use_viewer = viewer_enabled(cfg_for_viewer) if viewer is None else viewer
    use_viewer_host = viewer_host if viewer_host is not None else viewer_bind_host(cfg_for_viewer)
    use_viewer_port = viewer_port if viewer_port is not None else viewer_port_from_config(cfg_for_viewer)

    # Accumulated across preflight / EMG / HDF5 for the per-session integrity
    # summary file written next to session.h5 at the end of the take.
    from jtsmm.integrity.summary import (
        SessionIntegritySummary,
        set_emg_check,
        set_hdf5,
        set_preflight,
        write_session_integrity,
    )

    integrity_summary = SessionIntegritySummary(session_dir=Path("."))

    # Quality gates stop the run; operational failures always do. See
    # registry.record_abort_on_error for which is which.
    abort_on_error = record_abort_on_error(cfg_for_viewer) and not no_abort
    if not abort_on_error:
        click.echo(
            "note: quality gates will report but not stop this run "
            "(--no-abort / host.record.abort_on_error: false)",
            err=True,
        )

    if not no_doctor and not skip_preflight and record_preflight_enabled(cfg_for_viewer):
        # The take's own config (overrides applied) with --port folded in, so the
        # preflight probes the same serial port the recording will open.
        preflight_cfg = copy.deepcopy(cfg_for_viewer)
        if port:
            preflight_cfg.setdefault("hat", {})["serial_port"] = port
        issues = run_preflight(
            resolved=preflight_cfg,
            config_path=config_path,
            check_serial=not mock,
            check_mcu_selftest=not mock,
        )
        errors, warnings, infos = partition_preflight_issues(issues)
        set_preflight(integrity_summary, errors=errors, warnings=warnings, infos=infos)
        for w in warnings:
            click.echo(f"warning: {w}", err=True)
        for info in infos:
            click.echo(info)
        if errors:
            for err in errors:
                click.echo(f"error: {err}", err=True)
            if abort_on_error:
                raise SystemExit(1)
            click.echo(
                f"note: {len(errors)} preflight error(s) not aborting the run; "
                "they are recorded in integrity.json",
                err=True,
            )
    else:
        reason = (
            "--no-doctor/--skip-preflight"
            if (no_doctor or skip_preflight)
            else "host.record.preflight.enabled=false"
        )
        set_preflight(integrity_summary, skipped=True, skip_reason=reason)

    if emg_check and record_emg_check_enabled(cfg_for_viewer) and not mock and not skip_apply:
        from jtsmm.emg.check import (
            check_result_to_dict,
            format_check_table,
            hard_contact_failures,
            run_emg_check,
        )

        resolved_for_check = resolve_session_config(
            config_path,
            bias_drive=bias_drive,
            bias_channels=bias_channels,
            active_channels=active_channels,
            leadoff=leadoff,
            default_gain=int(default_gain) if default_gain is not None else None,
        )
        click.echo("Running pre-recording EMG check ('jtsmm emg check', ~15 s)...")
        check_result = run_emg_check(resolved_for_check, port=port)
        click.echo(format_check_table(check_result))
        # Split the verdict: bad electrode CONTACT is fixable here and now and
        # makes that muscle's data worthless, so it aborts by default. Noise and
        # line hum are environmental -- they still only warn, because blocking on
        # them would strand a subject mid-session for something re-prepping will
        # not fix.
        hard = hard_contact_failures(check_result)
        if hard and record_emg_check_block_on_contact(cfg_for_viewer) and not force_emg_check:
            click.echo(
                "error: EMG check found unusable electrode contact on: "
                + ", ".join(hard),
                err=True,
            )
            click.echo(
                "       Re-prep those electrodes (abrade + fresh gel, check the "
                "clip and its crimp, let the gel settle 3-5 min), then re-run.\n"
                "       Verify with: jtsmm emg impedance --current-na 24\n"
                "       To record anyway: jtsmm record --force-emg-check "
                "(or host.record.emg_check.block_on_contact_failure: false)",
                err=True,
            )
            set_emg_check(
                integrity_summary,
                result=check_result_to_dict(check_result),
                passed=False,
            )
            if abort_on_error:
                raise SystemExit(1)
            click.echo(
                "note: recording anyway with unusable contact on "
                f"{', '.join(hard)} (--no-abort)",
                err=True,
            )
        if not check_result.passed:
            detail = " (contact forced past: " + ", ".join(hard) + ")" if hard else ""
            click.echo(
                "warning: EMG check reported issues (see table above); "
                f"recording anyway{detail}",
                err=True,
            )
        emg_dict = check_result_to_dict(check_result)
        set_emg_check(integrity_summary, result=emg_dict, passed=check_result.passed)
        meta["emg_check_json"] = json.dumps(emg_dict, separators=(",", ":"))
        meta["emg_check_passed"] = "true" if check_result.passed else "false"
        # EMG check applies probe configs (250 SPS leadoff, DRL isolation, etc.).
        _restore_mcu_config_after_check(
            cfg_for_viewer,
            port,
            apply_off=not record_mcu_apply_enabled(cfg_for_viewer),
            meta=meta,
        )
    else:
        if mock:
            reason = "--mock (EMG check needs hardware)"
        elif skip_apply:
            # The check rewrites the MCU config (probe settings) and would then
            # have to push the YAML back -- exactly what --skip-apply forbids.
            reason = "--skip-apply (the check would change the MCU config)"
            if emg_check and record_emg_check_enabled(cfg_for_viewer):
                click.echo(
                    "warning: skipping the pre-recording EMG check because of --skip-apply",
                    err=True,
                )
        elif not emg_check:
            reason = "--no-emg-check"
        else:
            reason = "host.record.emg_check.enabled=false"
        set_emg_check(integrity_summary, skipped=True, skip_reason=reason)

    src_config = Path(config_path) if config_path is not None else default_config_path()
    from jtsmm.config.registry import (
        session_ct_calibration,
        session_ct_calibration_n_poses,
        session_ct_calibration_recordings_dir,
        session_ct_tip_calibration,
        session_experiment_plan,
        session_mvc,
        session_optical_config,
        session_optical_track,
        session_persist_ct_calibration,
        session_use_experiment_plan,
        session_write_jts_config,
    )

    try:
        use_mvc = session_mvc(cfg_for_viewer) if mvc is None else bool(mvc)
        if use_mvc and (use_experiment_plan is True or experiment_plan is not None):
            click.echo(
                "error: --mvc cannot be combined with --use-experiment-plan / --experiment-plan",
                err=True,
            )
            raise SystemExit(1)
        if use_mvc:
            if use_experiment_plan is None and session_use_experiment_plan(cfg_for_viewer):
                click.echo(
                    "warning: --mvc ignores session.experiment motion phases; "
                    "the experiment name is still used for the session folder. "
                    "Record without --mvc to walk landmarks + motion",
                    err=True,
                )
            plan_obj = load_mvc_plan(cfg_for_viewer)
        else:
            plan_obj = load_experiment_plan(
                cfg_for_viewer,
                use_plan=use_experiment_plan,
                plan_path=experiment_plan,
            )
    except ImportError as exc:
        click.echo(
            "error: experiment_plan requires the optical package "
            "(pip install -e software/optical)",
            err=True,
        )
        raise SystemExit(1) from exc
    except Exception as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from exc
    if plan_obj is not None:
        walked = plan_obj.all_phases
        n_mvc = sum(1 for p in walked if p.kind == "mvc")
        n_cal = sum(1 for p in walked if "calibration" in p.kind)
        n_mot = sum(1 for p in walked if p.kind == "experiment")
        bits = []
        if n_mvc and not n_mot and not n_cal:
            bits.append("MVC EMG-only")
        elif n_mvc:
            bits.append(f"{n_mvc} MVC")
        if n_cal:
            bits.append(f"{n_cal} landmark")
        if n_mot:
            bits.append(f"{n_mot} motion")
        click.echo(
            f"Experiment plan '{plan_obj.name}': "
            f"{len(walked)} phases ({', '.join(bits) or 'none'})"
        )
        if duration_s is not None and duration_s < 60:
            click.echo(
                f"warning: --duration {duration_s:g}s is short for interactive phases; "
                "duration is a safety timeout (SPACE walks the plan).",
                err=True,
            )

    if use_mvc:
        if optical_track or ct_calibration:
            click.echo(
                "warning: --mvc is EMG-only; ignoring --optical-track / --ct-calibration",
                err=True,
            )
        eff_optical_track = False
        eff_ct_cal = False
        eff_write_jts = False
        eff_use_plan = False
        plan_path = None
    else:
        eff_use_plan = plan_obj is not None
        plan_path = experiment_plan or session_experiment_plan(cfg_for_viewer)

        eff_optical_track = (
            session_optical_track(cfg_for_viewer) if optical_track is None else bool(optical_track)
        )
        eff_ct_cal = (
            session_ct_calibration(cfg_for_viewer) if ct_calibration is None else bool(ct_calibration)
        )
        eff_write_jts = (
            session_write_jts_config(cfg_for_viewer)
            if write_jts_config is None
            else bool(write_jts_config)
        )
    eff_n_poses = (
        session_ct_calibration_n_poses(cfg_for_viewer)
        if ct_calibration_n_poses is None
        else int(ct_calibration_n_poses)
    )
    eff_tip = ct_tip_calibration or session_ct_tip_calibration(cfg_for_viewer)
    eff_persist_ct = (
        session_persist_ct_calibration(cfg_for_viewer)
        if persist_ct_calibration is None
        else bool(persist_ct_calibration)
    )
    eff_rec_dir = (
        ct_calibration_recordings_dir
        or session_ct_calibration_recordings_dir(cfg_for_viewer)
    )
    eff_opt_cfg = optical_config or session_optical_config(cfg_for_viewer)

    reproduce = RecordReproduceSpec(
        argv=list(sys.argv),
        config_path=str(src_config),
        duration_s=duration_s,
        port=port,
        mock=mock,
        skip_apply=skip_apply,
        skip_preflight=skip_preflight or no_doctor,
        no_integrity=no_integrity,
        viewer=viewer,
        viewer_host=viewer_host,
        viewer_port=viewer_port,
        bias_drive=bias_drive,
        bias_channels=bias_channels,
        active_channels=active_channels,
        leadoff=leadoff,
        default_gain=int(default_gain) if default_gain is not None else None,
        subject=subject,
        protocol=protocol,
        meta_kv=tuple(meta_kv),
        emg_check=emg_check,
        force_emg_check=force_emg_check,
        no_abort=no_abort,
        optical_track=eff_optical_track,
        optical_config=str(eff_opt_cfg) if eff_opt_cfg is not None else None,
        ct_calibration=eff_ct_cal,
        ct_calibration_n_poses=eff_n_poses,
        ct_tip_calibration=str(eff_tip) if eff_tip is not None else None,
        persist_ct_calibration=eff_persist_ct,
        ct_calibration_recordings_dir=str(eff_rec_dir) if eff_rec_dir is not None else None,
        experiment_plan=str(plan_path) if plan_obj is not None and plan_path is not None else None,
        use_experiment_plan=eff_use_plan,
        mvc=use_mvc,
        write_jts_config=eff_write_jts,
        jts_config_path=str(jts_config_path) if jts_config_path is not None else None,
    )

    try:
        out = run_session(
            config_path,
            duration_s=duration_s,
            port=port,
            skip_mcu_apply=skip_apply,
            mock_only=mock,
            viewer=use_viewer,
            viewer_host=use_viewer_host,
            viewer_port=use_viewer_port,
            bias_drive=bias_drive,
            bias_channels=bias_channels,
            active_channels=active_channels,
            leadoff=leadoff,
            default_gain=int(default_gain) if default_gain is not None else None,
            optical_track=eff_optical_track,
            optical_config=Path(eff_opt_cfg) if eff_opt_cfg is not None else None,
            experiment_plan=plan_obj,
            use_experiment_plan=eff_use_plan,
            mvc=use_mvc,
            write_jts_config=eff_write_jts,
            jts_config_path=jts_config_path,
            ct_calibration=eff_ct_cal,
            ct_calibration_n_poses=eff_n_poses,
            ct_tip_calibration=Path(eff_tip) if eff_tip is not None else None,
            persist_ct_calibration=eff_persist_ct,
            ct_calibration_recordings_dir=eff_rec_dir,
            meta=meta,
            reproduce=reproduce,
        )
    except BaseException as exc:  # noqa: BLE001 - incl. Ctrl-C / SIGTERM during setup
        if isinstance(exc, (SystemExit, GeneratorExit)):
            raise
        # The supervisor closes the file on every exit and tags the exception
        # with its path; grade that partial file so the session dir still gets
        # an integrity.json that says what happened.
        interrupted = isinstance(exc, KeyboardInterrupt)
        # The supervisor tags where it failed; anything untagged happened
        # before it ran (config, apply, doctor), i.e. during setup.
        when = {
            "take": "during the take",
            "finish": "while finishing the take",
        }.get(getattr(exc, "jtsmm_phase", "setup"), "during setup")
        if interrupted:
            click.echo(f"record aborted (Ctrl-C {when})", err=True)
        elif not isinstance(exc, Exception):
            click.echo(f"record stopped by {exc} {when}", err=True)
        else:
            click.echo(f"record failed: {exc} ({when})", err=True)
        partial = getattr(exc, "jtsmm_session_h5", None)
        if partial is not None and Path(partial).is_file():
            click.echo(f"Partial session (closed) -> {partial}", err=True)
            if not no_integrity and record_integrity_enabled(cfg_for_viewer):
                try:
                    from jtsmm.integrity import check_session

                    integrity_summary.session_dir = Path(partial).parent
                    report = _check_session_with_config(
                        check_session, Path(partial), cfg_for_viewer, src_config
                    )
                    report.add_error(
                        "recording aborted: "
                        + (f"Ctrl-C {when}" if interrupted
                           else f"{type(exc).__name__}: {exc} ({when})")
                    )
                    set_hdf5(integrity_summary, report)
                    written = write_session_integrity(Path(partial).parent, integrity_summary)
                    click.echo(f"Wrote integrity summary -> {written['integrity.json']}", err=True)
                except Exception as exc2:  # noqa: BLE001 - never mask the original failure
                    click.echo(f"warning: integrity check of the partial file failed ({exc2})", err=True)
        raise SystemExit(130 if interrupted else 1) from exc

    click.echo(f"Wrote session -> {out}")
    reproduce_sh = out.parent / "reproduce.sh"
    if reproduce_sh.is_file():
        click.echo(f"Reproduce with -> {reproduce_sh}")

    integrity_summary.session_dir = out.parent
    if not no_integrity and record_integrity_enabled(cfg_for_viewer):
        from jtsmm.integrity import check_session, format_session_summary

        report = _check_session_with_config(check_session, out, cfg_for_viewer, src_config)
        set_hdf5(integrity_summary, report)
        written = write_session_integrity(out.parent, integrity_summary)
        click.echo("")
        click.echo(format_session_summary(integrity_summary))
        click.echo(f"Wrote integrity summary -> {written['integrity.json']}")
        if not report.ok:
            if abort_on_error:
                raise SystemExit(2)
            click.echo(
                f"note: integrity reported {len(report.issues)} error(s); not "
                "aborting (--no-abort). The session file and integrity.json are "
                "written either way",
                err=True,
            )
    else:
        reason = (
            "--no-integrity"
            if no_integrity
            else "host.record.integrity.enabled=false"
        )
        integrity_summary.hdf5.ran = True
        integrity_summary.hdf5.skipped = True
        integrity_summary.hdf5.skip_reason = reason
        # Still write a partial summary so the session dir always has one.
        written = write_session_integrity(out.parent, integrity_summary)
        click.echo(f"Wrote integrity summary -> {written['integrity.json']} ({reason})")


if __name__ == "__main__":
    record_cmd.main(standalone_mode=True)
