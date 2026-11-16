# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Record session orchestration."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from jtsmm.config import (
    apply_emg_overrides,
    apply_mcu,
    load,
    open_hat_serial,
    read_config_dump,
    stop_mcu_streams,
    stop_mcu_streams_robust,
)
from jtsmm.config.registry import (
    capture_duration_s,
    record_mcu_apply_enabled,
    session_ct_calibration,
    session_ct_calibration_n_poses,
    session_ct_calibration_recordings_dir,
    session_ct_tip_calibration,
    session_experiment,
    session_experiment_plan,
    session_meta_from_config,
    session_mvc,
    session_optical_config,
    session_optical_track,
    effective_optical_track_hz,
    session_persist_ct_calibration,
    session_use_experiment_plan,
    session_write_jts_config,
)
from jtsmm.io.session_layout import session_output_from_config
from jtsmm.io.session_reproduce import RecordReproduceSpec

from .serial_util import find_hat_port
from .supervisor import SessionSupervisor


def session_output_path(resolved: dict[str, Any]) -> Path:
    """Legacy single-file path; prefer session_output_from_config for new sessions."""
    _, h5 = session_output_from_config(resolved)
    return h5


def _teardown_mcu_streams(port: str | None) -> None:
    if not port:
        return
    try:
        stop_mcu_streams_robust(str(port))
    except Exception as exc:  # noqa: BLE001 - teardown must go on, but visibly
        print(
            f"WARNING: could not stop MCU streams on {port} ({exc}); the HAT may still "
            "be streaming -- run `jtsmm trigger stop` / power-cycle before the next take",
            file=sys.stderr,
        )


def _compact_json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def imu_streams_summary(cfg: dict[str, Any]) -> str:
    """``mta: game_rotation_vector@390 accelerometer@250 | cra: off`` from a
    `config dump` (what the MCU runs) or an apply payload."""
    parts = []
    for slot in ("imu_mta", "imu_cra"):
        block = cfg.get(slot) or {}
        reports = block.get("reports") or {}
        on = [f"{n}@{int(c.get('rate_hz', 0))}" for n, c in reports.items()
              if isinstance(c, dict) and c.get("enabled", True)]
        state = " ".join(on) if block.get("enabled", True) and on else "off"
        parts.append(f"{slot[4:]}: {state}")
    return " | ".join(parts)


def apply_or_snapshot_mcu_config(
    port: str,
    resolved: dict[str, Any],
    *,
    skip_reason: str | None,
    persist: bool = False,
) -> dict[str, str]:
    """Apply the session config to the MCU, or record why not. Returns /meta attrs.

    Streams are stopped first: a previous take or viewer that did not shut down
    cleanly can leave the ADS1299 streaming, and `config apply` into a live
    stream races the DRDY-driven reads on the shared SPI bus. When the apply is skipped
    the file would otherwise claim the host's config with nothing to show the
    MCU may run something else, so a read-only `config dump` snapshot is kept.
    """
    meta: dict[str, str] = {}
    if skip_reason is not None:
        meta["mcu_apply_skipped"] = "true"
        meta["mcu_apply_skip_reason"] = skip_reason
        print(
            f"note: MCU config apply skipped ({skip_reason}); the MCU keeps whatever "
            "config it already runs -- /meta/mcu_config_json is the HOST's config",
            file=sys.stderr,
        )
        dump = None
        try:
            with open_hat_serial(str(port)) as ser:
                dump = read_config_dump(ser)
        except Exception as exc:  # noqa: BLE001 - a snapshot must not block the take
            print(f"warning: could not read `config dump` ({exc})", file=sys.stderr)
        if dump is not None:
            meta["mcu_config_dump_json"] = _compact_json(dump)
        else:
            print(
                "warning: no `config dump` snapshot -- the MCU's actual config is not "
                "recorded for this session",
                file=sys.stderr,
            )
        return meta

    report: dict[str, Any] = {}
    ser = open_hat_serial(str(port))
    try:
        stop_mcu_streams(ser)
        if not apply_mcu(ser, resolved, persist=persist, report=report):
            raise RuntimeError(
                "MCU config apply did not complete "
                f"({report.get('line') or report.get('result', 'unknown')})"
            )
        # Whether the lead-off comparators run (CONFIG4), asked while the streams
        # are still stopped (ads_status is refused during streaming). The
        # in-recording viewer trusts the comparator bits only with this answer,
        # as `jtsmm view` does; without it the lead-off LEDs stayed "unknown" and
        # never showed a connected electrode (bench, 2026-09-28).
        from jtsmm.emg.capture import query_comparators_enabled

        cmp_on = query_comparators_enabled(ser)
        meta["emg_comparators_enabled"] = (
            "unknown" if cmp_on is None else ("true" if cmp_on else "false"))
    finally:
        ser.close()
    meta["mcu_apply_skipped"] = "false"
    if report.get("config_dump") is not None:
        meta["mcu_config_dump_json"] = _compact_json(report["config_dump"])
        print(f"MCU IMU reports -- {imu_streams_summary(report['config_dump'])}", flush=True)
    if report.get("verify_problems"):
        meta["mcu_config_verify_problems"] = _compact_json(report["verify_problems"])
    return meta


def resolve_session_config(
    config_path: Path | str | None = None,
    *,
    bias_drive: bool | None = None,
    bias_channels: str | None = None,
    active_channels: str | None = None,
    leadoff: bool | None = None,
    default_gain: int | None = None,
) -> dict[str, Any]:
    """Load + apply CLI overrides — the exact resolved config `run_session`
    (and thus the MCU) will use for a recording.

    Exposed separately so callers can run hardware checks (e.g. the
    automatic pre-recording ``jtsmm emg check`` in ``record_cmd``) against
    the *same* effective config before capture starts, instead of
    re-deriving it (and risking drift from what actually gets applied).
    """
    resolved = load(config_path)
    apply_emg_overrides(
        resolved,
        bias_drive=bias_drive,
        bias_channels=bias_channels,
        active_channels=active_channels,
        leadoff=leadoff,
        default_gain=default_gain,
        monitoring=False,
    )
    return resolved


def load_experiment_plan(
    resolved: dict[str, Any],
    *,
    use_plan: bool | None = None,
    plan_path: Path | str | None = None,
) -> Any | None:
    """Resolve the motion-phase plan for ``jtsmm record``.

    Enablement (first match):
    - explicit ``use_plan`` CLI flag
    - a CLI ``plan_path`` implies on (unless ``use_plan`` is False)
    - ``session.use_experiment_plan`` (legacy: a non-empty ``experiment_plan`` path)

    Source when enabled (first match):
    - ``plan_path`` / ``session.experiment_plan`` file (JSON or optical.yaml)
    - inline ``session.experiment``
    - ``experiment:`` inside ``session.optical_config``
    """
    if use_plan is None:
        enabled = True if plan_path is not None else session_use_experiment_plan(resolved)
    else:
        enabled = bool(use_plan)
    if not enabled:
        return None

    from optical.tracking.experiment import ExperimentPlan

    path = plan_path if plan_path is not None else session_experiment_plan(resolved)
    if path is not None:
        return ExperimentPlan.from_json(path)

    inline = session_experiment(resolved)
    if inline is not None:
        return ExperimentPlan.from_dict(inline)

    opt_cfg = session_optical_config(resolved)
    if opt_cfg is not None:
        try:
            from optical.config.loader import experiment_section, is_monolithic, load_config_file

            doc = load_config_file(opt_cfg)
            if is_monolithic(doc):
                return ExperimentPlan.from_dict(experiment_section(doc))
        except Exception:
            pass

    raise RuntimeError(
        "session.use_experiment_plan is true but no plan found "
        "(set session.experiment, session.experiment_plan, or --experiment-plan; "
        "MVC is jtsmm record --mvc)"
    )


def load_mvc_plan(resolved: dict[str, Any]) -> Any:
    """EMG MVC plan; folder/meta name comes from ``session.experiment.name`` when set."""
    from optical.tracking.experiment import mvc_experiment_plan
    from jtsmm.io.session_layout import experiment_name_from_config

    base = experiment_name_from_config(resolved)
    if not base:
        return mvc_experiment_plan()
    key = base.strip()
    if not key.lower().endswith("_mvc"):
        key = f"{key}_mvc"
    return mvc_experiment_plan(name=key)


def run_session(
    config_path: Path | str | None = None,
    *,
    duration_s: float | None = None,
    port: str | None = None,
    optical_track: bool | None = None,
    optical_config: Path | str | None = None,
    experiment_plan: Any | None = None,
    experiment_plan_path: Path | str | None = None,
    use_experiment_plan: bool | None = None,
    mvc: bool | None = None,
    write_jts_config: bool | None = None,
    jts_config_path: Path | str | None = None,
    ct_calibration: bool | None = None,
    ct_calibration_n_poses: int | None = None,
    ct_tip_calibration: Path | str | None = None,
    persist_ct_calibration: bool | None = None,
    ct_calibration_recordings_dir: Path | str | None = None,
    ct_calibration_only: bool = False,
    skip_mcu_apply: bool = False,
    mock_only: bool = False,
    persist_config: bool = False,
    viewer: bool = False,
    viewer_host: str = "0.0.0.0",
    viewer_port: int = 8770,
    enable_emg: bool = True,
    bias_drive: bool | None = None,
    bias_channels: str | None = None,
    active_channels: str | None = None,
    leadoff: bool | None = None,
    default_gain: int | None = None,
    meta: dict[str, str] | None = None,
    reproduce: RecordReproduceSpec | None = None,
) -> Path:
    """Load config, apply MCU settings, capture multimodal data to HDF5.

    Session YAML defaults (``session.optical_track``, ``session.use_experiment_plan``,
    ``session.mvc``, ``session.experiment``, ``session.ct_calibration``, …) apply when the
    corresponding CLI argument is omitted. Explicit CLI values always win.
    """
    resolved = resolve_session_config(
        config_path,
        bias_drive=bias_drive,
        bias_channels=bias_channels,
        active_channels=active_channels,
        leadoff=leadoff,
        default_gain=default_gain,
    )
    yaml_meta = session_meta_from_config(resolved)
    merged_meta = dict(yaml_meta)
    if meta:
        merged_meta.update(meta)

    if reproduce is not None and reproduce.config_path is None and config_path is not None:
        reproduce.config_path = str(Path(config_path))

    use_optical_track = (
        session_optical_track(resolved) if optical_track is None else bool(optical_track)
    )
    use_ct_cal = (
        session_ct_calibration(resolved) if ct_calibration is None else bool(ct_calibration)
    )
    n_poses = (
        session_ct_calibration_n_poses(resolved)
        if ct_calibration_n_poses is None
        else int(ct_calibration_n_poses)
    )
    tip_cfg = ct_tip_calibration
    if tip_cfg is None:
        tip_cfg = session_ct_tip_calibration(resolved)
    tip_path = Path(tip_cfg) if tip_cfg is not None else None
    persist_ct = (
        session_persist_ct_calibration(resolved)
        if persist_ct_calibration is None
        else bool(persist_ct_calibration)
    )
    rec_dir_cfg = ct_calibration_recordings_dir
    if rec_dir_cfg is None:
        rec_dir_cfg = session_ct_calibration_recordings_dir(resolved)
    calib_only = bool(ct_calibration_only)
    if calib_only:
        use_ct_cal = True
    opt_cfg = optical_config
    if opt_cfg is None:
        opt_cfg = session_optical_config(resolved)
    optical_cfg_path = Path(opt_cfg) if opt_cfg is not None else None

    use_write_jts = (
        session_write_jts_config(resolved) if write_jts_config is None else bool(write_jts_config)
    )
    jts_path = Path(jts_config_path) if jts_config_path is not None else None

    plan_obj = experiment_plan
    use_mvc = session_mvc(resolved) if mvc is None else bool(mvc)
    if calib_only:
        plan_obj = None
        use_optical_track = False
        use_write_jts = False
        use_mvc = False
        merged_meta["recording_mode"] = "ct_calibration"
    elif use_mvc:
        plan_obj = load_mvc_plan(resolved)
        use_optical_track = False
        use_ct_cal = False
        use_write_jts = False
        merged_meta["recording_mode"] = "mvc"
    elif plan_obj is None:
        plan_obj = load_experiment_plan(
            resolved,
            use_plan=use_experiment_plan,
            plan_path=experiment_plan_path,
        )

    duration = capture_duration_s(
        resolved, cli_duration=duration_s, plan_active=plan_obj is not None
    )
    if calib_only:
        duration = None

    if use_ct_cal and optical_cfg_path is None:
        raise RuntimeError("session.ct_calibration requires session.optical_config / --optical-config")
    plan_needs_optical = plan_obj is None or bool(getattr(plan_obj, "needs_optical", True))
    if use_optical_track and plan_needs_optical and optical_cfg_path is None:
        raise RuntimeError("optical_track requires session.optical_config / --optical-config")

    # Live feedback can run slower than the cameras. Tip pivot and IMU mounting
    # need a pose on every pair, so they ignore session.optical_track_hz.
    track_hz = effective_optical_track_hz(
        resolved,
        ct_calibration=use_ct_cal,
        protocol=str(merged_meta.get("protocol") or "") or None,
    )

    supervisor_kw = dict(
        viewer=viewer,
        viewer_host=viewer_host,
        viewer_port=viewer_port,
        extra_meta=merged_meta,
        reproduce=reproduce,
        optical_track=use_optical_track,
        optical_config=optical_cfg_path,
        optical_track_hz=track_hz,
        experiment_plan=plan_obj,
        write_jts_config=use_write_jts,
        jts_config_path=jts_path,
        ct_calibration=use_ct_cal,
        ct_calibration_n_poses=n_poses,
        ct_tip_calibration=tip_path,
        persist_ct_calibration=persist_ct,
        ct_calibration_recordings_dir=rec_dir_cfg,
        ct_calibration_only=calib_only,
        enable_emg=enable_emg and not calib_only,
    )

    if mock_only:
        sup = SessionSupervisor(resolved, port or "", **supervisor_kw)
        return sup.run(duration, mock_only=True, streams_running=False)

    serial_port = find_hat_port(port or (resolved.get("hat") or {}).get("serial_port"))
    skip_reason = None
    if skip_mcu_apply:
        skip_reason = "--skip-apply"
    elif not record_mcu_apply_enabled(resolved):
        skip_reason = "host.record.mcu_apply.enabled=false"
    apply_meta = apply_or_snapshot_mcu_config(
        str(serial_port), resolved, skip_reason=skip_reason, persist=persist_config
    )
    supervisor_kw["extra_meta"] = {**merged_meta, **apply_meta}
    sup = SessionSupervisor(resolved, serial_port, **supervisor_kw)
    try:
        return sup.run(duration, mock_only=False, streams_running=False)
    finally:
        _teardown_mcu_streams(str(serial_port))


def json_dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))
