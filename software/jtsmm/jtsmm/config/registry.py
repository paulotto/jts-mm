# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Typed accessors for host.* and session.* configuration keys."""

from __future__ import annotations

import os
from typing import Any

from .loader import resolve_config_path

_RESERVED_SESSION_META_KEYS = frozenset(
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
    }
)


def _host(resolved: dict[str, Any]) -> dict[str, Any]:
    return resolved.get("host") or {}


def _session(resolved: dict[str, Any]) -> dict[str, Any]:
    return resolved.get("session") or {}


def _env_bool(name: str) -> bool | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return raw.lower() not in ("0", "false", "no", "")


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    return int(raw)


def capture_settle_ms(resolved: dict[str, Any]) -> int:
    if (v := _env_int("JTSMM_SETTLE_MS")) is not None:
        return max(0, v)
    capture = _host(resolved).get("capture") or {}
    return max(0, int(capture.get("settle_ms", 500)))


def capture_stream_restart(resolved: dict[str, Any]) -> bool:
    if (v := _env_bool("JTSMM_STREAM_RESTART")) is not None:
        return v
    capture = _host(resolved).get("capture") or {}
    return bool(capture.get("stream_restart", True))


def capture_trigger_after_cameras(resolved: dict[str, Any]) -> bool:
    """Start the MCU camera trigger only after picamera2 is streaming (default on).

    Off reproduces the pre-2026-09 order (trigger first) for comparison; see
    ``start_mcu_trigger`` for why that mis-pairs camera frames.
    """
    if (v := _env_bool("JTSMM_TRIGGER_AFTER_CAMERAS")) is not None:
        return v
    capture = _host(resolved).get("capture") or {}
    return bool(capture.get("trigger_after_cameras", True))


def capture_camera_pps_anchor(resolved: dict[str, Any]) -> bool:
    """Anchor each camera's first frame on the trigger nearest its PPS-mapped
    SensorTimestamp (default on) instead of on the oldest buffered trigger.

    Needs ``pps_samples > 0``; falls back to the old behaviour per camera when
    no fit is available. Off reproduces the pre-2026-09-18 association.
    """
    if (v := _env_bool("JTSMM_CAMERA_PPS_ANCHOR")) is not None:
        return v
    capture = _host(resolved).get("capture") or {}
    return bool(capture.get("camera_pps_anchor", True))


def capture_camera_delta_tracking(resolved: dict[str, Any]) -> bool:
    """Advance each camera's trigger cursor by its own elapsed frame periods.

    Default on. Off restores advance-by-one-per-delivered-frame, which mislabels
    everything after a camera drops frames (measured: 59 frames of spread).
    """
    if (v := _env_bool("JTSMM_CAMERA_DELTA_TRACKING")) is not None:
        return v
    capture = _host(resolved).get("capture") or {}
    return bool(capture.get("camera_delta_tracking", True))


def capture_pps_samples(resolved: dict[str, Any]) -> int:
    """Host<->MCU clock handshake samples per pass (0 disables)."""
    if (v := _env_int("JTSMM_PPS_SAMPLES")) is not None:
        return max(0, v)
    capture = _host(resolved).get("capture") or {}
    return max(0, int(capture.get("pps_samples", 8)))


def capture_native_prefer(resolved: dict[str, Any]) -> bool:
    flag = os.environ.get("JTSMM_NATIVE")
    if flag is not None:
        return flag not in ("0", "false", "False", "")
    capture = _host(resolved).get("capture") or {}
    return bool(capture.get("native_prefer", True))


def _config_path(value: Any) -> str | None:
    """Absolute form of a path from jtsmm.yaml (``~`` expanded, CWD-independent)."""
    if value is None or value == "":
        return None
    return str(resolve_config_path(str(value)))


def paths_data_dir(resolved: dict[str, Any]) -> str:
    """Session output root: JTSMM_DATA_DIR > host.paths.data_dir > host.storage.base_dir
    (deprecated alias) > ~/jtsmm_data."""
    if env := os.environ.get("JTSMM_DATA_DIR"):
        return os.path.expanduser(env)
    host = _host(resolved)
    paths = host.get("paths") or {}
    if paths.get("data_dir"):
        return str(_config_path(paths["data_dir"]))
    storage = host.get("storage") or {}
    return str(_config_path(storage.get("base_dir") or "~/jtsmm_data"))


def viewer_enabled(resolved: dict[str, Any]) -> bool:
    viewer = _host(resolved).get("viewer") or {}
    return bool(viewer.get("enabled", False))


def viewer_bind_host(resolved: dict[str, Any]) -> str:
    viewer = _host(resolved).get("viewer") or {}
    return str(viewer.get("bind_host", "0.0.0.0"))


def viewer_port(resolved: dict[str, Any]) -> int:
    viewer = _host(resolved).get("viewer") or {}
    return int(viewer.get("port", 8770))


def record_emg_check_enabled(resolved: dict[str, Any]) -> bool:
    record = _host(resolved).get("record") or {}
    emg_check = record.get("emg_check") or {}
    return bool(emg_check.get("enabled", True))


def integrity_errors_as_warnings(resolved: dict[str, Any]) -> bool:
    """Report integrity errors as warnings, so no session is ever RESULT: FAIL.

    For a run that must not be interrupted by a verdict. Demotion is recorded --
    banner, `would-fail:` prefixes, and the original errors in the report summary
    and integrity.json -- so a session passed this way stays distinguishable from
    one that passed on its merits.
    """
    return bool((_host(resolved).get("integrity") or {}).get("errors_as_warnings", False))


def record_abort_on_error(resolved: dict[str, Any]) -> bool:
    """Whether `jtsmm record` stops on a QUALITY gate.

    Covers the preflight, the pre-recording EMG contact check and the
    post-session integrity verdict. False turns all three into messages and
    lets the run proceed -- for a session that must not be interrupted.

    Operational failures still abort regardless: a failed MCU config restore
    (the ADC would still be in its probe configuration), a malformed experiment
    plan, bad CLI usage, or an exception from the capture itself. Those do not
    produce a worse recording, they produce no usable recording at all.
    """
    record = _host(resolved).get("record") or {}
    return bool(record.get("abort_on_error", True))


def integrity_thresholds(resolved: dict[str, Any]) -> dict[str, float]:
    """Raw ``host.integrity.thresholds`` block (empty when unset).

    The typed view, with defaults and warn/err validation, is
    ``jtsmm.integrity.checker.IntegrityThresholds``. These gates decide whether a
    finished recording is called usable; they are read after the file is closed,
    so they cannot affect sampling. Note this is separate from
    ``host.record.integrity.enabled``, which only says whether `jtsmm record`
    runs the check at all -- the thresholds apply wherever check_session runs.
    """
    block = (_host(resolved).get("integrity") or {}).get("thresholds")
    if not isinstance(block, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in block.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def record_emg_check_block_on_contact(resolved: dict[str, Any]) -> bool:
    """Whether a hard *contact* failure in the pre-recording EMG check aborts.

    Default true: an electrode that is disconnected, railed, or above
    CONTACT_Z_MAX_KOHM yields unusable data for that muscle and takes ~2 min to
    re-prep, so recording a full take first is pure waste. Noise/line failures
    never block (see hard_contact_failures). Set false -- or pass
    `--force-emg-check` -- to record anyway.
    """
    record = _host(resolved).get("record") or {}
    emg_check = record.get("emg_check") or {}
    return bool(emg_check.get("block_on_contact_failure", True))


def record_preflight_enabled(resolved: dict[str, Any]) -> bool:
    record = _host(resolved).get("record") or {}
    preflight = record.get("preflight") or {}
    return bool(preflight.get("enabled", True))


def record_integrity_enabled(resolved: dict[str, Any]) -> bool:
    record = _host(resolved).get("record") or {}
    integrity = record.get("integrity") or {}
    return bool(integrity.get("enabled", True))


def record_mcu_apply_enabled(resolved: dict[str, Any]) -> bool:
    record = _host(resolved).get("record") or {}
    mcu_apply = record.get("mcu_apply") or {}
    return bool(mcu_apply.get("enabled", True))


def emg_filter_mains_mode(resolved: dict[str, Any]) -> str:
    """How the viewer removes line interference: 'track', 'notch' or 'off'.

    'track' (default) fits and subtracts the tone and locks onto the real grid
    frequency, costing ~1 Hz of band per harmonic. 'notch' is the older biquad
    bank: ~5 Hz per harmonic, fixed at the nominal frequency. See
    docs/software/s09_mains_removal.md.
    """
    f = _host(resolved).get("emg_filter") or {}
    mode = str(f.get("mains_mode", "track") or "track")
    return mode if mode in ("track", "notch", "off") else "track"


def emg_check_line_hz(resolved: dict[str, Any]) -> float:
    emg_check = _host(resolved).get("emg_check") or {}
    return float(emg_check.get("line_hz", 50.0))


def emg_check_settle_ms(resolved: dict[str, Any]) -> int:
    emg_check = _host(resolved).get("emg_check") or {}
    return max(0, int(emg_check.get("settle_ms", 400)))


def emg_check_leadoff_duration_s(resolved: dict[str, Any]) -> float:
    emg_check = _host(resolved).get("emg_check") or {}
    leadoff = emg_check.get("leadoff") or {}
    return float(leadoff.get("duration_s", 5.0))


def emg_check_noise_duration_s(resolved: dict[str, Any]) -> float:
    emg_check = _host(resolved).get("emg_check") or {}
    noise = emg_check.get("noise") or {}
    return float(noise.get("duration_s", 5.0))


def emg_check_impedance_enabled(resolved: dict[str, Any]) -> bool:
    emg_check = _host(resolved).get("emg_check") or {}
    impedance = emg_check.get("impedance") or {}
    return bool(impedance.get("enabled", True))


def emg_check_impedance_duration_s(resolved: dict[str, Any]) -> float:
    emg_check = _host(resolved).get("emg_check") or {}
    impedance = emg_check.get("impedance") or {}
    return float(impedance.get("duration_s", 2.0))


def emg_check_drl_isolation_duration_s(resolved: dict[str, Any]) -> float:
    emg_check = _host(resolved).get("emg_check") or {}
    drl = emg_check.get("drl_isolation") or {}
    return float(drl.get("duration_s", 2.0))


def session_mode(resolved: dict[str, Any]) -> str:
    return str(_session(resolved).get("mode", "acquisition"))


def session_duration_s(resolved: dict[str, Any]) -> float | None:
    duration = _session(resolved).get("duration_s")
    if duration is None:
        return None
    return float(duration)


def session_default_duration_s(resolved: dict[str, Any]) -> float:
    return float(_session(resolved).get("default_duration_s", 10.0))


def session_subject_id(resolved: dict[str, Any]) -> str | None:
    value = _session(resolved).get("subject_id")
    return None if value is None else str(value)


def session_protocol(resolved: dict[str, Any]) -> str | None:
    value = _session(resolved).get("protocol")
    return None if value is None else str(value)


def session_meta(resolved: dict[str, Any]) -> dict[str, str]:
    meta = _session(resolved).get("meta") or {}
    if not isinstance(meta, dict):
        return {}
    out: dict[str, str] = {}
    for key, value in meta.items():
        if key in _RESERVED_SESSION_META_KEYS:
            continue
        if value is None or value == "":
            continue
        out[str(key)] = str(value)
    return out


def session_ct_calibration(resolved: dict[str, Any]) -> bool:
    return bool(_session(resolved).get("ct_calibration", False))


def session_ct_calibration_n_poses(resolved: dict[str, Any]) -> int:
    return int(_session(resolved).get("ct_calibration_n_poses", 20))


def session_ct_tip_calibration(resolved: dict[str, Any]) -> str | None:
    """Reusable tip-offset JSON path (written on calib, loaded when not calibrating)."""
    return _config_path(_session(resolved).get("ct_tip_calibration"))


def session_imu_mounting_calibration(resolved: dict[str, Any]) -> str | None:
    """Stable IMU mounting JSON: updated by ``jtsmm imu calibrate`` when a fit is accepted,
    embedded into every recorded session (``/calibration/imu_*``)."""
    return _config_path(_session(resolved).get("imu_mounting_calibration"))


def session_persist_ct_calibration(resolved: dict[str, Any]) -> bool:
    """Save CT tip calib poses + camera frames when true."""
    return bool(_session(resolved).get("persist_ct_calibration", False))


def session_ct_calibration_recordings_dir(resolved: dict[str, Any]) -> str | None:
    """Template path for persisted CT tip calib recordings (``{experiment_name}``)."""
    return _config_path(_session(resolved).get("ct_calibration_recordings_dir"))


def session_persist_imu_calibration(resolved: dict[str, Any]) -> bool:
    """Save IMU mounting cal session + metrics when true (default: true)."""
    return bool(_session(resolved).get("persist_imu_calibration", True))


def session_imu_calibration_recordings_dir(resolved: dict[str, Any]) -> str | None:
    """Template path for persisted IMU hand-eye recordings (``{experiment_name}``)."""
    return _config_path(_session(resolved).get("imu_calibration_recordings_dir"))


def session_mvc(resolved: dict[str, Any]) -> bool:
    """Dedicated EMG MVC take (``jtsmm record --mvc``), independent of the motion plan."""
    return bool(_session(resolved).get("mvc", False))


def session_optical_track(resolved: dict[str, Any]) -> bool:
    return bool(_session(resolved).get("optical_track", False))


def session_optical_track_hz(resolved: dict[str, Any]) -> float | None:
    """Live pose rate. ``None`` detects every stereo pair (the camera rate)."""
    raw = _session(resolved).get("optical_track_hz", None)
    if raw is None or raw == "":
        return None
    return float(raw)


def effective_optical_track_hz(
    resolved: dict[str, Any],
    *,
    ct_calibration: bool = False,
    protocol: str | None = None,
) -> float | None:
    """Rate the recorder should use. Tip pivot and IMU mounting stay at every pair."""
    if ct_calibration or (protocol or "") == "imu_mounting":
        return None
    return session_optical_track_hz(resolved)


def session_optical_config(resolved: dict[str, Any]) -> str | None:
    return _config_path(_session(resolved).get("optical_config"))


def session_use_experiment_plan(resolved: dict[str, Any]) -> bool:
    """Whether ``jtsmm record`` walks SPACE phases.

    If the key is omitted (legacy YAML), a non-empty ``experiment_plan`` path
    still enables the walker — that used to be the only switch.
    """
    session = _session(resolved)
    if "use_experiment_plan" in session:
        return bool(session.get("use_experiment_plan"))
    path = session.get("experiment_plan")
    return bool(path)


def session_experiment(resolved: dict[str, Any]) -> dict[str, Any] | None:
    """Inline motion-phase plan (``session.experiment``, same shape as optical.yaml)."""
    value = _session(resolved).get("experiment")
    if not isinstance(value, dict):
        return None
    phases = value.get("phases")
    if not isinstance(phases, list) or not phases:
        return None
    return value


def session_experiment_plan(resolved: dict[str, Any]) -> str | None:
    """Optional path override (JSON or optical.yaml)."""
    return _config_path(_session(resolved).get("experiment_plan"))


def session_write_jts_config(resolved: dict[str, Any]) -> bool:
    return bool(_session(resolved).get("write_jts_config", True))


def session_meta_from_config(resolved: dict[str, Any]) -> dict[str, str]:
    """Build operator /meta attrs from session.* (CLI overrides applied separately)."""
    out = dict(session_meta(resolved))
    subject = session_subject_id(resolved)
    protocol = session_protocol(resolved)
    if subject:
        out["subject_id"] = subject
    if protocol:
        out["protocol"] = protocol
    return out


def effective_duration_s(resolved: dict[str, Any], *, cli_duration: float | None = None) -> float:
    if cli_duration is not None:
        return float(cli_duration)
    duration = session_duration_s(resolved)
    if duration is not None:
        return duration
    return session_default_duration_s(resolved)


def capture_duration_s(
    resolved: dict[str, Any],
    *,
    cli_duration: float | None = None,
    plan_active: bool = False,
) -> float | None:
    """Wall-clock cap for the capture loop.

    Timed sessions (no experiment plan) use ``session.duration_s`` or
    ``session.default_duration_s``. With an experiment plan, SPACE walks the
    phases and the default 10 s does **not** apply. An explicit ``--duration``
    or a non-null ``session.duration_s`` remains an optional safety timeout.
    """
    if cli_duration is not None:
        return float(cli_duration)
    duration = session_duration_s(resolved)
    if duration is not None:
        return duration
    if plan_active:
        return None
    return session_default_duration_s(resolved)
