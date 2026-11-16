# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Write a synthetic multimodal session.h5 (+ JTS config) for offline ML work.

Mirrors a complete ``jtsmm record --optical-track`` + ``--experiment-plan`` take
with the same datasets and attributes the recorder writes (``Hdf5Writer`` plus
the supervisor's ``/meta`` keys): EMG, IMU, triggers, cameras, sync, optical
poses + quality, events/phases, and the ``<slug>_jts_config.json`` sidecar.
Rates come from the operator template's MCU payload (4 kHz EMG, 40 Hz
trigger, 390/250 Hz IMU) unless overridden, and the stored config snapshot
always matches the data. Kinematics are synthetic (not biomechanical).

Imperfections can be injected to exercise gap handling downstream:
EMG dropouts, optical gaps (frames missing from ``/optical`` and
``/cameras``), random single dropped frames and NaN poses.

Example::

    python -m jtsmm.tools.mock_session data/mock_session.h5 --duration 30
    python -m jtsmm.tools.mock_session m.h5 --emg-dropout 5:0.8 --optical-gap 12:1.5 \\
        --drop-frame-frac 0.15 --nan-pose-frac 0.05
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import yaml

from jtsmm._version import __version__
from jtsmm.config import applied_config_hash, build_mcu_apply_payload
from jtsmm.config.catalog import DEFAULT_EMG_SAMPLE_RATE_SPS, FULL_CHANNEL_NAMES, channel_lsb_uv, lsb_uv
from jtsmm.config.loader import template_path
from jtsmm.capture.phase_walker import TimedPhase, optical_frame_intervals
from jtsmm.imu.gravity_lin_accel import G
from jtsmm.io.hdf5_writer import ACCURACY_NOTE
from jtsmm.protocol.records import FUSION_REPORT_IDS

# The shipped operator template (checkout copy, else the packaged one).
ETC_TEMPLATE = template_path()

# Fixed landmark calibration + a few motion phases (same kinds as live record).
_MOCK_PHASES: tuple[tuple[str, str], ...] = (
    ("mand_point_1", "calibration_mandibular"),
    ("mand_point_2", "calibration_mandibular"),
    ("mand_point_3", "calibration_mandibular"),
    ("max_point_1", "calibration_maxillary"),
    ("max_point_2", "calibration_maxillary"),
    ("max_point_3", "calibration_maxillary"),
    ("tmj_left", "calibration_tmj"),
    ("tmj_right", "calibration_tmj"),
    ("open_close", "experiment"),
    ("left_right", "experiment"),
    ("chewing", "experiment"),
)

_PLAN_NAME = "mock_experiment"
_PLAN_DESCRIPTION = "Synthetic multimodal session for offline ML"
_FIRMWARE_VERSION = "1.0.0-mock"
# Mirrors hdf5_writer._LEADOFF_STATUS_NOTE's meaning (the writer owns the text).
_LEADOFF_NOTE = (
    "columns=[statp_mask,statn_mask] from the ADS1299 status word; bit i = CH(i+1), "
    "muscle CH1-6 only (bits 6-7 always 0). 1 = lead-off comparator tripped. "
    "Synthetic session: always 0."
)
# Camera sensor timestamps are CLOCK_BOOTTIME ns (seconds since the Pi booted).
_SENSOR_T0_NS = 1_234_567_000_000


def _load_operator_template() -> dict[str, Any]:
    with ETC_TEMPLATE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _phase_schedule(duration_s: float) -> list[TimedPhase]:
    """Partition the session into sequential phases with tiny gaps."""
    n = len(_MOCK_PHASES)
    # Leave ~5% head/tail margin so every phase overlaps optical frames.
    usable = max(duration_s * 0.9, duration_s - 0.5)
    start0 = (duration_s - usable) * 0.5
    slot = usable / n
    gap = min(0.02, slot * 0.05)
    out: list[TimedPhase] = []
    for i, (name, kind) in enumerate(_MOCK_PHASES):
        t0 = start0 + i * slot
        t1 = start0 + (i + 1) * slot - gap
        out.append(
            TimedPhase(
                name=name,
                kind=kind,
                t_start_us=int(round(t0 * 1e6)),
                t_end_us=int(round(max(t1, t0 + 1e-3) * 1e6)),
                map_quality_start="exact",
                map_quality_end="exact",
            )
        )
    return out


def _synthetic_poses(n: int, *, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Identity SE(3) + small sinusoid translations (mm-scale, float32)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    pose_mta = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose_cra = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose_ct = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    pose_mta[:, 0, 3] = (5.0 * np.sin(t)).astype(np.float32)
    pose_mta[:, 1, 3] = (2.0 * np.cos(t)).astype(np.float32)
    pose_cra[:, 0, 3] = (1.0 * np.sin(0.5 * t)).astype(np.float32)
    pose_cra[:, 2, 3] = (3.0 * np.cos(0.5 * t)).astype(np.float32)
    pose_ct[:, 0, 3] = (0.5 * np.sin(t + 0.3)).astype(np.float32)
    pose_ct[:, 1, 3] = (0.5 * np.cos(t + 0.3)).astype(np.float32)
    # Tiny noise so poses are not perfectly deterministic across axes.
    pose_mta[:, :3, 3] += rng.normal(0.0, 0.05, size=(n, 3)).astype(np.float32)
    # Camera-world placement like s01_f_30 (HP-MP ~160 mm apart, ~0.7 m away).
    pose_cra[:, :3, 3] += np.float32([40.0, -65.0, 700.0])
    pose_mta[:, :3, 3] += np.float32([55.0, 80.0, 645.0])
    pose_ct[:, :3, 3] += np.float32([50.0, 45.0, 680.0])
    return pose_mta, pose_cra, pose_ct


def _in_windows(t_s: np.ndarray, windows: Sequence[tuple[float, float]]) -> np.ndarray:
    """Mask of ``t_s`` inside any ``(start_s, duration_s)`` window."""
    hit = np.zeros(t_s.shape, dtype=bool)
    for start, dur in windows:
        hit |= (t_s >= float(start)) & (t_s < float(start) + float(dur))
    return hit


def _apply_rate_overrides(
    resolved: dict[str, Any], *, emg_fs: int | None, trig_hz: int | None, imu_hz: float | None
) -> None:
    """Write CLI rate overrides into the config so the stored snapshot matches."""
    mcu = resolved.setdefault("mcu", {})
    if emg_fs is not None:
        mcu.setdefault("emg", {})["sample_rate_sps"] = int(emg_fs)
    if trig_hz is not None:
        mcu.setdefault("trigger", {})["rate_hz"] = int(trig_hz)
        for cam in ((resolved.get("host") or {}).get("cameras") or {}).get("list") or []:
            cam.setdefault("trigger", {})["expect_rate_hz"] = int(trig_hz)
    if imu_hz is not None:
        for slot in ("imu_mta", "imu_cra"):
            blk = mcu.setdefault(slot, {})
            fusion = blk.setdefault("fusion", {})
            fusion["rate_hz"] = int(round(imu_hz))
            primary = fusion.get("primary", "game_rotation_vector")
            blk.setdefault("reports", {}).setdefault(primary, {})["rate_hz"] = int(round(imu_hz))


def write_mock_session(
    path: Path | str,
    *,
    duration_s: float = 30.0,
    emg_fs: int | None = None,
    imu_hz: float | None = None,
    trig_hz: int | None = None,
    jts_config_path: Path | str | None = None,
    write_jts_config: bool = True,
    emg_dropouts: Sequence[tuple[float, float]] = (),
    optical_gaps: Sequence[tuple[float, float]] = (),
    drop_frame_frac: float = 0.0,
    nan_pose_frac: float = 0.0,
    seed: int = 0,
) -> dict[str, Path | None]:
    """Write ``session.h5`` (+ JTS sidecar). Returns paths for both outputs.

    Rates default to the operator template's MCU payload. ``emg_dropouts`` and
    ``optical_gaps`` are ``(start_s, duration_s)`` windows with no EMG samples /
    no camera or optical frames; ``drop_frame_frac`` drops that fraction of
    single frames at random; ``nan_pose_frac`` marks that fraction of optical
    rows untracked (NaN) per body. ``jts_config`` is ``None`` when the optical
    package (cv2) is unavailable or ``write_jts_config`` is false.
    """
    path = Path(path)
    duration_s = float(duration_s)
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    rng = np.random.default_rng(seed)

    resolved = copy.deepcopy(_load_operator_template())
    # The one data-dir key (host.storage.base_dir is deprecated, and next to a
    # different paths.data_dir it makes the stored config fail re-validation).
    resolved["host"].pop("storage", None)
    resolved["host"].setdefault("paths", {})["data_dir"] = str(path.parent)
    _apply_rate_overrides(resolved, emg_fs=emg_fs, trig_hz=trig_hz, imu_hz=imu_hz)
    mcu_payload = build_mcu_apply_payload(resolved)
    host_bundle = {
        "host": resolved.get("host", {}),
        "session": resolved.get("session", {}),
        "hat": resolved.get("hat", {}),
    }
    cfg_hash = applied_config_hash(resolved)

    emg_cfg = mcu_payload.get("emg") or {}
    emg_fs = int(emg_cfg.get("sample_rate_sps", DEFAULT_EMG_SAMPLE_RATE_SPS))
    gain = int(emg_cfg.get("default_gain", 6))
    lsb_cols = np.asarray(channel_lsb_uv(gain, 8), dtype=np.float64)
    trig_hz = int((mcu_payload.get("trigger") or {}).get("rate_hz", 30))
    slot_cfg = mcu_payload.get("imu_mta") or {}
    fusion = slot_cfg.get("fusion") or {}
    quat_hz = float(fusion.get("rate_hz", 390))
    fusion_name = str(fusion.get("primary", "game_rotation_vector"))
    fusion_id = {v: k for k, v in FUSION_REPORT_IDS.items()}.get(fusion_name, 0x08)
    slot_reports = slot_cfg.get("reports") or {}

    def _motion_hz(report: str) -> float | None:
        cfg = slot_reports.get(report)
        return float(cfg.get("rate_hz", 250)) if cfg and cfg.get("enabled", True) else None

    accel_hz = _motion_hz("accelerometer")
    lin_hz = _motion_hz("linear_acceleration")

    # --- EMG (counts + per-column µV, K5) -------------------------------------
    n_emg_all = max(2, int(round(emg_fs * duration_s)))
    seq_all = np.arange(n_emg_all, dtype=np.uint32)
    t_emg_all = np.round(seq_all.astype(np.float64) * (1e6 / emg_fs)).astype(np.uint64)
    keep_emg = ~_in_windows(t_emg_all / 1e6, emg_dropouts)
    seq = seq_all[keep_emg]
    t_emg = t_emg_all[keep_emg]
    k = seq.astype(np.float64)
    samples = np.zeros((seq.size, 8), dtype=np.int32)
    for ch in range(8):
        # Mild channel tone so loaders see non-constant EMG.
        samples[:, ch] = (ch + 1) * 1000 + np.round(
            200 * np.sin(2 * np.pi * (ch + 1) * k / emg_fs)
        ).astype(np.int32)
    samples[:, 6] = 0  # CH7 bias_debug reads zero on this board
    samples_uv = (samples.astype(np.float64) * lsb_cols).astype(np.float32)
    leadoff = np.zeros((seq.size, 2), dtype=np.uint8)

    # --- triggers / frames ------------------------------------------------------
    period_us = 1e6 / trig_hz
    n_trig = max(2, int(round(trig_hz * duration_s)))
    trig_t = np.round(np.arange(n_trig) * period_us).astype(np.uint64)
    trig_id = np.arange(1, n_trig + 1, dtype=np.uint64)
    frame_ok = ~_in_windows(trig_t / 1e6, optical_gaps)
    if drop_frame_frac > 0:
        frame_ok &= rng.random(n_trig) >= float(drop_frame_frac)
    frame_ok[0] = frame_ok[-1] = True  # grid spans the whole take
    frames = np.flatnonzero(frame_ok)

    # --- IMU (fusion quaternion + the config's accel / linear accel) ----------
    n_q = max(2, int(round(quat_hz * duration_s)))
    t_q = np.round(np.arange(n_q) * (1e6 / quat_hz)).astype(np.uint64)
    ang = 0.05 * np.sin(2 * np.pi * 0.2 * t_q / 1e6)  # slow nod about x
    quat = np.column_stack(
        [np.cos(ang / 2), np.sin(ang / 2), np.zeros(n_q), np.zeros(n_q)]
    ).astype(np.float32)
    # name -> (t, (n,3) m/s^2, rate): accel reads gravity at the nodding pose,
    # linear_accel is gravity-free.
    motion: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for name, hz in (("accel", accel_hz), ("linear_accel", lin_hz)):
        if hz is None:
            continue
        n_m = max(2, int(round(hz * duration_s)))
        t_m = np.round(np.arange(n_m) * (1e6 / hz)).astype(np.uint64)
        v = rng.normal(0.0, 0.02, size=(n_m, 3))
        if name == "accel":
            a_m = 0.05 * np.sin(2 * np.pi * 0.2 * t_m / 1e6)
            v += np.column_stack([np.zeros(n_m), G * np.sin(a_m), G * np.cos(a_m)])
        motion[name] = (t_m, v.astype(np.float32), hz)

    cam_ids = tuple(
        str(e.get("id"))
        for e in (host_bundle.get("host", {}).get("cameras", {}) or {}).get("list") or []
        if e.get("id")
    ) or ("cam0", "cam1")

    n_dropped = int(n_trig - frames.size)
    sync_doc: dict[str, Any] = {
        "pi_to_master": {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0},
        "cameras": {cid: {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0} for cid in cam_ids},
        "drops": {cid: n_dropped for cid in cam_ids},
        "slip_count": {},
        "trigger_stamp": {"counts": {"pio_irq": n_trig}, "fallback": 0},
        "optical": {"processed": int(frames.size), "written": int(frames.size), "queue_dropped": 0},
        "mcu_stats_end": {
            "missed_drdy": 0,
            "drdy_count": int(n_emg_all),
            "emg_drops": int(n_emg_all - seq.size),
            "imu_mta_drops": 0,
            "imu_cra_drops": 0,
            "tx_drops": 0,
        },
    }

    timed = _phase_schedule(duration_s)
    pose_mta, pose_cra, pose_ct = _synthetic_poses(n_trig, seed=seed)
    pose_mta, pose_cra, pose_ct = pose_mta[frames], pose_cra[frames], pose_ct[frames]
    t_opt = trig_t[frames]
    tracked = {}
    for name, poses in (("mta", pose_mta), ("cra", pose_cra), ("ct", pose_ct)):
        lost = rng.random(frames.size) < float(nan_pose_frac) if nan_pose_frac > 0 else np.zeros(frames.size, bool)
        poses[lost] = np.nan
        tracked[name] = ~lost

    opened_at = time.time()
    recording_date = datetime.fromtimestamp(opened_at, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    plan_dict = {
        "name": _PLAN_NAME,
        "description": _PLAN_DESCRIPTION,
        "phases": [name for name, kind in _MOCK_PHASES if kind == "experiment"],
    }
    session_cfg = host_bundle.get("session") or {}
    lores = next(
        (c.get("lores") for c in (host_bundle["host"].get("cameras") or {}).get("list") or [] if c.get("lores")),
        None,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["software_version"] = f"{__version__}-mock"
        meta.attrs["schema_version"] = int(mcu_payload.get("schema_version", 1))
        meta.attrs["jts_schema_version"] = "1.1"
        meta.attrs["created_unix_utc"] = opened_at
        meta.attrs["recording_date"] = recording_date
        meta.attrs["config_hash"] = cfg_hash
        meta.attrs["mcu_config_json"] = json.dumps(mcu_payload, separators=(",", ":"))
        meta.attrs["host_config_json"] = json.dumps(host_bundle, separators=(",", ":"))
        meta.attrs["firmware_version"] = _FIRMWARE_VERSION
        meta.attrs["session_duration_s"] = duration_s
        meta.attrs["file_hashes_json"] = "{}"
        # Supervisor-level keys (session_meta + OpticalTrackBridge.session_meta).
        meta.attrs["mode"] = str(session_cfg.get("mode", "acquisition"))
        meta.attrs["emg_enabled"] = "true"
        meta.attrs["experiment_name"] = _PLAN_NAME
        meta.attrs["experiment_plan_json"] = json.dumps(plan_dict, separators=(",", ":"))
        meta.attrs["optical_tracking"] = "true"
        meta.attrs["optical_config"] = str(session_cfg.get("optical_config", "software/optical/etc/optical.yaml"))
        meta.attrs["optical_pose_method"] = "triangulation"
        meta.attrs["optical_cam_ids"] = ",".join(cam_ids)
        meta.attrs["optical_pair_mode"] = "sensor_time"
        meta.attrs["optical_pair_match_ns"] = str(int(round(1e9 / (2.0 * trig_hz))))
        meta.attrs["optical_rgb_stream"] = "lores_y"
        meta.attrs["optical_rgb_size"] = f"{lores[0]}x{lores[1]}" if lores else "728x544"
        meta.attrs["optical_lores_format"] = "YUV420"

        emg_g = f.create_group("emg")
        emg_g.create_dataset("samples", data=samples)
        emg_g.create_dataset("samples_uv", data=samples_uv)
        emg_g.create_dataset("leadoff_status", data=leadoff)
        emg_g.create_dataset("t_master_us", data=t_emg)
        emg_g.create_dataset("seq", data=seq)  # K1 acquisition sequence (gaps = loss)
        emg_g.attrs["sample_rate_sps"] = emg_fs
        emg_g.attrs["gain"] = gain
        emg_g.attrs["lsb_uv"] = float(lsb_uv(gain))
        emg_g.attrs["lsb_uv_per_channel"] = lsb_cols
        emg_g.attrs["channel_names"] = np.array(FULL_CHANNEL_NAMES, dtype="S")
        emg_g.attrs["leadoff_status_note"] = _LEADOFF_NOTE

        for slot in ("imu_mta", "imu_cra"):
            imu_g = f.create_group(slot)
            imu_g.create_dataset("quat", data=quat)
            imu_g.create_dataset("t_master_us_quat", data=t_q)
            imu_g.create_dataset("quat_report_id", data=np.full(n_q, fusion_id, dtype=np.uint8))
            imu_g.create_dataset("quat_accuracy", data=np.full(n_q, 3, dtype=np.uint8))
            imu_g["quat_accuracy"].attrs["note"] = ACCURACY_NOTE
            for name, (t_m, v, hz) in motion.items():
                ds = imu_g.create_dataset(name, data=v)
                ds.attrs["units"] = "m/s^2"
                ds.attrs["device"] = "BNO085"
                ds.attrs["rate_hz"] = int(round(hz))
                imu_g.create_dataset(f"t_master_us_{name}", data=t_m)
                imu_g.create_dataset(f"{name}_accuracy", data=np.full(t_m.size, 3, dtype=np.uint8))
                imu_g[f"{name}_accuracy"].attrs["note"] = ACCURACY_NOTE
            imu_g.attrs["fusion_mode"] = fusion_name
            imu_g.attrs["quat_fusion"] = FUSION_REPORT_IDS.get(fusion_id, fusion_name)

        sync = f.create_group("sync")
        sync.attrs["report_json"] = json.dumps(sync_doc, separators=(",", ":"))
        for cid in cam_ids:
            sync.create_dataset(f"drops/{cid}", data=np.asarray([[0, n_dropped]], dtype=np.uint64))

        trig_g = f.create_group("triggers")
        trig_g.create_dataset("trigger_id", data=trig_id)
        trig_g.create_dataset("t_master_us", data=trig_t)
        trig_g.attrs["rate_hz"] = trig_hz
        trig_g.attrs["source"] = "mcu_pio"

        cam_root = f.create_group("cameras")
        for j, cam_id in enumerate(cam_ids):
            grp = cam_root.create_group(cam_id)
            grp.create_dataset("trigger_id", data=trig_id[frames])
            grp.create_dataset("t_master_us", data=trig_t[frames])
            grp.create_dataset("frame_idx", data=np.arange(1, frames.size + 1, dtype=np.uint64))
            sensor_ns = _SENSOR_T0_NS + j * 1_000 + np.round(trig_t[frames].astype(np.float64) * 1e3)
            grp.create_dataset("sensor_timestamp_ns", data=sensor_ns.astype(np.uint64))

        optical_g = f.create_group("optical")
        optical_g.create_dataset("t_master_us", data=t_opt)
        optical_g.create_dataset("pose_mta", data=pose_mta)
        optical_g.create_dataset("pose_cra", data=pose_cra)
        optical_g.create_dataset("pose_ct", data=pose_ct)
        qg = optical_g.create_group("quality")
        qg.attrs["rms_kind"] = "kabsch_mm"
        qg.attrs["note"] = (
            "per-body n_tags, n_cameras, reprojection_rms; "
            "rms is PnP corner reprojection (px) or triangulation Kabsch residual (mm)"
        )
        for name, ok in tracked.items():
            sg = qg.create_group(name)
            sg.create_dataset("n_tags", data=np.where(ok, 2, 0).astype(np.uint8))
            sg.create_dataset("n_cameras", data=np.where(ok, 2, 0).astype(np.uint8))
            sg.create_dataset("reprojection_rms", data=np.where(ok, 0.4, np.nan).astype(np.float32))

        # Events: phase_start / phase_end pairs (same shape as live record).
        ev_t: list[int] = []
        ev_kind: list[str] = []
        ev_text: list[str] = []
        ev_phase_kind: list[str] = []
        ev_host: list[int] = []
        ev_qual: list[str] = []
        for ph in timed:
            for edge, t_us in (("start", ph.t_start_us), ("end", ph.t_end_us)):
                ev_t.append(int(t_us))
                ev_kind.append(f"phase_{edge}")
                ev_text.append(ph.name)
                ev_phase_kind.append(ph.kind)
                ev_host.append(int(t_us) * 1000)
                ev_qual.append("exact")

        str_dt = h5py.string_dtype()
        ev_g = f.create_group("events")
        ev_g.create_dataset("t_master_us", data=np.asarray(ev_t, dtype=np.uint64))
        ev_g.create_dataset("kind", data=np.asarray(ev_kind, dtype=str_dt))
        ev_g.create_dataset("text", data=np.asarray(ev_text, dtype=str_dt))
        ev_g.create_dataset("phase_kind", data=np.asarray(ev_phase_kind, dtype=str_dt))
        ev_g.create_dataset("t_host_ns", data=np.asarray(ev_host, dtype=np.uint64))
        ev_g.create_dataset("map_quality", data=np.asarray(ev_qual, dtype=str_dt))
        # Browser display time of viewer notes; the mock has none (NaN).
        ev_g.create_dataset("viewer_t_s", data=np.full(len(ev_t), np.nan, dtype=np.float64))

        phases_json = [ph.to_dict() for ph in timed]
        ph_g = f.create_group("phases")
        ph_g.create_dataset(
            "name", data=np.asarray([ph.name for ph in timed], dtype=str_dt)
        )
        ph_g.create_dataset(
            "kind", data=np.asarray([ph.kind for ph in timed], dtype=str_dt)
        )
        ph_g.create_dataset(
            "t_start_us",
            data=np.asarray([ph.t_start_us for ph in timed], dtype=np.uint64),
        )
        ph_g.create_dataset(
            "t_end_us",
            data=np.asarray([ph.t_end_us for ph in timed], dtype=np.uint64),
        )
        ph_g.attrs["n_phases"] = len(timed)
        ph_g.attrs["phases_json"] = json.dumps(phases_json, separators=(",", ":"))

    jts_path: Path | None = None
    if write_jts_config:
        try:
            jts_path = _write_jts_sidecar(
                path,
                timed=timed,
                jts_config_path=Path(jts_config_path) if jts_config_path is not None else None,
                period_us=period_us,
            )
        except ImportError as exc:
            print(f"warning: skipped JTS config (optical package unavailable: {exc})", file=sys.stderr)
    return {"session_h5": path, "jts_config": jts_path}


def _write_jts_sidecar(
    h5_path: Path,
    *,
    timed: list[TimedPhase],
    jts_config_path: Path | None,
    period_us: float | None = None,
) -> Path:
    """Map phase intervals onto trigger-grid frames and write the JTS JSON sidecar."""
    from optical.tracking.experiment import RecordedPhase, build_jts_config, write_jts_config

    with h5py.File(h5_path, "r") as f:
        t_opt = f["optical"]["t_master_us"][()]

    intervals = optical_frame_intervals(timed, t_opt, period_us=period_us)
    if not intervals:
        raise RuntimeError("no optical frames overlapped mock phase intervals")

    recorded = [
        RecordedPhase(name=n, kind=k, start_frame=s, end_frame=e)
        for n, k, s, e in intervals
    ]
    from optical.tracking.experiment import slugify

    out = jts_config_path or (h5_path.parent / f"{slugify(_PLAN_NAME)}_jts_config.json")
    cfg = build_jts_config(
        recorded,
        session_path=h5_path.resolve(),
        description=_PLAN_DESCRIPTION,
    )
    write_jts_config(out, cfg)
    return Path(out)


def _window(text: str) -> tuple[float, float]:
    try:
        start, dur = text.split(":", 1)
        return float(start), float(dur)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected START_S:DURATION_S, got {text!r}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a synthetic multimodal session.h5 + JTS config for offline ML."
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output path for session.h5",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Session duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--jts-config",
        type=Path,
        default=None,
        help="JTS config output path (default: <dir>/<slug>_jts_config.json)",
    )
    parser.add_argument("--emg-fs", type=int, default=None, help="EMG rate (default: config, 4000)")
    parser.add_argument("--trigger-hz", type=int, default=None, help="Camera trigger rate (default: config, 40)")
    parser.add_argument("--imu-hz", type=float, default=None, help="Fusion quaternion rate (default: config)")
    parser.add_argument(
        "--emg-dropout", type=_window, action="append", default=[], metavar="START:DUR",
        help="EMG samples missing in this window (seconds); repeatable",
    )
    parser.add_argument(
        "--optical-gap", type=_window, action="append", default=[], metavar="START:DUR",
        help="Camera/optical frames missing in this window (seconds); repeatable",
    )
    parser.add_argument("--drop-frame-frac", type=float, default=0.0,
                        help="Fraction of single camera frames dropped at random")
    parser.add_argument("--nan-pose-frac", type=float, default=0.0,
                        help="Fraction of optical rows untracked (NaN) per body")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for injected faults")
    args = parser.parse_args(argv)

    try:
        paths = write_mock_session(
            args.output,
            duration_s=args.duration,
            emg_fs=args.emg_fs,
            imu_hz=args.imu_hz,
            trig_hz=args.trigger_hz,
            jts_config_path=args.jts_config,
            emg_dropouts=args.emg_dropout,
            optical_gaps=args.optical_gap,
            drop_frame_frac=args.drop_frame_frac,
            nan_pose_frac=args.nan_pose_frac,
            seed=args.seed,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote session -> {paths['session_h5']}")
    if paths["jts_config"] is not None:
        print(f"Wrote JTS config -> {paths['jts_config']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
