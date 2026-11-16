# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Test helpers — frame builders and HDF5 fixtures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from jtsmm.config import applied_config_hash, build_mcu_apply_payload
from jtsmm.config.catalog import FULL_CHANNEL_NAMES, channel_lsb_uv, lsb_uv
from jtsmm.config.loader import repo_root
from jtsmm.protocol.crc import crc16_ccitt

GOLDEN = repo_root() / "schema" / "v1" / "golden" / "frames.json"
ETC_TEMPLATE = Path(__file__).resolve().parents[1] / "etc" / "jtsmm.yaml"

# Operator template enables optical/CT/experiment; mock takes must opt out.
MOCK_RECORD_DISABLE_OPTICAL = (
    "--no-optical-track",
    "--no-ct-calibration",
    "--no-use-experiment-plan",
)


def build_frame(record_type: int, t_master_us: int, payload: bytes) -> bytes:
    """Encode a 0x55AA frame matching firmware host_link_send_record()."""
    body = bytes([record_type & 0xFF, len(payload) & 0xFF])
    body += int(t_master_us).to_bytes(8, "little")
    body += payload
    crc = crc16_ccitt(body).to_bytes(2, "little")
    return b"\x55\xaa" + body + crc


def load_golden_vectors() -> dict:
    with GOLDEN.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def load_operator_template() -> dict:
    with ETC_TEMPLATE.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_session_h5(
    path: Path,
    *,
    emg_samples: int = 1000,
    emg_fs: int = 1000,
    imu_hz: float = 400.0,
    trig_hz: int = 60,
    duration_s: float = 1.0,
    gap_at: int | None = None,
    bad_config_hash: bool = False,
    # Camera coverage knobs (defaults = full trigger coverage, 1:1 ids).
    cam_ids: tuple[str, ...] | None = None,
    cam_trailing_gap_s: float = 0.0,
    cam_frame_frac: float = 1.0,
    bad_sync_residual_us: float | None = None,
    bad_samples_uv: bool = False,
    omit_samples_uv: bool = False,
    noncontiguous_triggers: bool = False,
    emg_row_mismatch: bool = False,
    corrupt_quat: bool = False,
    sync_drops: dict[str, int] | None = None,
    skipped_cameras: tuple[str, ...] | None = None,
    optical: bool = False,
    optical_nan_frac: dict[str, float] | None = None,
    optical_bad_se3: bool = False,
    optical_jump_mm: float | None = None,
    optical_missing: bool = False,
    # Keep every Nth optical row and record that as the live-tracking stride.
    optical_track_stride: int = 1,
    # Coverage knobs: cut a stream short (seconds removed from its end).
    imu_truncate_s: float = 0.0,
    emg_truncate_s: float = 0.0,
    # Extra top-level keys merged into sync/report_json (trigger_stamp, host_capture, ...).
    sync_extra: dict[str, Any] | None = None,
    # EMG signal-quality knobs: mains sine amplitude (µV) on the muscle channels,
    # and the fraction of samples for which CH8 (BIAS_DRN readback) sits at its rail.
    emg_mains_uv: float = 0.0,
    # CH8 (BIAS_DRN readback): `drl_pinned` parks it at the rail with zero AC --
    # the HEALTHY state on this board, since IN8P is unwired. `drl_ac_mv` adds a
    # 50 Hz swing, which is the actual fault signature.
    drl_pinned: bool = False,
    drl_ac_mv: float = 0.0,
) -> dict[str, Any]:
    """Write a synthetic multimodal HDF5 session for integrity tests.

    By default the fixture mirrors a healthy v1 recording: calibrated EMG,
    contiguous triggers, and camera metadata covering the full trigger window.
    Pass the ``bad_*`` / gap knobs to inject the regressions the checker must catch.
    """
    resolved = load_operator_template()
    resolved["host"]["paths"] = {"data_dir": str(path.parent)}
    mcu_payload = build_mcu_apply_payload(resolved)
    host_bundle = {
        "host": resolved.get("host", {}),
        "session": resolved.get("session", {}),
        "hat": resolved.get("hat", {}),
    }
    cfg_hash = applied_config_hash(resolved)
    if bad_config_hash:
        cfg_hash = "0" * 64

    gain = int((mcu_payload.get("emg") or {}).get("default_gain", 6))
    lsb = float(lsb_uv(gain))
    # Per column, as the writers store it: CH7/CH8 run at a fixed gain of 24.
    lsb_cols = np.asarray(channel_lsb_uv(gain), dtype=np.float64)

    n_emg = min(emg_samples, int(emg_fs * max(0.0, duration_s - emg_truncate_s)))
    t_emg = np.arange(n_emg, dtype=np.uint64) * int(1e6 / emg_fs)
    if gap_at is not None and 0 < gap_at < n_emg - 1:
        t_emg[gap_at + 1 :] += int(5e6 / emg_fs)

    samples = np.zeros((n_emg, 8), dtype=np.int32)
    for ch in range(8):
        samples[:, ch] = (ch + 1) * 1000
    if emg_mains_uv > 0 and n_emg:
        # Coherent 50 Hz on the six muscle channels -- what mains pickup looks like.
        tt = np.arange(n_emg) / float(emg_fs)
        hum = (emg_mains_uv * np.sqrt(2.0) / lsb) * np.sin(2 * np.pi * 49.98 * tt)
        for ch in range(6):
            samples[:, ch] = samples[:, ch] + hum.astype(np.int32)
    if drl_pinned and n_emg:
        samples[:, 7] = int((1 << 23) * 0.99)
    if drl_ac_mv > 0 and n_emg:
        tt = np.arange(n_emg) / float(emg_fs)
        amp = (drl_ac_mv * 1000.0 * np.sqrt(2.0)) / lsb_cols[7]
        samples[:, 7] = samples[:, 7] + (amp * np.sin(2 * np.pi * 49.98 * tt)).astype(np.int32)
    samples_uv = (samples.astype(np.float64) * lsb_cols[None, :]).astype(np.float32)
    if bad_samples_uv:
        samples_uv = samples_uv + 50.0  # ≫ float32 quantization
    leadoff = np.zeros((n_emg, 2), dtype=np.uint8)

    n_trig = max(2, int(round(trig_hz * duration_s)))
    trig_t = (np.arange(n_trig, dtype=np.uint64) * int(round(1e6 / trig_hz)))
    trig_id = np.arange(1, n_trig + 1, dtype=np.uint64)
    if noncontiguous_triggers and n_trig >= 4:
        trig_id[2:] += 5  # skip several ids

    imu_span_s = max(0.01, duration_s - imu_truncate_s)
    n_imu = max(2, int(round(imu_hz * imu_span_s)))
    imu_t = np.linspace(0, imu_span_s * 1e6, n_imu, endpoint=False, dtype=np.uint64)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n_imu, 1))
    if corrupt_quat:
        quat[0] = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
        quat[1:] = quat[0]

    if cam_ids is None:
        cam_ids = tuple(
            str(e.get("id"))
            for e in (host_bundle.get("host", {}).get("cameras", {}) or {}).get("list") or []
            if e.get("id")
        ) or ("cam0", "cam1")

    residual = 0.0 if bad_sync_residual_us is None else float(bad_sync_residual_us)
    sync_doc: dict[str, Any] = {
        "pi_to_master": {"a": 1.0, "b": 0.0, "residual_rms_us": 0.0},
        "cameras": {
            cid: {"a": 1.0, "b": 0.0, "residual_rms_us": residual} for cid in cam_ids
        },
        "drops": dict(sync_drops or {}),
        "slip_count": {},
        "mcu_stats_end": {
            "missed_drdy": 0,
            "drdy_count": int(n_emg),
            "emg_drops": 0,
            "imu_mta_drops": 0,
            "imu_cra_drops": 0,
            "tx_drops": 0,
        },
    }
    if sync_extra:
        for key, value in sync_extra.items():
            if isinstance(value, dict) and isinstance(sync_doc.get(key), dict):
                sync_doc[key].update(value)
            else:
                sync_doc[key] = value

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["software_version"] = "0.1.0"
        meta.attrs["schema_version"] = 1
        meta.attrs["jts_schema_version"] = "1.1"
        meta.attrs["config_hash"] = cfg_hash
        meta.attrs["mcu_config_json"] = json.dumps(mcu_payload, separators=(",", ":"))
        meta.attrs["host_config_json"] = json.dumps(host_bundle, separators=(",", ":"))
        meta.attrs["session_duration_s"] = duration_s
        meta.attrs["file_hashes_json"] = "{}"
        if skipped_cameras:
            meta.attrs["skipped_cameras"] = ",".join(skipped_cameras)

        emg_g = f.create_group("emg")
        emg_rows = samples[:-5] if emg_row_mismatch and n_emg > 5 else samples
        emg_g.create_dataset("samples", data=emg_rows)
        if not omit_samples_uv:
            uv_rows = samples_uv[: emg_rows.shape[0]]
            emg_g.create_dataset("samples_uv", data=uv_rows)
        emg_g.create_dataset("leadoff_status", data=leadoff[: emg_rows.shape[0]])
        emg_g.create_dataset("t_master_us", data=t_emg)
        emg_g.attrs["sample_rate_sps"] = emg_fs
        emg_g.attrs["gain"] = gain
        emg_g.attrs["lsb_uv"] = lsb
        emg_g.attrs["lsb_uv_per_channel"] = lsb_cols
        emg_g.attrs["channel_names"] = np.array(FULL_CHANNEL_NAMES, dtype="S")

        # Match kinematics_default: quat @ imu_hz + linear_accel ~250 Hz.
        n_lin = max(2, int(round(250.0 * imu_span_s)))
        lin_t = np.linspace(0, imu_span_s * 1e6, n_lin, endpoint=False, dtype=np.uint64)
        lin = np.zeros((n_lin, 3), dtype=np.float32)
        for slot in ("imu_mta", "imu_cra"):
            imu_g = f.create_group(slot)
            imu_g.create_dataset("quat", data=quat)
            imu_g.create_dataset("t_master_us_quat", data=imu_t)
            imu_g.create_dataset("linear_accel", data=lin)
            imu_g.create_dataset("t_master_us_linear_accel", data=lin_t)
            imu_g.attrs["fusion_mode"] = "game_rotation_vector"

        sync = f.create_group("sync")
        sync.attrs["report_json"] = json.dumps(sync_doc, separators=(",", ":"))

        trig_g = f.create_group("triggers")
        trig_g.create_dataset("trigger_id", data=trig_id)
        trig_g.create_dataset("t_master_us", data=trig_t)
        trig_g.attrs["rate_hz"] = trig_hz
        trig_g.attrs["source"] = "mcu_pio"

        cam_root = f.create_group("cameras")
        # Keep camera metadata through the last trigger, optionally truncating
        # a trailing gap (the starvation regression) or thinning frames.
        usable_n = n_trig
        if cam_trailing_gap_s > 0 and trig_hz > 0:
            usable_n = max(2, n_trig - int(round(cam_trailing_gap_s * trig_hz)))
        keep_n = max(2, int(round(usable_n * float(cam_frame_frac))))
        keep_idx = np.linspace(0, usable_n - 1, keep_n, dtype=int)
        for cam_id in cam_ids:
            grp = cam_root.create_group(cam_id)
            c_trig = trig_id[keep_idx]
            c_t = trig_t[keep_idx]
            # Exact association (0 µs jitter) — matches production sequence consume.
            grp.create_dataset("trigger_id", data=c_trig)
            grp.create_dataset("t_master_us", data=c_t)
            grp.create_dataset("frame_idx", data=c_trig.copy())
            # Epoch-scale sensor timestamps (picamera2 SensorTimestamp style).
            sensor0 = np.uint64(1_784_532_021_847_000_000)
            period_ns = np.uint64(int(round(1e9 / trig_hz)))
            grp.create_dataset(
                "sensor_timestamp_ns",
                data=sensor0 + keep_idx.astype(np.uint64) * period_ns,
            )

        if optical_missing:
            meta.attrs["optical_tracking"] = "true"

        if optical:
            meta.attrs["optical_tracking"] = "true"
            meta.attrs["optical_pose_method"] = "triangulation"
            t_opt = trig_t[keep_idx]
            stride = int(optical_track_stride)
            if stride > 1:
                meta.attrs["optical_track_stride"] = str(stride)
                t_opt = t_opt[::stride]
            n_opt = int(t_opt.size)
            eye = np.eye(4, dtype=np.float32)
            nanp = np.full((4, 4), np.nan, dtype=np.float32)
            origins = {
                "HP": np.array([40.0, 80.0, 400.0], dtype=np.float32),
                "MP": np.array([80.0, 20.0, 450.0], dtype=np.float32),
                "CT": np.array([90.0, 10.0, 440.0], dtype=np.float32),
            }
            og = f.create_group("optical")
            og.create_dataset("t_master_us", data=t_opt)
            nan_frac = dict(optical_nan_frac or {})
            body_ds = (("HP", "pose_cra"), ("MP", "pose_mta"), ("CT", "pose_ct"))
            qg = og.create_group("quality")
            qg.attrs["rms_kind"] = "kabsch_mm"
            slot_by_body = {"HP": "cra", "MP": "mta", "CT": "ct"}
            for body, ds in body_ds:
                poses = np.tile(eye, (n_opt, 1, 1))
                poses[:, :3, 3] = origins[body]
                n_nan = int(round(float(nan_frac.get(body, 0.0)) * n_opt))
                if n_nan > 0:
                    poses[:n_nan] = nanp
                if optical_bad_se3 and n_nan < n_opt:
                    poses[n_nan, :3, :3] *= 2.0
                if optical_jump_mm is not None and n_opt >= 2 and n_nan < n_opt - 1:
                    j = n_nan + 1
                    poses[j, :3, 3] = origins[body] + np.array(
                        [float(optical_jump_mm), 0.0, 0.0], dtype=np.float32
                    )
                og.create_dataset(ds, data=poses)
                slot = slot_by_body[body]
                sg = qg.create_group(slot)
                locked = np.isfinite(poses).reshape(n_opt, -1).all(axis=1)
                sg.create_dataset("n_tags", data=np.where(locked, 2, 0).astype(np.uint8))
                sg.create_dataset("n_cameras", data=np.where(locked, 2, 0).astype(np.uint8))
                rms = np.where(locked, 0.4, np.nan).astype(np.float32)
                sg.create_dataset("reprojection_rms", data=rms)
            sync_doc.setdefault("optical", {})
            sync_doc["optical"].update(
                {
                    "enqueued": n_opt,
                    "processed": n_opt,
                    "written": n_opt,
                    "paired": n_opt,
                    "pair_dropped": 0,
                    "queue_dropped": 0,
                    "failed": 0,
                }
            )
            sync.attrs["report_json"] = json.dumps(sync_doc, separators=(",", ":"))

    return resolved
