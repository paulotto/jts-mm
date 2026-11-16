# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Post-session HDF5 integrity validation (Phase 4)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from jtsmm.config import applied_config_hash, config_hash
from jtsmm.config.catalog import AUX_CHANNEL_GAIN, channel_lsb_uv
from jtsmm.config.registry import integrity_thresholds
from jtsmm.io.session_reader import montage_corrected
from jtsmm.integrity.emg_signal import (
    DRL_AC_ERR_MV,
    DRL_AC_WARN_MV,
    MAINS_BAD_WINDOW_FRAC_ERR,
    MAINS_BAD_WINDOW_FRAC_WARN,
    analyse_emg_signal,
)
from jtsmm.integrity.optical import check_optical
from jtsmm.integrity.optical_imu_sync import check_optical_imu_sync

# Hardware-characterisation tolerances. NOT policy gates -- these describe how
# the BNO085 and the ADS1299 actually behave, so they are not configurable.
IMU_RATE_TOL_DEFAULT = 0.02
# BNO085 hub scheduling: next to a 3-axis vector the hub skips fusion slots
# (game RV 390 req -> ~348-354 Hz) and quantizes the vector (accel 250 req ->
# 232-244 Hz, linear accel 250 -> ~220 Hz); doctor imu-rates --sweep
# 2026-10-06, 0 SH-2 sequence gaps, i.e. nothing lost. Lost reports are caught
# by the sequence-gap gate, not here.
IMU_RATE_TOL: dict[str, float] = {
    "game_rotation_vector": 0.12,
    "rotation_vector": 0.12,
    "geomagnetic_rotation_vector": 0.12,
    "linear_acceleration": 0.15,
    "accelerometer": 0.10,
    "gyroscope": 0.05,
    "magnetometer": 0.05,  # was keyed "magnetic_field", which never matched
}
SAMPLES_UV_ABS_TOL_UV = 1.0


@dataclass(frozen=True)
class IntegrityThresholds:
    """Pass/fail gates for `check_session`, overridable from the operator YAML.

    Defaults are the values these checks shipped with. They are policy, not
    physics: how much loss or misalignment a session may carry and still be
    called usable. A study with a looser or tighter tolerance should say so in
    `host.integrity.thresholds` rather than patching this file, and the values
    actually applied are recorded in `report.summary["thresholds"]` so a verdict
    stays auditable after the fact.

    Every `*_err_*` gate must sit at or beyond its `*_warn_*` partner; an
    inverted pair makes the warn branch unreachable, so it is reported and the
    defaults are used for that pair instead.
    """

    # EMG continuity, in multiples of the nominal sample period.
    emg_gap_warn_periods: float = 2.5
    emg_gap_err_periods: float = 4.0
    # EMG samples missing anywhere in the take (single samples included; exact
    # from /emg/seq when present), as a fraction of samples expected.
    emg_loss_warn_frac: float = 0.0005
    emg_loss_err_frac: float = 0.005
    # Fraction of ADC conversions the MCU missed because SPI DMA was still busy.
    emg_missed_drdy_warn_frac: float = 0.001
    emg_missed_drdy_err_frac: float = 0.02
    # IMU samples lost (ring-buffer overflow, USB tx drops, or timestamp gaps).
    imu_ring_drop_err_frac: float = 0.005
    # A step longer than this many median steps is an IMU gap (tolerant: IMU
    # timestamps are poll time). Longest gap > stream_end_gap_err_s, or total
    # loss > imu_ring_drop_err_frac, is an error.
    imu_gap_warn_periods: float = 4.0
    # A stream that stops early, or starts late, leaves that span with no data
    # for its modality while every other stream still has some.
    stream_end_gap_err_s: float = 0.5
    stream_start_lag_warn_s: float = 2.0
    stream_start_lag_err_s: float = 10.0
    camera_coverage_min_frac: float = 0.95
    camera_trailing_gap_warn_s: float = 0.25
    camera_trailing_gap_err_s: float = 0.5
    sync_residual_warn_us: float = 1_000.0
    sync_residual_err_us: float = 50_000.0
    # Frame<->trigger association. See the long note on the anchor failure in
    # _check_sync_report; 3 frames is 75 ms at the 40 fps template rate
    # (100 ms at 30 fps), inside optical_imu_sync.LAG_ERR_US (120 ms) so the
    # two defences agree.
    assign_frames_warn: float = 1.5
    assign_frames_err: float = 3.0
    assign_spread_frames_warn: float = 1.0
    assign_spread_frames_err: float = 3.0
    assign_drift_warn_us_per_s: float = 50.0
    assign_drift_err_us_per_s: float = 500.0
    # Median residual minus the expected trigger-to-SensorTimestamp latency, in
    # periods: the anchor picks the trigger within half a period of the
    # latency-corrected prediction, so a correct association sits near 0 and a
    # one-frame error near +-1. The median gate above (|median| vs 1.5 / 3
    # periods, latency included) cannot see a one-frame error (2026-11-05).
    assign_offset_dev_frames_warn: float = 0.25
    assign_offset_dev_frames_err: float = 0.5
    # Frames written before a mid-take anchor correction keep the wrong label;
    # a few seconds' worth at the start warns, more fails.
    anchor_refined_frames_err: float = 300.0

    _WARN_ERR_PAIRS = (
        ("emg_gap_warn_periods", "emg_gap_err_periods"),
        ("emg_missed_drdy_warn_frac", "emg_missed_drdy_err_frac"),
        ("emg_loss_warn_frac", "emg_loss_err_frac"),
        ("stream_start_lag_warn_s", "stream_start_lag_err_s"),
        ("camera_trailing_gap_warn_s", "camera_trailing_gap_err_s"),
        ("sync_residual_warn_us", "sync_residual_err_us"),
        ("assign_frames_warn", "assign_frames_err"),
        ("assign_spread_frames_warn", "assign_spread_frames_err"),
        ("assign_drift_warn_us_per_s", "assign_drift_err_us_per_s"),
        ("assign_offset_dev_frames_warn", "assign_offset_dev_frames_err"),
    )

    @classmethod
    def from_config(cls, resolved: dict[str, Any] | None) -> "IntegrityThresholds | None":
        """Build from ``host.integrity.thresholds``; None when the block is absent."""
        if not isinstance(resolved, dict):
            return None
        block = integrity_thresholds(resolved)
        fields = {f.name for f in dataclasses.fields(cls)}
        vals = {k: v for k, v in block.items() if k in fields}
        return cls(**vals) if vals else None

    def validated(self) -> tuple["IntegrityThresholds", list[str]]:
        """Self, with inverted warn/err pairs reset to their defaults, plus notes."""
        defaults = type(self)()
        notes: list[str] = []
        fixes: dict[str, float] = {}
        # NaN compares False with everything, so a NaN gate never fires; inf or
        # a negative value is equally meaningless for a loss/offset budget.
        for f_ in dataclasses.fields(self):
            v = getattr(self, f_.name)
            if not np.isfinite(v) or v < 0:
                notes.append(
                    f"host.integrity.thresholds.{f_.name}={v!r} is not a finite, non-negative "
                    f"number -- using the default {getattr(defaults, f_.name):g}"
                )
                fixes[f_.name] = getattr(defaults, f_.name)
        cur = dataclasses.replace(self, **fixes) if fixes else self
        for warn_name, err_name in self._WARN_ERR_PAIRS:
            w, e = getattr(cur, warn_name), getattr(cur, err_name)
            if e < w:
                notes.append(
                    f"host.integrity.thresholds.{err_name}={e:g} is below {warn_name}={w:g}, "
                    "which makes the warning unreachable -- using the defaults for that pair"
                )
                fixes[warn_name] = getattr(defaults, warn_name)
                fixes[err_name] = getattr(defaults, err_name)
        return (dataclasses.replace(self, **fixes) if fixes else self), notes


@dataclass
class IntegrityReport:
    path: Path
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Structured stats gathered while checking, keyed by section
    # ("meta", "emg", "imu_mta", "imu_cra", "triggers", "cameras"). Used by
    # ``format_report`` to render a summary table without re-reading the file.
    summary: dict[str, Any] = field(default_factory=dict)
    # Carried on the report rather than threaded through a dozen signatures:
    # every check already takes `report`, and the gates are read-only here.
    th: "IntegrityThresholds" = field(default_factory=lambda: IntegrityThresholds())

    @property
    def ok(self) -> bool:
        return not self.issues

    def add_error(self, msg: str) -> None:
        self.issues.append(msg)

    def add_warn(self, msg: str) -> None:
        self.warnings.append(msg)


def _load_json_attr(meta: h5py.Group, key: str) -> dict[str, Any] | None:
    raw = meta.attrs.get(key)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        obj = json.loads(str(raw))
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _session_start_ref(f: h5py.File) -> int | None:
    """Where the session proper starts: first trigger, else first camera frame."""
    trig = f.get("triggers")
    if trig is not None and "t_master_us" in trig and trig["t_master_us"].shape[0]:
        return int(trig["t_master_us"][0])
    firsts = []
    cams = f.get("cameras")
    if cams is not None:
        for g in cams.values():
            if isinstance(g, h5py.Group) and "t_master_us" in g and g["t_master_us"].shape[0]:
                firsts.append(int(g["t_master_us"][0]))
    return min(firsts) if firsts else None


def _session_duration_s(meta: h5py.Group, emg_g: h5py.Group | None) -> float | None:
    if "session_duration_s" in meta.attrs:
        return float(meta.attrs["session_duration_s"])
    if emg_g is None or "t_master_us" not in emg_g:
        return None
    t = np.asarray(emg_g["t_master_us"])
    if t.size < 2:
        return None
    return float(t[-1] - t[0]) / 1e6


def _check_hdf5_schema(f: h5py.File, report: IntegrityReport) -> dict[str, Any] | None:
    required_groups = ("meta", "emg", "imu_mta", "imu_cra", "cameras", "triggers", "sync")
    for name in required_groups:
        if name not in f:
            report.add_error(f"missing HDF5 group /{name}")
    meta = f.get("meta")
    if meta is None:
        return None
    for attr in ("config_hash", "mcu_config_json", "host_config_json", "software_version"):
        if attr not in meta.attrs:
            report.add_warn(f"missing /meta/{attr}")
    return meta


def _check_config_hash(meta: h5py.Group, report: IntegrityReport) -> None:
    stored = meta.attrs.get("config_hash")
    if stored is None:
        return
    if isinstance(stored, bytes):
        stored = stored.decode("utf-8")
    stored = str(stored)

    mcu_payload = _load_json_attr(meta, "mcu_config_json")
    host_cfg = _load_json_attr(meta, "host_config_json")
    if mcu_payload is None or host_cfg is None:
        report.add_warn("cannot recompute config_hash — incomplete /meta snapshots")
        return

    reconstructed = {
        "schema_version": int(mcu_payload.get("schema_version", meta.attrs.get("schema_version", 1))),
        "hat": host_cfg.get("hat", {}),
        "mcu": {
            "platform": mcu_payload.get("platform", {}),
            "emg": mcu_payload.get("emg", {}),
            "imu_mta": mcu_payload.get("imu_mta", {}),
            "imu_cra": mcu_payload.get("imu_cra", {}),
            "trigger": mcu_payload.get("trigger", {}),
        },
        "host": host_cfg.get("host", {}),
        "session": host_cfg.get("session", {}),
    }
    expected = config_hash(reconstructed)
    if stored != expected:
        report.add_error(f"config_hash mismatch: stored={stored[:16]}… expected={expected[:16]}…")


# REC_STATS fields that are not event counters: rates, high-water marks and the
# record's own sequence number are reported as they stand at the end.
_MCU_STATS_GAUGES = frozenset({
    "sensor_loop_hz", "comms_loop_hz", "max_drdy_gap_us", "imu_svc_max_us",
    "leadoff_max_us", "poll_max_us", "misalign_max_streak", "stats_seq",
    "t_master_us",
})


def take_scoped_mcu_stats(
    end: dict[str, Any] | None, start: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """``end - start`` for every monotonic REC_STATS counter.

    The firmware resets its counters at different moments: ring pushes/drops
    only at boot, IMU seq gaps/resets never, drdy_count/missed_drdy at stream
    start, tx_drops_* at ads_start. Compared with this take's rows as they are,
    they blamed the take for everything since boot (emg_push_unaccounted on
    every take, BNO085 resets from earlier sessions, a dead IMU hidden because
    it once worked). The supervisor stores a snapshot right after arming the
    streams (``sync.mcu_stats_start``); without it the end values are used as
    before and ``scope`` says so. Differences are modulo 2**32 (u32 wrap); a
    counter that went backwards by less than half the range was reset during
    the take, so its end value is the count since that reset.
    """
    if end is None:
        return None, {"scope": None}
    if not isinstance(start, dict):
        return end, {
            "scope": "since_boot",
            "note": "no mcu_stats_start (older recorder): MCU counters cover "
                    "everything since the HAT booted, not just this take",
        }
    out: dict[str, Any] = dict(end)
    restarted: list[str] = []
    for key, v_end in end.items():
        if key in _MCU_STATS_GAUGES or key not in start:
            continue
        try:
            e, s0 = int(v_end), int(start[key])
        except (TypeError, ValueError):
            continue
        d = (e - s0) & 0xFFFFFFFF
        if d >= 0x80000000:
            d = e
            restarted.append(key)
        out[key] = d
    info: dict[str, Any] = {"scope": "take"}
    if restarted:
        info["reset_during_take"] = sorted(restarted)
    return out, info


def _check_mcu_stats(
    mcu_stats_end: dict[str, Any] | None,
    report: IntegrityReport,
    *,
    emg_rows: int | None,
    emg_on: bool,
    last_t_us: int | None,
    scope: dict[str, Any] | None = None,
) -> None:
    """HAT-side loss counters from the last REC_STATS (``sync.mcu_stats_end``).

    ``mcu_stats_end`` is already take-scoped (see :func:`take_scoped_mcu_stats`)
    when the session has a start snapshot; ``scope`` describes which it is and
    ``emg_rows`` then counts only the rows written after that snapshot.

    Absent stats used to skip every MCU gate silently while the summary showed
    zeros. Now that is a warning and the summary says None (unknown).
    """
    if mcu_stats_end is None:
        report.summary["mcu_stats"] = {"present": False}
        report.add_warn(
            "no mcu_stats_end in /sync: the MCU-side gates (missed DRDY, RDATA framing, "
            "ring and USB tx drops, dead IMU port) could not run -- loss on the HAT is "
            "unknown for this session, not zero"
        )
        return
    st = mcu_stats_end

    def _get(key: str) -> int | None:
        v = st.get(key)
        try:
            return int(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    tx_total = _get("tx_drops")
    tx_emg, tx_imu, tx_other = _get("tx_drops_emg"), _get("tx_drops_imu"), _get("tx_drops_other")
    out: dict[str, Any] = {
        "present": True,
        "stats_seq": _get("stats_seq"),
        "tx_drops": tx_total,
        "tx_drops_emg": tx_emg,
        "tx_drops_imu": tx_imu,
        "tx_drops_other": tx_other,
    }
    scope = scope or {}
    since_boot = scope.get("scope") == "since_boot"
    for key in ("scope", "note", "reset_during_take"):
        if scope.get(key) is not None:
            out["scope_note" if key == "note" else key] = scope[key]
    stamp = _get("t_master_us")
    if stamp is not None and last_t_us is not None:
        out["age_s"] = round((last_t_us - stamp) / 1e6, 3)
        if out["age_s"] > 2.0:
            report.add_warn(
                f"mcu_stats_end is {out['age_s']:.1f} s older than the last sample -- "
                "losses after it are not in the counters"
            )
    report.summary["mcu_stats"] = out

    per_type = any(v is not None for v in (tx_emg, tx_imu, tx_other))
    if not per_type and tx_total:
        report.add_warn(
            f"HAT dropped {tx_total} records at the USB FIFO (tx_drops; firmware without "
            "per-type counters, so which streams lost them is unknown)"
        )
    if tx_emg:
        frac = tx_emg / float((emg_rows or 0) + tx_emg)
        msg = f"HAT dropped {tx_emg} EMG records at the USB FIFO ({frac * 100:.3f}%)"
        if frac > report.th.emg_loss_err_frac:
            report.add_error(msg + f" > {report.th.emg_loss_err_frac * 100:g}% budget")
        else:
            report.add_warn(msg)
    if tx_imu:
        pushed = (_get("imu_mta_push") or 0) + (_get("imu_cra_push") or 0)
        frac = tx_imu / float(pushed + tx_imu) if pushed + tx_imu else 1.0
        msg = f"HAT dropped {tx_imu} IMU records at the USB FIFO ({frac * 100:.3f}%)"
        if frac > report.th.imu_ring_drop_err_frac:
            report.add_error(msg + f" > {report.th.imu_ring_drop_err_frac * 100:g}% budget")
        else:
            report.add_warn(msg)
    if tx_other:
        report.add_warn(f"HAT dropped {tx_other} event/other records at the USB FIFO")

    # Every EMG conversion the firmware pushed is either in the file or counted
    # as dropped somewhere. A remainder means host-side loss -- or stats taken
    # before the stream stopped, hence only a warning.
    push = _get("emg_push")
    if emg_on and push and emg_rows is not None:
        accounted = emg_rows + (_get("emg_drops") or 0) + (tx_emg or 0)
        diff = push - accounted
        out["emg_push_unaccounted"] = int(diff)
        if abs(diff) > max(10, report.th.emg_loss_warn_frac * push):
            report.add_warn(
                f"EMG: firmware pushed {push} conversions, file has {emg_rows} rows "
                f"(+{accounted - emg_rows} counted drops) -- {diff:+d} unaccounted "
                + (
                    "(counters since the HAT booted: no mcu_stats_start, so earlier "
                    "takes are included)"
                    if since_boot else
                    "(host-side loss, or stats taken before the stream stopped)"
                )
            )


def _check_emg_framing(
    mcu_stats_end: dict[str, Any] | None, report: IntegrityReport
) -> None:
    """Fail the session if the MCU saw a slipped ADS1299 RDATA byte stream.

    ``frame_misalign`` counts EMG frames whose status word did not carry the
    ADS1299's fixed ``0b1100`` prefix, i.e. the byte stream lost alignment. Those
    frames are not noisy samples, they are the wrong bytes: a one-byte offset
    scales values by ~256 and rotates data between channels. Firmware now drops
    them, so on a current build this shows up as missing samples rather than as
    corrupt ones — but either way the affected span is unusable and must not pass
    as OK, which is what happened before this check existed.
    """
    if mcu_stats_end is None:
        # Unknown, not zero: a table of zeros would read as a clean MCU.
        report.summary["emg_framing"] = {
            k: None for k in
            ("frame_misalign", "misalign_drops", "misalign_max_streak", "resyncs", "spi_wedges")
        }
        return
    stats = mcu_stats_end
    misalign = int(stats.get("frame_misalign", 0))
    dropped = int(stats.get("emg_misalign_drops", 0))
    resyncs = int(stats.get("emg_resyncs", 0))
    worst = int(stats.get("misalign_max_streak", 0))
    wedges = int(stats.get("spi_wedges", 0))

    report.summary["emg_framing"] = {
        "frame_misalign": misalign,
        "misalign_drops": dropped,
        "misalign_max_streak": worst,
        "resyncs": resyncs,
        "spi_wedges": wedges,
    }

    if misalign > 0:
        detail = f"{misalign} misaligned EMG frames"
        if dropped:
            detail += f", {dropped} dropped"
        if worst:
            detail += f", worst run {worst}"
        if resyncs:
            detail += f", {resyncs} resyncs"
        report.add_error(
            f"ADS1299 RDATA framing slipped: {detail}. Samples in the affected "
            "span are not valid microvolts — check for bench commands issued "
            "during the stream (ads_status/selftest/imu_status) and for SPI "
            "clock/wiring problems."
        )
    if wedges > 0:
        report.add_warn(
            f"SPI DMA wedged {wedges}× during the session (transfer timed out and "
            "was aborted); each event costs at least one EMG sample"
        )


def _leading_hole(
    t_us: np.ndarray, t_ref_us: int | None, min_gap_us: float, *, slack_us: float = 100_000.0
) -> dict[str, Any] | None:
    """Gap(s) that end before the session proper starts (first trigger / frame).

    Every recorded session so far starts with ~40 ms of EMG, then a ~0.8 s hole
    in EMG and IMU that ends just before the first trigger: the USB reads stall
    while the cameras start. It is a start-up artefact, not loss during the
    take, so it is reported once by name and kept out of the continuity, loss
    and rate budgets. Returns ``{"cut": index of first sample kept, ...}``.
    """
    if t_ref_us is None or t_us.size < 3:
        return None
    d = np.diff(t_us.astype(np.int64))
    idx = np.flatnonzero((d > min_gap_us) & (t_us[1:].astype(np.int64) <= int(t_ref_us) + slack_us))
    if not idx.size:
        return None
    last = int(idx[-1])
    return {
        "cut": last + 1,
        "samples_before_s": round(float(t_us[last] - t_us[0]) / 1e6, 3),
        "hole_s": round(float(np.sum(d[idx])) / 1e6, 3),
        "ends_before_ref_s": round((float(t_ref_us) - float(t_us[last + 1])) / 1e6, 3),
        "gaps": int(idx.size),
    }


def _seq_loss(seq: np.ndarray) -> tuple[int, int]:
    """(samples missing, repeated/backwards steps) from a u32 sequence (K1), wrap-safe."""
    if seq.size < 2:
        return 0, 0
    d = np.diff(seq.astype(np.int64)) % (1 << 32)
    forward = (d >= 1) & (d < (1 << 31))
    missing = int(np.sum(d[forward] - 1))
    return missing, int(np.sum(~forward))


def _check_emg(
    emg_g: h5py.Group,
    mcu_payload: dict[str, Any] | None,
    report: IntegrityReport,
    *,
    duration_s: float | None = None,
    mcu_stats_end: dict[str, Any] | None = None,
    host_cfg: dict[str, Any] | None = None,
    t_ref_us: int | None = None,
) -> None:
    if "t_master_us" not in emg_g:
        report.add_warn("no EMG timestamps in session")
        return

    t_all = np.asarray(emg_g["t_master_us"], dtype=np.uint64)
    if t_all.size < 2:
        report.add_warn("EMG sample count < 2")
        return

    if not np.all(np.diff(t_all.astype(np.int64)) > 0):
        report.add_error("EMG t_master_us not strictly monotonic")

    fs = int(emg_g.attrs.get("sample_rate_sps", (mcu_payload or {}).get("emg", {}).get("sample_rate_sps", 1000)))
    if fs <= 0:
        fs = 1000
    expected_dt = 1e6 / fs
    warn_thresh = report.th.emg_gap_warn_periods * expected_dt
    err_thresh = report.th.emg_gap_err_periods * expected_dt

    lead = _leading_hole(t_all, t_ref_us, err_thresh)
    cut = int(lead["cut"]) if lead else 0
    if lead:
        report.add_warn(
            f"EMG leading hole: {lead['samples_before_s'] * 1000:.0f} ms of data, then "
            f"{lead['hole_s']:.2f} s with none, ending {lead['ends_before_ref_s'] * 1000:.0f} ms "
            "before the first trigger (start-up stall while the cameras start) -- excluded "
            "from the EMG continuity, loss and rate checks"
        )
    t_us = t_all[cut:]
    if t_us.size < 2:
        report.add_warn("EMG: no samples after the leading hole")
        return

    dts = np.diff(t_us.astype(np.float64))
    warn_idx = np.where(dts > warn_thresh)[0]
    err_idx = np.where(dts > err_thresh)[0]
    warn_gaps = int(warn_idx.size)
    err_gaps = int(err_idx.size)
    missed = int((mcu_stats_end or {}).get("missed_drdy", 0) or 0)
    if warn_gaps > 0:
        est_missing = int(
            np.sum(np.maximum(0, np.round(dts[warn_idx] / expected_dt) - 1))
        )
        msg = (
            f"EMG continuity: {warn_gaps} jitter gaps > "
            f"{report.th.emg_gap_warn_periods:g} periods "
            f"({warn_thresh:.0f} µs), ~{est_missing} missing samples"
        )
        if missed > 0:
            msg += f"; MCU missed_drdy={missed}"
        report.add_warn(msg)
    if err_gaps > 0:
        est_missing = int(
            np.sum(np.maximum(0, np.round(dts[err_idx] / expected_dt) - 1))
        )
        msg = (
            f"EMG continuity: {err_gaps} gaps > {report.th.emg_gap_err_periods:g} periods "
            f"({err_thresh:.0f} µs), "
            f"~{est_missing} missing samples"
        )
        if missed > 0:
            msg += f"; MCU missed_drdy={missed}"
        report.add_error(msg)

    # Loss accounting over EVERY step, including single missing samples
    # (dt = 2T), which fall below the gap thresholds above and were never
    # counted. With /emg/seq (K1) the count is exact: a seq gap is every
    # conversion lost anywhere after the ADC.
    k = np.rint(dts / expected_dt)
    missing_t = int(np.sum(np.maximum(0.0, k - 1.0)))
    missing_seq: int | None = None
    seq_dups = 0
    if "seq" in emg_g:
        seq = np.asarray(emg_g["seq"], dtype=np.uint32)
        if seq.size == t_all.size:
            missing_seq, seq_dups = _seq_loss(seq[cut:])
        else:
            report.add_error(f"EMG seq rows {seq.size} != t_master_us length {t_all.size}")
    missing = missing_seq if missing_seq is not None else missing_t
    loss_frac = missing / float(t_us.size + missing) if (t_us.size + missing) else 0.0
    basis = "seq" if missing_seq is not None else "timestamps"
    if missing_seq is not None and seq_dups:
        report.add_warn(f"EMG seq: {seq_dups} repeated or backwards steps")
    if loss_frac > 0:
        detail = (
            f"EMG loss: {missing} samples missing ({loss_frac * 100:.3f}%, from {basis}"
            + (f"; timestamps say {missing_t}" if missing_seq is not None else "")
            + ")"
        )
        if loss_frac > report.th.emg_loss_err_frac:
            report.add_error(detail + f" > {report.th.emg_loss_err_frac * 100:g}% budget")
        elif loss_frac > report.th.emg_loss_warn_frac:
            report.add_warn(detail)

    if mcu_stats_end and duration_s and duration_s > 0:
        missed = int(mcu_stats_end.get("missed_drdy", 0))
        drdy = int(mcu_stats_end.get("drdy_count", 0))
        total = drdy + missed
        if total > 0 and missed > 0:
            rate = missed / total
            if rate >= report.th.emg_missed_drdy_err_frac:
                report.add_error(
                    f"MCU missed_drdy rate {rate * 100:.2f}% ({missed}/{total}) — the ADC "
                    "produced conversions the firmware never read, so those samples do not "
                    "exist anywhere; SPI DMA could not keep up with DRDY"
                )
            elif rate > report.th.emg_missed_drdy_warn_frac:
                report.add_warn(
                    f"MCU missed_drdy rate {rate * 100:.3f}% ({missed}/{total}) — "
                    "SPI DMA still busy when DRDY arrived"
                )

    _check_emg_framing(mcu_stats_end, report)

    span_s = float(t_us[-1] - t_us[0]) / 1e6
    measured_fs = None
    if span_s > 0.1:
        measured_fs = (t_us.size - 1) / span_s
        rel_err = abs(measured_fs - fs) / fs
        if rel_err > 0.01:
            report.add_error(
                f"EMG rate {measured_fs:.1f} Hz differs from configured {fs} Hz by {rel_err * 100:.2f}%"
            )

    report.summary["emg"] = {
        "samples": int(t_all.size),
        "span_s": round(span_s, 3),
        "configured_hz": fs,
        "measured_hz": round(measured_fs, 2) if measured_fs is not None else None,
        "gaps_warn": warn_gaps,
        "gaps_err": err_gaps,
        "missing": int(missing),
        "missing_from_timestamps": int(missing_t),
        "missing_from_seq": missing_seq,
        "loss_frac": round(loss_frac, 6),
        "leading_hole": lead,
    }

    _check_emg_datasets(emg_g, report)

    _check_emg_signal_quality(emg_g, mcu_payload, report, host_cfg=host_cfg)

    emg_meta = (mcu_payload or {}).get("emg") or {}
    leadoff = emg_meta.get("leadoff") or {}
    # DC lead-off during recording is the default and is safe: its constant
    # current only offsets connected inputs by a DC term the 20 Hz high-pass
    # removes, and pulls disconnected inputs to a rail (which is the point).
    # The AC tone is the unsafe case -- it sits at 31.2 Hz, inside the EMG
    # band, and directly contaminates the recorded signal if left on.
    if leadoff.get("enabled") and leadoff.get("during_recording") is True and leadoff.get("ac_tone"):
        report.add_warn("AC leadoff tone active during recording (contaminates the EMG band)")


def _check_emg_signal_quality(
    emg_g: h5py.Group,
    mcu_payload: dict[str, Any] | None,
    report: IntegrityReport,
    *,
    host_cfg: dict[str, Any] | None = None,
) -> None:
    """Fail a session whose EMG is mains, or whose DRL amplifier is saturated.

    Every other EMG check here judges rates and continuity, so a recording that
    was 98% line interference with the bias amplifier clipping passed as OK.
    See jtsmm/integrity/emg_signal.py for the two measures and why they need
    both a share and an amplitude to fire.
    """
    if "samples" not in emg_g:
        return
    emg_meta = (mcu_payload or {}).get("emg") or {}
    fs = float(emg_g.attrs.get("sample_rate_sps", emg_meta.get("sample_rate_sps", 0)) or 0)
    lsb = float(emg_g.attrs.get("lsb_uv", 0) or 0)
    if fs <= 0 or lsb <= 0:
        return

    # Judge only channels that were actually powered on.
    pd_mask = int(emg_meta.get("pd_channels", 0) or 0)
    active = [c for c in range(6) if not (pd_mask >> c) & 1]
    names = _emg_channel_names(emg_g)
    leadoff = emg_meta.get("leadoff") or {}
    bias = emg_meta.get("bias") or {}
    # CH8 only reads back the bias amplifier when bias drive is on and MUX8 is
    # BIAS_DRN; the isolated probe_drl session routes a plain electrode there.
    drl_meaningful = bool(bias.get("enabled")) and not bool(leadoff.get("probe_drl"))
    line_cfg = ((host_cfg or {}).get("host") or {}).get("emg_check") or {}
    line_hz = float(line_cfg["line_hz"]) if "line_hz" in line_cfg else None

    per_ch, lsb_note = emg_channel_lsb(emg_g, int(emg_g["samples"].shape[1]))
    res = analyse_emg_signal(
        emg_g["samples"], fs_hz=fs, lsb_uv=lsb, active_channels=active,
        drl_meaningful=drl_meaningful, line_hz=line_hz, lsb_uv_per_channel=per_ch,
    )
    if not res.get("windows"):
        return

    summary: dict[str, Any] = {
        "line_hz": res.get("line_hz"),
        "windows": res["windows"],
        "channels": {
            names[c]: {
                "mains_share": round(v["median_mains_share"], 3),
                "band_rms_uv": round(v["median_band_rms_uv"], 1),
                "affected_mains_share": round(v["bad_median_mains_share"], 3),
                "affected_band_rms_uv": round(v["bad_median_band_rms_uv"], 1),
                "bad_window_frac": round(v["bad_window_frac"], 3),
            }
            for c, v in res["channels"].items()
        },
    }
    if res.get("drl"):
        summary["drl_ac_mv"] = round(res["drl"]["ac_mv"], 1)
        summary["drl_rail_frac"] = round(res["drl"]["rail_frac"], 3)
        # "n/a" = CH8 railed (common: IN8P is unwired), not a finding.
        summary["drl_status"] = res["drl"]["status"]
    if lsb_note:
        summary["lsb_note"] = lsb_note
    report.summary["emg_signal"] = summary

    verdict = res["verdict"]
    if verdict["drl_warn"]:
        drl = res["drl"]
        report.add_warn(
            f"DRL readback: CH8 swings {drl['ac_mv']:.1f} mV RMS ({drl['status']}; warn "
            f"{DRL_AC_WARN_MV:g} / high {DRL_AC_ERR_MV:g} mV) -- the bias loop may be driven by "
            "common-mode instead of rejecting it. Check the DRL electrode site and its lead. "
            "Informational until the driven loop is bench-verified (TEST_INSTRUCTIONS Step 4-B)."
        )

    for c, v in res["channels"].items():
        frac = v["bad_window_frac"]
        # One bad window in 24 is a transient (a cable touched), not a finding;
        # the analysis already grades by MAINS_BAD_WINDOW_FRAC_WARN / _ERR.
        if frac <= MAINS_BAD_WINDOW_FRAC_WARN:
            continue
        share = v["bad_median_mains_share"]
        rms = v["bad_median_band_rms_uv"]
        line = res.get("line_hz") or 50.0
        msg = (
            f"EMG {names[c]}: mains interference dominates {frac * 100:.0f}% of the session "
            f"({share * 100:.0f}% of 20-450 Hz power at {line:.0f} Hz and harmonics, "
            f"{rms:.0f} µV RMS in-band vs 5-20 µV for resting skin EMG). These samples are "
            "interference, not muscle activity: re-prep the electrode and reference, and "
            "remove mains-coupled equipment."
        )
        if frac >= MAINS_BAD_WINDOW_FRAC_ERR:
            report.add_error(msg)
        else:
            report.add_warn(msg)


def emg_channel_lsb(emg_g: h5py.Group, n_channels: int = 8) -> tuple[list[float] | None, str | None]:
    """Per-column LSB (µV) of the counts in ``/emg/samples``, plus a note.

    Current files carry ``lsb_uv_per_channel`` (K5). Older ones only have the
    scalar muscle ``lsb_uv``, but CH7/CH8 always ran at gain 24, so their true
    LSB is the fixed-gain one; returned corrected, with a note saying so.
    """
    per = emg_g.attrs.get("lsb_uv_per_channel")
    if per is not None:
        vals = [float(v) for v in np.asarray(per, dtype=np.float64).ravel()]
        if len(vals) >= n_channels and all(np.isfinite(v) and v > 0 for v in vals):
            return vals[:n_channels], None
    gain = emg_g.attrs.get("gain")
    try:
        g = int(gain)
    except (TypeError, ValueError):
        g = 0
    if g <= 0:
        lsb = float(emg_g.attrs.get("lsb_uv", 0) or 0)
        return ([lsb] * n_channels if lsb > 0 else None), None
    return channel_lsb_uv(g, n_channels), (
        "no lsb_uv_per_channel attr (file predates it): CH7/CH8 re-scaled with their "
        f"fixed gain {AUX_CHANNEL_GAIN} instead of the muscle gain {g}; this file's "
        "samples_uv for those columns is off by that ratio"
    )


def _emg_channel_names(emg_g: h5py.Group) -> list[str]:
    raw = emg_g.attrs.get("channel_names")
    out: list[str] = []
    if raw is not None:
        for item in np.asarray(raw).ravel():
            out.append(item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item))
        out = montage_corrected(out)[0]          # as cabled (an03 §1)
    while len(out) < 8:
        out.append(f"ch{len(out) + 1}")
    return out


def _check_emg_datasets(emg_g: h5py.Group, report: IntegrityReport) -> None:
    """Shape / calibration consistency for /emg datasets."""
    t_us = np.asarray(emg_g["t_master_us"], dtype=np.uint64) if "t_master_us" in emg_g else None
    samples = np.asarray(emg_g["samples"]) if "samples" in emg_g else None
    if samples is None:
        report.add_warn("EMG samples dataset missing")
        return
    if samples.ndim != 2 or samples.shape[1] != 8:
        report.add_error(f"EMG samples shape {samples.shape} expected (N, 8)")
    if t_us is not None and samples.shape[0] != t_us.size:
        report.add_error(
            f"EMG samples rows {samples.shape[0]} != t_master_us length {t_us.size}"
        )
    if np.any(samples < -(1 << 23)) or np.any(samples >= (1 << 23)):
        report.add_error("EMG samples outside signed 24-bit range")

    if "leadoff_status" in emg_g:
        lo = np.asarray(emg_g["leadoff_status"])
        if lo.ndim != 2 or lo.shape[1] != 2:
            report.add_error(f"leadoff_status shape {lo.shape} expected (N, 2)")
        elif t_us is not None and lo.shape[0] != t_us.size:
            report.add_error(
                f"leadoff_status rows {lo.shape[0]} != t_master_us length {t_us.size}"
            )
        elif lo.size and int(np.max(lo[:, 0] | lo[:, 1]) & 0xC0) != 0:
            report.add_warn("leadoff_status has bits 6–7 set (unexpected for muscle-only masks)")

    if "samples_uv" not in emg_g:
        report.add_warn("EMG samples_uv dataset missing")
        return
    uv = np.asarray(emg_g["samples_uv"])
    if uv.shape != samples.shape:
        report.add_error(f"samples_uv shape {uv.shape} != samples shape {samples.shape}")
        return
    lsb = emg_g.attrs.get("lsb_uv")
    if lsb is None:
        report.add_warn("EMG lsb_uv attr missing; cannot verify samples_uv calibration")
        return
    lsb_f = float(lsb)
    if not np.isfinite(lsb_f) or lsb_f <= 0:
        report.add_error(f"EMG lsb_uv attr invalid: {lsb_f}")
        return
    # samples_uv is written per column (K5); older files wrote every column
    # with the scalar muscle LSB, and are checked against what they wrote.
    scale: Any = lsb_f
    per = emg_g.attrs.get("lsb_uv_per_channel")
    if per is not None:
        per_a = np.asarray(per, dtype=np.float64).ravel()
        if per_a.size != samples.shape[1] or not np.all(np.isfinite(per_a)) or np.any(per_a <= 0):
            report.add_error(f"EMG lsb_uv_per_channel attr invalid: {per_a.tolist()}")
            return
        scale = per_a[None, :]
    expect = samples.astype(np.float64) * scale
    max_abs = float(np.max(np.abs(uv.astype(np.float64) - expect))) if uv.size else 0.0
    if max_abs > SAMPLES_UV_ABS_TOL_UV:
        report.add_error(
            f"samples_uv inconsistent with samples*lsb_uv (max_abs={max_abs:.3g} µV)"
        )


def _imu_ring_drop_rate(label: str, mcu_stats_end: dict[str, Any] | None) -> float | None:
    """Firmware-counted IMU loss for one slot (true sample loss, not rate quantization).

    Ring overflow drops plus, when the firmware reports them (K2), SH-2
    sequence gaps -- reports the sensor sent that never reached the ring.
    """
    if not mcu_stats_end:
        return None
    slot = "mta" if label == "imu_mta" else "cra"
    push = int(mcu_stats_end.get(f"imu_{slot}_push", 0) or 0)
    drops = int(mcu_stats_end.get(f"imu_{slot}_drops", 0) or 0)
    drops += int(mcu_stats_end.get(f"imu_{slot}_seq_gaps", 0) or 0)
    total = push + drops
    return (drops / total) if total > 0 else None


def _imu_rate_tolerance(report_name: str) -> float:
    return IMU_RATE_TOL.get(report_name, IMU_RATE_TOL_DEFAULT)


def _check_imu_slot(
    imu_g: h5py.Group,
    label: str,
    mcu_payload: dict[str, Any] | None,
    report: IntegrityReport,
    *,
    mcu_stats_end: dict[str, Any] | None = None,
    t_ref_us: int | None = None,
) -> None:
    slot_cfg = ((mcu_payload or {}).get(label) or {})
    if not slot_cfg.get("enabled", True):
        return

    slot = "mta" if label == "imu_mta" else "cra"
    # An enabled slot that pushed nothing is a dead sensor/port, not a "legacy
    # session". The firmware counter is unambiguous, and absent on older files
    # (where this check simply does not run).
    push = (mcu_stats_end or {}).get(f"imu_{slot}_push")
    if push is not None and int(push) == 0:
        aborts = (mcu_stats_end or {}).get(f"imu_{slot}_i2c_aborts")
        if aborts is None and slot == "mta":
            total = (mcu_stats_end or {}).get("imu_i2c_aborts")
            cra = (mcu_stats_end or {}).get("imu_cra_i2c_aborts")
            if total is not None and cra is not None:
                aborts = int(total) - int(cra)
        detail = f", {int(aborts)} I2C aborts on that bus" if aborts is not None else ""
        report.add_error(
            f"{label}: enabled but produced no samples for the whole session{detail} -- "
            f"the sensor on that port never answered. Check the cable, the PCA9546 mux "
            f"channel and power for the {slot.upper()} port (`imu_status` shows per-slot "
            f"aborts); set mcu.{label}.enabled: false to record without it."
        )
        report.summary.setdefault("imu", {})[label] = {"quat_samples": 0, "pushes": 0}
        return

    # Applied reports map -> expected HDF5 datasets. A dataset must be present for
    # every enabled report; extra/legacy sessions only warn (older quat-only files
    # predate the per-stream storage layout).
    reports = slot_cfg.get("reports") or {}
    report_to_dataset = {
        "accelerometer": "accel",
        "gyroscope": "gyro",
        "magnetometer": "mag",
        "linear_acceleration": "linear_accel",
    }
    fusion_report_names = {
        "rotation_vector", "game_rotation_vector", "geomagnetic_rotation_vector",
        "gyro_integrated_rotation_vector", "arvr_stabilized_rotation_vector",
        "arvr_stabilized_game_rotation_vector",
    }
    enabled = {n for n, c in reports.items() if c.get("enabled", True)}
    fusion_enabled = enabled & fusion_report_names
    if fusion_enabled and "quat" not in imu_g:
        report.add_warn(f"{label}: fusion report enabled but no quat dataset (legacy session?)")
    for rname, dsname in report_to_dataset.items():
        if rname in enabled and dsname not in imu_g:
            report.add_warn(f"{label}: report {rname} enabled but /{label}/{dsname} missing")
        # Per-stream rate check (motion vectors), when present.
        if dsname in imu_g and f"t_master_us_{dsname}" in imu_g:
            _check_stream_rate(
                imu_g, label, dsname, rname, reports.get(rname, {}), report,
            )

    gap_summary: dict[str, Any] = {}
    for name in sorted(imu_g.keys()):
        if not str(name).startswith("t_master_us_"):
            continue
        stream = str(name)[len("t_master_us_"):]
        g = _check_imu_gaps(
            np.asarray(imu_g[name], dtype=np.uint64), label, stream, report, t_ref_us=t_ref_us
        )
        if g is not None:
            gap_summary[stream] = g
    if gap_summary:
        report.summary.setdefault("imu_gaps", {})[label] = gap_summary
        holes = [g["leading_hole_s"] for g in gap_summary.values() if g.get("leading_hole_s")]
        if holes:
            report.add_warn(
                f"{label} leading hole: {max(holes):.2f} s with no data before the first "
                "trigger (start-up stall while the cameras start) -- excluded from the gap check"
            )

    ring_drop = _imu_ring_drop_rate(label, mcu_stats_end)
    if ring_drop is not None and ring_drop > report.th.imu_ring_drop_err_frac:
        report.add_error(
            f"{label} ring drop rate {ring_drop * 100:.2f}% exceeds "
            f"{report.th.imu_ring_drop_err_frac * 100:.1f}% budget (ring overflow + SH-2 "
            "sequence gaps counted by the firmware)"
        )
    resets = int((mcu_stats_end or {}).get(f"imu_{slot}_resets", 0) or 0)
    if resets:
        report.add_warn(
            f"{label}: BNO085 reset {resets}x during the session -- samples around each "
            "reset are missing and fusion restarted (orientation may jump)"
        )

    t_key = "t_master_us_quat"
    if t_key not in imu_g:
        if not fusion_enabled:
            return
        report.add_warn(f"no IMU quaternion timestamps for {label}")
        return

    t_us = np.asarray(imu_g[t_key], dtype=np.uint64)
    if t_us.size < 2:
        report.summary.setdefault("imu", {})[label] = {"quat_samples": int(t_us.size)}
        return

    # Prefer the resolved fusion report's rate; fall back to fusion.rate_hz (390).
    fusion = slot_cfg.get("fusion") or {}
    fusion_report_name: str | None = None
    expected_hz = None
    for rname in fusion_enabled:
        fusion_report_name = rname
        expected_hz = float(reports[rname].get("rate_hz", 0)) or None
        if expected_hz:
            break
    if expected_hz is None:
        expected_hz = float(fusion.get("rate_hz", 390))
        if fusion_enabled:
            fusion_report_name = next(iter(fusion_enabled))
    duration_s = float(t_us[-1] - t_us[0]) / 1e6
    if duration_s <= 0.1:
        return

    measured_hz = (t_us.size - 1) / duration_s
    tol = _imu_rate_tolerance(fusion_report_name or "game_rotation_vector")
    rel_err = abs(measured_hz - expected_hz) / expected_hz
    if rel_err > tol:
        report.add_warn(
            f"{label} fusion rate {measured_hz:.0f} Hz vs configured "
            f"{expected_hz:.0f} Hz ({rel_err * 100:.1f}% off)"
        )

    if not np.all(np.diff(t_us.astype(np.int64)) > 0):
        report.add_warn(f"{label} timestamps not strictly monotonic")

    report.summary.setdefault("imu", {})[label] = {
        "quat_samples": int(t_us.size),
        "configured_hz": round(expected_hz, 1),
        "measured_hz": round(measured_hz, 1),
    }


def _check_imu_gaps(
    t: np.ndarray, label: str, stream: str, report: IntegrityReport, *, t_ref_us: int | None
) -> dict[str, Any] | None:
    """Mid-stream silences in one IMU stream.

    The rate check only compares sample count with span, so take1's imu_mta,
    silent for 26.9 s (7 % of the take), passed as "rate 324 vs 390 Hz". IMU
    timestamps are the MCU's poll time, so they jitter by a period or so and
    the hub throttles below the configured rate; gaps are therefore measured
    against the stream's own median step, tolerantly (``imu_gap_warn_periods``).
    """
    if t.size < 3:
        return None
    ti = t.astype(np.int64)
    d_all = np.diff(ti).astype(np.float64)
    med = float(np.median(d_all[d_all > 0])) if np.any(d_all > 0) else 0.0
    if med <= 0:
        return None
    gap_us = report.th.imu_gap_warn_periods * med
    lead = _leading_hole(t, t_ref_us, gap_us)
    cut = int(lead["cut"]) if lead else 0
    d = d_all[cut:]
    if d.size < 2:
        return None
    gaps = np.flatnonzero(d > gap_us)
    missing = int(np.sum(np.maximum(0.0, np.rint(d[gaps] / med) - 1.0))) if gaps.size else 0
    n = int(t.size - cut)
    loss = missing / float(n + missing) if (n + missing) else 0.0
    longest = float(np.max(d[gaps])) / 1e6 if gaps.size else 0.0
    out = {
        "median_dt_us": round(med, 1),
        "gaps": int(gaps.size),
        "missing": missing,
        "loss_frac": round(loss, 5),
        "longest_gap_s": round(longest, 3),
        "leading_hole_s": lead["hole_s"] if lead else None,
    }
    if not gaps.size:
        return out
    t0 = float(ti[0])
    i_long = int(gaps[np.argmax(d[gaps])]) + cut
    where = f"longest {longest:.2f} s at t={(float(ti[i_long]) - t0) / 1e6:.1f} s"
    msg = (
        f"{label}/{stream}: {gaps.size} gap(s) > {report.th.imu_gap_warn_periods:g}x the "
        f"median step ({med / 1000:.1f} ms), ~{missing} samples missing "
        f"({loss * 100:.2f}%), {where}"
    )
    if longest > report.th.stream_end_gap_err_s or loss > report.th.imu_ring_drop_err_frac:
        report.add_error(
            msg + f" -- over the budget ({report.th.stream_end_gap_err_s:g} s longest gap, "
            f"{report.th.imu_ring_drop_err_frac * 100:g}% loss); that span has no data for this "
            "IMU even though its rate check may pass"
        )
    else:
        report.add_warn(msg)
    return out


def _check_stream_rate(
    imu_g: h5py.Group,
    label: str,
    dsname: str,
    report_name: str,
    report_cfg: dict[str, Any],
    report: IntegrityReport,
) -> None:
    t = np.asarray(imu_g[f"t_master_us_{dsname}"], dtype=np.uint64)
    if t.size < 2:
        return
    if not np.all(np.diff(t.astype(np.int64)) > 0):
        report.add_warn(f"{label}/{dsname} timestamps not strictly monotonic")
    expected_hz = float(report_cfg.get("rate_hz", 0) or 0)
    if expected_hz <= 0:
        return
    duration_s = float(t[-1] - t[0]) / 1e6
    if duration_s <= 0.1:
        return
    measured_hz = (t.size - 1) / duration_s
    tol = _imu_rate_tolerance(report_name)
    rel_err = abs(measured_hz - expected_hz) / expected_hz
    if rel_err > tol:
        report.add_warn(
            f"{label}/{dsname} rate {measured_hz:.0f} Hz vs configured "
            f"{expected_hz:.0f} Hz ({rel_err * 100:.1f}% off)"
        )


def _check_triggers(
    trig_g: h5py.Group,
    mcu_payload: dict[str, Any] | None,
    duration_s: float | None,
    report: IntegrityReport,
) -> None:
    if "trigger_id" not in trig_g:
        report.add_warn("no trigger events recorded")
        return

    trig_id = np.asarray(trig_g["trigger_id"], dtype=np.uint64)
    report.summary["triggers"] = {"count": int(trig_id.size)}
    if trig_id.size < 2:
        return

    id_diff = np.diff(trig_id.astype(np.int64))
    if not np.all(id_diff > 0):
        report.add_error("trigger_id not strictly increasing")
    elif not np.all(id_diff == 1):
        bad = np.flatnonzero(id_diff != 1)
        missing = int(np.sum(id_diff[bad] - 1))
        first = ", ".join(f"{int(trig_id[i])}->{int(trig_id[i + 1])}" for i in bad[:5])
        report.add_error(
            f"trigger_id not contiguous (+1); {bad.size} bad steps, {missing} ids missing "
            f"(e.g. {first}) -- TRIGGER_EVENTs lost over USB or edges lost to IRQ latency"
        )

    if "t_master_us" in trig_g:
        t_us = np.asarray(trig_g["t_master_us"], dtype=np.uint64)
        if t_us.size != trig_id.size:
            report.add_error(
                f"trigger t_master_us length {t_us.size} != trigger_id length {trig_id.size}"
            )
        elif t_us.size >= 2 and not np.all(np.diff(t_us.astype(np.int64)) > 0):
            report.add_error("trigger t_master_us not strictly monotonic")
        else:
            _check_trigger_time_vs_id(trig_id, t_us, report)

    rate = trig_g.attrs.get("rate_hz")
    if rate is None:
        rate = ((mcu_payload or {}).get("trigger") or {}).get("rate_hz")
    if rate is None or float(rate) <= 0:
        return

    # Measure the actual trigger rate over the trigger data window (same method
    # as the EMG rate check). Using the wall-clock session duration would over-
    # expect whenever startup samples are discarded at capture start.
    span_s = None
    if "t_master_us" in trig_g:
        t_us = np.asarray(trig_g["t_master_us"], dtype=np.uint64)
        if t_us.size >= 2:
            span_s = float(t_us[-1] - t_us[0]) / 1e6
    if span_s is None or span_s <= 0.1:
        span_s = float(duration_s) if duration_s else None
    if span_s is None or span_s <= 0:
        return

    expected = span_s * float(rate)
    rel_err = abs(trig_id.size - expected) / max(expected, 1.0)
    if rel_err > 0.05:
        report.add_warn(
            f"trigger count {trig_id.size} vs expected ~{expected:.0f} ({rel_err * 100:.1f}% off)"
        )

    report.summary["triggers"].update({
        "span_s": round(span_s, 3),
        "configured_hz": float(rate),
        "measured_hz": round(trig_id.size / span_s, 2) if span_s > 0 else None,
    })


def _check_trigger_time_vs_id(
    trig_id: np.ndarray, t_us: np.ndarray, report: IntegrityReport
) -> None:
    """Fail on trigger edges lost without an id gap.

    Before firmware advanced trigger_id per elapsed PIO period (K4), an edge
    lost to IRQ latency left the ids contiguous while the time step was two
    periods. Sequence association then labels every later frame one pulse
    early (~3/4 of a take's frames shifted by exactly one period), and no other
    gate sees it. Newer firmware turns the same event into an id gap, which the
    contiguity check above reports; this one stays for older files and as a
    cross-check.
    """
    if trig_id.size < 3:
        return
    d_id = np.diff(trig_id.astype(np.int64))
    d_t = np.diff(t_us.astype(np.float64))
    ok = (d_id == 1) & (d_t > 0)
    if int(ok.sum()) < 2:
        return
    period = float(np.median(d_t[ok]))
    if period <= 0:
        return
    k = np.rint(d_t / period).astype(np.int64)
    bad = np.flatnonzero((d_id > 0) & (k > d_id))
    report.summary.setdefault("triggers", {})["edges_lost_without_id_gap"] = int(
        np.sum(k[bad] - d_id[bad])
    ) if bad.size else 0
    if not bad.size:
        return
    lost = int(np.sum(k[bad] - d_id[bad]))
    t0 = float(t_us[0])
    where = ", ".join(
        f"after trigger_id {int(trig_id[i])} (t={(float(t_us[i]) - t0) / 1e6:.2f} s, "
        f"{float(d_t[i]) / 1000:.1f} ms step)"
        for i in bad[:5]
    )
    more = f" and {bad.size - 5} more" if bad.size > 5 else ""
    report.add_error(
        f"triggers: {lost} trigger edge(s) lost without an id gap at {bad.size} place(s): "
        f"{where}{more} -- camera frames after each are labelled one pulse early "
        "(firmware without per-period trigger ids, K4)"
    )


def _trigger_period_s(trig_t: np.ndarray) -> float | None:
    """Median trigger spacing in seconds (None when not measurable)."""
    if trig_t.size < 3:
        return None
    d = np.diff(trig_t.astype(np.float64))
    d = d[d > 0]
    return float(np.median(d)) / 1e6 if d.size else None


def _assign_offset_frames(sync_report: dict[str, Any] | None, cam_name: str) -> float | None:
    """``assign_residual.median_frames`` for one camera, if the recorder stored it."""
    cams = (sync_report or {}).get("cameras")
    if not isinstance(cams, dict):
        return None
    entry = cams.get(cam_name)
    ar = entry.get("assign_residual") if isinstance(entry, dict) else None
    if not isinstance(ar, dict) or ar.get("median_frames") is None:
        return None
    return float(ar["median_frames"])


def _check_trigger_camera_pairing(
    f: h5py.File,
    report: IntegrityReport,
    *,
    sync_report: dict[str, Any] | None = None,
    host_cfg: dict[str, Any] | None = None,
    skipped_cameras: set[str] | None = None,
) -> None:
    trig_g = f.get("triggers")
    cam_root = f.get("cameras")
    if trig_g is None or cam_root is None:
        return
    if "trigger_id" not in trig_g or "t_master_us" not in trig_g:
        return

    trig_t = np.asarray(trig_g["t_master_us"], dtype=np.int64)
    if trig_t.size == 0:
        return
    trig_span_s = float(trig_t[-1] - trig_t[0]) / 1e6 if trig_t.size >= 2 else 0.0

    sync_report = sync_report or {}
    cam_fits = sync_report.get("cameras") if isinstance(sync_report.get("cameras"), dict) else {}
    drops = sync_report.get("drops") if isinstance(sync_report.get("drops"), dict) else {}

    configured = []
    host_cameras = ((host_cfg or {}).get("host") or {}).get("cameras") or {}
    for entry in host_cameras.get("list") or []:
        cid = str(entry.get("id", "")).strip()
        if cid:
            configured.append(cid)

    present = list(cam_root.keys())
    skipped_cameras = skipped_cameras or set()
    for cid in configured:
        if cid not in cam_root:
            if cid in skipped_cameras:
                report.add_warn(
                    f"configured camera {cid} skipped at startup — no picamera2 "
                    "device detected (sensor not connected/enumerated)"
                )
                continue
            # Unconnected cameras are allowed; only warn when sync recorded drops.
            dropped = int(drops.get(cid, 0) or 0)
            if dropped > 0:
                report.add_warn(
                    f"configured camera {cid} missing from /cameras "
                    f"(sync.drops[{cid}]={dropped}; sensor likely absent)"
                )

    cam_summary: dict[str, Any] = report.summary.setdefault("cameras", {})
    for cam_name, cam_g in cam_root.items():
        if "trigger_id" not in cam_g or "t_master_us" not in cam_g:
            report.add_warn(f"camera {cam_name}: missing trigger_id or t_master_us")
            continue
        cam_trig = np.asarray(cam_g["trigger_id"], dtype=np.int64)
        cam_t = np.asarray(cam_g["t_master_us"], dtype=np.int64)
        cam_summary[cam_name] = {"frames": int(cam_trig.size)}
        if cam_trig.size == 0:
            report.add_warn(f"camera {cam_name}: no frames recorded")
            continue
        if cam_trig.size != cam_t.size:
            report.add_error(
                f"camera {cam_name}: trigger_id length {cam_trig.size} != "
                f"t_master_us length {cam_t.size}"
            )
            continue
        if not np.all(np.diff(cam_t) > 0):
            nnon = int(np.sum(np.diff(cam_t) <= 0))
            report.add_error(
                f"camera {cam_name}: t_master_us not strictly increasing ({nnon} non-positive)"
            )

        # Coverage = frames present / pulses the camera should have seen in its
        # own span (trigger-id range, so TRIGGER_EVENTs lost over USB still
        # count as expected frames). The time span alone read 99.8 % on take1
        # while 12.7 % of the frames were missing.
        period_s = _trigger_period_s(trig_t)
        expected = int(cam_trig[-1] - cam_trig[0] + 1)
        if expected > 0 and cam_trig.size >= 2:
            coverage = cam_trig.size / float(expected)
            cam_summary[cam_name]["coverage_frac"] = round(coverage, 4)
            cam_summary[cam_name]["expected_frames"] = expected
            if coverage < report.th.camera_coverage_min_frac:
                report.add_error(
                    f"camera {cam_name}: {cam_trig.size} frames for {expected} trigger pulses in "
                    f"its span ({coverage * 100:.1f}% < "
                    f"{report.th.camera_coverage_min_frac * 100:.0f}%) -- frames dropped by the "
                    "camera pipeline or by the association"
                )

        # Camera stopped before the triggers did -- catches supervisor starvation.
        if trig_span_s > 0.5 and cam_t.size >= 2:
            trailing_gap_s = float(trig_t[-1] - cam_t[-1]) / 1e6
            cam_summary[cam_name]["trailing_gap_s"] = round(trailing_gap_s, 3)
            if trailing_gap_s >= report.th.camera_trailing_gap_err_s:
                # A trailing gap equal to the labelling offset is that offset,
                # not starvation: every frame carries a trigger time N pulses
                # early, so the last frame sits N pulses short of the last
                # pulse while the camera in fact covered the whole window.
                off_frames = _assign_offset_frames(sync_report, cam_name)
                gap_frames = trailing_gap_s / (period_s or 1e9)
                if off_frames is not None and abs(abs(off_frames) - gap_frames) < 2.0:
                    report.add_error(
                        f"camera {cam_name}: trailing gap {trailing_gap_s:.3f}s vs triggers "
                        f"— matches the {off_frames:+.1f}-frame labelling offset below "
                        "(frame↔trigger association, not poll starvation)"
                    )
                else:
                    report.add_error(
                        f"camera {cam_name}: trailing gap {trailing_gap_s:.3f}s vs triggers "
                        f"(>{report.th.camera_trailing_gap_err_s:.1f}s) — likely poll starvation"
                    )
            elif trailing_gap_s >= report.th.camera_trailing_gap_warn_s:
                report.add_warn(
                    f"camera {cam_name}: trailing gap {trailing_gap_s:.3f}s vs triggers"
                )

        # Frame labels against the camera's own clock, over the WHOLE take. The
        # stored t_master_us is the labelled trigger's time, so between two
        # frames it must advance as the SensorTimestamps do. A step that
        # disagrees by whole periods is a slip in the association, and every
        # later frame inherits it. (The recorder's residual_rms_us only covers
        # its last 30 frames, so it is kept as context only.)
        fit = cam_fits.get(cam_name) if isinstance(cam_fits, dict) else None
        if isinstance(fit, dict) and "residual_rms_us" in fit:
            cam_summary[cam_name]["sync_residual_recorder_us"] = round(
                float(fit["residual_rms_us"]), 1
            )
        rms: float | None = None
        sens = (
            np.asarray(cam_g["sensor_timestamp_ns"], dtype=np.int64)
            if "sensor_timestamp_ns" in cam_g else None
        )
        if sens is not None and sens.size == cam_t.size and cam_t.size >= 3 and np.any(sens):
            x = (sens - sens[0]).astype(np.float64) / 1000.0
            y = (cam_t - cam_t[0]).astype(np.float64)
            if float(np.ptp(x)) > 0:
                rms, slew_ppm = _label_residual_rms(x, y, period_s)
                if slew_ppm is not None:
                    cam_summary[cam_name]["host_clock_slew_ppm"] = round(slew_ppm, 0)
                    if abs(slew_ppm) >= _HOST_CLOCK_SLEW_WARN_PPM:
                        report.add_warn(
                            f"camera {cam_name}: the Pi clock ran up to {slew_ppm:+.0f} ppm "
                            "off the MCU clock during the take (NTP correcting it) -- frame "
                            "labels are unaffected (checked step by step), but times mapped "
                            "from the Pi clock (viewer notes, phase edges) can be off by ms; "
                            "record once the Pi clock has settled"
                        )
            if period_s:
                steps = np.rint((np.diff(y) - np.diff(x)) / (period_s * 1e6))
                bad = np.flatnonzero(steps != 0)
                cam_summary[cam_name]["label_steps"] = int(bad.size)
                if bad.size:
                    where = ", ".join(
                        f"{int(steps[i]):+d} at frame {int(i) + 1} "
                        f"(t={(cam_t[i + 1] - trig_t[0]) / 1e6:.1f} s)"
                        for i in bad[:5]
                    )
                    report.add_warn(
                        f"camera {cam_name}: frame labels jump against the camera's own clock "
                        f"{bad.size}x ({where}{' ...' if bad.size > 5 else ''}) -- the "
                        "frame↔trigger association changed there by that many periods, so "
                        "the frames on one side of each jump are mislabelled by it (a jump at "
                        "frame 2 means the first frame was anchored on the wrong pulse)"
                    )
        elif isinstance(fit, dict) and "residual_rms_us" in fit:
            rms = float(fit["residual_rms_us"])  # older files without SensorTimestamps
        if rms is not None:
            cam_summary[cam_name]["sync_residual_us"] = round(rms, 1)
            if rms >= report.th.sync_residual_err_us:
                report.add_error(
                    f"camera {cam_name}: sync residual_rms_us={rms:.0f} "
                    f"(>{report.th.sync_residual_err_us:.0f}) — frame labels vs SensorTimestamp "
                    "fit broken (association slipped, or affine fit broken)"
                )
            elif rms >= report.th.sync_residual_warn_us:
                report.add_warn(
                    f"camera {cam_name}: sync residual_rms_us={rms:.0f} "
                    f"(>{report.th.sync_residual_warn_us:.0f})"
                )

    if present and not configured:
        # Host snapshot omitted camera list — nothing further to cross-check.
        pass


def _check_imu_quaternions(imu_g: h5py.Group, label: str, report: IntegrityReport) -> None:
    if "quat" not in imu_g:
        return
    quat = np.asarray(imu_g["quat"], dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4:
        report.add_error(f"{label}/quat shape {quat.shape} expected (N, 4)")
        return
    if not np.isfinite(quat).all():
        report.add_error(f"{label}/quat contains NaN/Inf")
        return
    if "t_master_us_quat" in imu_g:
        t = np.asarray(imu_g["t_master_us_quat"])
        if t.size != quat.shape[0]:
            report.add_error(
                f"{label}/quat rows {quat.shape[0]} != t_master_us_quat length {t.size}"
            )
    norms = np.linalg.norm(quat, axis=1)
    bad = float(np.mean(np.abs(norms - 1.0) > 0.2)) if norms.size else 0.0
    if bad > 0.01:
        report.add_warn(f"{label}: {bad * 100:.1f}% of quaternions far from unit norm")


def _check_sync_report(sync_report: dict[str, Any], report: IntegrityReport) -> None:
    drops = sync_report.get("drops")
    if isinstance(drops, dict):
        for cam_id, n in drops.items():
            try:
                n_i = int(n)
            except (TypeError, ValueError):
                continue
            if n_i >= 100:
                report.add_warn(f"sync.drops[{cam_id}]={n_i} (alignment dropped many frames)")

    sync_summary: dict[str, Any] = report.summary.setdefault("sync", {})
    if "capture_order" in sync_report:
        sync_summary["capture_order"] = sync_report.get("capture_order")

    # --- How the take ended (supervisor end_reason) ---------------------------
    # A take that did not run to its planned end is still a valid file, but its
    # length is not what the protocol asked for; a watchdog or exception end
    # means data stopped arriving or the recorder broke mid-take.
    end_reason = sync_report.get("end_reason")
    if isinstance(end_reason, str) and end_reason:
        sync_summary["end_reason"] = end_reason
        if end_reason.startswith("exception") or end_reason in ("watchdog", "setup"):
            report.add_error(f"take ended early: end_reason={end_reason}")
        elif end_reason not in ("duration", "phases_complete", "ct_calibration_only"):
            report.add_warn(
                f"take ended before its planned end: end_reason={end_reason}"
                + (" (--duration ran out before the experiment plan's last phase)"
                   if end_reason == "plan_timeout" else "")
            )
    teardown = sync_report.get("teardown_errors")
    if isinstance(teardown, list) and teardown:
        report.add_warn(f"teardown steps failed: {', '.join(str(t) for t in teardown)}")
    if sync_report.get("mcu_stats_end_fresh") is False:
        report.add_warn(
            "mcu_stats_end predates the stream stop (no fresh REC_STATS after `stats`); "
            "drops in the last second of the take are not counted"
        )
    overflow = int(sync_report.get("cam_hold_overflow") or 0)
    if overflow:
        report.add_warn(f"cam_hold_overflow={overflow}: camera frames were discarded while waiting for triggers")

    # --- How the firmware timed TRIGGER_EVENT ---------------------------------
    stamp = sync_report.get("trigger_stamp")
    if isinstance(stamp, dict):
        counts = stamp.get("counts") if isinstance(stamp.get("counts"), dict) else {}
        fallback = int(stamp.get("fallback", 0) or 0)
        sync_summary["trigger_stamp"] = {"counts": dict(counts), "fallback": fallback}
        if fallback > 0:
            report.add_warn(
                f"trigger stamps: {fallback} TRIGGER_EVENTs came from the firmware timer "
                "fallback (PIO IRQ went silent) -- camera times in that span are "
                "predicted, not measured; check `trig_status` on the HAT"
            )
        elif counts and int(counts.get("pio_irq", 0) or 0) == 0:
            report.add_warn(
                "trigger stamps: timer-based, not edge-latched (firmware before 2026-09 or "
                "`trig_stamp timer`); expect ~4 µs/frame drift against the real pulses"
            )

    # --- Host-side loss -------------------------------------------------------
    host = sync_report.get("host_capture")
    if isinstance(host, dict):
        qd = host.get("queue_drops") if isinstance(host.get("queue_drops"), dict) else {}
        raw = int(qd.get("raw_chunks", 0) or 0)
        recs = int(qd.get("records", 0) or 0)
        sync_summary["host_queue_drops"] = {"raw_chunks": raw, "records": recs}
        if raw or recs:
            report.add_error(
                f"host capture queues overflowed (raw_chunks={raw}, records={recs}): "
                "records were lost on the Pi, not the HAT -- the Pi was too slow to keep up"
            )
        crc = int(host.get("crc_errors", 0) or 0)
        if crc:
            report.add_warn(f"host capture: {crc} USB frame CRC errors")

    # --- Pi clock handshake + camera pairing cross-check ----------------------
    pps = sync_report.get("pps")
    if isinstance(pps, dict):
        accepted = 0
        attempted = 0
        for pass_ in ("start", "end"):
            r = pps.get(pass_)
            if isinstance(r, dict):
                accepted += int(r.get("accepted", 0) or 0)
                attempted += int(r.get("attempted", 0) or 0)
        sync_summary["pps_accepted"] = accepted
        if attempted > 0 and accepted == 0:
            report.add_warn(
                "no `pps` replies from the HAT (firmware before 2026-09?) -- camera<->trigger "
                "pairing could not be cross-checked"
            )

    cams = sync_report.get("cameras")
    if isinstance(cams, dict):
        cam_summary: dict[str, Any] = report.summary.setdefault("cameras", {})
        weak = _weak_pps_reason(sync_report)
        # The residual is measured against the host<->MCU PPS fit. Without the
        # end pass (take ended by an exception) or with no real PPS at all, that
        # fit has no slope constraint, so drift/spread/offset can be the fit's
        # error rather than the association's: report, but do not fail on it.
        grade = report.add_warn if weak else report.add_error
        noted = False
        for cid, entry in cams.items():
            ar = entry.get("assign_residual") if isinstance(entry, dict) else None
            if not isinstance(ar, dict):
                continue
            if weak and not noted:
                noted = True
                sync_summary["assign_gates_downgraded"] = weak
                report.add_warn(
                    f"camera assignment-residual gates downgraded to warnings: {weak} -- "
                    "the PPS fit they are measured against is not trustworthy, so they "
                    "cannot tell an association error from a clock-fit error"
                )
            row = cam_summary.setdefault(str(cid), {})
            frames = ar.get("median_frames")
            drift = ar.get("drift_us_per_s")
            row["assign_residual_ms"] = round(float(ar.get("median_us", 0.0)) / 1000.0, 2)
            if frames is not None:
                row["assign_residual_frames"] = round(float(frames), 2)
                ms = float(ar.get("median_us", 0.0)) / 1000.0
                if abs(float(frames)) >= report.th.assign_frames_err:
                    grade(
                        f"camera {cid}: every frame is paired with a trigger "
                        f"{float(frames):+.1f} frames ({ms:+.0f} ms) from its own PPS-mapped "
                        "SensorTimestamp -- the first frame anchored on the wrong pulse and "
                        "the whole take inherited the offset. Camera times are wrong by that "
                        "much against EMG and IMU; do not use this session for kinematics. "
                        "Check host.capture.trigger_after_cameras and the anchor method below"
                    )
                elif abs(float(frames)) > report.th.assign_frames_warn:
                    report.add_warn(
                        f"camera {cid}: paired trigger is {float(frames):+.1f} frames away from "
                        "the frame's own PPS-mapped SensorTimestamp -- the sequence "
                        "association is probably off by that many pulses"
                    )
            if drift is not None:
                row["assign_drift_us_per_s"] = round(float(drift), 1)
                if abs(float(drift)) >= report.th.assign_drift_err_us_per_s:
                    grade(
                        f"camera {cid}: trigger stamps drift {float(drift):+.0f} µs/s against "
                        "the camera clock -- a whole frame every "
                        f"{1e6 / 30.0 / abs(float(drift)):.0f} s at 30 fps, so no constant "
                        "offset correction can rescue this take"
                    )
                elif abs(float(drift)) > report.th.assign_drift_warn_us_per_s:
                    report.add_warn(
                        f"camera {cid}: trigger stamps drift {float(drift):+.0f} µs/s against "
                        "the camera clock (uncompensated PIO period?)"
                    )
            anc = (sync_report.get("camera_anchor") or {}).get(cid)
            lat = anc.get("latency_us") if isinstance(anc, dict) else None
            med_us = ar.get("median_us")
            if frames is not None and lat is not None and med_us is not None \
                    and float(frames) != 0.0:
                period_us = float(med_us) / float(frames)
                dev = (float(med_us) + float(lat)) / period_us
                row["assign_offset_dev_frames"] = round(dev, 3)
                # NTP slewing the Pi clock moves the PPS-mapped reference, not the
                # labels (those are checked step by step against the camera clock).
                slew = row.get("host_clock_slew_ppm")
                slewed = slew is not None and abs(float(slew)) >= _HOST_CLOCK_SLEW_WARN_PPM
                grade_dev = report.add_warn if (weak or slewed) else report.add_error
                why = (" (Pi clock slewed, so this may be the fit's error)" if slewed and not
                       weak else "")
                if abs(dev) >= report.th.assign_offset_dev_frames_err:
                    grade_dev(
                        f"camera {cid}: median frame-to-trigger residual is {dev:+.2f} periods "
                        f"from the expected {float(lat) / 1000:.1f} ms trigger-to-sensor "
                        "latency -- the association is probably off by a whole frame" + why
                    )
                elif abs(dev) > report.th.assign_offset_dev_frames_warn:
                    report.add_warn(
                        f"camera {cid}: median frame-to-trigger residual is {dev:+.2f} periods "
                        f"from the expected {float(lat) / 1000:.1f} ms trigger-to-sensor "
                        "latency" + why
                    )
            spread = ar.get("spread_frames")
            if spread is not None:
                row["assign_spread_frames"] = round(float(spread), 2)
                if float(spread) >= report.th.assign_spread_frames_err:
                    grade(
                        f"camera {cid}: frame↔trigger offset is not constant "
                        f"({float(spread):.1f} frames p10..p90) -- the association slipped "
                        "during the take, so different parts of the session carry different "
                        "camera-time errors"
                    )
                elif float(spread) > report.th.assign_spread_frames_warn:
                    report.add_warn(
                        f"camera {cid}: frame↔trigger offset is not constant "
                        f"({float(spread):.1f} frames p10..p90) -- the association slipped "
                        "part-way through the take"
                    )

    anchors = sync_report.get("camera_anchor")
    if isinstance(anchors, dict):
        for cid, anc in anchors.items():
            if not isinstance(anc, dict):
                continue
            sync_summary.setdefault("camera_anchor", {})[str(cid)] = anc.get("method")
            if anc.get("method") != "pps_time":
                report.add_warn(
                    f"camera {cid}: first frame was anchored on the oldest buffered trigger "
                    f"({anc.get('note', 'no PPS fit')}), so frames may be labelled early -- "
                    "enable host.capture.pps_samples / camera_pps_anchor"
                )
            refined = anc.get("refined_by_frames")
            if refined is not None:
                # The recorder shifted the association mid-take and cleared the
                # residual history that showed the error, so the assignment gates
                # above only see the corrected part. Frames written before the
                # shift keep their old labels.
                n_bad = int(anc.get("refined_after_frames", 0) or 0)
                sync_summary.setdefault("anchor_refined", {})[str(cid)] = {
                    "by_frames": int(refined), "after_frames": n_bad,
                }
                msg = (
                    f"camera {cid}: frame<->trigger anchor was corrected by {int(refined):+d} "
                    f"frames after {n_bad} frames -- those first {n_bad} frames are still "
                    f"labelled {abs(int(refined))} trigger periods off (the gates above only "
                    "see the corrected part); trim them before kinematics"
                )
                if n_bad > report.th.anchor_refined_frames_err:
                    report.add_error(msg)
                else:
                    report.add_warn(msg)


def _weak_pps_reason(sync_report: dict[str, Any]) -> str | None:
    """Why the recorded PPS fit cannot back the assignment gates, or None."""
    pi = sync_report.get("pi_to_master")
    if isinstance(pi, dict) and pi.get("pps_samples") is not None and int(pi["pps_samples"]) == 0:
        return "no PPS samples (pps_samples == 0)"
    pps = sync_report.get("pps")
    if isinstance(pps, dict):
        end = pps.get("end")
        if not isinstance(end, dict):
            return "no end-of-take PPS pass (take ended early / by an exception)"
        if int(end.get("accepted", 0) or 0) == 0:
            return "end-of-take PPS pass got no replies"
    if isinstance(pi, dict) and pi.get("mode") == "offset_only":
        return "PPS samples span too little time for a slope (offset-only fit)"
    return None


def _stream_spans(f: h5py.File, mcu_payload: dict[str, Any] | None) -> dict[str, tuple[int, int]]:
    """First/last t_master_us of every stream that recorded anything."""
    spans: dict[str, tuple[int, int]] = {}

    def _span(ds: Any) -> tuple[int, int] | None:
        t = np.asarray(ds, dtype=np.uint64)
        if t.size < 2:
            return None
        return int(t[0]), int(t[-1])

    emg_g = f.get("emg")
    if emg_g is not None and "t_master_us" in emg_g:
        sp = _span(emg_g["t_master_us"])
        if sp:
            spans["emg"] = sp
    for label in ("imu_mta", "imu_cra"):
        slot_cfg = ((mcu_payload or {}).get(label) or {})
        if not slot_cfg.get("enabled", True):
            continue
        g = f.get(label)
        if g is None:
            continue
        best: tuple[int, int] | None = None
        for name in g.keys():
            if not str(name).startswith("t_master_us"):
                continue
            sp = _span(g[name])
            if sp and (best is None or sp[1] > best[1]):
                best = sp
        if best:
            spans[label] = best
    trig_g = f.get("triggers")
    if trig_g is not None and "t_master_us" in trig_g:
        sp = _span(trig_g["t_master_us"])
        if sp:
            spans["triggers"] = sp
    return spans


def _check_stream_coverage(
    f: h5py.File, mcu_payload: dict[str, Any] | None, report: IntegrityReport
) -> None:
    """Fail a stream that stopped before the others did (it passed on its own span before)."""
    spans = _stream_spans(f, mcu_payload)
    if len(spans) < 2:
        return
    ref_end = max(end for _, end in spans.values())
    sensors = {k: v for k, v in spans.items() if k in ("emg", "imu_mta", "imu_cra")}
    ref_start = min(start for start, _ in sensors.values()) if sensors else None
    coverage: dict[str, Any] = {}
    for name, (start, end) in spans.items():
        gap_s = float(ref_end - end) / 1e6
        coverage[name] = {"span_s": round(float(end - start) / 1e6, 3), "end_gap_s": round(gap_s, 3)}
        if gap_s > report.th.stream_end_gap_err_s:
            report.add_error(
                f"{name}: stream ends {gap_s:.2f} s before the end of the session -- it "
                "stopped early (IMU stall, panel button, or a bus fault); its rate check "
                "only covers the part that exists"
            )
        if name in sensors and ref_start is not None:
            lag_s = float(start - ref_start) / 1e6
            coverage[name]["start_lag_s"] = round(lag_s, 3)
            # Symmetry with end_gap above: a stream that joins late leaves that
            # span with no data for its modality, exactly as one that stops
            # early does, and every per-stream rate check still passes because
            # it only ever sees the part that exists.
            if lag_s >= report.th.stream_start_lag_err_s:
                report.add_error(
                    f"{name}: stream starts {lag_s:.2f} s after the first sensor stream -- "
                    "that span has no data for this modality, and its own rate check only "
                    "covers the part that exists"
                )
            elif lag_s > report.th.stream_start_lag_warn_s:
                report.add_warn(f"{name}: stream starts {lag_s:.2f} s after the first sensor stream")
    report.summary["coverage"] = coverage


def _check_file_checksum(meta: h5py.Group, path: Path, report: IntegrityReport) -> None:
    hashes = meta.attrs.get("file_hashes_json")
    if hashes is None:
        report.add_warn("no /meta/file_hashes_json sidecar checksums")
        return
    if isinstance(hashes, bytes):
        hashes = hashes.decode("utf-8")
    try:
        table = json.loads(str(hashes))
    except json.JSONDecodeError:
        report.add_warn("invalid file_hashes_json in /meta")
        return
    if not isinstance(table, dict):
        return

    # The table lists the session's sidecar files (e.g. preview_index.jsonl),
    # hashed at close, by name relative to the session directory. It used to be
    # looked up under the .h5's own name, which is never in it, so nothing was
    # ever verified.
    checked: dict[str, str] = {}
    for rel, expected in sorted(table.items()):
        side = path.parent / str(rel)
        if not side.is_file():
            checked[str(rel)] = "missing"
            report.add_warn(f"sidecar {rel} listed in file_hashes_json is missing next to the .h5")
            continue
        digest = hashlib.sha256(side.read_bytes()).hexdigest()
        if digest != str(expected):
            checked[str(rel)] = "mismatch"
            report.add_error(
                f"sidecar checksum mismatch for {rel}: modified or truncated after the session closed"
            )
        else:
            checked[str(rel)] = "ok"
    report.summary["sidecars"] = checked


_HOST_CLOCK_SLEW_WARN_PPM = 200.0


def _label_residual_rms(
    x_us: np.ndarray, y_us: np.ndarray, period_s: float | None
) -> tuple[float, float | None]:
    """Frame-label jitter against the camera clock, and the worst host-clock slew.

    ``x_us`` are SensorTimestamps (Pi clock), ``y_us`` the frames' trigger times
    (MCU clock). One straight line over the take is not a valid model: NTP can
    run the Pi clock hundreds of ppm off while it corrects it (bench, 2026-09-25:
    -3187 ppm settling to -321 ppm), which bends the relation by tens of ms and
    made a correct take read as a 7.7 ms residual. The residual is taken against
    a straight line per ~10 s window instead (label slips are caught by the
    step check); the slew is the largest per-window rate difference in ppm.
    """
    n = int(x_us.size)
    win = 300 if not period_s else max(30, int(round(10.0 / float(period_s))))
    if n < 3:
        return 0.0, None
    res: list[np.ndarray] = []
    worst: float | None = None
    for i in range(0, n, win):
        j = min(n, i + win)
        if j - i < 3 and res:
            break
        xs, ys = x_us[i:j], y_us[i:j]
        if float(np.ptp(xs)) <= 0:
            continue
        a, b = np.polyfit(xs, ys, 1)
        res.append(ys - (a * xs + b))
        if j - i >= 30:
            ppm = (float(a) - 1.0) * 1e6
            worst = ppm if worst is None or abs(ppm) > abs(worst) else worst
    r = np.concatenate(res) if res else np.zeros(1)
    return float(np.sqrt(np.mean(r ** 2))), worst


def _summarize_events(f: h5py.File, report: IntegrityReport) -> None:
    """Count /events rows by kind (informational: viewer notes, phase edges)."""
    g = f.get("events")
    if not isinstance(g, h5py.Group):
        return
    kinds = [str(k) for k in g["kind"].asstr()[()]] if "kind" in g else []
    report.summary["events"] = {
        "total": len(kinds),
        "notes": sum(1 for k in kinds if k == "note"),
        "phase_edges": sum(1 for k in kinds if k.startswith("phase_")),
        "mcu_logs": sum(1 for k in kinds if k == "mcu_log"),
    }


def _load_sync_report(f: h5py.File) -> dict[str, Any]:
    sync = f.get("sync")
    if sync is None:
        return {}
    raw = sync.attrs.get("report_json")
    if raw is None:
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        obj = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _errors_as_warnings(
    docs: list[tuple[dict[str, Any], str, str]],
) -> tuple[bool, str | None]:
    """``host.integrity.errors_as_warnings`` from the first config that sets it."""
    for doc, source, _where in docs:
        host = doc.get("host")
        host = host if isinstance(host, dict) else doc
        block = host.get("integrity") if isinstance(host, dict) else None
        if isinstance(block, dict) and "errors_as_warnings" in block:
            return bool(block["errors_as_warnings"]), source
    return False, None


def _demote_errors(report: IntegrityReport) -> None:
    """Move every error into warnings, loudly.

    A session passed only because the gates were switched off must never be
    indistinguishable from one that passed on its merits -- that is how a bad
    take ends up in a results table. So each demoted error keeps its full text
    behind a `would-fail:` marker, a banner goes first in the warning list, and
    the originals are preserved verbatim in
    ``summary["errors_as_warnings"]["demoted"]``, which `set_hdf5` persists into
    integrity.json.
    """
    demoted = list(report.issues)
    report.issues.clear()
    report.summary["errors_as_warnings"] = {"enabled": True, "demoted": demoted}
    if not demoted:
        return
    report.warnings[:0] = [
        f"host.integrity.errors_as_warnings is ON: {len(demoted)} error(s) demoted -- "
        "this session would otherwise be RESULT: FAIL",
        *(f"would-fail: {m}" for m in demoted),
    ]


def _gate_configs(
    resolved_config: dict[str, Any] | None,
    config_source: str | None,
    host_cfg: dict[str, Any] | None,
    path: Path,
) -> list[tuple[dict[str, Any], str, str]]:
    """Configs the gates are read from, in precedence order.

    1. the config the caller says applies (``jtsmm record`` passes the one it
       recorded with; ``jtsmm integrity --config X`` passes X);
    2. the host config embedded in the session (``/meta/host_config_json``),
       so a file stays checkable anywhere with the gates it was recorded under;
    3. defaults.
    Nothing is picked up implicitly from this machine: the packaged template
    is not an operator's config, and a verdict must not change with the cwd.
    """
    docs: list[tuple[dict[str, Any], str, str]] = []
    if isinstance(resolved_config, dict):
        docs.append((resolved_config, "explicit config", config_source or "<resolved_config>"))
    if isinstance(host_cfg, dict):
        docs.append((host_cfg, "session config", f"{path}:/meta/host_config_json"))
    return docs


def check_session(
    path: Path | str,
    *,
    expected_config: dict[str, Any] | None = None,
    resolved_config: dict[str, Any] | None = None,
    config_source: str | None = None,
) -> IntegrityReport:
    """Run post-session integrity checks; return structured report.

    @param expected_config  Fail unless the session's config_hash matches it.
    @param resolved_config  Config whose ``host.integrity`` gates apply (the one
                            the recording used, or an operator override); falls
                            back to the session-embedded config, then defaults.
    @param config_source    Where ``resolved_config`` came from (a path), for
                            the report.
    """
    path = Path(path)
    report = IntegrityReport(path=path)
    if not path.is_file():
        report.add_error(f"session file not found: {path}")
        return report

    try:
        f = h5py.File(path, "r")
    except OSError as exc:
        msg = str(exc)
        if "consistency flag" in msg or "already open for write" in msg:
            report.add_error(
                f"{path.name} was not closed cleanly (recorder killed or crashed). Data "
                "flushed up to ~1 s before the stop is recoverable: run "
                f"`h5clear -s {path}` (hdf5-tools), then check it again"
            )
        else:
            report.add_error(f"cannot open {path.name} as HDF5: {msg}")
        return report
    with f:
        meta = _check_hdf5_schema(f, report)
        if meta is None:
            return report

        mcu_payload = _load_json_attr(meta, "mcu_config_json")
        _check_config_hash(meta, report)

        if expected_config is not None:
            exp = applied_config_hash(expected_config)
            stored = meta.attrs.get("config_hash")
            if stored is not None:
                if isinstance(stored, bytes):
                    stored = stored.decode("utf-8")
                if str(stored) != exp:
                    report.add_error("stored config_hash does not match supplied expected_config")

        # A missing group is already an error from _check_hdf5_schema; the
        # remaining checks must still run instead of dying on a KeyError.
        emg_g = f.get("emg")
        duration_s = _session_duration_s(meta, emg_g)
        report.summary["meta"] = {
            "size_bytes": path.stat().st_size,
            "schema_version": meta.attrs.get("schema_version"),
            "software_version": meta.attrs.get("software_version"),
            "subject_id": meta.attrs.get("subject_id"),
            "protocol": meta.attrs.get("protocol"),
            "duration_s": round(duration_s, 3) if duration_s is not None else None,
            "config_hash": meta.attrs.get("config_hash"),
        }
        host_cfg = _load_json_attr(meta, "host_config_json")
        docs = _gate_configs(resolved_config, config_source, host_cfg, path)
        th, source, source_path = None, "defaults", None
        for doc, src, where in docs:
            th = IntegrityThresholds.from_config(doc)
            if th is not None:
                source, source_path = src, where
                break
        if th is None:
            th = IntegrityThresholds()
        th, th_notes = th.validated()
        report.th = th
        for note in th_notes:
            report.add_warn(note)
        report.summary["thresholds"] = {
            "source": source, "source_path": source_path, **dataclasses.asdict(th),
        }
        sync_report = _load_sync_report(f)
        mcu_stats_end = sync_report.get("mcu_stats_end")
        if not isinstance(mcu_stats_end, dict):
            mcu_stats_end = None
        # From here on the MCU counters describe this take only (when the
        # recorder stored a start snapshot); gauges keep their end values.
        mcu_stats_start = sync_report.get("mcu_stats_start")
        mcu_stats_end, stats_scope = take_scoped_mcu_stats(mcu_stats_end, mcu_stats_start)
        t_ref_us = _session_start_ref(f)
        emg_rows = (
            int(emg_g["t_master_us"].shape[0])
            if emg_g is not None and "t_master_us" in emg_g else None
        )
        emg_rows_take = emg_rows
        if emg_rows is not None and stats_scope.get("scope") == "take":
            # Rows the recorder wrote before the start snapshot are in its counters.
            before = int((mcu_stats_start or {}).get("host_emg_rows_before", 0) or 0)
            emg_rows_take = max(0, emg_rows - before)
        last_emg_t = int(emg_g["t_master_us"][-1]) if emg_rows else None
        emg_attr = meta.attrs.get("emg_enabled", "true")
        if isinstance(emg_attr, bytes):
            emg_attr = emg_attr.decode("utf-8", "replace")
        emg_on = str(emg_attr).strip().lower() not in ("0", "false", "no")
        if emg_on and emg_g is None:
            report.summary["emg"] = {"skipped": True, "reason": "no /emg group"}
        elif emg_on:
            _check_emg(
                emg_g, mcu_payload, report, duration_s=duration_s,
                mcu_stats_end=mcu_stats_end, host_cfg=host_cfg, t_ref_us=t_ref_us,
            )
        else:
            report.summary["emg"] = {"skipped": True, "reason": "emg_enabled=false"}
        _check_mcu_stats(
            mcu_stats_end, report, emg_rows=emg_rows_take, emg_on=emg_on,
            last_t_us=last_emg_t, scope=stats_scope,
        )
        for label in ("imu_mta", "imu_cra"):
            imu_g = f.get(label)
            if imu_g is None:
                continue
            _check_imu_slot(imu_g, label, mcu_payload, report,
                            mcu_stats_end=mcu_stats_end, t_ref_us=t_ref_us)
            _check_imu_quaternions(imu_g, label, report)
        if f.get("triggers") is not None:
            _check_triggers(f["triggers"], mcu_payload, duration_s, report)
        skipped_raw = meta.attrs.get("skipped_cameras")
        if isinstance(skipped_raw, bytes):
            skipped_raw = skipped_raw.decode("utf-8")
        skipped_cameras = (
            {c.strip() for c in str(skipped_raw).split(",") if c.strip()}
            if skipped_raw
            else set()
        )
        _check_trigger_camera_pairing(
            f, report, sync_report=sync_report, host_cfg=host_cfg,
            skipped_cameras=skipped_cameras,
        )
        _check_sync_report(sync_report, report)
        _check_stream_coverage(f, mcu_payload, report)
        check_optical(f, report)
        check_optical_imu_sync(f, report)
        _check_file_checksum(meta, path, report)
        _summarize_events(f, report)

        eaw, eaw_source = _errors_as_warnings(docs)
        if eaw:
            _demote_errors(report)
            report.summary["errors_as_warnings"]["source"] = eaw_source

    return report


def check_session_issues(path: Path | str, **kwargs: Any) -> list[str]:
    """Backward-compatible issue list (errors + warn:-prefixed warnings)."""
    report = check_session(path, **kwargs)
    out = list(report.issues)
    out.extend(f"warn: {w}" for w in report.warnings)
    return out


def _fmt_rate(measured: float | None, configured: float | None) -> str:
    m = f"{measured:.1f}" if measured is not None else "—"
    c = f"{configured:.0f}" if configured is not None else "—"
    return f"{m}/{c} Hz"


def _fmt_bytes(n: int | None) -> str:
    if not n:
        return "n/a"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _row_status(report: IntegrityReport, needles: tuple[str, ...]) -> str:
    def _hits(msgs: list[str]) -> bool:
        return any(any(n.lower() in m.lower() for n in needles) for m in msgs)

    if _hits(report.issues):
        return "FAIL"
    if _hits(report.warnings):
        return "WARN"
    return "OK"


def format_report(report: IntegrityReport, *, width: int = 88) -> str:
    """Render an :class:`IntegrityReport` as a human-friendly summary block.

    Uses only ``report.summary``/``issues``/``warnings`` gathered by
    :func:`check_session`, so it never re-opens the HDF5 file.
    """
    rule = "=" * width
    thin = "-" * width
    lines: list[str] = [rule, f" jtsmm integrity report — {report.path}", rule]

    meta = report.summary.get("meta", {})

    def _s(v: Any) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        return str(v)

    duration = meta.get("duration_s")
    lines.append(
        f" subject     {_s(meta.get('subject_id')):<18} protocol    {_s(meta.get('protocol'))}"
    )
    lines.append(
        f" duration    {(f'{duration:.2f} s' if duration is not None else 'n/a'):<18} "
        f"size        {_fmt_bytes(meta.get('size_bytes'))}"
    )
    lines.append(
        f" schema      {_s(meta.get('schema_version')):<18} software    {_s(meta.get('software_version'))}"
    )
    lines.append(thin)
    header = f" {'stream':<16}{'count':>10}  {'rate (measured/target)':<24}  status"
    lines.append(header)
    lines.append(thin)

    emg = report.summary.get("emg")
    if emg is not None:
        rate = _fmt_rate(emg.get("measured_hz"), emg.get("configured_hz"))
        status = _row_status(report, ("emg",))
        lines.append(f" {'emg':<16}{emg.get('samples', 0):>10}  {rate:<24}  {status}")

    for label, imu in report.summary.get("imu", {}).items():
        rate = _fmt_rate(imu.get("measured_hz"), imu.get("configured_hz"))
        status = _row_status(report, (label,))
        name = f"{label}.quat"
        lines.append(f" {name:<16}{imu.get('quat_samples', 0):>10}  {rate:<24}  {status}")

    trig = report.summary.get("triggers")
    if trig is not None:
        rate = _fmt_rate(trig.get("measured_hz"), trig.get("configured_hz"))
        status = _row_status(report, ("trigger",))
        lines.append(f" {'triggers':<16}{trig.get('count', 0):>10}  {rate:<24}  {status}")

    for cam_name, cam in report.summary.get("cameras", {}).items():
        cov = cam.get("coverage_frac")
        cov_str = f"{cov * 100:.1f}% coverage" if cov is not None else "n/a"
        status = _row_status(report, (f"camera {cam_name}",))
        lines.append(f" {cam_name:<16}{cam.get('frames', 0):>10}  {cov_str:<24}  {status}")

    events = report.summary.get("events")
    if events is not None:
        lines.append(f" {'viewer notes':<16}{events['notes']:>10}  {'in /events (kind=note)':<24}  info")
        n_log = events.get("mcu_logs", 0)
        lines.append(f" {'firmware log':<16}{n_log:>10}  {'in /events (kind=mcu_log)':<24}  info")

    lines.append(thin)
    lines.append(f" ERRORS ({len(report.issues)})")
    for err in report.issues:
        lines.append(f"   - {err}")
    lines.append(f" WARNINGS ({len(report.warnings)})")
    for w in report.warnings:
        lines.append(f"   - {w}")
    lines.append(rule)
    if not report.ok:
        verdict = f"RESULT: FAIL ({len(report.issues)} error(s), {len(report.warnings)} warning(s))"
    elif report.warnings:
        demoted = (report.summary.get("errors_as_warnings") or {}).get("demoted") or []
        verdict = f"RESULT: OK with {len(report.warnings)} warning(s)"
        if demoted:
            verdict += f" [{len(demoted)} error(s) demoted by errors_as_warnings]"
    else:
        verdict = "RESULT: OK"
    lines.append(f" {verdict}")
    lines.append(rule)
    return "\n".join(lines)
