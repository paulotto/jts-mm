# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG loss accounting, the session-start hole, and mid-stream IMU gaps."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jtsmm.integrity.checker import _seq_loss, check_session

from tests.helpers import make_session_h5


def _rewrite(g: h5py.Group, name: str, data: np.ndarray) -> None:
    del g[name]
    g.create_dataset(name, data=data)


def _emg_keep(path: Path, keep: np.ndarray, *,
              prefix_t: np.ndarray | None = None) -> None:
    """Keep EMG rows by mask/index; optionally prepend rows at given times."""
    with h5py.File(path, "a") as f:
        g = f["emg"]
        cols = {n: np.asarray(g[n])[keep] for n in ("samples", "samples_uv", "leadoff_status")}
        t = np.asarray(g["t_master_us"], dtype=np.uint64)[keep].astype(np.int64)
        if prefix_t is not None:
            n = prefix_t.size
            for k, v in cols.items():
                cols[k] = np.concatenate([np.repeat(v[:1], n, axis=0), v])
            t = np.concatenate([prefix_t.astype(np.int64), t])
        for k, v in cols.items():
            _rewrite(g, k, v)
        _rewrite(g, "t_master_us", t.astype(np.uint64))


def _shift_all_but_emg(path: Path, us: int) -> None:
    """Move triggers/cameras/IMU later so a prefix can precede the first trigger."""
    with h5py.File(path, "a") as f:
        for grp, names in (("triggers", ("t_master_us",)),):
            g = f[grp]
            for n in names:
                _rewrite(g, n, np.asarray(g[n], dtype=np.uint64) + np.uint64(us))
        for cid in f["cameras"]:
            g = f["cameras"][cid]
            _rewrite(g, "t_master_us", np.asarray(g["t_master_us"], dtype=np.uint64) + np.uint64(us))
        for slot in ("imu_mta", "imu_cra"):
            g = f[slot]
            for n in list(g.keys()):
                if n.startswith("t_master_us"):
                    _rewrite(g, n, np.asarray(g[n], dtype=np.uint64) + np.uint64(us))
        g = f["emg"]
        _rewrite(g, "t_master_us", np.asarray(g["t_master_us"], dtype=np.uint64) + np.uint64(us))


def test_leading_hole_is_named_and_excluded(tmp_path: Path):
    """40 ms of EMG, 0.87 s hole, then the take: the real start-up pattern."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=2.0, emg_samples=2000)
    _shift_all_but_emg(path, 1_000_000)
    with h5py.File(path, "r") as f:
        n = f["emg"]["t_master_us"].shape[0]
    _emg_keep(path, np.arange(n), prefix_t=np.arange(40, dtype=np.int64) * 1000 + 90_000)
    report = check_session(path)
    assert any("EMG leading hole" in w and "excluded" in w for w in report.warnings), report.warnings
    assert not any("EMG continuity" in e or "EMG rate" in e or "EMG loss" in e
                   for e in report.issues), report.issues
    lead = report.summary["emg"]["leading_hole"]
    assert lead["hole_s"] > 0.85 and lead["samples_before_s"] < 0.05


def test_a_mid_take_hole_is_not_mistaken_for_the_leading_hole(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=1.0, gap_at=500)
    report = check_session(path)
    assert report.summary["emg"]["leading_hole"] is None
    assert any("EMG continuity" in e for e in report.issues)


def test_single_missing_samples_are_counted_and_gated(tmp_path: Path):
    """dt = 2T never crossed the 2.5-period gap threshold, so it was never counted."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=2.0, emg_samples=2000)
    keep = np.ones(2000, dtype=bool)
    keep[5::100] = False  # 20 single missing samples = 1 %
    _emg_keep(path, keep)
    report = check_session(path)
    assert not any("EMG continuity" in m for m in report.issues + report.warnings)
    assert report.summary["emg"]["missing"] == 20
    assert any("EMG loss: 20 samples missing" in e for e in report.issues), report.issues


def test_small_loss_only_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=2.0, emg_samples=2000)
    keep = np.ones(2000, dtype=bool)
    keep[[300, 1500]] = False  # 0.1 %
    _emg_keep(path, keep)
    report = check_session(path)
    assert any("EMG loss: 2 samples" in w for w in report.warnings), report.warnings
    assert report.ok, report.issues


def test_seq_is_the_exact_loss_count_and_wraps(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=1.0)
    with h5py.File(path, "a") as f:
        n = f["emg"]["t_master_us"].shape[0]
        seq = (np.arange(n, dtype=np.int64) + (1 << 32) - 400)
        seq[600:] += 3  # 3 conversions lost after the ADC, invisible in timestamps
        f["emg"].create_dataset("seq", data=(seq % (1 << 32)).astype(np.uint32))
    report = check_session(path)
    emg = report.summary["emg"]
    assert emg["missing_from_seq"] == 3 and emg["missing_from_timestamps"] == 0
    assert any("from seq; timestamps say 0" in m for m in report.warnings + report.issues)


def test_seq_loss_helper():
    assert _seq_loss(np.array([0xFFFFFFFE, 0xFFFFFFFF, 0, 2], dtype=np.uint32)) == (1, 0)
    assert _seq_loss(np.array([5, 5, 6, 4], dtype=np.uint32)) == (0, 2)


def _imu_drop(path: Path, slot: str, t0_s: float, t1_s: float) -> None:
    with h5py.File(path, "a") as f:
        g = f[slot]
        for stream in ("quat", "linear_accel"):
            t = np.asarray(g[f"t_master_us_{stream}"], dtype=np.uint64)
            keep = (t < t0_s * 1e6) | (t >= t1_s * 1e6)
            _rewrite(g, stream, np.asarray(g[stream])[keep])
            _rewrite(g, f"t_master_us_{stream}", t[keep])


def test_mid_stream_imu_silence_is_an_error(tmp_path: Path):
    """take1: imu_mta silent 26.9 s -> only a rate warning before."""
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=4.0, emg_samples=4000)
    _imu_drop(path, "imu_mta", 1.5, 2.5)
    report = check_session(path)
    errs = [e for e in report.issues if e.startswith("imu_mta/quat") and "gap" in e]
    assert errs and "longest 1.00 s at t=1.5 s" in errs[0], report.issues
    g = report.summary["imu_gaps"]["imu_mta"]["quat"]
    assert g["missing"] == 400 and abs(g["longest_gap_s"] - 1.0) < 0.01


def test_short_imu_gap_warns(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=4.0, emg_samples=4000)
    _imu_drop(path, "imu_cra", 2.0, 2.01)  # 4 samples at 400 Hz: 0.25 % < 0.5 %
    report = check_session(path)
    assert not any(e.startswith("imu_cra/quat") for e in report.issues), report.issues
    assert any(w.startswith("imu_cra/quat") and "gap" in w for w in report.warnings)
