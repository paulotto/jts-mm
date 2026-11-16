# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Smaller integrity fixes: missing groups, NaN gates, mains grading, static takes."""

from __future__ import annotations

import math
from pathlib import Path

import h5py
import numpy as np

from jtsmm.config.catalog import lsb_uv
from jtsmm.integrity.checker import IntegrityThresholds, check_session
from jtsmm.integrity.optical_imu_sync import estimate_lag

from tests.helpers import make_session_h5


def test_missing_groups_are_reported_not_raised(tmp_path: Path):
    for group in ("emg", "imu_mta", "imu_cra", "triggers"):
        path = tmp_path / f"no_{group}.h5"
        make_session_h5(path)
        with h5py.File(path, "a") as f:
            del f[group]
        report = check_session(path)  # used to raise KeyError
        assert any(f"missing HDF5 group /{group}" in e for e in report.issues)


def test_missing_optical_pose_dataset_is_reported_not_raised(tmp_path: Path):
    path = tmp_path / "opt.h5"
    make_session_h5(path, optical=True, duration_s=2.0)
    with h5py.File(path, "a") as f:
        del f["optical"]["pose_mta"]
    report = check_session(path)  # used to raise KeyError in the stereo lock loop
    assert any("missing /optical/pose_mta" in e for e in report.issues)


def test_nan_and_negative_thresholds_are_rejected():
    th, notes = IntegrityThresholds(
        assign_frames_err=float("nan"), emg_loss_err_frac=-1.0, sync_residual_err_us=math.inf,
    ).validated()
    d = IntegrityThresholds()
    assert th.assign_frames_err == d.assign_frames_err
    assert th.emg_loss_err_frac == d.emg_loss_err_frac
    assert th.sync_residual_err_us == d.sync_residual_err_us
    assert sum("not a finite, non-negative" in n for n in notes) == 3


def test_nan_threshold_from_config_does_not_disable_the_gate(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=1.0, gap_at=500)
    cfg = {"host": {"integrity": {"thresholds": {"emg_gap_err_periods": float("nan")}}}}
    report = check_session(path, resolved_config=cfg)
    assert any("EMG continuity" in e for e in report.issues)
    assert any("emg_gap_err_periods=nan" in w for w in report.warnings)


def test_a_single_bad_mains_window_is_not_reported(tmp_path: Path):
    """1 of 24 windows (4 %) is below MAINS_BAD_WINDOW_FRAC_WARN (5 %)."""
    fs, dur = 1000, 48.0
    path = tmp_path / "hum.h5"
    make_session_h5(path, duration_s=dur, emg_samples=int(fs * dur), emg_fs=fs)
    lsb = lsb_uv(6)
    with h5py.File(path, "a") as f:
        g = f["emg"]
        x = np.asarray(g["samples"], dtype=np.int64)
        t = np.arange(2000) / fs
        hum = (1000.0 * np.sqrt(2) / lsb * np.sin(2 * np.pi * 49.98 * t)).astype(np.int64)
        x[10_000:12_000, :6] += hum[:, None]
        g["samples"][...] = x.astype(np.int32)
        del g["samples_uv"]
    report = check_session(path)
    assert report.summary["emg_signal"]["channels"]["masseter_l"]["bad_window_frac"] > 0
    assert not any("mains interference" in m for m in report.issues + report.warnings)


def test_static_imu_with_noisy_optical_is_inconclusive():
    """PnP jitter differentiates to tens of deg/s on a head that is not moving."""
    rng = np.random.default_rng(0)
    t_o = (np.arange(0, 20, 1 / 30.0) * 1e6).astype(np.int64)
    poses = np.tile(np.eye(4), (t_o.size, 1, 1))
    for i in range(t_o.size):
        a = np.radians(rng.normal(0, 1.0, 3))
        cx, sx, cy, sy, cz, sz = (math.cos(a[0]), math.sin(a[0]), math.cos(a[1]),
                                  math.sin(a[1]), math.cos(a[2]), math.sin(a[2]))
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        poses[i, :3, :3] = rz @ ry @ rx
    t_i = (np.arange(0, 20, 1 / 330.0) * 1e6).astype(np.int64)
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (t_i.size, 1))
    assert estimate_lag(t_o, poses, t_i, quats) is None
