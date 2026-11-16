# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-channel EMG LSB (K5) and the per-window DRL AC figure."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jtsmm.config.catalog import channel_lsb_uv, lsb_uv
from jtsmm.integrity.checker import check_session
from jtsmm.integrity.emg_signal import analyse_emg_signal
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.io.session_reader import Session
from jtsmm.sync.clock_align import AlignedRecord

from tests.helpers import make_session_h5

FS = 1000.0
AUX = lsb_uv(24)


def _drl_samples(dc_step_mv: float, ac_mv: float, n_windows: int = 8) -> np.ndarray:
    n = int(FS * 2) * n_windows
    x = np.zeros((n, 8))
    t = np.arange(n) / FS
    win = (np.arange(n) // int(FS * 2)).astype(float)
    uv = dc_step_mv * 1000.0 * win + ac_mv * 1000.0 * np.sqrt(2) * np.sin(2 * np.pi * 50 * t)
    x[:, 7] = uv / AUX
    return x


def test_drl_ac_excludes_dc_drift_between_windows():
    """A slow drift of 20 mV per window with 2 mV real AC must read ~2 mV."""
    res = analyse_emg_signal(
        _drl_samples(dc_step_mv=20.0, ac_mv=2.0), fs_hz=FS, lsb_uv=lsb_uv(6),
        active_channels=[], lsb_uv_per_channel=channel_lsb_uv(6),
    )
    assert abs(res["drl"]["ac_mv"] - 2.0) < 0.1, res["drl"]
    assert not res["verdict"]["drl_warn"] and not res["verdict"]["drl_error"]


def test_drl_uses_the_fixed_gain_lsb_not_the_muscle_lsb():
    x = _drl_samples(dc_step_mv=0.0, ac_mv=20.0)
    right = analyse_emg_signal(x, fs_hz=FS, lsb_uv=lsb_uv(6), active_channels=[],
                               lsb_uv_per_channel=channel_lsb_uv(6))
    wrong = analyse_emg_signal(x, fs_hz=FS, lsb_uv=lsb_uv(6), active_channels=[])
    assert abs(right["drl"]["ac_mv"] - 20.0) < 0.5
    assert abs(wrong["drl"]["ac_mv"] / right["drl"]["ac_mv"] - 4.0) < 0.01


def _write_session(path: Path) -> None:
    w = Hdf5Writer(path=path, config_hash="x", mcu_config={"emg": {"default_gain": 6,
                   "sample_rate_sps": 1000}}, host_config={})
    w.open()
    for i in range(10):
        w.append(AlignedRecord("emg", 1000 * i, {"channels": [1000] * 8, "status": b"\xc0\0\0"}))
    w.close()


def test_writer_stores_samples_uv_per_channel(tmp_path: Path):
    path = tmp_path / "w.h5"
    _write_session(path)
    with h5py.File(path, "r") as f:
        g = f["emg"]
        per = np.asarray(g.attrs["lsb_uv_per_channel"])
        assert np.allclose(per, channel_lsb_uv(6))
        assert abs(float(g.attrs["lsb_uv"]) - lsb_uv(6)) < 1e-12
        uv = np.asarray(g["samples_uv"])
        assert np.allclose(uv[0, :6], 1000 * lsb_uv(6), rtol=1e-6)
        assert np.allclose(uv[0, 6:], 1000 * AUX, rtol=1e-6)
    with Session(path) as s:
        assert s.emg.uv_note is None
        assert np.allclose(s.emg.samples_uv[0, 7], 1000 * AUX, rtol=1e-6)


def _make_legacy(path: Path) -> None:
    """Old writer: every column scaled with the muscle LSB, no per-channel attr."""
    make_session_h5(path)
    with h5py.File(path, "a") as f:
        g = f["emg"]
        del g.attrs["lsb_uv_per_channel"]
        samples = np.asarray(g["samples"], dtype=np.float64)
        g["samples_uv"][...] = (samples * float(g.attrs["lsb_uv"])).astype(np.float32)


def test_reader_corrects_legacy_ch7_ch8(tmp_path: Path):
    path = tmp_path / "legacy.h5"
    _make_legacy(path)
    with h5py.File(path, "r") as f:
        raw = np.asarray(f["emg"]["samples"][0], dtype=np.float64)
    with Session(path) as s:
        uv = s.emg.samples_uv
        _, win = s.emg.window(0, 3)
        assert s.emg.uv_note and "CH7/CH8" in s.emg.uv_note
    assert np.allclose(uv[0, :6], raw[:6] * lsb_uv(6), rtol=1e-5)
    assert np.allclose(uv[0, 6:], raw[6:] * AUX, rtol=1e-5)
    assert np.allclose(win[0], uv[0])


def test_checker_accepts_legacy_and_current_samples_uv(tmp_path: Path):
    cur = tmp_path / "cur.h5"
    make_session_h5(cur)
    assert not any("samples_uv inconsistent" in e for e in check_session(cur).issues)
    old = tmp_path / "old.h5"
    _make_legacy(old)
    report = check_session(old)
    assert not any("samples_uv inconsistent" in e for e in report.issues), report.issues
    # a current file whose samples_uv was written with the muscle LSB everywhere is wrong
    with h5py.File(cur, "a") as f:
        g = f["emg"]
        g["samples_uv"][...] = (np.asarray(g["samples"], dtype=np.float64)
                                * float(g.attrs["lsb_uv"])).astype(np.float32)
    assert any("samples_uv inconsistent" in e for e in check_session(cur).issues)


def test_checker_notes_the_legacy_drl_rescale(tmp_path: Path):
    path = tmp_path / "old.h5"
    make_session_h5(path, duration_s=5.0, emg_samples=5000, drl_ac_mv=5.0)
    with h5py.File(path, "a") as f:
        del f["emg"].attrs["lsb_uv_per_channel"]
    report = check_session(path)
    assert "CH7/CH8 re-scaled" in report.summary["emg_signal"]["lsb_note"]
    assert abs(report.summary["emg_signal"]["drl_ac_mv"] - 5.0) < 0.5
