# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The viewer adapts to the EMG rate: ~500 display rows/s, filters built for the real fs."""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.viewer.hub import TARGET_DISPLAY_FS, Hub, display_bucket
from jtsmm.viewer.sources import TapSource

RATES = (250, 500, 1000, 2000, 4000)


class _Cap:
    def __init__(self) -> None:
        self.bucket: int | None = None

    def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
        self.bucket = bucket_samples


@pytest.mark.parametrize("fs", RATES)
def test_recording_viewer_keeps_500_rows_per_second(fs, monkeypatch):
    """Full-rate samples from the tap; the hub decimates to ~500 rows/s itself."""
    cap = _Cap()
    src = TapSource(cap, emg_fs=fs, emg_gain=6)
    hub = src.make_hub()
    monkeypatch.setattr(src, "_spawn", lambda *_a: None)
    src.start(hub)
    assert cap.bucket == 1
    assert hub.decim == display_bucket(fs) == max(1, fs // 500)
    assert hub.display_fs == pytest.approx(TARGET_DISPLAY_FS)
    assert hub.emg_fs == fs


@pytest.mark.parametrize("fs", (1000, 2000, 4000))
def test_live_filter_removes_mains_at_every_rate(fs):
    """Filters are designed for the configured fs: 50 Hz + harmonics go, EMG stays."""
    hub = Hub(emg_fs=fs, emg_lsb_uv=0.1, n_channels=1)
    assert hub.decim == max(1, round(fs / TARGET_DISPLAY_FS))
    rng = np.random.default_rng(0)
    n = int(fs * 6)
    t_us = (np.arange(n) * (1e6 / fs)).astype(np.uint64) + 1_000_000
    t = np.arange(n) / fs
    emg = rng.normal(0.0, 20.0, n)  # broadband "EMG"
    mains = 500.0 * np.sin(2 * np.pi * 50 * t) + 200.0 * np.sin(2 * np.pi * 150 * t)
    block = int(fs * 0.1)
    for i in range(0, n, block):  # viewer-sized blocks, like the live source
        hub.append_emg_raw(t_us[i:i + block], (emg + mains)[i:i + block, None].astype(np.float32))
    filt = np.concatenate(hub._filt)[:, 0]
    tail = filt[len(filt) // 2:]  # after the tracker has locked
    spec = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
    fr = np.fft.rfftfreq(len(tail), 1.0 / hub.display_fs)

    def band(f0):
        return spec[(fr > f0 - 2) & (fr < f0 + 2)].max()

    floor = np.median(spec[(fr > 60) & (fr < 140)])
    assert band(50) < 3 * floor and band(150) < 3 * floor  # mains gone
    assert np.std(tail) > 5.0  # EMG-like content still there


class _Frame:
    """One native tap frame at bucket 1: min == max == mean == the samples (counts)."""

    def __init__(self, t_us, counts, status):
        self.emg_t = t_us
        self.emg_min = self.emg_max = counts
        self.emg_mean = counts.astype(np.float32)
        self.emg_seen = 0
        self.status = status
        self.stats = None
        self.trig_seen = 0


def test_recording_viewer_matches_jtsmm_view():
    """Same samples -> same filtered trace, lead-off and saturation as `jtsmm view`.

    Bench 2026-09-28: the recording viewer looked worse on the same signal (it
    filtered pre-bucketed rows at 500 Hz and never trusted the comparators).
    """
    from jtsmm.config.catalog import lsb_uv
    from jtsmm.viewer.sources import _counts_to_uv, _slice_emg_for_viewer

    fs, gain = 4000, 6
    rng = np.random.default_rng(1)
    n = fs * 3
    t = (np.arange(n) * 250).astype(np.uint64) + 1_000_000
    sig = rng.normal(0, 30, (n, 8)) + 400 * np.sin(2 * np.pi * 50 * np.arange(n) / fs)[:, None]
    counts = np.round(sig / lsb_uv(gain)).astype(np.int32)
    counts[:, 2] = 8388607                        # CH3 railed: lead-off by saturation
    status = bytes([0xC0, 0x00, 0x00])            # comparators: every channel connected

    cap = type("Cap", (), {"set_viewer_tap_enabled": lambda self, *a, **k: None})()
    tap = TapSource(cap, emg_fs=fs, emg_gain=gain, leadoff_during_recording=True,
                    comparators_enabled=True)
    tap_hub = tap.make_hub()
    view_hub = Hub(emg_fs=fs, emg_lsb_uv=lsb_uv(gain), n_channels=tap_hub.n_channels,
                   leadoff_during_recording=True, comparators_enabled=True)
    blk = 133                                     # ~33 ms of samples per poll
    for i in range(0, n, blk):
        cap.get_viewer_frame = (lambda fr: (lambda: fr))(_Frame(t[i:i + blk], counts[i:i + blk], status))
        tap.poll_once(tap_hub)
        view_hub.append_emg_raw(t[i:i + blk], _slice_emg_for_viewer(_counts_to_uv(counts[i:i + blk], gain)))
        view_hub.set_leadoff(status)
    a, b = tap_hub.snapshot().to_map(), view_hub.snapshot().to_map()
    assert a["emg"]["n"] == b["emg"]["n"] > 0
    for key in ("t", "lo", "hi", "filt", "env"):
        assert a["emg"][key] == b["emg"][key], key
    assert a["leadoff"] == b["leadoff"]
    assert a["leadoff"]["cmp"] == "on"                       # trusted from the start
    assert a["leadoff"]["p"][2] and not a["leadoff"]["p"][0]  # CH3 off (railed), CH1 on
    assert a["sat"] == b["sat"]
