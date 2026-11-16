# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Viewer stack tests (no hardware): protocol, reader, processing, hub, sources."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from jtsmm.config.catalog import FULL_CHANNEL_NAMES, lsb_uv
from jtsmm.io.session_reader import Session
from jtsmm.viewer.hub import Hub
from jtsmm.viewer.leadoff import decode_status_word, saturation_flags
from jtsmm.viewer.processing import CausalEmgFilter
from jtsmm.viewer.sources import MockSource, TapSource


def _write_session(path: Path, *, n_emg: int = 4000, fs: int = 2000, gain: int = 6) -> None:
    lsb = lsb_uv(gain)
    t = (np.arange(n_emg, dtype=np.uint64) * int(1e6 / fs))
    counts = np.zeros((n_emg, 8), dtype=np.int32)
    for ch in range(8):
        counts[:, ch] = (np.sin(np.arange(n_emg) * 0.05) * (ch + 1) * 1000).astype(np.int32)
    n_imu = n_emg // 5
    it = (np.arange(n_imu, dtype=np.uint64) * int(1e6 / (fs / 5)))
    quat = np.tile(np.array([1, 0, 0, 0], np.float32), (n_imu, 1))
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["session_duration_s"] = n_emg / fs
        meta.attrs["config_hash"] = "abc"
        g = f.create_group("emg")
        g.create_dataset("samples", data=counts)
        g.create_dataset("samples_uv", data=counts.astype(np.float32) * lsb)
        g.create_dataset("t_master_us", data=t)
        g.attrs["sample_rate_sps"] = fs
        g.attrs["gain"] = gain
        g.attrs["lsb_uv"] = lsb
        g.attrs["channel_names"] = np.array(FULL_CHANNEL_NAMES, dtype="S")
        for name in ("imu_mta", "imu_cra"):
            ig = f.create_group(name)
            ig.create_dataset("quat", data=quat)
            ig.create_dataset("t_master_us_quat", data=it)
            ig.attrs["fusion_mode"] = "game_rotation_vector"
        tg = f.create_group("triggers")
        tg.create_dataset("trigger_id", data=np.arange(1, 11, dtype=np.uint64))
        tg.create_dataset("t_master_us", data=np.linspace(0, t[-1], 10, dtype=np.uint64))
        sync = f.create_group("sync")
        sync.attrs["report_json"] = json.dumps({"mcu_stats_end": {"missed_drdy": 0}})


# --------------------------- session reader ---------------------------

def test_montage_as_cabled_and_legacy_names_corrected_on_read(tmp_path: Path):
    """CH3 is the right, CH4 the left digastric (cable labels, hardware check 2026-11-03).
    Files written before carry CH3/CH4 = digastric_l/digastric_r and are read as cabled; the
    samples are not reordered, only the names."""
    from jtsmm.config.catalog import LEGACY_CHANNEL_NAMES_DIGASTRIC_SWAPPED

    assert FULL_CHANNEL_NAMES[2:4] == ["digastric_r", "digastric_l"]
    new, old = tmp_path / "new.h5", tmp_path / "old.h5"
    _write_session(new, n_emg=400, fs=2000)
    _write_session(old, n_emg=400, fs=2000)
    with h5py.File(old, "a") as f:
        f["emg"].attrs["channel_names"] = np.array(LEGACY_CHANNEL_NAMES_DIGASTRIC_SWAPPED,
                                                   dtype="S")
    with Session.open(new) as a, Session.open(old) as b:
        assert a.emg.channel_names == list(FULL_CHANNEL_NAMES) and a.emg.montage_note is None
        assert b.emg.channel_names == list(FULL_CHANNEL_NAMES)
        assert "corrected on read" in b.emg.montage_note
        np.testing.assert_array_equal(a.emg.window(0, 50)[1], b.emg.window(0, 50)[1])


def test_session_reader_windows_and_meta(tmp_path: Path):
    p = tmp_path / "s.h5"
    _write_session(p, n_emg=4000, fs=2000)
    with Session.open(p) as s:
        assert s.emg is not None and s.emg.n == 4000
        assert s.emg.fs_hz == 2000 and s.emg.n_channels == 8
        assert s.emg.channel_names[0] == "masseter_r"
        t, uv = s.emg.window(0, 100)
        assert t.shape == (100,) and uv.shape == (100, 8)
        # index_at monotonic seek
        idx = s.emg.index_at(int(s.emg.t_us[500]))
        assert idx == 500
        assert s.imu_mta is not None and s.imu_mta.n == 800
        assert s.triggers is not None and s.triggers.n == 10
        assert s.sync_report["mcu_stats_end"]["missed_drdy"] == 0
        assert s.duration_s == pytest.approx(2.0, abs=0.01)
        assert len(s.event_times_us()) == 10


def test_session_reader_missing_groups(tmp_path: Path):
    p = tmp_path / "empty.h5"
    with h5py.File(p, "w") as f:
        f.create_group("meta").attrs["config_hash"] = "x"
    with Session.open(p) as s:
        assert s.emg is None and s.imu_mta is None and s.triggers is None
        assert s.sync_report == {}


# --------------------------- leadoff decode ---------------------------

def test_leadoff_decode_bits():
    # LOFF_STATP = 0x0F (ch0..3 off), LOFF_STATN = 0xF0 (ch4..7 off), gpio=0
    # 24-bit: 1100 | statp[8] | statn[8] | gpio[4]
    word = (0b1100 << 20) | (0x0F << 12) | (0xF0 << 4) | 0x0
    status = word.to_bytes(3, "big")
    st = decode_status_word(status)
    assert st.statp[:4] == [True, True, True, True]
    assert st.statp[4:] == [False] * 4
    assert st.statn[4:] == [True, True, True, True]
    assert st.statn[:4] == [False] * 4


def test_leadoff_empty():
    st = decode_status_word(None)
    assert st.statp == [False] * 8


def test_saturation_flags():
    lsb = lsb_uv(6)
    arr = np.zeros((4, 8), dtype=np.float32)
    arr[1, 3] = (1 << 23) * lsb  # rail channel 3
    flags = saturation_flags(arr, lsb)
    assert flags[3] is True
    assert flags[0] is False


# --------------------------- processing -------------------------------

def test_causal_filter_shapes_and_envelope():
    f = CausalEmgFilter(500.0, 8)
    x = (np.random.default_rng(0).standard_normal((200, 8)) * 30).astype(np.float32)
    filt, env = f.process(x)
    assert filt.shape == (200, 8) and env.shape == (200, 8)
    assert np.all(env >= 0)


def test_causal_filter_blocks_dc():
    f = CausalEmgFilter(500.0, 1)
    const = np.full((2000, 1), 1000.0, np.float32)
    filt, _ = f.process(const)
    # HPF removes DC — tail settles near zero.
    assert abs(float(filt[-200:].mean())) < 5.0


def test_causal_filter_line_harmonics():
    """50 Hz plus in-band harmonics; skip tones at/near Nyquist."""
    f2k = CausalEmgFilter(2000.0, 1)
    assert f2k.notch_freqs == [50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0]
    f500 = CausalEmgFilter(500.0, 1)
    assert f500.notch_freqs == [50.0, 100.0, 150.0, 200.0]


def test_causal_filter_notches_line_harmonics():
    fs = 2000.0
    n = int(fs * 2.0)
    t = np.arange(n) / fs
    tail = int(0.5 * fs)
    for f0 in (50.0, 100.0, 150.0, 250.0):
        x = (80.0 * np.sin(2 * np.pi * f0 * t)).astype(np.float32).reshape(-1, 1)
        filt, _ = CausalEmgFilter(fs, 1).process(x)
        rms = float(np.sqrt(np.mean(np.square(filt[-tail:]))))
        assert rms < 8.0, f"{f0} Hz residual RMS {rms:.2f} µV"


def test_hub_notches_250hz_before_decimation():
    """250 Hz is Nyquist at 500 Hz display — notch it at native 2 kHz first."""
    fs = 2000
    hub = Hub(emg_fs=fs, emg_lsb_uv=lsb_uv(6), n_channels=1, display_fs=500)
    n = fs * 2
    t_us = np.arange(n, dtype=np.uint64) * int(1e6 / fs)
    tt = np.arange(n) / fs
    uv = (80.0 * np.sin(2 * np.pi * 250.0 * tt)).astype(np.float32)[:, None]
    hub.append_emg_raw(t_us, uv)
    snap = hub.snapshot()
    assert snap.emg_filt is not None
    tail = snap.emg_filt[-250:]
    rms = float(np.sqrt(np.mean(np.square(tail))))
    assert rms < 8.0


# --------------------------- hub --------------------------------------

def test_hub_decimation_and_band():
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), n_channels=8, display_fs=500)
    assert hub.decim == 4
    n = 400
    t = np.arange(n, dtype=np.uint64) * 500
    uv = np.tile(np.linspace(-50, 50, n, dtype=np.float32)[:, None], (1, 8))
    hub.append_emg_raw(t, uv)
    snap = hub.snapshot()
    assert snap.emg_n == n // 4  # decimated
    assert np.all(snap.emg_lo <= snap.emg_hi + 1e-3)
    assert snap.emg_channels == 8


def test_hub_snapshot_clears_pending():
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=2000)
    hub.append_emg_raw(np.arange(10, dtype=np.uint64) * 500,
                       np.ones((10, 8), np.float32))
    assert hub.snapshot().emg_n == 10
    assert hub.snapshot().emg_n == 0  # drained


# --------------------------- sources ----------------------------------

def test_tap_source_fake_frame():
    """TapSource against a fake native cap (no device)."""
    n = 16
    frame = SimpleNamespace(
        emg_t=(np.arange(n, dtype=np.uint64) * 2000),
        emg_min=np.full((n, 8), -100, np.int32),
        emg_max=np.full((n, 8), 100, np.int32),
        emg_seen=n,
        status=bytes([0xC0, 0x00, 0x00]),
        mta_quat=(1.0, 0.0, 0.0, 0.0), mta_t=1000,
        cra_quat=(1.0, 0.0, 0.0, 0.0), cra_t=1000,
        stats={"missed_drdy": 0, "emg_drops": 0, "tx_drops": 0},
    )

    class FakeCap:
        def __init__(self):
            self.enabled = None

        def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
            self.enabled = (enabled, bucket_samples)

        def get_viewer_frame(self):
            return frame

    cap = FakeCap()
    src = TapSource(cap, emg_fs=2000, emg_gain=6, bucket_samples=4)
    hub = src.make_hub()
    src.poll_once(hub)
    snap = hub.snapshot()
    assert snap.emg_n == n
    lsb = lsb_uv(6)
    assert snap.emg_hi.max() == pytest.approx(100 * lsb, rel=1e-3)
    assert snap.stats["missed_drdy"] == 0


def test_mock_source_hub_smoke():
    src = MockSource(fs=2000, gain=6)
    hub = src.make_hub()
    src.start(hub)
    import time
    time.sleep(0.2)
    src.stop()
    snap = hub.snapshot()
    assert snap.emg_n > 0
    assert snap.imu_mta.quat is not None


def test_mock_source_synthesizes_all_imu_streams():
    """MockSource drives non-empty accel/gyro/mag/lin plus tagged quat + derived."""
    src = MockSource(fs=2000, gain=6)
    hub = src.make_hub()
    src.start(hub)
    import time
    time.sleep(0.25)
    src.stop()
    snap = hub.snapshot()
    assert snap.imu_mta.quat_report_id == MockSource._MOCK_QUAT_REPORT_ID
    for stream in ("accel", "gyro", "mag", "linear_accel"):
        t, v = snap.imu_mta.motion[stream]
        assert t is not None and v is not None
        assert t.shape[0] > 0 and v.shape[1] == 3
    # Derived relative orientation + ZUPT-bounded position estimate are present.
    assert snap.imu_q_rel is not None
    assert snap.imu_pos_rel is not None


def test_pipeline_snapshot_roundtrip_all_streams():
    """Snapshot with motion windows + derived encodes/decodes losslessly."""
    pytest.importorskip("msgpack")
    from jtsmm.viewer.protocol import decode, encode

    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=2000)
    hub.append_emg_raw(np.arange(4, dtype=np.uint64) * 500, np.zeros((4, 8), np.float32))
    hub.set_imu_mta((1.0, 0.0, 0.0, 0.0), 1000, report_id=0x08)
    hub.set_imu_cra((1.0, 0.0, 0.0, 0.0), 1000, report_id=0x08)
    for i in range(5):
        t = 1000 + i * 100
        hub.push_imu_motion("mta", "accel", t, (0.1 * i, 0.0, 9.81))
        hub.push_imu_motion("mta", "gyro", t, (0.0, 0.2 * i, 0.0))
        hub.push_imu_motion("mta", "mag", t, (25.0, -8.0, 40.0))
        hub.push_imu_motion("mta", "linear_accel", t, (0.05 * i, 0.0, 0.0))
        hub.push_imu_motion("cra", "linear_accel", t, (0.0, 0.0, 0.0))
    d = decode(encode(hub.snapshot()))
    mta = d["imu"]["mta"]
    assert mta["qid"] == 0x08
    assert mta["motion"]["accel"]["v"].shape == (5, 3)
    assert np.allclose(mta["motion"]["mag"]["v"][0], [25.0, -8.0, 40.0], atol=1e-3)
    assert d["imu"]["derived"] is not None
    assert "q_rel" in d["imu"]["derived"]


def test_hub_integrates_lin_accel_with_interpolated_quat():
    """Relative translation uses orientation at each linear-accel timestamp."""
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=2000)
    # Quats at 0 and 2 ms; linear accel only at 1 ms (between quat samples).
    hub.set_imu_mta((1.0, 0.0, 0.0, 0.0), 0)
    hub.set_imu_cra((1.0, 0.0, 0.0, 0.0), 0)
    hub.set_imu_mta((1.0, 0.0, 0.0, 0.0), 2000)
    hub.set_imu_cra((1.0, 0.0, 0.0, 0.0), 2000)
    hub.push_imu_motion("mta", "linear_accel", 1000, (0.5, 0.0, 0.0))
    hub.push_imu_motion("cra", "linear_accel", 1000, (0.0, 0.0, 0.0))
    snap = hub.snapshot()
    assert snap.imu_pos_rel is not None


def test_hub_camera_trigger_rate():
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=2000)
    for i in range(60):
        hub.note_trigger_events(1_000_000 + i * 16_667, 1)  # ~60 Hz
    snap = hub.snapshot()
    assert abs(snap.camera_hz - 60.0) < 5.0


def test_tap_source_quat_batch():
    """TapSource forwards batched quat/motion arrays when present."""
    n = 12
    times = np.arange(n, dtype=np.uint64) * 2500 + 1000
    quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))
    lin_t = np.arange(8, dtype=np.uint64) * 4500 + 1000
    lin_v = np.zeros((8, 3), np.float32)
    lin_v[:, 0] = 0.1
    frame = SimpleNamespace(
        emg_t=np.empty(0, dtype=np.uint64),
        emg_min=np.empty((0, 8), np.int32),
        emg_max=np.empty((0, 8), np.int32),
        status=None,
        mta_quat_times=times,
        mta_quat_vals=quats,
        mta_quat_report_id=0x08,
        mta_t=int(times[-1]),
        mta_cal={"quat": 3, "linear_accel": 2},
        cra_quat_times=times,
        cra_quat_vals=quats,
        cra_quat_report_id=0x08,
        cra_t=int(times[-1]),
        cra_cal={"quat": 3},
        mta_lin_t=lin_t,
        mta_lin_v=lin_v,
        cra_lin_t=lin_t,
        cra_lin_v=lin_v,
        trig_seen=2,
        stats=None,
    )

    class FakeCap:
        def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
            pass

        def get_viewer_frame(self):
            return frame

    src = TapSource(FakeCap(), emg_fs=2000, emg_gain=6, bucket_samples=4)
    hub = src.make_hub()
    src.poll_once(hub)
    # Measured rates need two tap windows; advance timestamps for a second poll.
    frame.mta_quat_times = times + 33_000
    frame.cra_quat_times = times + 33_000
    frame.mta_lin_t = lin_t + 33_000
    frame.cra_lin_t = lin_t + 33_000
    frame.mta_t = int(times[-1]) + 33_000
    frame.cra_t = int(times[-1]) + 33_000
    frame.trig_seen = 2
    src.poll_once(hub)
    snap = hub.snapshot()
    assert abs(snap.imu_mta.rates["quat"] - 400.0) < 80.0
    assert abs(snap.imu_mta.rates["linear_accel"] - (1e6 / 4500)) < 30.0
    t, v = snap.imu_mta.motion["linear_accel"]
    assert t.shape[0] == 16  # two tap polls × 8 lin samples
    assert snap.camera_hz > 0


def test_hub_reports_measured_rates():
    """Hub exposes true per-stream Hz (incl. the coalesced quat) in snapshots."""
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=2000)
    hub.append_emg_raw(np.arange(2, dtype=np.uint64) * 500, np.zeros((2, 8), np.float32))
    # 100 quats @ 400 Hz (2500 µs) and 100 gyro samples @ 200 Hz (5000 µs).
    for i in range(100):
        hub.set_imu_mta((1.0, 0.0, 0.0, 0.0), 1000 + i * 2500, report_id=0x08)
        hub.push_imu_motion("mta", "gyro", 1000 + i * 5000, (0.0, 0.0, 0.0))
    snap = hub.snapshot()
    rates = snap.imu_mta.rates
    assert abs(rates["quat"] - 400.0) < 8.0
    assert abs(rates["gyro"] - 200.0) < 4.0

    from jtsmm.viewer.protocol import decode, encode

    pytest.importorskip("msgpack")
    d = decode(encode(snap))
    assert abs(d["imu"]["mta"]["rates"]["quat"] - 400.0) < 8.0


def test_tap_source_uses_emg_seen_for_rate():
    """Tap path counts raw EMG samples via emg_seen delta, not buckets * decim."""
    n = 16
    frame = SimpleNamespace(
        emg_t=(np.arange(n, dtype=np.uint64) * 2000),
        emg_min=np.full((n, 8), -100, np.int32),
        emg_max=np.full((n, 8), 100, np.int32),
        emg_seen=64,
        status=None,
        mta_quat=None, mta_t=0, cra_quat=None, cra_t=0,
        stats=None,
    )

    class FakeCap:
        def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
            pass

        def get_viewer_frame(self):
            return frame

    src = TapSource(FakeCap(), emg_fs=2000, emg_gain=6, bucket_samples=4)
    hub = src.make_hub()
    src.poll_once(hub)  # seeds _last_emg_seen=64
    frame.emg_seen = 128
    frame.emg_t = (np.arange(n, dtype=np.uint64) * 2000) + 32_000
    src.poll_once(hub)
    snap = hub.snapshot()
    assert abs(snap.emg_hz - 2000.0) < 80.0


def test_hub_reports_measured_emg_rate():
    """Hub exposes the true EMG ADC rate (raw, pre-decimation) in snapshots."""
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=500)  # decim = 4
    # Two 1000-sample batches at 2000 Hz (500 µs spacing) → measured ~2000 Hz
    # even though display rows are decimated to 500 Hz.
    for base in (0, 1000):
        t = (np.arange(base, base + 1000, dtype=np.uint64)) * 500
        hub.append_emg_raw(t, np.zeros((1000, 8), np.float32))
    snap = hub.snapshot()
    assert abs(snap.emg_hz - 2000.0) < 40.0

    pytest.importorskip("msgpack")
    from jtsmm.viewer.protocol import decode, encode

    d = decode(encode(snap))
    assert abs(d["emg"]["hz"] - 2000.0) < 40.0
    assert d["emg"]["fs"] == 2000.0  # configured target unchanged


def test_hub_reports_calibration_status():
    hub = Hub(emg_fs=2000, emg_lsb_uv=lsb_uv(6), display_fs=2000)
    hub.set_imu_mta((1.0, 0.0, 0.0, 0.0), 1000, report_id=0x08, accuracy=3)
    hub.push_imu_motion("mta", "gyro", 1000, (0.0, 0.0, 0.0), accuracy=2)
    snap = hub.snapshot()
    assert snap.imu_mta.cal == {"quat": 3, "gyro": 2}

    pytest.importorskip("msgpack")
    from jtsmm.viewer.protocol import decode, encode

    d = decode(encode(snap))
    assert d["imu"]["mta"]["cal"] == {"quat": 3, "gyro": 2}


def test_tap_source_motion_and_tagged_quat():
    """TapSource forwards native motion vectors + tagged report ids."""
    n = 8
    frame = SimpleNamespace(
        emg_t=(np.arange(n, dtype=np.uint64) * 2000),
        emg_min=np.full((n, 8), -10, np.int32),
        emg_max=np.full((n, 8), 10, np.int32),
        emg_seen=n,
        status=None,
        mta_quat=(1.0, 0.0, 0.0, 0.0), mta_t=1000, mta_quat_report_id=0x05,
        cra_quat=(1.0, 0.0, 0.0, 0.0), cra_t=1000, cra_quat_report_id=0x05,
        mta_accel=(0.0, 0.0, 9.81), mta_gyro=(0.1, 0.0, 0.0),
        mta_mag=(25.0, 0.0, 40.0), mta_lin=(0.2, 0.0, 0.0),
        mta_cal={"quat": 3, "accel": 2, "gyro": 3, "linear_accel": 2},
        cra_accel=(0.0, 0.0, 9.81), cra_gyro=None, cra_mag=None, cra_lin=(0.0, 0.0, 0.0),
        cra_cal={"quat": 3},
        stats=None,
    )

    class FakeCap:
        def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
            pass

        def get_viewer_frame(self):
            return frame

    src = TapSource(FakeCap(), emg_fs=2000, emg_gain=6, bucket_samples=4)
    hub = src.make_hub()
    src.poll_once(hub)
    snap = hub.snapshot()
    assert snap.imu_mta.quat_report_id == 0x05
    assert snap.imu_mta.cal == {"quat": 3, "accel": 2, "gyro": 3, "linear_accel": 2}
    assert snap.imu_cra.cal == {"quat": 3}
    t, v = snap.imu_mta.motion["accel"]
    assert v is not None and np.allclose(v[-1], [0.0, 0.0, 9.81], atol=1e-3)
    assert "gyro" not in snap.imu_cra.motion  # cra_gyro was None


# --------------------------- protocol ---------------------------------

def test_protocol_roundtrip():
    pytest.importorskip("msgpack")
    from jtsmm.viewer.protocol import Snapshot, decode, encode

    s = Snapshot(
        seq=7, t_ref_us=123456, t0_us=1000, dropped=2,
        emg_n=3, emg_channels=8, emg_fs=2000.0, emg_lsb_uv=lsb_uv(6),
        emg_t=np.array([0.0, 0.001, 0.002], np.float32),
        emg_lo=np.full((3, 8), -1.0, np.float32),
        emg_hi=np.full((3, 8), 2.0, np.float32),
        emg_filt=np.zeros((3, 8), np.float32),
        emg_env=np.ones((3, 8), np.float32),
    )
    d = decode(encode(s))
    assert d["seq"] == 7 and d["dropped"] == 2
    assert d["emg"]["n"] == 3 and d["emg"]["c"] == 8
    assert d["emg"]["hi"].shape == (3, 8)
    assert np.allclose(d["emg"]["hi"], 2.0)
    assert np.allclose(d["emg"]["t"], [0.0, 0.001, 0.002], atol=1e-6)


def test_hub_camera_jpeg_slots():
    from jtsmm.viewer.hub import Hub
    from jtsmm.viewer.jpeg_util import PLACEHOLDER_JPEG, encode_jpeg_rgb, synthetic_preview_rgb

    hub = Hub(emg_fs=2000.0, emg_lsb_uv=lsb_uv(6), n_channels=7)
    hub.set_camera_ids(["cam0"])
    assert hub.camera_ids() == ["cam0"]
    assert hub.get_camera_jpeg("cam0") is None
    jpeg = encode_jpeg_rgb(synthetic_preview_rgb("cam0", t=0.5))
    hub.update_camera_jpeg("cam0", jpeg)
    assert hub.get_camera_jpeg("cam0") == jpeg
    hub.set_camera_ids([])
    assert hub.camera_ids() == []
    assert hub.get_camera_jpeg("cam0") is None
    assert len(PLACEHOLDER_JPEG) > 20 and PLACEHOLDER_JPEG[:2] == b"\xff\xd8"


def test_aruco_overlay_detects_generated_marker():
    cv2 = pytest.importorskip("cv2")
    from cv2 import aruco

    from jtsmm.viewer.aruco_overlay import draw_aruco_overlay_bgr

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    tag = aruco.generateImageMarker(dictionary, 7, 120)
    canvas = np.full((240, 320), 255, dtype=np.uint8)
    canvas[60:180, 100:220] = tag
    bgr = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    out = draw_aruco_overlay_bgr(bgr)
    assert out.shape == bgr.shape
    # Green pixels should appear where the box/label was drawn.
    assert np.any(out[:, :, 1] > out[:, :, 0])


def test_draw_marker_boxes_from_tracker_corners_no_detect():
    pytest.importorskip("cv2")
    from jtsmm.viewer.aruco_overlay import draw_marker_boxes_rgb

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    # Corners in 640×480; preview is 320×240 so they scale by 1/2.
    corners = np.array(
        [[100.0, 80.0], [200.0, 80.0], [200.0, 180.0], [100.0, 180.0]],
        dtype=np.float64,
    )
    out = draw_marker_boxes_rgb(rgb, [(7, corners)], src_wh=(640, 480))
    assert out.shape == rgb.shape
    assert np.any(out[:, :, 1] > out[:, :, 0])
    # Scaled top-left should be near (50, 40).
    assert out[40, 50, 1] >= 200
    assert out[40, 50, 0] == 0
    gray = np.zeros((240, 320), dtype=np.uint8)
    over = draw_marker_boxes_rgb(gray, [(7, corners)], src_wh=(640, 480))
    assert over.ndim == 3 and over.shape[2] == 3
    assert over[40, 50, 1] >= 200


def test_encode_jpeg_preview_quality_80():
    from jtsmm.viewer.cam_preview import PREVIEW_SIZE
    from jtsmm.viewer.jpeg_util import PREVIEW_JPEG_QUALITY, encode_jpeg, encode_jpeg_rgb, synthetic_preview_rgb

    assert PREVIEW_JPEG_QUALITY == 80
    assert PREVIEW_SIZE == (320, 240)
    rgb = synthetic_preview_rgb("cam0", size=PREVIEW_SIZE, t=0.5)
    q80 = encode_jpeg_rgb(rgb)
    q100 = encode_jpeg_rgb(rgb, quality=100)
    assert q80[:2] == b"\xff\xd8" and q100[:2] == b"\xff\xd8"
    assert len(q80) <= len(q100)
    gray = rgb[:, :, 0]
    gj = encode_jpeg(gray)
    assert gj[:2] == b"\xff\xd8"
    assert len(gj) > 20


def test_optical_overlay_markers_are_copied():
    import threading

    from jtsmm.capture.optical_bridge import OpticalTrackBridge

    corners = np.array(
        [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]], dtype=np.float64
    )
    marker = SimpleNamespace(marker_id=4, corners=corners)
    det = SimpleNamespace(markers=(marker,))
    bridge = object.__new__(OpticalTrackBridge)
    bridge._overlay_lock = threading.Lock()
    bridge.tracker = SimpleNamespace(last_detections={"cam0": det})
    rgb = np.zeros((544, 728, 3), dtype=np.uint8)
    bridge._store_overlay({"cam0": rgb})
    packed = bridge.overlay_markers("cam0")
    assert packed is not None
    markers, src_wh = packed
    assert src_wh == (728, 544)
    assert markers[0][0] == 4
    corners[0, 0] = 999.0
    assert markers[0][1][0, 0] == 10.0
    assert bridge.overlay_markers("cam1") is None


def test_server_ws_smoke():
    pytest.importorskip("msgpack")
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx2")
    from starlette.testclient import TestClient

    from jtsmm.viewer.protocol import decode
    from jtsmm.viewer.server import build_app

    src = MockSource(fs=2000, gain=6, n_cameras=1)
    hub = src.make_hub()
    src.start(hub)
    try:
        time.sleep(0.15)  # let mock preview publish one JPEG
        app = build_app(src, hub)
        client = TestClient(app)
        r = client.get("/api/session")
        assert r.status_code == 200 and r.json()["kind"] == "mock"
        assert r.json()["camera_ids"] == ["cam0"]
        assert hub.get_camera_jpeg("cam0") is not None
        with client.websocket_connect("/ws") as ws:
            frame = decode(ws.receive_bytes())
            assert frame["type"] == "frame"
            assert frame["emg"]["c"] == 7
        # Route exists (don't consume the infinite MJPEG generator).
        paths = {getattr(route, "path", None) for route in app.routes}
        assert "/stream/{cam_id}" in paths
    finally:
        src.stop()


def test_hub_filtered_trace_uses_bucket_mean_not_midpoint():
    """The filtered/envelope input must be the bucket mean (anti-aliased), not (min+max)/2."""
    from jtsmm.viewer.hub import Hub

    # 8 samples -> 2 buckets of 4 with values [0,0,0,100]: mean 25, midpoint 50.
    t = np.arange(8, dtype=np.uint64) * 500
    uv = np.array([[0], [0], [0], [100]] * 2, dtype=np.float32)
    hub = Hub(emg_fs=2000.0, emg_lsb_uv=1.0, n_channels=1, display_fs=500.0, filter_enabled=False)
    hub.append_emg_raw(t, uv)
    snap = hub.snapshot()
    assert snap.emg_filt.shape == (2, 1)
    assert np.allclose(snap.emg_filt[:, 0], 25.0)
    assert np.allclose(snap.emg_lo[:, 0], 0.0) and np.allclose(snap.emg_hi[:, 0], 100.0)

    # Tap path: explicit mean wins; without one the midpoint is the documented fallback.
    lo = np.array([[0.0]], dtype=np.float32)
    hi = np.array([[100.0]], dtype=np.float32)
    hub_mean = Hub(emg_fs=2000.0, emg_lsb_uv=1.0, n_channels=1, display_fs=500.0, filter_enabled=False)
    hub_mean.append_emg_rows(np.array([0], dtype=np.uint64), lo, hi, mean_uv=np.array([[25.0]]))
    assert np.allclose(hub_mean.snapshot().emg_filt[:, 0], 25.0)
    hub_mid = Hub(emg_fs=2000.0, emg_lsb_uv=1.0, n_channels=1, display_fs=500.0, filter_enabled=False)
    hub_mid.append_emg_rows(np.array([0], dtype=np.uint64), lo, hi)
    assert np.allclose(hub_mid.snapshot().emg_filt[:, 0], 50.0)


def test_tap_source_forwards_bucket_mean():
    n = 4
    frame = SimpleNamespace(
        emg_t=(np.arange(n, dtype=np.uint64) * 2000),
        emg_min=np.full((n, 8), 0, np.int32),
        emg_max=np.full((n, 8), 100, np.int32),
        emg_mean=np.full((n, 8), 10.0, np.float32),
        emg_seen=n * 4,
        status=None, mta_quat=None, cra_quat=None, mta_t=0, cra_t=0, stats=None,
    )

    class FakeCap:
        def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
            pass

        def get_viewer_frame(self):
            return frame

    src = TapSource(FakeCap(), emg_fs=2000, emg_gain=6, bucket_samples=4)
    hub = src.make_hub()
    hub._filter = None  # inspect the raw input to the display filter
    src.poll_once(hub)
    snap = hub.snapshot()
    # Muscle columns at the profile gain, the CH8/DRL column at its fixed gain 24.
    assert np.allclose(snap.emg_filt[:, :6], 10.0 * lsb_uv(6), rtol=1e-4)
    assert np.allclose(snap.emg_filt[:, 6], 10.0 * lsb_uv(24), rtol=1e-4)


def _tone_gain(fs: float, freq: float, **kw) -> float:
    """Steady-state gain of CausalEmgFilter at one frequency."""
    from jtsmm.viewer.processing import CausalEmgFilter

    t = np.arange(int(fs * 12)) / fs
    x = np.sin(2 * np.pi * freq * t).reshape(-1, 1)
    y, _ = CausalEmgFilter(fs, 1, **kw).process(x)
    tail = y[-int(fs * 4) :, 0]
    return float(np.sqrt(np.mean(tail**2)) * np.sqrt(2))


def test_filter_notches_four_mains_harmonics_by_default():
    """Clipped mains is harmonic-rich; notching only 50 Hz left 100/150/200 Hz on screen."""
    fs = 500.0  # the display rate the hub uses
    for f0 in (50.0, 100.0, 150.0, 200.0):
        assert _tone_gain(fs, f0) < 0.02, f0


def test_filter_preserves_the_emg_band_between_harmonics():
    fs = 500.0
    for f0 in (60.0, 75.0, 90.0, 110.0, 125.0, 160.0, 175.0):
        g = _tone_gain(fs, f0)
        assert g > 0.95, (f0, g)  # < 0.45 dB loss


def test_notch_q_scales_with_harmonic_so_width_stays_constant():
    """Each notch must stay ~5 Hz wide instead of widening with frequency."""
    fs = 500.0
    # 5 Hz off each harmonic the response is already back up; a constant-Q
    # notch at 200 Hz would be 20 Hz wide and still deep at 195/205 Hz.
    for f0 in (100.0, 150.0, 200.0):
        for off in (-7.5, 7.5):
            g = _tone_gain(fs, f0 + off)
            assert g > 0.80, (f0, off, g)


def test_notch_harmonics_is_still_overridable():
    fs = 500.0
    assert _tone_gain(fs, 100.0, notch_harmonics=1) > 0.95   # 2nd harmonic passes
    assert _tone_gain(fs, 50.0, notch_harmonics=1) < 0.02    # fundamental still notched
    assert _tone_gain(fs, 50.0, notch_harmonics=0) > 0.95    # notches off entirely


def test_harmonics_above_nyquist_are_skipped():
    """At a 150 Hz display rate only the 50 Hz fundamental fits below Nyquist."""
    from jtsmm.viewer.processing import CausalEmgFilter

    # Assert the harmonic list itself: with mains_mode="track" the whole bank is
    # one section, so a section count would pass here for the wrong reason.
    assert CausalEmgFilter(150.0, 1).notch_freqs == [50.0]
    f = CausalEmgFilter(150.0, 1, mains_mode="notch")
    # hpf + 1 notch + lpf = 3 sections; a 100 Hz notch would be above Nyquist.
    assert len(f._sections) == 3


# --- mains frequency comes from config / the recorded session -----------------


def test_hub_builds_its_notch_bank_at_the_configured_line_frequency():
    from jtsmm.viewer.hub import Hub

    h60 = Hub(emg_fs=2000.0, emg_lsb_uv=1.0, n_channels=1, display_fs=500.0, line_hz=60.0)
    assert h60.line_hz == 60.0
    # 60/120/180 notched, 50/100/150 left alone.
    for f0 in (60.0, 120.0, 180.0):
        assert _tone_gain_via_hub(h60, f0) < 0.05, f0
    for f0 in (50.0, 100.0, 150.0):
        assert _tone_gain_via_hub(h60, f0) > 0.90, f0


def _tone_gain_via_hub(hub, freq: float) -> float:
    """Steady-state gain of a hub's own display filter at one frequency."""
    fs = hub.display_fs
    t = np.arange(int(fs * 12)) / fs
    x = np.sin(2 * np.pi * freq * t).reshape(-1, 1)
    hub._filter.reset()
    y, _ = hub._filter.process(x)
    tail = y[-int(fs * 4) :, 0]
    return float(np.sqrt(np.mean(tail**2)) * np.sqrt(2))


def test_hub_defaults_to_50_hz():
    from jtsmm.viewer.hub import Hub

    h = Hub(emg_fs=2000.0, emg_lsb_uv=1.0, n_channels=1, display_fs=500.0)
    assert h.line_hz == 50.0
    assert _tone_gain_via_hub(h, 50.0) < 0.05


def test_tap_source_passes_line_hz_through_to_hub_and_meta():
    class FakeCap:
        def set_viewer_tap_enabled(self, enabled, bucket_samples=4):
            pass

        def get_viewer_frame(self):
            return None

    src = TapSource(FakeCap(), emg_fs=2000, emg_gain=6, bucket_samples=4, line_hz=60.0)
    assert src.meta()["line_hz"] == 60.0
    assert src.make_hub().line_hz == 60.0


def test_session_line_hz_read_from_recorded_host_config():
    import json

    from jtsmm.viewer.sources import _session_line_hz

    assert _session_line_hz({}) == 50.0
    assert _session_line_hz({"host_config_json": "not json"}) == 50.0
    doc = {"host": {"emg_check": {"line_hz": 60.0}}}
    assert _session_line_hz({"host_config_json": json.dumps(doc)}) == 60.0
    assert _session_line_hz({"host_config_json": json.dumps(doc).encode()}) == 60.0
    # Missing key, and a zero/garbage value, both fall back to the default.
    assert _session_line_hz({"host_config_json": json.dumps({"host": {}})}) == 50.0
    assert _session_line_hz(
        {"host_config_json": json.dumps({"host": {"emg_check": {"line_hz": 0}}})}
    ) == 50.0
