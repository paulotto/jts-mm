# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Native capture parse thread + HDF5 pump vs the Python path, fed raw wire bytes.

``NativeMcuCapture.inject_raw`` pushes bytes through the same queue and parse
thread the USB reader uses, so this covers the native hot path end to end
(minus libusb): decode of every record type and what lands in the HDF5 file.
"""

from __future__ import annotations

import struct
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm import native
from jtsmm.config.catalog import channel_lsb_uv
from jtsmm.protocol import FrameParser as PyFrameParser
from jtsmm.protocol.types import RecordType
from jtsmm.sync.clock_align import AlignedRecord

from tests.helpers import build_frame
from tests.test_native_capture import _as_comparable, _firmware_stream

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)

_MODALITY = {
    int(RecordType.EMG_BLOCK): "emg",
    int(RecordType.TRIGGER_EVENT): "trigger",
    **{int(t): "imu_mta" for t in (RecordType.IMU_MTA_QUAT, RecordType.IMU_MTA_RAW,
                                   RecordType.IMU_MTA_LIN_ACCEL)},
    **{int(t): "imu_cra" for t in (RecordType.IMU_CRA_QUAT, RecordType.IMU_CRA_RAW,
                                   RecordType.IMU_CRA_LIN_ACCEL)},
}


def _inject(cap, stream: bytes, expect: int, chunk: int = 61) -> None:
    for i in range(0, len(stream), chunk):  # odd chunking straddles frames
        cap.inject_raw(stream[i : i + chunk])
    deadline = time.monotonic() + 5.0
    while cap.records_rx < expect and time.monotonic() < deadline:
        time.sleep(0.005)
    assert cap.records_rx == expect


def _session_stream(n_emg: int = 3000) -> bytes:
    """A plausible take in the formats current firmware sends."""
    rng = np.random.default_rng(1)
    frames = []
    t = 10_000
    for i in range(n_emg):
        status = bytes([0xC0 | (i & 0x0F), (i * 7) & 0xFF, (i * 13) & 0xF0])
        chans = b"".join(int(v & 0xFFFFFF).to_bytes(3, "big")
                         for v in rng.integers(-(1 << 23), 1 << 23, size=8))
        frames.append(build_frame(0x10, t, status + chans + struct.pack("<I", 1000 + i)))
        if i % 5 == 0:
            for rt_q, rt_r, rt_l, off in ((0x11, 0x12, 0x13, 0), (0x14, 0x15, 0x16, 7)):
                q = struct.pack("<4f", *rng.normal(size=4).astype(np.float32))
                frames.append(build_frame(rt_q, t + 1 + off, bytes([0x08]) + q + bytes([i & 3])))
                for sid in (1, 2, 3):
                    xyz = struct.pack("<3h", *rng.integers(-32768, 32767, size=3))
                    frames.append(build_frame(rt_r, t + 2 + off + sid, bytes([sid]) + xyz + bytes([2])))
                lin = struct.pack("<3f", *rng.normal(size=3).astype(np.float32))
                frames.append(build_frame(rt_l, t + 6 + off, lin + bytes([1])))
        if i % 33 == 0:
            frames.append(build_frame(0x20, t + 3, struct.pack("<4BQ", 1, 0, 0, 0, i // 33)))
        if i == n_emg // 2:
            frames.append(build_frame(0x12, t + 4, struct.pack("<9h", *range(9))))  # legacy raw
            frames.append(build_frame(0x51, t + 5, struct.pack("<28I", *range(28))))
            frames.append(build_frame(0x22, t + 6, bytes([1, 1, 0, 0, 0])))
            frames.append(build_frame(0x70, t + 7, b"log line \xff"))
        t += 500
    frames.append(build_frame(0x51, t, struct.pack("<36I", *range(100, 136))))
    return b"".join(frames)


def test_inject_raw_decodes_like_python_for_every_record_type():
    stream = _firmware_stream()
    py_recs = PyFrameParser().feed(stream)
    cap = native.NativeMcuCapture()
    cap._start_parser_only()
    try:
        _inject(cap, stream, len(py_recs), chunk=7)
        c_recs = [cap.next_record(0.0) for _ in py_recs]
        assert cap.next_record(0.0) is None
    finally:
        cap.stop()
    for c, p in zip(c_recs, py_recs):
        assert _as_comparable(c) == _as_comparable(p), p.type_name


def test_inject_raw_pump_writes_same_hdf5_as_python_path(tmp_path: Path):
    from jtsmm.capture.native_backend import NativeWriter
    from jtsmm.io.hdf5_writer import Hdf5Writer

    mcu_cfg = {"emg": {"default_gain": 6, "sample_rate_sps": 2000}, "trigger": {"rate_hz": 60}}
    stream = _session_stream()
    py_recs = PyFrameParser().feed(stream)

    py_path = tmp_path / "py.h5"
    pw = Hdf5Writer(path=py_path, config_hash="x", mcu_config=mcu_cfg, host_config={})
    pw.open()
    for r in py_recs:
        mod = _MODALITY.get(r.record_type)
        if mod is not None:
            pw.append(AlignedRecord(mod, r.t_master_us, r.decoded))
    pw.close()

    c_path = tmp_path / "c.h5"
    nw = NativeWriter(path=c_path, config_hash="x", mcu_config=mcu_cfg, host_config={})
    nw.open()
    cap = native.NativeMcuCapture()
    cap._start_parser_only()
    by_type: dict[int, int] = {}
    last_stats = None
    logs: list[dict] = []
    try:
        _inject(cap, stream, len(py_recs))
        while True:
            out = cap.pump_to_writer(nw._writer, 1000)
            logs.extend(out["logs"])
            for k, v in out["by_type"].items():
                by_type[k] = by_type.get(k, 0) + v
            if out["last_stats"] is not None:
                last_stats = out["last_stats"]
            if not any(out[k] for k in ("emg", "mta", "cra", "trig")) and out["last_stats"] is None:
                break
    finally:
        cap.stop()
    nw.close()

    want = {}
    for r in py_recs:
        if r.record_type in _MODALITY:
            want[r.record_type] = want.get(r.record_type, 0) + 1
    assert by_type == want  # per record type, not quat/raw/lin lumped per slot
    # Firmware log lines come back to the caller (it prints and stores them), decoded
    # like the Python parser, with the MCU stamp.
    py_logs = [r for r in py_recs if r.record_type == int(RecordType.LOG)]
    assert [(lg["t_master_us"], lg["text"]) for lg in logs] == [
        (r.t_master_us, r.decoded["text"]) for r in py_logs
    ]
    assert logs[0]["text"] == "log line \ufffd"
    assert last_stats.decoded == py_recs[-1].decoded and len(last_stats.decoded) == 36

    with h5py.File(py_path, "r") as fp, h5py.File(c_path, "r") as fc:
        for name in ("samples", "t_master_us", "leadoff_status"):
            assert np.array_equal(fp["emg"][name][:], fc["emg"][name][:]), name
        seq = [r.decoded["seq"] for r in py_recs if r.record_type == 0x10]
        assert np.array_equal(fc["emg"]["seq"][:], np.asarray(seq, dtype=np.uint32))
        expect_uv = fc["emg"]["samples"][:].astype(np.float32) * np.asarray(
            channel_lsb_uv(6), dtype=np.float32)
        assert np.array_equal(fc["emg"]["samples_uv"][:], expect_uv)
        for grp in ("imu_mta", "imu_cra"):
            for name in ("quat", "t_master_us_quat", "quat_report_id", "accel", "gyro", "mag",
                         "linear_accel", "t_master_us_accel", "t_master_us_gyro",
                         "t_master_us_mag", "t_master_us_linear_accel", "quat_accuracy",
                         "accel_accuracy", "gyro_accuracy", "mag_accuracy",
                         "linear_accel_accuracy"):
                assert np.array_equal(fp[grp][name][:], fc[grp][name][:]), f"{grp}/{name}"
            # Calibration status per sample; the legacy 18 B raw record carries none.
            assert set(fc[grp]["quat_accuracy"][:]) == {0, 1, 2, 3}
            assert set(fc[grp]["linear_accel_accuracy"][:]) == {1}
            legacy = {2, 255} if grp == "imu_mta" else {2}
            assert set(fc[grp]["accel_accuracy"][:]) == legacy
            assert "unreliable" in fc[grp]["gyro_accuracy"].attrs["note"]
            assert fp[grp].attrs["quat_fusion"] == fc[grp].attrs["quat_fusion"]
        for name in ("trigger_id", "t_master_us"):
            assert np.array_equal(fp["triggers"][name][:], fc["triggers"][name][:])
        assert int(fc["triggers"].attrs["rate_hz"]) == 60


def test_link_health_reports_record_age():
    from jtsmm.capture.native_backend import NativeCapture

    cap = NativeCapture.__new__(NativeCapture)
    cap._cap = native.NativeMcuCapture()
    cap._cap._start_parser_only()
    try:
        time.sleep(0.15)
        h = cap.link_health()
        assert set(h) == {"device_lost", "usb_errors", "usb_stalls", "bytes_rx",
                          "last_record_age_s"}
        assert h["device_lost"] is False and h["last_record_age_s"] >= 0.1
        _inject(cap._cap, build_frame(0x10, 1, bytes(27)), 1)
        assert cap.link_health()["last_record_age_s"] < 0.1
        stats = cap.link_stats
        assert stats["records_rx"] == 1 and stats["device_lost"] == 0
    finally:
        cap.stop()


def test_reader_exposes_imu_accuracy(tmp_path: Path):
    from jtsmm.capture.native_backend import NativeWriter
    from jtsmm.io.session_reader import Session

    path = tmp_path / "acc.h5"
    nw = NativeWriter(path=path, config_hash="x", mcu_config={}, host_config={})
    nw.open()
    w = nw._writer
    w.append_quat(False, 100, [1.0, 0.0, 0.0, 0.0], 0x08, 3)
    w.append_quat(False, 200, [1.0, 0.0, 0.0, 0.0], 0x08)  # no status byte
    w.append_imu_motion(False, 0x01, 150, [0.0, 0.0, 9.81], 2)
    nw.close()

    with Session.open(path) as s:
        mta = s.imu_mta
        assert mta.quat_accuracy.tolist() == [3, 255]
        assert mta.accel.accuracy.tolist() == [2]
        assert mta.gyro is None
