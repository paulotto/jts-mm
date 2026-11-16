# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""M8c wiring: SessionSupervisor must drive the native backend end-to-end.

Uses a fake ``NativeMcuCapture`` (no hardware) so the full path is exercised:
supervisor -> NativeCapture adapter -> ClockAligner -> NativeWriter -> HDF5.
Skips when the native extension is not built.
"""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import pytest

from jtsmm import native
from jtsmm.protocol.types import RecordType

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)


class _FakeNativeCapture:
    """Stand-in for the compiled NativeMcuCapture; replays a fixed record set."""

    def __init__(self, queue_max: int = 0) -> None:
        self._recs = self._build()
        self.crc_errors = 0

    @staticmethod
    def _rec(rtype: int, t: int, decoded: dict):
        return SimpleNamespace(record_type=int(rtype), t_master_us=t, payload=b"", decoded=decoded)

    def _build(self):
        recs = []
        for i in range(200):
            recs.append(self._rec(RecordType.EMG_BLOCK, 1000 + i * 1000, {"channels": [i] * 8}))
        for i in range(80):
            recs.append(self._rec(RecordType.IMU_MTA_QUAT, 2000 + i * 2500, {"quat": (1.0, 0.0, 0.0, 0.0)}))
            recs.append(self._rec(RecordType.IMU_CRA_QUAT, 2100 + i * 2500, {"quat": (0.7, 0.0, 0.7, 0.0)}))
        for i in range(5):
            recs.append(self._rec(RecordType.TRIGGER_EVENT, 3000 + i * 16000, {"trigger_id": i + 1, "flags": (0, 0, 0, 0)}))
        recs.append(self._rec(RecordType.STATS, 9000, {"tx_drops": 0, "missed_drdy": 0}))
        return recs

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def discard_queued_records(self) -> None:
        pass

    def pump_to_writer(self, writer, max_records: int = 8192):
        out = {"emg": 0, "mta": 0, "cra": 0, "trig": 0, "last_stats": None, "triggers": []}
        n = 0
        while self._recs and n < max_records:
            rec = self._recs.pop(0)
            n += 1
            if rec.record_type == int(RecordType.STATS):
                out["last_stats"] = rec
                continue
            if rec.record_type == int(RecordType.EMG_BLOCK):
                writer.append_emg(rec.t_master_us, list(rec.decoded["channels"]))
                out["emg"] += 1
            elif rec.record_type == int(RecordType.IMU_MTA_QUAT):
                writer.append_quat(False, rec.t_master_us, list(rec.decoded["quat"]))
                out["mta"] += 1
            elif rec.record_type == int(RecordType.IMU_CRA_QUAT):
                writer.append_quat(True, rec.t_master_us, list(rec.decoded["quat"]))
                out["cra"] += 1
            elif rec.record_type == int(RecordType.TRIGGER_EVENT):
                writer.append_trigger(rec.t_master_us, int(rec.decoded["trigger_id"]))
                out["trig"] += 1
                out["triggers"].append(
                    {"trigger_id": int(rec.decoded["trigger_id"]), "t_master_us": rec.t_master_us}
                )
        return out

    def next_record(self, timeout_s: float):
        if self._recs:
            return self._recs.pop(0)
        return None


def test_supervisor_native_path_writes_hdf5(tmp_path, monkeypatch, resolved_config):
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setenv("JTSMM_NATIVE", "1")
    monkeypatch.setenv("JTSMM_SETTLE_MS", "0")
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    monkeypatch.setattr(native, "NativeMcuCapture", _FakeNativeCapture)
    monkeypatch.setenv("JTSMM_PPS_SAMPLES", "0")
    monkeypatch.setattr("jtsmm.capture.supervisor.start_mcu_streams", lambda _ser, **_k: None)
    monkeypatch.setattr("jtsmm.capture.supervisor.start_mcu_sensor_streams", lambda _ser, **_k: None)
    monkeypatch.setattr("jtsmm.capture.supervisor.start_mcu_trigger", lambda _ser: None)
    monkeypatch.setattr("jtsmm.capture.supervisor.stop_mcu_streams", lambda _ser: None)
    monkeypatch.setattr("jtsmm.capture.supervisor.stop_mcu_streams_robust", lambda _port, **_: None)
    monkeypatch.setattr(
        "jtsmm.capture.supervisor.open_hat_serial",
        lambda _port: __import__("contextlib").nullcontext(),
    )

    # Force the native branch: no cameras, storage under tmp.
    resolved_config.setdefault("host", {})["cameras"] = {}
    resolved_config["host"].setdefault("paths", {})["data_dir"] = str(tmp_path / "data")

    sup = SessionSupervisor(resolved_config, port="/dev/null")
    h5_path = sup.run(duration_s=0.5)

    assert h5_path.is_file()
    with h5py.File(h5_path, "r") as f:
        assert f["emg"]["samples"].shape[1] == 8
        assert f["emg"]["samples"].shape[0] == 200
        assert f["imu_mta"]["quat"].shape[0] == 80
        assert f["imu_cra"]["quat"].shape[0] == 80
        assert f["triggers"]["trigger_id"].shape[0] == 5
        assert f["meta"].attrs["config_hash"]
        assert "report_json" in f["sync"].attrs
