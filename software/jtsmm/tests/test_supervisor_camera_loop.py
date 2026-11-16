# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Supervisor camera/MCU interleaving — catches the cam0 starvation regression."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import h5py
import pytest

from jtsmm import native
from jtsmm.protocol.types import RecordType

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)


@dataclass
class _FakeCam:
    cam_id: str
    is_mock: bool = False
    n_frames: int = 40
    drain_calls: list[int] = field(default_factory=list)
    started: bool = False
    stopped: bool = False
    _next_idx: int = 1

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def drain_frames(
        self,
        *,
        max_frames: int = 8,
        with_rgb: bool = False,
        rgb_limit: int | None = None,
    ) -> list[Any]:
        self.drain_calls.append(max_frames)
        out: list[Any] = []
        for _ in range(max_frames):
            if self._next_idx > self.n_frames:
                break
            out.append(_frame(self._next_idx))
            self._next_idx += 1
        return out

    def poll_frame(self, *, with_rgb: bool = False) -> Any | None:
        frames = self.drain_frames(max_frames=1, with_rgb=with_rgb)
        return frames[0] if frames else None


class _RecordingNativeMcuCapture:
    """Stand-in for compiled NativeMcuCapture; records pump batch sizes."""

    def __init__(self, queue_max: int = 0) -> None:
        self.batch_sizes: list[int] = []
        self._recs = self._build()
        self.crc_errors = 0

    @staticmethod
    def _rec(rtype: int, t: int, decoded: dict):
        return SimpleNamespace(record_type=int(rtype), t_master_us=t, payload=b"", decoded=decoded)

    def _build(self):
        recs = []
        # Interleave triggers early so camera sequence-consume can lock, then
        # flood EMG so a single 8192 drain would starve cameras under the old
        # loop. With max_records=128 we need many interleaved pumps.
        for i in range(40):
            recs.append(
                self._rec(
                    RecordType.TRIGGER_EVENT,
                    2000 + i * 16_666,
                    {"trigger_id": i + 1, "flags": (0, 0, 0, 0)},
                )
            )
        for i in range(2000):
            recs.append(self._rec(RecordType.EMG_BLOCK, 1000 + i * 500, {"channels": [i] * 8}))
        for i in range(40, 120):
            recs.append(
                self._rec(
                    RecordType.TRIGGER_EVENT,
                    2000 + i * 16_666,
                    {"trigger_id": i + 1, "flags": (0, 0, 0, 0)},
                )
            )
        return recs

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def discard_queued_records(self) -> None:
        pass

    def pump_to_writer(self, writer, max_records: int = 8192):
        self.batch_sizes.append(int(max_records))
        out = {"emg": 0, "mta": 0, "cra": 0, "trig": 0, "last_stats": None, "triggers": []}
        n = 0
        while self._recs and n < max_records:
            rec = self._recs.pop(0)
            n += 1
            if rec.record_type == int(RecordType.EMG_BLOCK):
                writer.append_emg(rec.t_master_us, list(rec.decoded["channels"]))
                out["emg"] += 1
            elif rec.record_type == int(RecordType.TRIGGER_EVENT):
                writer.append_trigger(rec.t_master_us, int(rec.decoded["trigger_id"]))
                out["trig"] += 1
                out["triggers"].append(
                    {"trigger_id": int(rec.decoded["trigger_id"]), "t_master_us": rec.t_master_us}
                )
        return out

    def set_viewer_tap_enabled(self, enabled: bool, bucket_samples: int = 4) -> None:
        pass

    def get_viewer_frame(self) -> Any:
        return None

    def next_record(self, timeout_s: float = 0.0):
        if self._recs:
            return self._recs.pop(0)
        return None


def _frame(idx: int):
    return SimpleNamespace(
        cam_id="cam0",
        frame_idx=idx,
        sensor_timestamp_ns=1_700_000_000_000_000_000 + idx * 16_666_000,
        trigger_id=None,
        exposure_us=4000,
        gain=1.0,
        mock=False,
    )


def test_supervisor_interleaves_small_mcu_batches_with_camera_drains(
    tmp_path, monkeypatch, resolved_config
):
    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.capture.supervisor import SessionSupervisor

    fake_mcu = _RecordingNativeMcuCapture()
    real_cam = _FakeCam("cam0", is_mock=False, n_frames=40)
    mock_cam = _FakeCam("cam1", is_mock=True, n_frames=40)

    monkeypatch.setenv("JTSMM_NATIVE", "1")
    monkeypatch.setenv("JTSMM_SETTLE_MS", "0")
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    monkeypatch.setenv("JTSMM_PPS_SAMPLES", "0")
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: fake_mcu)
    monkeypatch.setattr(sup_mod, "start_mcu_streams", lambda _ser, **_k: None)
    monkeypatch.setattr(sup_mod, "start_mcu_trigger", lambda _ser: None)
    monkeypatch.setattr(sup_mod, "stop_mcu_streams", lambda _ser: None)
    monkeypatch.setattr(sup_mod, "stop_mcu_streams_robust", lambda _port, **_: None)
    monkeypatch.setattr(
        sup_mod,
        "open_hat_serial",
        lambda _port: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setattr(
        sup_mod,
        "cameras_from_config",
        lambda _cfg, **_kw: [real_cam, mock_cam],
    )

    resolved_config["host"].setdefault("paths", {})["data_dir"] = str(tmp_path / "data")
    resolved_config["host"]["cameras"] = {
        "list": [
            {
                "id": "cam0",
                "device": "csi0",
                "resolution": [64, 64],
                "trigger": {"expect_rate_hz": 60},
            },
            {
                "id": "cam1",
                "device": "csi1",
                "resolution": [64, 64],
                "trigger": {"expect_rate_hz": 60},
            },
        ]
    }

    sup = SessionSupervisor(resolved_config, port="/dev/null")
    h5_path = sup.run(duration_s=0.4, mock_only=False)

    assert fake_mcu.batch_sizes, "MCU pump was never called"
    assert all(b <= 128 for b in fake_mcu.batch_sizes), fake_mcu.batch_sizes
    assert len(fake_mcu.batch_sizes) > 5, "expected many interleaved small pumps"
    assert real_cam.started and real_cam.stopped
    assert real_cam.drain_calls, "real camera was never drained"
    assert not mock_cam.started, "mock cam1 must be skipped on real recordings"
    assert mock_cam.drain_calls == []

    assert h5_path.is_file()
    with h5py.File(h5_path, "r") as f:
        assert "cam0" in f["cameras"]
        assert "cam1" not in f["cameras"]
        assert f["cameras"]["cam0"]["trigger_id"].shape[0] > 0


def test_capture_cam_drain_frames_stops_on_empty(monkeypatch):
    import jtsmm.capture.capture_cam as cam_mod
    from jtsmm.capture.capture_cam import CameraCapture

    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)

    cam = CameraCapture(
        cam_id="cam0",
        device="csi0",
        resolution=(64, 64),
        exposure_us=1000,
        gain=1.0,
        expect_rate_hz=60.0,
    )
    assert cam.is_mock is True
    frames = cam.drain_frames(max_frames=8)
    assert len(frames) == 1
    assert frames[0].mock is True


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="needs OpenCV")
def test_drain_frames_attaches_rgb_only_within_limit(monkeypatch):
    """RGB has to be taken during the drain, not by a follow-up poll.

    The drain runs until picamera2's completed-request queue is empty, so a
    separate ``poll_frame(with_rgb=True)`` afterwards would practically always
    return None and the optical tracker would never see pixels.
    """
    import numpy as np

    import jtsmm.capture.capture_cam as cam_mod
    from jtsmm.capture.capture_cam import CameraCapture

    monkeypatch.setattr(cam_mod, "_try_open_picamera2", lambda _cam: False)
    cam = CameraCapture(
        cam_id="cam0",
        device="csi0",
        resolution=(8, 8),
        exposure_us=1000,
        gain=1.0,
        expect_rate_hz=60.0,
    )
    cam._mock = False
    cam._picam = object()

    arrays_made: list[str] = []

    class _Req:
        def get_metadata(self) -> dict[str, Any]:
            return {"SensorTimestamp": 1, "ExposureTime": 1000, "AnalogueGain": 1.0}

        def make_array(self, name: str) -> Any:
            arrays_made.append(name)
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

    pending = [_Req(), _Req(), _Req()]
    monkeypatch.setattr(
        cam,
        "_acquire_request",
        lambda *, blocking: pending.pop(0) if pending else None,
    )

    frames = cam.drain_frames(max_frames=8, with_rgb=True, rgb_limit=1)
    assert len(frames) == 3
    assert frames[0].rgb is not None
    assert [f.rgb for f in frames[1:]] == [None, None]
    assert arrays_made == ["main"]


def test_camera_trigger_is_armed_after_cameras_start(tmp_path, monkeypatch, resolved_config):
    """trig_start must come after cam.start().

    A TRIGGER_EVENT that fires before any frame exists cannot be consumed.
    Arming after cam.start() keeps that pre-frame buffer from forming.
    """
    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.capture.supervisor import SessionSupervisor

    fake_mcu = _RecordingNativeMcuCapture()
    cam = _FakeCam("cam0", is_mock=False, n_frames=40)
    order: list[str] = []

    monkeypatch.setenv("JTSMM_NATIVE", "1")
    monkeypatch.setenv("JTSMM_SETTLE_MS", "0")
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    monkeypatch.setenv("JTSMM_PPS_SAMPLES", "0")
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: fake_mcu)
    monkeypatch.setattr(
        sup_mod,
        "start_mcu_streams",
        lambda _ser, **kw: order.append(f"streams(trigger={kw.get('trigger', True)})"),
    )
    monkeypatch.setattr(
        sup_mod, "start_mcu_trigger", lambda _ser: order.append("trig_start")
    )
    monkeypatch.setattr(sup_mod, "stop_mcu_streams", lambda _ser: None)
    monkeypatch.setattr(sup_mod, "stop_mcu_streams_robust", lambda _port, **_: None)
    monkeypatch.setattr(
        sup_mod, "open_hat_serial", lambda _port: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(sup_mod, "cameras_from_config", lambda _cfg, **_kw: [cam])

    real_start = cam.start

    def _tracked_start() -> None:
        order.append("cam.start")
        real_start()

    monkeypatch.setattr(cam, "start", _tracked_start)

    resolved_config["host"].setdefault("paths", {})["data_dir"] = str(tmp_path / "data")
    resolved_config["host"]["cameras"] = {
        "list": [{"id": "cam0", "device": "csi0", "trigger": {"expect_rate_hz": 60}}]
    }

    SessionSupervisor(resolved_config, port="/dev/null").run(duration_s=0.3)

    assert "streams(trigger=False)" in order, order
    assert order.index("cam.start") < order.index("trig_start"), order


class _IdleKeys:
    def __enter__(self) -> _IdleKeys:
        return self

    def __exit__(self, *_args) -> None:
        return None

    def poll(self) -> int | None:
        return None


def test_mvc_wait_still_writes_camera_trigger_rows(tmp_path, monkeypatch, resolved_config):
    """MVC must keep sequence-consume alive; draining without ingest created optical lag."""
    experiment = pytest.importorskip("optical.tracking.experiment")
    ExperimentPlan, Phase = experiment.ExperimentPlan, experiment.Phase

    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.capture.supervisor import SessionSupervisor

    fake_mcu = _RecordingNativeMcuCapture()
    cam = _FakeCam("cam0", is_mock=False, n_frames=40)

    monkeypatch.setenv("JTSMM_NATIVE", "1")
    monkeypatch.setenv("JTSMM_SETTLE_MS", "0")
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    monkeypatch.setenv("JTSMM_PPS_SAMPLES", "0")
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: fake_mcu)
    monkeypatch.setattr(sup_mod, "start_mcu_streams", lambda _ser, **_k: None)
    monkeypatch.setattr(sup_mod, "start_mcu_trigger", lambda _ser: None)
    monkeypatch.setattr(sup_mod, "stop_mcu_streams", lambda _ser: None)
    monkeypatch.setattr(sup_mod, "stop_mcu_streams_robust", lambda _port, **_: None)
    monkeypatch.setattr(
        sup_mod, "open_hat_serial", lambda _port: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(sup_mod, "cameras_from_config", lambda _cfg, **_kw: [cam])
    monkeypatch.setattr("jtsmm.io.terminal_keys.TerminalKeys", lambda: _IdleKeys())

    resolved_config["host"].setdefault("paths", {})["data_dir"] = str(tmp_path / "data")
    resolved_config["host"]["cameras"] = {
        "list": [{"id": "cam0", "device": "csi0", "trigger": {"expect_rate_hz": 60}}]
    }

    plan = ExperimentPlan(
        name="mvc_then_motion",
        phases=[Phase("mvc_opening"), Phase("open_close")],
    )
    h5_path = SessionSupervisor(
        resolved_config, port="/dev/null", experiment_plan=plan
    ).run(duration_s=0.3)

    with h5py.File(h5_path, "r") as f:
        ids = list(f["cameras"]["cam0"]["trigger_id"][:])
    assert ids, "MVC wait used to drop every camera row and leave a stale trigger cursor"
    assert ids == list(range(int(ids[0]), int(ids[0]) + len(ids)))


class _LoggingCam(_FakeCam):
    """_FakeCam that records when start() happens relative to serial commands."""

    def __init__(self, cam_id: str, log: list, **kw) -> None:
        super().__init__(cam_id, **kw)
        self._log = log

    def start(self) -> None:
        super().start()
        self._log.append(("cam_start", self.cam_id))


class _LoggingSerial:
    """Context-manager serial stand-in that logs every command line."""

    def __init__(self, log: list) -> None:
        self._log = log

    def write(self, data: bytes) -> None:
        self._log.append(("cmd", data.decode("ascii").strip()))

    def flush(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


def _run_order_case(tmp_path, monkeypatch, resolved_config, *, trigger_after: bool) -> tuple[list, dict]:
    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.capture.supervisor import SessionSupervisor

    log: list = []
    fake_mcu = _RecordingNativeMcuCapture()
    cam = _LoggingCam("cam0", log, is_mock=False, n_frames=10)

    monkeypatch.setenv("JTSMM_NATIVE", "1")
    monkeypatch.setenv("JTSMM_SETTLE_MS", "0")
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    monkeypatch.setenv("JTSMM_PPS_SAMPLES", "0")
    monkeypatch.setenv("JTSMM_TRIGGER_AFTER_CAMERAS", "1" if trigger_after else "0")
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: fake_mcu)
    # Real start_mcu_* functions run against the logging serial (they only sleep).
    monkeypatch.setattr(sup_mod, "open_hat_serial", lambda _port: _LoggingSerial(log))
    monkeypatch.setattr(sup_mod, "stop_mcu_streams_robust", lambda _port, **_: None)
    monkeypatch.setattr(sup_mod, "cameras_from_config", lambda _cfg, **_kw: [cam])
    monkeypatch.setattr(sup_mod.time, "sleep", lambda _s: None)  # keep the test fast

    resolved_config["host"].setdefault("paths", {})["data_dir"] = str(tmp_path / "data")
    resolved_config["host"]["cameras"] = {
        "list": [{"id": "cam0", "device": "csi0", "resolution": [64, 64], "trigger": {"expect_rate_hz": 60}}]
    }
    sup = SessionSupervisor(resolved_config, port="/dev/null")
    h5_path = sup.run(duration_s=0.05, mock_only=False)
    with h5py.File(h5_path, "r") as f:
        import json

        sync = json.loads(f["sync"].attrs["report_json"])
    return log, sync


def test_trigger_starts_after_cameras_by_default(tmp_path, monkeypatch, resolved_config):
    """S2: the first pulse must not happen before the cameras are streaming."""
    log, sync = _run_order_case(tmp_path, monkeypatch, resolved_config, trigger_after=True)
    cmds = [(i, c) for i, (k, c) in enumerate(log) if k == "cmd"]
    cam_i = next(i for i, (k, _) in enumerate(log) if k == "cam_start")
    ads_i = next(i for i, c in cmds if c == "ads_start")
    trig_i = next(i for i, c in cmds if c == "trig_start")
    assert ads_i < cam_i < trig_i, log
    assert sync["capture_order"] == "sensors,cameras,trigger"


def test_trigger_order_legacy_when_disabled(tmp_path, monkeypatch, resolved_config):
    """host.capture.trigger_after_cameras=false reproduces the old order for comparison."""
    log, sync = _run_order_case(tmp_path, monkeypatch, resolved_config, trigger_after=False)
    cam_i = next(i for i, (k, _) in enumerate(log) if k == "cam_start")
    trig_i = next(i for i, (k, c) in enumerate(log) if k == "cmd" and c == "trig_start")
    assert trig_i < cam_i, log
    assert sync["capture_order"] == "sensors+trigger,cameras"


def test_pps_handshake_fits_host_clock(monkeypatch):
    """`pps <id>` round trips yield a Pi->master fit; a mute firmware yields none, quietly."""
    import time

    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.sync.clock_align import ClockAligner

    class FakeCap:
        def __init__(self) -> None:
            self.q: list = []

        def discard_queued(self) -> None:
            self.q.clear()

        def drain_records(self) -> list:
            out, self.q = self.q, []
            return out

    offset_us = 5_000_000  # MCU clock = host monotonic + 5 s

    class ReplyingSerial:
        def __init__(self, cap: FakeCap) -> None:
            self.cap = cap

        def write(self, data: bytes) -> None:
            pid = int(data.decode("ascii").split()[1])
            t_master = time.monotonic_ns() // 1000 + offset_us
            self.cap.q.append(
                SimpleNamespace(
                    record_type=int(RecordType.PPS_REPLY), t_master_us=t_master,
                    decoded={"pps_id": pid},
                )
            )

        def flush(self) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    cap = FakeCap()
    monkeypatch.setattr(sup_mod, "open_hat_serial", lambda _p: ReplyingSerial(cap))
    aligner = ClockAligner()
    res = sup_mod._pps_handshake("/dev/null", cap, aligner, 6, label="start")
    assert res["accepted"] == 6 and res["timeouts"] == 0
    assert aligner._pi_fit.n_samples == 6
    now_us = time.monotonic_ns() / 1000.0
    assert abs(aligner._pi_fit.map(now_us) - (now_us + offset_us)) < 3000.0  # < 3 ms

    class MuteSerial(ReplyingSerial):
        def write(self, data: bytes) -> None:  # old firmware: "unknown: pps", no record
            pass

    monkeypatch.setattr(sup_mod, "_PPS_REPLY_TIMEOUT_S", 0.02)
    monkeypatch.setattr(sup_mod, "open_hat_serial", lambda _p: MuteSerial(cap))
    res2 = sup_mod._pps_handshake("/dev/null", cap, ClockAligner(), 3, label="start")
    assert res2["accepted"] == 0 and res2["timeouts"] == 3 and "error" not in res2


def test_python_pump_path_records_mcu_stats_end(tmp_path, monkeypatch, resolved_config):
    """Regression: `mcu_stats_end` was absent from every Python-writer session.

    `use_native_writer` is false for optical / CT / experiment-plan runs, which
    take `_pump_native_python`. That path routed records through the aligner
    (which does not carry REC_STATS) and only handled STATS inside the viewer
    branch, so `last_stats` stayed None and the sync report lost every MCU
    counter — silently disabling the MCU-side integrity checks, because a
    missing field reads as zero.
    """
    import json

    from jtsmm.capture import supervisor as sup_mod
    from jtsmm.capture.supervisor import SessionSupervisor

    class _Cap(_RecordingNativeMcuCapture):
        def _build(self):
            recs = []
            for i in range(20):
                recs.append(
                    self._rec(
                        RecordType.TRIGGER_EVENT,
                        2000 + i * 16_666,
                        {"trigger_id": i + 1, "flags": (1, 0, 0, 0)},
                    )
                )
            for i in range(200):
                recs.append(
                    self._rec(RecordType.EMG_BLOCK, 1000 + i * 500, {"channels": [i] * 8})
                )
            recs.append(
                self._rec(
                    RecordType.STATS,
                    9000,
                    {"frame_misalign": 3, "emg_drops": 7, "tx_drops": 11, "missed_drdy": 2},
                )
            )
            return recs

    fake = _Cap()
    cam = _FakeCam("cam0", is_mock=False, n_frames=10)
    # JTSMM_NATIVE must stay unset: it overrides host.capture.native_prefer, and
    # this test needs the extension available (so `_require_native` passes) while
    # the *writer* is the Python one.
    monkeypatch.delenv("JTSMM_NATIVE", raising=False)
    monkeypatch.setenv("JTSMM_SETTLE_MS", "0")
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    monkeypatch.setenv("JTSMM_PPS_SAMPLES", "0")
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: fake)
    for name in ("start_mcu_streams", "start_mcu_sensor_streams", "start_mcu_trigger",
                 "stop_mcu_streams"):
        monkeypatch.setattr(sup_mod, name, lambda *_a, **_kw: None)
    monkeypatch.setattr(sup_mod, "stop_mcu_streams_robust", lambda _port, **_: None)
    monkeypatch.setattr(
        sup_mod, "open_hat_serial", lambda _port: __import__("contextlib").nullcontext()
    )
    monkeypatch.setattr(sup_mod, "cameras_from_config", lambda _cfg, **_kw: [cam])

    # native_prefer=False selects the Python writer -> `_pump_native_python`,
    # while the native extension itself stays available (so no _require_native).
    resolved_config["host"].setdefault("capture", {})["native_prefer"] = False
    resolved_config["host"].setdefault("paths", {})["data_dir"] = str(tmp_path / "data")
    resolved_config["host"]["cameras"] = {
        "list": [{"id": "cam0", "device": "csi0", "resolution": [64, 64],
                  "trigger": {"expect_rate_hz": 60}}]
    }

    sup = SessionSupervisor(resolved_config, port="/dev/null")
    h5_path = sup.run(duration_s=0.4, mock_only=False)

    with h5py.File(h5_path, "r") as f:
        sync = json.loads(f["sync"].attrs["report_json"])
    stats = sync.get("mcu_stats_end")
    assert stats is not None, "mcu_stats_end missing from the sync report"
    assert stats["frame_misalign"] == 3
    assert stats["emg_drops"] == 7
    assert stats["tx_drops"] == 11
