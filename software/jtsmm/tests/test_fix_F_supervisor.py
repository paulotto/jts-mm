# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""SessionSupervisor setup/teardown robustness (fix package F).

Fakes stand in for the native capture, cameras and MCU serial, so these run
without hardware. The native extension must be built (its writer class is
used for the availability check), like the other supervisor tests.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import time
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


def _rec(rtype: int, t: int, decoded: dict) -> Any:
    return SimpleNamespace(record_type=int(rtype), t_master_us=t, payload=b"", decoded=decoded)


class FakeMcu:
    """NativeMcuCapture stand-in with a live queue the test can feed."""

    def __init__(self, n_emg: int = 400) -> None:
        self.q: list[Any] = [_rec(RecordType.EMG_BLOCK, 1000 + i * 500, {"channels": [i] * 8})
                             for i in range(n_emg)]
        self.crc_errors = 0
        self.started = False
        self.stopped = False
        self.log: list[str] | None = None
        self.stop_raises: BaseException | None = None

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        if self.log is not None:
            self.log.append("native_stop")
        self.stopped = True
        if self.stop_raises is not None:
            raise self.stop_raises

    def discard_queued_records(self) -> None:
        self.q.clear()

    def next_record(self, timeout_s: float = 0.0):
        return self.q.pop(0) if self.q else None

    def pump_to_writer(self, writer, max_records: int = 8192):
        out = {"emg": 0, "mta": 0, "cra": 0, "trig": 0, "last_stats": None, "triggers": [],
               "logs": []}
        n = 0
        while self.q and n < max_records:
            rec = self.q.pop(0)
            n += 1
            if rec.record_type == int(RecordType.STATS):
                out["last_stats"] = rec
            elif rec.record_type == int(RecordType.LOG):  # as mcu_capture.cpp: handed back
                out["logs"].append({"t_master_us": rec.t_master_us, "text": rec.decoded["text"]})
            elif rec.record_type == int(RecordType.EMG_BLOCK):
                writer.append_emg(rec.t_master_us, list(rec.decoded["channels"]))
                out["emg"] += 1
        return out

    def set_viewer_tap_enabled(self, *_a, **_k) -> None:
        pass


@dataclass
class FakeCam:
    cam_id: str = "cam0"
    is_mock: bool = False
    start_raises: BaseException | None = None
    stop_raises: BaseException | None = None
    started: bool = False
    stopped: bool = False
    log: list | None = field(default=None, repr=False)

    def start(self) -> None:
        if self.start_raises is not None:
            raise self.start_raises
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        if self.log is not None:
            self.log.append(f"{self.cam_id}_stop")
        if self.stop_raises is not None:
            raise self.stop_raises

    def drain_frames(self, **_kw) -> list:
        return []


@pytest.fixture
def env(tmp_path, monkeypatch, resolved_config):
    from jtsmm.capture import supervisor as sup_mod

    state = SimpleNamespace(mcu=FakeMcu(), cams=[FakeCam()], log=[], made_mcu=0, made_cams=0)
    state.mcu.log = state.log

    def make_mcu(*_a, **_k):
        state.made_mcu += 1
        return state.mcu

    def make_cams(_cfg, **_kw):
        state.made_cams += 1
        return list(state.cams)

    for key, val in (("JTSMM_NATIVE", "1"), ("JTSMM_SETTLE_MS", "0"),
                     ("JTSMM_STREAM_RESTART", "0"), ("JTSMM_PPS_SAMPLES", "0")):
        monkeypatch.setenv(key, val)
    monkeypatch.setattr(native, "NativeMcuCapture", make_mcu)
    monkeypatch.setattr(sup_mod, "start_mcu_streams", lambda _ser, **_k: state.log.append("streams_on"))
    monkeypatch.setattr(sup_mod, "start_mcu_trigger", lambda _ser: state.log.append("trig_start"))
    monkeypatch.setattr(sup_mod, "stop_mcu_streams", lambda _ser: state.log.append("streams_off"))
    monkeypatch.setattr(
        sup_mod, "stop_mcu_streams_robust", lambda _p, **_k: state.log.append("robust_stop")
    )
    monkeypatch.setattr(sup_mod, "open_hat_serial", lambda _p: contextlib.nullcontext())
    monkeypatch.setattr(sup_mod, "cameras_from_config", make_cams)
    resolved_config["host"].setdefault("storage", {})["base_dir"] = str(tmp_path / "data")
    resolved_config["host"]["cameras"] = {
        "list": [{"id": "cam0", "device": "csi0", "trigger": {"expect_rate_hz": 60}}]
    }
    state.resolved = resolved_config
    state.sup_mod = sup_mod
    return state


def _sync(h5_path) -> dict:
    with h5py.File(h5_path, "r") as f:  # must open: the writer was closed
        return json.loads(f["sync"].attrs["report_json"])


def test_setup_exception_closes_writer_and_stops_threads(env):
    from jtsmm.capture.supervisor import SessionSupervisor

    env.cams[0].start_raises = RuntimeError("picamera2 start failed")
    with pytest.raises(RuntimeError, match="picamera2 start failed") as ei:
        SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=5.0)
    h5_path = ei.value.jtsmm_session_h5
    assert env.mcu.started and env.mcu.stopped, "native capture threads left running"
    assert env.cams[0].stopped, "camera left open"
    assert "streams_off" in env.log, "MCU streams left running"
    sync = _sync(h5_path)
    assert "capture_order" in sync


def test_teardown_step_failure_still_closes_writer(env):
    from jtsmm.capture.supervisor import SessionSupervisor

    env.mcu.stop_raises = RuntimeError("libusb wedged")
    env.cams[0].stop_raises = OSError("camera busy")
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.2)
    sync = _sync(h5_path)
    errs = "\n".join(sync.get("teardown_errors") or [])
    assert "libusb wedged" in errs and "camera busy" in errs
    # the steps after the failures still ran
    assert "streams_off" in env.log and env.cams[0].stopped
    with h5py.File(h5_path, "r") as f:
        assert f["emg"]["samples"].shape[0] > 0


def test_missing_tip_file_fails_before_any_hardware(env, tmp_path):
    from jtsmm.capture.supervisor import SessionSupervisor

    sup = SessionSupervisor(
        env.resolved,
        port="/dev/null",
        optical_track=True,
        optical_config=tmp_path / "optical.yaml",
        ct_tip_calibration=tmp_path / "missing_tip.json",
    )
    with pytest.raises(RuntimeError, match="ct_tip_calibration not found"):
        sup.run(duration_s=1.0)
    assert env.made_mcu == 0 and env.made_cams == 0


class _Plan:
    name = "fake_plan"
    needs_optical = False
    description = ""
    slug = "fake_plan"
    all_phases = [SimpleNamespace(name="p1", kind="experiment")]

    def to_dict(self) -> dict:
        return {"name": self.name}


def test_experiment_plan_without_tty_fails_before_any_hardware(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    def no_tty():
        raise RuntimeError("stdin is not a TTY")

    monkeypatch.setattr("jtsmm.io.terminal_keys.TerminalKeys", no_tty)
    sup = SessionSupervisor(env.resolved, port="/dev/null", experiment_plan=_Plan())
    with pytest.raises(RuntimeError, match="interactive TTY"):
        sup.run(duration_s=1.0)
    assert env.made_mcu == 0 and env.made_cams == 0


# --- signals / Ctrl-C ---------------------------------------------------------


class SignallingCam(FakeCam):
    """Sends ``signum`` to this process on the n-th drain (i.e. inside the loop)."""

    def __init__(self, signum: int, after: int = 3) -> None:
        super().__init__()
        self.signum = signum
        self.after = after
        self.calls = 0

    def drain_frames(self, **_kw) -> list:
        self.calls += 1
        if self.calls == self.after:
            os.kill(os.getpid(), self.signum)
        return []


@pytest.mark.parametrize(
    ("signum", "reason", "signame"),
    [(signal.SIGTERM, "signal", "SIGTERM"), (signal.SIGHUP, "signal", "SIGHUP"),
     (signal.SIGINT, "operator_stop", None)],
)
def test_signal_in_loop_is_a_graceful_stop(env, signum, reason, signame):
    from jtsmm.capture.supervisor import SessionSupervisor

    env.cams = [SignallingCam(signum)]
    prev = signal.getsignal(signum)
    t0 = time.monotonic()
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=30.0)
    assert time.monotonic() - t0 < 10.0, "signal did not end the take"
    sync = _sync(h5_path)
    assert sync["end_reason"] == reason
    assert sync.get("end_signal") == signame
    assert env.mcu.stopped and env.cams[0].stopped
    assert signal.getsignal(signum) == prev, "handler not restored after the run"


def test_sigterm_during_setup_closes_and_raises(env):
    from jtsmm.capture.supervisor import SessionStopSignal, SessionSupervisor

    class TermOnStart(FakeCam):
        def start(self) -> None:
            os.kill(os.getpid(), signal.SIGTERM)
            super().start()

    env.cams = [TermOnStart()]
    with pytest.raises(SessionStopSignal) as ei:
        SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=30.0)
    sync = _sync(ei.value.jtsmm_session_h5)
    assert sync["end_reason"] == "signal" and sync["end_signal"] == "SIGTERM"
    assert env.mcu.stopped


def test_duration_end_reason(env):
    from jtsmm.capture.supervisor import SessionSupervisor

    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.1)
    assert _sync(h5_path)["end_reason"] == "duration"


# --- stop order / fresh mcu_stats_end ----------------------------------------


def _answer_stats(env, monkeypatch, decoded: dict | None) -> None:
    """`stats` over CDC -> one REC_STATS on the vendor link (or silence)."""

    def send(_ser) -> None:
        if "streams_off" not in env.log:
            # the take's start snapshot, taken before the queued test records
            env.log.append("stats_start")
            start = {"stats_seq": 0} if decoded and "stats_seq" in decoded else {}
            env.mcu.q.insert(0, _rec(RecordType.STATS, 500, start))
            return
        env.log.append("stats")
        if decoded is not None:
            # a few tail EMG records still in flight, then the stats record
            env.mcu.q.extend(_rec(RecordType.EMG_BLOCK, 900_000 + i, {"channels": [0] * 8})
                             for i in range(5))
            env.mcu.q.append(_rec(RecordType.STATS, 999_000, dict(decoded)))

    monkeypatch.setattr(env.sup_mod, "_send_stats_request", send)


def test_stop_order_sends_stats_before_native_stop(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    # a stale periodic stats record from mid-take
    env.mcu.q.append(_rec(RecordType.STATS, 5000, {"emg_drops": 1, "stats_seq": 7}))
    _answer_stats(env, monkeypatch, {"emg_drops": 5, "stats_seq": 8})
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.2)
    log = env.log
    assert log.index("streams_off") < log.index("stats") < log.index("native_stop"), log
    assert "robust_stop" not in log, "CDC stop worked, no fallback needed"
    sync = _sync(h5_path)
    assert sync["mcu_stats_end"]["stats_seq"] == 8 and sync["mcu_stats_end"]["emg_drops"] == 5
    assert sync["mcu_stats_end_fresh"] is True
    with h5py.File(h5_path, "r") as f:  # the tail after the stop reached the file
        assert int(f["emg"]["t_master_us"][-1]) >= 900_000


def test_old_firmware_stats_without_seq_counts_as_fresh(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    _answer_stats(env, monkeypatch, {"emg_drops": 2})
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.1)
    sync = _sync(h5_path)
    assert sync["mcu_stats_end"]["emg_drops"] == 2 and sync["mcu_stats_end_fresh"] is True


def test_no_stats_reply_is_bounded_and_flagged(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    env.mcu.q.append(_rec(RecordType.STATS, 5000, {"emg_drops": 1, "stats_seq": 7}))
    _answer_stats(env, monkeypatch, None)
    t0 = time.monotonic()
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.1)
    assert time.monotonic() - t0 < 5.0
    sync = _sync(h5_path)
    assert sync["mcu_stats_end_fresh"] is False
    assert sync["mcu_stats_end"]["stats_seq"] == 7


def test_cdc_stop_failure_falls_back_to_robust_stop(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    def broken(_ser):
        raise OSError("ttyACM0 vanished")

    monkeypatch.setattr(env.sup_mod, "stop_mcu_streams", broken)
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.1)
    assert "robust_stop" in env.log
    assert any("ttyACM0 vanished" in e for e in _sync(h5_path)["teardown_errors"])


# --- start sequence, counters, timing, disk ---------------------------------


@pytest.mark.parametrize("python_writer", [False, True])
def test_records_queued_during_camera_start_are_not_lost(env, monkeypatch, python_writer):
    """Regression for the 40 ms + 0.8 s start hole seen in s01_f_30.

    picamera2 start blocks ~0.8 s while EMG/IMU pile up in the native queue.
    The first pump used to pop the whole backlog and keep only 128 records
    (~40 ms of EMG). Every record must reach the file, on both writer paths.
    """
    from jtsmm.capture.supervisor import SessionSupervisor

    if python_writer:
        monkeypatch.delenv("JTSMM_NATIVE", raising=False)
        env.resolved["host"].setdefault("capture", {})["native_prefer"] = False
    env.mcu.q = [_rec(RecordType.EMG_BLOCK, 1000 + i * 500, {"channels": [i] * 8}) for i in range(80)]

    class SlowStartCam(FakeCam):
        def start(self) -> None:  # 0.8 s of 2 kHz EMG arrives while "starting"
            env.mcu.q.extend(
                _rec(RecordType.EMG_BLOCK, 1000 + i * 500, {"channels": [i] * 8})
                for i in range(80, 1680)
            )
            super().start()

    env.cams = [SlowStartCam()]
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.5)
    with h5py.File(h5_path, "r") as f:
        t = f["emg"]["t_master_us"][:].astype("int64")
    assert len(t) == 1680
    assert set((t[1:] - t[:-1]).tolist()) == {500}
    startup = _sync(h5_path)["startup"]
    assert startup["camera_start_s"] is not None


def test_counters_and_timing_are_reported(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.delenv("JTSMM_NATIVE", raising=False)
    env.resolved["host"].setdefault("capture", {})["native_prefer"] = False
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.3)
    sync = _sync(h5_path)
    assert sync["cam_hold_overflow"] == 0 and sync["cam_hold_leftover"] == 0
    st = sync["session_timing"]
    assert st["capture_s"] >= 0.25 and st["wall_s"] >= st["capture_s"]
    assert st["data_span_s"] is not None and st["setup_s"] is not None
    assert sync["disk_preflight"]["free_gb"] > 0
    with h5py.File(h5_path, "r") as f:
        meta = f["meta"].attrs
        assert float(meta["capture_duration_s"]) == pytest.approx(st["data_span_s"])
        assert "includes setup" in str(meta["session_duration_kind"])


def test_disk_full_fails_before_hardware(env, monkeypatch):
    import shutil

    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(free=100_000_000))
    with pytest.raises(RuntimeError, match="not enough free disk space"):
        SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=60.0)
    assert env.made_mcu == 0 and env.made_cams == 0


def test_disk_space_preflight_warns_when_tight(monkeypatch, tmp_path, capsys):
    import shutil

    from jtsmm.capture.supervisor import disk_space_preflight, estimate_session_bytes_per_s

    rate = estimate_session_bytes_per_s(
        {"emg": {"sample_rate_sps": 2000}, "trigger": {"rate_hz": 30}}, 2,
        preview_jpegs=True, optical=True,
    )
    assert 0.5e6 < rate < 1e6  # previews dominate (~500 kB/s)
    monkeypatch.setattr(shutil, "disk_usage", lambda _p: SimpleNamespace(free=1_500_000_000))
    info = disk_space_preflight(tmp_path, 10e6)
    assert info["warning"] is True and "low disk space" in capsys.readouterr().err


# --- CT tip trust gate --------------------------------------------------------


class _FakeOptical:
    def __init__(self, *_a, **_k) -> None:
        self.stats: dict = {}
        self.tip = None

    def session_meta(self, *_a) -> dict:
        return {}

    def set_ct_tip_body_mm(self, tip) -> None:
        self.tip = tip

    def wants_cam(self, _cid) -> bool:
        return False

    def flush(self, **_k) -> None:
        pass

    def drain_to_writer(self, _w) -> None:
        pass

    def shutdown(self, _w) -> None:
        pass


@pytest.mark.parametrize(("rms", "shared_written"), [(1.5, True), (9.0, False)])
def test_untrusted_ct_tip_does_not_overwrite_shared_file(env, monkeypatch, tmp_path, rms, shared_written):
    import sys
    import types

    import numpy as np

    from jtsmm.capture.supervisor import SessionSupervisor

    written: list = []
    fake = types.ModuleType("jtsmm.capture.ct_pivot")
    fake.run_ct_pivot_calibration = lambda *a, **k: SimpleNamespace(
        tip_body_mm=np.array([1.0, 2.0, 3.0]), rms_mm=rms
    )
    fake.write_ct_tip_calibration = lambda path, res: written.append(path) or path
    fake.ct_tip_trustworthy = lambda res: res.rms_mm <= 5.0
    monkeypatch.setitem(sys.modules, "jtsmm.capture.ct_pivot", fake)
    monkeypatch.setattr(env.sup_mod, "OpticalTrackBridge", _FakeOptical)
    monkeypatch.setattr("jtsmm.io.terminal_keys.TerminalKeys", lambda: SimpleNamespace())
    shared = tmp_path / "shared_tip.json"
    h5_path = SessionSupervisor(
        env.resolved, port="/dev/null", ct_calibration=True, ct_calibration_only=True,
        optical_config=tmp_path / "optical.yaml", ct_tip_calibration=shared,
    ).run(duration_s=None)
    assert (written == [shared]) is shared_written
    with h5py.File(h5_path, "r") as f:
        assert f["meta"].attrs["ct_tip_trusted"] == ("true" if shared_written else "false")
    assert _sync(h5_path)["end_reason"] == "ct_calibration_only"


# --- dead-link watchdog -------------------------------------------------------


def test_watchdog_stops_take_when_emg_stops(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setattr(env.sup_mod, "_WATCHDOG_SILENCE_S", 0.3)
    t0 = time.monotonic()
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=30.0)
    assert time.monotonic() - t0 < 10.0, "watchdog never fired"
    sync = _sync(h5_path)
    assert sync["end_reason"] == "watchdog"
    assert sync["watchdog"]["reason"] == "no EMG records"
    assert sync["watchdog"]["silent_s"] >= 0.3
    with h5py.File(h5_path, "r") as f:  # the data before the silence is kept
        assert f["emg"]["samples"].shape[0] == 400


def test_watchdog_stops_take_on_device_lost(env, monkeypatch):
    from jtsmm.capture.native_backend import NativeCapture
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setattr(
        NativeCapture, "link_health", lambda self: {"device_lost": True}, raising=False
    )
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=30.0)
    sync = _sync(h5_path)
    assert sync["end_reason"] == "watchdog" and sync["watchdog"]["reason"] == "device_lost"


def test_watchdog_quiet_while_emg_flows(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setattr(env.sup_mod, "_WATCHDOG_SILENCE_S", 0.3)
    # an endless stream: the final drain / stats wait must be time-bounded
    monkeypatch.setattr(env.sup_mod, "_FINAL_DRAIN_MAX_S", 0.2)
    monkeypatch.setattr(env.sup_mod, "_FINAL_STATS_TIMEOUT_S", 0.2)

    class Streaming(FakeMcu):
        def pump_to_writer(self, writer, max_records: int = 8192):
            self.q.append(_rec(RecordType.EMG_BLOCK, 10_000_000 + len(self.q), {"channels": [0] * 8}))
            return super().pump_to_writer(writer, max_records=1)

    env.mcu = Streaming()
    env.mcu.log = env.log
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: env.mcu)
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.8)
    assert _sync(h5_path)["end_reason"] == "duration"


def test_cameras_stop_before_the_trigger(env):
    """A triggered IMX296 can wait in stop() for a frame; stop it while pulses still come."""
    from jtsmm.capture.supervisor import SessionSupervisor

    env.cams[0].log = env.log
    SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.2)
    assert env.log.index("cam0_stop") < env.log.index("streams_off")


def test_hanging_camera_stop_does_not_keep_the_file_open(env, monkeypatch):
    """Bench 2026-09-25: the duration ran out and the recorder never closed the file."""
    import threading

    from jtsmm.capture.supervisor import SessionSupervisor

    release = threading.Event()
    monkeypatch.setattr(env.cams[0], "stop", lambda: release.wait(30))
    monkeypatch.setattr(env.sup_mod, "_CAMERA_STOP_TIMEOUT_S", 0.2)
    try:
        h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.2)
    finally:
        release.set()
    sync = _sync(h5_path)  # opens only if the writer closed the file
    assert sync["end_reason"] == "duration"
    errs = "\n".join(sync.get("teardown_errors") or [])
    assert "camera cam0 stop did not return within 0.2 s" in errs
