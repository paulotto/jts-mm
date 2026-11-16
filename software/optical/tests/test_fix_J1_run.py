# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Live capture loop with fake cameras / HAT (run.py imports multicam -> needs cv2)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("flask")

import optical.tracking.run as run  # noqa: E402
from optical.calibration.multicam import CameraRig, CamSpec  # noqa: E402
from optical.sync.frame_meta import CamFrameMeta  # noqa: E402
from optical.tracking.experiment import free_recording_plan  # noqa: E402
from optical.tracking.io import FrameRecord, read_tracking_session  # noqa: E402

PERIOD_NS = 33_333_333
SPACE, Q = ord(" "), ord("q")


class _Keys:
    """Scripted key presses: {poll_index: key}."""

    def __init__(self, script):
        self.script = dict(script)
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def poll(self):
        k = self.script.get(self.n)
        self.n += 1
        return k


class _Rig(CameraRig):
    """Two cameras with independent frame queues (real capture_all_synced logic).

    ``lag``: cam1 has that many stale frames queued, so capture order hands
    back cam1 frame k-lag next to cam0 frame k. ``fail_at`` / ``interrupt_at``
    raise on that capture.
    """

    def __init__(self, log, *, lag=0, fail_at=None, interrupt_at=None):
        self.log, self.fail_at, self.interrupt_at = log, fail_at, interrupt_at
        self.next_idx = {"cam0": 0, "cam1": -lag}
        self.k = 0
        self.closed = False
        log.append("rig_open")

    def capture_all(self):
        return self.capture_all_with_meta()[0]

    def _capture_one(self, cam):
        idx = self.next_idx[cam]
        self.next_idx[cam] += 1
        ts = 5_000_000_000 + idx * PERIOD_NS + (30_000 if cam == "cam1" else 0)
        frame = np.full((8, 8), idx % 255, np.uint8)
        return frame, CamFrameMeta(cam_id=cam, sensor_timestamp_ns=ts, t_host_monotonic_ns=ts)

    def capture_all_with_meta(self):
        k = self.k
        self.k += 1
        if self.fail_at is not None and k == self.fail_at:
            raise RuntimeError("camera went away")
        if self.interrupt_at is not None and k == self.interrupt_at:
            raise KeyboardInterrupt
        frames, meta = {}, {}
        for cam in ("cam0", "cam1"):
            frames[cam], meta[cam] = self._capture_one(cam)
        return frames, meta

    def close(self):
        self.closed = True
        self.log.append("rig_close")


class _HatSync:
    def __init__(self, log):
        self.log = log

    def __call__(self, port=None):
        return self

    def start(self):
        self.log.append("hat_start")
        return "/dev/fake"

    def reset_trigger_backlog(self):
        self.log.append("reset_backlog")
        return 3

    def align_frame_bundle(self, meta, *, frame_idx, cam_order):
        t_cam = {c: m.sensor_timestamp_ns for c, m in meta.items()}
        return t_cam, int(meta["cam0"].sensor_timestamp_ns // 1000)

    def sync_report(self):
        return {"drops": {}, "slip_count": {}, "bundle_trigger_id_mismatch": 0}

    def stop(self):
        self.log.append("hat_stop")


class _Tracker:
    cam_ids = ["cam0", "cam1"]
    rig_id = "rig"

    def __init__(self):
        self.last_detections = {}
        self.frames_seen = []

    def process_frame(self, frames, *, frame_idx, t_cam_ns=None, t_master_us=None):
        self.frames_seen.append({c: int(f[0, 0]) for c, f in frames.items()})
        rec = FrameRecord(frame_idx=frame_idx, t_cam_ns=dict(t_cam_ns), t_master_us=t_master_us)
        rec.set_pose("CT", np.eye(4))
        rec.quality = {b: {"valid": True, "n_tags": 1, "n_cameras": 2} for b in ("HP", "MP", "CT")}
        return rec

    def detect_all(self, frames):
        return {}


@pytest.fixture
def harness(tmp_path, monkeypatch):
    log = []
    hs = _HatSync(log)
    monkeypatch.setattr(run, "HatSyncBridge", hs)
    monkeypatch.setattr(run, "arm_hat_trigger", lambda: log.append("arm"))
    monkeypatch.setattr(run, "active_trigger_rate_hz", lambda: 30)
    monkeypatch.setattr(run, "overlay_frames_bgr", lambda raw, det, as_bgr: raw)
    monkeypatch.setattr(run, "_stack_preview", lambda frames, order: None)

    def go(*, keys, rig_kw=None, ct_tip=None, preview_period=None):
        if preview_period is not None:
            monkeypatch.setattr(run, "_PREVIEW_PERIOD_S", preview_period)
        rig_holder = {}

        def open_rig(cams, backend):
            rig_holder["rig"] = _Rig(log, **(rig_kw or {}))
            return rig_holder["rig"]

        monkeypatch.setattr(run, "_open_rig", open_rig)
        monkeypatch.setattr(run, "_TerminalKeys", lambda: _Keys(keys))
        plan = free_recording_plan(name="t", condition="c")
        tracker = _Tracker()
        out = tmp_path / "s.h5"
        kwargs = dict(
            tracker=tracker,
            tcfg=SimpleNamespace(pose_method="triangulation"),
            plan=plan,
            phases=list(plan.phases),
            cams=[CamSpec("cam0", 0), CamSpec("cam1", 1)],
            preview=SimpleNamespace(update=lambda img: None),
            camera_backend="picamera2",
            output_path=out,
            jts_config_path=None,
            record_frames_dir=None,
            write_jts_config_file=False,
            ct_tip=ct_tip,
        )
        return run, kwargs, tracker, log, rig_holder, out

    return go


def test_trigger_armed_before_cameras_and_meta_persisted(harness):
    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE, 10: SPACE})
    assert run_mod._run_live_triggered(**kw) == out
    assert log[:4] == ["hat_start", "reset_backlog", "arm", "rig_open"]
    assert log[-2:] == ["rig_close", "hat_stop"]
    d = read_tracking_session(out)
    assert d["meta"]["n_frames"] == 10  # captures 1..10 (SPACE polled after capture 0)
    assert d["meta"]["trigger_rate_hz"] == 30.0
    assert json.loads(d["meta"]["sync_report"])["bundle_trigger_id_mismatch"] == 0
    assert json.loads(d["meta"]["pairing"])["resynced"] == 0


def test_lagging_camera_is_paired_by_timestamp(harness):
    run_mod, kw, tracker, log, rig, out = harness(keys={2: SPACE, 12: SPACE}, rig_kw={"lag": 1})
    run_mod._run_live_triggered(**kw)
    assert tracker.frames_seen and all(f["cam0"] == f["cam1"] for f in tracker.frames_seen)
    d = read_tracking_session(out)
    t0, t1 = d["timestamps"]["t_cam_ns"]["cam0"], d["timestamps"]["t_cam_ns"]["cam1"]
    assert np.all(np.abs(t1 - t0) < PERIOD_NS / 2)
    assert json.loads(d["meta"]["pairing"])["resynced"] >= 1


def test_exception_saves_captured_frames_then_reraises(harness):
    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE}, rig_kw={"fail_at": 7})
    with pytest.raises(RuntimeError, match="camera went away"):
        run_mod._run_live_triggered(**kw)
    d = read_tracking_session(out)
    assert d["meta"]["n_frames"] == 6
    assert "aborted: RuntimeError" in d["meta"]["notes"]
    assert "camera went away" in d["meta"]["capture_error"]
    assert rig["rig"].closed and log[-1] == "hat_stop"


def test_ctrl_c_saves_and_returns(harness):
    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE}, rig_kw={"interrupt_at": 5})
    assert run_mod._run_live_triggered(**kw) == out
    d = read_tracking_session(out)
    assert d["meta"]["n_frames"] == 4 and d["meta"]["notes"].endswith("[aborted]")


def test_quit_before_recording_writes_empty_session(harness):
    run_mod, kw, tracker, log, rig, out = harness(keys={0: Q})
    run_mod._run_live_triggered(**kw)
    assert read_tracking_session(out)["meta"]["n_frames"] == 0


def test_ct_tip_offset_applied(harness):
    tip = np.array([0.0, -230.0, 5.0])
    run_mod, kw, tracker, log, rig, out = harness(
        keys={0: SPACE, 4: SPACE}, ct_tip=(tip, {"ct_tip_source": "x.json"})
    )
    run_mod._run_live_triggered(**kw)
    d = read_tracking_session(out)
    assert np.allclose(d["bodies"]["CT"]["position_mm"], tip)
    assert d["meta"]["ct_tip_source"] == "x.json"


def test_preview_built_only_at_publish_rate(harness, monkeypatch):
    calls = []
    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE, 30: SPACE}, preview_period=3600.0)
    monkeypatch.setattr(run_mod, "overlay_frames_bgr", lambda raw, det, as_bgr: calls.append(1) or raw)
    run_mod._run_live_triggered(**kw)
    assert len(calls) == 1


def test_frame_dump_carries_rate_and_timestamps(harness, tmp_path):
    from optical.tracking.frame_store import (
        list_frame_indices,
        load_frame_timestamps,
        load_session_meta,
    )

    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE, 5: SPACE})
    kw["record_frames_dir"] = tmp_path / "leaf"
    run_mod._run_live_triggered(**kw)
    meta = load_session_meta(tmp_path / "leaf")
    assert meta["frame_rate_hz"] == 30.0 and meta["cam_ids"] == ["cam0", "cam1"]
    stamps = load_frame_timestamps(tmp_path / "leaf")
    assert sorted(stamps) == list_frame_indices(tmp_path / "leaf") == [0, 1, 2, 3, 4]
    t_cam, t_master = stamps[2]
    assert t_cam["cam1"] - t_cam["cam0"] == 30_000 and t_master is not None
    assert json.loads(read_tracking_session(out)["meta"]["frame_dump"])["dropped"] == 0


def test_jts_frame_intervals_are_inclusive(harness, tmp_path):
    """JTS frame_interval [start, end] includes end: 10 rows -> [0, 9]."""
    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE, 10: SPACE})
    kw["write_jts_config_file"] = True
    kw["jts_config_path"] = tmp_path / "jts.json"
    run_mod._run_live_triggered(**kw)
    n_rows = int(read_tracking_session(out)["meta"]["n_frames"])
    cfg = json.loads((tmp_path / "jts.json").read_text())
    exp = cfg["analysis"]["experiment"]
    assert n_rows == 10
    assert exp["frame_interval"] == [0, n_rows - 1]
    assert exp["sub_experiments"] == {"c": [0, 9]}


def test_interrupted_phase_is_inclusive_and_empty_phase_skipped(harness, tmp_path):
    from optical.tracking.experiment import Phase

    recorded = []
    run._close_phase(recorded, Phase("p", "experiment"), 5, 5)  # no frame captured
    assert recorded == []
    run._close_phase(recorded, Phase("p", "experiment"), 5, 8)
    assert recorded[0].interval == [5, 7]
    run_mod, kw, tracker, log, rig, out = harness(keys={0: SPACE}, rig_kw={"interrupt_at": 5})
    kw["write_jts_config_file"] = True
    kw["jts_config_path"] = tmp_path / "jts2.json"
    run_mod._run_live_triggered(**kw)
    cfg = json.loads((tmp_path / "jts2.json").read_text())
    assert cfg["analysis"]["experiment"]["frame_interval"] == [0, 3]  # rows 0..3
