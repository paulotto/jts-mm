# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Async frame dump and CT tip loading (pure numpy; write_frame stubbed)."""

from __future__ import annotations

import json
import threading
from types import SimpleNamespace

import numpy as np
import pytest

import optical.tracking.frame_store as fs
from optical.tracking.io import FrameRecord
from optical.tracking.tracker import apply_ct_tip, load_ct_tip_offset


def test_async_writer_writes_in_background_and_counts_drops(tmp_path, monkeypatch):
    gate = threading.Event()
    written = []

    def slow_write(session_dir, frame_idx, frames):
        gate.wait(5.0)
        written.append(frame_idx)

    monkeypatch.setattr(fs, "write_frame", slow_write)
    w = fs.AsyncFrameWriter(tmp_path, max_queue=2)
    results = [w.submit(i, {"cam0": np.zeros((2, 2), np.uint8)}) for i in range(6)]
    # At most one in flight + two queued; the rest are dropped, capture never blocks.
    assert results.count(True) in (2, 3) and w.dropped == 6 - results.count(True)
    gate.set()
    w.close()
    assert written == [i for i, ok in enumerate(results) if ok]
    assert w.stats()["written"] == len(written) and w.stats()["pending"] == 0


def test_async_writer_survives_write_errors(tmp_path, monkeypatch):
    def boom(session_dir, frame_idx, frames):
        raise OSError("disk full")

    monkeypatch.setattr(fs, "write_frame", boom)
    w = fs.AsyncFrameWriter(tmp_path)
    w.submit(0, {})
    w.submit(1, {})
    w.close()
    assert w.failed == 2 and "disk full" in w.first_error


def test_ct_tip_loaded_and_applied(tmp_path):
    p = tmp_path / "tip.json"
    p.write_text(json.dumps({"tip_body_mm": [1.0, -230.0, 4.0], "rms_mm": 1.25}))
    tip, meta = load_ct_tip_offset(SimpleNamespace(ct_tip_calibration_path=p))
    assert np.allclose(tip, [1.0, -230.0, 4.0])
    assert meta["ct_tip_rms_mm"] == "1.250000" and meta["ct_tip_source"] == str(p)
    rec = FrameRecord(frame_idx=0)
    T = np.eye(4)
    T[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    T[:3, 3] = [10.0, 20.0, 600.0]
    rec.set_pose("CT", T)
    apply_ct_tip(rec, tip)
    assert np.allclose(rec.pose_CT[:3, 3], T[:3, :3] @ tip + T[:3, 3])
    nan_rec = FrameRecord(frame_idx=1)
    apply_ct_tip(nan_rec, tip)  # untracked CT stays NaN
    assert np.all(np.isnan(nan_rec.pose_CT))


def test_ct_tip_unset_or_missing(tmp_path):
    assert load_ct_tip_offset(SimpleNamespace(ct_tip_calibration_path=None)) is None
    with pytest.raises(FileNotFoundError, match="ct_tip_calibration"):
        load_ct_tip_offset(SimpleNamespace(ct_tip_calibration_path=tmp_path / "nope.json"))


def test_shipped_config_points_at_shipped_tip():
    from optical.config.loader import package_etc_dir
    from optical.tracking.tracker import TrackerConfig

    cfg = TrackerConfig.from_path(package_etc_dir() / "optical.yaml", n_cameras=2)
    assert cfg.ct_tip_calibration_path == package_etc_dir() / "ct_tip_calibration.json"
    tip, _meta = load_ct_tip_offset(cfg)
    assert 200.0 < float(np.linalg.norm(tip)) < 260.0
