# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mid-take anchor refinement: fit reset, and the checker reporting it."""

from __future__ import annotations

from pathlib import Path

from jtsmm.integrity.checker import check_session
from jtsmm.sync.clock_align import ClockAligner

from tests.helpers import make_session_h5

P, T0, OFF_NS = 33_333, 2_000_000, 5_000_000_000


def _refined_aligner(delta: bool) -> ClockAligner:
    al = ClockAligner(pps_anchor=False, delta_tracking=delta, triggers_max=10_000)
    for k in range(6):
        tm = T0 + k * 50_000
        al.ingest_pps(tm * 1000 + OFF_NS, tm)
    for i in range(1, 600):
        al.note_trigger(i, T0 + i * P, stamp_source="pio_irq")
    # First delivered frame is really pulse 20; oldest_buffered labels it 1.
    for i in range(20, 600):
        al.ingest_cam("cam0", (T0 + i * P) * 1000 + OFF_NS, None, frame_idx=i)
    return al


def test_refinement_resets_the_camera_fit_so_no_frames_are_rejected():
    for delta in (True, False):
        al = _refined_aligner(delta)
        info = al.anchor_info["cam0"]
        assert info["refined_by_frames"] == 19, info
        assert info["refined_after_frames"] == 90
        assert info["pre_refine_median_frames"] < -18
        # A stale fit 19 periods off would reject every corrected frame until a
        # re-lock (validation_fail_count > 0, frames dropped).
        assert al.validation_fail_count.get("cam0", 0) == 0, delta
        assert al.drops.get("cam0", 0) == 0, delta
        st = al.assignment_residual_stats("cam0")
        assert abs(st["median_frames"]) < 0.1 and st["spread_frames"] < 0.1


def _anchor(n_bad: int) -> dict:
    return {"camera_anchor": {"cam0": {
        "method": "oldest_buffered", "trigger_id": 1, "note": "x",
        "refined_by_frames": 19, "refined_after_frames": n_bad,
    }}}


def test_checker_warns_about_frames_labelled_before_refinement(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_anchor(90))
    report = check_session(path)
    assert report.ok, report.issues
    assert any("first 90 frames" in w for w in report.warnings), report.warnings
    assert report.summary["sync"]["anchor_refined"]["cam0"] == {"by_frames": 19, "after_frames": 90}


def test_checker_fails_when_many_frames_were_mislabelled(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, sync_extra=_anchor(1200))
    report = check_session(path)
    assert any("first 1200 frames" in e for e in report.issues), report.issues
