# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trigger edges lost without an id gap (pre-K4 firmware) and id-gap handling."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jtsmm.integrity.checker import check_session
from jtsmm.sync.clock_align import ClockAligner

from tests.helpers import make_session_h5


def _drop_edge_keep_ids(path: Path, at: int) -> None:
    """Remove one trigger time but keep ids contiguous (merged PIO IRQ flag)."""
    with h5py.File(path, "a") as f:
        g = f["triggers"]
        t = np.asarray(g["t_master_us"], dtype=np.uint64)
        ids = np.asarray(g["trigger_id"], dtype=np.uint64)
        t2 = np.delete(t, at)
        ids2 = ids[: t2.size]
        del g["t_master_us"], g["trigger_id"]
        g.create_dataset("t_master_us", data=t2)
        g.create_dataset("trigger_id", data=ids2)


def test_lost_edge_without_id_gap_is_an_error_that_says_where(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=2.0)
    _drop_edge_keep_ids(path, 40)
    report = check_session(path)
    msgs = [e for e in report.issues if "lost without an id gap" in e]
    assert msgs, report.issues
    assert "after trigger_id 40" in msgs[0] and "t=0.65 s" in msgs[0], msgs[0]
    assert report.summary["triggers"]["edges_lost_without_id_gap"] == 1


def test_clean_triggers_do_not_trip_the_lost_edge_check(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=2.0)
    report = check_session(path)
    assert not any("lost without an id gap" in e for e in report.issues)
    assert report.summary["triggers"]["edges_lost_without_id_gap"] == 0


def test_id_gap_is_reported_with_its_location(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, noncontiguous_triggers=True)
    report = check_session(path)
    msg = next(e for e in report.issues if "not contiguous" in e)
    assert "5 ids missing" in msg and "2->8" in msg, msg
    # an id gap whose time step matches is NOT also a lost edge
    assert not any("lost without an id gap" in e for e in report.issues)


def test_aligner_period_estimate_ignores_id_gaps_and_counts_lost_edges():
    al = ClockAligner()
    period = 33_333
    tid, t = 0, 1_000_000
    for i in range(100):
        tid += 1
        t += period
        if i == 50:
            tid += 3          # K4: firmware advanced ids for 3 lost edges
            t += 3 * period
        if i == 70:
            t += period       # pre-K4: edge lost, ids contiguous
        al.note_trigger(tid, t)
    assert abs(al._trigger_period_us - period) < 50  # noqa: SLF001
    assert al.trigger_lost_edges == 1
    assert al.sync_report()["trigger_lost_edges"] == 1
