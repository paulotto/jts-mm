# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Viewer notes in /events are exposed by the reader, integrity summary and plots."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.io.session_reader import Session
from tests.helpers import make_session_h5


def _add_events(path: Path) -> None:
    """Append /events like the recorder: notes during the take, phases at close."""
    rows = [  # write order != time order
        (60_000, "note", "", "first note", 1.0),
        (20_000, "note", "", "Kiefer knackt – ü", None),
        (10_000, "phase_start", "experiment", "open_close", None),
        (90_000, "phase_end", "experiment", "open_close", None),
    ]
    s = h5py.string_dtype()
    with h5py.File(path, "a") as f:
        g = f.require_group("events")
        g.create_dataset("t_master_us", data=np.asarray([r[0] for r in rows], dtype=np.uint64))
        g.create_dataset("kind", data=np.asarray([r[1] for r in rows], dtype=s))
        g.create_dataset("phase_kind", data=np.asarray([r[2] for r in rows], dtype=s))
        g.create_dataset("text", data=np.asarray([r[3] for r in rows], dtype=s))
        g.create_dataset("t_host_ns", data=np.arange(len(rows), dtype=np.uint64))
        g.create_dataset("map_quality", data=np.asarray(["pi_fit"] * len(rows), dtype=s))
        g.create_dataset("viewer_t_s", data=np.asarray(
            [np.nan if r[4] is None else r[4] for r in rows], dtype=np.float64))


def test_reader_exposes_notes_sorted_by_time(tmp_path: Path):
    path = tmp_path / "s.h5"
    w = Hdf5Writer(path=path, config_hash="x", mcu_config={}, host_config={})
    w.open()
    w.append_event(t_master_us=5_000, kind="note", text="late write, early time",
                   t_host_ns=7, map_quality="pi_fit", viewer_t_s=0.5)
    w.flush()
    w.append_event(t_master_us=1_000, kind="phase_start", text="p", phase_kind="experiment")
    w.append_event(t_master_us=9_000, kind="note", text="second", t_host_ns=9)
    w.close()
    with Session.open(path) as s:
        ev = s.events()
        notes = s.notes()
    assert [e["t_master_us"] for e in ev] == [1_000, 5_000, 9_000]
    assert [n["text"] for n in notes] == ["late write, early time", "second"]
    assert notes[0]["viewer_t_s"] == 0.5 and notes[1]["viewer_t_s"] is None
    assert notes[0]["t_host_ns"] == 7 and notes[0]["map_quality"] == "pi_fit"


def test_reader_without_events_or_viewer_t_s(tmp_path: Path):
    path = tmp_path / "s.h5"
    make_session_h5(path, duration_s=0.2, emg_samples=40)
    with Session.open(path) as s:
        assert s.notes() == [] and s.events() == []
    with h5py.File(path, "a") as f:  # older file layout: no viewer_t_s column
        g = f.require_group("events")
        g.create_dataset("t_master_us", data=np.asarray([3], dtype=np.uint64))
        g.create_dataset("kind", data=np.asarray(["note"], dtype=h5py.string_dtype()))
    with Session.open(path) as s:
        assert s.notes() == [{"t_master_us": 3, "kind": "note", "text": "", "phase_kind": "",
                              "t_host_ns": 0, "map_quality": "", "viewer_t_s": None}]


def test_integrity_summary_counts_notes(tmp_path: Path):
    from jtsmm.integrity.checker import check_session, format_report

    path = tmp_path / "session.h5"
    make_session_h5(path, duration_s=0.2, emg_samples=40)
    _add_events(path)
    report = check_session(path)
    assert report.summary["events"] == {"total": 4, "notes": 2, "phase_edges": 2, "mcu_logs": 0}
    text = format_report(report)
    assert "viewer notes" in text and "in /events (kind=note)" in text
    assert not any("note" in m for m in report.issues + report.warnings)  # informational only


def test_visualize_session_marks_notes(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.visualize_session import load_session_viz, plot_matplotlib

    path = tmp_path / "session.h5"
    make_session_h5(path, duration_s=0.3, emg_samples=60)
    _add_events(path)
    viz = load_session_viz(path)
    assert viz.notes == [(20_000, "Kiefer knackt – ü"), (60_000, "first note")]
    written = plot_matplotlib(viz, save=tmp_path / "png", show=False, show_phases=True)
    assert any("emg" in p.name for p in written)
