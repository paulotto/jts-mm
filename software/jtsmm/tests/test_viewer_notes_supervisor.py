# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm record --viewer`: operator notes land in /events on both writer paths.

Notes are drained from the viewer's NoteLog during the take and once more
after the server stops, mapped host->master like phase edges, and re-mapped
in place with the final fit at close.
"""

from __future__ import annotations

import json

import h5py
import pytest

from jtsmm import native
from jtsmm.sync.clock_align import ClockAligner
from jtsmm.viewer.notes import NoteLog
from tests.test_fix_F_supervisor import env  # noqa: F401 - pytest fixture

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None
         and hasattr(native.NativeHdf5Writer, "append_event")),
    reason="native extension (with /events) not built",
)


class LateFitAligner(ClockAligner):
    """Live stamps map with offset 0; the 'end PPS' fit (at sync_report) adds 50 ms."""

    offset_us = 0

    def map_pi_ns(self, t_host_ns=None):
        return int(t_host_ns // 1000) + LateFitAligner.offset_us, "pi_fit"

    def sync_report(self):
        LateFitAligner.offset_us = 50_000
        return super().sync_report()


class FakeViewer:
    """ViewerServer stand-in: a real NoteLog, no hub, a late note at shutdown."""

    def __init__(self) -> None:
        self.notes = NoteLog()
        self.hub = None
        self.notes.add("first", 1.0)
        self.notes.add("Kiefer knackt – links", 2.5)

    def shutdown(self) -> None:
        # typed after the last in-loop flush, before the server stopped
        self.notes.add("after the loop", None)


def _run(env, monkeypatch):  # noqa: F811
    from jtsmm.capture.supervisor import SessionSupervisor

    LateFitAligner.offset_us = 0
    viewer = FakeViewer()
    monkeypatch.setattr(env.sup_mod, "ClockAligner", LateFitAligner)
    monkeypatch.setattr(env.sup_mod, "_FINAL_STATS_TIMEOUT_S", 0.05)
    monkeypatch.setattr(SessionSupervisor, "_maybe_start_viewer", lambda self, *a, **k: viewer)
    h5 = SessionSupervisor(env.resolved, port="/dev/null", viewer=True).run(duration_s=0.6)
    return h5, viewer


def _check(h5, viewer) -> None:
    with h5py.File(h5, "r") as f:
        ev = f["events"]
        kinds = list(ev["kind"].asstr()[()])
        texts = list(ev["text"].asstr()[()])
        t_master = [int(x) for x in ev["t_master_us"][()]]
        t_host = [int(x) for x in ev["t_host_ns"][()]]
        quals = list(ev["map_quality"].asstr()[()])
        view_t = [float(x) for x in ev["viewer_t_s"][()]]
        sync = json.loads(f["sync"].attrs["report_json"])
    assert kinds == ["note"] * 3
    assert texts == ["first", "Kiefer knackt – links", "after the loop"]  # arrival order
    assert t_host == sorted(t_host) and t_host[0] > 0
    # every stamp carries the final-fit correction (+50 ms), including the one
    # collected at teardown and the ones already on disk
    assert t_master == [h // 1000 + 50_000 for h in t_host]
    assert quals == ["pi_fit"] * 3
    assert view_t[:2] == [1.0, 2.5] and view_t[2] != view_t[2]  # NaN = no browser time
    notes = sync["viewer_notes"]
    assert notes["n_notes"] == 3 and notes["n_remapped"] == 3
    assert notes["max_abs_correction_us"] == 50_000 and notes["n_failed"] == 0
    assert viewer.notes.saved_to == h5.name


def test_notes_saved_native_writer_path(env, monkeypatch, capsys):  # noqa: F811
    h5, viewer = _run(env, monkeypatch)
    err = capsys.readouterr().err
    assert "HDF5 writer: native" in err
    assert "viewer notes: 3 saved to /events" in err
    _check(h5, viewer)


def test_notes_saved_python_writer_path(env, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("JTSMM_NATIVE")
    env.resolved["host"].setdefault("capture", {})["native_prefer"] = False
    h5, viewer = _run(env, monkeypatch)
    assert "HDF5 writer: Python" in capsys.readouterr().err
    _check(h5, viewer)


def test_stale_native_module_falls_back_to_python_writer(env, monkeypatch, capsys):  # noqa: F811
    """A native build without append_event must not silently drop the notes."""
    monkeypatch.setattr(native, "NativeHdf5Writer", object)
    h5, viewer = _run(env, monkeypatch)
    err = capsys.readouterr().err
    assert "predates /events" in err and "HDF5 writer: Python" in err
    _check(h5, viewer)


def test_remap_note_stamps_skips_mock_and_non_pps():
    from types import SimpleNamespace

    from jtsmm.capture.supervisor import remap_note_stamps

    calls = []
    writer = SimpleNamespace(set_event_time=lambda row, **kw: calls.append((row, kw)))
    notes = [{"row": 3, "t_host_ns": 2_000_000, "t_master_us": 1, "map_quality": "mcu_arrival"},
             {"row": 5, "t_host_ns": 4_000_000, "t_master_us": 1, "map_quality": "mock"}]
    out = remap_note_stamps(notes, LateFitAligner(), writer)
    assert out["n_remapped"] == 1 and calls == [(3, {"t_master_us": 2_000 + LateFitAligner.offset_us,
                                                     "map_quality": "pi_fit"})]
    fit = SimpleNamespace(map_pi_ns=lambda _t: (123, "mcu_arrival"))
    assert remap_note_stamps(notes, fit, writer)["n_remapped"] == 0
    assert remap_note_stamps(notes, LateFitAligner(), writer, enabled=False)["n_remapped"] == 0
