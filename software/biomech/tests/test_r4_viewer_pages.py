# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 viewer pages and privacy, with small fake contexts (fast): URL handling, cache key,
background exports, hidden tabs, pseudonyms, tracked array reads, video helpers."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("dash")

from dash import no_update  # noqa: E402

from biomech.viz.views import registry as R  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "biomech"


class FakeData:
    """The parts of DataContext the code under test touches."""

    def __init__(self, pids=("s01", "s02", "s10"), tables=None):
        self._pids = sorted(pids)
        self._tables = tables or {}

    def pids(self):
        return list(self._pids)

    def pseudonym(self, pid):
        return self._pids.index(pid) + 1

    def pseudonyms(self):
        return {p: i + 1 for i, p in enumerate(self._pids)}

    def table(self, stage, name, pids=None):
        return self._tables.get((stage, name), pd.DataFrame()).copy()


@pytest.fixture
def tmp_view():
    """A temporary dashboard view with typed pickers, alignments and perspectives."""
    vid = "zz_r4_test_view"

    @R.register(vid, "dashboard", "R4 test view", params=("align", "stat"),
                perspectives=("trials", "cohort"), aligns=("cycle", "time"), domain="qc",
                choices={"n": [1, 2, 3], "band": ["a", "b"]}, perspective="cohort",
                align="time", n=7)
    def _build(req):
        from biomech.viz.spec import Axis, FigureSpec, Panel

        return FigureSpec(vid, "t", [Panel("p", Axis("x"), Axis("y"), [])],
                          meta={"opts": dict(req.options), "align": req.align})

    yield vid
    R.REGISTRY.pop(vid, None)


# -- C1: URL ---------------------------------------------------------------------------------
def test_location_does_not_reload_on_search_writes():
    from biomech.viewer import app as A

    src = (SRC / "viewer" / "app.py").read_text()
    assert 'dcc.Location(id="url", refresh="callback-nav")' in src
    assert A.PAGES["/results"][1].__name__ == "layout"


def test_deep_link_restores_align_typed_pickers_and_validates(tmp_view):
    from biomech.viewer.pages import analysis as an

    ctx = SimpleNamespace(data=FakeData())
    c = an.controls_for(ctx, tmp_view, f"?view={tmp_view}&align=cycle&n=2&band=b"
                                       "&perspective=bogus&stat=median_iqr")
    assert c["align"] == "cycle"
    picks = {name: val for name, _, val in c["pickers"]}
    assert picks["n"] == 2 and isinstance(picks["n"], int)       # typed, not "2"
    assert picks["band"] == "b"
    assert c["persp"] == "cohort"                                # invalid -> view default
    assert c["stat"] == "median_iqr"
    # Defaults that are not offered fall back to the first value (n=7 is not a choice).
    d = an.controls_for(ctx, tmp_view, None)
    assert {name: val for name, _, val in d["pickers"]}["n"] == 1
    assert d["align"] == "time" and d["persp"] == "cohort"
    # A link to another view is ignored.
    o = an.controls_for(ctx, tmp_view, "?view=other&align=cycle")
    assert o["align"] == "time"


def test_show_writes_url_only_after_the_first_render(tmp_view, monkeypatch):
    from biomech.viewer.pages import analysis as an

    ctx = SimpleNamespace(data=FakeData(), config=SimpleNamespace(read_dir="x", snapshot=None),
                          store=SimpleNamespace(runs=lambda: []))
    an._CACHE.clear()
    body, url = an.show(ctx, tmp_view, "trials", None, None, None, "cycle", [2],
                        "light", 1200, [{"type": "an-choice", "name": "n"}], "/results",
                        initial=True)
    assert url is no_update
    body, url = an.show(ctx, tmp_view, "trials", None, None, None, "cycle", [2],
                        "light", 1200, [{"type": "an-choice", "name": "n"}], "/results")
    assert url.startswith("?") and "align=cycle" in url and "n=2" in url
    assert "perspective=trials" in url


# -- C6: cache key ---------------------------------------------------------------------------
def test_store_token_follows_run_progress_and_snapshot(tmp_path: Path):
    from biomech.results import ResultStore
    from biomech.viewer.pages.analysis import _store_token

    store = ResultStore(tmp_path)
    ctx = SimpleNamespace(config=SimpleNamespace(read_dir=tmp_path, snapshot=None),
                          store=store)
    t0 = _store_token(ctx)
    rd = store.run_dir("20250318T100000Z_aaaaaa")
    rd.mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({"run_id": "x"}))
    (rd / "events.jsonl").write_text('{"kind": "run_start"}\n')
    t1 = _store_token(ctx)
    with open(rd / "events.jsonl", "a") as fh:
        fh.write('{"kind": "unit_done"}\n')
    t2 = _store_token(ctx)
    (rd / "manifest.json").write_text(json.dumps({"finished_utc": "2026-10-18T10:05:00Z"}))
    t3 = _store_token(ctx)
    assert len({t0, t1, t2, t3}) == 4 and _store_token(ctx) == t3
    snap = SimpleNamespace(config=SimpleNamespace(read_dir=tmp_path / "snap", snapshot="v1"),
                           store=ResultStore(tmp_path / "snap"))
    assert _store_token(snap) != t0 and "v1" in _store_token(snap)


# -- C12: background exports -----------------------------------------------------------------
def test_export_job_runs_in_background_once_with_progress_and_cancel(monkeypatch, tmp_path):
    from biomech.viewer.jobs import Job
    from biomech.viewer.pages import paper as pb

    gate = threading.Event()
    seen = {}

    def fake_export(ctx, items, out, *, on_progress=None, cancel=None, **kw):
        res = {"figures": [], "tables": [], "errors": [], "problems": [], "cancelled": []}
        for i in range(50):
            if cancel():
                res["cancelled"].append(f"cancelled after {i}")
                break
            on_progress(i, 50, f"item{i}")
            seen["last"] = i
            gate.wait(5)
            res["figures"].append(f"f{i}")
        return res

    monkeypatch.setattr(pb, "export", fake_export)
    ctx = SimpleNamespace(exports=Job(), data=None, paper_items=[], paper_sets=[],
                          export_dir=tmp_path,
                          config=SimpleNamespace(out_dir=tmp_path))
    assert pb.start_export(ctx, "paper", ["c"]) is True
    assert pb.start_export(ctx, "every") is False                # double click: refused
    for _ in range(100):
        if ctx.exports.snapshot().total:
            break
        time.sleep(0.01)
    d = ctx.exports.describe()
    assert d["running"] and d["max"] == 50 and "paper export" in d["text"]
    ctx.exports.cancel()
    gate.set()
    ctx.exports.join(5)
    s = ctx.exports.snapshot()
    assert not s.running and s.cancelled and seen["last"] < 49
    assert "cancelled" in ctx.exports.describe()["text"]
    assert pb.start_export(ctx, "every") is True                 # a new job after the first
    ctx.exports.join(5)


def test_export_reports_progress_and_honours_cancel(tmp_path, monkeypatch):
    from biomech.viz import export as X

    calls = []
    built = []
    monkeypatch.setattr(X.R, "request", lambda ctx, vid, **kw: vid)
    monkeypatch.setattr(X.R, "build", lambda vid, req: built.append(vid) or
                        __import__("biomech.viz.spec", fromlist=["TableSpec"]).TableSpec(
                            vid, "c", [], []))
    stop = {"n": 2}
    res = X.export(None, ["a", "b", "c", "d"], tmp_path, previews=False,
                   on_progress=lambda d, t, lab: calls.append((d, t, lab)),
                   cancel=lambda: len(built) >= stop["n"])
    assert calls[:2] == [(0, 4, "a"), (1, 4, "b")] and built == ["a", "b"]
    assert res["cancelled"] and res["tables"] == ["a", "b"]


# -- C9: hidden tabs, cohort arrays through the context ---------------------------------------
def test_hidden_tabs_compute_nothing():
    from biomech.viewer.pages import analysis as an
    from biomech.viewer.pages import explorer as ex

    boom = SimpleNamespace()                    # any access would raise AttributeError
    assert an.grid(boom, "kinematics", "summary", None, "views") == (no_update,) * 6
    assert an.quick_plot(boom, "x", "pid", "s", "t", None, "light", 1200, "views") is no_update
    assert ex.atlas_body(boom, "data_timeline", None, None, "s01", None, "light", 1200,
                         "signals") is no_update


def test_no_direct_cohort_array_reads_in_views():
    files = list((SRC / "viz" / "views").glob("*.py")) + [SRC / "viewer" / "data.py",
                                                          SRC / "anim" / "source.py"]
    bad = [f.name for f in files if re.search(r"store\.read_arrays\([^)]*(_cohort|COHORT_UNIT)",
                                              f.read_text())]
    assert not bad


# -- C13: privacy ----------------------------------------------------------------------------
def test_table_browser_uses_pseudonyms_in_natural_order(monkeypatch):
    from biomech.viewer.pages import analysis as an

    pids = [f"s{i:02d}" for i in range(1, 12)]
    df = pd.DataFrame({"pid": ["s11", "s02", "s99", "s10", "s01"], "v": [1.0, 2, 3, 4, 5],
                       "unit": "x"})
    data = FakeData(pids, {("kinematics", "summary"): df})
    ctx = SimpleNamespace(data=data)
    monkeypatch.setattr("biomech.pipeline.get_stage",
                        lambda s: SimpleNamespace(scope="participant"))
    out = an._browse(ctx, "kinematics", "summary", None)
    assert list(out.pid) == ["P1", "P2", "P10", "P11", "excluded"]
    assert "unit" not in out and not out.pid.str.contains(r"s\d\d").any()
    body, info, *_ = an.grid(ctx, "kinematics", "summary", None, "tables")
    assert [r["pid"] for r in body.data] == ["P1", "P2", "P10", "P11", "excluded"]


def test_frame_table_and_pseudo_never_fall_back_to_study_ids():
    from biomech.viz.views import blocks as B

    df = pd.DataFrame({"pid": ["s01", "s77"], "x": [1.0, 2.0]})
    t = B.frame_table(df, "tt", "cap", ctx=FakeData(("s01",)))
    assert [r["pid"] for r in t.rows] == ["P1", "excluded"]
    with pytest.raises(ValueError):
        B.frame_table(df, "tt", "cap")
    with pytest.raises(ValueError):
        R._pseudo(FakeData(("s01",)), "s77")
    assert R._pseudo(FakeData(("s01",)), "s01") == "P1"


def test_build_says_when_another_participant_is_shown(tmp_view):
    vid = "zz_r4_pid_view"

    @R.register(vid, "dashboard", "pid view", params=("pid",), domain="qc")
    def _b(req):
        from biomech.viz.spec import Axis, FigureSpec, Panel

        shown = req.pid if req.pid in req.ctx.pids() else req.ctx.pids()[0]
        return FigureSpec(vid, "t", [Panel("p", Axis("x"), Axis("y"), [])], caption="cap.",
                          meta={"pid": shown})

    try:
        ctx = FakeData(("s01", "s02"))
        spec = R.build(vid, R.request(ctx, vid, pid="s55"))
        assert spec.id == vid                                   # no pid token
        assert "not included" in spec.caption and "showing P1" in spec.caption
        assert "showing P1" in spec.panels[0].note and spec.meta["substituted"]
        ok = R.build(vid, R.request(ctx, vid, pid="s02"))
        assert ok.id == f"{vid}--pid-P2" and "not included" not in ok.caption
    finally:
        R.REGISTRY.pop(vid, None)


def test_privacy_scan_covers_ids_file_names_and_video_manifests(tmp_path):
    from biomech.check import outputs as O
    from biomech.viz.spec import Axis, FigureSpec, Panel

    spec = FigureSpec("fig--pid-s03", "t", [Panel("p", Axis("x"), Axis("y"), [])])
    assert O.pid_leaks(O._spec_text(spec)) == ["s03"]
    (tmp_path / "paper" / "figures" / "x_s04").mkdir(parents=True)
    vids = tmp_path / "videos"
    vids.mkdir()
    (vids / "multimodal_P1_chewing.json").write_text(json.dumps({"pid": "s05"}))
    (vids / "multimodal_s06_chewing.mp4").write_bytes(b"")
    rep = O.Report("t")
    O.check_file_names(rep, tmp_path / "paper", "exported files")
    O.check_video_privacy(rep, vids)
    fails = {i.name: i.detail for i in rep.items if i.status == "fail"}
    assert any("s04" in d for d in fails.values())
    assert any("s05" in d for d in fails.values())
    assert any("s06" in d for d in fails.values())


def test_tracking_dict_covers_values_copies_and_unpacking():
    import copy

    from biomech.results.access import track, tracking

    d = tracking("st", {"open_close__x": 1, "chewing__y": 2})
    for f in (lambda: dict(d), lambda: {**d}, lambda: list(d.values()),
              lambda: {}.update(d), lambda: d | {}):
        with track() as log:
            f()
        assert log.arrays == {("st", "open_close__x"), ("st", "chewing__y")}
    with track() as log:
        list(d)                                 # keys only: nothing read
        c = d.copy()
        c2 = copy.copy(d)
    assert not log.arrays and type(c).__name__ == "TrackingDict" and c2.stage == "st"
    with track() as log:
        c["chewing__y"]
    assert log.arrays == {("st", "chewing__y")}


# -- C14: robustness -------------------------------------------------------------------------
def test_landmark_qc_on_empty_tables():
    from biomech.viz.views.dashboards import landmark_qc

    spec = landmark_qc(SimpleNamespace(ctx=FakeData()))
    assert len(spec.panels) == 2 and "no anatomy" in spec.panels[1].note
    from biomech.viz.render_plotly import to_plotly

    to_plotly(spec)


def test_viewer_tables_render_units_and_plain_text():
    from biomech.viewer.pages.analysis import _table
    from biomech.viz.spec import Column, TableSpec

    spec = TableSpec("t", "Mean $\\pm$ SD", [Column("m", "Metric", align="l"),
                                             Column("v", "Opening $\\theta$", "\\degree", 1)],
                     [{"m": "range $\\geq$ 40", "v": 41.26}, {"_rule": True},
                      {"m": "x", "v": float("nan")}], notes="\\si{\\milli\\metre} note")
    div = _table(spec)
    cap, grid, notes = div.children
    assert cap.children == "Mean ± SD" and notes.children == "mm note"
    assert [c["name"] for c in grid.columns] == ["Metric", "Opening θ (°)"]
    assert grid.data[0] == {"m": "range ≥ 40", "v": "41.3"} and grid.data[1]["v"] == "–"


# -- C17: videos -----------------------------------------------------------------------------
def test_video_stem_fps_and_never_a_study_id():
    from biomech.anim.layouts import video_stem

    a = video_stem("multimodal", "s01", ["chewing"], pseudonym=3)
    assert a == "multimodal_P3_chewing"
    assert video_stem("multimodal", "s01", ["chewing"], pseudonym=3, fps=30) == a
    assert video_stem("multimodal", "s01", ["chewing"], pseudonym=3, fps=60).endswith("_60fps")
    with pytest.raises(ValueError):
        video_stem("multimodal", "s01", ["chewing"])
    assert video_stem("cohort", None, ["open_close"]) == "cohort_open_close"


def test_decode_stops_after_the_last_wanted_frame(monkeypatch):
    from biomech.anim import render_video as RV

    W, H, N = 4, 2, 1000
    reads = {"n": 0}

    class Out:
        def read(self, n):
            reads["n"] += 1
            return bytes([reads["n"] % 256]) * n if reads["n"] <= N else b""

        def close(self):
            pass

    class Proc:
        stdout = Out()
        killed = False

        def poll(self):
            return None

        def kill(self):
            Proc.killed = True

        def wait(self):
            return 0

    monkeypatch.setattr(RV, "video_frame_size", lambda p: (W, H))
    monkeypatch.setattr(RV.subprocess, "Popen", lambda *a, **k: Proc())
    frames = RV.decode("x.mp4", [2, 5])
    assert len(frames) == 2 and reads["n"] == 6 and Proc.killed
    assert frames[0][0, 0, 0] == 3 and frames[1][0, 0, 0] == 6


def test_load_clip_cuts_every_phase(monkeypatch):
    from biomech.anim import source as S

    phases = {"open_close": (0.0, 10e6), "chewing": (20e6, 25e6)}
    ctx = SimpleNamespace(
        pids=lambda: ["s01"], pseudonym=lambda p: 1,
        phase=lambda pid, ph: ("motion", SimpleNamespace(t_start_us=phases[ph][0],
                                                         t_end_us=phases[ph][1])))
    monkeypatch.setattr(S, "_kinematics", lambda *a: (None, None))
    monkeypatch.setattr(S, "_stage", lambda *a: None)
    monkeypatch.setattr(S, "_session", lambda *a: ([], None, None))
    monkeypatch.setattr(S, "_posterior", lambda *a: None)
    clip = S.load_clip(ctx, "s01", ["open_close", "chewing"], 1.0, 3.0)
    assert clip.segments == [("open_close", 1e6, 3e6), ("chewing", 21e6, 23e6)]
    with pytest.raises(ValueError, match="chewing"):
        S.load_clip(ctx, "s01", ["open_close", "chewing"], 6.0, 8.0)
    assert np.isclose(clip.t1_us, 23e6)
