# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""biomech check-outputs: a clean study passes, and every kind of corruption is caught."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from biomech.check import outputs as C
from biomech.io.context import DataContext
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views import registry as R

pytestmark = pytest.mark.slow


def _fails(rep: C.Report) -> list[C.Item]:
    return [i for i in rep.items if i.status == "fail"]


def test_clean_study_passes(synth_study, tmp_path):
    cfg = synth_study[0]
    rep = C.check_outputs(cfg, tmp_path, kinds=("paper", "dashboard"), videos=False,
                          pgf=shutil.which("lualatex") is not None)
    assert _fails(rep) == [], _fails(rep)
    assert (tmp_path / "report.html").is_file() and (tmp_path / "report.json").is_file()
    assert any(p.name.startswith("sheet_paper") for p in tmp_path.glob("sheet_*.png"))


def test_wrong_table_cell_is_caught(synth_study, monkeypatch):
    ctx = DataContext(synth_study[0])
    orig = R.get("table_reference")

    def tampered(req):
        spec = orig.builder(req)
        spec.rows[0]["mean"] = float(spec.rows[0]["mean"]) + 0.5
        return spec

    monkeypatch.setitem(R.REGISTRY, "table_reference", replace(orig, builder=tampered))
    rep = C.Report("t")
    C.check_tables(rep, ctx)
    assert any("table_reference" in i.name for i in _fails(rep))


def _bad_view(req):
    long = "Averyveryveryverylongunbreakabletitlethatcannotwrapanywhereatall" * 2
    p = Panel(long, Axis("x"), Axis("y"), [Series("line", np.arange(5.0), np.arange(5.0)),
                                           Series("text", np.array([1.0]), np.array([1.0]),
                                                  text=["participant s03"])])
    return FigureSpec("bad_view", "bad", [p], width="column")


def test_clipped_figure_and_leaked_id_are_caught(synth_study, tmp_path, monkeypatch):
    monkeypatch.setitem(R.REGISTRY, "bad_view", R.ViewDef("bad_view", "dashboard", "bad",
                                                          _bad_view))
    ctx = DataContext(synth_study[0])
    rep = C.Report("t")
    C.check_figures(rep, ctx, tmp_path, pgf=False, kinds=("dashboard",))
    bad = [i for i in _fails(rep) if "bad_view" in i.name]
    assert any("overlapping or clipped" in i.name for i in bad)
    assert any("participant ids" in i.name for i in bad)


@pytest.mark.skipif(shutil.which("lualatex") is None, reason="no LuaLaTeX")
def test_oversized_pgf_figure_is_caught(synth_study, tmp_path):
    from biomech.viz.export import check_figure_size
    from biomech.viz import render_pgf

    spec = _bad_view(None)
    render_pgf.write_style(tmp_path)
    render_pgf.render_figure(spec, tmp_path)
    assert any("wide" in p for p in check_figure_size(tmp_path, spec))


@pytest.mark.skipif(shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
                    reason="no ffmpeg/ffprobe (the video check probes with ffprobe)")
def test_off_sync_video_is_caught(tmp_path):
    vids = tmp_path / "videos"
    vids.mkdir()
    mp4 = vids / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10",
                    "-t", "1", "-pix_fmt", "yuv420p", "-c:v", "libx264", str(mp4)], check=True)
    meta = {"fps": 10, "width": 160, "height": 120, "expected_duration_s": 1.0,
            "with_camera": False, "frames": [[0, 1_000_000], [5, 1_500_000]]}
    (vids / "clip.json").write_text(json.dumps(meta))
    rep = C.Report("t")
    C.check_videos(rep, vids, read_timecode=lambda img: 1_000_000.0)    # frame 5 is off
    names = {i.name: i.status for i in rep.items}
    assert names["clip.mp4: stream"] == "pass"
    assert names["clip.mp4: embedded timecode matches the timeline"] == "fail"
    rep2 = C.Report("t")
    C.check_videos(rep2, vids, read_timecode=lambda img, it=iter([1_000_000.0, 1_500_000.0]):
                   next(it))
    assert {i.name: i.status for i in rep2.items}[
        "clip.mp4: embedded timecode matches the timeline"] == "pass"


def test_stray_unit_outside_cohort_is_flagged(synth_study):
    cfg, store = synth_study[0], synth_study[1]
    store.write("qc", "s99", "x", tables={"sessions": pd.DataFrame({"pid": ["s99"]})})
    try:
        rep = C.Report("t")
        C.check_coverage(rep, DataContext(cfg))
        assert any(i.status == "warn" and "s99" in i.detail for i in rep.items)
    finally:
        shutil.rmtree(store._unit_dir("qc", "s99"))


def test_documented_unavailable_unit_is_reported_not_failed(tmp_path):
    p = tmp_path / "cohort.yaml"
    p.write_text("participants:\n- pid: S09\n  unavailable:\n    imu: no IMU orientation\n"
                 "- pid: s08\n", encoding="utf-8")
    assert C.unavailable_units(p) == {("s09", "imu"): "no IMU orientation"}
    assert C.unavailable_units(tmp_path / "missing.yaml") == {}
