# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Spec renderers (pgf, matplotlib, Plotly) and viewer smoke tests."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from biomech.viz import render_mpl, render_pgf
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec


def _spec() -> FigureSpec:
    x = np.linspace(0, 10, 5000)
    y = np.sin(x)
    y[100] = np.nan
    p1 = Panel("(a) line & band", Axis("Time", "\\second"), Axis("Angle", "\\degree"),
               [Series("band", x, y - 0.2, y + 0.2, role="band"),
                Series("line", x, y, label="mean $\\pm$ SD", role="primary"),
                Series("hline", y=np.array([0.0]), role="reference"),
                Series("hspan", np.array([2.0]), y2=np.array([3.0]), role="muted")],
               legend="below")
    p2 = Panel("(b) heatmap", Axis(""), Axis(""),
               [Series("heatmap", z=np.arange(9.0).reshape(3, 3),
                       text=[str(i) for i in range(9)])], legend=None)
    p2.x.ticks, p2.x.ticklabels = [0, 1, 2], ["a_b", "c", "d"]
    return FigureSpec("t_fig", "test", [p1, p2], rows=1, cols=2, width="text")


def _table() -> TableSpec:
    return TableSpec("t_tab", "cap", [Column("m", "Metric", align="l"),
                                      Column("v", "Value", "\\milli\\metre", 1)],
                     [{"m": "Opening ≥ 40 %", "v": 25.04}, {"_rule": True},
                      {"m": "x", "v": float("nan")}])


def test_pgf_structure(tmp_path: Path):
    render_pgf.write_style(tmp_path)
    f = render_pgf.render_figure(_spec(), tmp_path)
    tex = f.read_text()
    assert tex.count("\\begin{axis}") == 2 and "anchor=north west" in tex
    assert "figures/t_fig/p0s1.dat" in tex and "matrix plot*" in tex
    dat = (tmp_path / "figures/t_fig/p0s1.dat").read_text().splitlines()
    assert len(dat) - 1 <= render_pgf.MAX_POINTS + 2 and any("nan" in r for r in dat)
    t = render_pgf.render_table(_table(), tmp_path).read_text()
    assert "\\toprule" in t and "\\jtsmmsym{≥}{\\geq}" in t and "\\%" in t and "{--}" in t


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="no LaTeX")
def test_pgf_compiles(tmp_path: Path):
    render_pgf.write_style(tmp_path)
    render_pgf.render_figure(_spec(), tmp_path)
    render_pgf.render_table(_table(), tmp_path)
    tex = render_pgf.standalone_preview(tmp_path, [("figure", "t_fig"), ("table", "t_tab")])
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                        tex.name], cwd=tmp_path, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and "pgfplots.sty' not found" in r.stdout:
        pytest.skip("pgfplots not installed")
    assert r.returncode == 0, r.stdout[-2000:]


def test_mpl_and_plotly(tmp_path: Path):
    from biomech.viz.render_plotly import apply_zoom, to_plotly

    assert render_mpl.render_figure(_spec(), tmp_path / "x.png").is_file()
    fig = to_plotly(_spec())
    assert len(fig.data) >= 3
    z = apply_zoom(_spec(), {"xaxis.range[0]": 1.0, "xaxis.range[1]": 4.0,
                             "yaxis2.autorange": True})
    assert z.panels[0].x.lim == (1.0, 4.0) and z.panels[1].y.lim is None


def test_viewer_builds_pages(study, tmp_path: Path):
    pytest.importorskip("dash")
    from biomech.viewer.app import create_app
    from biomech.viewer.context import ViewerContext
    from biomech.viewer.pages import animation, control, explorer, paper

    cfg = study()
    ctx = ViewerContext.from_study(cfg.source)
    app = create_app(ctx)
    assert app.layout is not None
    control.layout(ctx)                  # status matrix against an empty store
    explorer.layout(ctx)
    animation.layout(ctx)
    page = paper.layout(ctx)    # every paper item fails gracefully without results
    assert page is not None
    assert ctx.runs.progress() == (0, 0)
