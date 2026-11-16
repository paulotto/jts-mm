# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J2.2: the ``surface3d`` series kind in the three renderers."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np

from biomech.viz import render_mpl, render_pgf, render_plotly
from biomech.viz.spec import Axis, FigureSpec, Panel, Series

LUALATEX = shutil.which("lualatex") is not None


def _spec() -> FigureSpec:
    x, y = np.linspace(0, 1, 12), np.linspace(0, 2, 8)
    z = np.outer(np.sin(y), np.cos(x))
    p = Panel("Surface", Axis("Phase"), Axis("Task"),
              [Series("surface3d", x=x, y=y, z=z, style={"ramp": "diverging"})],
              projection="3d", z=Axis("Level"))
    return FigureSpec("surf_fig", "surface", [p], width="column", height_in=2.2)


def test_mpl_and_plotly_draw_a_surface(tmp_path):
    spec = _spec()
    render_mpl.render_figure(spec, tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000
    fig = render_plotly.to_plotly(spec)
    assert any(t.type == "surface" for t in fig.data)


def test_pgf_writes_a_surf_plot(tmp_path):
    spec = _spec()
    render_pgf.write_style(tmp_path, "ieee")
    tex = render_pgf.render_figure(spec, tmp_path, venue="ieee").read_text()
    assert "\\addplot3[surf" in tex and "mesh/rows=8" in tex and "colormap name=jtdiv" in tex
    if LUALATEX:
        doc = render_pgf.standalone_figure(tmp_path, spec.id, "ieee", natural=False)
        r = subprocess.run(["lualatex", "-interaction=nonstopmode", doc.name], cwd=tmp_path,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-1500:]
