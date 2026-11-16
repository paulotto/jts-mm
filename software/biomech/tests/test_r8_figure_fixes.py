# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Figure-review fixes that are renderer or number-rule behaviour (not one view's text)."""

from __future__ import annotations

import numpy as np

from biomech.viz import layout as L
from biomech.viz import render_pgf
from biomech.viz.spec import Axis, FigureSpec, Panel, Series
from biomech.viz.views import blocks as B


def test_signed_numbers_never_print_a_negative_zero():
    assert B.snum(-0.004, 1) == "0.0"
    assert B.snum(0.04, 1) == "0.0"
    assert B.snum(0.4, 1) == "+0.4"
    assert B.snum(-0.4, 1) == "−0.4"
    assert B.snum(np.nan, 1) == "--"


def test_p_text_reads_as_running_text():
    assert B.ptext(0.0123) == "p = 0.012"
    assert B.ptext(0.0001) == "p < 0.001"


def test_pgf_image_takes_no_legend_slot(tmp_path):
    """A graphics plot before labelled series must not shift their legend images (
    emg_onset_raster, emg_cycle_spectrogram)."""
    img = np.arange(20.0).reshape(4, 5)
    p = Panel("Raster", Axis("Time"), Axis("Row"),
              [Series("image", img=img, extent=(0, 5, 0, 4), style={"cmap": "sequential"}),
               Series("scatter", np.array([1.0]), np.array([1.0]), role="primary",
                      label="onset")], legend="below")
    spec = FigureSpec("img_legend", "image legend", [p], width="column", height_in=1.5)
    render_pgf.write_style(tmp_path, "ieee")
    tex = render_pgf.render_figure(spec, tmp_path, venue="ieee").read_text()
    assert "\\addplot[forget plot] graphics[" in tex


def test_3d_axes_get_at_most_a_few_ticks():
    x = np.linspace(-15, 15, 50)
    p = Panel("Cloud", Axis("Anterior"), Axis("Left"),
              [Series("scatter3d", x, x, z=x)], projection="3d", z=Axis("Superior"))
    lay = L.compute(FigureSpec("cloud3d", "cloud", [p], width="text", height_in=3.0))
    pl = lay.panels[0]
    assert len(pl.x.ticks) <= 5 and len(pl.y.ticks) <= 5


def test_pgf_colorbar_ticks_are_text_labels(tmp_path):
    """Colour-bar ticks as explicit labels (text font, real minus), not pgfplots math numbers
   ."""
    import shutil
    import subprocess

    from biomech.viz.spec import Colorbar

    img = np.linspace(-2.0, 2.0, 20).reshape(4, 5)
    p = Panel("Map", Axis("Time"), Axis("Row"),
              [Series("image", img=img, extent=(0, 5, 0, 4),
                      style={"cmap": "diverging", "lim": (-2.0, 2.0)})],
              colorbar=Colorbar("Change", "", lim=(-2.0, 2.0), ramp="diverging"))
    spec = FigureSpec("cb_ticks", "colorbar", [p], width="column", height_in=1.5)
    render_pgf.write_style(tmp_path, "ieee")
    tex = render_pgf.render_figure(spec, tmp_path, venue="ieee").read_text()
    assert "yticklabels={{{\\jtsmmminus}2}" in tex
    if shutil.which("lualatex"):
        doc = render_pgf.standalone_figure(tmp_path, spec.id, "ieee", natural=False)
        r = subprocess.run(["lualatex", "-interaction=nonstopmode", doc.name], cwd=tmp_path,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stdout[-1500:]


def test_corner_tick_labels_do_not_collide():
    """Both lowest ticks at the bottom-left corner: the y label there is blanked."""
    x = np.linspace(-600, 200, 50)
    p = Panel("Trace", Axis("Time", lim=(-600.0, 200.0)), Axis("Level", lim=(-20.0, 20.0)),
              [Series("line", x, np.sin(x / 50) * 15)])
    pl = L.compute(FigureSpec("corner", "corner", [p], width="column", height_in=1.6)).panels[0]
    assert pl.x.labels[0] and pl.y.ticks[0] == -20.0 and pl.y.labels[0] == ""


def test_pictures_are_drawn_without_a_frame(tmp_path):
    """Explicitly empty ticks on both axes: no axis lines or grid (hardware renders, the
    activation clock)."""
    p = Panel("Render", Axis("", ticks=[], ticklabels=[]), Axis("", ticks=[], ticklabels=[]),
              [Series("image", img=np.zeros((4, 5, 3)), extent=(0, 5, 0, 4))])
    q = Panel("Plot", Axis("Time"), Axis("Level"), [Series("line", np.arange(3.0),
                                                           np.arange(3.0))])
    assert L.frameless(p) and not L.frameless(q)
    spec = FigureSpec("frameless", "frameless", [p], width="column", height_in=1.5)
    render_pgf.write_style(tmp_path, "ieee")
    tex = render_pgf.render_figure(spec, tmp_path, venue="ieee").read_text()
    assert "axis line style={draw=none}" in tex


def test_pgf_errorbar_keeps_the_series_marker(tmp_path):
    """The participant shape set on an errorbar series reaches the PGF (systems X1)."""
    p = Panel("Lag", Axis("Window"), Axis("Lag"),
              [Series("errorbar", np.array([1.0, 2.0]), np.array([0.5, 0.6]),
                      y2=np.array([0.7, 0.8]), ylo=np.array([0.3, 0.4]), role="primary",
                      style={"ms": "s"})])
    spec = FigureSpec("eb_mark", "errorbar", [p], width="column", height_in=1.5)
    render_pgf.write_style(tmp_path, "ieee")
    tex = render_pgf.render_figure(spec, tmp_path, venue="ieee").read_text()
    assert render_pgf._mark("s") in tex
