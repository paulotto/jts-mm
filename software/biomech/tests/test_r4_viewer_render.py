# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 renderer fixes: notes, colourbars, log annotations, ramps, legends, colour lines,
point budget, matplotlib without pyplot, pgf bands / colour limits (fast, no study)."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from biomech.viz import render_mpl, render_pgf, style
from biomech.viz import render_plotly as rp
from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series


def _fig(*panels, cols=None, **kw):
    return FigureSpec("t", "t", list(panels), rows=1, cols=cols or len(panels), width="text",
                      **kw)


def _ann_texts(fig):
    return [a.text for a in fig.layout.annotations]


# -- C2: panel notes ---------------------------------------------------------------------------
def test_plotly_draws_panel_notes_2d_and_3d():
    x = np.linspace(0, 1, 20)
    p2 = Panel("a", Axis("x"), Axis("y"), [Series("line", x, x)], note="p95 3.1 mm")
    p3 = Panel("b", Axis("x"), Axis("y"), [Series("line3d", x, x, z=x)], projection="3d",
               note="3-D note")
    fig = rp.to_plotly(_fig(p2, p3))
    notes = {a.text: a for a in fig.layout.annotations if a.text in ("p95 3.1 mm", "3-D note")}
    assert set(notes) == {"p95 3.1 mm", "3-D note"}
    a = notes["p95 3.1 mm"]
    assert a.xref == "x domain" and a.yref == "y domain" and a.xanchor == "left"
    assert a.yanchor == "top" and a.x < 0.1 and a.y > 0.9
    assert notes["3-D note"].xref == "paper"


# -- C3: colourbars beyond heatmaps --------------------------------------------------------
def _bars(fig):
    out = []
    for tr in fig.data:
        m = getattr(tr, "marker", None)
        if m is not None and m.showscale:
            out.append(tr)
        elif tr.type == "heatmap" and tr.showscale:
            out.append(tr)
    return out


@pytest.mark.parametrize("kind", ["colorline", "scatter", "image"])
def test_colourbar_for_colour_lines_scatters_and_maps(kind):
    x = np.linspace(0, 1, 50)
    if kind == "image":
        s = Series("image", img=np.outer(x[:5], x[:6]), extent=(0, 1, 0, 1),
                   style={"cmap": "sequential", "lim": (0.0, 1.0)})
    else:
        s = Series(kind, x, np.sin(x), c=x * 10)
    p = Panel("p", Axis("x"), Axis("y"), [s], colorbar=Colorbar("Speed", "\\milli\\metre"))
    fig = rp.to_plotly(_fig(p))
    bars = _bars(fig)
    assert len(bars) == 1
    cb = bars[0].marker.colorbar
    assert cb.title.text == "Speed (mm)" and cb.lenmode == "fraction" and 0 < cb.x < 1.05
    if kind != "image":
        assert (bars[0].marker.cmin, bars[0].marker.cmax) == (0.0, 10.0)
    assert not hasattr(fig, "_bm_size")


def test_heatmap_colourbar_still_on_the_trace():
    p = Panel("h", Axis(""), Axis(""), [Series("heatmap", z=np.arange(6.0).reshape(2, 3))],
              colorbar=Colorbar("v"))
    fig = rp.to_plotly(_fig(p))
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert heat and heat[0].showscale and heat[0].colorbar.title.text == "v"
    assert len(_bars(fig)) == 1


# -- C4: annotations on log axes -----------------------------------------------------------
def test_log_axis_annotations_use_log10():
    f = np.logspace(0, 3, 50)
    p = Panel("psd", Axis("Frequency", "\\hertz", log=True), Axis("PSD", log=True),
              [Series("line", f, 1 / f),
               Series("vline", x=np.array([50.0]), label="mains", role="reference"),
               Series("hline", y=np.array([0.01]), label="floor", role="reference"),
               Series("text", np.array([100.0]), np.array([0.1]), text=["here"])])
    fig = rp.to_plotly(_fig(p))
    by = {a.text: a for a in fig.layout.annotations}
    assert by["mains"].x == pytest.approx(np.log10(50.0))
    assert by["floor"].y == pytest.approx(np.log10(0.01))
    assert by["here"].x == pytest.approx(2.0) and by["here"].y == pytest.approx(-1.0)
    # Shapes take data values on log axes (plotly.js shapePositionToRange = d2r).
    xs = [s.x0 for s in fig.layout.shapes if s.yref == "y domain"]
    assert xs == [50.0]


# -- C5: diverging heatmaps ------------------------------------------------------------------
def test_mpl_heatmap_diverging_symmetric_and_contrast():
    z = np.array([[-1.0, 0.0, 3.0]])
    p = Panel("d", Axis(""), Axis(""), [Series("heatmap", z=z, text=["a", "b", "c"])],
              colorbar=Colorbar("Δ", ramp="diverging"))
    b = render_mpl.build(_fig(p))
    im = b.axes[0].images[0]
    assert im.get_clim() == (-3.0, 3.0)
    assert im.get_cmap().name == "jt-diverging"
    colors = {t.get_text(): t.get_color() for t in b.axes[0].texts}
    assert colors["c"] == "#FFFFFF" and colors["b"] == "#1F2328"   # pole dark, centre light
    assert render_mpl.heat_lim(z, None) == (-1.0, 3.0)


def test_pgf_diverging_colourbar_limits_and_colorline_meta(tmp_path: Path):
    z = np.array([[-1.0, 0.5], [2.0, 0.0]])
    p1 = Panel("d", Axis(""), Axis(""), [Series("heatmap", z=z)],
               colorbar=Colorbar("Δ", ramp="diverging"))
    x = np.linspace(0, 1, 30)
    p2 = Panel("c", Axis("x"), Axis("y"), [Series("colorline", x, x, c=np.full(30, 4.0))],
               colorbar=Colorbar("speed"))
    render_pgf.write_style(tmp_path)
    tex = render_pgf.render_figure(_fig(p1, p2), tmp_path).read_text()
    first = tex.split("\\begin{axis}")[1]
    assert first.count("point meta min=-2") == 2 and first.count("point meta max=2") == 2
    second = tex.split("\\begin{axis}")[2]
    lo = float(second.split("mesh, point meta=explicit")[1].split("point meta min=")[1]
               .split(",")[0])
    hi = float(second.split("point meta max=")[1].split("]")[0])
    assert lo < 4.0 < hi                    # never degenerate (pgfplots aborts at min == max)


def test_pgf_band_split_at_gaps(tmp_path: Path):
    x = np.arange(10.0)
    y = np.sin(x)
    y[4:6] = np.nan
    p = Panel("b", Axis("x"), Axis("y"), [Series("band", x, y - 0.1, y + 0.1, label="IQR"),
                                         Series("line", x, y, label="median")],
              legend="below")
    tex = render_pgf.render_figure(_fig(p), tmp_path).read_text()
    fills = [ln for ln in tex.splitlines() if "fill opacity" in ln]
    assert len(fills) == 2 and "forget plot" in fills[1] and "forget plot" not in fills[0]
    assert tex.count("\\addlegendentry{IQR}") == 1
    d0 = (tmp_path / "figures/t/p0s0.dat").read_text().split()
    assert "nan" not in d0


# -- C10: hover, legends, reflow -------------------------------------------------------------
def test_context_roles_skip_hover_and_legend_entrywidth():
    x = np.linspace(0, 1, 20)
    p = Panel("p", Axis("x"), Axis("y"), [
        Series("line", x, x, role="context", label="trials"),
        Series("line", x, x ** 2, role="faint"),
        Series("scatter", x, x, role="muted"),
        Series("line", x, x ** 3, label="mean")], legend="below")
    fig = rp.to_plotly(_fig(p))
    by = {(t.name, t.mode): t for t in fig.data}
    assert all(t.hoverinfo == "skip" for t in fig.data if t.line.color in (
        style.hexcolor("grey"), style.hexcolor("lightgrey")) or t.mode == "markers")
    assert by[("mean", "lines")].hovertemplate
    lg = fig.layout.legend
    assert lg.entrywidthmode == "pixels" and lg.entrywidth >= 20


def test_inside_legend_corner_honoured():
    x = np.linspace(0, 1, 20)
    p = Panel("p", Axis("x"), Axis("y"), [Series("line", x, x, label="a"),
                                         Series("line", x, -x, label="b")],
              legend="inside:south west")
    fig = rp.to_plotly(_fig(p))
    lg = fig.layout.legend
    assert lg.xanchor == "left" and lg.yanchor == "bottom" and lg.orientation == "v"
    assert lg.x == pytest.approx(fig.layout.xaxis.domain[0], abs=1e-6)
    assert lg.y == pytest.approx(fig.layout.yaxis.domain[0], abs=1e-6)


def test_reflow_restores_grid_trimmed_axis_titles():
    from biomech.viz.views.frequency import _grid_labels

    x = np.linspace(0, 1, 10)
    panels = [Panel(f"m{i}", Axis("Frequency", "\\hertz"), Axis("PSD"), [Series("line", x, x)])
              for i in range(6)]
    _grid_labels(panels, 3)
    spec = FigureSpec("g", "g", panels, rows=2, cols=3, width="text")
    assert panels[1].y.label == "" and panels[0].x.label == ""
    fig = rp.to_plotly(spec, viewport_px=420, fit=True)         # phone: one column
    ytitles = [fig.layout[f"yaxis{'' if k == 0 else k + 1}"].title.text for k in range(6)]
    xtitles = [fig.layout[f"xaxis{'' if k == 0 else k + 1}"].title.text for k in range(6)]
    assert all(t == "PSD" for t in ytitles) and all(t == "Frequency (Hz)" for t in xtitles)
    assert panels[1].y.label == ""                              # the cached spec untouched
    wide = rp.to_plotly(spec, viewport_px=1500, fit=True)
    assert wide.layout.yaxis2.title.text in (None, "")


# -- C11: colour lines as coloured segments -------------------------------------------------
def test_colorline_is_coloured_segments_not_grey_markers():
    t = np.linspace(0, 1, 400)
    p = Panel("p", Axis("x"), Axis("y"), [Series("colorline", np.cos(6 * t), np.sin(6 * t),
                                                 c=t)], colorbar=Colorbar("speed"))
    fig = rp.to_plotly(_fig(p))
    lines = [tr for tr in fig.data if tr.mode == "lines"]
    assert len(lines) > 5 and not any(tr.mode == "markers" and tr.marker.size == 4
                                      for tr in fig.data)
    colors = {tr.line.color for tr in lines}
    assert len(colors) == len(lines)                            # one colour per bin
    assert style.hexcolor("grey") not in colors
    n = sum(np.isfinite(np.asarray(tr.x, float)).sum() for tr in lines)
    assert 400 <= n <= 2 * 400                                 # bounded payload
    # Pieces join: the last point of one piece is the first of the next bin's piece.
    bins = rp.colorline_bins(np.arange(4.0), np.zeros(4), np.array([0, 0, 1, 1.0]), (0, 1))
    assert [b[0] for b in bins] == [0, rp.COLOR_BINS // 2, rp.COLOR_BINS - 1] or len(bins) >= 2
    assert bins[0][1][-1] == bins[1][1][0]


def test_constant_colorline_single_exact_colour():
    x = np.linspace(0, 1, 10)
    p = Panel("p", Axis("x"), Axis("y"), [Series("colorline", x, x, c=np.full(10, 2.0),
                                                 style={"lim": (0.0, 4.0)})])
    fig = rp.to_plotly(_fig(p))
    assert len(fig.data) == 1
    assert fig.data[0].line.color == style.ramp_color(0.5)


# -- C15: point budget -------------------------------------------------------------------------
def test_point_budget_keeps_extremes_and_gaps():
    n = 60_000
    x = np.arange(n, dtype=float)
    y = np.sin(x / 500.0)
    y[123] = 9.0                            # a spike
    y[30_000:30_010] = np.nan               # a gap
    p = Panel("p", Axis("x"), Axis("y"), [Series("line", x, y)])
    fig = rp.to_plotly(_fig(p))
    tx, ty = np.asarray(fig.data[0].x, float), np.asarray(fig.data[0].y, float)
    assert len(tx) <= rp.POINT_BUDGET * 1.3
    assert np.nanmax(ty) == 9.0 and tx[0] == 0 and tx[-1] == n - 1
    assert np.isnan(ty).sum() >= 1 and np.isnan(ty).sum() <= 2
    gap = np.flatnonzero(np.isnan(ty))[0]
    assert tx[gap - 1] < 30_000 and tx[gap + 1] >= 30_010


# -- C16: theme-aware ramps ---------------------------------------------------------------------
def test_dark_ramps_do_not_glare_and_keep_high_end_visible():
    def lum(h):
        r, g, b = (int(h.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    panel = lum("#" + style.THEMES["dark"]["panel"])
    seq = rp._colorscale("sequential", "dark")
    assert lum(seq[0][1]) < 0.35 and lum(seq[-1][1]) > 0.75 and lum(seq[0][1]) > panel
    div = rp._colorscale("diverging", "dark")
    mid = div[len(div) // 2][1]
    assert lum(mid) < 0.35 and lum(div[0][1]) > 0.5 and lum(div[-1][1]) > 0.5
    assert rp._colorscale("sequential", "light")[0][1] == "#" + style.SEQUENTIAL[0]
    z = np.array([[1.0, np.nan], [2.0, 3.0]])
    p = Panel("h", Axis(""), Axis(""), [Series("heatmap", z=z, text=["1", "", "2", "3"])])
    fig = rp.to_plotly(_fig(p), theme="dark")
    heat = [t for t in fig.data if t.type == "heatmap"][0]
    assert heat.colorscale[0][1] == seq[0][1]


# -- C14: matplotlib without pyplot, anim preview frames ----------------------------------
def test_render_mpl_without_pyplot_is_thread_safe(tmp_path: Path):
    import matplotlib.pyplot as plt

    x = np.linspace(0, 1, 100)
    spec = _fig(Panel("p", Axis("x"), Axis("y"), [Series("line", x, x, label="a")]))
    before = plt.get_fignums()
    b = render_mpl.build(spec)
    assert plt.get_fignums() == before and b.fig.canvas is not None
    out, errs = [], []

    def work(i):
        try:
            out.append(render_mpl.render_figure(spec, tmp_path / f"f{i}.png", dpi=80))
        except Exception as exc:              # pragma: no cover - reported below
            errs.append(exc)

    ts = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errs and len(out) == 6
    sizes = {p.read_bytes() for p in out}
    assert len(sizes) == 1                                   # identical output


def test_anim_preview_trace_count_matches_frames_for_constant_colorline():
    from biomech.anim import render_plotly as arp
    from biomech.viz import layout as L

    x = np.linspace(0, 1, 30)
    const = Series("colorline", x, x, c=np.full(30, 1.0), style={"lim": (0.0, 2.0)})
    vary = Series("colorline", x, x, c=x * 2, style={"lim": (0.0, 2.0)})
    spec = _fig(Panel("p", Axis("x"), Axis("y"), [const]))
    pl = L.compute(spec).panels[0]
    traces = arp._traces(const, 0, pl, "dark")
    assert len(traces) == len(arp._frame_data(const, pl)) == len(arp._frame_data(vary, pl))


def test_scalar_image_nan_transparent_in_all_renderers(tmp_path: Path):
    img = np.array([[1.0, np.nan], [0.5, 0.0]])
    s = Series("image", img=img, extent=(0, 2, 0, 2), style={"cmap": "sequential",
                                                              "lim": (0.0, 1.0),
                                                              "mask": np.array([[0, 0],
                                                                                [0, 1]], bool)})
    rgba = render_mpl.scalar_rgba(s)
    assert rgba[0, 1, 3] == 0.0 and rgba[0, 0, 3] == 1.0
    assert tuple(np.round(rgba[1, 1, :3] * 255).astype(int)) == tuple(
        int(style.THEMES["light"]["muted"][i:i + 2], 16) for i in (0, 2, 4))
    p = Panel("m", Axis("x"), Axis("y"), [s], colorbar=Colorbar("share", "\\percent"))
    fig = rp.to_plotly(_fig(p))
    heats = [t for t in fig.data if t.type == "heatmap"]
    assert len(heats) == 2 and not any(t.type == "image" for t in fig.data)
    main = heats[-1]
    assert np.isnan(np.asarray(main.z, float)).sum() == 1
    assert list(main.y) == sorted(main.y)                    # increasing y, rows flipped
    assert np.asarray(main.z, float)[0, 0] == 0.5            # bottom row first
    render_pgf.render_figure(_fig(p), tmp_path)
    assert (tmp_path / "figures/t/p0s0.png").is_file()
