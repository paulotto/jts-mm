# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Layout engine and renderers: overlap-free by construction, and checked on real renders.

* unit tests of the engine (ticks, wrapping, legends, rotation, equal scales);
* every registered paper/dashboard view x perspective on the synthetic study:
  no overlapping or clipped text in matplotlib, a clean layout check, one Plotly
  legend per panel that needs one, and (with LuaLaTeX) each paper figure compiled
  alone fits its IEEE width with no overfull boxes or pgfplots warnings;
* zoom export and LaTeX escaping regressions (audit #19, #24).
"""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from biomech.viz import layout as L
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import apply_zoom, merge_relayout, to_plotly
from biomech.viz.spec import Axis, FigureSpec, Panel, Series


# -- engine ----------------------------------------------------------------------------
def test_log_ticks_are_plain_numbers():
    ticks, labels = L.log_ticks(0.05, 8.0)
    assert labels == ["0.1", "0.3", "1", "3"]
    _, labels = L.log_ticks(0.3, 2.2)
    assert labels == ["0.5", "1", "2"]
    assert all("e" not in s and "×" not in s for s in L.log_ticks(1e-3, 1e3)[1])


def test_linear_ticks_and_decimals():
    ticks, dec = L.linear_ticks(-3.2, 47.1, 6)
    assert ticks == [0, 10, 20, 30, 40] and dec == 0
    assert L._decimals(2.5) == 1 and L._decimals(0.25) == 2 and L._decimals(5) == 0
    assert L.fmt_tick(-0.0, 1) == "0.0" and L.fmt_tick(-2, 0) == "−2"


def test_wrap_keeps_titles_inside_their_cell():
    lines = L.wrap("(a) Processing spread / between-subject SD", 1.5, 8)
    assert len(lines) > 1 and all(L.text_width_in(t, 8) <= 1.5 for t in lines)
    assert "/ between" in " ".join(lines) or lines[0].endswith("/")
    assert L.wrap("(b) Between/within-subject variance", 1.2, 8)[0] == "(b) Between/"


def _cat_panel(n: int, long: bool) -> Panel:
    names = [("a very long category name " if long else "c") + str(i) for i in range(n)]
    p = Panel("t", Axis(""), Axis("value"), [Series("bar", np.arange(n), np.arange(n) + 1.0)])
    p.x.ticks, p.x.ticklabels = list(range(n)), names
    return p


def test_crowded_categories_rotate_and_reserve_space():
    lay = L.compute(FigureSpec("f", "", [_cat_panel(12, True)], width="column"))
    pl = lay.panels[0]
    assert pl.x.rotate == 90 and pl.xticks_band.h > 0.5
    assert not L.check(lay)
    lay2 = L.compute(FigureSpec("f", "", [_cat_panel(3, False)], width="column"))
    assert lay2.panels[0].x.rotate == 0


def test_legend_columns_fit_the_panel():
    p = Panel("t", Axis("x"), Axis("y"), [Series("line", np.arange(3.0), np.arange(3.0),
                                                 label=f"series number {i}") for i in range(5)])
    lay = L.compute(FigureSpec("f", "", [p], width="column"))
    pl = lay.panels[0]
    assert pl.mode_legend == "below" and pl.legend is not None
    assert pl.legend.x1 <= lay.width + 1e-6 and 1 <= pl.legend_cols <= 5
    one = Panel("t", Axis("x"), Axis("y"), [Series("line", np.arange(3.0), np.arange(3.0),
                                                   label="only")])
    assert L.compute(FigureSpec("g", "", [one])).panels[0].mode_legend == "none"


def test_shared_equal_scale():
    a = Panel("sag", Axis("x"), Axis("z"), [Series("line", np.array([0, 40.0]),
                                                   np.array([0, 20.0]))], equal=True)
    b = Panel("front", Axis("y"), Axis("z"), [Series("line", np.array([0, 10.0]),
                                                     np.array([0, 20.0]))], equal=True)
    spec = FigureSpec("o", "", [a, b], cols=2, width="text", meta={"equal_scale": "shared"})
    lay = L.compute(spec)
    s = [pl.axis.w / (pl.x.lim[1] - pl.x.lim[0]) for pl in lay.panels]
    s += [pl.axis.h / (pl.y.lim[1] - pl.y.lim[0]) for pl in lay.panels]
    assert np.allclose(s, s[0])


def test_mpl_matches_layout_boxes():
    p = Panel("t", Axis("time", "\\second"), Axis("y"), [Series("line", np.arange(10.0),
                                                                np.arange(10.0))])
    b = render_mpl.build(FigureSpec("f", "", [p, p], cols=2, width="text"))
    W, H = b.layout.width, b.layout.height
    for ax, pl in zip(b.axes, b.layout.panels):
        x0, y0, w, h = ax.get_position().bounds
        assert x0 * W == pytest.approx(pl.axis.x0, abs=1e-6)
        assert w * W == pytest.approx(pl.axis.w, abs=1e-6)
        assert (1 - y0 - h) * H == pytest.approx(pl.axis.y0, abs=1e-6)


# -- regressions -------------------------------------------------------------------------
def test_apply_zoom_handles_autorange_lists_and_reversal():
    p = Panel("t", Axis("x"), Axis("y"), [Series("line", np.arange(5.0), np.arange(5.0))])
    q = Panel("u", Axis("x", log=True), Axis("y"), [Series("line", np.arange(1, 5.0),
                                                           np.arange(4.0))])
    spec = FigureSpec("z", "", [p, q], cols=2)
    z = merge_relayout({}, {"xaxis.range[0]": 1, "xaxis.range[1]": 3})
    z = merge_relayout(z, {"xaxis.autorange": True})
    z = merge_relayout(z, {"xaxis.range": [2, 4], "yaxis.range": [5, 1], "xaxis2.range[0]": 0,
                           "xaxis2.range[1]": 1})
    out = apply_zoom(spec, z)
    assert out.panels[0].x.lim == (2.0, 4.0) and out.panels[0].y.lim == (1.0, 5.0)
    assert out.panels[1].x.lim == pytest.approx((1.0, 10.0))
    assert apply_zoom(spec, {"xaxis.autorange": True}).panels[0].x.lim is None


def test_tex_escape_keeps_math():
    assert render_pgf._esc("$\\theta_y$ > 3 and 5 %") == "$\\theta_y$ $>$ 3 and 5 \\%"
    assert render_pgf._esc("a_b ≤ c") == "a\\_b \\jtsmmsym{≤}{\\leq} c"


# -- every registered view on the synthetic study -------------------------------------------
def _cases():
    from biomech.viz.views import registry as R

    out = []
    # the synthetic study is a dataset study; the validation views have their own synthetic
    # study (tests/test_v_*.py)
    for v in R.views(groups=("dataset",)):
        if v.kind not in ("paper", "dashboard"):
            continue
        for persp in (v.perspectives or (None,)):
            out.append((v.id, persp))
    return out


@pytest.mark.slow
@pytest.mark.parametrize("view,persp", _cases())
def test_views_render_cleanly(synth_study, view, persp):
    from biomech.io.context import DataContext
    from biomech.viz.views import registry as R

    cfg = synth_study[0]
    ctx = DataContext(cfg)
    pid = ctx.pids()[1] if persp not in (None, "cohort") else None
    spec = R.build(view, R.request(ctx, view, perspective=persp, pid=pid))
    if not isinstance(spec, FigureSpec):
        assert spec.rows
        return
    b = render_mpl.build(spec)
    try:
        assert render_mpl.text_overlaps(b) == []
        assert L.check(b.layout) == []
    finally:
        import matplotlib.pyplot as plt

        plt.close(b.fig)
    for theme in ("light", "dark"):
        fig = to_plotly(spec, theme=theme)
        legends = [k for k in fig.layout.to_plotly_json() if k.startswith("legend")]
        need = [pl for pl in L.compute(spec).panels if pl.mode_legend != "none"]
        assert len(legends) >= len(need)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lualatex") is None, reason="no LuaLaTeX")
def test_paper_figures_fit_and_compile(synth_study, tmp_path):
    from biomech.io.context import DataContext
    from biomech.viz.export import export

    res = export(DataContext(synth_study[0]), None, tmp_path, previews=False, compile_pdf=True,
                 check_size=True)
    assert res["errors"] == [] and res["problems"] == [], res
    assert (tmp_path / "preview.pdf").is_file()


def test_dark_theme_uses_validated_steps():
    from biomech.viz import style

    for name in style.CATEGORICAL:
        assert style.hexcolor(name, "dark") == "#" + style.PALETTE_DARK[name]
        assert style.hexcolor(name, "light") == "#" + style.PALETTE[name]


@pytest.mark.skipif(shutil.which("lualatex") is None, reason="no LuaLaTeX")
def test_wrapped_title_in_last_column_stays_inside_the_figure(tmp_path):
    from biomech.viz.export import check_figure_size

    x = np.arange(5.0)
    panels = [Panel(t, Axis("x"), Axis("y"), [Series("line", x, x)]) for t in
              ("(a) Short", "(b) Short", "(c) A title long enough to wrap onto a second line")]
    spec = FigureSpec("wrap_last", "wrap", panels, rows=1, cols=3, width="text", height_in=1.4)
    render_pgf.write_style(tmp_path)
    render_pgf.render_figure(spec, tmp_path)
    assert not [p for p in check_figure_size(tmp_path, spec) if "wide" in p]


def test_wrapping_table_columns_share_the_width(tmp_path):
    from biomech.viz.spec import Column, TableSpec

    spec = TableSpec("t_wrap", "wrap", [Column("a", "A", align="l"), Column("b", "B", align="X"),
                                        Column("c", "C", align="X")],
                     [{"a": "x", "b": "long text " * 20, "c": "more text " * 10}], width="text")
    tex = render_pgf.render_table(spec, tmp_path).read_text()
    assert "\\begin{tabularx}{\\textwidth}" in tex and tex.count("arraybackslash}X") == 2
    assert "p{2cm}" not in tex and "\\end{tabularx}" in tex
    plain = TableSpec("t_plain", "plain", [Column("a", "A", align="l")], [{"a": "x"}])
    assert "tabular*" in render_pgf.render_table(plain, tmp_path).read_text()


def test_pgf_thinning_keeps_gaps_and_run_ends():
    from biomech.viz import render_pgf as RP

    x = np.arange(20000, dtype=float)
    y = np.sin(x / 50.0)
    y[7000:7100] = np.nan                       # a gap
    keep = RP._thin(x, y)
    assert len(keep) < 6000
    assert {0, 6999, 7100, 19999} <= set(keep.tolist())      # run ends kept
    assert np.isnan(y[keep]).any()                            # the gap stays a gap
    k = keep[np.isfinite(y[keep])]
    # no kept finite point inside the gap and no segment across it (a NaN separates them)
    left = keep[keep < 7000].max()
    right = keep[keep >= 7100].min()
    between = keep[(keep > left) & (keep < right)]
    assert between.size and np.isnan(y[between]).all() and k.size > 100
