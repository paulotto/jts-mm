# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Style consistency (plan D2): one meaning per colour, distinct markers, panel tags, the
U+2212 minus, reader wording instead of internal plan codes, and tick-label bands that hold
their longest label in every renderer (viewer regression: "hinge through palpated condyles"
clipped at the left edge of fig_kinematics c)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

from biomech.viz import layout as L
from biomech.viz import render_mpl, render_pgf, style
from biomech.viz.render_plotly import VIEW_FONT, to_plotly
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec
from biomech.viz.views import vocab as V


# -- colour and marker roles -------------------------------------------------------------------
def _col(r: str) -> str:
    return style.role(r)[0]


def test_one_meaning_per_colour():
    hues = set(style.CATEGORICAL)
    assert _col("accent") not in hues and _col("band") == _col("accent")   # cohort summary
    assert _col("group:openers") == style.ENTITY["digastric_r"]
    assert _col("group:closers") == style.ENTITY["masseter_r"]
    groups = {_col("group:openers"), _col("group:closers")}
    for r in style.ROLES:
        if r.startswith(("set:", "protocol:")):
            assert _col(r) not in groups, r
    for r in ("reference", "threshold"):
        col, dash, _ = style.role(r)
        assert col == "ink2" and dash == "dashed"
    for m in V.MUSCLES:                       # muscles keep their entity colour
        assert _col(f"entity:{m}") == style.ENTITY[m]


@pytest.mark.parametrize("family", ["set:", "protocol:", "model:", "cat:", "sex:", "side:"])
def test_markers_and_colours_distinct_within_a_family(family):
    looks = [(style.role(r)[0], style.role(r)[1], style.role(r)[2]) for r in style.ROLES
             if r.startswith(family)]
    assert len(set(looks)) == len(looks)


def test_down_triangle_is_not_an_up_triangle(tmp_path):
    sty = render_pgf.write_style(tmp_path).read_text()
    line = next(ln for ln in sty.splitlines() if "jtsmm/mark/setemgnodigimu/" in ln)
    assert "triangle*" in line and "rotate=180" in line
    up = next(ln for ln in sty.splitlines() if "jtsmm/mark/setemgimu/" in ln)
    assert "rotate" not in up
    assert "rotate=180" in render_pgf._mark("v") and "rotate" not in render_pgf._mark("^")


def test_legend_conflicts_are_reported():
    from biomech.check.outputs import legend_conflicts

    x = np.arange(3.0)
    bad = Panel("(a) t", Axis("x"), Axis("y"), [Series("line", x, x, label="A", role="cat:0"),
                                                Series("line", x, x, label="B", role="primary")])
    ok = Panel("(b) t", Axis("x"), Axis("y"), [Series("line", x, x, label="A", role="cat:0"),
                                               Series("line", x, x, label="B", role="cat:1"),
                                               Series("line", x, x, role="context"),
                                               Series("line", x, x, role="context")])
    out = legend_conflicts(FigureSpec("f", "", [bad, ok], cols=2))
    assert len(out) == 1 and "panel a" in out[0]


# -- panel tags ----------------------------------------------------------------------------------
def _two_panels() -> FigureSpec:
    x = np.linspace(0, 10, 20)
    return FigureSpec("tags", "", [
        Panel("(a) Opening angle", Axis("Cycle", "\\percent"), Axis("Angle", "\\degree"),
              [Series("line", x, x)]),
        Panel("Second", Axis("Cycle", "\\percent"), Axis("Angle", "\\degree"),
              [Series("line", x, -x)], tag="b")], cols=2, width="text")


def test_tag_derived_from_title_prefix():
    p = Panel("(c) Misalignment at the incisor")
    assert p.heading() == ("c", "Misalignment at the incisor") and p.tag == ""
    assert p.title == "(c) Misalignment at the incisor"          # the spec is not rewritten
    assert Panel("Plain").heading() == ("", "Plain")
    assert Panel("(a) x", tag="z").heading() == ("z", "x")


def test_tags_sit_top_left_outside_the_axis():
    lay = L.compute(_two_panels())
    assert L.check(lay) == []
    for pl, t in zip(lay.panels, "ab"):
        assert pl.tag_text == t and pl.title_lines[0] in ("Opening angle", "Second")
        assert pl.tag.x1 <= pl.axis.x0 and pl.tag.x1 <= pl.title.x0     # left of axis/title
        assert pl.tag.y1 <= pl.axis.y0 + 1e-9                           # above the axis
    a, b = lay.panels
    assert a.tag.y0 == pytest.approx(b.tag.y0) and a.tag.x0 == pytest.approx(0.0)
    for pl in lay.panels:                                   # on the first title line
        assert pl.tag.y0 == pytest.approx(pl.title.y0)


def test_tags_in_every_renderer(tmp_path):
    spec = _two_panels()
    b = render_mpl.build(spec)
    tags = [t for t in b.fig.texts if t.get_gid() == "tag"]
    assert [t.get_text() for t in tags] == ["a", "b"]
    assert all(t.get_fontweight() in ("bold", 700) for t in tags)
    assert render_mpl.text_overlaps(b) == []
    render_pgf.write_style(tmp_path)
    tex = render_pgf.render_figure(spec, tmp_path).read_text()
    assert tex.count("\\jtsmmfonttag") == 2 and "(a)" not in tex and "title=" not in tex
    ann = [a.text for a in to_plotly(spec).layout.annotations]
    assert "<b>a</b>" in ann and "<b>b</b>" in ann and "Opening angle" in ann


def test_auto_tags_for_unlettered_figures():
    from biomech.viz.export import auto_tags

    x = np.arange(3.0)
    spec = FigureSpec("f", "", [Panel("A", series=[Series("line", x, x)]),
                                Panel("B", series=[Series("line", x, x)])], cols=2)
    assert [p.heading()[0] for p in auto_tags(spec).panels] == ["a", "b"]
    one = FigureSpec("g", "", [Panel("A", series=[Series("line", x, x)])])
    assert auto_tags(one).panels[0].heading()[0] == ""
    lettered = _two_panels()
    assert [p.heading()[0] for p in auto_tags(lettered).panels] == ["a", "b"]


# -- minus sign ------------------------------------------------------------------------------------
def test_unicode_minus_everywhere(tmp_path):
    m = style.MINUS
    assert style.unicode_minus("bias -0.4") == f"bias {m}0.4"
    assert style.unicode_minus("LoA [-2.9, 2.1]") == f"LoA [{m}2.9, 2.1]"
    for keep in ("10-20", "open-close", "1e-3", "s-01"):
        assert style.unicode_minus(keep) == keep
    assert L.fmt_tick(-3, 0) == f"{m}3" and L.plain_text("$-3$ mm") == f"{m}3 mm"
    assert "\\jtsmmminus" in render_pgf._esc("bias -0.4") and "-" not in render_pgf._esc("-1")
    sty = render_pgf.write_style(tmp_path).read_text()
    assert "\\def\\jtsmmminus{\\ifmmode-\\else\\textminus\\fi}" in sty
    # table cells (text columns); numeric S columns are typeset by siunitx with a real minus
    t = TableSpec("t_minus", "c", [Column("a", "A", align="l")], [{"a": "bias -1.5 mm"}])
    assert "{\\jtsmmminus}1.5" in render_pgf.render_table(t, tmp_path).read_text()
    # matplotlib: labels and in-panel text
    x = np.arange(3.0)
    p = Panel("(a) t", Axis("x"), Axis("y"), [Series("line", x, -x),
                                              Series("hline", y=np.array([-1.0]),
                                                     role="reference", text=["bias -1.0"])])
    b = render_mpl.build(FigureSpec("m", "", [p]))
    texts = [t.get_text() for ax in b.axes for t in ax.texts]
    assert any(f"{m}1.0" in t for t in texts)
    assert all("-" not in t.get_text() for ax in b.axes for t in ax.get_yticklabels())


# -- reader wording ---------------------------------------------------------------------------------
def test_internal_codes_become_reader_wording():
    R = V.reader_text
    assert R("the driving burst (deviation D6)") == "the driving burst (post hoc)"
    assert R("ensemble events (exploratory X2)") == "ensemble events (exploratory)"
    assert R("X2 envelopes, X3 detector (exploratory)") == "envelopes, detector (exploratory)"
    assert R("C1") == "pre-specified" and R("see erratum E17") == "see erratum"
    assert R("P3 at P95, 5 ms") == "P3 at P95, 5 ms"           # pseudonyms, percentiles
    assert V.internal_codes("D6 and X2, C3a; E17 not P95") == ["D6", "X2", "C3a", "E17"]


def test_reader_spec_copies_and_maps(tmp_path):
    from biomech.check.outputs import exported_code_leaks
    from biomech.viz import serialize

    x = np.arange(3.0)
    spec = FigureSpec("codes", "", [Panel("(a) t", note="X2", series=[Series(
        "line", x, x, label="rule D6")])], caption="Onset (deviation D6), cue (X2).")
    out = V.reader_spec(spec)
    assert spec.caption.endswith("(X2).") and out.caption == "Onset (post hoc), cue (exploratory)."
    assert out.panels[0].note == "exploratory" and out.panels[0].series[0].label == "rule post hoc"
    tab = TableSpec("tab", "T", [Column("b", "Basis", align="l")], [{"b": "C2, TKEO"}],
                    notes="X1 bandwidth")
    tout = V.reader_spec(tab)
    assert tout.rows[0]["b"] == "pre-specified, TKEO" and tout.notes == "exploratory bandwidth"
    # the check lists exported items that still carry codes
    serialize.write_figure(spec, tmp_path / "figures" / spec.id)
    serialize.write_figure(out, tmp_path / "figures" / "clean")
    serialize.write_table(tab, tmp_path)
    leaks = exported_code_leaks(tmp_path)
    assert set(leaks) == {"codes", "tab"} and "D6" in leaks["codes"]


# -- tick-label bands hold the longest label (viewer clipping regression) ---------------------------
LONG = "hinge through palpated condyles"


def _misalignment() -> FigureSpec:
    x = np.arange(5.0)
    a = Panel("(a) Path", Axis("Anterior", "\\milli\\metre"), Axis("Superior", "\\milli\\metre"),
              [Series("line", x, x)])
    c = Panel("(c) Misalignment at the incisor", Axis("RMS residual", "\\milli\\metre", log=True),
              Axis(""), [Series("scatter", np.array([0.5, 2.0, 6.0]), np.array([0.0, 1.0, 2.0]))])
    c.y.ticks, c.y.ticklabels = [0, 1, 2], [LONG, "best-fit hinge", "hinge + translation"]
    c.y.reverse = True
    # As fig_kinematics: 2 x 2, the long labels in the first column (left figure edge).
    return FigureSpec("misalign", "", [a, a, c, a], rows=2, cols=2, width="text",
                      height_in=1.6)


def _real_width_px(text: str, px: float, family: str) -> float:
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath

    w, _, _ = TextToPath().get_text_width_height_descent(
        text, FontProperties(family=family, size=px), ismath=False)
    return w


@pytest.mark.parametrize("width_in", [7.0, 11.0, 14.9])
def test_plotly_y_tick_band_holds_the_longest_label(width_in):
    spec = _misalignment()
    fig = to_plotly(spec, width_in=width_in)
    lay = fig.layout
    assert (lay.margin.l, lay.margin.r) == (0, 0)             # paper fractions = layout's
    assert lay.yaxis3.automargin is True                      # narrower card: grow, not clip
    W = L.compute(spec, width_in, font=VIEW_FONT, glyph=1.15).width
    band_px = lay.xaxis3.domain[0] * W * 96.0                 # figure edge .. panel c's axis
    size = VIEW_FONT["tick"] * 1.333
    # The widest substitute a browser may use for Inter (DejaVu Sans) + tick length and gap.
    need = _real_width_px(LONG, size, "DejaVu Sans") + 3 + 2
    assert band_px >= need, (band_px, need)


def test_mpl_and_pgf_y_tick_band_holds_the_longest_label(tmp_path):
    spec = _misalignment()
    b = render_mpl.build(spec)
    assert render_mpl.text_overlaps(b) == []                  # includes "outside the figure"
    lab = next(t for t in b.axes[2].get_yticklabels() if t.get_text() == LONG)
    b.fig.canvas.draw()
    bb = lab.get_window_extent(b.fig.canvas.get_renderer())
    ax0 = b.axes[2].get_window_extent(b.fig.canvas.get_renderer()).x0
    assert bb.x0 >= 0 and bb.x1 <= ax0
    if shutil.which("lualatex") is None:
        return
    from biomech.viz.export import check_figure_size

    render_pgf.write_style(tmp_path)
    render_pgf.render_figure(spec, tmp_path)
    assert check_figure_size(tmp_path, spec) == []
