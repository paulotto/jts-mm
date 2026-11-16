# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Journal figure set: the new building blocks (ECDF with a marked percentile,
slopegraph, paired forest, specification curve with its indicator matrix, strip with a marked
statistic) and, on the synthetic study, every ``jnl_*`` view in all renderers for the IEEE and
Springer venues."""

from __future__ import annotations

import numpy as np
import pytest

from biomech.viz import layout as L
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import to_plotly
from biomech.viz.spec import Axis, FigureSpec
from biomech.viz.views import blocks as B

VENUES = ("ieee", "springer")


def _clean(spec: FigureSpec, venue: str) -> None:
    b = render_mpl.build(spec, venue=venue)
    try:
        assert render_mpl.text_overlaps(b) == []
        assert L.check(b.layout) == []
    finally:
        b.fig.clf()


# -- ECDF ------------------------------------------------------------------------------------------
def test_ecdf_steps_run_from_zero_to_hundred():
    x, y = B.ecdf_steps([3.0, 1.0, np.nan, 2.0])
    assert list(x) == [1, 1, 2, 2, 3, 3]
    assert np.allclose(y, [0, 100 / 3, 100 / 3, 200 / 3, 200 / 3, 100])
    assert B.ecdf_steps([np.nan])[0].size == 0


def test_ecdf_marks_percentile_and_ci():
    v = np.linspace(40, 130, 12)
    p = B.ecdf(v, "(b) Velocity", Axis("Peak velocity", "\\degree\\per\\second"),
               mark={"p": 95, "value": 126.0, "lo": 110.0, "hi": 130.0, "note": "P95 126"})
    kinds = [s.kind for s in p.series]
    assert kinds.count("line") == 1 and "hspan" in kinds and "vline" in kinds
    span = next(s for s in p.series if s.kind == "hspan")
    assert span.x[0] == 110 and span.y2[0] == 130
    diamond = next(s for s in p.series if s.kind == "scatter" and s.style.get("ms") == "d")
    assert diamond.x[0] == 126 and diamond.y[0] == 95 and p.note == "P95 126"
    # several curves: one role each, no points when asked
    q = B.ecdf({"A": v, "B": v + 5}, "t", Axis("x", "\\milli\\second"),
               roles={"A": "set:emg_imu", "B": "set:imu"}, points=False)
    assert [s.role for s in q.series] == ["set:emg_imu", "set:imu"]
    assert q.legend_mode() == "below"
    assert B.ecdf([], "t", Axis("x")).note == "no data"


# -- slopegraph ------------------------------------------------------------------------------------
def test_slopegraph_pairs_lone_values_and_summary():
    a = np.array([100.0, 50.0, np.nan, 20.0])
    b = np.array([40.0, np.nan, 10.0, -5.0])
    p = B.slopegraph(["first onset", "driving burst"], a, b, "(b) Lead", Axis("Lead", "\\milli\\second"),
                     summary=[(60.0, 20.0, 100.0), (10.0, -5.0, 40.0)],
                     annot=[["n = 3", "p = 0.02"], ["n = 3", "p = 0.2"]], ref=0.0)
    lines = [s for s in p.series if s.kind == "line" and s.role == "context"]
    assert len(lines) == 2                                   # only complete pairs are joined
    lone = next(s for s in p.series if s.role == "faint")
    assert sorted(lone.y.tolist()) == [10.0, 50.0] and sorted(lone.x.tolist()) == [0.0, 1.0]
    diamonds = [s for s in p.series if s.kind == "scatter" and s.role == "accent"]
    assert [d.x[0] for d in diamonds] == [-0.2, 1.2] and diamonds[0].label
    texts = [s for s in p.series if s.kind == "text"]
    assert len(texts) == 4 and all(t.y[0] > 100 for t in texts)
    first_col = sorted((t for t in texts if t.x[0] == 0), key=lambda t: -t.y[0])
    assert [t.text[0] for t in first_col] == ["n = 3", "p = 0.02"]     # first line on top
    assert p.x.ticklabels == ["first onset", "driving burst"]


# -- paired forest, strip with a mark --------------------------------------------------------------
def test_paired_forest_offsets_and_legend():
    p = B.paired_forest(["x", "y"], ([1.0, 2.0], [0.5, np.nan], [1.5, np.nan]),
                        ([1.2, np.nan], [0.8, 1.0], [1.6, 3.0]), "t", Axis("v", "\\milli\\metre"),
                        label_a="primary", label_b="variant")
    sc = [s for s in p.series if s.kind == "scatter"]
    assert [s.label for s in sc] == ["primary", "variant"]
    assert np.allclose(sc[0].y, [-0.17, 0.83]) and np.allclose(sc[1].y, [0.17])
    assert p.legend_mode() == "below"


def test_strip_mark_draws_marks_and_flagged():
    p = B.strip_mark(["a", "b"], [[1.0, 2.0, 3.0], [4.0]], "t", Axis("v", "\\hertz"),
                     marks=[(np.nan,) * 3, (5.0, 4.0, 6.0)], flagged=[[], [9.0]],
                     flagged_label="noise-limited")
    labels = [s.label for s in p.series if s.label]
    assert labels == ["participants", "noise-limited", "P95 [95 % CI]"]
    bar = next(s for s in p.series if s.style.get("ms") == "|")
    assert bar.x[0] == 5.0 and bar.y[0] == 1.0
    assert p.y.lim == (-0.5, 1.5) and p.y.reverse


# -- specification curve -----------------------------------------------------------------------------
def _specs():
    rng = np.random.default_rng(0)
    specs, est = [], []
    for d in ("A", "B", "C"):
        for r in ("first", "drive"):
            specs.append({"detector": d, "rule": r})
            est.append(rng.normal(50, 40))
    est[2] = np.nan                                           # a missing estimate is dropped
    est = np.array(est)
    return specs, est, est - 30, est + 30


def test_spec_curve_sorts_and_builds_the_matrix():
    specs, est, lo, hi = _specs()
    dims = [("detector", ["A", "B", "C"]), ("rule", ["first", "drive"])]
    curve, mat = B.spec_curve(est, lo, hi, specs, "(c) Lead", Axis("Lead", "\\milli\\second"),
                              dims=dims, highlight={0: ("pre-specified", "d")}, key="k")
    pts = np.concatenate([s.y for s in curve.series if s.kind == "scatter"])
    xs = np.concatenate([s.x for s in curve.series if s.kind == "scatter"])
    order = np.argsort(xs)
    assert np.all(np.diff(pts[order]) >= 0) and len(pts) == 5      # ranked, NaN dropped
    hl = next(s for s in curve.series if s.label == "pre-specified")
    assert hl.y[0] == pytest.approx(est[0]) and hl.role == "accent"
    on = next(s for s in mat.series if s.kind == "scatter" and s.role == "accent")
    off = next(s for s in mat.series if s.kind == "scatter" and s.role == "faint")
    assert len(on.x) == 5 * 2 and len(off.x) == 5 * 3             # one level per choice
    assert mat.y.ticklabels[0] == "detector: A" and mat.y.ticklabels[-1] == "rule: drive"
    assert curve.sharex == mat.sharex == "k" and curve.x.lim == mat.x.lim
    seps = [s for s in mat.series if s.kind == "hline"]
    assert len(seps) == 1 and seps[0].y[0] == 2.5


@pytest.mark.parametrize("venue", VENUES)
def test_blocks_render_cleanly(tmp_path, venue):
    specs, est, lo, hi = _specs()
    curve, mat = B.spec_curve(est, lo, hi, specs, "(c) Specification curve",
                              Axis("Lead", "\\milli\\second"),
                              dims=[("detector", ["A", "B", "C"]), ("rule", ["first", "drive"])],
                              highlight={0: ("pre-specified", "d"), 1: ("post hoc", "s")})
    curve.legend = "inside:north west"
    v = np.linspace(40, 130, 12)
    pe = B.ecdf(v, "(a) Velocity", Axis("Velocity", "\\degree\\per\\second"),
                mark={"p": 95, "value": 126.0, "lo": 110.0, "hi": 130.0, "note": "P95 126"})
    ps = B.slopegraph(["first", "drive"], v[:6], v[6:], "(b) Lead",
                      Axis("Lead", "\\milli\\second"), summary=[(80, 60, 100), (110, 90, 125)],
                      annot=[["n = 6"], ["n = 6"]], ref=0.0)
    spec = FigureSpec("jnl_blocks", "blocks", [pe, curve, ps, mat], rows=2, cols=2,
                      width="text", height_in=1.4)
    _clean(spec, venue)
    to_plotly(spec, theme="dark")
    render_pgf.write_style(tmp_path, venue)
    tex = render_pgf.render_figure(spec, tmp_path, venue=venue).read_text()
    assert f"venue: {venue}" in tex


def test_number_helpers():
    from biomech.viz.views import paper_journal as J

    assert J._n(-12.4) == "−12" and J._peq(0.0004) == "< 0.001" and J._peq(0.2) == "= 0.200"


def test_few_shot_reads_the_stored_tests():
    """The journal view computes no test: p values come from ml_eval.tests (R5 step 8)."""
    import pandas as pd

    from biomech.viz.views import paper_journal as J

    rows = [{"variant": "few_shot", "feature_set": fs, "k": k, "pid": f"s{i:02d}",
             "balanced_accuracy": 0.5 + (0.01 * i if fs == "emg_imu" else 0.0)}
            for i in range(1, 8) for k in (0, 50) for fs in ("imu", "emg_imu")]
    tests = pd.DataFrame([{"test": "wilcoxon|svm|emg_imu_vs_imu", "p": 0.2, "p_holm": 0.4},
                          {"test": "fewshot|svm|emg_imu_vs_imu|k=0", "stat": 0.04, "p": 0.03,
                           "n": 7, "n_better": 7, "p_holm": 0.06, "n_amounts": 2},
                          {"test": "fewshot|svm|emg_imu_vs_imu|k=50", "stat": 0.04,
                           "p": 0.016, "n": 7, "n_better": 6, "p_holm": 0.06,
                           "n_amounts": 2}])
    tables = {"variants": pd.DataFrame(rows), "tests": tests}

    class Ctx:
        pass

    orig = J._t
    try:
        J._t = lambda ctx, stage, name: tables.get(name, pd.DataFrame())
        fsh = J._few_shot(Ctx())
        assert sorted(fsh) == [0, 50]
        assert fsh[50]["p"] == 0.016 and fsh[50]["p_holm"] == 0.06
        assert fsh[50]["better"] == 6 and fsh[50]["n"] == 7       # stored, not recomputed
        assert J._n_amounts(fsh) == 2
        assert np.allclose(fsh[0]["d"].to_numpy(), 0.01 * np.arange(1, 8))
        tables["tests"] = tests.iloc[:1]                          # computed before ml_eval stored it
        fsh = J._few_shot(Ctx())
        assert np.isnan(fsh[50]["p"]) and np.isnan(fsh[50]["p_holm"])
        assert fsh[50]["better"] == 7 and J._n_amounts(fsh) == 2
    finally:
        J._t = orig


# -- the journal views on the synthetic study ---------------------------------------------------------
JNL = ("jnl_requirements", "jnl_c2", "jnl_x3", "jnl_decoding", "jnl_online", "jnl_robustness",
       "jnl_table_decoding", "jnl_table_robustness")


@pytest.mark.slow
@pytest.mark.parametrize("venue", VENUES)
def test_journal_views_render_cleanly(synth_study, tmp_path, venue):
    from biomech.check.outputs import _spec_text, code_leaks, legend_conflicts, pid_leaks
    from biomech.io.context import DataContext
    from biomech.viz.views import registry as R
    from biomech.viz.views import vocab as V

    ctx = DataContext(synth_study[0])
    for vid in JNL:
        spec = R.build(vid, R.request(ctx, vid))
        assert not pid_leaks(_spec_text(spec)), vid
        assert not code_leaks(_spec_text(V.reader_spec(spec))[1:]), vid
        if not isinstance(spec, FigureSpec):
            assert spec.rows, vid
            render_pgf.render_table(spec, tmp_path)
            continue
        assert legend_conflicts(spec) == [], vid
        assert spec.caption and all(p.heading()[0] or not p.title for p in spec.panels), vid
        _clean(V.reader_spec(spec), venue)
        for theme in ("light", "dark"):
            to_plotly(spec, theme=theme)
        render_pgf.write_style(tmp_path, venue)
        render_pgf.render_figure(spec, tmp_path, venue=venue)
