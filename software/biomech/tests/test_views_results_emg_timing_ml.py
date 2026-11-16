# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Result views of the EMG, timing and decoding stages (results_emg/_timing/_ml).

Every view is built for every declared phase and picker value on the synthetic study,
rendered with all three backends, checked against the stored numbers it draws, and
scanned for study ids (figures show pseudonyms only).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from biomech.io.context import DataContext
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import to_plotly
from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import registry as R

pytestmark = pytest.mark.slow

MODULES = ("results_emg", "results_timing", "results_ml")
STUDY_ID = re.compile(r"(?<![A-Za-z0-9])s\d{2}(?![0-9])")


@pytest.fixture(scope="module")
def ctx(synth_study):
    c = DataContext(synth_study[0])
    yield c
    c.close()


def _views():
    return [v for v in R.views() if v.builder.__module__.rsplit(".", 1)[-1] in MODULES]


def _variants(v, ctx):
    base = [{"phase": p} for p in R.expand_phases(v, ctx)] if "phase" in v.params else [{}]
    out = list(base)
    for name in v.choices:
        out += [{**base[0], name: val} for val in v.choice_values(name, ctx)]
    for kw in out:
        if "pid" in v.params:
            kw["pid"] = ctx.pids()[0]
    return out


def _build(ctx, view_id, **kw):
    if "pid" in R.get(view_id).params:
        kw.setdefault("pid", ctx.pids()[0])
    return R.build(view_id, R.request(ctx, view_id, **kw))


def _texts(spec) -> list[str]:
    out = [spec.id, spec.caption]
    if isinstance(spec, TableSpec):
        out += [spec.notes] + [c.header for c in spec.columns]
        out += [str(v) for r in spec.rows for v in r.values()]
        return out
    out.append(spec.title)
    for p in spec.panels:
        out += [p.title, p.note, p.x.label, p.y.label, *(p.x.ticklabels or []),
                *(p.y.ticklabels or [])]
        for s in p.series:
            out += [s.label or "", *(s.text or [])]
    return out


def _series(spec, kind=None, panel=None):
    ps = spec.panels if panel is None else [spec.panels[panel]]
    return [s for p in ps for s in p.series if kind is None or s.kind == kind]


def _heat(spec, panel=0) -> np.ndarray:
    return next(s for s in spec.panels[panel].series if s.kind == "heatmap").z


def test_views_registered():
    ids = {v.id for v in _views()}
    assert {"emg_activation", "emg_cycles", "emg_refs", "emg_refs_table",
            "emg_mask_sensitivity", "emg_envelope", "emg_indices", "emg_rest_activity",
            "emg_cci", "emg_activation_lmm", "emg_spm", "emg_spm_table", "emg_varratio",
            "timing_onsets", "timing_release", "timing_release_sweep", "timing_usability",
            "timing_cohort", "timing_cohort_table", "ml_confusion", "ml_timeline",
            "ml_summary", "ml_permutation", "ml_variants", "ml_fusion", "ml_tests",
            "ml_generalization", "ml_windows_table"} <= ids
    dom = {"results_emg": "emg_time", "results_timing": "timing", "results_ml": "ml"}
    for v in _views():
        assert v.domain == dom[v.builder.__module__.rsplit(".", 1)[-1]], v.id


def test_every_variant_builds_renders_and_is_pseudonymous(ctx, tmp_path):
    n = 0
    for v in _views():
        for kw in _variants(v, ctx):
            spec = R.build(v.id, R.request(ctx, v.id, **kw))
            assert spec.caption and spec.sources, (v.id, kw)
            bad = [t for t in _texts(spec) if t and STUDY_ID.search(t)]
            assert not bad, (v.id, kw, bad[:3])
            if isinstance(spec, TableSpec):
                assert render_pgf.render_table(spec, tmp_path).is_file()
            else:
                assert isinstance(spec, FigureSpec)
                to_plotly(spec)
                render_mpl.render_figure(spec, tmp_path / f"{spec.id}.png", dpi=80)
                render_pgf.render_figure(spec, tmp_path)
            n += 1
    assert n >= 60


# -- EMG ------------------------------------------------------------------------------------
def test_emg_activation_matches_table(ctx):
    spec = _build(ctx, "emg_activation", phase="chewing")
    a = ctx.table("emg", "activation")
    a = a[a.task == "chewing"]
    z = _heat(spec)
    assert z.shape == (len(ctx.pids()), 6)
    pid, m = ctx.pids()[1], "masseter_r"
    col = list(spec.panels[0].x.ticklabels).index("masseter R")
    want = a[(a.pid == pid) & (a.muscle == m)].median_pct.iloc[0]
    assert z[1, col] == pytest.approx(want)
    assert spec.panels[0].y.ticklabels == [f"P{i + 1}" for i in range(len(ctx.pids()))]


def test_emg_cycles_profiles_and_cci(ctx):
    spec = _build(ctx, "emg_cycles", phase="open_close")
    prof = _series(spec, "line", panel=2)
    assert len(prof) == 6 and all(np.isfinite(s.y).any() for s in prof)
    c = ctx.table("emg", "cycles")
    pts = np.concatenate([s.x for s in _series(spec, "scatter", panel=3)
                          if s.role == "context"])
    assert np.sort(pts) == pytest.approx(np.sort(c[c.task == "open_close"].cci_opening))
    assert "cycles" in spec.panels[2].note


def test_emg_refs_and_table(ctx):
    r = ctx.table("emg", "refs")
    spec = _build(ctx, "emg_refs")
    assert np.nanmax(_heat(spec)) == pytest.approx(r.mvc_over_task.max())
    t = _build(ctx, "emg_refs_table")
    assert len(t.rows) == len(r)
    keys = {c.key for c in t.columns}
    assert {"ref_source", "mvc_over_task", "ref_uv", "usable_shape"} <= keys
    assert {row["pid"] for row in t.rows} == {f"P{i + 1}" for i in range(len(ctx.pids()))}


def test_emg_mask_sensitivity(ctx):
    s = ctx.table("emg", "mask_sensitivity")
    spec = _build(ctx, "emg_mask_sensitivity", phase="open_close")
    a, b = [x for x in _series(spec, "scatter", panel=0) if x.label]
    g = s[s.task == "open_close"].groupby("muscle")
    want = 100 * g.kept.sum() / g.n_cycles.sum()
    assert a.x[0] == pytest.approx(want["masseter_r"])
    assert "lowfreq" in b.label
    d = (s.median_pct_alt - s.median_pct)[s.task == "open_close"]
    pts = np.concatenate([x.x for x in _series(spec, "scatter", panel=2) if x.role == "context"])
    assert np.sort(pts) == pytest.approx(np.sort(d.to_numpy()))
    other = _build(ctx, "emg_mask_sensitivity", phase="left_right")
    assert "no kinematic cycles" in other.panels[0].note


def test_emg_envelope_units(ctx):
    pid = ctx.pids()[0]
    arr = ctx.arrays("emg", pid)
    names = [str(n) for n in arr["profile__names"]]
    uv = _build(ctx, "emg_envelope", phase="chewing")
    pct = _build(ctx, "emg_envelope", phase="chewing", units="norm")
    i = names.index("masseter_r")
    line = [s for s in uv.panels[0].series if s.kind == "line"][0]
    assert line.y == pytest.approx(arr["chewing__env_uv"][:, i].astype(float))
    line = [s for s in pct.panels[0].series if s.kind == "line"][0]
    assert line.y == pytest.approx(arr["chewing__norm_pct"][:, i].astype(float), nan_ok=True)
    assert uv.panels[0].y.unit == "\\micro\\volt" and pct.id.endswith("units-norm")
    assert len(uv.panels) == 7 and "P1" in uv.caption


def test_emg_cohort_strips(ctx):
    ind = _build(ctx, "emg_indices")
    st = ctx.table("stats_emg", "indices").set_index("metric")
    iv = [s for s in ind.panels[0].series if s.kind == "line"]
    assert iv[0].x == pytest.approx([st.loc["asym_masseter_pct", "ci_lo"],
                                     st.loc["asym_masseter_pct", "ci_hi"]])
    rest = _build(ctx, "emg_rest_activity")
    assert len([s for s in rest.panels[0].series if s.role == "context"]) == 6
    cci = _build(ctx, "emg_cci")
    assert cci.panels[0].y.ticklabels == ["open-close", "chewing"]


def test_emg_lmm_forest(ctx, monkeypatch):
    spec = _build(ctx, "emg_activation_lmm")
    assert "no model" in spec.panels[0].note          # 4 synthetic participants < 5
    fake = pd.DataFrame({"term": ["Intercept", "C(task)[T.open_close]",
                                  "C(muscle)[T.masseter_r]",
                                  "C(task)[T.open_close]:C(muscle)[T.masseter_r]"],
                         "coef": [1.0, 0.5, -0.2, 0.1], "ci_lo": [0.8, 0.3, -0.4, -0.1],
                         "ci_hi": [1.2, 0.7, 0.0, 0.3], "optimizer": "lbfgs"})
    orig = ctx.table
    monkeypatch.setattr(ctx, "table", lambda st, name, *a: fake.copy()
                        if (st, name) == ("stats_emg", "activation_lmm") else orig(st, name, *a))
    spec = _build(ctx, "emg_activation_lmm")
    p = spec.panels[0]
    assert p.y.ticklabels == ["open-close", "masseter R", "open-close × masseter R"]
    assert [s.x[0] for s in p.series if s.kind == "scatter"] == [0.5, -0.2, 0.1]
    assert "intercept 1.00" in spec.caption


def test_emg_spm_views(ctx):
    for test in ("masseter_L_vs_R_chewing", "openers_vs_closers_open_close"):
        spec = _build(ctx, "emg_spm", test=test)
        assert len([s for s in spec.panels[1].series if s.kind == "line"]) == 2
        assert spec.panels[0].series or "not computed" in spec.panels[0].note
    t = _build(ctx, "emg_spm_table")
    spm = ctx.table("stats_emg", "spm")
    assert len(t.rows) == len(spm)
    assert any(str(r["test"]).endswith("| P1") for r in t.rows)
    var = _build(ctx, "emg_varratio")
    assert all(p.series for p in var.panels)
    assert sum(1 for s in _series(var, "line", panel=0) if s.label) == 2


# -- timing ---------------------------------------------------------------------------------
def test_timing_onsets_pickers(ctx):
    on = ctx.table("timing", "onsets")
    prim = ctx.onset_rule()
    v = R.get("timing_onsets")
    assert v.choice_values("rule", ctx)[0] == prim
    spec = _build(ctx, "timing_onsets", phase="open_close")
    assert spec.id == "timing_onsets"                   # defaults: tkeo, primary rule
    g = on[(on.task == "open_close") & (on.detector == "tkeo") & (on.rule == prim)
           & (on.direction == "opening") & on.usable & (on.pid == ctx.pids()[0])]
    row0 = [s for s in spec.panels[0].series if s.role == "context" and np.all(s.y < 0.5)]
    assert np.sort(row0[0].x) == pytest.approx(np.sort(g.lead_ms.dropna().to_numpy()))
    other = _build(ctx, "timing_onsets", phase="open_close", detector="envelope", rule="first")
    assert other.id.endswith("detector-envelope--rule-first")
    assert "first onset" in other.caption


def test_timing_release_views(ctx):
    rel = ctx.table("timing", "release")
    spec = _build(ctx, "timing_release", phase="open_close")
    pts = np.concatenate([s.x for s in spec.panels[0].series if s.role == "context"])
    want = rel[rel.task == "open_close"].anticipation_ms.dropna().to_numpy()
    assert np.sort(pts) == pytest.approx(np.sort(want))
    sw = _build(ctx, "timing_release_sweep", phase="open_close")
    assert _heat(sw, 0).shape == (3, 3) and _heat(sw, 1).shape == (3, 3)
    assert sum(t.endswith("*") for t in sw.panels[1].series[0].text) == 1
    rc = ctx.table("stats_timing", "release_cohort")
    prim = rc[(rc.task == "open_close") & rc.primary]
    assert np.nanmax(_heat(sw, 1)) >= float(prim["median"].iloc[0])


def test_timing_cohort_views(ctx):
    us = _build(ctx, "timing_usability")
    z = _heat(us)
    assert z.shape == (9, 4) and np.nanmin(z) >= 0 and np.nanmax(z) <= 100
    tc = _build(ctx, "timing_cohort", phase="open_close")
    c = ctx.table("stats_timing", "timing_cohort")
    want = c[(c.task == "open_close") & (c.direction == "opening") & (c.detector == "tkeo")
             & (c.rule == ctx.onset_rule()) & (c.metric == "lead_ms")]["median"].iloc[0]
    first = [s for s in tc.panels[0].series if s.kind == "scatter"][0]
    assert first.x[0] == pytest.approx(want)
    assert tc.panels[0].y.ticklabels[0].endswith("*")
    assert len([s for s in tc.panels[2].series if s.kind == "scatter"]) == 4
    t = _build(ctx, "timing_cohort_table", phase="chewing")
    assert len(t.rows) == ((c.task == "chewing")).sum()


# -- decoding -------------------------------------------------------------------------------
def test_ml_confusion_and_timeline(ctx):
    arr = ctx.store.read_arrays("ml_eval", "_cohort")
    cfgs = R.get("ml_confusion").choice_values("config", ctx)
    assert cfgs[0] == "emg_imu|svm" and len(cfgs) == sum(k.startswith("confusion__")
                                                         for k in arr.keys())
    spec = _build(ctx, "ml_confusion", config="imu|lda")
    z = _heat(spec)
    assert z.sum(axis=1) == pytest.approx(np.full(7, 100.0), abs=1e-6)
    assert spec.panels[0].x.ticklabels[0] == "REST"
    pid = ctx.pids()[0]
    tl = _build(ctx, "ml_timeline", fset="emg")
    pred = [s for s in tl.panels[0].series if s.kind == "scatter"][0]
    assert len(pred.x) == len(arr[f"pred__{pid}__emg__t_us"])
    assert sorted(pred.y) == sorted(arr[f"pred__{pid}__emg__yhat"].astype(float))
    assert "P1" in tl.caption and tl.panels[0].y.ticklabels[1] == "OPEN"


def test_ml_tables_and_figures(ctx):
    s = ctx.table("ml_eval", "summary")
    spec = _build(ctx, "ml_summary")
    assert len(spec.panels[0].y.ticklabels) == (s.metric == "balanced_accuracy").sum()
    lat = _build(ctx, "ml_summary", metric="latency_ms_median")
    assert lat.panels[0].x.unit == "\\milli\\second"
    perm = _build(ctx, "ml_permutation")
    tests = ctx.table("ml_eval", "tests")
    obs = [s for s in perm.panels[0].series if s.kind == "vline" and s.label == "observed"]
    assert obs[0].x[0] == pytest.approx(tests.stat.iloc[0])
    assert perm.panels[1].y.ticklabels == ["EMG", "IMU"]
    var = _build(ctx, "ml_variants")
    assert {s.label for s in var.panels[0].series if s.label} == {"EMG + IMU", "IMU"}
    # (b) one dot per fold: one fold per held-out participant
    assert sum(int(np.isfinite(s.x).sum()) for s in var.panels[1].series
               if s.kind == "scatter" and s.role == "context") == len(ctx.pids())
    assert len(var.panels[2].y.ticklabels) == len(ctx.pids())
    fu = _build(ctx, "ml_fusion")
    fd = ctx.table("ml_eval", "fusion_diagnostics")
    assert len(fu.panels[0].y.ticklabels) == len(fd)
    assert len(_build(ctx, "ml_tests").rows) == len(tests)
    assert "five" in _build(ctx, "ml_generalization").notes
    w = _build(ctx, "ml_windows_table")
    meta = ctx.table("ml_windows", "meta")
    assert [r["n_windows"] for r in w.rows] == list(meta.set_index("pid").loc[ctx.pids(),
                                                                            "n_windows"])
    assert all("mvc" in r["ref_source"] and "emg" in r["features"] for r in w.rows)
