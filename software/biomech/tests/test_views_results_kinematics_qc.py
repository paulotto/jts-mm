# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Result views of kinematics, uncertainty, variability, QC, IMU and labels on the synthetic study.

Every view of ``views.results_kinematics`` / ``views.results_qc`` is built for every
declared phase and picker value and rendered with all three backends (Plotly, matplotlib,
pgfplots); the numbers drawn are checked against the store, and no output names a study id.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from biomech.check import coverage as C
from biomech.io.context import DataContext
from biomech.viz import render_mpl, render_pgf
from biomech.viz.render_plotly import to_plotly
from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import registry as R

pytestmark = pytest.mark.slow

MODULES = ("biomech.viz.views.results_kinematics", "biomech.viz.views.results_qc")
STUDY_ID = re.compile(r"\bs\d{2}\b")
MINE = {"kinematics", "stats_kinematics", "uncertainty", "qc", "qc_cohort", "imu", "labels",
        "variability", "validation"}
EXPECTED = {
    "kin_summary": "kinematics", "kin_anatomy": "kinematics", "kin_anatomy_table": "kinematics",
    "kin_opening_model": "kinematics", "kin_coupling": "kinematics",
    "kin_coupling_cohort": "kinematics", "kin_hinge_frames": "kinematics",
    "kin_theta_raw": "kinematics", "kin_fha_cohort": "kinematics",
    "kin_condyle_sensitivity": "kinematics", "kin_norms": "kinematics",
    "kin_norms_table": "kinematics", "kin_influence": "kinematics",
    "kin_subjects": "kinematics", "kin_reliability": "kinematics",
    "kin_hinge_summary": "kinematics", "unc_null": "uncertainty", "unc_ci": "uncertainty",
    "unc_hinge_noise": "uncertainty", "var_spec_values": "variability",
    "var_cycles_needed": "variability", "var_protocol": "variability",
    "var_spec_curve": "variability", "qc_heatmap_emg": "qc", "qc_heatmap_imu": "qc",
    "qc_heatmap_optical": "qc", "qc_heatmap_phases": "qc", "qc_heatmap_sessions": "qc",
    "qc_heatmap_emg_check": "qc",
    "qc_overview": "qc", "qc_emg_flags": "qc", "qc_demographics": "qc", "qc_validation": "qc",
    "imu_calibration": "imu", "imu_cycles": "imu", "imu_quasi_static": "imu",
    "labels_timeline": "ml", "labels_counts": "ml"}


@pytest.fixture(scope="module")
def ctx(synth_study):
    c = DataContext(synth_study[0])
    yield c
    c.close()


def _variants(v, ctx) -> list[dict]:
    out = [{"phase": ph} if ph else {} for ph in R.expand_phases(v, ctx)]
    for name in v.choices:
        vals = v.choice_values(name, ctx)
        assert vals, f"{v.id}: picker {name} has no values"
        out += [{name: val} for val in vals]
    return out


def _build(ctx, view_id: str, **kw):
    v = R.get(view_id)
    if "pid" in v.params:
        kw.setdefault("pid", ctx.pids()[1])
    return R.build(view_id, R.request(ctx, view_id, **kw))


def _drawn(spec: FigureSpec) -> int:
    return sum(1 for p in spec.panels for s in p.series
               if s.kind not in ("vline", "hline", "text")
               and any(a is not None and np.size(a) and np.isfinite(np.asarray(a, float)).any()
                       for a in (s.y, s.z, s.y2)))


def test_views_registered():
    got = {v.id: v.domain for v in R.views() if v.builder.__module__ in MODULES}
    assert got == EXPECTED


@pytest.mark.parametrize("view_id", sorted(EXPECTED))
def test_view_builds_and_renders(ctx, view_id, tmp_path):
    v = R.get(view_id)
    for kw in _variants(v, ctx):
        spec = _build(ctx, view_id, **kw)
        assert not STUDY_ID.search(repr(spec)), (spec.id, STUDY_ID.findall(repr(spec))[:3])
        assert spec.sources, spec.id
        if isinstance(spec, TableSpec):
            assert spec.columns and spec.caption
            if view_id != "qc_emg_flags":                  # no flags is a valid result
                assert spec.rows, spec.id
            tex = render_pgf.render_table(spec, tmp_path)
            assert not STUDY_ID.search(tex.read_text(encoding="utf-8")), spec.id
            continue
        assert spec.caption and spec.panels
        if view_id != "kin_influence":                     # synthetic: no older participant
            assert _drawn(spec) > 0, (spec.id, [p.note for p in spec.panels])
        for p in spec.panels:                              # legends only for >= 2 series
            assert p.legend_mode() == "none" or len(p.legend_entries()) >= 2, (spec.id, p.title)
        fig = to_plotly(spec)
        assert len(fig.data) > 0
        assert not STUDY_ID.search(fig.to_json()), spec.id
        assert render_mpl.render_figure(spec, tmp_path / f"{spec.id}.png", dpi=60).exists()
        tex = render_pgf.render_figure(spec, tmp_path)
        assert not STUDY_ID.search(tex.read_text(encoding="utf-8")), tex.name


def test_numbers_match_the_store(ctx):
    pid = ctx.pids()[1]
    # cohort coupling: the drawn median is the stored one
    arr = ctx.store.read_arrays("stats_kinematics", "_cohort")
    spec = _build(ctx, "kin_coupling_cohort")
    acc = [s for s in spec.panels[0].series if s.role == "accent"][0]
    np.testing.assert_allclose(acc.y, arr["opening_mm__median"])
    assert any(s.kind == "band" for s in spec.panels[0].series)
    # H0 histograms: the observed residual is marked at the stored value
    hn = ctx.table("uncertainty", "hinge_noise").set_index("pid")
    spec = _build(ctx, "unc_null", pid=pid)
    obs = [s for s in spec.panels[0].series if s.kind == "vline" and s.role == "accent"][0]
    assert obs.x[0] == pytest.approx(hn.at[pid, "obs_hinge_fit_mm"])
    # bootstrap CI: the interval equals the stored percentile CI
    spec = _build(ctx, "unc_ci")
    i = ctx.pids().index(pid)
    line = [s for s in spec.panels[1].series if s.kind == "line"][i]
    assert line.x == pytest.approx([hn.at[pid, "obs_hinge_fit_ci_lo"],
                                    hn.at[pid, "obs_hinge_fit_ci_hi"]], rel=1e-6)
    # quasi-static spans cover exactly the stored mask
    a = ctx.arrays("imu", pid)
    spec = _build(ctx, "imu_quasi_static", pid=pid, phase="chewing")
    spans = [s for s in spec.panels[0].series if s.kind == "hspan"]
    assert spans and len(spans) == int(np.sum(np.diff(np.r_[0, a["chewing__quasi_static"]
                                                            .astype(int)]) == 1))
    # QC heatmap: participant x muscle values of the stored table
    spec = _build(ctx, "qc_heatmap_emg", column="snr_db")
    z = spec.panels[0].series[0].z
    e = ctx.table("qc", "emg")
    assert z.shape == (len(ctx.pids()), 6)
    assert np.nanmax(z) == pytest.approx(e.snr_db.max())


def test_privacy(ctx):
    demo = _build(ctx, "qc_demographics")
    assert len(demo.rows) == 1                         # the aggregate row only
    assert not {"sex", "age", "pid"} & {c.key for c in demo.columns}
    sess = _build(ctx, "qc_heatmap_sessions", column="duration_s")
    assert "session" not in repr(sess.panels[0].x.ticklabels)
    tab = _build(ctx, "kin_anatomy_table")
    assert [c.header for c in tab.columns[2:]] == [f"P{i + 1}" for i in range(len(ctx.pids()))]


def test_no_output_of_these_stages_uncovered(ctx):
    log, errors = C.view_reads(ctx)
    assert not [e for e in errors if e[0].split()[0] in EXPECTED], errors[:5]
    u = C.uncovered(ctx.store, log)
    left = [x for x in u["tables"] + u["arrays"]
            if x[0] in MINE and not x[1].startswith("bandwidth")]
    assert not left, left
