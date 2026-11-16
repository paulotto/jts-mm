# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J2.4 / J2.5 (dataset): the 3-D kinematics views and the claim-coverage views of
``viz/views/kin_r8.py`` -- registration, the geometry helpers, and every view and form built on a
synthetic in-memory store (with and without the tables only the next re-run stores) and on an
empty one: no exception, a structured caption that passes the caption rubric, no plan codes or
raw identifiers in the reader text, and an annotated empty panel where data are missing.
Fast: no result store, no rendering beyond one matplotlib build per view."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.check.text import caption_rubric_issues, text_issues
from biomech.stats import cohort_kinematics as ck
from biomech.stats.requirements import coverage_table
from biomech.viz import render_mpl
from biomech.viz.spec import FigureSpec
from biomech.viz.views import kin_r8 as K
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

PIDS = [f"s{k:02d}" for k in range(1, 9)]
VIEWS = ("kin_workspace3d", "kin_condyle3d", "kin_coupling_surface", "cov_outcome_n",
         "req_lowpass_bound", "req_envelope", "kin_null_variants", "kin_geometry_lit",
         "timing_window_null", "timing_single_cycle", "emg_spm_sigbar", "var_cmc_cycles")
NEW = {("stats_kinematics", "coupling_summary"), ("requirements", "max_coverage")}
VARIANT_SUBJECT_VALUES = K._variant_subject_values   # before the autouse fixture patches it


class _Config:
    seed = 0
    signals = {"kinematics": {"lowpass_hz": 6.0, "min_condyle_posterior_mm": 40.0}}
    raw = {"sensitivity": {"lowpass_8hz": {"set": {"signals.kinematics.lowpass_hz": 8.0}},
                           "other": {"set": {"stages.timing.emg_pre_s": 0.4}}}}

    def params(self, stage):
        return {"stats_timing": {"min_cycles": 3},
                "timing": {"emg_pre_s": 0.6, "emg_post_s": 0.2}}.get(stage, {})


class _Store:
    def __init__(self, stages):
        self._stages = sorted(stages)

    def stages(self):
        return self._stages


class Ctx:
    """The DataContext surface the views read: tables by (stage, name), arrays by
    (stage, unit), pseudonyms 1..n."""

    def __init__(self, tables, arrays):
        self.tables, self.arr = tables, arrays
        self.store = _Store({s for s, _ in tables} | {s for s, _ in arrays})
        self.config = _Config()

    def table(self, stage, name, pids=None):
        if (stage, name) not in self.tables:
            raise KeyError(name)
        d = self.tables[(stage, name)].copy()
        return d[d.pid.isin(pids)] if pids is not None and "pid" in d else d

    def arrays(self, stage, unit):
        return dict(self.arr.get((stage, unit), {}))

    def pids(self):
        return list(PIDS)

    def pseudonym(self, pid):
        return PIDS.index(pid) + 1


def _kin_arrays(rng: np.random.Generator, k: int) -> dict[str, np.ndarray]:
    a: dict[str, np.ndarray] = {}
    for j, t in enumerate(("rest",) + V.TASKS):
        n = 120
        s = np.linspace(0, 1, n)
        depth = 0.0 if t == "rest" else 20.0 + k
        inc = np.column_stack([-8 * s + rng.normal(0, 1, n) + (3 * j if t == "protrusion_retrusion"
                                                                  else 0),
                               rng.normal(0, 2 + j, n), -depth * s + rng.normal(0, 0.5, n)])
        a[f"{t}__incisor"] = inc
        for side, y in (("left", 55.0), ("right", -55.0)):
            a[f"{t}__tmj_{side}"] = np.column_stack([-60 + 8 * s, np.full(n, y), 30 - 3 * s])
        a[f"{t}__theta_vel"] = 50 * np.sin(6 * s) * (1 + 0.1 * k)
        a[f"{t}__theta_acc"] = 900 * np.cos(6 * s) * (1 + 0.1 * k)
    g = np.linspace(0, 20, 21)
    a["coupling__grid_deg"] = g
    slope = 0.2 - 0.06 * k
    a["coupling__tmj_mid_ant_mm"] = np.array([slope * g + rng.normal(0, 0.2, g.size)
                                              for _ in range(5)])
    return a


def synthetic(full: bool = True) -> Ctx:
    """Eight synthetic participants; ``full`` adds the tables of the next re-run."""
    rng = np.random.default_rng(3)
    arrays = {("kinematics", p): _kin_arrays(rng, k) for k, p in enumerate(PIDS)}
    g = arrays[("kinematics", PIDS[0])]["coupling__grid_deg"]
    curves = {"tmj_mid_ant_mm": [arrays[("kinematics", p)]["coupling__tmj_mid_ant_mm"]
                                 for p in PIDS]}
    band = ck.coupling_band(curves, g)
    arrays[("stats_kinematics", "_cohort")] = {"grid_deg": g, **band}
    x = np.linspace(0, 1, 101)
    arrays[("stats_emg", "_cohort")] = {
        "spm2__openers": np.array([np.sin(np.pi * x) + 0.05 * i for i in range(8)]),
        "spm2__closers": np.array([1 - np.sin(np.pi * x) + 0.05 * i for i in range(8)]),
        "spm1__left": np.array([10 + x + i for i in range(8)]),
        "spm1__right": np.array([10 + x + i for i in range(8)])}
    plaus = [1, 1, 0, 1, 1, 1, 0, 1]
    subj = pd.DataFrame({
        "pid": PIDS, "condyles_plausible": plaus,
        "max_observed_opening_mm": rng.uniform(25, 40, 8),
        "protrusion_mm": rng.uniform(5, 12, 8), "laterotrusion_left_mm": rng.uniform(6, 12, 8),
        "laterotrusion_right_mm": rng.uniform(6, 12, 8),
        "opening_angle_deg": rng.uniform(10, 25, 8),
        "peak_open_vel_max_dps": rng.uniform(60, 130, 8),
        "condyle_left_disp_mm": rng.uniform(4, 20, 8),
        "condyle_right_disp_mm": rng.uniform(4, 20, 8),
        **{c: rng.uniform(20, 100, 8) for c, _, _ in K.GEOMETRY}})
    if full:
        env = ck.dynamics_envelope({p: arrays[("kinematics", p)] for p in PIDS})
        subj = subj.merge(env, on="pid").assign(peak_close_vel_max_dps=rng.uniform(60, 130, 8))
    req = pd.DataFrame([{"key": k, "value": v, "ci_lo": v * 0.9}
                        for k, v in (("opening_mm", 38.0), ("protrusion_mm", 11.0),
                                     ("laterotrusion_mm", 11.0), ("peak_vel_dps", 120.0),
                                     ("close_vel_dps", 125.0), ("peak_acc_dps2", 1500.0))])
    fha = pd.DataFrame([{"pid": p, "cycle": c, "theta_to_deg": th, "n_x": 0.05, "n_y": 1.0,
                         "n_z": 0.05, "pierce_x_mm": -90 + th, "pierce_z_mm": -5.0 + c}
                        for p in PIDS for c in range(3) for th in (6.0, 12.0)])
    nulls = [k for k, _, _ in K.NULLS]
    hn = pd.DataFrame({"pid": PIDS, "obs_hinge_fit_mm": rng.uniform(1, 3, 8),
                       "obs_hinge_tmj_mm": rng.uniform(3, 9, 8),
                       **{f"ratio_{k}": rng.uniform(0.5, 30, 8) for k in nulls}})
    tests = pd.DataFrame([{"noise": k, "group": g_, "n_tested": 8, "above_n": 5,
                           "noise_ratio_median": 2.0, "noise_ratio_ci_lo": 1.5,
                           "noise_ratio_ci_hi": 3.0}
                          for k in nulls for g_ in ("all", "plausible_condyles")])
    cs = pd.DataFrame([{"metric": m, "group": g_, "n": 8, "median": 5.0 + i, "ci_lo": 4.0 + i,
                        "ci_hi": 6.0 + i} for i, m in enumerate(K.SENS_METRICS)
                       for g_, _, _ in K.SENS_GROUPS])
    rows = [(t, d, r) for t, d in K.TIMING_ROWS for r, _ in K.RULES]
    tn = pd.DataFrame([{"pid": p, "task": t, "direction": d, "detector": "tkeo", "rule": r,
                        "lead_ms": 80.0 + i, "lead_null_ms": 250.0 - i,
                        "lead_minus_null_ms": -170.0 + 2 * i}
                       for t, d, r in rows for i, p in enumerate(PIDS)])
    tc = pd.DataFrame([{"task": t, "direction": d, "detector": "tkeo", "rule": r,
                        "metric": "lead_ms", "p_null_holm": 1.0} for t, d, r in rows])
    tu = pd.DataFrame([{"pid": p, "task": t, "direction": d, "detector": "tkeo", "rule": r,
                        "usable_frac": 0.05 + 0.1 * i, "active_at_start": 0.5, "n": 20}
                       for t, d, r in rows for i, p in enumerate(PIDS)])
    spm = pd.DataFrame([{"family": "confirmatory", "test": "openers_vs_closers_open_close",
                         "clusters": "[(0, 11, 0.003), (59, 100, 0.0)]", "p_holm": 0.0},
                        {"family": "confirmatory", "test": "masseter_L_vs_R_chewing",
                         "clusters": "[]", "p_holm": 1.0},
                        {"family": "variability", "test": "cmc_opening_angle",
                         "clusters": None, "p_holm": np.nan, "cmc_within_median": 0.88,
                         "cmc_between": 0.87}])
    cn = pd.DataFrame([{"task": "open_close", "metric": "opening_mm", "side": "", "pid": p,
                        "target_halfwidth": 3.9, "n_available": 10, "n_needed": 2 + i % 4}
                       for i, p in enumerate(PIDS)])
    rel = pd.DataFrame([{"task": "open_close", "metric": "opening_mm", "side": "",
                         "mdc95": 7.8}])
    tables = {("stats_kinematics", "subjects"): subj, ("requirements", "requirements"): req,
              ("kinematics", "fha"): fha, ("uncertainty", "hinge_noise"): hn,
              ("stats_kinematics", "hinge_noise_tests"): tests,
              ("stats_kinematics", "condyle_sensitivity"): cs,
              ("stats_timing", "timing_null"): tn, ("stats_timing", "timing_cohort"): tc,
              ("stats_timing", "timing_usability"): tu, ("stats_emg", "spm"): spm,
              ("variability", "cycles_needed"): cn, ("stats_kinematics", "reliability"): rel}
    if full:
        tables[("stats_kinematics", "coupling_summary")] = ck.coupling_summary(
            curves, g, PIDS, {p for p, ok in zip(PIDS, plaus) if ok})
        tables[("requirements", "max_coverage")] = coverage_table()
    return Ctx(tables, arrays)


def _builds(ctx) -> list[tuple[str, str | None, FigureSpec]]:
    out = []
    for vid in VIEWS:
        v = R.get(vid)
        for f in (v.choice_values("form", ctx) or [None]):
            params = {} if f is None else {"form": f}
            out.append((vid, f, R.build(vid, R.request(ctx, vid, **params))))
    return out


@pytest.fixture(scope="module")
def full():
    return synthetic(True)


@pytest.fixture(autouse=True)
def _lowpass(monkeypatch):
    """The low-pass sensitivity comparison of the synthetic store (no comparison file)."""
    def values(ctx, name, column):
        s = K._t(ctx, "stats_kinematics", "subjects")
        return s.set_index("pid")[column] * 1.08 if column in s else pd.Series(dtype=float)
    monkeypatch.setattr(K, "_variant_subject_values", values)


def test_registered_with_forms():
    for vid in VIEWS:
        v = R.get(vid)
        assert v.kind == "paper" and v.group == "dataset", vid
    assert R.get("kin_workspace3d").choice_values("form") == list(K.WORKSPACE_FORMS)
    assert R.get("kin_coupling_surface").choice_values("form") == list(K.COUPLING_FORMS)
    assert R.get("kin_condyle3d").choice_values("form") == list(K.CONDYLE_FORMS)
    assert len(R.get("emg_spm_sigbar").choice_values("form")) == 2
    assert all(V.choice_label("form", f) != f for f in K.WORKSPACE_FORMS + K.CONDYLE_FORMS
               + K.COUPLING_FORMS + K.SPM_FORMS)                  # vocabulary has every form


def test_geometry_helpers(tmp_path):
    rng = np.random.default_rng(0)
    P = rng.normal(size=(400, 3)) * [5, 4, 10]
    cage = K.hull_cage(P, rings=3)
    assert cage is not None and np.isnan(cage).any(axis=1).sum() == 5     # 2 + 3 sections
    assert K.hull_cage(P[:3]) is None
    box = K.box_edges([0, -1, -2], [1, 1, 0])
    assert np.isnan(box).any(axis=1).sum() == 12
    rq = pd.DataFrame({"value": [30.0, 10.0, 12.0]},
                      index=pd.Index(["opening_mm", "protrusion_mm", "laterotrusion_mm"],
                                     name="key"))
    lo, hi = K.design_box(rq)
    assert lo.tolist() == [0.0, -12.0, -30.0] and hi.tolist() == [10.0, 12.0, 0.0]
    assert K.design_box(rq.iloc[:2]) is None
    assert K._outside(np.array([[0.5, 0, -1], [-2, 0, -1]]), (lo, hi)) == 50.0
    assert K._behind(np.array([[0.5, 0, -1], [-2, 0, -1]]), (lo, hi)) == 50.0
    lit = tmp_path / "literature.yaml"
    lit.write_text("geometry:\n  balkwill_deg:\n    - {low: 18, high: 26, ref: R3}\n"
                   "  bonwill_side_mm: {value: 101.6, ref: R1}\n")
    got = K.literature(lit)
    assert got["balkwill_deg"][0]["low"] == 18 and got["bonwill_side_mm"][0]["value"] == 101.6
    assert K.literature(tmp_path / "missing.yaml") == {}


def test_every_view_builds_with_a_rubric_caption(full):
    for vid, form, spec in _builds(full):
        assert isinstance(spec, FigureSpec), vid
        cap = spec.meta.get("caption")
        assert cap and cap["n"] and cap["marks"], (vid, form)
        reader = V.reader_spec(spec)
        assert caption_rubric_issues(reader) == [], (vid, form, caption_rubric_issues(reader))
        assert text_issues(reader) == [], (vid, form, text_issues(reader))
        assert not any("s0" in t for t in (spec.caption, *[p.title for p in spec.panels])), vid
        render_mpl.build(spec)                                 # draws (3-D, surface, sigbars)


def test_values_reach_the_figures(full):
    built = {(v, f): s for v, f, s in _builds(full)}
    ov = built[("kin_workspace3d", "overlay")]
    assert any(s.kind == "line3d" and s.role == "threshold" for s in ov.panels[0].series)
    assert ov.panels[1].series and not ov.panels[1].note             # share outside the box
    cv = built[("kin_coupling_surface", "curves")]
    assert all(p.series and not p.note for p in cv.panels)           # slope and miss drawn
    neg = (full.tables[("stats_kinematics", "coupling_summary")].slope_mm_per_deg < 0).sum()
    assert sum(s.role == "quaternary" for s in cv.panels[0].series) == neg
    sf = built[("kin_coupling_surface", "3d")]
    assert any(s.kind == "surface3d" for s in sf.panels[0].series)
    env = built[("req_envelope", None)]
    assert not env.panels[0].note and env.panels[2].series            # rows and coverage
    lp = built[("req_lowpass_bound", None)]
    assert lp.panels[0].x.ticks == [6.0, 8.0]                        # primary + the one run
    sb = built[("emg_spm_sigbar", "band")]
    assert any(s.kind == "sigbar" for s in sb.panels[0].series)
    assert sb.panels[1].note.startswith("no supra-threshold cluster")
    sc = built[("timing_single_cycle", None)]
    texts = [t for s in sc.panels[1].series if s.kind == "text" for t in s.text]
    assert texts and all(t.endswith(f"/{len(PIDS)}") for t in texts)
    cov = built[("cov_outcome_n", None)]
    flagged = [s for s in cov.panels[0].series if s.role == "secondary"]
    assert flagged                                                   # implausible condyles


def test_tables_of_the_next_rerun_degrade_to_notes():
    ctx = synthetic(False)
    built = {(v, f): s for v, f, s in _builds(ctx)}
    cv = built[("kin_coupling_surface", "curves")]
    assert cv.panels[0].series and cv.panels[1].note == K.NOT_STORED
    assert any("next re-run" in n for n in cv.meta["caption"]["notes"])
    env = built[("req_envelope", None)]
    assert env.panels[0].note.endswith(K.NOT_STORED) and env.panels[2].note == K.NOT_STORED
    assert caption_rubric_issues(V.reader_spec(env)) == []


def test_empty_store_builds_every_view():
    ctx = Ctx({}, {})
    for vid, form, spec in _builds(ctx):
        assert isinstance(spec, FigureSpec) and spec.caption, (vid, form)
        assert any(p.note for p in spec.panels) or not any(p.series for p in spec.panels), vid


def test_variant_subject_values_one_value_per_participant(monkeypatch):
    """A run that changes a participant's task label in the subjects key splits that participant
    into a primary-only and a variant-only comparison row; the lookup keeps the one value."""
    from biomech.viz import numbers as N

    rows = [("s01", "free_motion", 100.0, 108.0), ("s02", "chewing", 90.0, np.nan),
            ("s02", "free_motion", np.nan, 99.0)]
    cmp = pd.DataFrame({"stage": "stats_kinematics", "table": "subjects",
                        "column": "peak_open_vel_max_dps", "@pid": [r[0] for r in rows],
                        "@peak_acc_task": [r[1] for r in rows],
                        "primary": [r[2] for r in rows], "variant": [r[3] for r in rows]})
    monkeypatch.setattr(N, "sensitivity_table", lambda ctx, name: cmp)
    v = VARIANT_SUBJECT_VALUES(synthetic(), "lowpass_8hz", "peak_open_vel_max_dps")
    assert v.index.is_unique
    assert v.to_dict() == {"s01": 108.0, "s02": 99.0}
