# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""(I1, I2, I5): the C2-E views (event sequence, locking, cohort table, the
per-cycle dashboards), the camera-free oracle dashboard, the X6 / X5b / X5c in-text numbers and
the completeness of the new tables -- on an empty store and on a synthetic store.

The synthetic store is built in memory: C2-E from :func:`biomech.testing.synth.timing_arrays`
participants run through :func:`biomech.timing.c2e.analyse` and
:func:`biomech.stats.cohort_c2e.c2e_cohort` (the timing and stats_timing stages' own functions,
with few surrogates), X5b / X5c from synthetic per-participant rows through
:func:`biomech.stats.cohort_camerafree.oracle` (the camerafree stage itself needs a recorded
session; its per-participant tables are written in their stored form). Fast: about 5 s.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from biomech import papers
from biomech.check.outputs import code_leaks, pid_leaks, _spec_text
from biomech.results.access import note_table, track, tracking_frame
from biomech.stats import cohort_c2e as CC
from biomech.stats.cohort_camerafree import ORACLE_TASKS, oracle
from biomech.testing import synth as S
from biomech.timing import c2e
from biomech.viz import numbers as N
from biomech.viz.spec import FigureSpec
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

ROOT = Path(__file__).resolve().parents[3] / "experiments" / "icorr2027"
PIDS = [f"s{k:02d}" for k in range(1, 7)]
P = {"tasks": ["open_close", "chewing"], "vel_frac": 0.05, "acc_k": 3.0, "pre_s": 0.3,
     "emg_pre_s": 0.6, "emg_post_s": 0.2, "k_sd": 3.0, "burst_min_db": 6.0, "ensemble": {},
     "release": {}}
QUICK = {"n_surrogates": 20, "n_s2": 20, "coupling": {"n_null": 10}}
SPEC = dict(emg_fs=1000, task_s={"open_close": 12.0, "chewing": 8.0})
# the gate decision of etc/biomech.yaml (timing.c2e; X6 (change log), 2026-10-31)
GATES = {"delta_ms": 17.0, "gates_passed": True, "gate_power_40ms": 1.0,
         "s2_min_sd_ms": 232.0}
NEW_VIEWS = ("c2e_sequence", "c2e_locking", "c2e_table", "c2e_cycles", "c2e_release",
             "cf_oracle_tasks")
NEW_TABLES = {("timing", "c2e_cycles"), ("timing", "c2e_null"), ("timing", "c2e_release"),
              ("timing", "coupling"), ("stats_timing", "c2e_cohort"),
              ("camerafree", "cf_decomposition"), ("camerafree", "cf_task_model"),
              ("camerafree_cohort", "cf_oracle_participant"),
              ("camerafree_cohort", "cf_oracle_cohort"),
              ("camerafree_cohort", "cf_oracle_contrasts")}


class _Config:
    seed = 0

    def params(self, stage):
        return {"timing": {"c2e": {"min_usable": 5, "flag_usable": 3, **GATES}}}.get(stage, {})


class _Store:
    def __init__(self, tables):
        self._stages = sorted({st for st, _ in tables})

    def stages(self):
        return self._stages


class Ctx:
    """The DataContext surface the views use: tables by (stage, name), read like
    ``DataContext.table`` (noted, column-tracked under ``track()``), pseudonyms 1..n."""

    def __init__(self, tables: dict[tuple[str, str], pd.DataFrame]):
        self.tables = tables
        self.store = _Store(tables)
        self.config = _Config()

    def table(self, stage, name, pids=None):
        if (stage, name) not in self.tables:
            raise KeyError(name)
        note_table(stage, name)
        d = self.tables[(stage, name)].copy()
        if pids and "pid" in d:
            d = d[d["pid"].isin(pids)]
        return tracking_frame(d, stage, name)

    def pids(self):
        return list(PIDS)

    def pseudonym(self, pid):
        return PIDS.index(pid) + 1

    def pseudonyms(self):
        return {p: i + 1 for i, p in enumerate(PIDS)}


def _cf_rows(est: str, base: float) -> list[dict]:
    group = {"open_close": "rotational", "chewing": "rotational",
             "protrusion_retrusion": "translational", "left_right": "translational"}
    out = []
    for i, pid in enumerate(PIDS):
        for j, t in enumerate(ORACLE_TASKS):
            out.append({"pid": pid, "estimator": est, "task": t, "scored": True,
                        "task_group": group[t], "rmse_inc_mm": base + 0.1 * i + 0.5 * j})
        out.append({"pid": pid, "estimator": est, "task": "scored", "scored": True,
                    "task_group": "all", "rmse_inc_mm": base + 1.0})
    return out


def synthetic_tables() -> dict[tuple[str, str], pd.DataFrame]:
    """The new tables of six synthetic participants (closers 40 ms before closing)."""
    outs = []
    for k, pid in enumerate(PIDS):
        sp = S.Spec(seed=10 + k, closer_lead_ms=40.0, lead_jitter_ms=20.0,
                    period_jitter_s=0.15, **SPEC)
        ta = S.timing_arrays(sp)
        outs.append(c2e.analyse(ta.prep, pid, ta.imu_arrays, ta.kin_cycles,
                                {**P, "c2e": QUICK}, seed=1, kin_arrays=ta.kin_arrays))
    t = {("timing", n): pd.concat([o[n] for o in outs], ignore_index=True) for n in outs[0]}
    t[("stats_timing", "c2e_cohort")] = CC.c2e_cohort(t[("timing", "c2e_null")],
                                                      t[("timing", "coupling")], GATES, 0)
    dec = pd.DataFrame(_cf_rows("X5b_model", 1.0) + _cf_rows("X5b_rotation", 2.0)
                       + _cf_rows("E4own", 3.0))
    x5c = pd.DataFrame(_cf_rows("X5c", 2.5)).assign(label_class="OPEN+CLOSE",
                                                     n_cal_samples=900, note="")
    x5c.loc[x5c.index[:1], ["rmse_inc_mm", "note"]] = [np.nan, "no model"]
    t[("camerafree", "cf_decomposition")] = dec
    t[("camerafree", "cf_task_model")] = x5c
    tr = dec[dec["estimator"] == "E4own"]
    for n, d in oracle(dec, x5c, tr, "E3b", seed=1, n_boot=200).items():
        t[("camerafree_cohort", n)] = d
    return t


@pytest.fixture(scope="module")
def tables() -> dict[tuple[str, str], pd.DataFrame]:
    return synthetic_tables()


def _variants(vid: str, ctx) -> list[dict]:
    v = R.get(vid)
    kws = [{}]
    for name in v.choices:
        kws += [{name: val} for val in v.choice_values(name, ctx)]
    if "phase" in v.params:
        kws += [{"phase": ph} for ph in v.phases]
    return kws


def _build_all(ctx) -> list:
    specs = []
    for vid in NEW_VIEWS:
        for kw in _variants(vid, ctx):
            specs.append(R.build(vid, R.request(ctx, vid, **kw)))
    return specs


# -- views ----------------------------------------------------------------------------------
def test_views_build_on_an_empty_store():
    for spec in _build_all(Ctx({})):
        assert spec.caption, spec.id
        if isinstance(spec, FigureSpec):
            assert not any(s.kind == "scatter" for p in spec.panels for s in p.series)


def test_views_build_on_the_synthetic_store(tables):
    ctx = Ctx(tables)
    specs = _build_all(ctx)
    assert {s.id.split("--")[0] for s in specs} == set(NEW_VIEWS)
    for spec in specs:
        text = _spec_text(spec)
        assert not pid_leaks(text), (spec.id, pid_leaks(text))         # pseudonyms only
        assert "meta" in dir(spec) and spec.meta.get("caption"), spec.id    # structured
        if R.get(spec.id.split("--")[0]).kind == "paper":
            assert not code_leaks(text), (spec.id, code_leaks(text))
    seq = R.build("c2e_sequence", R.request(ctx, "c2e_sequence"))
    assert [p.y.ticklabels for p in seq.panels][0][0] == "closer release"
    diamonds = [s for p in seq.panels for s in p.series if s.style.get("ms") == "d"]
    assert len(diamonds) == 8                                # 4 events x 2 tasks
    rel = diamonds[0].x[0]                                   # release before the opening
    assert rel < 0 and "n = 6 participants" in seq.caption
    closer = [s for s in seq.panels[0].series if s.style.get("ms") == "d"][3].x[0]
    closing = [s for s in seq.panels[0].series if s.style.get("ms") == "d"][2].x[0]
    assert closing - 80 < closer < closing                   # truth: closers 40 ms first
    lock = R.build("c2e_locking", R.request(ctx, "c2e_locking"))
    assert lock.panels[0].y.ticklabels == [f"P{i}" for i in range(1, 7)]
    tab = R.build("c2e_table", R.request(ctx, "c2e_table"))
    keys = list(tables[("stats_timing", "c2e_cohort")]["key"])
    data = [r for r in tab.rows if not r.get("_rule") and not r.get("_head")]
    assert len(data) == len(keys) and data[-1]["question"] == "Decision"
    heads = [r["question"] for r in tab.rows if r.get("_head")]
    assert heads[0] == "Primary hypothesis" and len(heads) == len(set(heads))
    s2 = [r for r, k in zip(data, keys) if k.startswith("s2_")]
    assert s2 and all(r["result"] == "descriptive; cohort p withheld" for r in s2)
    heat = R.build("cf_oracle_tasks", R.request(ctx, "cf_oracle_tasks", estimator="X5c"))
    assert "without a value" in heat.caption


def test_new_forms_and_pickers_have_reader_names():
    for form in ("timeline", "aligned", "raincloud", "dumbbell", "slopegraph"):
        assert V.choice_label("form", form) != form.replace("_", " "), form
    for src in R.get("c2e_cycles").choices["lead"]:
        assert V.choice_label("lead", src) != src, src


# -- completeness ---------------------------------------------------------------------------
def test_new_tables_are_consumed_by_an_artefact(tables):
    """Every new table is read by a figure or curated table (not only its complete table
    view), and every column of it reaches some view (complete table views included) --
    the check-outputs completeness rule on these tables."""
    ctx = Ctx(tables)
    curated = [v.id for v in R.views() if v.kind in ("paper", "dashboard")
               and v.domain != "tables"]
    from biomech.viz.views.tables import declarations, view_id

    complete = [view_id(k, d) for k, d in declarations().items()
                if tuple(k.split(":", 1)) in NEW_TABLES]
    with track() as log:
        for vid in curated:
            for kw in _variants(vid, ctx):
                try:
                    R.build(vid, R.request(ctx, vid, **kw))
                except Exception:                  # other views need tables not built here
                    pass
    assert NEW_TABLES <= log.tables, NEW_TABLES - log.tables
    with track() as log_all:
        for vid in complete:
            R.build(vid, R.request(ctx, vid))
    cols = {(st, n, c) for (st, n), d in tables.items() if (st, n) in NEW_TABLES
            for c in d.columns}
    assert not cols - log.columns - log_all.columns


# -- paper set and numbers ------------------------------------------------------------------
def test_paper_set_carries_the_c2e_items():
    ps = papers.load(ROOT / "paper_dataset.yaml")
    parts = {i.view_id: i.part for i in ps.items}
    # c2e_sequence left the JNER set on 2026-11-12; the view itself stays tested above
    assert "c2e_sequence" not in parts and parts["c2e_table"] == "supporting"
    assert parts["c2e_locking"] == "supplement"
    for vid in ("jnl_c2", "jnl_x3", "cf_decomposition", "cf_table_oracle"):  # kept
        assert vid in parts


def test_x6_x5b_x5c_numbers_resolve_on_the_synthetic_store(tables):
    decl = N.load(ROOT / "numbers_dataset.yaml")
    new = {k: d for k, d in decl.items() if str(d.get("source", "")) in
           ("stats_timing:c2e_cohort", "camerafree_cohort:cf_oracle_cohort",
            "camerafree_cohort:cf_oracle_contrasts")}
    assert len(new) >= 40
    got = {n.key: n for n in N.resolve(Ctx(tables), new)}
    assert set(got) == set(new)
    coh = tables[("stats_timing", "c2e_cohort")].set_index("key")
    assert got["cTwoEHOne"].value == pytest.approx(coh.loc["h1", "median"])
    assert got["cTwoEHOne"].n == coh.loc["h1", "n"]
    assert got["cTwoEDecision"].text == coh.loc["decision", "hypothesis"]
    assert got["cTwoEDelta"].text == "17"
    assert got["cfOracleConRotModelRot"].value == pytest.approx(1.0)
    assert got["cfOracleConTaskOwn"].n == len(PIDS)
    # every number states its n within the cohort
    assert all(n.n is None or n.n <= len(PIDS) for n in got.values())
