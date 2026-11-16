# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Run API + ResultStore: caching, staleness, failure isolation, cancellation."""

from __future__ import annotations

import threading

import numpy as np
import pandas as pd
import pytest

from biomech import pipeline
from biomech.pipeline import StageResult, run, stage, stage_status
from biomech.results import COHORT_UNIT, ResultStore

CALLS: list[str] = []


@pytest.fixture(autouse=True)
def toy_stages():
    """Register throw-away stages; restore the registry afterwards."""
    pipeline.registry()  # load built-ins (and plugins) first, so the snapshot holds them
    saved = dict(pipeline._REGISTRY)
    CALLS.clear()

    @stage("t_a", version="1")
    def a(ctx):
        CALLS.append(f"a:{ctx.pid}")
        if ctx.params.get("fail_for") == ctx.pid:
            raise RuntimeError("boom")
        return StageResult(tables={"x": pd.DataFrame({"pid": [ctx.pid], "v": [ctx.params["k"]]})},
                           arrays={"sig": np.arange(3)})

    @stage("t_b", scope="cohort", depends=("t_a",))
    def b(ctx):
        CALLS.append("b")
        x = ctx.upstream("t_a", "x")
        return StageResult(tables={"sum": pd.DataFrame({"total": [x["v"].sum()]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


def test_run_caches_and_reads_back(study):
    cfg = study({"t_a": {"k": 2}})
    s1 = run(cfg, ["t_b"])
    assert s1.statuses["t_a"] == {"s01": "ran", "s02": "ran"}  # s03 excluded
    assert s1.statuses["t_b"] == {COHORT_UNIT: "ran"}
    store = ResultStore(cfg.out_dir)
    assert store.read_table("t_b", "sum")["total"].item() == 4
    assert store.read_arrays("t_a", "s01")["sig"].tolist() == [0, 1, 2]

    CALLS.clear()
    s2 = run(cfg, ["t_b"])
    assert CALLS == [] and set(s2.statuses["t_a"].values()) == {"cached"}


def test_param_change_marks_downstream_stale(study):
    run(study({"t_a": {"k": 1}}), ["t_b"])
    cfg = study({"t_a": {"k": 5}})
    st = stage_status(cfg, ["t_b"])
    assert set(st["status"]) == {"stale"}
    CALLS.clear()
    run(cfg, ["t_b"])
    assert sorted(CALLS) == ["a:s01", "a:s02", "b"]
    assert ResultStore(cfg.out_dir).read_table("t_b", "sum")["total"].item() == 10


def test_failure_is_isolated_and_blocks_downstream(study):
    cfg = study({"t_a": {"k": 1, "fail_for": "s02"}})
    s = run(cfg, ["t_b"])
    assert s.statuses["t_a"] == {"s01": "ran", "s02": "failed"}
    # The cohort stage runs on the participants whose upstream is current (R4 fail policy).
    assert s.statuses["t_b"] == {COHORT_UNIT: "ran"}
    store = ResultStore(cfg.out_dir)
    assert store.read_table("t_b", "sum")["total"].item() == 1
    cov = store.read_table("t_b", "coverage").set_index("pid")
    assert cov.loc["s02", "reason"] == "t_a failed" and not bool(cov.loc["s02", "used"])
    meta = ResultStore(cfg.out_dir).meta("t_a", "s02")
    assert meta.status == "failed" and "boom" in meta.error
    events = ResultStore(cfg.out_dir).run_events(s.run_id)
    assert events[0]["kind"] == "run_start" and events[-1]["kind"] == "run_end"


def test_cancel_stops_after_current_unit(study):
    cfg = study({"t_a": {"k": 1}})
    CALLS.clear()
    s = run(cfg, ["t_b"], cancel=lambda: len(CALLS) >= 1)
    # The running unit finishes; what never ran is reported as cancelled.
    assert s.cancelled and s.statuses["t_a"] == {"s01": "ran", "s02": "cancelled"}
    assert s.statuses["t_b"] == {COHORT_UNIT: "cancelled"} and CALLS == ["a:s01"]


def test_cancel_before_start_runs_nothing(study):
    cfg = study({"t_a": {"k": 1}})
    ev = threading.Event()
    ev.set()
    CALLS.clear()
    s = run(cfg, ["t_b"], cancel=ev)
    assert s.cancelled and CALLS == [] and s.statuses == {}


def test_cohort_stage_stopped_by_the_token_is_cancelled_not_failed(study):
    """A long cohort stage that checks the token mid-way (ml_eval does, between
    configurations) raises; the unit is reported cancelled and nothing is stored."""
    from biomech.pipeline import StageResult, stage

    ev = threading.Event()

    @stage("t_c", scope="cohort", depends=("t_a",))
    def c(ctx):
        ev.set()                                  # the user presses Ctrl-C mid-stage
        if ctx.cancelled():
            raise RuntimeError("cancelled")
        return StageResult(tables={})

    cfg = study({"t_a": {"k": 1}})
    s = run(cfg, ["t_c"], cancel=ev)
    assert s.statuses["t_c"] == {COHORT_UNIT: "cancelled"} and s.n_failed == 0
    assert ResultStore(cfg.out_dir).meta("t_c", COHORT_UNIT) is None


def test_sensitivity_seed_cache_reuses_only_matching_units(study):
    """A variant that changes t_b only: t_a units are linked from the primary (same key), t_b
    is not and re-runs; an outdated variant unit with the primary's wanted key is replaced."""
    from dataclasses import replace

    from biomech.sensitivity import seed_cache

    cfg = study({"t_a": {"k": 1}})
    run(cfg, ["t_b"])
    vcfg = replace(study({"t_a": {"k": 1}, "t_b": {"x": 2}}), out_dir=cfg.out_dir / "variant")
    assert seed_cache(cfg, vcfg) == 2                       # the two t_a units, not t_b
    CALLS.clear()
    s = run(vcfg, ["t_b"])
    assert CALLS == ["b"] and set(s.statuses["t_a"].values()) == {"cached"}
    assert seed_cache(cfg, vcfg) == 0                       # nothing left to link


def test_subject_filter_and_force(study):
    cfg = study({"t_a": {"k": 1}})
    run(cfg, ["t_a"], ["s01"])
    assert CALLS == ["a:s01"]
    CALLS.clear()
    run(cfg, ["t_a"], ["s01"], force=True)
    assert CALLS == ["a:s01"]


def test_manifest_records_provenance(study):
    cfg = study({"t_a": {"k": 1}})
    s = run(cfg, ["t_a"])
    man = ResultStore(cfg.out_dir).run_manifest(s.run_id)
    assert man["stages"] == ["t_a"] and man["participants"] == ["s01", "s02"]
    assert "numpy" in man["versions"] and man["statuses"]["t_a"]["s01"] == "ran"


def test_stage_groups_keep_studies_apart(study):
    """A study runs only its pipeline.groups; a stage of another group is refused."""
    from dataclasses import replace

    from biomech.pipeline import resolve_order, stage

    @stage("t_v", scope="cohort", group="validation")
    def v(ctx):
        CALLS.append("v")
        return StageResult(tables={"x": pd.DataFrame({"a": [1]})})

    cfg = study({"t_a": {"k": 1}})
    assert "t_v" not in resolve_order(None, cfg.groups)
    with pytest.raises(ValueError, match="group"):
        resolve_order(["t_v"], cfg.groups)
    vcfg = replace(cfg, raw={**cfg.raw, "pipeline": {"groups": ["validation"]}})
    assert vcfg.groups == ("validation",)
    s = run(vcfg, ["t_v"])
    assert s.statuses["t_v"] == {COHORT_UNIT: "ran"} and "a:s01" not in CALLS
