# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pipeline robustness: seed in the key, per-participant seeds, gc restores
an interrupted unit, out-of-segment dependencies are checked, coverage reasons."""

from __future__ import annotations

import os

import pandas as pd
import pytest

from biomech import pipeline
from biomech.pipeline import StageResult, participant_seed, run, stage, stage_status
from biomech.results import COHORT_UNIT, ResultStore

SEEDS: dict[str, int] = {}


@pytest.fixture(autouse=True)
def toy_stages():
    pipeline.registry()
    saved = dict(pipeline._REGISTRY)
    SEEDS.clear()

    @stage("r_a", version="1")
    def a(ctx):
        SEEDS[ctx.pid] = ctx.seed
        if ctx.params.get("fail_for") == ctx.pid:
            raise RuntimeError("boom")
        return StageResult(tables={"x": pd.DataFrame({"pid": [ctx.pid], "v": [1.0]})})

    @stage("r_c", depends=("r_a",))
    def c(ctx):
        return StageResult(tables={"y": ctx.upstream("r_a", "x")})

    @stage("r_s", scope="cohort", depends=("r_c",))
    def s(ctx):
        return StageResult(tables={"n": pd.DataFrame({"n": [len(ctx.participants)]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


def test_seed_enters_the_key_and_differs_per_participant(study, tmp_path):
    cfg = study({"r_a": {}})
    run(cfg, ["r_a"])
    assert SEEDS["s01"] != SEEDS["s02"] and SEEDS["s01"] == participant_seed(cfg.seed, "s01")
    import dataclasses

    cfg7 = dataclasses.replace(cfg, seed=cfg.seed + 1)
    assert set(stage_status(cfg7, ["r_a"]).status) == {"stale"}      # new seed -> re-run
    assert set(stage_status(cfg, ["r_a"]).status) == {"up-to-date"}


def test_gc_restores_a_unit_moved_aside_by_an_interrupted_write(tmp_path):
    store = ResultStore(tmp_path)
    store.write("st", "s01", "k1", tables={"t": pd.DataFrame({"a": [1]})})
    final = tmp_path / "cache" / "st" / "s01"
    old = final.parent / f".s01.old.{os.getpid()}.123"
    final.rename(old)                         # process died between the two renames
    assert store.units("st") == []
    store.gc()
    assert store.units("st") == ["s01"] and store.read_table("st", "t")["a"].item() == 1


def test_unrequested_upstream_is_checked_not_assumed(study):
    cfg = study({"r_a": {}})
    run(cfg, ["r_c"])
    cfg2 = study({"r_a": {"changed": 1}})      # r_a stale now, but only r_c is forced
    s = run(cfg2, ["r_c"], force=["r_c"])
    # r_a is a dependency of r_c, so it is part of the plan and re-runs first; r_c then
    # runs on the fresh r_a (never on a stale upstream).
    assert s.statuses["r_a"] == {"s01": "ran", "s02": "ran"}
    assert s.statuses["r_c"] == {"s01": "ran", "s02": "ran"}


def test_cohort_reruns_when_an_excluded_participant_recovers(study):
    s = run(study({"r_a": {"fail_for": "s02"}}), ["r_s"])
    store = ResultStore(study({"r_a": {"fail_for": "s02"}}).out_dir)
    assert s.statuses["r_s"] == {COHORT_UNIT: "ran"}
    assert store.read_table("r_s", "n")["n"].item() == 1
    s = run(study({"r_a": {}}), ["r_s"])
    assert s.statuses["r_s"] == {COHORT_UNIT: "ran"}                 # unit list changed
    assert store.read_table("r_s", "n")["n"].item() == 2
    assert not store.read_table("r_s", "coverage").used.eq(False).any()
