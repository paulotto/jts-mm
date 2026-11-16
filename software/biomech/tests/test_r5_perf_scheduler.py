# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R5 scheduler (plan B2): detached stages as their own tasks, cohort stages released per
dependency, longest-first order -- the same store and statuses as before."""

from __future__ import annotations

import json
import threading
import time

import pandas as pd
import pytest
from joblib import parallel_config

from biomech import pipeline
from biomech.pipeline import StageResult, _TaskPool, run, stage
from biomech.results import COHORT_UNIT, ResultStore

EVENTS: list[dict] = []


@pytest.fixture(autouse=True)
def toy_stages():
    """t_a -> t_c -> t_s (cohort); t_a -> t_d (detached, slow) -> t_e -> t_u (cohort)."""
    pipeline.registry()
    saved = dict(pipeline._REGISTRY)
    EVENTS.clear()

    @stage("t_a", version="1")
    def a(ctx):
        if ctx.pid in (ctx.params.get("fail_for") or []):
            raise RuntimeError("boom")
        return StageResult(tables={"x": pd.DataFrame({"pid": [ctx.pid], "v": [ctx.params["k"]]})})

    @stage("t_c", depends=("t_a",))
    def c(ctx):
        return StageResult(tables={"y": ctx.upstream("t_a", "x").assign(v=lambda d: d.v * 10)})

    @stage("t_d", depends=("t_a",), detach=True)
    def d(ctx):
        time.sleep(ctx.params.get("sleep", 0.0))
        return StageResult(tables={"z": ctx.upstream("t_a", "x").assign(v=lambda d: d.v + 1)})

    @stage("t_e", depends=("t_d", "t_c"))
    def e(ctx):
        return StageResult(tables={"w": ctx.upstream("t_d", "z")})

    @stage("t_s", scope="cohort", depends=("t_c",))
    def s(ctx):
        return StageResult(tables={"sum": pd.DataFrame({"total": [ctx.upstream("t_c", "y").v.sum()]})})

    @stage("t_u", scope="cohort", depends=("t_e",))
    def u(ctx):
        return StageResult(tables={"n": pd.DataFrame({"n": [len(ctx.upstream("t_e", "w"))]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


def _store_dump(out) -> dict:
    st = ResultStore(out)
    dump = {}
    for name in st.stages():
        for unit in st.units(name):
            m = st.meta(name, unit)
            dump[(name, unit)] = (m.key, m.status, m.error, tuple(m.tables),
                                  {t: st.read_table(name, t, [unit]).to_json() for t in m.tables})
    return dump


def _run(cfg, n_jobs, **kw):
    if n_jobs == 1:
        return run(cfg, ["t_s", "t_u"], n_jobs=1, on_event=EVENTS.append, **kw)
    with parallel_config(backend="threading"):
        return run(cfg, ["t_s", "t_u"], n_jobs=n_jobs, on_event=EVENTS.append, **kw)


@pytest.mark.parametrize("fail", [[], ["s02"]])
def test_detached_tasks_give_the_serial_store_and_statuses(study, tmp_path, fail):
    from biomech.config import load_config

    cfg = study({"t_a": {"k": 2, "fail_for": fail}})
    serial = _run(cfg, 1)
    par_cfg = load_config(cfg.source, out_dir=tmp_path / "parallel")
    par = _run(par_cfg, 2)
    assert par.statuses == serial.statuses
    assert _store_dump(par_cfg.out_dir) == _store_dump(cfg.out_dir)
    if fail:     # the detached stage is skipped with the same stored reason as in a chain
        m = ResultStore(par_cfg.out_dir).meta("t_d", "s02")
        assert m.status == "skipped" and m.error == "upstream ['t_a'] failed"
        assert par.statuses["t_e"]["s02"] == "skipped"


def test_cohort_stage_does_not_wait_for_detached_stages(study):
    cfg = study({"t_a": {"k": 1}, "t_d": {"sleep": 1.0}})
    _run(cfg, 2)
    t = {(e["kind"], e.get("stage"), e.get("unit")): e["t"] for e in EVENTS}
    s_done = t[("unit_done", "t_s", COHORT_UNIT)]
    d_done = max(v for (k, st, _), v in t.items() if k == "unit_done" and st == "t_d")
    assert s_done < d_done                         # t_s ran while t_d was still running
    assert t[("unit_done", "t_u", COHORT_UNIT)] > d_done


def test_second_run_is_cached_and_nothing_is_resubmitted(study):
    cfg = study({"t_a": {"k": 1}})
    _run(cfg, 2)
    again = _run(cfg, 2)
    assert {u: s for st in again.statuses.values() for u, s in st.items()} == {
        "s01": "cached", "s02": "cached", COHORT_UNIT: "cached"}


def test_forced_head_failure_rewrites_a_cached_detached_unit_as_skipped(study, tmp_path):
    """As in one chain: a cached downstream unit whose upstream fails in this run is stored
    as skipped -- the detached task is still submitted for it."""
    from biomech.config import load_config

    good = pipeline._REGISTRY["t_a"]

    def boom(ctx):
        raise RuntimeError("boom")

    for sub, n_jobs in (("serial", 1), ("parallel", 2)):
        base = study({"t_a": {"k": 1}})
        cfg = load_config(base.source, out_dir=tmp_path / sub)
        pipeline._REGISTRY["t_a"] = good
        first = _run(cfg, n_jobs)
        assert first.statuses["t_d"] == {"s01": "ran", "s02": "ran"}
        pipeline._REGISTRY["t_a"] = pipeline.Stage("t_a", boom, "participant", (), "1", "")
        s = _run(cfg, n_jobs, force=["t_a"])
        assert s.statuses["t_a"] == {"s01": "failed", "s02": "failed"}, sub
        assert s.statuses["t_d"] == {"s01": "skipped", "s02": "skipped"}, sub
        assert ResultStore(cfg.out_dir).meta("t_d", "s01").status == "skipped"


def test_cancel_drops_detached_tasks_not_started(study):
    cfg = study({"t_a": {"k": 1}, "t_d": {"sleep": 0.3}})
    ev = threading.Event()

    def on_event(e):
        EVENTS.append(e)
        if e["kind"] == "unit_done" and e["stage"] == "t_c":
            ev.set()

    with parallel_config(backend="threading"):
        s = run(cfg, ["t_s", "t_u"], n_jobs=2, on_event=on_event, cancel=ev)
    assert s.cancelled
    assert all(v in ("ran", "cancelled") for v in s.statuses["t_d"].values())
    assert s.statuses["t_u"] == {COHORT_UNIT: "cancelled"}


def test_task_pool_runs_by_priority_then_order():
    pool = _TaskPool(1)
    for prio, tag in (((2, 0.0), "tail"), ((1, -5.0), "long"), ((1, -1.0), "short"),
                      ((1, -1.0), "short2"), ((0,), "cohort")):
        pool.add(prio, tag, lambda t=tag: t)
    got = []
    while pool.busy():
        got += [r for _, r in pool.wait()]
    assert got == ["cohort", "long", "short", "short2", "tail"]


def test_previous_elapsed_orders_longest_first(tmp_path):
    from biomech.pipeline import _previous_elapsed

    st = ResultStore(tmp_path)
    for pid, dt in (("s01", 3.0), ("s02", 9.0)):
        st.write("t_a", pid, "k", tables={}, elapsed_s=dt)
    est = _previous_elapsed(st, ["t_a", "t_c"], ["s01", "s02", "s03"])
    assert est(["t_a"], "s02") == 9.0 and est(["t_a"], "s03") == 6.0   # never run: stage mean
    assert est(["t_c"], "s01") == 0.0


def test_segment_writes():
    from biomech.pipeline import _segment_writes

    plan = [("t_d", "cached", "k1"), ("t_e", "cached", "k2")]
    assert not _segment_writes(plan, {"t_a": "ran", "t_c": "cached"})
    assert _segment_writes(plan, {"t_a": "failed", "t_c": "cached"})
    assert _segment_writes(plan, {"t_a": "cached", "t_c": "skipped"})
    assert _segment_writes([("t_d", "todo", "k")], {"t_a": "cached"})


def test_events_log_is_valid_json_lines(study):
    cfg = study({"t_a": {"k": 1}})
    s = _run(cfg, 2)
    path = ResultStore(cfg.out_dir).run_dir(s.run_id) / "events.jsonl"
    kinds = [json.loads(line)["kind"] for line in path.read_text().splitlines()]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"


def test_default_jobs(monkeypatch):
    from biomech import pipeline as P

    monkeypatch.setattr(P, "performance_cores", lambda: 8)
    assert [P.default_jobs(n) for n in (1, 4, 12, 14, 30)] == [1, 4, 12, 12, 12]
    monkeypatch.setattr(P, "performance_cores", lambda: None)
    assert P.default_jobs(100) >= 1 and P.default_jobs(0) == 1
