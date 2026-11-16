# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pipeline/store robustness: stale upstream, code hash, plugins, atomic writes, locks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from joblib import parallel_config

from biomech import pipeline
from biomech.pipeline import StageResult, run, stage, stage_status
from biomech.results import COHORT_UNIT, ResultStore

CALLS: list[str] = []


@pytest.fixture(autouse=True)
def chain_stages():
    """t_a (participant) -> t_c (participant) -> t_s (cohort)."""
    pipeline.registry()
    saved = dict(pipeline._REGISTRY)
    CALLS.clear()

    @stage("t_a", version="1")
    def a(ctx):
        CALLS.append(f"a:{ctx.pid}")
        if ctx.pid in (ctx.params.get("fail_for") or []):
            raise RuntimeError("boom")
        return StageResult(tables={"x": pd.DataFrame({"v": [ctx.params["k"]]})})

    @stage("t_c", depends=("t_a",))
    def c(ctx):
        CALLS.append(f"c:{ctx.pid}")
        x = ctx.upstream("t_a", "x")
        return StageResult(tables={"y": pd.DataFrame({"v": x["v"] * 10})})

    @stage("t_s", scope="cohort", depends=("t_c",))
    def s(ctx):
        CALLS.append("s")
        y = ctx.upstream("t_c", "y")
        return StageResult(tables={"sum": pd.DataFrame({"total": [y["v"].sum()]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


def test_stale_upstream_blocks_cohort_stage(study):
    """Audit P1-1: a skipped participant unit keeps its old `ok` meta; the cohort stage
    must not be built on it (and then stay cached under the expected key for good)."""
    cfg = study({"t_a": {"k": 1}})
    run(cfg, ["t_s"])
    store = ResultStore(cfg.out_dir)
    assert store.read_table("t_s", "sum")["total"].item() == 20

    cfg2 = study({"t_a": {"k": 2, "fail_for": ["s02"]}})
    s = run(cfg2, ["t_s"])
    assert s.statuses["t_a"]["s02"] == "failed"
    assert s.statuses["t_c"]["s02"] == "skipped"       # recorded as a "skipped" unit
    # Fail policy (R4): the cohort stage runs on the participants whose upstream is
    # current -- never on s02's old ok result -- and records why s02 is left out.
    assert s.statuses["t_s"] == {COHORT_UNIT: "ran"}
    assert store.read_table("t_s", "sum")["total"].item() == 20      # s01 alone (k=2)
    cov = store.read_table("t_s", "coverage").set_index("pid")
    assert bool(cov.loc["s01", "used"]) and not bool(cov.loc["s02", "used"])
    assert "t_c skipped (upstream failed)" in cov.loc["s02", "reason"]

    cfg3 = study({"t_a": {"k": 2}})
    s = run(cfg3, ["t_s"])
    assert s.statuses["t_s"] == {COHORT_UNIT: "ran"}
    assert store.read_table("t_s", "sum")["total"].item() == 40


def test_stale_participant_upstream_blocks_participant_stage(study):
    cfg = study({"t_a": {"k": 1}})
    run(cfg, ["t_c"])
    store = ResultStore(cfg.out_dir)
    # Make t_a/s01 stale by hand (as after a code change that was not re-run).
    meta = store._unit_dir("t_a", "s01") / "meta.json"
    d = json.loads(meta.read_text())
    d["key"] = "outdated"
    meta.write_text(json.dumps(d))
    CALLS.clear()
    s = run(cfg, ["t_c"], ["s01"])
    # t_a is rebuilt first (it is in the order) under the key it should have; t_c's key
    # chains through that *expected* key, so it is unchanged and t_c stays cached.
    assert CALLS == ["a:s01"] and s.statuses["t_c"]["s01"] == "cached"
    # Without t_a in the order (only t_c requested and t_a stale), t_c must be skipped.
    d = json.loads(meta.read_text())
    d["key"] = "outdated"
    meta.write_text(json.dumps(d))
    reg = dict(pipeline._REGISTRY)
    try:
        pipeline._REGISTRY["t_c"] = pipeline.Stage("t_c", reg["t_c"].func, "participant",
                                                   ("t_a",), "1", "")
        CALLS.clear()
        s = pipeline._run_stages(["t_c"], [s_ for s_ in _cohort(cfg).included(["s01"])],
                                 _cohort(cfg), cfg, store, pipeline.KeyResolver(cfg, _cohort(cfg)),
                                 set(), 1, "r", lambda *a, **k: None, None, statuses := {},
                                 lambda st, u: store.status(
                                     st, u, pipeline.KeyResolver(cfg, _cohort(cfg)).key(st, u))
                                 == "up-to-date")
    finally:
        pipeline._REGISTRY.clear()
        pipeline._REGISTRY.update(reg)
    assert CALLS == [] and statuses["t_c"]["s01"] == "skipped"


def _cohort(cfg):
    from biomech.cohort import load_cohort

    return load_cohort(cfg.cohort_path, cfg.data_dir)


def test_code_hash_file_selection():
    files = {f.relative_to(root).as_posix() for root, f in pipeline.hashed_files()
             if root.name == "biomech"}
    assert "emg/pipeline.py" in files                 # analysis code with a clashing basename
    assert "pipeline.py" not in files and "cli.py" not in files
    assert not any(f.startswith(("viz/", "viewer/", "results/")) for f in files)
    assert "norms/norms.yaml" in files                # data that changes results
    jfiles = {f.relative_to(root.parent).as_posix() for root, f in pipeline.hashed_files()
              if root.parent.name == "jtsmm"}
    assert "config/catalog.py" in jfiles              # µV scaling


def test_failing_plugin_is_skipped(monkeypatch):
    class EP:
        name = "broken"

        def load(self):
            raise ValueError("bad plugin")

    monkeypatch.setattr(pipeline, "_PLUGINS_LOADED", False)
    monkeypatch.setattr(pipeline.importlib.metadata, "entry_points", lambda group: [EP()])
    with pytest.warns(UserWarning, match="broken"):
        reg = pipeline.registry()
    assert "kinematics" in reg


def test_namespace_package_roots(tmp_path):
    class NS:
        __path__ = [str(tmp_path)]

    assert pipeline._package_roots(NS()) == [tmp_path]


def test_missing_input_is_reported_not_crashing(study, data_dir):
    cfg = study({"t_a": {"k": 1}})
    for d in data_dir.iterdir():
        if d.name.startswith("session_s02_f_44_2025"):
            (d / "session.h5").unlink()
    st = stage_status(cfg, ["t_a"])
    assert set(st[st.unit == "s02"].status) == {"missing-input"}
    with pytest.raises(ValueError, match="missing session_s02"):
        run(cfg, ["t_a"])                            # run() refuses an invalid cohort up front


def test_parallel_workers(study):
    cfg = study({"t_a": {"k": 3}})
    with parallel_config(backend="threading"):
        s = run(cfg, ["t_s"], n_jobs=2)
    assert s.statuses["t_a"] == {"s01": "ran", "s02": "ran"}
    assert ResultStore(cfg.out_dir).read_table("t_s", "sum")["total"].item() == 60


# -- store -----------------------------------------------------------------------------
def test_phantom_temp_units_invisible_and_collected(tmp_path):
    store = ResultStore(tmp_path)
    store.write("st", "s01", "k1", tables={"t": pd.DataFrame({"a": [1]})})
    phantom = tmp_path / "cache" / "st" / ".s02.abc"
    phantom.mkdir()
    (phantom / "meta.json").write_text(json.dumps({"stage": "st", "unit": "s02", "key": "k",
                                                   "status": "ok", "created_utc": "",
                                                   "tables": ["t"]}))
    assert store.units("st") == ["s01"]
    assert len(store.read_table("st", "t")) == 1
    removed = store.gc()
    assert removed == [phantom] and not phantom.exists()


def test_replace_keeps_old_unit_when_rename_fails(tmp_path, monkeypatch):
    store = ResultStore(tmp_path)
    store.write("st", "s01", "k1", arrays={"x": np.arange(2)})
    real = Path.rename

    def flaky(self, target):
        if Path(target).name == "s01" and self.name.startswith(".s01.") and ".old." not in self.name:
            raise OSError("disk full")
        return real(self, target)

    monkeypatch.setattr(Path, "rename", flaky)
    with pytest.raises(OSError):
        store.write("st", "s01", "k2", arrays={"x": np.arange(5)})
    monkeypatch.setattr(Path, "rename", real)
    assert store.meta("st", "s01").key == "k1"
    assert store.read_arrays("st", "s01")["x"].tolist() == [0, 1]
    assert [p.name for p in (tmp_path / "cache" / "st").iterdir()] == ["s01"]


def test_run_lock(tmp_path):
    store = ResultStore(tmp_path)
    lock = store.acquire_run_lock("r1")
    with pytest.raises(RuntimeError, match="already active"):
        store.acquire_run_lock("r2")
    lock.release()
    # A lock left by a dead process is taken over.
    (tmp_path / ".run.lock").write_text(json.dumps({"pid": 2 ** 22 + 12345, "run_id": "dead"}))
    store.acquire_run_lock("r3").release()
    assert not (tmp_path / ".run.lock").exists()


def test_gc_refuses_during_live_run(tmp_path):
    store = ResultStore(tmp_path)
    (tmp_path / ".run.lock").write_text(json.dumps({"pid": os.getppid(), "run_id": "x"}))
    with pytest.raises(RuntimeError):
        store.gc()
