# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Frozen result snapshots (results/snapshot.py)."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from biomech import pipeline
from biomech import sensitivity as S
from biomech.io.context import DataContext
from biomech.pipeline import StageResult, run, stage
from biomech.results import ResultStore
from biomech.results import snapshot as SN


@pytest.fixture(autouse=True)
def toy_stages():
    pipeline.registry()
    saved = dict(pipeline._REGISTRY)

    @stage("t_a", version="1")
    def a(ctx):
        return StageResult(tables={"x": pd.DataFrame({"pid": [ctx.pid], "v": [ctx.params["k"]]})},
                           arrays={"sig": np.arange(3) * ctx.params["k"]})

    @stage("t_b", scope="cohort", depends=("t_a",))
    def b(ctx):
        x = ctx.upstream("t_a", "x")
        return StageResult(tables={"sum": pd.DataFrame({"total": [x["v"].sum()]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


def test_snapshot_freezes_and_is_isolated_from_later_runs(study):
    cfg = study({"t_a": {"k": 2}})
    run(cfg, ["t_b"])
    p = SN.create(cfg, "first", note="k=2")
    meta = SN.info(cfg.out_dir, "first")
    assert meta["note"] == "k=2" and meta["runs"] and (p / "cache" / "t_a" / "s01").is_dir()
    f = p / "cache" / "t_a" / "s01" / "x.parquet"
    live = cfg.out_dir / "cache" / "t_a" / "s01" / "x.parquet"
    assert os.stat(f).st_ino != os.stat(live).st_ino and os.stat(f).st_nlink == 1  # a copy
    assert f.read_bytes() == live.read_bytes()
    cfg2 = study({"t_a": {"k": 5}})
    run(cfg2, ["t_b"])                                            # live store changes
    assert ResultStore(cfg2.out_dir).read_table("t_b", "sum")["total"].item() == 10
    snap = DataContext(cfg2.with_snapshot("first"))
    assert snap.table("t_b", "sum")["total"].item() == 4           # frozen values
    assert snap.arrays("t_a", "s01")["sig"].tolist() == [0, 2, 4]
    assert cfg2.with_snapshot("first").snapshot == "first" and cfg2.with_snapshot(None) is cfg2
    d = S.compare(ResultStore(SN.path(cfg2.out_dir, "first")), ResultStore(cfg2.out_dir))
    row = d[(d.table == "sum") & (d.column == "total")].iloc[0]
    assert row.primary == 4 and row.variant == 10


def test_snapshot_names_and_errors(study):
    cfg = study({"t_a": {"k": 1}})
    run(cfg, ["t_a"])
    SN.create(cfg, "a")
    with pytest.raises(FileExistsError):
        SN.create(cfg, "a")
    SN.create(cfg, "a", overwrite=True)
    for bad in ("", "live", "../x", ".hidden"):
        with pytest.raises(ValueError):
            SN.create(cfg, bad)
    with pytest.raises(FileNotFoundError, match="no snapshot"):
        cfg.with_snapshot("nope")
    assert SN.available(cfg.out_dir) == ["a"] and SN.path(cfg.out_dir, "live") == cfg.out_dir


def test_snapshot_reads_its_own_cohort_and_parameters(study):
    """A participant added (or a parameter changed) after the freeze does not appear in
    views on the snapshot."""
    import yaml

    cfg = study({"t_a": {"k": 2}})
    run(cfg, ["t_b"])
    SN.create(cfg, "frozen")
    doc = yaml.safe_load(cfg.cohort_path.read_text(encoding="utf-8"))
    for row in doc["participants"]:
        row["include"] = row["pid"] == "s01"            # live cohort shrinks after the freeze
    cfg.cohort_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    live = study({"t_a": {"k": 5}})
    snap = live.with_snapshot("frozen")
    assert DataContext(live).pids() == ["s01"]
    assert DataContext(snap).pids() == ["s01", "s02"]
    assert snap.params("t_a")["k"] == 2 and live.params("t_a")["k"] == 5
