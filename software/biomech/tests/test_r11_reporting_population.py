# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""the reporting-population switch. A paper set with ``variant:`` reads the
store of that sensitivity variant (and its snapshots) for numbers, exports and checks; the one
resolver is :func:`biomech.sensitivity.store_config`.

A three-participant study with one participant flagged ``outside_protocol`` and two toy stages (a
participant stage and a cohort stage counting the participants) is run for the primary
analysis and the ``reporting_population`` variant, as in ``test_snapshot.py``; no real data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from biomech import papers, pipeline
from biomech import sensitivity as S
from biomech.cohort import draft_manifest
from biomech.config import ConfigError, load_config
from biomech.io.context import DataContext
from biomech.pipeline import StageResult, run, stage
from biomech.results import COHORT_UNIT, ResultStore
from biomech.viz import numbers as N

NUMBERS = {
    "cohortN": {"source": "t_b:sum", "column": "n", "fmt": "d", "unit": ""},
    "total": {"source": "t_b:sum", "column": "total", "n": "n", "fmt": ".0f", "unit": ""},
    # the n = all value next to it (D35: the all-participant robustness row)
    "totalAll": {"source": "sensitivity:reporting_population", "column": "primary", "n": 3,
                 "filter": {"stage": "t_b", "table": "sum", "column": "total"},
                 "fmt": ".0f", "unit": ""},
}


@pytest.fixture(autouse=True)
def toy_stages():
    pipeline.registry()
    saved = dict(pipeline._REGISTRY)

    @stage("t_a", version="1")
    def a(ctx):
        return StageResult(tables={"x": pd.DataFrame({"pid": [ctx.pid], "v": [int(ctx.pid[1:])]})},
                           arrays={"sig": np.arange(3)})

    @stage("t_b", scope="cohort", depends=("t_a",))
    def b(ctx):
        x = ctx.upstream("t_a", "x")
        return StageResult(tables={"sum": pd.DataFrame({"total": [x["v"].sum()],
                                                        "n": [x["pid"].nunique()]})})

    yield
    pipeline._REGISTRY.clear()
    pipeline._REGISTRY.update(saved)


@pytest.fixture
def ran(tmp_path, data_dir):
    """(primary config, paper-set path): s03 is flagged outside_protocol; both stores run, compared."""
    doc = draft_manifest(data_dir, "t")
    for row in doc["participants"]:
        if row["pid"] == "s03":
            row["flags"] = ["outside_protocol"]
    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
    study = tmp_path / "study.yaml"
    study.write_text(yaml.safe_dump({"study": "t", "paths": {
        "data_dir": str(data_dir), "out_dir": "out", "cohort": "cohort.yaml"}}),
        encoding="utf-8")
    base = load_config(study)
    run(base, ["t_b"])
    run(S.variant_config(study, "reporting_population"), ["t_b"])
    S.write_comparison(base, "reporting_population")
    (tmp_path / "numbers_t.yaml").write_text(yaml.safe_dump(NUMBERS), encoding="utf-8")
    ps = tmp_path / "paper_t.yaml"
    ps.write_text("variant: reporting_population\nnumbers: numbers_t.yaml\n"
                  "targets: {journal: {venue: springer, out_dir: export}}\n"
                  "items: []\n", encoding="utf-8")
    return base, ps


def _numbers(cfg) -> dict[str, N.Number]:
    ctx = DataContext(cfg)
    try:
        return {n.key: n for n in N.resolve(ctx, N.load(cfg.source.parent / "numbers_t.yaml"))}
    finally:
        ctx.close()


def _pin(ps, snapshot: str) -> None:
    ps.write_text(f"snapshot: {snapshot}\n" + ps.read_text(encoding="utf-8"), encoding="utf-8")


def test_paper_set_with_variant_resolves_every_number_from_the_variant_store(ran):
    base, path = ran
    ps = papers.load(path)
    assert ps.variant == "reporting_population" and ps.snapshot is None
    cfg = ps.store_config(base)
    assert cfg.variant == "reporting_population" and cfg.out_dir == S.variant_dir(base, "reporting_population")
    assert S.primary_out_dir(cfg) == base.out_dir
    assert DataContext(cfg).pids() == ["s01", "s02"]                 # the flagged participant left out
    nums = _numbers(cfg)
    assert nums["cohortN"].text == "2" and nums["total"].text == "3" and nums["total"].n == 2
    assert nums["totalAll"].text == "6"                              # primary (n = 3) value
    # the primary store, for comparison (and the override "primary")
    prim = _numbers(ps.store_config(cfg, variant=S.PRIMARY))
    assert prim["cohortN"].text == "3" and prim["total"].text == "6"
    assert prim["totalAll"].text == "6"          # the same comparison, frozen in the primary
    # the comparison is found at the variant root and under sensitivity/ of the primary
    assert S.comparison_path(cfg, "reporting_population") == cfg.out_dir / S.COMPARISON_FILE
    assert S.comparison_path(base, "reporting_population") == \
        base.out_dir / "sensitivity" / "reporting_population" / S.COMPARISON_FILE
    # another variant's comparison (of the primary run): the live variant store reads the
    # primary's copy
    assert S.comparison_path(cfg, "lowfreq_mask") == \
        base.out_dir / "sensitivity" / "lowfreq_mask" / S.COMPARISON_FILE


def test_variant_snapshot_is_frozen_with_its_cohort_and_comparison(ran):
    from biomech.cli import main

    base, path = ran
    other = base.out_dir / "sensitivity" / "lowfreq_mask" / S.COMPARISON_FILE
    other.parent.mkdir(parents=True)
    other.write_bytes(b"primary comparison")
    r = CliRunner().invoke(main, ["snapshot", "--config", str(base.source), "--variant",
                                  "reporting_population", "ra", "--note", "adults"])
    assert r.exit_code == 0, r.output
    snap_dir = S.variant_dir(base, "reporting_population") / "snapshots" / "ra"
    assert (snap_dir / "comparison.parquet").is_file()
    # the primary's comparisons are frozen with it (read there from now on)
    frozen_other = snap_dir / "sensitivity" / "lowfreq_mask" / S.COMPARISON_FILE
    assert frozen_other.read_bytes() == b"primary comparison"
    assert S.comparison_path(S.store_config(base, "reporting_population", "ra"), "lowfreq_mask") == \
        frozen_other
    frozen = yaml.safe_load((snap_dir / "cohort.yaml").read_text(encoding="utf-8"))
    assert [p["pid"] for p in frozen["participants"] if p.get("include", True)] == \
        ["s01", "s02"]
    r = CliRunner().invoke(main, ["snapshot", "--config", str(base.source), "--variant",
                                  "reporting_population", "--list"])
    assert r.exit_code == 0 and r.output.startswith("ra ")
    _pin(path, "ra")
    ps = papers.load(path)
    cfg = ps.store_config(base)
    assert cfg.snapshot == "ra" and cfg.read_dir == snap_dir and cfg.variant == "reporting_population"
    # a later change of the live variant store does not reach the pinned numbers
    ResultStore(S.variant_dir(base, "reporting_population")).write(
        "t_b", COHORT_UNIT, "changed", tables={"sum": pd.DataFrame({"total": [99], "n": [2]})})
    nums = _numbers(cfg)
    assert nums["total"].text == "3" and nums["cohortN"].text == "2"
    assert nums["totalAll"].text == "6"
    assert _numbers(ps.store_config(base, "live"))["total"].text == "99"
    # the same resolution through the CLI (dry run: nothing written)
    r = CliRunner().invoke(main, ["numbers", "--config", str(base.source), "--paper",
                                  str(path), "--dry-run"])
    assert r.exit_code == 0, r.output
    assert "reporting_population store, snapshot ra" in r.output and "\\numTotal" in r.output
    assert not (path.parent / "export" / "numbers.tex").exists()


def test_store_config_errors_name_the_problem(ran):
    base, path = ran
    _pin(path, "r9")                        # a primary snapshot name, not one of the variant
    with pytest.raises(FileNotFoundError, match="biomech snapshot --variant reporting_population"):
        papers.load(path).store_config(base)
    with pytest.raises(ConfigError, match="unknown sensitivity variant"):
        S.store_config(base, "nope")
    with pytest.raises(ConfigError, match="has no results"):
        S.store_config(base, "lowfreq_mask")          # known, never run
    assert not S.variant_dir(base, "lowfreq_mask").exists()     # and nothing written
    assert S.store_config(base) is base and S.store_config(base, S.PRIMARY, "live") is base
    with pytest.raises(ValueError, match="variant must name"):
        path.write_text("variant: [a]\nitems: []\n", encoding="utf-8")
        papers.load(path)


def test_number_checks_bound_the_paper_by_its_population(ran):
    from biomech.check import outputs as C

    base, path = ran
    rep = C.Report("t")
    ctx = DataContext(base)
    try:
        C.check_numbers_n(rep, ctx)
    finally:
        ctx.close()
    items = [i for i in rep.items if i.area == "n"]
    assert items and all(i.status == "pass" for i in items), [i.detail for i in items]
    assert "reporting_population: 2" in items[0].name and "analysis 3" in items[0].name

    def num(key, n, **kw):
        return N.Number(key, 1.0, "1.0", n=n, source=kw.pop("source", "a:b"), **kw)

    # the paper's own numbers: at most the variant's cohort; the primary column of a
    # comparison: at most the analysis of record
    nums = [num("ok", 2), num("tooMany", 3),
            num("allOk", 3, source="sensitivity:reporting_population", column="primary"),
            num("allBad", 4, source="sensitivity:reporting_population", column="primary"),
            num("variantBad", 3, source="sensitivity:reporting_population", column="variant")]
    assert C.number_n_issues(nums, 2, 2, None, 3) == [
        "tooMany: n = 3 > 2 participants",
        "allBad: n = 4 > 3 participants of the analysis",
        "variantBad: n = 3 > 2 adults"]


def test_exports_record_and_check_their_population(ran, tmp_path):
    from biomech.viz import serialize

    base, path = ran
    cfg = papers.load(path).store_config(base)
    prov = serialize.provenance(DataContext(cfg))
    assert prov["variant"] == "reporting_population" and prov["snapshot"] == "live"
    assert serialize.provenance(DataContext(base))["variant"] is None
    a, b = tmp_path / "ea", tmp_path / "eb"
    for d, v in ((a, "reporting_population"), (b, None)):
        d.mkdir()
        (d / "numbers.json").write_text(json.dumps({"provenance": {"variant": v},
                                                    "numbers": {}, "cited": {}}))
    tex = tmp_path / "tex"
    tex.mkdir()
    (tex / "main.tex").write_text("text")
    r = N.check_manuscript(tex, [a, b])
    assert r["populations"] == {str(a): "reporting_population", str(b): "primary"}
    assert N.check_manuscript(tex, [a])["populations"] == {str(a): "reporting_population"}
