# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""View groups (Part 3 step 3): a study lists, exports and checks only the views of
its own groups (``pipeline.groups``: dataset | validation)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from biomech.config import load_config
from biomech.viz.spec import Column, TableSpec
from biomech.viz.views import registry as R


def _table(vid: str):
    def build(req):
        return TableSpec(vid, f"{vid} caption.", [Column("a", "A", align="l")],
                         [{"a": "x"}], width="column")
    return build


@pytest.fixture
def two_groups(monkeypatch):
    """A registry with one paper and one data view per group (nothing else)."""
    R._load()
    reg: dict[str, R.ViewDef] = {}
    monkeypatch.setattr(R, "REGISTRY", reg)
    for vid, kind, group in (("ds_paper", "paper", "dataset"), ("ds_data", "data", "dataset"),
                             ("val_paper", "paper", "validation"),
                             ("val_dash", "dashboard", "validation")):
        R.register(vid, kind, vid, group=group,
                   domain="validation" if group == "validation" else "")(_table(vid))
    return reg


def _study(tmp_path: Path, data_dir: Path, groups: list[str] | None):
    from biomech.cohort import draft_manifest

    (tmp_path / "cohort.yaml").write_text(yaml.safe_dump(draft_manifest(data_dir, "t")),
                                          encoding="utf-8")
    cfg = {"study": "t", "paths": {"data_dir": str(data_dir), "out_dir": "out",
                                   "cohort": "cohort.yaml"}}
    if groups is not None:
        cfg["pipeline"] = {"groups": groups}
    p = tmp_path / f"study_{'_'.join(groups or ['default'])}.yaml"
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return load_config(p)


def test_register_group_default_and_validation():
    R._load()
    assert all(v.group in R.GROUPS for v in R.views())
    assert {v.group for v in R.views()} >= {"dataset"}
    assert "validation" in R.DOMAINS
    with pytest.raises(ValueError, match="unknown group"):
        R.register("x_bad_group", "paper", "x", group="other")


def test_views_filter_by_groups(two_groups, tmp_path, data_dir):
    ids = lambda vs: sorted(v.id for v in vs)                          # noqa: E731
    assert ids(R.views()) == ["ds_data", "ds_paper", "val_dash", "val_paper"]
    assert ids(R.views(groups=("validation",))) == ["val_dash", "val_paper"]
    assert ids(R.views("paper", ("dataset",))) == ["ds_paper"]
    assert ids(R.views("paper", "validation")) == ["val_paper"]
    # a config (or a context with .config) stands for its study groups
    cfg_v = _study(tmp_path, data_dir, ["validation"])
    cfg_d = _study(tmp_path, data_dir, None)                       # default: dataset
    assert cfg_d.groups == ("dataset",)
    assert ids(R.views(groups=cfg_v)) == ["val_dash", "val_paper"]
    assert ids(R.views(groups=cfg_d)) == ["ds_data", "ds_paper"]

    class Ctx:
        config = cfg_v

    assert ids(R.views("paper", Ctx())) == ["val_paper"]
    assert set(R.by_domain(("paper", "dashboard"), cfg_v)) == {"validation"}
    assert [v.id for vs in R.by_domain(("paper", "dashboard"), cfg_d).values()
            for v in vs] == ["ds_paper"]


def test_viewer_lists_only_the_study_groups(two_groups, tmp_path, data_dir):
    pytest.importorskip("dash")
    from biomech.viewer.pages import analysis

    cfg = _study(tmp_path, data_dir, ["validation"])
    vals = {o["value"] for o in analysis.view_options(groups=cfg.groups)}
    assert vals == {"val_paper", "val_dash"}
    assert {o["value"] for o in analysis.view_options(groups=("dataset",))} == {"ds_paper"}


@pytest.mark.parametrize("groups, mine, other", [(["validation"], "val_paper", "ds_paper"),
                                                 (None, "ds_paper", "val_paper")])
def test_export_writes_only_the_study_groups(two_groups, tmp_path, data_dir, groups, mine,
                                             other):
    from biomech.io.context import DataContext
    from biomech.viz.export import all_items, export

    cfg = _study(tmp_path, data_dir, groups)
    ctx = DataContext(cfg)
    try:
        res = export(ctx, None, tmp_path / "paper", previews=False,
                     groups=ctx.config.groups)
        assert res["tables"] == [mine] and not res["errors"]
        # an explicit item of the other group is refused, the own one exported
        res = export(ctx, [other, mine], tmp_path / "paper2", previews=False)
        assert res["tables"] == [mine]
        assert len(res["errors"]) == 1 and res["errors"][0].startswith(f"{other}:")
        assert not (tmp_path / "paper2" / "tables" / f"{other}.tex").exists()
        ids = {i["id"] for i in all_items(ctx, ("paper", "dashboard", "data"), "default")}
        assert ids == ({"val_paper", "val_dash"} if groups else {"ds_paper", "ds_data"})
    finally:
        ctx.close()


def test_cli_export_paper_for_a_validation_study(two_groups, tmp_path, data_dir):
    from click.testing import CliRunner

    from biomech.cli import main

    cfg = _study(tmp_path, data_dir, ["validation"])
    paper = tmp_path / "paper_systems.yaml"
    paper.write_text(yaml.safe_dump({"items": ["val_paper"], "out_dir": "paper_systems"}),
                     encoding="utf-8")
    r = CliRunner().invoke(main, ["export", "--config", str(cfg.source), "--paper",
                                  str(paper), "-j", "1"])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "paper_systems" / "tables" / "val_paper.tex").is_file()
    paper.write_text(yaml.safe_dump({"items": ["ds_paper"], "out_dir": "p2"}), encoding="utf-8")
    r = CliRunner().invoke(main, ["export", "--config", str(cfg.source), "--paper",
                                  str(paper), "-j", "1"])
    assert r.exit_code != 0 and "a dataset view" in r.output


def test_coverage_and_figure_checks_use_the_study_groups(two_groups, tmp_path, data_dir):
    from biomech.check import coverage as COV
    from biomech.check import outputs as O
    from biomech.io.context import DataContext

    built: list[str] = []
    for vid, v in list(two_groups.items()):
        def b(req, _vid=vid, _fn=v.builder):
            built.append(_vid)
            return _fn(req)
        two_groups[vid] = R.ViewDef(**{**v.__dict__, "builder": b})
    cfg = _study(tmp_path, data_dir, ["validation"])
    ctx = DataContext(cfg)
    try:
        log, errors = COV.view_reads(ctx)
        assert not errors and set(built) == {"val_paper", "val_dash"}
        built.clear()
        rep = O.Report("t")
        O.check_figures(rep, ctx, tmp_path / "check", pgf=False)
        assert set(built) == {"val_paper", "val_dash"}
        assert all("ds_" not in i.name for i in rep.items)
        built.clear()
        COV.view_reads(ctx, groups=("dataset",))
        assert set(built) == {"ds_paper", "ds_data"}
    finally:
        ctx.close()


def test_stored_outputs_of_other_groups_are_not_required(tmp_path):
    import pandas as pd

    from biomech.check import coverage as COV
    from biomech.pipeline import registry
    from biomech.results import COHORT_UNIT, ResultStore

    store = ResultStore(tmp_path / "out")

    val = next(n for n, s in registry().items() if s.group == "validation")
    ds = next(n for n, s in registry().items() if s.group == "dataset")
    for st in (val, ds):
        store.write(st, COHORT_UNIT, key="k", tables={"t": pd.DataFrame({"a": [1.0]})},
                    run_id="r")
    t_all, _ = COV.stored_outputs(store)
    assert {s for s, _ in t_all} == {val, ds}
    t_v, _ = COV.stored_outputs(store, ("validation",))
    assert {s for s, _ in t_v} == {val}
    t_d, _ = COV.stored_outputs(store, ("dataset",))
    assert {s for s, _ in t_d} == {ds}
