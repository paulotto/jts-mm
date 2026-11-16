# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Export of every registered view (biomech export --all) and the viewer page split."""

from __future__ import annotations

import pytest

from biomech.io.context import DataContext
from biomech.viz.export import all_items
from biomech.viz.views import registry as R

pytestmark = pytest.mark.slow


def test_all_items_cover_every_view_perspective_and_participant(synth_study):
    ctx = DataContext(synth_study[0])
    items = all_items(ctx)
    ids = {i["id"] for i in items}
    # only the views of the study's groups (the synthetic study is a dataset study)
    assert ids == {v.id for v in R.views(groups=ctx) if v.kind in ("paper", "dashboard", "data")}
    for v in R.views(groups=ctx):
        mine = [i for i in items if i["id"] == v.id]
        if "pid" in v.params:
            assert {i.get("pid") for i in mine if i.get("perspective") != "cohort"} \
                == set(ctx.pids())
        assert {i.get("perspective") for i in mine} == (set(v.perspectives) or {None})
    one = all_items(ctx, ("data",), "default", [ctx.pids()[0]])
    assert all(i.get("perspective") is None and i.get("pid") in (None, ctx.pids()[0])
               for i in one)


def test_cli_export_all_writes_every_dashboard(synth_study, tmp_path):
    from click.testing import CliRunner

    from biomech.cli import main

    cfg = synth_study[0]
    r = CliRunner().invoke(main, ["export", "--config", str(cfg.source), "--all", "--kind",
                                  "dashboard", "--perspectives", "default", "--to",
                                  str(tmp_path)])
    assert r.exit_code == 0, r.output
    for v in R.views("dashboard", groups=cfg):
        assert any(p.name.startswith(v.id) for p in (tmp_path / "figures").iterdir()) or \
            (tmp_path / "tables" / f"{v.id}.tex").is_file(), v.id
    assert (tmp_path / "previews" / "timing_ensemble.png").is_file()
    r = CliRunner().invoke(main, ["export", "--config", str(cfg.source)])
    assert r.exit_code != 0 and "exactly one" in r.output


def test_analysis_page_lists_no_data_views(synth_study):
    pytest.importorskip("dash")
    from biomech.viewer.context import ViewerContext
    from biomech.viewer.pages import analysis

    page = analysis.layout(ViewerContext.from_study(synth_study[0].source))

    def find(node, cid):
        if getattr(node, "id", None) == cid:
            return node
        kids = getattr(node, "children", None)
        for k in (kids if isinstance(kids, (list, tuple)) else [kids] if kids is not None
                  else []):
            hit = find(k, cid)
            if hit is not None:
                return hit
        return None

    dd = find(page, "an-view")
    kinds = {R.get(o["value"]).kind for o in dd.options}
    assert kinds == {"paper", "dashboard"}
    # every listed view has a domain (Results-page section) and the domains are known
    assert all(R.get(o["value"]).domain in R.DOMAINS for o in dd.options)
    # a deep link selects its view at build time
    deep = analysis.layout(ViewerContext.from_study(synth_study[0].source),
                           search="?view=emg_psd&pid=s02")
    assert find(deep, "an-view").value == "emg_psd" and find(deep, "an-pid").value == "s02"
