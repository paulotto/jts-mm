# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""G3: exported spec files re-render every figure and table identically, without the store."""

from __future__ import annotations

import filecmp
import json

import numpy as np
import pytest

from biomech.viz import render_pgf, serialize
from biomech.viz.spec import Axis, Colorbar, FigureSpec, Panel, Series, TableSpec


def _same_tree(a, b) -> list[str]:
    diffs = []
    for p in sorted(a.rglob("*")):
        if p.is_file():
            q = b / p.relative_to(a)
            if not q.is_file() or not filecmp.cmp(p, q, shallow=False):
                diffs.append(str(p.relative_to(a)))
    return diffs


def test_figure_round_trip_is_lossless(tmp_path):
    s = [Series("line", np.arange(5.0), np.r_[1.0, np.nan, 3, 4, 5], label="a", role="cat:0"),
         Series("image", img=np.random.default_rng(0).random((3, 4)), extent=(0, 1, 0, 1),
                style={"cmap": "sequential", "lim": (0.0, 1.0),
                       "mask": np.eye(3, 4, dtype=bool)}),
         Series("text", np.r_[0.5], np.r_[0.5], text=["−1"])]
    spec = FigureSpec("rt", "Round trip", [Panel("(a) x", Axis("Time", "\\second", lim=(0, 4)),
                                                 Axis("y", log=False), s,
                                                 colorbar=Colorbar("c", lim=(0.0, 1.0)))],
                      caption="cap", sources=["x:y"], meta={"n": np.int64(3), "f": np.nan})
    serialize.write_figure(spec, tmp_path / "f", {"snapshot": "live"})
    back = serialize.load_figure(tmp_path / "f" / "spec.json")
    assert back.panels[0].x.lim == (0, 4) and isinstance(back.panels[0].x.lim, tuple)
    assert back.panels[0].series[1].extent == (0, 1, 0, 1)
    np.testing.assert_array_equal(back.panels[0].series[1].style["mask"], np.eye(3, 4, dtype=bool))
    np.testing.assert_array_equal(back.panels[0].series[0].y, spec.panels[0].series[0].y)
    assert back.meta["n"] == 3 and np.isnan(back.meta["f"])
    assert back.panels[0].colorbar.lim == (0.0, 1.0)
    rows = (tmp_path / "f" / "source_data.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("# rt") and len(rows) == 2 + 5 + 1      # header, 5 + 1 points



def test_heatmap_source_data_has_one_row_per_cell(tmp_path):
    """2026-11-11 (JNER release folder): a heatmap's matrix is source data too; cells are
    named by the axis tick labels where the axis has them, else by their coordinate."""
    z = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]])
    spec = FigureSpec("hm", "Heat", [Panel("(a) map",
                                           Axis("", ticks=[0.0, 1.0, 2.0],
                                                ticklabels=["m1", "m2", "m3"]),
                                           Axis("Task"), [Series("heatmap", z=z)])])
    serialize.write_figure(spec, tmp_path / "f", {"snapshot": "live"})
    rows = (tmp_path / "f" / "source_data.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2 + 6
    head = rows[1].split(",")
    cells = [dict(zip(head, r.split(","))) for r in rows[2:]]
    assert [(c["x"], c["y"], c["z"]) for c in cells[:4]] == [
        ("m1", "0.0", "1.0"), ("m2", "0.0", "2.0"), ("m3", "0.0", "3.0"), ("m1", "1.0", "4.0")]
    assert cells[4]["z"] == "" and cells[4]["kind"] == "heatmap"


def test_table_round_trip(tmp_path):
    from biomech.viz.spec import Column

    t = TableSpec("tb", "Cap", [Column("m", "Metric", align="l"),
                                Column("v", "Value", "\\milli\\metre", 2)],
                  [{"m": "a", "v": 1.234}, {"m": "b", "v": float("nan")}], notes="n")
    cp, jp = serialize.write_table(t, tmp_path, {"snapshot": "r4"})
    back = serialize.load_table(jp)
    assert back.columns[1].unit == "\\milli\\metre" and back.rows[0]["v"] == 1.234
    assert np.isnan(back.rows[1]["v"])
    lines = cp.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "Metric,Value (mm)" and lines[2] == "a,1.234" and lines[3] == "b,"


@pytest.mark.slow
def test_every_paper_view_rerenders_identically(synth_study, tmp_path):
    from biomech.io.context import DataContext
    from biomech.viz.views import registry as R

    ctx = DataContext(synth_study[0])
    n = 0
    for v in R.views("paper") + R.views("dashboard") + R.views("data"):
        spec = R.build(v.id, R.request(ctx, v.id))
        if not isinstance(spec, FigureSpec):
            continue
        a, b = tmp_path / "a" / v.id, tmp_path / "b" / v.id
        render_pgf.render_figure(spec, a)
        serialize.write_figure(spec, tmp_path / "spec" / v.id)
        render_pgf.render_figure(serialize.load_figure(tmp_path / "spec" / v.id / "spec.json"), b)
        assert _same_tree(a, b) == [], v.id
        json.loads((tmp_path / "spec" / v.id / "spec.json").read_text(encoding="utf-8"))
        n += 1
    assert n >= 40
