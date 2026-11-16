# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Integrity and completeness on the synthetic study: the export manifest and
captions, export <-> store verification with seeded mismatches, column-level completeness and
the complete table views."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from biomech.check import coverage as C
from biomech.check import integrity as I
from biomech.io.context import DataContext
from biomech.results.access import AccessLog, track
from biomech.viz import manifest as M
from biomech.viz.export import export

pytestmark = pytest.mark.slow

ITEMS = [{"id": "fig_kinematics", "_label": "kinematics", "_part": "core"},
         {"id": "table_reference", "_label": "tabReference", "_part": "supporting"},
         {"id": "tab_kinematics__hinge", "_label": "tabHinge", "_part": "supplement"}]


@pytest.fixture(scope="module")
def ctx(synth_study):
    c = DataContext(synth_study[0])
    yield c
    c.close()


@pytest.fixture(scope="module")
def exported(ctx, tmp_path_factory):
    out = tmp_path_factory.mktemp("export")
    res = export(ctx, ITEMS, out, previews=False, meta={"paper": "test", "target": "t"})
    assert not res["errors"], res["errors"]
    return out


def _copy(src, tmp_path):
    dst = tmp_path / "copy"
    shutil.copytree(src, dst)
    return dst


def test_manifest_and_captions(exported):
    man = M.read(exported)
    assert man["meta"]["paper"] == "test" and man["schema"] == M.SCHEMA
    by = {i["label"]: i for i in man["items"]}
    assert set(by) == {"kinematics", "tabReference", "tabHinge"}
    assert by["kinematics"]["kind"] == "figure" and by["tabHinge"]["kind"] == "table"
    assert by["kinematics"]["view"] == "fig_kinematics" and by["kinematics"]["part"] == "core"
    for i in man["items"]:
        assert i["files"] and all(len(h) == 64 for h in i["files"].values())
        assert i["caption"] and i["sources"]
    cap = (exported / "captions.tex").read_text()
    for lb in by:
        assert f"jtsmmcap@{lb}" in cap and f"jtsmmsrc@{lb}" in cap


def test_clean_export_verifies(exported, ctx):
    assert I.verify(exported, ctx) == []


def test_edited_file_is_caught(exported, ctx, tmp_path):
    out = _copy(exported, tmp_path)
    dat = next((out / "figures").rglob("*.dat"))
    dat.write_text(dat.read_text() + "\n1 2\n")
    assert any("changed after the export" in p for p in I.verify_files(out))


def _rehash(out, rel):
    man = M.read(out)
    for it in man["items"]:
        if rel in it["files"]:
            it["files"][rel] = M.sha256(out / rel)
    (out / "manifest.json").write_text(json.dumps(man))


def test_changed_values_are_caught(exported, ctx, tmp_path):
    out = _copy(exported, tmp_path)
    npz = out / "figures" / "fig_kinematics" / "spec.npz"
    arrays = dict(np.load(npz))
    k = next(k for k, v in arrays.items() if v.dtype.kind == "f" and v.size > 3)
    arrays[k] = arrays[k] + 1.0
    np.savez_compressed(npz, **arrays)
    _rehash(out, "figures/fig_kinematics/spec.npz")
    assert I.verify_files(out) == []                    # hashes consistent ...
    probs = I.verify_rebuild(out, ctx)                  # ... but not the results
    assert any("fig_kinematics" in p and "values differ" in p for p in probs)


def test_changed_caption_and_cell_are_caught(exported, ctx, tmp_path):
    out = _copy(exported, tmp_path)
    sj = out / "figures" / "fig_kinematics" / "spec.json"
    d = json.loads(sj.read_text())
    d["figure"]["caption"] += " Edited."
    sj.write_text(json.dumps(d))
    tj = out / "tables" / "tab_kinematics__hinge.json"
    t = json.loads(tj.read_text())
    row = t["table"]["rows"][0]
    key = next(k for k, v in row.items() if isinstance(v, float))
    row[key] += 0.5
    tj.write_text(json.dumps(t))
    for rel in ("figures/fig_kinematics/spec.json", "tables/tab_kinematics__hinge.json"):
        _rehash(out, rel)
    probs = " | ".join(I.verify_rebuild(out, ctx))
    assert "caption differs" in probs and "cells differ" in probs


def test_unlisted_export_is_caught(exported, tmp_path):
    out = _copy(exported, tmp_path)
    shutil.copytree(out / "figures" / "fig_kinematics", out / "figures" / "old_figure")
    assert any("not in the manifest" in p for p in I.verify_files(out))


def test_every_stored_table_has_a_complete_view(ctx):
    from biomech.viz.views import registry as R
    from biomech.viz.views.tables import declarations

    tables, _ = C.stored_outputs(ctx.store)
    decl = declarations()
    assert {f"{st}:{t}" for st, t in tables} <= set(decl)
    R._load()
    vid = "tab_kinematics__hinge"
    with track() as log:
        R.build(vid, R.request(ctx, vid))
    cols = C.stored_columns(ctx.store, {("kinematics", "hinge")})
    assert cols and {c for c in cols if not C.column_exempt(*c)} <= log.columns


def test_unused_columns_are_reported(ctx):
    tables, _ = C.stored_outputs(ctx.store)
    one = sorted(tables)[0]
    log = AccessLog(tables={one})                       # table "shown", no column read
    r = C.uncovered(ctx.store, log)
    missing = {(st, t) for st, t, _ in r["columns"]}
    assert one in missing
    assert all(c != "unit" for _, _, c in r["columns"])  # bookkeeping is exempt


@pytest.mark.skipif(shutil.which("lualatex") is None, reason="needs LuaLaTeX")
def test_size_check_finds_small_text_and_tall_figures(exported):
    import dataclasses

    from biomech.viz import style
    from biomech.viz.export import _Fig, check_figure_size

    v = style.venue("ieee")
    assert check_figure_size(exported, _Fig("fig_kinematics", "text"), v) == []
    strict = dataclasses.replace(v, min_pt=v.min_pt + 3, max_height_in=0.5)
    probs = " | ".join(check_figure_size(exported, _Fig("fig_kinematics", "text"), strict))
    assert "pt <" in probs and "in high >" in probs


def test_gallery_expands_every_form(ctx, tmp_path):
    from biomech import papers
    from biomech.viz.export import export_gallery, gallery_items
    from biomech.viz.views import registry as R

    entries = gallery_items(ctx, [{"id": "imu_agreement", "_label": "imu"},
                                  {"id": "fig_kinematics", "_label": "kin"}])
    forms = R.get("imu_agreement").choice_values("form", ctx)
    assert [e.get("form") for e in entries[:-1]] == list(forms)
    assert entries[0]["_label"] == "imuStrip" and entries[-1] == {"id": "fig_kinematics",
                                                                  "_label": "kin"}
    (tmp_path / "paper_g.yaml").write_text(
        "targets: {t: {venue: ieee, out_dir: out}}\nitems: [{id: imu_agreement, label: imu}]\n")
    res = export_gallery(ctx, papers.load(tmp_path / "paper_g.yaml"), compile_pdf=False,
                         previews=False)
    assert not res["errors"] and len(res["figures"]) == len(forms)
    assert res["target_dir"][0].endswith("out_gallery")


def test_long_table_headers_wrap_instead_of_widening_the_table(tmp_path):
    from biomech.viz import render_pgf as R
    from biomech.viz.spec import Column, TableSpec

    cols = [Column(f"c{i}", f"A rather long column header number {i}", "\\milli\\metre", 1, "S")
            for i in range(8)]
    t = TableSpec("wide", "", cols, [{f"c{i}": 1.0 for i in range(8)}], width="text")
    tex = R.render_table(t, tmp_path).read_text()
    assert tex.count("\\shortstack{") == 8 and "(\\si[per-mode=symbol]{\\milli\\metre})" in tex
    assert R._wrap_header("n", "", 13) == "n"                       # short headers unchanged


def test_coverage_matrix_names_the_artefacts_of_each_column(ctx, tmp_path):
    import csv

    log, errors = C.view_reads(ctx, kinds=("dashboard",))
    assert log.consumers and all(v for v in log.consumers.values())
    assert set(log.consumers) <= log.columns | set(log.consumers)
    n = C.write_matrix(log.consumers, tmp_path / "m.csv")
    rows = list(csv.DictReader((tmp_path / "m.csv").open()))
    assert len(rows) == n and rows[0].keys() >= {"stage", "table", "column", "artefacts"}
    hinge = [r for r in rows if r["stage"] == "kinematics" and r["table"] == "hinge"]
    assert hinge and all(int(r["n_artefacts"]) >= 1 for r in hinge)
