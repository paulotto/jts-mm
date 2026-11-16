# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""hardware/mechanical/parts.yaml (biomech.hwparts) and the 3D-printed-parts paper items.

The manifest is valid against the committed files; unspecified usage falls back to the original
JTS renders (so the setup figures are unchanged until a study's versions are stated); stating
a version switches the setup render, the parts figure and the parts table; every form of the
parts items builds and renders.
"""

from __future__ import annotations

import copy
import re
import dataclasses

import pytest

from biomech import hwparts
from biomech.io.context import DataContext
from biomech.viz import render_pgf, style
from biomech.viz.spec import FigureSpec, TableSpec
from biomech.viz.views import hardware
from biomech.viz.views import registry as R
from biomech.viz.views import vocab as V

STUDIES = ("icorr2027", "jtsmm_validation")
MAN = hwparts.load()
# the design analyses of the parts are not in the public release: checks
# that need them are skipped there, and run in full where the folder exists
HAS_ANALYSES = (MAN.root / "analyses").is_dir()
pytestmark = pytest.mark.skipif(not MAN.parts, reason="no hardware/mechanical/parts.yaml")


def with_usage(study: str, **parts: str) -> hwparts.Manifest:
    doc = copy.deepcopy(MAN.doc)
    doc["usage"][study].update(parts)
    return hwparts.Manifest(doc, MAN.root)


def test_manifest_is_valid():
    errors, notes = MAN.problems(STUDIES)
    if not HAS_ANALYSES:
        errors = [e for e in errors if not re.fullmatch(r"analysis \S+: path analyses/\S+ missing", e)]
    assert not errors, errors
    assert set(MAN.usage) == set(STUDIES)


def test_every_variant_has_a_render_and_geometry():
    import json

    index = json.loads((MAN.root / "renders" / "index.json").read_text())
    for p in MAN.parts.values():
        for v in p.variants.values():
            assert v.stl is not None and v.stl.is_file(), (p.id, v.id)
            assert f"{p.id}/{v.id}" in index, (p.id, v.id)
            assert MAN.render(p.id, v.id).is_file(), (p.id, v.id)


def test_cad_backed_variants_name_a_body_of_their_part_file():
    import zipfile

    for p in MAN.parts.values():
        bodies = {v.fcstd_body for v in p.variants.values() if v.fcstd_body}
        if not bodies:
            continue
        with zipfile.ZipFile(p.fcstd) as z:
            doc = z.read("Document.xml").decode()
        for b in bodies:
            assert f'name="{b}"' in doc, (p.id, b)


def test_unspecified_usage_falls_back_to_the_original_renders():
    for study in STUDIES:
        for a in MAN.assemblies.values():
            if MAN.used(study, a.parts[0]) is None:
                assert MAN.assembly_render(study, a.id) == a.legacy_render
    assert MAN.assembly_render(None, "head_piece") == MAN.assemblies["head_piece"].legacy_render


def test_stated_usage_switches_render_and_overrides():
    m = with_usage("icorr2027", head_marker="qualisys_aruco")
    assert m.used("icorr2027", "head_marker").id == "qualisys_aruco"
    assert m.assembly_render("icorr2027", "head_piece") == m.render("head_marker",
                                                                     "qualisys_aruco")
    doc = copy.deepcopy(m.doc)
    doc["usage"]["overrides"] = {"icorr2027": {"s03": {"head_marker": "aruco_2026-08-18"}}}
    m2 = hwparts.Manifest(doc, MAN.root)
    assert m2.used("icorr2027", "head_marker", pid="s03").id == "aruco_2026-08-18"
    assert m2.used("icorr2027", "head_marker", pid="s01").id == "qualisys_aruco"


def test_titles_survive_the_reader_text():
    """Paper text replaces plan codes ("V5" -> "pre-specified"); no part, variant or figure
    title may look like one."""
    titles = [p.title for p in MAN.parts.values()]
    titles += [v.title for p in MAN.parts.values() for v in p.variants.values()]
    titles += [c for a in MAN.analyses.values() for c in (a.get("figures") or {})]
    assert [t for t in titles if V.reader_text(t) != t] == []


def test_candidates_are_variants():
    for p in MAN.parts.values():
        assert p.candidate in p.variants, p.id
        assert MAN.shown("icorr2027", p.id)[0] is not None or p.id not in MAN.usage["icorr2027"]


def test_unknown_usage_is_an_error():
    doc = copy.deepcopy(MAN.doc)
    doc["usage"]["icorr2027"]["head_marker"] = "v99"
    errors, _ = hwparts.Manifest(doc, MAN.root).problems(STUDIES)
    assert any("v99" in e for e in errors)


@pytest.fixture
def ctx_for(study):
    made = []

    def make(name: str):
        c = DataContext(dataclasses.replace(study(), study=name))
        made.append(c)
        return c

    yield make
    for c in made:
        c.close()


@pytest.mark.parametrize("study_id,fig,tab", [("icorr2027", "hw_parts", "hw_table"),
                                              ("jtsmm_validation", "val_hw_parts",
                                               "val_hw_table")])
def test_every_form_builds_renders_and_has_a_reader_name(ctx_for, tmp_path, study_id, fig, tab):
    ctx = ctx_for(study_id)
    for vid in (fig, tab):
        for form in R.get(vid).choice_values("form", ctx):
            if form in ("concepts", "incisor") and not HAS_ANALYSES:
                continue                                    # needs the design analyses
            assert V.choice_label("form", form) != form.replace("_", " "), form
            spec = R.build(vid, R.request(ctx, vid, form=form))
            assert spec.caption, (vid, form)
            if isinstance(spec, FigureSpec):
                assert all(any(s.kind == "image" for s in p.series) for p in spec.panels), form
                with style.use_venue("ieee"):
                    render_pgf.render_figure(spec, tmp_path / f"{vid}_{form}")
            else:
                assert isinstance(spec, TableSpec) and spec.rows, (vid, form)


def test_a_study_without_usage_lists_every_part_with_a_candidate(ctx_for):
    """A study missing from `usage` (the synthetic test study) still gets a full table."""
    ctx = ctx_for("synthetic")
    spec = R.build("hw_table", R.request(ctx, "hw_table", form="parts"))
    assert len(spec.rows) == sum(1 for p in MAN.parts.values() if p.candidate)
    fig = R.build("hw_parts", R.request(ctx, "hw_parts", form="used"))
    assert len(fig.panels) == len(spec.rows)


def test_unspecified_versions_are_said_in_the_caption(ctx_for, monkeypatch):
    ctx = ctx_for("icorr2027")
    # usage not yet specified (the repository manifest has it filled since 2026-11-04)
    none = with_usage("icorr2027", teeth_attachment=None, mouth_marker=None,
                      head_marker=None, calibration_tool=None)
    monkeypatch.setattr(hwparts, "load", lambda path=None: none)
    spec = R.build("hw_table", R.request(ctx, "hw_table", form="parts"))
    # the reader is told the versions are unconfirmed, without the manifest path
    assert "still to be confirmed" in spec.caption
    assert "parts.yaml" not in spec.caption
    m = with_usage("icorr2027", teeth_attachment="v5_5mm", mouth_marker="qualisys_aruco",
                   head_marker="qualisys_aruco", calibration_tool="qualisys_aruco")
    monkeypatch.setattr(hwparts, "load", lambda path=None: m)
    spec = R.build("hw_table", R.request(ctx, "hw_table", form="parts"))
    assert "still to be confirmed" not in spec.caption
    assert any(r["version"] == "Version 5, 5 mm" for r in spec.rows)
    assert hardware.usage_note("icorr2027")[0].startswith("Part versions:")
