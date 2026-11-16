# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J1: captions inside the exported PDFs (preview, captioned/<id>.pdf), the caption
rubric, no stale files after a re-export, and the integrity check of a --standalone export."""

from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pytest

from biomech.check import integrity as I
from biomech.check.text import caption_rubric_issues
from biomech.viz import caption as C
from biomech.viz import export as ex
from biomech.viz import render_pgf
from biomech.viz.spec import Axis, Column, FigureSpec, Panel, Series, TableSpec

LUALATEX = shutil.which("lualatex") is not None and shutil.which("pdftotext") is not None


def _fig(sid: str = "cap_fig") -> FigureSpec:
    x = np.linspace(0, 1, 20)
    spec = FigureSpec(sid, "", [Panel("Angle", Axis("Time", "\\second"), Axis("Angle", "\\degree"),
                                      [Series("line", x, 3 * x, label="P1")])],
                      width="column", height_in=1.2)
    return C.attach(spec, C.CaptionSpec(
        "Opening angle of one participant over one open-close cycle",
        marks=[("line", "participant P1")], n="n = 1 participant",
        statistic="raw signal, no summary", evidence="descriptive"))


def _tab(notes: str = "") -> TableSpec:
    spec = TableSpec("cap_tab", "", [Column("what", "Quantity", align="l"),
                                     Column("v", "RMSE", unit="\\milli\\metre", align="S",
                                            decimals=1)],
                     [{"what": "incisor", "v": 1.234}], notes=notes)
    return C.attach(spec, C.CaptionSpec("Agreement of the incisor position with the reference",
                                        n="n = 2 participants", statistic="median"))


def test_preview_carries_the_real_caption(tmp_path):
    t = render_pgf.standalone_preview(
        tmp_path, [("figure", "cap_fig"), ("table", "cap_tab")],
        captions={"cap_fig": ("Opening angle shown.", "kinematics:arrays")},
        prov={"snapshot": "r99", "variant": "reporting_population", "git": {"sha": "abcdef123456"}}
    ).read_text()
    assert "\\caption[cap\\_fig]{Opening angle shown.}" in t
    assert "Sources: kinematics:arrays" in t and "Data: r99, code" in t
    assert "adults" not in t        # the footer names the snapshot, not the variant folder
    assert "\\caption[cap\\_tab]{cap\\_tab}" in t          # no caption: the id as before


@pytest.mark.skipif(not LUALATEX, reason="needs lualatex and pdftotext")
def test_captioned_pdfs_contain_caption_sources_and_provenance(tmp_path):
    fig, tab = _fig(), _tab()
    render_pgf.write_style(tmp_path, "ieee")
    render_pgf.render_figure(fig, tmp_path, venue="ieee")
    render_pgf.render_table(tab, tmp_path)
    prov = {"snapshot": "r99", "git": {"sha": "abcdef123456"}}
    for kind, spec in (("figure", fig), ("table", tab)):
        pdf, probs = ex.captioned_pdf(tmp_path, kind, spec.id, spec.caption, ["src:tab"], prov,
                                      spec.width, "ieee", label="lbl")
        assert probs == [] and pdf == tmp_path / "captioned" / f"{spec.id}.pdf"
        text = " ".join(subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                                       text=True).stdout.split())
        assert spec.caption.split(".")[0] in text
        assert "src:tab" in text and "r99" in text and spec.id in text
    assert not list(tmp_path.glob("captioned_*"))            # no build leftovers


def test_caption_rubric():
    assert caption_rubric_issues(_fig()) == ["caption short (21 words < 25)"]
    long = _fig()
    long.caption = "word " * 300
    assert any("caption long" in i for i in caption_rubric_issues(long))
    bare = C.attach(_fig(), C.CaptionSpec("x " * 30))
    iss = caption_rubric_issues(bare)
    assert "no n" in iss and "encodings not stated (marks)" in iss and "no statistic" in iss
    assert "table abbreviations not spelled out: RMSE" in caption_rubric_issues(_tab())
    assert not any("RMSE" in i for i in caption_rubric_issues(_tab("RMSE: root mean square")))
    sym = _tab()
    sym.columns[1].header = "$\\rho$"
    assert "table symbols without a note" in caption_rubric_issues(sym)


def test_reexport_removes_stale_figure_files(tmp_path, monkeypatch):
    from biomech.viz.views import registry as R

    monkeypatch.setattr(R, "build", lambda vid, req: _fig())
    monkeypatch.setattr(R, "request", lambda ctx, vid, **kw: None)
    fdir = tmp_path / "figures" / "cap_fig"
    fdir.mkdir(parents=True)
    (fdir / "p9s9.dat").write_text("stale")
    (fdir / "cap_fig.pdf").write_text("upload of older results")
    res = ex._export_one(None, "cap_fig", tmp_path, ex.style.venue("ieee"), {}, False)
    assert res["errors"] == [] and (fdir / "fig.tex").is_file()
    assert not (fdir / "p9s9.dat").exists() and not (fdir / "cap_fig.pdf").exists()


def test_standalone_export_needs_upload_and_captioned_pdfs(tmp_path):
    (tmp_path / "figures" / "f").mkdir(parents=True)
    (tmp_path / "figures" / "f" / "fig.tex").write_text("x")
    from biomech.viz import manifest as M

    files = {"figures/f/fig.tex": M.sha256(tmp_path / "figures" / "f" / "fig.tex")}
    man = {"schema": 1, "meta": {"standalone": True},
           "items": [{"id": "f", "kind": "figure", "files": files}]}
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    probs = I.verify_files(tmp_path)
    assert any("captioned/f.pdf" in p for p in probs)
    assert any("figures/f/f.pdf" in p for p in probs)
    man["meta"]["standalone"] = False
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    assert I.verify_files(tmp_path) == []
