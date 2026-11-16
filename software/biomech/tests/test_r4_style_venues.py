# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Venue profiles (plan D1): IEEE / Springer / Nature widths and absolute font sizes, the same
geometry in every renderer, standalone upload PDFs, and the venue selection paths (export
argument, ``biomech export --venue``, ``venue:`` in a paper yaml)."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from biomech.viz import layout as L
from biomech.viz import render_mpl, render_pgf, style
from biomech.viz.spec import Axis, FigureSpec, Panel, Series

MM = 1 / 25.4
LUALATEX = shutil.which("lualatex") is not None


def _spec(width: str = "text", sid: str = "venue_fig") -> FigureSpec:
    x = np.linspace(-5, 5, 30)
    panels = [Panel(f"({t}) Panel {t}", Axis("Time", "\\milli\\second"), Axis("Angle", "\\degree"),
                    [Series("line", x, x * k, label="openers", role="group:openers"),
                     Series("line", x, -x * k, label="closers", role="group:closers"),
                     Series("hline", y=np.array([-2.0]), role="reference", text=["bias -2.0"])])
              for k, t in ((1, "a"), (2, "b"))]
    return FigureSpec(sid, "venue test", panels, cols=2, width=width, height_in=1.4)


def test_venue_profiles_map_width_and_fonts():
    assert style.venue().name == "ieee"                      # default unchanged
    ieee, spr, nat = (style.venue(n) for n in ("ieee", "springer", "nature"))
    assert (ieee.column_in, ieee.text_in) == (3.5, 7.16)
    assert spr.column_in == pytest.approx(85 * MM) and spr.text_in == pytest.approx(170 * MM)
    assert nat.column_in == pytest.approx(89 * MM) and nat.text_in == pytest.approx(183 * MM)
    assert ieee.font_pt["label"] == 8 and ieee.font_pt["tick"] == 7 and ieee.font_pt["note"] == 6
    assert min(spr.font_pt.values()) >= 7 and spr.family == "sans"
    assert min(nat.font_pt.values()) >= 5 and max(nat.font_pt.values()) <= 8
    for v in (ieee, spr, nat):
        for width in ("column", "text"):
            lay = L.compute(_spec(width), venue=v.name)
            assert lay.width == pytest.approx(v.width_in(width))
            assert lay.font["tick"] == v.font_pt["tick"] and lay.venue == v.name
    with pytest.raises(ValueError):
        style.venue("acm")


def test_use_venue_is_scoped():
    with style.use_venue("nature") as v:
        assert v.name == "nature" and L.compute(_spec()).width == pytest.approx(183 * MM)
        with style.use_venue(None):
            assert style.venue().name == "nature"
    assert style.venue().name == "ieee"
    assert L.compute(_spec()).width == pytest.approx(7.16)


@pytest.mark.parametrize("venue", ["ieee", "springer", "nature"])
def test_renderers_share_the_venue_geometry(tmp_path, venue):
    spec = _spec()
    v = style.venue(venue)
    b = render_mpl.build(spec, venue=venue)
    assert b.fig.get_figwidth() == pytest.approx(v.text_in)
    assert b.layout.font["label"] == v.font_pt["label"]
    assert render_mpl.text_overlaps(b) == [] and L.check(b.layout) == []
    render_pgf.write_style(tmp_path, venue)
    tex = render_pgf.render_figure(spec, tmp_path, venue=venue).read_text()
    m = re.search(r"% Size: ([\d.]+) x", tex)
    assert float(m.group(1)) == pytest.approx(v.text_in, abs=0.01)
    assert f"venue: {venue}" in tex


def test_pgf_fonts_are_absolute(tmp_path):
    for venue in ("ieee", "springer", "nature"):
        sty = render_pgf.write_style(tmp_path, venue).read_text()
        v = style.venue(venue)
        lab = v.font_pt["label"]
        assert f"\\def\\jtsmmfontlabel{{\\fontsize{{{lab:g}}}" in sty
        assert f"\\def\\jtsmmfontnote{{\\fontsize{{{v.font_pt['note']:g}}}" in sty
        assert "\\bfseries" in sty.split("\\def\\jtsmmfonttag", 1)[1].split("\n", 1)[0]
        assert ("\\sffamily" in sty) == (v.family == "sans")
        assert "\\footnotesize" not in sty and "\\scriptsize" not in sty
    render_pgf.write_style(tmp_path, "ieee")
    tex = render_pgf.render_figure(_spec(), tmp_path).read_text()
    assert "\\footnotesize" not in tex and "\\scriptsize" not in tex
    assert "\\jtsmmfigsetup" in tex


@pytest.mark.skipif(not LUALATEX, reason="no LuaLaTeX")
def test_no_size_drift_with_a_12pt_document(tmp_path):
    """The same figure in a 10-pt and a 12-pt document has the same natural size."""
    from biomech.viz.export import _compile, pdf_size_in

    spec = _spec("column")
    render_pgf.write_style(tmp_path)
    render_pgf.render_figure(spec, tmp_path)
    sizes = []
    for pt in (10, 12):
        doc = (f"\\documentclass[{pt}pt]{{standalone}}\n\\def\\jtsmmnaturalbbox{{}}\n"
               + render_pgf.PREAMBLE
               + f"\\begin{{document}}\\input{{figures/{spec.id}/fig.tex}}\\end{{document}}\n")
        tex = tmp_path / f"doc{pt}.tex"
        tex.write_text(doc)
        ok, log = _compile(tex)
        assert ok, log[-800:]
        sizes.append(pdf_size_in(tex.with_suffix(".pdf")))
    if None in sizes:
        pytest.skip("no pdfinfo")
    assert sizes[0] == pytest.approx(sizes[1], abs=0.01)


@pytest.mark.skipif(not LUALATEX, reason="no LuaLaTeX")
@pytest.mark.parametrize("venue", ["ieee", "springer"])
def test_standalone_upload_pdf(tmp_path, venue):
    from biomech.viz.export import check_figure_size, pdf_size_in, standalone_pdf

    spec = _spec()
    render_pgf.write_style(tmp_path, venue)
    render_pgf.render_figure(spec, tmp_path, venue=venue)
    pdf, probs = standalone_pdf(tmp_path, spec, venue)
    assert probs == [] and pdf == tmp_path / "figures" / spec.id / f"{spec.id}.pdf"
    size = pdf_size_in(pdf)
    if size is not None:                      # page = exactly the venue's text width
        assert size[0] == pytest.approx(style.venue(venue).text_in, abs=2 / 72)
    assert not list(tmp_path.glob("upload_*"))  # no leftovers next to the export
    assert check_figure_size(tmp_path, spec, venue) == []
    if shutil.which("pdffonts"):
        import subprocess

        fonts = subprocess.run(["pdffonts", str(pdf)], capture_output=True, text=True).stdout
        assert ("Heros" in fonts) == (venue == "springer")
        assert ("Termes" in fonts) == (venue == "ieee")


def test_preview_uses_the_venue_page(tmp_path):
    t = render_pgf.standalone_preview(tmp_path, [], venue="springer").read_text()
    assert "a4paper" in t and "textwidth=6.6929in" in t and "TeX Gyre Heros" in t
    t = render_pgf.standalone_preview(tmp_path, []).read_text()
    assert "letterpaper" in t and "columnsep=0.24in" in t


def test_cli_venue_from_paper_yaml_and_flag(tmp_path, monkeypatch):
    from click.testing import CliRunner

    import biomech.cli as cli
    import biomech.io.context as ctxmod
    import biomech.viz.export as ex

    calls = []

    class _Cfg:
        out_dir = tmp_path

    monkeypatch.setattr(cli, "_load", lambda *a, **k: _Cfg())
    monkeypatch.setattr(ctxmod, "DataContext", lambda cfg: object())

    def fake_export(ctx, items, target, **kw):
        calls.append(kw)
        return {"figures": [], "tables": [], "errors": [], "problems": [], "standalone": []}

    monkeypatch.setattr(ex, "export", fake_export)
    study = tmp_path / "study.yaml"
    study.write_text("study: x\n")
    paper = tmp_path / "paper.yaml"
    paper.write_text("venue: springer\nitems: [fig_x]\n")
    r = CliRunner().invoke(cli.main, ["export", "--config", str(study), "--paper", str(paper),
                                      "--to", str(tmp_path / "o"), "--standalone"])
    assert r.exit_code == 0, r.output
    assert calls[-1]["venue"] == "springer" and calls[-1]["standalone"] is True
    r = CliRunner().invoke(cli.main, ["export", "--config", str(study), "--paper", str(paper),
                                      "--to", str(tmp_path / "o"), "--venue", "nature"])
    assert r.exit_code == 0 and calls[-1]["venue"] == "nature"
    paper.write_text("items: [fig_x]\n")
    r = CliRunner().invoke(cli.main, ["export", "--config", str(study), "--paper", str(paper),
                                      "--to", str(tmp_path / "o")])
    assert r.exit_code == 0 and calls[-1]["venue"] == "ieee"


def test_paper_yamls_name_their_venue():
    """The paper sets name a venue per target, list only registered views and pin a
    snapshot that exists (the legacy paper_icorr / paper_journal sets were removed, R8)."""
    from biomech import papers
    from biomech.viz.views import registry as R

    root = Path(__file__).resolve().parents[3] / "experiments"
    ds = root / "icorr2027" / "paper_dataset.yaml"
    if not ds.is_file():
        pytest.skip("not in the repository checkout")
    ps = papers.load(ds)
    assert ps.targets["journal"].venue == "springer"
    assert ps.targets["conference"].venue == "ieee"
    known = {v.id for v in R.views()}
    assert {i.view_id for i in ps.items} <= known
    assert not (root / "icorr2027" / "paper_icorr.yaml").exists()
    store = root.parent / "data" / "analysis" / "icorr2027" / "sensitivity" / "reporting_population"
    if ps.snapshot and (store / "snapshots").is_dir():
        assert (store / "snapshots" / ps.snapshot / "snapshot.json").is_file(), \
            f"pinned snapshot {ps.snapshot} missing"
