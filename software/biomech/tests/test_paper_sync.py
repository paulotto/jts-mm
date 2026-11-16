# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""biomech paper-sync: a paper export copied into a manuscript's results folder."""

from __future__ import annotations

import hashlib
import json

import pytest

from biomech.viz import numbers as N
from biomech.viz import paper_sync as P


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _export(root, items=("fig_a", "tab_b")):
    """A minimal export folder: one figure, one table, numbers, captions, style."""
    ex = root / "export"
    (ex / "figures" / "fig_a").mkdir(parents=True)
    (ex / "tables").mkdir()
    pdf, tex = b"%PDF-1.5 figure", b"\\begin{tabularx}{\\textwidth}{l}x\\end{tabularx}"
    (ex / "figures" / "fig_a" / "fig_a.pdf").write_bytes(pdf)
    (ex / "tables" / "tab_b.tex").write_bytes(tex)
    man = {"provenance": {"study": "s", "snapshot": "r1", "variant": "reporting_population"},
           "meta": {"paper_file": "paper_x.yaml", "target": "journal"}, "items": []}
    if "fig_a" in items:
        man["items"].append({"label": "figA", "id": "fig_a", "kind": "figure", "part": "core",
                             "section": "A", "files": {"figures/fig_a/fig_a.pdf": _sha(pdf)}})
    if "tab_b" in items:
        man["items"].append({"label": "tabB", "id": "tab_b", "kind": "table", "part": "core",
                             "section": "A", "files": {"tables/tab_b.tex": _sha(tex)}})
    (ex / "manifest.json").write_text(json.dumps(man))
    (ex / "numbers.tex").write_text("\\newcommand{\\numLeadX}{\\num{1}}\n")
    (ex / "numbers.json").write_text(json.dumps({"schema": 2, "provenance": man["provenance"],
                                                 "numbers": {"leadX": {"macro": "\\numLeadX",
                                                                       "value": 1.0}},
                                                 "cited": {}}))
    (ex / "captions.tex").write_text("% captions\n")
    (ex / "jtsmm-pgfstyle.tex").write_text(
        "\\pgfplotsset{compat=1.18}\n\\def\\jtsmmminus{\\ifmmode-\\else\\textminus\\fi}\n"
        "\\definecolor{jtblue}{HTML}{0072B2}\n")
    return ex


def test_sync_copies_by_label_and_is_idempotent(tmp_path):
    ex, dest = _export(tmp_path), tmp_path / "paper" / "results"
    r = P.sync(ex, dest)
    assert (dest / "figures" / "figA.pdf").read_bytes() == b"%PDF-1.5 figure"
    assert (dest / "tables" / "tabB.tex").is_file() and (dest / "numbers.tex").is_file()
    res = (dest / "results.tex").read_text()
    # pgfplots-free helpers, the full style only when pgfplots is loaded
    assert "\\def\\jtsmmminus" in res and "\\input{results/jtsmm-pgfstyle.tex}" in res
    assert "\\pgfplotsset" not in res.split("\\else")[1]
    mf = json.loads((dest / "manifest.json").read_text())
    assert mf["sync"]["generator"] == P.GENERATOR and {i["label"] for i in mf["items"]} == {
        "figA", "tabB"}
    assert "manifest.json" in r["written"]
    again = P.sync(ex, dest)                     # nothing changed: nothing written
    assert again["written"] == [] and again["removed"] == []


def test_sync_removes_only_its_own_stale_files(tmp_path):
    dest = tmp_path / "paper" / "results"
    P.sync(_export(tmp_path), dest)
    (dest / "notes.txt").write_text("mine")      # not ours: never touched
    ex2 = _export(tmp_path / "v2", items=("fig_a",))
    r = P.sync(ex2, dest)
    assert r["removed"] == ["tables/tabB.tex"] and not (dest / "tables" / "tabB.tex").exists()
    assert (dest / "notes.txt").read_text() == "mine"


def test_sync_refuses_a_changed_export_and_a_foreign_folder(tmp_path):
    ex = _export(tmp_path)
    (ex / "figures" / "fig_a" / "fig_a.pdf").write_bytes(b"edited")
    with pytest.raises(ValueError, match="differs from the export manifest"):
        P.sync(ex, tmp_path / "out")
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "image.png").write_bytes(b"x")
    with pytest.raises(FileExistsError):
        P.sync(_export(tmp_path / "ok"), foreign)


def test_check_paper_reads_the_synced_folder(tmp_path):
    paper = tmp_path / "paper"
    P.sync(_export(tmp_path), paper / "results")
    (paper / "main.tex").write_text("\\input{results/results}\nLead \\numLeadX, "
                                    "\\figcaption{figA} and \\numMissing.\n")
    r = N.check_manuscript(paper, [paper / "results"])
    assert r["undefined"] == ["numMissing"] and r["populations"]


def _with_additional(root):
    """The export plus one interactive HTML (in the export) and one video (outside it)."""
    ex = _export(root)
    html, mp4 = b"<html>3-D</html>", b"\x00\x00\x00 ftypmp42"
    (ex / "additional").mkdir()
    (ex / "additional" / "interactiveWs.html").write_bytes(html)
    vid = root / "videos" / "v1_hinge_P8.mp4"
    vid.parent.mkdir()
    vid.write_bytes(mp4)
    man = json.loads((ex / "manifest.json").read_text())
    man["additional"] = [
        {"label": "interactiveWs", "kind": "interactive", "file": "additional/interactiveWs.html",
         "sha256": _sha(html)},
        {"label": "videoHinge", "kind": "video", "source": str(vid), "sha256": _sha(mp4)}]
    (ex / "manifest.json").write_text(json.dumps(man))
    return ex


def test_additional_files_are_opt_in_and_kept_by_a_plain_sync(tmp_path):
    ex, dest = _with_additional(tmp_path), tmp_path / "paper" / "results"
    P.sync(ex, dest)
    assert not (dest / "additional").exists()            # large files: opt-in only
    P.sync(ex, dest, additional=True)
    assert (dest / "additional" / "interactiveWs.html").read_bytes() == b"<html>3-D</html>"
    assert (dest / "additional" / "videoHinge.mp4").is_file()
    r = P.sync(ex, dest)                                  # a plain sync keeps them
    assert (dest / "additional" / "videoHinge.mp4").is_file() and not r["removed"]


def test_additional_video_checked_against_the_manifest(tmp_path):
    ex, dest = _with_additional(tmp_path), tmp_path / "paper" / "results"
    (tmp_path / "videos" / "v1_hinge_P8.mp4").write_bytes(b"re-rendered")
    with pytest.raises(ValueError, match="differs from the export manifest"):
        P.sync(ex, dest, additional=True)
