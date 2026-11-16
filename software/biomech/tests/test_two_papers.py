# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""2026-11-12: the study in two papers -- per-item width in paper sets,
the IEEE venue with an 8-pt floor, count macros (``\\num...N``) in written captions, and the
two paper sets of icorr2027 load."""

from __future__ import annotations

from pathlib import Path

import pytest

from biomech import papers
from biomech.viz import style
from biomech.viz.export import caption_override_problems

EXP = Path(__file__).resolve().parents[3] / "experiments" / "icorr2027"


def _set(tmp_path: Path, width: str) -> Path:
    p = tmp_path / "p.yaml"
    p.write_text("targets:\n  t: {venue: ieee8, out_dir: out}\nitems:\n"
                 f"  - {{id: cf_drift, label: suppCfDrift, width: {width}}}\n", encoding="utf-8")
    return p


def test_item_width_reaches_the_export_entry(tmp_path):
    ps = papers.load(_set(tmp_path, "column"))
    it = ps.items[0]
    assert it.width == "column" and it.params == {}
    assert it.entry()["_width"] == "column"


def test_item_width_must_be_column_or_text(tmp_path):
    with pytest.raises(ValueError, match="width"):
        papers.load(_set(tmp_path, "page"))


def test_ieee8_venue_has_ieee_widths_and_an_8_pt_floor():
    v, ieee = style.VENUES["ieee8"], style.VENUES["ieee"]
    assert (v.column_in, v.text_in) == (ieee.column_in, ieee.text_in)
    assert v.min_pt == 8.0 and min(v.font_pt.values()) == 8.0


def test_written_caption_may_use_the_count_macro():
    items = [{"label": "x", "figure": "f", "part": "core",
              "caption_tex": "Dots: participants. $n=\\numCfOpenEOneN$."}]
    assert caption_override_problems(items, {}, {"cfOpenEOne"}) == []


def test_both_icorr2027_paper_sets_load_with_their_captions():
    jner = papers.load(EXP / "paper_dataset.yaml")
    cf = papers.load(EXP / "paper_camerafree.yaml")
    labels = {i.label for i in jner.items}
    assert "sensing" not in labels and "timingEvents" not in labels and "imu" in labels
    assert {i.label for i in cf.items} == {"cfLadder", "suppCfDrift", "suppCfAxes",
                                           "suppCfDecomposition", "tabCameraFree",
                                           "suppTabCfOracle"}
    over = cf.caption_overrides("conference")
    assert all(over[k]["text"].count("rung") == 0 for k in over if k != "_limits")
    assert "imu" in jner.caption_overrides("journal")
