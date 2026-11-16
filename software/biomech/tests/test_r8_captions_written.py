# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Written captions of a paper target (paper set ``captions:``, 2026-11-10 compression round)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biomech import papers
from biomech.viz.export import caption_override_problems, caption_words

STUDY = Path(__file__).resolve().parents[3] / "experiments" / "icorr2027"


def test_caption_words_skips_commands_and_panel_letters():
    assert caption_words(r"(a)~Opening of \acs{IMU} in $95\,\%$. (b, c)~Two.") == 6
    assert caption_words(r"$n=\numCohortN$. Descriptive.") == 3


def test_problems_limit_macro_and_typed_decimal():
    over = {"_limits": {"supplement": {"figure": 3}}}
    items = [{"label": "x", "figure": "f", "part": "supplement",
              "caption_tex": r"One two three four \numNope."},
             {"label": "y", "figure": "f", "part": "supplement",
              "caption_tex": r"Lead 1.23 ms."},
             {"label": "z", "figure": "f", "part": "core",
              "caption_tex": r"Calibration $2.5$ s \numCohortN."}]
    p = caption_override_problems(items, over, {"cohortN"})
    assert any("x: 5 words" in m for m in p)
    assert any("numNope" in m for m in p)
    assert any("y: a typed decimal" in m for m in p)
    assert not any(m.startswith("caption of z") for m in p)


def test_jner_written_captions_load_and_fit():
    ps = papers.load(STUDY / "paper_dataset.yaml")
    over = ps.caption_overrides("journal")
    parts = {i.label: i.part for i in ps.items_for("journal")}
    lim = over["_limits"]
    for lab, e in over.items():
        if lab == "_limits":
            continue
        kind = "figure" if e["panels"] else "table"
        assert caption_words(e["text"]) <= lim[parts[lab]][kind], lab


def test_unknown_label_is_an_error(tmp_path):
    (tmp_path / "caps.yaml").write_text("nosuch: text\n")
    (tmp_path / "p.yaml").write_text("targets: {journal: {venue: springer}}\n"
                                     "captions: {journal: caps.yaml}\nitems:\n"
                                     "  - {id: icorr_hinge, label: hinge}\n")
    with pytest.raises(ValueError, match="nosuch"):
        papers.load(tmp_path / "p.yaml").caption_overrides("journal")
