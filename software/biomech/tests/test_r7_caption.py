# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""structured captions and abbreviation expansion."""

from __future__ import annotations

from biomech.viz.caption import CaptionSpec, attach, expand_abbreviations, render
from biomech.viz.spec import Column, FigureSpec, TableSpec
from biomech.viz.views import vocab as V


def test_render_order_and_case():
    c = CaptionSpec("per participant the hinge residual", [("dots", "participants"),
                    ("line", "95 % CI")], "n = 14 participants", "median and BCa 95 % CI",
                    "one-sided Wilcoxon signed-rank test", "pre-specified", ["* note"],
                    codes=["C1"])
    t = render(c)
    # 2026-11-10 (JNER revision 6): each mark and the statistic and test are own sentences
    assert t.startswith("Per participant the hinge residual. Dots: participants. Line: 95 % CI.")
    assert "n = 14 participants." in t and "Median and BCa 95 % CI. One-sided" in t
    assert t.endswith("Pre-specified. * note.") and "C1" not in t


def test_expand_abbreviations_first_use_only():
    t = expand_abbreviations("SPM bars mark clusters; SPM uses the IMU. Mean ± SD.")
    assert t.startswith("Statistical parametric mapping (SPM) bars")
    assert t.count("statistical parametric mapping") + t.count("Statistical parametric") == 1
    assert "inertial measurement unit (IMU)" in t and "standard deviation (SD)" in t
    # already defined, or part of another word: untouched
    assert expand_abbreviations("standard deviation (SD) and SD") == \
        "standard deviation (SD) and SD"
    assert expand_abbreviations("SDK and ISD") == "SDK and ISD"


def test_plain_prose_sentences_and_panel_starts():
    from biomech.viz.caption import plain_prose

    t = plain_prose("Shown; (a) medians over x. (b, c) one value; n = 3. (a) and (b) as is. "
                    "Range 0–12° in (b–d).")
    assert t == ("Shown. (a) Medians over x. (b, c) One value. n = 3. (a) and (b) as is. "
                 "Range 0 to 12° in (b to d).")
    # an abbreviation that starts a sentence after the panel letters is capitalised
    assert expand_abbreviations("(c) RMS misalignment").startswith("(c) Root mean square")


def test_attach_and_reader_spec_keep_parts():
    f = FigureSpec("x", "t", [], caption="old")
    attach(f, CaptionSpec("the opening", [("band", "IQR")]))
    assert f.caption == "The opening. Band: IQR." and f.meta["caption"]["shows"] == "the opening"
    r = V.reader_spec(f)
    assert r.caption == "The opening. Band: interquartile range (IQR)."
    t = attach(TableSpec("t", "", [Column("a", "A")], [{"a": 1}]),
               CaptionSpec("values", n="n = 2"))
    assert t.meta["caption"]["n"] == "n = 2" and t.caption == "Values. n = 2."
