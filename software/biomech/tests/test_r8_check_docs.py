# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J8: ``biomech check-docs`` on a fixture doc."""

from __future__ import annotations

import pytest

from biomech.check import docs as D

MACROS = {"numA": "7", "numB": "6", "numC": "40.89", "numACI": "[1, 2]"}
DOC = """# Results

- Cohort `\\numA` / `\\numB` (7 / 6), ratio `\\numC` (40.9), unknown `\\numZed`.
- Example `\\numKey`, family `\\numLp<Eight|Ten>`, list `\\numA…`.
- Typed by hand: 12.5 mm and 128 °/s.
- A constant 6 Hz <!-- num-ok: method constant -->
- Code `software/biomech/biomech/check/docs.py:1` and `software/biomech/nope.py:3`.
- Read the store r9 snapshot; n = 14 participants.
- The participant recorded outside the protocol is excluded from both papers.

## Status (history)

- Old run r9 snapshot: 99 mm.
"""


def test_check_doc_finds_each_kind(tmp_path):
    p = tmp_path / "an17_paper_arguments.md"
    p.write_text(DOC, encoding="utf-8")
    r = D.check_doc(p, MACROS, {"r13a", "v11"})
    assert r.undefined == ["an17_paper_arguments.md:3: \\numZed"]
    assert len(r.mismatched) == 1 and "numC" in r.mismatched[0]
    assert {h.split(": ", 1)[1] for h in r.hand_typed} == {"12.5 mm", "128 °/s"}
    assert r.bad_refs == ["an17_paper_arguments.md:7: software/biomech/nope.py not found"]
    assert r.stale_snapshots == ["an17_paper_arguments.md:8: r9"]          # not in history
    assert len(r.population) == 1 and ":8:" in r.population[0]            # "excluded" line ok
    assert r.failures() == 3


def test_non_strict_doc_skips_hand_typed_numbers(tmp_path):
    p = tmp_path / "an02_frames_and_kinematics.md"
    p.write_text(DOC, encoding="utf-8")
    r = D.check_doc(p, MACROS, set())
    assert r.hand_typed == [] and r.population == []


def test_fill_updates_quoted_values(tmp_path):
    p = tmp_path / "an17_paper_arguments.md"
    p.write_text("A `\\numA` / `\\numB` (8 / 5 mm), ratio `\\numC` (40.9), "
                 "range `\\numA` (1–9).\n", encoding="utf-8")
    n, left = D.fill_doc(p, MACROS)
    assert n == 3
    assert p.read_text() == ("A `\\numA` / `\\numB` (7 / 6 mm), ratio `\\numC` (40.89), "
                             "range `\\numA` (1–9).\n")
    assert len(left) == 1 and "(1–9)" in left[0]


def test_every_storyboard_claim_has_an_item_of_its_paper_set(tmp_path):
    """J2.5: a claim row without an item, or an item its paper set does not list, fails."""
    from biomech.check.docs import STORYBOARD, claim_coverage

    if not STORYBOARD.exists():
        pytest.skip("the paper storyboard is not in this checkout (public release)")
    fails, _ = claim_coverage()
    assert fails == []
    doc = tmp_path / "an16.md"
    doc.write_text("## 1. Dataset\n\n| Section | Claim | Items (part) | Macros |\n|---|---|---|---|\n"
                   "| **X** | a claim | – | – |\n| **Y** | other | `no_such_item` (core) | – |\n",
                   encoding="utf-8")
    fails, _ = claim_coverage(doc)
    assert any("without an item" in f for f in fails)
    assert any("no_such_item" in f for f in fails)


def test_stale_snapshot_names_respect_time_context(tmp_path):
    """Dated sections and lines that place a snapshot in time are records, not stale text."""
    doc = tmp_path / "an99_x.md"
    doc.write_text(
        "# Doc\n\n"
        "The paper store is r10.\n"                          # stale: a current claim
        "The paper store is r15a (unchanged since r13a).\n"   # names the pin: fine
        "In the first r10 run the store had 12 of 13.\n"       # placed in time: fine
        "## Entry (2026-10-29)\n\n"
        "The run used snapshot r12.\n")                       # dated section: fine
    rep = D.check_doc(doc, {}, {"r15a", "r15", "v13"}, strict=False)
    assert rep.stale_snapshots == ["an99_x.md:3: r10"]
