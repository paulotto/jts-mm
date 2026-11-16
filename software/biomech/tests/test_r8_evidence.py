# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J5 / E43: one evidence map for the requirement rows, equal to the analysis plan's
evidence register, with a reader label for every level, and the paper view showing it."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from biomech.stats import evidence as EV

STUDY = Path(__file__).resolve().parents[3] / "experiments" / "icorr2027"
PLAN = STUDY / "analysis_plan.md"
LOG = STUDY / "change_log.md"
# the working plan holds the register under E43; the public release has the plan as frozen and
# the register in the change log (2026-11-15)
SOURCES = ((PLAN, "### Evidence register (E43)", "\n### "), (LOG, "## 6. Evidence register (E43)", "\n## "))


def _register(path: Path, marker: str, end: str) -> dict[str, str] | None:
    """Key -> level from the "Evidence register (E43)" table of a document, None without it."""
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if marker not in text:
        return None
    block = text[text.index(marker):]
    stop = block.find(end, 5)
    block = block if stop < 0 else block[:stop]
    out: dict[str, str] = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].startswith("`"):
            continue
        for key in re.findall(r"`([a-z0-9_]+)`", cells[0]):
            out[key] = cells[1]
    return out


def test_map_levels_are_known_and_labelled():
    from biomech.viz.views import vocab as V

    labels = V.labels("evidence")
    assert set(EV.REQUIREMENT_EVIDENCE.values()) <= set(EV.LEVELS)
    assert all(level in labels for level in EV.LEVELS), \
        sorted(set(EV.LEVELS) - set(labels))
    with pytest.raises(KeyError):
        EV.requirement("unregistered_row")


@pytest.mark.skipif(not LOG.is_file(), reason="not in the repository checkout")
def test_map_equals_the_plan_register():
    regs = [(p.name, r) for p, m, e in SOURCES if (r := _register(p, m, e)) is not None]
    assert regs, "evidence register table not found"
    assert "change_log.md" in [name for name, _ in regs]
    for name, reg in regs:
        assert reg, name
        for key, level in EV.REQUIREMENT_EVIDENCE.items():
            assert reg.get(key) == level, (name, key, level, reg.get(key))
        extra = set(reg) - set(EV.REQUIREMENT_EVIDENCE)
        assert not extra, (name, extra)


def test_condyle_hinge_is_secondary_post_hoc_and_paper_view_agrees():
    from biomech.viz.views import vocab as V

    assert EV.requirement("misalign_hinge_tmj_mm") == "secondary_post_hoc"
    assert EV.requirement("misalign_hinge_fit_mm") == "confirmatory"
    shown = V.labels("requirement_evidence")
    for key, level in shown.items():                  # the view override agrees with the map
        assert EV.REQUIREMENT_EVIDENCE[key] == level, key
    # 2026-11-10 (JNER revision 1.1): reader words; the hinge at the joint was defined after
    # the results (D33), so it reads "exploratory (post hoc)"
    assert V.labels("evidence")["secondary_post_hoc"] == "exploratory (post hoc)"
