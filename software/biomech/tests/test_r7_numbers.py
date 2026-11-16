# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""in-text numbers resolved from the store, written as macros, verified against
the store, and checked against a manuscript (no store needed: a stub context)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from biomech.viz import numbers as N


class Ctx:
    def __init__(self, tables):
        self.tables = tables

    def table(self, stage, name):
        return self.tables.get(f"{stage}:{name}", pd.DataFrame()).copy()


REQ = pd.DataFrame({"key": ["opening_mm", "peak_vel_dps"], "value": [39.69, 124.447],
                    "ci_lo": [35.15, 78.55], "ci_hi": [41.16, 133.2], "n": [14, 14]})
SEG = pd.DataFrame({"pid": ["s10", "s10", "s14"], "segment": ["a", "b", "a"],
                    "incisor_rms_mm": [1.0, 3.0, 2.0]})
DECL = {
    "reqOpening": {"source": "requirements:requirements", "filter": {"key": "opening_mm"},
                   "column": "value", "ci": ["ci_lo", "ci_hi"], "n": "n", "fmt": ".1f",
                   "unit": "\\milli\\metre"},
    "segMedian": {"source": "ref_kinematics:segments", "filter": {"segment": "a"},
                  "column": "incisor_rms_mm", "agg": "median", "fmt": ".2f"},
}


@pytest.fixture
def ctx():
    return Ctx({"requirements:requirements": REQ, "ref_kinematics:segments": SEG})


def test_resolve_one_row_and_aggregate(ctx):
    nums = {n.key: n for n in N.resolve(ctx, DECL)}
    o = nums["reqOpening"]
    assert o.text == "39.7" and o.ci_text == ("35.1", "41.2") and o.n == 14
    assert o.macro() == "numReqOpening"
    s = nums["segMedian"]
    assert s.text == "1.50" and s.n == 2                 # median over the two "a" rows


def test_resolve_fails_loudly(ctx):
    bad = {"x": {"source": "ref_kinematics:segments", "column": "incisor_rms_mm"}}
    with pytest.raises(ValueError, match="selects 3 rows"):
        N.resolve(ctx, bad)
    with pytest.raises(ValueError, match="column 'nope'"):
        N.resolve(ctx, {"x": {"source": "requirements:requirements",
                              "filter": {"key": "opening_mm"}, "column": "nope"}})
    with pytest.raises(ValueError, match="empty or missing"):
        N.resolve(ctx, {"x": {"source": "no:table", "column": "v"}})


def test_keys_are_letters_only(tmp_path):
    p = tmp_path / "n.yaml"
    p.write_text("req1: {source: a:b, column: c}\n")
    with pytest.raises(ValueError, match="letters only"):
        N.load(p)


def test_write_macros_and_verify(ctx, tmp_path):
    decl = tmp_path / "numbers.yaml"
    import yaml

    decl.write_text(yaml.safe_dump(DECL))
    nums = N.resolve(ctx, N.load(decl))
    jp, tp = N.write(tmp_path / "out", nums, {"snapshot": "r9"}, decl)
    tex = tp.read_text()
    assert "\\newcommand{\\numReqOpening}{\\qty{39.7}{\\milli\\metre}}" in tex
    assert "\\newcommand{\\numReqOpeningCI}{[\\num{35.1}, \\num{41.2}]\\,\\unit{\\milli\\metre}}" in tex
    assert "\\newcommand{\\numReqOpeningN}{14}" in tex and "\\numSegMedianVal" in tex
    assert N.verify(tmp_path / "out", ctx) == []
    # a changed store value (or a hand-edited numbers.json) is caught
    ctx.tables["requirements:requirements"] = REQ.assign(value=[40.2, 124.4])
    probs = N.verify(tmp_path / "out", ctx)
    assert probs and "reqOpening" in probs[0] and "40.2" in probs[0]


def test_check_manuscript(ctx, tmp_path):
    decl = tmp_path / "numbers.yaml"
    import yaml

    decl.write_text(yaml.safe_dump(DECL))
    exp = tmp_path / "export"
    N.write(exp, N.resolve(ctx, N.load(decl)), {"snapshot": "r9"}, decl)
    (exp / "manifest.json").write_text(json.dumps({"items": [{"label": "hinge"}]}))
    tex = tmp_path / "paper"
    tex.mkdir()
    (tex / "main.tex").write_text("Opening \\numReqOpening\\ \\numReqOpeningCI, "
                                  "\\numTypo; \\figcaption{hinge} \\figcaption{nofig}")
    r = N.check_manuscript(tex, [exp])
    assert r["undefined"] == ["numTypo", "figcaption{nofig}"]
    assert r["unused"] == ["numSegMedian"]
