# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J8: method parameters (``config:``) and hardware tables (``hw:``) as numbers."""

from __future__ import annotations

import pandas as pd
import pytest

from biomech.viz import numbers as N


class _Cfg:
    raw = {"stages": {"timing": {"emg_pre_s": 0.6, "k_sd": 3.0, "rule": "drive",
                                 "window": [1, 2]}}}


class _Ctx:
    config = _Cfg()

    def table(self, stage, name):
        return pd.DataFrame()


def test_config_numbers_read_the_store_configuration():
    decl = {"winPre": {"source": "config:stages.timing.emg_pre_s", "scale": 1000, "fmt": ".0f",
                       "unit": "\\milli\\second", "text": "EMG search window before onset"},
            "kSd": {"source": "config:stages.timing.k_sd"},
            "rule": {"source": "config:stages.timing.rule"}}
    got = {n.key: n for n in N._resolve_rows(_Ctx(), decl)}
    assert got["winPre"].text == "600" and got["winPre"].unit == "\\milli\\second"
    assert got["winPre"].source == "config:stages.timing.emg_pre_s" and got["winPre"].n is None
    assert got["kSd"].text == "3" and got["rule"].value == "drive"
    with pytest.raises(ValueError, match="not in the configuration"):
        N._resolve_rows(_Ctx(), {"x": {"source": "config:stages.timing.nope"}})
    with pytest.raises(ValueError, match="not a single value"):
        N._resolve_rows(_Ctx(), {"x": {"source": "config:stages.timing.window"}})


def test_hw_numbers_read_the_parts_tables(tmp_path, monkeypatch):
    (tmp_path / "parts.csv").write_text("part,mass_g\nmouth_marker,15.9\nhead_marker,62.0\n")
    monkeypatch.setattr(N, "HW_TABLES", tmp_path)
    got = N._resolve_rows(_Ctx(), {"massMouth": {"source": "hw:parts",
                                                  "filter": {"part": "mouth_marker"},
                                                  "column": "mass_g", "fmt": ".1f",
                                                  "unit": "\\gram"}})
    assert got[0].text == "15.9" and got[0].source == "hw:parts"
    with pytest.raises(ValueError, match="empty or missing"):
        N._resolve_rows(_Ctx(), {"x": {"source": "hw:nope", "column": "mass_g"}})


def test_check_paper_rejects_numbers_of_the_analysis_of_record(tmp_path):
    import json

    exp = tmp_path / "export"
    exp.mkdir()
    nums = {"cohortN": {"macro": "\\numCohortN", "value": 13, "text": "13",
                        "source": "qc_cohort:demographics", "column": "n"},
            "cohortNAll": {"macro": "\\numCohortNAll", "value": 14, "text": "14",
                           "source": "sensitivity:reporting_population", "column": "primary"},
            "lpEight": {"macro": "\\numLpEight", "value": 1.0, "text": "1.0",
                        "source": "sensitivity:lowpass_8hz", "column": "variant"}}
    (exp / "numbers.json").write_text(json.dumps(
        {"provenance": {"variant": "reporting_population"}, "numbers": nums}))
    tex = tmp_path / "paper"
    tex.mkdir()
    (tex / "main.tex").write_text("We had \\numCohortN{} adults (\\numCohortNAllN{} recorded); "
                                  "\\numLpEight.")
    r = N.check_manuscript(tex, [exp])
    assert r["other_population"] == ["numCohortNAll"] and r["undefined"] == []


def test_numbers_print_rounded_zero_without_sign():
    from biomech.viz.numbers import _fmt

    assert _fmt(-0.0004, ".3f") == "0.000" and _fmt(-0.0006, ".3f") == "-0.001"
    assert _fmt(-0.4, ".0f") == "0" and _fmt(-12.5, ".1f") == "-12.5" and _fmt(0.0, ".2f") == "0.00"


def test_write_refuses_a_macro_defined_twice(tmp_path):
    """A key "XN" next to a key "X" with an n field defines \\numXN twice (LaTeX stops)."""
    nums = [N.Number(key="leadX", value=1.0, text="1", n=12),
            N.Number(key="leadXN", value=12.0, text="12")]
    with pytest.raises(ValueError, match="numLeadXN"):
        N.write(tmp_path, nums, {"snapshot": "test"})
    N.write(tmp_path, nums[:1], {"snapshot": "test"})       # one definition: fine


def test_check_manuscript_ignores_comments_and_definitions(tmp_path):
    """Documentation in comments and the \\newcommand of a copied numbers file are not uses."""
    import json

    exp = tmp_path / "results"
    exp.mkdir()
    (exp / "numbers.json").write_text(json.dumps({"schema": 2, "provenance": {}, "numbers": {
        "leadX": {"macro": "\\numLeadX", "value": 1.0, "text": "1"}}, "cited": {}}))
    (tmp_path / "paper.tex").write_text(
        "% use \\numX, \\numXVal and \\figcaption{label}\n"
        "\\newcommand{\\numOldAll}{3}\n"
        "The lead was \\numLeadX.\n")
    r = N.check_manuscript(tmp_path, [exp])
    assert r["undefined"] == [] and r["unused"] == []


def test_derived_source_reads_a_registered_table(monkeypatch):
    """``source: derived:<name>`` (2026-11-05): a table computed from the store by
    viz/derived.py, then the usual filter / column / ci / agg / n handling."""
    import pandas as pd

    from biomech.viz import derived
    from biomech.viz import numbers as N

    monkeypatch.setitem(derived.TABLES, "toy", lambda ctx: pd.DataFrame({
        "angle_deg": [9.0, 10.0], "cond_trans_mm": [1.0, 1.7], "ci_lo": [0.1, -1.6],
        "ci_hi": [2.0, 3.5], "n": [12, 12], "pid": ["s01", "s02"]}))
    decl = {"tTen": {"source": "derived:toy", "filter": {"angle_deg": 10.0},
                     "column": "cond_trans_mm", "ci": ["ci_lo", "ci_hi"], "n": "n",
                     "fmt": ".1f", "unit": "\\milli\\metre"},
            "tMax": {"source": "derived:toy", "column": "cond_trans_mm", "agg": "max",
                     "n": {"distinct": "pid"}, "fmt": ".1f"}}
    got = {n.key: n for n in N.resolve(object(), decl)}
    assert got["tTen"].text == "1.7" and got["tTen"].ci_text == ("-1.6", "3.5")
    assert got["tTen"].n == 12 and got["tMax"].n == 2
    with pytest.raises(ValueError, match="unknown"):
        N.resolve(object(), {"x": {"source": "derived:nope", "column": "a"}})


def test_p_floor_prints_the_bound_not_zero():
    """A p value that rounds to zero prints as "< 0.001"; other
    columns and non-zero p values keep their text."""
    from biomech.viz.numbers import _fmt, _p_floor

    assert _p_floor("p_holm", 0.0002, ".3f", _fmt(0.0002, ".3f")) == "< 0.001"
    assert _p_floor("p", 0.00004, ".2f", _fmt(0.00004, ".2f")) == "< 0.01"
    assert _p_floor("p_holm", 0.023, ".3f", _fmt(0.023, ".3f")) == "0.023"
    assert _p_floor("median", 0.0002, ".3f", _fmt(0.0002, ".3f")) == "0.000"
    assert _p_floor("pct", 0.0002, ".3f", _fmt(0.0002, ".3f")) == "0.000"
