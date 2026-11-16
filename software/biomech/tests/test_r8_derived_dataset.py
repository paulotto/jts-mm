# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Dataset-paper numbers from derived tables and Table I rounding (2026-11-05): the SVM search
grid as text macros, and ``fmt: req`` printing a requirement macro with Table I's digits
(decimals from the row's design value, one below 20, none above)."""

from __future__ import annotations

import pandas as pd

from biomech.viz import derived
from biomech.viz import numbers as N
from biomech.viz.views.paper import _value_text, req_decimals


def test_svm_search_grid_is_the_ml_registry_grid():
    from ml.models.registry import REGISTRY

    t = derived.table(object(), "svm_search")
    assert set(t["param"]) == set(REGISTRY["svm"].grid)
    row = t[t["param"] == "C"].iloc[0]
    assert row["grid_values"] == ", ".join(f"{v:g}" for v in REGISTRY["svm"].grid["C"])
    got = {n.key: n for n in N.resolve(object(), {
        "g": {"source": "derived:svm_search", "filter": {"param": "C"},
              "column": "grid_values"}})}
    assert got["g"].text == row["grid_values"]


class _Ctx:
    def __init__(self, df):
        self.df = df

    def table(self, stage, name, pids=None):
        return self.df


def test_req_format_matches_table_one():
    rq = pd.DataFrame([
        {"key": "opening_mm", "value": 40.55, "ci_lo": 37.53, "ci_hi": float("nan"),
         "sample_max": 41.16, "unit": "mm", "n": 13},
        {"key": "translation_mm", "value": 23.17, "ci_lo": 17.44, "ci_hi": float("nan"),
         "sample_max": 24.6, "unit": "mm", "n": 13},
        {"key": "hinge_mm", "value": 1.948, "ci_lo": 1.47, "ci_hi": 2.22,
         "sample_max": 3.0, "unit": "mm", "n": 13}])
    decl = {
        "op": {"source": "requirements:requirements", "filter": {"key": "opening_mm"},
               "column": "value", "fmt": "req"},
        "opLow": {"source": "requirements:requirements", "filter": {"key": "opening_mm"},
                  "column": "ci_lo", "fmt": "req"},
        "trLow": {"source": "requirements:requirements", "filter": {"key": "translation_mm"},
                  "column": "ci_lo", "fmt": "req"},
        "hinge": {"source": "requirements:requirements", "filter": {"key": "hinge_mm"},
                  "column": "value", "ci": ["ci_lo", "ci_hi"], "fmt": "req"}}
    got = {n.key: n for n in N.resolve(_Ctx(rq), decl)}
    assert got["op"].text == "41" and got["opLow"].text == "38"
    assert got["trLow"].text == "17"           # the row's value (23.2) sets the digits
    assert got["hinge"].text == "1.9" and got["hinge"].ci_text == ("1.5", "2.2")
    for r in rq.itertuples():                 # Table I prints the same digits
        txt = _value_text(pd.Series(r._asdict()))
        assert txt.startswith(f"{r.value:.{req_decimals(r.value)}f}")
    assert _value_text(pd.Series(rq.iloc[0])) == "41 mm (bound 38)"
