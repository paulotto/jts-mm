# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J32.1: the IMU-only heading against the optical one, only where it was identifiable."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from biomech.viz.derived_dataset import cf_heading_identifiable


def _ctx(ident: dict[str, bool], e2a: dict[str, float], e2b: dict[str, float]):
    summ = pd.DataFrame({"pid": list(ident), "psi_b_identifiable": list(ident.values())})
    rows = [{"pid": p, "estimator": e, "outcome": "rmse_open_deg", "value": v}
            for e, d in (("E2a", e2a), ("E2b", e2b)) for p, v in d.items()]
    rows.append({"pid": "s1", "estimator": "E2b", "outcome": "rmse_geo_deg", "value": 9.0})
    tabs = {("camerafree", "cf_summary"): summ,
            ("camerafree_cohort", "cf_participant"): pd.DataFrame(rows)}
    return SimpleNamespace(table=lambda s, n: tabs[(s, n)])


def test_only_identifiable_participants_count():
    ctx = _ctx({"s1": True, "s2": True, "s3": False, "s4": True},
               {"s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 1.0},
               {"s1": 1.5, "s2": 0.9, "s3": 1.0, "s4": 1.1})
    t = cf_heading_identifiable(ctx)
    r = t.iloc[0]
    assert r["n"] == 3 and r["n_up"] == 2
    assert r["median"] == pytest.approx(0.1)
    assert r["min"] == pytest.approx(-0.1) and r["max"] == pytest.approx(0.5)


def test_empty_without_identifiable_participants():
    ctx = _ctx({"s1": False}, {"s1": 1.0}, {"s1": 1.0})
    assert cf_heading_identifiable(ctx).empty
