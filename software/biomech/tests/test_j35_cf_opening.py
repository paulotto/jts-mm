# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J35: opening per cycle from the camera-free estimators (statistics of the table)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.viz.derived_dataset import cf_opening_table


def _rows(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(6):
        for c in range(8):
            opt = 30.0 + rng.normal(0, 2)
            base = rng.normal(0, 1)
            for est, extra in (("E1", 0.0), ("E2a", 0.0), ("E3", 0.5 * (-1) ** c)):
                rows.append({"pid": f"s{i}", "task": "open_close", "cycle": c, "estimator": est,
                             "opening_est_mm": opt + base + extra, "opening_opt_mm": opt})
    return pd.DataFrame(rows)


def test_identical_estimators_have_equal_limits_and_zero_contrast():
    t = cf_opening_table(_rows()).set_index("estimator")
    assert t.at["E1", "n"] == 6 and t.at["E1", "n_cycles"] == 48
    assert t.at["E2a", "max_abs_loa"] == pytest.approx(t.at["E1", "max_abs_loa"])
    assert t.at["E2a", "hl"] == pytest.approx(0.0)
    assert np.isnan(t.at["E1", "hl"])


def test_a_noisier_estimator_has_wider_limits_and_a_positive_contrast():
    t = cf_opening_table(_rows()).set_index("estimator")
    assert t.at["E3", "max_abs_loa"] > t.at["E1", "max_abs_loa"]
    assert t.at["E3", "hl"] > 0
    assert t.at["E3", "loa_lo"] < t.at["E3", "bias"] < t.at["E3", "loa_hi"]
    assert t.at["E3", "max_abs_loa_ci_hi"] > t.at["E3", "max_abs_loa"]
