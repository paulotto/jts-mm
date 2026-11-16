# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Requirements table (percentile policy, CIs) and Bland-Altman cluster-bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from biomech.stats.core import bland_altman_rm, bland_altman_rm_ci
from biomech.stats.requirements import COLUMNS, p95_design, requirements_table


def test_p95_design_block_is_bounded_by_the_sample():
    """E31: Harrell-Davis value next to the linear P95, order-statistic lower bound X(10) at
    n = 12 (P(Bin(12, .95) >= 10) = 0.980), no finite upper bound, the maximum."""
    x = np.arange(1.0, 13.0)
    d = p95_design(x)
    assert d["n"] == 12 and d["value_linear"] == np.percentile(x, 95)
    assert d["lower_k"] == 10 and d["ci_lo"] == 10.0
    assert d["ci_lo"] <= d["value"] <= d["sample_max"] == x.max()
    assert np.isnan(d["ci_hi"]) and d["upper_note"] == "none finite at n = 12"


def test_ba_cluster_ci_contains_the_estimates():
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(12), 10)
    ref = rng.normal(30, 5, 120)
    est = ref + 0.5 + rng.normal(0, 1, 12)[g] + rng.normal(0, 0.8, 120)
    ba, ci = bland_altman_rm(est, ref, g), bland_altman_rm_ci(est, ref, g, seed=2, n_boot=400)
    for k in ("bias", "loa_lo", "loa_hi"):
        assert ci[f"{k}_ci_lo"] < ba[k] < ci[f"{k}_ci_hi"]
    assert ci["prop_slope_ci_lo"] < ci["prop_slope"] < ci["prop_slope_ci_hi"]


def test_requirements_table_rows_and_policy():
    pids = [f"s{i:02d}" for i in range(1, 13)]
    rng = np.random.default_rng(3)
    subj = pd.DataFrame({"pid": pids, "max_observed_opening_mm": rng.normal(35, 5, 12),
                         "protrusion_mm": rng.normal(9, 2, 12),
                         "laterotrusion_left_mm": rng.normal(11, 2, 12),
                         "laterotrusion_right_mm": rng.normal(11, 2, 12),
                         "opening_angle_deg": rng.normal(15, 3, 12),
                         "condyle_left_disp_mm": rng.normal(10, 3, 12),
                         "condyle_right_disp_mm": rng.normal(10, 3, 12),
                         "peak_open_vel_max_dps": rng.normal(70, 20, 12)})
    hinge = pd.DataFrame({"pid": pids * 2, "model": ["hinge_fit"] * 12 + ["hinge_tmj"] * 12,
                          "rms_incisor_mm": rng.normal(2, 0.5, 24)})
    hn = pd.DataFrame({"pid": pids, "above_invivo": [True] * 11 + [False]})
    empty = pd.DataFrame()
    t = requirements_table(subj, hinge, hn, empty, empty, empty, empty, empty, empty, seed=0)
    assert list(t.columns) == COLUMNS
    cap = t[t.statistic.str.startswith("P95")]
    assert len(cap) == 6 and (cap.ci_lo <= cap.value).all()
    assert (cap.value <= cap.sample_max).all() and cap.ci_hi.isna().all()
    assert (cap.lower_k == 10).all() and (cap.upper_note == "none finite at n = 12").all()
    assert (t[t.group == "workspace"].note.str.contains("lower bound")).all()
    fit = t[t.key == "misalign_hinge_fit_mm"].iloc[0]
    assert fit.evidence == "confirmatory" and "11/12" in fit.note
