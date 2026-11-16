# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J4: requirement envelope (X7), coverage of the sample maximum (X8) and the
condyle-incisor coupling summary (X9), on synthetic inputs with known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from biomech.stats import cohort_kinematics as ck
from biomech.stats import requirements as R


def test_dynamics_envelope_takes_peaks_per_participant():
    t = np.linspace(0, 10, 401)
    vel = 100 * np.sin(2 * np.pi * 0.5 * t)                 # +-100 deg/s
    acc = np.gradient(vel, t)
    arrays = {"s01": {"free_motion__theta_vel": vel, "free_motion__theta_acc": acc,
                      "open_close__theta_acc": 2 * acc},
              "s02": {"chewing__theta_acc": np.array([np.nan, 5.0, -7.0])}}
    env = ck.dynamics_envelope(arrays).set_index("pid")
    assert env.loc["s01", "free_peak_open_vel_dps"] == pytest.approx(100, rel=1e-3)
    assert env.loc["s01", "free_peak_close_vel_dps"] == pytest.approx(100, rel=1e-3)
    assert env.loc["s01", "peak_acc_task"] == "open_close"
    assert env.loc["s01", "peak_acc_dps2"] == pytest.approx(2 * np.pi * 0.5 * 200, rel=1e-2)
    assert np.isnan(env.loc["s02", "free_peak_open_vel_dps"])
    assert env.loc["s02", "peak_acc_dps2"] == 7.0 and env.loc["s02", "peak_acc_task"] == "chewing"


def test_envelope_rows_use_the_design_rule_and_post_hoc_label():
    s = pd.DataFrame({"peak_close_vel_max_dps": np.linspace(100, 200, 13),
                      "free_peak_open_vel_dps": np.linspace(120, 240, 13),
                      "peak_acc_dps2": np.linspace(800, 3000, 13)})
    rows = {r["key"]: r for r in R._envelope_rows(s)}
    assert set(rows) == {"close_vel_dps", "free_peak_open_vel_dps", "peak_acc_dps2"}
    for r in rows.values():
        assert r["evidence"] == "post_hoc_descriptive" and r["n"] == 13
        assert r["statistic"].startswith("P95") and "lower bound" in r["note"]
    assert rows["peak_acc_dps2"]["value"] == pytest.approx(
        R.p95_design(s.peak_acc_dps2.to_numpy())["value"])
    assert R._envelope_rows(pd.DataFrame({"peak_acc_dps2": [np.nan]})) == []


def test_coverage_table_closed_form():
    c = R.coverage_table()
    need = c[c.smallest_n].set_index("q").n
    assert need[0.80] == 14 and need[0.90] == 29
    row = c[(c.n == 13) & (c.q == 0.80) & ~c.smallest_n].iloc[0]
    assert row.p_max_exceeds == pytest.approx(1 - 0.8 ** 13)
    assert (c.p_max_exceeds.diff()[~c.smallest_n & (c.n > 13)] > 0).all()


def test_coupling_summary_slope_and_generic_miss():
    grid = np.arange(0, 21, 1.0)
    rng = np.random.default_rng(0)
    slopes = [0.10, 0.15, 0.20, 0.25, 0.30]

    def cycles(k: float) -> np.ndarray:
        return np.vstack([k * grid + rng.normal(0, 0.01, grid.size) for _ in range(4)])

    curves = {"tmj_mid_ant_mm": [cycles(k) for k in slopes]}
    pids = [f"s0{i}" for i in range(1, 6)]
    out = ck.coupling_summary(curves, grid, pids, plausible={"s01"}).set_index("pid")
    assert out.slope_mm_per_deg.to_numpy() == pytest.approx(slopes, abs=0.01)
    # the middle participant is closest to the generic (others') median curve
    assert out.generic_rms_mm.idxmin() == "s03"
    assert out.loc["s01", "condyles_plausible"] and not out.loc["s02", "condyles_plausible"]
    assert out.loc["s01", "angle_max_deg"] == 20.0


def test_posterior_excursion_from_rest_position():
    rest = np.tile([10.0, 0.0, 0.0], (50, 1))
    back = np.column_stack([np.linspace(10, -5, 101), np.zeros(101), np.zeros(101)])
    arrays = {"s01": {"rest__incisor": rest, "open_close__incisor": back},
              "s02": {"open_close__incisor": back}}
    d = ck.posterior_excursion(arrays).set_index("pid")
    assert d.loc["s01", "posterior_excursion_max_mm"] == pytest.approx(15.0)
    assert d.loc["s01", "posterior_excursion_mm"] == pytest.approx(np.percentile(
        np.clip(10 - back[:, 0], 0, None), ck.POSTERIOR_Q))
    assert np.isnan(d.loc["s02", "posterior_excursion_mm"])          # no rest phase
    s = pd.DataFrame({"posterior_excursion_mm": np.linspace(5, 16, 13)})
    assert R.p95_design(s.posterior_excursion_mm.to_numpy())["n"] == 13
