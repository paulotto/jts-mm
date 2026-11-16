# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Uncertainty blocks of the dataset views (Part 5 E): bootstrap bands of cohort
curves, BCa CIs of the median on strips, paired strips instead of dumbbells of medians,
n-aware heights, and the CMC / variance-ratio bootstrap of the variability view."""

from __future__ import annotations

import numpy as np
import pytest

from biomech.viz.spec import Panel
from biomech.viz.views import blocks as B


def _curves(n: int = 12, pts: int = 101, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, pts)
    return np.sin(np.pi * x)[None] * (10 + rng.normal(0, 2, (n, 1))) + rng.normal(0, 1, (n, pts))


def test_boot_curve_ci_deterministic_and_brackets_the_mean():
    S = _curves()
    lo, hi = B.boot_curve_ci(S, seed=42)
    lo2, hi2 = B.boot_curve_ci(S, seed=42)
    assert np.array_equal(lo, lo2) and np.array_equal(hi, hi2)
    m = S.mean(axis=0)
    assert np.all(lo <= m + 1e-12) and np.all(m <= hi + 1e-12)
    sd = S.std(axis=0, ddof=1)
    assert np.all(hi - lo < 2 * sd)                     # a CI of the mean, not the spread
    lo_m, hi_m = B.boot_curve_ci(S, centre="median", seed=42)
    med = np.median(S, axis=0)
    assert np.all(lo_m <= med + 1e-12) and np.all(med <= hi_m + 1e-12)


def test_boot_curve_ci_needs_three_units_per_sample():
    S = _curves(n=5)
    S[2:, :10] = np.nan                                  # only 2 units at the first samples
    lo, hi = B.boot_curve_ci(S, seed=0)
    assert np.isnan(lo[:10]).all() and np.isfinite(lo[10:]).all()
    lo, hi = B.boot_curve_ci(S[:2], seed=0)
    assert np.isnan(lo).all() and np.isnan(hi).all()
    lo, hi = B.boot_curve_ci(np.empty((0, 0)))
    assert lo.size == 0


def test_add_curves_ci_two_bands_line_on_top():
    p = Panel("t")
    B.add_curves_ci(p, np.arange(101.0), _curves(), label="mean", role="cat:0",
                    show_subjects=False, seed=3)
    kinds = [s.kind for s in p.series]
    assert kinds == ["band", "band", "line"]
    assert p.series[0].style["alpha"] == B.SD_BAND_ALPHA and "alpha" not in p.series[1].style
    assert all(s.role == "cat:0" for s in p.series) and p.series[-1].label == "mean"
    sd_w = np.nanmean(p.series[0].y2 - p.series[0].y)
    ci_w = np.nanmean(p.series[1].y2 - p.series[1].y)
    assert ci_w < sd_w


def test_median_ci_is_the_stats_stage_bca():
    from biomech.stats.core import describe

    v = np.random.default_rng(5).normal(3, 1, 14)
    d = describe(v, stat="median", seed=42)
    assert B.median_ci(v, 42) == (d["ci_lo"], d["ci_hi"])
    assert np.isnan(B.median_ci([1.0, 2.0, 3.0])).all()          # fewer than 4 values


def test_strip_ci_and_paired_strip():
    rng = np.random.default_rng(2)
    vals = [rng.normal(i, 1, 12) for i in range(3)]
    p = B.strip_ci(["a", "b", "c"], vals, "t", B.Axis("x"), seed=42)
    ivs = [s for s in p.series if s.kind == "line"]
    assert len(ivs) == 3
    for i, s in enumerate(ivs):
        assert s.x[0] <= np.median(vals[i]) <= s.x[1]
    q = B.paired_strip(["a", "b"], vals[:2], [v + 1 for v in vals[:2]], "t", B.Axis("x"),
                       label_a="A", label_b="B", seed=42)
    assert [s.label for s in q.legend_entries()] == ["A", "B"]
    diamonds = [s for s in q.series if s.style.get("ms") == "d"]
    assert len(diamonds) == 2 and all(len(s.x) == 2 for s in diamonds)
    assert len([s for s in q.series if s.kind == "line"]) == 4
    assert q.y.ticklabels == ["a", "b"]
    empty = B.paired_strip(["a"], [[]], [[]], "t", B.Axis("x"), label_a="A", label_b="B")
    assert empty.note == "no data"


def test_rows_height_grows_with_n():
    assert B.rows_height(1) == B.rows_height(2) == 1.3
    assert B.rows_height(14) > B.rows_height(12) > B.rows_height(6)
    assert B.rows_height(14, per=0.2, extra=0.6) == pytest.approx(3.4)


def test_cmc_batch_matches_waveforms_cmc():
    from biomech.stats.waveforms import cmc
    from biomech.viz.views.results_emg import _cmc_batch

    S = _curves(n=9, seed=4)
    assert _cmc_batch(S) == pytest.approx(cmc(S), rel=1e-12)
    batch = np.stack([S, S[::-1] * 0.5 + 1.0])
    assert _cmc_batch(batch) == pytest.approx([cmc(S), cmc(S[::-1] * 0.5 + 1.0)], rel=1e-12)


class _Ctx:
    """Minimal context: emg profiles of four participants and the stored CMC row."""

    def __init__(self, theta: dict, stored_within: float, stored_between: float):
        import pandas as pd

        self._theta = theta
        self._spm = pd.DataFrame([{"test": "cmc_opening_angle",
                                   "cmc_within_median": stored_within,
                                   "cmc_between": stored_between}])

    def pids(self):
        return list(self._theta)

    def arrays(self, stage, pid):
        return {"profile__open_close__theta_deg": self._theta[pid]}

    def table(self, stage, name):
        import pandas as pd

        return self._spm if name == "spm" else pd.DataFrame()


def test_cmc_ci_reproduces_the_stored_values_or_draws_nothing():
    from biomech.stats.waveforms import cmc, variance_ratio
    from biomech.viz.views.results_emg import cmc_ci, varratio_ci

    theta = {f"s{i}": _curves(n=8, seed=10 + i) for i in range(4)}
    w = float(np.nanmedian([cmc(P) for P in theta.values()]))
    b = cmc(np.array([P.mean(0) for P in theta.values()]))
    out = cmc_ci(_Ctx(theta, w, b), seed=42, n_boot=300)
    assert out is not None and out["median_within"][0] == w and out["between"][0] == b
    for est, lo, hi in [out["median_within"], out["between"], *out["within"].values()]:
        assert lo <= hi and np.isfinite(lo)
    assert cmc_ci(_Ctx(theta, w + 0.01, b), seed=42) is None      # stale store: no CI
    inputs = {"theta": theta}
    ratio = variance_ratio(theta)["ratio"]
    lo, hi = varratio_ci(None, "theta", ratio, seed=42, n_boot=300, inputs=inputs)
    assert np.all(lo <= hi)
    assert varratio_ci(None, "theta", ratio * 1.01, seed=42, inputs=inputs) is None
