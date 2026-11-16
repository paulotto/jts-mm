# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R10 calculation audit: reference implementations and coverage simulations.

* ``holm`` against statsmodels ``multipletests(method="holm")``;
* ``icc_a1`` against Shrout & Fleiss (1979): McGraw & Wong's ICC(A,1) (two-way, absolute
  agreement, single measurement) has the same estimator as Shrout & Fleiss' ICC(2,1). Their
  6 targets x 4 judges example gives ICC(2,1) = 0.29 (Table 4 of S&F 1979; BMS = 11.24,
  JMS = 32.49, EMS = 1.02); a reference ICC(2,1) from statsmodels' two-way ANOVA is checked
  against that value, then ``icc_a1`` (two raters only) against the reference on every pair
  of judges;
* ``ccc`` against Lin's (1989) formula written out with 1/n moments (EV9; the pre-EV9
  pipeline used unbiased n - 1 moments) and against a worked example by hand;
* ``bland_altman_rm`` against a hand-derived example of the Bland & Altman (2007)
  "true value varies" method (the paper's own worked numbers are not reproduced here
  because they could not be verified reliably), the equal-replicates special case
  (divisor = m) and a large simulated sample;
* coverage of the BCa median CI (``describe``) at n = 9 and 14 and of the participant-
  cluster bootstrap (``bland_altman_rm_ci``) with 13 clusters (``slow``);
* E30 (impedance): two muscles of one participant must not double the n of the test; the
  test is the exact Wilcoxon over the participants' median changes; the pairs stay
  descriptive; Holm runs within the two families (check vs recorded, check vs check).
"""

from __future__ import annotations

from fractions import Fraction
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

from biomech.stats.core import bland_altman_rm, bland_altman_rm_ci, describe, holm
from biomech.validation.agreement import ccc, icc_a1


# -- Holm ------------------------------------------------------------------------------------
@pytest.mark.parametrize("p", [
    [0.01, 0.04, 0.03, 0.005],
    [0.2, 0.2, 0.2],                                   # ties
    [1e-6, 0.5, 0.049, 0.051, 0.9, 0.0125],
    [0.3],                                             # family of one
    [0.04, 0.04, 0.5, 1.0],                            # capped at 1
    [0.0234375, 0.0234375, 0.23046875, 0.0546875],     # the r9 C2 family (raw p)
])
def test_holm_matches_statsmodels(p):
    ref = multipletests(np.asarray(p), method="holm")[1]
    np.testing.assert_allclose(holm(p), ref, rtol=0, atol=1e-12)


def test_holm_keeps_nan_out_of_the_family():
    p = [0.01, np.nan, 0.04, 0.03]
    out = np.asarray(holm(p))
    assert np.isnan(out[1])
    ref = multipletests(np.array([0.01, 0.04, 0.03]), method="holm")[1]
    np.testing.assert_allclose(out[[0, 2, 3]], ref, atol=1e-12)


# -- ICC(A,1) = ICC(2,1) ---------------------------------------------------------------------
# Shrout & Fleiss (1979), Psychol Bull 86:420, Table 2: 6 targets (rows) x 4 judges.
SF_DATA = np.array([[9, 2, 5, 8],
                    [6, 1, 3, 2],
                    [8, 4, 6, 8],
                    [7, 1, 2, 6],
                    [10, 5, 6, 9],
                    [6, 2, 4, 7]], float)


def _icc21_reference(Y: np.ndarray) -> tuple[float, dict[str, float]]:
    n, k = Y.shape
    long = pd.DataFrame({"y": Y.ravel(), "t": np.repeat(np.arange(n), k),
                         "r": np.tile(np.arange(k), n)})
    a = anova_lm(smf.ols("y ~ C(t) + C(r)", long).fit(), typ=1)
    ms = {"BMS": a.loc["C(t)", "mean_sq"], "JMS": a.loc["C(r)", "mean_sq"],
          "EMS": a.loc["Residual", "mean_sq"]}
    icc = (ms["BMS"] - ms["EMS"]) / (ms["BMS"] + (k - 1) * ms["EMS"]
                                     + k * (ms["JMS"] - ms["EMS"]) / n)
    return float(icc), ms


def test_reference_icc21_reproduces_shrout_fleiss():
    icc, ms = _icc21_reference(SF_DATA)
    assert round(ms["BMS"], 2) == 11.24
    assert round(ms["JMS"], 2) == 32.49
    assert round(ms["EMS"], 2) == 1.02
    assert round(icc, 2) == 0.29                       # published ICC(2,1)


@pytest.mark.parametrize("j", list(combinations(range(4), 2)))
def test_icc_a1_equals_icc21_on_every_pair_of_judges(j):
    Y = SF_DATA[:, list(j)]
    ref, _ = _icc21_reference(Y)
    assert icc_a1(Y[:, 0], Y[:, 1]) == pytest.approx(ref, abs=1e-12)


def test_icc_a1_on_random_data_and_nan_handling():
    rng = np.random.default_rng(3)
    t = rng.normal(0, 2, 40)
    a, b = t + rng.normal(0, 1, 40), t + 0.7 + rng.normal(0, 1, 40)
    ref, _ = _icc21_reference(np.c_[a, b])
    assert icc_a1(a, b) == pytest.approx(ref, abs=1e-12)
    a2 = a.copy()
    a2[[3, 7]] = np.nan                                # pairs with a NaN are dropped
    keep = np.isfinite(a2)
    assert icc_a1(a2, b) == pytest.approx(_icc21_reference(np.c_[a2[keep], b[keep]])[0],
                                          abs=1e-12)


# -- Lin's CCC -------------------------------------------------------------------------------
def _lin(x: np.ndarray, y: np.ndarray, ddof: int) -> float:
    n = len(x)
    mx, my = x.mean(), y.mean()
    sxy = np.sum((x - mx) * (y - my)) / (n - ddof)
    sx2 = np.sum((x - mx) ** 2) / (n - ddof)
    sy2 = np.sum((y - my) ** 2) / (n - ddof)
    return float(2 * sxy / (sx2 + sy2 + (mx - my) ** 2))


def test_ccc_worked_example():
    """x = 1..5, y = (2, 3, 3, 5, 6): x̄ = 3, ȳ = 3.8; with 1/n moments s_x² = 10/5 = 2,
    s_y² = 10.8/5 = 2.16, s_xy = 10/5 = 2, so ρ_c = 2·2 / (2 + 2.16 + 0.8²) = 4/4.8 = 5/6.
    (The n − 1 moments would give 5/5.84 = 0.856, the pre-EV9 value.)"""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 3.0, 3.0, 5.0, 6.0])
    assert ccc(x, y) == pytest.approx(5 / 6, abs=1e-12)
    assert ccc(y, x) == pytest.approx(5 / 6, abs=1e-12)              # symmetric


@pytest.mark.parametrize("seed,n,shift", [(0, 5, 0.8), (1, 8, 0.0), (2, 18, 2.0), (3, 200, 0.3)])
def test_ccc_is_lins_formula_with_population_moments(seed, n, shift):
    rng = np.random.default_rng(seed)
    x = rng.normal(10, 2, n)
    y = x + shift + rng.normal(0, 1, n)
    assert ccc(x, y) == pytest.approx(_lin(x, y, ddof=0), abs=1e-12)


def test_ccc_edge_cases():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert ccc(x, x) == pytest.approx(1.0)
    assert ccc(x, 5.0 - x) == pytest.approx(-1.0)                    # mirrored, same mean
    assert np.isnan(ccc(x[:2], x[:2]))                               # below 3 pairs
    # with a mean shift and small n the 1/n value is the smaller one (EV9 direction)
    y = x + 1.0 + np.array([0.3, -0.2, 0.1, -0.4])
    assert ccc(x, y) < _lin(x, y, ddof=1)


def test_ccc_nan_handling():
    x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    y = np.array([1.1, 2.3, 3.0, np.nan, 5.2, 5.9])
    k = np.isfinite(x) & np.isfinite(y)
    assert ccc(x, y) == pytest.approx(_lin(x[k], y[k], ddof=0), abs=1e-12)


# -- Bland & Altman (2007) repeated measures -------------------------------------------------
def test_bland_altman_rm_hand_derived_example():
    """Three subjects, differences S1 = [1, 3], S2 = [5, 7, 9], S3 = [2, 3, 4, 3].

    N = 9, k = 3, m = (2, 3, 4), subject means 2, 7, 3, grand mean 37/9.
    SS_b = 2(19/9)^2 + 3(26/9)^2 + 4(10/9)^2 = 350/9, MS_b = 175/9.
    SS_w = 2 + 8 + 2 = 12, MS_w = 12 / 6 = 2.
    D = (N^2 - sum m^2) / ((k - 1) N) = (81 - 29) / 18 = 26/9.
    s_b^2 = (175/9 - 2) / (26/9) = 157/26; var(single difference) = 157/26 + 2 = 209/26.
    """
    groups = np.array(["a", "a", "b", "b", "b", "c", "c", "c", "c"])
    d = np.array([1, 3, 5, 7, 9, 2, 3, 4, 3], float)
    ref = np.random.default_rng(0).normal(50, 10, d.size)      # arbitrary reference values
    out = bland_altman_rm(ref + d, ref, groups)
    bias = Fraction(37, 9)
    var = Fraction(157, 26) + 2
    assert out["bias"] == pytest.approx(float(bias), abs=1e-12)
    assert out["sd_within"] == pytest.approx(np.sqrt(2.0), abs=1e-12)
    assert out["sd_between"] == pytest.approx(np.sqrt(157 / 26), abs=1e-12)
    assert out["loa_lo"] == pytest.approx(float(bias) - 1.96 * np.sqrt(float(var)), abs=1e-12)
    assert out["loa_hi"] == pytest.approx(float(bias) + 1.96 * np.sqrt(float(var)), abs=1e-12)
    assert out["n"] == 9 and out["n_subjects"] == 3


def test_bland_altman_rm_equal_replicates_divisor_is_m():
    """With m replicates per subject the B&A divisor reduces to m (B&A 1999, sec. 5)."""
    rng = np.random.default_rng(11)
    k, m = 6, 4
    g = np.repeat(np.arange(k), m)
    d = rng.normal(0, 1, k)[g] + rng.normal(0, 0.5, k * m)
    means = np.array([d[g == i].mean() for i in range(k)])
    msb = m * np.sum((means - d.mean()) ** 2) / (k - 1)
    msw = np.sum((d - means[g]) ** 2) / (k * m - k)
    sd = np.sqrt(max((msb - msw) / m, 0.0) + msw)
    out = bland_altman_rm(d, np.zeros_like(d), g)
    assert out["loa_hi"] - out["bias"] == pytest.approx(1.96 * sd, abs=1e-12)


def test_bland_altman_rm_recovers_total_sd_on_a_large_sample():
    rng = np.random.default_rng(12)
    k = 3000
    m = rng.integers(2, 9, k)
    g = np.repeat(np.arange(k), m)
    d = 0.4 + rng.normal(0, 1.0, k)[g] + rng.normal(0, 0.6, g.size)   # sd_total = sqrt(1.36)
    out = bland_altman_rm(d, np.zeros_like(d), g)
    assert out["bias"] == pytest.approx(0.4, abs=0.06)
    assert (out["loa_hi"] - out["bias"]) / 1.96 == pytest.approx(np.sqrt(1.36), rel=0.03)


# -- E30 (impedance): one value per participant ----------------------------------------------
def _impedance_rows(n_pid: int = 7, drop: float = 3.0) -> pd.DataFrame:
    rows = []
    for k in range(n_pid):
        for j, muscle in enumerate(("masseter_l", "digastric_l")):
            for order, take in ((1, "mvc"), (2, "motion")):
                rows.append({"pid": f"p{k}", "take": take, "order": order,
                             "minutes": 0.0 if order == 1 else 3.0, "muscle": muscle,
                             "z_kohm": 40.0 - (drop + 0.5 * k + 0.1 * j) * (order - 1) - k,
                             "contact_ok": True, "noise_ok": True, "line_ok": True,
                             "check_passed": True})
    return pd.DataFrame(rows)


def test_impedance_change_n_counts_participants_not_channels():
    from scipy import stats as sps

    from biomech.qc import impedance

    s = impedance.summary(_impedance_rows())
    row = s[s["take"] == "second_minus_first"].iloc[0]
    assert row["muscle"] == "all"
    assert int(row["n"]) == 7 and int(row["n_channels"]) == 14 and int(row["n_negative"]) == 7
    # all 7 participant medians negative: exact two-sided p = 2 / 2^7
    assert row["wilcoxon_p"] == pytest.approx(2 / 2 ** 7, abs=1e-12)
    per = [-(3.0 + 0.5 * k + 0.05) for k in range(7)]          # median of the two channels
    assert row["z_median_kohm"] == pytest.approx(np.median(per), abs=1e-9)
    assert row["wilcoxon_p"] == pytest.approx(sps.wilcoxon(per, method="exact").pvalue)
    pairs = s[(s["take"] == "second_minus_first") & (s["muscle"] == "all_channels")].iloc[0]
    assert int(pairs["n"]) == 14 and np.isnan(pairs["wilcoxon_p"])   # descriptive only


def test_impedance_r9_shape_gives_the_triaged_p():
    """14 participants, all medians negative (the r9 situation): exact p = 2/2^14 = 1.2e-4."""
    from biomech.qc import impedance

    row = impedance.summary(_impedance_rows(14)).iloc[-2]
    assert row["muscle"] == "all" and row["wilcoxon_p"] == pytest.approx(2 / 2 ** 14)


def test_impedance_holm_families_are_split():
    from biomech.qc import impedance

    rng = np.random.default_rng(4)
    rows = []
    for k in range(8):
        for muscle in ("masseter_l", "masseter_r", "digastric_l", "digastric_r"):
            rows.append({"pid": f"p{k}", "take": "motion", "order": 2, "minutes": 3.0,
                         "muscle": muscle, "z_kohm": rng.uniform(5, 60),
                         "noise_rms_uv": rng.uniform(1, 9), "line_peak_uv": rng.uniform(1, 40)})
    chk = pd.DataFrame(rows)
    emg = chk[["pid", "muscle"]].assign(snr_db=rng.normal(20, 5, len(chk)),
                                        rest_rms_uv=rng.uniform(1, 9, len(chk)),
                                        rest_lf_uv=rng.uniform(1, 20, len(chk)),
                                        mains_share_raw=rng.uniform(0, 0.5, len(chk)))
    a = impedance.associations(chk, emg, n_boot=50, seed=1)
    assert set(a.holm_family) == {"check_vs_recorded", "check_vs_check"}
    for fam, g in a.groupby("holm_family"):
        np.testing.assert_allclose(g.p_holm, multipletests(g.p_adj, method="holm")[1],
                                   atol=1e-12)
    assert (a[a.holm_family == "check_vs_recorded"].check_quantity == "z_kohm").all()
    assert len(a[a.holm_family == "check_vs_check"]) == 2


# -- coverage simulations (slow) -------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("n", [9, 14])
def test_bca_median_ci_coverage(n, record_property):
    """Coverage of the 95 % BCa CI of the median from ``describe`` (4000 resamples) on a
    skewed population (lognormal, sigma = 0.6), 400 samples. Measured (2026-10-29, seeds
    as written): n = 9 -> 0.968, n = 14 -> 0.920 (Monte-Carlo SE about 0.013). The
    bootstrap distribution of a median is discrete, so the coverage jumps with n rather than
    approaching 0.95 smoothly."""
    rng = np.random.default_rng(100 + n)
    true = 1.0                                         # median of lognormal(0, 0.6)
    hit = 0
    reps = 400
    for _ in range(reps):
        x = rng.lognormal(0.0, 0.6, n)
        d = describe(x, stat="median", seed=int(rng.integers(1 << 30)))
        hit += d["ci_lo"] <= true <= d["ci_hi"]
    cov = hit / reps
    record_property("coverage", cov)
    print(f"BCa median CI coverage at n = {n}: {cov:.3f}")
    assert cov >= 0.85


@pytest.mark.slow
def test_cluster_bootstrap_coverage_13_clusters(record_property):
    """Participant-cluster bootstrap (``bland_altman_rm_ci``) with 13 clusters, unequal
    cycles per cluster (5-12), between SD 1.0, within SD 0.8, true bias 0.5: coverage of
    the 95 % CI of the bias (asserted >= 0.85) and of the upper limit of agreement
    (recorded). Measured (2026-10-29, seeds as written): bias 0.937, upper LoA 0.837 --
    the percentile cluster bootstrap under-covers the limits of agreement with 13
    clusters (Monte-Carlo SE about 0.02)."""
    rng = np.random.default_rng(2026)
    sb, sw, bias = 1.0, 0.8, 0.5
    loa_true = bias + 1.96 * np.sqrt(sb ** 2 + sw ** 2)
    reps, hit_b, hit_l = 300, 0, 0
    for _ in range(reps):
        m = rng.integers(5, 13, 13)
        g = np.repeat(np.arange(13), m)
        d = bias + rng.normal(0, sb, 13)[g] + rng.normal(0, sw, g.size)
        ref = rng.normal(30, 5, g.size)
        ci = bland_altman_rm_ci(ref + d, ref, g, seed=int(rng.integers(1 << 30)), n_boot=1000)
        hit_b += ci["bias_ci_lo"] <= bias <= ci["bias_ci_hi"]
        hit_l += ci["loa_hi_ci_lo"] <= loa_true <= ci["loa_hi_ci_hi"]
    cov_b, cov_l = hit_b / reps, hit_l / reps
    record_property("coverage_bias", cov_b)
    record_property("coverage_loa_hi", cov_l)
    print(f"cluster bootstrap, 13 clusters: bias {cov_b:.3f}, upper LoA {cov_l:.3f}")
    assert cov_b >= 0.85
