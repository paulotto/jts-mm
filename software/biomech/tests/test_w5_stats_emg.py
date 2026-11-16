# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Audit W5 regressions: cohort statistics and EMG references."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from biomech.emg.prepare import effort_mask_frac, max_window_mean
from biomech.stats import cohort_emg as ce
from biomech.stats import cohort_kinematics as ck
from biomech.stats.core import signed_rank_z, wilcoxon_one_sample


def test_coupling_band_tolerates_missing_participants():
    grid = np.arange(0.0, 20.0)
    good = np.tile(grid * 0.5, (5, 1))
    curves = {"opening_mm": [good, None, np.empty((0, 0)), good + 1, good - 1]}
    out = ck.coupling_band(curves, grid)
    assert out["opening_mm__subjects"].shape == (5, grid.size)
    assert np.isnan(out["opening_mm__subjects"][1]).all()
    np.testing.assert_allclose(out["opening_mm__median"], grid * 0.5)


@pytest.mark.parametrize("masked", [0.0, 0.4])
def test_max1s_ignores_masked_samples(masked):
    rng = np.random.default_rng(0)
    fs = 1000
    env = np.full((5 * fs, 2), 10.0)
    env[2000:3000] = 100.0                          # 1-s plateau = the true max-1-s value
    m = rng.random(env.shape) < masked
    env[m] = np.nan
    out = max_window_mean(env, fs)
    np.testing.assert_allclose(out, 100.0, rtol=0.05)


def test_max1s_nan_when_too_little_valid():
    fs = 100
    env = np.full((500, 1), 50.0)
    env[::5] = 50.0
    env[np.arange(500) % 5 != 0] = np.nan             # 80 % masked everywhere
    assert np.isnan(max_window_mean(env, fs)).all()


def test_wilcoxon_r_bounded_and_sided():
    x = np.array([3.0, 5, 2, 8, 6, 7, 4, 9, 1, 10])
    for alt in ("two-sided", "greater"):
        r = wilcoxon_one_sample(x, alt)["r"]
        assert 0 < r <= 1
    assert wilcoxon_one_sample(x, "two-sided")["r"] == pytest.approx(
        wilcoxon_one_sample(x, "greater")["r"])
    assert wilcoxon_one_sample(-x, "less")["r"] < 0
    # z equals scipy's normal approximation (two-sided, no ties)
    from scipy import stats

    v = np.array([1.5, -0.3, 2.2, 0.7, -1.1, 3.4, 0.9, 1.8])
    z, n = signed_rank_z(v)
    res = stats.wilcoxon(v, method="approx", correction=False)
    assert abs(z) == pytest.approx(stats.norm.isf(res.pvalue / 2), rel=1e-6)


def _profiles(order, rng):
    base = {m: rng.normal(size=(6, 101)) + (2.0 if m == "masseter_l" else 0.0)
            for m in ce.MUSCLES}
    P = np.stack([base[m] for m in order], axis=2)
    return {"names": np.array(order), "chewing__pct": P, "open_close__pct": P,
            "open_close__theta_deg": rng.normal(size=(6, 101))}


def test_spm_indexes_muscles_by_name():
    pytest.importorskip("spm1d")
    canon = list(ce.MUSCLES)
    perm = canon[::-1]
    ok = set(canon)
    res_a, res_b = {}, {}
    for k in range(8):
        rng = np.random.default_rng(k)
        res_a[f"s{k:02d}"] = _profiles(canon, rng)
        rng = np.random.default_rng(k)
        res_b[f"s{k:02d}"] = _profiles(perm, rng)
    shape_ok = {p: ok for p in res_a}
    ta, aa = ce.spm_tests(res_a, shape_ok)
    tb, ab = ce.spm_tests(res_b, shape_ok)
    np.testing.assert_allclose(aa["spm1__z"], ab["spm1__z"])
    np.testing.assert_allclose(aa["spm2__z"], ab["spm2__z"])


def test_spm_skips_participant_without_channel():
    pytest.importorskip("spm1d")
    profs = {}
    for k in range(6):
        rng = np.random.default_rng(k)
        order = [m for m in ce.MUSCLES if not (k == 0 and m == "masseter_l")]
        pr = _profiles(list(ce.MUSCLES), rng)
        keep = [list(ce.MUSCLES).index(m) for m in order]
        pr["chewing__pct"] = pr["chewing__pct"][:, :, keep]
        pr["open_close__pct"] = pr["open_close__pct"][:, :, keep]
        pr["names"] = np.array(order)
        profs[f"s{k:02d}"] = pr
    shape_ok = {p: set(ce.MUSCLES) for p in profs}
    _, arr = ce.spm_tests(profs, shape_ok)
    assert arr["spm1__left"].shape[0] == 5


def test_import_does_not_change_global_warning_filters():
    import importlib

    before = list(warnings.filters)
    importlib.reload(ce)
    importlib.reload(ck)
    assert warnings.filters == before


def test_per_phase_line_frequency_follows_grid_drift():
    from scipy.signal import butter, sosfiltfilt

    from biomech.emg.pipeline import clean, refine_line_hz, take_line_hz

    fs, T = 2000.0, 300.0
    t = np.arange(0, T, 1 / fs)
    rng = np.random.default_rng(0)
    sos = butter(4, [20, 450], btype="band", fs=fs, output="sos")
    emg = sosfiltfilt(sos, rng.normal(size=t.size))
    emg *= 3.0 / emg.std()
    f = 50.0 + 0.05 * t / T                                  # grid drifts 0.05 Hz in 5 min
    ph = 2 * np.pi * np.cumsum(f) / fs
    hum = sum(a * np.sin(k * ph + k) for k, a in ((1, 15.0), (3, 5.0), (5, 3.0), (9, 1.0)))
    x = (emg + hum)[:, None]
    t_us = (t * 1e6).astype(np.int64)
    take = take_line_hz(x[: int(60 * fs)], fs, t_us[: int(60 * fs)])
    seg = slice(int(250 * fs), int(260 * fs))
    refined = refine_line_hz(x[seg], fs, t_us[seg], take)
    assert abs(refined - f[seg].mean()) < 0.003 and abs(take - f[seg].mean()) > 0.03
    res_take = np.std(clean(x[seg], fs, t_us[seg], line_hz=take)[0][:, 0] - emg[seg])
    res_ref = np.std(clean(x[seg], fs, t_us[seg], line_hz=refined)[0][:, 0] - emg[seg])
    assert res_ref < 0.7 < res_take


def test_reference_usability_ignores_masked_pauses():
    fs = 1000
    env = np.full((10 * fs, 1), 5.0)                     # pauses
    env[1000:3000] = env[5000:7000] = 300.0             # two efforts
    mask = np.zeros(env.shape, bool)
    mask[3200:4800] = True                               # LF artifact between the efforts
    p95 = np.percentile(env, 95, axis=0)
    assert mask.mean() == pytest.approx(0.16)
    assert effort_mask_frac(mask, env, p95)[0] == 0.0
    mask[1000:1400] = True                               # artifact inside an effort counts
    assert effort_mask_frac(mask, env, p95)[0] == pytest.approx(0.1)


def test_mixed_model_rejects_a_diverged_fit(monkeypatch):
    """A fit with +inf log-likelihood / infinite SE is not accepted (lbfgs did that)."""
    from biomech.stats import core

    rng = np.random.default_rng(3)
    pid = np.repeat([f"p{i}" for i in range(10)], 12)
    task = np.tile(["a", "b", "c"], 40)
    y = 2.0 + 0.5 * (task == "b") + np.repeat(rng.normal(0, 0.5, 10), 12) \
        + rng.normal(0, 0.3, 120)
    df = pd.DataFrame({"pid": pid, "task": task, "y": y})
    good = core.mixed_model(df, "y ~ C(task)")
    assert good.loc[good.term == "Intercept", "coef"].iloc[0] == pytest.approx(2.0, abs=0.4)
    assert np.isfinite(good.se).all() and good.group_var.iloc[0] > 0

    import statsmodels.regression.mixed_linear_model as mlm

    real_fit = mlm.MixedLM.fit

    def fit(self, *a, method=None, **kw):
        r = real_fit(self, *a, method="bfgs" if method != "lbfgs" else "bfgs", **kw)
        if method == "lbfgs":                                # simulate the divergence
            r.llf = np.inf
        return r

    monkeypatch.setattr(mlm.MixedLM, "fit", fit)
    t = core.mixed_model(df, "y ~ C(task)")
    assert t.optimizer.iloc[0] == "bfgs"


def test_cycles_needed_uses_each_sides_own_mdc():
    from biomech.stats.cohort_misc import cycles_needed_table

    rng = np.random.default_rng(0)
    rows = []
    for side, sd in (("left", 0.5), ("right", 3.0)):          # right is far noisier
        for k in range(12):
            rows.append({"pid": "s01", "task": "left_right", "side": side, "ok": True,
                         "cycle": k, "lateral_mm": (1 if side == "left" else -1) * (10 + rng.normal(0, sd))})
    cyc = pd.DataFrame(rows)
    t = cycles_needed_table(cyc, {("left_right", "lateral_mm", "left"): 2.0,
                                  ("left_right", "lateral_mm", "right"): 2.0})
    n = t.set_index("side").n_needed
    # the noisy side needs more cycles, or never reaches the target (NaN, no extrapolation)
    assert set(t.side) == {"left", "right"} and (np.isnan(n["right"]) or n["left"] < n["right"])
    assert np.isfinite(n["left"])
    assert (t.n_available == 12).all()                        # sides not pooled
