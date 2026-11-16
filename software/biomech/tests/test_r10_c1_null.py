# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""the C1 hinge null (Monte-Carlo p, slow and motion noise,
cohort Wilcoxon), the teeth-only hinge fit, the condyle sensitivities and the D18 depth
neighbours. Known truth: an exact synthetic jaw (perfect hinge, optional translation)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from scipy.signal import lfilter

from biomech.kinematics import se3
from biomech.kinematics.clinical import REST_HP_S, rest_residuals, theta_deg
from biomech.kinematics.coupling import HINGE_POINTS, Opening, hinge_fits
from biomech.kinematics.hinge_fit import TEETH, fit_hinge
from biomech.stats import cohort_kinematics as ck
from biomech.timebase import odd_window
from biomech.uncertainty import montecarlo as mc

PTS = {"incisor": np.array([0.0, 0, -3]), "left": np.array([-37.0, 23, -1]),
       "right": np.array([-37.0, -23, -1]), "tmj_left": np.array([-88.0, 52, 18]),
       "tmj_right": np.array([-88.0, -52, 18])}
N_AX, S_AX = np.array([0.0, 1.0, 0.0]), np.array([-88.0, 0.0, 18.0])


def _stack(T: np.ndarray, pts: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    P = np.stack([pts[r] for r in HINGE_POINTS])
    A = np.broadcast_to(se3.apply(T[0][None], P[None]), (len(T), len(P), 3))
    B = np.einsum("fij,kj->fki", T[:, :3, :3], P) + T[:, None, :3, 3]
    return np.ascontiguousarray(A), B


def _hinge_motion(translate_mm: float = 0.0, n: int = 60) -> np.ndarray:
    """Opening 0 -> 25 deg about the true axis; translation (anterior-inferior) after 10 deg."""
    th = np.linspace(0, np.radians(25.0), n)
    T = mc._hinge_pose(N_AX, S_AX, th)
    frac = np.clip((th - np.radians(10.0)) / (th[-1] - np.radians(10.0)), 0, 1)
    T[:, :3, 3] += translate_mm * frac[:, None] * np.array([1.0, 0.0, -0.3])
    return T


# -- D25: the teeth-only hinge fit ----------------------------------------------------------------
def test_teeth_fit_exact_on_a_perfect_hinge():
    A, B = _stack(_hinge_motion(), PTS)
    h = fit_hinge(A, B, s0=S_AX + [5.0, 0, -5.0], n0=N_AX, points=TEETH, model="hinge_teeth")
    assert h.model == "hinge_teeth" and h.residual.shape == (60, 5)
    assert h.rms(slice(None)) < 1e-3                  # residual at every point, condyles too
    assert np.allclose(np.abs(h.n), N_AX, atol=1e-4)


def test_teeth_fit_does_not_depend_on_the_palpated_condyles():
    """Moving the palpated condyle points (all points are rigid with the mandible, so the
    poses stay) changes the five-point fit but not the teeth-only fit at the incisor."""
    T = _hinge_motion(translate_mm=6.0)
    moved = dict(PTS, tmj_left=PTS["tmj_left"] + [25.0, 5, -15],
                 tmj_right=PTS["tmj_right"] + [25.0, -5, -15])
    fits = [hinge_fits(*_stack(T, p)) for p in (PTS, moved)]
    assert set(fits[0]) == {"hinge_fit", "hinge_tmj", "rot_trans", "hinge_teeth"}
    a, b = (f["hinge_teeth"].rms(0) for f in fits)
    assert a == pytest.approx(b, rel=1e-4)
    assert abs(fits[0]["hinge_fit"].rms(0) - fits[1]["hinge_fit"].rms(0)) > 1e-3
    # weighting only the teeth can only lower the teeth residual
    assert fits[0]["hinge_teeth"].rms(slice(0, 3)) <= fits[0]["hinge_fit"].rms(slice(0, 3)) + 1e-9
    assert "hinge_teeth" not in hinge_fits(*_stack(T, PTS), teeth=False)


def test_fit_hinge_unchanged_without_points():
    A, B = _stack(_hinge_motion(translate_mm=5.0), PTS)
    a = fit_hinge(A, B, s0=S_AX, n0=N_AX)
    b = fit_hinge(A, B, s0=S_AX, n0=N_AX, points=(0, 1, 2, 3, 4))
    assert a.model == "hinge_fit" and np.allclose(a.residual, b.residual, atol=1e-9)


# -- D24: Monte-Carlo p, variants, summarize ---------------------------------------------------
def test_mc_pvalue():
    d = np.array([1.0, 2.0, 3.0, np.nan])
    assert mc.mc_pvalue(2.0, d) == pytest.approx(3 / 4)       # k = 2 (2 and 3), N = 3
    assert mc.mc_pvalue(5.0, d) == pytest.approx(1 / 4)       # never 0
    assert np.isnan(mc.mc_pvalue(np.nan, d)) and np.isnan(mc.mc_pvalue(1.0, np.array([np.nan])))


def test_summarize_p_and_teeth_tag():
    null = {"invivo": np.arange(1, 101, dtype=float) / 100,
            "invivo_teeth": np.arange(1, 101, dtype=float) / 50}
    row = mc.summarize({"hinge_fit": 0.995, "hinge_teeth": 0.5}, null, {})
    assert row["n_sims_invivo"] == 100 and row["p_mc_invivo"] == pytest.approx(2 / 101)
    assert row["above_invivo"] is True
    # the teeth null is compared with the teeth-only residual: 0.5 of 0.02 ... 2.0
    assert row["above_invivo_teeth"] is False
    assert row["p_mc_invivo_teeth"] == pytest.approx(77 / 101)    # k = 76 (25 ... 100)
    assert row["ratio_invivo_teeth"] == pytest.approx(0.5 / np.percentile(null["invivo_teeth"], 99))


def test_residual_noise_scale_uses_the_same_draws():
    runs = [np.random.default_rng(0).normal(size=(80, 6))]
    a = mc.ResidualNoise(runs, 9).draw(200, np.random.default_rng(5))
    b = mc.ResidualNoise(runs, 9, scale=1.7).draw(200, np.random.default_rng(5))
    np.testing.assert_allclose(b, 1.7 * a)


def _ops_from(T: np.ndarray, starts: list[int], peaks: list[int]) -> list[Opening]:
    return [Opening(k, T[a:b + 1], theta_deg(T[a:b + 1], T[a])[0])
            for k, (a, b) in enumerate(zip(starts, peaks))]


def _continuous(fs=40.0, dur=0.3, total=12.0):
    per = 2 * dur + 0.4
    t = np.arange(0, total, 1 / fs)
    ph = t % per
    th = np.where(ph < dur, 0.5 * (1 - np.cos(np.pi * ph / dur)),
                  np.where(ph < 2 * dur, 0.5 * (1 + np.cos(np.pi * (ph - dur) / dur)), 0.0))
    T = mc._hinge_pose(N_AX, S_AX, th * np.radians(25.0))
    starts = [int(round(k * per * fs)) for k in range(1, int(total / per) - 1)]
    return T, starts, [a + int(round(dur * fs)) for a in starts]


def _rest(n=2400, rho=0.8, seed=3):
    """A rest phase: AR(1) pose noise (0.05 deg, 0.03 mm) plus a slow drift."""
    rng = np.random.default_rng(seed)

    def ar(sd):
        return lfilter([1.0], [1.0, -rho], rng.normal(0, sd * np.sqrt(1 - rho ** 2), (n, 3)),
                       axis=0)

    from scipy.spatial.transform import Rotation as R
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R.from_rotvec(np.radians(ar(0.05))).as_matrix()
    T[:, :3, 3] = np.linspace(0, 0.5, n)[:, None] * np.array([1.0, 0, 0]) + ar(0.03)
    return T


def test_hinge_nulls_tags_seeds_and_primary_unchanged():
    fs = 40.0
    jm = SimpleNamespace(fs=fs, points_mp=PTS)
    Tfull, starts, peaks = _continuous(fs)
    ops = _ops_from(Tfull, starts, peaks)
    T_rest = _rest()
    sd_t, sd_r = np.full(3, 0.03), np.full(3, 0.05)
    p = {"n_sims": 4, "slow_noise": True, "motion_noise_scale": None, "teeth_null": True}
    out = mc.hinge_nulls(jm, ops, N_AX, S_AX, T_rest, sd_t, sd_r, p, 6.0, 2, 30, 10)
    assert set(out) == {"bench", "invivo", "invivo_teeth", "invivo_slow"}
    # the primary in-vivo null is the former stage's call: 1-s residuals, seed + 1
    noise = mc.ResidualNoise(rest_residuals(T_rest, fs), odd_window(REST_HP_S, fs))
    ref = mc.null_distribution(jm, ops, N_AX, S_AX, sd_t, sd_r, 4, 6.0, 30, 11, 2, noise=noise)
    np.testing.assert_array_equal(out["invivo"], ref)
    # the teeth fit on the same draws leaves the five-point null unchanged
    p2 = dict(p, teeth_null=False, slow_noise=False, motion_noise_scale=1.5)
    out2 = mc.hinge_nulls(jm, ops, N_AX, S_AX, T_rest, sd_t, sd_r, p2, 6.0, 2, 30, 10)
    assert set(out2) == {"bench", "invivo", "invivo_motion"}
    np.testing.assert_array_equal(out2["invivo"], out["invivo"])
    np.testing.assert_array_equal(out2["bench"], out["bench"])
    # more noise, larger null residual
    assert np.median(out2["invivo_motion"]) > np.median(out["invivo"])
    # no rest phase: every in-vivo null falls back to white noise of the floor
    out3 = mc.hinge_nulls(jm, ops, N_AX, S_AX, None, sd_t, sd_r, dict(p, motion_noise_scale=2.0),
                          6.0, 2, 30, 10)
    assert all(np.isfinite(v).all() for v in out3.values())


def test_condyle_axis_and_tmj_null_tags():
    """D33: the condyle axis is the one through the closed-pose condyle points; tmj_null adds
    bench_tmj and invivo_tmj (own seeds) and leaves every other null unchanged."""
    fs = 40.0
    jm = SimpleNamespace(fs=fs, points_mp=PTS)
    Tfull, starts, peaks = _continuous(fs)
    ops = _ops_from(Tfull, starts, peaks)
    n_c, s_c = mc.condyle_axis(jm, ops, 30)
    # left minus right (+y), the direction in which the measured opening angle is positive:
    # the ideal null motion opens the jaw like the data (the incisor moves down)
    assert n_c @ N_AX > 1.0 - 1e-9 and np.allclose(s_c, S_AX)
    op = ops[0]
    T = mc._hinge_pose(n_c, s_c, np.radians(op.theta[-1:]))[0] @ op.T[0]
    inc0, inc1 = se3.apply(op.T[0], PTS["incisor"]), se3.apply(T, PTS["incisor"])
    assert inc1[2] < inc0[2] - 5.0
    np.testing.assert_allclose(inc1, se3.apply(op.T[-1], PTS["incisor"]), atol=1e-6)
    T_rest = _rest()
    sd_t, sd_r = np.full(3, 0.03), np.full(3, 0.05)
    p = {"n_sims": 4, "slow_noise": False, "motion_noise_scale": None, "teeth_null": False}
    base = mc.hinge_nulls(jm, ops, N_AX, S_AX, T_rest, sd_t, sd_r, p, 6.0, 2, 30, 10)
    out = mc.hinge_nulls(jm, ops, N_AX, S_AX, T_rest, sd_t, sd_r, dict(p, tmj_null=True),
                         6.0, 2, 30, 10)
    assert set(out) == {"bench", "invivo", "bench_tmj", "invivo_tmj"}
    for k in base:
        np.testing.assert_array_equal(out[k], base[k])
    assert np.isfinite(out["invivo_tmj"]).all()
    assert mc.model_of("invivo_tmj") == "hinge_tmj" and mc.model_of("invivo") == "hinge_fit"


def test_tmj_null_separates_hinge_and_translation():
    """D33: a perfect hinge through the condyles stays inside its condyle-hinge null; the same
    opening with 4 mm of condylar translation after 10 deg is far above it."""
    fs = 40.0
    jm = SimpleNamespace(fs=fs, points_mp=PTS)
    P = np.stack([PTS[r] for r in HINGE_POINTS])
    rng = np.random.default_rng(5)
    noise = mc.ResidualNoise(rest_residuals(_rest(), fs), odd_window(REST_HP_S, fs))
    res = {}
    for name, mm in (("hinge", 0.0), ("translating", 4.0)):
        Tfull, starts, peaks = _continuous(fs)
        if mm:
            th = theta_deg(Tfull, Tfull[0])[0]
            frac = np.clip((th - 10.0) / 15.0, 0, 1)
            Tfull = Tfull.copy()
            Tfull[:, :3, 3] += mm * frac[:, None] * np.array([1.0, 0.0, -0.3])
        Tn = se3.lowpass(mc._noisy(Tfull, 0.0, 0.0, rng, noise), fs, 6.0, 2)
        ops = _ops_from(Tn, starts, peaks)
        A, B = mc._stack(P, [o.T for o in ops], 30)
        obs = hinge_fits(A, B, teeth=False)["hinge_tmj"].rms(0)
        n_c, s_c = mc.condyle_axis(jm, ops, 30)
        null = mc.null_distributions(jm, ops, n_c, s_c, 0.0, 0.0, 99, 6.0, 30, 7,
                                     noise=noise, through_condyles=True)["hinge_tmj"]
        res[name] = mc.mc_pvalue(obs, null)
    assert res["hinge"] > 0.05
    assert res["translating"] <= 0.01


@pytest.mark.slow
def test_c1_null_is_calibrated_on_a_perfect_hinge():
    """H0 true (a perfect hinge, pose noise resampled from the same ResidualNoise, the data
    low-passed as one segment): the Monte-Carlo p of the observed residual is about uniform,
    so "above P99" rejects about 1 % and the null is neither anti-conservative nor void."""
    fs = 40.0
    Tfull, starts, peaks = _continuous(fs)
    jm = SimpleNamespace(fs=fs, points_mp=PTS)
    noise = mc.ResidualNoise(rest_residuals(_rest(), fs), odd_window(REST_HP_S, fs))
    P = np.stack([PTS[r] for r in HINGE_POINTS])
    rng = np.random.default_rng(1)
    ps = []
    for rep in range(80):
        Tn = se3.lowpass(mc._noisy(Tfull, 0.0, 0.0, rng, noise), fs, 6.0, 2)
        ops = _ops_from(Tn, starts, peaks)
        A, B = mc._stack(P, [o.T for o in ops], 30)
        h = fit_hinge(A, B, s0=S_AX, n0=N_AX)
        null = mc.null_distribution(jm, ops, h.n, h.s, 0.0, 0.0, 99, 6.0, 30, 100 + rep,
                                    noise=noise)
        ps.append(mc.mc_pvalue(h.rms(0), null))
    ps = np.array(ps)
    assert sps.kstest(ps, "uniform").pvalue > 0.01
    assert 0.3 < np.median(ps) < 0.7
    assert np.mean(ps <= 0.05) < 0.15


# -- D24 cohort test, D25 / D29 condyle sensitivities ------------------------------------------
def test_wilcoxon_greater_exact():
    w, p, exact = ck.wilcoxon_greater(np.linspace(0.1, 1.4, 14))
    assert exact and w == pytest.approx(105.0) and p == pytest.approx(0.5 ** 14)
    w, p, exact = ck.wilcoxon_greater(np.array([0.1, -0.1, 0.3]))
    assert not exact                                  # tied |x|: normal approximation
    assert np.isnan(ck.wilcoxon_greater(np.array([0.2]))[1])


def _noise_table():
    ratio = np.array([1.5, 1.26, 1.75, 1.08, 1.02, 1.37, 1.58, 2.67, 1.12, 3.22, 0.48, 1.09,
                      1.11, 1.18])
    pids = [f"s{i:02d}" for i in range(1, 15)]
    return pd.DataFrame({"pid": pids, "ratio_invivo": ratio, "above_invivo": ratio > 1,
                         "p_mc_invivo": np.where(ratio > 1, 0.001, 0.6),
                         "ratio_invivo_slow": ratio * 0.8, "above_invivo_slow": ratio * 0.8 > 1})


def test_hinge_noise_tests_rows():
    noise = _noise_table()
    plaus = set(noise.pid) - {"s02", "s03", "s11"}
    t = ck.hinge_noise_tests(noise, plaus, 0).set_index(["noise", "group"])
    r = t.loc[("invivo", "all")]
    assert r.n_tested == 14 and r.above_n == 13
    ref = sps.wilcoxon(np.log(noise.ratio_invivo), alternative="greater", method="exact")
    assert r.wilcoxon_log_ratio_p == pytest.approx(ref.pvalue) and r.wilcoxon_log_ratio_exact
    assert r.p_mc_median == pytest.approx(0.001)
    rp = t.loc[("invivo", "plausible_condyles")]
    assert rp.n_tested == 11 and rp.above_n == 11          # the one below (s11) is implausible
    assert ("invivo_slow", "all") in t.index and ("bench", "all") not in t.index
    assert np.isnan(t.loc[("invivo_slow", "all")].p_mc_median)    # no p column: NaN


def _subj():
    """Condyle depths and reasons as in r9 (D18 at 40 mm: s02 too shallow, s03 too close,
    s11 below the incisal level)."""
    depth = {"s01": 53.9, "s02": 22.3, "s03": 46.2, "s06": 43.1, "s08": 43.6, "s11": 65.1,
             "s14": 78.9, "s99": np.nan}
    reason = {"s02": "too_shallow", "s03": "too_close", "s11": "below_incisal_level"}
    rows = []
    for k, (pid, d) in enumerate(depth.items()):
        cp = np.nan if pid == "s99" else float(pid not in reason)
        rows.append({"pid": pid, "condyles_plausible": cp, "condyles_reason": reason.get(pid, ""),
                     "frame_tmj_left_x": -d - 3.0, "frame_tmj_right_x": -d + 3.0,
                     "condyle_left_disp_mm": 10.0 + k, "condyle_right_disp_mm": 12.0 + k})
    return pd.DataFrame(rows)


def test_plausible_at_depth_reproduces_the_stored_flag_and_neighbours():
    s = _subj()
    at40 = ck.plausible_at_depth(s, 40.0)
    assert at40.tolist() == pd.to_numeric(s.condyles_plausible).eq(1).tolist()
    assert set(s.pid[ck.plausible_at_depth(s, 30.0)]) == set(s.pid[at40])       # s02 at 22 mm
    assert set(s.pid[at40]) - set(s.pid[ck.plausible_at_depth(s, 50.0)]) == {"s06", "s08"}
    assert ck.condyle_depth_mm(s).iloc[0] == pytest.approx(53.9)


def test_condyle_sensitivity_translation_and_groups():
    s = _subj()
    hinge = pd.DataFrame([{"pid": p, "model": m, "rms_incisor_mm": v,
                           "translation_range_mm": 5.0 if m == "rot_trans" else np.nan,
                           "rms_condyles_mm": 1.0}
                          for p in s.pid for m, v in (("hinge_fit", 1.0), ("hinge_tmj", 3.0),
                                                      ("rot_trans", 0.5), ("hinge_teeth", 0.9))])
    fha = pd.DataFrame({"pid": s.pid, "dist_tmj": 20.0})
    cs = ck.condyle_sensitivity(s, hinge, 0, fha).set_index(["metric", "group"])
    assert {"plausible_condyles_30mm", "plausible_condyles_50mm", "condyles_unknown"} <= \
        set(cs.index.get_level_values("group"))
    assert cs.loc[("translation_max_mm", "all"), "median"] == pytest.approx(
        np.median(12.0 + np.arange(8)))
    assert cs.loc[("translation_mean_mm", "all"), "median"] == pytest.approx(
        np.median(11.0 + np.arange(8)))
    assert cs.loc[("hinge_rms_incisor_hinge_teeth", "plausible_condyles"), "n"] == 4
    assert cs.loc[("translation_max_mm", "plausible_condyles_50mm"), "n"] == 2
    assert cs.loc[("dist_tmj", "plausible_condyles"), "median"] == pytest.approx(20.0)
