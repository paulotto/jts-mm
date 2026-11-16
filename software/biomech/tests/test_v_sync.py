# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Temporal sync + joint registration on a synthetic take with a known truth (plan D2/D3).

Truth: offset c, α = 30 ppm, world X, body Y_b; JTS-MM 40 Hz with 12-ms exposure blur,
3 % dropped frames, 0.2 mm / 0.1° noise, CT detected 10 ms late/early; Qualisys 200 Hz,
0.05 mm / 0.03°, short occlusions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.validation import registration as REG
from biomech.validation import selection as SEL
from biomech.validation import sync as SC
from biomech.validation.synthetic import make_take


@pytest.fixture(scope="module")
def take():
    return make_take(seed=1)


@pytest.fixture(scope="module")
def frames(take):
    tS = SC.frame_times(take.t_trig_s * 1e6)
    ok = se3.valid(take.S["CT"])
    return tS, ok, SEL.sync_windows(take.phases, tS, ok), SEL.movement_intervals(take.phases,
                                                                                 tS, ok)


@pytest.fixture(scope="module")
def result(take, frames):
    tS, _, win, held = frames
    return SC.synchronise(tS, take.S, take.t_q_s, take.Q, win, heldout=held,
                          lag_hint_s=take.model.offset_s + 2.0, sigma_pos_mm=0.2,
                          sigma_rot_deg=0.1)


def _pose_err(A, B):
    dp, ang = REG.pose_error(A[None], B[None])
    return float(np.linalg.norm(dp)), float(ang[0])


def test_selection_windows(take, frames):
    _, _, win, held = frames
    assert [w.name for w in win] == ["sync_start", "sync_end"]
    t0 = take.phases[0].t_start_us * 1e-6
    assert win[0].t0 < t0 + 7.5 and win[0].t1 > t0 + 45.0       # bursts + motion + hold
    names = [h.name for h in held]
    assert names == ["rest", "open_close", "protrusion_retrusion", "left_right", "chewing",
                     "free_motion"]
    fm = held[-1]
    assert fm.t1 <= win[1].t0                                    # end sync cut off (D0b)
    for h in held:
        assert all(h.t1 <= w.t0 or h.t0 >= w.t1 for w in win)    # held out of the fit


def test_time_model_recovered(take, result):
    true = take.model.recentred(result.t_ref_s)
    m = result.fine.model
    assert abs(m.offset_s - true.offset_s) < 0.5e-3               # τ within 0.5 ms
    assert abs(m.alpha - true.alpha) < 2e-6                       # α within 2 ppm
    assert m.se_offset_s < 0.5e-3 and m.se_alpha < 2e-6
    # equivalent centred form t_Q = (1 + α)(t_S − t_ref) + τ
    t = np.array([1010.0, 1290.0])
    np.testing.assert_allclose(m.to_q(t), (1 + m.alpha) * (t - m.t_ref_s) + m.tau_s,
                               atol=1e-9)
    np.testing.assert_allclose(m.to_s(m.to_q(t)), t, atol=1e-9)


def test_registration_recovered(take, result):
    f = result.fine
    dp, ang = _pose_err(f.X, take.X)
    assert dp < 0.1 and ang < 0.05
    for b in ("CT", "MP", "HP"):
        dp, ang = _pose_err(f.Y[b], take.Y[b])
        assert dp < 0.1 and ang < 0.05, b


def test_heldout_rms_matches_noise(take, result):
    n = take.noise
    exp_pos = np.hypot(n["pos_s"], n["pos_q"])                    # 3-D RMS of both systems
    exp_rot = np.hypot(n["rot_s_deg"], n["rot_q_deg"])
    h = result.heldout
    mov = h[h["interval"].isin(SEL.MOVEMENT_TASKS) & h["body"].isin(["MP", "HP"])]
    assert len(mov) == 10 and (mov["n"] > 100).all()
    assert mov["pos_rms_mm"].between(0.85 * exp_pos, 1.15 * exp_pos).all()
    assert mov["rot_rms_deg"].between(0.85 * exp_rot, 1.15 * exp_rot).all()
    jaw = mov[mov["body"] == "MP"]
    assert (jaw["lag_resid_ms"].abs() < 0.5).all()                # no residual misalignment
    rel = h[h["body"] == "MP|HP"]
    assert (rel["pos_rms_mm"] < 2 * exp_pos).all()


def test_coarse_methods(take, result):
    true = take.model.recentred(result.t_ref_s)
    assert abs(result.coarse_lag_s - true.offset_s) < 0.03
    ef = result.edge_fit
    assert abs(ef.model.offset_s - true.offset_s) < 4e-3
    assert ef.delta_s == pytest.approx(-0.015, abs=0.005)         # 10 ms late + 5 ms early
    for r in result.xcorr:                                        # |ω| check, sub-ms
        assert abs(r["lag_s"] - float(true.lag(r["t_centre_s"]))) < 1e-3
        assert r["rho"] > 0.9


@pytest.mark.parametrize("err", [-2.0, 2.0])
def test_edges_converge_from_2s_error(take, frames, err):
    tS, ok, win, _ = frames
    eS = SC.visibility_edges(tS, ok, max_bracket_s=0.0375).within(win)
    eQ = SC.visibility_edges(take.t_q_s, se3.valid(take.Q["CT"]))
    lag0, votes = SC.coarse_offset_from_edges(eS, eQ, center=take.model.offset_s + err,
                                              half_range=3.0)
    t_ref = take.model.t_ref_s
    assert abs(lag0 - take.model.offset_s) < 0.03 and votes >= 10
    ef = SC.fit_edges(eS, eQ, lag0, t_ref)
    assert abs(ef.model.offset_s - take.model.offset_s) < 4e-3
    assert ef.n_rise >= 8 and ef.n_fall >= 8


def test_edges_vote_without_hint(take, frames):
    tS, ok, win, _ = frames
    eS = SC.visibility_edges(tS, ok, max_bracket_s=0.0375).within(win)
    eQ = SC.visibility_edges(take.t_q_s, se3.valid(take.Q["CT"]))
    lag0, _ = SC.coarse_offset_from_edges(eS, eQ)
    assert abs(lag0 - take.model.offset_s) < 0.03


def test_sliding_lags_linear(take, result):
    for method, tol in (("pose", 1.0), ("joint", 6.0)):
        ln = result.line[method]
        assert ln["n_windows"] >= 20, method
        assert abs(ln["slope_ppm"] - take.model.alpha * 1e6) < tol, method
        assert ln["n_steps"] == 0 and abs(ln["quad_ms_per_100s2"]) < 3 * ln["quad_se"] + 0.5
    sl = result.sliding
    assert {"MP", "HP", "joint", "pose"} <= set(sl["body"])


def test_sliding_pose_detects_frame_step(take, result, frames):
    """A one-frame (25 ms) re-association half-way through shows up as a step."""
    tS, *_ = frames
    tS2 = tS.copy()
    half = tS2 > take.model.t_ref_s
    tS2[half] -= 0.025                       # JTS-MM stamps one frame early after mid-take
    f = result.fine
    Qspl = {b: SC.PoseSpline(take.t_q_s, take.Q[b]) for b in ("MP", "HP")}
    sl = SC.sliding_pose_lags(tS2, take.S, Qspl, f.model, f.X, f.Y, ["MP", "HP"],
                              float(tS[0]), float(tS[-1]))
    ok = sl[sl["ok"]]
    t_mid = take.model.t_ref_s
    before = ok[ok["t_centre_s"] < t_mid - 8.0]["resid_ms"].median()
    after = ok[ok["t_centre_s"] > t_mid + 8.0]["resid_ms"].median()
    assert after - before == pytest.approx(25.0, abs=2.0)     # Q − S lag grows by one frame
    ln = SC.lag_line(sl, f.model, body="pose")
    assert ln["max_step_ms"] > 7.0 and ln["n_steps"] + ln["n_outliers"] >= 1


def test_robust_to_flipped_ct_branch(take, frames):
    """A 180° flip of the JTS-MM CT pose during the static hold (seen in real data) is
    gated out instead of biasing the fit."""
    tS, _, win, held = frames
    S = {b: v.copy() for b, v in take.S.items()}
    t0 = take.phases[0].t_start_us * 1e-6
    hold = (tS > t0 + 47.0) & (tS < t0 + 58.0) & se3.valid(S["CT"])
    F = np.eye(4)
    F[:3, :3] = Rotation.from_euler("x", 180, degrees=True).as_matrix()
    S["CT"][hold] = S["CT"][hold] @ F
    res = SC.synchronise(tS, S, take.t_q_s, take.Q, win, heldout=held[:2],
                         lag_hint_s=take.model.offset_s, sigma_pos_mm=0.2, sigma_rot_deg=0.1)
    true = take.model.recentred(res.t_ref_s)
    assert abs(res.fine.model.offset_s - true.offset_s) < 0.5e-3
    assert res.fine.n_gated["CT"] >= 0.9 * hold.sum()
    dp, ang = _pose_err(res.fine.Y["CT"], take.Y["CT"])
    assert dp < 0.1 and ang < 0.05


def test_ct_only_clock_is_robust_to_a_corrupted_head_body(take, frames):
    """R5b: the head body disagrees by several degrees while the CT moves next to it
    (real takes: QTM label swaps / occlusion). The CT-only fit keeps the clock and X;
    the joint fit is kept as a sensitivity result."""
    tS, _, win, held = frames
    Q = {b: v.copy() for b, v in take.Q.items()}
    start = win[0]
    q_in = (take.t_q_s >= take.model.to_q(np.array([start.t0]))[0]) & (
        take.t_q_s <= take.model.to_q(np.array([start.t1]))[0])
    R = np.eye(4)
    R[:3, :3] = Rotation.from_euler("y", 6.0, degrees=True).as_matrix()
    R[:3, 3] = [3.0, 0.0, -2.0]
    Q["HP"][q_in] = Q["HP"][q_in] @ R
    res = SC.synchronise(tS, take.S, take.t_q_s, Q, win, heldout=held[:2],
                         lag_hint_s=take.model.offset_s, sigma_pos_mm=0.2, sigma_rot_deg=0.1)
    true = take.model.recentred(res.t_ref_s)
    assert abs(res.fine.model.offset_s - true.offset_s) < 0.5e-3
    assert abs(res.fine.model.alpha - true.alpha) < 2e-6
    dp, ang = _pose_err(res.fine.X, take.X)
    assert dp < 0.1 and ang < 0.05
    assert res.fine_joint is not None and "HP" in res.fine_joint.n_frames
    assert set(res.fine.n_frames) == {"CT"} and "HP" in res.fine.Y
    assert np.isnan(res.fine.block_se("Y_HP")[0])


def test_clock_variants_and_criteria(take, frames):
    """R5c.3/V6: the variant fits recover the clock; the pre-registered criteria pass for
    the primary CT-only clock on clean synthetic data and choose it."""
    tS, _, win, held = frames
    res = SC.synchronise(tS, take.S, take.t_q_s, take.Q, win, heldout=held[:2],
                         lag_hint_s=take.model.offset_s, sigma_pos_mm=0.2, sigma_rot_deg=0.1,
                         variants=True)
    assert {"ct", "joint", "two_stage", "hybrid", "tukey"} <= set(res.variants)
    true = take.model.recentred(res.t_ref_s)
    for name, f in res.variants.items():
        assert abs(f.model.offset_s - true.offset_s) < 1e-3, name
    assert set(res.window_lags["window"]) == {"sync_start", "sync_end"}
    crit = {}
    for name, f in res.variants.items():
        wl = SC.window_lags(tS, take.S["CT"], SC.PoseSpline(take.t_q_s, take.Q["CT"]),
                            f.model, f.X, f.Y["CT"], res.windows)
        crit[name] = SC.clock_criteria(wl, res.line.get("joint", {}), f.model)
    assert crit["ct"]["windows_within_tol"] and crit["ct"]["windows_agree"]
    assert SC.choose_clock(crit) in ("ct", "two_stage")


# -- VD19: clock SEs from per-window HAC offset SEs; sliding-slope time model -------------------
def _ar1_series(rng, n, phi, sd=1.0, dims=None):
    shape = (n,) if dims is None else (n, dims)
    e = np.zeros(shape)
    z = rng.normal(size=shape) * sd * np.sqrt(1 - phi ** 2)
    e[0] = rng.normal(size=shape[1:]) * sd
    for i in range(1, n):
        e[i] = phi * e[i - 1] + z[i]
    return e


def test_newey_west_long_run_variance_of_ar1():
    """Var(Σu) of an AR(1) series ≈ n σ²(1 + φ)/(1 − φ); the white formula is far below."""
    rng = np.random.default_rng(0)
    n, phi = 40000, 0.6
    u = _ar1_series(rng, n, phi)
    v, L = SC.newey_west_var(u - u.mean())
    assert v / n == pytest.approx((1 + phi) / (1 - phi), rel=0.15)
    assert L >= SC.hac_bandwidth(rng.normal(size=n)) and L < n
    w = rng.normal(size=1000)
    assert SC.hac_bandwidth(w) == int(np.floor(4 * (1000 / 100) ** (2 / 9)))  # NW (1994) floor
    v0, L0 = SC.newey_west_var(w, bandwidth=0)
    assert L0 == 0 and v0 == pytest.approx(np.sum(w * w))


def test_drift_from_two_windows_is_the_plan_formula():
    """SE_drift = √(SE₁² + SE₂²)/Δt (ms/s → ppm ×1e3); offset at t_ref from the line."""
    wl = pd.DataFrame({"window": ["a", "b"], "t_centre_s": [100.0, 340.0],
                       "lag_ms": [0.02, -0.04], "se_ms_hac": [0.01, 0.02]})
    d = SC.drift_from_windows(wl, t_ref_s=220.0)
    assert d["n_windows"] == 2
    assert d["d_alpha_ppm"] == pytest.approx((-0.04 - 0.02) / 240.0 * 1e3)
    assert d["se_alpha_ppm_hac"] == pytest.approx(np.hypot(0.01, 0.02) / 240.0 * 1e3)
    assert d["d_offset_ms"] == pytest.approx(-0.01)                     # midpoint of the line
    assert d["se_offset_ms_hac"] == pytest.approx(0.5 * np.hypot(0.01, 0.02))
    assert np.isnan(d["se_alpha_ppm_jk"])                               # only two windows


def test_drift_jackknife_with_three_windows():
    wl = pd.DataFrame({"window": list("abc"), "t_centre_s": [0.0, 100.0, 200.0],
                       "lag_ms": [0.0, 0.05, 0.0], "se_ms_hac": [0.01, 0.01, 0.01]})
    d = SC.drift_from_windows(wl, t_ref_s=100.0)
    b = np.array([(0.0 - 0.05) / 100, 0.0, (0.05 - 0.0) / 100]) * 1e3   # leave one out
    assert d["se_alpha_ppm_jk"] == pytest.approx(np.sqrt(2 / 3 * np.sum((b - b.mean()) ** 2)))
    assert d["d_alpha_ppm"] == pytest.approx(0.0, abs=1e-12)
    one = SC.drift_from_windows(wl.iloc[:1], 0.0)
    assert one["n_windows"] == 1 and np.isnan(one["se_alpha_ppm_hac"])


def test_window_offsets_and_hac_columns(take, frames, result):
    """6-DoF window offsets under the fitted clock are ~0 with SEs that match the fit's
    information (white synthetic noise: HAC ≈ white); the V6 window lags carry both SEs."""
    tS, _, win, _ = frames
    f = result.fine
    Q = SC.PoseSpline(take.t_q_s, take.Q["CT"])
    wo = SC.window_offsets(tS, take.S["CT"], Q, f.model, f.X, f.Y["CT"], win,
                           scale=f.scale["CT"])
    assert list(wo["window"]) == ["sync_start", "sync_end"]
    assert (np.abs(wo["lag_ms"]) < 3 * wo["se_ms_hac"] + 0.01).all()
    assert np.allclose(wo["se_ms_hac"], wo["se_ms_white"], rtol=0.3)
    d = SC.drift_from_windows(wo, f.model.t_ref_s)
    assert 0.3 < d["se_alpha_ppm_hac"] / (f.model.se_alpha * 1e6) < 3.0
    wl = result.window_lags
    assert {"se_ms_ar1", "se_ms_hac", "hac_lags"} <= set(wl.columns)
    c = SC.clock_criteria(wl, result.line.get("joint", {}), f.model,
                          alpha_se_hac_ppm=d["se_alpha_ppm_hac"])
    for k in ("window_diff_se_ms_ar1", "window_diff_se_ms_hac", "windows_agree_hac",
              "sliding_consistent_hac", "pass_hac"):
        assert k in c
    assert c["alpha_se_hac_ppm"] == d["se_alpha_ppm_hac"]


def test_sliding_variant_time_model(result):
    f = result.fine
    line = {"slope_ppm": -44.0, "intercept_ms": 0.5}
    v = SC.sliding_variant(f, line)
    assert v.model.alpha == pytest.approx(-44e-6)
    assert v.model.offset_s == pytest.approx(f.model.offset_s + 0.5e-3)
    assert v.model.t_ref_s == f.model.t_ref_s and v.X is f.X
    assert SC.sliding_variant(f, {"slope_ppm": np.nan, "intercept_ms": 0.0}) is None
    # the stage's re-scoring input: same t_ref, so evaluate.time_model reproduces it
    from biomech.validation.evaluate import time_model
    m = time_model({"offset_s": v.model.offset_s, "alpha_ppm": v.model.alpha * 1e6,
                    "t_ref_s": f.model.t_ref_s})
    assert m.lag(f.model.t_ref_s + 100.0) == pytest.approx(v.model.lag(f.model.t_ref_s + 100.0))


@pytest.mark.slow
def test_clock_se_coverage_on_synthetic_takes():
    """VD19 coverage: 30 synthetic takes (make_take) with AR(1) pose noise added to the
    JTS-MM CT (φ = 0.9 per 40-Hz frame, 0.3 mm / 0.15°), CT-only fine fit, SEs of α and of
    the offset from the per-window HAC offset SEs vs the true clock. Recorded 2026-10-29:
    see the printed line (HAC / AR(1)-inflated / plain fit covariance)."""
    from scipy.spatial.transform import Rotation as Ro

    hits = {k: [] for k in ("a_hac", "a_ar1", "a_plain", "o_hac", "o_ar1")}
    ratio = []
    for k in range(30):
        tk = make_take(seed=200 + k)
        rng = np.random.default_rng(1000 + k)
        S = tk.S["CT"].copy()
        n = len(S)
        S[:, :3, 3] += _ar1_series(rng, n, 0.9, 0.3 / np.sqrt(3), 3)
        ok = np.isfinite(S[:, 0, 0])
        rv = _ar1_series(rng, n, 0.9, np.radians(0.15) / np.sqrt(3), 3)
        S[ok, :3, :3] = np.einsum("nij,njk->nik", S[ok, :3, :3], Ro.from_rotvec(rv[ok]).as_matrix())
        tS = SC.frame_times(tk.t_trig_s * 1e6)
        ok = se3.valid(S)
        win = SEL.sync_windows(tk.phases, tS, ok)
        inw = np.zeros(len(tS), bool)
        for w in win:
            inw |= (tS >= w.t0) & (tS <= w.t1)
        t_ref = float(np.mean(tS[inw & ok]))
        true = tk.model.recentred(t_ref)
        m0 = SC.TimeModel(true.offset_s + 2e-3, true.alpha + 5e-6, t_ref)
        Q = SC.PoseSpline(tk.t_q_s, tk.Q["CT"])
        sel = np.flatnonzero(inw & ok)
        X0, Y0 = REG.init_xy(Q(m0.to_q(tS[sel])), S[sel])
        f = SC.fit_fine(tS, {"CT": S}, {"CT": Q}, win, m0, X0, {"CT": Y0}, sigma_pos_mm=0.2,
                        sigma_rot_deg=0.1)
        d = SC.drift_from_windows(SC.window_offsets(tS, S, Q, f.model, f.X, f.Y["CT"], win,
                                                    scale=f.scale["CT"]), t_ref)
        ea, eo = (f.model.alpha - true.alpha) * 1e6, (f.model.offset_s - true.offset_s) * 1e3
        se_a, se_o = f.model.se_alpha * 1e6, f.model.se_offset_s * 1e3
        hits["a_hac"].append(abs(ea) <= 1.96 * d["se_alpha_ppm_hac"])
        hits["a_ar1"].append(abs(ea) <= 1.96 * se_a * f.se_inflation)
        hits["a_plain"].append(abs(ea) <= 1.96 * se_a)
        hits["o_hac"].append(abs(eo) <= 1.96 * d["se_offset_ms_hac"])
        hits["o_ar1"].append(abs(eo) <= 1.96 * se_o * f.se_inflation)
        ratio.append(d["se_alpha_ppm_hac"] / se_a)
    cov = {k: float(np.mean(v)) for k, v in hits.items()}
    print("VD19 coverage", cov, "HAC/plain drift SE ratio median", float(np.median(ratio)))
    assert cov["a_hac"] >= 0.9 and cov["o_hac"] >= 0.9
