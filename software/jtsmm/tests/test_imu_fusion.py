# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic tests of the optical + IMU orientation fusion (jtsmm.imu.fusion)."""

from __future__ import annotations

import numpy as np
import pytest

from jtsmm.imu import fusion as FU
from jtsmm.imu.batch import quat_mul, quat_prepare, quat_to_matrix


def _rms(e: np.ndarray) -> float:
    e = e[np.isfinite(e)]
    return float(np.sqrt(np.mean(e ** 2)))


@pytest.fixture(scope="module")
def scen():
    """90-s scenario: rest 2-10 s, two dropouts, 1 % flips; calibrate on everything except
    the evaluation window 50-80 s."""
    sim = FU.simulate(90.0, seed=3, dropouts=((30.0, 30.5), (60.0, 61.0)))
    t, R = sim["t_opt_s"], sim["R_opt"]
    imu = FU.ImuTrack.from_stream(sim["imu_t_us"], sim["imu_q"])
    ev = (t >= 50.0) & (t <= 80.0)
    train = ~ev & (t > 10.0)
    cal = FU.calibrate(t, R, imu, train)
    st = FU.settings_from_sensors(t, R, imu, cal, [sim["rest"]], sim["quality"],
                                  train_mask=train)
    tq = sim["t_query_s"]
    f1 = FU.f1_complementary(t, R, imu, cal, st, tq, quality=sim["quality"])
    f2 = FU.f2_kalman(t, R, imu, cal, st, tq, quality=sim["quality"])
    return {"sim": sim, "imu": imu, "cal": cal, "st": st, "ev": ev, "train": train,
            "f1": f1, "f2": f2, "f0": FU.f0_optical(t, R, tq)}


# -- SO(3) and stream helpers -----------------------------------------------------------
def test_exp_log_roundtrip_including_near_pi():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 3))
    v *= (rng.uniform(0, np.pi - 1e-3, 200) / np.linalg.norm(v, axis=1))[:, None]
    v[:3] = [[0, 0, 0], [1e-9, 0, 0], [np.pi - 1e-6, 0, 0]]
    back = FU.log_so3(FU.exp_so3(v))
    assert np.allclose(back, v, atol=1e-6)
    assert np.isnan(FU.log_so3(np.full((1, 3, 3), np.nan))).all()


def test_increments_telescope_to_relative_rotation():
    rng = np.random.default_rng(1)
    q = quat_prepare(rng.normal(size=(50, 4)))
    d = FU.imu_increments(q)
    acc = q[0]
    for k in range(len(d)):
        acc = quat_mul(acc, d[k])
    R_end = quat_to_matrix(acc)
    assert FU.rot_error_deg(R_end, quat_to_matrix(q[-1])) < 1e-6


def test_imu_track_sanitizes_and_splits_at_gaps():
    t = np.array([0, 2000, 2000, 6000, 4000, 8000, 200_000, 202_000], dtype=np.int64)
    q = np.tile([1.0, 0, 0, 0], (len(t), 1))
    q[3] *= -1.0                                       # sign flip
    acc = np.array([3, 3, 3, 3, 3, 0, 3, 3])
    tr = FU.ImuTrack.from_stream(t, q, acc, max_gap_s=0.05)
    assert np.all(np.diff(tr.t_us) > 0)
    assert tr.n_dropped == 2                           # repeated stamp + accuracy 0
    seg = tr.segment(np.array([0.001, 0.005, 0.1, 0.201, 0.3]))
    assert seg[0] == seg[1] >= 0 and seg[2] == -1 and seg[3] > seg[1] and seg[4] == -1
    assert np.isnan(tr.rot(np.array([0.1]))).all()
    assert np.allclose(tr.rot(np.array([0.003])), np.eye(3))


def test_chi2_and_quality_scale():
    assert FU.chi2_3_ppf(0.95) == pytest.approx(7.8147, abs=1e-3)
    assert FU.chi2_3_ppf(0.999) == pytest.approx(16.266, abs=1e-2)
    q = {"reprojection_rms": np.array([1.0, 2.0, 1.0, 1.0]),
         "n_tags": np.array([2, 2, 1, 2]), "n_cameras": np.array([2, 2, 2, 1])}
    s = FU.quality_scale(q, 4, {"rms": 1.0, "n_tags": 2.0, "n_cameras": 2.0})
    assert s == pytest.approx([1.0, 4.0, 2.0, 2.0])
    assert np.all(FU.quality_scale(None, 3) == 1.0)


def test_tau_cost_balances_noise_and_drift():
    taus = np.geomspace(0.01, 10, 300)

    def best(sig2, q, b):
        return taus[np.argmin([FU.tau_cost(x, 0.025, sig2, q, b) for x in taus])]

    base = best(np.radians(0.5) ** 2, 1e-5, np.radians(1.0) / 60)
    assert 0.01 < base < 10
    assert best(np.radians(1.0) ** 2, 1e-5, np.radians(1.0) / 60) > base   # noisier optics
    assert best(np.radians(0.5) ** 2, 1e-3, np.radians(1.0) / 60) < base   # noisier IMU


# -- calibration ------------------------------------------------------------------------
def test_calibration_recovers_mounting_world_lag_and_drift(scen):
    sim, cal = scen["sim"], scen["cal"]
    assert FU.rot_error_deg(cal.X, sim["X"]) < 0.3
    # Z at t = 0 (the fit centres the drift on t_ref)
    Z0 = FU.exp_so3(-cal.drift_rad_s * cal.t_ref_s) @ cal.Z
    assert FU.rot_error_deg(Z0, sim["Z"]) < 0.3
    assert abs(cal.lag_s - sim["lag_s"]) < 0.0015
    assert np.isfinite(cal.lag_se_s) and cal.lag_se_s < 0.002
    true_head = float(np.degrees(sim["drift_rad_s"] @ sim["Z"][:, 2]) * 60)
    assert cal.drift_heading_deg_min == pytest.approx(true_head, abs=0.3)
    assert cal.inlier_frac > 0.95
    assert cal.span_s >= 60


def test_calibration_sees_only_the_training_mask(scen):
    sim = scen["sim"]
    t = sim["t_opt_s"]
    R = sim["R_opt"].copy()
    ev = scen["ev"]
    R[ev] = np.einsum("nij,jk->nik", R[ev], FU.exp_so3(np.radians([30.0, -20.0, 10.0])))
    cal = FU.calibrate(t, R, scen["imu"], scen["train"])
    assert np.allclose(cal.X, scen["cal"].X) and np.allclose(cal.Z, scen["cal"].Z)
    assert cal.lag_s == scen["cal"].lag_s


def test_settings_from_sensor_characteristics(scen):
    st, sim = scen["st"], scen["sim"]
    sd = np.degrees(np.sqrt(np.diag(st.sigma_opt)))
    # true per-axis SD at the reference quality: (0.35, 0.35, 0.7)
    assert sd == pytest.approx([0.35, 0.35, 0.7], rel=0.3)
    rest = st.derivation["sigma_opt_rest_deg_axis"]
    assert rest == pytest.approx([0.35, 0.35, 0.7], rel=0.3)
    assert st.derivation["source"] == "train_structure"
    assert 0.02 <= st.tau_s <= 5.0
    assert st.gate_chi2 == pytest.approx(FU.chi2_3_ppf(0.999))
    assert st.q_theta > 0 and st.q_bias > 0 and st.bias0_sd > 0
    assert np.allclose(st.bias0, scen["cal"].drift_rad_s)
    d = st.to_dict()
    assert {"tau_s", "sigma_opt_deg_axis", "derivation", "quality_ref"} <= set(d)
    # the rest-only variant gives the same optical noise level here (white noise model)
    st_r = FU.settings_from_sensors(sim["t_opt_s"], sim["R_opt"], scen["imu"], scen["cal"],
                                    [sim["rest"]], sim["quality"])
    assert st_r.derivation["source"] == "rest"
    assert np.degrees(np.sqrt(np.diag(st_r.sigma_opt))) == pytest.approx(sd, rel=0.35)


def test_residual_structure_separates_noise_and_random_walk():
    """Optical noise plus a slow random-walk offset: the structure function returns the
    white-noise level and a positive random-walk density."""
    sim = FU.simulate(90.0, seed=5, flip_frac=0.0, dropouts=(), scale_err=0.0)
    t, R = sim["t_opt_s"], sim["R_opt"].copy()
    rng = np.random.default_rng(0)
    walk = np.cumsum(rng.normal(size=(len(t), 3)) * np.radians(0.05), axis=0)
    R = np.einsum("nij,njk->nik", FU.exp_so3(walk), R)
    imu = FU.ImuTrack.from_stream(sim["imu_t_us"], sim["imu_q"])
    train = t > 10
    cal = FU.calibrate(t, R, imu, train)
    st = FU.residual_structure(t, R, imu, cal, train, sim["quality"],
                               {"rms": 0.9, "n_tags": 2.0, "n_cameras": 2.0})
    assert np.degrees(np.sqrt(st["sigma2"])) == pytest.approx([0.35, 0.35, 0.7], rel=0.3)
    q_true = np.radians(0.05) ** 2 / (t[1] - t[0])
    assert np.mean(st["q"]) == pytest.approx(q_true, rel=0.6)


def test_calibration_with_fixed_lag(scen):
    sim = scen["sim"]
    cal = FU.calibrate(sim["t_opt_s"], sim["R_opt"], scen["imu"], scen["train"], lag_s=0.02)
    assert cal.lag_s == 0.02 and np.isnan(cal.lag_se_s)
    assert FU.rot_error_deg(cal.X, sim["X"]) < 1.0


def test_calibration_with_fixed_mounting(scen):
    """A mounting from a separate hand-eye calibration is kept; Z, lag and drift are fitted."""
    sim = scen["sim"]
    t, R = sim["t_opt_s"], sim["R_opt"]
    cal = FU.calibrate(t, R, scen["imu"], scen["train"], X_fixed=sim["X"])
    assert FU.rot_error_deg(cal.X, sim["X"]) < 1e-4
    Z0 = FU.exp_so3(-cal.drift_rad_s * cal.t_ref_s) @ cal.Z
    assert FU.rot_error_deg(Z0, sim["Z"]) < 0.3
    assert abs(cal.lag_s - sim["lag_s"]) < 0.0015
    # a wrong mounting is kept as given and shows up in the residual
    off = FU.exp_so3(np.radians([3.0, 0.0, 0.0])) @ sim["X"]
    bad = FU.calibrate(t, R, scen["imu"], scen["train"], X_fixed=off)
    assert FU.rot_error_deg(bad.X, off) < 1e-4 and bad.rms_deg > cal.rms_deg


# -- estimators -------------------------------------------------------------------------
def test_f1_f2_beat_optical_only(scen):
    sim = scen["sim"]
    tq, t = sim["t_query_s"], sim["t_opt_s"]
    Rt = sim["truth"](tq)
    e0 = FU.rot_error_deg(scen["f0"], Rt)
    e1 = FU.rot_error_deg(scen["f1"].R_query, Rt)
    e2c = FU.rot_error_deg(scen["f2"].R_query, Rt)
    e2 = FU.rot_error_deg(scen["f2"].R_query_smooth, Rt)
    # F0 without its flip-contaminated neighbourhoods, for a fair comparison
    near_flip = np.zeros(len(tq), bool)
    for k in sim["flips"]:
        near_flip |= np.abs(tq - t[k]) < 0.03
    for e in (e1, e2c, e2):
        assert _rms(e) < 0.6 * _rms(e0[~near_flip])
    assert _rms(e2) < _rms(e2c) < _rms(e1)
    assert np.nanmax(e2) < 1.0
    # at the optical frame times too
    tf = t[np.isfinite(sim["R_opt"].reshape(len(t), -1)).all(1)]
    tf = tf[(tf > 11) & (tf < 89)]
    Rf = sim["truth"](tf)
    f2 = FU.f2_kalman(t, sim["R_opt"], scen["imu"], scen["cal"], scen["st"], tf,
                      quality=sim["quality"])
    good = ~np.isin(np.searchsorted(t, tf), sim["flips"])
    e0f = FU.rot_error_deg(FU.f0_optical(t, sim["R_opt"], tf), Rf)[good]
    assert _rms(FU.rot_error_deg(f2.R_query_smooth, Rf)[good]) < 0.5 * _rms(e0f)


def test_gating_rejects_injected_flips(scen):
    sim, f2, f1 = scen["sim"], scen["f2"], scen["f1"]
    flips = sim["flips"]
    flips = flips[np.isfinite(sim["R_opt"][flips].reshape(len(flips), -1)).all(1)]
    flips = flips[flips > f2.init_frame]
    assert flips.size >= 5
    assert np.all(f2.gated[flips])
    assert np.all(f1.gated[flips])
    false = np.setdiff1d(np.flatnonzero(f2.gated), flips)
    assert false.size <= 0.005 * f2.accepted.sum()
    assert f2.diagnostics["n_gated"] == int(f2.gated.sum())
    assert f2.resets == []


def test_nis_is_consistent(scen):
    d = scen["f2"].diagnostics
    assert 1.5 < d["nis_mean"] < 4.5
    assert 0.9 < d["nis_frac95_accepted"] < 0.995


def test_causal_outputs_ignore_future_frames(scen):
    sim = scen["sim"]
    t, R = sim["t_opt_s"], sim["R_opt"].copy()
    tq = sim["t_query_s"]
    cut = 70.0
    R[t > cut] = np.einsum("nij,jk->nik", R[t > cut], FU.exp_so3(np.radians([5.0, 0, 0])))
    f2 = FU.f2_kalman(t, R, scen["imu"], scen["cal"], scen["st"], tq, quality=sim["quality"])
    f1 = FU.f1_complementary(t, R, scen["imu"], scen["cal"], scen["st"], tq,
                             quality=sim["quality"])
    before = tq < cut - 1e-6
    assert np.allclose(f2.R_query[before], scen["f2"].R_query[before], equal_nan=True)
    assert np.allclose(f1.R_query[before], scen["f1"].R_query[before], equal_nan=True)


def test_dropout_bridging_error_grows_smoothly(scen):
    sim, imu, cal, st = scen["sim"], scen["imu"], scen["cal"], scen["st"]
    t, R = sim["t_opt_s"], sim["R_opt"]
    lengths = (0.1, 0.25, 0.5, 1.0, 2.0)
    res = {}
    for L in lengths:
        drop = np.zeros(len(t), bool)
        s = 50.5
        while s + L < 79.5:
            drop |= (t >= s) & (t < s + L)
            s += L + 1.5
        k = drop & np.isfinite(R.reshape(len(t), -1)).all(1)
        Rt = sim["truth"](t[k])
        f2 = FU.f2_kalman(t, R, imu, cal, st, t[k], quality=sim["quality"], drop_mask=drop)
        f1 = FU.f1_complementary(t, R, imu, cal, st, t[k], quality=sim["quality"],
                                 drop_mask=drop)
        hold = FU.f0_optical(t, np.where(drop[:, None, None], np.nan, R), t[k], "hold")
        res[L] = {"F2": _rms(FU.rot_error_deg(f2.R_query_smooth, Rt)),
                  "F2c": _rms(FU.rot_error_deg(f2.R_query, Rt)),
                  "F1": _rms(FU.rot_error_deg(f1.R_query, Rt)),
                  "hold": _rms(FU.rot_error_deg(hold, Rt))}
    for est in ("F2", "F2c", "F1"):
        v = np.array([res[L][est] for L in lengths])
        # smooth: no jump between neighbouring gap lengths, bounded at 2 s
        assert np.all(np.abs(np.diff(v)) < 0.15), (est, v)
        assert v[-1] < 1.0, (est, v)
    v2 = [res[L]["F2"] for L in lengths]                 # the smoother: non-decreasing
    assert all(b >= a - 0.03 for a, b in zip(v2, v2[1:])), v2
    for L in lengths[1:]:
        assert res[L]["F2"] < 0.3 * res[L]["hold"]


def test_f0_modes():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    R = FU.exp_so3(np.array([[0, 0, 0], [0, 0, 0.2], [np.nan] * 3, [0, 0, 0.6]]))
    R[2] = np.nan
    tq = np.array([0.5, 2.0, 2.5, 3.5])
    s = FU.log_so3(FU.f0_optical(t, R, tq, "slerp"))
    assert s[:3, 2] == pytest.approx([0.1, 0.4, 0.5])
    assert np.isnan(s[3]).all()
    h = FU.log_so3(FU.f0_optical(t, R, tq, "hold"))
    assert h[:, 2] == pytest.approx([0.0, 0.2, 0.2, 0.6])
    g = FU.f0_optical(t, R, tq, "slerp", max_gap_s=1.5)
    assert np.isnan(g[1:3]).all() and np.isfinite(g[0]).all()


def test_calibration_of_a_nearly_static_body():
    """A head that barely rotates (< 3° within 1 s): X still comes from longer pairs."""
    sim = FU.simulate(90.0, seed=7, amp_deg=1.5, flip_frac=0.0, opt_noise_deg=(0.1, 0.1, 0.2))
    t = sim["t_opt_s"]
    imu = FU.ImuTrack.from_stream(sim["imu_t_us"], sim["imu_q"])
    cal = FU.calibrate(t, sim["R_opt"], imu, t > 10.0, lag_s=sim["lag_s"])
    assert FU.rot_error_deg(cal.X, sim["X"]) < 2.0
    assert cal.rms_deg < 0.5
