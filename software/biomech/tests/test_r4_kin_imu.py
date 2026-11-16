# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""R4 kinematics / IMU / uncertainty: E19 null filtered like the data, A3 empty guards,
E23 IMU lag / tilt agreement / optical gap, E24 condyle NaN check and common noise window."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biomech.imu import analysis as ia
from biomech.kinematics import se3
from biomech.kinematics.clinical import noise_floor, theta_deg
from biomech.kinematics.coupling import HINGE_POINTS, Opening
from biomech.kinematics.hinge_fit import fit_hinge
from biomech.uncertainty import montecarlo as mc

PTS = {"incisor": np.array([0.0, 0, -3]), "left": np.array([-37.0, 23, -1]),
       "right": np.array([-37.0, -23, -1]), "tmj_left": np.array([-88.0, 52, 18]),
       "tmj_right": np.array([-88.0, -52, 18])}
N_AX, S_AX = np.array([0.0, 1.0, 0.0]), np.array([-88.0, 0.0, 18.0])


def _continuous_open_close(fs=40.0, dur=0.3, total=12.0):
    per = 2 * dur + 0.4
    t = np.arange(0, total, 1 / fs)
    ph = t % per
    th = np.where(ph < dur, 0.5 * (1 - np.cos(np.pi * ph / dur)),
                  np.where(ph < 2 * dur, 0.5 * (1 + np.cos(np.pi * (ph - dur) / dur)), 0.0))
    T = mc._hinge_pose(N_AX, S_AX, th * np.radians(25.0))
    starts = [int(round(k * per * fs)) for k in range(1, int(total / per) - 1)]
    return T, starts, [a + int(round(dur * fs)) for a in starts]


def test_e19_null_matches_data_filtered_as_one_segment():
    """Data: a continuous open-close sequence with pose noise, low-passed as one segment and
    cut into openings (13 frames each, below se3.lowpass min_len). The H0 null must
    reproduce the residual of such data; filtering each bare opening (pad 0, the former
    method) overstated it."""
    fs, sd_t, sd_r = 40.0, 0.1, 0.07
    Tfull, starts, peaks = _continuous_open_close(fs)
    P = np.stack([PTS[r] for r in HINGE_POINTS])
    rng = np.random.default_rng(0)
    jm = SimpleNamespace(fs=fs, points_mp=PTS)

    def draw():
        Tn = se3.lowpass(mc._noisy(Tfull, sd_t, sd_r, rng), fs, 6.0, 2)
        ops = [Opening(k, Tn[a:b + 1], theta_deg(Tn[a:b + 1], Tn[a])[0])
               for k, (a, b) in enumerate(zip(starts, peaks))]
        A, B = mc._stack(P, [o.T for o in ops], 30)
        return fit_hinge(A, B, s0=S_AX, n0=N_AX).rms(0), ops

    data = np.median([draw()[0] for _ in range(20)])
    ops = draw()[1]
    new = np.median(mc.null_distribution(jm, ops, N_AX, S_AX, sd_t, sd_r, 20, 6.0, 30, 1))
    old = np.median(mc.null_distribution(jm, ops, N_AX, S_AX, sd_t, sd_r, 20, 6.0, 30, 1,
                                         pad_s=0.0))
    assert abs(new - data) < 0.15 * data
    assert abs(old - data) > 2 * abs(new - data)


def test_e19_padding_is_even_reflection():
    th = np.array([0.0, 1.0, 3.0, 4.0])
    np.testing.assert_array_equal(mc._padded_theta(th, 2), [1.0, 0.0, 0.0, 1.0, 3.0, 4.0, 4.0, 3.0])
    assert mc._padded_theta(th, 0).tolist() == th.tolist()


def test_a3_montecarlo_empty_inputs():
    jm = SimpleNamespace(fs=40.0, points_mp=PTS, landmarks=SimpleNamespace(items={}),
                         lower=SimpleNamespace(incisor="i", left="l", right="r"))
    null = mc.null_distribution(jm, [], N_AX, S_AX, 0.1, 0.1, 5, 6.0, 30, 0)
    assert null.shape == (5,) and np.isnan(null).all()
    T = mc._hinge_pose(N_AX, S_AX, np.radians(np.linspace(0, 10, 4)))
    short = [Opening(0, T, np.linspace(0, 10, 4))]                    # 4 frames < MIN_FRAMES
    assert np.isnan(mc.null_distribution(jm, short, N_AX, S_AX, 0.1, 0.1, 3, 6.0, 30, 0)).all()
    ci = mc.observed_ci(jm, [], 5, 2.0, 5.0, 30, 0)
    assert all(v.size == 0 for v in ci.values())
    row = mc.summarize({"hinge_fit": 0.5}, {"bench": null}, ci)
    assert np.isnan(row["null_bench_p99_mm"]) and row["above_bench"] is False


def test_e23_lag_ignores_gap_edges():
    fs = 200.0
    t = (np.arange(0, 20, 1 / fs) * 1e6).astype(np.int64)
    ts = t / 1e6
    ref = 15.0 + 10.0 * np.sin(2 * np.pi * 0.7 * ts) + 3.0 * np.sin(2 * np.pi * 1.9 * ts)
    x = 15.0 + 10.0 * np.sin(2 * np.pi * 0.7 * (ts - 0.02)) + 3.0 * np.sin(2 * np.pi * 1.9 * (ts - 0.02))
    for a in range(300, 3800, 400):                                   # optical gaps
        ref[a:a + 40] = np.nan
    assert ia._lag_ms(t, ref, x) == pytest.approx(20.0, abs=5.0 + 1e-9)


def test_e23_tilt_agreement_uses_quasi_static_samples():
    n = 1000
    opt = np.zeros(n)
    qs = np.zeros(n, np.uint8)
    qs[:500] = 1
    tilt = np.where(qs.astype(bool), 1.0, 30.0)                       # wrong while moving
    sig = {"theta_opt": opt, "quasi_static": qs, "theta_quat": opt + 2.0,
           "theta_tilt": tilt, "theta_comp": opt}
    a = ia.agreement(sig)
    assert a["rmse_tilt_deg"] == pytest.approx(1.0) and a["bias_tilt_deg"] == pytest.approx(1.0)
    assert a["rmse_quat_deg"] == pytest.approx(2.0)                   # others: every sample


def test_e23_optical_gap_from_frame_interval():
    assert ia._optical_gap_us(np.arange(0, 1e6, 25_000)) == pytest.approx(62_500)
    assert ia._optical_gap_us(np.arange(0, 1e6, 10_000)) == pytest.approx(25_000)


def test_e24_condyle_level_check_nan_is_order_independent():
    from biomech.frames.landmarks import CondyleCheck
    from biomech.kinematics.pose_chain import condyle_level_check

    ok = CondyleCheck(80.0, 80.0, 100.0, True)
    for fc in ({"tmj_left_z": np.nan, "tmj_right_z": 12.0},
               {"tmj_left_z": 12.0, "tmj_right_z": np.nan}, {}):
        r = condyle_level_check(ok, fc, 5.0)
        assert not r.plausible and r.reason == "condyle_level_unknown"
    assert condyle_level_check(ok, {"tmj_left_z": 12.0, "tmj_right_z": 3.0}, 5.0).reason == \
        "below_incisal_level"
    assert condyle_level_check(ok, {"tmj_left_z": 12.0, "tmj_right_z": 9.0}, 5.0).plausible


def test_e24_noise_floor_theta_from_the_incisor_quietest_window():
    """Window A: rotation noise about a y axis through the incisor (incisor still, theta
    noisy); window B: translation noise (incisor moves, theta constant). All floors come
    from A, the incisor-quietest window; the theta floor was the minimum over windows (B)."""
    fs, n = 40.0, 120
    rng = np.random.default_rng(0)
    p = PTS["incisor"]
    T = np.tile(np.eye(4), (n, 1, 1))
    ang = np.radians(rng.normal(0, 0.2, 60))
    R = Rotation.from_rotvec(np.outer(ang, [0.0, 1.0, 0.0])).as_matrix()
    T[:60, :3, :3] = R
    T[:60, :3, 3] = p - np.einsum("nij,j->ni", R, p)
    T[60:, :3, 3] = rng.normal(0, 0.5, (60, 3))
    t_us = (np.arange(n) / fs * 1e6).astype(np.int64)
    jm = SimpleNamespace(t_us=t_us, fs=fs, T=T, T_raw=T, points_mp=PTS,
                         window=lambda a, b: (t_us >= a) & (t_us <= b))
    out = noise_floor(jm, 0, int(t_us[-1]), window_s=1.0)
    assert out["floor_incisor_rms_mm_raw"] < 1e-9
    th = theta_deg(T[:40], T[0])[0]
    assert out["floor_theta_sd_deg_raw"] == pytest.approx(np.std(th), rel=1e-9)
    assert out["floor_theta_sd_deg_raw"] > 0.05


def test_e28_rest_residuals_keep_correlated_noise_and_drop_drift():
    from scipy.signal import lfilter
    from scipy.spatial.transform import Rotation as R

    from biomech.kinematics.clinical import pose_noise_residual, rest_residuals
    from biomech.uncertainty.montecarlo import ResidualNoise

    rng = np.random.default_rng(3)
    n, fs, rho = 4000, 40.0, 0.8

    def ar(sd):                                     # AR(1) noise of total SD ``sd``
        return lfilter([1.0], [1.0, -rho], rng.normal(0, sd * np.sqrt(1 - rho ** 2), (n, 3)),
                       axis=0)

    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R.from_rotvec(np.radians(ar(0.1))).as_matrix()          # 0.1 deg SD
    drift = np.linspace(0, 3.0, n)[:, None] * np.array([1.0, 0.0, 0.0])    # slow 3 mm
    T[:, :3, 3] = drift + ar(0.05)                                          # 0.05 mm SD
    rot, tr = pose_noise_residual(T, fs)
    # The 1-s running median also takes some low-frequency power of AR(1) noise with
    # rho = 0.8 (~10-17 % of the SD); first differences would read only ~45 %.
    np.testing.assert_allclose(rot, 0.1, rtol=0.2)
    np.testing.assert_allclose(tr, 0.05, rtol=0.2)                          # drift removed
    runs = rest_residuals(T, fs)
    e = ResidualNoise(runs, 41).draw(3000, rng)
    assert e.shape == (3000, 6)
    r1 = np.corrcoef(e[1:, 3], e[:-1, 3])[0, 1]
    assert r1 > 0.6                                                         # correlation kept
