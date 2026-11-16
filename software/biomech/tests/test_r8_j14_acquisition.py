# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J14: the new estimators on synthetic data with known truth.

X11 acquisition quality (drops, time-base replay, IMU lag), the stereo-calibration checks,
VD26 fault injection, VD27 pivot fit, VD28 ladder scale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from biomech.qc import acquisition as A
from biomech.qc import stereo_cal as S


# -- X11: drops ------------------------------------------------------------------------------------
def _cam(ids, period_us=25_000, t0=1_000_000):
    ids = np.asarray(ids, np.int64)
    t = t0 + (ids - 1) * period_us
    return A.CameraStream("cam0", ids, (t * 1000 + 7).astype(np.int64), t.astype(np.int64))


def test_drops_startup_gap_and_bursts():
    trig = np.arange(1, 2001)
    tt = 1_000_000 + (trig - 1) * 25_000
    ids = [1, 2] + list(range(22, 2001))           # start-up gap (19 ids) within 1 s
    ids = [i for i in ids if i not in (100, 500, 501, 900, 901, 902)]  # bursts 1, 2, 3
    r = A.camera_drops(_cam(ids), trig, tt, startup_s=1.0)
    assert r["startup_triggers"] == 19 and r["pre_first_triggers"] == 0
    assert r["dropped"] == 6 and r["n_bursts"] == 3 and r["burst_max"] == 3
    assert r["regular_triggers"] == 2000 - 22 + 1
    assert r["drop_rate_pct"] == pytest.approx(100 * 6 / 1979)


# -- X11: the time-base replay equals the live fit -------------------------------------------------
def test_timebase_residuals_match_clockaligner_fit():
    from jtsmm.sync.clock_align import AffineFit, ClockAligner

    rng = np.random.default_rng(1)
    n = 300
    ids = np.arange(1, n + 1)
    t_master = 5_000_000 + (ids - 1) * 25_000.0
    sensor_us = 1.7e9 + (t_master - 12_500) * (1 + 40e-6) + rng.normal(0, 3, n)
    cam = A.CameraStream("cam0", ids, np.round(sensor_us * 1000).astype(np.int64),
                         t_master.astype(np.int64))
    r = A.timebase_residuals(cam, 30)
    al = ClockAligner()
    fit = AffineFit()
    live = []
    for x, y in zip(cam.sensor_ns * 1e-3, t_master):
        al._update_fit(fit, float(x), float(y))
        live.append(y - (fit.a * x + fit.b))
    live = np.asarray(live)
    ok = np.isfinite(r)
    assert ok[2:].all()
    np.testing.assert_allclose(r[ok][5:], live[ok][5:], atol=1e-3)   # µs


# -- X11: IMU lag recovers a known delay despite glitches -------------------------------------------
def test_imu_camera_lag_recovers_known_delay():
    rng = np.random.default_rng(2)
    true_ms = 7.3
    t_fine = np.arange(0, 60e6, 1e6 / 390)                           # IMU 390 Hz, µs
    ang = lambda t: 20 * np.sin(2 * np.pi * 0.8 * t * 1e-6) * (1 + 0.5 * np.sin(2 * np.pi * 0.13 * t * 1e-6))  # noqa: E731
    axis = np.array([0.2, 1.0, 0.1]) / np.linalg.norm([0.2, 1.0, 0.1])
    R_imu = Rotation.from_rotvec(np.radians(ang(t_fine - true_ms * 1e3))[:, None] * axis)
    q = R_imu.as_quat()
    q_wxyz = np.c_[q[:, 3], q[:, :3]]
    q_wxyz[::997] = [0.0, 1.0, 0.0, 0.0]                              # isolated jumps
    t_trig = np.arange(0, 60e6, 25_000.0)                             # optics 40 Hz
    t_exp = t_trig + 6_000                                            # exposure centre
    R_opt = Rotation.from_rotvec(np.radians(ang(t_exp))[:, None] * axis).as_matrix()
    R_opt = R_opt @ Rotation.from_rotvec(rng.normal(0, np.radians(0.05), (len(t_trig), 3))).as_matrix()
    p = {**A.DEFAULTS["lag"]}
    out, tasks = A.imu_camera_lag(t_trig, R_opt, t_fine, q_wxyz,
                                  [(1_000_000, 29_000_000, "a"), (31_000_000, 59_000_000, "b")],
                                  p, 0.012)
    assert out["lag_ms"] == pytest.approx(true_ms, abs=0.6)
    assert out["r_peak"] > 0.9 and out["n_glitch_imu"] > 0
    assert len(tasks) == 2


# -- stereo calibration ----------------------------------------------------------------------------
def _toy_cal():
    K = np.array([[900.0, 0, 364], [0, 900.0, 272], [0, 0, 1]])
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("y", -30, degrees=True).as_matrix()
    T[:3, 3] = [430.0, 0, 120.0]
    return S.StereoCal(K, np.zeros(5), K.copy(), np.zeros(5), T)


def test_epipolar_zero_and_reprojection_on_exact_points():
    import cv2

    cal = _toy_cal()
    X = np.random.default_rng(3).uniform([-80, -60, 450], [80, 60, 800], (200, 3))
    p0, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), cal.K0, cal.d0)
    rv, _ = cv2.Rodrigues(cal.R)
    p1, _ = cv2.projectPoints(X, rv, cal.t, cal.K1, cal.d1)
    n0 = S._undistort(p0.reshape(-1, 2), cal.K0, cal.d0)
    n1 = S._undistort(p1.reshape(-1, 2), cal.K1, cal.d1)
    assert np.max(S.epipolar_px(cal, n0, n1)) < 1e-6
    np.testing.assert_allclose(S.triangulate(cal, n0, n1), X, atol=1e-6)
    # a 1 deg pitch of cam1 moves the points off their epipolar lines (a yaw about the
    # vertical axis would mostly slide them along the near-horizontal lines)
    T = cal.T_c0_c1.copy()
    T[:3, :3] = Rotation.from_euler("x", 1, degrees=True).as_matrix() @ T[:3, :3]
    assert np.median(S.epipolar_px(cal.with_pose(T), n0, n1)) > 5.0


def test_batch_kabsch_recovers_transform():
    rng = np.random.default_rng(4)
    P = rng.normal(0, 30, (8, 3))
    R = Rotation.from_rotvec([0.3, -0.2, 0.5]).as_matrix()
    t = np.array([10.0, -5.0, 600.0])
    Q = (R @ P.T).T + t
    W = np.ones((2, 8))
    W[1, :3] = 0                                   # missing points in the second frame
    Qs = np.stack([Q, np.where(W[1][:, None] > 0, Q, np.nan)])
    Rh, th = S._batch_kabsch(P, Qs, W)
    np.testing.assert_allclose(Rh[0], R, atol=1e-9)
    np.testing.assert_allclose(Rh[1], R, atol=1e-9)
    np.testing.assert_allclose(th[1], t, atol=1e-7)


def test_sigma_model_standard_error():
    sig = S.sigma_model({"n": 49, "rms_rot_deg": 0.7, "rms_trans_mm": 7.0})
    assert sig["expected"]["sigma_trans_mm"] == pytest.approx(7.0 / np.sqrt(3) / 7)
    assert sig["snapshot"]["sigma_rot_deg"] == pytest.approx(0.7 / np.sqrt(3))


# -- VD26: fault injection --------------------------------------------------------------------------
def _recording(n=1500, drop_every=97):
    ids_t = np.arange(1, n + 1)
    tt = 2_000_000 + (ids_t - 1) * 25_000
    cams = {}
    for c, off in (("cam0", 0), ("cam1", 21)):
        ids = ids_t[off:]
        ids = ids[(ids % drop_every) != 0]
        t = 2_000_000 + (ids - 1) * 25_000
        sens = (1.9e12 + (t - 12_500) * 1000 * (1 + 30e-6)).astype(np.int64)
        cams[c] = A.CameraStream(c, ids.astype(np.int64), sens, t.astype(np.int64))
    return cams, ids_t.astype(np.int64), tt.astype(np.int64)


def test_fault_replay_clean_and_burst_limit():
    from biomech.validation import assoc_faults as F

    cams, tid, tt = _recording()
    p = {**F.DEFAULTS}
    clean = F.replay(cams, tid, tt, "clean", 0, 0, p)
    assert clean["cam0_wrong"] == 0 and clean["cam0_unassigned"] == 0
    assert clean["cam1_correct"] == clean["cam1_n"]
    small = F.replay(cams, tid, tt, "burst", 10, 0, p)
    assert small["cam0_wrong"] == 0 and small["cam1_wrong"] == 0
    loss = F.replay(cams, tid, tt, "trigger_loss", 5, 0, p)
    assert loss["cam0_wrong"] == 0 and loss["cam0_unassigned"] > 0


def test_fault_summary_tolerance_limit():
    from biomech.validation import assoc_faults as F

    runs = pd.DataFrame([
        {"fault": "burst", "level": 10, "seed": 0, "wrong": 0, "wrong_pct": 0.0,
         "unassigned_pct": 0.0},
        {"fault": "burst", "level": 20, "seed": 0, "wrong": 0, "wrong_pct": 0.0,
         "unassigned_pct": 0.0},
        {"fault": "burst", "level": 30, "seed": 0, "wrong": 50, "wrong_pct": 60.0,
         "unassigned_pct": 1.0},
        {"fault": "clean", "level": 0, "seed": 0, "wrong": 0, "wrong_pct": 0.0,
         "unassigned_pct": 0.0}])
    lev, lim = F.summarise(runs)
    r = lim.set_index("fault").loc["burst"]
    assert r["tolerated_level"] == 20 and r["first_failing_level"] == 30


# -- VD27: pivot fit ----------------------------------------------------------------------------------
def test_pivot_fit_recovers_tip():
    from biomech.validation.calrepeat import pivot_fit

    rng = np.random.default_rng(5)
    d = np.array([0.5, -230.0, -5.0])
    c = np.array([-20.0, 230.0, 510.0])
    T = []
    for _ in range(20):
        R = Rotation.from_rotvec(rng.normal(0, 0.4, 3)).as_matrix()
        M = np.eye(4)
        M[:3, :3] = R
        M[:3, 3] = c - R @ d
        T.append(M)
    f = pivot_fit(np.stack(T))
    np.testing.assert_allclose(f["tip"], d, atol=1e-8)
    assert f["rms_mm"] < 1e-8


# -- VD28: ladder scale --------------------------------------------------------------------------------
def test_ladder_scale_error_and_relative_scale():
    from biomech.validation.trueness import SlotSamples, ladder_metrics, scale_table

    rng = np.random.default_rng(6)
    slots = []
    for sysm, k in (("jtsmm", 0.993), ("qualisys", 1.0)):
        for i, mm in enumerate(range(0, 201, 20)):
            p = np.array([mm * k, 0.0, 600.0]) + rng.normal(0, 0.01, (40, 3))
            slots.append(SlotSamples(sysm, "lateral", "near", f"d{mm:03d}", float(mm),
                                     np.arange(40) / 40.0, p))
    _, tr, _ = ladder_metrics(slots)
    sc = scale_table(tr).set_index("system")
    assert sc.loc["jtsmm", "scale_error_pct"] == pytest.approx(-0.7, abs=0.02)
    assert sc.loc["jtsmm", "scale_vs_ref_pct"] == pytest.approx(-0.7, abs=0.03)
    assert sc.loc["jtsmm", "line_resid_scale_share_pct"] > 95


def test_bundle_keeps_a_table_without_columns_as_parquet():
    """An empty notes table (no columns) is stored as Parquet bytes, so the bundle
    verification sees the same column index (the column encoding lost its type)."""
    from biomech.results.bundle import _encode_table

    assert _encode_table(pd.DataFrame()) is None
    assert _encode_table(pd.DataFrame({"note": ["x"]})) is not None


def test_p95_upper_bound_needs_59_participants():
    """E31: a one-sided upper 95 % bound of a P95 first exists at
    n = ceil(ln 0.05 / ln 0.95) = 59 (the sample maximum), and at n = 13 the maximum lies
    above the P95 only with probability 1 - 0.95^13."""
    from types import SimpleNamespace

    from biomech.viz.derived_dataset import p95_upper_bound

    req = pd.DataFrame({"key": ["opening_mm"], "n": [13]})
    ctx = SimpleNamespace(table=lambda stage, name: req)
    r = p95_upper_bound(ctx).iloc[0]
    assert r["n_min_upper"] == 59 == int(np.ceil(np.log(0.05) / np.log(0.95)))
    assert r["max_above_p95_conf"] == pytest.approx(1 - 0.95 ** 13)
    assert round(r["max_above_p95_conf"], 2) == 0.49


def test_cf_free_motion_scores_only_free_motion_rows():
    """X5d (J15): the cohort median is over the free-motion rows only, E3 is the
    primary E3 of cf_summary, E4gen comes from the cohort table, and the in-sample
    estimators carry role in_sample."""
    from types import SimpleNamespace

    from biomech.viz.derived_dataset import cf_free_motion

    pids = [f"s{i:02d}" for i in range(1, 8)]
    rot = pd.DataFrame([{"pid": p, "estimator": e, "task": t,
                         "rmse_open_deg": (1.0 if t == "free_motion" else 9.0) + i / 10}
                        for i, p in enumerate(pids) for e in ("E1", "E2a", "E2b", "E3b")
                        for t in ("free_motion", "open_close")])
    tr = pd.DataFrame([{"pid": p, "estimator": e, "task": "free_motion", "rmse_inc_mm": 5.0}
                       for p in pids for e in ("E4hinge", "E4own", "E5", "E6")])
    gen = pd.DataFrame([{"pid": p, "estimator": "E4gen", "task": "free_motion",
                         "rmse_inc_mm": 6.0} for p in pids])
    summ = pd.DataFrame({"pid": pids, "primary_e3": "E3b"})
    tabs = {("camerafree", "cf_summary"): summ, ("camerafree", "cf_rotation"): rot,
            ("camerafree", "cf_translation"): tr, ("camerafree_cohort", "cf_generic"): gen}
    ctx = SimpleNamespace(table=lambda s, n: tabs[(s, n)], config=SimpleNamespace(seed=1))
    out = cf_free_motion(ctx).set_index("estimator")
    assert out.loc["E3", "source_estimator"] == "E3b"
    assert out.loc["E2a", "median"] == pytest.approx(1.3) and out.loc["E2a", "n"] == 7
    assert out.loc["E4gen", "median"] == pytest.approx(6.0)
    assert set(out.index[out.role == "in_sample"]) == {"E1", "E4own", "E6"}


def test_cf_anchor_offsets_per_task_participant_medians():
    """J16: absolute opening offset, primary E3 rows only, participant medians per task."""
    from types import SimpleNamespace

    from biomech.viz.derived_dataset import cf_anchor_offsets

    pids = [f"s{i:02d}" for i in range(1, 7)]
    rows = []
    for i, p in enumerate(pids):
        for k, sign in enumerate((-1, 1, 1)):
            rows.append({"pid": p, "estimator": "E3b", "task": "open_close",
                         "opt_open_deg": sign * (0.5 + i / 10), "opt_inc_mm": 1.0 + k})
        rows.append({"pid": p, "estimator": "E3a", "task": "open_close",
                     "opt_open_deg": 99.0, "opt_inc_mm": 99.0})
    tabs = {("camerafree", "cf_summary"): pd.DataFrame({"pid": pids, "primary_e3": "E3b"}),
            ("camerafree", "cf_anchors"): pd.DataFrame(rows)}
    ctx = SimpleNamespace(table=lambda s, n: tabs[(s, n)], config=SimpleNamespace(seed=1))
    out = cf_anchor_offsets(ctx).set_index(["task", "outcome"])
    assert out.loc[("open_close", "offset_open_deg"), "median"] == pytest.approx(0.75)
    assert out.loc[("open_close", "offset_inc_mm"), "median"] == pytest.approx(2.0)
    assert out.loc[("open_close", "offset_inc_mm"), "n_anchors"] == 18
    assert out.loc[("chewing", "offset_inc_mm"), "n"] == 0


def test_c2_null_period_desc_reads_band_zero_medians(tmp_path):
    """J17: the descriptives are cohort medians of the band-0 rows of the audit file."""
    from types import SimpleNamespace

    from biomech.viz.derived_dataset import c2_null_period_desc

    (tmp_path / "audit").mkdir()
    pd.DataFrame({"pid": ["a", "b", "c", "a"], "task": ["chewing"] * 4,
                  "band": [0.0, 0.0, 0.0, 0.1], "junction_share": [0.05, 0.07, 0.09, 0.5],
                  "near_share": [0.2, 0.2, 0.2, 0.0], "interval_sd_rel": [0.1, 0.2, 0.3, 9],
                  "period_s": [1.0, 1.1, 1.2, 9]}
                 ).to_parquet(tmp_path / "audit" / "c2_null_period_r1_desc.parquet")
    ctx = SimpleNamespace(config=SimpleNamespace(out_dir=tmp_path, snapshot="r1"))
    out = c2_null_period_desc(ctx).set_index("metric")
    assert out.loc["junction_share", "median"] == pytest.approx(0.07)
    assert out.loc["interval_sd_rel", "n"] == 3
