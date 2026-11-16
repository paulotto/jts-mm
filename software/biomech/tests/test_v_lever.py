# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU lever arm from a hand-eye session (VD15): a simulated tumbling plate with a known
lever arm, mounting, gravity direction and accelerometer bias is recovered."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biomech.validation import lever as LV


def _session(path, r_true, R_mount, bias, noise_mm=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t_o = np.arange(0.0, 30.0, 0.025)                     # 40 Hz optics
    t_a = np.arange(0.0, 30.0, 1 / 250.0)                 # 250 Hz accelerometer
    g_w = Rotation.from_euler("xyz", [20, -10, 0], degrees=True).apply([0, 0, -9806.65])

    def pose(t):
        rv = np.column_stack([0.9 * np.sin(2 * np.pi * 0.7 * t), 0.7 * np.sin(2 * np.pi * 1.1 * t + 1),
                              0.8 * np.sin(2 * np.pi * 0.5 * t + 2)])
        p = np.column_stack([30 * np.sin(2 * np.pi * 0.4 * t), 20 * np.cos(2 * np.pi * 0.6 * t),
                             600 + 15 * np.sin(2 * np.pi * 0.3 * t)])
        return Rotation.from_rotvec(rv).as_matrix(), p

    R_o, p_o = pose(t_o)
    T = np.tile(np.eye(4), (len(t_o), 1, 1))
    T[:, :3, :3], T[:, :3, 3] = R_o, p_o + rng.normal(0, noise_mm, p_o.shape)
    # specific force at the IMU site from the exact kinematics (fine differences)
    h = 1e-3
    R_a, p_a = pose(t_a)
    def site(t):
        return pose(t)[1] + np.einsum("nij,j->ni", pose(t)[0], r_true)
    acc_w = (site(t_a + h) - 2 * site(t_a) + site(t_a - h)) / h ** 2
    f = np.einsum("ij,nkj,nk->ni", R_mount.T, R_a, acc_w - g_w) + bias
    quat = Rotation.from_matrix(np.einsum("nij,jk->nik", R_a, R_mount)).as_quat()[:, [3, 0, 1, 2]]
    with h5py.File(path, "w") as h5:
        h5["optical/t_master_us"] = (t_o * 1e6).astype(np.uint64)
        h5["optical/pose_mta"] = T.astype(np.float32)
        g = h5.create_group("imu_mta")
        g["t_master_us_accel"] = (t_a * 1e6).astype(np.uint64)
        g["accel"] = (f / 1e3).astype(np.float32)
        g["t_master_us_quat"] = (t_a * 1e6).astype(np.uint64)
        g["quat"] = quat.astype(np.float32)
        c = h5.create_group("calibration/imu_mta")
        M = np.eye(4)
        M[:3, :3] = R_mount
        c["mounting_transform"] = M
        c.attrs.update(accepted=True, rms_deg=1.0, date="2025-03-15T00:00:00Z")
        h5.create_group("phases").attrs["phases_json"] = json.dumps(
            [{"name": "imu_mount_mta", "t_start_us": int(1e6), "t_end_us": int(29e6)}])


def test_lever_arm_is_recovered(tmp_path):
    r_true = np.array([8.0, -30.0, 12.0])
    R_mount = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    bias = np.array([40.0, -300.0, 100.0])
    path = tmp_path / "session.h5"
    _session(path, r_true, R_mount, bias)
    d = LV.read_slot(path, "mta")
    assert d is not None and LV.read_slot(path, "cra") is None
    p = {**LV.DEFAULTS, "n_boot": 50}
    M, y, t = LV.design(d, 4.0, p)
    out = LV.fit(M, y, t, p, seed=0)
    r = np.array([out[f"r_{ax}_mm"] for ax in "xyz"])
    assert np.allclose(r, r_true, atol=1.5)
    assert out["g_ratio"] == pytest.approx(1.0, abs=0.01)
    assert out["bias_y_mm_s2"] == pytest.approx(-300.0, abs=20.0)
    for ax, v in zip("xyz", r_true):
        assert out[f"r_{ax}_ci_lo_mm"] - 1.5 <= v <= out[f"r_{ax}_ci_hi_mm"] + 1.5
    # the stage function: one session listed twice -> identical estimates, zero difference
    tables, _ = LV.run_ref_imu_lever({"imu_mounting_sessions": ["a/session.h5", "b/session.h5"]},
                                     tmp_path, {**p, "bodies": {"mta": "MP"}})
    assert tables["lever"].empty                          # paths a/, b/ do not exist
    (tmp_path / "a").mkdir()
    _session(tmp_path / "a" / "session.h5", r_true, R_mount, bias)
    (tmp_path / "b").mkdir()
    _session(tmp_path / "b" / "session.h5", r_true, R_mount, bias)
    tables, _ = LV.run_ref_imu_lever({"imu_mounting_sessions": ["a/session.h5", "b/session.h5"]},
                                     tmp_path, {**p, "bodies": {"mta": "MP"}})
    lev, rep = tables["lever"], tables["repeatability"]
    assert lev.primary.sum() == 2 and (lev.fc_spread_mm < 3.0).all()
    assert rep.iloc[0].diff_norm_mm == pytest.approx(0.0, abs=1e-6)
    assert rep.iloc[0].mount_change_deg == pytest.approx(0.0, abs=1e-6)
