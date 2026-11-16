# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mounting-rotation composition: fit R_mount, apply it, recover the optical body."""

from __future__ import annotations

import math

import numpy as np

from jtsmm.imu.kinematics import apply_mounting_quat, quat_from_rotation_matrix
from jtsmm.imu.mounting import fit_mounting_rotation, rotation_from_quat


def _rot(axis, deg):
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    th = math.radians(deg)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def _figure_eight(n=60):
    out = []
    for i in range(n):
        u = i / (n - 1)
        out.append(
            _rot([1, 0, 0], 50 * math.sin(2 * math.pi * u))
            @ _rot([0, 1, 0], 40 * math.sin(4 * math.pi * u + 0.5))
            @ _rot([0, 0, 1], 35 * math.sin(2 * math.pi * u + 1.2))
        )
    return np.stack(out)


def _angle_deg(a, b):
    # Frobenius norm resolves tiny angles; arccos(trace) cannot below ~1e-6 deg.
    return math.degrees(np.linalg.norm(a - b) / math.sqrt(2.0))


def test_fitted_mount_applied_to_imu_quats_recovers_the_optical_body():
    """R_imu = W R_opt M  (W: IMU world vs optical world, M: body<-chip).

    Fitting M and applying it must give W R_opt exactly: the orientation of the
    array body in the IMU world frame. The old composition q(M) (x) q_chip did not.
    """
    W = _rot([0.3, -0.5, 0.8], 73.0)
    M = _rot([0.9, 0.2, -0.4], 38.0)
    R_opt = _figure_eight()
    R_imu = np.stack([W @ Ro @ M for Ro in R_opt])
    n = R_opt.shape[0]
    t = np.arange(n, dtype=np.uint64) * np.uint64(150_000)
    poses = np.zeros((n, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(R) for R in R_imu])

    fit = fit_mounting_rotation(t, poses, t, quats)
    assert _angle_deg(fit.R_mount, M) < 1e-9

    for q, Ro in zip(quats, R_opt):
        q_body = apply_mounting_quat(tuple(q), fit.R_mount)
        assert _angle_deg(rotation_from_quat(q_body), W @ Ro) < 1e-9
