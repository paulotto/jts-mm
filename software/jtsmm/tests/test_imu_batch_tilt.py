# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Vectorized quaternion helpers and gravity tilt."""

from __future__ import annotations

import numpy as np

from jtsmm.imu.batch import (angular_velocity, matrix_to_quat, quat_mul, quat_resample,
                             quat_to_matrix, quat_to_rotvec, rotvec_to_quat, sanitize_stream)
from jtsmm.imu.tilt import G, complementary, quasi_static, signed_angle_about


def _rz(a):
    return rotvec_to_quat(np.array([0.0, 0.0, a]))


def test_sanitize_sorts_and_dedupes():
    t, x = sanitize_stream(np.array([0, 5, 5, 3, 9]), np.array([0, 1, 2, 3, 4]))
    assert t.tolist() == [0, 3, 5, 9] and x.tolist() == [0, 3, 1, 4]


def test_rotvec_roundtrip_and_matrix():
    rng = np.random.default_rng(0)
    r = rng.normal(size=(50, 3))
    r = r / np.linalg.norm(r, axis=1, keepdims=True) * rng.uniform(0, 3.0, (50, 1))
    q = rotvec_to_quat(r)
    assert np.allclose(quat_to_rotvec(q), r, atol=1e-9)
    assert np.allclose(np.abs(matrix_to_quat(quat_to_matrix(q))), np.abs(q), atol=1e-9)
    R = quat_to_matrix(quat_mul(q[:1], q[1:2]))[0]
    assert np.allclose(R, quat_to_matrix(q[:1])[0] @ quat_to_matrix(q[1:2])[0])


def test_angular_velocity_constant_rate():
    t = np.arange(0, 1_000_000, 2_500)                   # 400 Hz
    q = rotvec_to_quat(np.outer(t * 1e-6, [0.0, 0.0, 2.0]))
    w = angular_velocity(t, q)
    assert np.allclose(w[:, 2], 2.0, atol=1e-6) and np.allclose(w[:, :2], 0.0, atol=1e-9)


def test_resample_slerp_and_gap():
    t = np.array([0, 10_000, 200_000])
    q = np.vstack([_rz(0.0), _rz(0.2), _rz(0.4)])
    out = quat_resample(t, q, np.array([5_000, 100_000]), max_gap_us=50_000)
    assert np.isclose(quat_to_rotvec(out[0])[2], 0.1) and np.isnan(out[1]).all()


def test_signed_angle_and_quasi_static():
    g = np.array([0.0, 0.0, G])
    th = 0.3
    rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    g_body = rot.T @ g                                   # body pitched by +th about y
    ang = signed_angle_about(g, g_body[None, :], np.array([0.0, 1.0, 0.0]))
    assert np.isclose(ang[0], -th)
    assert quasi_static(np.array([[0, 0, G], [0, 0, 1.3 * G]])).tolist() == [True, False]


def test_complementary_removes_drift():
    t = np.arange(0, 20, 0.01)
    truth = 0.2 * np.sin(2 * np.pi * 0.5 * t)
    fast = truth + 0.02 * t                              # drifting
    slow = truth + np.random.default_rng(0).normal(0, 0.02, t.size)
    out = complementary(t, fast, slow, np.ones(t.size, bool), tau_s=0.5)
    assert np.abs(out[-500:] - truth[-500:]).max() < 0.05
    assert np.abs(fast[-500:] - truth[-500:]).max() > 0.35
