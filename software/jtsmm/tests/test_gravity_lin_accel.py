# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for gravity removal from raw accelerometer."""

import numpy as np

from jtsmm.imu.gravity_lin_accel import gravity_body, linear_accel_from_raw


def test_gravity_body_identity():
  q = np.array([1.0, 0.0, 0.0, 0.0])
  g = gravity_body(q)
  assert np.allclose(g, [0, 0, 9.80665], atol=1e-4)


def test_linear_accel_static():
  q = np.array([1.0, 0.0, 0.0, 0.0])
  accel = np.array([0.0, 0.0, 9.80665])
  lin = linear_accel_from_raw(accel, q)
  assert np.allclose(lin, [0, 0, 0], atol=1e-3)


def test_gravity_body_tilted_90_deg():
  """world<-body rotation of 90 deg about x: a still sensor reads +g on body y."""
  h = np.sqrt(0.5)
  q = np.array([h, h, 0.0, 0.0])
  assert np.allclose(gravity_body(q), [0.0, 9.80665, 0.0], atol=1e-9)
  lin = linear_accel_from_raw(np.array([0.0, 9.80665, 0.0]), q)
  assert np.allclose(lin, [0.0, 0.0, 0.0], atol=1e-9)   # used to be 19.6 m/s^2


def test_gravity_body_matches_rotation_matrix_transpose():
  from jtsmm.imu.mounting import rotation_from_quat

  rng = np.random.default_rng(0)
  for _ in range(20):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    want = rotation_from_quat(q).T @ np.array([0.0, 0.0, 9.80665])
    assert np.allclose(gravity_body(q), want, atol=1e-9)
