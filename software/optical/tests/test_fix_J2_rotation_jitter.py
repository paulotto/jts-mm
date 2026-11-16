# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Rotation jitter must see noise about every axis (runs without cv2)."""

from __future__ import annotations

import numpy as np
import pytest

from _fix_J2_imports import import_optical

geometry, metrics = import_optical("optical.calibration.geometry", "optical.evaluation.metrics")


def _noisy_rvecs(R0: np.ndarray, axis: int | None, sigma_deg: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        d = rng.normal(0.0, np.radians(sigma_deg), 3)
        if axis is not None:
            m = np.zeros(3)
            m[axis] = 1.0
            d = d * m
        R = R0 @ geometry.rodrigues(d)
        out.append(_log(R))
    return np.asarray(out)


def _log(R: np.ndarray) -> np.ndarray:
    theta = geometry.rotation_angle_rad(R)
    if theta < 1e-12:
        return np.zeros(3)
    if np.pi - theta < 1e-6:
        # Axis from the dominant column of (R + I) / 2.
        B = 0.5 * (R + np.eye(3))
        k = B[:, int(np.argmax(np.diag(B)))]
        return theta * k / np.linalg.norm(k)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return theta * w / (2.0 * np.sin(theta))


@pytest.mark.parametrize(
    "R0",
    [
        np.eye(3),
        geometry.rodrigues([np.pi, 0.0, 0.0]),  # tag facing the camera
        geometry.rodrigues([0.0, 0.0, np.pi / 6]),
    ],
    ids=["identity", "180x", "30z"],
)
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["x", "y", "z"])
def test_rotation_jitter_sees_each_axis(R0: np.ndarray, axis: int) -> None:
    sigma = 0.5
    rvecs = _noisy_rvecs(R0, axis, sigma, 2000, seed=axis)
    jit = metrics.rotation_jitter_deg(rvecs)
    # Single-axis Gaussian noise: RMS geodesic angle == sigma.
    assert jit == pytest.approx(sigma, rel=0.08)


def test_old_metric_was_blind_for_facing_tag() -> None:
    """Documents the bug: std(|rvec|) is ~0 for y/z noise on a 180° pose."""
    R0 = geometry.rodrigues([np.pi, 0.0, 0.0])
    rvecs = _noisy_rvecs(R0, 2, 0.5, 500, seed=7)
    assert float(np.std(metrics.rotation_magnitude_deg(rvecs))) < 0.05
    assert metrics.rotation_jitter_deg(rvecs) > 0.4


def test_aggregate_uses_geodesic_jitter() -> None:
    R0 = geometry.rodrigues([np.pi, 0.0, 0.0])
    rvecs = _noisy_rvecs(R0, 1, 0.5, 400, seed=3)
    samples = [
        metrics.MarkerFrameSample(
            frame_idx=i, marker_id=1, size_mm=30.0, side_px=60.0,
            tvec_mm=np.array([0.0, 0.0, 500.0]), rvec=r, reprojection_rms_px=0.2,
        )
        for i, r in enumerate(rvecs)
    ]
    tri = [
        metrics.MarkerTriSample(
            frame_idx=i, marker_id=1, size_mm=30.0, translation_mm=np.zeros(3),
            reprojection_rms_px=0.3, fit_rms_mm=0.1, rotation_rvec=r,
        )
        for i, r in enumerate(rvecs)
    ]
    res = metrics.aggregate_marker_results(
        marker_id=1, size_mm=30.0, label="30mm", total_frames=len(rvecs),
        detected_flags=[True] * len(rvecs), samples=samples, thresholds={},
        tri_detected_flags=[True] * len(rvecs), tri_samples=tri,
        tri_thresholds={"max_rotation_jitter_deg": 0.3},
    )
    assert res.rotation_jitter_deg == pytest.approx(0.5, rel=0.12)
    assert res.tri_rotation_jitter_deg == pytest.approx(0.5, rel=0.12)
    assert res.passes["tri_rotation_jitter"] is False


def test_rodrigues_and_mean_rotation_basics() -> None:
    R = geometry.rodrigues([0.1, -0.2, 0.3])
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert geometry.rotation_angle_rad(R) == pytest.approx(np.linalg.norm([0.1, -0.2, 0.3]))
    assert np.allclose(geometry.mean_rotation([R, R, R]), R)
    assert np.isnan(geometry.rotation_rms_deg([R]))
