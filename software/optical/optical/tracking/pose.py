# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
SE(3) helpers for marker pose estimation and fusion.

Used by :mod:`optical.tracking.pose_estimation` and :mod:`optical.tracking.triangulation`.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def tag_object_points(tag_size_mm: float) -> np.ndarray:
    """
    3D tag corners in the tag frame (z=0), OpenCV ``detectMarkers`` order.

    Corner layout (see ``docs/optical/01_marker_setup.md``):
    (-s/2,+s/2), (+s/2,+s/2), (+s/2,-s/2), (-s/2,-s/2).
    """
    s = float(tag_size_mm)
    half = s / 2.0
    return np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def rodrigues_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector → 3×3 rotation (same result as ``cv2.Rodrigues``)."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = r / theta
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * (Kx @ Kx)


def matrix_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """3×3 rotation → axis-angle vector (inverse of :func:`rodrigues_to_matrix`)."""
    q = rotation_to_quat(R)
    if q[0] < 0:
        q = -q
    s = float(np.linalg.norm(q[1:]))
    if s < 1e-12:
        return np.zeros(3, dtype=np.float64)
    theta = 2.0 * np.arctan2(s, q[0])
    return q[1:] / s * theta


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotations in degrees."""
    c = (np.trace(np.asarray(R_a, dtype=np.float64).T @ np.asarray(R_b, dtype=np.float64)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def T_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Build homogeneous ``T_cam←obj`` from OpenCV ``solvePnP`` output."""
    R = rodrigues_to_matrix(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4×4 homogeneous transform."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def compose_T(T_a_b: np.ndarray, T_b_c: np.ndarray) -> np.ndarray:
    """Compose transforms: ``T_a←c = T_a←b @ T_b←c``."""
    return np.asarray(T_a_b, dtype=np.float64) @ np.asarray(T_b_c, dtype=np.float64)


def rotation_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → unit quaternion (w, x, y, z)."""
    Rm = np.asarray(R, dtype=np.float64)
    tr = np.trace(Rm)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (Rm[2, 1] - Rm[1, 2]) / S
        y = (Rm[0, 2] - Rm[2, 0]) / S
        z = (Rm[1, 0] - Rm[0, 1]) / S
    elif Rm[0, 0] > Rm[1, 1] and Rm[0, 0] > Rm[2, 2]:
        S = np.sqrt(1.0 + Rm[0, 0] - Rm[1, 1] - Rm[2, 2]) * 2
        w = (Rm[2, 1] - Rm[1, 2]) / S
        x = 0.25 * S
        y = (Rm[0, 1] + Rm[1, 0]) / S
        z = (Rm[0, 2] + Rm[2, 0]) / S
    elif Rm[1, 1] > Rm[2, 2]:
        S = np.sqrt(1.0 + Rm[1, 1] - Rm[0, 0] - Rm[2, 2]) * 2
        w = (Rm[0, 2] - Rm[2, 0]) / S
        x = (Rm[0, 1] + Rm[1, 0]) / S
        y = 0.25 * S
        z = (Rm[1, 2] + Rm[2, 1]) / S
    else:
        S = np.sqrt(1.0 + Rm[2, 2] - Rm[0, 0] - Rm[1, 1]) * 2
        w = (Rm[1, 0] - Rm[0, 1]) / S
        x = (Rm[0, 2] + Rm[2, 0]) / S
        y = (Rm[1, 2] + Rm[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def quat_to_rotation(q: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) → 3×3 rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64).reshape(4)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def average_rotations_quat(Rs: np.ndarray) -> np.ndarray:
    """
    Average rotation matrices via Markley's quaternion eigenmethod.

    Args:
        Rs: Stack of shape (N, 3, 3).

    Returns:
        Single 3×3 rotation matrix.
    """
    Rs = np.asarray(Rs, dtype=np.float64)
    if Rs.shape[0] == 1:
        return Rs[0]

    A = np.zeros((4, 4), dtype=np.float64)
    for Rm in Rs:
        q = rotation_to_quat(Rm)
        if q[0] < 0:
            q = -q
        A += np.outer(q, q)
    A /= Rs.shape[0]
    _, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, -1]
    if q_avg[0] < 0:
        q_avg = -q_avg
    return quat_to_rotation(q_avg / np.linalg.norm(q_avg))


def median_translation(ts: np.ndarray) -> np.ndarray:
    """Component-wise median of translation vectors (N, 3)."""
    return np.median(np.asarray(ts, dtype=np.float64), axis=0)


def fuse_transforms(Ts: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """
    Fuse multiple rigid transforms into one estimate.

    Rotation: quaternion average. Translation: component-wise median.
    """
    if not Ts:
        return None
    Ts = [np.asarray(T, dtype=np.float64) for T in Ts]
    if len(Ts) == 1:
        return Ts[0].copy()

    Rs = np.stack([T[:3, :3] for T in Ts], axis=0)
    ts = np.stack([T[:3, 3] for T in Ts], axis=0)
    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = average_rotations_quat(Rs)
    T_out[:3, 3] = median_translation(ts)
    return T_out


def nan_pose() -> np.ndarray:
    """4×4 pose matrix filled with ``nan`` (untracked frame)."""
    return np.full((4, 4), np.nan, dtype=np.float64)


def is_valid_pose(T: np.ndarray) -> bool:
    """True if ``T`` is a finite 4×4 transform."""
    T = np.asarray(T)
    return T.shape == (4, 4) and np.all(np.isfinite(T))


def is_rigid_se3(T: np.ndarray, *, atol: float = 1e-3) -> bool:
    """True if ``T`` is a finite proper rigid transform (R in SO(3), last row [0,0,0,1])."""
    if not is_valid_pose(T):
        return False
    T = np.asarray(T, dtype=np.float64)
    if not np.allclose(T[3, :], (0.0, 0.0, 0.0, 1.0), atol=atol):
        return False
    r = T[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=atol):
        return False
    return abs(float(np.linalg.det(r)) - 1.0) <= max(atol * 10.0, 1e-2)


def pose_to_position_rotation(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 4×4 pose into translation (mm) and 3×3 rotation."""
    T = np.asarray(T, dtype=np.float64)
    return T[:3, 3].copy(), T[:3, :3].copy()


def weighted_fuse_transforms(
    Ts: Sequence[np.ndarray],
    weights: Sequence[float],
) -> Optional[np.ndarray]:
    """
    Fuse transforms with per-estimate weights (e.g. inverse reprojection error).

    Weights are normalized; rotation uses weighted quaternion average.
    """
    if not Ts:
        return None
    Ts = [np.asarray(T, dtype=np.float64) for T in Ts]
    w = np.asarray(list(weights), dtype=np.float64)
    if w.shape[0] != len(Ts):
        raise ValueError("weights length must match Ts")
    if np.all(w <= 0):
        return fuse_transforms(Ts)
    w = w / np.sum(w)

    if len(Ts) == 1:
        return Ts[0].copy()

    A = np.zeros((4, 4), dtype=np.float64)
    ts = np.zeros((len(Ts), 3), dtype=np.float64)
    for i, T in enumerate(Ts):
        q = rotation_to_quat(T[:3, :3])
        if q[0] < 0:
            q = -q
        A += w[i] * np.outer(q, q)
        ts[i] = T[:3, 3]

    _, eigvecs = np.linalg.eigh(A)
    q_avg = eigvecs[:, -1]
    if q_avg[0] < 0:
        q_avg = -q_avg

    T_out = np.eye(4, dtype=np.float64)
    T_out[:3, :3] = quat_to_rotation(q_avg / np.linalg.norm(q_avg))
    T_out[:3, 3] = np.average(ts, axis=0, weights=w)
    return T_out
