# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""
Pure-numpy rotation / rigid-transform / projection helpers (no OpenCV).

Used by calibration (extrinsics fusion, board-spread gate, cross-view error)
and evaluation (rotation jitter). Kept free of cv2 so the maths is unit-tested
on hosts without OpenCV. Conventions match OpenCV: Rodrigues vectors, 4×4
``T_a←b`` homogeneous transforms, and the ``calibrateCamera`` distortion model.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _vee(M: np.ndarray) -> np.ndarray:
    """Axis vector of the skew part ``(M - Mᵀ)`` (not halved)."""
    return np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]], dtype=np.float64)


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation matrix from a Rodrigues vector (same result as ``cv2.Rodrigues``)."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(r))
    if theta < 1e-12:
        return np.eye(3)
    k = r / theta
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * (Kx @ Kx)


def rotation_angle_rad(R: np.ndarray) -> float:
    """Rotation angle of ``R`` in [0, π]; atan2 form stays accurate near 0 and π."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = 0.5 * (np.trace(R) - 1.0)
    s = 0.5 * float(np.linalg.norm(_vee(R)))
    return float(np.arctan2(s, c))


def geodesic_angle_rad(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle of the relative rotation ``Raᵀ Rb`` (``‖log(Raᵀ Rb)‖``)."""
    return rotation_angle_rad(np.asarray(Ra, dtype=np.float64).T @ np.asarray(Rb, dtype=np.float64))


def project_to_so3(M: np.ndarray) -> np.ndarray:
    """Closest rotation to ``M`` in the Frobenius sense (SVD, det = +1)."""
    U, _s, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64).reshape(3, 3))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)) or 1.0)])
    return U @ D @ Vt


def mean_rotation(Rs: Sequence[np.ndarray] | np.ndarray) -> np.ndarray:
    """Chordal L2 mean: project ``Σ R_i`` back onto SO(3).

    Unlike averaging Rodrigues vectors or their norms, this is independent of
    the reference orientation (fine near 180°, where rvec flips sign).
    """
    stack = np.asarray(Rs, dtype=np.float64).reshape(-1, 3, 3)
    if stack.shape[0] == 0:
        raise ValueError("mean_rotation needs at least one rotation")
    return project_to_so3(stack.sum(axis=0))


def rotation_rms_deg(Rs: Sequence[np.ndarray] | np.ndarray) -> float:
    """RMS geodesic angle (deg) of each rotation from their chordal mean.

    Sensitive to rotation noise about every axis; ``nan`` for fewer than two samples.
    """
    stack = np.asarray(Rs, dtype=np.float64).reshape(-1, 3, 3)
    if stack.shape[0] < 2:
        return float("nan")
    R_mean = mean_rotation(stack)
    ang = np.array([geodesic_angle_rad(R_mean, R) for R in stack])
    return float(np.degrees(np.sqrt(np.mean(ang**2))))


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """4×4 homogeneous transform from ``R`` (3×3) and ``t`` (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4×4 homogeneous transform."""
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)


def project_points(
    obj_points: np.ndarray,
    T_cam_obj: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray | None,
) -> np.ndarray:
    """Project object points into pixels with OpenCV's pinhole + distortion model.

    Supports the 4/5/8/12-coefficient vectors that ``cv2.calibrateCamera``
    produces (radial k1–k6, tangential p1/p2, thin prism s1–s4). Tilted-sensor
    coefficients (14-vector with non-zero τx/τy) raise ``NotImplementedError``.

    Returns:
        (N, 2) float64 pixel coordinates.
    """
    X = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T_cam_obj, dtype=np.float64)
    Xc = X @ T[:3, :3].T + T[:3, 3]
    x = Xc[:, 0] / Xc[:, 2]
    y = Xc[:, 1] / Xc[:, 2]

    d = np.zeros(14, dtype=np.float64)
    if dist is not None:
        dv = np.asarray(dist, dtype=np.float64).ravel()
        if dv.size > 14:
            raise ValueError(f"unsupported distortion vector length {dv.size}")
        d[: dv.size] = dv
    k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4, tx, ty = d
    if tx != 0.0 or ty != 0.0:
        raise NotImplementedError("tilted-sensor distortion (tauX/tauY) is not supported")

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6)
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x) + s1 * r2 + s2 * r4
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y + s3 * r2 + s4 * r4

    Km = np.asarray(K, dtype=np.float64).reshape(3, 3)
    u = Km[0, 0] * xd + Km[0, 1] * yd + Km[0, 2]
    v = Km[1, 1] * yd + Km[1, 2]
    return np.stack([u, v], axis=1)


def board_points_well_spread(
    obj_points: np.ndarray,
    *,
    min_points: int = 4,
    min_minor_rms_frac: float = 0.25,
) -> bool:
    """Reject board detections too degenerate for PnP / calibration.

    A partial ChArUco view whose corners lie on (or near) one line gives a
    PnP solution with an unobservable rotation about that line; OpenCV then
    returns garbage with ``ok=True`` (or ``calibrateCamera`` asserts). Require:

    - at least ``min_points`` corners,
    - ≥2 distinct rows **and** ≥2 distinct columns on the board grid, and
    - the RMS spread along the minor principal axis of the centred board
      points ≥ ``min_minor_rms_frac`` × grid pitch (catches diagonal lines,
      which pass the row/column count).
    """
    P = np.asarray(obj_points, dtype=np.float64).reshape(-1, 3)[:, :2]
    n = P.shape[0]
    if n < max(3, int(min_points)):
        return False
    # Board corners sit on an exact grid; 1 µm rounding merges float32 noise.
    xs = np.unique(np.round(P[:, 0], 3))
    ys = np.unique(np.round(P[:, 1], 3))
    if xs.size < 2 or ys.size < 2:
        return False
    pitch = float(min(np.min(np.diff(xs)), np.min(np.diff(ys))))
    if not np.isfinite(pitch) or pitch <= 0:
        return False
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    minor_rms = float(s[-1]) / np.sqrt(n)
    return minor_rms >= float(min_minor_rms_frac) * pitch


# Resizes whose x/y scale factors differ by more than this are a crop or an
# anamorphic stretch, not a resize of the calibrated image: refuse to scale K.
MAX_SCALE_ASPECT_MISMATCH = 0.005


def scale_K(K: np.ndarray, src_wh: Sequence[int], dst_wh: Sequence[int]) -> np.ndarray:
    """Scale a pinhole ``K`` for an image resized from ``src_wh`` to ``dst_wh``.

    Pixel *centres* map as ``x' = (x + 0.5)·s − 0.5`` (OpenCV / ISP resizers
    keep the image area aligned), so the principal point is
    ``c' = (c + 0.5)·s − 0.5``, not ``c·s`` (off by 0.25 px at half
    resolution). Focal lengths scale by ``s``.

    Raises:
        ValueError: non-positive sizes, or x/y scale factors differing by more
            than :data:`MAX_SCALE_ASPECT_MISMATCH` (0.5 %) — that is a crop or a
            different sensor mode, for which scaling ``K`` would be wrong.
    """
    src_w, src_h = int(src_wh[0]), int(src_wh[1])
    dst_w, dst_h = int(dst_wh[0]), int(dst_wh[1])
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source image size {tuple(src_wh)!r}")
    if dst_w <= 0 or dst_h <= 0:
        raise ValueError(f"invalid target image size {tuple(dst_wh)!r}")
    sx = dst_w / float(src_w)
    sy = dst_h / float(src_h)
    mismatch = abs(sx / sy - 1.0)
    if mismatch > MAX_SCALE_ASPECT_MISMATCH:
        raise ValueError(
            f"cannot scale intrinsics {src_w}x{src_h} -> {dst_w}x{dst_h}: aspect ratio "
            f"changes by {100.0 * mismatch:.2f}% (> {100.0 * MAX_SCALE_ASPECT_MISMATCH:.1f}%); "
            "this is a crop / different sensor mode — recalibrate at this size"
        )
    K2 = np.asarray(K, dtype=np.float64).reshape(3, 3).copy()
    K2[0, 0] *= sx
    K2[0, 1] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K2[0, 2] + 0.5) * sx - 0.5
    K2[1, 2] = (K2[1, 2] + 0.5) * sy - 0.5
    return K2
