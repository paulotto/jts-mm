# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Rigid-body helpers on (N, 4, 4) pose arrays (mm). NaN marks an invalid pose."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from jtsmm.tools.run_jts_analysis import interpolate_se3_gaps

from biomech import timebase


def valid(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T)
    return np.isfinite(T.reshape(T.shape[0], -1)).all(axis=1)


def inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    out = np.empty_like(T)
    R = T[..., :3, :3]
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, T[..., :3, 3])
    out[..., 3, :] = (0.0, 0.0, 0.0, 1.0)
    return out


def compose(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.einsum("...ij,...jk->...ik", np.asarray(A, float), np.asarray(B, float))


def apply(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Transform point(s) ``p`` (3,) or (N, 3) by pose(s) T."""
    T = np.asarray(T, dtype=np.float64)
    return np.einsum("...ij,...j->...i", T[..., :3, :3], p) + T[..., :3, 3]


def relative(T_w_ref: np.ndarray, T_w_body: np.ndarray) -> np.ndarray:
    """``T_ref<-body = inv(T_w<-ref) T_w<-body``, NaN where either is invalid."""
    out = compose(inv(T_w_ref), T_w_body)
    bad = ~(valid(T_w_ref) & valid(T_w_body))
    out[bad] = np.nan
    return out


def bridge_gaps(T: np.ndarray, max_gap: int) -> tuple[np.ndarray, np.ndarray]:
    """SE(3)-interpolate interior gaps of <= ``max_gap`` frames; returns (poses, bridged mask)."""
    ok = valid(T)
    if ok.all() or not ok.any():
        return np.asarray(T, dtype=np.float64).copy(), np.zeros(len(T), bool)
    filled, _ = interpolate_se3_gaps(T, max_gap_frames=max_gap)
    return filled, valid(filled) & ~ok


_segments = timebase.segments


def lowpass(T: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 2,
            min_len: int = 15) -> np.ndarray:
    """Zero-phase low-pass of translation and rotation, per contiguous valid segment.

    Rotation is filtered as rotation vectors relative to the segment's mean
    rotation (small angles around it, so the vector space is near-linear).
    Segments shorter than ``min_len`` are left as they are.
    """
    T = np.asarray(T, dtype=np.float64)
    out = T.copy()
    sos = butter(order, cutoff_hz / (0.5 * fs_hz), output="sos")
    for a, b in _segments(valid(T)):
        if b - a < min_len:
            continue
        seg = T[a:b]
        out[a:b, :3, 3] = sosfiltfilt(sos, seg[:, :3, 3], axis=0)
        rot = Rotation.from_matrix(seg[:, :3, :3])
        mean = rot.mean()
        rv = (mean.inv() * rot).as_rotvec()
        rv = sosfiltfilt(sos, rv, axis=0)
        out[a:b, :3, :3] = (mean * Rotation.from_rotvec(rv)).as_matrix()
    return out


def lowpass_batch(Ts: np.ndarray, fs_hz: float, cutoff_hz: float, order: int = 2,
                  min_len: int = 15) -> np.ndarray:
    """:func:`lowpass` of every pose sequence in ``Ts`` (K, N, 4, 4), in one pass.

    For K sequences with the same valid frames (e.g. Monte-Carlo noise draws on one ideal
    motion): the segments are shared, so filters and rotation conversions run once over
    all K (``sosfiltfilt`` along the frame axis and the per-element ``Rotation`` methods
    treat every sequence independently); only the segment mean rotation is taken per
    sequence. Identical to ``[lowpass(T, ...) for T in Ts]`` (test_r5_perf_vectorized);
    sequences with differing validity fall back to that loop.
    """
    Ts = np.asarray(Ts, dtype=np.float64)
    K = Ts.shape[0]
    if K == 0:
        return Ts.copy()
    ok = np.stack([valid(T) for T in Ts])
    if not (ok == ok[0]).all():
        return np.stack([lowpass(T, fs_hz, cutoff_hz, order, min_len) for T in Ts])
    out = Ts.copy()
    sos = butter(order, cutoff_hz / (0.5 * fs_hz), output="sos")
    for a, b in _segments(ok[0]):
        if b - a < min_len:
            continue
        L = b - a
        seg = Ts[:, a:b]
        out[:, a:b, :3, 3] = sosfiltfilt(sos, seg[:, :, :3, 3], axis=1)
        rot = Rotation.from_matrix(seg[:, :, :3, :3].reshape(K * L, 3, 3))
        means = Rotation.concatenate([rot[k * L:(k + 1) * L].mean() for k in range(K)])
        rep = np.repeat(np.arange(K), L)
        rv = (means.inv()[rep] * rot).as_rotvec().reshape(K, L, 3)
        rv = sosfiltfilt(sos, rv, axis=1)
        out[:, a:b, :3, :3] = (means[rep] * Rotation.from_rotvec(rv.reshape(K * L, 3))
                               ).as_matrix().reshape(K, L, 3, 3)
    return out


def rotvec_rel(T: np.ndarray, T_ref: np.ndarray) -> np.ndarray:
    """Rotation vector (rad) of ``R(t) R_ref^T``: the body's rotation since ``T_ref``,
    expressed in the reference (parent) frame."""
    R = np.asarray(T, dtype=np.float64)[..., :3, :3]
    Rr = np.asarray(T_ref, dtype=np.float64)[..., :3, :3]
    D = np.einsum("...ij,...kj->...ik", R, np.broadcast_to(Rr, R.shape))
    out = np.full(R.shape[:-2] + (3,), np.nan)
    ok = np.isfinite(D.reshape(-1, 9)).all(axis=1).reshape(D.shape[:-2])
    if ok.any():
        out[ok] = Rotation.from_matrix(D[ok]).as_rotvec()
    return out
