# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical + accelerometer position fusion (validation plan VD6; reported as a secondary result, VD8).

The orientation fusion (:mod:`jtsmm.imu.fusion`) leaves the position optical and
interpolated, so an optical dropout of a moving plate is bridged by a straight line. Here
the BNO085 accelerometer (raw ``accelerometer`` report, 250 Hz, chip frame, m/s², **gravity
included**) carries the position through the gap:

1. **Gravity removal.** ``a_imuworld = R_imu(t) f(t) − g`` with ``R_imu`` the chip's own
   game-rotation-vector orientation (world z up, ``g = (0, 0, 9.80665)``): the heading drift
   of the game rotation vector does not matter for gravity. An orientation (tilt) error of
   δ leaks ``g sin δ`` (≈ 171 mm/s² per degree); its slowly varying part is absorbed by the
   bias state, the rest by the smoother's two-sided constraint.
2. **Into the optical world** with the rotation calibration of :func:`fusion.calibrate`:
   ``a = Exp(w (t − t_ref)) Z a_imuworld``, at the optical-clock time ``t_imu − lag``,
   in mm/s².
3. **Kalman filter** per axis on ``[p, v, b]`` (position, velocity, world-frame
   accelerometer bias as a random walk), driven by ``a − b`` and corrected by the optical
   positions; Rauch–Tung–Striebel smoother (both gap ends constrain the bridge) or
   forward only (causal).

The accelerometer sits a few cm from the optical body origin; the lever-arm terms
(``α×r + ω×(ω×r)``) are not modelled (stated as a limitation). The settings come from the
JTS-MM data only: the optical position noise from the rest phase, the acceleration noise
from the training-frame residual between the optical and the IMU acceleration; no
reference data, no tuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from jtsmm.imu.fusion import Calibration, ImuTrack, exp_so3

G_MPS2 = np.array([0.0, 0.0, 9.80665])


@dataclass
class PositionSettings:
    """Kalman settings (mm, s): optical position noise, white acceleration noise density,
    bias random-walk density, and how they were derived."""

    r_pos_mm: float
    q_acc: float                 # (mm/s²)² · s, per axis
    q_bias: float                # (mm/s³)² · s, per axis
    derivation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def world_acceleration(cal: Calibration, imu: ImuTrack, t_acc_s: np.ndarray, f_mps2: np.ndarray,
                       max_gap_s: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Gravity-free acceleration in the optical world (mm/s²) at the optical-clock times
    of the accelerometer samples. Samples outside the orientation stream (or in its gaps)
    are dropped."""
    ta = np.asarray(t_acc_s, dtype=np.float64)
    f = np.asarray(f_mps2, dtype=np.float64)
    ok = np.isfinite(ta) & np.isfinite(f).all(axis=1)
    ta, f = ta[ok], f[ok]
    Ri = imu.rot(ta, 0.0)                                   # the IMU's own clock
    a_iw = np.einsum("nij,nj->ni", Ri, f) - G_MPS2
    t_opt = ta - cal.lag_s                                  # R_opt(t) ~ R_imu(t + lag)
    E = exp_so3(np.outer(t_opt - cal.t_ref_s, cal.drift_rad_s))
    a = np.einsum("nij,jk,nk->ni", E, cal.Z, a_iw) * 1000.0
    keep = np.isfinite(a).all(axis=1)
    return t_opt[keep], a[keep]


def _frame_acceleration(t: np.ndarray, p: np.ndarray, half_win: int = 3) -> np.ndarray:
    """Second derivative of the optical position at the frames (local quadratic fit over
    ±``half_win`` frames inside runs of valid frames); NaN elsewhere."""
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p).all(axis=1)
    for i in range(half_win, len(t) - half_win):
        sl = slice(i - half_win, i + half_win + 1)
        if not ok[sl].all():
            continue
        tt = t[sl] - t[i]
        if np.ptp(tt) <= 0:
            continue
        A = np.column_stack([np.ones_like(tt), tt, 0.5 * tt ** 2])
        out[i] = np.linalg.lstsq(A, p[sl], rcond=None)[0][2]
    return out


def settings_from_data(t: np.ndarray, p: np.ndarray, train_mask: np.ndarray,
                       rest_intervals: Sequence[tuple[float, float]], t_acc: np.ndarray,
                       a_acc: np.ndarray, *, bias_rw: float = 20.0) -> PositionSettings:
    """Settings from the JTS-MM data only.

    * ``r_pos``: the SD of the optical position about its median over the rest phase(s)
      (RMS over axes), at least 0.05 mm.
    * ``q_acc``: the robust variance of (IMU − optical) acceleration on the training frames
      (the IMU acceleration averaged over each frame interval), times the frame interval:
      the white-noise density that reproduces that disagreement at the frame rate.
    * ``q_bias``: ``bias_rw²`` (mm/s³)² · s, a slow random walk (fixed, physical scale)."""
    t = np.asarray(t, dtype=np.float64)
    ok = np.isfinite(p).all(axis=1)
    rest = np.zeros(len(t), bool)
    for a0, a1 in rest_intervals:
        rest |= (t >= a0) & (t <= a1)
    rs = rest & ok
    r_pos = float(np.sqrt(np.mean(np.var(p[rs], axis=0)))) if rs.sum() > 10 else 0.3
    r_pos = max(r_pos, 0.05)
    dt = float(np.median(np.diff(t)))
    a_opt = _frame_acceleration(t, p)
    a_imu = np.full_like(a_opt, np.nan)
    if len(t_acc) > 2:                                      # mean over each frame interval
        cs = np.vstack([np.zeros((1, 3)), np.cumsum(a_acc, axis=0)])
        lo = np.searchsorted(t_acc, t - dt / 2)
        hi = np.searchsorted(t_acc, t + dt / 2)
        n = hi - lo
        m = n > 0
        a_imu[m] = (cs[hi[m]] - cs[lo[m]]) / n[m, None]
    tr = np.asarray(train_mask, bool) & np.isfinite(a_opt).all(axis=1) & np.isfinite(a_imu).all(axis=1)
    if tr.sum() > 50:
        d = a_imu[tr] - a_opt[tr]
        d -= np.median(d, axis=0)                           # a constant offset is the bias state's
        sd = 1.4826 * np.median(np.abs(d), axis=0)
        sigma2 = float(np.mean(sd ** 2))
    else:
        sigma2 = 1000.0 ** 2
    q_acc = sigma2 * dt
    return PositionSettings(r_pos, q_acc, bias_rw ** 2,
                            {"r_pos_from": "rest SD", "q_acc_from": "train residual",
                             "sigma_acc_mm_s2": float(np.sqrt(sigma2)), "dt_frame_s": dt,
                             "n_train": int(tr.sum()), "n_rest": int(rs.sum())})


def position_kf(t_meas: np.ndarray, p_meas: np.ndarray, meas_ok: np.ndarray,
                t_acc: np.ndarray, a_acc: np.ndarray, st: PositionSettings,
                t_query: np.ndarray, *, window: tuple[float, float] | None = None,
                r_meas_mm: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Positions (mm, optical world) at ``t_query``: ``(smoothed, forward)`` -- the RTS
    smoother (both ends of a gap constrain it) and the causal forward filter. Inputs: the
    optical positions (``meas_ok`` selects the frames used) and the gravity-free world
    acceleration; the three axes share the dynamics and are filtered together. ``window``
    (s) limits the filter to that span (queries outside stay NaN). NaN before the first
    used optical frame. ``r_meas_mm`` (one per measurement, mm): per-measurement position
    SDs instead of the common ``st.r_pos_mm`` (e.g. anchors and pseudo-measurements of
    different precision)."""
    t_meas = np.asarray(t_meas, float)
    tq = np.asarray(t_query, float)
    t_acc = np.asarray(t_acc, float)
    ok = np.asarray(meas_ok, bool) & np.isfinite(p_meas).all(axis=1)
    if window is not None:
        ok &= (t_meas >= window[0]) & (t_meas <= window[1])
        ka = (t_acc >= window[0]) & (t_acc <= window[1])
        t_acc, a_acc = t_acc[ka], np.asarray(a_acc)[ka]
    out = np.full((len(tq), 3), np.nan)
    out_f = np.full((len(tq), 3), np.nan)
    if ok.sum() < 2:
        return out, out_f
    if window is not None:
        tq_in = (tq >= window[0]) & (tq <= window[1])
    else:
        tq_in = np.ones(len(tq), bool)
    qi = np.flatnonzero(tq_in)
    # one event list: 0 = acceleration sample, 1 = optical frame, 2 = query
    tt = np.concatenate([t_acc, t_meas[ok], tq[qi]])
    kind = np.concatenate([np.zeros(len(t_acc), int), np.ones(ok.sum(), int),
                           np.full(len(qi), 2)])
    ref = np.concatenate([np.arange(len(t_acc)), np.flatnonzero(ok), qi])
    order = np.lexsort((kind, tt))
    tt, kind, ref = tt[order], kind[order], ref[order]
    start = np.flatnonzero(kind == 1)[0]
    tt, kind, ref = tt[start:], kind[start:], ref[start:]
    n = len(tt)
    x = np.zeros((3, 3))                                      # rows: axes; cols: p, v, b
    x[:, 0] = p_meas[ref[0]]
    P = np.diag([st.r_pos_mm ** 2, 200.0 ** 2, 500.0 ** 2])
    xf = np.empty((n, 3, 3))
    Pf = np.empty((n, 3, 3))
    xp = np.empty((n, 3, 3))
    Pp = np.empty((n, 3, 3))
    Fk = np.empty((n, 3, 3))
    u = np.zeros(3)
    H = np.array([1.0, 0.0, 0.0])
    r2 = (np.full(len(t_meas), st.r_pos_mm ** 2) if r_meas_mm is None
          else np.asarray(r_meas_mm, float) ** 2)
    t_prev = tt[0]
    for k in range(n):
        dt = tt[k] - t_prev
        if dt > 0:
            F = np.array([[1.0, dt, -0.5 * dt * dt], [0.0, 1.0, -dt], [0.0, 0.0, 1.0]])
            Bv = np.array([0.5 * dt * dt, dt, 0.0])
            # white acceleration noise and bias random walk (continuous-time densities)
            Q = st.q_acc * np.array([[dt ** 3 / 3, dt ** 2 / 2, 0.0],
                                     [dt ** 2 / 2, dt, 0.0], [0.0, 0.0, 0.0]])
            Q[2, 2] = st.q_bias * dt
            x = x @ F.T + np.outer(u, Bv)
            P = F @ P @ F.T + Q
        else:
            F = np.eye(3)
        Fk[k], xp[k], Pp[k] = F, x, P
        if kind[k] == 0:
            u = a_acc[ref[k]]
        elif kind[k] == 1:
            S = H @ P @ H + r2[ref[k]]
            K = P @ H / S
            x = x + np.outer(p_meas[ref[k]] - x[:, 0], K)
            P = P - np.outer(K, H @ P)
        xf[k], Pf[k] = x, P
        t_prev = tt[k]
    xs = xf.copy()
    for k in range(n - 2, -1, -1):
        C = Pf[k] @ Fk[k + 1].T @ np.linalg.pinv(Pp[k + 1])
        xs[k] = xf[k] + (xs[k + 1] - xp[k + 1]) @ C.T
    q = kind == 2
    out[ref[q]] = xs[q][:, :, 0]
    out_f[ref[q]] = xf[q][:, :, 0]
    return out, out_f
