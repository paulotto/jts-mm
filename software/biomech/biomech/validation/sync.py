# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Temporal synchronisation of JTS-MM with Qualisys (plan D2; the sync note's methods).

Time model (affine, centred on ``t_ref`` on the JTS-MM master clock)::

    t_Q = t_S + c + α (t_S − t_ref)     ≡     t_Q = (1 + α)(t_S − t_ref) + τ,  τ = t_ref + c

``c`` (:attr:`TimeModel.offset_s`) is the Qualisys-minus-master lag at ``t_ref`` and ``α``
the relative clock-rate error. Centring decorrelates ``c`` and ``α``. ``t_S`` is the
JTS-MM **exposure centre** (cam0 trigger = exposure start, + 6 ms for the 12-ms exposure).

Methods, each usable on its own:

1. **CT visibility edges** (sync note Method 1): rise and fall midpoints in both systems;
   a vote over all edge differences gives the coarse lag without a start value (or within a
   hint range), then least squares on the matched edges gives ``c``, ``α`` and the
   rise/fall threshold bias ``δ`` (rises late by ``δ``, falls early by ``δ``; fitting it
   removes it from ``c``).
2. **CT angular-speed cross-correlation** (frame- and tip-offset invariant ``|ω|``) on a
   1-ms grid with a parabolic peak, per sync window -- a check of the edge result.
3. **Fine fit** (Method 2, joint with the spatial registration D3): cubic splines of the
   Qualisys poses (positions: cubic spline; rotations: ``RotationSpline``) evaluated at the
   mapped JTS-MM times; robust (Huber IRLS) Gauss–Newton over ``c``, ``α``, ``X`` and every
   ``Y_b`` on the start and end windows; ``Σ = σ² (JᵀJ)⁻¹``.
4. **Validation:** residuals on held-out movement phases (with the residual lag that best
   explains them, ``Σ e·v / Σ v·v``), and sliding-window lags of the MP/HP ``|ω|`` across
   the take, which must lie on a line of slope ``α`` (steps = dropped frames, curvature =
   unstable clocks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, RotationSpline

from biomech import timebase
from biomech.kinematics import se3
from biomech.validation import registration as REG
from biomech.validation.selection import Interval

EXPOSURE_S = 0.012          # JTS-MM global-shutter exposure; trigger time = exposure start


def frame_times(t_us: np.ndarray, exposure_s: float = EXPOSURE_S) -> np.ndarray:
    """JTS-MM optical frame times in seconds at the exposure centre."""
    return np.asarray(t_us, dtype=np.float64) * 1e-6 + 0.5 * exposure_s


# -- time model -------------------------------------------------------------------------
@dataclass
class TimeModel:
    offset_s: float                    # c = t_Q − t_S at t_ref
    alpha: float = 0.0                 # relative rate error
    t_ref_s: float = 0.0               # centring time (master clock)
    cov: np.ndarray | None = None      # 2x2 over (offset_s, alpha)

    def lag(self, t_s: np.ndarray | float) -> np.ndarray | float:
        return self.offset_s + self.alpha * (np.asarray(t_s, dtype=np.float64) - self.t_ref_s)

    def to_q(self, t_s: np.ndarray | float) -> np.ndarray | float:
        return np.asarray(t_s, dtype=np.float64) + self.lag(t_s)

    def to_s(self, t_q: np.ndarray | float) -> np.ndarray | float:
        t_q = np.asarray(t_q, dtype=np.float64)
        return (t_q - self.offset_s + self.alpha * self.t_ref_s) / (1.0 + self.alpha)

    @property
    def tau_s(self) -> float:
        """``τ`` of the centred form ``t_Q = (1 + α)(t_S − t_ref) + τ``."""
        return self.t_ref_s + self.offset_s

    @property
    def se_offset_s(self) -> float:
        return float(np.sqrt(self.cov[0, 0])) if self.cov is not None else float("nan")

    @property
    def se_alpha(self) -> float:
        return float(np.sqrt(self.cov[1, 1])) if self.cov is not None else float("nan")

    def recentred(self, t_ref_s: float) -> "TimeModel":
        """The same mapping centred on another ``t_ref`` (covariance propagated)."""
        d = t_ref_s - self.t_ref_s
        cov = None
        if self.cov is not None:
            A = np.array([[1.0, d], [0.0, 1.0]])
            cov = A @ self.cov @ A.T
        return TimeModel(self.offset_s + self.alpha * d, self.alpha, t_ref_s, cov)


# -- Method 1: visibility edges ---------------------------------------------------------
@dataclass
class Edges:
    rise: np.ndarray
    fall: np.ndarray

    def within(self, windows: Sequence[Interval]) -> "Edges":
        def keep(t: np.ndarray) -> np.ndarray:
            m = np.zeros(len(t), bool)
            for w in windows:
                m |= (t >= w.t0) & (t <= w.t1)
            return t[m]
        return Edges(keep(self.rise), keep(self.fall))

    def __len__(self) -> int:
        return len(self.rise) + len(self.fall)


def visibility_edges(t_s: np.ndarray, ok: np.ndarray, *, max_bracket_s: float | None = None
                     ) -> Edges:
    """Rise (reappearance) and fall (disappearance) times: the midpoint between the last
    invisible and first visible frame, and vice versa. Edges whose bracketing frames are
    more than ``max_bracket_s`` apart (a dropped frame next to the edge) are skipped."""
    t = np.asarray(t_s, dtype=np.float64)
    ok = np.asarray(ok, bool)
    d = np.diff(ok.astype(np.int8))
    i_rise = np.flatnonzero(d == 1)          # ok[i] False, ok[i+1] True
    i_fall = np.flatnonzero(d == -1)         # ok[i] True, ok[i+1] False

    def mid(i: np.ndarray) -> np.ndarray:
        gap = t[i + 1] - t[i]
        m = np.ones(len(i), bool) if max_bracket_s is None else gap <= max_bracket_s
        return 0.5 * (t[i[m]] + t[i[m] + 1])

    return Edges(mid(i_rise), mid(i_fall))


def coarse_offset_from_edges(eS: Edges, eQ: Edges, *, center: float | None = None,
                             half_range: float | None = None, tol: float = 0.05
                             ) -> tuple[float, int]:
    """Vote over all same-type edge differences ``t_Q − t_S``: the lag supported by the most
    edges (within ``±tol``). Returns ``(lag, votes)``. Without ``center`` the whole range of
    differences is searched."""
    diffs = [np.subtract.outer(q, s).ravel() for s, q in ((eS.rise, eQ.rise),
                                                           (eS.fall, eQ.fall))]
    d = np.sort(np.concatenate(diffs)) if diffs else np.array([])
    if center is not None and half_range is not None:
        d = d[np.abs(d - center) <= half_range]
    if d.size == 0:
        raise ValueError("no edge differences in the search range")
    lo = np.searchsorted(d, d - tol, "left")
    hi = np.searchsorted(d, d + tol, "right")
    votes = hi - lo
    k = int(np.argmax(votes))
    sel = d[lo[k]:hi[k]]
    return float(np.median(sel)), int(votes[k])


@dataclass
class EdgeFit:
    model: TimeModel
    delta_s: float                      # rise/fall threshold bias (rises late by δ in Q − S)
    se_delta_s: float
    n_rise: int
    n_fall: int
    resid_sd_s: float
    pairs: np.ndarray                   # (K, 3): t_S, t_Q, type (+1 rise, −1 fall)


def _match(tS: np.ndarray, tQ: np.ndarray, lag: np.ndarray, tol: float) -> np.ndarray:
    """Mutually nearest pairs (index into tS, index into tQ) with |t_Q − (t_S + lag)| ≤ tol."""
    if len(tS) == 0 or len(tQ) == 0:
        return np.zeros((0, 2), int)
    pred = tS + lag
    j = np.clip(np.searchsorted(tQ, pred), 1, len(tQ) - 1) if len(tQ) > 1 else np.zeros(
        len(tS), int)
    j = np.where(np.abs(tQ[j - 1] - pred) < np.abs(tQ[j] - pred), j - 1, j) if len(tQ) > 1 \
        else j
    good = np.abs(tQ[j] - pred) <= tol
    pairs = []
    for i in np.flatnonzero(good):
        # mutual: tS[i] must be the nearest S edge to tQ[j[i]] as well
        back = np.argmin(np.abs(pred - tQ[j[i]]))
        if back == i:
            pairs.append((i, j[i]))
    return np.array(pairs, int).reshape(-1, 2)


def fit_edges(eS: Edges, eQ: Edges, lag0: float, t_ref_s: float, *, tol: float = 0.08,
              n_iter: int = 5, fit_alpha: bool = True, fit_delta: bool = True) -> EdgeFit:
    """Least squares ``t_Q − t_S = c + α (t_S − t_ref) + s δ`` over matched edges (``s`` =
    +1 rise, −1 fall), re-matching with the current model and dropping > 3.5 robust SD."""
    qR, qF = np.sort(eQ.rise), np.sort(eQ.fall)
    theta = np.array([lag0, 0.0, 0.0])
    cov = np.full((3, 3), np.nan)
    rows: np.ndarray = np.zeros((0, 3))
    sd = float("nan")
    for it in range(n_iter):
        lagR = theta[0] + theta[1] * (eS.rise - t_ref_s) + theta[2]
        lagF = theta[0] + theta[1] * (eS.fall - t_ref_s) - theta[2]
        pr = _match(eS.rise, qR, lagR, tol)
        pf = _match(eS.fall, qF, lagF, tol)
        rows = np.vstack([np.c_[eS.rise[pr[:, 0]], qR[pr[:, 1]], np.ones(len(pr))],
                          np.c_[eS.fall[pf[:, 0]], qF[pf[:, 1]], -np.ones(len(pf))]])
        if len(rows) < 2:
            raise ValueError(f"fit_edges: only {len(rows)} matched edges")
        y = rows[:, 1] - rows[:, 0]
        cols = [np.ones(len(rows))]
        span = np.ptp(rows[:, 0])
        use_alpha = fit_alpha and span > 30.0 and len(rows) >= 4
        use_delta = fit_delta and (rows[:, 2] > 0).sum() >= 2 and (rows[:, 2] < 0).sum() >= 2
        if use_alpha:
            cols.append(rows[:, 0] - t_ref_s)
        if use_delta:
            cols.append(rows[:, 2])
        A = np.column_stack(cols)
        keep = np.ones(len(y), bool)
        for _ in range(3):
            sol, *_ = np.linalg.lstsq(A[keep], y[keep], rcond=None)
            r = y - A @ sol
            mad = 1.4826 * np.median(np.abs(r[keep] - np.median(r[keep])))
            new = np.abs(r) <= max(3.5 * mad, 1e-3)
            if (new == keep).all():
                break
            keep = new
        dof = max(int(keep.sum()) - A.shape[1], 1)
        sd = float(np.sqrt(np.sum(r[keep] ** 2) / dof))
        c = sd ** 2 * np.linalg.pinv(A[keep].T @ A[keep])
        full = np.zeros(3)
        full[0] = sol[0]
        k = 1
        cov = np.full((3, 3), np.nan)
        idx = [0]
        if use_alpha:
            full[1] = sol[k]
            idx.append(1)
            k += 1
        if use_delta:
            full[2] = sol[k]
            idx.append(2)
        for a_, ia in enumerate(idx):
            for b_, ib in enumerate(idx):
                cov[ia, ib] = c[a_, b_]
        theta = full
        rows = rows[keep]
        tol = min(tol, max(4.0 * sd, 0.02)) if it >= 1 else tol
    mcov = None if not np.isfinite(cov[:2, :2]).all() else cov[:2, :2]
    if mcov is None and np.isfinite(cov[0, 0]):
        mcov = np.array([[cov[0, 0], 0.0], [0.0, np.nan]])
    return EdgeFit(model=TimeModel(float(theta[0]), float(theta[1]), t_ref_s, mcov),
                   delta_s=float(theta[2]),
                   se_delta_s=float(np.sqrt(cov[2, 2])) if np.isfinite(cov[2, 2]) else np.nan,
                   n_rise=int((rows[:, 2] > 0).sum()), n_fall=int((rows[:, 2] < 0).sum()),
                   resid_sd_s=sd, pairs=rows)


# -- Method 2a: angular-speed cross-correlation -----------------------------------------
def angular_speed(t_s: np.ndarray, T: np.ndarray, *, max_gap_s: float | None = None,
                  baseline_s: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """``|ω|`` (deg/s) at pair mid-times (frame- and body-offset invariant).

    Pairs are consecutive valid frames, or -- with ``baseline_s`` -- each valid frame and
    the valid frame closest to ``baseline_s`` later (within a quarter of it), so that two
    systems at different rates use the same differentiation filter (a ``baseline_s``
    difference quotient). Pairs further apart than ``max_gap_s`` are dropped.
    """
    t = np.asarray(t_s, dtype=np.float64)
    idx = np.flatnonzero(se3.valid(T))
    if len(idx) < 2:
        return np.zeros(0), np.zeros(0)
    if baseline_s is None:
        a, b = idx[:-1], idx[1:]
    else:
        tv = t[idx]
        j = np.clip(np.searchsorted(tv, tv + baseline_s), 1, len(tv) - 1)
        j = np.where(np.abs(tv[j - 1] - tv - baseline_s) < np.abs(tv[j] - tv - baseline_s),
                     j - 1, j)
        good = (j > np.arange(len(tv))) & (np.abs(tv[j] - tv - baseline_s) <= 0.25 * baseline_s)
        a, b = idx[good], idx[j[good]]
    dt = t[b] - t[a]
    keep = dt > 0
    if max_gap_s is not None:
        keep &= dt <= max_gap_s
    a, b, dt = a[keep], b[keep], dt[keep]
    R = np.asarray(T, dtype=np.float64)[:, :3, :3]
    D = np.einsum("nij,nkj->nik", R[b], R[a])
    ang = np.degrees(np.linalg.norm(Rotation.from_matrix(D).as_rotvec(), axis=1))
    tm = 0.5 * (t[a] + t[b])
    o = np.argsort(tm, kind="stable")
    return tm[o], (ang / dt)[o]


def _on_grid(t_mid: np.ndarray, w: np.ndarray, grid: np.ndarray, smooth_s: float,
             max_gap_s: float) -> np.ndarray:
    """Speed samples on a uniform grid (linear within runs, NaN across gaps), then a centred
    NaN-aware moving mean of ``smooth_s`` (symmetric, so no lag)."""
    out = np.full(len(grid), np.nan)
    if len(t_mid) < 2:
        return out
    dt = grid[1] - grid[0]
    for a, b in timebase.time_segments(t_mid * 1e6, None, max_gap_s):
        if b - a < 2:
            continue
        sel = (grid >= t_mid[a]) & (grid <= t_mid[b - 1])
        out[sel] = np.interp(grid[sel], t_mid[a:b], w[a:b])
    n = timebase.odd_window(smooth_s, 1.0 / dt)
    if n > 1:
        fin = np.isfinite(out)
        k = np.ones(n)
        num = np.convolve(np.where(fin, out, 0.0), k, mode="same")
        den = np.convolve(fin.astype(float), k, mode="same")
        out = np.where(den >= 0.8 * n, num / np.maximum(den, 1), np.nan)
    return out


def _corr_curve(a: np.ndarray, bq: np.ndarray, n_lags: int, min_overlap: int
                ) -> np.ndarray:
    """Pearson correlation of ``a`` with ``bq[m : m + len(a)]`` for m = 0..n_lags-1 (NaN-aware)."""
    L = len(a)
    fa = np.isfinite(a)
    rho = np.full(n_lags, np.nan)
    for m in range(n_lags):
        b = bq[m:m + L]
        mk = fa & np.isfinite(b)
        if mk.sum() < min_overlap:
            continue
        x, y = a[mk], b[mk]
        x = x - x.mean()
        y = y - y.mean()
        den = np.sqrt(np.dot(x, x) * np.dot(y, y))
        if den > 0:
            rho[m] = np.dot(x, y) / den
    return rho


def _peak(lags: np.ndarray, rho: np.ndarray, curv_half_s: float = 0.005
          ) -> tuple[float, float, float, float]:
    """Parabolic peak ``(lag, rho_max, peak ratio, curvature)``; ratio = main peak / best
    value outside the main lobe (contiguous region above half the peak); curvature =
    ``ρ''`` at the peak (1/s², from the points ``±curv_half_s`` away)."""
    if not np.isfinite(rho).any():
        return float("nan"), float("nan"), float("nan"), float("nan")
    r = np.where(np.isfinite(rho), rho, -np.inf)
    k = int(np.argmax(r))
    lag = lags[k]
    if 0 < k < len(r) - 1 and np.isfinite(r[k - 1]) and np.isfinite(r[k + 1]):
        den = r[k - 1] - 2 * r[k] + r[k + 1]
        if den < 0:
            lag = lags[k] + 0.5 * (r[k - 1] - r[k + 1]) / den * (lags[1] - lags[0])
    lo, hi = k, k
    while lo > 0 and r[lo - 1] > 0.5 * r[k]:
        lo -= 1
    while hi < len(r) - 1 and r[hi + 1] > 0.5 * r[k]:
        hi += 1
    outside = np.r_[r[:lo], r[hi + 1:]]
    outside = outside[np.isfinite(outside)]
    ratio = float(r[k] / max(outside.max(), 1e-6)) if outside.size else float("inf")
    dl = lags[1] - lags[0] if len(lags) > 1 else 1.0
    m = max(int(round(curv_half_s / dl)), 1)
    curv = float("nan")
    if m <= k < len(r) - m and np.isfinite(r[k - m]) and np.isfinite(r[k + m]):
        curv = float((r[k - m] - 2 * r[k] + r[k + m]) / (m * dl) ** 2)
    return float(lag), float(r[k]), ratio, curv


@dataclass
class SpeedSeries:
    """``|ω|`` of one body in one system, ready for gridding."""
    t_mid: np.ndarray
    w: np.ndarray
    max_gap_s: float


def speed_series(t_s: np.ndarray, T: np.ndarray, max_gap_s: float,
                 baseline_s: float | None = 0.1, power: int = 2) -> SpeedSeries:
    """``|ω|^power`` of one body over a ``baseline_s`` difference quotient (the same filter in
    both systems; 100 ms = four JTS-MM frames: the JTS-MM rotation noise of ~0.5–1° per frame
    swamps a one-frame quotient). ``power = 2`` (default): white pose noise adds a constant
    to ``E|ω|²`` (removed by the correlation's mean), whereas ``|ω|`` is distorted
    non-linearly at low speed by noise of different size in the two systems."""
    tm, w = angular_speed(t_s, T, max_gap_s=None if baseline_s else max_gap_s,
                          baseline_s=baseline_s)
    return SpeedSeries(tm, w ** power, max_gap_s)


def xcorr_lag(sS: SpeedSeries | Sequence[SpeedSeries], sQ: SpeedSeries | Sequence[SpeedSeries],
              t0: float, t1: float, lag0: float, half_range: float, *, dt: float = 1e-3,
              smooth_s: float = 0.1, min_overlap_frac: float = 0.3, curve: bool = False
              ) -> dict[str, Any]:
    """Lag ``t_Q − t_S`` of the ``|ω|`` signals in the master window ``[t0, t1]``, searched in
    ``lag0 ± half_range`` on a ``dt`` grid with a parabolic peak. Several bodies (paired
    lists) are combined by averaging their correlation curves, weighted by each curve's squared
    peak (a noise-dominated body barely contributes)."""
    sS = [sS] if isinstance(sS, SpeedSeries) else list(sS)
    sQ = [sQ] if isinstance(sQ, SpeedSeries) else list(sQ)
    g = np.arange(t0, t1, dt)
    n_l = int(round(2 * half_range / dt)) + 1
    lags = lag0 - half_range + dt * np.arange(n_l)
    h = g[0] + lags[0] + dt * np.arange(len(g) + n_l - 1)
    curves = []
    n_ok = 0
    for a_s, b_s in zip(sS, sQ):
        a = _on_grid(a_s.t_mid, a_s.w, g, smooth_s, a_s.max_gap_s)
        b = _on_grid(b_s.t_mid, b_s.w, h, smooth_s, b_s.max_gap_s)
        n_ok = max(n_ok, int(np.isfinite(a).sum()))
        curves.append(_corr_curve(a, b, n_l, max(int(min_overlap_frac * len(g)), 10)))
    C = np.vstack(curves)
    pk = np.array([np.nanmax(c) if np.isfinite(c).any() else 0.0 for c in curves])
    wgt = np.clip(pk, 0.0, None) ** 2
    if wgt.sum() <= 0:
        wgt = np.ones(len(curves))
    fin = np.isfinite(C)
    num = np.nansum(np.where(fin, C, 0.0) * wgt[:, None], axis=0)
    den = np.sum(fin * wgt[:, None], axis=0)
    rho = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    lag, rmax, ratio, curv = _peak(lags, rho)
    at_edge = bool(np.isfinite(lag) and (abs(lag - lag0) > half_range - 2 * dt))
    # Approximate SE of the lag: var ≈ (1 − ρ) / (−ρ'' · N_eff), with N_eff the number of
    # independent samples after the ``smooth_s`` moving mean (a Cramér–Rao-type bound;
    # used to weight and screen windows, not reported as the sync uncertainty).
    n_eff = n_ok * dt / max(smooth_s, dt)
    se = float(np.sqrt(max(1.0 - rmax, 1e-6) / (-curv * n_eff))) \
        if np.isfinite(curv) and curv < 0 and n_eff > 0 else float("nan")
    out: dict[str, Any] = {"lag_s": lag, "rho": rmax, "peak_ratio": ratio, "n_grid": n_ok,
                           "se_s": se, "t_centre_s": 0.5 * (t0 + t1), "at_edge": at_edge}
    if curve:                         # for the sync views (R5b)
        out["lags_s"], out["rho_curve"] = lags, rho
    return out


# -- Method 2b: continuous-time fine fit, joint with the registration -------------------
class PoseSpline:
    """C² interpolant of a 200-Hz pose stream, per run of valid frames (no gap is bridged).

    Positions: cubic spline per axis; rotations: :class:`scipy.spatial.transform.RotationSpline`.
    Queries outside every run (or within ``margin_s`` of a run end) are NaN.
    """

    def __init__(self, t_s: np.ndarray, T: np.ndarray, min_frames: int = 4) -> None:
        t = np.asarray(t_s, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        self.runs: list[tuple[float, float, CubicSpline, RotationSpline]] = []
        for a, b in timebase.segments(se3.valid(T)):
            if b - a < min_frames:
                continue
            ts = t[a:b]
            self.runs.append((ts[0], ts[-1], CubicSpline(ts, T[a:b, :3, 3], axis=0),
                              RotationSpline(ts, Rotation.from_matrix(T[a:b, :3, :3]))))
        self.starts = np.array([r[0] for r in self.runs])
        self.ends = np.array([r[1] for r in self.runs])

    def covered(self, tq: np.ndarray, margin_s: float = 0.0) -> np.ndarray:
        tq = np.asarray(tq, dtype=np.float64)
        if not self.runs:
            return np.zeros(len(tq), bool)
        k = np.searchsorted(self.starts, tq, "right") - 1
        kk = np.clip(k, 0, None)
        return (k >= 0) & (tq >= self.starts[kk] + margin_s) & (tq <= self.ends[kk] - margin_s)

    def __call__(self, tq: np.ndarray) -> np.ndarray:
        tq = np.asarray(tq, dtype=np.float64)
        out = np.full((len(tq), 4, 4), np.nan)
        if not self.runs:
            return out
        k = np.searchsorted(self.starts, tq, "right") - 1
        kk = np.clip(k, 0, None)
        inside = (k >= 0) & (tq <= self.ends[kk]) & (tq >= self.starts[kk])
        for run in np.unique(k[inside]):
            sel = inside & (k == run)
            _, _, ps, rs = self.runs[run]
            out[sel, :3, 3] = ps(tq[sel])
            out[sel, :3, :3] = rs(tq[sel]).as_matrix()
            out[sel, 3, 3] = 1.0
            out[sel, 3, :3] = 0.0
        return out

    def velocity(self, tq: np.ndarray) -> np.ndarray:
        """Translational velocity (mm/s), NaN outside the runs."""
        tq = np.asarray(tq, dtype=np.float64)
        out = np.full((len(tq), 3), np.nan)
        if not self.runs:
            return out
        k = np.searchsorted(self.starts, tq, "right") - 1
        kk = np.clip(k, 0, None)
        inside = (k >= 0) & (tq <= self.ends[kk]) & (tq >= self.starts[kk])
        for run in np.unique(k[inside]):
            sel = inside & (k == run)
            out[sel] = self.runs[run][2](tq[sel], 1)
        return out


@dataclass
class FineFit:
    model: TimeModel
    X: np.ndarray
    Y: dict[str, np.ndarray]
    cov: np.ndarray | None              # full parameter covariance (ms, ppm, rad, mm)
    names: list[str]                    # parameter names in cov order
    sigma2: float                       # weighted residual variance (normalised units)
    se_inflation: float                 # sqrt((1 + ρ1)/(1 − ρ1)) from the residual autocorrelation
    n_frames: dict[str, int]
    rms_pos_mm: dict[str, float]
    rms_rot_deg: dict[str, float]
    n_iter: int = 0
    resid: dict[str, np.ndarray] = field(default_factory=dict)  # body -> (n, 6) mm / deg
    n_gated: dict[str, int] = field(default_factory=dict)       # frames given weight 0
    scale: dict[str, tuple[float, float]] = field(default_factory=dict)  # σ (mm, deg) per body

    def se(self, name: str) -> float:
        if self.cov is None:
            return float("nan")
        i = self.names.index(name)
        return float(np.sqrt(max(self.cov[i, i], 0.0)))

    def block_se(self, block: str) -> tuple[float, float]:
        """(rotation SE deg, translation SE mm) of ``X`` or ``Y_<body>`` (RMS over axes)."""
        if self.cov is None or f"{block}.rx" not in self.names:
            return float("nan"), float("nan")
        i = self.names.index(f"{block}.rx")
        d = np.sqrt(np.clip(np.diag(self.cov)[i:i + 6], 0, None))
        return (float(np.degrees(np.sqrt(np.mean(d[:3] ** 2)))),
                float(np.sqrt(np.mean(d[3:] ** 2))))


def _lag1_autocorr(r: np.ndarray, runs: list[tuple[int, int]]) -> float:
    num = den = 0.0
    for a, b in runs:
        if b - a < 3:
            continue
        x = r[a:b] - r[a:b].mean(axis=0)
        num += float(np.sum(x[1:] * x[:-1]))
        den += float(np.sum(x * x))
    return num / den if den > 0 else 0.0


def fit_fine(tS: np.ndarray, S: dict[str, np.ndarray], Qspl: dict[str, PoseSpline],
             windows: Sequence[Interval], model0: TimeModel, X0: np.ndarray,
             Y0: dict[str, np.ndarray], *, fit_time: bool = True, sigma_pos_mm: float = 0.3,
             sigma_rot_deg: float = 0.15, huber_k: float = 2.5, gate_k: float = 10.0,
             loss: str = "huber",
             n_irls: int = 5, margin_s: float = 0.0025, h_t: float = 2e-4,
             fd_eps: float = 1e-6) -> FineFit:
    """Joint robust fit of ``(c, α)`` (if ``fit_time``), ``X`` and every ``Y_b``.

    Residual per body ``b`` and JTS-MM frame ``i`` in a window:
    ``X T_S,b(t_i) Y_b`` against the Qualisys spline at ``t_i + c + α (t_i − t_ref)``,
    position in mm / ``σ_p,b`` and rotation (``log(R_Qᵀ R)``) in rad / ``σ_R,b``. ``σ`` starts at
    ``sigma_pos_mm`` / ``sigma_rot_deg`` and, from the second round on, is re-estimated per
    body from the median residual norm of its non-gated frames (``median |r| / 1.538`` for a
    3-D Gaussian), so each body is weighted by its own precision. Huber weights are
    per frame (position and rotation parts separately), refreshed with the frame mask in each
    IRLS round; a frame whose position or rotation residual exceeds ``gate_k`` σ
    (``3 gate_k`` in the first round) gets weight 0 (gross outliers such as a flipped pose branch; counted in
    ``n_gated``). The time columns of the Jacobian are ``∂r/∂t'`` (central difference of the
    spline, step ``h_t``) times ``∂t'/∂(c, α)``; the transform columns are forward differences
    with the spline values held fixed.
    """
    bodies = [b for b in S if b in Qspl]
    tS = np.asarray(tS, dtype=np.float64)
    sr = np.radians(sigma_rot_deg)
    t_ref = model0.t_ref_s
    in_win = np.zeros(len(tS), bool)
    for w in windows:
        in_win |= (tS >= w.t0) & (tS <= w.t1)
    base = {b: np.flatnonzero(in_win & se3.valid(S[b])) for b in bodies}
    nb = len(bodies)
    names = (["offset_ms", "alpha_ppm"] if fit_time else []) + [
        f"{blk}.{c}" for blk in ["X", *(f"Y_{b}" for b in bodies)]
        for c in ("rx", "ry", "rz", "tx", "ty", "tz")]
    nt = 2 if fit_time else 0

    def tprime(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        if not fit_time:
            return model0.to_q(t)
        c = model0.offset_s + x[0] * 1e-3
        a = model0.alpha + x[1] * 1e-6
        return t + c + a * (t - t_ref)

    def transforms(x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        X = REG.perturb(X0, x[nt:nt + 6])
        Y = {b: REG.perturb(Y0[b], x[nt + 6 + 6 * k: nt + 12 + 6 * k])
             for k, b in enumerate(bodies)}
        return X, Y

    x = np.zeros(nt + 6 + 6 * nb)
    idx: dict[str, np.ndarray] = {}
    w: dict[str, np.ndarray] = {}
    sig = {b: (float(sigma_pos_mm), float(sr)) for b in bodies}
    n_iter = 0

    def resid_body(b: str, X: np.ndarray, Yb: np.ndarray, Q: np.ndarray) -> np.ndarray:
        P = REG.apply(X, S[b][idx[b]], Yb)
        r = REG.pose_residuals(P, Q, *sig[b])
        # A trial step that maps a frame outside the Qualisys runs is penalised (never
        # rewarded with a zero residual); the frame mask is refreshed every IRLS round.
        return np.where(np.isfinite(r), r, 4.0 * huber_k)

    def fun(z: np.ndarray) -> np.ndarray:
        X, Y = transforms(z)
        out = [resid_body(b, X, Y[b], Qspl[b](tprime(z, tS[idx[b]]))) * np.sqrt(w[b])
               for b in bodies]
        return np.vstack(out).ravel()

    def jac(z: np.ndarray) -> np.ndarray:
        X, Y = transforms(z)
        blocks = []
        for k, b in enumerate(bodies):
            ti = tS[idx[b]]
            tp = tprime(z, ti)
            Q = Qspl[b](tp)
            sw = np.sqrt(w[b])
            r0 = resid_body(b, X, Y[b], Q)
            Jb = np.zeros((r0.size, len(z)))
            if fit_time:
                dr = (resid_body(b, X, Y[b], Qspl[b](tp + h_t))
                      - resid_body(b, X, Y[b], Qspl[b](tp - h_t))) / (2 * h_t)
                Jb[:, 0] = (dr * sw).ravel() * 1e-3
                Jb[:, 1] = (dr * sw * (ti - t_ref)[:, None]).ravel() * 1e-6
            for j in range(6):
                e = np.zeros(6)
                e[j] = fd_eps
                Xp = REG.perturb(X, e)
                Jb[:, nt + j] = ((resid_body(b, Xp, Y[b], Q) - r0) * sw).ravel() / fd_eps
                Yp = REG.perturb(Y[b], e)
                Jb[:, nt + 6 + 6 * k + j] = ((resid_body(b, X, Yp, Q) - r0) * sw).ravel() / fd_eps
            blocks.append(Jb)
        return np.vstack(blocks)

    for n_iter in range(1, n_irls + 1):
        # frame mask: the mapped time must lie inside a Qualisys run (no gap bridging)
        for b in bodies:
            tp = tprime(x, tS[base[b]])
            idx[b] = base[b][Qspl[b].covered(tp, margin_s)]
        X, Y = transforms(x)
        if n_iter > 1:                       # per-body robust scale from the last round
            for b in bodies:
                r = resid_body(b, X, Y[b], Qspl[b](tprime(x, tS[idx[b]])))
                keep = (np.linalg.norm(r[:, :3], axis=1) <= gate_k) & (
                    np.linalg.norm(r[:, 3:], axis=1) <= gate_k)
                if keep.sum() >= 10:
                    sp = sig[b][0] * float(np.median(np.linalg.norm(r[keep, :3], axis=1))) / 1.538
                    so = sig[b][1] * float(np.median(np.linalg.norm(r[keep, 3:], axis=1))) / 1.538
                    sig[b] = (max(sp, 0.2 * sigma_pos_mm), max(so, 0.2 * sr))
        for b in bodies:
            r = resid_body(b, X, Y[b], Qspl[b](tprime(x, tS[idx[b]])))
            # round 1: least squares behind a loose gate (3 gate_k) around the robust
            # initialisation; later rounds: Huber weights behind the gate
            w[b] = (np.ones_like(r) if n_iter == 1 else REG.tukey_weights(r)
                    if loss == "tukey" else REG.huber_weights(r, huber_k))
            g = gate_k * (3.0 if n_iter == 1 else 1.0)
            gross = ((np.linalg.norm(r[:, :3], axis=1) > g)
                     | (np.linalg.norm(r[:, 3:], axis=1) > g))
            w[b][gross] = 0.0
        if sum(len(idx[b]) for b in bodies) < len(x):
            raise ValueError("fit_fine: too few frames in the sync windows")
        sol = least_squares(fun, x, jac=jac, method="trf", x_scale="jac", xtol=1e-10,
                            ftol=1e-10, gtol=1e-10, max_nfev=50)
        step = np.abs(sol.x - x)
        x = sol.x
        if n_iter > 2 and step[:nt].max(initial=0) < 1e-4 and step[nt:].max() < 1e-7:
            break
    J = jac(x)
    r = fun(x)
    dof = max(r.size - x.size, 1)
    s2 = float(np.sum(r ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = None
    X, Y = transforms(x)
    rho1 = []
    res_mm: dict[str, np.ndarray] = {}
    rms_p: dict[str, float] = {}
    rms_r: dict[str, float] = {}
    scale: dict[str, tuple[float, float]] = {}
    n_gated = {b: int((w[b][:, 0] == 0).sum()) for b in bodies}
    for b in bodies:
        keep = w[b][:, 0] > 0
        ti = tS[idx[b]][keep]
        Q = Qspl[b](tprime(x, ti))
        P = REG.apply(X, S[b][idx[b]][keep], Y[b])
        dp, ang = REG.pose_error(P, Q)
        rr = REG.pose_residuals(P, Q, 1.0, 1.0)
        rr[:, 3:] = np.degrees(rr[:, 3:])
        res_mm[b] = rr
        scale[b] = (sig[b][0], float(np.degrees(sig[b][1])))
        rms_p[b] = float(np.sqrt(np.nanmean(np.sum(dp ** 2, axis=1))))
        rms_r[b] = float(np.sqrt(np.nanmean(ang ** 2)))
        runs = timebase.time_segments(ti * 1e6, None, 0.0375)
        rho1.append(_lag1_autocorr(rr / np.r_[[scale[b][0]] * 3, [scale[b][1]] * 3], runs))
    rho = float(np.clip(np.mean(rho1), 0.0, 0.95)) if rho1 else 0.0
    model = model0
    if fit_time:
        c = model0.offset_s + x[0] * 1e-3
        a = model0.alpha + x[1] * 1e-6
        mc = None
        if cov is not None:
            mc = cov[:2, :2] * np.array([[1e-6, 1e-9], [1e-9, 1e-12]])
        model = TimeModel(float(c), float(a), t_ref, mc)
    return FineFit(model=model, X=X, Y=Y, cov=cov, names=names, sigma2=s2,
                   se_inflation=float(np.sqrt((1 + rho) / (1 - rho))),
                   n_frames={b: int(len(idx[b]) - n_gated[b]) for b in bodies},
                   rms_pos_mm=rms_p, rms_rot_deg=rms_r, n_iter=n_iter, resid=res_mm,
                   n_gated=n_gated, scale=scale)


# -- validation ---------------------------------------------------------------------------
def evaluate_intervals(tS: np.ndarray, S: dict[str, np.ndarray], Qspl: dict[str, PoseSpline],
                       model: TimeModel, X: np.ndarray, Y: dict[str, np.ndarray],
                       intervals: Sequence[Interval], *, head: str = "HP", jaw: str = "MP",
                       margin_s: float = 0.0025) -> pd.DataFrame:
    """Residuals of the fitted model on held-out intervals, per interval and body, plus the
    frame-free jaw-in-head pose (``body = "<jaw>|<head>"``).

    ``lag_resid_ms`` is the time shift that best explains the position error
    (``Σ e·v / Σ v·v``, e = JTS-MM − Qualisys, v = Qualisys velocity): ≈ 0 when the time
    model holds in that interval.
    """
    tS = np.asarray(tS, dtype=np.float64)
    rows = []
    for iv in intervals:
        sel = np.flatnonzero((tS >= iv.t0) & (tS <= iv.t1))
        tq = model.to_q(tS[sel])
        Qs: dict[str, np.ndarray] = {}
        for b in S:
            if b not in Qspl or b not in Y:
                continue
            Q = Qspl[b](tq)
            ok = Qspl[b].covered(tq, margin_s)
            Q[~ok] = np.nan
            Qs[b] = Q
            P = REG.apply(X, S[b][sel], Y[b])
            dp, ang = REG.pose_error(P, Q)
            good = np.isfinite(ang)
            v = Qspl[b].velocity(tq)
            vv = np.nansum(np.sum(v[good] ** 2, axis=1))
            ev = np.nansum(np.sum(dp[good] * v[good], axis=1))
            rows.append(_err_row(iv, b, dp[good], ang[good],
                                 lag_ms=1e3 * ev / vv if vv > 0 else np.nan,
                                 speed=np.linalg.norm(v[good], axis=1)))
        if head in Qs and jaw in Qs:
            rel_S, rel_Q = REG.relative_pose_pair(S[head][sel], S[jaw][sel], Qs[head], Qs[jaw],
                                                  Y[head], Y[jaw])
            dp, ang = REG.pose_error(rel_S, rel_Q)
            good = np.isfinite(ang)
            rows.append(_err_row(iv, f"{jaw}|{head}", dp[good], ang[good], lag_ms=np.nan,
                                 speed=None))
    return pd.DataFrame(rows)


def _err_row(iv: Interval, body: str, dp: np.ndarray, ang: np.ndarray, *, lag_ms: float,
             speed: np.ndarray | None) -> dict[str, object]:
    n = len(ang)
    e = np.linalg.norm(dp, axis=1) if n else np.zeros(0)
    slope = np.nan
    if speed is not None and n > 10 and np.ptp(speed) > 0:
        slope = float(np.polyfit(speed, e, 1)[0]) * 1e3        # mm per (m/s)
    return {"interval": iv.name, "t0_s": iv.t0, "t1_s": iv.t1, "body": body, "n": n,
            "pos_rms_mm": float(np.sqrt(np.mean(e ** 2))) if n else np.nan,
            "pos_p95_mm": float(np.percentile(e, 95)) if n else np.nan,
            "bias_x_mm": float(dp[:, 0].mean()) if n else np.nan,
            "bias_y_mm": float(dp[:, 1].mean()) if n else np.nan,
            "bias_z_mm": float(dp[:, 2].mean()) if n else np.nan,
            "rot_rms_deg": float(np.sqrt(np.mean(ang ** 2))) if n else np.nan,
            "lag_resid_ms": float(lag_ms),
            "err_vs_speed_mm_per_mps": slope,
            "speed_rms_mm_s": float(np.sqrt(np.mean(speed ** 2))) if speed is not None and n
            else np.nan}


def sliding_lags(sS: dict[str, SpeedSeries], sQ: dict[str, SpeedSeries], model: TimeModel,
                 t0: float, t1: float, *, win_s: float = 15.0, step_s: float = 5.0,
                 half_range_s: float = 0.1, smooth_s: float = 0.1, min_rho: float = 0.5,
                 min_ratio: float = 1.2, max_se_ms: float = 3.0) -> pd.DataFrame:
    """Lag of the ``|ω|`` signals in sliding windows (per body and all bodies combined),
    searched around the model's lag; ``resid_ms`` = estimate − model at the window centre.
    A window is ``ok`` when its peak is clear (``rho``, peak ratio, not at the search edge)
    and its approximate SE (from the peak curvature) is at most ``max_se_ms``; windows
    without enough motion fail the SE test."""
    rows = []
    bodies = [b for b in sS if b in sQ]
    starts = np.arange(t0, max(t1 - win_s, t0) + 1e-9, step_s)
    for a in starts:
        b_ = min(a + win_s, t1)
        lag0 = float(model.lag(0.5 * (a + b_)))
        for name, ss, qq in [(b, [sS[b]], [sQ[b]]) for b in bodies] + [
                ("joint", [sS[b] for b in bodies], [sQ[b] for b in bodies])]:
            r = xcorr_lag(ss, qq, a, b_, lag0, half_range_s, smooth_s=smooth_s)
            ok = (np.isfinite(r["lag_s"]) and r["rho"] >= min_rho
                  and r["peak_ratio"] >= min_ratio and not r["at_edge"]
                  and np.isfinite(r["se_s"]) and r["se_s"] * 1e3 <= max_se_ms)
            rows.append({"t_centre_s": 0.5 * (a + b_), "body": name, "lag_s": r["lag_s"],
                         "resid_ms": 1e3 * (r["lag_s"] - lag0), "se_ms": 1e3 * r["se_s"],
                         "rho": r["rho"], "peak_ratio": r["peak_ratio"], "ok": bool(ok)})
    return pd.DataFrame(rows)


def sliding_pose_lags(tS: np.ndarray, S: dict[str, np.ndarray], Qspl: dict[str, PoseSpline],
                      model: TimeModel, X: np.ndarray, Y: dict[str, np.ndarray],
                      bodies: Sequence[str], t0: float, t1: float, *, win_s: float = 15.0,
                      step_s: float = 5.0, min_speed_mm_s: float = 5.0, max_se_ms: float = 3.0,
                      n_iter: int = 3, margin_s: float = 0.0025) -> pd.DataFrame:
    """Per-window time shift of the registered poses (``X``, ``Y`` fixed): Gauss–Newton on
    ``δ`` with ``e = X T_S Y − Q(t' + δ)`` → ``δ += Σ e·v / Σ v·v`` over the bodies' positions.
    Uses the registration, so it is not independent of the fit, but it is far more precise
    than the ``|ω|`` correlation and resolves single-frame steps. Rows: ``body = "pose"``."""
    tS = np.asarray(tS, dtype=np.float64)
    rows = []
    P = {b: REG.apply(X, S[b], Y[b]) for b in bodies}
    for a in np.arange(t0, max(t1 - win_s, t0) + 1e-9, step_s):
        b_ = min(a + win_s, t1)
        sel = np.flatnonzero((tS >= a) & (tS <= b_))
        lag0 = float(model.lag(0.5 * (a + b_)))
        tq0 = model.to_q(tS[sel])
        d = 0.0
        se = np.nan
        n = 0
        vrms = np.nan
        for _ in range(n_iter):
            num = den = ss = 0.0
            n = 0
            for b in bodies:
                tq = tq0 + d
                ok = Qspl[b].covered(tq, margin_s) & se3.valid(P[b][sel])
                if not ok.any():
                    continue
                e = P[b][sel][ok, :3, 3] - Qspl[b](tq[ok])[:, :3, 3]
                v = Qspl[b].velocity(tq[ok])
                num += float(np.sum(e * v))
                den += float(np.sum(v * v))
                ss += float(np.sum(e * e))
                n += int(ok.sum())
            if den <= 0 or n < 10:
                break
            step = num / den
            d += step
            sig2 = max(ss / (3 * n) - step ** 2 * den / (3 * n), 1e-12)
            se = float(np.sqrt(sig2 / den))
            vrms = float(np.sqrt(den / n))
        ok_w = bool(np.isfinite(se) and se * 1e3 <= max_se_ms and vrms >= min_speed_mm_s)
        rows.append({"t_centre_s": 0.5 * (a + b_), "body": "pose", "lag_s": lag0 + d,
                     "resid_ms": 1e3 * d, "se_ms": 1e3 * se, "rho": np.nan,
                     "peak_ratio": np.nan, "ok": ok_w})
    return pd.DataFrame(rows)


def lag_line(df: pd.DataFrame, model: TimeModel, body: str = "joint",
             step_ms: float = 10.0) -> dict[str, float]:
    """Straight-line check of the sliding lags ``t_Q − t_S − c`` against ``t_S − t_ref``:
    the slope (ppm; compare with the model's α), the intercept (ms; ≈ 0 when the fit windows
    agree with the take), residual SD, largest jump between neighbouring windows (a step =
    dropped/re-associated frames) and the quadratic term (curvature = unstable clocks).
    Weighted by the windows' SEs, > 4 robust SD outliers dropped."""
    d = df[(df["body"] == body) & df["ok"]].sort_values("t_centre_s")
    out = {"n_windows": int(len(d)), "n_windows_total": int((df["body"] == body).sum()),
           "slope_ppm": np.nan, "slope_se_ppm": np.nan, "intercept_ms": np.nan,
           "resid_sd_ms": np.nan, "max_step_ms": np.nan, "n_steps": 0,
           "quad_ms_per_100s2": np.nan, "quad_se": np.nan, "model_alpha_ppm": model.alpha * 1e6,
           "n_outliers": 0, "chi2_dof": np.nan}
    if len(d) < 3:
        return out
    t_all = d["t_centre_s"].to_numpy() - model.t_ref_s
    y_all = d["lag_s"].to_numpy() - model.offset_s
    se = d["se_ms"].to_numpy() * 1e-3 if "se_ms" in d else np.ones(len(d))
    w_all = 1.0 / np.maximum(se, 1e-5) ** 2
    keep = np.ones(len(d), bool)
    for _ in range(5):                       # weighted LS, dropping > 4 robust SD outliers
        t, y, w = t_all[keep], y_all[keep], w_all[keep]
        A = np.c_[np.ones_like(t), t]
        Aw = A * np.sqrt(w)[:, None]
        sol, *_ = np.linalg.lstsq(Aw, y * np.sqrt(w), rcond=None)
        z = (y_all - (sol[0] + sol[1] * t_all)) * np.sqrt(w_all)
        mad = 1.4826 * np.median(np.abs(z[keep] - np.median(z[keep])))
        new = np.abs(z) <= max(4.0 * mad, 1e-9)
        if (new == keep).all() or new.sum() < 3:
            break
        keep = new
    t, y, w = t_all[keep], y_all[keep], w_all[keep]
    A = np.c_[np.ones_like(t), t]
    r = y - A @ sol
    dof = max(len(t) - 2, 1)
    chi2 = float(np.sum(w * r ** 2) / dof)
    cov = np.linalg.inv((A * w[:, None]).T @ A) * max(chi2, 1.0)
    out.update(n_windows=int(keep.sum()), n_outliers=int((~keep).sum()),
               slope_ppm=float(sol[1] * 1e6),
               slope_se_ppm=float(np.sqrt(cov[1, 1]) * 1e6), intercept_ms=float(sol[0] * 1e3),
               resid_sd_ms=float(np.sqrt(np.sum(r ** 2) / dof) * 1e3), chi2_dof=chi2)
    d = d[keep]
    jumps = np.abs(np.diff(d["resid_ms"].to_numpy()))
    out["max_step_ms"] = float(jumps.max()) if len(jumps) else np.nan
    out["n_steps"] = int((jumps > step_ms).sum())
    if len(t) >= 4:
        u = t / 100.0
        B = np.c_[np.ones_like(u), u, u ** 2]
        sw = np.sqrt(w)[:, None]
        q, *_ = np.linalg.lstsq(B * sw, y * sw[:, 0], rcond=None)
        rq = y - B @ q
        s2q = float(np.sum(w * rq ** 2) / max(len(t) - 3, 1))
        cq = max(s2q, 1.0) * np.linalg.inv((B * w[:, None]).T @ B)
        out.update(quad_ms_per_100s2=float(q[2] * 1e3), quad_se=float(np.sqrt(cq[2, 2]) * 1e3))
    return out


# -- clock variants and pre-registered criteria (R5c.3, V6) ---------------------------------
def clock_variants(tS: np.ndarray, S: dict[str, np.ndarray], Qspl: dict[str, PoseSpline],
                   windows: Sequence[Interval], in_win: np.ndarray, f_ct: FineFit, *,
                   fit_joint: FineFit | None = None, ct: str = "CT", gate_sd: float = 3.0,
                   **kw: Any) -> dict[str, FineFit]:
    """The sensitivity fits of the clock (each evaluated later with the same primary
    outcome): ``ct`` (primary), ``joint``, ``two_stage`` (CT-only, then the MP/HP frames of
    the sync windows whose registered residual lies within ``gate_sd`` robust SD of the
    stage-1 solution, refitted jointly), ``hybrid`` (clock from ``ct``, X from
    ``two_stage``) and ``tukey`` (CT-only with a redescending loss)."""
    out: dict[str, FineFit] = {"ct": f_ct}
    if fit_joint is not None:
        out["joint"] = fit_joint
    bodies = [b for b in Qspl if b != ct]
    S2 = {ct: S[ct]}
    Y0 = {ct: f_ct.Y[ct]}
    for b in bodies:
        sb = np.flatnonzero(in_win & se3.valid(S[b]))
        if len(sb) < 10:
            continue
        tq = f_ct.model.to_q(tS[sb])
        Q = Qspl[b](tq)
        ok = Qspl[b].covered(tq, 0.0025)
        dp, ang = REG.pose_error(REG.apply(f_ct.X, S[b][sb], f_ct.Y[b]), Q)
        e = np.linalg.norm(dp, axis=1)
        good = ok & np.isfinite(e)
        if good.sum() < 10:
            continue
        zs = []
        for v in (e[good], ang[good]):
            med = np.median(v)
            zs.append(np.abs(v - med) / max(1.4826 * np.median(np.abs(v - med)), 1e-9))
        keep = np.zeros(len(sb), bool)
        keep[np.flatnonzero(good)[(zs[0] <= gate_sd) & (zs[1] <= gate_sd)]] = True
        Sb = np.full_like(S[b], np.nan)
        Sb[sb[keep]] = S[b][sb[keep]]
        S2[b] = Sb
        Y0[b] = f_ct.Y[b]
    try:
        two = fit_fine(tS, S2, {b: Qspl[b] for b in S2}, windows, f_ct.model, f_ct.X, Y0, **kw)
        out["two_stage"] = two
        hyb = FineFit(model=f_ct.model, X=two.X, Y=dict(two.Y), cov=None, names=[],
                      sigma2=np.nan, se_inflation=f_ct.se_inflation, n_frames=two.n_frames,
                      rms_pos_mm={}, rms_rot_deg={})
        out["hybrid"] = hyb
    except ValueError:
        pass
    try:
        out["tukey"] = fit_fine(tS, {ct: S[ct]}, {ct: Qspl[ct]}, windows, f_ct.model, f_ct.X,
                                {ct: f_ct.Y[ct]}, loss="tukey", **kw)
    except ValueError:
        pass
    return out


def hac_bandwidth(u: np.ndarray) -> int:
    """Bartlett (Newey–West) bandwidth: the Andrews (1991) AR(1) plug-in
    ``L = ⌈1.1447 (α̂ n)^{1/3}⌉`` with ``α̂ = 4ρ̂² / ((1 − ρ̂)² (1 + ρ̂)²)`` and ``ρ̂`` the lag-1
    autocorrelation of ``u`` clipped to [0, 0.97]; at least the Newey–West (1994) fixed
    rule ``⌊4 (n/100)^{2/9}⌋``, at most ``n − 1``."""
    n = len(u)
    if n < 3:
        return 0
    x = u - u.mean()
    den = float(np.sum(x * x))
    rho = float(np.clip(np.sum(x[1:] * x[:-1]) / den, 0.0, 0.97)) if den > 0 else 0.0
    a = 4 * rho ** 2 / ((1 - rho) ** 2 * (1 + rho) ** 2)
    L = max(int(np.ceil(1.1447 * (a * n) ** (1 / 3))), int(np.floor(4 * (n / 100) ** (2 / 9))))
    return int(min(L, n - 1))


def newey_west_var(u: np.ndarray, bandwidth: int | None = None) -> tuple[float, int]:
    """HAC (Newey–West, Bartlett weights ``1 − l/(L+1)``) variance of ``Σ uᵢ`` for a
    mean-zero score series in time order: ``Σ uᵢ² + 2 Σ_{l=1..L} w_l Σᵢ uᵢ uᵢ₋ₗ``. ``L`` from
    :func:`hac_bandwidth` unless given. Returns ``(variance, L)``."""
    u = np.asarray(u, float)
    L = hac_bandwidth(u) if bandwidth is None else int(bandwidth)
    v = float(np.sum(u * u))
    for lag in range(1, L + 1):
        v += 2.0 * (1.0 - lag / (L + 1.0)) * float(np.sum(u[lag:] * u[:-lag]))
    return max(v, 0.0), L


def window_lags(tS: np.ndarray, S_ct: np.ndarray, Q_ct: PoseSpline, model: TimeModel,
                X: np.ndarray, Y_ct: np.ndarray, windows: Sequence[Interval]) -> pd.DataFrame:
    """Per sync window, the residual time shift of the CT that best explains its position
    error under the fitted clock (``lag = Σ e·v / Σ v·v``). Two windows give two independent
    clock offsets; their difference tests the affine model (V6).

    SEs of the lag: ``se_ms_hac`` (VD19, primary): Newey–West HAC of the frame scores
    ``uᵢ = rᵢ·vᵢ`` (``r = e − lag·v``) in time order, ``SE = √(HAC var(Σu)) / Σ v·v``,
    bandwidth by :func:`hac_bandwidth` (``hac_lags``); ``se_ms_ar1`` (the pre-VD19 SE): the
    white-noise SE inflated by ``√((1 + ρ)/(1 − ρ))`` with ``ρ`` the lag-1 autocorrelation
    of the residual norms."""
    rows = []
    for w in windows:
        sel = np.flatnonzero((tS >= w.t0) & (tS <= w.t1) & se3.valid(S_ct))
        tq = model.to_q(tS[sel])
        ok = Q_ct.covered(tq, 0.0025)
        sel, tq = sel[ok], tq[ok]
        if len(sel) < 10:
            rows.append({"window": w.name, "n": len(sel), "lag_ms": np.nan,
                         "se_ms_ar1": np.nan, "se_ms_hac": np.nan, "hac_lags": 0})
            continue
        P = REG.apply(X, S_ct[sel], Y_ct)[:, :3, 3]
        Q = Q_ct(tq)[:, :3, 3]
        v = Q_ct.velocity(tq)
        e = P - Q
        vv = float(np.sum(v * v))
        lag = float(np.sum(e * v) / vv) if vv > 0 else np.nan
        r = e - lag * v
        s2 = float(np.sum(r * r) / max(3 * len(sel) - 1, 1))
        rr = np.linalg.norm(r, axis=1)
        rho = float(np.corrcoef(rr[1:], rr[:-1])[0, 1]) if len(rr) > 3 else 0.0
        infl = np.sqrt((1 + max(rho, 0)) / (1 - min(max(rho, 0), 0.95)))
        hv, L = newey_west_var(np.sum(r * v, axis=1))
        rows.append({"window": w.name, "n": int(len(sel)), "t_centre_s": float(np.mean(tS[sel])),
                     "lag_ms": lag * 1e3,
                     "se_ms_ar1": float(np.sqrt(s2 / vv)) * 1e3 * infl if vv > 0 else np.nan,
                     "se_ms_hac": float(np.sqrt(hv)) / vv * 1e3 if vv > 0 else np.nan,
                     "hac_lags": int(L)})
    return pd.DataFrame(rows)


def window_offsets(tS: np.ndarray, S_ct: np.ndarray, Q_ct: PoseSpline, model: TimeModel,
                   X: np.ndarray, Y_ct: np.ndarray, windows: Sequence[Interval], *,
                   scale: tuple[float, float] = (0.3, 0.15), h_t: float = 2e-4,
                   gate_k: float = 10.0, margin_s: float = 0.0025) -> pd.DataFrame:
    """VD19: per sync window, the clock offset (residual time shift under ``model``) from
    the same whitened 6-DoF CT residuals as the fine fit (position / ``σ_p`` mm, rotation
    / ``σ_R`` deg; ``scale`` = the fit's per-body scale), ``X`` and ``Y`` held fixed:
    one Gauss–Newton step ``δ = −Σ jᵢ·rᵢ / Σ jᵢ·jᵢ`` with ``jᵢ = ∂rᵢ/∂t'`` (central
    difference of the spline, step ``h_t``); frames beyond ``gate_k`` σ are left out.
    ``se_ms_hac``: Newey–West HAC of the scores ``uᵢ = jᵢ·rᵢ`` in time order (bandwidth
    :func:`hac_bandwidth`, ``hac_lags``), ``SE = √(HAC var Σu) / Σ j·j``; ``se_ms_white``:
    the same without autocorrelation (lag 0). Unlike :func:`window_lags` (position only, the
    V6 criteria) this carries the timing information of the rotations, as the fit does.
    Frames are taken in time order across visibility gaps."""
    sp, sr = float(scale[0]), float(np.radians(scale[1]))
    rows = []
    for w in windows:
        sel = np.flatnonzero((tS >= w.t0) & (tS <= w.t1) & se3.valid(S_ct))
        tq = model.to_q(tS[sel])
        ok = Q_ct.covered(tq, margin_s + h_t)
        sel, tq = sel[ok], tq[ok]
        row = {"window": w.name, "n": int(len(sel)), "t_centre_s": np.nan, "lag_ms": np.nan,
               "se_ms_hac": np.nan, "se_ms_white": np.nan, "hac_lags": 0}
        if len(sel) >= 10:
            P = REG.apply(X, S_ct[sel], Y_ct)
            r = REG.pose_residuals(P, Q_ct(tq), sp, sr)
            j = (REG.pose_residuals(P, Q_ct(tq + h_t), sp, sr)
                 - REG.pose_residuals(P, Q_ct(tq - h_t), sp, sr)) / (2 * h_t)
            g = (np.isfinite(r).all(axis=1) & np.isfinite(j).all(axis=1)
                 & (np.linalg.norm(r[:, :3], axis=1) <= gate_k)
                 & (np.linalg.norm(r[:, 3:], axis=1) <= gate_k))
            r, j = r[g], j[g]
            jj = float(np.sum(j * j))
            if len(r) >= 10 and jj > 0:
                d = -float(np.sum(j * r)) / jj
                u = np.sum(j * (r + d * j), axis=1)        # scores at the solution
                hv, L = newey_west_var(u)
                row.update(n=int(len(r)), t_centre_s=float(np.mean(tS[sel][g])),
                           lag_ms=d * 1e3, se_ms_hac=float(np.sqrt(hv)) / jj * 1e3,
                           se_ms_white=float(np.sqrt(np.sum(u * u))) / jj * 1e3,
                           hac_lags=int(L))
        rows.append(row)
    return pd.DataFrame(rows)


def _wls_line(t: np.ndarray, y: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Weighted LS line ``y = a + b t`` with known SEs (no χ² rescaling): ``(coef, cov)``."""
    A = np.c_[np.ones_like(t), t]
    W = 1.0 / se ** 2
    cov = np.linalg.inv((A * W[:, None]).T @ A)
    return cov @ (A * W[:, None]).T @ y, cov


def drift_from_windows(wl: pd.DataFrame, t_ref_s: float, se_col: str = "se_ms_hac"
                       ) -> dict[str, float]:
    """VD19: the clock correction and its SEs from the per-window offsets (independent
    windows; :func:`window_offsets` for the reported clock SEs, :func:`window_lags` for the
    HAC version of the V6 checks) -- weighted LS of ``lag_ms`` on
    ``t_centre − t_ref``. Two windows: ``SE_drift = √(SE₁² + SE₂²)/Δt`` exactly. With three
    or more windows also the delete-one-window jackknife SE of the drift
    (``se_alpha_ppm_jk``; NaN otherwise). Offsets in ms at ``t_ref``, drift in ppm."""
    out = {"n_windows": 0, "d_offset_ms": np.nan, "d_alpha_ppm": np.nan,
           "se_offset_ms_hac": np.nan, "se_alpha_ppm_hac": np.nan, "se_alpha_ppm_jk": np.nan}
    d = wl.dropna(subset=["lag_ms", se_col]) if len(wl) else wl
    d = d[d[se_col] > 0] if len(d) else d
    out["n_windows"] = int(len(d))
    if len(d) == 1:
        out.update(d_offset_ms=float(d["lag_ms"].iloc[0]),
                   se_offset_ms_hac=float(d[se_col].iloc[0]))
        return out
    if len(d) < 2:
        return out
    t = d["t_centre_s"].to_numpy(float) - t_ref_s
    y = d["lag_ms"].to_numpy(float)
    se = d[se_col].to_numpy(float)
    coef, cov = _wls_line(t, y, se)
    out.update(d_offset_ms=float(coef[0]), d_alpha_ppm=float(coef[1]) * 1e3,   # ms/s → ppm
               se_offset_ms_hac=float(np.sqrt(cov[0, 0])),
               se_alpha_ppm_hac=float(np.sqrt(cov[1, 1])) * 1e3)
    K = len(d)
    if K >= 3:
        b = np.array([_wls_line(np.delete(t, i), np.delete(y, i), np.delete(se, i))[0][1]
                      for i in range(K)])
        out["se_alpha_ppm_jk"] = float(np.sqrt((K - 1) / K * np.sum((b - b.mean()) ** 2))) * 1e3
    return out


def sliding_variant(fit: FineFit, line: dict[str, float]) -> FineFit | None:
    """VD19 sensitivity: the time model of the sliding-window lag line (:func:`lag_line`,
    relative to ``fit.model``) -- offset ``c + intercept`` and drift = the line's slope --
    with ``fit``'s registration. ``None`` when the line is undefined."""
    sl, ic = line.get("slope_ppm", np.nan), line.get("intercept_ms", np.nan)
    if not (np.isfinite(sl) and np.isfinite(ic)):
        return None
    m = fit.model
    model = TimeModel(m.offset_s + ic * 1e-3, sl * 1e-6, m.t_ref_s, None)
    return FineFit(model=model, X=fit.X, Y=dict(fit.Y), cov=None, names=[], sigma2=np.nan,
                   se_inflation=fit.se_inflation, n_frames=dict(fit.n_frames), rms_pos_mm={},
                   rms_rot_deg={})


def clock_criteria(window_lag: pd.DataFrame, line_omega: dict[str, float], model: TimeModel,
                   *, tol_ms: float = 1.0, alpha_se_hac_ppm: float | None = None
                   ) -> dict[str, Any]:
    """The pre-registered checks of the primary (CT-only) clock (V6), none of which uses the
    agreement outcome: per-window residual lags within ``tol_ms``; start and end windows
    agree within 2 SE; the frame-free sliding ``|ω|`` slope consistent with α within 2 SE.
    Synthetic recovery is verified by the test suite (tests/test_v_sync.py).

    ``windows_agree``, ``sliding_consistent`` and ``pass`` keep the pre-registered SEs (the
    AR(1)-inflated window SEs, ``window_diff_se_ms_ar1``; the fit-covariance SE of α) and
    drive :func:`choose_clock`. VD19 adds the same checks with the HAC window SEs and the
    drift SE (``alpha_se_hac_ppm``: the reported VD19 drift SE when given, from
    :func:`window_offsets`; else from the window lags' HAC SEs): ``window_diff_se_ms_hac``,
    ``windows_agree_hac``, ``alpha_se_hac_ppm``, ``sliding_consistent_hac``, ``pass_hac``."""
    wl = window_lag.dropna(subset=["lag_ms"])
    out: dict[str, Any] = {"n_windows": int(len(wl))}
    out["window_lag_max_abs_ms"] = float(np.abs(wl["lag_ms"]).max()) if len(wl) else np.nan
    out["windows_within_tol"] = bool(len(wl) and (np.abs(wl["lag_ms"]) <= tol_ms).all())
    if len(wl) >= 2:
        a, b = wl.iloc[0], wl.iloc[-1]
        diff = abs(a["lag_ms"] - b["lag_ms"])
        se = float(np.hypot(a["se_ms_ar1"], b["se_ms_ar1"]))
        se_h = float(np.hypot(a["se_ms_hac"], b["se_ms_hac"]))
        out.update(window_diff_ms=float(diff), window_diff_se_ms_ar1=se,
                   windows_agree=bool(diff <= 2 * se), window_diff_se_ms_hac=se_h,
                   windows_agree_hac=bool(diff <= 2 * se_h))
    else:
        out.update(window_diff_ms=np.nan, window_diff_se_ms_ar1=np.nan, windows_agree=False,
                   window_diff_se_ms_hac=np.nan, windows_agree_hac=False)
    sl, sse = line_omega.get("slope_ppm", np.nan), line_omega.get("slope_se_ppm", np.nan)
    a_ppm = model.alpha * 1e6
    a_se = model.se_alpha * 1e6 if np.isfinite(model.se_alpha) else 0.0
    a_se_h = (float(alpha_se_hac_ppm) if alpha_se_hac_ppm is not None
              else drift_from_windows(window_lag, model.t_ref_s)["se_alpha_ppm_hac"])
    out.update(sliding_slope_ppm=sl, alpha_ppm=a_ppm,
               sliding_consistent=bool(np.isfinite(sl) and np.isfinite(sse)
                                       and abs(sl - a_ppm) <= 2 * np.hypot(sse, a_se)),
               alpha_se_hac_ppm=a_se_h,
               sliding_consistent_hac=bool(np.isfinite(sl) and np.isfinite(sse)
                                           and np.isfinite(a_se_h)
                                           and abs(sl - a_ppm) <= 2 * np.hypot(sse, a_se_h)))
    out["synthetic_recovery"] = "tests/test_v_sync.py"
    out["pass"] = bool(out["windows_within_tol"] and out["windows_agree"]
                       and out["sliding_consistent"])
    out["pass_hac"] = bool(out["windows_within_tol"] and out["windows_agree_hac"]
                           and out["sliding_consistent_hac"])
    return out


def choose_clock(crit: dict[str, dict[str, Any]]) -> str:
    """V6 rule: the CT-only clock stays primary unless it fails a criterion while the
    two-stage clock passes all of them."""
    if crit.get("ct", {}).get("pass", False):
        return "ct"
    if crit.get("two_stage", {}).get("pass", False):
        return "two_stage"
    return "ct"


# -- orchestration ----------------------------------------------------------------------
@dataclass
class SyncResult:
    windows: list[Interval]
    t_ref_s: float
    edges_S: Edges
    edges_Q: Edges
    coarse_lag_s: float
    coarse_votes: int
    edge_fit: EdgeFit | None
    xcorr: list[dict[str, float]]
    xcorr_model: TimeModel | None
    fine: FineFit
    heldout: pd.DataFrame
    sliding: pd.DataFrame
    line: dict[str, dict[str, float]]      # "joint" (|ω|² correlation) and "pose" (registered)
    fine_joint: FineFit | None = None      # sensitivity: the other fit variant (R5b)
    variants: dict[str, FineFit] = field(default_factory=dict)   # clock variants (R5c.3)
    window_lags: pd.DataFrame | None = None                      # per-window residual lag
    speed_ct: tuple[SpeedSeries, SpeedSeries] | None = None   # CT |ω| (JTS-MM, Qualisys)


def synchronise(tS: np.ndarray, S: dict[str, np.ndarray], tQ: np.ndarray,
                Q: dict[str, np.ndarray], windows: Sequence[Interval], *,
                heldout: Sequence[Interval] = (), ct: str = "CT", head: str = "HP",
                jaw: str = "MP", lag_hint_s: float | None = None, hint_range_s: float = 5.0,
                edge_tol_s: float = 0.05, frame_s: float = 0.025,
                xcorr_half_range_s: float = 0.3, sliding_win_s: float = 15.0,
                sliding_step_s: float = 5.0, sigma_pos_mm: float = 0.3,
                sigma_rot_deg: float = 0.15, huber_k: float = 2.5,
                speed_baseline_s: float = 0.1, fit_bodies: str = "ct",
                compare_joint: bool = True, variants: bool = False) -> SyncResult:
    """Full chain: edges → coarse lag (vote) → edge LS → ω cross-correlation per window →
    closed-form ``X``/``Y`` → joint fine fit → held-out residuals → sliding-window lags.

    ``tS`` are JTS-MM exposure-centre times (master s), ``S``/``Q`` map the shared body names
    (Qualisys naming: CT, MP, HP) to (N, 4, 4) poses in mm of each system.

    ``fit_bodies="ct"`` (default, R5b): the time model and ``X`` come from the CT alone; each
    other body's ``Y`` is then the robust closed form on the sync windows at that model
    (:func:`registration.y_given_x`). The head and jaw bodies are unsafe in the sync windows
    (the CT moves next to them: label swaps, occlusion; errors of several degrees in the
    real takes), so they must not steer the clock. ``"all"``: the joint fit of all bodies.
    With ``compare_joint`` the other variant is fitted too and kept as ``fine_joint``.
    """
    if fit_bodies not in ("ct", "all"):
        raise ValueError(f"fit_bodies must be 'ct' or 'all', got {fit_bodies!r}")
    windows = list(windows)
    if not windows:
        raise ValueError("no sync windows")
    tS = np.asarray(tS, dtype=np.float64)
    tQ = np.asarray(tQ, dtype=np.float64)
    okS = se3.valid(S[ct])
    in_win = np.zeros(len(tS), bool)
    for w in windows:
        in_win |= (tS >= w.t0) & (tS <= w.t1)
    t_ref = float(np.mean(tS[in_win & okS])) if (in_win & okS).any() else float(
        np.mean([0.5 * (w.t0 + w.t1) for w in windows]))

    # 1. edges
    eS = visibility_edges(tS, okS, max_bracket_s=1.5 * frame_s).within(windows)
    eQ = visibility_edges(tQ, se3.valid(Q[ct]), max_bracket_s=1.5 * float(
        np.median(np.diff(tQ))))
    if lag_hint_s is not None:
        lag0, votes = coarse_offset_from_edges(eS, eQ, center=lag_hint_s,
                                               half_range=hint_range_s, tol=edge_tol_s)
    else:
        lag0, votes = coarse_offset_from_edges(eS, eQ, tol=edge_tol_s)
    try:
        ef: EdgeFit | None = fit_edges(eS, eQ, lag0, t_ref, tol=max(2 * edge_tol_s, 0.08))
        model_e = ef.model
    except ValueError:
        ef = None
        model_e = TimeModel(lag0, 0.0, t_ref)

    # 2. ω cross-correlation per window (CT)
    gapS = gapQ = 3.5 * frame_s          # grid gaps: same rule for both systems
    spS = {b: speed_series(tS, S[b], gapS, speed_baseline_s) for b in S}
    spQ = {b: speed_series(tQ, Q[b], gapQ, speed_baseline_s) for b in Q}
    xc = []
    for w in windows:
        r = xcorr_lag(spS[ct], spQ[ct], w.t0, w.t1, float(model_e.lag(0.5 * (w.t0 + w.t1))),
                      xcorr_half_range_s, curve=True)
        r["window"] = w.name
        xc.append(r)
    xmodel = None
    good = [r for r in xc if np.isfinite(r["lag_s"]) and not r["at_edge"]]
    if len(good) >= 2:
        tc = np.array([r["t_centre_s"] for r in good])
        lg = np.array([r["lag_s"] for r in good])
        A = np.c_[np.ones_like(tc), tc - t_ref]
        sol, *_ = np.linalg.lstsq(A, lg, rcond=None)
        xmodel = TimeModel(float(sol[0]), float(sol[1]), t_ref)
    elif len(good) == 1:
        xmodel = TimeModel(float(good[0]["lag_s"] - model_e.alpha * (good[0]["t_centre_s"]
                                                                        - t_ref)),
                           model_e.alpha, t_ref)

    # 3. initial registration at the best coarse model, then the joint fine fit
    m0 = xmodel if xmodel is not None else model_e
    Qspl = {b: PoseSpline(tQ, Q[b]) for b in Q if b in S}
    sel = np.flatnonzero(in_win & okS)
    A_ct = Qspl[ct](m0.to_q(tS[sel]))
    X0, Yct = REG.init_xy(A_ct, S[ct][sel])
    Y0 = {ct: Yct}
    for b in Qspl:
        if b == ct:
            continue
        sb = np.flatnonzero(in_win & se3.valid(S[b]))
        Y0[b] = REG.y_given_x(Qspl[b](m0.to_q(tS[sb])), S[b][sb], X0)
    kw = dict(sigma_pos_mm=sigma_pos_mm, sigma_rot_deg=sigma_rot_deg, huber_k=huber_k)

    def joint() -> FineFit:
        return fit_fine(tS, {b: S[b] for b in Qspl}, Qspl, windows, m0, X0, Y0, **kw)

    def ct_only() -> FineFit:
        f = fit_fine(tS, {ct: S[ct]}, {ct: Qspl[ct]}, windows, m0, X0, {ct: Yct}, **kw)
        for b in Qspl:
            if b != ct:
                sb = np.flatnonzero(in_win & se3.valid(S[b]))
                f.Y[b] = REG.y_given_x(Qspl[b](f.model.to_q(tS[sb])), S[b][sb], f.X)
        return f

    fine = ct_only() if fit_bodies == "ct" else joint()
    other = None
    var_fits: dict[str, FineFit] = {}
    if compare_joint and len(Qspl) > 1:
        other = joint() if fit_bodies == "ct" else ct_only()
        if variants:                            # clock variants (R5c.3; the stage asks)
            f_ct = fine if fit_bodies == "ct" else other
            var_fits = clock_variants(tS, S, Qspl, windows, in_win, f_ct, fit_joint=(
                other if fit_bodies == "ct" else fine), ct=ct, **kw)

    # 4. validation
    held = evaluate_intervals(tS, {b: S[b] for b in Qspl}, Qspl, fine.model, fine.X, fine.Y,
                              heldout, head=head, jaw=jaw)
    sb = [b for b in (jaw, head) if b in spS and b in spQ]
    sl = sliding_lags({b: spS[b] for b in sb}, {b: spQ[b] for b in sb}, fine.model,
                      float(tS[0]), float(tS[-1]), win_s=sliding_win_s, step_s=sliding_step_s)
    sp = sliding_pose_lags(tS, S, Qspl, fine.model, fine.X, fine.Y, sb, float(tS[0]),
                           float(tS[-1]), win_s=sliding_win_s, step_s=sliding_step_s)
    sl = pd.concat([sl, sp], ignore_index=True)
    line = {m: lag_line(sl, fine.model, body=m) for m in ("joint", "pose")}
    return SyncResult(windows=windows, t_ref_s=t_ref, edges_S=eS, edges_Q=eQ,
                      coarse_lag_s=lag0, coarse_votes=votes, edge_fit=ef, xcorr=xc,
                      xcorr_model=xmodel, fine=fine, heldout=held, sliding=sl, line=line,
                      fine_joint=other, speed_ct=(spS[ct], spQ[ct]), variants=var_fits,
                      window_lags=window_lags(tS, S[ct], Qspl[ct], fine.model, fine.X,
                                              fine.Y[ct], windows))
