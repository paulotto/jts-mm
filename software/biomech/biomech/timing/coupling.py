# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Lagged correlation and the continuous EMG-to-kinematics coupling (C2-E, X6 P5).

* :func:`lagged_corr` -- the one lagged-correlation helper: normalised correlation of
  ``a[i]`` with ``b[i + L]`` over the sample pairs where both are finite, for every lag
  L in -max_lag ... +max_lag (the whole curve). Used by the IMU lag of C3a
  (``imu.analysis._lag_ms``, uncentred, on derivatives) and by P5 (centred).
* :func:`coupling` -- P5: rectified closer EMG and -alpha through the *same* zero-phase
  low-pass on the same grid (identical |H|^2, so the relative lag is unbiased), alpha masked
  around occlusion, both cycle-residualised (the mean waveform over cycle phase subtracted),
  then the lag of the peak correlation over +-``max_lag_ms`` (> 0: EMG first, i.e. the
  acceleration follows the EMG -- the electromechanical delay). Its S1 null: the EMG feature
  circularly shifted (:mod:`biomech.timing.surrogates`) gives the null of the peak
  correlation. Raw (non-residualised) lags are reported too; they are pre-stated as not
  interpretable as onset order (X6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from biomech import timebase
from biomech.timing import surrogates


def lagged_corr(a: np.ndarray, b: np.ndarray, max_lag: int, min_pairs: int = 50,
                center: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(lags in samples, correlation) of ``a[i]`` with ``b[i + L]`` (L > 0: ``b`` late).

    Per lag only the pairs where both are finite count (gaps are never zero-filled); a lag
    with fewer than ``min_pairs`` pairs, or zero energy, is NaN. ``center`` False: the
    uncentred normalised correlation sum(a b) / sqrt(sum a^2 sum b^2) (the C3a IMU lag);
    True: Pearson (each lag's pair means removed)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    lags = np.arange(-max_lag, max_lag + 1)
    n = len(a)
    c = np.full(lags.size, np.nan)
    for j, L in enumerate(lags):
        aa = a[max(0, -L): n - max(0, L)]
        bb = b[max(0, L): n - max(0, -L)]
        v = np.isfinite(aa) & np.isfinite(bb)
        if v.sum() < min_pairs:
            continue
        av, bv = aa[v], bb[v]
        if center:
            av, bv = av - av.mean(), bv - bv.mean()
        den = np.sqrt(np.dot(av, av) * np.dot(bv, bv))
        if den > 0:
            c[j] = np.dot(av, bv) / den
    return lags, c


def peak_lag(lags: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """(lag in samples, correlation) at the maximum; NaN when the curve has no value."""
    if not np.isfinite(c).any():
        return np.nan, np.nan
    j = int(np.nanargmax(c))
    return float(lags[j]), float(c[j])


@dataclass(frozen=True)
class CouplingParams:
    """``stages.timing.c2e.coupling``."""

    smooth_s: float = 0.005          # EMG rectification: centred moving mean before sampling
    occlusion_mask_s: float = 0.05   # alpha masked +- this around each occlusion (cycle end)
    max_lag_ms: float = 150.0
    n_phase_bins: int = 50           # cycle-phase bins of the residualisation
    min_pairs: int = 200
    n_null: int = 200
    gap_s: float = 1.0               # S1 shift range [gap, T - gap]

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CouplingParams":
        return cls(**dict(d or {}))


def emg_feature(t_emg_us: np.ndarray, x: np.ndarray, t_grid_us: np.ndarray, fs: float,
                smooth_s: float) -> np.ndarray:
    """Rectified EMG (mean over the given channels, NaN = masked) on the grid: a centred
    moving mean of ``smooth_s`` (mask-aware, as the envelope) sampled at the grid times
    (nearest sample); NaN where the moving window has < 50 % valid samples."""
    from scipy.ndimage import uniform_filter1d

    r = np.abs(np.asarray(x, float))
    r = r.reshape(len(r), -1)
    valid = np.isfinite(r)
    w = timebase.odd_window(smooth_s, fs)
    num = uniform_filter1d(np.where(valid, r, 0.0), w, axis=0, mode="nearest")
    den = uniform_filter1d(valid.astype(float), w, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        m = num / den
    m[den < 0.5] = np.nan
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        f = np.nanmean(m, axis=1)
    j = np.clip(np.searchsorted(t_emg_us, t_grid_us), 1, len(t_emg_us) - 1)
    j = np.where(np.abs(t_emg_us[j - 1] - t_grid_us) <= np.abs(t_emg_us[j] - t_grid_us),
                 j - 1, j)
    out = f[j]
    out[(t_grid_us < t_emg_us[0]) | (t_grid_us > t_emg_us[-1])] = np.nan
    return out


def lowpass(t_us: np.ndarray, x: np.ndarray, cutoff_hz: float, order: int) -> np.ndarray:
    """Zero-phase Butterworth low-pass per finite segment (shorter than the padding: NaN)."""
    from scipy.signal import butter, sosfiltfilt

    fs = 1e6 / float(np.median(np.diff(t_us)))
    sos = butter(order, cutoff_hz, fs=fs, output="sos")
    pad = 3 * (2 * len(sos) + 1)
    out = np.full(len(x), np.nan)
    for a, b in timebase.segments(np.isfinite(x)):
        if b - a > pad:
            out[a:b] = sosfiltfilt(sos, x[a:b])
    return out


def residualise(t_us: np.ndarray, x: np.ndarray, starts_us: np.ndarray, ends_us: np.ndarray,
                n_bins: int) -> np.ndarray:
    """``x`` minus its mean waveform over cycle phase: phase = (t - start) / (end - start)
    in [0, 1) per cycle; the mean over all cycles in ``n_bins`` phase bins, linearly
    interpolated between the bin centres (a piecewise-constant mean would leave a
    within-bin sawtooth that is common to every cycle and distorts the lag); NaN outside
    the cycles."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    ph = np.full(x.shape, np.nan)
    for s, e in zip(starts_us, ends_us):
        m = (t_us >= s) & (t_us < e)
        if e > s and m.any():
            ph[m] = (t_us[m] - s) / (e - s)
    ok = np.isfinite(ph) & np.isfinite(x)
    if not ok.any():
        return out
    b = np.minimum((ph[ok] * n_bins).astype(np.int64), n_bins - 1)
    sums = np.bincount(b, weights=x[ok], minlength=n_bins)
    cnt = np.bincount(b, minlength=n_bins)
    have = cnt > 0
    if not have.any():
        return out
    centre = (np.arange(n_bins) + 0.5) / n_bins
    mean = np.interp(ph[ok], centre[have], sums[have] / cnt[have])
    out[ok] = x[ok] - mean
    return out


def coupling(t_us: np.ndarray, emg: np.ndarray, alpha: np.ndarray, starts_us: np.ndarray,
             ends_us: np.ndarray, cutoff_hz: float, order: int, p: CouplingParams,
             rng: np.random.Generator, occlusions_us: np.ndarray | None = None
             ) -> dict[str, Any]:
    """P5 of one task on the IMU grid ``t_us``: ``emg`` = :func:`emg_feature` (rectified
    closers), ``alpha`` = the *unfiltered* second derivative of the opening angle (deg/s²);
    cycles [``starts_us``, ``ends_us``) (the residualisation phase); ``occlusions_us`` (default
    the cycle ends) are the occlusion times. Both go through the same
    :func:`lowpass` (``cutoff_hz``, ``order``); -alpha is masked +-``occlusion_mask_s``
    around each occlusion. Returns ``emd_lag_ms`` / ``emd_r`` (residualised), ``emd_lag_raw_ms``
    / ``emd_r_raw``, ``emd_r_null_median`` and ``p_coupling`` (Monte-Carlo p of the S1 peak
    correlation >= the observed, residualised), ``n_pairs`` and ``reason``."""
    out = {"emd_lag_ms": np.nan, "emd_r": np.nan, "emd_lag_raw_ms": np.nan,
           "emd_r_raw": np.nan, "emd_r_null_median": np.nan, "p_coupling": np.nan,
           "n_pairs": 0, "reason": ""}
    t_us = np.asarray(t_us)
    if t_us.size < 4 or len(starts_us) < 3:
        out["reason"] = "few_cycles"
        return out
    dt_ms = float(np.median(np.diff(t_us))) / 1e3
    e = lowpass(t_us, np.asarray(emg, float), cutoff_hz, order)
    a = -np.asarray(alpha, float)
    for t_occ in (ends_us if occlusions_us is None else occlusions_us):
        a[np.abs(t_us - t_occ) <= p.occlusion_mask_s * 1e6] = np.nan
    # The filter runs per finite segment: the masked samples bound the segments and stay NaN.
    a = lowpass(t_us, a, cutoff_hz, order)
    k = int(round(p.max_lag_ms / dt_ms))
    er = residualise(t_us, e, starts_us, ends_us, p.n_phase_bins)
    ar = residualise(t_us, a, starts_us, ends_us, p.n_phase_bins)
    lags, c = lagged_corr(er, ar, k, p.min_pairs, center=True)
    lag, r = peak_lag(lags, c)
    lags_r, c_r = lagged_corr(e, a, k, p.min_pairs, center=True)
    lag_r, r_r = peak_lag(lags_r, c_r)
    out.update(emd_lag_ms=lag * dt_ms, emd_r=r, emd_lag_raw_ms=lag_r * dt_ms, emd_r_raw=r_r,
               n_pairs=int((np.isfinite(er) & np.isfinite(ar)).sum()))
    if not np.isfinite(r):
        out["reason"] = "no_pairs"
        return out
    shifts = surrogates.s1_sample_shifts(rng, len(t_us), 1e3 / dt_ms, p.n_null, p.gap_s)
    null = np.full(shifts.size, np.nan)
    for j, s in enumerate(shifts):
        es = residualise(t_us, np.roll(e, int(s)), starts_us, ends_us, p.n_phase_bins)
        null[j] = peak_lag(*lagged_corr(es, ar, k, p.min_pairs, center=True))[1]
    fin = null[np.isfinite(null)]
    out["emd_r_null_median"] = float(np.median(fin)) if fin.size else np.nan
    out["p_coupling"] = surrogates.mc_p(fin, r, "greater")
    return out
