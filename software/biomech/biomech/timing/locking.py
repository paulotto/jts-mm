# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Locking and location of per-cycle EMG leads against their surrogates (C2-E, X6 P1-P3).

Pure functions on arrays (ms; lead > 0 = EMG before the mechanical event):

* :func:`iqr` / :func:`locking` -- P1 locking (A): the spread (IQR) of the usable per-cycle
  leads against the same statistic of every S1 surrogate (:mod:`biomech.timing.surrogates`):
  a time-locked EMG gives a smaller spread than chance. Per participant the Monte-Carlo p and
  log(S_obs / median S_null) (the cohort tests it < 0, :mod:`biomech.stats.families`).
* :func:`mixture_location` -- P1 location (B): the per-cycle leads as a mixture of a locked
  component, a normal N(mu, sigma) truncated to the search window, and the background, the
  density of the S1 surrogate leads (the same window, detector and gates applied to EMG
  unrelated to the motion). mu is the location of the locked bursts, free of the window bias
  that makes the median of a random burst ~ +250 ms (D34).
* :func:`theil_sen` / :func:`s2_slope` -- P3 cycle structure: the Theil-Sen slope b of
  (t_EMG - a) on (t_mech - a) (a = the S2 anchor) and its S2 null (offsets permuted across
  cycles); b ~ 1: locked to the upcoming onset, b ~ 0: locked to the anchor (the rhythm).
* :func:`s2_iqr` -- P2: the locking IQR under the S2 cycle shuffle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from biomech.timing import surrogates


def iqr(x: np.ndarray, min_n: int = 1) -> float:
    """Interquartile range of the finite values (NaN below ``min_n`` values)."""
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    if v.size < max(min_n, 1):
        return np.nan
    q1, q3 = np.percentile(v, [25, 75])
    return float(q3 - q1)


def locking(obs: np.ndarray, null: list[np.ndarray] | np.ndarray, min_n: int = 5
            ) -> dict[str, float]:
    """P1 locking of the usable leads ``obs`` against the surrogate leads ``null`` (one array
    per surrogate). ``iqr_obs``, ``iqr_null_median`` (over surrogates with >= ``min_n``
    usable leads), ``log_ratio`` = log(iqr_obs / iqr_null_median) (< 0: locked), ``p_lock`` =
    Monte-Carlo p of a null IQR <= the observed, ``n_surr`` (surrogates with a value)."""
    s_obs = iqr(obs, min_n)
    s_null = np.array([iqr(v, min_n) for v in null], float)
    fin = s_null[np.isfinite(s_null)]
    med = float(np.median(fin)) if fin.size else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = float(np.log(s_obs / med)) if s_obs > 0 and med > 0 else np.nan
    return {"iqr_obs": s_obs, "iqr_null_median": med, "log_ratio": lr,
            "p_lock": surrogates.mc_p(fin, s_obs, "less"), "n_surr": int(fin.size)}


@dataclass(frozen=True)
class MixtureParams:
    """Bounds and grid of :func:`mixture_location` (``stages.timing.c2e.mixture``)."""

    sigma_ms: tuple[float, float] = (5.0, 150.0)   # locked-component SD bounds
    pi_min: float = 0.1                            # smaller locked share: no location
    bandwidth_ms: float = 20.0                     # Gaussian KDE of the surrogate leads
    floor: float = 0.02                            # uniform share mixed into the background
    grid_ms: float = 5.0                           # start grid of mu

    @classmethod
    def from_dict(cls, d: dict | None) -> "MixtureParams":
        d = dict(d or {})
        if "sigma_ms" in d:
            d["sigma_ms"] = tuple(float(v) for v in d["sigma_ms"])
        return cls(**d)


def background_density(null_leads: np.ndarray, lo: float, hi: float, p: MixtureParams
                       ) -> tuple[np.ndarray, np.ndarray]:
    """(grid ms, density) of the surrogate leads on [lo, hi]: Gaussian KDE (bandwidth
    ``bandwidth_ms``, reflected at both window edges) mixed with ``floor`` of a uniform,
    normalised to 1 on the window. Uniform when there are no surrogate leads."""
    g = np.arange(lo, hi + 0.5, 1.0)
    v = np.asarray(null_leads, float)
    v = v[np.isfinite(v) & (v >= lo) & (v <= hi)]
    uni = np.full(g.size, 1.0 / (hi - lo))
    if v.size < 2:
        return g, uni
    # Histogram of the leads reflected at both edges on a 1-ms grid, then a Gaussian kernel
    # (O(grid) instead of O(n x grid)); the window part is kept.
    pts = np.concatenate([v, 2 * lo - v, 2 * hi - v])
    h = p.bandwidth_ms
    half = int(np.ceil(4 * h))
    cnt, _ = np.histogram(pts, np.arange(lo - half - 0.5, hi + half + 1.0, 1.0))
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / h) ** 2)
    f = np.convolve(cnt, k / k.sum(), mode="same")[half:half + g.size]
    f = f / f.sum() if f.sum() > 0 else uni
    f = (1 - p.floor) * f + p.floor * uni / uni.sum()
    return g, f / f.sum()


_SQRT2PI = np.sqrt(2 * np.pi)


def _tn_pdf(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Density (per ms) of N(mu, sd) truncated to [lo, hi]."""
    z = special.ndtr((hi - mu) / sd) - special.ndtr((lo - mu) / sd)
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * _SQRT2PI) / np.maximum(z, 1e-300)


def mixture_location(leads: np.ndarray, null_leads: np.ndarray, lo: float, hi: float,
                     p: MixtureParams = MixtureParams(), min_n: int = 5) -> dict[str, float]:
    """P1 location (B): maximum-likelihood fit of
    f(l) = pi TN(l; mu, sigma, [lo, hi]) + (1 - pi) f_bg(l)
    to the usable leads (ms), f_bg = :func:`background_density` of the surrogate leads.
    Start: the best point of a grid (mu every ``grid_ms``, 8 log-spaced sigma, pi by EM
    steps); then L-BFGS-B within the bounds. Returns ``mu_ms``, ``sigma_ms``,
    ``pi``, ``loglik_gain`` (log-likelihood over the background alone) and ``reason``
    ("" when fitted; ``few_leads`` below ``min_n``; ``no_locked_component`` when pi <
    ``pi_min``: mu is then NaN)."""
    v = np.asarray(leads, float)
    v = v[np.isfinite(v) & (v >= lo) & (v <= hi)]
    out = {"mu_ms": np.nan, "sigma_ms": np.nan, "pi": np.nan, "loglik_gain": np.nan,
           "reason": ""}
    if v.size < min_n:
        out["reason"] = "few_leads"
        return out
    g, f = background_density(null_leads, lo, hi, p)
    fb = np.interp(v, g, f)                       # density per ms
    s_lo, s_hi = p.sigma_ms

    mus = np.arange(lo, hi + 1e-9, p.grid_ms)
    sds = np.geomspace(s_lo, s_hi, 8)
    # Start grid: pi per (mu, sigma) by EM fixed-point steps (the background is fixed), then
    # L-BFGS-B from the best grid point (kept if the refinement does not improve on it).
    M, S = np.meshgrid(mus, sds, indexing="ij")
    lk = _tn_pdf(v[None, None, :], M[..., None], S[..., None], lo, hi)
    q = np.full(M.shape, 0.5)
    for _ in range(25):
        r = q[..., None] * lk / (q[..., None] * lk + (1 - q[..., None]) * fb)
        q = r.mean(axis=-1)
    ll = np.sum(np.log(q[..., None] * lk + (1 - q[..., None]) * fb), axis=-1)
    i, j = np.unravel_index(int(np.argmax(ll)), ll.shape)
    x0 = np.array([M[i, j], np.log(S[i, j]), np.clip(q[i, j], 1e-3, 1 - 1e-3)])

    def nll(z: np.ndarray) -> float:
        mu, sd, w = z[0], np.exp(z[1]), z[2]
        return -float(np.sum(np.log(w * _tn_pdf(v, mu, sd, lo, hi) + (1 - w) * fb)))

    res = optimize.minimize(nll, x0, method="L-BFGS-B",
                            bounds=[(lo, hi), (np.log(s_lo), np.log(s_hi)), (0.0, 1.0)])
    fun, z = (res.fun, res.x) if res.fun <= nll(x0) else (nll(x0), x0)
    ll0 = float(np.sum(np.log(fb)))
    out.update(sigma_ms=float(np.exp(z[1])), pi=float(z[2]), loglik_gain=float(-fun - ll0))
    if z[2] < p.pi_min:
        out["reason"] = "no_locked_component"
    else:
        out["mu_ms"] = float(z[0])
    return out


def theil_sen(x: np.ndarray, y: np.ndarray) -> float:
    """Theil-Sen slope (median of the pairwise slopes) of the finite pairs; NaN below 3."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return np.nan
    i, j = np.triu_indices(x.size, 1)
    dx = x[j] - x[i]
    keep = dx != 0
    return float(np.median((y[j] - y[i])[keep] / dx[keep])) if keep.any() else np.nan


def s2_slope(anchor: np.ndarray, mech: np.ndarray, event: np.ndarray,
             rng: np.random.Generator, n: int) -> dict[str, float]:
    """P3 cycle structure: Theil-Sen slope b of (event - anchor) on (mech - anchor) (µs in,
    unitless out) against its S2 null (the offsets event - anchor permuted across cycles;
    :func:`biomech.timing.surrogates.s2_permutations`). ``b``, ``b_null_median``,
    ``p_s2`` (Monte-Carlo p of a null slope >= b), ``n_cycles`` and ``x_sd_ms`` (SD of
    mech - anchor: with little variation S2 has no power, X6 P3)."""
    a, m, e = (np.asarray(v, float) for v in (anchor, mech, event))
    ok = np.isfinite(a) & np.isfinite(m) & np.isfinite(e)
    a, m, e = a[ok], m[ok], e[ok]
    out = {"b": np.nan, "b_null_median": np.nan, "p_s2": np.nan, "n_cycles": int(a.size),
           "x_sd_ms": float(np.std((m - a) / 1e3, ddof=1)) if a.size > 1 else np.nan}
    if a.size < 3:
        return out
    x, y = m - a, e - a
    out["b"] = theil_sen(x, y)
    perms = surrogates.s2_permutations(rng, a.size, n)
    i, j = np.triu_indices(a.size, 1)
    dx = x[j] - x[i]
    keep = dx != 0
    yp = y[perms]                                         # (n, cycles)
    slopes = (yp[:, j] - yp[:, i])[:, keep] / dx[keep]
    null = np.median(slopes, axis=1) if keep.any() else np.full(n, np.nan)
    out["b_null_median"] = float(np.nanmedian(null)) if np.isfinite(null).any() else np.nan
    out["p_s2"] = surrogates.mc_p(null, out["b"], "greater")
    return out


def s2_iqr(anchor: np.ndarray, mech: np.ndarray, onset: np.ndarray, usable: np.ndarray,
           rng: np.random.Generator, n: int, min_n: int = 5) -> dict[str, float]:
    """P2 locking against S2: per cycle k (rows) and channel (columns) the EMG onset
    ``onset`` (µs) with its ``usable`` flag, the cycle's ``anchor`` and mechanical onset
    ``mech`` (µs, per cycle). Under S2 the cycles' onset offsets from their anchors (all
    channels of a cycle together) are permuted; the lead = mech - (anchor + offset). Returns
    ``iqr_obs``, ``iqr_s2_median`` and ``p_s2`` (Monte-Carlo p of an S2 IQR <= observed)."""
    a = np.asarray(anchor, float)
    m = np.asarray(mech, float)
    on = np.asarray(onset, float)
    u = np.asarray(usable, bool) & np.isfinite(on)
    ok = np.isfinite(a) & np.isfinite(m)
    a, m, on, u = a[ok], m[ok], on[ok], u[ok]
    lead = (m[:, None] - on) / 1e3
    s_obs = iqr(lead[u], min_n)
    out = {"iqr_obs": s_obs, "iqr_s2_median": np.nan, "p_s2": np.nan}
    if a.size < 3 or not np.isfinite(s_obs):
        return out
    off = on - a[:, None]
    perms = surrogates.s2_permutations(rng, a.size, n)
    null = np.array([iqr(((m - a)[:, None] - off[pm]) [u[pm]] / 1e3, min_n) for pm in perms])
    fin = null[np.isfinite(null)]
    out["iqr_s2_median"] = float(np.median(fin)) if fin.size else np.nan
    out["p_s2"] = surrogates.mc_p(fin, s_obs, "less")
    return out
