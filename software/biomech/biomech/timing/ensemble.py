# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Detector-free EMG timing from event-aligned ensembles (X2 (change log), exploratory).

Single-cycle onset detection (``timing.lead``) needs a burst that clears a rest
threshold inside a fixed window. In self-paced opening the in-task baseline sits well
above rest and the opener burst often starts after the window, so most channel-cycles
are unusable and the usable ones over-represent early bursts. Here every cycle counts:

* the envelope (50 ms RMS over the unmasked samples, NaN where fewer than half of the
  window is valid; dB re the channel's rest median) is
  aligned to each cycle's mechanical onset and averaged over the muscle group;
* the participant's median curve over cycles gives two times relative to that onset:
  **agonist rise** (openers for opening, closers for closing): the last time before the
  post-onset peak at which the curve is below ``base + rise_frac x (peak - base)``;
  **antagonist release** (closers for opening, openers for closing): the first time the
  curve falls below ``pre - release_frac x (pre - post minimum)``;
* a change smaller than ``min_change_db`` is "no event" (NaN); 95 % CIs bootstrap the
  cycles.

``lead_ms = -t_ms``, as in ``timing.lead`` (> 0: the EMG event precedes the motion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from biomech.emg.artifacts import masked_envelope
from biomech.io.session import CLOSERS, OPENERS

GROUPS = {"opening": (OPENERS, CLOSERS), "closing": (CLOSERS, OPENERS)}   # agonist, antagonist


@dataclass(frozen=True)
class EnsembleParams:
    window_ms: tuple[float, float] = (-1000.0, 1000.0)
    step_ms: float = 5.0
    envelope_s: float = 0.05
    base_ms: tuple[float, float] = (-600.0, -300.0)
    peak_ms: tuple[float, float] = (-300.0, 800.0)
    release_ms: tuple[float, float] = (-200.0, 300.0)
    rise_frac: float = 0.25
    release_frac: float = 0.5
    min_change_db: float = 3.0
    n_boot: int = 500
    # The release search starts this long before the end of the base window (ms);
    # formerly a hard-coded -100 (review).
    release_lookback_ms: float = 100.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "EnsembleParams":
        d = dict(d or {})
        for k in ("window_ms", "base_ms", "peak_ms", "release_ms"):
            if k in d:
                d[k] = tuple(float(v) for v in d[k])
        return cls(**d)


def grid_ms(p: EnsembleParams) -> np.ndarray:
    return np.arange(p.window_ms[0], p.window_ms[1] + 1e-9, p.step_ms)


def aligned_db(t_us: np.ndarray, env: np.ndarray, rest_level: np.ndarray, events_us: np.ndarray,
               g: np.ndarray) -> np.ndarray:
    """(cycles, grid, channels) envelope in dB re rest around each event (NaN off-record)."""
    t_us = np.asarray(t_us)
    out = np.full((len(events_us), g.size, env.shape[1]), np.nan)
    if t_us.size == 0:
        return out
    # Nearest sample within half a grid step (timestamp jitter); longer gaps stay NaN.
    tol = 0.5 * (g[1] - g[0]) * 1e3 if g.size > 1 else np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        db = 20.0 * np.log10(np.maximum(env, 1e-9) / rest_level[None, :])
    for k, t0 in enumerate(events_us):
        tq = t0 + g * 1e3
        j = np.searchsorted(t_us, tq).clip(1, t_us.size - 1)
        i = np.where(np.abs(t_us[j - 1] - tq) <= np.abs(t_us[j] - tq), j - 1, j)
        ok = np.abs(t_us[i] - tq) <= tol
        out[k, ok] = db[i[ok]]
    return out


def rise_time(curve: np.ndarray, g: np.ndarray, p: EnsembleParams) -> tuple[float, float]:
    """(t_ms, change_db) of the agonist rise; NaN when the rise is below ``min_change_db``."""
    bsel = (g >= p.base_ms[0]) & (g <= p.base_ms[1])
    win = np.flatnonzero((g > p.peak_ms[0]) & (g < p.peak_ms[1]))
    if not np.isfinite(curve[bsel]).any() or not np.isfinite(curve[win]).any():
        return np.nan, np.nan
    base = float(np.nanmedian(curve[bsel]))
    pk = int(win[np.nanargmax(curve[win])])
    change = float(curve[pk] - base)
    if change < p.min_change_db:
        return np.nan, change
    below = np.flatnonzero(curve[:pk] < base + p.rise_frac * change)
    below = below[g[below] >= p.base_ms[0]]
    return (float(g[below[-1] + 1]) if below.size else np.nan), change


def release_time(curve: np.ndarray, g: np.ndarray, p: EnsembleParams) -> tuple[float, float]:
    """(t_ms, change_db) of the antagonist release (a drop); NaN below ``min_change_db``."""
    bsel = (g >= p.base_ms[0]) & (g <= p.base_ms[1])
    psel = (g >= p.release_ms[0]) & (g <= p.release_ms[1])
    if not np.isfinite(curve[bsel]).any() or not np.isfinite(curve[psel]).any():
        return np.nan, np.nan
    pre = float(np.nanmedian(curve[bsel]))
    change = pre - float(np.nanmin(curve[psel]))
    if change < p.min_change_db:
        return np.nan, -change
    hit = np.flatnonzero((g >= p.base_ms[1] - p.release_lookback_ms)
                         & (curve < pre - p.release_frac * change))
    return (float(g[hit[0]]) if hit.size else np.nan), -change


def _group_curve(a: np.ndarray, names: list[str], group: tuple[str, ...]) -> np.ndarray | None:
    idx = [names.index(m) for m in group if m in names]
    if not idx:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a[:, :, idx], axis=2)                       # cycles x grid


BOOT_CHUNK = 64              # resamples per batch (bounds the (chunk, cycles, grid) copy)


def _nanmedian_lanes(Y: np.ndarray) -> np.ndarray:
    """``np.nanmedian(Y, axis=1)`` for (batch, n, grid), by one sort: the mean of the two
    middle non-NaN values (NaN for an all-NaN lane), as numpy computes it for n < 600.

    Equal values have equal bits except +0/-0, whose order after a sort is not defined;
    lanes whose middle values are zero are therefore taken from ``np.nanmedian`` itself.
    """
    s = np.sort(Y, axis=1)                                   # NaN sorts last
    k = (~np.isnan(s)).sum(axis=1)                           # non-NaN count per lane
    h = k // 2
    lo = np.where(k % 2 == 1, h, h - 1).clip(0)
    a = np.take_along_axis(s, lo[:, None, :], axis=1)[:, 0]
    b = np.take_along_axis(s, h.clip(max=Y.shape[1] - 1)[:, None, :], axis=1)[:, 0]
    out = (a + b) / 2.0
    out[k == 0] = np.nan
    zero = ((a == 0) | (b == 0)) & (k > 0)
    for r in np.flatnonzero(zero.any(axis=1)):
        out[r] = np.nanmedian(Y[r], axis=0)
    return out


def _boot_medians(y: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """(n_boot, grid) median curves of cycle resamples of ``y`` (cycles x grid).

    One ``rng.integers`` call for all resamples draws the same numbers, and leaves the
    generator in the same state, as one call per resample (numpy ``Generator``), and
    :func:`_nanmedian_lanes` equals ``np.nanmedian(.., axis=0)`` of each resample: identical
    to the former per-resample loop (test_r5_perf_vectorized).
    """
    n = len(y)
    idx = rng.integers(0, n, (n_boot, n))
    if n >= 600:                    # numpy switches median algorithms there: keep its own
        parts = [np.nanmedian(y[i], axis=0)[None] for i in idx]
    else:
        parts = [_nanmedian_lanes(y[idx[a:a + BOOT_CHUNK]])
                 for a in range(0, n_boot, BOOT_CHUNK)]
    return np.concatenate(parts) if parts else np.empty((0, y.shape[1]), y.dtype)


def ensemble_events(y: np.ndarray, g: np.ndarray, kind: str, p: EnsembleParams,
                    rng: np.random.Generator) -> dict[str, float]:
    """Median-curve event time with a bootstrap CI over cycles."""
    import warnings

    fn = rise_time if kind == "rise" else release_time
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        t, change = fn(np.nanmedian(y, axis=0), g, p)
        boots = np.array([fn(m, g, p)[0] for m in _boot_medians(y, p.n_boot, rng)]) \
            if len(y) >= 3 else np.empty(0)
    fin = boots[np.isfinite(boots)]
    lo, hi = (np.percentile(fin, [2.5, 97.5]) if fin.size > p.n_boot / 2 else (np.nan, np.nan))
    return {"t_ms": t, "lead_ms": -t, "ci_lo_ms": float(lo), "ci_hi_ms": float(hi),
            "change_db": change, "boot_found": float(fin.size / max(boots.size, 1))}


def analyse(names: list[str], phases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
            rest_level: np.ndarray, events: dict[tuple[str, str], np.ndarray], pid: str,
            p: EnsembleParams, seed: int = 0) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    """Rows of the ``ensemble`` table and the participant's median curves.

    ``phases[task] = (t_us, cleaned, mask)``; ``events[(task, direction)]`` holds the
    mechanical onsets (us) of that task's cycles.
    """
    g = grid_ms(p)
    rng = np.random.default_rng(seed)
    rows, arrays = [], {"ensemble__grid_ms": g}
    for task, (t_us, cleaned, mask) in phases.items():
        fs = 1e6 / np.median(np.diff(t_us)) if len(t_us) > 1 else 1.0
        # Masked samples are left out of the RMS (NaN where < 50 % of the window is valid),
        # not zero-filled: zeros pulled the envelope down next to every mask (E18).
        env = masked_envelope(cleaned, mask, fs, p.envelope_s)
        env[mask] = np.nan
        for direction, (agon, anta) in GROUPS.items():
            ev = np.asarray(events.get((task, direction), []), dtype=float)
            ev = ev[np.isfinite(ev)]
            if ev.size == 0:
                continue
            a = aligned_db(t_us, env, rest_level, ev, g)
            for role, group, kind in (("agonist", agon, "rise"), ("antagonist", anta, "release")):
                y = _group_curve(a, names, group)
                if y is None:
                    continue
                arrays[f"ensemble__{task}__{direction}__{role}"] = np.nanmedian(y, axis=0) \
                    .astype(np.float32) if np.isfinite(y).any() else np.full(g.size, np.nan,
                                                                              np.float32)
                rows.append({"pid": pid, "task": task, "direction": direction, "role": role,
                             "event": kind, "n_cycles": int(len(y)),
                             **ensemble_events(y, g, kind, p, rng)})
    return rows, arrays


# -- C2-E P3: per-cycle release and its S1 null (X6 (change log)) ---------------------------
def envelope_db(cleaned: np.ndarray, mask: np.ndarray, fs: float, rest_level: np.ndarray,
                envelope_s: float) -> np.ndarray:
    """(N, C) envelope in dB re rest, as :func:`analyse` builds it (masked samples left out
    of the RMS, NaN where masked)."""
    env = masked_envelope(cleaned, mask, fs, envelope_s)
    env[np.asarray(mask, bool)] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(np.maximum(env, 1e-9) / rest_level[None, :])


def aligned_db_circular(db: np.ndarray, i_ref: np.ndarray, g_samples: np.ndarray,
                        shift: int = 0) -> np.ndarray:
    """(events, grid, channels) of ``db`` around the event samples ``i_ref`` with the EMG
    circularly shifted by ``shift`` samples (S1, :mod:`biomech.timing.surrogates`; 0 = the
    observed alignment). Grid points outside the phase *at the real event* are NaN for every
    shift, so observed and surrogate curves have the same support."""
    N = db.shape[0]
    idx = np.asarray(i_ref, np.int64)[:, None] + np.asarray(g_samples, np.int64)[None, :]
    out = db[(idx - int(shift)) % N]
    out[(idx < 0) | (idx >= N)] = np.nan
    return out


def release_per_cycle(y: np.ndarray, g: np.ndarray, p: EnsembleParams
                      ) -> tuple[np.ndarray, np.ndarray]:
    """:func:`release_time` of every cycle's own group curve ``y`` (cycles x grid): (t_ms,
    change_db), NaN where the cycle has no release of ``min_change_db``."""
    import warnings

    t = np.full(len(y), np.nan)
    ch = np.full(len(y), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k, row in enumerate(y):
            t[k], ch[k] = release_time(row, g, p)
    return t, ch


def release_s1(db: np.ndarray, names: list[str], i_ref: np.ndarray, fs: float,
               p: EnsembleParams, shifts: np.ndarray, group: tuple[str, ...] = CLOSERS
               ) -> dict[str, np.ndarray | float]:
    """P3 (X6): the closer release before opening, re-aligned to the events ``i_ref``
    (phase samples of t_omega), with the X2 parameters, and its S1 null.

    Observed: ``t_ms`` and ``depth_db`` (pre - minimum, dB; > 0 = a drop) of the median group
    curve (:func:`release_time`), per-cycle ``cycle_t_ms`` / ``cycle_change_db``
    (:func:`release_per_cycle`). Null, one value per S1 shift: ``null_depth_db`` (median-curve
    depth) and ``null_cycle_t_ms`` (list of per-cycle release times). Empty arrays without the
    group or events."""
    import warnings

    g = grid_ms(p)
    gs = np.round(g * fs / 1e3).astype(np.int64)
    idx = [names.index(m) for m in group if m in names]
    out: dict[str, Any] = {"t_ms": np.nan, "depth_db": np.nan, "cycle_t_ms": np.empty(0),
                           "cycle_change_db": np.empty(0), "null_depth_db": np.empty(0),
                           "null_cycle_t_ms": []}
    if not idx or len(i_ref) == 0:
        return out

    def curves(shift: int) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(aligned_db_circular(db[:, idx], i_ref, gs, shift), axis=2)

    def median_depth(y: np.ndarray) -> tuple[float, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            t, ch = release_time(np.nanmedian(y, axis=0), g, p)
        return t, -ch

    y = curves(0)
    out["t_ms"], out["depth_db"] = median_depth(y)
    out["cycle_t_ms"], out["cycle_change_db"] = release_per_cycle(y, g, p)
    nd, nc = np.full(len(shifts), np.nan), []
    for j, s in enumerate(shifts):
        ys = curves(int(s))
        nd[j] = median_depth(ys)[1]
        nc.append(release_per_cycle(ys, g, p)[0])
    out["null_depth_db"], out["null_cycle_t_ms"] = nd, nc
    return out
