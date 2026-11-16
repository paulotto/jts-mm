# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Causal detection of the closer release before opening (X3 (change log), exploratory).

The ensemble analysis (X2) shows the elevators relaxing ~100 ms before the jaw opens.
This module asks what a real-time controller could do with that: a causal detector
that sees only past samples.

* Envelope: per closer, causal moving RMS over ``window_s`` of the cleaned EMG (masked
  samples excluded), in dB re the channel's rest median; the group signal E(t) is the mean
  over the closers.
* Baseline: B(t) = median of E over [t − ``base_hi_s``, t − ``base_lo_s``] (lagged, so the
  drop itself does not lower it).
* Release: E(t) ≤ B(t) − ``drop_db`` continuously for ``hold_s``; the decision time is the
  end of that run (causal). After a decision, ``refractory_s`` must pass before the next.
* Per opening cycle (mechanical onset t_mech from ``timing.lead``): the first decision in
  [t_mech − ``search_pre_s``, t_mech + ``search_post_s``]; anticipation_ms = t_mech − t_dec
  (> 0: the controller knew before the jaw moved).
* False releases: decisions with no mechanical opening onset within
  [t_dec − ``search_post_s``, t_dec + ``search_pre_s``], per minute of the task.
* Chance level (:func:`chance_level`): the detection rate at random pseudo-onsets ≥ 1 s
  away from every real onset (``chance_detected_frac``); for the tasks of
  ``lead.NULL_SHIFT_TASKS`` (rhythmic, no movement-free second) the D23a circular shift of
  the decision train against the real onsets instead (``surrogates`` S1). A detected share
  above it is evidence of a release cue, and anticipation values are biased early by up to
  that chance share. The median anticipation at the detected pseudo-events
  (``chance_anticipation_ms``, D23) is what a decision unrelated to the opening yields;
  ``stats_timing.release_cohort`` compares the observed anticipation with it per participant
  (paired signed-rank test).

The parameters were fixed before the pilot data were analysed (X3 (change log)); a
sensitivity sweep over ``drop_db`` and ``hold_s`` is reported alongside.

Input signal (E27): ``timing.lead`` feeds the *causally* cleaned EMG
(:func:`biomech.emg.causal.causal_clean`: tracking mains canceller + causal band-pass from a
2-s warm-up before the phase, analysis mask applied after filtering), so the whole chain
(cleaning, envelope, baseline, decision) uses past samples only. Masked stretches: a NaN group
envelope is replaced in the baseline by the last valid value (causal hold); before the first
valid value there is no baseline and no decision (E21; the former whole-phase median looked
ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from biomech.io.session import CLOSERS
from biomech.timing import surrogates


@dataclass(frozen=True)
class ReleaseParams:
    window_s: float = 0.05
    base_lo_s: float = 0.3
    base_hi_s: float = 0.6
    drop_db: float = 3.0
    hold_s: float = 0.03
    refractory_s: float = 0.3
    search_pre_s: float = 0.6
    search_post_s: float = 0.2
    sweep_drop_db: tuple[float, ...] = (2.0, 3.0, 6.0)
    sweep_hold_s: tuple[float, ...] = (0.02, 0.03, 0.05)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "ReleaseParams":
        d = dict(d or {})
        for k in ("sweep_drop_db", "sweep_hold_s"):
            if k in d:
                d[k] = tuple(float(v) for v in d[k])
        return cls(**d)


def _trailing_sum(x: np.ndarray, n: int) -> np.ndarray:
    """Sum of the last ``n`` samples (fewer at the start) along axis 0, O(N) per column.

    Cumulative sum of the mean-removed values (keeps the running sum small, so the
    differences stay exact to ~1e-12 relative), equal to ``np.convolve(x, ones(n))[:N]``.
    Boolean/integer input is summed exactly in integers.
    """
    x = np.asarray(x)
    i = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(i - n, 0)
    if x.dtype.kind in "bui":
        ci = np.zeros((x.shape[0] + 1,) + x.shape[1:], dtype=np.int64)
        np.cumsum(x, axis=0, out=ci[1:])
        return (ci[i] - ci[lo]).astype(np.float64)
    x = x.astype(np.float64, copy=False)
    m = x.mean(axis=0) if x.shape[0] else np.zeros(x.shape[1:])
    c = np.zeros((x.shape[0] + 1,) + x.shape[1:])
    np.cumsum(x - m, axis=0, out=c[1:])
    cnt = (i - lo).astype(np.float64).reshape((-1,) + (1,) * (x.ndim - 1))
    return c[i] - c[lo] + cnt * m


def causal_db(t_us: np.ndarray, cleaned: np.ndarray, mask: np.ndarray, rest_level: np.ndarray,
              names: list[str], window_s: float) -> np.ndarray:
    """Group closer envelope in dB re rest, causal moving RMS (NaN where no closer valid)."""
    fs = 1e6 / np.median(np.diff(t_us)) if len(t_us) > 1 else 1.0
    n = max(int(round(window_s * fs)), 1)
    idx = [names.index(m) for m in CLOSERS if m in names]
    if not idx:
        return np.full(len(t_us), np.nan)
    c = np.asarray(cleaned, dtype=np.float64)[:, idx]
    valid = ~np.asarray(mask, dtype=bool)[:, idx] & np.isfinite(c)
    x = np.where(valid, c, 0.0)
    num = np.maximum(_trailing_sum(x * x, n), 0.0)
    den = _trailing_sum(valid, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        env = np.sqrt(num / den)
        env[den < 0.5 * n] = np.nan
        db = 20.0 * np.log10(np.maximum(env, 1e-9) / rest_level[idx][None, :])
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(db, axis=1)


def _causal_fill(v: np.ndarray) -> tuple[np.ndarray, int]:
    """NaN replaced by the last valid value; leading NaN by the first valid one (only read
    by windows that end at or after it, see :func:`detect`). Returns (filled, first valid)."""
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros_like(v), v.size
    first = int(np.argmax(ok))
    idx = np.where(ok, np.arange(v.size), -1)
    np.maximum.accumulate(idx, out=idx)
    idx[:first] = first
    return v[idx], first


def _decisions(below: np.ndarray, hold: int, refr: int) -> np.ndarray:
    """Indices where ``below`` has held for ``hold`` samples, at least ``refr`` samples
    (exclusive) after the previous decision. Run lengths by a running index of the last
    False sample; only the (few) candidates are looped over for the refractory rule."""
    n = below.size
    i = np.arange(n)
    last_false = np.maximum.accumulate(np.where(~below, i, -1))
    run = i - last_false
    cand = np.flatnonzero(run == hold)
    out, last = [], -refr - 1
    for c in cand.tolist():
        if c - last > refr:
            out.append(c)
            last = c
    return np.asarray(out, dtype=np.int64)


def detect(t_us: np.ndarray, e_db: np.ndarray, p: ReleaseParams) -> np.ndarray:
    """Causal release decision times (µs).

    The baseline at sample t is the median of the 10-ms decimated envelope over
    [t − ``base_hi_s``, t − ``base_lo_s``]; NaN envelope values are held at the last valid
    value, and a window ending before the first valid value gives no baseline (E21).
    """
    t = np.asarray(t_us, dtype=np.int64)
    e_db = np.asarray(e_db, dtype=np.float64)
    if t.size < 4:
        return np.empty(0, dtype=np.int64)
    fs = 1e6 / np.median(np.diff(t))
    lo, hi = int(round(p.base_lo_s * fs)), int(round(p.base_hi_s * fs))
    hold, refr = max(int(round(p.hold_s * fs)), 1), int(round(p.refractory_s * fs))
    # Lagged running median of E over [t - hi, t - lo], on a 10 ms decimated copy for speed.
    step = max(int(round(0.01 * fs)), 1)
    ed = e_db[::step]
    w = max((hi - lo) // step, 1)
    from scipy.ndimage import median_filter

    filled, first = _causal_fill(ed)
    med = median_filter(filled, size=w, mode="nearest", origin=(w - 1) // 2)  # trailing window
    med[: min(first, med.size)] = np.nan          # no valid value seen yet: no baseline
    base_d = np.full(ed.size, np.nan)
    shift = lo // step
    base_d[shift:] = med[: ed.size - shift] if shift else med
    base = np.repeat(base_d, step)[: e_db.size]
    with np.errstate(invalid="ignore"):
        below = np.isfinite(e_db) & np.isfinite(base) & (e_db <= base - p.drop_db)
    return t[_decisions(below, hold, refr)]


def evaluate(dec_us: np.ndarray, onsets_us: np.ndarray, p: ReleaseParams
             ) -> tuple[list[dict[str, Any]], int]:
    """Per-onset first decision in the search window, and the number of false decisions."""
    rows = []
    for t0 in onsets_us:
        sel = dec_us[(dec_us >= t0 - p.search_pre_s * 1e6) & (dec_us <= t0 + p.search_post_s * 1e6)]
        rows.append({"t_mech_us": float(t0), "t_release_us": float(sel[0]) if sel.size else np.nan,
                     "anticipation_ms": float((t0 - sel[0]) / 1e3) if sel.size else np.nan,
                     "detected": bool(sel.size)})
    false = 0
    for d in dec_us:
        near = (onsets_us >= d - p.search_post_s * 1e6) & (onsets_us <= d + p.search_pre_s * 1e6)
        false += int(not near.any())
    return rows, false


def chance_level(dec_us: np.ndarray, onsets_us: np.ndarray, t0_us: float, t1_us: float,
                 p: ReleaseParams, n: int = 200, seed: int = 0, shift: bool = False
                 ) -> dict[str, float]:
    """Chance level of the release detector, the same search window applied.

    * pseudo-onsets (``shift`` False): drawn uniformly in the phase, at least 1 s from every
      real opening onset and from the phase edges;
    * circular shift (``shift`` True; the tasks of ``lead.NULL_SHIFT_TASKS``, D23a): the
      decision train is shifted circularly within the phase by ``n`` uniform shifts of
      1 s ... T - 1 s and scored against the real onsets, so a rhythmic task with no
      movement-free second still has a chance level.

    Returns ``detected_frac`` (the chance level of ``detected``), ``anticipation_ms`` (median
    anticipation over the detected pseudo-events: what a decision unrelated to the opening
    yields, analysis_plan D23) and ``n`` (pseudo-onsets, or shifts, used).
    """
    rng = np.random.default_rng(seed)
    if shift:
        T = t1_us - t0_us
        dec = np.asarray(dec_us, float)
        if T <= 2e6 or onsets_us.size == 0:
            return {"detected_frac": np.nan, "anticipation_ms": np.nan, "n": 0}
        det, ant = [], []
        for s in surrogates.uniform_shifts(rng, T, n, 1e6):           # S1 of the decisions
            rows, _ = evaluate(np.sort(surrogates.shift_times(dec, t0_us, T, s)), onsets_us, p)
            det += [r["detected"] for r in rows]
            ant += [r["anticipation_ms"] for r in rows]
        a_ = np.asarray(ant, float)
        return {"detected_frac": float(np.mean(det)),
                "anticipation_ms": float(np.nanmedian(a_)) if np.isfinite(a_).any() else np.nan,
                "n": int(n)}
    cand = rng.uniform(t0_us + 1e6, t1_us - 1e6, size=20 * n) if t1_us - t0_us > 2e6 else []
    far = [c for c in cand if np.all(np.abs(onsets_us - c) >= 1e6)][:n]
    if not far:
        return {"detected_frac": np.nan, "anticipation_ms": np.nan, "n": 0}
    rows, _ = evaluate(dec_us, np.asarray(far, float), p)
    ant = np.array([r["anticipation_ms"] for r in rows], float)
    return {"detected_frac": float(np.mean([r["detected"] for r in rows])),
            "anticipation_ms": float(np.nanmedian(ant)) if np.isfinite(ant).any() else np.nan,
            "n": len(far)}


def chance_detection(dec_us: np.ndarray, onsets_us: np.ndarray, t0_us: float, t1_us: float,
                     p: ReleaseParams, n: int = 200, seed: int = 0, shift: bool = False
                     ) -> float:
    """Detection rate at chance (the chance level of ``detected``, :func:`chance_level`):
    random pseudo-onsets, or with ``shift`` the circular shift of the decision train."""
    return chance_level(dec_us, onsets_us, t0_us, t1_us, p, n, seed, shift)["detected_frac"]


def analyse(names: list[str], phases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
            rest_level: np.ndarray, onsets: dict[str, np.ndarray], pid: str, p: ReleaseParams,
            seed: int = 0, shift_tasks: tuple[str, ...] = ()
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``phases[task] = (t_us, cleaned, mask)``; ``onsets[task]`` = mechanical opening onsets.

    ``cleaned`` should be the causally cleaned EMG (E27); ``seed`` (the configured study
    seed) draws the chance level; ``shift_tasks`` (``lead.NULL_SHIFT_TASKS``) get the
    circular-shift chance level of :func:`chance_level` (D23a), the others pseudo-onsets.

    Returns per-cycle rows (primary parameters) and per-task summary rows for every
    (drop_db, hold_s) of the sensitivity sweep (``primary`` marks the frozen setting).
    """
    cyc_rows, task_rows = [], []
    for task, (t_us, cleaned, mask) in phases.items():
        ev = np.asarray(onsets.get(task, []), dtype=float)
        ev = ev[np.isfinite(ev)]
        if ev.size == 0 or len(t_us) < 10:
            continue
        e = causal_db(t_us, cleaned, mask, rest_level, names, p.window_s)
        minutes = (t_us[-1] - t_us[0]) / 6e7
        for drop in p.sweep_drop_db:
            for hold in p.sweep_hold_s:
                q = replace(p, drop_db=drop, hold_s=hold)
                dec = detect(t_us, e, q)
                rows, false = evaluate(dec, ev, q)
                ch = chance_level(dec, ev, float(t_us[0]), float(t_us[-1]), q, seed=seed,
                                  shift=task in shift_tasks)
                chance = ch["detected_frac"]
                primary = bool(np.isclose(drop, p.drop_db) and np.isclose(hold, p.hold_s))
                if primary:
                    cyc_rows += [{"pid": pid, "task": task, "cycle": k, **r}
                                 for k, r in enumerate(rows)]
                ant = np.array([r["anticipation_ms"] for r in rows], float)
                task_rows.append({"pid": pid, "task": task, "drop_db": drop, "hold_s": hold,
                                  "primary": primary, "n_openings": int(ev.size),
                                  "detected_frac": float(np.mean([r["detected"] for r in rows])),
                                  "chance_detected_frac": chance,
                                  "chance_anticipation_ms": ch["anticipation_ms"],
                                  "anticipation_ms": float(np.nanmedian(ant))
                                  if np.isfinite(ant).any() else np.nan,
                                  "n_false": int(false),
                                  "false_per_min": float(false / minutes) if minutes > 0
                                  else np.nan})
    return cyc_rows, task_rows
