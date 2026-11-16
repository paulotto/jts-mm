# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Metrics of the pseudo-online evaluation (ml08 §5). Inputs are per-window arrays of one
participant in time order; labels are used for scoring only.

* ``balanced_accuracy`` of the raw and smoothed window decisions over the scored windows
  (decodable, labelled, inside the mode's scored portion; free motion separately).
* Decision latency (:func:`decision_latency`): per REST -> class onset of the *optical label
  track* (time ``t_on``, class held until ``t_off``, preceded by REST since ``t_rest``) the
  first window of the scored portion, ending in ``[max(t_on - pre_s, t_rest), t_off]``, at
  which the smoothed decision *enters* that class; latency = its *decision time* (the chunk at
  which the stream emitted it) - ``t_on``. Negative = anticipatory; no entry = not detected
  (a state held over from the previous movement is not a detection). The window-end latency
  of ``ml_eval`` (:func:`ml.protocols.evaluate.latency`, raw decisions, 2 consecutive windows,
  onsets of the window labels) is reported alongside (:func:`window_latency`).
* False activations (:func:`false_activations_free`): smoothed onsets in ``free_motion`` at a
  window whose label is REST and with no labelled movement onset within ``guard_s`` after it,
  per minute of REST-labelled free-motion windows (count x stride). In the ``rest`` phase
  :func:`ml.protocols.evaluate.false_activations` (onsets per minute of the phase); the rest
  windows are decoded at the end of the rest phase with statistics from that phase itself
  (in-sample, as offline).
* X3 release (:func:`release_metrics`): per task, the first stream release decision in
  ``[t_mech - search_pre_s, t_mech + search_post_s]`` of every mechanical opening onset
  (:func:`biomech.timing.release.evaluate`), its chance level at random pseudo-onsets
  (:func:`biomech.timing.release.chance_detection`), false releases per minute.
* Compute (:func:`compute_stats`): per-chunk processing time, median / P99 / max (ms).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ml.config.classes import CLASS_COUNT, FREE_PHASE, REST_PHASE
from ml.evaluation.metrics import balanced_accuracy
from ml.protocols import evaluate as ev

REST = 0


def label_onsets(labels: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    """REST -> class transitions of the label track per phase: phase, t_on, t_off, cls and
    ``t_rest_us``, the start of the REST run before the onset."""
    out = []
    for ph, (t, y) in labels.items():
        t = np.asarray(t, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)
        ok = (y >= 0) & (y < CLASS_COUNT)
        t, y = t[ok], y[ok]
        if y.size < 2:
            continue
        period = int(np.median(np.diff(t)))
        for i in np.flatnonzero((y[1:] != REST) & (y[:-1] == REST)) + 1:
            j = i
            while j < y.size and y[j] == y[i]:
                j += 1
            t_off = int(t[j]) if j < y.size else int(t[-1]) + period
            k = i - 1
            while k > 0 and y[k - 1] == REST:
                k -= 1
            out.append({"phase": ph, "t_on_us": int(t[i]), "t_off_us": t_off, "cls": int(y[i]),
                        "t_rest_us": int(t[k])})
    return out


def decision_latency(on: list[dict[str, Any]], t_end: np.ndarray, t_dec: np.ndarray,
                     s: np.ndarray, phase: np.ndarray, portion: np.ndarray,
                     pre_s: float = 0.5) -> list[dict[str, Any]]:
    """Per label onset (inside the scored portion): hit window index and latency (ms)."""
    rows = []
    for o in on:
        sel = np.flatnonzero((phase == o["phase"]) & portion)
        lo = max(o["t_on_us"] - pre_s * 1e6, o.get("t_rest_us", -np.inf))
        if sel.size == 0 or t_end[sel[0]] > lo:
            continue                                   # onset not fully inside the portion
        cand = sel[(t_end[sel] >= lo) & (t_end[sel] <= o["t_off_us"])]
        prev = np.where(cand > 0, s[np.maximum(cand - 1, 0)], REST)
        same = (cand > 0) & (phase[np.maximum(cand - 1, 0)] == o["phase"])
        entry = (s[cand] == o["cls"]) & ~(same & (prev == o["cls"]))
        hit = cand[entry]
        j = int(hit[0]) if hit.size else -1
        rows.append({**o, "hit": j, "detected": j >= 0,
                     "latency_ms": (t_dec[j] - o["t_on_us"]) / 1e3 if j >= 0 else np.nan,
                     "latency_end_ms": (t_end[j] - o["t_on_us"]) / 1e3 if j >= 0 else np.nan})
    return rows


def window_latency(t_end: np.ndarray, y: np.ndarray, p: np.ndarray, phase: np.ndarray,
                   rows: np.ndarray, pre_s: float = 0.5, confirm: int = 2) -> list[float]:
    """:func:`ml.protocols.evaluate.latency` per phase on ``rows`` (the offline definition)."""
    out: list[float] = []
    for ph in dict.fromkeys(phase[rows]):
        r = rows[phase[rows] == ph]
        out += ev.latency(t_end[r], y[r], p[r], pre_s=pre_s, confirm=confirm)
    return out


def false_activations_free(ev_idx: np.ndarray, t_end: np.ndarray, y: np.ndarray,
                           phase: np.ndarray, portion: np.ndarray, calibrated: np.ndarray,
                           on: list[dict[str, Any]], stride_s: float, guard_s: float = 0.5
                           ) -> dict[str, float]:
    """False onsets per minute of REST-labelled free-motion windows."""
    free = (phase == FREE_PHASE) & portion & calibrated
    rest_free = free & (y == REST)
    minutes = rest_free.sum() * stride_s / 60.0
    t_on = np.array(sorted(o["t_on_us"] for o in on), dtype=np.int64)
    n = 0
    for j in ev_idx:
        if not rest_free[j]:
            continue
        k = np.searchsorted(t_on, t_end[j], side="left")
        if k < t_on.size and t_on[k] - t_end[j] <= guard_s * 1e6:
            continue                                     # an early detection, not false
        n += 1
    return {"fa_free_n": int(n), "fa_free_minutes": float(minutes),
            "fa_per_min_free": n / minutes if minutes > 0 else np.nan}


def classify_events(ev_idx: np.ndarray, hits: set[int], t_end: np.ndarray, y: np.ndarray,
                    on: list[dict[str, Any]], guard_s: float = 0.5) -> list[str]:
    """``hit`` (the first detection of a labelled onset), ``false`` (onset at a REST window
    with no labelled onset within ``guard_s``) or ``other`` (e.g. a wrong class mid-movement)."""
    t_on = np.array(sorted(o["t_on_us"] for o in on), dtype=np.int64)
    out = []
    for j in ev_idx:
        if int(j) in hits:
            out.append("hit")
            continue
        k = np.searchsorted(t_on, t_end[j], side="left")
        near = k < t_on.size and t_on[k] - t_end[j] <= guard_s * 1e6
        out.append("false" if y[j] == REST and not near else "other")
    return out


def scores(y: np.ndarray, p: np.ndarray, rows: np.ndarray) -> float:
    return balanced_accuracy(y[rows], p[rows]) if rows.size else np.nan


def rest_false_activations(t_end: np.ndarray, p: np.ndarray, rows: np.ndarray
                           ) -> dict[str, float]:
    return ev.false_activations(t_end[rows], p[rows]) if rows.size >= 2 else \
        {"fa_frac": np.nan, "fa_per_min": np.nan}


def release_metrics(dec_us: np.ndarray, phases: dict[str, tuple[int, int]],
                    onsets_us: dict[str, np.ndarray], p: Any, seed: int = 0
                    ) -> list[dict[str, Any]]:
    """X3 metrics per task with stream decisions (offline definitions, biomech.timing.release;
    the chance level as offline: circular shift of the decisions in the rhythmic tasks of
    ``lead.NULL_SHIFT_TASKS``, pseudo-onsets otherwise)."""
    from biomech.timing.lead import NULL_SHIFT_TASKS
    from biomech.timing.release import chance_detection, evaluate

    rows = []
    dec_us = np.asarray(dec_us, dtype=np.int64)
    for task, (t0, t1) in phases.items():
        on = np.asarray(onsets_us.get(task, []), dtype=float)
        on = on[np.isfinite(on)]
        if on.size == 0:
            continue
        dec = dec_us[(dec_us >= t0) & (dec_us <= t1)]
        cyc, false = evaluate(dec, on, p)
        ant = np.array([r["anticipation_ms"] for r in cyc], dtype=float)
        minutes = (t1 - t0) / 6e7
        rows.append({"task": task, "n_openings": int(on.size),
                     "detected_frac": float(np.mean([r["detected"] for r in cyc])),
                     "chance_detected_frac": chance_detection(
                         dec, on, float(t0), float(t1), p, seed=seed,
                         shift=task in NULL_SHIFT_TASKS),
                     "anticipation_ms": float(np.nanmedian(ant)) if np.isfinite(ant).any()
                     else np.nan,
                     "n_false": int(false),
                     "false_per_min": float(false / minutes) if minutes > 0 else np.nan,
                     "n_decisions": int(dec.size)})
    return rows


def compute_stats(chunk_ns: np.ndarray, chunk_s: float) -> dict[str, float]:
    ms = np.asarray(chunk_ns, dtype=np.float64) / 1e6
    if ms.size == 0:
        return {"n_chunks": 0, "chunk_ms_median": np.nan, "chunk_ms_p99": np.nan,
                "chunk_ms_max": np.nan, "realtime_factor": np.nan}
    return {"n_chunks": int(ms.size), "chunk_ms_median": float(np.median(ms)),
            "chunk_ms_p99": float(np.percentile(ms, 99)), "chunk_ms_max": float(ms.max()),
            "realtime_factor": float(ms.sum() / (ms.size * chunk_s * 1e3))}


__all__ = ["REST_PHASE", "classify_events", "compute_stats", "decision_latency",
           "false_activations_free", "label_onsets", "release_metrics",
           "rest_false_activations", "scores", "window_latency"]
