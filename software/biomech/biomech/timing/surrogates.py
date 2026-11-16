# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Surrogates of the timing analyses: the one engine behind every chance null.

Used by the D23 lead null (:func:`biomech.timing.lead.pseudo_onset_null`), the X3 chance level
(:func:`biomech.timing.release.chance_level`) and the C2-E family (X6,
:mod:`biomech.timing.c2e`):

* **S1, circular shift.** The EMG phase (EMG, analysis mask and causal chain together) is
  shifted circularly against the motion by a uniform 1 s ... T - 1 s (T = phase length), so
  every coupling between EMG and motion is destroyed while the EMG's own statistics (burst
  rate, durations, amplitudes, masks) are kept. Two uses: one shift per pseudo-onset (D23,
  :func:`shifted_windows` with one shift per window) or one shift of the whole phase with
  every real cycle re-evaluated (C2-E, :func:`s1_sample_shifts`). Shifting the EMG by ``s``
  is the same as reading it at index ``i - s`` (mod N): only index arithmetic, no copies.
  An event train (decision times) is shifted by :func:`shift_times`.
* **S2, anchor-preserving cycle shuffle.** Per cycle k an anchor a_k (end of the previous
  closing for opening, the opening onset for closing) and an event e_k (an EMG onset or
  release); the offsets e_k - a_k are permuted across cycles and re-attached to the anchors.
  It keeps everything locked to the rhythm (the anchor) and destroys only the locking to
  the upcoming movement onset (X6, B3).
* **Seeding.** :func:`null_seed`: one generator per (run seed, key), so each null is
  reproducible and independent of the order in which the nulls are computed.

Units: times in µs unless named otherwise; ``shift`` in the unit of ``span`` (s or µs).
"""

from __future__ import annotations

import zlib

import numpy as np


def null_seed(seed: int, key: str) -> np.random.Generator:
    """Generator of one null: the run seed and a key (the task name for D23)."""
    return np.random.default_rng([int(seed) & 0x7FFFFFFF, zlib.crc32(key.encode())])


def uniform_shifts(rng: np.random.Generator, span: float, n: int, gap: float) -> np.ndarray:
    """``n`` circular shifts uniform on [gap, span - gap] (same unit as ``span``)."""
    return rng.uniform(gap, span - gap, n)


def shifted_windows(i_ref: np.ndarray, shift: np.ndarray, off: np.ndarray, n: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """S1 by index arithmetic: the window of reference sample ``i_ref`` in an EMG shifted
    by ``shift`` samples. Returns ``(i_e, sel)``: the EMG sample that sits at the reference
    (``(i_ref - shift) mod n``) and the window indices ``(i_e + off) mod n`` (one row per
    reference; ``off`` = sample offsets of the window, e.g. ``arange(-n_pre, n_post + 1)``).
    """
    i_e = (np.asarray(i_ref, np.int64) - np.asarray(shift, np.int64)) % n
    return i_e, (i_e[..., None] + np.asarray(off, np.int64)) % n


def shift_times(t: np.ndarray, t0: float, period: float, s: float) -> np.ndarray:
    """An event train circularly shifted by ``s`` within [t0, t0 + period)."""
    return t0 + np.mod(t - t0 + s, period)


def s1_sample_shifts(rng: np.random.Generator, n_samples: int, fs: float, n: int,
                     gap_s: float) -> np.ndarray:
    """``n`` whole-phase S1 shifts in samples, uniform on [gap_s, T - gap_s] s (T =
    ``n_samples / fs``); empty when the phase is shorter than 2 ``gap_s``."""
    T = n_samples / fs
    if T <= 2 * gap_s or n <= 0:
        return np.empty(0, np.int64)
    return np.round(uniform_shifts(rng, T, n, gap_s) * fs).astype(np.int64)


def s2_permutations(rng: np.random.Generator, n_cycles: int, n: int) -> np.ndarray:
    """(n, n_cycles) independent permutations of the cycle indices (S2)."""
    return rng.permuted(np.tile(np.arange(n_cycles), (max(n, 0), 1)), axis=1)


def s2_shuffle(anchor: np.ndarray, event: np.ndarray, perms: np.ndarray) -> np.ndarray:
    """S2 surrogate events (n, n_cycles): anchor_k + (event - anchor)[perm_k]."""
    a = np.asarray(anchor, float)
    d = np.asarray(event, float) - a
    return a[None, :] + d[perms]


def mc_p(null: np.ndarray, obs: float, alternative: str = "less") -> float:
    """Monte-Carlo p (k + 1) / (N + 1) of ``obs`` against the finite ``null`` values:
    ``less`` counts null <= obs (e.g. a locking IQR smaller than chance), ``greater`` null
    >= obs. NaN without an observed value or a null."""
    v = np.asarray(null, float)
    v = v[np.isfinite(v)]
    if not np.isfinite(obs) or v.size == 0:
        return np.nan
    k = np.sum(v <= obs) if alternative == "less" else np.sum(v >= obs)
    return float((k + 1) / (v.size + 1))
