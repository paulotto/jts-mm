# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""sEMG onset detection: offline detectors and their causal (real-time) twins.

Three detectors, all operating on a band-passed signal of one channel:

* ``tkeo``        Teager–Kaiser energy (Solnik et al. 2010), rectified and
                  smoothed, then a double threshold. Primary detector.
* ``hodges_bui``  Rectified signal low-passed at 50 Hz, mean + h·SD threshold
                  held for ``min_on_s`` (Hodges & Bui 1996).
* ``envelope``    RMS envelope above baseline mean + 3·SD.

Each detector returns onset indices into the input. The *causal* variant uses
only past samples (single-pass filters, trailing windows) and reports the index
at which the decision is made, i.e. ``onset + detection delay``; the difference
to the offline onset is the latency a controller would pay.

Baseline statistics come from a caller-chosen quiet window, never from the
whole signal, so a session full of activity cannot raise its own threshold.

SciPy (the ``analysis`` extra) is imported inside the filters, so importing
this module works on a minimal install.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "OnsetParams",
    "tkeo",
    "detector_signal",
    "detect_onsets",
    "first_onset",
    "odd_window",
    "baseline_values",
    "baseline_threshold",
    "baseline_stats",
]


@dataclass(frozen=True)
class OnsetParams:
    method: str = "tkeo"          # tkeo | hodges_bui | envelope
    k_sd: float = 3.0             # threshold = baseline mean + k_sd * baseline SD
    min_on_s: float = 0.025       # supra-threshold run needed to accept an onset
    min_off_s: float = 0.05       # quiet run needed before a new onset can start
    smooth_s: float = 0.02        # smoothing of the detector signal (tkeo/hodges_bui)
    envelope_s: float = 0.05      # RMS window for the envelope detector
    edge_s: float = 0.05          # runs touching the first/last edge_s are filter edge effects


def odd_window(window_s: float, fs_hz: float) -> int:
    """Samples of a centred moving window: the odd count nearest ``window_s * fs_hz`` (>= 1).

    An even-length centred moving average is asymmetric and shifts its output by half a
    sample. Same rule as ``biomech.timebase.odd_window`` (kept here because jtsmm does not
    import biomech); an exact even product rounds up (300.0 -> 301).
    """
    x = float(window_s) * float(fs_hz)
    if not np.isfinite(x) or x < 1.0:
        return 1
    return int(2 * np.floor(x / 2.0) + 1)


def _odd(n: int) -> int:
    n = max(int(n), 1)
    return n if n % 2 else n + 1


def tkeo(x: np.ndarray) -> np.ndarray:
    """Teager–Kaiser energy ``x[n]^2 - x[n-1] x[n+1]`` (edges repeated)."""
    a = np.asarray(x, dtype=np.float64)
    out = np.empty_like(a)
    out[1:-1] = a[1:-1] ** 2 - a[:-2] * a[2:]
    out[0], out[-1] = out[1], out[-2]
    return out


def _smooth(x: np.ndarray, n: int, causal: bool) -> np.ndarray:
    """Moving mean over ``n`` samples: trailing (causal) or centred.

    The centred branch uses an odd length (an even ``n`` is raised by one) so it does
    not shift the signal by half a sample; the causal branch keeps ``n``.
    """
    n = max(int(n), 1)
    if causal:
        from scipy.signal import lfilter

        return lfilter(np.ones(n) / n, [1.0], x)
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(x, size=_odd(n), mode="nearest")


def detector_signal(x: np.ndarray, fs_hz: float, p: OnsetParams, *, causal: bool = False
                    ) -> np.ndarray:
    """The non-negative signal each detector thresholds."""
    a = np.asarray(x, dtype=np.float64)
    if p.method == "tkeo":
        n = odd_window(p.smooth_s, fs_hz) if not causal else round(p.smooth_s * fs_hz)
        return _smooth(np.abs(tkeo(a)), n, causal)
    if p.method == "hodges_bui":
        from scipy.signal import sosfilt, sosfiltfilt

        sos = _hb_sos(float(fs_hz)).copy()        # scipy needs a writable buffer
        r = np.abs(a)
        return sosfilt(sos, r) if causal else sosfiltfilt(sos, r)
    if p.method == "envelope":
        n = (max(int(round(p.envelope_s * fs_hz)), 1) if causal
             else odd_window(p.envelope_s, fs_hz))
        return np.sqrt(np.maximum(_smooth(a * a, n, causal), 0.0))
    raise ValueError(f"unknown onset method {p.method!r}")


@lru_cache(maxsize=16)
def _hb_sos(fs_hz: float) -> np.ndarray:
    """The Hodges–Bui 50 Hz low-pass (designed once per rate; read-only)."""
    from scipy.signal import butter

    sos = butter(2, 50.0 / (0.5 * fs_hz), output="sos")
    sos.flags.writeable = False
    return sos


def _raw_signal(x: np.ndarray, method: str) -> np.ndarray:
    return np.abs(tkeo(x)) if method == "tkeo" else np.abs(x)


def _half_window(fs_hz: float, p: OnsetParams) -> int:
    if p.method == "envelope":
        return (odd_window(p.envelope_s, fs_hz) - 1) // 2
    if p.method == "tkeo":
        return (odd_window(p.smooth_s, fs_hz) - 1) // 2
    return int(round(0.01 * fs_hz))       # 50 Hz low-pass: ~10 ms of zero-phase spread


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


MIN_BASELINE_RUN_S = 0.05     # shorter NaN-free baseline runs are skipped (filter edges)


def baseline_values(x: np.ndarray, fs_hz: float, p: OnsetParams, *, causal: bool = False,
                    min_run_s: float = MIN_BASELINE_RUN_S) -> tuple[np.ndarray, np.ndarray]:
    """Detector-signal and raw-signal values of a quiet recording, pooled over its NaN-free runs.

    NaN marks excluded (masked) samples. Each run is processed on its own and the values
    are pooled: concatenating the runs would create seams whose TKEO / filter transients
    raise the threshold (E18). Runs shorter than ``min_run_s`` (at least 16 samples) are
    skipped. A NaN-free input is one run, i.e. the same values as processing it whole.
    """
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    n_min = max(int(round(min_run_s * fs_hz)), 16)
    s_parts, r_parts = [], []
    for i0, i1 in _runs(np.isfinite(a)):
        if i1 - i0 < n_min:
            continue
        seg = a[i0:i1]
        s_parts.append(detector_signal(seg, fs_hz, p, causal=causal))
        r_parts.append(_raw_signal(seg, p.method))
    if not s_parts:
        return np.empty(0), np.empty(0)
    sv, rv = np.concatenate(s_parts), np.concatenate(r_parts)
    return sv[np.isfinite(sv)], rv[np.isfinite(rv)]


def baseline_threshold(x: np.ndarray, fs_hz: float, p: OnsetParams, *, causal: bool = False,
                       min_valid_s: float = 0.0) -> float:
    """``mean + k_sd * SD`` of the pooled detector signal (:func:`baseline_values`); NaN when
    fewer than ``min_valid_s`` seconds (and 8 samples) of baseline survive."""
    sv, _ = baseline_values(x, fs_hz, p, causal=causal)
    if sv.size < max(8, int(round(min_valid_s * fs_hz))):
        return float("nan")
    return float(sv.mean() + p.k_sd * sv.std())


def baseline_stats(x: np.ndarray, fs_hz: float, p: OnsetParams, *, causal: bool = False
                   ) -> tuple[float, float] | None:
    """(detector threshold, raw threshold) that :func:`detect_onsets` derives from a
    ``baseline_signal``; None when fewer than 8 baseline samples survive (no onsets).

    Pass it as ``detect_onsets(..., baseline_stats=...)`` to threshold many windows against
    the same quiet recording without recomputing its detector signal each time (same
    thresholds, so the same onsets)."""
    base, rb = baseline_values(x, fs_hz, p, causal=causal)
    base = base[np.isfinite(base)]
    if base.size < 8:
        return None
    return (float(base.mean() + p.k_sd * base.std()), float(rb.mean() + p.k_sd * rb.std()))


def detect_onsets(
    x: np.ndarray,
    fs_hz: float,
    baseline: slice | np.ndarray | None,
    p: OnsetParams = OnsetParams(),
    *,
    causal: bool = False,
    baseline_signal: np.ndarray | None = None,
    baseline_stats: tuple[float, float] | None | bool = False,
) -> np.ndarray:
    """Onset indices (offline) or decision indices (``causal=True``).

    @param baseline         a slice or boolean mask selecting a quiet stretch of ``x``,
    @param baseline_signal  or a separate quiet recording of the same channel
                            (e.g. the rest phase) whose statistics set the threshold;
                            NaN marks excluded samples, the statistics pool the NaN-free
                            runs (:func:`baseline_values`).
    A run counts when it stays above threshold for ``min_on_s`` and follows at
    least ``min_off_s`` below it. Offline, the onset is the run's first sample;
    causally, the decision falls ``min_on_s`` later (the earliest moment the run
    is known to be long enough). NaN samples never exceed the threshold.

    Offline, a centred smoother lets the detector signal cross up to half a
    window *before* the burst, which would bias an EMG-to-motion lead:
      * tkeo: each run is refined to the first sample within +-half a window
        whose unsmoothed |TKEO| exceeds the k-SD threshold of the unsmoothed
        baseline (synthetic bursts: |error| <= 10 ms);
      * envelope: the run start is shifted by half the RMS window;
      * hodges_bui: zero-phase 50 Hz low-pass, used as is (about 10 ms early).
    The residual biases differ by detector on purpose: the spread across the
    three is reported as a sensitivity analysis (analysis_plan §5.3).

    @param baseline_stats   instead of ``baseline_signal``: its precomputed
                            :func:`baseline_stats` (None = too little baseline, no onsets).
    """
    xa = np.nan_to_num(np.asarray(x, dtype=np.float64))
    if len(xa) < 16:                  # too short for TKEO / the 50 Hz zero-phase filter
        return np.empty(0, dtype=np.int64)
    s = detector_signal(xa, fs_hz, p, causal=causal)
    if baseline_stats is not False:
        if baseline_stats is None:
            return np.empty(0, dtype=np.int64)
        thr, raw_thr_pre = baseline_stats
    else:
        if baseline_signal is not None:
            base, rb = baseline_values(baseline_signal, fs_hz, p, causal=causal)
        else:
            base = s[baseline]
        base = base[np.isfinite(base)]
        if base.size < 8:
            return np.empty(0, dtype=np.int64)
        thr = float(base.mean() + p.k_sd * base.std())
    above = np.isfinite(np.asarray(x, dtype=np.float64)) & (s > thr)
    n_on = max(int(round(p.min_on_s * fs_hz)), 1)
    n_off = max(int(round(p.min_off_s * fs_hz)), 1)
    h = _half_window(fs_hz, p)
    if not causal and p.method == "tkeo":
        raw = _raw_signal(xa, p.method)
        if baseline_stats is not False:
            raw_thr = raw_thr_pre
        else:
            if baseline_signal is None:
                rb = raw[baseline]
            raw_thr = float(rb.mean() + p.k_sd * rb.std())
    edge = int(round(p.edge_s * fs_hz))
    n = len(xa)
    out: list[int] = []
    last_end = -n_off - 1
    for a, b in _runs(above):
        if b - a < n_on or a < edge or (not causal and b > n - edge and b - a < 2 * n_on):
            last_end = b if a < edge else last_end
            continue
        if a - last_end >= n_off:
            if causal:
                out.append(a + n_on - 1)
            elif p.method == "tkeo":
                lo, hi = max(a - h, 0), min(a + h + 1, n)
                hit = np.flatnonzero(raw[lo:hi] > raw_thr)
                out.append(int(lo + hit[0]) if hit.size else int(a))
            elif p.method == "envelope":
                out.append(int(min(a + h, b - 1)))
            else:
                out.append(int(a))
        last_end = b
    return np.asarray(out, dtype=np.int64)


def first_onset(x: np.ndarray, fs_hz: float, baseline: slice | np.ndarray | None,
                p: OnsetParams = OnsetParams(), *, causal: bool = False,
                start: int = 0, baseline_signal: np.ndarray | None = None) -> int | None:
    """First onset (or decision) at or after ``start``; None when none is found."""
    idx = detect_onsets(x, fs_hz, baseline, p, causal=causal, baseline_signal=baseline_signal)
    idx = idx[idx >= start]
    return int(idx[0]) if idx.size else None
