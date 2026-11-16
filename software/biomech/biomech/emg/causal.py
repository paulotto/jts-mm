# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Causal (real-time) cleaning of recorded EMG for the causal timing quantities (E27).

The offline chain (``biomech.emg.pipeline.clean``: fitted mains harmonics + zero-phase
band-pass) reads future samples, so a "causal" onset decision or release detector fed with
it still looks ahead by the filter's spread. Quantities that stand for what a real-time
controller could do (the causal onset twins in ``timing.lead``: ``t_decision_us``,
``causal_delay_ms``, ``anticipation_ms``; the X3 release detector) use this chain instead:

* :class:`jtsmm.viewer.processing.CausalEmgFilter` (DC block -> tracking mains canceller at
  the take/phase grid frequency -> 20 Hz high-pass -> 450 Hz low-pass, ``mains_mode="track"``),
  fed with the timestamps (``process_timed``) so dropped samples keep the canceller on the grid;
* started ``warmup_s`` (2 s) before the phase and the warm-up discarded, as
  ``ml.core.signals.causal_emg`` / ``ml.core.dataset`` do;
* masked raw samples (analysis mask; rail and non-finite samples also in the warm-up) are
  replaced by the last valid sample of the channel *before* filtering (``hold_fill``, causal),
  so an artifact does not ring through the IIR sections; the analysis mask is applied (NaN)
  *after* filtering.

Offline onsets keep the zero-phase signal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from biomech.emg.prepare import EmgPrep

WARMUP_S = 2.0


def hold_fill(x: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Masked samples replaced by the last valid sample of the same channel (0 before any).

    Causal: never reads a later sample (same rule as ``ml.core.signals.hold_fill``).
    """
    x = np.asarray(x, dtype=np.float64)
    if mask is None or not np.any(mask):
        return x
    n = x.shape[0]
    idx = np.where(mask, -1, np.arange(n)[:, None])
    np.maximum.accumulate(idx, axis=0, out=idx)
    cols = np.broadcast_to(np.arange(x.shape[1]), idx.shape)
    return np.where(idx >= 0, x[np.maximum(idx, 0), cols], 0.0)


def causal_filter(t_us: np.ndarray, raw_uv: np.ndarray, fs: float, line_hz: float, *,
                  keep: np.ndarray | None = None, mask: np.ndarray | None = None,
                  bad: np.ndarray | None = None, hpf_hz: float = 20.0, lpf_hz: float = 450.0,
                  apply_mask: bool = True) -> np.ndarray:
    """Causally filter ``raw_uv`` (N, C) and return the rows ``keep`` (default: all).

    @param mask  (n_keep, C) analysis mask of the kept rows: hold-filled before filtering,
                 NaN after (``apply_mask``).
    @param bad   (N, C) further samples to hold-fill before filtering (e.g. rail hits in the
                 warm-up); non-finite samples always are.
    """
    from jtsmm.viewer.processing import CausalEmgFilter

    raw = np.asarray(raw_uv, dtype=np.float64)
    if raw.ndim == 1:
        raw = raw[:, None]
    n, c = raw.shape
    keep = np.ones(n, bool) if keep is None else np.asarray(keep, bool)
    fill = ~np.isfinite(raw)
    if bad is not None:
        fill |= np.asarray(bad, bool)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.shape != (int(keep.sum()), c):
            raise ValueError(f"mask shape {m.shape} != kept rows {(int(keep.sum()), c)}")
        fill[keep] |= m
    x = hold_fill(raw, fill)
    if n == 0:
        return np.zeros((0, c))
    f = CausalEmgFilter(fs, c, hpf_hz=hpf_hz, lpf_hz=lpf_hz, notch_hz=float(line_hz),
                        mains_mode="track")
    out, _ = f.process_timed(np.asarray(t_us), x)
    y = np.asarray(out, dtype=np.float64)[keep]
    if mask is not None and apply_mask:
        y = np.where(np.asarray(mask, bool), np.nan, y)
    return y


def causal_clean(prep_or_raw: "EmgPrep | np.ndarray", phase: str | None = None, *,
                 take: str = "motion", warmup_s: float = WARMUP_S, apply_mask: bool = True,
                 t_us: np.ndarray | None = None, fs: float | None = None,
                 line_hz: float | None = None, t0_us: float | None = None,
                 mask: np.ndarray | None = None, hpf_hz: float = 20.0,
                 lpf_hz: float = 450.0) -> np.ndarray:
    """Causally cleaned EMG of one phase, (n_phase, C) uV, NaN at analysis-masked samples.

    Either an :class:`~biomech.emg.prepare.EmgPrep` and a ``phase`` name: the raw signal is
    read from ``warmup_s`` before the phase start, the grid frequency is the one the offline
    chain used for that phase (take value refined per phase), the band edges come from the
    EMG parameters, and the mask is the phase's analysis mask (identical rows to
    ``prep.phase(phase)``). Or a raw (N, C) array with ``t_us``, ``fs``, ``line_hz``, the
    phase start ``t0_us`` (samples before it are warm-up) and the phase ``mask``.
    """
    if isinstance(prep_or_raw, np.ndarray):
        if t_us is None or fs is None or line_hz is None:
            raise ValueError("raw input needs t_us, fs and line_hz")
        t = np.asarray(t_us)
        keep = t >= (t[0] if t0_us is None else t0_us)
        return causal_filter(t, prep_or_raw, fs, line_hz, keep=keep, mask=mask, hpf_hz=hpf_hz,
                             lpf_hz=lpf_hz, apply_mask=apply_mask)
    prep = prep_or_raw
    if phase is None:
        raise ValueError("causal_clean(prep, phase): phase missing")
    sd = prep.motion if take == "motion" else prep.mvc
    if sd is None:
        raise LookupError(f"no {take} take")
    pe = prep.phase(phase, take=take)
    ph = sd.phase(phase)
    t, raw, _ = sd.emg_window(int(ph.t_start_us - warmup_s * 1e6), ph.t_end_us, prep.names)
    t = np.asarray(t, dtype=np.int64)
    # The phase rows of the warm-up window are exactly the rows of ``pe`` (same reader);
    # match by timestamp so a differing edge sample cannot misalign the mask.
    keep = np.isin(t, pe.t_us)
    if int(keep.sum()) != len(pe.t_us):
        raise ValueError(f"{phase}: warm-up window does not contain the phase samples")
    band = getattr(prep.p, "bandpass_hz", (hpf_hz, lpf_hz))
    rail = np.abs(raw) >= prep.p.mask.rail_frac * sd.emg_full_scale_uv(prep.names)[None, :]
    return causal_filter(t, raw, sd.emg_fs, prep.phase_line_hz(sd, phase), keep=keep,
                         mask=pe.mask, bad=rail & ~keep[:, None], hpf_hz=float(band[0]),
                         lpf_hz=float(band[1]), apply_mask=apply_mask)
