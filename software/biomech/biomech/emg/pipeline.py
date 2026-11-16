# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Offline sEMG cleaning chain (an03 §2, analysis_plan §4.2).

``remove_mains`` (fitted line harmonics <= 450 Hz at the real sample times) ->
zero-phase Butterworth band-pass -> RMS envelope. Artifacts are *masked*
afterwards (``biomech.emg.artifacts``), never zeroed: the former spike
suppression lowered MVC references by up to 9 %.
The primitives come from ``jtsmm.emg`` so the offline chain and the causal
viewer/controller chain share one implementation of each step.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from jtsmm.emg.mains import measure_line_hz, remove_mains

from biomech.timebase import odd_window


def take_line_hz(uv: np.ndarray, fs_hz: float, t_us: np.ndarray | None = None,
                 nominal_hz: float = 50.0) -> float:
    """Grid frequency of one take, measured on its loudest channel.

    Estimate it once per take on a long stretch and pass it to :func:`clean`:
    on a short, clean window the free estimator latches onto a noise peak
    (48.7 Hz on s01's 10-s rest), and fitting there removes nothing.
    ``measure_line_hz`` returns None when the tone is not clearly present; the
    nominal frequency is then the better guess.
    """
    a = np.asarray(uv, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    loud = int(np.argmax(a.std(axis=0)))
    # The nominal grid first; the other one (50 <-> 60 Hz) only if the nominal tone is
    # absent, so a recording on a 60-Hz grid is not "cleaned" at 50 Hz.
    for nominal in (nominal_hz, 60.0 if round(nominal_hz) == 50 else 50.0):
        f = measure_line_hz(a[:, loud], fs_hz, nominal, t_us=t_us)
        if f is not None:
            return float(f)
    return float(nominal_hz)


def refine_line_hz(uv: np.ndarray, fs_hz: float, t_us: np.ndarray | None, take_hz: float,
                   max_dev_hz: float = 0.1) -> float:
    """The grid frequency in this window (a phase), falling back to the take's value.

    The take value is measured on the first minute only; the grid wanders (pilot:
    50.016 vs 49.986 Hz within an hour), and a fixed frequency leaves a residual
    that grows with the harmonic number (9x the error at 450 Hz). Synthetic
    check: 0.05 Hz drift over 5 min left 1.44 uV of residual hum late in the take
    with the take value, 0.55 uV (the no-drift floor) with the per-phase value.
    The window value is used only when its tone is clearly measurable and within
    ``max_dev_hz`` of the take value.
    """
    a = np.asarray(uv, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    loud = int(np.argmax(a.std(axis=0)))
    f = measure_line_hz(a[:, loud], fs_hz, take_hz, t_us=t_us, max_dev_hz=max_dev_hz)
    return float(f) if f is not None else float(take_hz)


def bandpass(uv: np.ndarray, fs_hz: float, band_hz: tuple[float, float] = (20.0, 450.0),
             order: int = 4) -> np.ndarray:
    """Zero-phase band-pass along axis 0; the upper edge is clipped below Nyquist."""
    a = np.asarray(uv, dtype=np.float64)
    ny = 0.5 * fs_hz
    lo, hi = max(band_hz[0], 0.1), min(band_hz[1], 0.98 * ny)
    if hi <= lo or a.shape[0] < 3 * (2 * order + 1):
        return a.copy()
    sos = butter(order, [lo / ny, hi / ny], btype="band", output="sos")
    return sosfiltfilt(sos, a, axis=0)


def rms_envelope(uv: np.ndarray, fs_hz: float, window_s: float = 0.15) -> np.ndarray:
    """Centred moving RMS along axis 0 (O(N), edges use nearest-value padding).

    The window is the odd sample count nearest ``window_s`` (``timebase.odd_window``), so
    the envelope is not shifted by half a sample (E17).
    """
    a = np.asarray(uv, dtype=np.float64)
    w = odd_window(window_s, fs_hz)
    return np.sqrt(np.maximum(uniform_filter1d(a * a, size=w, axis=0, mode="nearest"), 0.0))


def clean(uv: np.ndarray, fs_hz: float, t_us: np.ndarray | None = None, *,
          line_hz: float | None = None, band_hz: tuple[float, float] = (20.0, 450.0),
          order: int = 4, robust_mains: bool = True,
          amp_order: int | str = "auto") -> tuple[np.ndarray, float]:
    """Full offline cleaning of an (N, C) block; returns ``(cleaned_uv, line_hz)``.

    Pass ``line_hz`` from :func:`take_line_hz`; None re-estimates it from this block.
    ``robust_mains`` down-weights bursts in the harmonic fit (analysis_plan D8): an
    ordinary fit leaks burst energy back into quiet samples as a residual hum.
    """
    a = np.asarray(uv, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    demained, f0 = remove_mains(a, fs_hz, t_us=t_us, line_hz=line_hz, robust=robust_mains,
                                amp_order=amp_order)
    return bandpass(demained, fs_hz, band_hz, order), float(f0)
