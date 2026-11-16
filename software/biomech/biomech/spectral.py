# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Frequency-domain descriptors (exploratory, X1 (change log)).

* **EMG spectra** -- Welch PSD of the cleaned (20-450 Hz, mains-removed) EMG over
  windows free of artifact masks, per muscle and task; mean and median frequency
  (MNF, MDF), relative band powers. Spectral shape depends on electrode placement,
  inter-electrode distance and subcutaneous tissue as much as on the muscle, so
  between-participant spread of MNF vs the within-participant spread across windows
  quantifies how much a standardized electrode protocol matters (S2).
* **Movement bandwidth** -- PSD of the jaw angular velocity; the frequencies holding
  50/95/99 % of its power are the control bandwidth an exoskeleton must track (a
  measured replacement for "5 x chewing frequency"). Taken from the IMU (200 Hz
  grid) because the optical poses are low-passed at 6 Hz.

Everything is computed from windows, so per-window values give within-participant
variability and the spectra can be compared with SPM along the frequency axis.
"""

from __future__ import annotations

import numpy as np

EMG_BANDS = ((20.0, 50.0), (50.0, 100.0), (100.0, 200.0), (200.0, 450.0))


def welch_masked(x: np.ndarray, fs: float, mask: np.ndarray | None = None,
                 seg_s: float = 0.5, overlap: float = 0.5,
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Welch PSD per column over segments without masked/NaN samples.

    Returns ``(f, psd (C, F), per_segment (C, S, F))``; columns without any clean
    segment are NaN. Hann window, density scaling (units^2 / Hz).
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    n, C = a.shape
    m = np.zeros(a.shape, bool) if mask is None else np.asarray(mask, bool).reshape(a.shape)
    m = m | ~np.isfinite(a)
    L = max(int(round(seg_s * fs)), 16)
    hop = max(int(round(L * (1 - overlap))), 1)
    win = np.hanning(L)
    scale = 1.0 / (fs * np.sum(win ** 2))
    f = np.fft.rfftfreq(L, 1.0 / fs)
    starts = np.arange(0, max(n - L + 1, 0), hop)
    per = np.full((C, len(starts), f.size), np.nan)
    for c in range(C):
        for j, s in enumerate(starts):
            if m[s:s + L, c].any():
                continue
            seg = a[s:s + L, c]
            seg = seg - seg.mean()
            p = np.abs(np.fft.rfft(seg * win)) ** 2 * scale
            p[1:-1] *= 2.0
            per[c, j] = p
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            psd = np.nanmean(per, axis=1)
    return f, psd, per


def mean_median_freq(f: np.ndarray, psd: np.ndarray, band: tuple[float, float] = (20.0, 450.0)
                     ) -> tuple[np.ndarray, np.ndarray]:
    """MNF and MDF (Hz) of PSDs (..., F) within ``band``; NaN where the power is zero/NaN."""
    sel = (f >= band[0]) & (f <= band[1])
    fb, p = f[sel], np.asarray(psd, float)[..., sel]
    tot = np.nansum(p, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mnf = np.nansum(p * fb, axis=-1) / tot
        cum = np.nancumsum(p, axis=-1) / tot[..., None]
    idx = np.argmax(cum >= 0.5, axis=-1)
    mdf = fb[idx]
    bad = ~np.isfinite(tot) | (tot <= 0)
    return np.where(bad, np.nan, mnf), np.where(bad, np.nan, mdf)


def band_shares(f: np.ndarray, psd: np.ndarray, bands=EMG_BANDS) -> np.ndarray:
    """Relative power per band (..., len(bands)) within the union of the bands."""
    p = np.asarray(psd, float)
    total = np.nansum(p[..., (f >= bands[0][0]) & (f <= bands[-1][1])], axis=-1)
    out = []
    for lo, hi in bands:
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.nansum(p[..., (f >= lo) & (f < hi)], axis=-1) / total)
    return np.stack(out, axis=-1)


def power_quantiles(f: np.ndarray, psd: np.ndarray, qs=(0.5, 0.95, 0.99),
                    f_min: float = 0.0) -> np.ndarray:
    """Frequencies below which the given fractions of the power lie (cumulative PSD)."""
    sel = f >= f_min
    fb, p = f[sel], np.asarray(psd, float)[sel]
    tot = np.nansum(p)
    if not np.isfinite(tot) or tot <= 0:
        return np.full(len(qs), np.nan)
    cum = np.nancumsum(p) / tot
    return np.array([fb[min(int(np.argmax(cum >= q)), fb.size - 1)] for q in qs])


def log_norm_psd(psd: np.ndarray, f: np.ndarray, band: tuple[float, float] = (20.0, 450.0)
                 ) -> np.ndarray:
    """log10 of the PSD normalized to unit power in ``band`` (shape only, for comparisons)."""
    sel = (f >= band[0]) & (f <= band[1])
    p = np.asarray(psd, float)
    tot = np.nansum(p[..., sel], axis=-1, keepdims=True) * (f[1] - f[0])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(p / tot)
