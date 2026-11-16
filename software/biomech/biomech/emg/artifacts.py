# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sample-level EMG artifact masks (analysis_plan §3.1). Mask, never zero.

A masked sample is excluded (NaN) from every statistic; nothing is replaced by
an invented value. Rules, per channel:

* ``rail``     raw sample within 1 % of the ADC full scale (saturation);
* ``leadoff``  the channel's lead-off comparator bit is set;
* ``envelope`` RMS envelope above ``env_factor`` x the channel's own MVC/RVC
               reference (p95), i.e. far beyond any voluntary level;
* ``lowfreq``  1-20 Hz raw level above ``lf_factor`` x the channel's rest level
               AND above ``lf_ratio`` x the concurrent 20-450 Hz envelope:
               movement artifacts (electrode/skin motion) live below 20 Hz,
               EMG does not. The ratio condition (Amendment 1) keeps ordinary
               movement, which raises the 1-20 Hz level above rest everywhere,
               from masking 30-65 % of task data; at ratio 3 the residual
               leakage through the 20 Hz high-pass reaches ~25 % of the EMG.

Roles (analysis_plan D14): every rule is computed and kept in ``MaskResult.hits``, but
only ``analysis_rules`` (default rail, lead-off, envelope: physically missing, saturated or
grossly corrupted samples) form the analysis mask that excludes samples from statistics.
The low-frequency rule is not standard practice (a 20 Hz high-pass is, De Luca et al. 2010)
and a check found no contamination in the samples it hit (an03 §2), so by default it is a
QC flag and a sensitivity analysis (``analysis_rules`` including ``lowfreq``).

The analysis mask is dilated by ``dilate_s`` on both sides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

from biomech.emg.pipeline import rms_envelope
from biomech.timebase import odd_window

RULES = ("rail", "leadoff", "envelope", "lowfreq")
ANALYSIS_RULES = ("rail", "leadoff", "envelope")       # D14 default


def dilate_rows(m: np.ndarray, k: int) -> np.ndarray:
    """Grow True runs by ``k`` samples on both sides along axis 0.

    Same result as ``scipy.ndimage.binary_dilation`` with a (2k+1, 1) box element, but
    O(N) via a cumulative sum instead of O(N k).
    """
    m = np.asarray(m, dtype=bool)
    if k <= 0 or not m.any():
        return m.copy()
    n = m.shape[0]
    c = np.zeros((n + 1,) + m.shape[1:], dtype=np.int64)
    np.cumsum(m, axis=0, out=c[1:])
    i = np.arange(n)
    return (c[np.minimum(i + k + 1, n)] - c[np.maximum(i - k, 0)]) > 0


@dataclass(frozen=True)
class MaskParams:
    env_factor: float = 3.0
    lf_factor: float = 10.0
    lf_ratio: float = 3.0
    lf_band_hz: tuple[float, float] = (1.0, 20.0)
    lf_window_s: float = 0.25
    rail_frac: float = 0.99
    dilate_s: float = 0.1
    envelope_s: float = 0.15
    analysis_rules: tuple[str, ...] = ANALYSIS_RULES

    def __post_init__(self) -> None:
        bad = set(self.analysis_rules) - set(RULES)
        if bad:
            raise ValueError(f"unknown mask rules {sorted(bad)}; known: {RULES}")


@dataclass
class MaskResult:
    mask: np.ndarray                                  # (N, C) bool, True = excluded (analysis)
    counts: dict[str, np.ndarray] = field(default_factory=dict)  # rule -> (C,) samples hit
    hits: dict[str, np.ndarray] = field(default_factory=dict)    # rule -> (N, C), undilated
    analysis_rules: tuple[str, ...] = ANALYSIS_RULES

    def any_hit(self) -> np.ndarray:
        """Samples hit by any rule, analysis or flag-only (the rule hits undilated), or inside
        the analysis mask, which is dilated (±``dilate_s``)."""
        m = np.zeros_like(self.mask)
        for h in self.hits.values():
            m |= h
        return m | self.mask

    def flagged(self, rule: str) -> np.ndarray:
        """Hits of a rule that are not excluded by the analysis mask (QC flag only)."""
        h = self.hits.get(rule)
        return np.zeros_like(self.mask) if h is None else h & ~self.mask

    def rules(self, *names: str, dilate: int = 0) -> np.ndarray:
        """Mask from a subset of rules (e.g. only physically missing data)."""
        m = np.zeros_like(self.mask)
        for n in names:
            if n in self.hits:
                m |= self.hits[n]
        return dilate_rows(m, dilate) if dilate else m


def lowfreq_level(raw_uv: np.ndarray, fs_hz: float, p: MaskParams = MaskParams()) -> np.ndarray:
    """Moving RMS (N, C) of the 1-20 Hz band of the raw signal."""
    a = np.asarray(raw_uv, dtype=np.float64)
    a = a - np.median(a, axis=0)
    ny = 0.5 * fs_hz
    if a.shape[0] < 64:
        return np.zeros_like(a)
    sos = butter(2, [p.lf_band_hz[0] / ny, p.lf_band_hz[1] / ny], btype="band", output="sos")
    lf = sosfiltfilt(sos, a, axis=0)
    w = odd_window(p.lf_window_s, fs_hz)              # odd: centred, no half-sample shift
    return np.sqrt(np.maximum(uniform_filter1d(lf * lf, size=w, axis=0, mode="nearest"), 0.0))


def artifact_mask(
    raw_uv: np.ndarray,
    cleaned_uv: np.ndarray,
    fs_hz: float,
    *,
    full_scale_uv: np.ndarray | None = None,
    leadoff: np.ndarray | None = None,
    ref_uv: np.ndarray | None = None,
    rest_lf_uv: np.ndarray | None = None,
    p: MaskParams = MaskParams(),
) -> MaskResult:
    raw = np.asarray(raw_uv, dtype=np.float64)
    n, c = raw.shape
    hits: dict[str, np.ndarray] = {}
    if full_scale_uv is not None:
        hits["rail"] = np.abs(raw) >= p.rail_frac * np.asarray(full_scale_uv)[None, :]
    if leadoff is not None and leadoff.shape == raw.shape:
        hits["leadoff"] = np.asarray(leadoff, dtype=bool)
    if ref_uv is not None:
        env = rms_envelope(cleaned_uv, fs_hz, p.envelope_s)
        ref = np.asarray(ref_uv, dtype=np.float64)
        lim = np.where(np.isfinite(ref) & (ref > 0), p.env_factor * ref, np.inf)
        hits["envelope"] = env > lim[None, :]
    if rest_lf_uv is not None:
        lvl = np.asarray(rest_lf_uv, dtype=np.float64)
        lim = np.where(np.isfinite(lvl) & (lvl > 0), p.lf_factor * lvl, np.inf)
        lf = lowfreq_level(raw, fs_hz, p)
        inband = rms_envelope(cleaned_uv, fs_hz, p.lf_window_s)
        hits["lowfreq"] = (lf > lim[None, :]) & (lf > p.lf_ratio * inband)
    mask = np.zeros((n, c), dtype=bool)
    for r, m in hits.items():
        if r in p.analysis_rules:
            mask |= m
    mask = dilate_rows(mask, max(int(round(p.dilate_s * fs_hz)), 0))
    return MaskResult(mask=mask, counts={r: m.sum(axis=0) for r, m in hits.items()}, hits=hits,
                      analysis_rules=tuple(p.analysis_rules))


def masked_envelope(cleaned_uv: np.ndarray, mask: np.ndarray, fs_hz: float,
                    window_s: float = 0.15, min_valid: float = 0.5) -> np.ndarray:
    """Centred moving RMS over unmasked samples; NaN where < ``min_valid`` of the window is valid."""
    x = np.asarray(cleaned_uv, dtype=np.float64)
    valid = ~np.asarray(mask, dtype=bool) & np.isfinite(x)
    sq = np.where(valid, x * x, 0.0)
    w = odd_window(window_s, fs_hz)                   # odd: centred, no half-sample shift
    num = uniform_filter1d(sq, size=w, axis=0, mode="nearest")
    den = uniform_filter1d(valid.astype(np.float64), size=w, axis=0, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        env = np.sqrt(num / den)
    env[den < min_valid] = np.nan
    return env
