# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Vectorized window features with one registry that is the real dispatch.

Every EMG feature is a function of a block of windows ``W`` with shape
``(..., L)`` (NaN already zeroed) and returns ``W.shape[:-1]``; the same
functions therefore serve whole windows ``(n, C, L)`` for SVM/MLP/LDA and
sub-frames ``(n, C, S, L/S)`` for the GRU. Names in configs and feature tables
are registry keys; an unknown name is an error listing the known ones.

All windows are trailing: a window ending at sample ``e`` uses ``e-L+1 .. e``.
Amplitude features are computed on EMG in % of the channel reference. ZC, SSC
and WAMP all threshold *sample differences* ``|x[k+1] - x[k]|`` (Hudgins), so
their thresholds are a multiple of the SD of the sample differences in the
participant's unlabelled rest phase (:func:`rest_thresholds`), not of the
amplitude SD.

Label-free per-participant normalization (analysis_plan D13,
:func:`rest_relative_stats` / :func:`apply_rest_relative`): amplitude features
become ``log(f) - median_rest log(f)``; count and spectral features are z-scored
with the mean and SD over the participant's rest-phase windows.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

EPS = 1e-6
EMG_BAND_HZ = (20.0, 450.0)
DEFAULT_EMG_FEATURES: tuple[str, ...] = (
    "log_mav", "log_rms", "log_wl", "zc", "ssc", "wamp",
    "band_20_50", "band_50_100", "band_100_200", "band_200_450", "mnf",
)
DEFAULT_SEQ_FEATURES: tuple[str, ...] = ("log_mav", "log_rms", "log_wl", "zc")
IMU_SIGNALS = ("w_mta_x", "w_mta_y", "w_mta_z", "w_cra_x", "w_cra_y", "w_cra_z",
               "a_mta_x", "a_mta_y", "a_mta_z", "a_cra_x", "a_cra_y", "a_cra_z")


class _Win:
    """Windows plus lazily shared intermediates (diff, spectrum)."""

    def __init__(self, W: np.ndarray, thr: np.ndarray, fs: float) -> None:
        self.W = W
        self.t = np.asarray(thr)[..., None]
        self.fs = fs

    @cached_property
    def d(self) -> np.ndarray:
        return np.diff(self.W, axis=-1)

    @cached_property
    def spec(self) -> tuple[np.ndarray, np.ndarray]:
        L = self.W.shape[-1]
        p = np.abs(np.fft.rfft(self.W * np.hanning(L), axis=-1)) ** 2
        return np.fft.rfftfreq(L, 1.0 / self.fs), p

    def band(self, lo: float, hi: float, *, closed: bool = False) -> np.ndarray:
        f, p = self.spec
        sel = (f >= lo) & ((f <= hi) if closed else (f < hi))
        return p[..., sel].sum(axis=-1)

    @cached_property
    def total(self) -> np.ndarray:
        return self.band(*EMG_BAND_HZ, closed=True) + EPS


def _mav(w: _Win) -> np.ndarray:
    return np.abs(w.W).mean(axis=-1)


def _rms(w: _Win) -> np.ndarray:
    return np.sqrt((w.W ** 2).mean(axis=-1))


def _wl(w: _Win) -> np.ndarray:
    return np.abs(w.d).mean(axis=-1)


# ZC: sign change AND |x[k+1] - x[k]| >= thr; SSC: slope sign change AND either
# adjacent |difference| >= thr; WAMP: |difference| >= thr. ``w.t`` is therefore on the
# scale of sample differences (rest-phase SD of np.diff), see rest_thresholds().
def _zc(w: _Win) -> np.ndarray:
    return ((w.W[..., :-1] * w.W[..., 1:] < 0) & (np.abs(w.d) >= w.t)).mean(axis=-1)


def _ssc(w: _Win) -> np.ndarray:
    d = w.d
    return (((d[..., :-1] * d[..., 1:]) < 0)
            & ((np.abs(d[..., :-1]) >= w.t) | (np.abs(d[..., 1:]) >= w.t))).mean(axis=-1)


def _wamp(w: _Win) -> np.ndarray:
    return (np.abs(w.d) >= w.t).mean(axis=-1)


def _mnf(w: _Win) -> np.ndarray:
    f, p = w.spec
    sel = (f >= EMG_BAND_HZ[0]) & (f <= EMG_BAND_HZ[1])
    return (p[..., sel] * f[sel]).sum(axis=-1) / w.total / 100.0


def _mdf(w: _Win) -> np.ndarray:
    f, p = w.spec
    sel = (f >= EMG_BAND_HZ[0]) & (f <= EMG_BAND_HZ[1])
    c = np.cumsum(p[..., sel], axis=-1)
    k = (c < 0.5 * c[..., -1:]).sum(axis=-1)
    return f[sel][np.minimum(k, sel.sum() - 1)] / 100.0


@dataclass(frozen=True)
class Feature:
    name: str
    fn: Callable[[_Win], np.ndarray]
    doc: str


REGISTRY: dict[str, Feature] = {f.name: f for f in (
    Feature("mav", _mav, "mean absolute value (% ref)"),
    Feature("log_mav", lambda w: np.log(_mav(w) + EPS), "log MAV"),
    Feature("rms", _rms, "root mean square (% ref)"),
    Feature("log_rms", lambda w: np.log(_rms(w) + EPS), "log RMS"),
    Feature("wl", _wl, "waveform length per sample"),
    Feature("log_wl", lambda w: np.log(_wl(w) + EPS), "log waveform length"),
    Feature("dasdv", lambda w: np.sqrt((w.d ** 2).mean(axis=-1)), "difference abs. SD"),
    Feature("log_ssi", lambda w: np.log((w.W ** 2).sum(axis=-1) + EPS),
            "log simple square integral"),
    Feature("zc", _zc, "zero-crossing rate above the rest threshold"),
    Feature("ssc", _ssc, "slope-sign-change rate above the rest threshold"),
    Feature("wamp", _wamp, "Willison amplitude rate (|difference| above the rest threshold)"),
    Feature("mnf", _mnf, "mean frequency 20-450 Hz (Hz/100)"),
    Feature("mdf", _mdf, "median frequency 20-450 Hz (Hz/100)"),
    Feature("log_ttp", lambda w: np.log(w.total), "log total power 20-450 Hz"),
)}
_BAND = re.compile(r"^band_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")


def get_feature(name: str) -> Feature:
    """Registry entry; ``band_<lo>_<hi>`` is relative power in [lo, hi) Hz."""
    if name in REGISTRY:
        return REGISTRY[name]
    m = _BAND.match(name)
    if m:
        lo, hi = float(m[1]), float(m[2])
        return Feature(name, lambda w, lo=lo, hi=hi: w.band(lo, hi) / w.total,
                       f"relative power {lo:g}-{hi:g} Hz")
    raise KeyError(f"unknown EMG feature {name!r}; known: {sorted(REGISTRY)} + band_<lo>_<hi>")


def compute(W: np.ndarray, thr: np.ndarray, fs: float, names: tuple[str, ...] | list[str]
            ) -> np.ndarray:
    """Stack ``names`` on a new last axis: ``W.shape[:-1] + (len(names),)``."""
    fns = [get_feature(n).fn for n in names]
    w = _Win(W, thr, fs)
    return np.stack([f(w) for f in fns], axis=-1)


# -- thresholds and label-free normalization ----------------------------------------------
def rest_thresholds(x_rest: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """(C,) ZC/SSC/WAMP thresholds: ``sigma`` x SD of the rest-phase sample differences.

    All three features compare ``|x[k+1] - x[k]|`` with the threshold, so its scale is
    that of the differences. (At 4 kHz the difference SD of 20-450 Hz noise is ~0.4 x
    its amplitude SD; an amplitude-SD threshold hardly ever fired at rest and the
    features were near-constant.) NaN (masked) samples are ignored.
    """
    x = np.asarray(x_rest, dtype=np.float64)
    if x.shape[0] < 3:
        raise ValueError("rest thresholds need at least 3 samples")
    return float(sigma) * np.nanstd(np.diff(x, axis=0), axis=0)


AMPLITUDE_LINEAR = frozenset({"mav", "rms", "wl", "dasdv"})
AMPLITUDE_LOG = frozenset({"log_mav", "log_rms", "log_wl", "log_ssi", "log_ttp"})
EMG_NORMS = ("rest_relative", "mvc")
SD_FLOOR = 1e-3        # count / spectral features are O(1) rates, Hz/100 or power shares


def feature_kind(name: str) -> str:
    """``amp`` (linear amplitude), ``log_amp`` (already log) or ``z`` (count / spectral).

    ``name`` is a registry key or a column name ``<channel>|<feature>``.
    """
    base = name.rsplit("|", 1)[-1]
    get_feature(base)
    if base in AMPLITUDE_LINEAR:
        return "amp"
    if base in AMPLITUDE_LOG:
        return "log_amp"
    return "z"


def to_norm_scale(X: np.ndarray, names: list[str]) -> np.ndarray:
    """Columns on the scale D13 centres: log for linear amplitude features, else as is."""
    kinds = [feature_kind(n) for n in names]
    X = np.asarray(X, dtype=np.float64).copy()
    for j, k in enumerate(kinds):
        if k == "amp":
            X[..., j] = np.log(X[..., j] + EPS)
    return X


def rest_relative_stats(X_rest: np.ndarray, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """(center, scale) per column from the participant's rest windows (labels unused).

    Amplitude columns: centre = median of the log feature, scale 1. Count / spectral
    columns: mean and SD (floored at ``SD_FLOOR``). ``X_rest`` is (n, F) or
    (n, S, F) (sub-frames pooled).
    """
    G = to_norm_scale(X_rest, names).reshape(-1, len(names))
    G = G[np.isfinite(G).all(axis=1)]
    if G.shape[0] < 2:
        raise ValueError(f"rest-relative normalization needs >= 2 valid rest windows, "
                         f"got {G.shape[0]}")
    amp = np.array([feature_kind(n) != "z" for n in names])
    center = np.where(amp, np.median(G, axis=0), G.mean(axis=0))
    scale = np.where(amp, 1.0, np.maximum(G.std(axis=0), SD_FLOOR))
    return center, scale


def apply_rest_relative(X: np.ndarray, names: list[str], center: np.ndarray,
                        scale: np.ndarray) -> np.ndarray:
    """``(to_norm_scale(X) - center) / scale`` on the last axis."""
    return (to_norm_scale(X, names) - center) / scale


# -- EMG ---------------------------------------------------------------------------------
def emg_feature_names(channels: list[str], names=DEFAULT_EMG_FEATURES) -> list[str]:
    return [f"{c}|{f}" for c in channels for f in names]


def _windows(x: np.ndarray, ends: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    ok = ends >= win - 1
    v = sliding_window_view(x, win, axis=0)          # (N - win + 1, C, win)
    return v[(ends - win + 1)[ok]], ok


def emg_window_features(x: np.ndarray, fs: float, ends: np.ndarray, win: int, thr: np.ndarray,
                        names=DEFAULT_EMG_FEATURES) -> tuple[np.ndarray, np.ndarray]:
    """Features for windows ending at sample indices ``ends`` (inclusive).

    @param x    (N, C) filtered EMG in % reference; NaN = masked
    @param thr  (C,) ZC/SSC/WAMP thresholds on the sample-difference scale (same units)
    @return (features (n_win, C * F) channel-major, masked fraction (n_win, C))
    """
    n, c = x.shape
    W, ok = _windows(x, ends, win)
    nanfrac = np.ones((ends.size, c))
    out = np.full((ends.size, c * len(names)), np.nan)
    if W.shape[0]:
        nanfrac[ok] = np.isnan(W).mean(axis=2)
        per = compute(np.nan_to_num(W), np.asarray(thr, float), fs, names)   # (n_ok, C, F)
        out[ok] = per.reshape(per.shape[0], -1)
    return out, nanfrac


def seq_feature_names(channels: list[str], names=DEFAULT_SEQ_FEATURES) -> list[str]:
    return emg_feature_names(channels, names)


def emg_sequence_features(x: np.ndarray, fs: float, ends: np.ndarray, win: int, thr: np.ndarray,
                          n_sub: int, names=DEFAULT_SEQ_FEATURES) -> np.ndarray:
    """(n_win, S, C * F) features of ``n_sub`` equal sub-frames per window (GRU input).

    The oldest ``win % n_sub`` samples are dropped so every sub-frame has the same
    length; sub-frame s = 0 is the oldest.
    """
    n, c = x.shape
    L = win // n_sub
    if L < 2:
        raise ValueError(f"{n_sub} sub-frames need windows of at least {2 * n_sub} samples")
    W, ok = _windows(x, ends, win)
    out = np.full((ends.size, n_sub, c * len(names)), np.nan)
    if W.shape[0]:
        S = np.nan_to_num(W[..., win - n_sub * L:]).reshape(W.shape[0], c, n_sub, L)
        per = compute(S, np.asarray(thr, float)[:, None], fs, names)          # (n, C, S, F)
        out[ok] = per.transpose(0, 2, 1, 3).reshape(per.shape[0], n_sub, -1)
    return out


# -- IMU ---------------------------------------------------------------------------------
def imu_feature_names() -> list[str]:
    return ([f"{s}|mean" for s in IMU_SIGNALS] + [f"{s}|sd" for s in IMU_SIGNALS]
            + ["w_mta_speed|mean"])


def imu_window_features(sig: np.ndarray, ends: np.ndarray, win: int) -> np.ndarray:
    """(n_win, 25) features from (N, 12) IMU signals on a regular grid."""
    W, ok = _windows(sig, ends, win)                 # (n_ok, 12, win)
    out = np.full((ends.size, 25), np.nan)
    if W.shape[0]:
        speed = np.linalg.norm(W[:, 0:3, :], axis=1).mean(axis=1, keepdims=True)
        out[ok] = np.concatenate([np.nanmean(W, axis=2), np.nanstd(W, axis=2), speed], axis=1)
    return out


def imu_sequence_features(sig: np.ndarray, ends: np.ndarray, win: int, n_sub: int) -> np.ndarray:
    """(n_win, S, 12) sub-frame means of the IMU signals, aligned with the EMG sub-frames."""
    L = max(win // n_sub, 1)
    W, ok = _windows(sig, ends, win)
    out = np.full((ends.size, n_sub, sig.shape[1]), np.nan)
    if W.shape[0]:
        S = W[..., win - n_sub * L:].reshape(W.shape[0], sig.shape[1], n_sub, L)
        out[ok] = S.mean(axis=3).transpose(0, 2, 1)
    return out
