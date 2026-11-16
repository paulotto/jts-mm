# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Decoder of the pseudo-online evaluation: fold models, calibration, smoothing (ml08 §3).

Models (per LOSO fold, i.e. per held-out participant): trained offline on the *other*
participants' stored ``ml_windows`` windows with the fold's hyper-parameters from the main
``ml_eval`` LOSO run (``best_params``; an inner search when missing) by
:func:`ml.protocols.evaluate.fit` -- the stage passes ``stages.ml_eval.max_train`` /
``inner_folds`` and the study seed, so this is the very model ``ml_eval`` scored. The held-out participant's windows come from the stream.

Calibration modes (the stream's own windows with their optical labels):

``none``       the fold model throughout;
``first``      a calibration block: the first ``first_k`` decodable labelled windows after the
               rest phase (the rest phase is the normalization block and holds REST only); the
               model is refit with them (weights as the few-shot variant,
               :func:`ml.protocols.variants.user_weight`: per window
               ``n_src / (n_classes * n_cal)``, n_cal = calibration windows of the refit) and used from ``guard_s`` after the
               last of them; scored on the windows after that;
``per_phase``  progressive guided calibration: the first ``per_phase_k`` windows of every
               structured phase; after each phase's k-th window the model is refit with the
               calibration windows of that and all *earlier* phases (``variants`` uses those of
               every phase at once, which in a stream would need windows of later phases);
               scored on each phase's windows from ``guard_s`` after its k-th window.

Refits are idealized as instantaneous (the fit time is reported). A model is a fixed function
of a window's features, so decoding the stored stream windows with the model in force at each
window's time gives the decisions an online decoder would have made.

Smoothing (:func:`smooth`), causal over the window decisions of a phase: majority over the last
``n`` decodable windows with hysteresis -- a movement class is entered when it holds at least
``enter`` of them (and more than the current class), and held while it holds at least ``hold``;
REST otherwise. Undecodable windows (masked, gap, non-finite) hold the current state. The state
starts at REST at every phase start (an operator marker, known online).
"""

from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, fields
from typing import Any

import numpy as np

from ml.config.classes import CLASS_COUNT, FREE_PHASE, REST_PHASE
from ml.core.dataset import WindowDataset
from ml.models import registry as reg
from ml.protocols import evaluate as ev
from ml.protocols import variants as va

REST = 0
MODES = ("none", "first", "per_phase")


@dataclass(frozen=True)
class Smoothing:
    n: int = 3
    enter: int = 2
    hold: int = 1

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Smoothing":
        d = dict(d or {})
        bad = sorted(set(d) - {f.name for f in fields(cls)})
        if bad:
            raise ValueError(f"unknown smoothing keys {bad}")
        s = cls(**{k: int(v) for k, v in d.items()})
        if not (1 <= s.hold <= s.enter <= s.n):
            raise ValueError(f"smoothing needs 1 <= hold <= enter <= n, got {s}")
        return s


@dataclass(frozen=True)
class CalibrationPlan:
    modes: tuple[str, ...] = MODES
    first_k: int = 100
    per_phase_k: int = 20
    guard_s: float = 0.35
    weight: str | float = "balanced"

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CalibrationPlan":
        d = dict(d or {})
        bad = sorted(set(d) - {f.name for f in fields(cls)})
        if bad:
            raise ValueError(f"unknown calibration keys {bad}")
        if "modes" in d:
            d["modes"] = tuple(d["modes"])
            for m in d["modes"]:
                if m not in MODES:
                    raise ValueError(f"calibration mode must be one of {MODES}, got {m!r}")
        return cls(**d)


# -- smoothing and events ------------------------------------------------------------------
def smooth(yhat: np.ndarray, valid: np.ndarray, segment: np.ndarray,
           sm: Smoothing = Smoothing()) -> np.ndarray:
    """Smoothed decision per window (see module doc); ``segment`` = phase per window."""
    yhat = np.asarray(yhat)
    out = np.full(yhat.size, REST, dtype=np.int64)
    hist: deque[int] = deque(maxlen=sm.n)
    state, prev = REST, None
    for i in range(yhat.size):
        if segment[i] != prev:
            hist.clear()
            state, prev = REST, segment[i]
        if valid[i] and 0 <= yhat[i] < CLASS_COUNT:
            hist.append(int(yhat[i]))
            cnt = np.bincount(np.asarray(hist), minlength=CLASS_COUNT)
            mov = cnt.copy()
            mov[REST] = -1
            cand = int(np.argmax(mov))
            if state != REST and cand != state and cnt[cand] >= sm.enter and cnt[cand] > cnt[state]:
                state = cand
            elif state != REST and cnt[state] < sm.hold:
                state = cand if cnt[cand] >= sm.enter else REST
            elif state == REST and cnt[cand] >= sm.enter:
                state = cand
        out[i] = state
    return out


def onsets(s: np.ndarray, segment: np.ndarray) -> np.ndarray:
    """Indices where the (smoothed) decision enters a movement class."""
    s = np.asarray(s)
    if s.size == 0:
        return np.zeros(0, dtype=np.int64)
    new_seg = np.r_[True, segment[1:] != segment[:-1]]
    change = np.r_[True, s[1:] != s[:-1]] | new_seg
    return np.flatnonzero(change & (s != REST))


# -- feature sets ------------------------------------------------------------------------------
def fset_columns(names: dict[str, list[str]], fset: str) -> tuple[np.ndarray, list[slice]]:
    """Columns of ``hstack([emg, imu])`` that :meth:`WindowDataset.matrix` selects for ``fset``
    (computed by that method on index-coded blocks, so the two can never differ)."""
    blocks, off = {}, 0
    for b in ("emg", "imu"):
        if b in names:
            blocks[b] = (off + np.arange(len(names[b]), dtype=np.float64))[None, :]
            off += len(names[b])
    one = np.zeros(1, dtype=np.int64)
    ds = WindowDataset(blocks=blocks, y=one, pid=np.array(["x"]), session=np.array(["x"]),
                       phase=np.array(["x"]), t_end_us=one, groups=one, names=dict(names))
    X, slices = ds.matrix(fset)
    return X[0].astype(np.int64), slices


def stream_matrix(X_emg: np.ndarray, X_imu: np.ndarray, names: dict[str, list[str]],
                  fset: str) -> tuple[np.ndarray, list[slice]]:
    cols, slices = fset_columns(names, fset)
    return np.hstack([X_emg, X_imu])[:, cols], slices


# -- models ------------------------------------------------------------------------------------
def fit_fold(X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, model: str,
             params: dict[str, Any] | None, slices: list[slice], seed: int,
             max_train: int | None, inner: int = 3, cache_dir: str | None = None
             ) -> tuple[Any, dict[str, Any]]:
    """The fold model (``ml_eval``'s when ``params`` are its ``best_params``; with
    ``cache_dir`` loaded from the model cache ``ml_eval`` filled, :mod:`ml.models.cache`)."""
    est, info = ev.fit(model, X, y, groups, seed=seed, inner=inner, max_train=max_train,
                       search=params is None, params=params or {}, slices=slices,
                       cache_dir=cache_dir)
    return est, {k: v for k, v in info.items() if k != "inner_macro_f1"}


def fit_calibrated(X: np.ndarray, y: np.ndarray, groups: np.ndarray, X_cal: np.ndarray,
                   y_cal: np.ndarray, *, unit: str, model: str, params: dict[str, Any],
                   slices: list[slice], seed: int, max_train: int | None,
                   weight: str | float = "balanced") -> Any:
    """Refit on the fold's training subsample plus the user's calibration windows
    (as :func:`ml.protocols.variants.few_shot_predictions`)."""
    src = ev.subsample(len(y), max_train, seed)
    Xs, ys, gs = X[src], y[src], groups[src]
    w_user = va.user_weight(weight, len(ys), np.unique(ys).size, len(y_cal))
    sw = np.r_[np.ones(len(ys)), np.full(len(y_cal), w_user)]
    est = reg.get(model).make(seed, slices=slices, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(np.vstack([Xs, X_cal]), np.r_[ys, y_cal],
                groups=np.concatenate([gs.astype(str), np.full(len(y_cal), str(unit))]),
                sample_weight=sw)
    return est


@dataclass
class ModePlan:
    """Model switches of one calibration mode: model i+1 (calibrated with ``cal[i]``) is in
    force for windows ending after ``t_switch[i]``; ``portion`` marks the scored windows."""

    mode: str
    t_switch: list[int]
    cal: list[np.ndarray]
    portion: np.ndarray
    applicable: bool = True


def plan(mode: str, t_end: np.ndarray, phase: np.ndarray, y: np.ndarray,
         decodable: np.ndarray, cp: CalibrationPlan) -> ModePlan:
    n = t_end.size
    structured = (phase != FREE_PHASE)
    cand = decodable & (y >= 0)
    order = np.argsort(t_end, kind="stable")
    if mode == "none":
        return ModePlan(mode, [], [], structured.copy())
    g = int(cp.guard_s * 1e6)
    if mode == "first":
        c = order[(cand & structured & (phase != REST_PHASE))[order]][:cp.first_k]
        if c.size < cp.first_k or cp.first_k <= 0:
            return ModePlan(mode, [], [], np.zeros(n, bool), applicable=False)
        ts = int(t_end[c[-1]]) + g
        return ModePlan(mode, [ts], [c], structured & (t_end > ts))
    if mode == "per_phase":
        portion = np.zeros(n, dtype=bool)
        switches, cals, acc = [], [], []
        for ph in dict.fromkeys(phase[order][structured[order]]):
            rows = order[(cand & (phase == ph))[order]]
            if rows.size < max(cp.per_phase_k, 1):
                continue
            c = rows[:cp.per_phase_k]
            acc.append(c)
            ts = int(t_end[c[-1]]) + g
            switches.append(ts)
            cals.append(np.concatenate(acc))
            portion |= (phase == ph) & (t_end > ts)
        return ModePlan(mode, switches, cals, portion, applicable=bool(switches))
    raise ValueError(f"unknown calibration mode {mode!r}")


def decode(models: list[Any], mp: ModePlan, X: np.ndarray, t_end: np.ndarray,
           decodable: np.ndarray) -> np.ndarray:
    """Raw window decisions (-1 = undecodable) with the model in force at each window."""
    out = np.full(t_end.size, -1, dtype=np.int64)
    which = np.searchsorted(np.asarray(mp.t_switch, np.int64), t_end, side="left")
    for i, m in enumerate(models):
        sel = decodable & (which == i)
        if sel.any():
            out[sel] = np.asarray(m.predict(X[sel]), dtype=np.int64)
    return out


__all__ = ["CalibrationPlan", "MODES", "ModePlan", "Smoothing", "decode", "fit_calibrated",
           "fit_fold", "fset_columns", "onsets", "plan", "smooth", "stream_matrix"]
