# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Fit and score models under the evaluation protocols (C3b, analysis_plan §5.5).

One implementation serves ``ml train`` and the biomech stage ``ml_eval``:

* :func:`fit` subsamples (optional), runs the grouped inner hyper-parameter search
  (``GroupKFold`` over the fold's inner groups, scored with the one macro-F1 over
  the classes present) and refits the best setting with the same groups.
* :func:`run` fits every outer fold (:mod:`ml.protocols.folds`), optionally in
  parallel, and returns predictions keyed by *row index* into the dataset.
* :func:`fold_scores` turns one fold into balanced accuracy, macro-F1, a fixed
  7-class confusion matrix, detection latency, false activations in the rest
  phase and free-motion scores.
* :func:`loso_many` runs the nested LOSO of several configurations in one process pool
  (every inner fit a task, longest first, refits queued as their fold's search completes;
  :mod:`ml.protocols.pool`) with the rows :func:`run` would give;
* :func:`loso`, :func:`blocked`, :func:`permutation` and :func:`summarize` give
  the tables of the biomech stage; ``loso(..., importance=n)`` adds block
  permutation importance (balanced-accuracy drop when one block's columns are
  permuted jointly across the test fold's rows, ``n`` repeats), split off into its
  own table by :func:`importance_table`.
"""

from __future__ import annotations

import hashlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, ParameterGrid

from ml.config.classes import CLASS_COUNT, REST_PHASE
from ml.core.dataset import WindowDataset
from ml.core.windows import label_runs
from ml.evaluation.metrics import balanced_accuracy, macro_f1
from ml.models import cache as model_cache
from ml.models import registry as reg
from ml.protocols import pool as pl
from ml.protocols.folds import Fold, make_folds

REST = 0


def resolve_jobs(n_jobs: int | str | None) -> int:
    """0, None, -1 or "auto": every core; otherwise the given count."""
    import os

    if n_jobs in (0, None, -1, "auto"):
        return max(os.cpu_count() or 1, 1)
    return max(int(n_jobs), 1)


# -- fitting ----------------------------------------------------------------------------
def grouped_search(model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   grid: dict[str, list[Any]], *, inner: int, seed: int,
                   slices: list[slice] | None = None, params: dict[str, Any] | None = None
                   ) -> tuple[dict[str, Any], float]:
    """Best grid point by mean inner macro-F1; GroupKFold over ``groups``.

    Serial; the parallel runs search all folds' grids in one pool instead
    (:func:`_flat_search`, same splits and choice)."""
    n_groups = np.unique(groups).size
    if not grid or n_groups < 2:
        return {}, float("nan")
    splits = list(GroupKFold(n_splits=min(int(inner), n_groups)).split(X, y, groups))
    points = list(ParameterGrid(grid))
    flat = [_search_task(X, y, groups, model, slices, seed, params, pt, tr, te)
            for pt in points for tr, te in splits]
    return _best_point(points, flat, len(splits))


def _best_point(points: list[dict[str, Any]], flat: list[float], n_splits: int
                ) -> tuple[dict[str, Any], float]:
    """First grid point (grid order) with the highest mean score over its splits."""
    best, best_score = {}, -np.inf
    for i, point in enumerate(points):
        m = float(np.mean(flat[i * n_splits:(i + 1) * n_splits]))
        if m > best_score:
            best, best_score = dict(point), m
    return best, best_score


def subsample(n: int, max_train: int | None, seed: int) -> np.ndarray:
    """Row indices :func:`fit` trains on: all rows, or ``max_train`` drawn at random
    (sorted) with ``default_rng(seed)``; shared with :mod:`ml.protocols.variants`."""
    if max_train and n > max_train:
        return np.sort(np.random.default_rng(seed).choice(n, int(max_train), replace=False))
    return np.arange(n)


def fit(model: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, *, seed: int = 42,
        inner: int = 3, max_train: int | None = None, search: bool = True,
        grid: dict[str, list[Any]] | None = None, params: dict[str, Any] | None = None,
        slices: list[slice] | None = None, probability: bool = False,
        cache_dir: str | None = None) -> tuple[Any, dict]:
    """Subsample, inner grid search, refit. :func:`run` with workers can do the search
    of all folds in one flat pool instead (``flat_search``, same result). ``cache_dir``:
    the refit goes through the model cache (:mod:`ml.models.cache`; same model, its
    original fit time in ``est.fit_time_s_``)."""
    spec = reg.get(model)
    want = "sequences" if X.ndim == 3 else "features"
    if spec.input != want:
        raise ValueError(f"model {model!r} takes {spec.input}, the feature set gives {want}")
    idx = subsample(len(y), max_train, seed)          # random subsample for speed
    X, y, groups = X[idx], y[idx], groups[idx]
    best, score = ({}, float("nan"))
    if search:
        best, score = grouped_search(model, X, y, groups, spec.grid if grid is None else grid,
                                     inner=inner, seed=seed, slices=slices, params=params)
    est = spec.make(seed, slices=slices, probability=probability, **{**(params or {}), **best})
    est, fit_s, hit = model_cache.fit(est, X, y, groups=groups, cache_dir=cache_dir)
    est.fit_time_s_, est.fit_cache_hit_ = fit_s, hit
    info = dict(best)
    if np.isfinite(score):
        info["inner_macro_f1"] = score
    return est, info


@dataclass
class FoldResult:
    fold: Fold
    y_pred: np.ndarray
    proba: np.ndarray | None
    free_pred: np.ndarray
    best_params: dict[str, Any]
    fit_time_s: float
    model: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


def block_importance(est: Any, X: np.ndarray, y: np.ndarray, slices: list[slice], *,
                     n_repeats: int, seed: int) -> list[float]:
    """Mean balanced accuracy per block with that block's columns permuted across rows.

    The rows of one block are shuffled jointly (within-block structure kept, its link to
    the label and to the other blocks broken); the model is not refit.
    """
    rng = np.random.default_rng(seed)
    out = []
    for s in slices:
        ba = []
        for _ in range(int(n_repeats)):
            Xp = X.copy()
            Xp[..., s] = X[rng.permutation(len(X))][..., s]
            ba.append(balanced_accuracy(y, est.predict(Xp)))
        out.append(float(np.mean(ba)))
    return out


def _run_fold(f: Fold, X: np.ndarray, y: np.ndarray, model: str, slices, kw: dict[str, Any],
              probability: bool, keep_model: bool, importance: int = 0,
              cache_dir: str | None = None) -> FoldResult | None:
    if np.unique(y[f.train]).size < 2:
        return None
    t0 = time.perf_counter()
    est, best = fit(model, X[f.train], y[f.train], f.groups, slices=slices,
                    probability=probability, cache_dir=cache_dir, **kw)
    dt = time.perf_counter() - t0
    proba = est.predict_proba(X[f.test]) if probability else None
    extra = {"fit_info": getattr(est, "fit_info_", None),
             "proba_method": getattr(est, "proba_method_", None)}
    if importance and slices and len(slices) > 1:
        extra["perm_ba"] = block_importance(est, X[f.test], y[f.test], slices,
                                            n_repeats=importance, seed=kw.get("seed", 42))
    return FoldResult(f, est.predict(X[f.test]), proba,
                      est.predict(X[f.free]) if f.free.size else np.zeros(0, dtype=int),
                      best, dt, est if keep_model else None, extra)


def run(ds: WindowDataset, fset: str, model: str, protocol: str, *, seed: int = 42,
        inner: int = 3, max_train: int | None = None, search: bool = True,
        grid: dict[str, list[Any]] | None = None, params: dict[str, Any] | None = None,
        probability: bool = False, keep_models: bool = False, n_jobs: int = 1, k: int = 5,
        guard_s: float = 0.35, y: np.ndarray | None = None, importance: int = 0,
        flat_search: bool = False, fold_params: dict[str, dict[str, Any]] | None = None
        ) -> list[FoldResult]:
    """All outer folds of ``protocol``; ``y`` overrides the labels (permutation test);
    ``importance`` > 0 adds block permutation importance (``extra["perm_ba"]``);
    ``fold_params`` (``{pid: params}``) fixes the hyper-parameters of the folds whose test
    subject is ``pid`` (merged over ``params``; with ``search=False``, e.g. blocked CV with
    the LOSO choices) and records them as the fold's ``best_params``.

    With workers, one task per fold (search + refit in the worker); ``flat_search``
    instead runs every fold's inner grid in one flat pool of fits, then the refits (same
    result; meant for machines with more cores than folds -- benchmark before enabling)."""
    X, slices = ds.matrix(fset)
    yy = ds.y if y is None else y
    folds = make_folds(ds, protocol, k=k, guard_s=guard_s)
    n_jobs = resolve_jobs(n_jobs)
    kw = dict(seed=seed, inner=inner, max_train=max_train, search=search, grid=grid,
              params=params)

    def fold_kw(f: Fold) -> dict[str, Any]:
        if not fold_params or f.pid not in fold_params:
            return kw
        return {**kw, "params": {**(params or {}), **fold_params[f.pid]}}

    def tag(r: FoldResult | None, f: Fold) -> FoldResult | None:
        if r is not None and fold_params and f.pid in fold_params and not search:
            r.best_params = dict(fold_params[f.pid])
        return r

    if n_jobs == 1:
        res = [tag(_run_fold(f, X, yy, model, slices, fold_kw(f), probability, keep_models,
                             importance), f) for f in folds]
        return [r for r in res if r is not None]
    spec = reg.get(model)
    g = spec.grid if grid is None else grid
    best: dict[int, tuple[dict[str, Any], float]] = {}
    with parallel_config(backend="loky", inner_max_num_threads=1):
        if search and g and flat_search:
            # The inner grid search of every fold as ONE flat pool of (fold, grid point,
            # inner split) fits: with one task per fold the run waited for the slowest fold
            # (12 folds on 24 cores ran no faster than on 6). Same subsample, splits and
            # choice (first best in grid order) as fit()/grouped_search(), so the result does
            # not depend on n_jobs.
            best = _flat_search(folds, X, yy, model, slices, g, seed=seed, inner=inner,
                                max_train=max_train, params=params, n_jobs=n_jobs)
        jobs = []
        for i, f in enumerate(folds):
            fkw = fold_kw(f)
            if i in best:
                fkw = {**fkw, "search": False,
                       "params": {**(fkw["params"] or {}), **best[i][0]}}
            jobs.append(delayed(_run_fold)(f, X, yy, model, slices, fkw, probability,
                                           keep_models, importance))
        res = Parallel(n_jobs=min(n_jobs, len(folds)))(jobs)
    out = []
    for i, r in enumerate(res):
        r = tag(r, folds[i])
        if r is None:
            continue
        if i in best:                       # report the grid choice as the search would
            choice, score = best[i]
            r.best_params = {**choice, **({"inner_macro_f1": score}
                                          if np.isfinite(score) else {})}
        out.append(r)
    return out


def _point_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str, slices,
                 seed: int, params: dict[str, Any] | None, point: dict[str, Any]) -> Any:
    """One inner-search fit: ``model`` at grid ``point`` on (X, y, groups)."""
    est = reg.get(model).make(seed, slices=slices, **{**(params or {}), **point})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X, y, groups=groups)
    return est


def _search_task(X: np.ndarray, y: np.ndarray, groups: np.ndarray, model: str, slices,
                 seed: int, params: dict[str, Any] | None, point: dict[str, Any],
                 tr: np.ndarray, te: np.ndarray) -> float:
    est = _point_model(X[tr], y[tr], groups[tr], model, slices, seed, params, point)
    return macro_f1(y[te], est.predict(X[te]))


def _flat_search(folds: list[Fold], X: np.ndarray, y: np.ndarray, model: str, slices,
                 grid: dict[str, list[Any]], *, seed: int, inner: int, max_train: int | None,
                 params: dict[str, Any] | None, n_jobs: int
                 ) -> dict[int, tuple[dict[str, Any], float]]:
    """fold index -> (best grid point, its mean inner macro-F1), as grouped_search."""
    points = list(ParameterGrid(grid))
    tasks, layout = [], {}
    for i, f in enumerate(folds):
        yt = y[f.train]
        if np.unique(yt).size < 2:
            continue
        idx = subsample(len(yt), max_train, seed)
        Xf, yf, gf = X[f.train][idx], yt[idx], f.groups[idx]
        n_groups = np.unique(gf).size
        if n_groups < 2:
            continue
        splits = list(GroupKFold(n_splits=min(int(inner), n_groups)).split(Xf, yf, gf))
        layout[i] = (len(points), len(splits), len(tasks))
        tasks += [delayed(_search_task)(Xf, yf, gf, model, slices, seed, params, pt, tr, te)
                  for pt in points for tr, te in splits]
    scores = Parallel(n_jobs=n_jobs)(tasks)
    return {i: _best_point(points, scores[start:start + n_pt * n_sp], n_sp)
            for i, (n_pt, n_sp, start) in layout.items()}


# -- scoring ----------------------------------------------------------------------------
def scores(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return {"balanced_accuracy": balanced_accuracy(y, p), "macro_f1": macro_f1(y, p),
            "n": int(len(y)),
            "confusion": confusion_matrix(y, p, labels=list(range(CLASS_COUNT)))}


def segments(t_us: np.ndarray, *keys: np.ndarray, stride_us: float | None = None,
             gap_strides: float = 2.0) -> np.ndarray:
    """Segment id per row: consecutive rows with equal ``keys`` (e.g. take and phase) and
    no window gap -- a step between window ends that is not positive or exceeds
    ``gap_strides`` x the stride (dropped or rejected windows) starts a new segment.
    ``stride_us`` defaults to the median positive step within equal keys."""
    t = np.asarray(t_us, dtype=np.int64)
    if t.size == 0:
        return np.zeros(0, dtype=np.int64)
    run = label_runs(np.zeros(t.size, dtype=np.int8), *keys)
    dt = np.diff(t)
    if stride_us is None:
        pos = dt[(dt > 0) & (run[1:] == run[:-1])]
        stride_us = float(np.median(pos)) if pos.size else np.inf
    cut = np.r_[False, (dt <= 0) | (dt > gap_strides * float(stride_us))]
    return run + np.cumsum(cut)


def latency(t_us: np.ndarray, y: np.ndarray, p: np.ndarray, *, pre_s: float = 0.5,
            confirm: int = 2, segment: np.ndarray | None = None,
            stride_us: float | None = None, gap_strides: float = 2.0) -> list[float]:
    """For each REST->class onset in ``y``: first time (relative to the onset, ms) at
    which the prediction equals that class for ``confirm`` consecutive windows,
    searching from ``pre_s`` before the onset to the end of that labelled segment.
    Negative values are anticipatory; NaN = never detected.

    Onsets, the search range and the confirmation run stay inside one segment
    (:func:`segments` of ``t_us`` and ``segment``, e.g. take and phase ids): rows that
    are adjacent in the table but separated by a phase boundary or a window gap
    (> ``gap_strides`` strides) are not adjacent in time, so a REST window before the
    boundary does not make the first window after it an onset.
    """
    keys = () if segment is None else (np.asarray(segment),)
    seg = segments(t_us, *keys, stride_us=stride_us, gap_strides=gap_strides)
    out = []
    same = seg[1:] == seg[:-1]
    on = np.flatnonzero((y[1:] != REST) & (y[:-1] == REST) & same) + 1
    seg_start = np.r_[0, np.flatnonzero(~same) + 1]
    for i in on:
        c = y[i]
        j_end = i
        while j_end < len(y) and y[j_end] == c and seg[j_end] == seg[i]:
            j_end += 1
        lo = seg_start[np.searchsorted(seg_start, i, side="right") - 1]
        j0 = int(lo) + int(np.searchsorted(t_us[lo:i + 1], t_us[i] - pre_s * 1e6))
        hit = np.nan
        run_len = 0
        for j in range(j0, j_end):
            run_len = run_len + 1 if p[j] == c else 0
            if run_len >= confirm:
                hit = (t_us[j - confirm + 1] - t_us[i]) / 1e3
                break
        out.append(hit)
    return out


def false_activations(t_us: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Rest-phase windows predicted as movement: share and events per minute."""
    if len(p) < 2:
        return {"fa_frac": np.nan, "fa_per_min": np.nan}
    act = p != REST
    events = int(np.sum(act[1:] & ~act[:-1]) + act[0])
    minutes = (t_us[-1] - t_us[0]) / 6e7
    return {"fa_frac": float(act.mean()), "fa_per_min": events / minutes if minutes > 0 else np.nan}


def fold_scores(ds: WindowDataset, r: FoldResult, y: np.ndarray | None = None) -> dict[str, Any]:
    yy = ds.y if y is None else y
    te = r.fold.test
    out: dict[str, Any] = scores(yy[te], r.y_pred)
    lat = latency(ds.t_end_us[te], yy[te], r.y_pred,
                  segment=label_runs(np.zeros(te.size, np.int8), ds.session[te], ds.phase[te]))
    out["latency_ms_median"] = float(np.nanmedian(lat)) if np.isfinite(lat).any() else np.nan
    out["detected_frac"] = float(np.mean(np.isfinite(lat))) if lat else np.nan
    rest = ds.phase[te] == REST_PHASE
    out.update(false_activations(ds.t_end_us[te][rest], r.y_pred[rest]) if rest.any()
               else {"fa_frac": np.nan, "fa_per_min": np.nan})
    if r.fold.free.size:
        fr = r.fold.free
        out["free_balanced_accuracy"] = balanced_accuracy(yy[fr], r.free_pred)
        out["free_macro_f1"] = macro_f1(yy[fr], r.free_pred)
    return out


# -- biomech stage tables ---------------------------------------------------------------
def loso(ds: WindowDataset, fset: str, model: str, *, max_train: int | None, seed: int = 42,
         inner: int = 3, n_jobs: int = 1, y: np.ndarray | None = None,
         importance: int = 0, flat_search: bool = False,
         cache_dir: str | None = None) -> list[dict[str, Any]]:
    """Nested leave-one-subject-out, one row per subject (``max_train`` from the caller,
    ``stages.ml_eval.max_train``; None = no subsample) (``perm_ba|<block>`` keys with
    ``importance`` > 0 and a multi-block feature set). ``_pred`` holds the subject's
    window-level decoder output (``t_end_us``, true and predicted class), dropped from the
    table by :func:`summarize`. One configuration of :func:`loso_many` (``flat_search`` is
    kept for callers; every search is flat now)."""
    return loso_many(ds, [(model, fset, importance)], max_train=max_train, seed=seed,
                     inner=inner, n_jobs=n_jobs, y=y,
                     cache_models={(model, fset)} if cache_dir else (),
                     cache_dir=cache_dir)[0]


# Expected cost of one SVM fit + prediction relative to the others (C, gamma_mult), measured
# on a real emg_imu fold (8.5k training rows); only the scheduling order depends on it.
_SVM_COST = {(0.01, 0.25): 5.8, (0.01, 1.0): 5.2, (0.01, 4.0): 6.2, (0.1, 0.25): 3.6,
             (0.1, 1.0): 3.3, (0.1, 4.0): 4.6, (1.0, 0.25): 2.6, (1.0, 1.0): 2.6,
             (1.0, 4.0): 3.8, (10.0, 0.25): 2.2, (10.0, 1.0): 2.2, (10.0, 4.0): 4.3}


def fit_cost(model: str, n: int, d: int, params: dict[str, Any] | None = None) -> float:
    """Rough seconds of one fit (+ prediction) on ``n`` rows x ``d`` features (scheduling)."""
    params = params or {}
    scale = (n / 8500.0) ** 2 * max(d, 1) / 91.0
    base = model.split(":")[-1]
    if base == "svm":
        try:
            w = _SVM_COST.get((float(params.get("C", 1.0)),
                               float(params.get("gamma_mult", 1.0))), 4.0)
        except (TypeError, ValueError):
            w = 4.0
        return w * scale * (4.0 if model.startswith("late_") else 1.0)
    if base == "mlp":
        return 0.4 * n / 8500.0
    if base == "gru":
        return 30.0 * n / 8500.0
    return 0.02 * n / 8500.0 * max(d, 1) / 91.0


def _search_fit(xref: pl.Ref, yref: pl.Ref, train: np.ndarray, groups: np.ndarray, model: str,
                slices, seed: int, params: dict[str, Any] | None, point: dict[str, Any],
                tests: list[np.ndarray]) -> list[float]:
    """One inner fit (worker) on the rows ``train`` of the shared matrix (inner groups
    ``groups``), scored on each row set of ``tests``: the inner splits of several outer folds
    that train on the very same rows share it (:func:`loso_many`)."""
    X, y = xref.get(), yref.get()
    est = _point_model(X[train], y[train], groups, model, slices, seed, params, point)
    return [macro_f1(y[te], est.predict(X[te])) for te in tests]


def _inner_fit_key(train: np.ndarray) -> bytes:
    """Hash of an inner training row set (the duplicate check compares the rows too)."""
    return hashlib.blake2b(np.ascontiguousarray(train, dtype=np.int64).tobytes(),
                           digest_size=16).digest()


def _refit(xref: pl.Ref, yref: pl.Ref, f: Fold, model: str, slices, kw: dict[str, Any],
           importance: int, cache_dir: str | None) -> tuple | None:
    """The fold's refit and test predictions (worker); the fold stays with the caller."""
    r = _run_fold(f, xref.get(), yref.get(), model, slices, kw, False, False, importance,
                  cache_dir=cache_dir)
    return None if r is None else (r.y_pred, r.proba, r.free_pred, r.best_params,
                                   r.fit_time_s, r.extra)


def loso_many(ds: WindowDataset, configs: list[tuple[str, str, int]], *,
              max_train: int | None, seed: int = 42, inner: int = 3, n_jobs: int = 1,
              y: np.ndarray | None = None, cancelled: Callable[[], bool] | None = None,
              cache_dir: str | None = None, cache_models: Any = ()
              ) -> list[list[dict[str, Any]]]:
    """Nested LOSO of several ``(model, feature set, importance)`` configurations in ONE pool.

    Every inner fit (configuration x fold x grid point x inner split) is a task; a fold's
    refit (+ test predictions, block importance) is queued as soon as its last inner fit is
    in, with the grid choice of :func:`grouped_search` (first best mean in grid order).
    Tasks run longest expected first (:func:`fit_cost`) with no barrier between folds or
    configurations; workers read the feature matrices memory-mapped (:class:`pool.Shared`).
    Inner splits of different outer folds that train on the very same rows (same subject
    set, same subsample) with the same grid point are one fit, scored on each fold's inner
    test rows: the same computation, done once.
    The rows equal those of :func:`run` / ``fit`` fold by fold (same subsample, splits,
    choice and refit), for any ``n_jobs``. ``cache_models``: the ``(model, feature set)``
    pairs whose fold models go to the model cache ``cache_dir`` (``ml_online`` reuses them).
    Returns one row list per configuration (as :func:`loso`)."""
    yy = ds.y if y is None else np.asarray(y)
    folds = make_folds(ds, "loso_subject")
    n_jobs = resolve_jobs(n_jobs)
    want_cache = {(str(m), str(f)) for m, f in cache_models}
    results: dict[tuple[int, int], FoldResult | None] = {}
    with pl.Shared(prefix="ml_eval_") as sh:
        yref = sh.put(np.asarray(yy))
        plans: dict[int, dict[str, Any]] = {}
        tasks: list[pl.Task] = []
        pending: dict[tuple[int, int], dict[str, Any]] = {}
        # one search fit per (configuration, grid point, inner training rows): its requesters
        fits: dict[tuple[int, int, bytes], list[dict[str, Any]]] = {}
        requesters: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
        for ci, (model, fset, imp) in enumerate(configs):
            spec = reg.get(model)
            want = "sequences" if spec.input == "sequences" else "features"
            X, slices = ds.matrix(fset)
            if want != ("sequences" if X.ndim == 3 else "features"):
                raise ValueError(f"model {model!r} takes {spec.input}, the feature set gives "
                                 f"{'sequences' if X.ndim == 3 else 'features'}")
            xref = sh.put(X)
            cdir = cache_dir if (model, fset) in want_cache else None
            plans[ci] = dict(model=model, fset=fset, imp=int(imp), xref=xref, slices=slices,
                             cdir=cdir, d=int(X.shape[-1]))
            points = list(ParameterGrid(spec.grid)) if spec.grid else []
            for fi, f in enumerate(folds):
                yt = yy[f.train]
                if np.unique(yt).size < 2:
                    results[(ci, fi)] = None
                    continue
                idx = subsample(len(yt), max_train, seed)
                n_fit = int(idx.size)
                n_groups = np.unique(f.groups[idx]).size
                kw = dict(seed=seed, inner=inner, max_train=max_train, search=True,
                          grid=None, params=None)
                if not points or n_groups < 2:
                    # nothing to search (fit() returns its own empty choice)
                    tasks.append(pl.Task(("refit", ci, fi), _refit,
                                         (xref, yref, f, model, slices, kw, int(imp), cdir),
                                         cost=fit_cost(model, n_fit, plans[ci]["d"])))
                    continue
                rows, gf = f.train[idx], f.groups[idx]
                splits = list(GroupKFold(n_splits=min(int(inner), n_groups))
                              .split(rows, yt[idx], gf))
                pending[(ci, fi)] = {"points": points, "n_splits": len(splits),
                                     "scores": {}, "n_fit": n_fit}
                for pi, pt in enumerate(points):
                    for si, (tr, te) in enumerate(splits):
                        train = rows[tr]
                        same = fits.setdefault((ci, pi, _inner_fit_key(train)), [])
                        ft = next((e for e in same if np.array_equal(e["train"], train)), None)
                        if ft is None:
                            ft = {"gid": len(requesters), "train": train, "groups": gf[tr],
                                  "pt": pt, "tests": []}
                            same.append(ft)
                            requesters[(ci, ft["gid"])] = []
                        ft["tests"].append(rows[te])
                        requesters[(ci, ft["gid"])].append((fi, pi, si))
        for (ci, pi, _), ft in ((k, e) for k, es in fits.items() for e in es):
            pc = plans[ci]
            tasks.append(pl.Task(("search", ci, ft["gid"]), _search_fit,
                                 (pc["xref"], yref, ft["train"], ft["groups"], pc["model"],
                                  pc["slices"], seed, None, ft["pt"], ft["tests"]),
                                 cost=fit_cost(pc["model"], len(ft["train"]), pc["d"],
                                               ft["pt"])))

        def on_result(key, r):
            if key[0] == "refit":
                return None
            _, ci, gid = key
            new = []
            for (fi, pi, si), score in zip(requesters[(ci, gid)], r):
                st = pending[(ci, fi)]
                st["scores"][(pi, si)] = score
                if len(st["scores"]) < len(st["points"]) * st["n_splits"]:
                    continue
                flat = [st["scores"][(pi_, si_)] for pi_ in range(len(st["points"]))
                        for si_ in range(st["n_splits"])]
                best, score = _best_point(st["points"], flat, st["n_splits"])
                st["best"] = (best, score)
                pc = plans[ci]
                kw = dict(seed=seed, inner=inner, max_train=max_train, search=False,
                          grid=None, params=dict(best))
                new.append(pl.Task(("refit", ci, fi), _refit,
                                   (pc["xref"], yref, folds[fi], pc["model"], pc["slices"], kw,
                                    pc["imp"], pc["cdir"]),
                                   cost=1.5 * fit_cost(pc["model"], st["n_fit"], pc["d"],
                                                       best)))
            return new

        out = pl.run_tasks(tasks, n_jobs, on_result=on_result, cancelled=cancelled)
    for (ci, fi), st in pending.items():
        r = out[("refit", ci, fi)]
        if r is None:
            results[(ci, fi)] = None
            continue
        choice, score = st["best"]
        y_pred, proba, free_pred, _, dt, extra = r
        results[(ci, fi)] = FoldResult(folds[fi], y_pred, proba, free_pred,
                                       {**choice, **({"inner_macro_f1": score}
                                                     if np.isfinite(score) else {})},
                                       dt, None, extra)
    for key, r in out.items():
        if key[0] == "refit" and (key[1], key[2]) not in pending:
            results[(key[1], key[2])] = None if r is None else FoldResult(
                folds[key[2]], r[0], r[1], r[2], r[3], r[4], None, r[5])
    return [_loso_rows(ds, plans[ci]["fset"], plans[ci]["model"],
                       [results[(ci, fi)] for fi in range(len(folds))
                        if results.get((ci, fi)) is not None],
                       y=y, max_train=max_train, seed=seed)
            for ci in range(len(configs))]


def _loso_rows(ds: WindowDataset, fset: str, model: str, res: list[FoldResult], *,
               y: np.ndarray | None, max_train: int | None, seed: int) -> list[dict[str, Any]]:
    from ml.core.dataset import feature_set_blocks

    blocks = feature_set_blocks(fset)
    rows = []
    for r in res:
        row = {"pid": r.fold.unit, "feature_set": fset, "model": model,
               "best_params": str({k: v for k, v in r.best_params.items()
                                   if k != "inner_macro_f1"}),
               **fold_scores(ds, r, y)}
        for b, v in zip(blocks, r.extra.get("perm_ba") or []):
            row[f"perm_ba|{b}"] = v
        yy = ds.y if y is None else y
        row["_pred"] = {"t_end_us": ds.t_end_us[r.fold.test], "y": yy[r.fold.test],
                        "yhat": np.asarray(r.y_pred), "rows": r.fold.test,
                        "max_train": max_train, "seed": seed}
        rows.append(row)
    return rows


def importance_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Pop the ``perm_ba|<block>`` keys of LOSO rows into a (pid, block) table, in place."""
    out = []
    for r in rows:
        for k in [k for k in r if k.startswith("perm_ba|")]:
            v = r.pop(k)
            out.append({"pid": r["pid"], "feature_set": r["feature_set"], "model": r["model"],
                        "block": k.split("|", 1)[1], "balanced_accuracy": r["balanced_accuracy"],
                        "ba_permuted": v, "ba_drop": r["balanced_accuracy"] - v})
    return pd.DataFrame(out)


def blocked(ds: WindowDataset, fset: str, model: str, *, k: int = 5, guard_s: float = 0.35,
            seed: int = 42, n_jobs: int = 1,
            params_by_pid: dict[str, dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Within-subject blocked CV, one row per subject, no inner search.

    ``params_by_pid`` (``{pid: params}``): the hyper-parameters of each subject's folds --
    in ``ml_eval`` those the nested LOSO search chose for the fold that held that subject
    out (like-for-like with LOSO, and chosen without the subject's data); ``hyper`` =
    ``loso`` for those subjects. Without a choice the model defaults apply (``hyper`` =
    ``default``; the r9 protocol, kept as a sensitivity).

    One configuration of :func:`blocked_many`; every fold's fit is deterministic, so the
    table does not depend on ``n_jobs``."""
    return blocked_many(ds, [(fset, model, params_by_pid)], k=k, guard_s=guard_s, seed=seed,
                        n_jobs=n_jobs)[0]


def blocked_many(ds: WindowDataset,
                 configs: list[tuple[str, str, dict[str, dict[str, Any]] | None]], *,
                 k: int = 5, guard_s: float = 0.35, seed: int = 42, n_jobs: int = 1,
                 cancelled: Callable[[], bool] | None = None) -> list[list[dict[str, Any]]]:
    """Blocked CV (:func:`blocked`) of several ``(feature set, model, params_by_pid)``
    configurations in ONE pool (:mod:`ml.protocols.pool`): every fold fit is a task, longest
    expected first, no barrier between configurations. A fold fit that two configurations
    share -- same feature set, model, fold and hyper-parameters (e.g. ``ml_eval``'s LOSO
    choice of a model without a grid, ``{}``, and the defaults) -- runs once. Returns one row
    list per configuration (as :func:`blocked`)."""
    folds = make_folds(ds, "blocked", k=k, guard_s=guard_s)
    n_jobs = resolve_jobs(n_jobs)
    jobs: dict[tuple, pl.Task] = {}
    keys: list[list[tuple]] = []
    with pl.Shared(prefix="ml_blocked_") as sh:
        yref = sh.put(np.asarray(ds.y))
        for fset, model, by_pid in configs:
            X, slices = ds.matrix(fset)
            xref = sh.put(X)
            ck = []
            for fi, f in enumerate(folds):
                p = dict((by_pid or {}).get(f.pid) or {})
                key = ("fold", fset, model, fi, repr(sorted(p.items())))
                if key not in jobs:
                    kw = dict(seed=seed, inner=3, max_train=None, search=False, grid=None,
                              params=p or None)
                    jobs[key] = pl.Task(key, _refit, (xref, yref, f, model, slices, kw, 0, None),
                                        cost=fit_cost(model, int(f.train.size),
                                                      int(X.shape[-1]), p))
                ck.append(key)
            keys.append(ck)
        out = pl.run_tasks(list(jobs.values()), n_jobs, cancelled=cancelled)
    tables = []
    for (fset, model, by_pid), ck in zip(configs, keys):
        res = [(f, out[key]) for f, key in zip(folds, ck) if out[key] is not None]
        rows = []
        for pid in sorted({f.pid for f, _ in res}):
            rs = [(f, r) for f, r in res if f.pid == pid]
            sc = scores(np.concatenate([ds.y[f.test] for f, _ in rs]),
                        np.concatenate([r[0] for _, r in rs]))
            chosen = by_pid is not None and pid in by_pid
            rows.append({"pid": pid, "feature_set": fset, "model": model,
                         "balanced_accuracy": sc["balanced_accuracy"],
                         "macro_f1": sc["macro_f1"], "n": sc["n"],
                         "hyper": "loso" if chosen else "default",
                         "best_params": str(by_pid[pid]) if chosen else "{}"})
        tables.append(rows)
    return tables


def shift_labels(ds: WindowDataset, rng: np.random.Generator) -> np.ndarray:
    """Each subject's label sequence circularly shifted (keeps label autocorrelation)."""
    y = ds.y.copy()
    for pid in np.unique(ds.pid):
        idx = np.flatnonzero(ds.pid == pid)
        y[idx] = np.roll(ds.y[idx], int(rng.integers(len(idx) // 4, 3 * len(idx) // 4)))
    return y


def permutation(ds: WindowDataset, fset: str, model: str, n_perm: int, *, inner: int,
                max_train: int | None, seed: int = 42, n_jobs: int = 1,
                cancelled: Callable[[], bool] | None = None) -> np.ndarray:
    """Null distribution of the cohort mean LOSO balanced accuracy (structured tasks).

    ``inner`` and ``max_train`` must be those of the observed run (``ml_eval``), so the
    null models are fitted exactly like the observed ones. With ``cancelled`` the
    permutations run in batches and ``RuntimeError("cancelled")`` is raised between
    them (the label shifts are drawn up front, so batching does not change the null)."""
    rng = np.random.default_rng(seed)

    def one(kk: int, y_perm: np.ndarray) -> float:
        rs = run(ds, fset, model, "loso_subject", seed=seed + kk, inner=inner,
                 max_train=max_train, y=y_perm)
        return float(np.mean([balanced_accuracy(y_perm[r.fold.test], r.y_pred) for r in rs]))

    perms = [shift_labels(ds, rng) for _ in range(n_perm)]
    jobs = resolve_jobs(n_jobs)
    step = len(perms) if cancelled is None else max(4 * jobs, 1)
    out: list[float] = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        for b0 in range(0, len(perms), step):
            if cancelled is not None and cancelled():
                raise RuntimeError("cancelled")
            out += Parallel(n_jobs=jobs)(delayed(one)(kk, perms[kk])
                                         for kk in range(b0, min(b0 + step, len(perms))))
    return np.array(out)


def summarize(rows: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "confusion"
                        and not k.startswith("_")} for r in rows])
    conf: dict[str, np.ndarray] = {}
    for r in rows:
        if "confusion" in r:
            key = f"{r['feature_set']}|{r['model']}"
            conf[key] = conf.get(key, 0) + r["confusion"]
    return df, conf
