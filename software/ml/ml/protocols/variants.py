# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Exploratory decoding variants on the LOSO folds (biomech ``ml_eval``, ml05 §6.1).

All variants use the SVM and the *same* ``loso_subject`` folds as the main run
(:func:`ml.protocols.folds.make_folds`), the same training subsample
(:func:`ml.protocols.evaluate.subsample`, ``max_train``) and, with ``hyper: main``
(default), the hyper-parameters the main nested LOSO chose for that fold (the
``best_params`` of the ``loso`` rows, :func:`main_params`); a fold without such a
choice gets one grouped inner search (``hyper: search`` forces it everywhere).

``late_fusion``
    one SVM on the ``emg`` block and one on the ``imu`` block (each with its own
    fold choice: ``svm|emg`` and ``svm|imu``), posteriors (grouped sigmoid
    calibration) combined as ``w P_emg + (1 - w) P_imu``; ``w`` in steps of
    ``w_step`` (default 0.25: {0, .25, .5, .75, 1}) maximizing the macro-F1 of the
    grouped (subject) out-of-fold posteriors of the training fold
    (:class:`ml.fusion.late.LateFusionClassifier`, ``method="weighted"``); ties go
    to the smaller ``w``. Scored on the whole held-out fold.
``reduced_emg``
    ``emg_imu`` with the EMG block cut to the per-channel features in ``features``
    (default ``log_rms``, ``mnf``), parameters of ``svm|emg_imu``. Whole fold.
``muscle_ablation``
    ``emg_imu`` with the EMG columns of one muscle group left out (``drop``: group name ->
    muscles, default the masseter pair, the temporalis pair, both digastrics and each
    digastric alone; X10, exploratory), parameters of ``svm|emg_imu``, whole fold. Off
    unless configured. :func:`ablation_deltas` gives the paired difference to the full
    feature set's main LOSO run (same fold, parameters, training subsample and seed).
``few_shot``
    per feature set (default ``imu`` and ``emg_imu``): the held-out participant's
    windows; ``mode: per_phase`` (default, a guided calibration routine): the
    calibration set is the first k windows of *every* phase (each movement once),
    the evaluation set, common to all k, the windows of each phase ending more than
    ``guard_s`` after its K-th window (K = max(k)); ``mode: first``: the first k
    windows of the recording (only the first task's classes). The SVM is refit on
    the fold's training subsample plus the calibration windows (``n_cal`` = k x phases in ``per_phase``
    mode, k in ``first`` mode), each weighted ``w_user`` (``weight: balanced`` ->
    ``n_src / (n_classes_src * n_cal)``, so the calibration windows *together* weigh
    as much as one average training class, as in ml05 §6.1; a number is a fixed
    per-window multiplier); training windows weigh 1, and ``class_weight="balanced"``
    multiplies both. ``k = 0`` is the plain LOSO model on the evaluation set: the main
    run's fold model, so its predictions are reused when the main run is passed in
    (``main_pred``, :func:`main_predictions`; ``fit_time_s`` is then NaN).

:func:`run_variants` returns one row per variant x feature set x k x participant
(``variants``); :func:`summarize_variants` gives median and IQR over participants
(``variants_summary``), plus paired few-shot gains.
"""

from __future__ import annotations

import ast
import time
import warnings
from typing import Any, Callable

import numpy as np
import pandas as pd
from ml.core.dataset import WindowDataset, feature_set_blocks
from ml.evaluation.metrics import balanced_accuracy, macro_f1
from ml.fusion.late import LateFusionClassifier
from ml.models import registry as reg
from ml.config.classes import CLASS_COUNT
from ml.protocols import evaluate as ev
from ml.protocols import pool
from ml.protocols.folds import Fold, make_folds

MODEL = "svm"
DEFAULTS: dict[str, Any] = {
    "hyper": "main",
    "late_fusion": {"w_step": 0.25},
    "reduced_emg": {"feature_set": "emg_imu", "features": ["log_rms", "mnf"]},
    "muscle_ablation": {"feature_set": "emg_imu",
                        "drop": {"masseter": ["masseter_r", "masseter_l"],
                                 "temporalis": ["temporalis_r", "temporalis_l"],
                                 "digastric": ["digastric_r", "digastric_l"],
                                 "digastric_r": ["digastric_r"],
                                 "digastric_l": ["digastric_l"]}},
    "few_shot": {"feature_sets": ["imu", "emg_imu"], "k": [0, 20, 50, 100],
                 "guard_s": 0.35, "weight": "balanced",
                 "mode": "per_phase"},
}


OPT_IN = ("muscle_ablation",)


def config(cfg: dict[str, Any] | None | bool) -> dict[str, Any] | None:
    """Defaults merged with ``cfg``; ``None``/``False`` = all off, a variant set to
    ``null``/``false`` is off; unknown keys are errors."""
    if cfg is None or cfg is False:
        return None
    cfg = {} if cfg is True else dict(cfg)
    bad = sorted(set(cfg) - set(DEFAULTS))
    if bad:
        raise ValueError(f"unknown ml_eval.variants keys {bad}; known: {sorted(DEFAULTS)}")
    out: dict[str, Any] = {"hyper": str(cfg.get("hyper", DEFAULTS["hyper"]))}
    if out["hyper"] not in ("main", "search"):
        raise ValueError(f"ml_eval.variants.hyper must be main or search, got {out['hyper']!r}")
    for name in ("late_fusion", "reduced_emg", "muscle_ablation", "few_shot"):
        # muscle_ablation (X10, 2026-11-03) is off unless configured, so configurations
        # written before it give the same tables
        v = cfg.get(name, None if name in OPT_IN else {})
        if v is None or v is False:
            out[name] = None
            continue
        v = {} if v is True else dict(v)
        bad = sorted(set(v) - set(DEFAULTS[name]))
        if bad:
            raise ValueError(f"unknown ml_eval.variants.{name} keys {bad}; "
                             f"known: {sorted(DEFAULTS[name])}")
        out[name] = {**DEFAULTS[name], **v}
    ab = out["muscle_ablation"]
    if ab is not None:
        ab["drop"] = {str(g): [str(m) for m in ms] for g, ms in dict(ab["drop"]).items()}
        if not ab["drop"] or not all(ab["drop"].values()):
            raise ValueError("muscle_ablation.drop: groups with at least one muscle each")
    fs = out["few_shot"]
    if fs is not None:
        fs["k"] = sorted({int(k) for k in fs["k"]})
        if fs["k"][0] < 0:
            raise ValueError("few_shot.k must be >= 0")
        if fs["mode"] not in ("per_phase", "first"):
            raise ValueError("few_shot.mode must be per_phase or first")
        if fs["weight"] != "balanced" and not float(fs["weight"]) > 0:
            raise ValueError("few_shot.weight must be 'balanced' or a positive number")
    return out


def main_predictions(rows: list[dict[str, Any]], model: str = MODEL
                     ) -> dict[str, dict[str, dict[str, Any]]]:
    """``{feature_set: {pid: _pred}}`` of the main LOSO rows of ``model``: the fold's test
    rows, predictions and the ``max_train`` / seed they were fitted with."""
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        q = r.get("_pred")
        if r.get("model") == model and isinstance(q, dict) and "rows" in q:
            out.setdefault(str(r["feature_set"]), {})[str(r["pid"])] = q
    return out


def main_params(rows: list[dict[str, Any]], model: str = MODEL) -> dict[str, dict[str, dict]]:
    """``{feature_set: {pid: params}}`` from the ``best_params`` of the main LOSO rows."""
    out: dict[str, dict[str, dict]] = {}
    for r in rows:
        if r.get("model") != model:
            continue
        try:
            p = ast.literal_eval(str(r.get("best_params", "")))
        except (ValueError, SyntaxError):
            continue
        if isinstance(p, dict):
            out.setdefault(str(r["feature_set"]), {})[str(r["pid"])] = p
    return out


# -- pieces (also used by the tests) ----------------------------------------------------
def reduced_matrix(ds: WindowDataset, features: list[str]) -> tuple[np.ndarray, list[slice],
                                                                    list[str]]:
    """``emg`` columns whose feature (``channel|feature``) is in ``features``, then ``imu``."""
    names = ds.names.get("emg", [])
    keep = [i for i, n in enumerate(names) if n.split("|", 1)[-1] in set(features)]
    if not keep:
        raise ValueError(f"reduced_emg: no EMG columns with features {features}")
    emg, imu = ds.block("emg")[:, keep].astype(np.float64), ds.block("imu").astype(np.float64)
    d = len(keep)
    return (np.hstack([emg, imu]), [slice(0, d), slice(d, d + imu.shape[1])],
            [names[i] for i in keep])


def ablation_matrix(ds: WindowDataset, drop: list[str]) -> tuple[np.ndarray, list[slice],
                                                                 list[str]]:
    """``emg`` columns of every muscle not in ``drop`` (``channel|feature``), then ``imu``."""
    names = ds.names.get("emg", [])
    keep = [i for i, n in enumerate(names) if n.split("|", 1)[0] not in set(drop)]
    if len(keep) == len(names):
        raise ValueError(f"muscle_ablation: no EMG columns of {drop}")
    if not keep:
        raise ValueError(f"muscle_ablation: dropping {drop} leaves no EMG column")
    emg, imu = ds.block("emg")[:, keep].astype(np.float64), ds.block("imu").astype(np.float64)
    d = len(keep)
    return (np.hstack([emg, imu]), [slice(0, d), slice(d, d + imu.shape[1])],
            [names[i] for i in keep])


def few_shot_split(ds: WindowDataset, rows: np.ndarray, ks: list[int], guard_s: float,
                   mode: str = "per_phase") -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Calibration rows per k and the evaluation rows common to all k.

    ``per_phase`` (default): a guided calibration routine -- the first k windows of every
    phase (each movement once, recording order within the phase); evaluation = the
    windows of each phase that end more than ``guard_s`` after its K-th window, K = max(k).
    ``first``: the first k windows of the recording (all from the first task).
    """
    rows = np.asarray(rows)
    order = rows[np.lexsort((ds.t_end_us[rows], ds.session[rows]))]
    K = max(ks) if ks else 0
    if mode == "first":
        cal = {k: order[:k] for k in ks}
        if K == 0 or order.size <= K:
            return cal, (order if K == 0 else order[:0])
        last, after = order[K - 1], order[K:]
        near = (ds.session[after] == ds.session[last]) & (
            ds.t_end_us[after] - ds.t_end_us[last] <= guard_s * 1e6)
        return cal, after[~near]
    if mode != "per_phase":
        raise ValueError(f"unknown few-shot mode {mode!r}")
    cal = {k: [] for k in ks}
    test = []
    keys = np.char.add(ds.session[order].astype(str), "|" + ds.phase[order].astype(str))
    for key in dict.fromkeys(keys):
        o = order[keys == key]
        for k in ks:
            cal[k].append(o[:k])
        if K == 0:
            test.append(o)
            continue
        if o.size <= K:
            continue
        last, after = o[K - 1], o[K:]
        test.append(after[ds.t_end_us[after] - ds.t_end_us[last] > guard_s * 1e6])
    cal = {k: (np.concatenate(v) if v else rows[:0]) for k, v in cal.items()}
    return cal, (np.concatenate(test) if test else rows[:0])


def user_weight(weight: str | float, n_src: int, n_classes: int, n_cal: int) -> float:
    """Per-window weight of ``n_cal`` calibration windows: ``balanced`` ->
    ``n_src / (n_classes * n_cal)`` (all calibration windows together weigh as one average
    training class); a number is used as is; 0 without calibration windows."""
    if n_cal == 0:
        return 0.0
    if weight == "balanced":
        return n_src / (max(n_classes, 1) * n_cal)
    return float(weight)


def few_shot_plan(ds: WindowDataset, fold: Fold, *, ks: list[int], guard_s: float,
                  weight: str | float, seed: int, max_train: int | None,
                  mode: str = "per_phase",
                  k0_pred: tuple[np.ndarray, np.ndarray] | None = None
                  ) -> list[dict[str, Any]]:
    """Per k: calibration and evaluation rows and, unless k = 0 reuses ``k0_pred``
    (``y_pred`` set), the refit's rows, groups and sample weights (``fit``); see
    :func:`few_shot_predictions`."""
    cals, test = few_shot_split(ds, fold.test, ks, guard_s, mode)
    if test.size == 0:
        return []
    k0 = None
    if k0_pred is not None:
        rows0, yhat0 = (np.asarray(v) for v in k0_pred)
        if rows0.size:
            order = np.argsort(rows0, kind="stable")
            pos = order[np.minimum(np.searchsorted(rows0, test, sorter=order), rows0.size - 1)]
            if np.array_equal(rows0[pos], test):
                k0 = yhat0[pos]
    src = ev.subsample(fold.train.size, max_train, seed)
    tr, g_src = fold.train[src], fold.groups[src]
    n_cls = np.unique(ds.y[tr]).size
    out = []
    for k in ks:
        cal = cals[k]
        if cal.size == 0 and k0 is not None:
            out.append({"k": k, "cal": cal, "test": test, "y_pred": k0, "w_user": 0.0,
                        "fit_time_s": float("nan")})
            continue
        w_user = user_weight(weight, tr.size, n_cls, cal.size)
        out.append({"k": k, "cal": cal, "test": test, "w_user": w_user,
                    "fit": {"rows": np.concatenate([tr, cal]),
                            "groups": np.concatenate([g_src.astype(str),
                                                      np.full(cal.size, str(fold.unit))]),
                            "sample_weight": None if cal.size == 0 else
                            np.r_[np.ones(tr.size), np.full(cal.size, w_user)]}})
    return out


def few_shot_fit(X: np.ndarray, y: np.ndarray, slices: list[slice] | None,
                 params: dict[str, Any], *, rows: np.ndarray, groups: np.ndarray,
                 sample_weight: np.ndarray | None, test: np.ndarray, seed: int
                 ) -> tuple[np.ndarray, float]:
    """One few-shot refit (training subsample + calibration rows): test predictions and
    fit time."""
    t0 = time.perf_counter()
    est = reg.get(MODEL).make(seed, slices=slices, **params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est.fit(X[rows], y[rows], groups=groups, sample_weight=sample_weight)
    return est.predict(X[test]), time.perf_counter() - t0


def few_shot_predictions(ds: WindowDataset, fold: Fold, X: np.ndarray, slices: list[slice],
                         params: dict[str, Any], *, ks: list[int], guard_s: float,
                         weight: str | float, seed: int, max_train: int | None,
                         mode: str = "per_phase",
                         k0_pred: tuple[np.ndarray, np.ndarray] | None = None
                         ) -> list[dict[str, Any]]:
    """Per k: calibration rows, evaluation rows, predictions, user weight, fit time.

    ``k0_pred`` = (rows, predictions) of the main run's model of this fold with the same
    ``params``, ``max_train`` and seed: k = 0 then takes them instead of refitting that
    model (``fit_time_s`` NaN). Serial; :func:`run_variants` runs the per-k refits
    (:func:`few_shot_plan`, :func:`few_shot_fit`) as separate pool tasks."""
    out = few_shot_plan(ds, fold, ks=ks, guard_s=guard_s, weight=weight, seed=seed,
                        max_train=max_train, mode=mode, k0_pred=k0_pred)
    for r in out:
        spec = r.pop("fit", None)
        if spec is not None:
            r["y_pred"], r["fit_time_s"] = few_shot_fit(X, ds.y, slices, params, test=r["test"],
                                                        seed=seed, **spec)
    return out


def late_fusion_model(X: np.ndarray, y: np.ndarray, groups: np.ndarray, slices: list[slice],
                      block_params: list[dict[str, Any]], *, w_step: float, inner: int,
                      seed: int) -> LateFusionClassifier:
    m = LateFusionClassifier(base=MODEL, method="weighted", inner=inner, seed=seed,
                             slices=slices, step=w_step, block_params=block_params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return m.fit(X, y, groups=groups)


# -- tasks ------------------------------------------------------------------------------
# Every fit is its own pool task (ml.protocols.pool): a fold's late fusion splits into its
# grouped out-of-fold block fits and the two final block fits, few-shot into one refit per
# k, and missing fold hyper-parameters into a search task first. Workers read the feature
# matrices memory-mapped; the caller assembles the rows in table order.
def _score(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    return {"balanced_accuracy": balanced_accuracy(y, p), "macro_f1": macro_f1(y, p),
            "n_test": int(len(y))}


def _search_job(xref, yref, rows: np.ndarray, cols: slice | None, groups: np.ndarray,
                slices: list[slice] | None, seed: int, inner: int, max_train: int | None
                ) -> dict[str, Any]:
    """A fold's inner search when the main run has no choice for it (``hyper: search``)
    (worker): ``X[rows][:, cols]`` is the fold's matrix, subsampled as ``fit`` does."""
    X = xref.get()[rows]
    if cols is not None:
        X = X[:, cols]
    y = yref.get()[rows]
    idx = ev.subsample(len(y), max_train, seed)
    p, _ = ev.grouped_search(MODEL, X[idx], y[idx], groups[idx], reg.get(MODEL).grid,
                             inner=inner, seed=seed, slices=slices)
    return p


def _late_piece(xref, yref, train: np.ndarray, idx: np.ndarray, groups: np.ndarray,
                test: np.ndarray, m: LateFusionClassifier, b: int,
                split: tuple[np.ndarray, np.ndarray] | None) -> tuple[np.ndarray, float]:
    """One block fit of a late-fusion model (worker): out-of-fold posteriors on the split's
    test rows, or (``split`` None) the block model on all rows and its posteriors on the
    held-out fold."""
    X, y = xref.get(), yref.get()
    Xtr, ytr = X[train], y[train]
    Xf, yf = Xtr[idx], ytr[idx]
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if split is not None:
            P = m.oof_piece(b, Xf, np.asarray(yf), groups, *split)
        else:
            P = m.block_proba(b, m.fit_block(b, Xf, np.asarray(yf), groups), X[test])
    return P, time.perf_counter() - t0


def _reduced_job(xref, yref, f: Fold, slices: list[slice], p: dict[str, Any], *, seed: int,
                 inner: int, max_train: int | None) -> tuple[np.ndarray, float]:
    X, y = xref.get(), yref.get()
    t0 = time.perf_counter()
    est, _ = ev.fit(MODEL, X[f.train], y[f.train], f.groups, seed=seed, inner=inner,
                    max_train=max_train, search=False, params=p, slices=slices)
    dt = time.perf_counter() - t0
    return est.predict(X[f.test]), dt


def _few_shot_job(xref, yref, slices: list[slice], params: dict[str, Any], spec: dict,
                  test: np.ndarray, seed: int) -> tuple[np.ndarray, float]:
    return few_shot_fit(xref.get(), yref.get(), slices, params, test=test, seed=seed, **spec)


class _Unit:
    """One table entry (variant x fold [x feature set]): its pending pieces and rows."""

    def __init__(self, kind: str, f: Fold, fset: str | None = None) -> None:
        self.kind, self.f, self.fset = kind, f, fset
        self.state: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] | None = None


def run_variants(ds: WindowDataset, cfg: dict[str, Any] | None, main: dict[str, dict], *,
                 seed: int = 42, inner: int = 3, max_train: int | None = None,
                 n_jobs: int = 1, main_pred: dict[str, dict] | None = None,
                 cancelled: Callable[[], bool] | None = None) -> pd.DataFrame:
    """The ``variants`` table (empty when ``cfg`` is None or nothing applies).

    ``main_pred`` (:func:`main_predictions` of the main LOSO rows) lets few-shot k = 0
    reuse the main fold model's predictions (same table). All fits run in one pool, longest
    expected first, without barriers (:mod:`ml.protocols.pool`); the rows come in table order
    whatever ``n_jobs``. ``cancelled`` is polled after every finished fit
    (``RuntimeError("cancelled")``)."""
    if cfg is None:
        return pd.DataFrame()
    have = set(ds.blocks)
    units: list[_Unit] = []
    for f in make_folds(ds, "loso_subject"):
        if np.unique(ds.y[f.train]).size < 2:
            continue
        if cfg["late_fusion"] is not None and {"emg", "imu"} <= have:
            units.append(_Unit("late", f))
        if cfg["reduced_emg"] is not None and {"emg", "imu"} <= have:
            units.append(_Unit("reduced", f))
        if cfg.get("muscle_ablation") is not None and {"emg", "imu"} <= have:
            for g in cfg["muscle_ablation"]["drop"]:
                units.append(_Unit("ablation", f, g))
        if cfg["few_shot"] is not None:
            for fset in cfg["few_shot"]["feature_sets"]:
                if set(b if b != "emg_nodig" else "emg" for b in feature_set_blocks(fset)) \
                        <= have:
                    units.append(_Unit("few_shot", f, fset))
    if not units:
        return pd.DataFrame()
    n_jobs = ev.resolve_jobs(n_jobs)
    hyper = cfg["hyper"]
    with pool.Shared(prefix="ml_variants_") as sh:
        yref = sh.put(np.asarray(ds.y))
        mats: dict[str, tuple[Any, np.ndarray, list[slice]]] = {}

        def mat(name: str):
            if name not in mats:
                if name == "reduced":
                    X, sl, names = reduced_matrix(ds, list(cfg["reduced_emg"]["features"]))
                    mats[name] = (sh.put(X), X, sl, names)
                elif name.startswith("ablation:"):
                    drop = cfg["muscle_ablation"]["drop"][name.split(":", 1)[1]]
                    X, sl, names = ablation_matrix(ds, list(drop))
                    mats[name] = (sh.put(X), X, sl, names)
                else:
                    X, sl = ds.matrix(name)
                    mats[name] = (sh.put(X), X, sl, None)
            return mats[name]

        def params_or_task(u: int, tag: Any, key: str, xname: str, cols: slice | None,
                           slices: list[slice] | None) -> list[pool.Task]:
            """Main-run parameters into the unit state, or a search task for them."""
            f = units[u].f
            if hyper == "main" and f.unit in main.get(key, {}):
                units[u].state[("p", tag)] = (dict(main[key][f.unit]), f"main_loso:{key}")
                return []
            xref, X, _, _ = mat(xname)
            d = X.shape[1] if cols is None else len(range(*cols.indices(X.shape[1])))
            return [pool.Task(("hyper", u, tag), _search_job,
                              (xref, yref, f.train, cols, f.groups, slices, seed, inner,
                               max_train),
                              cost=12 * 3 * ev.fit_cost(MODEL, int(min(
                                  f.train.size, max_train or f.train.size) * 2 / 3), d))]

        def ready(u: int) -> list[pool.Task]:
            """The fit tasks of unit ``u`` once its parameters are known."""
            un, f = units[u], units[u].f
            n_fit = int(min(f.train.size, max_train or f.train.size))
            if un.kind == "late":
                xref, X, slices, _ = mat("emg+imu")
                bp = [un.state[("p", b)][0] for b in ("emg", "imu")]
                m = LateFusionClassifier(base=MODEL, method="weighted", inner=inner, seed=seed,
                                         slices=slices, step=float(cfg["late_fusion"]["w_step"]),
                                         block_params=bp)
                idx = ev.subsample(f.train.size, max_train, seed)
                ytr = ds.y[f.train][idx]
                gf = f.groups[idx]
                splits = m.oof_splits(ytr, gf)
                if splits is None:
                    raise ValueError("late_weighted needs >= 2 groups for out-of-fold posteriors")
                un.state.update(m=m, splits=splits, y=ytr, n=len(splits) * 2 + 2)
                ts = []
                for si, sp in enumerate(splits):
                    for b in range(2):
                        ts.append(pool.Task(("late", u, si, b), _late_piece,
                                            (xref, yref, f.train, idx, gf, f.test, m, b, sp),
                                            cost=4 * ev.fit_cost(MODEL, len(sp[0]),
                                                                 slices[b].stop - slices[b].start,
                                                                 bp[b])))
                for b in range(2):
                    ts.append(pool.Task(("late", u, None, b), _late_piece,
                                        (xref, yref, f.train, idx, gf, f.test, m, b, None),
                                        cost=4 * ev.fit_cost(MODEL, n_fit,
                                                             slices[b].stop - slices[b].start,
                                                             bp[b])))
                return ts
            if un.kind in ("reduced", "ablation"):
                xref, X, slices, _ = mat("reduced" if un.kind == "reduced"
                                         else f"ablation:{un.fset}")
                p = un.state[("p", 0)][0]
                return [pool.Task(("reduced", u), _reduced_job, (xref, yref, f, slices, p),
                                  cost=ev.fit_cost(MODEL, n_fit, X.shape[1], p),
                                  kwargs={"seed": seed, "inner": inner,
                                          "max_train": max_train})]
            fc = cfg["few_shot"]
            xref, X, slices, _ = mat(un.fset)
            p, h = un.state[("p", 0)]
            q = (main_pred or {}).get(un.fset, {}).get(str(f.unit))
            k0 = None
            if h.startswith("main_loso") and q is not None and q.get("seed") == seed and \
                    q.get("max_train") == max_train:
                k0 = (q["rows"], q["yhat"])
            plan = few_shot_plan(ds, f, ks=fc["k"], guard_s=float(fc["guard_s"]),
                                 mode=str(fc.get("mode", "per_phase")), weight=fc["weight"],
                                 seed=seed, max_train=max_train, k0_pred=k0)
            un.state["plan"] = plan
            ts = [pool.Task(("few_shot", u, i), _few_shot_job,
                            (xref, yref, slices, p, r["fit"], r["test"], seed),
                            cost=ev.fit_cost(MODEL, int(r["fit"]["rows"].size), X.shape[1], p))
                  for i, r in enumerate(plan) if "fit" in r]
            un.state["n"] = len(ts)
            if not ts:
                finish(u)
            return ts

        def finish(u: int) -> None:
            un, f = units[u], units[u].f
            if un.kind == "late":
                st = un.state
                m, n = st["m"], len(st["y"])
                oof = [np.zeros((n, CLASS_COUNT)) for _ in range(2)]
                for si, (_, te) in enumerate(st["splits"]):
                    for b in range(2):
                        oof[b][te] = st[("late", si, b)][0]
                m.combine(st["y"], oof)
                P = [st[("late", None, b)][0] for b in range(2)]
                yhat = m.proba_from_blocks(P).argmax(axis=1)
                dt = float(sum(v[1] for k, v in st.items()
                               if isinstance(k, tuple) and k[0] == "late"))
                bp = [st[("p", b)][0] for b in ("emg", "imu")]
                un.rows = [{"variant": "late_fusion", "feature_set": "emg+imu", "k": 0,
                            "pid": f.unit, **_score(ds.y[f.test], yhat),
                            "test_portion": "fold", "w_emg": float(m.weights_[0]),
                            "params": str(bp),
                            "hyper": "|".join(st[("p", b)][1] for b in ("emg", "imu")),
                            "fit_time_s": dt}]
            elif un.kind == "reduced":
                rc = cfg["reduced_emg"]
                yhat, dt = un.state["fit"]
                p, h = un.state[("p", 0)]
                un.rows = [{"variant": "reduced_emg",
                            "feature_set": f"emg[{','.join(rc['features'])}]+imu",
                            "k": 0, "pid": f.unit, **_score(ds.y[f.test], yhat),
                            "test_portion": "fold", "n_emg_features": len(mat("reduced")[3]),
                            "params": str(p), "hyper": h, "fit_time_s": dt}]
            elif un.kind == "ablation":
                yhat, dt = un.state["fit"]
                p, h = un.state[("p", 0)]
                un.rows = [{"variant": "muscle_ablation", "feature_set": f"emg-{un.fset}+imu",
                            "k": 0, "pid": f.unit, **_score(ds.y[f.test], yhat),
                            "test_portion": "fold",
                            "n_emg_features": len(mat(f"ablation:{un.fset}")[3]),
                            "dropped": ",".join(cfg["muscle_ablation"]["drop"][un.fset]),
                            "params": str(p), "hyper": h, "fit_time_s": dt}]
            else:
                p, h = un.state[("p", 0)]
                rows = []
                for i, r in enumerate(un.state["plan"]):
                    if "fit" in r:
                        r["y_pred"], r["fit_time_s"] = un.state[("few_shot", i)]
                    rows.append({"variant": "few_shot", "feature_set": un.fset, "k": r["k"],
                                 "pid": f.unit, **_score(ds.y[r["test"]], r["y_pred"]),
                                 "test_portion": "after_calibration",
                                 "n_cal": int(r["cal"].size),
                                 "cal_classes": int(np.unique(ds.y[r["cal"]]).size),
                                 "w_user": r["w_user"], "params": str(p), "hyper": h,
                                 "fit_time_s": r["fit_time_s"]})
                un.rows = rows

        def start(u: int) -> list[pool.Task]:
            un = units[u]
            if un.kind == "late":
                _, X, slices, _ = mat("emg+imu")
                ts = []
                for b, s in zip(("emg", "imu"), slices):
                    ts += params_or_task(u, b, b, "emg+imu", s, None)
            elif un.kind == "reduced":
                _, X, slices, _ = mat("reduced")
                ts = params_or_task(u, 0, str(cfg["reduced_emg"]["feature_set"]), "reduced",
                                    None, slices)
            elif un.kind == "ablation":
                _, X, slices, _ = mat(f"ablation:{un.fset}")
                ts = params_or_task(u, 0, str(cfg["muscle_ablation"]["feature_set"]),
                                    f"ablation:{un.fset}", None, slices)
            else:
                _, X, slices, _ = mat(un.fset)
                ts = params_or_task(u, 0, un.fset, un.fset, None, slices)
            return ts or ready(u)

        def hyper_src(u: int) -> int:
            return 2 if units[u].kind == "late" else 1

        def on_result(key, r):
            kind, u = key[0], key[1]
            un = units[u]
            if kind == "hyper":
                un.state[("p", key[2])] = (r, "inner_search")
                if sum(1 for k in un.state if isinstance(k, tuple) and k[0] == "p") \
                        == hyper_src(u):
                    return ready(u)
                return None
            if kind == "late":
                un.state[("late", key[2], key[3])] = r
                if sum(1 for k in un.state if isinstance(k, tuple) and k[0] == "late") \
                        == un.state["n"]:
                    finish(u)
                return None
            if kind == "reduced":
                un.state["fit"] = r
                finish(u)
                return None
            un.state[("few_shot", key[2])] = r
            if sum(1 for k in un.state if isinstance(k, tuple) and k[0] == "few_shot") \
                    == un.state["n"]:
                finish(u)
            return None

        tasks: list[pool.Task] = []
        for u in range(len(units)):
            tasks += start(u)
        pool.run_tasks(tasks, n_jobs, on_result=on_result, cancelled=cancelled)
    return pd.DataFrame([r for un in units for r in (un.rows or [])])


def _describe(v: np.ndarray) -> dict[str, float]:
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan, "iqr": np.nan,
                "mean": np.nan}
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return {"n": int(v.size), "median": float(med), "q1": float(q1), "q3": float(q3),
            "iqr": float(q3 - q1), "mean": float(v.mean())}


def summarize_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Median [IQR] over participants per variant x feature set x k and metric.

    Metrics: ``balanced_accuracy``, ``macro_f1``, ``fit_time_s`` (per fold), and for
    ``few_shot`` the paired ``ba_gain_vs_k0`` (BA(k) - BA(0), same evaluation set) and
    ``ba_emg_imu_minus_imu`` (both feature sets, same k)."""
    if df.empty:
        return pd.DataFrame()
    out = []
    for (v, fs, k), g in df.groupby(["variant", "feature_set", "k"], sort=True):
        for m in ("balanced_accuracy", "macro_f1", "fit_time_s"):
            out.append({"variant": v, "feature_set": fs, "k": int(k), "metric": m,
                        **_describe(g[m].to_numpy())})
    fs_df = df[df.variant == "few_shot"]
    if not fs_df.empty:
        w = fs_df.pivot_table(index="pid", columns=["feature_set", "k"],
                              values="balanced_accuracy")
        for fs, k in w.columns:
            if k != 0 and (fs, 0) in w.columns:
                out.append({"variant": "few_shot", "feature_set": fs, "k": int(k),
                            "metric": "ba_gain_vs_k0",
                            **_describe((w[(fs, k)] - w[(fs, 0)]).to_numpy())})
        for k in sorted({k for _, k in w.columns}):
            if ("emg_imu", k) in w.columns and ("imu", k) in w.columns:
                out.append({"variant": "few_shot", "feature_set": "emg_imu-imu", "k": int(k),
                            "metric": "ba_emg_imu_minus_imu",
                            **_describe((w[("emg_imu", k)] - w[("imu", k)]).to_numpy())})
    return pd.DataFrame(out)


def ablation_deltas(df: pd.DataFrame, loso: pd.DataFrame, feature_set: str = "emg_imu",
                    model: str = MODEL) -> pd.DataFrame:
    """Paired ``muscle_ablation`` - full: per left-out group the balanced accuracy minus that
    of the main LOSO ``model|feature_set`` run of the same participant (the ablation reuses
    its fold parameters, training subsample and seed), median [IQR] over participants
    (``variants_summary`` rows of the metric ``ba_minus_full``) plus the participants whose
    accuracy dropped (``n_lower``)."""
    if df.empty or "variant" not in df or loso.empty:
        return pd.DataFrame()
    a = df[df.variant == "muscle_ablation"]
    full = loso[(loso.model == model) & (loso.feature_set == feature_set)]
    if a.empty or full.empty:
        return pd.DataFrame()
    ref = full.set_index(full.pid.astype(str))["balanced_accuracy"]
    out = []
    for fs, g in a.groupby("feature_set", sort=True):
        d = (g.set_index(g.pid.astype(str))["balanced_accuracy"] - ref).dropna().to_numpy(float)
        out.append({"variant": "muscle_ablation", "feature_set": fs, "k": 0,
                    "metric": "ba_minus_full", **_describe(d), "n_lower": int((d < 0).sum())})
    return pd.DataFrame(out)


def _wilcoxon_p(d: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p (scipy's default, exact for small samples without
    ties/zeros); NaN for < 5 pairs or all-zero differences."""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size < 5 or not np.any(d != 0):
        return float("nan")
    from scipy.stats import wilcoxon

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(wilcoxon(d).pvalue)


def few_shot_tests(df: pd.DataFrame, a: str = "emg_imu", b: str = "imu",
                   model: str = MODEL) -> list[dict[str, Any]]:
    """Paired ``a`` - ``b`` few-shot balanced accuracy per participant (same evaluation set)
    for every calibration amount k (k = 0 included: the family the analysis plan logged,
    D13 "Holm over the four k"): median difference, Wilcoxon p, Holm over the k
    (``n_amounts`` = the family size, the k with a finite p), participants and those better
    with ``a``. Rows ``fewshot|<model>|<a>_vs_<b>|k=<k>`` of ``ml_eval.tests``."""
    from biomech.stats.core import holm

    f = df[df.variant == "few_shot"] if not df.empty and "variant" in df else pd.DataFrame()
    if f.empty:
        return []
    w = f.pivot_table(index="pid", columns=["feature_set", "k"], values="balanced_accuracy")
    out = []
    for k in sorted({int(c[1]) for c in w.columns}):
        if (a, k) in w.columns and (b, k) in w.columns:
            d = (w[(a, k)] - w[(b, k)]).dropna().to_numpy(float)
            out.append({"test": f"fewshot|{model}|{a}_vs_{b}|k={k}",
                        "stat": float(np.median(d)) if d.size else float("nan"),
                        "p": _wilcoxon_p(d), "n": int(d.size), "n_better": int((d > 0).sum())})
    ps = [r["p"] for r in out]
    m = int(np.isfinite(ps).sum()) if ps else 0
    for r, ph in zip(out, holm(ps)):
        r["p_holm"] = ph
        r["n_amounts"] = m
    return out


__all__ = ["DEFAULTS", "few_shot_tests", "config", "few_shot_predictions", "few_shot_split", "late_fusion_model",
           "main_params", "main_predictions", "reduced_matrix", "run_variants", "summarize_variants", "user_weight"]
