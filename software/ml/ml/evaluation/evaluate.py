# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``ml eval``: score saved predictions or reloaded fold models with the shared metrics.

Fold predictions are tied to the dataset by **row index** (``indices`` in
``predictions.npz``), never by timestamp: pooled takes can share timestamps, and
a timestamp lookup silently mapped rows of one take onto another.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.core.dataset import WindowDataset

from .metrics import compute_metrics
from .report import write_run_artifacts
from .types import EvaluationResult, PredictionBundle


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, *,
                         y_proba: np.ndarray | None = None, t_end_us: np.ndarray | None = None,
                         groups: np.ndarray | None = None, indices: np.ndarray | None = None,
                         extra: dict[str, Any] | None = None) -> EvaluationResult:
    return EvaluationResult(
        metrics=compute_metrics(y_true, y_pred, y_proba=y_proba),
        predictions=PredictionBundle(np.asarray(y_true), np.asarray(y_pred), y_proba, t_end_us,
                                     groups, indices),
        extra=dict(extra or {}))


def load_model(path: str | Path):
    """A fitted model saved by ``ml train`` (``model.joblib``)."""
    return joblib.load(Path(path))


def run_summary(run_dir: str | Path) -> dict[str, Any]:
    p = Path(run_dir) / "summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing (not an `ml train` run directory?)")
    return json.loads(p.read_text(encoding="utf-8"))


def _fold_dirs(run_dir: Path) -> list[Path]:
    dirs = sorted((run_dir / "per_fold").glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    if not dirs:
        raise FileNotFoundError(f"no per_fold/fold_* under {run_dir}")
    return dirs


def _blob(fold_dir: Path) -> dict[str, np.ndarray]:
    p = fold_dir / "predictions.npz"
    if not p.is_file():
        raise FileNotFoundError(p)
    z = np.load(p, allow_pickle=False)
    if "indices" not in z.files:
        raise ValueError(f"{p} has no row indices (written before 2026-10-15); retrain the run")
    return {k: z[k] for k in z.files}


def evaluate_run_predictions(run_dir: str | Path) -> EvaluationResult:
    """Pooled metrics from a run's saved fold predictions."""
    run_dir = Path(run_dir)
    blobs = [_blob(d) for d in _fold_dirs(run_dir)]
    cat = np.concatenate
    proba = [b["y_proba"] for b in blobs if "y_proba" in b]
    info = run_summary(run_dir)
    return evaluate_predictions(
        cat([b["y_true"] for b in blobs]), cat([b["y_pred"] for b in blobs]),
        y_proba=cat(proba) if len(proba) == len(blobs) else None,
        t_end_us=cat([b["t_end_us"] for b in blobs]), groups=cat([b["groups"] for b in blobs]),
        indices=cat([b["indices"] for b in blobs]),
        extra={"run_dir": str(run_dir.resolve()), "n_folds": len(blobs),
               "source": "saved_predictions", "model": info.get("model"),
               "feature_set": info.get("feature_set"), "protocol": info.get("protocol")})


def evaluate_run_models(run_dir: str | Path, ds: WindowDataset) -> EvaluationResult:
    """Reload every fold model and re-score its test rows (by index) on ``ds``."""
    run_dir = Path(run_dir)
    info = run_summary(run_dir)
    fset = info["feature_set"]
    X, _ = ds.matrix(fset)
    yt, yp, pr, idx_all, per_fold = [], [], [], [], []
    for d in _fold_dirs(run_dir):
        b = _blob(d)
        idx = np.asarray(b["indices"], dtype=np.int64)
        if idx.size and (idx.max() >= len(ds) or not np.array_equal(ds.y[idx], b["y_true"])):
            raise ValueError(f"{d.name}: the dataset does not match the run (row indices out of "
                             "range or labels differ); pass the dataset.npz used for training")
        m = load_model(d / "model.joblib")
        p = m.predict(X[idx])
        prob = m.predict_proba(X[idx])
        per_fold.append({"fold": d.name, **{k: v for k, v in compute_metrics(
            ds.y[idx], p, y_proba=prob).items() if k != "classification_report"}})
        yt.append(ds.y[idx])
        yp.append(p)
        pr.append(prob)
        idx_all.append(idx)
    idx = np.concatenate(idx_all)
    return evaluate_predictions(
        np.concatenate(yt), np.concatenate(yp), y_proba=np.concatenate(pr),
        t_end_us=ds.t_end_us[idx], groups=ds.groups[idx], indices=idx,
        extra={"run_dir": str(run_dir.resolve()), "n_folds": len(idx_all), "model": info["model"],
               "feature_set": fset, "protocol": info.get("protocol"),
               "source": "reloaded_fold_models", "per_fold": per_fold})


def evaluate_model_path(model_path: str | Path, ds: WindowDataset, fset: str, *,
                        indices: np.ndarray | None = None) -> EvaluationResult:
    """Score one fitted model on (a subset of) a dataset. Includes its training rows
    unless ``indices`` restricts it, so this is not a cross-validated score."""
    m = load_model(model_path)
    X, _ = ds.matrix(fset)
    idx = np.arange(len(ds)) if indices is None else np.asarray(indices, dtype=np.int64)
    return evaluate_predictions(
        ds.y[idx], m.predict(X[idx]), y_proba=m.predict_proba(X[idx]), t_end_us=ds.t_end_us[idx],
        groups=ds.groups[idx], indices=idx,
        extra={"model_path": str(Path(model_path).resolve()), "feature_set": fset,
               "model": type(m).__name__})


def write_evaluation(out_dir: str | Path, result: EvaluationResult, *,
                     write_predictions: bool = True) -> Path:
    out_dir = Path(out_dir)
    write_run_artifacts(out_dir, result.metrics, extra=result.extra or None)
    if write_predictions:
        p = result.predictions
        payload = {k: v for k, v in {"y_true": p.y_true, "y_pred": p.y_pred, "y_proba": p.y_proba,
                                     "t_end_us": p.t_reference_us, "groups": p.groups,
                                     "indices": p.indices}.items() if v is not None}
        np.savez_compressed(out_dir / "predictions.npz", **payload)
    return out_dir
