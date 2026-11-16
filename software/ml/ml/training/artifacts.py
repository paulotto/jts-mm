# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Run-directory artifact writers."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ml.config.classes import CLASS_NAMES
from ml.evaluation.metrics import compute_metrics
from ml.evaluation.plots import plot_confusion_matrix, plot_per_class_f1

__all__ = [
    "new_run_dir",
    "write_aggregate_artifacts",
    "write_env",
    "write_fold_artifacts",
    "write_git_commit",
    "write_json",
    "write_text",
    "write_yaml",
]


def new_run_dir(runs_root: str | Path, *, model: str, config_hash: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(runs_root) / f"{stamp}_{model}_{config_hash}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "per_fold").mkdir()
    return run_dir


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_git_commit(path: Path, cwd: Path | None = None) -> None:
    commit = "unknown"
    dirty = False
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
    except Exception:
        pass
    write_text(path, f"{commit}{' (dirty)' if dirty else ''}")


def write_env(path: Path) -> None:
    versions = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "sklearn": _pkg_version("sklearn"),
        "joblib": _pkg_version("joblib"),
    }
    try:
        versions["torch"] = _pkg_version("torch")
    except Exception:
        versions["torch"] = None
    write_json(path, versions)


def write_fold_artifacts(
    fold_dir: Path,
    *,
    metrics: dict[str, Any],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    t_end_us: np.ndarray,
    groups: np.ndarray,
    indices: np.ndarray,
    free_indices: np.ndarray | None = None,
    free_pred: np.ndarray | None = None,
) -> None:
    """``predictions.npz`` keyed by dataset row ``indices`` (never by timestamp)."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    write_json(fold_dir / "metrics.json", metrics)
    extra = {}
    if free_indices is not None and len(free_indices):
        extra = {"free_indices": np.asarray(free_indices), "free_pred": np.asarray(free_pred)}
    np.savez_compressed(
        fold_dir / "predictions.npz",
        indices=np.asarray(indices, dtype=np.int64),
        y_true=np.asarray(y_true),
        y_pred=np.asarray(y_pred),
        y_proba=np.asarray(y_proba),
        t_end_us=np.asarray(t_end_us),
        groups=np.asarray(groups),
        **extra,
    )


def write_aggregate_artifacts(
    run_dir: Path,
    *,
    fold_metrics: list[dict[str, Any]],
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_proba_all: np.ndarray | None = None,
) -> dict[str, Any]:
    pooled = compute_metrics(y_true_all, y_pred_all, y_proba=y_proba_all)
    aggregate = {
        "n_folds": len(fold_metrics),
        "pooled": _metrics_without_report(pooled),
        "per_fold": [_metrics_without_report(m) for m in fold_metrics],
        "mean": {},
        "std": {},
    }
    scalar_keys = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "cohen_kappa",
        "mcc",
        "macro_auroc",
        "macro_average_precision",
        "fit_time_s",
    ]
    for key in scalar_keys:
        vals = [m.get(key) for m in fold_metrics if m.get(key) is not None]
        if not vals:
            continue
        aggregate["mean"][key] = float(np.mean(vals))
        # Sample SD across folds (ddof = 1); undefined for a single fold.
        aggregate["std"][key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None

    write_json(run_dir / "metrics.json", aggregate)
    report = pooled.get("classification_report")
    if report:
        write_text(run_dir / "classification_report.txt", str(report))

    cm = pooled.get("confusion_matrix")
    if cm is not None:
        plot_confusion_matrix(cm, run_dir / "confusion_matrix.png")
        _write_cm_csv(cm, run_dir / "confusion_matrix.csv")

    per_class = pooled.get("per_class")
    if isinstance(per_class, dict) and per_class:
        plot_per_class_f1(per_class, run_dir / "per_class_f1.png")

    return aggregate


def _metrics_without_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in metrics.items() if k != "classification_report"}


def _write_cm_csv(cm: list[list[int]], path: Path) -> None:
    labels = sorted(CLASS_NAMES)
    lines = ["true\\pred," + ",".join(CLASS_NAMES[i] for i in labels)]
    for i, row in enumerate(cm):
        lines.append(f"{CLASS_NAMES[labels[i]]}," + ",".join(str(v) for v in row))
    write_text(path, "\n".join(lines))


def _pkg_version(name: str) -> str:
    mod = __import__(name)
    return str(getattr(mod, "__version__", "unknown"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
