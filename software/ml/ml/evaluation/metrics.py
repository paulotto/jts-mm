# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Classification metrics."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize

from ml.config.classes import CLASS_COUNT, CLASS_NAMES


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in ``y_true`` (the one scorer of the package).

    Used for inner hyper-parameter search, early stopping and reported scores, so a
    fold that lacks a class is not penalized for it and per-subject values compare.
    """
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        return float("nan")
    return float(f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro",
                          zero_division=0))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else float("nan")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, *,
                    y_proba: np.ndarray | None = None) -> dict[str, Any]:
    """Return scalar metrics and per-class tables."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = sorted(CLASS_NAMES)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )

    balanced_acc = balanced_accuracy(y_true, y_pred)

    ap_per_class = _per_class_average_precision(y_true, y_proba, labels)

    out: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # Over the classes present in y_true, so per-subject values are comparable.
        "macro_f1": macro_f1(y_true, y_pred),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": balanced_acc,
        "cohen_kappa": _quiet(cohen_kappa_score, y_true, y_pred),
        "mcc": _quiet(matthews_corrcoef, y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": float(precision[j]),
                "recall": float(recall[j]),
                "f1": float(f1[j]),
                "support": int(support[j]),
                "average_precision": ap_per_class.get(CLASS_NAMES[i]),
            }
            for j, i in enumerate(labels)
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, target_names=[CLASS_NAMES[i] for i in labels],
            zero_division=0
        ),
    }

    out["macro_auroc"] = _macro_auroc(y_true, y_proba, labels)
    ap_vals = [v for v in ap_per_class.values() if v is not None]
    out["macro_average_precision"] = float(np.mean(ap_vals)) if ap_vals else None
    return out


def _per_class_average_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray | None,
    labels: list[int],
) -> dict[str, float | None]:
    """One-vs-rest average precision per class (needs ``y_proba``)."""
    out: dict[str, float | None] = {CLASS_NAMES[i]: None for i in labels}
    if y_proba is None:
        return out
    y_proba = np.asarray(y_proba, dtype=np.float64)
    if y_proba.ndim != 2 or y_proba.shape[1] < CLASS_COUNT:
        return out

    y_bin = label_binarize(y_true, classes=labels)
    if y_bin.ndim == 1:
        y_bin = y_bin.reshape(-1, 1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        for j, class_id in enumerate(labels):
            name = CLASS_NAMES[class_id]
            if j >= y_bin.shape[1]:
                continue
            if int(y_bin[:, j].sum()) == 0:
                out[name] = None
                continue
            try:
                scores = y_proba[:, class_id]
                out[name] = float(average_precision_score(y_bin[:, j], scores))
            except ValueError:
                out[name] = None
    return out


def _macro_auroc(y_true: np.ndarray, y_proba: np.ndarray | None, labels: list[int]) -> float | None:
    if y_proba is None:
        return None
    if len(np.unique(y_true)) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UndefinedMetricWarning)
            return float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro", labels=labels)
            )
    except ValueError:
        return None


def _quiet(fn, *args) -> float:
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore")
        return float(fn(*args))
