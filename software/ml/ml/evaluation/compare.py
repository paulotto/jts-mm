# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Aggregate metrics across training run directories (``ml train --compare``)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ml.config.classes import CLASS_NAMES
from ml.evaluation.plots import plot_per_class_f1_across_models


_SCALAR_KEYS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "balanced_accuracy",
    "cohen_kappa",
    "mcc",
    "macro_auroc",
    "macro_average_precision",
    "fit_time_s",
)


def compare_runs(runs_root: str | Path) -> list[dict[str, Any]]:
    """Load ``metrics.json`` from each run dir under ``runs_root`` and summarize."""
    runs_root = Path(runs_root)
    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)

    rows: list[dict[str, Any]] = []
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        mean = metrics.get("mean") or {}
        std = metrics.get("std") or {}
        info = _run_info(run_dir, metrics)
        row: dict[str, Any] = {
            "run": run_dir.name,
            "model": info.get("model", "unknown"),
            "feature_set": info.get("feature_set", ""),
            "protocol": info.get("protocol", ""),
            "n_folds": metrics.get("n_folds"),
        }
        pooled = metrics.get("pooled") if isinstance(metrics.get("pooled"), dict) else {}
        for key in _SCALAR_KEYS:
            if key in mean:
                row[key] = mean[key]
                row[f"{key}_std"] = std.get(key)
            elif key in pooled and isinstance(pooled[key], (int, float)):
                row[key] = pooled[key]
            elif key in metrics and isinstance(metrics[key], (int, float)):
                # Single-evaluation dirs from ``ml eval`` / write_run_artifacts
                row[key] = metrics[key]

        per_class = _extract_per_class(metrics)
        if per_class:
            row["per_class_f1"] = {name: entry.get("f1") for name, entry in per_class.items()}
            row["per_class_average_precision"] = {
                name: entry.get("average_precision") for name, entry in per_class.items()
            }
        rows.append(row)

    rows.sort(key=lambda r: (-(r.get("macro_f1") or float("-inf")), r["run"]))
    return rows


def write_comparison_table(rows: list[dict[str, Any]], out_dir: str | Path) -> tuple[Path, Path]:
    """Write Markdown + CSV comparison tables. Returns ``(md_path, csv_path)``.

    Also writes ``per_class_f1.png`` when at least one row has per-class F1
    (one best run per model, ranked by macro F1).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "comparison.md"
    csv_path = out_dir / "comparison.csv"

    headers = [
        "Model",
        "Features",
        "Protocol",
        "Run",
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "Balanced acc.",
        "κ",
        "MCC",
        "Macro AUROC",
        "Macro AP",
        "Fit time",
        "Folds",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("model", "")),
                    str(row.get("feature_set", "")),
                    str(row.get("protocol", "")),
                    str(row.get("run", "")),
                    _fmt(row.get("accuracy"), row.get("accuracy_std")),
                    _fmt(row.get("macro_f1"), row.get("macro_f1_std")),
                    _fmt(row.get("weighted_f1"), row.get("weighted_f1_std")),
                    _fmt(row.get("balanced_accuracy"), row.get("balanced_accuracy_std")),
                    _fmt(row.get("cohen_kappa"), row.get("cohen_kappa_std")),
                    _fmt(row.get("mcc"), row.get("mcc_std")),
                    _fmt(row.get("macro_auroc"), row.get("macro_auroc_std")),
                    _fmt(row.get("macro_average_precision"),
                         row.get("macro_average_precision_std")),
                    _fmt(row.get("fit_time_s"), row.get("fit_time_s_std")),
                    str(row.get("n_folds", "")),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "model",
        "feature_set",
        "protocol",
        "run",
        "n_folds",
        *[k for key in _SCALAR_KEYS for k in (key, f"{key}_std")],
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    chart_series = _best_per_model_for_chart(rows)
    if chart_series:
        plot_per_class_f1_across_models(chart_series, out_dir / "per_class_f1.png")

    return md_path, csv_path


def _best_per_model_for_chart(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One series entry per model (highest macro F1 run that has per_class_f1)."""
    best: dict[str, dict[str, Any]] = {}
    for row in rows:
        pc = row.get("per_class_f1")
        if not isinstance(pc, dict) or not pc:
            continue
        model = str(row.get("model", "unknown"))
        score = row.get("macro_f1")
        prev = best.get(model)
        if prev is None or (score is not None and score > (prev.get("macro_f1") or float("-inf"))):
            best[model] = {
                "model": model,
                "macro_f1": score,
                "per_class_f1": {
                    CLASS_NAMES[i]: pc.get(CLASS_NAMES[i]) for i in sorted(CLASS_NAMES)
                },
            }
    # Stable model order: registry order, then others (late fusion etc.).
    from ml.models.registry import names

    order = {n: i for i, n in enumerate(names())}
    return sorted(best.values(), key=lambda r: (order.get(str(r["model"]), 99), str(r["model"])))


def _extract_per_class(metrics: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    pooled = metrics.get("pooled")
    if isinstance(pooled, dict) and isinstance(pooled.get("per_class"), dict):
        return pooled["per_class"]
    if isinstance(metrics.get("per_class"), dict):
        return metrics["per_class"]
    return None


def _fmt(mean: Any, std: Any = None) -> str:
    if mean is None:
        return ""
    try:
        m = float(mean)
    except (TypeError, ValueError):
        return str(mean)
    if std is None:
        return f"{m:.3f}"
    try:
        return f"{m:.3f}±{float(std):.3f}"
    except (TypeError, ValueError):
        return f"{m:.3f}"


def _run_info(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    """Model, feature set and protocol from ``summary.json`` (``ml train``) or ``extra``."""
    summary = run_dir / "summary.json"
    if summary.is_file():
        try:
            return dict(json.loads(summary.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            pass
    extra = metrics.get("extra")
    return dict(extra) if isinstance(extra, dict) else {}
