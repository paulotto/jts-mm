# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Evaluation plots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ml.config.classes import CLASS_NAMES


def plot_confusion_matrix(cm: list[list[int]] | np.ndarray, path: Path) -> None:
    cm = np.asarray(cm, dtype=np.float64)
    labels = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    row_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in (
        (axes[0], cm, "Confusion matrix (counts)"),
        (axes[1], row_norm, "Confusion matrix (row-normalized)"),
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1(
    per_class: Mapping[str, Mapping[str, Any] | float],
    path: Path,
    *,
    title: str = "Per-class F1",
) -> None:
    """Single-model per-class F1 bars (class order matches CLASS_NAMES)."""
    class_order = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    values = [_f1_value(per_class.get(name)) for name in class_order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(class_order))
    bars = ax.bar(x, values, color="#4C78A8", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.axhline(0.0, color="#333333", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar, val in zip(bars, values):
        if val is None or np.isnan(val):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(val + 0.02, 1.02),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_per_class_f1_across_models(
    series: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    title: str = "Per-class F1 across models",
) -> None:
    """Grouped bar chart: x = class, hue = model (side-by-side comparison).

    Each entry in ``series`` needs ``model`` and ``per_class_f1``
    (mapping class name → float).
    """
    class_order = [CLASS_NAMES[i] for i in sorted(CLASS_NAMES)]
    models = [str(s.get("model", "model")) for s in series]
    n_models = len(series)
    if n_models == 0:
        return

    x = np.arange(len(class_order), dtype=np.float64)
    width = min(0.8 / n_models, 0.25)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * width
    # Neutral palette (avoid purple/glow defaults).
    colors = ("#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, entry in enumerate(series):
        pc = entry.get("per_class_f1") or {}
        vals = [float(pc.get(name) or 0.0) for name in class_order]
        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            label=models[i],
            color=colors[i % len(colors)],
            edgecolor="white",
            linewidth=0.4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(class_order, rotation=30, ha="right")
    ax.set_ylabel("F1")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0.0, color="#333333", linewidth=0.6)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _f1_value(entry: Mapping[str, Any] | float | None) -> float:
    if entry is None:
        return float("nan")
    if isinstance(entry, (int, float)):
        return float(entry)
    val = entry.get("f1")
    if val is None:
        return float("nan")
    return float(val)
