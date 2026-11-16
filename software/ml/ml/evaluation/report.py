# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Write evaluation run artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .plots import plot_confusion_matrix, plot_per_class_f1


def write_run_artifacts(
    run_dir: str | Path,
    metrics: dict[str, Any],
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = dict(metrics)
    payload["created_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if extra:
        payload["extra"] = extra

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n",
                            encoding="utf-8")

    report = metrics.get("classification_report")
    if report:
        (run_dir / "classification_report.txt").write_text(str(report) + "\n", encoding="utf-8")

    cm = metrics.get("confusion_matrix")
    if cm is not None:
        plot_confusion_matrix(cm, run_dir / "confusion_matrix.png")
        _write_cm_csv(cm, run_dir / "confusion_matrix.csv")

    per_class = metrics.get("per_class")
    if isinstance(per_class, dict) and per_class:
        plot_per_class_f1(per_class, run_dir / "per_class_f1.png")

    return run_dir


def _write_cm_csv(cm: list[list[int]], path: Path) -> None:
    from ml.config.classes import CLASS_NAMES

    labels = sorted(CLASS_NAMES)
    lines = ["true\\pred," + ",".join(CLASS_NAMES[i] for i in labels)]
    for i, row in enumerate(cm):
        lines.append(f"{CLASS_NAMES[labels[i]]}," + ",".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
