#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Accuracy from the precision and trueness reports.

Reads ``results/precision.json`` and ``results/trueness.json``. For each
ladder orientation, camera distance (near / mid / far), method, and world
axis:

* precision is the static per-axis σ
* trueness is the RMS of the metric-line residuals on that axis
* accuracy is ``sqrt(trueness² + precision²)`` for one frame

Millimetres are the plotted unit. Each bar is the world-frame error on that
axis. Percent of the ladder travel is still stored in the JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from array_pose import AXES
from optical.evaluation.recordings import write_json

STUDY_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / STUDY_DIR.name

METHODS = (("pnp", "PnP"), ("triangulation", "Triangulation"))
METRICS_MM = (("precision_mm", "Precision"), ("trueness_mm", "Trueness"), ("accuracy_mm", "Accuracy"))
METRIC_COLORS = {"precision_mm": "C0", "trueness_mm": "C1", "accuracy_mm": "C2"}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing {path}. Run the precision and trueness eval first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _rms(values: List[float]) -> float:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(v * v)))


def _precision_index(precision: dict, body: str) -> Dict[str, dict]:
    """``condition → {method → {axis → σ}}``."""
    out: Dict[str, dict] = {}
    for report in precision.get("reports") or []:
        cond = str(report.get("condition") or "")
        block = (report.get("bodies") or {}).get(body) or {}
        methods = {}
        for key, _label in METHODS:
            sigma = (block.get(key) or {}).get("axis_sigma_mm") or {}
            methods[key] = {axis: float(sigma.get(axis, float("nan"))) for axis in AXES}
        if cond:
            out[cond] = methods
    return out


def _axis_trueness(slots: List[dict], method: str, axis: str) -> float:
    residuals = []
    for slot in slots:
        block = (slot.get("methods") or {}).get(method) or {}
        err = (block.get("error_mm") or {}).get(axis, float("nan"))
        residuals.append(float(err))
    return _rms(residuals)


def _distance_trueness(slots: List[dict], method: str) -> float:
    """RMS spacing error. The reference slot is 0 by construction and is left out."""
    errors = []
    for slot in slots:
        if abs(float(slot.get("hand_mm") or 0.0)) < 1e-9:
            continue
        block = (slot.get("methods") or {}).get(method) or {}
        errors.append(float(block.get("distance_error_mm", float("nan"))))
    return _rms(errors)


def _accuracy(trueness: float, precision: float) -> float:
    if not np.isfinite(trueness) or not np.isfinite(precision):
        return float("nan")
    return float(np.sqrt(trueness * trueness + precision * precision))


def _pct(value_mm: float, travel_mm: float) -> float:
    if not np.isfinite(value_mm) or not np.isfinite(travel_mm) or travel_mm < 1.0:
        return float("nan")
    return float(100.0 * value_mm / travel_mm)


def _span_mm(slots: List[dict]) -> float:
    hand = [float(slot["hand_mm"]) for slot in slots]
    if len(hand) < 2:
        return float("nan")
    return float(max(hand) - min(hand))


def _direction(slots: List[dict], method: str) -> np.ndarray | None:
    for slot in slots:
        u = ((slot.get("methods") or {}).get(method) or {}).get("direction_u")
        if isinstance(u, list) and len(u) == 3:
            return np.asarray(u, dtype=np.float64)
    return None


def _combine(trueness: dict, precision_index: Dict[str, dict]) -> Dict[str, dict]:
    """Per orientation, range, method, and axis."""
    orientations = {}
    for name, block in (trueness.get("orientations") or {}).items():
        ranges_out = {}
        for rng, range_block in (block.get("ranges") or {}).items():
            slots = range_block.get("slots") or []
            if not slots:
                continue
            prec = precision_index.get(rng) or {}
            span = _span_mm(slots)
            methods = {}
            for key, _label in METHODS:
                direction = _direction(slots, key)
                axes = {}
                for i, axis in enumerate(AXES):
                    b = _axis_trueness(slots, key, axis)
                    sigma = (prec.get(key) or {}).get(axis, float("nan"))
                    accuracy = _accuracy(b, sigma)
                    travel = float("nan") if direction is None else float(abs(direction[i]) * span)
                    axes[axis] = {
                        "travel_mm": travel,
                        "precision_mm": sigma,
                        "trueness_mm": b,
                        "accuracy_mm": accuracy,
                        "precision_pct": _pct(sigma, travel),
                        "trueness_pct": _pct(b, travel),
                        "accuracy_pct": _pct(accuracy, travel),
                    }
                distance = _distance_trueness(slots, key)
                methods[key] = {
                    "axes": axes,
                    "trueness_distance_mm": distance,
                    "trueness_distance_pct": _pct(distance, span),
                }
            ranges_out[rng] = methods
        if ranges_out:
            orientations[name] = {
                "label": block.get("label") or name,
                "ranges": ranges_out,
            }
    return orientations


def _fmt(value: float) -> str:
    if value != value:
        return f"{'n/a':>8}"
    return f"{value:8.3f}"


def _print_report(orientations: Dict[str, dict], ranges: List[str]) -> None:
    for name, block in orientations.items():
        print(f"\n=== {name} ({block['label']}) ===")
        print("millimetres;  accuracy = sqrt(trueness² + precision²)")
        for rng in ranges:
            methods = (block.get("ranges") or {}).get(rng)
            if not methods:
                print(f"\n{rng}: no trueness rows")
                continue
            print(f"\n{rng}")
            header = f"{'method':<16} {'':<12}" + "".join(f"{axis:>8}" for axis in AXES)
            print(header)
            for key, label in METHODS:
                block_m = methods.get(key) or {}
                axes = block_m.get("axes") or {}
                for i, (metric, metric_label) in enumerate(METRICS_MM):
                    vals = "".join(_fmt((axes.get(axis) or {}).get(metric, float("nan"))) for axis in AXES)
                    tag = label if i == 0 else ""
                    print(f"{tag:<16} {metric_label:<12}{vals}")
                dist = block_m.get("trueness_distance_mm", float("nan"))
                print(f"{'':<16} {'spacing':<12}{_fmt(dist)} mm")


def plot_accuracy(
    name: str,
    label: str,
    ranges_payload: dict,
    ranges: List[str],
    out: Path,
) -> Path | None:
    """Rows are PnP / triangulation, columns are x / y / z, bars are the three distances."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not available — skipped accuracy plot")
        return None

    present = [rng for rng in ranges if rng in ranges_payload]
    if not present:
        return None
    fig, axes = plt.subplots(len(METHODS), 3, figsize=(11.2, 3.4 * len(METHODS)), squeeze=False)
    x = np.arange(len(present))
    width = 0.25
    for row, (key, method_label) in enumerate(METHODS):
        for col, axis in enumerate(AXES):
            ax = axes[row, col]
            for i, (metric, metric_label) in enumerate(METRICS_MM):
                vals = []
                for rng in present:
                    cell = (((ranges_payload.get(rng) or {}).get(key) or {}).get("axes") or {}).get(axis) or {}
                    vals.append(cell.get(metric, np.nan))
                ax.bar(
                    x + (i - 1) * width,
                    vals,
                    width,
                    label=metric_label,
                    color=METRIC_COLORS[metric],
                    alpha=0.9,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(present)
            ax.set_title(f"{method_label} — {axis}")
            if col == 0:
                ax.set_ylabel("mm")
            ax.grid(True, axis="y", alpha=0.3)
            if row == 0 and col == 2:
                ax.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Accuracy — {name} ({label});  sqrt(trueness² + precision²)",
        fontsize=11,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Accuracy from precision and trueness reports")
    ap.add_argument("--precision", type=Path, default=HOME_STUDY / "results" / "precision.json")
    ap.add_argument("--trueness", type=Path, default=HOME_STUDY / "results" / "trueness.json")
    ap.add_argument("--output", type=Path, default=HOME_STUDY / "results" / "accuracy.json")
    ap.add_argument("--plot-dir", type=Path, default=HOME_STUDY / "results")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    precision = _load(args.precision)
    trueness = _load(args.trueness)
    body = str(trueness.get("array_body") or precision.get("array_body") or "")
    ranges = [str(r) for r in (trueness.get("ranges") or precision.get("conditions") or ["near", "mid", "far"])]
    combined = _combine(trueness, _precision_index(precision, body))
    if not combined:
        raise SystemExit("No overlapping precision and trueness rows to combine.")

    _print_report(combined, ranges)
    plot_paths = []
    if not args.no_plot:
        for name, block in combined.items():
            path = plot_accuracy(
                name,
                block["label"],
                block["ranges"],
                ranges,
                args.plot_dir / f"accuracy_{name}.png",
            )
            if path is not None:
                plot_paths.append(str(path))
                print(f"Wrote {path}")

    payload = {
        "study": "5.1.1_accuracy",
        "array_body": body,
        "formula": "accuracy = sqrt(trueness^2 + precision^2)",
        "trueness": "RMS of metric-line residual on that axis",
        "precision": "static per-axis sigma from the precision hold",
        "percent": "value_mm / travel on that axis, with travel = |direction_u| * ladder span",
        "trueness_distance": "RMS of distance_error_mm over slots, reference omitted; percent of the ladder span",
        "ranges": ranges,
        "orientations": combined,
        "plots": plot_paths,
    }
    write_json(args.output, payload)
    print(f"\nWrote {args.output}")

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
