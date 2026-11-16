#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""CT pivot calibration repeatability study.

Loads independent ``ct_tip_calibration.json`` runs from
``~/experiments/5_2_1_pivot_calibration_repeatability/recordings/`` and
reports tip-offset scatter plus per-run OLS residual metrics.

Usage
-----
  python analyze_repeatability.py
  python analyze_repeatability.py --input ~/experiments/5_2_1_pivot_calibration_repeatability/recordings
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / SCRIPT_DIR.name
DEFAULT_RECORDINGS = HOME_STUDY / "recordings"
TIP_JSON = "ct_tip_calibration.json"


@dataclass(frozen=True)
class Run:
    """One independent pivot calibration."""

    path: Path
    label: str
    tip_body_mm: tuple[float, float, float]
    tip_length_mm: float
    pivot_world_mm: tuple[float, float, float]
    rms_mm: float
    median_err_mm: float
    max_err_mm: float
    residual_norm: float
    n_poses: int
    cond_A: float


def _f3(v: Sequence[float] | None) -> tuple[float, float, float]:
    if not v or len(v) < 3:
        return (float("nan"), float("nan"), float("nan"))
    return (float(v[0]), float(v[1]), float(v[2]))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in v))


def discover_runs(recordings_dir: Path) -> list[Path]:
    """Find tip JSON files under ``recordings_dir`` (recursive)."""
    if not recordings_dir.is_dir():
        return []
    found = sorted(recordings_dir.rglob(TIP_JSON))
    seen: set[Path] = set()
    out: list[Path] = []
    for p in found:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        out.append(p)
    return out


def load_run(path: Path) -> Run:
    data = json.loads(path.read_text(encoding="utf-8"))
    tip = _f3(data.get("tip_body_mm"))
    tip_len = float(data.get("tip_length_mm", _norm(tip)))
    return Run(
        path=path,
        label=path.parent.name,
        tip_body_mm=tip,
        tip_length_mm=tip_len,
        pivot_world_mm=_f3(data.get("pivot_world_mm")),
        rms_mm=float(data.get("rms_mm", float("nan"))),
        median_err_mm=float(data.get("median_err_mm", float("nan"))),
        max_err_mm=float(data.get("max_err_mm", float("nan"))),
        residual_norm=float(data.get("residual_norm", float("nan"))),
        n_poses=int(data.get("n_poses", 0)),
        cond_A=float(data.get("cond_A", float("nan"))),
    )


def _mean(xs: Sequence[float]) -> float:
    return statistics.fmean(xs) if xs else float("nan")


def _stdev(xs: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0 if xs else float("nan")
    return statistics.stdev(xs)


def _rms(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def analyze(runs: Sequence[Run]) -> dict[str, Any]:
    """Tip-offset scatter + residual summary across independent calibrations."""
    tips = [r.tip_body_mm for r in runs]
    n = len(tips)
    mean_tip = tuple(_mean([t[i] for t in tips]) for i in range(3))
    std_tip = tuple(_stdev([t[i] for t in tips]) for i in range(3))
    deltas = [_norm((t[0] - mean_tip[0], t[1] - mean_tip[1], t[2] - mean_tip[2])) for t in tips]
    pairwise: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tips[i], tips[j]
            pairwise.append(_norm((a[0] - b[0], a[1] - b[1], a[2] - b[2])))

    lengths = [r.tip_length_mm for r in runs]
    rms_vals = [r.rms_mm for r in runs]
    med_vals = [r.median_err_mm for r in runs]
    max_vals = [r.max_err_mm for r in runs]
    cond_vals = [r.cond_A for r in runs]

    return {
        "n_runs": n,
        "tip_body_mm": {
            "mean": list(mean_tip),
            "std": list(std_tip),
            "std_norm_mm": _norm(std_tip),
            "max_dev_from_mean_mm": max(deltas) if deltas else float("nan"),
            "rms_dev_from_mean_mm": _rms(deltas),
            "pairwise_max_mm": max(pairwise) if pairwise else float("nan"),
            "pairwise_mean_mm": _mean(pairwise),
            "per_run_dev_from_mean_mm": deltas,
        },
        "tip_length_mm": {
            "mean": _mean(lengths),
            "std": _stdev(lengths),
            "min": min(lengths) if lengths else float("nan"),
            "max": max(lengths) if lengths else float("nan"),
            "range": (max(lengths) - min(lengths)) if lengths else float("nan"),
        },
        "ols_residuals_mm": {
            "rms_mean": _mean(rms_vals),
            "rms_std": _stdev(rms_vals),
            "rms_min": min(rms_vals) if rms_vals else float("nan"),
            "rms_max": max(rms_vals) if rms_vals else float("nan"),
            "median_err_mean": _mean(med_vals),
            "median_err_std": _stdev(med_vals),
            "max_err_mean": _mean(max_vals),
            "max_err_std": _stdev(max_vals),
            "cond_A_mean": _mean(cond_vals),
            "cond_A_max": max(cond_vals) if cond_vals else float("nan"),
        },
        "runs": [
            {
                **{k: v for k, v in asdict(r).items() if k != "path"},
                "path": str(r.path),
                "dev_from_mean_mm": deltas[i],
            }
            for i, r in enumerate(runs)
        ],
    }


def format_report(summary: dict[str, Any]) -> str:
    tip = summary["tip_body_mm"]
    length = summary["tip_length_mm"]
    ols = summary["ols_residuals_mm"]
    lines = [
        "=== CT Pivot Calibration Repeatability ===",
        f"n_runs = {summary['n_runs']}",
        "",
        "Tip offset t_tip (body frame, mm)",
        f"  mean          = [{tip['mean'][0]:8.3f}, {tip['mean'][1]:8.3f}, {tip['mean'][2]:8.3f}]",
        f"  std           = [{tip['std'][0]:8.3f}, {tip['std'][1]:8.3f}, {tip['std'][2]:8.3f}]",
        f"  ||std||       = {tip['std_norm_mm']:.3f} mm",
        f"  max |Δ mean|  = {tip['max_dev_from_mean_mm']:.3f} mm",
        f"  RMS |Δ mean|  = {tip['rms_dev_from_mean_mm']:.3f} mm",
        f"  pairwise max  = {tip['pairwise_max_mm']:.3f} mm",
        f"  pairwise mean = {tip['pairwise_mean_mm']:.3f} mm",
        "",
        "Tip length |t_tip| (mm)",
        f"  mean ± std = {length['mean']:.3f} ± {length['std']:.3f}",
        f"  range         = {length['min']:.3f} … {length['max']:.3f}  (Δ={length['range']:.3f})",
        "",
        "OLS residual metrics across runs (mm)",
        f"  rms           = {ols['rms_mean']:.3f} ± {ols['rms_std']:.3f}  [{ols['rms_min']:.3f} … {ols['rms_max']:.3f}]",
        f"  median_err    = {ols['median_err_mean']:.3f} ± {ols['median_err_std']:.3f}",
        f"  max_err       = {ols['max_err_mean']:.3f} ± {ols['max_err_std']:.3f}",
        f"  cond(A)       = {ols['cond_A_mean']:.2e}  (max {ols['cond_A_max']:.2e})",
        "",
        f"{'run':40s}  {'tx':>8} {'ty':>8} {'tz':>8}  {'|t|':>8}  {'rms':>6}  {'max':>6}  {'Δmean':>6}",
        "-" * 100,
    ]
    for r in summary["runs"]:
        t = r["tip_body_mm"]
        lines.append(
            f"{r['label'][:40]:40s}  "
            f"{t[0]:8.3f} {t[1]:8.3f} {t[2]:8.3f}  "
            f"{r['tip_length_mm']:8.2f}  "
            f"{r['rms_mm']:6.3f}  {r['max_err_mm']:6.3f}  "
            f"{r['dev_from_mean_mm']:6.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "label",
        "path",
        "tip_x_mm",
        "tip_y_mm",
        "tip_z_mm",
        "tip_length_mm",
        "rms_mm",
        "median_err_mm",
        "max_err_mm",
        "residual_norm",
        "n_poses",
        "cond_A",
        "dev_from_mean_mm",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in summary["runs"]:
            t = r["tip_body_mm"]
            writer.writerow(
                {
                    "label": r["label"],
                    "path": r["path"],
                    "tip_x_mm": t[0],
                    "tip_y_mm": t[1],
                    "tip_z_mm": t[2],
                    "tip_length_mm": r["tip_length_mm"],
                    "rms_mm": r["rms_mm"],
                    "median_err_mm": r["median_err_mm"],
                    "max_err_mm": r["max_err_mm"],
                    "residual_norm": r["residual_norm"],
                    "n_poses": r["n_poses"],
                    "cond_A": r["cond_A"],
                    "dev_from_mean_mm": r["dev_from_mean_mm"],
                }
            )


def try_plot(out_dir: Path, summary: dict[str, Any]) -> Path | None:
    """Optional scatter / residual bar chart (needs matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    runs = summary["runs"]
    labels = [r["label"] for r in runs]
    tips = [r["tip_body_mm"] for r in runs]
    mean = summary["tip_body_mm"]["mean"]
    rms = [r["rms_mm"] for r in runs]
    deltas = [r["dev_from_mean_mm"] for r in runs]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))

    ax = axes[0]
    ax.scatter([t[0] for t in tips], [t[1] for t in tips], c="C0", s=40, zorder=3)
    ax.scatter([mean[0]], [mean[1]], c="C3", marker="x", s=80, zorder=4, label="mean")
    for i, _lab in enumerate(labels):
        ax.annotate(str(i), (tips[i][0], tips[i][1]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("t_x (mm)")
    ax.set_ylabel("t_y (mm)")
    ax.set_title("Tip offset (x–y)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1]
    xs = list(range(len(runs)))
    ax.bar(xs, deltas, color="C0", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs])
    ax.set_xlabel("run index")
    ax.set_ylabel("|t − mean| (mm)")
    ax.set_title("Deviation from mean tip")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(xs, rms, color="C1", alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs])
    ax.set_xlabel("run index")
    ax.set_ylabel("OLS rms (mm)")
    ax.set_title("Per-run fit residual")
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("CT pivot calibration repeatability", fontsize=11)
    fig.tight_layout()
    plot_path = out_dir / "repeatability.png"
    fig.savefig(plot_path, dpi=140)
    plt.close(fig)
    return plot_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RECORDINGS,
        help=f"Recordings directory (default: {DEFAULT_RECORDINGS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: this script's directory / results)",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip matplotlib figure")
    args = parser.parse_args(argv)

    out_dir = (args.out or (HOME_STUDY / "results")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = args.input.expanduser()
    paths = discover_runs(recordings)
    if not paths:
        print(f"error: no {TIP_JSON} under {recordings}", file=sys.stderr)
        return 1

    runs = [load_run(p) for p in paths]
    summary = analyze(runs)
    summary["meta"] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(recordings),
        "n_files": len(paths),
    }

    report = format_report(summary)
    print(report)

    txt_path = out_dir / "repeatability_report.txt"
    json_path = out_dir / "repeatability_summary.json"
    csv_path = out_dir / "repeatability_runs.csv"
    txt_path.write_text(report + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, summary)

    plot_path = None if args.no_plot else try_plot(out_dir, summary)

    print(f"Wrote {txt_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    if plot_path is not None:
        print(f"Wrote {plot_path}")
    elif not args.no_plot:
        print("note: matplotlib not available — skipped plot", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
