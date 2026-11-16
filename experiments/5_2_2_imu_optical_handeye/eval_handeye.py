#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""IMU–optical hand-eye calibration repeatability.

Independent repetitions of the same rigid IMU–array mount (not per-subject).
Reports geodesic scatter of ``R_mount`` plus Park–Martin fit residuals,
matching the structure of the CT tip-offset scatter + OLS residuals of
``5_2_1_pivot_calibration_repeatability``.

Modes
-----
* default — use stored ``imu_mounting.json``
* ``--from-session`` — re-run Park–Martin on each leaf ``session.h5``

Usage
-----
  python eval_handeye.py
  python eval_handeye.py --from-session
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / SCRIPT_DIR.name
DEFAULT_RECORDINGS = HOME_STUDY / "recordings"
IMU_JSON = "imu_mounting.json"
SESSION_H5 = "session.h5"
SLOTS = ("imu_mta", "imu_cra")


@dataclass(frozen=True)
class SlotRun:
    """One IMU slot from one persist leaf."""

    path: Path
    label: str
    run: int
    slot: str
    accepted: bool
    rms_deg: float | None
    n_pairs: int | None
    n_samples: int | None
    error: str | None
    R_mount: np.ndarray | None
    axes_3d: bool | None
    axis_ratio_opt: tuple[float, float] | None
    axis_ratio_imu: tuple[float, float] | None


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x):
        return None
    return x


def _i(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_R(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.shape == (4, 4):
        arr = arr[:3, :3]
    if arr.shape != (3, 3) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _ratio_pair(raw: Any) -> tuple[float, float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    a, b = _f(raw[0]), _f(raw[1])
    if a is None or b is None:
        return None
    return (a, b)


def _run_index(data: dict[str, Any]) -> int:
    raw = data.get("run", data.get("attempt", 0))
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def discover_runs(recordings_dir: Path) -> list[Path]:
    if not recordings_dir.is_dir():
        return []
    found = sorted(recordings_dir.rglob(IMU_JSON))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(path)
    return out


def _slot_run(leaf: Path, data: dict[str, Any], slot: str, rec: dict[str, Any]) -> SlotRun:
    return SlotRun(
        path=leaf / IMU_JSON,
        label=leaf.name,
        run=_run_index(data),
        slot=slot,
        accepted=bool(rec.get("accepted")),
        rms_deg=_f(rec.get("rms_deg")),
        n_pairs=_i(rec.get("n_pairs")),
        n_samples=_i(rec.get("n_samples")),
        error=str(rec["error"]) if rec.get("error") else None,
        R_mount=_as_R(rec.get("R_mount") or rec.get("T")),
        axes_3d=None if rec.get("axes_3d") is None else bool(rec.get("axes_3d")),
        axis_ratio_opt=_ratio_pair(rec.get("axis_ratio_opt")),
        axis_ratio_imu=_ratio_pair(rec.get("axis_ratio_imu")),
    )


def load_leaf_slots(path: Path) -> list[SlotRun]:
    data = json.loads(path.read_text(encoding="utf-8"))
    leaf = path.parent
    slots = data.get("slots") or {}
    out: list[SlotRun] = []
    for slot in SLOTS:
        rec = slots.get(slot)
        if not isinstance(rec, dict):
            continue
        out.append(_slot_run(leaf, data, slot, rec))
    return out


def recompute_leaf_slots(path: Path, *, max_rms_deg: float) -> list[SlotRun]:
    from jtsmm.imu.mounting import PHASE_BY_SLOT, MountingFitError, fit_session_slot, read_phase_windows
    from jtsmm.imu.persist import slot_record_from_error

    data = json.loads(path.read_text(encoding="utf-8"))
    leaf = path.parent
    session = leaf / SESSION_H5
    if not session.is_file():
        raise FileNotFoundError(f"{leaf}: missing {SESSION_H5}")
    windows = read_phase_windows(session)
    slots_out: dict[str, dict[str, Any]] = {}
    for slot in SLOTS:
        phase = PHASE_BY_SLOT.get(slot)
        t0 = t1 = None
        if phase and phase in windows:
            t0, t1 = windows[phase]
        try:
            result = fit_session_slot(
                session,
                slot,
                t0_us=t0,
                t1_us=t1,
                phase=phase,
                max_rms_deg=max_rms_deg,
                strict=False,
            )
            slots_out[slot] = result.to_slot_dict()
        except MountingFitError as exc:
            slots_out[slot] = slot_record_from_error(slot, str(exc), phase=phase)
    data["slots"] = slots_out
    return [_slot_run(leaf, data, slot, rec) for slot, rec in slots_out.items()]


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


def _project_so3(M: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u = u.copy()
        u[:, -1] *= -1
        R = u @ vt
    return R


def _so3_log(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64)
    cos_th = float(np.clip((np.trace(m) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(cos_th))
    if th < 1e-9:
        return np.zeros(3, dtype=np.float64)
    w = np.array(
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]],
        dtype=np.float64,
    )
    return w * (th / (2.0 * np.sin(th)))


def geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(_so3_log(R_a.T @ R_b))))


def chordal_mean(Rs: Sequence[np.ndarray]) -> np.ndarray:
    acc = np.zeros((3, 3), dtype=np.float64)
    for R in Rs:
        acc += np.asarray(R, dtype=np.float64)
    return _project_so3(acc)


def _scatter_block(group: Sequence[SlotRun]) -> dict[str, Any]:
    usable = [r for r in group if r.R_mount is not None]
    rms_vals = [r.rms_deg for r in usable if r.rms_deg is not None]
    pair_vals = [float(r.n_pairs) for r in usable if r.n_pairs is not None]
    if not usable:
        return {
            "n": len(group),
            "n_with_R": 0,
            "R_mount": None,
            "fit_rms_deg": {},
            "n_pairs": {},
        }
    Rs = [r.R_mount for r in usable if r.R_mount is not None]
    mean_R = chordal_mean(Rs)
    deltas = [geodesic_deg(mean_R, R) for R in Rs]
    pairwise: list[float] = []
    for i in range(len(Rs)):
        for j in range(i + 1, len(Rs)):
            pairwise.append(geodesic_deg(Rs[i], Rs[j]))
    rotvecs = [np.degrees(_so3_log(mean_R.T @ R)).tolist() for R in Rs]
    return {
        "n": len(group),
        "n_with_R": len(usable),
        "R_mount": {
            "mean": mean_R.tolist(),
            "std_deg": _stdev(deltas),
            "max_dev_from_mean_deg": max(deltas) if deltas else float("nan"),
            "rms_dev_from_mean_deg": _rms(deltas),
            "pairwise_max_deg": max(pairwise) if pairwise else float("nan"),
            "pairwise_mean_deg": _mean(pairwise),
            "per_run_dev_from_mean_deg": deltas,
            "per_run_rotvec_deg": rotvecs,
        },
        "fit_rms_deg": {
            "mean": _mean(rms_vals),
            "std": _stdev(rms_vals),
            "min": min(rms_vals) if rms_vals else float("nan"),
            "max": max(rms_vals) if rms_vals else float("nan"),
        },
        "n_pairs": {
            "mean": _mean(pair_vals),
            "std": _stdev(pair_vals),
            "min": min(pair_vals) if pair_vals else float("nan"),
            "max": max(pair_vals) if pair_vals else float("nan"),
        },
    }


def analyze(runs: Sequence[SlotRun]) -> dict[str, Any]:
    per_slot: dict[str, Any] = {}
    run_rows: list[dict[str, Any]] = []
    for slot in SLOTS:
        group = [r for r in runs if r.slot == slot]
        block = _scatter_block(group)
        per_slot[slot] = block
        deltas = (block.get("R_mount") or {}).get("per_run_dev_from_mean_deg") or []
        usable = [r for r in group if r.R_mount is not None]
        for i, r in enumerate(usable):
            run_rows.append(
                {
                    "label": r.label,
                    "path": str(r.path),
                    "run": r.run,
                    "slot": r.slot,
                    "accepted": r.accepted,
                    "rms_deg": r.rms_deg,
                    "n_pairs": r.n_pairs,
                    "n_samples": r.n_samples,
                    "error": r.error,
                    "dev_from_mean_deg": deltas[i] if i < len(deltas) else None,
                    "axes_3d": r.axes_3d,
                    "axis_ratio_opt": list(r.axis_ratio_opt) if r.axis_ratio_opt else None,
                    "axis_ratio_imu": list(r.axis_ratio_imu) if r.axis_ratio_imu else None,
                }
            )
        for r in group:
            if r.R_mount is not None:
                continue
            run_rows.append(
                {
                    "label": r.label,
                    "path": str(r.path),
                    "run": r.run,
                    "slot": r.slot,
                    "accepted": r.accepted,
                    "rms_deg": r.rms_deg,
                    "n_pairs": r.n_pairs,
                    "n_samples": r.n_samples,
                    "error": r.error,
                    "dev_from_mean_deg": None,
                    "axes_3d": r.axes_3d,
                    "axis_ratio_opt": list(r.axis_ratio_opt) if r.axis_ratio_opt else None,
                    "axis_ratio_imu": list(r.axis_ratio_imu) if r.axis_ratio_imu else None,
                }
            )
    return {
        "n_runs": len({r.label for r in runs}),
        "n_slot_runs": len(runs),
        "slots": per_slot,
        "runs": run_rows,
    }


def format_report(summary: dict[str, Any]) -> str:
    lines = [
        "=== IMU–optical hand-eye calibration repeatability ===",
        f"n_runs = {summary['n_runs']}   n_slot_runs = {summary['n_slot_runs']}",
        "Independent repetitions of one rigid IMU–array pair (not per-subject).",
        "",
    ]
    for slot, block in summary["slots"].items():
        scatter = block.get("R_mount") or {}
        rms = block.get("fit_rms_deg") or {}
        pairs = block.get("n_pairs") or {}
        lines.append(f"{slot}  (n={block.get('n', 0)}, with R={block.get('n_with_R', 0)})")
        if scatter:
            mean = scatter.get("mean") or []
            lines.append("  R_mount geodesic scatter vs chordal mean (deg)")
            if mean and len(mean) == 3:
                lines.append(
                    "  mean R        = ["
                    + "; ".join(
                        "[" + ", ".join(f"{v:7.4f}" for v in row) + "]" for row in mean
                    )
                    + "]"
                )
            lines.append(f"  std           = {scatter.get('std_deg', float('nan')):.3f}°")
            lines.append(
                f"  max |Δ mean|  = {scatter.get('max_dev_from_mean_deg', float('nan')):.3f}°"
            )
            lines.append(
                f"  RMS |Δ mean|  = {scatter.get('rms_dev_from_mean_deg', float('nan')):.3f}°"
            )
            lines.append(
                f"  pairwise max  = {scatter.get('pairwise_max_deg', float('nan')):.3f}°"
            )
            lines.append(
                f"  pairwise mean = {scatter.get('pairwise_mean_deg', float('nan')):.3f}°"
            )
        if rms:
            lines.append("  Park–Martin residual RMS (deg)")
            lines.append(
                f"  rms           = {rms.get('mean', float('nan')):.3f} ± "
                f"{rms.get('std', float('nan')):.3f}  "
                f"[{rms.get('min', float('nan')):.3f} … {rms.get('max', float('nan')):.3f}]"
            )
        if pairs:
            lines.append(
                f"  n_pairs       = {pairs.get('mean', float('nan')):.1f} ± "
                f"{pairs.get('std', float('nan')):.1f}  "
                f"[{pairs.get('min', float('nan')):.0f} … {pairs.get('max', float('nan')):.0f}]"
            )
        n_axes = sum(1 for r in summary["runs"] if r.get("slot") == slot and r.get("axes_3d") is True)
        n_known = sum(1 for r in summary["runs"] if r.get("slot") == slot and r.get("axes_3d") is not None)
        if n_known:
            lines.append(f"  axes 3/3      = {n_axes}/{n_known} takes")
        lines.append("")
    lines.append(
        f"{'run':40s} {'slot':8s} {'rms':>7} {'pairs':>6} {'Δmean':>7} {'axes':>5}"
    )
    lines.append("-" * 84)
    for r in summary["runs"]:
        rms = r.get("rms_deg")
        rms_s = f"{rms:7.3f}" if rms is not None else f"{'n/a':>7}"
        pairs = r.get("n_pairs")
        pairs_s = f"{pairs:6d}" if pairs is not None else f"{'n/a':>6}"
        dlt = r.get("dev_from_mean_deg")
        dlt_s = f"{dlt:7.3f}" if dlt is not None else f"{'n/a':>7}"
        if r.get("axes_3d") is True:
            axes_s = "  3/3"
        elif r.get("axes_3d") is False:
            axes_s = "  2/3"
        else:
            axes_s = "  n/a"
        lines.append(
            f"{r['label'][:40]:40s} {r['slot']:8s} {rms_s} {pairs_s} {dlt_s} {axes_s}"
        )
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, summary: dict[str, Any]) -> None:
    fields = [
        "label",
        "path",
        "run",
        "slot",
        "accepted",
        "n_pairs",
        "n_samples",
        "rms_deg",
        "dev_from_mean_deg",
        "axes_3d",
        "axis_ratio_opt",
        "axis_ratio_imu",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in summary["runs"]:
            writer.writerow({k: r.get(k) for k in fields})


SLOT_STYLE = {
    "imu_mta": {"color": "C0", "short": "MTA"},
    "imu_cra": {"color": "C1", "short": "CRA"},
}


def _take_keys(rows: Sequence[dict[str, Any]]) -> list[tuple[int, str]]:
    """Unique persist takes, ordered by run index then leaf name."""
    seen: list[tuple[int, str]] = []
    for r in rows:
        key = (int(r.get("run") or 0), str(r.get("label") or ""))
        if key not in seen:
            seen.append(key)
    return sorted(seen, key=lambda k: (k[0] if k[0] else 10**9, k[1]))


def _row_for(rows: Sequence[dict[str, Any]], take: tuple[int, str], slot: str) -> dict[str, Any] | None:
    run, label = take
    for r in rows:
        if int(r.get("run") or 0) == run and str(r.get("label") or "") == label and r.get("slot") == slot:
            return r
    return None


def _grouped_slot_bars(
    ax: Any,
    rows: Sequence[dict[str, Any]],
    *,
    value_key: str,
    ylabel: str,
    title: str,
) -> None:
    takes = _take_keys(rows)
    n = len(takes)
    if n == 0:
        ax.set_title(title)
        return
    width = 0.36
    xs = list(range(n))
    for i, slot in enumerate(SLOTS):
        style = SLOT_STYLE[slot]
        vals: list[float] = []
        accepted: list[bool] = []
        for take in takes:
            row = _row_for(rows, take, slot)
            raw = None if row is None else row.get(value_key)
            vals.append(float("nan") if raw is None else float(raw))
            accepted.append(bool(row.get("accepted")) if row is not None else False)
        offset = (i - 0.5) * width
        bars = ax.bar(
            [x + offset for x in xs],
            [0.0 if math.isnan(v) else v for v in vals],
            width=width,
            color=style["color"],
            alpha=0.9,
            label=style["short"],
            zorder=3,
        )
        for bar, ok, val in zip(bars, accepted, vals):
            if math.isnan(val):
                bar.set_visible(False)
            elif not ok:
                bar.set_hatch("///")
                bar.set_alpha(0.55)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(run) if run else label[-8:] for run, label in takes])
    ax.set_xlabel("take (run)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8, title="IMU")
    ax.grid(True, axis="y", alpha=0.3)


def try_plot(out_dir: Path, summary: dict[str, Any]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    rows = list(summary["runs"])

    ax = axes[0]
    for slot in SLOTS:
        style = SLOT_STYLE[slot]
        group = [r for r in rows if r.get("slot") == slot]
        vecs = (summary["slots"].get(slot) or {}).get("R_mount") or {}
        rotvecs = vecs.get("per_run_rotvec_deg") or []
        usable = [r for r in group if r.get("dev_from_mean_deg") is not None]
        if not rotvecs or len(rotvecs) != len(usable):
            continue
        xs = [v[0] for v in rotvecs]
        ys = [v[1] for v in rotvecs]
        ax.scatter(xs, ys, c=style["color"], s=44, zorder=3, label=style["short"])
        for r, x, y in zip(usable, xs, ys):
            ax.annotate(
                str(r.get("run") or "?"),
                (x, y),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.axvline(0.0, color="0.6", lw=0.8)
    ax.set_xlabel("rotvec x vs mean (deg)")
    ax.set_ylabel("rotvec y vs mean (deg)")
    ax.set_title("R_mount scatter (axis-angle)")
    ax.legend(loc="best", fontsize=8, title="IMU")
    ax.grid(True, alpha=0.3)

    _grouped_slot_bars(
        axes[1],
        rows,
        value_key="dev_from_mean_deg",
        ylabel="|R − mean| (deg)",
        title="Deviation from mean R_mount",
    )
    _grouped_slot_bars(
        axes[2],
        rows,
        value_key="rms_deg",
        ylabel="Park–Martin RMS (deg)",
        title="Per-run fit residual",
    )

    fig.suptitle("IMU–optical hand-eye repeatability", fontsize=11)
    if any(not r.get("accepted") for r in rows):
        fig.text(
            0.01,
            0.01,
            "hatched = fit rejected (RMS > 2°)",
            fontsize=8,
            color="0.35",
        )
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
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
    parser.add_argument(
        "--from-session",
        action="store_true",
        help="Re-run Park–Martin on each leaf session.h5 instead of trusting JSON",
    )
    try:
        from jtsmm.imu.mounting import MAX_RMS_DEG as default_max_rms
    except ImportError:
        default_max_rms = 3.0
    parser.add_argument("--max-rms-deg", type=float, default=float(default_max_rms))
    args = parser.parse_args(argv)

    out_dir = (args.out or (HOME_STUDY / "results")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = args.input.expanduser()
    json_paths = discover_runs(recordings)
    if not json_paths:
        print(f"error: no {IMU_JSON} under {recordings}", file=sys.stderr)
        return 1

    runs: list[SlotRun] = []
    errors: list[str] = []
    for path in json_paths:
        try:
            if args.from_session:
                runs.extend(recompute_leaf_slots(path, max_rms_deg=args.max_rms_deg))
            else:
                runs.extend(load_leaf_slots(path))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path.parent.name}: {exc}")

    if not runs:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1

    summary = analyze(runs)
    summary["meta"] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(recordings),
        "n_files": len(json_paths),
        "n_evaluated": len(runs),
        "mode": "from_session" if args.from_session else "stored_json",
        "errors": errors,
    }

    report = format_report(summary)
    if errors:
        report += "Skipped / failed leaves\n"
        for err in errors:
            report += f"  • {err}\n"
        report += "\n"
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
    n_h5 = sum(1 for p in json_paths if (p.parent / SESSION_H5).is_file())
    if n_h5 < len(json_paths):
        print(
            f"note: {n_h5}/{len(json_paths)} leaves have {SESSION_H5}; "
            "re-record with --persist-imu-calibration for --from-session",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
