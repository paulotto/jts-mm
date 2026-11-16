#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Precision: static hold of the two-marker array — PnP vs triangulation.

The metric is unchanged (r_mean±σ_r of pose about the temporal mean, first 30 s).
Poses are the array rigid body from ``study.yaml`` (``array_body``), not the CRA.
Per-axis σ (world x, y, z) is written to ``results/precision_axes.png``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np

from array_pose import (
    AXES,
    axis_sigma_mm,
    collect_body_poses,
    warn_missing_body,
)
from optical.config.loader import default_config_path, load_config_file
from optical.evaluation.precision import (
    mean_pose_from_samples,
    rotation_residual_deg,
    translation_residual_mm,
)
from optical.evaluation.recordings import (
    iter_recording_leaves,
    latest_leaf_per_condition,
    parse_condition_from_leaf,
    write_json,
)
from optical.tracking.tracker import load_tracker_for_recording

STUDY_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / STUDY_DIR.name

DEFAULT_PRECISION_CONDITIONS = ["near", "mid", "far"]
PRECISION_WINDOW_S = 30.0


def _named_leaves(recordings_dir: Path, token: str) -> List[Path]:
    """Recording leaves whose directory name contains ``token``."""
    token = token.lower()
    return [p for p in iter_recording_leaves(recordings_dir) if token in p.name.lower()]


def _frames_in_window(leaf: Path, window_s: float) -> tuple[int, int, float]:
    """Leading stored frames that fall inside ``window_s`` of the first frame.

    Uses ``t_master_us`` when every frame has one. Otherwise falls back to
    ``frame_rate_hz`` in ``meta.json``. Returns ``(n_keep, n_stored, span_s)``.
    """
    from optical.tracking.frame_store import (
        list_frame_indices,
        load_frame_timestamps,
        load_session_meta,
    )

    indices = list_frame_indices(leaf)
    n_stored = len(indices)
    if n_stored == 0:
        return 0, 0, 0.0
    stamps = load_frame_timestamps(leaf)
    times: List[int] = []
    for idx in indices:
        rec = stamps.get(idx)
        t_m = None if rec is None else rec[1]
        if t_m is None:
            times = []
            break
        times.append(int(t_m))
    if times:
        t0 = times[0]
        limit_us = window_s * 1_000_000.0
        n_keep = 0
        for t in times:
            if (t - t0) > limit_us:
                break
            n_keep += 1
        span = (times[n_keep - 1] - t0) / 1_000_000.0 if n_keep else 0.0
        return n_keep, n_stored, span
    meta = load_session_meta(leaf)
    hz = float(meta.get("frame_rate_hz") or 0.0)
    if hz <= 0:
        return n_stored, n_stored, float("nan")
    n_keep = min(n_stored, max(1, int(round(hz * window_s))))
    span = (n_keep - 1) / hz if n_keep else 0.0
    return n_keep, n_stored, span


def _fmt_pm(stats: dict, *, unit: str) -> str:
    rm = stats.get("r_mean", float("nan"))
    sr = stats.get("sigma_r", float("nan"))
    return f"{rm:.3f}±{sr:.3f}{unit}"


def _fmt_t(t) -> str:
    if not t:
        return "n/a"
    return f"[{t[0]:8.2f}, {t[1]:8.2f}, {t[2]:8.2f}] mm"


def _fmt_R(R) -> str:
    if not R:
        return "n/a"
    rows = ", ".join("[" + ", ".join(f"{v:7.4f}" for v in row) + "]" for row in R)
    return f"[{rows}]"


def _fmt_axes(sigma: dict) -> str:
    parts = [f"{axis} {sigma.get(axis, float('nan')):.3f}" for axis in AXES]
    return "[" + ", ".join(parts) + "] mm"


def _method_block(ts: np.ndarray, Rs: np.ndarray) -> dict:
    return {
        "n_valid": int(ts.shape[0]),
        "translation_mm": translation_residual_mm(ts),
        "rotation_deg": rotation_residual_deg(Rs),
        "axis_sigma_mm": axis_sigma_mm(ts),
        # Mean pose is not a precision metric — logged for sanity / setup checks.
        "mean_pose": mean_pose_from_samples(ts, Rs),
    }


def evaluate_leaf(
    leaf: Path,
    *,
    tracker,
    tcfg,
    bodies: List[str],
    pnp_camera_id: str,
    max_frames: int | None,
    condition: str,
) -> dict:
    n_frames, packed = collect_body_poses(
        leaf,
        tracker=tracker,
        tcfg=tcfg,
        bodies=bodies,
        pnp_camera_id=pnp_camera_id,
        max_frames=max_frames,
    )
    out_bodies = {}
    for body in bodies:
        sample = packed[body]
        out_bodies[body] = {
            "pnp": _method_block(sample["pnp_t"], sample["pnp_R"]),
            "triangulation": _method_block(sample["tri_t"], sample["tri_R"]),
        }
    return {
        "recording": str(leaf),
        "condition": condition,
        "n_frames": n_frames,
        "bodies": out_bodies,
    }


def _print_range_table(reports: List[dict], bodies: List[str], order: List[str]) -> None:
    """Compact comparison: r_mean±σ_r (mm / deg) and per-axis σ."""
    by_cond = {r["condition"]: r for r in reports}
    shown = [c for c in order if c in by_cond] or sorted(by_cond)
    print("\n=== Precision by range (r_mean±σ_r about temporal mean) ===")
    header = f"{'body':<8}  {'method':<14}  " + "  ".join(f"{c:>22}" for c in shown)
    print(header)
    print("-" * len(header))
    for body in bodies:
        for method, key in (("PnP", "pnp"), ("Triangulation", "triangulation")):
            cells = []
            for cond in shown:
                m = by_cond[cond]["bodies"].get(body, {}).get(key, {})
                t = _fmt_pm(m.get("translation_mm", {}), unit="mm")
                r = _fmt_pm(m.get("rotation_deg", {}), unit="°")
                cells.append(f"{t}/{r}")
            print(f"{body:<8}  {method:<14}  " + "  ".join(f"{c:>22}" for c in cells))
    print("\n=== Per-axis translation σ about the temporal mean ===")
    header = f"{'body':<8}  {'method':<14}  " + "  ".join(f"{c:>28}" for c in shown)
    print(header)
    print("-" * len(header))
    for body in bodies:
        for method, key in (("PnP", "pnp"), ("Triangulation", "triangulation")):
            cells = []
            for cond in shown:
                m = by_cond[cond]["bodies"].get(body, {}).get(key, {})
                cells.append(_fmt_axes(m.get("axis_sigma_mm") or {}))
            print(f"{body:<8}  {method:<14}  " + "  ".join(f"{c:>28}" for c in cells))


def _print_mean_poses(reports: List[dict], bodies: List[str]) -> None:
    """Print temporal mean poses (informational; not used as a precision metric)."""
    print("\n=== Mean poses (temporal mean; not a precision metric) ===")
    for r in reports:
        print(f"\n{r['condition']}  ({Path(r['recording']).name})")
        for body in bodies:
            block = r["bodies"].get(body, {})
            for label, key in (("PnP", "pnp"), ("Tri", "triangulation")):
                mp = block.get(key, {}).get("mean_pose") or {}
                print(
                    f"  {body} {label:<3}  t {_fmt_t(mp.get('t_mm'))}  "
                    f"n={mp.get('n', 0)}"
                )
                print(f"         R {_fmt_R(mp.get('R'))}")


def _condition_labels(reports: List[dict]) -> List[str]:
    totals: dict[str, int] = {}
    for r in reports:
        totals[r["condition"]] = totals.get(r["condition"], 0) + 1
    running: dict[str, int] = {}
    labels = []
    for r in reports:
        cond = r["condition"]
        running[cond] = running.get(cond, 0) + 1
        if totals[cond] == 1:
            labels.append(cond)
        else:
            labels.append(f"{cond} #{running[cond]}")
    return labels


def plot_precision_axes(reports: List[dict], body: str, out: Path) -> Path | None:
    """Grouped bars of per-axis σ for PnP and triangulation."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("note: matplotlib not available — skipped precision plot")
        return None

    labels = _condition_labels(reports)
    x = np.arange(len(reports))
    width = 0.36
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), sharey=False)
    for ax, axis in zip(axes, AXES):
        pnp = [
            r["bodies"].get(body, {}).get("pnp", {}).get("axis_sigma_mm", {}).get(axis, np.nan)
            for r in reports
        ]
        tri = [
            r["bodies"]
            .get(body, {})
            .get("triangulation", {})
            .get("axis_sigma_mm", {})
            .get(axis, np.nan)
            for r in reports
        ]
        ax.bar(x - width / 2, pnp, width, label="PnP", color="C0", alpha=0.9)
        ax.bar(x + width / 2, tri, width, label="Triangulation", color="C1", alpha=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(f"{axis}")
        ax.set_ylabel("σ about mean (mm)")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Precision — {body} per-axis translation σ", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return out


def _precision_cfg(study: dict) -> tuple[str, List[str], float]:
    block = study.get("precision") if isinstance(study.get("precision"), dict) else {}
    body = str(study.get("array_body") or "").strip()
    if not body:
        raise SystemExit("study.yaml: set array_body to the two-marker array (not the CRA)")
    conditions = [str(c) for c in (block.get("conditions") or DEFAULT_PRECISION_CONDITIONS)]
    window_s = float(block.get("window_s", PRECISION_WINDOW_S))
    return body, conditions, window_s


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Static precision of the two-marker array (near / mid / far)"
    )
    ap.add_argument("--study", type=Path, default=STUDY_DIR / "study.yaml")
    ap.add_argument("--recordings", type=Path, default=HOME_STUDY / "recordings")
    ap.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated conditions (default: study.yaml precision.conditions)",
    )
    ap.add_argument(
        "--all-leaves",
        action="store_true",
        help="Evaluate every matching leaf (default: latest per condition only)",
    )
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--output", type=Path, default=HOME_STUDY / "results" / "precision.json")
    ap.add_argument(
        "--plot",
        type=Path,
        default=HOME_STUDY / "results" / "precision_axes.png",
        help="Per-axis σ figure (x, y, z)",
    )
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    study = load_config_file(args.study)
    cfg = Path(study["optical_config"]) if study.get("optical_config") else default_config_path()
    body, default_conditions, window_s = _precision_cfg(study)
    bodies = [body]
    pnp_cam = str(study.get("pnp_camera_id", "cam0"))

    if args.conditions:
        want = [c.strip() for c in args.conditions.split(",") if c.strip()]
    else:
        want = list(default_conditions)
    if not want:
        want = list(DEFAULT_PRECISION_CONDITIONS)

    leaves = _named_leaves(args.recordings, "precision")
    if not leaves:
        raise SystemExit(
            f"No recordings with 'precision' in the name under {args.recordings}"
        )
    if args.all_leaves:
        selected: List[tuple[str, Path]] = []
        for leaf in leaves:
            cond = parse_condition_from_leaf(leaf, known=want)
            if cond in want:
                selected.append((cond, leaf))
    else:
        latest = latest_leaf_per_condition(leaves, want)
        selected = [(c, latest[c]) for c in want if c in latest]
        missing = [c for c in want if c not in latest]
        if missing:
            print(f"Warning: no recording for condition(s): {', '.join(missing)}")

    if not selected:
        raise SystemExit(
            f"No precision recordings matching {want} under {args.recordings} "
            f"(leaf name must contain 'precision'; …_{{condition}}_{{stamp}})"
        )

    print(f"Array body: {body} (not the CRA)")
    print(f"Precision window: first {window_s:.0f}s of each take")
    reports = []
    warned = False
    for cond, leaf in selected:
        n_keep, n_stored, span_s = _frames_in_window(leaf, window_s)
        limit = n_keep
        cap = ""
        if args.max_frames is not None and args.max_frames < n_keep:
            limit = args.max_frames
            cap = f", capped at {limit} frames"
        print(
            f"Evaluating {leaf.name}  [{cond}]  "
            f"{n_keep}/{n_stored} frames ({span_s:.2f}s){cap} …"
        )
        tracker, tcfg = load_tracker_for_recording(cfg, leaf)
        if not warned:
            warn_missing_body(tracker, body)
            warned = True
        report = evaluate_leaf(
            leaf,
            tracker=tracker,
            tcfg=tcfg,
            bodies=bodies,
            pnp_camera_id=pnp_cam,
            max_frames=limit,
            condition=cond,
        )
        report["window_s"] = window_s
        report["n_frames_stored"] = n_stored
        reports.append(report)

    payload = {
        "study": "5.1.1_precision",
        "metric": "r_mean±sigma_r about temporal mean pose; per-axis sigma of translation",
        "array_body": body,
        "window_s": window_s,
        "config": str(cfg),
        "bodies": bodies,
        "conditions": want,
        "reports": reports,
    }
    plot_path = None
    if not args.no_plot:
        plot_path = plot_precision_axes(reports, body, args.plot)
        if plot_path is not None:
            payload["plot"] = str(plot_path)
    write_json(args.output, payload)
    print(f"\nWrote {args.output} ({len(reports)} recording(s))")
    if plot_path is not None:
        print(f"Wrote {plot_path}")
    for r in reports:
        print(f"\n{r['condition']}  frames={r['n_frames']}  ({Path(r['recording']).name})")
        for b in bodies:
            m = r["bodies"].get(b, {})
            if not m:
                continue
            print(
                f"  {b}: PnP  t {_fmt_pm(m['pnp']['translation_mm'], unit='mm')}  "
                f"R {_fmt_pm(m['pnp']['rotation_deg'], unit='°')} "
                f"axes {_fmt_axes(m['pnp']['axis_sigma_mm'])} | "
                f"Tri  t {_fmt_pm(m['triangulation']['translation_mm'], unit='mm')}  "
                f"R {_fmt_pm(m['triangulation']['rotation_deg'], unit='°')} "
                f"axes {_fmt_axes(m['triangulation']['axis_sigma_mm'])}"
            )
    _print_mean_poses(reports, bodies)
    if not args.all_leaves:
        _print_range_table(reports, bodies, want)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
