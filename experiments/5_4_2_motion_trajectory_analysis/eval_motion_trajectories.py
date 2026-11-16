#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Motion amplitude and rep-to-rep consistency.

Prefers the JTS landmark (``jaw_motion.h5`` next to the take, or produced with
``--run-jts``); otherwise the optical relative pose (mandibular marker origin
in the cranial marker frame). Traces and the opening signal come from
``biomech.kinematics.traces``; cycles are peak-to-peak
(``biomech.kinematics.peak_cycles``, the definition of the earlier evaluation,
kept for comparability), time-normalised with ``biomech.kinematics.cycles.time_normalize``.

Usage
-----
  python eval_motion_trajectories.py
  python eval_motion_trajectories.py --data-dir ~/jtsmm_data --subject s03
  python eval_motion_trajectories.py --run-jts
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from biomech.io.inventory import TakeInventory, scan, subject_pids
from biomech.io.session import SessionData
from biomech.kinematics.cycles import time_normalize
from biomech.kinematics.peak_cycles import nan_fraction, peak_to_peak_cycles
from biomech.kinematics.traces import load_kinematic_trace, opening_1d, sample_rate_hz
from biomech.paths import resolve_data_dir, results_dir
from biomech.results.provenance import write_csv, write_json, write_provenance

STUDY = "5_4_2_motion_trajectory_analysis"
CYCLE_N = 101
TRAJ_MOTIONS = ("open_close", "protrusion_retrusion", "left_right", "chewing")


def maybe_run_jts(rec: TakeInventory, dest: Path) -> Path | None:
    """``jaw_motion.h5`` for the take: next to it, cached in ``dest``, or computed now.

    Runs ``jtsmm.tools.run_jts_analysis`` with this interpreter (``sys.executable``);
    the raw take is never modified (no ``--append-session``).
    """
    existing = rec.path / "jaw_motion.h5"
    if existing.is_file():
        return existing
    cached = dest / "jaw_motion.h5"
    if cached.is_file():
        return cached
    if not rec.has_jts_config:
        return None
    dest.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "jtsmm.tools.run_jts_analysis", str(rec.h5),
           "--out", str(dest), "--no-show"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"JTS skip {rec.name}: {exc}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"JTS failed {rec.name}: {proc.stderr[-400:]}", file=sys.stderr)
        return None
    return cached if cached.is_file() else None


AXIS_NAMES = ("X", "Y", "Z")


def _axis_pair(xyz: np.ndarray) -> tuple[int, int]:
    rom = np.nanmax(xyz, axis=0) - np.nanmin(xyz, axis=0)
    rom = np.where(np.isfinite(rom), rom, -1.0)
    order = np.argsort(rom)[::-1]
    return int(order[0]), int(order[1] if order.size > 1 else 0)


def _phase_mask(t: np.ndarray, ph: Any) -> np.ndarray:
    return (t >= int(ph.t_start_us)) & (t <= int(ph.t_end_us))


def eval_session(rec: TakeInventory, derived: Path, run_jts: bool
                 ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    jts_path = rec.path / "jaw_motion.h5"
    if run_jts:
        got = maybe_run_jts(rec, derived / rec.name)
        if got is not None:
            jts_path = got
    packed = load_kinematic_trace(rec.path, jts_path)
    if packed is None:
        return [], {"session": rec.name, "subject_id": rec.subject_id, "source": None,
                    "error": "no trajectory", "cycles": {}}, {
            "session": rec.name, "subject_id": rec.subject_id, "xyz": np.zeros((0, 3)),
            "motions": {}}
    t, xyz, source = packed
    with SessionData(rec.path) as sd:
        phases = list(sd.phases)
    rest = next((p for p in phases if p.name == "rest"), None)
    rest_xyz = xyz[_phase_mask(t, rest)] if rest is not None else None
    amp = opening_1d(xyz, rest_xyz)
    fs = sample_rate_hz(t, 60.0)
    rows: list[dict[str, Any]] = []
    cycles_out: dict[str, list[list[float]]] = {}
    nan_out: dict[str, list[float]] = {}
    motion_spans: dict[str, dict[str, Any]] = {}
    for p in phases:
        if p.name not in TRAJ_MOTIONS:
            continue
        m = _phase_mask(t, p)
        idx = np.flatnonzero(m)
        yy = amp[m]
        y = yy[np.isfinite(yy)]
        if y.size < 8:
            continue
        baseline = float(np.percentile(y, 5))
        peak = float(np.max(y))
        rom = peak - baseline
        segs = peak_to_peak_cycles(yy, fs, t_s=t[m].astype(np.float64) * 1e-6)
        norms, nan_fracs = [], []
        for a, b in segs:
            cyc = yy[a: b + 1]
            if np.isfinite(cyc).sum() < 4:
                continue
            norms.append(time_normalize(cyc, CYCLE_N))
            nan_fracs.append(nan_fraction(yy, a, b))
        motion_spans[p.name] = {"idx": idx, "segs": segs}
        if norms:
            stack = np.vstack(norms)
            mean = np.nanmean(stack, axis=0)
            rms = float(np.sqrt(np.nanmean((stack - mean) ** 2)))
            cv = float(np.nanstd([np.nanmax(c) - np.nanmin(c) for c in stack]) / max(rom, 1e-6))
            cycles_out[p.name] = [c.tolist() for c in norms]
            nan_out[p.name] = nan_fracs
        else:
            rms = cv = float("nan")
        rows.append(
            {
                "session": rec.name,
                "pid": rec.pid,
                "subject_id": rec.subject_id,
                "motion": p.name,
                "source": source,
                "n_samples": int(y.size),
                "n_cycles": int(len(norms)),
                "n_cycles_gappy": int(sum(f > 0.1 for f in nan_fracs)),
                "amplitude_mm": rom,
                "peak_mm": peak,
                "baseline_mm": baseline,
                "rep_rms_mm": rms,
                "rep_cv": cv,
                "fs_hz": fs,
            }
        )
    doc = {
        "session": rec.name,
        "pid": rec.pid,
        "subject_id": rec.subject_id,
        "source": source,
        "cycle_definition": "peak_to_peak",
        "cycles": cycles_out,
        "cycle_nan_frac": nan_out,
    }
    view = {"session": rec.name, "subject_id": rec.subject_id, "xyz": xyz,
            "motions": motion_spans}
    return rows, doc, view


def plot_amp(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    motions = [m for m in TRAJ_MOTIONS if any(r["motion"] == m for r in rows)]
    subjects = sorted({r["subject_id"] for r in rows if r["subject_id"]})
    x = np.arange(len(motions))
    width = 0.8 / max(len(subjects), 1)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for i, sid in enumerate(subjects):
        vals = []
        for m in motions:
            vs = [r["amplitude_mm"] for r in rows if r["subject_id"] == sid and r["motion"] == m]
            vals.append(float(np.nanmean(vs)) if vs else np.nan)
        ax.bar(x + i * width, vals, width, label=sid)
    ax.set_xticks(x + width * (len(subjects) - 1) / 2)
    ax.set_xticklabels(motions, rotation=15)
    ax.set_ylabel("amplitude (mm)")
    ax.set_title("Movement amplitude")
    ax.legend()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_cycles(cycle_docs: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    motions = [m for m in TRAJ_MOTIONS]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    xi = np.linspace(0, 100, CYCLE_N)
    for ax, motion in zip(axes.ravel(), motions):
        for doc in cycle_docs:
            for cyc in doc.get("cycles", {}).get(motion, []):
                ax.plot(xi, cyc, color="#7f8c8d", alpha=0.35, lw=0.8)
        ax.set_title(motion)
        ax.set_xlabel("% cycle")
        ax.set_ylabel("opening (mm)")
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _session_clock(name: str) -> str:
    for p in str(name).split("_"):
        if "T" in p and p[0:2].isdigit():
            i = p.find("T")
            hhmm = p[i + 1 : i + 5]
            if len(hhmm) == 4 and hhmm.isdigit():
                return f"{hhmm[:2]}:{hhmm[2:]}"
    return str(name)[-8:]


def _draw_cycle_xy_panel(ax, xyz: np.ndarray, bag: dict[str, Any], *, cmap) -> None:
    idx = np.asarray(bag.get("idx") if bag else [], dtype=int)
    segs = list((bag or {}).get("segs") or [])
    if idx.size < 2:
        ax.text(0.5, 0.5, "no phase", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        return
    pts = xyz[idx]
    ai, aj = _axis_pair(pts)
    ax.plot(pts[:, ai], pts[:, aj], color="#d5d5d5", lw=0.75, zorder=1)
    nseg = max(len(segs), 1)
    for k, pair in enumerate(segs):
        a, b = int(pair[0]), int(pair[1])
        if a < 0 or b >= len(pts) or b <= a:
            continue
        cyc = pts[a : b + 1]
        color = cmap(0.08 + 0.84 * k / nseg)
        ax.plot(cyc[:, ai], cyc[:, aj], color=color, lw=1.1, zorder=2, solid_capstyle="round")
        ax.plot(cyc[0, ai], cyc[0, aj], "o", color=color, ms=3.2, zorder=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(f"{AXIS_NAMES[ai]} (mm)")
    ax.set_ylabel(f"{AXIS_NAMES[aj]} (mm)")
    ax.text(
        0.98,
        0.03,
        f"n={len(segs)}",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=7,
        color="#444",
    )


def plot_cycle_xy(views: list[dict[str, Any]], out: Path) -> None:
    """2D path per motion; each peak-to-peak cycle in a different colour."""
    import matplotlib.pyplot as plt

    views = [v for v in views if np.asarray(v.get("xyz", [])).size >= 8]
    if not views:
        return
    n = len(views)
    fig, axes = plt.subplots(n, 4, figsize=(12.8, 2.6 * n), squeeze=False, constrained_layout=True)
    fig.suptitle("Peak-to-peak cycles on the 2D trajectory (plane = two largest ROM axes)", fontsize=11)
    cmap = plt.cm.plasma
    for row, view in enumerate(views):
        xyz = np.asarray(view["xyz"], dtype=float)
        sid = view.get("subject_id") or "?"
        clock = _session_clock(str(view.get("session") or ""))
        for col, motion in enumerate(TRAJ_MOTIONS):
            ax = axes[row, col]
            _draw_cycle_xy_panel(ax, xyz, (view.get("motions") or {}).get(motion) or {}, cmap=cmap)
            if row == 0 and col != 0:
                ax.set_title(motion)
            if col == 0:
                extra = f"{motion}\n" if row == 0 else ""
                ax.set_title(f"{extra}{sid}  {clock}", fontsize=9)
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_cycle_xy_session(view: dict[str, Any], out: Path) -> None:
    import matplotlib.pyplot as plt

    xyz = np.asarray(view.get("xyz", []), dtype=float)
    if xyz.size < 8:
        return
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.8), constrained_layout=True)
    sid = view.get("subject_id") or ""
    clock = _session_clock(str(view.get("session") or ""))
    fig.suptitle(f"Peak-to-peak cycles on 2D path — {sid}  {clock}", fontsize=11)
    cmap = plt.cm.plasma
    for ax, motion in zip(axes.ravel(), TRAJ_MOTIONS):
        _draw_cycle_xy_panel(ax, xyz, (view.get("motions") or {}).get(motion) or {}, cmap=cmap)
        ax.set_title(motion)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=None,
                    help=f"default: $BIOMECH_RESULTS_DIR or ~/experiments, then /{STUDY}/results")
    ap.add_argument("--run-jts", action="store_true")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    recs = [r for r in scan(data_dir, pids=subject_pids(args.subject), kind="motion")
            if r.has_optical]

    out = results_dir(STUDY, args.out_dir).resolve()
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    cycle_docs: list[dict[str, Any]] = []
    xy_views: list[dict[str, Any]] = []
    for rec in recs:
        r, doc, view = eval_session(rec, derived, args.run_jts)
        rows.extend(r)
        cycle_docs.append(doc)
        xy_views.append(view)
        write_json(derived / f"cycles_{rec.name}.json", doc)

    write_csv(out / "trajectory_metrics.csv", rows)
    write_json(out / "trajectory_summary.json", {"n_rows": len(rows), "n_sessions": len(recs)})
    write_provenance(derived, STUDY, "eval_motion_trajectories.py", recs, data_dir=data_dir,
                     extra={"cycle_definition": "peak_to_peak", "run_jts": args.run_jts})
    if not args.no_plot and rows:
        import matplotlib

        matplotlib.use("Agg")
        plot_amp(rows, out / "amplitude_bars.png")
        plot_cycles(cycle_docs, out / "cycle_overlays.png")
        plot_cycle_xy(xy_views, out / "cycle_xy_paths.png")
        if xy_views:
            def _open_rom(v: dict[str, Any]) -> float:
                xyz = np.asarray(v.get("xyz"), dtype=float)
                bag = (v.get("motions") or {}).get("open_close") or {}
                idx = np.asarray(bag.get("idx", []), dtype=int)
                if idx.size < 2 or xyz.size < 8:
                    return -1.0
                pts = xyz[idx]
                rom = np.nanmax(pts, axis=0) - np.nanmin(pts, axis=0)
                return float(np.nansum(np.where(np.isfinite(rom), rom, 0.0)))

            plot_cycle_xy_session(max(xy_views, key=_open_rom), out / "cycle_xy_example.png")
    print(f"sessions={len(recs)} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
