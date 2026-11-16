#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Timing residuals of the reconstructed camera time base.

For every accepted camera row, recompute the rolling affine residual used by
``ClockAligner`` (window=30):

    r_f = T_trigger − (a · SensorTimestamp_µs + b)
    σ_c(t) = residual_rms of the current window

Usage
-----
  python eval_timing_residuals.py
  python eval_timing_residuals.py --data-dir ~/jtsmm_data --subject s03
  python eval_timing_residuals.py --example session_s03_m_33_20250306T110000Z_00000001
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from biomech.io.inventory import TakeInventory, scan, subject_pids
from biomech.io.session import SessionData
from biomech.paths import resolve_data_dir, results_dir
from biomech.results.provenance import write_json, write_provenance

STUDY = "5_3_2_timing_residuals"


def rolling_residuals(sensor_ns: np.ndarray, t_master: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from jtsmm.sync.clock_align import AffineFit, ClockAligner

    aligner = ClockAligner(window=30)
    fit = AffineFit()
    n = min(len(sensor_ns), len(t_master))
    residual = np.full(n, np.nan, dtype=np.float64)
    sigma = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        tm = float(t_master[i])
        ts = float(sensor_ns[i])
        if tm <= 0.0 or ts <= 0.0:
            continue
        t_local = ts / 1000.0
        aligner._update_fit(fit, t_local, tm)
        if int(fit.n_samples) < 8:
            continue
        residual[i] = tm - float(fit.map(t_local))
        sigma[i] = float(fit.residual_rms_us)
    return residual, sigma


def load_cam_times(h5: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import h5py

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with h5py.File(h5, "r") as f:
        if "cameras" not in f:
            return out
        for cid, g in f["cameras"].items():
            if "t_master_us" not in g or "sensor_timestamp_ns" not in g:
                continue
            t = np.asarray(g["t_master_us"], dtype=np.float64)
            s = np.asarray(g["sensor_timestamp_ns"], dtype=np.float64)
            out[str(cid)] = (s, t)
    return out


def named_phase_mask(t_master: np.ndarray, phases: list[Any]) -> np.ndarray:
    """True on samples whose ``t_master`` falls in any named phase interval."""
    t_master = np.asarray(t_master, dtype=np.float64)
    m = np.zeros(len(t_master), dtype=bool)
    for p in phases:
        t0 = getattr(p, "t_start_us", None)
        t1 = getattr(p, "t_end_us", None)
        if t0 is None or t1 is None:
            continue
        m |= (t_master >= int(t0)) & (t_master <= int(t1))
    return m


def percentile(x: np.ndarray, q: float) -> float:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def _cam_stats(r: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(r)
    abs_r = np.abs(r[ok])
    return {
        "n": int(ok.sum()),
        "rms_us": float(np.sqrt(np.mean(r[ok] ** 2))) if ok.any() else None,
        "p50_abs_us": percentile(abs_r, 50),
        "p95_abs_us": percentile(abs_r, 95),
        "p99_abs_us": percentile(abs_r, 99),
    }


def session_residuals(rec: TakeInventory) -> dict[str, Any]:
    packed = load_cam_times(rec.h5)
    try:
        with SessionData(rec.path) as sd:
            phases = list(sd.phases)
    except Exception:  # noqa: BLE001 - a take without phases still has residuals
        phases = []
    cams: dict[str, Any] = {}
    abs_all: list[np.ndarray] = []
    abs_phase: list[np.ndarray] = []
    for cid, (sensor, master) in packed.items():
        r, sig = rolling_residuals(sensor, master)
        in_phase = named_phase_mask(master, phases)
        full = _cam_stats(r)
        ph = _cam_stats(np.where(in_phase, r, np.nan))
        abs_all.append(np.abs(r[np.isfinite(r)]))
        r_ph = r[in_phase & np.isfinite(r)]
        if r_ph.size:
            abs_phase.append(np.abs(r_ph))
        cams[cid] = {
            **full,
            "n_phases": ph["n"],
            "rms_phases_us": ph["rms_us"],
            "p50_abs_phases_us": ph["p50_abs_us"],
            "p95_abs_phases_us": ph["p95_abs_us"],
            "p99_abs_phases_us": ph["p99_abs_us"],
            "t_master_us": master,
            "residual_us": r,
            "sigma_c_us": sig,
            "in_phase": in_phase,
        }
    return {
        "session": rec,
        "cams": cams,
        "abs_all": abs_all,
        "abs_phase": abs_phase,
        "phases": phases,
    }


def pool_stats(chunks: list[np.ndarray]) -> dict[str, float | None]:
    if not chunks:
        return {"n": 0, "rms_us": None, "p50_abs_us": None, "p95_abs_us": None, "p99_abs_us": None}
    v = np.concatenate(chunks) if chunks else np.array([])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "rms_us": None, "p50_abs_us": None, "p95_abs_us": None, "p99_abs_us": None}
    return {
        "n": int(v.size),
        "rms_us": float(np.sqrt(np.mean(v**2))),
        "p50_abs_us": percentile(v, 50),
        "p95_abs_us": percentile(v, 95),
        "p99_abs_us": percentile(v, 99),
    }


def pick_example(results: list[dict[str, Any]], name: str | None) -> dict[str, Any] | None:
    if name:
        for r in results:
            if r["session"].name == name or name in r["session"].name:
                return r
    best = None
    best_n = -1
    for r in results:
        n = sum(c["n"] for c in r["cams"].values())
        if n > best_n:
            best, best_n = r, n
    return best


def plot_hist(abs_all: np.ndarray, out: Path, stats: dict[str, Any], *, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    v = abs_all[np.isfinite(abs_all)]
    if v.size:
        ax[0].hist(v, bins=80, color="#1a5276", log=True)
        ax[0].set_xlabel("|r_f| (µs)")
        ax[0].set_ylabel("count")
        ax[0].set_title(title)
        s = np.sort(v)
        ecdf = np.arange(1, s.size + 1) / s.size
        ax[1].plot(s, ecdf, color="#1a5276")
        ax[1].set_xlabel("|r_f| (µs)")
        ax[1].set_ylabel("ECDF")
        ax[1].set_xscale("log")
        rms = stats.get("rms_us")
        p95 = stats.get("p95_abs_us")
        ax[1].set_title(
            f"n={stats['n']}  RMS={rms:.2f} µs  p95={p95:.2f} µs"
            if rms is not None and p95 is not None
            else f"n={stats['n']}"
        )
    fig.savefig(out, dpi=140)
    plt.close(fig)


def _span_phases(ax, phases: list[Any], t0: float) -> None:
    for p in phases:
        t0p = (float(p.t_start_us) - t0) / 1e6
        t1p = (float(p.t_end_us) - t0) / 1e6
        ax.axvspan(t0p, t1p, color="#1a5276", alpha=0.07, zorder=0)


def plot_example(result: dict[str, Any], out: Path, *, phases_only: bool = False) -> None:
    import matplotlib.pyplot as plt

    cams = result["cams"]
    fig, axes = plt.subplots(len(cams), 1, figsize=(10, 3.2 * max(len(cams), 1)), sharex=True, constrained_layout=True)
    if len(cams) == 1:
        axes = [axes]
    rec: TakeInventory = result["session"]
    phases = result.get("phases") or []
    scope = "named phases" if phases_only else "full take"
    for ax, (cid, block) in zip(axes, cams.items()):
        t = np.asarray(block["t_master_us"], dtype=np.float64)
        t0 = t[t > 0][0] if np.any(t > 0) else 0.0
        ts = (t - t0) / 1e6
        r = np.abs(np.asarray(block["residual_us"], dtype=np.float64))
        sig = np.asarray(block["sigma_c_us"], dtype=np.float64)
        if phases_only:
            m = np.asarray(block.get("in_phase"), dtype=bool)
            if m.shape != r.shape:
                m = named_phase_mask(t, phases)
            r = np.where(m, r, np.nan)
            sig = np.where(m, sig, np.nan)
            _span_phases(ax, phases, t0)
        ax.plot(ts, r, color="#7f8c8d", lw=0.6, label="|r_f|")
        ax.plot(ts, sig, color="#c0392b", lw=1.2, label="σ_c (rolling RMS)")
        ax.set_ylabel("µs")
        ax.set_title(f"{rec.name}  {cid}  ({scope})")
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("t_master (s)")
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
    ap.add_argument("--example", type=str, default=None, help="session folder name to plot")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    recs = [r for r in scan(data_dir, pids=subject_pids(args.subject), kind="motion")
            if r.has_cameras]

    results = [session_residuals(r) for r in recs]
    abs_chunks = [c for r in results for c in r["abs_all"]]
    abs_phase_chunks = [c for r in results for c in r.get("abs_phase") or []]
    stats = pool_stats(abs_chunks)
    stats_phases = pool_stats(abs_phase_chunks)

    out = results_dir(STUDY, args.out_dir).resolve()
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    per_session = []
    for r in results:
        rec: TakeInventory = r["session"]
        for cid, block in r["cams"].items():
            np.savez_compressed(
                derived / f"residuals_{rec.name}_{cid}.npz",
                t_master_us=block["t_master_us"],
                residual_us=block["residual_us"],
                sigma_c_us=block["sigma_c_us"],
                in_phase=block["in_phase"],
                cam_id=np.array(cid),
            )
            per_session.append(
                {
                    "session": rec.name,
                    "subject_id": rec.subject_id,
                    "cam": cid,
                    "n": block["n"],
                    "rms_us": block["rms_us"],
                    "p50_abs_us": block["p50_abs_us"],
                    "p95_abs_us": block["p95_abs_us"],
                    "p99_abs_us": block["p99_abs_us"],
                    "n_phases": block["n_phases"],
                    "rms_phases_us": block["rms_phases_us"],
                    "p50_abs_phases_us": block["p50_abs_phases_us"],
                    "p95_abs_phases_us": block["p95_abs_phases_us"],
                    "p99_abs_phases_us": block["p99_abs_phases_us"],
                }
            )
    write_json(
        out / "residual_summary.json",
        {"pool": stats, "pool_phases": stats_phases, "sessions": per_session},
    )
    write_provenance(derived, STUDY, "eval_timing_residuals.py", recs, data_dir=data_dir,
                     extra={"pool": stats, "pool_phases": stats_phases, "subject": args.subject})

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        if abs_chunks:
            plot_hist(
                np.concatenate(abs_chunks),
                out / "residual_hist.png",
                stats,
                title="Timing residual magnitude (full take)",
            )
        if abs_phase_chunks:
            plot_hist(
                np.concatenate(abs_phase_chunks),
                out / "residual_hist_phases.png",
                stats_phases,
                title="Timing residual magnitude (named phases)",
            )
        ex = pick_example(results, args.example)
        if ex is not None:
            plot_example(ex, out / "sigma_c_example.png", phases_only=False)
            plot_example(ex, out / "sigma_c_example_phases.png", phases_only=True)
    print(
        f"sessions={len(recs)} pool_n={stats['n']} rms_us={stats['rms_us']}  "
        f"phases_n={stats_phases['n']} rms_phases_us={stats_phases['rms_us']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
