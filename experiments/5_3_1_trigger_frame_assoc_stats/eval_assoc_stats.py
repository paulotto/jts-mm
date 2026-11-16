#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Trigger–frame association statistics (cadence / slip / hold).

Reads live ClockAligner counters from ``session.h5:/sync.attrs.report_json``
for every real session. Per-frame labels are not stored; shares are defined
in the README. Cadence is the association: the cursor advances by the number
of trigger periods in the camera's own SensorTimestamps. Predict-then-match
is only the fallback when that interval is not a plausible drop burst.

Usage
-----
  python eval_assoc_stats.py                       # $JTSMM_DATA_DIR or the repo pilot data
  python eval_assoc_stats.py --data-dir ~/jtsmm_data --subject s03 --no-plot
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from biomech.io.inventory import TakeInventory, scan, subject_pids
from biomech.io.session import SessionData
from biomech.paths import resolve_data_dir, results_dir
from biomech.results.provenance import write_csv, write_json, write_provenance

STUDY = "5_3_1_trigger_frame_assoc_stats"
_STAMP_RE = re.compile(r"(\d{8}T\d{6}Z)")
COUNTER_KEYS = (
    "drops",
    "slip_count",
    "camera_drop_recovered",
    "waiting_for_trigger",
    "validation_fail_count",
    "validation_relock_count",
    "anomaly_count",
    "cadence_advance",
    "cadence_missing_trigger",
)


def _imap(report: dict[str, Any], key: str, cam: str) -> int:
    raw = report.get(key) or {}
    if not isinstance(raw, dict):
        return 0
    try:
        return int(raw.get(cam, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _isum(report: dict[str, Any], key: str) -> int:
    raw = report.get(key)
    if isinstance(raw, dict):
        return sum(_imap(report, key, k) for k in raw)
    try:
        return int(raw or 0)
    except (TypeError, ValueError):
        return 0


def n_written(h5: Path, cam: str) -> tuple[int, int]:
    import h5py

    with h5py.File(h5, "r") as f:
        if "cameras" not in f or cam not in f["cameras"]:
            return 0, 0
        g = f["cameras"][cam]
        if "t_master_us" not in g:
            return 0, 0
        t = np.asarray(g["t_master_us"])
        n = int(t.shape[0])
        if np.issubdtype(t.dtype, np.floating):
            n_bad = int((~np.isfinite(t) | (t <= 0)).sum())
        else:
            n_bad = int((t == 0).sum())
        return n, n_bad


def take_order_key(rec: TakeInventory) -> str:
    m = _STAMP_RE.search(rec.name)
    return m.group(1) if m else rec.name


def assign_take_index(recs: list[TakeInventory]) -> dict[str, int]:
    """1-based take index per subject, chronological by UTC stamp in the folder name."""
    by_subj: dict[str, list[TakeInventory]] = defaultdict(list)
    for rec in recs:
        by_subj[rec.subject_id or rec.name].append(rec)
    out: dict[str, int] = {}
    for _sid, group in by_subj.items():
        ordered = sorted(group, key=take_order_key)
        for i, rec in enumerate(ordered, start=1):
            out[rec.name] = i
    return out


def optical_nan_frac(h5: Path) -> float | None:
    import h5py

    with h5py.File(h5, "r") as f:
        if "optical" not in f or "t_master_us" not in f["optical"]:
            return None
        t = np.asarray(f["optical"]["t_master_us"])
        if t.size == 0:
            return None
        if np.issubdtype(t.dtype, np.floating):
            return float((~np.isfinite(t)).mean())
        return float((t == 0).mean())


def eval_session(rec: TakeInventory, take_index: int | None = None) -> list[dict[str, Any]]:
    with SessionData(rec.path) as sd:
        report = sd.sync_report
    cams = sorted(rec.n_cam_frames.keys()) or sorted(
        set().union(
            *[
                set((report.get(k) or {}).keys())
                for k in COUNTER_KEYS
                if isinstance(report.get(k), dict)
            ]
        )
    )
    leftover = _isum(report, "cam_hold_leftover")
    overflow = _isum(report, "cam_hold_overflow")
    opt_nan = optical_nan_frac(rec.h5)
    rows = []
    for cam in cams or ["(none)"]:
        written, n_zero = n_written(rec.h5, cam) if cam != "(none)" else (0, 0)
        drops = _imap(report, "drops", cam)
        slip = _imap(report, "slip_count", cam)
        recovered = _imap(report, "camera_drop_recovered", cam)
        hold = _imap(report, "waiting_for_trigger", cam)
        reject = _imap(report, "validation_fail_count", cam)
        relock = _imap(report, "validation_relock_count", cam)
        anomaly = _imap(report, "anomaly_count", cam)
        cadence = _imap(report, "cadence_advance", cam)
        missing_trig = _imap(report, "cadence_missing_trigger", cam)
        # Both paths add to camera_drop_recovered. Cadence is counted on its
        # own; whatever remains is the fit-based predict-then-match fallback.
        predict = max(0, recovered - cadence)
        attempts = written + drops + leftover
        cadence_share = cadence / written if written else float("nan")
        missing_share = missing_trig / written if written else float("nan")
        predict_share = predict / written if written else float("nan")
        slip_share = slip / written if written else float("nan")
        # Lower bound: written frames not explained by a cadence skip, the
        # predict fallback, or a slip.
        nominal = 1.0
        for part in (cadence_share, predict_share, slip_share):
            if np.isfinite(part):
                nominal -= part
        nominal = float(np.clip(nominal, 0.0, 1.0)) if written else float("nan")
        hold_share = hold / attempts if attempts else float("nan")
        unstamped = drops / (written + drops) if (written + drops) else float("nan")
        zero_share = n_zero / written if written else float("nan")
        rows.append(
            {
                "session": rec.name,
                "subject_id": rec.subject_id,
                "take_index": take_index,
                "take_kind": rec.take_kind,
                "config_hash": rec.config_hash,
                "cam": cam,
                "n_written": written,
                "n_zero_stamp": n_zero,
                "drops": drops,
                "slip_count": slip,
                "camera_drop_recovered": recovered,
                "predict_then_match": predict,
                "waiting_for_trigger": hold,
                "validation_fail_count": reject,
                "validation_relock_count": relock,
                "anomaly_count": anomaly,
                "cadence_advance": cadence,
                "cadence_missing_trigger": missing_trig,
                "cam_hold_overflow": overflow,
                "cam_hold_leftover": leftover,
                "n_ingest_attempts": attempts,
                "share_nominal": nominal,
                "share_cadence": cadence_share,
                "share_missing_trigger": missing_share,
                "share_predict": predict_share,
                "share_slip": slip_share,
                "share_hold": hold_share,
                "share_unstamped": unstamped,
                "share_zero_stamp": zero_share,
                "optical_nan_frac": opt_nan,
                "has_sync_report": rec.has_sync,
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n_rows": 0}
    keys = (
        "share_nominal",
        "share_cadence",
        "share_missing_trigger",
        "share_predict",
        "share_slip",
        "share_hold",
        "share_unstamped",
        "validation_fail_count",
        "validation_relock_count",
    )
    out: dict[str, Any] = {"n_rows": len(rows), "n_sessions": len({r["session"] for r in rows})}
    for k in keys:
        v = np.asarray([r[k] for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = {
            "mean": float(v.mean()) if v.size else None,
            "median": float(np.median(v)) if v.size else None,
            "p95": float(np.percentile(v, 95)) if v.size else None,
            "sum": float(v.sum()) if v.size else None,
        }
    return out


def plot_label(row: dict[str, Any]) -> str:
    sid = row.get("subject_id") or "unknown"
    idx = row.get("take_index")
    take = f"t{idx}" if idx else "?"
    cam = row.get("cam") or ""
    return f"{sid} {take}\n{cam}"


def plot_rows(rows: list[dict[str, Any]], out: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [plot_label(r) for r in rows]
    nom = np.array([r["share_nominal"] or 0.0 for r in rows], dtype=float)
    cad = np.array([r["share_cadence"] or 0.0 for r in rows], dtype=float)
    pred = np.array([r["share_predict"] or 0.0 for r in rows], dtype=float)
    slip = np.array([r["share_slip"] or 0.0 for r in rows], dtype=float)
    x = np.arange(len(rows))
    fig, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(rows)), 8), constrained_layout=True)
    ax = axes[0]
    ax.bar(x, nom, label="nominal (lb)", color="#1a5276")
    ax.bar(x, cad, bottom=nom, label="cadence", color="#1e8449")
    ax.bar(x, pred, bottom=nom + cad, label="predict-then-match", color="#f4d03f")
    ax.bar(x, slip, bottom=nom + cad + pred, label="slip", color="#c0392b")
    ax.set_ylabel("share of written frames")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Association mode shares")
    ax2 = axes[1]
    ax2.bar(x - 0.25, [r["cadence_missing_trigger"] for r in rows], 0.25, label="missing trigger")
    ax2.bar(x, [r["validation_fail_count"] for r in rows], 0.25, label="validation rejects")
    ax2.bar(x + 0.25, [r["validation_relock_count"] for r in rows], 0.25, label="fit relocks")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=90, fontsize=7)
    ax2.set_ylabel("count")
    ax2.legend()
    ax2.set_title("Rejects and affine fit resets")
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
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    # Association study: real motion takes with cameras and a sync snapshot.
    recs = [r for r in scan(data_dir, pids=subject_pids(args.subject), kind="motion")
            if r.has_cameras and r.has_sync]

    take_idx = assign_take_index(recs)
    rows: list[dict[str, Any]] = []
    for rec in recs:
        rows.extend(eval_session(rec, take_idx.get(rec.name)))

    out = results_dir(STUDY, args.out_dir).resolve()
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    write_csv(out / "assoc_sessions.csv", rows)
    summary = summarize(rows)
    write_json(out / "assoc_summary.json", summary)
    write_json(derived / "assoc_sessions.json", rows)
    write_provenance(derived, STUDY, "eval_assoc_stats.py", recs, data_dir=data_dir,
                     extra=summary)
    if not args.no_plot and rows:
        import matplotlib

        matplotlib.use("Agg")
        plot_rows(rows, out / "assoc_shares.png")
    print(f"sessions={len(recs)} rows={len(rows)}")
    print(f"wrote {out / 'assoc_sessions.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
