#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""CT pivot calibration repeatability eval.

Loads independent CT tip calibration leaves under ``recordings/`` and reports
tip-offset scatter plus OLS residual metrics.

Modes
-----
* default — use stored ``ct_tip_calibration.json`` (live OLS result)
* ``--from-poses`` — re-solve OLS from ``poses.json`` (checks tip JSON integrity)
* ``--from-frames`` — re-track ``frames/`` with the leaf-local
  ``calibration_bundle.h5`` / ``optical.yaml`` snapshots, then OLS

Usage
-----
  python eval_repeatability.py
  python eval_repeatability.py --from-poses
  python eval_repeatability.py --from-frames --optical-config ~/optical.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from analyze_repeatability import (
    DEFAULT_RECORDINGS,
    TIP_JSON,
    Run,
    analyze,
    discover_runs,
    format_report,
    load_run,
    try_plot,
    write_csv,
)

SCRIPT_DIR = Path(__file__).resolve().parent
HOME_STUDY = Path.home() / "experiments" / SCRIPT_DIR.name

CALIB_BUNDLE = "calibration_bundle.h5"
POSES_JSON = "poses.json"


def _default_optical_config() -> Path:
    """``~/optical.yaml`` when present, else the package default (``optical.paths``)."""
    from optical.config.loader import default_config_path

    return Path(default_config_path())


def _load_pose_matrices(leaf: Path) -> list[np.ndarray]:
    doc = json.loads((leaf / POSES_JSON).read_text(encoding="utf-8"))
    out: list[np.ndarray] = []
    for raw in doc.get("poses_world_body") or []:
        T = np.asarray(raw, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"{leaf}: bad pose shape {T.shape}")
        out.append(T)
    return out


def _run_from_ols(leaf: Path, result: Any, *, label: str | None = None) -> Run:
    tip_arr = np.asarray(result.tip_body_mm, dtype=np.float64).reshape(3)
    pivot_arr = np.asarray(result.pivot_world_mm, dtype=np.float64).reshape(3)
    tip = (float(tip_arr[0]), float(tip_arr[1]), float(tip_arr[2]))
    pivot = (float(pivot_arr[0]), float(pivot_arr[1]), float(pivot_arr[2]))
    tip_len = float(getattr(result, "tip_length_mm", float(np.linalg.norm(tip_arr))))
    return Run(
        path=leaf / TIP_JSON,
        label=label or leaf.name,
        tip_body_mm=tip,
        tip_length_mm=tip_len,
        pivot_world_mm=pivot,
        rms_mm=float(result.rms_mm),
        median_err_mm=float(result.median_err_mm),
        max_err_mm=float(result.max_err_mm),
        residual_norm=float(result.residual_norm),
        n_poses=int(result.n_poses),
        cond_A=float(result.cond_A),
    )


def recompute_from_poses(leaf: Path) -> Run:
    from optical.tracking.pivot import solve_pivot_ols

    poses = _load_pose_matrices(leaf)
    if len(poses) < 3:
        raise ValueError(f"{leaf}: need ≥3 poses, got {len(poses)}")
    return _run_from_ols(leaf, solve_pivot_ols(poses), label=f"{leaf.name}#poses")


def _iter_sample_frames(leaf: Path) -> list[dict[str, np.ndarray]]:
    import cv2

    frames_root = leaf / "frames"
    if not frames_root.is_dir():
        return []
    samples: list[dict[str, np.ndarray]] = []
    for sample_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        bundle: dict[str, np.ndarray] = {}
        for jpg in sorted(sample_dir.glob("*.jpg")):
            raw = cv2.imread(str(jpg), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            if raw.ndim == 3:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
            bundle[jpg.stem] = raw
        if bundle:
            samples.append(bundle)
    return samples


def recompute_from_frames(leaf: Path, optical_config: Path) -> Run:
    """Re-track SPACE samples with the leaf-local calib snapshot, then OLS."""
    from optical.paths import recording_calibration_bundle
    from optical.tracking.pivot import solve_pivot_ols
    from optical.tracking.pose import is_valid_pose
    from optical.tracking.tracker import load_tracker_for_recording

    if recording_calibration_bundle(leaf) is None:
        raise FileNotFoundError(
            f"{leaf}: missing {CALIB_BUNDLE} — re-record with --persist-ct-calibration"
        )
    samples = _iter_sample_frames(leaf)
    if len(samples) < 3:
        raise ValueError(f"{leaf}: need ≥3 frame samples, got {len(samples)}")

    tracker, _tcfg = load_tracker_for_recording(optical_config, leaf)
    poses: list[np.ndarray] = []
    for i, frames in enumerate(samples):
        record = tracker.process_frame(frames, frame_idx=i)
        if is_valid_pose(record.pose_CT):
            poses.append(np.asarray(record.pose_CT, dtype=np.float64).copy())
    if len(poses) < 3:
        raise ValueError(f"{leaf}: only {len(poses)} valid CT poses after re-track")
    return _run_from_ols(leaf, solve_pivot_ols(poses), label=f"{leaf.name}#frames")


def _leaf_dirs(tip_paths: Sequence[Path]) -> list[Path]:
    return [p.parent for p in tip_paths]


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--from-poses",
        action="store_true",
        help="Re-solve OLS from poses.json instead of trusting tip JSON",
    )
    mode.add_argument(
        "--from-frames",
        action="store_true",
        help="Re-track frames/ with leaf calibration_bundle.h5, then OLS",
    )
    parser.add_argument(
        "--optical-config",
        type=Path,
        default=None,
        help="Fallback optical.yaml when a leaf has no snapshot (default: ~/optical.yaml)",
    )
    args = parser.parse_args(argv)

    out_dir = (args.out or (HOME_STUDY / "results")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    recordings = args.input.expanduser()
    tip_paths = discover_runs(recordings)
    if not tip_paths:
        print(f"error: no {TIP_JSON} under {recordings}", file=sys.stderr)
        return 1

    leaves = _leaf_dirs(tip_paths)
    optical_cfg = (args.optical_config or _default_optical_config()).expanduser()
    runs: list[Run] = []
    errors: list[str] = []

    for leaf in leaves:
        try:
            if args.from_frames:
                runs.append(recompute_from_frames(leaf, optical_cfg))
            elif args.from_poses:
                runs.append(recompute_from_poses(leaf))
            else:
                runs.append(load_run(leaf / TIP_JSON))
        except Exception as exc:  # noqa: BLE001 — report and continue
            errors.append(f"{leaf.name}: {exc}")

    if not runs:
        for err in errors:
            print(f"error: {err}", file=sys.stderr)
        return 1

    summary = analyze(runs)
    mode_name = (
        "from_frames" if args.from_frames else "from_poses" if args.from_poses else "stored_tip"
    )
    summary["meta"] = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "input": str(recordings),
        "n_files": len(tip_paths),
        "n_evaluated": len(runs),
        "mode": mode_name,
        "errors": errors,
    }

    # Snapshot coverage (for reproducibility notes).
    n_with_bundle = sum(1 for leaf in leaves if (leaf / CALIB_BUNDLE).is_file())
    summary["meta"]["n_leaves_with_calibration_bundle"] = n_with_bundle

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
    if n_with_bundle < len(leaves):
        print(
            f"note: {n_with_bundle}/{len(leaves)} leaves have {CALIB_BUNDLE}; "
            "re-record with --persist-ct-calibration for full offline reproducibility",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
