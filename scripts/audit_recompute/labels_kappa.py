#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Auto-label validity check on a frozen snapshot.

Since 2026-11-01 the check is the pipeline stage ``labels_check`` (rule, variants and
levels: :mod:`biomech.labels.kappa`). This script runs the same code on a snapshot that
predates the stage (e.g. ``r9``, where the audit was first computed) and writes
``data/analysis/icorr2027/audit/labels_kappa_<TAG>.csv`` with pseudonyms only.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/labels_kappa.py --snapshot r9
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

REPO = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    from biomech.config import load_config
    from biomech.io.context import DataContext
    from biomech.labels.kappa import DEFAULTS, run_labels_check

    ap = argparse.ArgumentParser(description="Auto-label kappa check against a second rule")
    ap.add_argument("--snapshot", default=_pins.dataset_record())
    ap.add_argument("--windows", type=int, default=DEFAULTS["windows"])
    ap.add_argument("--window-s", type=float, default=DEFAULTS["window_s"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    args = ap.parse_args(argv)

    cfg = load_config(REPO / "experiments/icorr2027/study.yaml").with_snapshot(args.snapshot)
    ctx = DataContext(cfg)
    data = {}
    for pid in ctx.pids():
        try:
            data[pid] = (ctx.arrays("labels", pid), ctx.arrays("kinematics", pid))
        except Exception:                            # noqa: BLE001 -- no labels for pid
            continue
    df = run_labels_check(data, {"windows": args.windows, "window_s": args.window_s,
                                 "seed": args.seed})["kappa"]
    df["pid"] = df["pid"].map(lambda p: f"P{ctx.pseudonym(p)}" if p else "all")
    out = REPO / "data/analysis/icorr2027/audit" / f"labels_kappa_{args.snapshot}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.rename(columns={"pid": "participant"}).to_csv(out, index=False, float_format="%.6g")
    with pd.option_context("display.width", 220):
        show = df[df.level.isin(["participant", "phase", "overall_sampled",
                                  "overall_all_frames", "boundary_tolerant"])]
        print(show.drop(columns=["agreement", "kappa_all_frames"], errors="ignore")
              .to_string(index=False))
    print(f"-> {out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
