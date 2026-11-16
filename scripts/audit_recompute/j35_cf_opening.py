# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J35: incisal opening per cycle without cameras (exploratory, post hoc; definitions
fixed in J35 before this ran).

C3a of the dataset paper converts the peak inertial opening angle of a cycle into an incisal
opening with the generic model (mean coefficients of all other participants) and compares it
with the optical opening, for the in-take fit (E1). This script does the same for the
camera-free estimators of X5: E2a (mountings calibrated once, optical heading) and the primary
resets of the participant (``camerafree:cf_summary.primary_e3``, causal), with E1 as the
reproduction of C3a on the camera-free grid.

Per participant and valid cycle (``kinematics:cycles``, ``ok``) of open-close and chewing: the
samples of the camera-free grid from the cycle start to its end (at least 10), the peak angle
of an estimator as the largest value of its opening angle minus the angle at the first sample
(as the ``imu`` stage), the opening a1*theta + a2*theta^2 with the generic coefficients (as
``stats.cohort_misc.imu_tables``), and the optical ``opening_mm`` of the cycle as reference.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j35_cf_opening.py

Output: ``<adults store>/audit/j35_cf_opening_<snap>.parquet`` (cycle x estimator rows) and
``.json`` (settings, summary per task and estimator, the stored C3a row, sha256). The cohort
statistics of the paper are computed from the cycle rows by the derived table
``cf_opening_cycles``; the summary in the json is for the record of this run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
TASKS = ("open_close", "chewing")
MIN_SAMPLES = 10
ESTIMATORS = ("E1", "E2a", "E3")       # E3: the participant's primary resets (causal)


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def cycle_rows(dc: Any) -> pd.DataFrame:
    """One row per participant, task, cycle, and estimator."""
    cyc = dc.table("kinematics", "cycles")
    cyc = cyc[cyc.task.isin(TASKS) & cyc.ok.astype(bool)]
    models = dc.table("kinematics", "opening_model").drop_duplicates("pid").set_index("pid")
    summ = dc.table("camerafree", "cf_summary").drop_duplicates("pid").set_index("pid")
    rows = []
    for pid in dc.pids():
        a = dc.arrays("camerafree", pid)
        if "t_s" not in a or pid not in summ.index:
            continue
        t_us = np.asarray(a["t_s"], float) * 1e6
        prim = str(summ.at[pid, "primary_e3"])
        src = {"E1": "open_deg__E1", "E2a": "open_deg__E2a", "E3": f"open_deg__{prim}"}
        others = models.drop(index=pid, errors="ignore")
        a1, a2 = float(others.a1_mm_per_deg.mean()), float(others.a2_mm_per_deg2.mean())
        opt = np.asarray(a["open_deg__opt"], float)
        for c in cyc[cyc.pid == pid].itertuples():
            sel = (t_us >= c.t_start_us) & (t_us <= c.t_end_us)
            if sel.sum() < MIN_SAMPLES:
                continue
            o = opt[sel] - opt[sel][0]
            th_opt = float(np.nanmax(o)) if np.isfinite(o).any() else np.nan
            for est, key in src.items():
                if key not in a:
                    continue
                x = np.asarray(a[key], float)[sel]
                x = x - x[0]
                th = float(np.nanmax(x)) if np.isfinite(x).any() else np.nan
                rows.append({"pid": pid, "task": c.task, "cycle": int(c.cycle),
                             "estimator": est, "source": key.split("__")[1],
                             "n_samples": int(sel.sum()), "theta_peak_deg": th,
                             "theta_peak_opt_deg": th_opt, "a1_generic": a1, "a2_generic": a2,
                             "n_others": int(len(others)),
                             "opening_est_mm": a1 * th + a2 * th ** 2,
                             "opening_opt_mm": float(c.opening_mm)})
    return pd.DataFrame(rows)


def summary(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Repeated-measures limits of agreement per task and estimator (record of the run)."""
    from biomech.stats.cohort_misc import max_abs_loa_mover
    from biomech.stats.core import bland_altman_rm

    out = []
    for (task, est), g in rows.groupby(["task", "estimator"]):
        e, r, pid = g.opening_est_mm.to_numpy(), g.opening_opt_mm.to_numpy(), g.pid.to_numpy()
        ba = bland_altman_rm(e, r, pid)
        ok = np.isfinite(e) & np.isfinite(r)
        m, lo, hi = max_abs_loa_mover((e - r)[ok], pid[ok])
        out.append({"task": task, "estimator": est, **ba, "max_abs_loa": m,
                    "max_abs_loa_ci_lo": lo, "max_abs_loa_ci_hi": hi})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    a = ap.parse_args()
    from biomech.io.context import DataContext

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    try:
        rows = cycle_rows(dc)
        ba = dc.table("stats_imu", "imu_opening_ba")
    finally:
        dc.close()
    stored = ba[(ba.task == "open_close") & (ba.estimator == "quat") & (ba.model == "loso")]
    stored_row = {k: float(stored[k].iloc[0]) for k in ("bias", "loa_lo", "loa_hi",
                                                        "max_abs_loa", "max_abs_loa_ci_hi", "n")
                  if k in stored and len(stored)}
    summ = summary(rows)
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"j35_cf_opening_{a.snapshot}.parquet"
    rows.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    meta = {"snapshot": a.snapshot, "git_sha": sha, "variant": VARIANT, "tasks": list(TASKS),
            "min_samples": MIN_SAMPLES, "estimators": list(ESTIMATORS),
            "n_participants": int(rows.pid.nunique()), "summary": summ,
            "stored_c3a_open_close_quat_loso": stored_row,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 200)
    cols = ["task", "estimator", "n", "n_subjects", "bias", "loa_lo", "loa_hi", "max_abs_loa",
            "max_abs_loa_ci_hi"]
    print(pd.DataFrame(summ)[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("stored C3a (open-close, quat, loso):", stored_row)
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
