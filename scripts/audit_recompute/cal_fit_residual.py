# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J21: indirect check of the stereo calibration of the takes of days 1-4 through the
live tracker's rigid-fit residual (exploratory; definitions fixed in J21 before this
ran).

* J21.1 -- per take (all 13 of the paper store, pin r16a) and body (cra, mta): the stored
  per-frame residual (``optical/quality/<body>/reprojection_rms``, ``rms_kind = kabsch_mm``)
  over the frames inside the experiment phases, median and P95;
* J21.2 -- the 5 takes with stored corners: the corners re-triangulated with the own and with
  every other stored calibration, the rigid tag model fitted per frame and body (unweighted
  Kabsch, all corners of the body, at least one full tag), residual = RMS distance of the
  corners to the fitted model; per (take, calibration, body) the median over the frames of the
  experiment phases.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/cal_fit_residual.py

Output: ``<adults store>/audit/cal_fit_residual_<snap>.parquet`` (``part`` = stored / cross)
and ``cal_fit_residual_<snap>.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import _pins  # noqa: E402

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
PHASES = ("rest", "open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")
BODIES = {"cra": "HP", "mta": "MP"}          # quality group -> optical.yaml body (stereo_cal)


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def _in_phases(t_us: np.ndarray, phases: list) -> np.ndarray:
    m = np.zeros(t_us.size, bool)
    for ph in phases:
        if ph.name in PHASES:
            m |= (t_us >= ph.t_start_us) & (t_us <= ph.t_end_us)
    return m


def fit_residual(fit, X: np.ndarray) -> np.ndarray:
    """Per frame the RMS distance (mm) of the body's corners to the fitted rigid model."""
    from biomech.qc.stereo_cal import _batch_kabsch

    M = len(fit.tags) * 4
    Q = np.full((fit.F, M, 3), np.nan)
    W = np.zeros((fit.F, M))
    for k in range(4):
        Q[fit.f, fit.c * 4 + k] = X[fit.src, k]
        W[fit.f, fit.c * 4 + k] = 1.0
    ok = W.sum(axis=1) >= 4
    out = np.full(fit.F, np.nan)
    if ok.any():
        R, t = _batch_kabsch(fit.P, Q[ok], W[ok])
        pred = np.einsum("fij,mj->fmi", R, fit.P) + t[:, None]
        d2 = np.where(W[ok] > 0, np.sum((np.nan_to_num(Q[ok]) - pred) ** 2, axis=2), 0.0)
        out[ok] = np.sqrt(d2.sum(axis=1) / W[ok].sum(axis=1))
    return out


def main() -> None:
    from biomech.io.context import DataContext
    from biomech.io.session import SessionData
    from biomech.qc import stereo_cal as sc

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    a = ap.parse_args()
    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    try:
        pids = sorted(dc.table("acquisition_cohort", "per_session").pid.unique())
        cal_tab = dc.table("acquisition_cohort", "calibrations")
        days = dc.table("acquisition_cohort", "per_session").set_index("pid")["day_index"]
        paths = {p: Path(dc.cohort.session_path(dc.cohort.get(p).motion)) for p in pids}
    finally:
        dc.close()
    rows = []
    phases, trow = {}, {}
    for pid in pids:
        sd = SessionData(paths[pid])
        try:
            phases[pid] = list(sd.phases)
        finally:
            sd.close()
        with h5py.File(paths[pid] / "session.h5", "r") as f:
            kind = f["optical/quality"].attrs.get("rms_kind")
            t = np.asarray(f["optical/t_master_us"][()], np.int64)
            trow[pid] = t
            sel = _in_phases(t, phases[pid])
            for body in BODIES:
                r = np.asarray(f[f"optical/quality/{body}/reprojection_rms"][()], float)[sel]
                r = r[np.isfinite(r)]
                rows.append({"part": "stored", "pid": pid, "day_index": int(days[pid]),
                             "body": body, "calibration": "in use", "own": True,
                             "rms_kind": str(kind), "n_frames": int(r.size),
                             "median_mm": float(np.median(r)) if r.size else np.nan,
                             "p95_mm": float(np.percentile(r, 95)) if r.size else np.nan})
    # J21.2 cross test on the takes with stored corners
    labels = {p: lab for lab, ps in zip(cal_tab["label"], cal_tab["participants"])
              for p in ps.split(", ")}
    cals = {lab: sc.session_calibration(paths[ps.split(", ")[0]] / "session.h5")
            for lab, ps in zip(cal_tab["label"], cal_tab["participants"])}
    model, ranges = sc.marker_model()
    for pid, own in labels.items():
        pairs = sc.corner_pairs(paths[pid] / "session.h5")
        frames = np.unique(pairs.row)
        sel = _in_phases(trow[pid][frames], phases[pid])
        fits = {b: sc._BodyFit(pairs, frames, list(range(ranges[n][0], ranges[n][1] + 1)), model)
                for b, n in BODIES.items()}
        for lab, cal in cals.items():
            n0 = sc._undistort(pairs.p0, cal.K0, cal.d0)
            n1 = sc._undistort(pairs.p1, cal.K1, cal.d1)
            X = sc.triangulate(cal, n0, n1).reshape(-1, 4, 3)
            for b, fit in fits.items():
                r = fit_residual(fit, X)[sel]
                r = r[np.isfinite(r)]
                rows.append({"part": "cross", "pid": pid, "day_index": int(days[pid]),
                             "body": b, "calibration": lab, "own": lab == own,
                             "rms_kind": "kabsch_mm", "n_frames": int(r.size),
                             "median_mm": float(np.median(r)) if r.size else np.nan,
                             "p95_mm": float(np.percentile(r, 95)) if r.size else np.nan})
    df = pd.DataFrame(rows)
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"cal_fit_residual_{a.snapshot}.parquet"
    df.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    meta = {"snapshot": a.snapshot, "git_sha": sha, "phases": PHASES,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    (out_dir / f"cal_fit_residual_{a.snapshot}.json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 200)
    print(df.drop(columns=["rms_kind"]).round(3).to_string())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
