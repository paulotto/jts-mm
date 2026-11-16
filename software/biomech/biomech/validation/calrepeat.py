# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Repeatability of the two tool calibrations (VD27, J14; ported from and checked
against the earlier evaluations 5.2.1 / 5.2.2 of the recording system, an14
§11f): ten consecutive pivot calibrations of the calibration tool (CT) and ten consecutive
hand-eye calibrations of both IMU mountings, each run refitted here with the project's code
(not read from the run's stored result, which is only compared).

* **Pivot** (:func:`pivot_fit`): least squares of ``R_i d + t_i = c`` over the run's poses
  (tip ``d`` in the CT body frame, pivot point ``c`` in the world); the residual of a pose is
  ``|R_i d + t_i - c|``. Repeatability: per-axis SD of ``d`` over the runs, the 3-D SD and the
  largest deviation from the mean tip; per pose, the tilt from the run's mean orientation
  against the residual (Spearman), since a lever arm of about 230 mm amplifies marker noise
  at oblique views.
* **IMU mounting**: ``jtsmm.imu.mounting.fit_session_slot`` (Park-Martin AX = XB, the
  recorder's own fit) on each run's ``session.h5``; deviations of the runs from their mean
  rotation (chordal mean) as rotation vectors in the IMU frame: per-axis SD, RMS angle,
  largest angle, and the Park-Martin residual per run.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULTS: dict[str, Any] = {
    "reference": "reference.yaml",
    "pivot_dir": "calibration_repeatability/ct_recordings",
    "imu_dir": "calibration_repeatability/recordings_imu",
    "slots": ["imu_mta", "imu_cra"],
}

_RUN = re.compile(r"_(\d+)_\d{8}T\d{6}Z$")


def _run_no(name: str) -> int:
    m = _RUN.search(name)
    return int(m.group(1)) if m else -1


def pivot_fit(T: np.ndarray) -> dict[str, Any]:
    """OLS pivot of poses ``T`` (N, 4, 4) world <- body."""
    R, t = T[:, :3, :3], T[:, :3, 3]
    n = len(T)
    A = np.zeros((3 * n, 6))
    b = np.zeros(3 * n)
    for i in range(n):
        A[3 * i:3 * i + 3, :3] = R[i]
        A[3 * i:3 * i + 3, 3:] = -np.eye(3)
        b[3 * i:3 * i + 3] = -t[i]
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    d, c = x[:3], x[3:]
    res = np.linalg.norm(np.einsum("nij,j->ni", R, d) + t - c, axis=1)
    return {"tip": d, "pivot": c, "resid": res, "rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "cond": float(np.linalg.cond(A))}


def _mean_rotation(Rs: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(Rs.sum(axis=0))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def _rotvec_deg(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return np.degrees(Rotation.from_matrix(R).as_rotvec())


def pivot_runs(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per run, per pose, and the repeatability summary."""
    from scipy.stats import spearmanr

    runs, poses = [], []
    for leaf in sorted((q for q in folder.iterdir() if (q / "poses.json").is_file()),
                       key=lambda q: _run_no(q.name)):
        doc = json.loads((leaf / "poses.json").read_text())
        T = np.asarray(doc["poses_world_body"], float).reshape(-1, 4, 4)
        f = pivot_fit(T)
        stored = doc.get("ols", {})
        Rm = _mean_rotation(T[:, :3, :3])
        tilt = np.array([np.degrees(np.arccos(np.clip((np.trace(Rm.T @ Ri) - 1) / 2, -1, 1)))
                         for Ri in T[:, :3, :3]])
        runs.append({"run": _run_no(leaf.name), "leaf": leaf.name, "n_poses": len(T),
                     "tip_x_mm": f["tip"][0], "tip_y_mm": f["tip"][1], "tip_z_mm": f["tip"][2],
                     "tip_length_mm": float(np.linalg.norm(f["tip"])), "rms_mm": f["rms_mm"],
                     "stored_rms_mm": float(stored.get("rms_mm", np.nan)),
                     "stored_tip_diff_mm": float(np.linalg.norm(
                         f["tip"] - np.asarray(stored.get("tip_body_mm", [np.nan] * 3), float))),
                     "cond": f["cond"]})
        poses += [{"run": _run_no(leaf.name), "pose": i, "resid_mm": float(r), "tilt_deg": float(a)}
                  for i, (r, a) in enumerate(zip(f["resid"], tilt))]
    rd, pdf = pd.DataFrame(runs), pd.DataFrame(poses)
    if rd.empty:
        return rd, pdf, pd.DataFrame()
    tips = rd[["tip_x_mm", "tip_y_mm", "tip_z_mm"]].to_numpy()
    dev = np.linalg.norm(tips - tips.mean(axis=0), axis=1)
    sd = tips.std(axis=0, ddof=1)
    rho, pval = spearmanr(pdf["tilt_deg"], pdf["resid_mm"])
    summ = pd.DataFrame([{"tool": "ct_pivot", "n_runs": len(rd),
                          "sd_x_mm": sd[0], "sd_y_mm": sd[1], "sd_z_mm": sd[2],
                          "sd_3d_mm": float(np.sqrt((sd ** 2).sum())),
                          "dev_max_mm": float(dev.max()),
                          "rms_min_mm": float(rd["rms_mm"].min()),
                          "rms_max_mm": float(rd["rms_mm"].max()),
                          "rms_median_mm": float(rd["rms_mm"].median()),
                          "tip_length_mean_mm": float(rd["tip_length_mm"].mean()),
                          "tilt_resid_spearman": float(rho), "tilt_resid_p": float(pval),
                          "n_poses": int(len(pdf)),
                          "stored_rms_maxdiff_mm": float((rd["rms_mm"] - rd["stored_rms_mm"])
                                                         .abs().max())}])
    return rd, pdf, summ


def imu_runs(folder: Path, slots: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    from jtsmm.imu.mounting import MountingFitError, fit_session_slot

    rows, Rs = [], {s: [] for s in slots}
    leaves = sorted((q for q in folder.iterdir() if (q / "session.h5").is_file()),
                    key=lambda q: _run_no(q.name))
    for leaf in leaves:
        stored = {}
        js = leaf / "imu_mounting.json"
        if js.is_file():
            stored = json.loads(js.read_text()).get("slots", {})
        for s in slots:
            try:
                r = fit_session_slot(leaf / "session.h5", s, strict=False)
            except MountingFitError as exc:
                rows.append({"run": _run_no(leaf.name), "slot": s, "note": str(exc)[:200]})
                continue
            Rst = np.asarray(stored.get(s, {}).get("R_mount", np.full((3, 3), np.nan)), float)
            diff = (np.degrees(np.arccos(np.clip((np.trace(Rst.T @ r.R_mount) - 1) / 2, -1, 1)))
                    if np.isfinite(Rst).all() else np.nan)
            rows.append({"run": _run_no(leaf.name), "slot": s, "rms_deg": float(r.rms_deg),
                         "n_pairs": int(r.n_pairs), "accepted": bool(r.accepted),
                         "stored_rms_deg": float(stored.get(s, {}).get("rms_deg", np.nan)),
                         "stored_diff_deg": float(diff)})
            Rs[s].append((_run_no(leaf.name), r.R_mount))
    rd = pd.DataFrame(rows)
    summ = []
    for s, lst in Rs.items():
        if len(lst) < 2:
            continue
        R = np.stack([x[1] for x in lst])
        Rm = _mean_rotation(R)
        dv = np.stack([_rotvec_deg(Rm.T @ Ri) for Ri in R])
        ang = np.linalg.norm(dv, axis=1)
        sd = dv.std(axis=0, ddof=1)
        for (run, _), v in zip(lst, dv):
            rd.loc[(rd["run"] == run) & (rd["slot"] == s), ["dev_x_deg", "dev_y_deg", "dev_z_deg"]] = v
        sub = rd[rd["slot"] == s]
        summ.append({"tool": s, "n_runs": len(lst), "sd_x_deg": sd[0], "sd_y_deg": sd[1],
                     "sd_z_deg": sd[2], "rms_dev_deg": float(np.sqrt(np.mean(ang ** 2))),
                     "dev_max_deg": float(ang.max()),
                     "pm_rms_min_deg": float(sub["rms_deg"].min()),
                     "pm_rms_max_deg": float(sub["rms_deg"].max()),
                     "pm_rms_mean_deg": float(sub["rms_deg"].mean()),
                     "stored_diff_max_deg": float(sub["stored_diff_deg"].max())})
    return rd, pd.DataFrame(summ)


def run_ref_cal_repeat(data_dir: Path, p: dict[str, Any]) -> dict[str, pd.DataFrame]:
    p = {**DEFAULTS, **(p or {})}
    out: dict[str, pd.DataFrame] = {}
    notes = []
    pdir = Path(data_dir) / p["pivot_dir"]
    if pdir.is_dir():
        out["pivot_runs"], out["pivot_poses"], ps = pivot_runs(pdir)
    else:
        ps = pd.DataFrame()
        notes.append({"note": f"{pdir} not found: pivot repeatability skipped"})
    idir = Path(data_dir) / p["imu_dir"]
    if idir.is_dir():
        out["imu_runs"], isumm = imu_runs(idir, list(p["slots"]))
    else:
        isumm = pd.DataFrame()
        notes.append({"note": f"{idir} not found: IMU mounting repeatability skipped"})
    out["summary"] = pd.concat([ps, isumm], ignore_index=True)
    out["notes"] = pd.DataFrame(notes)
    return out
