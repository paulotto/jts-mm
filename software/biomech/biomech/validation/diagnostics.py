# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stored diagnostics of the validation study (VD24; stage ``ref_diagnostics``).

Values the docs quoted from audit scripts or by hand, computed here exactly as the audits did
so that every quoted number is a stage result (and can get a numbers macro):

* **Frame timing** (``frame_timing``, ``frame_gaps``): per take the SD of the JTS-MM
  camera-frame intervals without the long ones, the share of intervals longer than 1.5 times
  the median, and the long intervals by their length in frames -- the computation of the
  methods table (``val_table_methods``), on the same ``ref_kinematics`` frame times.
* **Mounting diagnostics** (VD14; the audit of 2026-10-29, reproduced line
  by line): per leave-one-task-out fold and IMU the in-take calibration and the calibration
  with each hand-eye mounting held fixed (``FU.calibrate`` with a per-body lag, as the audit;
  ``ref_fusion`` pools the lag over the IMUs, so its stored residuals differ in the second
  decimal), their fit residuals (``mounting_folds``, summary ``mounting_residual``), the
  rotation between the hand-eye sessions (``mounting_sessions``), the singular values of the
  relative-rotation vectors of the training frames (``mounting_excitation``: how well the
  in-take mounting is excited), the fold-to-fold spread of the in-take mounting
  (``mounting_spread``) and the dataset (imu-stage) estimator on the take's ``free_motion``
  against each hand-eye mounting (``mounting_imu_stage``).
* **IMU dynamic calibration** (``dyncal_summary``, ``dyncal_periods``,
  ``dyncal_transitions``): the validation part of ``scripts/audit_recompute/imu_dyncal.py``
  -- still periods of both IMUs, steps of the IMU-alone rotation error against Qualisys
  (``ref_fusion`` ``err_rot_I_deg``) and their alignment with stillness, drift inside and
  outside the still periods, the jumps across phase boundaries. Implemented separately from
  the script (which stays the independent re-computation; a test checks that both agree);
  the dataset part of that audit needs the dataset study's camera-free store and stays in
  the script.

Descriptive (n = 2), no test. The CT triangle is stored in ``ref_tip:ct_geometry``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from biomech.imu.dyncal import DEFAULTS as DYNCAL_DEFAULTS
from biomech.imu.dyncal import (  # noqa: F401  (re-exported: the stage and the tests use them)
    angular_speed_dps, block_medians, dyncal_trace, edge_change_deg, imu_grid, ols_slope_dpm,
    phase_at, phase_jumps, step_summary, still_period_rows, still_runs, true_runs)

DEFAULTS: dict[str, Any] = {
    # frame timing (val_table_methods)
    "long_interval_factor": 1.5,     # an interval > 1.5 x median is a dropped-frame gap
    # mounting audit (Part H, 2026-10-29)
    "exposure_s": 0.012,
    "imu_max_gap_s": 0.05,           # IMU track gaps (as the audit)
    "lag_range_s": (-0.05, 0.08),
    "seed": 0,
    "rel_rot_lag_frames": 25,        # frame pairs of the excitation SVD
    "quat_match_s": 0.02,            # imu-stage estimator: nearest quaternion within 20 ms
    "imu_stage_task": "free_motion",
    # dynamic-calibration audit (scripts/audit_recompute/imu_dyncal.py)
    **DYNCAL_DEFAULTS,
}
BODY_SLOT = {"HP": "cra", "MP": "mta"}       # validation body -> IMU slot / optical body


# == frame timing =================================================================================
def frame_timing(pid: str, t_s: np.ndarray, factor: float = 1.5
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Camera-frame intervals of a take (``t_s``: JTS-MM frame times, s), as
    ``val_table_methods``: the median interval, the SD of the intervals that are not longer
    than ``factor`` x median (hardware trigger jitter), the share of longer ones (dropped
    frames) and, per length in frame periods, how many there are."""
    t = np.asarray(t_s, float)
    t = t[np.isfinite(t)]
    dt = np.diff(t) if t.size > 1 else np.zeros(0)
    if not dt.size:
        return (pd.DataFrame([{"pid": pid, "n_intervals": 0}]),
                pd.DataFrame(columns=["pid", "interval_frames", "missing_frames",
                                      "interval_ms", "n", "share_of_intervals"]))
    med = float(np.median(dt))
    long_ = dt > factor * med
    summ = {"pid": pid, "n_intervals": int(dt.size), "frame_interval_ms": med * 1e3,
            "frame_rate_hz": 1.0 / med, "interval_sd_us": float(1e6 * np.std(dt[~long_])),
            "n_long": int(long_.sum()), "long_share": float(long_.mean()),
            "long_max_ms": float(dt[long_].max() * 1e3) if long_.any() else np.nan}
    k = np.round(dt[long_] / med).astype(int)
    gaps = [{"pid": pid, "interval_frames": int(v), "missing_frames": int(v) - 1,
             "interval_ms": float(np.median(dt[long_][k == v]) * 1e3), "n": int((k == v).sum()),
             "share_of_intervals": float((k == v).sum() / dt.size)} for v in np.unique(k)]
    return pd.DataFrame([summ]), pd.DataFrame(gaps, columns=[
        "pid", "interval_frames", "missing_frames", "interval_ms", "n", "share_of_intervals"])


# == mounting diagnostics =========================================================================
def angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two rotations (deg)."""
    return float(np.degrees(np.linalg.norm(Rot.from_matrix(R1.T @ R2).as_rotvec())))


def session_label(path: str | Path) -> str:
    """``YYYY-MM-DD`` of a hand-eye session folder (``..._20250314T110000Z_...``), else the
    folder name."""
    p = Path(path)
    name = p.parent.name if p.suffix else p.name
    m = re.search(r"(\d{4})(\d{2})(\d{2})T\d{6}Z", name)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else name


def session_rotations(pid: str, he: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """Rotation of each IMU mounting between every pair of hand-eye sessions (``he``:
    label -> slot -> R_mount); a remount shows as a large angle."""
    labels = sorted(he)
    rows = []
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            for body, slot in BODY_SLOT.items():
                if slot in he[a] and slot in he[b]:
                    rows.append({"pid": pid, "body": body, "slot": slot, "session_a": a,
                                 "session_b": b,
                                 "mounting_change_deg": angle_deg(he[a][slot], he[b][slot])})
    return pd.DataFrame(rows, columns=["pid", "body", "slot", "session_a", "session_b",
                                       "mounting_change_deg"])


def excitation_sv_deg(R: np.ndarray, lag: int = 25) -> np.ndarray:
    """Singular values (deg) of the relative-rotation vectors ``R_k⁻¹ R_{k+lag}`` of a
    sequence of training rotations (consecutive in index), scaled by 1/√N (the audit's
    measure of how well a hand-eye mounting is excited; small = rotation about few axes)."""
    R = np.asarray(R, float)
    if len(R) <= lag + 2:
        return np.full(3, np.nan)
    rel = Rot.from_matrix(R[:-lag]).inv() * Rot.from_matrix(R[lag:])
    sv = np.linalg.svd(rel.as_rotvec(), compute_uv=False) / np.sqrt(len(R))
    return np.degrees(sv)


def pairwise_spread(Xs: Sequence[np.ndarray]) -> tuple[float, float]:
    """(max, median) pairwise angle (deg) between rotations (the fold-to-fold spread)."""
    pair = [angle_deg(a, b) for i, a in enumerate(Xs) for b in Xs[i + 1:]]
    return (float(max(pair)), float(np.median(pair))) if pair else (np.nan, np.nan)


def nearest_quat(t_track_s: np.ndarray, q: np.ndarray, t_s: np.ndarray, tol_s: float
                 ) -> np.ndarray:
    """Nearest-sample quaternions of a track at ``t_s`` (NaN farther than ``tol_s``)."""
    ts = np.asarray(t_track_s, float)
    j = np.clip(np.searchsorted(ts, t_s), 1, len(ts) - 1)
    j = np.where(np.abs(ts[j - 1] - t_s) < np.abs(ts[j] - t_s), j - 1, j)
    out = np.asarray(q, float)[j].copy()
    out[np.abs(ts[j] - t_s) > tol_s] = np.nan
    return out


def run_mounting(session: Any, pid: str, he: dict[str, dict[str, np.ndarray]], own: str | None,
                 params: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """The Part H mounting audit on one comparison take (``he``: hand-eye session label ->
    slot -> R_mount (body<-chip); ``own``: the label VD14 assigns to this take)."""
    from jtsmm.imu import fusion as FU

    from biomech.validation import fusion as VF
    from biomech.validation import kinematics as VK
    from biomech.validation import selection as SEL
    from biomech.validation.sync import frame_times

    p = {**DEFAULTS, **dict(params or {})}
    opt = session.optical()
    if opt is None:
        raise LookupError("comparison take has no optical group")
    t = frame_times(opt.t_us, p["exposure_s"])
    ct = opt.poses["ct"]
    ct_ok = np.isfinite(ct.reshape(len(ct), -1)).all(1)
    held = SEL.movement_intervals(session.phases, t, ct_ok)
    mov = [h for h in held if h.name in SEL.MOVEMENT_TASKS]
    wins = SEL.sync_windows(session.phases, t, ct_ok)
    tracks = {s: VF._imu_track(session, s, p["imu_max_gap_s"]) for s in BODY_SLOT.values()}
    Ropt = {s: np.asarray(opt.poses[s], float)[:, :3, :3] for s in BODY_SLOT.values()}
    lag_range = tuple(p["lag_range_s"])
    fold_rows, exc_rows = [], []
    X_free: dict[str, list[np.ndarray]] = {b: [] for b in BODY_SLOT}
    vs_he: dict[str, list[float]] = {b: [] for b in BODY_SLOT}
    for h in mov:
        tr = VK.mask_of(t, list(wins) + [m for m in mov if m is not h]) & ~VK.mask_of(t, [h])
        for body, s in BODY_SLOT.items():
            if tracks[s] is None:
                continue
            R, ok = FU.valid_rotations(Ropt[s])
            cfree = FU.calibrate(t, R, tracks[s], tr, lag_range_s=lag_range, fit_drift=None,
                                 seed=int(p["seed"]))
            X_free[body].append(cfree.X)
            fold_rows.append({"pid": pid, "fold": h.name, "body": body, "mounting": "in_take",
                              "own": False, "rms_deg": float(cfree.rms_deg),
                              "lag_ms": float(cfree.lag_s * 1e3), "angle_to_intake_deg": 0.0})
            for lab, M in sorted(he.items()):
                if s not in M:
                    continue
                cpre = FU.calibrate(t, R, tracks[s], tr, lag_range_s=lag_range, fit_drift=None,
                                    seed=int(p["seed"]), X_fixed=M[s].T)
                fold_rows.append({"pid": pid, "fold": h.name, "body": body, "mounting": lab,
                                  "own": lab == own, "rms_deg": float(cpre.rms_deg),
                                  "lag_ms": float(cpre.lag_s * 1e3),
                                  "angle_to_intake_deg": angle_deg(cfree.X.T, M[s])})
                if lab == own:
                    vs_he[body].append(angle_deg(cfree.X.T, M[s]))
            Rk = R[tr & ok]
            sv = excitation_sv_deg(Rk, int(p["rel_rot_lag_frames"]))
            exc_rows.append({"pid": pid, "fold": h.name, "body": body, "n_frames": int(len(Rk)),
                             "sv1_deg": sv[0], "sv2_deg": sv[1], "sv3_deg": sv[2]})
    folds = pd.DataFrame(fold_rows, columns=["pid", "fold", "body", "mounting", "own", "rms_deg",
                                             "lag_ms", "angle_to_intake_deg"])
    resid = []
    for (body, lab), g in folds.groupby(["body", "mounting"], sort=False):
        resid.append({"pid": pid, "body": body, "mounting": lab, "own": bool(g["own"].iloc[0]),
                      "n_folds": int(len(g)), "rms_median_deg": float(g["rms_deg"].median()),
                      "rms_min_deg": float(g["rms_deg"].min()),
                      "rms_max_deg": float(g["rms_deg"].max()),
                      "angle_to_intake_median_deg": float(g["angle_to_intake_deg"].median())})
    spread = []
    for body, Xs in X_free.items():
        if not Xs:
            continue
        mx, md = pairwise_spread(Xs)
        spread.append({"pid": pid, "body": body, "n_folds": len(Xs), "spread_max_deg": mx,
                       "spread_median_deg": md,
                       "intake_vs_handeye_median_deg": float(np.median(vs_he[body]))
                       if vs_he[body] else np.nan})
    return {"mounting_sessions": session_rotations(pid, he),
            "mounting_folds": folds, "mounting_residual": pd.DataFrame(resid),
            "mounting_excitation": pd.DataFrame(exc_rows),
            "mounting_spread": pd.DataFrame(spread),
            "mounting_imu_stage": imu_stage_estimator(t, held, Ropt, tracks, pid, he, own, p)}


def imu_stage_estimator(t: np.ndarray, held: Sequence[Any], Ropt: dict[str, np.ndarray],
                        tracks: dict[str, Any], pid: str,
                        he: dict[str, dict[str, np.ndarray]], own: str | None,
                        p: dict[str, Any]) -> pd.DataFrame:
    """The dataset's imu-stage calibration (``biomech.imu.calibration.fit``: head mounting
    ``A``, heading offset and jaw mounting ``X`` jointly) on the take's ``free_motion``, with
    the HP marker frame as the maxillary frame, against each hand-eye mounting."""
    from jtsmm.imu import fusion as FU

    from biomech.imu import calibration as IC
    from biomech.validation import kinematics as VK

    cols = ["pid", "task", "handeye_session", "own", "n", "rms_deg", "head_angle_deg",
            "jaw_angle_deg"]
    task = str(p["imu_stage_task"])
    iv = [h for h in held if h.name == task]
    if not iv or any(tracks[s] is None for s in ("cra", "mta")):
        return pd.DataFrame(columns=cols)
    k = VK.mask_of(t, iv)
    Rh, okh = FU.valid_rotations(Ropt["cra"])
    Rm, okm = FU.valid_rotations(Ropt["mta"])
    Rrel = np.einsum("nji,njk->nik", Rh, Rm)
    q = {s: nearest_quat(tracks[s].t_s, tracks[s].q, t, float(p["quat_match_s"]))
         for s in ("cra", "mta")}
    sel = k & okh & okm
    cal = IC.fit(q["cra"][sel], q["mta"][sel], Rrel[sel])
    rows = [{"pid": pid, "task": task, "handeye_session": lab, "own": lab == own, "n": int(sel.sum()),
             "rms_deg": float(cal.rms_deg),
             "head_angle_deg": angle_deg(cal.A, M["cra"]) if "cra" in M else np.nan,
             "jaw_angle_deg": angle_deg(cal.X, M["mta"].T) if "mta" in M else np.nan}
            for lab, M in sorted(he.items())]
    return pd.DataFrame(rows, columns=cols)


# == IMU dynamic calibration (validation part of the I3c audit; helpers in biomech.imu.dyncal) ====
def run_dyncal(session: Any, pid: str, fusion_arrays: dict[str, Any],
               params: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Validation part of the I3c audit: the IMU-alone rotation error against Qualisys
    (``ref_fusion`` ``err_rot_I_deg`` at ``t_master_s``) on the take's still periods."""
    p = {**DEFAULTS, **dict(params or {})}
    g = imu_grid(session, float(p["grid_hz"]))
    if g is None or "err_rot_I_deg" not in fusion_arrays:
        note = "IMU streams missing" if g is None else "no IMU-alone error trace"
        return {"dyncal_summary": pd.DataFrame([{"pid": pid, "note": note}])}
    # one span per phase name, its first occurrence (as the audit: ``sd.phase(name)``)
    first: dict[str, Any] = {}
    for ph in session.phases:
        first.setdefault(ph.name, ph)
    phases = [(n, float(ph.t_start_us), float(ph.t_end_us)) for n, ph in first.items()]
    s, per, tr = dyncal_trace(pid, *g, phases, np.asarray(fusion_arrays["t_master_s"], float),
                              np.asarray(fusion_arrays["err_rot_I_deg"], float),
                              "IMU-alone rotation error vs Qualisys", p)
    return {"dyncal_summary": s, "dyncal_periods": per, "dyncal_transitions": tr}
