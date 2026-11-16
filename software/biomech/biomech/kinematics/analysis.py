# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One participant's kinematics: tables and signals for the ``kinematics`` stage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.frames.landmarks import LANDMARKS
from biomech.io.session import SessionData
from biomech.kinematics.derivatives import angular_velocity, derivatives
from biomech.kinematics import se3
from biomech.kinematics.clinical import (TASKS, anatomy, cycle_metrics, find_task_cycles,
                                         noise_floor, theta_deg)
from biomech.kinematics.coupling import (coupling_curves, fha_rows, hinge_data, hinge_fits,
                                         hinge_summary, openings)
from biomech.kinematics.geometry import jaw_geometry
from biomech.kinematics.pose_chain import JawMotion, KinParams, build

SUMMARY_METRICS = ("opening_mm", "incisal_disp_mm", "protrusion_mm", "retrusion_mm",
                   "lateral_mm", "deviation_mm", "theta_y_deg", "theta_total_deg", "condyle_left_disp_mm",
                   "condyle_right_disp_mm", "condyle_left_ant_mm", "condyle_right_ant_mm",
                   "peak_open_vel_dps", "peak_close_vel_dps", "peak_incisal_speed_mm_s",
                   "duration_s")


def _summary(cyc: pd.DataFrame, pid: str) -> pd.DataFrame:
    rows = []
    ok = cyc[cyc["ok"]] if not cyc.empty else cyc
    for (task, side), g in ok.groupby(["task", "side"]):
        for m in SUMMARY_METRICS:
            v = g[m].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            if v.size == 0:
                continue
            mean = v.mean()
            rows.append({"pid": pid, "task": task, "side": side, "metric": m,
                         "n_cycles": int(v.size), "median": float(np.median(v)),
                         "max": float(v.max()), "mean": float(mean),
                         "sd": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                         # CV only for strictly positive (ratio-scale) metrics; a signed
                         # metric (deviation, protrusion) can have a mean near 0.
                         "cv": float(v.std(ddof=1) / mean)
                         if v.size > 1 and (v > 0).all() else np.nan})
    return pd.DataFrame(rows)


def _opening_fit(theta: np.ndarray, opening: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares opening = a1·θ + a2·θ² (through the origin): ((a1, a2), RMSE)."""
    X = np.column_stack([theta, theta ** 2])
    coef, *_ = np.linalg.lstsq(X, opening, rcond=None)
    resid = opening - X @ coef
    return coef, float(np.sqrt(np.mean(resid ** 2)))


MIN_LOCO_FRAMES = 10           # frames the leave-one-cycle-out fit needs (2 coefficients)


def opening_model_loco(theta: np.ndarray, opening: np.ndarray, cycle: np.ndarray, pid: str
                       ) -> pd.DataFrame:
    """D26: the opening model refitted once per open_close opening with that cycle's frames
    left out (``cycle``: the kinematics cycle number of every frame), so the IMU stage can
    predict each fitted cycle out of sample. One row per fitted cycle; NaN coefficients
    when the other cycles hold fewer than ``MIN_LOCO_FRAMES`` frames (a single cycle)."""
    rows = []
    for c in np.unique(cycle):
        keep = cycle != c
        coef, rmse = ((np.full(2, np.nan), np.nan) if keep.sum() < MIN_LOCO_FRAMES
                      else _opening_fit(theta[keep], opening[keep]))
        rows.append({"pid": pid, "cycle": int(c), "a1_mm_per_deg": float(coef[0]),
                     "a2_mm_per_deg2": float(coef[1]), "rmse_mm": rmse,
                     "n": int(keep.sum())})
    return pd.DataFrame(rows, columns=["pid", "cycle", "a1_mm_per_deg", "a2_mm_per_deg2",
                                       "rmse_mm", "n"])


def analyse(sd: SessionData, pid: str, kp: KinParams, p: dict[str, Any]
            ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray], JawMotion]:
    jm = build(sd, kp)
    tables: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {"T_max_hp": jm.T_max_hp}

    roles = {jm.upper.incisor: "incisor", jm.upper.left: "left", jm.upper.right: "right",
             jm.lower.incisor: "incisor", jm.lower.left: "left", jm.lower.right: "right"}
    rows = []
    for lm in jm.landmarks.items.values():
        base = lm.name.removesuffix("_hp")
        spec = LANDMARKS.get(base)
        rows.append({"pid": pid, "name": lm.name, "landmark": base,
                     "jaw": spec.jaw if spec else "", "description": spec.description if spec
                     else "", "role_used": roles.get(base, spec.role if spec else ""),
                     "body": lm.body, "x": lm.point[0], "y": lm.point[1], "z": lm.point[2],
                     "mad_mm": lm.mad_mm, "n": lm.n})
    tables["landmarks"] = pd.DataFrame(rows)

    task_cycles = {}
    for task in TASKS:
        ph = sd.phase(task)
        if ph is None:
            continue
        for tc in find_task_cycles(jm, task, ph.t_start_us, ph.t_end_us, p["cycles"]):
            task_cycles[(task, tc.side)] = tc

    # Closed reference: the open_close cycle start whose incisal height is the
    # median of all good starts (an actual recorded pose, not an average).
    oc = task_cycles.get(("open_close", ""))
    closed_T, inc0 = None, None
    if oc is not None:
        idx = np.flatnonzero(oc.sel)
        starts = np.array([idx[c.start] for c in oc.cycles if c.ok], dtype=int)
        starts = starts[se3.valid(jm.T[starts])] if starts.size else starts
        if starts.size:
            z = se3.apply(jm.T[starts], jm.points_mp["incisor"])[:, 2]
            closed_T = jm.T[starts[np.argsort(z)[len(z) // 2]]]
            inc0 = se3.apply(closed_T, jm.points_mp["incisor"])

    cyc_rows: list[dict[str, Any]] = []
    for tc in task_cycles.values():
        cyc_rows += cycle_metrics(jm, tc, pid, closed_T)
    cyc = pd.DataFrame(cyc_rows)
    tables["cycles"] = cyc
    tables["summary"] = _summary(cyc, pid)

    rest = sd.phase("rest")
    anat = anatomy(jm, inc0)
    if rest is not None:
        anat.update(noise_floor(jm, rest.t_start_us, rest.t_end_us))
    tables["anatomy"] = pd.DataFrame([{"pid": pid, **anat}])
    if closed_T is not None:
        # Frame-pitch QC (E13): the lower occlusal line (incisor -> molar midpoint) at closure
        # in the maxillary frame. Near 0 deg when both jaws' landmarks agree; a large pitch
        # points at misplaced tooth points (a tilted frame), not at a low condyle palpation.
        low = {r: se3.apply(closed_T, jm.points_mp[r]) for r in ("incisor", "left", "right")}
        d = 0.5 * (low["left"] + low["right"]) - low["incisor"]
        tables["anatomy"]["occlusal_pitch_deg"] = float(np.degrees(np.arctan2(d[2], -d[0])))

    # Largest opening seen in any task (lower bound of MIO; no maximal-opening
    # task was recorded, Deviation D1).
    if inc0 is not None:
        zmin, where = np.inf, ""
        for ph in sd.phases:
            if ph.kind != "experiment":
                continue
            z = jm.track("incisor")[jm.window(ph.t_start_us, ph.t_end_us)][:, 2]
            if np.isfinite(z).any() and np.nanmin(z) < zmin:
                zmin, where = float(np.nanmin(z)), ph.name
        tables["anatomy"]["max_observed_opening_mm"] = float(inc0[2] - zmin)
        tables["anatomy"]["max_observed_opening_task"] = where

    # Signals per motion phase (for the IMU/EMG/timing stages and the viewer).
    for ph in sd.phases:
        if ph.kind != "experiment":
            continue
        sel = jm.window(ph.t_start_us, ph.t_end_us)
        key = ph.name
        arrays[f"{key}__t_us"] = jm.t_us[sel]
        arrays[f"{key}__T"] = jm.T[sel].astype(np.float32)
        for role in ("incisor", "left", "right", "tmj_left", "tmj_right"):
            arrays[f"{key}__{role}"] = jm.track(role)[sel].astype(np.float32)
        # Derivatives (kinematics.derivatives): one estimate for metrics, views, videos.
        t_ph = jm.t_us[sel]
        v, a = derivatives(t_ph, jm.track("incisor")[sel])
        arrays[f"{key}__incisor_vel"] = v.astype(np.float32)
        arrays[f"{key}__incisor_acc"] = a.astype(np.float32)
        arrays[f"{key}__omega"] = angular_velocity(t_ph, jm.T[sel][:, :3, :3]).astype(np.float32)
        if closed_T is not None:
            th = theta_deg(jm.T[sel], closed_T)[0]
            arrays[f"{key}__theta_deg"] = th.astype(np.float32)
            # Unsmoothed angle (before the 6 Hz low-pass): the independent reference for the
            # movement-bandwidth noise check (an04 §6, D12).
            arrays[f"{key}__theta_raw_deg"] = theta_deg(jm.T_raw[sel], closed_T)[0] \
                .astype(np.float32)
            tv, ta = derivatives(t_ph, th)
            arrays[f"{key}__theta_vel"] = tv.astype(np.float32)
            arrays[f"{key}__theta_acc"] = ta.astype(np.float32)
    if closed_T is not None:
        arrays["T_closed"] = closed_T

    # C1: openings of open_close.
    fits = None
    if oc is not None:
        ops = openings(jm, oc)
        grid = np.arange(0.0, p["coupling"]["max_deg"] + 1e-9, p["coupling"]["step_deg"])
        curves = coupling_curves(jm, ops, grid)
        arrays["coupling__grid_deg"] = grid
        for k, v in curves.items():
            arrays[f"coupling__{k}"] = v
        tables["fha"] = pd.DataFrame(fha_rows(jm, ops, pid, p["fha"]["step_deg"]))
        A, B, _ = hinge_data(jm, ops, p["hinge"]["max_frames_per_cycle"])
        if A.shape[0] >= 10:
            fits = hinge_fits(A, B)
            tables["hinge"] = pd.DataFrame(hinge_summary(fits, pid))
            arrays["hinge__residual_incisor"] = fits["hinge_fit"].residual[:, 0]
            arrays["hinge__theta_deg"] = fits["hinge_fit"].theta_deg
            arrays["hinge__tau_mm"] = fits["rot_trans"].tau_mm
        # Coupling model for the IMU stage: opening = a1*theta + a2*theta^2 (through 0).
        th, op, cy = [], [], []
        for c in ops:
            inc = se3.apply(c.T, jm.points_mp["incisor"])
            th.append(c.theta)
            op.append(-(inc[:, 2] - inc[0, 2]))
            cy.append(np.full(len(c.theta), c.cycle))
        if th:
            th_a, op_a, cy_a = np.concatenate(th), np.concatenate(op), np.concatenate(cy)
            ok = np.isfinite(th_a) & np.isfinite(op_a)
            coef, rmse = _opening_fit(th_a[ok], op_a[ok])
            tables["opening_model"] = pd.DataFrame([{
                "pid": pid, "a1_mm_per_deg": coef[0], "a2_mm_per_deg2": coef[1],
                "rmse_mm": rmse, "n": int(ok.sum())}])
            tables["opening_model_loco"] = opening_model_loco(th_a[ok], op_a[ok], cy_a[ok], pid)

    # Jaw geometry per participant (D21): landmark triangles and planes, hinge axis.
    tables["geometry"] = pd.DataFrame([{"pid": pid, **jaw_geometry(
        jm, closed_T, fits["hinge_fit"] if fits else None)}])

    # Specification-curve variants for MIO (analysis_plan §5.3).
    spec = []
    if oc is not None and not cyc.empty:
        g = cyc[(cyc.task == "open_close") & cyc.ok]
        idx = np.flatnonzero(oc.sel)
        if not g.empty:
            spec.append({"outcome": "mio_mm", "variant": "cycle_start|vertical|lowpass",
                         "value": float(g["opening_mm"].max())})
            spec.append({"outcome": "mio_mm", "variant": "cycle_start|3d|lowpass",
                         "value": float(g["incisal_disp_mm"].max())})
            raw_inc = se3.apply(jm.T_raw[idx], jm.points_mp["incisor"])
            vals = [raw_inc[c.start, 2] - np.nanmin(raw_inc[c.start:c.end + 1, 2])
                    for c in oc.cycles if c.ok]
            spec.append({"outcome": "mio_mm", "variant": "cycle_start|vertical|raw",
                         "value": float(np.nanmax(vals))})
            if rest is not None:
                rsel = jm.window(rest.t_start_us, rest.t_end_us)
                z_rest = np.nanmedian(jm.track("incisor")[rsel][:, 2])
                inc = jm.track("incisor")[idx]
                vals = [z_rest - np.nanmin(inc[c.start:c.end + 1, 2]) for c in oc.cycles if c.ok]
                spec.append({"outcome": "mio_mm", "variant": "rest|vertical|lowpass",
                             "value": float(np.nanmax(vals))})
    tables["spec"] = pd.DataFrame([{"pid": pid, **r} for r in spec])
    return tables, arrays, jm
