# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Opening-phase analyses: coupling curves, helical axes and hinge fits (C1).

Every function works on the *opening halves* (start -> peak) of the good
``open_close`` cycles, each referenced to its own closed start pose.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from biomech.kinematics import se3
from biomech.kinematics.clinical import TaskCycles, theta_deg
from biomech.kinematics.helical import stepped_screws
from biomech.kinematics.hinge_fit import (TEETH, HingeResult, fit_hinge, fit_rot_trans,
                                          hinge_through)
from biomech.kinematics.pose_chain import JawMotion

HINGE_POINTS = ("incisor", "left", "right", "tmj_left", "tmj_right")


@dataclass
class Opening:
    cycle: int
    T: np.ndarray            # (n, 4, 4)
    theta: np.ndarray        # (n,) sagittal angle since start, deg


def openings(jm: JawMotion, tc: TaskCycles) -> list[Opening]:
    idx = np.flatnonzero(tc.sel)
    out = []
    for k, c in enumerate(tc.cycles):
        if not c.ok:
            continue
        T = jm.T[idx[c.start]: idx[c.peak] + 1]
        ok = se3.valid(T)
        if ok.sum() < 5 or not ok[0]:
            continue
        th, _ = theta_deg(T, T[0])
        out.append(Opening(k, T, th))
    return out


def coupling_curves(jm: JawMotion, ops: list[Opening], grid_deg: np.ndarray
                    ) -> dict[str, np.ndarray]:
    """Per-cycle curves on an angle grid: incisal opening, condyle anterior/3D displacement."""
    names = {"opening_mm": [], "tmj_mid_ant_mm": [], "tmj_mid_disp_mm": [],
             "tmj_left_disp_mm": [], "tmj_right_disp_mm": []}
    for op in ops:
        inc = se3.apply(op.T, jm.points_mp["incisor"])
        mid = se3.apply(op.T, jm.points_mp["tmj_mid"])
        tl = se3.apply(op.T, jm.points_mp["tmj_left"])
        tr = se3.apply(op.T, jm.points_mp["tmj_right"])
        th = np.maximum.accumulate(np.nan_to_num(op.theta, nan=-np.inf))
        th[~np.isfinite(th)] = np.nan
        series = {
            "opening_mm": -(inc[:, 2] - inc[0, 2]),
            "tmj_mid_ant_mm": mid[:, 0] - mid[0, 0],
            "tmj_mid_disp_mm": np.linalg.norm(mid - mid[0], axis=1),
            "tmj_left_disp_mm": np.linalg.norm(tl - tl[0], axis=1),
            "tmj_right_disp_mm": np.linalg.norm(tr - tr[0], axis=1),
        }
        ok = np.isfinite(th)
        thu, first = np.unique(th[ok], return_index=True)
        for k, v in series.items():
            vv = v[ok][first]
            names[k].append(np.interp(grid_deg, thu, vv, left=np.nan, right=np.nan)
                            if thu.size >= 2 else np.full(grid_deg.shape, np.nan))
    return {k: np.array(v) if v else np.empty((0, grid_deg.size)) for k, v in names.items()}


def fha_rows(jm: JawMotion, ops: list[Opening], pid: str, step_deg: float) -> list[dict[str, Any]]:
    rows = []
    for op in ops:
        mid0 = se3.apply(op.T[0], jm.points_mp["tmj_mid"])
        for i, j, sc in stepped_screws(op.T, op.theta, step_deg):
            pp = sc.pierce(1, 0.0)
            rows.append({
                "pid": pid, "cycle": op.cycle, "theta_from_deg": float(op.theta[i]),
                "theta_to_deg": float(op.theta[j]), "phi_deg": sc.phi_deg, "t_mm": sc.t_mm,
                "n_x": sc.n[0], "n_y": sc.n[1], "n_z": sc.n[2],
                "pierce_x_mm": pp[0] if pp is not None else np.nan,
                "pierce_z_mm": pp[2] if pp is not None else np.nan,
                "dist_tmj_mid_mm": float(np.hypot(pp[0] - mid0[0], pp[2] - mid0[2]))
                if pp is not None else np.nan,
            })
    return rows


def hinge_data(jm: JawMotion, ops: list[Opening], max_frames: int = 30
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacked (A, B, theta) for the hinge fits; each opening subsampled to ``max_frames``."""
    P = np.stack([jm.points_mp[r] for r in HINGE_POINTS])
    A, B, TH = [], [], []
    for op in ops:
        ok = np.flatnonzero(se3.valid(op.T))
        if ok.size < 3:
            continue
        pick = ok[np.unique(np.linspace(1, ok.size - 1, min(max_frames, ok.size - 1)).astype(int))]
        a = se3.apply(op.T[0], P)
        b = np.einsum("fij,kj->fki", op.T[pick, :3, :3], P) + op.T[pick, None, :3, 3]
        A.append(np.broadcast_to(a, b.shape))
        B.append(b)
        TH.append(op.theta[pick])
    if not A:
        return np.empty((0, len(P), 3)), np.empty((0, len(P), 3)), np.empty(0)
    return np.ascontiguousarray(np.concatenate(A)), np.concatenate(B), np.concatenate(TH)


def hinge_fits(A: np.ndarray, B: np.ndarray,
               init: dict[str, HingeResult] | None = None, teeth: bool = True
               ) -> dict[str, HingeResult]:
    """The three models, plus (``teeth``) the D25 teeth-only hinge ``hinge_teeth``: the
    fixed axis fitted to the incisor and molar residuals only (sensitivity of C1 to the
    skin-palpated condyle points; started from the five-point fit, so it needs no condyle
    position of its own). ``init`` (e.g. the full-data fits) warm-starts the optimizers of a
    bootstrap resample (an06 §2)."""
    tmj_l = A[:, 3].mean(axis=0)
    tmj_r = A[:, 4].mean(axis=0)
    if init is not None:
        hinge = fit_hinge(A, B, s0=init["hinge_fit"].s, n0=init["hinge_fit"].n)
    else:
        hinge = fit_hinge(A, B, s0=0.5 * (tmj_l + tmj_r), n0=tmj_l - tmj_r)
    out = {
        "hinge_fit": hinge,
        "hinge_tmj": hinge_through(A, B, tmj_l, tmj_r),
        "rot_trans": fit_rot_trans(A, B, hinge,
                                   start=init.get("rot_trans") if init is not None else None),
    }
    if teeth:
        h0 = init["hinge_teeth"] if init is not None and "hinge_teeth" in init else hinge
        out["hinge_teeth"] = fit_hinge(A, B, s0=h0.s, n0=h0.n, points=TEETH,
                                       model="hinge_teeth")
    return out


def hinge_summary(res: dict[str, HingeResult], pid: str) -> list[dict[str, Any]]:
    rows = []
    for name, r in res.items():
        pierce = r.s - (r.s[1] / r.n[1]) * r.n if abs(r.n[1]) > 0.5 else np.full(3, np.nan)
        rows.append({
            "pid": pid, "model": name, "n_frames": int(r.residual.shape[0]),
            "rms_incisor_mm": r.rms(0), "max_incisor_mm": r.max(0),
            "rms_molars_mm": r.rms(slice(1, 3)), "rms_condyles_mm": r.rms(slice(3, 5)),
            "max_condyles_mm": r.max(slice(3, 5)),
            "axis_x": r.n[0], "axis_y": r.n[1], "axis_z": r.n[2],
            "axis_pierce_x_mm": pierce[0], "axis_pierce_z_mm": pierce[2],
            "theta_max_deg": float(np.nanmax(r.theta_deg)),
            "translation_range_mm": float(np.nanpercentile(r.tau_mm, 95)
                                          - np.nanpercentile(r.tau_mm, 5))
            if r.tau_mm is not None else np.nan,
            "trans_dir_x": r.u[0] if r.u is not None else np.nan,
            "trans_dir_z": r.u[2] if r.u is not None else np.nan,
        })
    return rows
