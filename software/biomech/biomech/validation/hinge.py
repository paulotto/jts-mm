# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD21 (exploratory): the dataset's C1 hinge models on both systems of the
comparison takes -- is the hinge residual anatomy or tracking error?

The dataset fits a fixed hinge to the JTS-MM openings (C1: best-fit free axis; D33: the
axis through the condyle points). Here the same fits run on the JTS-MM and the Qualisys
trajectories of the same openings (``ref_agreement`` cycles, start -> peak), each system in
its own head frame, with three points per frame (incisor, condyle L, condyle R; the
comparison arrays carry no molars) and the openings subsampled as in the dataset
(``max_frames`` per opening). Qualisys tracks with sub-millimetre precision, so its residual
bounds the anatomical non-hinge motion from below; the JTS-MM minus Qualisys difference is
the part that tracking error adds. Per participant, descriptive (n = 2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.kinematics.hinge_fit import fit_hinge, hinge_through

SYSTEMS = (("s", "jtsmm"), ("q", "qualisys"))
TASKS = ("open_close", "chewing")


def opening_points(arr: dict[str, np.ndarray], cycles: pd.DataFrame, task: str, key: str,
                   max_frames: int = 30) -> tuple[np.ndarray, np.ndarray, int]:
    """(A, B, n_openings) of one system: closed-pose and current positions of the incisor and
    both condyle points, per opening subsampled to ``max_frames`` frames after the start; an
    opening is used only when it is tracked at its start and has at least 4 tracked frames."""
    P = np.stack([arr[f"incisor_{key}_mm"], arr[f"tmj_left_{key}_mm"],
                  arr[f"tmj_right_{key}_mm"]], axis=1)
    A, B, used = [], [], 0
    for r in cycles[cycles.task == task].itertuples():
        idx = np.arange(int(r.start), int(r.peak) + 1)
        ok = idx[np.isfinite(P[idx]).all(axis=(1, 2))]
        if ok.size < 4 or ok[0] != int(r.start):
            continue
        pick = ok[np.unique(np.linspace(1, ok.size - 1, min(max_frames, ok.size - 1)).astype(int))]
        A.append(np.broadcast_to(P[ok[0]], (pick.size, 3, 3)))
        B.append(P[pick])
        used += 1
    if not A:
        return np.empty((0, 3, 3)), np.empty((0, 3, 3)), 0
    return np.ascontiguousarray(np.concatenate(A)), np.concatenate(B), used


def run_ref_hinge(arr: dict[str, np.ndarray], cycles: pd.DataFrame, pid: str,
                  max_frames: int = 30, min_frames: int = 10) -> dict[str, pd.DataFrame]:
    """Table ``hinge``: per task and system the incisor RMS of the best-fit fixed hinge and
    of the hinge through the condyle points, the best-fit axis's distance from the condyle
    midpoint and the largest fitted opening angle; table ``hinge_diff``: JTS-MM minus
    Qualisys per task."""
    rows: list[dict[str, Any]] = []
    for task in TASKS:
        for key, system in SYSTEMS:
            A, B, used = opening_points(arr, cycles, task, key, max_frames)
            row: dict[str, Any] = {"pid": pid, "task": task, "system": system,
                                   "n_openings": used, "n_frames": int(len(A))}
            if len(A) >= min_frames:
                tl, tr = A[:, 1].mean(axis=0), A[:, 2].mean(axis=0)
                mid = 0.5 * (tl + tr)
                fit = fit_hinge(A, B, s0=mid, n0=tl - tr)
                tmj = hinge_through(A, B, tl, tr)
                row.update({"hinge_fit_mm": fit.rms(0), "hinge_tmj_mm": tmj.rms(0),
                            "axis_to_condyles_mm": float(np.linalg.norm(
                                np.cross(mid - fit.s, fit.n))),
                            "opening_max_deg": float(np.nanmax(np.abs(fit.theta_deg)))})
            rows.append(row)
    hinge = pd.DataFrame(rows)
    diff = []
    for task, g in hinge.groupby("task", sort=False):
        s, q = g[g.system == "jtsmm"], g[g.system == "qualisys"]
        if s.empty or q.empty or "hinge_fit_mm" not in g:
            continue
        diff.append({"pid": pid, "task": task,
                     "d_hinge_fit_mm": float(s.hinge_fit_mm.iloc[0] - q.hinge_fit_mm.iloc[0]),
                     "d_hinge_tmj_mm": float(s.hinge_tmj_mm.iloc[0] - q.hinge_tmj_mm.iloc[0])})
    return {"hinge": hinge, "hinge_diff": pd.DataFrame(diff)}
