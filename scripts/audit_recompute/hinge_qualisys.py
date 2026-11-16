# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""C1 cross-check on the validation takes: the hinge fits of the dataset (free fixed axis,
axis through the condyle points) on the JTS-MM and on the Qualisys trajectories of the same
openings, so the hinge residual of a reference with ~0.05 mm precision can be compared with
the JTS-MM one (is the residual anatomy or tracking noise?).

Uses ``ref_kinematics`` arrays (incisor and condyle points of both systems, each in its own
head frame) and the ``ref_agreement`` cycles (start -> peak = opening). Three points per frame
(incisor, condyle L, condyle R) instead of the dataset's five (no molars in the validation
arrays); both systems are fitted identically.

    python hinge_qualisys.py VALIDATION_STORE [--tasks open_close chewing]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from biomech.kinematics.hinge_fit import fit_hinge, hinge_through


def openings(arr: dict, cyc: pd.DataFrame, task: str, sysk: str, max_frames: int = 30
             ) -> tuple[np.ndarray, np.ndarray]:
    P = np.stack([arr[f"incisor_{sysk}_mm"], arr[f"tmj_left_{sysk}_mm"],
                  arr[f"tmj_right_{sysk}_mm"]], axis=1)                  # (N, 3, 3)
    A, B = [], []
    for r in cyc[cyc.task == task].itertuples():
        idx = np.arange(int(r.start), int(r.peak) + 1)
        ok = idx[np.isfinite(P[idx]).all(axis=(1, 2))]
        if ok.size < 4 or ok[0] != r.start:
            continue
        pick = ok[np.unique(np.linspace(1, ok.size - 1, min(max_frames, ok.size - 1)).astype(int))]
        b = P[pick]
        A.append(np.broadcast_to(P[ok[0]], b.shape))
        B.append(b)
    if not A:
        return np.empty((0, 3, 3)), np.empty((0, 3, 3))
    return np.ascontiguousarray(np.concatenate(A)), np.concatenate(B)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("store", type=Path)
    ap.add_argument("--tasks", nargs="+", default=["open_close", "chewing"])
    a = ap.parse_args()
    rows = []
    for pdir in sorted((a.store / "cache" / "ref_kinematics").glob("s*")):
        pid = pdir.name
        arr = dict(np.load(pdir / "arrays.npz"))
        cyc = pd.read_parquet(a.store / "cache" / "ref_agreement" / pid / "cycles.parquet")
        for task in a.tasks:
            for sysk, name in (("s", "JTS-MM"), ("q", "Qualisys")):
                A, B = openings(arr, cyc, task, sysk)
                if len(A) < 10:
                    continue
                tl, tr = A[:, 1].mean(axis=0), A[:, 2].mean(axis=0)
                fit = fit_hinge(A, B, s0=0.5 * (tl + tr), n0=tl - tr)
                tmj = hinge_through(A, B, tl, tr)
                d_axis = float(np.linalg.norm(np.cross(0.5 * (tl + tr) - fit.s, fit.n)))
                rows.append({"pid": pid, "task": task, "system": name,
                             "n_openings": int((cyc.task == task).sum()), "n_frames": len(A),
                             "opening_deg_max": float(np.nanmax(np.abs(fit.theta_deg))),
                             "hinge_fit_incisor_rms_mm": rms(fit.residual[:, 0]),
                             "hinge_tmj_incisor_rms_mm": rms(tmj.residual[:, 0]),
                             "axis_to_condyle_mid_mm": d_axis})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
