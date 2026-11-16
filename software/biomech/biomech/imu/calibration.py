# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One-time per-participant IMU calibration against optical (an04 §3).

The two BNO085 game rotation vectors have independent heading (yaw)
references, and the IMU mountings on the arrays were not calibrated in the
pilot. The mandible orientation in the maxillary frame is modelled as

    R_max<-mp(t) = A · R_cra(t)^T · Rz(psi) · R_mta(t) · X

with R_* the IMU orientations (world<-body), Rz(psi) the heading offset
between the IMUs' worlds, X = R_mta<-mp the mandible mounting and A =
R_max<-cra (head mounting + maxillary frame). Seven parameters, fitted by
nonlinear least squares on a calibration task (``free_motion``) and tested on
the other tasks. X is initialized by Park-Martin (AX = XB on mandible-IMU vs
optical relative rotations), psi by a grid. If the head barely rotates, A and
psi are not separately identifiable; their product still is, which is all the
prediction needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot

from jtsmm.imu.batch import quat_to_matrix
from jtsmm.imu.mounting import MountingFitError, park_martin_axxb


@dataclass
class ImuCalibration:
    A: np.ndarray            # (3, 3) R_max<-cra
    psi_rad: float
    X: np.ndarray            # (3, 3) R_mta<-mp
    rms_deg: float           # fit residual on the calibration samples
    n: int
    init: str

    def predict(self, q_cra: np.ndarray, q_mta: np.ndarray) -> np.ndarray:
        """(N, 3, 3) predicted R_max<-mp; NaN where either quaternion is NaN."""
        Rc = quat_to_matrix(q_cra)
        Rm = quat_to_matrix(q_mta)
        Rz = Rot.from_rotvec([0, 0, self.psi_rad]).as_matrix()
        out = self.A @ np.swapaxes(Rc, 1, 2) @ Rz @ Rm @ self.X
        bad = ~(np.isfinite(q_cra).all(1) & np.isfinite(q_mta).all(1))
        out[bad] = np.nan
        return out


def _residual(x: np.ndarray, Rc: np.ndarray, Rm: np.ndarray, Ro: np.ndarray) -> np.ndarray:
    A = Rot.from_rotvec(x[0:3]).as_matrix()
    Rz = Rot.from_rotvec([0, 0, x[3]]).as_matrix()
    X = Rot.from_rotvec(x[4:7]).as_matrix()
    pred = A @ np.swapaxes(Rc, 1, 2) @ Rz @ Rm @ X
    err = np.swapaxes(pred, 1, 2) @ Ro
    return Rot.from_matrix(err).as_rotvec().ravel()


def _park_martin_X(Rm: np.ndarray, Ro: np.ndarray, n_pairs: int = 400,
                   seed: int = 0) -> np.ndarray | None:
    rng = np.random.default_rng(seed)
    n = len(Rm)
    A, B = [], []
    for _ in range(n_pairs * 4):
        i, j = rng.integers(0, n, 2)
        a = Ro[i].T @ Ro[j]
        b = Rm[i].T @ Rm[j]
        if np.linalg.norm(Rot.from_matrix(a).as_rotvec()) < np.radians(3):
            continue
        A.append(a)
        B.append(b)
        if len(A) >= n_pairs:
            break
    if len(A) < 20:
        return None
    try:
        Xpm, _, _ = park_martin_axxb(A, B)          # R_mp<-mta
    except MountingFitError:
        return None
    return Xpm.T


def fit(q_cra: np.ndarray, q_mta: np.ndarray, R_opt: np.ndarray, max_samples: int = 800,
        seed: int = 0) -> ImuCalibration:
    ok = np.isfinite(q_cra).all(1) & np.isfinite(q_mta).all(1) & np.isfinite(
        R_opt.reshape(len(R_opt), -1)).all(1)
    idx = np.flatnonzero(ok)
    if idx.size < 30:
        raise ValueError("too few synchronized IMU/optical samples for calibration")
    if idx.size > max_samples:
        idx = idx[np.linspace(0, idx.size - 1, max_samples).astype(int)]
    Rc, Rm, Ro = quat_to_matrix(q_cra[idx]), quat_to_matrix(q_mta[idx]), R_opt[idx]

    inits = []
    X0 = _park_martin_X(Rm, Ro, seed=seed)
    rng = np.random.default_rng(seed)
    x_list = [Rot.from_matrix(X0).as_rotvec()] if X0 is not None else []
    x_list += [Rot.random(random_state=rng.integers(1 << 31)).as_rotvec() for _ in range(6)]
    for k, xr in enumerate(x_list):
        X = Rot.from_rotvec(xr).as_matrix()
        for psi in np.radians(np.arange(0, 360, 45)):
            Rz = Rot.from_rotvec([0, 0, psi]).as_matrix()
            M = np.swapaxes(Rc, 1, 2) @ Rz @ Rm @ X
            A0 = Rot.from_matrix(Ro @ np.swapaxes(M, 1, 2)).mean().as_matrix()
            x0 = np.concatenate([Rot.from_matrix(A0).as_rotvec(), [psi], xr])
            cost = float(np.mean(_residual(x0, Rc, Rm, Ro) ** 2))
            inits.append((cost, x0, "park_martin" if (k == 0 and X0 is not None) else "random"))
    inits.sort(key=lambda r: r[0])
    best = None
    for cost0, x0, how in inits[:4]:
        sol = least_squares(_residual, x0, args=(Rc, Rm, Ro), method="lm")
        if best is None or sol.cost < best[0].cost:
            best = (sol, how)
    sol, how = best
    r = sol.fun.reshape(-1, 3)
    rms = float(np.degrees(np.sqrt(np.mean(np.sum(r ** 2, axis=1)))))
    return ImuCalibration(Rot.from_rotvec(sol.x[0:3]).as_matrix(), float(sol.x[3]),
                          Rot.from_rotvec(sol.x[4:7]).as_matrix(), rms, int(idx.size), how)
