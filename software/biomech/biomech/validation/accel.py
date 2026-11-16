# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU accelerometer against Qualisys at the IMU site (validation plan VD17).

The reference is the Qualisys acceleration of the IMU site: the 200-Hz body pose ``T_Q,b``
carries the lever arm from the hand-eye session (VD15, JTS-MM body frame ``r_J``) into the
Qualisys body frame, ``r_Q = Y_b⁻¹ r_J`` (registration ``T_Q,b = X T_S,b Y_b``, plan D3), so
the site is ``p(t) = T_Q,b(t) r_Q`` and its acceleration ``p̈`` (low-pass ``fc`` on gap-free
runs, central second differences).

**Model** (chip frame, gravity included, the accelerometer's specific force)::

    f(t + lag) = R · R_Q,b(t)ᵀ (p̈(t) − g_Q) + b

with the constant rotation ``R`` (chip ← Qualisys body), the gravity vector ``g_Q`` of the
Qualisys world, the accelerometer bias ``b`` and the IMU–Qualisys ``lag``. A per-axis scale
(``fit_scale``, off) is a sensitivity only: with the little tilt of the jaw tasks it trades
off against bias, gravity and rotation (first real-data run: scales 0.63–1.12, while the
uncalibrated accelerometer reads 9.65–9.89 m/s² at rest, i.e. within ≈ 1.6 % of g). They are fitted on
the **other** held intervals of the take (leave-one-task-out, as every validation fit), never
on the scored task: alternating least squares (``g_Q, b`` linear given ``R``; ``R`` by Kabsch
given ``g_Q, b``), started from the prior ``R_mount,Jᵀ R_Y`` (hand-eye mounting and
registration), on a lag grid, then a joint refinement at the best lag. ``|g_Q|`` is held at
``g_mm_s2`` (only its direction is fitted): when a plate tilts little, a free gravity
magnitude trades off against the bias (first real-data run: 6.9–10.7 m/s²).

**Outcomes per held-out task** (dynamic = gravity removed, chip frame):

* ``sensor``: measured minus predicted specific force; it isolates the accelerometer, the
  gravity term coming from the reference orientation;
* ``own_orientation``: the dynamic acceleration a wearable obtains alone, ``f − b − R_imuᵀ ĝ``
  with the IMU's own (game rotation vector) orientation and ``ĝ`` the training-frame mean of
  ``R_imu f``; the difference to ``sensor`` is the gravity leak of the IMU's tilt error
  (≈ 171 mm/s² per degree);
* ``no_lever``: ``sensor`` with the lever arm set to zero (refitted): if the VD15 lever arm
  is right, it lowers the error. This checks VD15 on independent data.

Per outcome: RMS error, RMS of the reference dynamic acceleration (scale), the variance of
the reference dynamic acceleration explained (``r2_dyn``) and the band-mean
magnitude-squared coherence (0.5–``fc`` Hz, 3-D: summed cross and auto spectra), plus the
rest-phase RMS of both dynamic accelerations as their noise floors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import butter, csd, sosfiltfilt, welch
from scipy.spatial.transform import Rotation

from biomech.imu.preprocess import GAP_S as IMU_GAP_S

DEFAULTS: dict[str, Any] = {
    "fc_hz": 6.0,                       # the dataset's pose low-pass (VD15 uses the same)
    "fc_sweep_hz": [4.0, 10.0],
    "lag_grid_ms": [-40.0, 80.0, 5.0],  # start, stop, step
    "n_iter": 15,
    "fit_rate_hz": 50.0,                # training frames thinned to this rate for the fit
    "g_mm_s2": 9806.65,                 # gravity magnitude held fixed
    "fit_scale": False,                 # per-axis accelerometer scale (sensitivity only)
    "min_run_s": 1.0,
    "coh_band_lo_hz": 0.5,
    "coh_nperseg": 256,
}


def _runs(ok: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    i = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [(a, b) for a, b in zip(i[::2], i[1::2]) if b - a >= min_len]


def site_acceleration(t: np.ndarray, T: np.ndarray, r: np.ndarray, fc: float,
                      min_run_s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """World acceleration (mm/s²) of the body point ``r`` (body frame, mm) and the body
    rotation, at the pose times ``t``; NaN outside gap-free runs of ≥ ``min_run_s``."""
    t = np.asarray(t, float)
    T = np.asarray(T, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    p = T[:, :3, 3] + np.einsum("nij,j->ni", T[:, :3, :3], np.asarray(r, float))
    ok = np.isfinite(p).all(axis=1) & np.isfinite(T[:, :3, :3]).all(axis=(1, 2))
    gap_ok = np.r_[True, np.diff(t) < 1.5 / fs]
    a = np.full_like(p, np.nan)
    sos = butter(4, fc, fs=fs, output="sos")
    for i0, i1 in _runs(ok & gap_ok, int(min_run_s * fs)):
        pl = sosfiltfilt(sos, p[i0:i1], axis=0)
        a[i0 + 1:i1 - 1] = (pl[2:] - 2 * pl[1:-1] + pl[:-2]) * fs ** 2
    R = np.where(ok[:, None, None], T[:, :3, :3], np.nan)
    return a, R


def accel_at(t_query: np.ndarray, t_acc: np.ndarray, acc: np.ndarray, fc: float
             ) -> np.ndarray:
    """Accelerometer (any unit) low-passed at ``fc`` and interpolated at ``t_query``."""
    o = np.argsort(t_acc, kind="stable")
    t_a, A = np.asarray(t_acc, float)[o], np.asarray(acc, float)[o]
    keep = np.r_[True, np.diff(t_a) > 0] & np.isfinite(A).all(axis=1)
    t_a, A = t_a[keep], A[keep]
    fs = 1.0 / float(np.median(np.diff(t_a)))
    A = sosfiltfilt(butter(4, fc, fs=fs, output="sos"), A, axis=0)
    return np.column_stack([np.interp(t_query, t_a, A[:, j], left=np.nan, right=np.nan)
                            for j in range(3)])


def _kabsch(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotation R minimising Σ‖v − R u‖²."""
    H = u.T @ v
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def fit(a: np.ndarray, R_Q: np.ndarray, f: np.ndarray, R0: np.ndarray, n_iter: int = 15,
        G: float = 9806.65, refine: bool = True, fit_scale: bool = False
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """``R, g_Q, b, s`` of ``f = diag(s) R R_Qᵀ (a − g_Q) + b`` with ``|g_Q| = G`` on the
    frames given, and the residual RMS; ``s = 1`` unless ``fit_scale``. The alternation
    starts it; ``refine`` adds the joint least squares (the lag scan uses the alternation
    alone)."""
    R = np.asarray(R0, float)
    g = np.zeros(3)
    b = np.zeros(3)
    sc = np.ones(3)
    n = len(a)
    for _ in range(n_iter):
        # g, b given R: f − R R_Qᵀ a = −R R_Qᵀ g + b, then |g| = G and b re-estimated
        M = np.einsum("ij,nkj->nik", R, R_Q)                 # R R_Qᵀ per frame
        y = (f - np.einsum("nij,nj->ni", M, a)).reshape(-1)
        A = np.concatenate([-M, np.tile(np.eye(3), (n, 1, 1))], axis=2).reshape(-1, 6)
        x, *_ = np.linalg.lstsq(A, y, rcond=None)
        g = G * x[:3] / max(np.linalg.norm(x[:3]), 1e-9)
        b = np.mean(f - np.einsum("nij,nj->ni", M, a - g), axis=0)
        # R given g, b
        u = np.einsum("nji,nj->ni", R_Q, a - g)              # R_Qᵀ (a − g)
        R = _kabsch(u, f - b)
    if refine:
        # joint refinement (the alternation converges slowly when the body tilts little);
        # g = Rot(δ) g0 keeps |g| = G; s = exp(σ)
        from scipy.optimize import least_squares

        u_a = np.einsum("nji,nj->ni", R_Q, a)                # R_Qᵀ a
        g0 = g.copy()

        def resid(x: np.ndarray) -> np.ndarray:
            Rx = Rotation.from_rotvec(x[:3]).as_matrix() @ R
            gb = np.einsum("nji,j->ni", R_Q, Rotation.from_rotvec(x[3:6]).apply(g0))
            return (f - np.exp(x[9:12]) * ((u_a - gb) @ Rx.T) - x[6:9]).reshape(-1)

        n_par = 12 if fit_scale else 9

        def resid_n(x: np.ndarray) -> np.ndarray:
            return resid(np.r_[x, np.zeros(12 - len(x))])

        sol = least_squares(resid_n, np.r_[np.zeros(6), b, np.zeros(n_par - 9)], method="lm")
        x = np.r_[sol.x, np.zeros(12 - n_par)]
        R = Rotation.from_rotvec(x[:3]).as_matrix() @ R
        g, b, sc = Rotation.from_rotvec(x[3:6]).apply(g0), x[6:9], np.exp(x[9:12])
    res = f - sc * np.einsum("ij,nkj,nk->ni", R, R_Q, a - g) - b
    return R, g, b, sc, float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))


def _coherence(x: np.ndarray, y: np.ndarray, fs: float, lo: float, hi: float,
               nperseg: int, min_len: int) -> float:
    """Band-mean 3-D magnitude-squared coherence over the finite runs of both signals."""
    ok = np.isfinite(x).all(axis=1) & np.isfinite(y).all(axis=1)
    Sxy = Sxx = Syy = None
    for i0, i1 in _runs(ok, max(min_len, nperseg)):
        f_, pxy = csd(x[i0:i1], y[i0:i1], fs=fs, nperseg=nperseg, axis=0)
        _, pxx = welch(x[i0:i1], fs=fs, nperseg=nperseg, axis=0)
        _, pyy = welch(y[i0:i1], fs=fs, nperseg=nperseg, axis=0)
        w = i1 - i0
        Sxy = w * pxy.sum(axis=1) if Sxy is None else Sxy + w * pxy.sum(axis=1)
        Sxx = w * pxx.sum(axis=1) if Sxx is None else Sxx + w * pxx.sum(axis=1)
        Syy = w * pyy.sum(axis=1) if Syy is None else Syy + w * pyy.sum(axis=1)
        freqs = f_
    if Sxy is None:
        return np.nan
    msc = np.abs(Sxy) ** 2 / (Sxx * Syy)
    k = (freqs >= lo) & (freqs <= hi)
    return float(np.mean(msc[k])) if k.any() else np.nan


def _rms(x: np.ndarray) -> float:
    n = np.sum(x ** 2, axis=1)
    n = n[np.isfinite(n)]
    return float(np.sqrt(np.mean(n))) if n.size else np.nan


def evaluate_body(t_q: np.ndarray, T_Q: np.ndarray, r_Q: np.ndarray, t_acc: np.ndarray,
                  acc_mm: np.ndarray, R_imu_at: Any, R0: np.ndarray, folds: list,
                  rest: list, p: dict[str, Any], fc: float
                  ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray]]:
    """Leave-one-task-out comparison of one body's accelerometer with Qualisys.

    ``folds``: the scored intervals (``.name``, ``.t0``, ``.t1``); each is scored with
    ``R, g_Q, b, lag`` fitted on the other folds and the ``rest`` intervals. ``R_imu_at(t)``
    gives the IMU's own orientation (chip → IMU world) or None. Returns the outcome rows,
    the calibration rows and the dynamic accelerations (reference and measured, chip frame)
    of the scored frames."""
    t_q = np.asarray(t_q, float)
    fs = 1.0 / float(np.median(np.diff(t_q)))
    lags = np.arange(*p["lag_grid_ms"]) * 1e-3
    a, R_Q = site_acceleration(t_q, T_Q, r_Q, fc, p["min_run_s"])
    a0, _ = site_acceleration(t_q, T_Q, np.zeros(3), fc, p["min_run_s"])
    def mask(ivs):
        return (np.any([(t_q >= iv.t0) & (t_q <= iv.t1) for iv in ivs], axis=0) if ivs
                else np.zeros(len(t_q), bool))
    f_lag = {lag: accel_at(t_q + lag, t_acc, acc_mm, fc) for lag in lags}
    rows, cal_rows = [], []
    out = {"dyn_ref": np.full((len(t_q), 3), np.nan), "dyn_meas": np.full((len(t_q), 3), np.nan),
           "dyn_own": np.full((len(t_q), 3), np.nan)}
    k_rest = mask(rest)
    stride = max(1, int(round(fs / float(p["fit_rate_hz"]))))
    for h in folds:
        k_h = mask([h])
        k_tr = mask([g for g in folds if g is not h] + list(rest))
        best = None
        for variant, acc_site in (("sensor", a), ("no_lever", a0)):
            fits = []
            for lag in lags:
                ok = np.flatnonzero(k_tr & np.isfinite(acc_site).all(axis=1)
                                    & np.isfinite(f_lag[lag]).all(axis=1))[::stride]
                if ok.size < 100:
                    continue
                *_, rms = fit(acc_site[ok], R_Q[ok], f_lag[lag][ok], R0, p["n_iter"],
                              float(p["g_mm_s2"]), refine=False)
                fits.append((rms, lag, ok))
            if not fits:
                continue
            _, lag, ok_tr = min(fits, key=lambda z: z[0])
            R, g, b, sc, rms_tr = fit(acc_site[ok_tr], R_Q[ok_tr], f_lag[lag][ok_tr], R0,
                                      p["n_iter"], float(p["g_mm_s2"]),
                                      fit_scale=bool(p["fit_scale"]))
            f = f_lag[lag]
            ok = k_h & np.isfinite(acc_site).all(axis=1) & np.isfinite(f).all(axis=1)
            M = np.einsum("ij,nkj->nik", R, R_Q)
            dyn_ref = np.einsum("nij,nj->ni", M, acc_site)
            grav = -np.einsum("nij,j->ni", M, g)
            dyn_meas = (f - b) / sc - grav
            row = {"body_fold": h.name, "variant": variant}
            rows.append({**row, **_scores(dyn_meas, dyn_ref, ok, k_rest, fs, fc, p)})
            cal_rows.append({"fold": h.name, "variant": variant, "lag_ms": lag * 1e3,
                             "fit_rms_mm_s2": rms_tr, "g_norm_mm_s2": float(np.linalg.norm(g)),
                             **{f"bias_{ax}_mm_s2": float(v) for ax, v in zip("xyz", b)},
                             **{f"scale_{ax}": float(v) for ax, v in zip("xyz", sc)},
                             "prior_vs_fit_deg": float(np.degrees(Rotation.from_matrix(
                                 R0.T @ R).magnitude()))})
            if variant == "sensor":
                best = (lag, R, g, b, sc, dyn_ref, ok)
                out["dyn_ref"][ok], out["dyn_meas"][ok] = dyn_ref[ok], dyn_meas[ok]
        if best is None or R_imu_at is None:
            continue
        lag, R, g, b, sc, dyn_ref, ok = best
        Ri = R_imu_at(t_q + lag)                              # chip -> IMU world
        f = (f_lag[lag] - b) / sc + b                         # scale-corrected
        tr = k_tr & np.isfinite(f).all(axis=1) & np.isfinite(Ri).all(axis=(1, 2))
        g_w = np.mean(np.einsum("nij,nj->ni", Ri[tr], f[tr] - b), axis=0)
        dyn_own = f - b - np.einsum("nji,j->ni", Ri, g_w)
        ok_o = ok & np.isfinite(dyn_own).all(axis=1)
        rows.append({"body_fold": h.name, "variant": "own_orientation",
                     **_scores(dyn_own, dyn_ref, ok_o, k_rest, fs, fc, p)})
        out["dyn_own"][ok_o] = dyn_own[ok_o]
    return rows, cal_rows, out


def _scores(meas: np.ndarray, ref: np.ndarray, ok: np.ndarray, k_rest: np.ndarray,
            fs: float, fc: float, p: dict[str, Any]) -> dict[str, Any]:
    m = np.where(ok[:, None], meas, np.nan)
    r = np.where(ok[:, None], ref, np.nan)
    e = m - r
    ss_ref = np.nansum((r - np.nanmean(r, axis=0)) ** 2)
    return {"n": int(ok.sum()), "rms_err_mm_s2": _rms(e), "rms_ref_mm_s2": _rms(r),
            "r2_dyn": float(1 - np.nansum(e ** 2) / ss_ref) if ss_ref > 0 else np.nan,
            "coherence": _coherence(m, r, fs, p["coh_band_lo_hz"], fc, p["coh_nperseg"],
                                    int(p["min_run_s"] * fs)),
            "rest_rms_meas_mm_s2": _rms(np.where(k_rest[:, None], meas, np.nan)),
            "rest_rms_ref_mm_s2": _rms(np.where(k_rest[:, None], ref, np.nan))}


def run_ref_imu_accel(sy: Any, session: Any, mounting: dict[str, np.ndarray],
                      levers: dict[str, np.ndarray], Y: dict[str, np.ndarray],
                      slots: dict[str, str], params: dict[str, Any] | None = None
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Per body (``slots``: validation body -> IMU slot) with a hand-eye mounting and lever
    arm: the leave-one-task-out comparison at the primary cut-off and the sweep."""
    from jtsmm.imu.fusion import ImuTrack

    from biomech.validation import selection as SEL

    p = {**DEFAULTS, **(params or {})}
    folds = [h for h in sy.held if h.name in SEL.MOVEMENT_TASKS]
    rest = [h for h in sy.held if h.name in SEL.STATIC_TASKS]
    rows, cal_rows, arrays = [], [], {"t_q_master_s": np.asarray(sy.t_qm, float)}
    for body, slot in slots.items():
        if body not in mounting or body not in levers or body not in Y:
            continue
        imu = session.imu(slot)
        acc = getattr(imu, "accel", None) if imu is not None else None
        if acc is None or len(acc.t_us) < 100:
            continue
        track = ImuTrack.from_stream(np.asarray(imu.t_us), np.asarray(imu.quat, np.float64),
                                     getattr(imu, "quat_accuracy", None), IMU_GAP_S)

        def R_imu_at(t: np.ndarray, track=track) -> np.ndarray:
            out = np.full((len(t), 3, 3), np.nan)
            seg = track.segment(t)
            ok = seg >= 0
            out[ok] = track.rot(t[ok])
            return out

        Yb = np.asarray(Y[body], float)
        r_Q = np.linalg.solve(Yb, np.r_[levers[body], 1.0])[:3]     # Y⁻¹ r_J
        R0 = mounting[body].T @ Yb[:3, :3]                          # chip <- J body <- Q body
        T_Q = sy.qtm.pose(body)
        t_acc = np.asarray(acc.t_us, float) * 1e-6
        acc_mm = np.asarray(acc.data, float) * 1e3
        for fc in [float(p["fc_hz"])] + [float(v) for v in p["fc_sweep_hz"]]:
            r_, c_, out = evaluate_body(sy.t_qm, T_Q, r_Q, t_acc, acc_mm, R_imu_at, R0, folds,
                                        rest, p, fc)
            primary = fc == float(p["fc_hz"])
            base = {"pid": sy.pid, "body": body, "fc_hz": fc, "primary": primary}
            rows += [{**base, "task": x.pop("body_fold"), **x} for x in r_]
            cal_rows += [{**base, **x} for x in c_]
            if primary:
                for k, v in out.items():
                    arrays[f"{body}__{k}_mm_s2"] = v
    return {"accel": pd.DataFrame(rows), "calibration": pd.DataFrame(cal_rows)}, arrays
