# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU lever arm from the hand-eye mounting sessions (validation plan VD15).

Shared method code: the validation stage ``ref_imu_lever`` (re-exported as
:mod:`biomech.validation.lever`) and the dataset stage ``camerafree`` (X5, E5/E6 lever arms
from the hand-eye session itself) use it; it lives outside ``biomech.validation`` so that the
dataset study's code hash holds no validation code (``pipeline.group_files``).

The hand-eye calibration (``jtsmm imu calibrate``) fits only the mounting rotation
``R_mount`` (body←chip). The accelerometer comparison and a position fusion that models the
IMU site also need the IMU position on its plate, the lever arm ``r`` (body frame, mm, from
the optical body origin). It is estimated from the same session, in which each plate is
tumbled by hand in front of the cameras while its IMU records:

* **model** (chip frame, gravity included, the accelerometer's specific force):
  ``f = R_cb R_wbᵀ (p̈_b + R̈_wb r − g_w) + b`` with ``R_cb = R_mountᵀ``, the optical body
  pose ``(R_wb, p_b)``, the unknown gravity vector ``g_w`` in the optical world (the camera
  frame is not levelled) and an accelerometer bias ``b``. It is linear in ``(r, g_w, b)``:
  ordinary least squares on all frames of the plate's own tumbling phase;
* **derivatives:** the optical position and the rotation-matrix entries are low-passed
  (4th-order Butterworth, zero phase, at ``fc``) on gap-free 40-Hz runs and differentiated
  twice by central second differences (error (ωh)²/12, < 1 % below 6 Hz; a Savitzky–Golay
  derivative attenuates the motion band and biased the lever arm by ≈ 13 % in the
  synthetic test); the accelerometer is low-passed at the same ``fc``. The lag between the
  streams (a few ms) is negligible below ``fc``;
* **cut-off:** ``fc_hz`` = 6 Hz (the dataset's pose low-pass), the sweep ``fc_sweep_hz``
  (4, 8 Hz) as a sensitivity. The cut-off must lie well above the angular-rate band: the
  rotation-matrix entries and their products with the accelerations carry harmonics, and a
  cut-off near the band biases the fit even without noise (synthetic test: exact from 4 Hz,
  +11 mm at 2 Hz). The first rule (the 95 % angular-rate band clipped to 2–6 Hz) failed this
  test and was replaced (VD15). The 95 % band is reported (``motion_f95_hz``);
* **uncertainty:** moving-block bootstrap over ``block_s`` blocks of frames; the OLS
  standard errors and the condition number are reported next to it;
* **checks:** ``|g_w|`` must come out near 9.81 m/s² (scale and model); the estimate must
  be stable over the cut-off sweep (``fc_spread_mm``: the largest per-axis range over the
  sweep; noise in the double-differentiated 40-Hz optics biases the fit as ``fc`` rises);
  and the lever arm of an IMU that was not remounted (``mount_change_deg`` between the
  sessions' ``R_mount``) must agree between sessions (repeatability).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

DEFAULTS: dict[str, Any] = {
    "reference": "reference.yaml",
    "bodies": {"mta": "MP", "cra": "HP"},     # IMU slot -> validation body name
    "fc_hz": 6.0,
    "fc_sweep_hz": [4.0, 8.0],
    "f_power": 0.95,
    "edge_s": 0.25,                 # frames dropped at both ends of each filtered run
    "min_run_s": 1.5,               # shortest gap-free optical run used
    "max_gap_s": 0.03,              # a larger step between optical frames ends a run
    "n_boot": 1000,
    "block_s": 1.0,
}
G_MM_S2 = 9806.65


def read_slot(path: str | Path, slot: str) -> dict[str, Any] | None:
    """Optical pose, accelerometer, quaternions, accepted ``R_mount`` and the tumbling phase
    of IMU ``slot`` (``mta`` / ``cra``) from a hand-eye session; None if any is missing."""
    with h5py.File(path, "r") as f:
        cal = f.get(f"calibration/imu_{slot}")
        if cal is None or not bool(cal.attrs.get("accepted", False)):
            return None
        phases = {p["name"]: (p["t_start_us"], p["t_end_us"])
                  for p in json.loads(f["phases"].attrs["phases_json"])}
        win = phases.get(f"imu_mount_{slot}")
        if win is None or f"optical/pose_{slot}" not in f or f"imu_{slot}/accel" not in f:
            return None
        g = f[f"imu_{slot}"]
        return {"t_opt": f["optical/t_master_us"][()].astype(float) * 1e-6,
                "T": f[f"optical/pose_{slot}"][()].astype(float),
                "t_acc": g["t_master_us_accel"][()].astype(float) * 1e-6,
                "acc": g["accel"][()].astype(float) * 1e3,          # m/s² -> mm/s²
                "t_q_us": g["t_master_us_quat"][()], "quat": g["quat"][()].astype(float),
                "R_mount": cal["mounting_transform"][()][:3, :3].astype(float),
                "window": (win[0] * 1e-6, win[1] * 1e-6),
                "date": str(cal.attrs.get("date", "")), "fit_rms_deg": float(cal.attrs.get(
                    "rms_deg", np.nan))}


def _clean(t: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    o = np.argsort(t, kind="stable")
    t, x = t[o], x[o]
    keep = np.r_[True, np.diff(t) > 0] & np.isfinite(x).all(axis=1)
    return t[keep], x[keep]


def motion_band_hz(d: dict[str, Any], f_power: float) -> float:
    """Frequency below which ``f_power`` of the IMU angular-rate power of the tumbling
    phase lies (angular velocity from consecutive quaternions on a 200-Hz grid)."""
    from jtsmm.imu.fusion import ImuTrack

    tr = ImuTrack.from_stream(d["t_q_us"], d["quat"], None, 0.05)
    grid = np.arange(*d["window"], 0.005)
    grid = grid[(grid >= tr.t_s[0]) & (grid <= tr.t_s[-1])]
    if len(grid) < 400:
        return np.nan
    R = tr.rot(grid)
    from scipy.spatial.transform import Rotation

    w = Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-1], R[1:])).as_rotvec() / 0.005
    f, P = welch(w, fs=200.0, nperseg=512, axis=0)
    c = np.cumsum(P.sum(axis=1))
    return float(f[np.searchsorted(c, f_power * c[-1])])


def _runs(ok: np.ndarray) -> list[tuple[int, int]]:
    i = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return list(zip(i[::2], i[1::2]))


def design(d: dict[str, Any], fc: float, p: dict[str, Any]
           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows ``M x = y`` of the lever-arm model (three per frame) and the frame times."""
    t_o, T = d["t_opt"], d["T"]
    fs = 1.0 / float(np.median(np.diff(t_o)))
    t_a, A = _clean(d["t_acc"], d["acc"])
    fs_a = 1.0 / float(np.median(np.diff(t_a)))
    A = sosfiltfilt(butter(4, fc, fs=fs_a, output="sos"), A, axis=0)
    sos = butter(4, fc, fs=fs, output="sos")
    ok = (np.isfinite(T[:, :3, :]).all(axis=(1, 2)) & (t_o >= d["window"][0])
          & (t_o <= d["window"][1]))
    edge = int(round(p["edge_s"] * fs))
    Ms, ys, ts = [], [], []
    Rc = d["R_mount"].T                                 # chip <- body
    for a, b in _runs(ok):
        t = t_o[a:b]
        if t[-1] - t[0] < p["min_run_s"] or np.max(np.diff(t)) > p["max_gap_s"]:
            continue
        pb = sosfiltfilt(sos, T[a:b, :3, 3], axis=0)
        Rf = sosfiltfilt(sos, T[a:b, :3, :3].reshape(-1, 9), axis=0).reshape(-1, 3, 3)
        dt = float(np.median(np.diff(t)))
        pdd, Rdd = np.full_like(pb, np.nan), np.full_like(Rf, np.nan)
        pdd[1:-1] = (pb[2:] - 2 * pb[1:-1] + pb[:-2]) / dt ** 2
        Rdd[1:-1] = (Rf[2:] - 2 * Rf[1:-1] + Rf[:-2]) / dt ** 2
        U, _, Vt = np.linalg.svd(Rf)
        R = U @ Vt                                      # nearest rotation (frame change)
        fa = np.column_stack([np.interp(t, t_a, A[:, j]) for j in range(3)])
        for i in range(edge, len(t) - edge):
            Rcw = Rc @ R[i].T
            Ms.append(np.hstack([Rcw @ Rdd[i], -Rcw, np.eye(3)]))
            ys.append(fa[i] - Rcw @ pdd[i])
            ts.append(t[i])
    if not Ms:
        return np.zeros((0, 9)), np.zeros(0), np.zeros(0)
    return np.concatenate(Ms), np.concatenate(ys), np.asarray(ts)


def fit(M: np.ndarray, y: np.ndarray, t: np.ndarray, p: dict[str, Any], seed: int
        ) -> dict[str, Any]:
    """OLS estimate of ``(r, g_w, b)`` with SEs, block-bootstrap CIs of ``r`` and checks."""
    out: dict[str, Any] = {"n_frames": int(len(t))}
    if len(t) < 50:
        return out
    x, *_ = np.linalg.lstsq(M, y, rcond=None)
    res = y - M @ x
    s2 = float(res @ res) / (len(y) - M.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(M.T @ M)))
    blk = np.floor((t - t[0]) / p["block_s"]).astype(int)
    rows_of = {b: np.flatnonzero(np.repeat(blk, 3) == b) for b in np.unique(blk)}
    keys = np.array(list(rows_of))
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(int(p["n_boot"])):
        idx = np.concatenate([rows_of[k] for k in rng.choice(keys, len(keys))])
        xb, *_ = np.linalg.lstsq(M[idx], y[idx], rcond=None)
        boot.append(xb[:3])
    lo, hi = (np.percentile(np.asarray(boot), [2.5, 97.5], axis=0) if boot
              else (np.full(3, np.nan), np.full(3, np.nan)))      # n_boot = 0: point estimate
    for j, ax in enumerate("xyz"):
        out.update({f"r_{ax}_mm": float(x[j]), f"r_{ax}_se_mm": float(se[j]),
                    f"r_{ax}_ci_lo_mm": float(lo[j]), f"r_{ax}_ci_hi_mm": float(hi[j]),
                    f"bias_{ax}_mm_s2": float(x[6 + j])})
    out.update(r_norm_mm=float(np.linalg.norm(x[:3])), g_norm_mm_s2=float(np.linalg.norm(x[3:6])),
               g_ratio=float(np.linalg.norm(x[3:6]) / G_MM_S2),
               res_rms_mm_s2=float(np.sqrt(np.mean(res ** 2))), cond=float(np.linalg.cond(M)))
    return out


def run_ref_imu_lever(doc: dict[str, Any], data_dir: Path, p: dict[str, Any], seed: int = 0
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Lever arm per hand-eye session (``reference.yaml: imu_mounting_sessions``), IMU slot
    and cut-off (the primary ``fc`` and the sweep), and the between-session differences of
    the primary estimates per slot."""
    from scipy.spatial.transform import Rotation

    rows: list[dict[str, Any]] = []
    mounts: dict[tuple[str, str], np.ndarray] = {}
    for rel in doc.get("imu_mounting_sessions") or []:
        path = Path(data_dir) / str(rel)
        if not path.exists():
            continue
        for slot, body in p["bodies"].items():
            d = read_slot(path, slot)
            if d is None:
                continue
            mounts[(Path(rel).parent.name, slot)] = d["R_mount"]
            f95 = motion_band_hz(d, p["f_power"])
            fc0 = float(p["fc_hz"])
            for fc in [fc0] + [float(v) for v in p["fc_sweep_hz"] if abs(v - fc0) > 1e-9]:
                M, y, t = design(d, fc, p)
                rows.append({"session": Path(rel).parent.name, "date": d["date"],
                             "slot": slot, "body": body, "fc_hz": fc, "primary": fc == fc0,
                             "motion_f95_hz": f95, "mount_fit_rms_deg": d["fit_rms_deg"],
                             **fit(M, y, t, p, seed)})
    lev = pd.DataFrame(rows)
    rep: list[dict[str, Any]] = []
    if not lev.empty and "r_x_mm" in lev:
        cols = [f"r_{ax}_mm" for ax in "xyz"]
        spread = lev.groupby(["session", "slot"])[cols].agg(lambda v: v.max() - v.min())
        lev["fc_spread_mm"] = [float(spread.loc[(s, sl)].max())
                               for s, sl in zip(lev.session, lev.slot)]
        prim = lev[lev.primary].sort_values("date")
        for slot, g in prim.groupby("slot", sort=False):
            if len(g) < 2:
                continue
            a, b = g.iloc[0], g.iloc[-1]
            d = np.array([b[f"r_{ax}_mm"] - a[f"r_{ax}_mm"] for ax in "xyz"])
            rep.append({"slot": slot, "body": a["body"], "session_a": a["session"],
                        "session_b": b["session"], **{f"diff_{ax}_mm": float(v)
                                                       for ax, v in zip("xyz", d)},
                        "diff_norm_mm": float(np.linalg.norm(d)),
                        "mount_change_deg": float(np.degrees(Rotation.from_matrix(
                            mounts[(a["session"], slot)].T @ mounts[(b["session"], slot)])
                            .magnitude()))})
    return {"lever": lev, "repeatability": pd.DataFrame(rep)}, {}
