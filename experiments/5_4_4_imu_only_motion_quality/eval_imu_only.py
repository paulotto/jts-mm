#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""IMU-only motion quality: naive double integration vs orientation.

Most pilot takes only recorded the cranial IMU. When both slots exist, relative
mandible kinematics are used; otherwise CRA vs optical HP (pose_cra).

Position: trapezoidal double integration of linear acceleration with **no**
ZUPT / periodic reset. Drift is compared to optical translation on 0.5/1/2 s
windows, comparing displacement magnitudes ``|‖Δp_IMU‖ − ‖Δp_opt‖|`` (yaw-
invariant). Orientation: geodesic of IMU vs optical *attitude change from t0*
(after ``R_mount``), so a constant world/yaw offset does not enter the RMS.

``R_mount`` per slot comes from the newest ``imu_mounting.json`` of
``5_2_2_imu_optical_handeye`` under
``--mounting-dir`` (default: ``session.imu_calibration_recordings_dir`` of the
operator config, i.e. ``~/experiments/5_2_2_imu_optical_handeye/recordings``),
found with ``jtsmm.imu.persist.discover_runs`` / ``load_run``.

Usage
-----
  python eval_imu_only.py
  python eval_imu_only.py --data-dir ~/jtsmm_data --subject s01 --mounting-dir /path/to/recordings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from biomech.io.inventory import TakeInventory, scan, subject_pids
from biomech.io.session import SessionData
from biomech.paths import resolve_data_dir, results_dir
from biomech.results.provenance import write_csv, write_json, write_provenance

STUDY = "5_4_4_imu_only_motion_quality"
WINDOWS_S = (0.5, 1.0, 2.0)
EARLY_S = 8.0


def percentile(x: np.ndarray, q: float) -> float:
    v = np.asarray(x, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    return float(np.percentile(v, q)) if v.size else float("nan")


def default_mounting_dir() -> Path:
    """``session.imu_calibration_recordings_dir`` of the operator config, else the layout default.

    A ``{experiment_name}`` placeholder is cut off: ``discover_runs`` searches below it.
    """
    from jtsmm.io.session_layout import IMU_CALIB_RECORDINGS_DEFAULT

    raw = None
    try:
        from jtsmm.config.loader import default_config_path, load_yaml
        from jtsmm.config.registry import session_imu_calibration_recordings_dir

        raw = session_imu_calibration_recordings_dir(load_yaml(default_config_path()))
    except Exception:  # noqa: BLE001 - no usable operator config on an analysis host
        raw = None
    text = str(raw or IMU_CALIB_RECORDINGS_DEFAULT).split("{", 1)[0]
    return Path(text).expanduser()


def load_latest_mounting(slot: str, recordings_dir: Path) -> tuple[np.ndarray | None, str | None]:
    """``(R_mount, source path)`` of ``slot`` from the newest ``imu_mounting.json``.

    The newest file is the last one by path (leaf names carry the UTC stamp). A
    run that missed the 2° acceptance is still used: it is the calibration the
    rig had.
    """
    from jtsmm.imu.persist import discover_runs, load_run

    runs = discover_runs(recordings_dir)
    if not runs:
        return None, None
    doc = load_run(runs[-1])
    R = ((doc.get("slots") or {}).get(slot) or {}).get("R_mount")
    if R is None:
        return None, str(runs[-1])
    arr = np.asarray(R, dtype=np.float64)
    return (arr if arr.shape == (3, 3) else None), str(runs[-1])


def _at_time(t: np.ndarray, y: np.ndarray, t_query: float) -> float | None:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size == 0:
        return None
    i = int(np.clip(np.searchsorted(t, t_query), 0, len(y) - 1))
    v = y[i]
    return float(v) if np.isfinite(v) else None


def _rms_until(t: np.ndarray, y: np.ndarray, t_end: float) -> float | None:
    m = (np.asarray(t) <= t_end) & np.isfinite(y)
    if not np.any(m):
        return None
    return float(np.sqrt(np.mean(np.asarray(y)[m] ** 2)))


def _quat_interp(t_src: np.ndarray, q: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Nlerp onto ``t_dst`` (searchsorted; hemisphere-aware)."""
    t_src = np.asarray(t_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if t_src.size == 0 or t_dst.size == 0:
        return np.full((len(t_dst), 4), np.nan)
    idx = np.clip(np.searchsorted(t_src, t_dst, side="left"), 1, len(t_src) - 1)
    t0 = t_src[idx - 1]
    t1 = t_src[idx]
    dt = t1 - t0
    u = np.divide(t_dst - t0, dt, out=np.zeros_like(t_dst), where=dt > 0)
    u = np.clip(u, 0.0, 1.0)[:, None]
    q0 = q[idx - 1]
    q1 = q[idx]
    q1 = np.where(np.sum(q0 * q1, axis=1, keepdims=True) < 0.0, -q1, q1)
    out = (1.0 - u) * q0 + u * q1
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def _apply_mount(q: np.ndarray, R: np.ndarray | None) -> np.ndarray:
    if R is None:
        return np.asarray(q, dtype=np.float64)
    from jtsmm.imu.kinematics import quat_from_rotation_matrix

    q_m = np.asarray(quat_from_rotation_matrix(R), dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # q_body = q_mount ⊗ q_chip  (Hamilton, vectorised)
    w1, x1, y1, z1 = q_m
    w2, x2, y2, z2 = q.T
    out = np.column_stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def geodesic_deg(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    d = np.abs(np.sum(q1 * q2, axis=1))
    d = np.clip(d, 0.0, 1.0)
    return 2.0 * np.degrees(np.arccos(d))


def _quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product; ``a`` is (4,) or (N, 4), ``b`` is (N, 4)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim == 1:
        a = np.broadcast_to(a, b.shape)
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.column_stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def attitude_change_from_t0(q: np.ndarray) -> np.ndarray:
    """``q_Δ(t) = conj(q(t0)) ⊗ q(t)`` — strips a constant world/yaw offset."""
    q = np.asarray(q, dtype=np.float64)
    ok = np.isfinite(q).all(axis=1)
    if not ok.any():
        return np.full_like(q, np.nan)
    q0 = q[int(np.flatnonzero(ok)[0])]
    conj0 = np.array([q0[0], -q0[1], -q0[2], -q0[3]], dtype=np.float64)
    out = _quat_multiply(conj0, q)
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-12)


def orientation_error_from_t0(q_imu: np.ndarray, q_opt: np.ndarray) -> np.ndarray:
    """Geodesic angle between IMU and optical *motion since t0* (deg)."""
    return geodesic_deg(attitude_change_from_t0(q_imu), attitude_change_from_t0(q_opt))


def R_to_quat(R: np.ndarray) -> np.ndarray:
    from jtsmm.imu.kinematics import quat_from_rotation_matrix

    return np.asarray(quat_from_rotation_matrix(R), dtype=np.float64)


def naive_double_integrate(t_s: np.ndarray, a: np.ndarray) -> np.ndarray:
    p = np.zeros_like(a, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    last = t_s[0]
    for i in range(1, len(t_s)):
        dt = float(t_s[i] - last)
        last = t_s[i]
        if dt <= 0.0 or dt > 0.5 or not np.isfinite(a[i]).all():
            p[i] = p[i - 1]
            continue
        v = v + 0.5 * (a[i] + a[i - 1]) * dt
        p[i] = p[i - 1] + v * dt  # v already updated; close enough for short windows
    return p


def load_imu_slot(h5: Path, slot: str) -> dict[str, Any] | None:
    import h5py

    with h5py.File(h5, "r") as f:
        if slot not in f:
            return None
        g = f[slot]
        if "quat" not in g:
            return None
        t_q = np.asarray(g["t_master_us_quat"], dtype=np.int64)
        q = np.asarray(g["quat"], dtype=np.float64)
        if "linear_accel" in g:
            t_a = np.asarray(g["t_master_us_linear_accel"], dtype=np.int64)
            a = np.asarray(g["linear_accel"], dtype=np.float64)
        elif "accel" in g:
            # The pilot recorded the raw accelerometer *including* gravity; remove it with
            # the game-rotation-vector orientation at each accelerometer sample.
            from jtsmm.imu.batch import quat_resample, quat_to_matrix
            from jtsmm.imu.gravity_lin_accel import G

            t_a = np.asarray(g["t_master_us_accel"], dtype=np.int64)
            raw = np.asarray(g["accel"], dtype=np.float64)
            R = quat_to_matrix(quat_resample(t_q, q, t_a, 50_000))        # world <- body
            a = raw - np.einsum("nji,j->ni", R, np.array([0.0, 0.0, G]))  # R^T (0, 0, g)
        else:
            t_a = a = None
        return {"t_q": t_q, "q": q, "t_a": t_a, "a": a}


def load_optical(h5: Path) -> dict[str, Any] | None:
    import h5py

    with h5py.File(h5, "r") as f:
        if "optical" not in f:
            return None
        g = f["optical"]
        return {
            "t": np.asarray(g["t_master_us"], dtype=np.int64),
            "mta": np.asarray(g["pose_mta"], dtype=np.float64) if "pose_mta" in g else None,
            "cra": np.asarray(g["pose_cra"], dtype=np.float64) if "pose_cra" in g else None,
        }


def displacement_mag_err(p_imu: np.ndarray, p_opt: np.ndarray, p0_imu: np.ndarray, p0_opt: np.ndarray) -> np.ndarray:
    """``|‖p_IMU − p0‖ − ‖p_opt − p0‖|`` — yaw-invariant path-length residual (m)."""
    di = np.linalg.norm(p_imu - p0_imu, axis=1)
    do = np.linalg.norm(p_opt - p0_opt, axis=1)
    return np.abs(di - do)


def window_drift(t_s: np.ndarray, p_imu: np.ndarray, p_opt: np.ndarray, win_s: float) -> list[float]:
    errs = []
    i0 = 0
    n = len(t_s)
    while i0 < n:
        t1 = t_s[i0] + win_s
        i1 = int(np.searchsorted(t_s, t1))
        if i1 <= i0 + 4:
            i0 += 1
            continue
        di = p_imu[i1 - 1] - p_imu[i0]
        do = p_opt[i1 - 1] - p_opt[i0]
        if np.isfinite(di).all() and np.isfinite(do).all():
            errs.append(float(abs(np.linalg.norm(di) - np.linalg.norm(do))))
        i0 = i1
    return errs


def eval_session(rec: TakeInventory, mounting_dir: Path) -> dict[str, Any]:
    from jtsmm.imu.kinematics import relative_linear_accel, relative_quaternion

    mta = load_imu_slot(rec.h5, "imu_mta")
    cra = load_imu_slot(rec.h5, "imu_cra")
    opt = load_optical(rec.h5)
    mode = "relative" if mta and cra else ("cra_only" if cra else None)
    if mode is None or opt is None:
        return {"session": rec.name, "subject_id": rec.subject_id, "mode": mode, "error": "missing imu/optical"}

    R_mta, mount_src = load_latest_mounting("imu_mta", mounting_dir)
    R_cra, _ = load_latest_mounting("imu_cra", mounting_dir)
    t_opt = opt["t"]
    pose = opt["mta"] if mode == "relative" else opt["cra"]
    ref = opt["cra"] if mode == "relative" else None
    finite = np.isfinite(pose).reshape(len(pose), -1).all(axis=1)
    if ref is not None:
        finite &= np.isfinite(ref).reshape(len(ref), -1).all(axis=1)
    t_opt = t_opt[finite]
    pose = pose[finite]
    if ref is not None:
        ref = ref[finite]
    if t_opt.size < 10:
        return {"session": rec.name, "mode": mode, "error": "too few optical samples"}

    q_opt = np.vstack([R_to_quat(P[:3, :3]) for P in pose])
    if mode == "relative":
        q_rel_opt = []
        for Pm, Pc in zip(pose, ref):
            Rrel = Pc[:3, :3].T @ Pm[:3, :3]
            q_rel_opt.append(R_to_quat(Rrel))
        q_opt = np.vstack(q_rel_opt)
        p_opt = np.vstack([(Pc[:3, :3].T @ (Pm[:3, 3] - Pc[:3, 3])) for Pm, Pc in zip(pose, ref)])
        # mm → m for comparison with accel integration (m/s^2)
        if np.nanmax(np.abs(p_opt)) > 2.0:
            p_opt = p_opt / 1000.0
        q_mta = _apply_mount(_quat_interp(mta["t_q"], mta["q"], t_opt), R_mta)
        q_cra = _apply_mount(_quat_interp(cra["t_q"], cra["q"], t_opt), R_cra)
        q_imu = np.empty((len(q_mta), 4), dtype=np.float64)
        for i in range(len(q_mta)):
            q_imu[i] = relative_quaternion(tuple(q_mta[i]), tuple(q_cra[i]))
        if mta["a"] is None or cra["a"] is None:
            p_imu = np.full_like(p_opt, np.nan)
            t_a = t_opt.astype(np.float64) / 1e6
        else:
            t_a = mta["t_a"].astype(np.float64) / 1e6
            a_rel = []
            qma = _apply_mount(_quat_interp(mta["t_q"], mta["q"], mta["t_a"]), R_mta)
            qca = _apply_mount(_quat_interp(cra["t_q"], cra["q"], mta["t_a"]), R_cra)
            ac = np.zeros_like(mta["a"])
            # interpolate CRA accel onto MTA accel times
            for k in range(3):
                ac[:, k] = np.interp(mta["t_a"].astype(float), cra["t_a"].astype(float), cra["a"][:, k])
            for i in range(len(mta["a"])):
                a_rel.append(relative_linear_accel(tuple(mta["a"][i]), tuple(qma[i]), tuple(ac[i]), tuple(qca[i])))
            p_full = naive_double_integrate(t_a, np.asarray(a_rel))
            p_imu = np.column_stack(
                [np.interp(t_opt.astype(float) / 1e6, t_a, p_full[:, k]) for k in range(3)]
            )
    else:
        p_opt = pose[:, :3, 3].copy()
        if np.nanmax(np.abs(p_opt)) > 2.0:
            p_opt = p_opt / 1000.0
        q_imu = _apply_mount(_quat_interp(cra["t_q"], cra["q"], t_opt), R_cra)
        if cra["a"] is None:
            p_imu = np.full_like(p_opt, np.nan)
            t_a = t_opt.astype(np.float64) / 1e6
        else:
            t_a = cra["t_a"].astype(np.float64) / 1e6
            a = cra["a"]
            q_a = _apply_mount(_quat_interp(cra["t_q"], cra["q"], cra["t_a"]), R_cra)
            aw = np.zeros_like(a)
            from jtsmm.imu.kinematics import quat_rotate

            for i in range(len(a)):
                aw[i] = quat_rotate(tuple(q_a[i]), tuple(a[i]))
            p_full = naive_double_integrate(t_a, aw)
            p_imu = np.column_stack(
                [np.interp(t_opt.astype(float) / 1e6, t_a, p_full[:, k]) for k in range(3)]
            )

    ident = np.zeros_like(q_imu)
    ident[:, 0] = 1.0
    ang_imu = geodesic_deg(attitude_change_from_t0(q_imu), ident)
    ang_opt = geodesic_deg(attitude_change_from_t0(q_opt), ident)
    ang = geodesic_deg(attitude_change_from_t0(q_imu), attitude_change_from_t0(q_opt))
    t_s = t_opt.astype(np.float64) / 1e6
    t_s = t_s - t_s[0]
    disp_imu = np.linalg.norm(p_imu - p_imu[:1], axis=1)
    disp_opt = np.linalg.norm(p_opt - p_opt[:1], axis=1)
    err_pos = np.abs(disp_imu - disp_opt)
    drifts = {f"win_{w}s_m": window_drift(t_s, p_imu, p_opt, w) for w in WINDOWS_S}
    # Quadratic growth diagnostic on the longest stretch (open_close if present)
    with SessionData(rec.path) as sd:
        phases = list(sd.phases)
    stretch = None
    for p in phases:
        if p.name == "open_close":
            m = (t_opt >= int(p.t_start_us)) & (t_opt <= int(p.t_end_us))
            if m.sum() > 10:
                stretch = m
            break
    if stretch is None:
        stretch = np.ones(len(t_s), dtype=bool)
    e = np.abs(
        np.linalg.norm(p_imu[stretch] - p_imu[stretch][:1], axis=1)
        - np.linalg.norm(p_opt[stretch] - p_opt[stretch][:1], axis=1)
    )
    tt = t_s[stretch] - t_s[stretch][0]
    # fit |‖Δp_IMU‖ − ‖Δp_opt‖| ≈ c t^2  (c in m/s^2)
    A = (tt**2)[:, None]
    mask = np.isfinite(e) & (tt > 0)
    coef = None
    if mask.sum() >= 4:
        c, _, _, _ = np.linalg.lstsq(A[mask], e[mask], rcond=None)
        coef = float(c[0])
    return {
        "session": rec.name,
        "subject_id": rec.subject_id,
        "mode": mode,
        "mounting": mount_src,
        "R_mount_mta": R_mta is not None,
        "R_mount_cra": R_cra is not None,
        "n": int(len(t_s)),
        "orient_rms_deg": float(np.sqrt(np.nanmean(ang**2))) if np.isfinite(ang).any() else None,
        "orient_p95_deg": percentile(ang, 95),
        "drift_p50_m": {k: percentile(np.array(v), 50) for k, v in drifts.items()},
        "drift_p95_m": {k: percentile(np.array(v), 95) for k, v in drifts.items()},
        "quad_c_m_s2": coef,
        "pos_err_1s_m": _at_time(t_s, err_pos, 1.0),
        "pos_err_2s_m": _at_time(t_s, err_pos, 2.0),
        "pos_err_5s_m": _at_time(t_s, err_pos, 5.0),
        "orient_rms_5s_deg": _rms_until(t_s, ang, 5.0),
        "orient_rms_10s_deg": _rms_until(t_s, ang, 10.0),
        "t_s": t_s,
        "disp_imu_m": disp_imu,
        "disp_opt_m": disp_opt,
        "err_pos_m": err_pos,
        "ang_imu_deg": ang_imu,
        "ang_opt_deg": ang_opt,
        "err_ang_deg": ang,
        "drifts": drifts,
    }


def plot_session(block: dict[str, Any], out: Path) -> None:
    """2×2: first 8 s tracking overlay vs full-take drift."""
    import matplotlib.pyplot as plt

    t = np.asarray(block["t_s"])
    early = t <= EARLY_S
    disp_i = np.asarray(block["disp_imu_m"]) * 1e3  # mm
    disp_o = np.asarray(block["disp_opt_m"]) * 1e3
    ang_i = np.asarray(block["ang_imu_deg"])
    ang_o = np.asarray(block["ang_opt_deg"])
    err_a = np.asarray(block["err_ang_deg"])
    sid = block.get("subject_id") or block["session"]
    mode = block["mode"]
    e1 = block.get("pos_err_1s_m")
    e5 = block.get("pos_err_5s_m")
    o5 = block.get("orient_rms_5s_deg")

    fig, ax = plt.subplots(2, 2, figsize=(11.2, 7.2), constrained_layout=True)
    fig.suptitle(
        f"{sid}  {mode}   top: first {EARLY_S:.0f} s (tracking)   bottom: whole take (drift)",
        fontsize=10,
    )

    ax[0, 0].plot(t[early], disp_o[early], color="#1a5276", lw=1.4, label="optical")
    ax[0, 0].plot(t[early], disp_i[early], color="#c0392b", lw=1.1, label="IMU (naive)")
    ax[0, 0].set_xlim(0, EARLY_S)
    ax[0, 0].set_xlabel("t (s)")
    ax[0, 0].set_ylabel("|p(t) − p(t₀)| (mm)")
    ax[0, 0].set_title(
        "Position, first 8 s"
        + (f"   |Δ|={e1 * 1e3:.1f} mm @1 s" if e1 is not None else "")
    )
    ax[0, 0].legend(loc="upper left", fontsize=8)
    ax[0, 0].grid(True, alpha=0.3)

    ax[0, 1].plot(t[early], ang_o[early], color="#1a5276", lw=1.4, label="optical |ΔR|")
    ax[0, 1].plot(t[early], ang_i[early], color="#c0392b", lw=1.1, label="IMU |ΔR|")
    ax[0, 1].plot(t[early], err_a[early], color="#7f8c8d", lw=1.0, ls="--", label="error")
    ax[0, 1].set_xlim(0, EARLY_S)
    ax[0, 1].set_xlabel("t (s)")
    ax[0, 1].set_ylabel("deg")
    ax[0, 1].set_title(
        "Orientation from t₀, first 8 s"
        + (f"   err RMS={o5:.2f}° (5 s)" if o5 is not None else "")
    )
    ax[0, 1].legend(loc="upper left", fontsize=8)
    ax[0, 1].grid(True, alpha=0.3)

    ax[1, 0].plot(t, np.asarray(block["disp_opt_m"]), color="#1a5276", lw=1.2, label="optical")
    ax[1, 0].plot(t, np.asarray(block["disp_imu_m"]), color="#c0392b", lw=0.8, label="IMU (naive)")
    ax[1, 0].axvline(EARLY_S, color="#7f8c8d", ls=":", lw=0.9)
    ax[1, 0].set_xlabel("t (s)")
    ax[1, 0].set_ylabel("|p(t) − p(t₀)| (m)")
    ax[1, 0].set_title(
        "Position, full take (optical stays cm-scale)"
        + (f"   |Δ|={e5 * 1e3:.0f} mm @5 s" if e5 is not None else "")
    )
    ax[1, 0].legend(loc="upper left", fontsize=8)
    ax[1, 0].grid(True, alpha=0.3)

    ax[1, 1].plot(t, ang_o, color="#1a5276", lw=1.0, label="optical |ΔR|")
    ax[1, 1].plot(t, ang_i, color="#c0392b", lw=0.8, label="IMU |ΔR|")
    ax[1, 1].plot(t, err_a, color="#7f8c8d", lw=0.9, ls="--", label="error")
    ax[1, 1].axvline(EARLY_S, color="#7f8c8d", ls=":", lw=0.9)
    ax[1, 1].set_xlabel("t (s)")
    ax[1, 1].set_ylabel("deg")
    ax[1, 1].set_title(f"Orientation from t₀, full take   err RMS={block['orient_rms_deg']:.1f}°")
    ax[1, 1].legend(loc="upper left", fontsize=8)
    ax[1, 1].grid(True, alpha=0.3)

    fig.savefig(out, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--out-dir", "--out", dest="out_dir", type=Path, default=None,
                    help=f"default: $BIOMECH_RESULTS_DIR or ~/experiments, then /{STUDY}/results")
    ap.add_argument("--mounting-dir", type=Path, default=None,
                    help="5_2_2_imu_optical_handeye recordings with imu_mounting.json (default: operator config)")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    data_dir = resolve_data_dir(args.data_dir)
    mounting_dir = (args.mounting_dir or default_mounting_dir()).expanduser()
    recs = [r for r in scan(data_dir, pids=subject_pids(args.subject), kind="motion")
            if r.has_optical]

    out = results_dir(STUDY, args.out_dir).resolve()
    derived = out / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    rows = []
    blocks = []
    for rec in recs:
        try:
            block = eval_session(rec, mounting_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {rec.name}: {exc}", file=sys.stderr)
            continue
        blocks.append(block)
        if block.get("error"):
            print(f"skip {rec.name}: {block['error']}", file=sys.stderr)
            continue
        np.savez_compressed(
            derived / f"imu_only_{rec.name}.npz",
            t_s=block["t_s"],
            disp_imu_m=block["disp_imu_m"],
            disp_opt_m=block["disp_opt_m"],
            err_pos_m=block["err_pos_m"],
            ang_imu_deg=block["ang_imu_deg"],
            ang_opt_deg=block["ang_opt_deg"],
            err_ang_deg=block["err_ang_deg"],
        )
        rows.append(
            {
                "session": rec.name,
                "subject_id": rec.subject_id,
                "mode": block["mode"],
                "R_mount_mta": block["R_mount_mta"],
                "R_mount_cra": block["R_mount_cra"],
                "n": block["n"],
                "orient_rms_deg": block["orient_rms_deg"],
                "orient_p95_deg": block["orient_p95_deg"],
                "orient_rms_5s_deg": block["orient_rms_5s_deg"],
                "orient_rms_10s_deg": block["orient_rms_10s_deg"],
                "pos_err_1s_m": block["pos_err_1s_m"],
                "pos_err_2s_m": block["pos_err_2s_m"],
                "pos_err_5s_m": block["pos_err_5s_m"],
                "quad_c_m_s2": block["quad_c_m_s2"],
                **{f"{k}_p50": block["drift_p50_m"][k] for k in block["drift_p50_m"]},
                **{f"{k}_p95": block["drift_p95_m"][k] for k in block["drift_p95_m"]},
            }
        )
        if not args.no_plot:
            import matplotlib

            matplotlib.use("Agg")
            plot_session(block, out / f"imu_only_{rec.name[-8:]}.png")

    write_csv(out / "imu_only_metrics.csv", rows)
    write_json(out / "imu_only_summary.json", {"n": len(rows), "rows": rows})
    mount_files = sorted({b["mounting"] for b in blocks if b.get("mounting")})
    write_provenance(derived, STUDY, "eval_imu_only.py", recs, data_dir=data_dir,
                     extra={"mounting_dir": str(mounting_dir), "mounting_files": mount_files})
    print(f"sessions={len(recs)} ok={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
