# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Clinical jaw-motion metrics per cycle, anatomy and noise floor (an02 §3-4).

All positions are maxillary coordinates (mm; x anterior, y left, z up) of
mandibular points tracked by :class:`~biomech.kinematics.pose_chain.JawMotion`.
Reference pose: for open_close and chewing, a cycle's own start frame (closed,
analysis_plan §4.1). For protrusion_retrusion and left_right the start frame is
the *opposite* excursion (retruded / other side), so those tasks are referenced
to the closed pose ``ref_T`` (median start pose of the open_close cycles).
Angles are rotations since the reference about the maxillary y axis (positive =
opening).

Task -> excursion signal used for cycle detection:

==================== ==========================================
open_close, chewing  opening = -z of the incisal point
protrusion_retrusion x of the incisal point (protrusion > 0)
left_right           +y (left excursions) and -y (right ones)
==================== ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.kinematics.cycles import Cycle, detect
from biomech.kinematics.derivatives import derivatives, speed
from biomech.kinematics.pose_chain import JawMotion

TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing")


@dataclass
class TaskCycles:
    task: str
    side: str                 # "", "left", "right"
    sel: np.ndarray           # boolean frame mask of the phase
    cycles: list[Cycle]


def phase_signals(jm: JawMotion, t0: int, t1: int) -> dict[str, np.ndarray]:
    sel = jm.window(t0, t1)
    return {
        "sel": sel, "t_us": jm.t_us[sel], "T": jm.T[sel],
        "incisor": jm.track("incisor")[sel], "tmj_left": jm.track("tmj_left")[sel],
        "tmj_right": jm.track("tmj_right")[sel], "tmj_mid": jm.track("tmj_mid")[sel],
    }


def find_task_cycles(jm: JawMotion, task: str, t0: int, t1: int, p: dict[str, Any]
                     ) -> list[TaskCycles]:
    sig = phase_signals(jm, t0, t1)
    inc = sig["incisor"]
    kw = {k: p[k] for k in ("min_period_s", "prominence_frac", "min_amplitude",
                            "max_nan_frac") if k in p}
    if task in ("open_close", "chewing"):           # start and end closed
        kw["return_frac"] = p.get("return_frac")
    if task in ("open_close", "chewing"):
        series = [("", -inc[:, 2])]
    elif task == "protrusion_retrusion":
        series = [("", inc[:, 0])]
    elif task == "left_right":
        series = [("left", inc[:, 1]), ("right", -inc[:, 1])]
    else:
        return []
    out = []
    for side, x in series:
        out.append(TaskCycles(task, side, sig["sel"], detect(x, jm.fs, **kw)))
    return out


def theta_deg(T: np.ndarray, T_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sagittal opening angle about +y, total rotation angle) since T_ref, degrees."""
    rv = se3.rotvec_rel(T, T_ref)
    return np.degrees(rv[:, 1]), np.degrees(np.linalg.norm(rv, axis=1))


CLOSED_REFERENCED = ("protrusion_retrusion", "left_right")


def cycle_metrics(jm: JawMotion, tc: TaskCycles, pid: str, ref_T: np.ndarray | None = None
                  ) -> list[dict[str, Any]]:
    idx = np.flatnonzero(tc.sel)
    t = jm.t_us[idx] * 1e-6
    T = jm.T[idx]
    inc = jm.track("incisor")[idx]
    tl, tr = jm.track("tmj_left")[idx], jm.track("tmj_right")[idx]
    use_ref = ref_T is not None and tc.task in CLOSED_REFERENCED
    if use_ref:
        r_inc = se3.apply(ref_T, jm.points_mp["incisor"])
        r_tl = se3.apply(ref_T, jm.points_mp["tmj_left"])
        r_tr = se3.apply(ref_T, jm.points_mp["tmj_right"])
    rows = []
    for k, c in enumerate(tc.cycles):
        s, pk, e = c.start, c.peak, c.end
        T0 = ref_T if use_ref else T[s]
        th, tot = theta_deg(T[s:e + 1], T0)
        i0 = r_inc if use_ref else inc[s]
        l0 = r_tl if use_ref else tl[s]
        q0 = r_tr if use_ref else tr[s]
        d = inc[s:e + 1] - i0
        # Same estimator as the stored phase derivatives (kinematics.derivatives).
        tw = t[s:e + 1] * 1e6
        vel = derivatives(tw, th)[0] if e > s + 2 else np.full(th.shape, np.nan)
        spd = speed(derivatives(tw, inc[s:e + 1])[0]) if e > s + 2 else np.full(th.shape, np.nan)
        j = pk - s
        # Movement onset/offset: incisal speed crosses 5 % of its peak, searched outward
        # from the opening and closing speed peaks. The detected start/end lie anywhere in
        # the flat rest around a cycle, which misaligns time-normalized averages; a
        # displacement threshold (the first version) fires when the velocity is already
        # at ~40 % of its peak.
        on, off = movement_bounds(spd, j)
        on, off = s + on, s + off
        row = {
            "pid": pid, "task": tc.task, "side": tc.side, "cycle": k, "ok": c.ok,
            "nan_frac": c.nan_frac, "flags": ",".join(c.flags),
            "t_start_us": int(jm.t_us[idx[s]]), "t_peak_us": int(jm.t_us[idx[pk]]),
            "t_end_us": int(jm.t_us[idx[e]]),
            "t_on_us": int(jm.t_us[idx[on]]), "t_off_us": int(jm.t_us[idx[off]]),
            "duration_s": float(t[e] - t[s]), "rise_s": float(t[pk] - t[s]),
            "opening_mm": float(-d[j, 2]),
            "incisal_disp_mm": float(np.linalg.norm(d[j])),
            "protrusion_mm": float(d[j, 0]),
            "lateral_mm": float(d[j, 1]),
            "deviation_mm": float(np.nanmax(np.abs(d[: j + 1, 1]))) if j > 0 else np.nan,
            "retrusion_mm": float(-np.nanmin(d[:, 0])) if tc.task == "protrusion_retrusion"
            else np.nan,
            "reference": "closed" if use_ref else "cycle_start",
            "theta_y_deg": float(th[j]), "theta_total_deg": float(tot[j]),
            "condyle_left_disp_mm": float(np.linalg.norm(tl[pk] - l0)),
            "condyle_right_disp_mm": float(np.linalg.norm(tr[pk] - q0)),
            "condyle_left_ant_mm": float(tl[pk, 0] - l0[0]),
            "condyle_right_ant_mm": float(tr[pk, 0] - q0[0]),
            "peak_open_vel_dps": float(np.nanmax(vel[: j + 1])) if j > 0 else np.nan,
            "peak_close_vel_dps": float(-np.nanmin(vel[j:])) if e > pk else np.nan,
            "peak_incisal_speed_mm_s": float(np.nanmax(spd)) if spd.size else np.nan,
        }
        rows.append(row)
    return rows


def movement_bounds(speed: np.ndarray, peak: int, frac: float = 0.05) -> tuple[int, int]:
    """(onset, offset) indices of a cycle from its speed profile (``peak`` = excursion peak).

    Onset: the last sample before the opening speed maximum that is below ``frac`` of it;
    offset: the first sample after the closing speed maximum below ``frac`` of that.
    Falls back to the window edges when the speed never drops that low.
    """
    v = np.nan_to_num(np.asarray(speed, float), nan=0.0)
    n = v.size
    if n < 3:
        return 0, max(n - 1, 0)
    i_up = int(np.argmax(v[: max(peak, 1) + 1]))
    i_dn = peak + int(np.argmax(v[peak:])) if peak < n else n - 1
    low = np.flatnonzero(v[: i_up + 1] < frac * v[i_up])
    on = int(low[-1]) if low.size else 0
    low = np.flatnonzero(v[i_dn:] < frac * v[i_dn])
    off = i_dn + int(low[0]) if low.size else n - 1
    return on, off


def anatomy(jm: JawMotion, closed_inc: np.ndarray | None) -> dict[str, float]:
    """Size proxies and incisal overlap (an02 §2). ``closed_inc``: incisal point when closed."""
    pm = jm.points_mp
    up = jm.upper_max
    out = {
        "upper_intermolar_mm": float(np.linalg.norm(up["left"] - up["right"])),
        "lower_intermolar_mm": float(np.linalg.norm(pm["left"] - pm["right"])),
        "intercondylar_mm": jm.condyles.intercondylar_mm,
        "mandibular_length_mm": float(np.linalg.norm(pm["incisor"] - pm["tmj_mid"])),
        "condyle_incisor_left_mm": jm.condyles.incisor_left_mm,
        "condyle_incisor_right_mm": jm.condyles.incisor_right_mm,
        "condyles_plausible": float(jm.condyles.plausible),
        "condyles_reason": jm.condyles.reason,
        "upper_incisor_mad_mm": jm.landmarks.items[jm.upper.incisor].mad_mm,
        "lower_incisor_mad_mm": jm.landmarks.items[jm.lower.incisor].mad_mm,
        "tmj_left_mad_mm": jm.landmarks.items["tmj_left"].mad_mm,
        "tmj_right_mad_mm": jm.landmarks.items["tmj_right"].mad_mm,
        "roles_upper_geometry_agrees": float(jm.upper.agrees),
        "roles_lower_geometry_agrees": float(jm.lower.agrees),
        "roles_relabelled": float(jm.upper.relabelled or jm.lower.relabelled),
        "upper_incisor_label": jm.upper.incisor, "lower_incisor_label": jm.lower.incisor,
        **{f"frame_{k}": v for k, v in jm.frame_check.items()},
    }
    if closed_inc is not None:
        # Offset between the two digitized incisor landmarks at closure (QC only):
        # the pointer sat on labial surfaces, not incisal edges, so this is not
        # the clinical overbite/overjet (Deviation D2).
        out["incisor_landmark_dz_mm"] = float(closed_inc[2])
        out["incisor_landmark_dx_mm"] = float(closed_inc[0])
    return out


def noise_floor(jm: JawMotion, t0: int, t1: int, window_s: float = 1.0) -> dict[str, float]:
    """In-vivo noise floor and rest stability of the incisal point.

    ``rest_*``: 3D RMS about the mean over the whole rest phase (includes small
    real movements: swallowing, posture). ``floor_incisor_rms_mm_*`` and
    ``floor_theta_sd_deg_*``: the quietest ``window_s`` window of rest (least incisor
    RMS; one window for both), a descriptive lower bound of rest stability.

    ``floor_rot_sd_deg_*_{x,y,z}`` / ``floor_trans_sd_mm_*_{x,y,z}``: SD of the fast pose
    fluctuation over the whole rest phase (:func:`rest_residuals`, E28) in the
    parametrization the Monte Carlo null injects (left-multiplied rotation about each axis,
    translation of the pose origin). The in-vivo null resamples these residuals themselves
    (their temporal correlation included: at 40 Hz the rest fluctuation is far from white,
    lag-1 autocorrelation 0.6-0.8), not white noise of this SD. Before E28 the SD of the
    quietest 1-s window was used -- a minimum over many windows, biased low. The incisor
    RMS mixes both (rotation moves the incisor), so using it as the translational noise
    would count rotation noise twice.
    """
    sel = jm.window(t0, t1)
    out: dict[str, float] = {}
    w = max(int(round(window_s * jm.fs)), 5)
    for tag, T in (("smooth", jm.T[sel]), ("raw", jm.T_raw[sel])):
        ok = se3.valid(T)
        if ok.sum() < w:
            continue
        inc = se3.apply(T, jm.points_mp["incisor"])
        good = inc[ok]
        out[f"rest_incisor_rms_mm_{tag}"] = float(np.sqrt(((good - good.mean(0)) ** 2).sum(1).mean()))
        th = theta_deg(T, T[np.flatnonzero(ok)[0]])[0]
        out[f"rest_theta_sd_deg_{tag}"] = float(np.nanstd(th))
        # One common quietest window (least incisor RMS) for every floor (E24): the
        # theta floor was the minimum over all windows, possibly a different one.
        best_mm, best_i = np.inf, None
        for i in range(0, len(inc) - w + 1, max(w // 4, 1)):
            seg = inc[i:i + w]
            if not np.isfinite(seg).all():
                continue
            rms = float(np.sqrt(((seg - seg.mean(0)) ** 2).sum(1).mean()))
            if rms < best_mm:
                best_mm, best_i = rms, i
        best_deg = float(np.std(th[best_i:best_i + w])) if best_i is not None else np.nan
        out[f"floor_incisor_rms_mm_{tag}"] = best_mm if np.isfinite(best_mm) else np.nan
        out[f"floor_theta_sd_deg_{tag}"] = best_deg if np.isfinite(best_deg) else np.nan
        rot_sd, trans_sd = pose_noise_residual(T, jm.fs)
        for ax, r, t in zip("xyz", rot_sd, trans_sd):
            out[f"floor_rot_sd_deg_{tag}_{ax}"] = float(r)
            out[f"floor_trans_sd_mm_{tag}_{ax}"] = float(t)
    return out


REST_HP_S = 1.0     # rest residuals: motion slower than a centred 1-s running median removed


def rest_residuals(T: np.ndarray, fs: float, hp_s: float = REST_HP_S) -> list[np.ndarray]:
    """Fast pose fluctuation at rest, per run of consecutive tracked frames (E28).

    Per frame the left-multiplied perturbation of :func:`pose_noise` -- rotation vector of
    dR_k = R_k·R̄ᵀ (deg) and dt_k = t_k − dR_k·t̄ (mm), R̄/t̄ the mean pose of the valid frames
    -- minus its centred running median over ``hp_s`` (odd window), so slow real movement
    (swallowing, posture drift) is removed while the fast, temporally correlated
    fluctuation stays. Runs shorter than the window are dropped. Returns one (n, 6) array
    [rot x, y, z (deg), trans x, y, z (mm)] per run.
    """
    from scipy.ndimage import median_filter

    from biomech.timebase import odd_window

    ok = se3.valid(T)
    w = odd_window(hp_s, fs)
    if ok.sum() < w:
        return []
    idx = np.flatnonzero(ok)
    Tk = T[idx]
    R_mean = Rotation.from_matrix(Tk[:, :3, :3]).mean().as_matrix()
    dR = Tk[:, :3, :3] @ R_mean.T
    p = np.column_stack([np.degrees(Rotation.from_matrix(dR).as_rotvec()),
                         Tk[:, :3, 3] - np.einsum("nij,j->ni", dR, Tk[:, :3, 3].mean(axis=0))])
    br = np.flatnonzero(np.diff(idx) != 1) + 1
    out = []
    for run in np.split(p, br):
        if len(run) >= w:
            out.append(run - median_filter(run, size=(w, 1), mode="nearest"))
    return out


def pose_noise_residual(T: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis SD (ddof 1) of :func:`rest_residuals`: (rotation deg, translation mm)."""
    runs = rest_residuals(T, fs)
    if not runs or sum(len(r) for r in runs) < 5:
        return np.full(3, np.nan), np.full(3, np.nan)
    sd = np.concatenate(runs).std(axis=0, ddof=1)
    return sd[:3], sd[3:]


def pose_noise(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis SD of the left-multiplied pose perturbation ``T_k = dT_k · mean(T)``.

    ``dT_k = [dR_k, dt_k]`` with ``dR_k = R_k R̄ᵀ`` (rotation vector, deg) and
    ``dt_k = t_k − dR_k t̄`` (mm): the translation left after removing what the
    rotation noise does to the pose origin. This is exactly the perturbation the
    Monte Carlo null applies (``montecarlo._noisy``), so estimated and simulated
    noise share one parametrization.
    """
    ok = se3.valid(T)
    if ok.sum() < 3:
        return np.full(3, np.nan), np.full(3, np.nan)
    Tk = T[ok]
    R_mean = Rotation.from_matrix(Tk[:, :3, :3]).mean().as_matrix()
    dR = Tk[:, :3, :3] @ R_mean.T
    rv = Rotation.from_matrix(dR).as_rotvec()
    t_mean = Tk[:, :3, 3].mean(axis=0)
    dt = Tk[:, :3, 3] - np.einsum("nij,j->ni", dR, t_mean)
    return np.degrees(rv.std(axis=0, ddof=1)), (dt - dt.mean(0)).std(axis=0, ddof=1)
