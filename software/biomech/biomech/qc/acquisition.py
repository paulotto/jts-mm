# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Acquisition quality of one recording: camera frame drops and the frame-trigger
association, the camera time base, the IMU-to-camera lag and the recording day (X11, J14; ported from and checked against the earlier evaluations 5.3.1 / 5.3.2 of the
recording system, an20).

Pure functions on a recording's ``session.h5``; :func:`analyse` returns the stage tables.

* **Frame drops** (:func:`camera_drops`): an increase of the assigned trigger id by k between
  two consecutive frames of a camera is k - 1 dropped frames. The start-up gap (the cameras
  deliver no frames for about 0.5 s after their start, cam0 after its first one or two
  frames) is counted separately: every gap that ends within ``startup_s`` of the first
  trigger. The drop rate is the dropped share of the triggers of regular operation (first
  frame after the start-up gap to the last frame).
* **Association counters** (:func:`association_counters`): the recorder's own counters from
  ``/sync`` ``report_json`` (which path assigned each frame). ``cadence_advance`` minus the
  start-up ids equals the regular drops when every drop was resolved by the cadence step.
* **Time base** (:func:`timebase_residuals`): the live clock fit of every camera
  (``jtsmm.sync.clock_align.ClockAligner._update_fit``: least squares of the trigger master
  time on the SensorTimestamp over a trailing window of ``fit_window`` frames that includes
  the current frame), replayed offline; the residual of each frame from its own window's
  fit. A frame paired with a neighbouring trigger would sit a whole period (25 ms at 40 Hz)
  off; a constant whole-period shift is absorbed by the offset and is checked by the IMU
  lag instead (the IMU is stamped independently of the association).
* **IMU-to-camera lag** (:func:`imu_camera_lag`): the magnitude of the angular velocity of
  the mandibular array from the optics (frame-to-frame rotation, exposure-centre times:
  trigger + ``exposure_s``/2) and from the IMU game rotation vector (sample-to-sample), each
  body-frame component on a common grid through the same zero-phase smoothing before the
  magnitude is taken (smoothing |omega| itself keeps the positive noise floor of the 40 Hz
  pose differences; the correlation then fell to 0.35-0.5). Sample-to-sample rotations faster
  than ``max_omega_dps`` are glitches (isolated IMU quaternion jumps of up to 1e5 deg/s) and
  are masked in both streams. The lag of the peak Pearson correlation (parabolic
  refinement) over the movement tasks, positive when the IMU shows the motion later, and per
  task (constancy within the take; the tasks last 10-24 s). |omega| needs no hand-eye
  calibration (frame-independent).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULTS: dict[str, Any] = {
    "fit_window": 30,                 # frames: ClockAligner.window (the live fit)
    "startup_s": 1.0,                 # gaps ending this soon after the first trigger: start-up
    "spike_us": 50.0,                 # residual above this: an isolated spike (reported)
    "exposure_s": 0.012,              # pulse width = exposure; frame time = trigger + half
    "lag": {
        "body": "mta",
        "tasks": ["open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion"],
        "grid_hz": 1000.0,
        "smooth_s": 0.175,            # centred Savitzky-Golay on both |omega| (order 2)
        "max_lag_ms": 100.0,
        "min_r": 0.8,                 # a task counts when its peak correlation reaches this
        "max_gap_s": 0.0625,          # optical gaps > 2.5 frame periods are not bridged
        "max_omega_dps": 1000.0,      # sample-to-sample rotations faster than this are glitches
                                      # (IMU quaternion jumps up to 1e5 deg/s; jaw < 500 deg/s)
    },
}

COUNTERS = ("cadence_advance", "camera_drop_recovered", "cadence_missing_trigger",
            "slip_count", "validation_fail_count", "validation_relock_count", "anomaly_count",
            "drops", "waiting_for_trigger", "trigger_lost_edges", "discarded_trigger_backlog",
            "cam_hold_overflow", "camera_missing_sensor_ts")


@dataclass
class CameraStream:
    cam: str
    trigger_id: np.ndarray      # assigned trigger id per frame (0 = none)
    sensor_ns: np.ndarray
    t_master_us: np.ndarray     # the assigned trigger's master time


def read_streams(h5: str | Path) -> tuple[dict[str, CameraStream], np.ndarray, np.ndarray,
                                          dict[str, Any]]:
    """Camera streams, trigger ids and times, and the sync report of a recording."""
    import h5py

    with h5py.File(h5, "r") as f:
        cams = {}
        for c in sorted(f["cameras"].keys()):
            g = f["cameras"][c]
            cams[c] = CameraStream(c, np.asarray(g["trigger_id"][:], np.int64),
                                   np.asarray(g["sensor_timestamp_ns"][:], np.int64),
                                   np.asarray(g["t_master_us"][:], np.int64))
        tid = np.asarray(f["triggers/trigger_id"][:], np.int64)
        tt = np.asarray(f["triggers/t_master_us"][:], np.int64)
        rep = json.loads(f["sync"].attrs["report_json"]) if "sync" in f else {}
    return cams, tid, tt, rep


def camera_drops(cam: CameraStream, trig_id: np.ndarray, trig_t_us: np.ndarray,
                 startup_s: float = 1.0) -> dict[str, Any]:
    """Drops, bursts and the start-up gap of one camera (see the module docstring)."""
    ok = cam.trigger_id > 0
    ids = cam.trigger_id[ok]
    if ids.size < 2 or trig_id.size == 0:
        return {"cam": cam.cam, "n_frames": int(ids.size), "note": "too few frames"}
    t_of = dict(zip(trig_id.tolist(), trig_t_us.tolist()))
    t0 = float(trig_t_us.min())
    first_trig = int(trig_id.min())
    d = np.diff(ids)
    gaps = np.flatnonzero(d > 1)
    startup = int(ids[0] - first_trig)              # triggers before the first frame
    regular_from = 0
    bursts = []
    for g in gaps:
        end_t = t_of.get(int(ids[g + 1]), np.nan)
        if np.isfinite(end_t) and (end_t - t0) <= startup_s * 1e6:
            startup += int(d[g] - 1)
            regular_from = g + 1
        else:
            bursts.append(int(d[g] - 1))
    span = int(ids[-1] - ids[regular_from] + 1)      # triggers of regular operation
    dropped = int(sum(bursts))
    back = int(np.sum(d < 1))                        # non-increasing ids (never expected)
    b = np.asarray(bursts, int)
    return {"cam": cam.cam, "n_frames": int(ids.size), "startup_triggers": startup,
            "pre_first_triggers": int(ids[0] - first_trig),
            "regular_triggers": span, "dropped": dropped,
            "drop_rate_pct": 100.0 * dropped / span if span else np.nan,
            "n_bursts": int(b.size), "burst_single_frac": float(np.mean(b == 1)) if b.size else np.nan,
            "burst_max": int(b.max()) if b.size else 0, "id_backsteps": back}


def association_counters(report: dict[str, Any]) -> pd.DataFrame:
    """The recorder's association counters per camera (``cam`` = "all" for scalars)."""
    rows = []
    for k in COUNTERS:
        v = report.get(k)
        if isinstance(v, dict):
            for c, x in v.items():
                if isinstance(x, (int, float)):
                    rows.append({"counter": k, "cam": c, "value": float(x)})
            if not v:
                rows.append({"counter": k, "cam": "all", "value": 0.0})
        elif isinstance(v, (int, float)):
            rows.append({"counter": k, "cam": "all", "value": float(v)})
        else:
            rows.append({"counter": k, "cam": "all", "value": np.nan})
    return pd.DataFrame(rows)


def timebase_residuals(cam: CameraStream, window: int = 30) -> np.ndarray:
    """Residual (µs) of every frame from the live trailing-window affine fit (NaN for the
    first two frames and unassigned frames). Vectorised replay of ``_update_fit``."""
    ok = cam.trigger_id > 0
    x = cam.sensor_ns[ok].astype(np.float64) * 1e-3          # µs
    y = cam.t_master_us[ok].astype(np.float64)
    n = x.size
    r = np.full(n, np.nan)
    if n < 3:
        out = np.full(cam.trigger_id.size, np.nan)
        return out
    w = int(window)
    pad = w - 1
    xp = np.concatenate([np.full(pad, np.nan), x])
    yp = np.concatenate([np.full(pad, np.nan), y])
    X = np.lib.stride_tricks.sliding_window_view(xp, w)       # (n, w), window ends at i
    Y = np.lib.stride_tricks.sliding_window_view(yp, w)
    x0 = X[:, -1:]
    y0 = Y[:, -1:]
    xc, yc = X - x0, Y - y0                                   # centred on the current frame
    m = np.isfinite(xc)
    cnt = m.sum(axis=1)
    xc0, yc0 = np.where(m, xc, 0.0), np.where(m, yc, 0.0)
    mx = xc0.sum(1) / cnt
    my = yc0.sum(1) / cnt
    sxx = (np.where(m, (xc - mx[:, None]) ** 2, 0.0)).sum(1)
    sxy = (np.where(m, (xc - mx[:, None]) * (yc - my[:, None]), 0.0)).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(sxx > 1e-9, sxy / sxx, 1.0)
    pred0 = my - a * mx                                        # fit at the current frame (xc = 0)
    r = -pred0                                                 # y_i - pred_i, centred: 0 - pred0
    r[cnt < 3] = np.nan
    out = np.full(cam.trigger_id.size, np.nan)
    out[ok] = r
    return out


def residual_summary(r: np.ndarray, spike_us: float = 50.0) -> dict[str, float]:
    v = np.abs(r[np.isfinite(r)])
    if v.size == 0:
        return {"n": 0}
    big = v > spike_us
    consec = int(np.sum(big[1:] & big[:-1])) if v.size > 1 else 0
    return {"n": int(v.size), "rms_us": float(np.sqrt(np.mean(v ** 2))),
            "p99_us": float(np.percentile(v, 99)), "max_us": float(v.max()),
            "n_above_spike": int(big.sum()), "frac_above_spike": float(big.mean()),
            "consecutive_spikes": consec}


# -- IMU-to-camera lag ------------------------------------------------------------------------
def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = np.asarray(q, float)
    return Rotation.from_quat(np.c_[q[:, 1:], q[:, :1]]).as_matrix()


def omega_vec(t_us: np.ndarray, R: np.ndarray, max_dt_us: float | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame angular velocity (deg/s, (N-1, 3)) from consecutive rotations, at the
    interval midpoints. Intervals longer than ``max_dt_us`` (gaps) or with an invalid
    rotation give NaN. The vector, not its magnitude, is smoothed downstream: frame-to-frame
    pose noise is zero-mean in the components but adds a positive floor to |omega|."""
    from scipy.spatial.transform import Rotation

    t = np.asarray(t_us, float)
    dt = np.diff(t)
    Rr = np.einsum("nji,njk->nik", R[:-1], R[1:])              # R_k^T R_{k+1}
    w = np.degrees(Rotation.from_matrix(Rr).as_rotvec())
    with np.errstate(invalid="ignore", divide="ignore"):
        w = w / (dt[:, None] * 1e-6)
    bad = ~np.isfinite(w).all(axis=1) | (dt <= 0)
    if max_dt_us is not None:
        bad |= dt > max_dt_us
    w[bad] = np.nan
    return 0.5 * (t[:-1] + t[1:]), w


def _to_grid(t: np.ndarray, x: np.ndarray, grid: np.ndarray, max_gap_us: float) -> np.ndarray:
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return np.full(grid.shape, np.nan)
    tv, xv = t[ok], x[ok]
    y = np.interp(grid, tv, xv, left=np.nan, right=np.nan)
    j = np.clip(np.searchsorted(tv, grid), 1, tv.size - 1)
    y[(tv[j] - tv[j - 1]) > max_gap_us] = np.nan
    return y


def _smooth(x: np.ndarray, n: int) -> np.ndarray:
    """Centred Savitzky-Golay (order 2) on finite runs; NaN stays NaN (zero phase)."""
    from scipy.signal import savgol_filter

    from biomech import timebase

    y = np.full(x.shape, np.nan)
    n = max(5, n | 1)
    for a, b in timebase.segments(np.isfinite(x)):
        if b - a >= n:
            y[a:b] = savgol_filter(x[a:b], n, 2)
    return y


def _peak(lags: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """Peak lag (samples, parabolic refinement) and peak correlation."""
    if not np.isfinite(c).any():
        return np.nan, np.nan
    j = int(np.nanargmax(c))
    if 0 < j < c.size - 1 and np.isfinite(c[j - 1]) and np.isfinite(c[j + 1]):
        den = c[j - 1] - 2 * c[j] + c[j + 1]
        off = 0.5 * (c[j - 1] - c[j + 1]) / den if den < 0 else 0.0
        return float(lags[j] + off), float(c[j])
    return float(lags[j]), float(c[j])


def lag_of(a: np.ndarray, b: np.ndarray, max_lag: int, min_pairs: int = 200
           ) -> tuple[float, float]:
    """Lag (samples, > 0: ``b`` late) and peak Pearson correlation of ``a`` and ``b``."""
    from biomech.timing.coupling import lagged_corr

    lags, c = lagged_corr(a, b, max_lag, min_pairs=min_pairs, center=True)
    return _peak(lags, c)


def imu_camera_lag(t_opt_us: np.ndarray, R_opt: np.ndarray, t_imu_us: np.ndarray,
                   q_imu_wxyz: np.ndarray, intervals: list[tuple[int, int]],
                   p: dict[str, Any], exposure_s: float = 0.012
                   ) -> tuple[dict[str, Any], pd.DataFrame]:
    """IMU-to-camera lag over the task ``intervals`` (master µs) and per window."""
    fs = float(p["grid_hz"])
    step = 1e6 / fs
    kmax = int(round(p["max_lag_ms"] * 1e-3 * fs))
    nsm = int(round(p["smooth_s"] * fs))
    t_o = np.asarray(t_opt_us, float) + 0.5 * exposure_s * 1e6
    ok = np.isfinite(R_opt).all(axis=(1, 2))
    to, wo = omega_vec(t_o[ok], R_opt[ok], max_dt_us=p["max_gap_s"] * 1e6)
    ti, wi = omega_vec(np.asarray(t_imu_us, float), _quat_wxyz_to_R(q_imu_wxyz),
                       max_dt_us=50e3)
    n_glitch = {}
    for name, w in (("optical", wo), ("imu", wi)):
        g = np.linalg.norm(w, axis=1) > float(p["max_omega_dps"])
        w[g] = np.nan
        n_glitch[name] = int(g.sum())

    def mag(t: np.ndarray, w: np.ndarray, grid: np.ndarray, gap_us: float) -> np.ndarray:
        comp = np.stack([_smooth(_to_grid(t, w[:, k], grid, gap_us), nsm) for k in range(3)], 1)
        return np.linalg.norm(comp, axis=1)
    segs_a, segs_b, wins = [], [], []
    sep = np.full(kmax + 1, np.nan)                    # never correlate across intervals
    for a0, a1, task in intervals:
        grid = np.arange(a0, a1, step)
        if grid.size < 4 * kmax:
            continue
        ga = mag(to, wo, grid, p["max_gap_s"] * 1e6)
        gb = mag(ti, wi, grid, 50e3)
        segs_a += [ga, sep]
        segs_b += [gb, sep]
        L, r = lag_of(ga, gb, kmax)
        wins.append({"task": task, "t_mid_s": float(0.5 * (a0 + a1) * 1e-6),
                     "lag_ms": L / fs * 1e3, "r": r})
    if not segs_a:
        return {"lag_ms": np.nan, "note": "no movement interval long enough"}, pd.DataFrame()
    L, r = lag_of(np.concatenate(segs_a), np.concatenate(segs_b), kmax)
    wdf = pd.DataFrame(wins)
    good = wdf[wdf["r"] >= p["min_r"]] if not wdf.empty else wdf
    out = {"lag_ms": L / fs * 1e3, "r_peak": r,
           "lag_vs_trigger_ms": L / fs * 1e3 + 0.5 * exposure_s * 1e3,
           "n_tasks": int(len(wdf)), "n_tasks_used": int(len(good)),
           "task_lag_median_ms": float(good["lag_ms"].median()) if len(good) else np.nan,
           "task_lag_range_ms": float(good["lag_ms"].max() - good["lag_ms"].min())
           if len(good) else np.nan,
           "task_lag_slope_ms_per_min": float(np.polyfit(good["t_mid_s"] / 60.0,
                                                         good["lag_ms"], 1)[0])
           if len(good) >= 3 else np.nan,
           "n_glitch_imu": n_glitch["imu"], "n_glitch_optical": n_glitch["optical"]}
    return out, wdf


# -- orchestration ------------------------------------------------------------------------------
def analyse(session_dir: str | Path, pid: str, take: str, phases: list[Any],
            imu_stream: Any, optical: Any, meta: dict[str, Any],
            p: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Tables of one recording: ``cameras`` (drops and time base per camera), ``counters``,
    ``lag`` (one row), ``lag_windows``, ``recording`` (day and time)."""
    p = {**DEFAULTS, **(p or {})}
    lp = {**DEFAULTS["lag"], **(p.get("lag") or {})}
    h5 = Path(session_dir) / "session.h5"
    cams, tid, tt, rep = read_streams(h5)
    task_iv = [(int(ph.t_start_us), int(ph.t_end_us), ph.name) for ph in phases
               if getattr(ph, "name", None) in lp["tasks"]]
    rows = []
    for c, cs in cams.items():
        row = {"pid": pid, "take": take, **camera_drops(cs, tid, tt, float(p["startup_s"]))}
        r = timebase_residuals(cs, int(p["fit_window"]))
        row.update({f"resid_{k}": v for k, v in residual_summary(r, p["spike_us"]).items()})
        if task_iv:
            tm = cs.t_master_us
            inside = np.zeros(tm.size, bool)
            for a0, a1, _ in task_iv:
                inside |= (tm >= a0) & (tm < a1)
            row.update({f"resid_tasks_{k}": v
                        for k, v in residual_summary(np.where(inside, r, np.nan),
                                                     p["spike_us"]).items()})
        rows.append(row)
    cnt = association_counters(rep)
    cnt.insert(0, "take", take)
    cnt.insert(0, "pid", pid)
    lag_row: dict[str, Any] = {"pid": pid, "take": take, "body": lp["body"]}
    wdf = pd.DataFrame()
    if imu_stream is not None and optical is not None and lp["body"] in optical.poses:
        R = optical.poses[lp["body"]][:, :3, :3]
        res, wdf = imu_camera_lag(optical.t_us.astype(np.int64), R,
                                  np.asarray(imu_stream.t_us, np.int64),
                                  np.asarray(imu_stream.quat, float), task_iv, lp,
                                  float(p["exposure_s"]))
        lag_row.update(res)
        if not wdf.empty:
            wdf.insert(0, "take", take)
            wdf.insert(0, "pid", pid)
    else:
        lag_row["note"] = "no IMU or optical stream"
    date = str(meta.get("recording_date", "")) or None
    rec = pd.DataFrame([{"pid": pid, "take": take, "recording_utc": date,      # private
                         "date": date[:10] if date else None,                  # private
                         "firmware": meta.get("firmware_version"),
                         "software": meta.get("software_version"),
                         "pose_method": meta.get("optical_pose_method"),
                         "ct_tip_rms_mm": float(meta["ct_tip_rms_mm"])
                         if meta.get("ct_tip_rms_mm") not in (None, "") else np.nan}])
    return {"cameras": pd.DataFrame(rows), "counters": cnt, "lag": pd.DataFrame([lag_row]),
            "lag_windows": wdf, "recording": rec}
