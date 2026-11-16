# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU dynamic-calibration audit helpers shared by both studies (stored by the validation stage ``ref_diagnostics`` and the dataset stage ``imu_dyncal``): still
periods of both IMUs, steps of an error trace and their alignment with stillness, drift inside
and outside the still periods, and the jumps across phase boundaries. Kept outside
``biomech.validation`` so the dataset code never depends on the validation package (separate
code hashes). ``scripts/audit_recompute/imu_dyncal.py`` stays the independent re-computation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

DEFAULTS: dict[str, Any] = {
    "grid_hz": 100.0,
    "smooth_s": 0.2,
    "still_dps": 2.0,
    "still_min_s": 2.0,
    "block_s": 0.25,
    "max_gap_s": 1.0,
    "step_min_deg": 0.25,
    "step_k": 5.0,
    "after_s": 2.0,
    "jump_s": 1.0,
}


def _quat_matrix(q: np.ndarray) -> np.ndarray:
    """(N, 3, 3) from wxyz quaternions (NaN rows stay NaN)."""
    q = np.asarray(q, float)
    out = np.full((len(q), 3, 3), np.nan)
    ok = np.isfinite(q).all(1)
    if ok.any():
        out[ok] = Rot.from_quat(q[ok][:, [1, 2, 3, 0]]).as_matrix()
    return out


def angular_speed_dps(q: np.ndarray, grid_hz: float, smooth_s: float) -> np.ndarray:
    """Angular speed (deg/s) of a wxyz quaternion series on a regular grid: geodesic
    increment per sample, the last one repeated, centred ``smooth_s`` running mean."""
    R = Rot.from_quat(np.asarray(q, float)[:, [1, 2, 3, 0]])
    w = np.degrees((R[:-1].inv() * R[1:]).magnitude()) * grid_hz
    n = max(1, int(round(smooth_s * grid_hz)))
    return np.convolve(np.r_[w, w[-1]], np.ones(n) / n, "same")


def true_runs(mask: np.ndarray, min_n: int) -> list[tuple[int, int]]:
    """``[a, b)`` index runs of ``mask`` at least ``min_n`` long."""
    d = np.diff(np.r_[0, np.asarray(mask).astype(np.int8), 0])
    a, b = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(i), int(j)) for i, j in zip(a, b) if j - i >= min_n]


def still_runs(qc: np.ndarray, qm: np.ndarray, p: dict[str, Any]) -> list[tuple[int, int]]:
    """Grid runs where both IMUs turn slower than ``still_dps`` for ``still_min_s``."""
    g = float(p["grid_hz"])
    still = ((angular_speed_dps(qc, g, p["smooth_s"]) < p["still_dps"])
             & (angular_speed_dps(qm, g, p["smooth_s"]) < p["still_dps"]))
    return true_runs(still, int(round(p["still_min_s"] * g)))


def phase_at(phases: Sequence[tuple[str, float, float]], t0_us: float, t1_us: float) -> str:
    """The phase (name, start µs, end µs) a span overlaps most; ``gap`` outside all."""
    best, name = 0.0, "gap"
    for n, a, b in phases:
        ov = min(t1_us, b) - max(t0_us, a)
        if ov > best:
            best, name = ov, n
    return name


def still_period_rows(tg_us: np.ndarray, qc: np.ndarray, qm: np.ndarray,
                      runs_: Sequence[tuple[int, int]],
                      phases: Sequence[tuple[str, float, float]], grid_hz: float
                      ) -> list[dict[str, Any]]:
    """Per still period: span, duration, phase and the net change of the pure-IMU relative
    orientation ``R_craᵀ R_mta`` (mean of the last against the first 0.5 s)."""
    M = np.swapaxes(_quat_matrix(qc), 1, 2) @ _quat_matrix(qm)
    n_edge = max(1, int(round(0.5 * grid_hz)))
    out = []
    for a, b in runs_:
        rel = M[a:b] @ M[a].T
        ok = np.isfinite(rel.reshape(len(rel), -1)).all(1)
        rv = np.full((len(rel), 3), np.nan)
        if ok.any():
            rv[ok] = Rot.from_matrix(rel[ok]).as_rotvec()
        rv = np.degrees(rv)
        dur = (tg_us[b - 1] - tg_us[a]) * 1e-6
        net = float(np.linalg.norm(rv[-n_edge:].mean(0) - rv[:n_edge].mean(0)))
        out.append({"t0_s": tg_us[a] * 1e-6, "t1_s": tg_us[b - 1] * 1e-6, "duration_s": dur,
                    "phase": phase_at(phases, float(tg_us[a]), float(tg_us[b - 1])),
                    "rel_net_deg": net, "rel_rate_dpm": net / dur * 60.0})
    return out


def block_medians(t_s: np.ndarray, e: np.ndarray, block_s: float
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Block medians of the finite samples (time and value), blocks of ``block_s`` from the
    first finite sample."""
    ok = np.isfinite(e)
    t, e = np.asarray(t_s, float)[ok], np.asarray(e, float)[ok]
    if t.size == 0:
        return np.array([]), np.array([])
    k = np.floor((t - t[0]) / block_s).astype(int)
    _, idx = np.unique(k, return_index=True)
    ends = np.r_[idx[1:], t.size]
    return (np.array([np.median(t[i:j]) for i, j in zip(idx, ends)]),
            np.array([np.median(e[i:j]) for i, j in zip(idx, ends)]))


def _near(t: np.ndarray, spans: Sequence[tuple[float, float]], after: float) -> np.ndarray:
    m = np.zeros(t.size, bool)
    for a, b in spans:
        m |= (t >= a) & (t <= b + after)
    return m


def ols_slope_dpm(t_s: np.ndarray, e: np.ndarray) -> float:
    """OLS slope (deg/min); NaN with < 20 finite samples or < 1 s span."""
    ok = np.isfinite(e)
    if ok.sum() < 20 or np.ptp(t_s[ok]) < 1.0:
        return np.nan
    return float(np.polyfit(t_s[ok] / 60.0, e[ok], 1)[0])


def _nanmedian(x: Sequence[float]) -> float:
    v = np.asarray(list(x), float)
    v = v[np.isfinite(v)]
    return float(np.median(v)) if v.size else np.nan


def step_summary(t_s: np.ndarray, e: np.ndarray, still: Sequence[tuple[float, float]],
                 p: dict[str, Any]) -> dict[str, Any]:
    """Steps of the block-median error trace (change between consecutive blocks at most
    ``max_gap_s`` apart above max(``step_min_deg``, ``step_k`` robust SDs)), how many lie in
    or up to ``after_s`` after a still period against the share expected without alignment,
    and the drift (deg/min) inside the still periods (median of per-period slopes), outside
    them and over the whole trace."""
    t_s, e = np.asarray(t_s, float), np.asarray(e, float)
    tb, eb = block_medians(t_s, e, float(p["block_s"]))
    out: dict[str, Any] = {"n_samples": int(np.isfinite(e).sum())}
    if tb.size < 3:
        return {**out, "n_changes": 0}
    dt, de = np.diff(tb), np.diff(eb)
    tm = 0.5 * (tb[1:] + tb[:-1])
    keep = dt <= p["max_gap_s"]
    de, tm = de[keep], tm[keep]
    sd = 1.4826 * float(np.median(np.abs(de - np.median(de)))) if de.size else np.nan
    thr = max(p["step_min_deg"], p["step_k"] * sd) if np.isfinite(sd) else p["step_min_deg"]
    step = np.abs(de) > thr
    near = _near(tm, still, float(p["after_s"]))
    inside = _near(t_s, still, 0.0)
    return {**out, "n_changes": int(de.size), "change_sd_deg": sd, "step_threshold_deg": thr,
            "n_err_steps": int(step.sum()), "n_err_steps_aligned": int((step & near).sum()),
            "share_aligned_expected": float(near.mean()) if near.size else np.nan,
            "max_step_deg": float(np.max(np.abs(de))) if de.size else np.nan,
            "max_step_aligned_deg": float(np.max(np.abs(de[near]))) if near.any() else np.nan,
            "drift_in_still_dpm": _nanmedian(ols_slope_dpm(t_s[m], e[m]) for m in (
                (t_s >= a) & (t_s <= b) for a, b in still)),
            "drift_outside_dpm": ols_slope_dpm(t_s[~inside], e[~inside]),
            "drift_all_dpm": ols_slope_dpm(t_s, e),
            "still_share_of_trace": float(inside[np.isfinite(e)].mean())}


def phase_jumps(t_s: np.ndarray, e: np.ndarray, spans: Sequence[tuple[str, float, float]],
                still: Sequence[tuple[float, float]], drift_dpm: float, jump_s: float
                ) -> list[dict[str, Any]]:
    """Per pair of consecutive phases with >= 5 error samples: the jump of the trace across
    the boundary (median of the first ``jump_s`` of the next phase minus the last ``jump_s``
    of the previous one), the jump the drift slope predicts over the gap, their difference,
    the still time in the gap and the within-window SD."""
    t_s, e = np.asarray(t_s, float), np.asarray(e, float)
    seg = []
    for name, a, b in spans:
        m = (t_s >= a) & (t_s <= b) & np.isfinite(e)
        if m.sum() >= 5:
            seg.append((name, t_s[m], e[m]))
    out = []
    for (n0, t0, e0), (n1, t1, e1) in zip(seg[:-1], seg[1:]):
        w0, w1 = e0[t0 >= t0[-1] - jump_s], e1[t1 <= t1[0] + jump_s]
        gap = float(t1[0] - t0[-1])
        st = sum(max(0.0, min(b, t1[0]) - max(a, t0[-1])) for a, b in still)
        jump = float(np.median(w1) - np.median(w0))
        exp = drift_dpm * gap / 60.0 if np.isfinite(drift_dpm) else np.nan
        out.append({"from_phase": n0, "to_phase": n1, "gap_s": gap, "still_in_gap_s": st,
                    "jump_deg": jump, "expected_deg": exp, "residual_deg": jump - exp,
                    "window_sd_deg": float(np.std(np.r_[w0 - np.median(w0),
                                                        w1 - np.median(w1)]))})
    return out


def edge_change_deg(t_s: np.ndarray, e: np.ndarray) -> float:
    """Error change over a still period: median of the last 0.5 s minus the first 0.5 s."""
    ok = np.isfinite(e)
    if ok.sum() < 10:
        return np.nan
    t, e = np.asarray(t_s, float)[ok], np.asarray(e, float)[ok]
    a, b = e[t <= t[0] + 0.5], e[t >= t[-1] - 0.5]
    return float(np.median(b) - np.median(a)) if a.size and b.size else np.nan


def dyncal_trace(pid: str, tg_us: np.ndarray, qc: np.ndarray, qm: np.ndarray,
                 phases: Sequence[tuple[str, float, float]], t_s: np.ndarray, e: np.ndarray,
                 trace: str, params: dict[str, Any] | None = None
                 ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The dynamic-calibration audit of one take: IMU grid ``tg_us`` with the head / jaw
    quaternions (wxyz), its phases (name, start µs, end µs) and an error trace ``e`` at
    ``t_s`` (motion removed by a reference). Returns (summary, still periods, phase
    transitions)."""
    p = {**DEFAULTS, **dict(params or {})}
    runs_ = still_runs(qc, qm, p)
    prow = still_period_rows(tg_us, qc, qm, runs_, phases, float(p["grid_hz"]))
    still = [(r["t0_s"], r["t1_s"]) for r in prow]
    t_s, e = np.asarray(t_s, float), np.asarray(e, float)
    tr = step_summary(t_s, e, still, p)
    spans = [(n, a * 1e-6, b * 1e-6) for n, a, b in phases]
    jumps = phase_jumps(t_s, e, spans, still, tr.get("drift_all_dpm", np.nan), float(p["jump_s"]))
    summ = {"pid": pid, "trace": trace, **tr,
            "max_abs_residual_jump_deg": float(np.max(np.abs([j["residual_deg"] for j in jumps])))
            if jumps else np.nan,
            "n_still": len(prow), "still_s": float(sum(r["duration_s"] for r in prow)),
            "rel_rate_median_dpm": float(np.median([r["rel_rate_dpm"] for r in prow]))
            if prow else np.nan}
    per = []
    for r in prow:
        sel = (t_s >= r["t0_s"]) & (t_s <= r["t1_s"])
        per.append({"pid": pid, **r, "err_change_deg": edge_change_deg(t_s[sel], e[sel])})
    return (pd.DataFrame([summ]), pd.DataFrame(per),
            pd.DataFrame([{"pid": pid, **j} for j in jumps]))


def imu_grid(session: Any, grid_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(grid µs, head quaternions, jaw quaternions) of the sanitized IMU streams on their
    common span (``biomech.imu.preprocess.load``), or None."""
    from biomech.imu.preprocess import load

    c, m = load(session, "cra"), load(session, "mta")
    if c is None or m is None:
        return None
    t0, t1 = max(c.t_q[0], m.t_q[0]), min(c.t_q[-1], m.t_q[-1])
    tg = np.arange(t0, t1, int(round(1e6 / grid_hz)), dtype=np.int64)
    return tg, c.quat_at(tg), m.quat_at(tg)
