# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical ↔ IMU angular-speed lag. Catches a constant frame↔trigger offset
that camera affine residuals cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from jtsmm.imu.mounting import SLOT_POSE

# Physical lag (fusion latency, exposure midpoint vs trigger edge) is tens of ms.
LAG_WARN_US = 40_000
LAG_ERR_US = 120_000
# Search span: one bad anchor is typically a handful of trigger periods.
MAX_LAG_US = 1_500_000
LAG_STEP_US = 2_500
# Below these the take is too static / too poorly tracked to conclude anything.
MIN_PAIRS = 60
MIN_MEDIAN_SPEED_DEG_S = 5.0
MIN_CORRELATION = 0.5


@dataclass
class LagEstimate:
    slot: str
    lag_us: float
    correlation: float
    n_samples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "lag_ms": round(self.lag_us / 1000.0, 1),
            "correlation": round(self.correlation, 3),
            "n": int(self.n_samples),
        }


def _pose_speed(t_us: np.ndarray, poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid-interval times and |ω| (deg/s) from consecutive valid pose rotations."""
    t = np.asarray(t_us, dtype=np.float64)
    p = np.asarray(poses, dtype=np.float64)
    if p.ndim != 3 or p.shape[0] < 3:
        return np.zeros(0), np.zeros(0)
    R = p[:, :3, :3]
    finite = np.isfinite(R).all(axis=(1, 2))
    valid = np.zeros(R.shape[0], dtype=bool)
    if np.any(finite):
        valid[finite] = np.abs(np.linalg.det(R[finite])) > 0.5
    R = np.where(finite[:, None, None], R, np.eye(3))
    # R[i]^T @ R[i+1]
    M = np.einsum("nji,njk->nik", R[:-1], R[1:])
    tr = np.trace(M, axis1=1, axis2=2)
    ang = np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))
    dt = np.diff(t) / 1e6
    pair_ok = valid[:-1] & valid[1:] & np.isfinite(ang) & (dt > 0)
    if not np.any(pair_ok):
        return np.zeros(0), np.zeros(0)
    # Drop frame drops: keep intervals near the nominal period.
    med_dt = float(np.median(dt[pair_ok]))
    pair_ok &= (dt > 0.5 * med_dt) & (dt < 1.8 * med_dt)
    if not np.any(pair_ok):
        return np.zeros(0), np.zeros(0)
    t_mid = 0.5 * (t[:-1] + t[1:])
    return t_mid[pair_ok], (ang[pair_ok] / dt[pair_ok])


def _quat_speed(t_us: np.ndarray, quats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mid-interval times and |ω| (deg/s) from consecutive unit quaternions."""
    t = np.asarray(t_us, dtype=np.float64)
    q = np.asarray(quats, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != 4 or q.shape[0] < 3:
        return np.zeros(0), np.zeros(0)
    norm = np.linalg.norm(q, axis=1)
    ok = np.isfinite(norm) & (norm > 1e-6)
    if not np.any(ok):
        return np.zeros(0), np.zeros(0)
    qn = np.zeros_like(q)
    qn[ok] = q[ok] / norm[ok, None]
    dot = np.abs(np.sum(qn[:-1] * qn[1:], axis=1))
    ang = 2.0 * np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    dt = np.diff(t) / 1e6
    pair_ok = ok[:-1] & ok[1:] & np.isfinite(ang) & (dt > 0)
    if not np.any(pair_ok):
        return np.zeros(0), np.zeros(0)
    t_mid = 0.5 * (t[:-1] + t[1:])
    return t_mid[pair_ok], (ang[pair_ok] / dt[pair_ok])


def _boxcar(x: np.ndarray, width: int) -> np.ndarray:
    if width < 2 or x.size < width:
        return x
    kernel = np.ones(int(width), dtype=np.float64) / float(width)
    return np.convolve(x, kernel, mode="same")


def estimate_lag(
    t_opt_us: np.ndarray,
    poses: np.ndarray,
    t_imu_us: np.ndarray,
    quats: np.ndarray,
    *,
    slot: str = "imu_mta",
    max_lag_us: int = MAX_LAG_US,
    step_us: int = LAG_STEP_US,
) -> LagEstimate | None:
    """Lag (µs) to add to optical times so they line up with the IMU.

    Positive means the optical rows are stamped *earlier* than the motion they
    show. Returns None when the take is too static or too sparse to conclude.
    """
    t_o, s_o = _pose_speed(t_opt_us, poses)
    t_i, s_i = _quat_speed(t_imu_us, quats)
    if t_o.size < MIN_PAIRS or t_i.size < MIN_PAIRS:
        return None
    # Match bandwidth: the IMU runs ~10x the camera rate.
    ratio = max(1, int(round(t_i.size / max(t_o.size, 1))))
    s_i = _boxcar(s_i, ratio)
    order = np.argsort(t_i)
    t_i, s_i = t_i[order], s_i[order]
    # "Too static" is judged on the IMU's angular speed over the optical span.
    # The optical speed is differentiated pose noise: ~1 deg of PnP jitter per
    # frame at 30 Hz already reads as ~30 deg/s on a head that is not moving,
    # which let static takes through to a meaningless correlation.
    span = (t_i >= t_o[0]) & (t_i <= t_o[-1])
    if int(span.sum()) < MIN_PAIRS or float(np.median(s_i[span])) < MIN_MEDIAN_SPEED_DEG_S:
        return None

    o_c = s_o - s_o.mean()
    o_norm = float(np.sqrt(np.sum(o_c * o_c)))
    if o_norm < 1e-9:
        return None

    best_lag = 0.0
    best_corr = -2.0
    curve: list[tuple[float, float]] = []
    for lag in range(-int(max_lag_us), int(max_lag_us) + 1, int(step_us)):
        sampled = np.interp(
            t_o + float(lag), t_i, s_i, left=np.nan, right=np.nan
        )
        good = np.isfinite(sampled)
        if int(good.sum()) < MIN_PAIRS:
            continue
        a = o_c[good]
        a = a - a.mean()
        b = sampled[good] - sampled[good].mean()
        den = float(np.sqrt(np.sum(a * a)) * np.sqrt(np.sum(b * b)))
        if den < 1e-12:
            continue
        corr = float(np.sum(a * b) / den)
        curve.append((float(lag), corr))
        if corr > best_corr:
            best_corr = corr
            best_lag = float(lag)
    if not curve or best_corr < MIN_CORRELATION:
        return None

    # Parabolic refinement on the correlation peak.
    lags = np.array([c[0] for c in curve])
    corrs = np.array([c[1] for c in curve])
    k = int(np.argmax(corrs))
    if 0 < k < len(curve) - 1:
        y0, y1, y2 = corrs[k - 1], corrs[k], corrs[k + 1]
        den = y0 - 2.0 * y1 + y2
        if abs(den) > 1e-12:
            delta = 0.5 * (y0 - y2) / den
            if abs(delta) <= 1.0:
                best_lag = float(lags[k] + delta * step_us)
    return LagEstimate(
        slot=slot, lag_us=best_lag, correlation=best_corr, n_samples=int(t_o.size)
    )


def check_optical_imu_sync(f: Any, report: Any) -> None:
    """Flag a constant optical↔IMU timing offset (bad frame↔trigger anchor)."""
    og = f.get("optical")
    if og is None or "t_master_us" not in og:
        return
    t_opt = np.asarray(og["t_master_us"], dtype=np.int64)
    if t_opt.size < MIN_PAIRS:
        return

    results: dict[str, Any] = {}
    for slot, pose_ds in SLOT_POSE.items():
        grp = f.get(slot)
        if grp is None or pose_ds not in og or "quat" not in grp:
            continue
        if "t_master_us_quat" not in grp:
            continue
        est = estimate_lag(
            t_opt,
            np.asarray(og[pose_ds]),
            np.asarray(grp["t_master_us_quat"], dtype=np.int64),
            np.asarray(grp["quat"], dtype=np.float64),
            slot=slot,
        )
        if est is None:
            results[slot] = {"inconclusive": True}
            continue
        results[slot] = est.as_dict()
        lag_ms = est.lag_us / 1000.0
        if abs(est.lag_us) > LAG_ERR_US:
            report.add_error(
                f"{slot}: optical vs IMU timing offset {lag_ms:+.0f} ms "
                f"(corr {est.correlation:.2f}) — camera frames are paired with the "
                "wrong TRIGGER_EVENT; poses are unusable for kinematics"
            )
        elif abs(est.lag_us) > LAG_WARN_US:
            report.add_warn(
                f"{slot}: optical vs IMU timing offset {lag_ms:+.0f} ms "
                f"(corr {est.correlation:.2f})"
            )
    if results:
        report.summary["optical_imu_sync"] = results
        # Silence here used to read as a pass. It is not: an inconclusive
        # cross-check means the take was too static, too poorly tracked or too
        # short to correlate, so the frame<->trigger association was never
        # independently confirmed -- exactly the case where a constant offset
        # survives. Say so, and point at the gate that still applies.
        if all(r.get("inconclusive") for r in results.values()):
            report.add_warn(
                "optical vs IMU timing could not be checked (too static, too few "
                "paired samples, or correlation too low) -- the frame↔trigger "
                "association is only covered by the camera assignment residual"
            )
