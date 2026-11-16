# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Static-hold precision of a tracked pose (§5.1.1 / §5.1.2).

Precision is the spread of the samples about their own temporal mean:

* translation: ``r_i = ‖p_i − p̄‖`` (mm), reported as ``r_mean ± σ_r``;
* rotation: ``θ_i = ∠(R̄, R_i)`` geodesic (deg) to the chordal mean ``R̄``,
  reported as ``r_mean ± σ_r``.

The SO(3) maths is :mod:`optical.calibration.geometry` (atan2 angle, accurate
for the sub-degree jitter measured here).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from optical.calibration.geometry import geodesic_angle_rad, mean_rotation


def residual_stats(values: np.ndarray) -> dict[str, Any]:
    """``{"r_mean", "sigma_r", "n"}`` of a 1-D residual series (population σ); NaN when n < 2."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size < 2:
        return {"r_mean": float("nan"), "sigma_r": float("nan"), "n": int(v.size)}
    return {"r_mean": float(np.mean(v)), "sigma_r": float(np.std(v)), "n": int(v.size)}


def translation_residual_mm(ts: np.ndarray) -> dict[str, Any]:
    """Residual stats of ``‖p_i − p̄‖`` for (N, 3) positions."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[0] < 2:
        return residual_stats(np.array([]))
    return residual_stats(np.linalg.norm(ts - ts.mean(axis=0), axis=1))


def rotation_residual_deg(Rs: np.ndarray) -> dict[str, Any]:
    """Residual stats of the geodesic angle (deg) of each (N, 3, 3) rotation to their mean."""
    Rs = np.asarray(Rs, dtype=np.float64)
    if Rs.ndim != 3 or Rs.shape[0] < 2:
        return residual_stats(np.array([]))
    R_bar = mean_rotation(Rs)
    return residual_stats(np.degrees([geodesic_angle_rad(R_bar, R) for R in Rs]))


def mean_pose_from_samples(ts: np.ndarray, Rs: np.ndarray) -> dict[str, Any]:
    """Temporal mean pose (arithmetic ``t``, chordal ``R``); informational only."""
    ts = np.asarray(ts, dtype=np.float64)
    Rs = np.asarray(Rs, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[0] < 1:
        return {"t_mm": None, "R": None, "n": 0}
    R_bar = mean_rotation(Rs) if Rs.ndim == 3 and Rs.shape[0] >= 1 else None
    return {
        "t_mm": [float(x) for x in ts.mean(axis=0)],
        "R": R_bar.tolist() if R_bar is not None else None,
        "n": int(ts.shape[0]),
    }


__all__ = ["mean_pose_from_samples", "residual_stats", "rotation_residual_deg",
           "translation_residual_mm"]
