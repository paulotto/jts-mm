# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Pivot (cone) calibration of a removable tip offset via OLS.

Seat the tip in a fixed cone and capture diverse orientations of the tracked
body. The tip is stationary in world, so for each pose ``T_world←body``:

    R_i @ tip_body + t_i = p_world

Stacking ``[R_i | -I] [tip_body; p_world] = -t_i`` and solving with ordinary
least squares yields the tip offset in the body frame (mm).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from optical.tracking.pose import is_valid_pose


@dataclass(frozen=True)
class PivotCalibrationResult:
    """OLS tip offset and fixed cone point in world (mm)."""

    tip_body_mm: np.ndarray
    pivot_world_mm: np.ndarray
    rms_mm: float
    n_poses: int
    residual_norm: float
    tip_length_mm: float = 0.0
    max_err_mm: float = 0.0
    median_err_mm: float = 0.0
    cond_A: float = float("nan")
    per_pose_err_mm: tuple[float, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "tip_body_mm": [float(x) for x in self.tip_body_mm.tolist()],
            "pivot_world_mm": [float(x) for x in self.pivot_world_mm.tolist()],
            "rms_mm": float(self.rms_mm),
            "n_poses": int(self.n_poses),
            "residual_norm": float(self.residual_norm),
            "tip_length_mm": float(self.tip_length_mm),
            "max_err_mm": float(self.max_err_mm),
            "median_err_mm": float(self.median_err_mm),
            "cond_A": float(self.cond_A),
            "per_pose_err_mm": [float(x) for x in self.per_pose_err_mm],
        }


def solve_pivot_ols(poses_world_body: Sequence[np.ndarray]) -> PivotCalibrationResult:
    """Solve tip offset from ≥3 SE(3) body poses with the tip fixed in world."""
    valid = [np.asarray(T, dtype=np.float64) for T in poses_world_body if is_valid_pose(T)]
    if len(valid) < 3:
        raise ValueError(f"pivot OLS needs ≥3 valid poses, got {len(valid)}")

    n = len(valid)
    A = np.zeros((3 * n, 6), dtype=np.float64)
    b = np.zeros(3 * n, dtype=np.float64)
    for i, T in enumerate(valid):
        R = T[:3, :3]
        t = T[:3, 3]
        A[3 * i : 3 * i + 3, 0:3] = R
        A[3 * i : 3 * i + 3, 3:6] = -np.eye(3)
        b[3 * i : 3 * i + 3] = -t

    x, residuals, _rank, s = np.linalg.lstsq(A, b, rcond=None)
    tip = x[:3].copy()
    pivot = x[3:].copy()

    errs = np.asarray(
        [np.linalg.norm(T[:3, :3] @ tip + T[:3, 3] - pivot) for T in valid],
        dtype=np.float64,
    )
    rms = float(np.sqrt(np.mean(errs * errs)))
    residual_norm = (
        float(np.sqrt(float(residuals[0])))
        if len(residuals)
        else float(np.linalg.norm(A @ x - b))
    )
    cond = float(s[0] / s[-1]) if len(s) and s[-1] > 0 else float("inf")
    return PivotCalibrationResult(
        tip_body_mm=tip,
        pivot_world_mm=pivot,
        rms_mm=rms,
        n_poses=n,
        residual_norm=residual_norm,
        tip_length_mm=float(np.linalg.norm(tip)),
        max_err_mm=float(np.max(errs)),
        median_err_mm=float(np.median(errs)),
        cond_A=cond,
        per_pose_err_mm=tuple(float(e) for e in errs.tolist()),
    )


def apply_tip_offset(T_world_body: np.ndarray, tip_body_mm: np.ndarray) -> np.ndarray:
    """Return body pose with translation moved to the tip (orientation unchanged)."""
    T = np.asarray(T_world_body, dtype=np.float64).copy()
    tip = np.asarray(tip_body_mm, dtype=np.float64).reshape(3)
    T[:3, 3] = T[:3, :3] @ tip + T[:3, 3]
    return T
