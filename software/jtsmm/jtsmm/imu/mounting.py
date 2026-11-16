# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU–array mounting rotation (Park–Martin AX=XB). Translation is left at zero."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np

from jtsmm.imu.kinematics import quat_slerp

__all__ = [
    "MAX_RMS_DEG",
    "MOUNTING_FILE_KIND",
    "embed_mounting",
    "load_mounting_file",
    "update_mounting_file",
    "MountingFitError",
    "MountingFitResult",
    "PHASE_BY_SLOT",
    "SELECT_N",
    "SLOT_POSE",
    "fit_mounting_rotation",
    "fit_session_slot",
    "geodesic_deg",
    "load_mounting_rotation",
    "MAX_PAIR_GAP_US",
    "pair_optical_imu",
    "park_martin_axxb",
    "project_so3",
    "read_phase_windows",
    "relative_axis_coverage",
    "select_so3_fps",
    "write_mounting_transform",
]

MAX_RMS_DEG = 3.0
MIN_REL_DEG = 8.0
MIN_SAMPLES = 12
MIN_PAIRS = 8
MIN_AXIS_RATIO = 0.12
MIN_AXIS3_RATIO = 0.28
SELECT_N = 40
MIN_OMEGA_DEG_S = 10.0
MAX_OMEGA_DEG_S = 80.0
OMEGA_FALLBACK_DEG_S = 150.0
MAX_JUMP_DEG_S = 1000.0
MIN_TAGS = 2
MAX_KABSCH_RMS_MM = 4.0
MAX_PNP_RMS_PX = 6.0
# Never SLERP an IMU orientation across a hole in the quaternion stream wider
# than this (or extrapolate this far past its ends): during a 10-80 deg/s tumble
# a 50 ms hole is already a few degrees of invented rotation.
MAX_PAIR_GAP_US = 50_000
PHASE_BY_SLOT = {"imu_mta": "imu_mount_mta", "imu_cra": "imu_mount_cra"}
SLOT_POSE = {"imu_mta": "pose_mta", "imu_cra": "pose_cra"}
SLOT_QUALITY = {"imu_mta": "mta", "imu_cra": "cra"}


class MountingFitError(ValueError):
    """Unusable capture (too few samples, nod-only, or RMS above acceptance)."""


@dataclass
class MountingFitResult:
    """Rotation-only mounting fit for one IMU slot."""

    slot: str
    R_mount: np.ndarray
    T: np.ndarray
    rms_deg: float
    n_pairs: int
    n_samples: int
    n_selected: int | None = None
    phase: str | None = None
    accepted: bool = True
    error: str | None = None
    axes_3d: bool | None = None
    axis_ratio_opt: tuple[float, float] | None = None
    axis_ratio_imu: tuple[float, float] | None = None

    def summary(self) -> str:
        ph = f" phase={self.phase}" if self.phase else ""
        status = "ok" if self.accepted else "REJECTED"
        used = self.n_selected if self.n_selected is not None else self.n_samples
        extra = f", {used} used" if used != self.n_samples else ""
        axes = _format_axes_coverage(self.axes_3d, self.axis_ratio_opt, self.axis_ratio_imu)
        return (
            f"{self.slot}{ph}: {status} RMS {self.rms_deg:.2f}° "
            f"(n={self.n_samples} frames{extra}, {self.n_pairs} pairs){axes}"
        )

    def to_slot_dict(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "accepted": bool(self.accepted),
            "rms_deg": float(self.rms_deg),
            "n_pairs": int(self.n_pairs),
            "n_samples": int(self.n_samples),
            "n_selected": int(self.n_selected) if self.n_selected is not None else int(self.n_samples),
            "phase": self.phase,
            "error": self.error,
            "axes_3d": None if self.axes_3d is None else bool(self.axes_3d),
            "axis_ratio_opt": list(self.axis_ratio_opt) if self.axis_ratio_opt is not None else None,
            "axis_ratio_imu": list(self.axis_ratio_imu) if self.axis_ratio_imu is not None else None,
            "R_mount": np.asarray(self.R_mount, dtype=np.float64).tolist(),
            "T": np.asarray(self.T, dtype=np.float64).tolist(),
        }


def rotation_from_quat(q: np.ndarray | Sequence[float]) -> np.ndarray:
    """Chip/world rotation matrix from a w-first unit quaternion."""
    w, x, y, z = (float(c) for c in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation-vector (axis-angle) of ``R`` ∈ SO(3)."""
    m = np.asarray(R, dtype=np.float64)
    cos_th = float(np.clip((np.trace(m) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(cos_th))
    if th < 1e-9:
        return np.zeros(3, dtype=np.float64)
    if th > np.pi - 1e-4:
        # Near pi the skew part vanishes; use (R + I)/2 ~ a a^T instead. Its
        # column with the largest diagonal is a * a_k, so dividing by sqrt(a_k^2)
        # gives every component with the right RELATIVE sign -- the old
        # per-component sign rules read the wrong off-diagonals and returned
        # (1, 1, 0) for a pi turn about (1, -1, 0). The global sign follows the
        # (small) skew part so the result stays continuous just below pi.
        b = 0.25 * (m + m.T) + 0.5 * np.eye(3)
        k = int(np.argmax(np.diag(b)))
        if b[k, k] <= 1e-12:
            return np.array([th, 0.0, 0.0])
        axis = b[:, k] / np.sqrt(b[k, k])
        skew = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
        if float(np.dot(axis, skew)) < 0.0:
            axis = -axis
        n = float(np.linalg.norm(axis))
        if n < 1e-9:
            return np.array([th, 0.0, 0.0])
        return axis / n * th
    w = np.array(
        [m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]],
        dtype=np.float64,
    )
    return w * (th / (2.0 * np.sin(th)))


def geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(so3_log(R_a.T @ R_b))))


def project_so3(M: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(M)
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def relative_axis_coverage(R_rel: Sequence[np.ndarray]) -> dict[str, Any]:
    """SVD of relative-rotation logs: ``ratio_12 = s1/s0``, ``ratio_13 = s2/s0``."""
    if not R_rel:
        return {
            "sv": [0.0, 0.0, 0.0],
            "ratio_12": 0.0,
            "ratio_13": 0.0,
            "axes_3d": False,
        }
    logs = np.stack([so3_log(R) for R in R_rel])
    s = np.linalg.svd(logs, compute_uv=False)
    s3 = np.zeros(3, dtype=np.float64)
    s3[: min(3, int(s.size))] = s[:3]
    s0 = float(s3[0])
    r12 = float(s3[1] / s0) if s0 > 1e-12 else 0.0
    r13 = float(s3[2] / s0) if s0 > 1e-12 else 0.0
    return {
        "sv": [float(s3[0]), float(s3[1]), float(s3[2])],
        "ratio_12": r12,
        "ratio_13": r13,
        "axes_3d": bool(r12 >= MIN_AXIS_RATIO and r13 >= MIN_AXIS3_RATIO),
    }


def _format_axes_coverage(
    axes_3d: bool | None,
    axis_ratio_opt: tuple[float, float] | None,
    axis_ratio_imu: tuple[float, float] | None,
) -> str:
    if axes_3d is None and axis_ratio_opt is None and axis_ratio_imu is None:
        return ""
    mark = "3/3" if axes_3d else "2/3"
    bits: list[str] = []
    if axis_ratio_opt is not None:
        bits.append(f"opt {axis_ratio_opt[0]:.2f}/{axis_ratio_opt[1]:.2f}")
    if axis_ratio_imu is not None:
        bits.append(f"imu {axis_ratio_imu[0]:.2f}/{axis_ratio_imu[1]:.2f}")
    extra = f" ({', '.join(bits)})" if bits else ""
    hint = "" if axes_3d else " — not all 3 axes; repeat figure-eight, not nod"
    return f"  axes {mark}{extra}{hint}"


def _pose_rotation(T: np.ndarray) -> np.ndarray | None:
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return None
    R = np.asarray(T[:3, :3], dtype=np.float64)
    if not np.all(np.isfinite(R)):
        return None
    det = float(np.linalg.det(R))
    if abs(det) < 0.5:
        return None
    return project_so3(R)


def _enforce_hemisphere(quats: np.ndarray) -> np.ndarray:
    out = np.asarray(quats, dtype=np.float64).copy()
    if out.shape[0] == 0:
        return out
    for i in range(1, out.shape[0]):
        if float(np.dot(out[i], out[i - 1])) < 0.0:
            out[i] = -out[i]
    return out


def _geodesic_to_many(R: np.ndarray, Rs: np.ndarray) -> np.ndarray:
    """Geodesic degrees from one rotation to each row of ``Rs`` (N, 3, 3)."""
    rel = np.einsum("ab,nbc->nac", R.T, Rs)
    tr = rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]
    return np.degrees(np.arccos(np.clip((tr - 1.0) * 0.5, -1.0, 1.0)))


def select_so3_fps(Rs: np.ndarray, k: int) -> np.ndarray:
    """Farthest-point sample of ``k`` indices on SO(3), sorted."""
    mats = np.asarray(Rs, dtype=np.float64)
    n = int(mats.shape[0])
    if n <= k:
        return np.arange(n, dtype=int)
    mean_R = project_so3(np.sum(mats, axis=0))
    start = int(np.argmax(_geodesic_to_many(mean_R, mats)))
    selected = [start]
    dmin = _geodesic_to_many(mats[start], mats)
    dmin[start] = -np.inf
    while len(selected) < k:
        i = int(np.argmax(dmin))
        if not np.isfinite(dmin[i]):
            break
        selected.append(i)
        dmin = np.minimum(dmin, _geodesic_to_many(mats[i], mats))
        dmin[i] = -np.inf
    return np.sort(np.asarray(selected, dtype=int))


def _omega_deg_s(t_us: np.ndarray, Rs: np.ndarray) -> np.ndarray:
    n = int(Rs.shape[0])
    w = np.zeros(n, dtype=np.float64)
    if n < 2:
        return w
    t = np.asarray(t_us, dtype=np.float64)
    for i in range(n):
        j0 = 0 if i == 0 else i - 1
        j1 = n - 1 if i == n - 1 else i + 1
        if j0 == j1:
            continue
        dt = (t[j1] - t[j0]) * 1e-6
        if dt <= 1e-9:
            continue
        w[i] = geodesic_deg(Rs[j0], Rs[j1]) / dt
    return w


def _jump_mask(t_us: np.ndarray, Rs: np.ndarray) -> np.ndarray:
    """Drop frames whose geodesic rate from the last kept pose is a flip/glitch."""
    n = int(Rs.shape[0])
    keep = np.ones(n, dtype=bool)
    if n < 2:
        return keep
    t = np.asarray(t_us, dtype=np.float64)
    last = 0
    for i in range(1, n):
        dt = (t[i] - t[last]) * 1e-6
        if dt <= 1e-9:
            keep[i] = False
            continue
        if geodesic_deg(Rs[last], Rs[i]) / dt > MAX_JUMP_DEG_S:
            keep[i] = False
            continue
        last = i
    return keep


def _quality_mask(
    n: int,
    *,
    n_tags: np.ndarray | None,
    n_cameras: np.ndarray | None,
    rms: np.ndarray | None,
    rms_kind: str,
) -> np.ndarray:
    keep = np.ones(n, dtype=bool)
    if n_cameras is not None and int(n_cameras.shape[0]) == n:
        cams = np.asarray(n_cameras, dtype=np.float64)
        finite = np.isfinite(cams)
        if finite.any() and float(np.nanmax(cams)) >= 2.0:
            keep &= (~finite) | (cams >= 2.0)
    if n_tags is not None and int(n_tags.shape[0]) == n:
        tags = np.asarray(n_tags, dtype=np.float64)
        finite = np.isfinite(tags)
        if finite.any() and float(np.nanmax(tags)) >= MIN_TAGS:
            keep &= (~finite) | (tags >= MIN_TAGS)
    if rms is not None and int(rms.shape[0]) == n:
        vals = np.asarray(rms, dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.any():
            cap = MAX_KABSCH_RMS_MM if "mm" in str(rms_kind).lower() else MAX_PNP_RMS_PX
            keep &= (~finite) | (vals <= cap)
    return keep


def _omega_mask(keep: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """Prefer the slow-tumble band; widen if that would starve Park–Martin."""
    base = np.asarray(keep, dtype=bool)
    in_band = base & (omega >= MIN_OMEGA_DEG_S) & (omega <= MAX_OMEGA_DEG_S)
    if int(in_band.sum()) >= MIN_SAMPLES:
        return in_band
    wide = base & (omega >= MIN_OMEGA_DEG_S) & (omega <= OMEGA_FALLBACK_DEG_S)
    if int(wide.sum()) >= MIN_SAMPLES:
        return wide
    return base


def _select_pair_frames(
    t_us: np.ndarray,
    R_opt: np.ndarray,
    *,
    n_tags: np.ndarray | None = None,
    n_cameras: np.ndarray | None = None,
    rms: np.ndarray | None = None,
    rms_kind: str = "reprojection_px",
    k: int = SELECT_N,
) -> np.ndarray:
    """Quality / rate gate, then farthest-point sample on SO(3)."""
    n = int(R_opt.shape[0])
    keep = _jump_mask(t_us, R_opt)
    keep &= _quality_mask(
        n, n_tags=n_tags, n_cameras=n_cameras, rms=rms, rms_kind=rms_kind
    )
    keep = _omega_mask(keep, _omega_deg_s(t_us, R_opt))
    idx = np.flatnonzero(keep)
    if idx.size < MIN_SAMPLES:
        keep = _quality_mask(
            n, n_tags=n_tags, n_cameras=n_cameras, rms=rms, rms_kind=rms_kind
        )
        idx = np.flatnonzero(keep)
        if idx.size < MIN_SAMPLES:
            idx = np.arange(n, dtype=int)
    picked = select_so3_fps(R_opt[idx], k)
    return np.asarray(idx[picked], dtype=int)


def pair_optical_imu(
    t_opt_us: np.ndarray,
    poses: np.ndarray,
    t_imu_us: np.ndarray,
    quats: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Valid optical rotations and SLERP'd IMU rotations, same length (N, 3, 3).

    Also returns master-clock times and source indices into ``poses``. An
    optical frame is dropped when the IMU samples around it are more than
    ``MAX_PAIR_GAP_US`` apart (or it lies that far outside the IMU stream):
    interpolating across a dropout invents orientation.
    """
    t_opt = np.asarray(t_opt_us, dtype=np.int64)
    t_imu = np.asarray(t_imu_us, dtype=np.int64)
    poses = np.asarray(poses)
    quats = _enforce_hemisphere(np.asarray(quats, dtype=np.float64))
    if t_imu.size < 2 or t_opt.size == 0:
        raise MountingFitError("need optical poses and at least two IMU quaternions")

    R_opt: list[np.ndarray] = []
    R_imu: list[np.ndarray] = []
    t_kept: list[int] = []
    src: list[int] = []
    for i, t in enumerate(t_opt):
        R = _pose_rotation(poses[i])
        if R is None:
            continue
        if t <= t_imu[0]:
            if int(t_imu[0]) - int(t) > MAX_PAIR_GAP_US // 2:
                continue
            q = quats[0]
        elif t >= t_imu[-1]:
            if int(t) - int(t_imu[-1]) > MAX_PAIR_GAP_US // 2:
                continue
            q = quats[-1]
        else:
            j = int(np.searchsorted(t_imu, t, side="right"))
            t0, t1 = int(t_imu[j - 1]), int(t_imu[j])
            if t0 != int(t) and t1 - t0 > MAX_PAIR_GAP_US:
                continue
            u = 0.0 if t1 == t0 else (int(t) - t0) / (t1 - t0)
            q0 = tuple(float(c) for c in quats[j - 1])
            q1 = tuple(float(c) for c in quats[j])
            q = quat_slerp(q0, q1, float(u))  # type: ignore[arg-type]
        R_opt.append(R)
        R_imu.append(rotation_from_quat(q))
        t_kept.append(int(t))
        src.append(i)
    if len(R_opt) < MIN_SAMPLES:
        raise MountingFitError(
            f"only {len(R_opt)} valid paired frames (need ≥{MIN_SAMPLES}); "
            "keep tags visible and tumble about more than one axis"
        )
    return (
        np.stack(R_opt),
        np.stack(R_imu),
        np.asarray(t_kept, dtype=np.int64),
        np.asarray(src, dtype=int),
    )


def _relative_pairs(
    R_opt: np.ndarray, R_imu: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    n = int(R_opt.shape[0])
    A: list[np.ndarray] = []
    B: list[np.ndarray] = []
    min_rad = np.radians(MIN_REL_DEG)
    for i in range(n):
        for j in range(i + 1, n):
            Ai = R_opt[i].T @ R_opt[j]
            Bi = R_imu[i].T @ R_imu[j]
            if np.linalg.norm(so3_log(Ai)) < min_rad:
                continue
            if np.linalg.norm(so3_log(Bi)) < min_rad:
                continue
            A.append(Ai)
            B.append(Bi)
    if len(A) < MIN_PAIRS:
        raise MountingFitError(
            f"only {len(A)} relative-rotation pairs above {MIN_REL_DEG:.0f}° "
            f"(need ≥{MIN_PAIRS}); nod-only / single-axis motion is degenerate — "
            "use a slow figure-eight about ~3 axes"
        )
    return A, B


def park_martin_axxb(
    A: Sequence[np.ndarray], B: Sequence[np.ndarray]
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Park–Martin rotation solution of ``A X = X B``.

    Returns ``(X, rms_deg, imu_axis_coverage)``.
    """
    alphas = [so3_log(a) for a in A]
    betas = [so3_log(b) for b in B]
    imu_cov = relative_axis_coverage(B)
    if imu_cov["ratio_12"] < MIN_AXIS_RATIO:
        raise MountingFitError(
            f"relative IMU rotations are nearly collinear "
            f"(axis ratio {imu_cov['ratio_12']:.3f}); "
            "tumble about more than one axis (figure-eight, not nod-only)"
        )
    H = np.zeros((3, 3), dtype=np.float64)
    for a, b in zip(alphas, betas):
        H += np.outer(a, b)
    X = project_so3(H)
    errs = [geodesic_deg(Ai @ X, X @ Bi) for Ai, Bi in zip(A, B)]
    rms = float(np.sqrt(np.mean(np.square(errs))))
    return X, rms, imu_cov


def fit_mounting_rotation(
    t_opt_us: np.ndarray,
    poses: np.ndarray,
    t_imu_us: np.ndarray,
    quats: np.ndarray,
    *,
    slot: str = "imu_mta",
    phase: str | None = None,
    max_rms_deg: float = MAX_RMS_DEG,
    strict: bool = True,
    n_tags: np.ndarray | None = None,
    n_cameras: np.ndarray | None = None,
    reprojection_rms: np.ndarray | None = None,
    rms_kind: str = "reprojection_px",
) -> MountingFitResult:
    """Estimate ``R_mount = body←chip``; translation of ``T`` is zero.

    Degenerate captures (too few pairs / collinear motion) still raise
    ``MountingFitError``. When ``strict`` is false, an RMS above
    ``max_rms_deg`` is returned with ``accepted=False`` instead of raising.

    Optional ``n_tags`` / ``n_cameras`` / ``reprojection_rms`` must align with
    ``poses``. Stereo sessions drop 1-view rows; then frames in the slow-tumble
    ω band are farthest-point sampled on SO(3) down to ``SELECT_N``.
    """
    R_opt, R_imu, t_kept, src = pair_optical_imu(t_opt_us, poses, t_imu_us, quats)
    n_paired = int(R_opt.shape[0])

    def _take(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        vals = np.asarray(arr)
        if vals.shape[0] != int(np.asarray(poses).shape[0]):
            return None
        return vals[src]

    pick = _select_pair_frames(
        t_kept,
        R_opt,
        n_tags=_take(n_tags),
        n_cameras=_take(n_cameras),
        rms=_take(reprojection_rms),
        rms_kind=rms_kind,
    )
    A, B = _relative_pairs(R_opt[pick], R_imu[pick])
    X, rms, imu_cov = park_martin_axxb(A, B)
    opt_cov = relative_axis_coverage(A)
    axes_3d = bool(opt_cov["axes_3d"] and imu_cov["axes_3d"])
    axis_ratio_opt = (float(opt_cov["ratio_12"]), float(opt_cov["ratio_13"]))
    axis_ratio_imu = (float(imu_cov["ratio_12"]), float(imu_cov["ratio_13"]))
    msg = None
    accepted = True
    if rms > max_rms_deg:
        msg = (
            f"{slot}: mounting residual {rms:.2f}° RMS exceeds {max_rms_deg:.1f}° "
            "(run `jtsmm integrity check` for optical↔IMU lag; else tags / glue)"
        )
        if strict:
            raise MountingFitError(msg)
        accepted = False
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = X
    return MountingFitResult(
        slot=slot,
        R_mount=X,
        T=T,
        rms_deg=rms,
        n_pairs=len(A),
        n_samples=n_paired,
        n_selected=int(pick.size),
        phase=phase,
        accepted=accepted,
        error=msg,
        axes_3d=axes_3d,
        axis_ratio_opt=axis_ratio_opt,
        axis_ratio_imu=axis_ratio_imu,
    )


def _as_str(v: Any) -> str:
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.ndarray):
        if v.shape == ():
            return _as_str(v.item())
        return str(v)
    return str(v)


def read_phase_windows(path: Path | str) -> dict[str, tuple[int, int]]:
    """Map ``/phases/name`` → ``(t_start_us, t_end_us)``."""
    out: dict[str, tuple[int, int]] = {}
    with h5py.File(path, "r") as f:
        if "phases" not in f:
            return out
        g = f["phases"]
        if "name" not in g or "t_start_us" not in g:
            return out
        names = [_as_str(n) for n in g["name"][()]]
        t0 = np.asarray(g["t_start_us"][()], dtype=np.uint64)
        t1 = np.asarray(g["t_end_us"][()], dtype=np.uint64)
        for name, a, b in zip(names, t0, t1):
            out[name] = (int(a), int(b))
    return out


def _mask_time(t: np.ndarray, t0: int | None, t1: int | None) -> np.ndarray:
    if t0 is None or t1 is None:
        return np.ones(t.shape[0], dtype=bool)
    return (t >= t0) & (t <= t1)


def _read_optical_quality(
    f: Any, slot: str, *, n: int
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    """``(n_tags, n_cameras, rms, rms_kind)`` aligned with ``/optical`` rows."""
    body = SLOT_QUALITY.get(slot)
    qg = f.get("optical/quality") if "optical" in f else None
    rms_kind = "reprojection_px"
    if qg is None or body is None:
        return None, None, None, rms_kind
    raw_kind = qg.attrs.get("rms_kind") if hasattr(qg, "attrs") else None
    if raw_kind is not None:
        rms_kind = str(raw_kind)
    sg = qg.get(body)
    if sg is None:
        return None, None, None, rms_kind

    def _col(name: str) -> np.ndarray | None:
        if name not in sg:
            return None
        arr = np.asarray(sg[name])
        if arr.shape[0] != n:
            return None
        return arr

    return _col("n_tags"), _col("n_cameras"), _col("reprojection_rms"), rms_kind


def fit_session_slot(
    path: Path | str,
    slot: str,
    *,
    t0_us: int | None = None,
    t1_us: int | None = None,
    phase: str | None = None,
    max_rms_deg: float = MAX_RMS_DEG,
    strict: bool = True,
) -> MountingFitResult:
    pose_ds = SLOT_POSE.get(slot)
    if pose_ds is None:
        raise MountingFitError(f"unknown IMU slot {slot!r}")
    with h5py.File(path, "r") as f:
        if "optical" not in f or pose_ds not in f["optical"]:
            raise MountingFitError(
                f"{path}: missing /optical/{pose_ds}. Cameras may have recorded, "
                "but ArUco tracking was off (session.optical_track defaults to false "
                "unless set). Re-record with `jtsmm imu calibrate --mode record`, which forces "
                "optical tracking and skips EMG."
            )
        if "t_master_us" not in f["optical"]:
            raise MountingFitError(
                f"{path}: missing /optical/t_master_us. Re-record with `jtsmm imu calibrate --mode record`."
            )
        if slot not in f or "quat" not in f[slot]:
            raise MountingFitError(f"{path}: missing /{slot}/quat")
        t_opt = np.asarray(f["optical"]["t_master_us"], dtype=np.uint64)
        poses = np.asarray(f["optical"][pose_ds])
        t_imu = np.asarray(f[slot]["t_master_us_quat"], dtype=np.uint64)
        quats = np.asarray(f[slot]["quat"], dtype=np.float64)
        n_tags, n_cams, rms, rms_kind = _read_optical_quality(f, slot, n=int(t_opt.shape[0]))
    m_opt = _mask_time(t_opt, t0_us, t1_us)
    m_imu = _mask_time(t_imu, t0_us, t1_us)
    return fit_mounting_rotation(
        t_opt[m_opt],
        poses[m_opt],
        t_imu[m_imu],
        quats[m_imu],
        slot=slot,
        phase=phase,
        max_rms_deg=max_rms_deg,
        strict=strict,
        n_tags=None if n_tags is None else n_tags[m_opt],
        n_cameras=None if n_cams is None else n_cams[m_opt],
        reprojection_rms=None if rms is None else rms[m_opt],
        rms_kind=rms_kind,
    )


def load_mounting_rotation(path: Path | str, slot: str) -> np.ndarray | None:
    """Accepted ``R_mount`` from ``/calibration/<slot>/mounting_transform``, or None.

    None unless the group says ``accepted=True`` and the block is a proper
    rotation: a fit that failed its RMS gate must never be applied (it used to
    be, silently, by the replay viewer and visualize_session).
    """
    key = slot if slot.startswith("imu_") else f"imu_{slot}"
    with h5py.File(path, "r") as f:
        g = f.get(f"calibration/{key}")
        if g is None or "mounting_transform" not in g:
            return None
        if not bool(g.attrs.get("accepted", False)):
            return None
        T = np.asarray(g["mounting_transform"], dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return None
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0.5:
        return None
    return R


_MOUNT_ATTRS = (
    "rms_deg", "n_pairs", "n_samples", "n_selected", "axes_3d", "axis_ratio_opt",
    "axis_ratio_imu", "accepted", "date", "phase", "error", "source", "source_session",
    "embedded",
)


def write_mounting_transform(
    path: Path | str,
    result: MountingFitResult,
) -> None:
    """Write ``/calibration/<slot>/mounting_transform`` (4×4, translation zero)."""
    path = Path(path)
    with h5py.File(path, "a") as f:
        g = f.require_group(f"calibration/{result.slot}")
        if "mounting_transform" in g:
            del g["mounting_transform"]
        # A rewrite must not inherit the previous fit's optional attrs (a stale
        # "error" or "phase" on a new, accepted fit).
        for key in _MOUNT_ATTRS:
            if key in g.attrs:
                del g.attrs[key]
        ds = g.create_dataset("mounting_transform", data=np.asarray(result.T, dtype=np.float64))
        ds.attrs["translation"] = "zero"
        ds.attrs["note"] = (
            "Rotation body←chip from Park–Martin AX=XB; translation reserved "
            "for a later lever-arm fit (not estimated here)."
        )
        g.attrs["rms_deg"] = float(result.rms_deg)
        g.attrs["n_pairs"] = int(result.n_pairs)
        g.attrs["n_samples"] = int(result.n_samples)
        if result.n_selected is not None:
            g.attrs["n_selected"] = int(result.n_selected)
        if result.axes_3d is not None:
            g.attrs["axes_3d"] = bool(result.axes_3d)
        if result.axis_ratio_opt is not None:
            g.attrs["axis_ratio_opt"] = np.asarray(result.axis_ratio_opt, dtype=np.float64)
        if result.axis_ratio_imu is not None:
            g.attrs["axis_ratio_imu"] = np.asarray(result.axis_ratio_imu, dtype=np.float64)
        g.attrs["accepted"] = bool(result.accepted)
        g.attrs["date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if result.phase:
            g.attrs["phase"] = result.phase
        if result.error:
            g.attrs["error"] = str(result.error)


# -- the stable mounting file (like the CT tip's ct_tip_calibration.json) ------------------
MOUNTING_FILE_KIND = "imu_mounting_calibration"


def update_mounting_file(path: Path | str, records: dict[str, dict[str, Any]],
                         source_session: Path | str) -> list[str]:
    """Merge the *accepted* slot fits (``MountingFitResult.to_slot_dict()`` records) into
    the stable mounting file that every later recording embeds; a rejected fit never
    replaces a good one. Returns the updated slots."""
    path = Path(path).expanduser()
    doc: dict[str, Any] = {"kind": MOUNTING_FILE_KIND, "slots": {}}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(old, dict) and isinstance(old.get("slots"), dict):
                doc["slots"] = dict(old["slots"])
        except (OSError, ValueError):
            pass
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    done = []
    for slot, rec in records.items():
        if not rec.get("accepted") or rec.get("R_mount") is None:
            continue
        doc["slots"][slot] = {**rec, "date": now, "source_session": str(source_session)}
        done.append(slot)
    if done:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        tmp.replace(path)
    return done


def load_mounting_file(path: Path | str) -> dict[str, dict[str, Any]]:
    """Accepted slot entries of the stable mounting file whose ``R_mount`` is a proper
    rotation (``{}`` when the file is missing or unreadable)."""
    path = Path(path).expanduser()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    out = {}
    for slot, e in (doc.get("slots") or {}).items() if isinstance(doc, dict) else []:
        if not isinstance(e, dict) or not e.get("accepted"):
            continue
        R = np.asarray(e.get("R_mount", np.nan), dtype=np.float64)
        if R.shape == (3, 3) and np.isfinite(R).all() and \
                np.allclose(R.T @ R, np.eye(3), atol=1e-6) and np.linalg.det(R) > 0.5:
            out[slot] = e
    return out


def embed_mounting(h5_path: Path | str, mounting_file: Path | str) -> list[str]:
    """Copy the accepted mountings of the stable file into a recorded session as
    ``/calibration/<slot>/mounting_transform`` (attrs: ``embedded=True``, ``source``,
    ``source_session``, the calibration ``date`` and fit metrics), so that every take
    carries the mounting it was recorded with -- as the CT tip in ``/meta``. Returns the
    embedded slots."""
    entries = load_mounting_file(mounting_file)
    if not entries:
        return []
    with h5py.File(h5_path, "a") as f:
        for slot, e in entries.items():
            g = f.require_group(f"calibration/{slot}")
            if "mounting_transform" in g:
                del g["mounting_transform"]
            for key in _MOUNT_ATTRS:
                if key in g.attrs:
                    del g.attrs[key]
            T = np.eye(4)
            T[:3, :3] = np.asarray(e["R_mount"], dtype=np.float64)
            ds = g.create_dataset("mounting_transform", data=T)
            ds.attrs["translation"] = "zero"
            ds.attrs["note"] = "Rotation body←chip (Park–Martin AX=XB), from the stable mounting file."
            g.attrs["accepted"] = True
            g.attrs["embedded"] = True
            g.attrs["source"] = str(Path(mounting_file).expanduser())
            g.attrs["source_session"] = str(e.get("source_session", ""))
            g.attrs["date"] = str(e.get("date", ""))
            for key in ("rms_deg", "n_pairs", "n_samples"):
                if e.get(key) is not None:
                    g.attrs[key] = e[key]
            if e.get("phase"):
                g.attrs["phase"] = str(e["phase"])
    return sorted(entries)
