#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Preview: keep optical translation, take fast cranial rotation from the IMU.

The jaw path is the mandibular marker origin in the cranial marker frame,

    p = R_HP^T (t_MP - t_HP).

R_HP is the noisy part. This script aligns the cranial IMU to the optical
pose on the rest phase, then replaces only the fast band of that rotation
(above ``SLOW_S``) with the IMU. Translation stays optical.

Writes one PNG per subject (latest motion take) under
``<results root>/data/imu_rotation_preview/``. ``eval_trajectory_quality.py``
imports the fusion helpers below.

Usage
-----
  python preview_imu_rotation.py
  python preview_imu_rotation.py --data-dir ~/jtsmm_data --subject s01 --out-dir /tmp/prev
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

# Frequencies faster than this follow the IMU. Slower head motion stays optical.
SLOW_S = 1.0
FS_HZ = 40.0

MOTION = ("rest", "open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")


def default_out_dir() -> Path:
    from biomech.paths import results_root

    return results_root() / "data" / "imu_rotation_preview"


def _dec(x: object) -> str:
    return x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x)


def _finite_pose(P: np.ndarray) -> np.ndarray:
    return np.isfinite(P).reshape(len(P), -1).all(axis=1)


def _phase_mask(names: list[str], t0: np.ndarray, t1: np.ndarray, t: np.ndarray, wanted: tuple[str, ...]) -> np.ndarray:
    m = np.zeros(len(t), dtype=bool)
    want = set(wanted)
    for n, a, b in zip(names, t0, t1):
        if n in want:
            m |= (t >= a) & (t <= b)
    return m


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    w, x, y, z = (q / n).T
    R = np.empty((len(q), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    R[:, 0, 1] = 2.0 * (x * y - z * w)
    R[:, 0, 2] = 2.0 * (x * z + y * w)
    R[:, 1, 0] = 2.0 * (x * y + z * w)
    R[:, 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    R[:, 1, 2] = 2.0 * (y * z - x * w)
    R[:, 2, 0] = 2.0 * (x * z - y * w)
    R[:, 2, 1] = 2.0 * (y * z + x * w)
    R[:, 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return R


def _slerp_R(t_src: np.ndarray, R_src: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Interpolate rotations onto ``t_out``. Times are microseconds."""
    keep = np.ones(len(t_src), dtype=bool)
    keep[1:] = np.diff(t_src.astype(np.float64)) > 0
    t_src = t_src[keep].astype(np.float64)
    R_src = R_src[keep]
    # Hemisphere continuity so slerp does not take the long way.
    # Slerp in scipy uses the quaternion shortest path already.
    order = np.argsort(t_src)
    t_src, R_src = t_src[order], R_src[order]
    uniq = np.ones(len(t_src), dtype=bool)
    uniq[1:] = np.diff(t_src) > 0
    t_src, R_src = t_src[uniq], R_src[uniq]
    slerp = Slerp(t_src, Rotation.from_matrix(R_src))
    tq = np.clip(t_out.astype(np.float64), t_src[0], t_src[-1])
    return slerp(tq).as_matrix()


def _project_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _align(R_opt: np.ndarray, R_imu: np.ndarray) -> np.ndarray:
    """Constant camera<-imu map from R_opt ≈ R_align @ R_imu."""
    return _project_so3(np.mean(R_opt @ np.transpose(R_imu, (0, 2, 1)), axis=0))


def _unwrap(R: np.ndarray) -> np.ndarray:
    acc = np.zeros((len(R), 3), dtype=np.float64)
    cur = np.zeros(3, dtype=np.float64)
    for i in range(1, len(R)):
        cur = cur + Rotation.from_matrix(R[i - 1].T @ R[i]).as_rotvec()
        acc[i] = cur
    return acc


def _fuse_rotation(R_opt: np.ndarray, R_imu_cam: np.ndarray, slow_s: float) -> np.ndarray:
    """Optical orientation, with the fast disagreement replaced by the IMU."""
    rel = np.einsum("nji,njk->nik", R_opt, R_imu_cam)
    step = _unwrap(rel)
    window = int(round(slow_s * FS_HZ))
    if window % 2 == 0:
        window += 1
    window = min(window, step.shape[0] if step.shape[0] % 2 else step.shape[0] - 1)
    if window < 7:
        return R_opt
    slow = np.column_stack([savgol_filter(step[:, k], window, 3, mode="interp") for k in range(3)])
    fast = step - slow
    R_delta = Rotation.from_rotvec(fast).as_matrix()
    return np.einsum("nij,njk->nik", R_opt, R_delta)


def _origin(R_hp: np.ndarray, t_hp: np.ndarray, t_mp: np.ndarray) -> np.ndarray:
    return np.einsum("nji,nj->ni", R_hp, t_mp - t_hp)


def _jitter_mm(xyz: np.ndarray) -> float:
    window = 11
    if len(xyz) < window:
        return float("nan")
    trend = np.column_stack([savgol_filter(xyz[:, k], window, 3, mode="interp") for k in range(3)])
    r = np.linalg.norm(xyz - trend, axis=1)
    return float(np.mean(r))


def _load(path: Path) -> dict:
    with h5py.File(path) as f:
        names = [_dec(x) for x in f["/phases/name"][:]]
        t0 = f["/phases/t_start_us"][:].astype(np.float64)
        t1 = f["/phases/t_end_us"][:].astype(np.float64)
        t = f["/optical/t_master_us"][:].astype(np.float64)
        cra = f["/optical/pose_cra"][:]
        mta = f["/optical/pose_mta"][:]
        n_hp = f["/optical/quality/cra/n_tags"][:]
        t_imu = f["/imu_cra/t_master_us_quat"][:].astype(np.float64)
        q = f["/imu_cra/quat"][:].astype(np.float64)
    ok = _finite_pose(cra) & _finite_pose(mta)
    return {
        "names": names,
        "t0": t0,
        "t1": t1,
        "t": t,
        "cra": cra,
        "mta": mta,
        "n_hp": n_hp,
        "t_imu": t_imu,
        "q": q,
        "ok": ok,
    }


def _limits(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    finite = xyz[np.isfinite(xyz).all(1)]
    c = finite.mean(axis=0) if len(finite) else np.zeros(3)
    r = float(np.max(np.ptp(finite, axis=0)) / 2) if len(finite) else 1.0
    return c, max(r, 1.0)


def _draw(ax, xyz: np.ndarray, color: str, title: str, center: np.ndarray, radius: float) -> None:
    finite = np.isfinite(xyz).all(1)
    ax.plot(xyz[finite, 0], xyz[finite, 1], xyz[finite, 2], color=color, lw=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def run_one(sid: str, path: Path, out_dir: Path) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = _load(path)
    t, cra, mta, ok = d["t"], d["cra"], d["mta"], d["ok"]
    rest = _phase_mask(d["names"], d["t0"], d["t1"], t, ("rest",)) & ok & (d["n_hp"] == 2)
    if int(rest.sum()) < 50:
        return f"{sid}: rest too short"
    R_imu = _slerp_R(d["t_imu"], _quat_wxyz_to_R(d["q"]), t)
    R_align = _align(cra[rest, :3, :3], R_imu[rest])
    R_imu_cam = np.einsum("ij,njk->nik", R_align, R_imu)
    both = ok & np.isfinite(R_imu_cam).reshape(len(t), -1).all(1)
    R_opt = cra[:, :3, :3]
    R_fused = np.array(R_opt, copy=True)
    idx = np.flatnonzero(both)
    # Fuse on the contiguous recorded stream; poses outside ``both`` stay optical.
    R_fused[idx] = _fuse_rotation(R_opt[idx], R_imu_cam[idx], SLOW_S)
    p_opt = np.full((len(t), 3), np.nan)
    p_fus = np.full((len(t), 3), np.nan)
    p_opt[ok] = _origin(R_opt[ok], cra[ok, :3, 3], mta[ok, :3, 3])
    p_fus[ok] = _origin(R_fused[ok], cra[ok, :3, 3], mta[ok, :3, 3])

    motion = _phase_mask(d["names"], d["t0"], d["t1"], t, MOTION) & ok
    oc = _phase_mask(d["names"], d["t0"], d["t1"], t, ("open_close",)) & ok
    j_rest_opt = _jitter_mm(p_opt[rest])
    j_rest_fus = _jitter_mm(p_fus[rest])
    j_oc_opt = _jitter_mm(p_opt[oc])
    j_oc_fus = _jitter_mm(p_fus[oc])

    fig = plt.figure(figsize=(11, 8), constrained_layout=True)
    axes = (
        fig.add_subplot(2, 2, 1, projection="3d"),
        fig.add_subplot(2, 2, 2, projection="3d"),
        fig.add_subplot(2, 2, 3, projection="3d"),
        fig.add_subplot(2, 2, 4, projection="3d"),
    )
    c_motion, r_motion = _limits(np.vstack([p_opt[motion], p_fus[motion]]))
    c_oc, r_oc = _limits(np.vstack([p_opt[oc], p_fus[oc]]))
    _draw(axes[0], p_opt[motion], "#1f4e79", "optical, all motion phases", c_motion, r_motion)
    _draw(axes[1], p_fus[motion], "#b85c38", "IMU rotation, all motion phases", c_motion, r_motion)
    _draw(axes[2], p_opt[oc], "#1f4e79", "optical, opening/closing", c_oc, r_oc)
    _draw(axes[3], p_fus[oc], "#b85c38", "IMU rotation, opening/closing", c_oc, r_oc)
    fig.suptitle(
        f"{sid}   rest jitter {j_rest_opt:.2f} → {j_rest_fus:.2f} mm"
        f"    opening {j_oc_opt:.2f} → {j_oc_fus:.2f} mm",
        fontsize=12,
    )
    dest = out_dir / f"{sid}.png"
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return (
        f"{sid}  rest {j_rest_opt:.2f} → {j_rest_fus:.2f} mm   "
        f"open_close {j_oc_opt:.2f} → {j_oc_fus:.2f} mm   {dest.name}"
    )


def main(argv: list[str] | None = None) -> int:
    from biomech.cohort import discover_sessions
    from biomech.io.inventory import select_takes, subject_pids
    from biomech.paths import resolve_data_dir

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="session root (default: $JTSMM_DATA_DIR, repo pilot data, ~/jtsmm_data)")
    ap.add_argument("--subject", default=None,
                    help="exact participant id(s): s03 or s03_m_33, comma-separated")
    ap.add_argument("--take", choices=("latest", "first"), default="latest")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args(argv)

    pids = subject_pids(args.subject)
    motion = [s for s in discover_sessions(resolve_data_dir(args.data_dir))
              if s.kind == "motion" and (pids is None or s.pid in pids)]
    out = (args.out_dir or default_out_dir()).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for s in select_takes(motion, args.take):
        print(run_one(f"{s.pid}_{s.sex}_{s.age}", s.h5, out))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
