# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""
Robust fusion and acceptance gates for multi-camera extrinsics (no OpenCV).

``multicam --mode extrinsics`` collects, per SPACE snapshot, one board pose
``T_cam←board`` per camera (PnP). Relative poses
``T_world←cam_i = T_cam0←board · T_board←cam_i`` (world = cam0) are fused
here with:

1. **PnP gate** — views whose PnP RMS exceeds ``max_pnp_rms_px`` are dropped.
2. **Robust fusion** — a rotation medoid and a component-wise median
   translation give an outlier-free reference; samples further than
   ``max(floor, k·median deviation)`` in rotation or translation are rejected,
   the reference is re-estimated on the inliers once, and inliers are averaged
   (chordal mean rotation, mean translation). One near-degenerate snapshot
   used to move the fused baseline by ~20 mm.
3. **Cross-view error** — for every paired snapshot, the corners cam_i
   detected are predicted from cam0's board pose through the fused
   ``T_cam_i←cam0`` and compared with cam_i's detections (px). This measures
   the camera-to-camera transform itself; it is the acceptance gate.

Bundles are only written automatically when all gates pass.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from optical.calibration.geometry import (
    geodesic_angle_rad,
    invert_T,
    make_T,
    mean_rotation,
    project_points,
)
from optical.calibration.io import CameraExtrinsics, CameraIntrinsics


@dataclass
class BoardView:
    """One camera's board observation in one snapshot."""

    obj_points: np.ndarray  # (N, 3) board frame, mm
    img_points: np.ndarray  # (N, 2) px
    T_cam_board: np.ndarray  # (4, 4) from PnP
    pnp_rms_px: float


Snapshot = Dict[str, BoardView]


@dataclass(frozen=True)
class ExtrinsicsGates:
    """Thresholds for accepting an extrinsics solve (``calibration.capture`` keys)."""

    max_pnp_rms_px: float = 1.5
    max_cross_view_rms_px: float = 4.0
    min_inlier_fraction: float = 0.6
    min_inliers: int = 3
    outlier_k: float = 3.0
    outlier_floor_deg: float = 0.1
    outlier_floor_mm: float = 1.0

    @classmethod
    def from_capture_cfg(cls, cap: Mapping[str, object] | None) -> "ExtrinsicsGates":
        cap = dict(cap or {})
        d = cls()
        return cls(
            max_pnp_rms_px=float(cap.get("max_pnp_rms_px", d.max_pnp_rms_px)),
            max_cross_view_rms_px=float(cap.get("max_cross_view_rms_px", d.max_cross_view_rms_px)),
            min_inlier_fraction=float(cap.get("min_inlier_fraction", d.min_inlier_fraction)),
            min_inliers=d.min_inliers,
            outlier_k=d.outlier_k,
            outlier_floor_deg=d.outlier_floor_deg,
            outlier_floor_mm=d.outlier_floor_mm,
        )


@dataclass
class ExtrinsicsFit:
    """Fused extrinsics plus everything needed to judge them."""

    world_id: str
    extrinsics: Dict[str, CameraExtrinsics]
    n_snapshots: int
    # Per non-world camera, arrays of length n_snapshots (NaN / False when unpaired).
    per_snapshot_cross_view_px: Dict[str, np.ndarray] = field(default_factory=dict)
    inliers: Dict[str, np.ndarray] = field(default_factory=dict)
    paired: Dict[str, np.ndarray] = field(default_factory=dict)
    rot_dev_deg: Dict[str, np.ndarray] = field(default_factory=dict)
    trans_dev_mm: Dict[str, np.ndarray] = field(default_factory=dict)
    cross_view_rms_px: Dict[str, float] = field(default_factory=dict)
    cross_view_rms_all_px: Dict[str, float] = field(default_factory=dict)
    n_pnp_rejected: Dict[str, int] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    @property
    def gates_passed(self) -> bool:
        return not self.problems

    def quality_arrays(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Per-camera arrays for the bundle (``extrinsics/quality``) and cache."""
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for cid in self.per_snapshot_cross_view_px:
            out[cid] = {
                "cross_view_rms_px_per_snapshot": self.per_snapshot_cross_view_px[cid],
                "inlier_snapshot": self.inliers[cid],
                "paired_snapshot": self.paired[cid],
                "rot_dev_deg": self.rot_dev_deg[cid],
                "trans_dev_mm": self.trans_dev_mm[cid],
                "cross_view_rms_all_px": np.asarray(self.cross_view_rms_all_px[cid]),
            }
        return out

    def summary_lines(self) -> List[str]:
        lines = []
        for cid in self.per_snapshot_cross_view_px:
            n_pair = int(self.paired[cid].sum())
            n_inl = int(self.inliers[cid].sum())
            lines.append(
                f"{self.world_id}->{cid}: {n_inl}/{n_pair} paired snapshots kept, "
                f"cross-view RMS {self.cross_view_rms_px[cid]:.2f} px "
                f"(all paired: {self.cross_view_rms_all_px[cid]:.2f} px), "
                f"PnP-rejected views: {self.n_pnp_rejected.get(cid, 0)}"
            )
        return lines


def _robust_inliers(
    Ts: Sequence[np.ndarray], gates: ExtrinsicsGates
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Inlier mask + fused transform + deviations (deg, mm) from the robust reference."""
    stack = np.stack([np.asarray(T, dtype=np.float64) for T in Ts], axis=0)
    n = stack.shape[0]
    Rs = stack[:, :3, :3]
    ts = stack[:, :3, 3]

    def deviations(R_ref: np.ndarray, t_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rot = np.degrees([geodesic_angle_rad(R_ref, R) for R in Rs])
        trans = np.linalg.norm(ts - t_ref, axis=1)
        return np.asarray(rot), trans

    def select(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
        thr_r = max(gates.outlier_floor_deg, gates.outlier_k * float(np.median(rot)))
        thr_t = max(gates.outlier_floor_mm, gates.outlier_k * float(np.median(trans)))
        return (rot <= thr_r) & (trans <= thr_t)

    if n <= 2:
        inl = np.ones(n, dtype=bool)
    else:
        # Rotation medoid: the sample with the smallest median distance to all
        # others. Unaffected by up to ~50 % outliers, unlike a mean.
        D = np.array([[geodesic_angle_rad(Ra, Rb) for Rb in Rs] for Ra in Rs])
        R_ref = Rs[int(np.argmin(np.median(D, axis=1)))]
        t_ref = np.median(ts, axis=0)
        inl = select(*deviations(R_ref, t_ref))
        # Refine the reference on the first inlier set, then select once more.
        if inl.any():
            inl = select(*deviations(mean_rotation(Rs[inl]), ts[inl].mean(axis=0)))
        if not inl.any():
            inl = np.ones(n, dtype=bool)

    T_fused = make_T(mean_rotation(Rs[inl]), ts[inl].mean(axis=0))
    rot, trans = deviations(T_fused[:3, :3], T_fused[:3, 3])
    return inl, T_fused, rot, trans


def cross_view_rms_px(
    view_ref: BoardView,
    view_i: BoardView,
    T_cami_camref: np.ndarray,
    intr_i: CameraIntrinsics,
) -> Tuple[float, np.ndarray]:
    """RMS px of cam_i's detected corners vs. projection through ``T_cam_i←cam_ref``.

    Returns ``(rms, squared_errors)`` (per-corner squared px distances).
    """
    T = np.asarray(T_cami_camref, dtype=np.float64) @ np.asarray(view_ref.T_cam_board, dtype=np.float64)
    pred = project_points(view_i.obj_points, T, intr_i.K, intr_i.dist)
    diff = pred - np.asarray(view_i.img_points, dtype=np.float64).reshape(-1, 2)
    sq = np.sum(diff**2, axis=1)
    return float(np.sqrt(np.mean(sq))), sq


def fuse_extrinsics(
    snapshots: Sequence[Snapshot],
    cam_order: Sequence[str],
    intrinsics: Mapping[str, CameraIntrinsics],
    gates: ExtrinsicsGates = ExtrinsicsGates(),
) -> ExtrinsicsFit:
    """Fuse board snapshots into ``T_world←cam`` (world = ``cam_order[0]``) and grade them.

    Raises:
        ValueError: a non-world camera has no snapshot paired with the world camera.
    """
    cam_order = list(cam_order)
    world_id = cam_order[0]
    n = len(snapshots)

    # PnP gate: a view with a poor single-camera fit gives a wrong board pose.
    kept: List[Dict[str, BoardView]] = []
    n_rej = {cid: 0 for cid in cam_order}
    for snap in snapshots:
        k: Dict[str, BoardView] = {}
        for cid, view in snap.items():
            if cid not in n_rej:
                continue
            if np.isfinite(view.pnp_rms_px) and view.pnp_rms_px <= gates.max_pnp_rms_px:
                k[cid] = view
            else:
                n_rej[cid] += 1
        kept.append(k)

    def mean_pnp(cid: str) -> float:
        vals = [s[cid].pnp_rms_px for s in kept if cid in s]
        return float(np.mean(vals)) if vals else float("nan")

    fit = ExtrinsicsFit(world_id=world_id, extrinsics={}, n_snapshots=n, n_pnp_rejected=n_rej)
    if len(cam_order) == 1:
        fit.extrinsics[world_id] = CameraExtrinsics(
            T_world_cam=np.eye(4), rms_px=mean_pnp(world_id), cross_view_rms_px=float("nan")
        )
        return fit

    worst_cross = float("nan")
    for cid in cam_order[1:]:
        idx = [i for i, s in enumerate(kept) if world_id in s and cid in s]
        if not idx:
            raise ValueError(
                f"No snapshot where both {world_id} and {cid} saw the board with PnP RMS "
                f"<= {gates.max_pnp_rms_px:.2f} px. Capture with the board visible in both "
                "cameras on the same SPACE press."
            )
        samples = [kept[i][world_id].T_cam_board @ invert_T(kept[i][cid].T_cam_board) for i in idx]
        inl_local, T_world_ci, rot_dev, trans_dev = _robust_inliers(samples, gates)

        T_ci_c0 = invert_T(T_world_ci)
        per_snap = np.full(n, np.nan)
        inliers = np.zeros(n, dtype=bool)
        paired = np.zeros(n, dtype=bool)
        rot_full = np.full(n, np.nan)
        trans_full = np.full(n, np.nan)
        sq_inl: List[np.ndarray] = []
        sq_all: List[np.ndarray] = []
        for j, i in enumerate(idx):
            rms, sq = cross_view_rms_px(kept[i][world_id], kept[i][cid], T_ci_c0, intrinsics[cid])
            per_snap[i] = rms
            paired[i] = True
            inliers[i] = bool(inl_local[j])
            rot_full[i] = rot_dev[j]
            trans_full[i] = trans_dev[j]
            sq_all.append(sq)
            if inl_local[j]:
                sq_inl.append(sq)
        cross = float(np.sqrt(np.mean(np.concatenate(sq_inl))))
        cross_all = float(np.sqrt(np.mean(np.concatenate(sq_all))))

        fit.per_snapshot_cross_view_px[cid] = per_snap
        fit.inliers[cid] = inliers
        fit.paired[cid] = paired
        fit.rot_dev_deg[cid] = rot_full
        fit.trans_dev_mm[cid] = trans_full
        fit.cross_view_rms_px[cid] = cross
        fit.cross_view_rms_all_px[cid] = cross_all
        fit.extrinsics[cid] = CameraExtrinsics(
            T_world_cam=T_world_ci, rms_px=mean_pnp(cid), cross_view_rms_px=cross
        )
        worst_cross = cross if not np.isfinite(worst_cross) else max(worst_cross, cross)

        n_pair = len(idx)
        n_inl = int(inl_local.sum())
        need = max(gates.min_inliers, int(math.ceil(gates.min_inlier_fraction * n_pair)))
        if n_inl < need:
            fit.problems.append(
                f"{world_id}->{cid}: only {n_inl}/{n_pair} paired snapshots agree "
                f"(need >= {need}); board poses are inconsistent"
            )
        if not (cross <= gates.max_cross_view_rms_px):
            fit.problems.append(_cross_view_problem(world_id, cid, cross, gates))

    fit.extrinsics[world_id] = CameraExtrinsics(
        T_world_cam=np.eye(4), rms_px=mean_pnp(world_id), cross_view_rms_px=worst_cross
    )
    # Keep cam_order in the output mapping.
    fit.extrinsics = {cid: fit.extrinsics[cid] for cid in cam_order}
    return fit


# ---------------------------------------------------------------------------
# Caches
# ---------------------------------------------------------------------------


def _cross_view_problem(world_id: str, cid: str, cross: float, gates: ExtrinsicsGates) -> str:
    return (
        f"{world_id}->{cid}: cross-view RMS {cross:.2f} px > "
        f"{gates.max_cross_view_rms_px:.2f} px"
    )


def rejudge_cached_gates(
    data: Mapping[str, np.ndarray],
    cam_ids: Sequence[str],
    gates: ExtrinsicsGates,
) -> Optional[Tuple[bool, List[str]]]:
    """Apply the current gates to a stored extrinsics cache.

    The cache records the pass/fail text from the threshold in force at solve
    time. ``--mode bundle`` calls this so a later change of
    ``max_cross_view_rms_px`` is what decides, not that old sentence.

    Returns ``None`` when the cache has no cross-view numbers (bundles from
    before the metric). The caller then keeps the stored decision.
    """
    if len(cam_ids) < 2:
        return True, []
    world_id = cam_ids[0]
    problems: List[str] = []
    judged = False
    for cid in cam_ids[1:]:
        key = f"{cid}_cross_view_rms_px"
        if key not in data:
            continue
        judged = True
        cross = float(np.asarray(data[key]))
        paired_key = f"{cid}_q_paired_snapshot"
        inlier_key = f"{cid}_q_inlier_snapshot"
        if paired_key in data and inlier_key in data:
            n_pair = int(np.asarray(data[paired_key]).astype(bool).sum())
            n_inl = int(np.asarray(data[inlier_key]).astype(bool).sum())
            need = max(gates.min_inliers, int(math.ceil(gates.min_inlier_fraction * n_pair)))
            if n_inl < need:
                problems.append(
                    f"{world_id}->{cid}: only {n_inl}/{n_pair} paired snapshots agree "
                    f"(need >= {need}); board poses are inconsistent"
                )
        if not (cross <= gates.max_cross_view_rms_px):
            problems.append(_cross_view_problem(world_id, cid, cross, gates))
    if not judged:
        return None
    return not problems, problems


def calibration_cache_paths(bundle_path: Path | str) -> Tuple[Path, Path]:
    """``(intrinsics_cache, extrinsics_cache)`` named after the bundle stem.

    The 1-camera and 2-camera profiles write different bundles into the same
    directory; fixed names (``intrinsics_cache.npz``) let one profile's
    ``--mode bundle`` silently merge the other profile's calibration.
    """
    bp = Path(bundle_path)
    return (
        bp.with_name(f"{bp.stem}.intrinsics_cache.npz"),
        bp.with_name(f"{bp.stem}.extrinsics_cache.npz"),
    )


def intrinsics_fingerprint(intrinsics: Mapping[str, CameraIntrinsics], cam_ids: Optional[Sequence[str]] = None) -> str:
    """SHA-256 over K, dist and image_size of the given cameras (sorted by id).

    Stored in the extrinsics cache: extrinsics solved with one set of
    intrinsics are invalid for another (PnP board poses depend on K / dist).
    """
    ids = sorted(cam_ids if cam_ids is not None else intrinsics.keys())
    h = hashlib.sha256()
    for cid in ids:
        intr = intrinsics[cid]
        h.update(cid.encode())
        h.update(np.ascontiguousarray(np.asarray(intr.K, dtype=np.float64).reshape(3, 3)).tobytes())
        h.update(np.ascontiguousarray(np.asarray(intr.dist, dtype=np.float64).ravel()).tobytes())
        h.update(np.asarray([int(v) for v in intr.image_size], dtype=np.int64).tobytes())
    return h.hexdigest()
