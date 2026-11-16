# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stereo calibration in use (X11, J14; ported from and checked against the earlier
evaluations 5.1.1 of the recording system, an20 §3).

Recordings from s10 on store the live tracker's camera model and every detected tag corner
(``/optical/corners``; s01-s09 store neither, so their calibration cannot be checked this
way). Conventions: ``T_world_cam`` is a camera's pose in the world (= cam0) frame; the
stereo pair maps cam0 into cam1 coordinates as ``x1 = R x0 + t``.

* :func:`epipolar_px` -- symmetric epipolar distance of a corner pair, in pixels (the
  distance in normalised coordinates times the focal length; fx = fy within 0.3 % here).
  It needs no scene knowledge (the essential matrix depends only on the calibration), is
  evaluated on the recording a calibration was used for (working distance, image region,
  motion), and is blind to the baseline length (t enters only up to scale).
* :func:`reprojection_px` -- RMS over both cameras of the reprojection error after linear
  triangulation.
* **Cross test** -- the median epipolar error of every recording with every stored
  calibration: a mismatch between a recording and its calibration would show up as an
  error of several pixels.
* **Sensitivity** (:func:`sensitivity`) -- the recorded corners re-triangulated with the
  cam1 pose perturbed by a random rotation and translation (per-axis N(0, sigma^2)),
  both arrays refitted per frame (weighted Kabsch on the tag model of ``optical.yaml``),
  and the mandibular array's origin in the cranial frame compared with the unperturbed
  refit. Uncertainty model from a calibration bundle's extrinsic snapshots (inliers): the
  final extrinsics are the mean of the inlier snapshots (``optical.calibration.extrinsics``:
  "inliers are averaged"), so the expected uncertainty is the standard error,
  sigma = RMS(snapshot deviation) / sqrt(3) / sqrt(n) per axis; snapshots of one board
  session are not independent, so this is optimistic and the single-snapshot scatter
  (no sqrt(n)) is reported as the conservative bound. Intrinsics are kept fixed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULTS: dict[str, Any] = {
    "max_pairs": 60000,             # corner pairs per recording for the epipolar statistics
    "seed": 0,
    "draws": 50,
    "scenarios": ["expected", "snapshot"],
    "tasks": ["open_close", "protrusion_retrusion", "left_right", "chewing"],
    "rom_percentile": 99.0,
    "head_body": "HP",              # optical.yaml body names: cranial (CRA) and mandibular (MTA)
    "jaw_body": "MP",
}


@dataclass
class StereoCal:
    K0: np.ndarray
    d0: np.ndarray
    K1: np.ndarray
    d1: np.ndarray
    T_c0_c1: np.ndarray                 # pose of cam1 in the cam0 frame
    extrinsics_rms_px: tuple[float, float] = (np.nan, np.nan)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def R(self) -> np.ndarray:
        return self.T_c0_c1[:3, :3].T

    @property
    def t(self) -> np.ndarray:
        return -self.R @ self.T_c0_c1[:3, 3]

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.T_c0_c1[:3, 3]))

    @property
    def convergence_deg(self) -> float:
        c = (np.trace(self.T_c0_c1[:3, :3]) - 1.0) / 2.0
        return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))

    def key(self) -> str:
        a = np.concatenate([self.K0.ravel(), self.d0.ravel(), self.K1.ravel(), self.d1.ravel(),
                            self.T_c0_c1.ravel()])
        return hashlib.sha1(np.round(a, 6).tobytes()).hexdigest()[:10]

    def with_pose(self, T_c0_c1: np.ndarray) -> "StereoCal":
        return StereoCal(self.K0, self.d0, self.K1, self.d1, T_c0_c1, self.extrinsics_rms_px,
                         self.extra)


def session_calibration(h5: str | Path) -> StereoCal | None:
    """The live tracker's camera model stored with the corners (None before s10)."""
    import h5py

    with h5py.File(h5, "r") as f:
        if "optical/corners/calibration" not in f:
            return None
        g = f["optical/corners/calibration"]
        c = {k: {n: np.asarray(g[k][n][()], float) for n in ("K", "dist", "T_world_cam")}
             for k in ("cam0", "cam1")}
        ex = tuple(float(g[k].attrs.get("extrinsics_rms_px", np.nan)) for k in ("cam0", "cam1"))
    T = np.linalg.inv(c["cam0"]["T_world_cam"]) @ c["cam1"]["T_world_cam"]
    return StereoCal(c["cam0"]["K"], c["cam0"]["dist"].ravel(), c["cam1"]["K"],
                     c["cam1"]["dist"].ravel(), T, ex)


def bundle_snapshots(path: str | Path) -> dict[str, Any]:
    """Inlier extrinsic snapshot deviations (deg, mm) of cam1 in a ``calibration_bundle.h5``."""
    import h5py

    with h5py.File(path, "r") as f:
        q = f["cameras/cam1"].get("extrinsics/quality")
        if q is None:
            return {"n": 0}
        keep = np.ones(q["rot_dev_deg"].shape, bool)
        for flag in ("inlier_snapshot", "paired_snapshot"):
            if flag in q:
                keep &= np.asarray(q[flag][()], bool)
        rot = np.asarray(q["rot_dev_deg"][()], float)[keep]
        tra = np.asarray(q["trans_dev_mm"][()], float)[keep]
    return {"n": int(rot.size), "rms_rot_deg": float(np.sqrt(np.mean(rot ** 2))),
            "rms_trans_mm": float(np.sqrt(np.mean(tra ** 2)))}


@dataclass
class CornerPairs:
    p0: np.ndarray      # (N, 4, 2) px, cam0
    p1: np.ndarray      # (N, 4, 2) px, cam1
    row: np.ndarray     # (N,) optical row
    tag: np.ndarray     # (N,) tag id

    def subset(self, idx: np.ndarray) -> "CornerPairs":
        return CornerPairs(self.p0[idx], self.p1[idx], self.row[idx], self.tag[idx])


def corner_pairs(h5: str | Path) -> CornerPairs | None:
    """Tags detected by both cameras in the same optical row."""
    import h5py

    with h5py.File(h5, "r") as f:
        if "optical/corners/corners_px" not in f:
            return None
        g = f["optical/corners"]
        cam = np.asarray(g["cam"][()])
        row = np.asarray(g["optical_row"][()], np.int64)
        tag = np.asarray(g["tag_id"][()], np.int64)
        cp = np.asarray(g["corners_px"][()], float)
        ids = [c.decode() if isinstance(c, bytes) else str(c)
               for c in g.attrs.get("cam_ids", [b"cam0", b"cam1"])]
    i0 = np.flatnonzero(cam == ids.index("cam0"))
    i1 = np.flatnonzero(cam == ids.index("cam1"))
    k0, k1 = row[i0] * 4096 + tag[i0], row[i1] * 4096 + tag[i1]
    u0, f0 = np.unique(k0, return_index=True)
    u1, f1 = np.unique(k1, return_index=True)
    _, a, b = np.intersect1d(u0, u1, return_indices=True)
    s0, s1 = i0[f0[a]], i1[f1[b]]
    return CornerPairs(cp[s0], cp[s1], row[s0], tag[s0])


def _undistort(px: np.ndarray, K: np.ndarray, d: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.undistortPoints(np.ascontiguousarray(px.reshape(-1, 1, 2), float), K,
                               d).reshape(-1, 2)


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], float)


def epipolar_px(cal: StereoCal, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Symmetric epipolar distance per point (px) from normalised coordinates."""
    E = _skew(cal.t) @ cal.R
    h0 = np.c_[n0, np.ones(len(n0))]
    h1 = np.c_[n1, np.ones(len(n1))]
    l1 = h0 @ E.T
    l0 = h1 @ E
    alg = np.abs(np.sum(l1 * h1, axis=1))
    e1 = alg / np.hypot(l1[:, 0], l1[:, 1]) * cal.K1[0, 0]
    e0 = alg / np.hypot(l0[:, 0], l0[:, 1]) * cal.K0[0, 0]
    return 0.5 * (e0 + e1)


def triangulate(cal: StereoCal, n0: np.ndarray, n1: np.ndarray) -> np.ndarray:
    """Linear triangulation from normalised coordinates; points in the cam0 frame (mm)."""
    import cv2

    P0 = np.c_[np.eye(3), np.zeros(3)]
    P1 = np.c_[cal.R, cal.t]
    X = cv2.triangulatePoints(P0, P1, n0.T.astype(float), n1.T.astype(float))
    return (X[:3] / X[3]).T


def reprojection_px(cal: StereoCal, X: np.ndarray, px0: np.ndarray, px1: np.ndarray
                    ) -> np.ndarray:
    import cv2

    r0, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), cal.K0, cal.d0)
    rv, _ = cv2.Rodrigues(cal.R)
    r1, _ = cv2.projectPoints(X, rv, cal.t, cal.K1, cal.d1)
    e0 = np.sum((r0.reshape(-1, 2) - px0.reshape(-1, 2)) ** 2, axis=1)
    e1 = np.sum((r1.reshape(-1, 2) - px1.reshape(-1, 2)) ** 2, axis=1)
    return np.sqrt(0.5 * (e0 + e1))


def in_use(cal: StereoCal, pairs: CornerPairs) -> dict[str, float]:
    """Epipolar and reprojection error of a recording with its own calibration."""
    n0 = _undistort(pairs.p0, cal.K0, cal.d0)
    n1 = _undistort(pairs.p1, cal.K1, cal.d1)
    ep = epipolar_px(cal, n0, n1)
    X = triangulate(cal, n0, n1)
    rp = reprojection_px(cal, X, pairs.p0, pairs.p1)
    return {"n_pairs": int(len(pairs.row)), "n_points": int(ep.size),
            "epipolar_median_px": float(np.median(ep)),
            "epipolar_p95_px": float(np.percentile(ep, 95)),
            "reprojection_median_px": float(np.median(rp)),
            "depth_median_mm": float(np.median(X[:, 2])),
            "baseline_mm": cal.baseline_mm, "convergence_deg": cal.convergence_deg,
            "f0_px": float(cal.K0[0, 0]), "f1_px": float(cal.K1[0, 0]),
            "extrinsics_rms0_px": cal.extrinsics_rms_px[0],
            "extrinsics_rms1_px": cal.extrinsics_rms_px[1], "cal_key": cal.key()}


def epipolar_median(cal: StereoCal, pairs: CornerPairs) -> float:
    n0 = _undistort(pairs.p0, cal.K0, cal.d0)
    n1 = _undistort(pairs.p1, cal.K1, cal.d1)
    return float(np.median(epipolar_px(cal, n0, n1)))


def thin(pairs: CornerPairs, max_pairs: int, seed: int = 0) -> CornerPairs:
    n = len(pairs.row)
    if not max_pairs or n <= max_pairs:
        return pairs
    return pairs.subset(np.sort(np.random.default_rng(seed).choice(n, max_pairs,
                                                                   replace=False)))


# -- sensitivity ----------------------------------------------------------------------------------
def marker_model(optical_yaml: str | Path | None = None) -> tuple[dict[int, np.ndarray],
                                                                  dict[str, tuple[int, int]]]:
    """Tag corners in their body frame (mm, (4, 3) per tag id) and the body id ranges."""
    from optical.markers.config import load_marker_config
    from optical.tracking.pose import tag_object_points

    if optical_yaml is None:
        from optical.config.loader import default_config_path

        optical_yaml = default_config_path()
    mc = load_marker_config(optical_yaml)
    obj = tag_object_points(mc.tag_size_mm)
    model = {int(i): (T[:3, :3] @ obj.T).T + T[:3, 3] for i, T in mc.mounts_T_body_tag.items()}
    return model, {k: (int(v[0]), int(v[1])) for k, v in mc.id_ranges.items()}


def _batch_kabsch(P: np.ndarray, Q: np.ndarray, W: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Weighted rigid fit per row: R, t with Q ≈ R P + t (P model (M, 3), Q (F, M, 3))."""
    w = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)
    Qz = np.where(W[..., None] > 0, Q, 0.0)
    cp = np.einsum("fm,mk->fk", w, P)
    cq = np.einsum("fm,fmk->fk", w, Qz)
    Pc = P[None] - cp[:, None]
    Qc = Qz - cq[:, None]
    H = np.einsum("fm,fmi,fmj->fij", w, Pc, Qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum("fji,fkj->fik", Vt, U)))
    D = np.zeros_like(H)
    D[:, 0, 0] = D[:, 1, 1] = 1.0
    D[:, 2, 2] = d
    R = np.einsum("fji,fjk,fkl->fil", Vt, D, np.transpose(U, (0, 2, 1)))
    t = cq - np.einsum("fij,fj->fi", R, cp)
    return R, t


class _BodyFit:
    """Per-frame layout of one body's corners for the batch Kabsch."""

    def __init__(self, pairs: CornerPairs, frames: np.ndarray, tags: list[int],
                 model: dict[int, np.ndarray]) -> None:
        self.tags = [t for t in tags if t in model]
        self.P = np.concatenate([model[t] for t in self.tags]) if self.tags else np.zeros((0, 3))
        col = {t: i for i, t in enumerate(self.tags)}
        fi = {int(r): i for i, r in enumerate(frames)}
        sel = np.flatnonzero(np.isin(pairs.tag, self.tags))
        self.src = sel                                  # pair index
        self.f = np.array([fi[int(pairs.row[i])] for i in sel], int)
        self.c = np.array([col[int(pairs.tag[i])] for i in sel], int)
        self.F = len(frames)

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """X: (N_pairs, 4, 3) triangulated corners -> R, t per frame and valid mask."""
        M = len(self.tags) * 4
        Q = np.full((self.F, M, 3), np.nan)
        W = np.zeros((self.F, M))
        for k in range(4):
            Q[self.f, self.c * 4 + k] = X[self.src, k]
            W[self.f, self.c * 4 + k] = 1.0
        ok = W.sum(axis=1) >= 4                         # at least one full tag
        R = np.full((self.F, 3, 3), np.nan)
        t = np.full((self.F, 3), np.nan)
        if ok.any():
            R[ok], t[ok] = _batch_kabsch(self.P, Q[ok], W[ok])
        return R, t, ok


def _rot_exp(w: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(w).as_matrix()


def _relative(fc: _BodyFit, fm: _BodyFit, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mandibular-array origin (mm) and rotation in the cranial frame per frame."""
    Rc, tc, okc = fc.fit(X)
    Rm, tm, okm = fm.fit(X)
    p = np.einsum("fji,fj->fi", Rc, tm - tc)          # Rc^T (tm - tc)
    R = np.einsum("fji,fjk->fik", Rc, Rm)
    bad = ~(okc & okm)
    p[bad] = np.nan
    R[bad] = np.nan
    return p, R


def _rom(p: np.ndarray, t_us: np.ndarray, intervals: dict[str, tuple[int, int]],
         rest: tuple[int, int] | None, pct: float) -> dict[str, float]:
    """Range of motion per task: the ``pct`` percentile of the distance from the median rest
    position (or the task's own median without a rest phase)."""
    out = {}
    ref = None
    if rest is not None:
        m = (t_us >= rest[0]) & (t_us < rest[1]) & np.isfinite(p).all(1)
        if m.sum() > 10:
            ref = np.median(p[m], axis=0)
    for task, (a, b) in intervals.items():
        m = (t_us >= a) & (t_us < b) & np.isfinite(p).all(1)
        if m.sum() < 10:
            continue
        r = ref if ref is not None else np.median(p[m], axis=0)
        out[task] = float(np.percentile(np.linalg.norm(p[m] - r, axis=1), pct))
    return out


def sensitivity(cal: StereoCal, pairs: CornerPairs, t_row_us: np.ndarray,
                phases: list[Any], sigma: dict[str, dict[str, float]],
                model: dict[int, np.ndarray], ranges: dict[str, tuple[int, int]],
                p: dict[str, Any], seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per draw: RMS and P95 of the trajectory change, RMS rotation change; per task: the
    change of the range of motion. Returns (draws, rom)."""
    n0 = _undistort(pairs.p0, cal.K0, cal.d0)
    n1 = _undistort(pairs.p1, cal.K1, cal.d1)
    frames = np.unique(pairs.row)
    t_us = t_row_us[frames].astype(np.int64)
    tags = lambda b: list(range(ranges[b][0], ranges[b][1] + 1))    # noqa: E731
    fc = _BodyFit(pairs, frames, tags(p["head_body"]), model)
    fm = _BodyFit(pairs, frames, tags(p["jaw_body"]), model)
    iv = {ph.name: (int(ph.t_start_us), int(ph.t_end_us)) for ph in phases
          if ph.name in p["tasks"]}
    rest = next(((int(ph.t_start_us), int(ph.t_end_us)) for ph in phases if ph.name == "rest"),
                None)

    def run(c: StereoCal) -> tuple[np.ndarray, np.ndarray]:
        X = triangulate(c, n0, n1).reshape(-1, 4, 3)
        return _relative(fc, fm, X)

    p0, R0 = run(cal)
    rom0 = _rom(p0, t_us, iv, rest, p["rom_percentile"])
    rng = np.random.default_rng(seed)
    draws, roms = [], []
    for scen in p["scenarios"]:
        s = sigma[scen]
        for k in range(int(p["draws"])):
            dR = _rot_exp(np.radians(rng.normal(0.0, s["sigma_rot_deg"], 3)))
            dt = rng.normal(0.0, s["sigma_trans_mm"], 3)
            T = cal.T_c0_c1.copy()
            T[:3, :3] = dR @ T[:3, :3]
            T[:3, 3] = T[:3, 3] + dt
            pk, Rk = run(cal.with_pose(T))
            dev = np.linalg.norm(pk - p0, axis=1)
            Rr = np.einsum("fji,fjk->fik", R0, Rk)
            ang = np.degrees(np.arccos(np.clip((np.trace(Rr, axis1=1, axis2=2) - 1) / 2, -1, 1)))
            ok = np.isfinite(dev) & np.isfinite(ang)
            draws.append({"scenario": scen, "draw": k,
                          "traj_rms_mm": float(np.sqrt(np.mean(dev[ok] ** 2))),
                          "traj_p95_mm": float(np.percentile(dev[ok], 95)),
                          "rot_rms_deg": float(np.sqrt(np.mean(ang[ok] ** 2)))})
            for task, v in _rom(pk, t_us, iv, rest, p["rom_percentile"]).items():
                if task in rom0:
                    roms.append({"scenario": scen, "draw": k, "task": task,
                                 "rom_mm": rom0[task], "d_rom_mm": v - rom0[task]})
    return pd.DataFrame(draws), pd.DataFrame(roms)


def sigma_model(snap: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Per-axis sigma of the cam1 pose: standard error of the mean of the snapshots
    (``expected``) and the scatter of one snapshot (``snapshot``)."""
    n = max(int(snap.get("n", 0)), 1)
    r, t = snap["rms_rot_deg"] / np.sqrt(3), snap["rms_trans_mm"] / np.sqrt(3)
    return {"expected": {"sigma_rot_deg": r / np.sqrt(n), "sigma_trans_mm": t / np.sqrt(n)},
            "snapshot": {"sigma_rot_deg": r, "sigma_trans_mm": t}}
