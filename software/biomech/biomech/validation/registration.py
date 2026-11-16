# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Spatial registration of JTS-MM to Qualisys (plan D3).

Model, per rigid body ``b`` and synchronised instant ``t``::

    T_Q,b(t) = X · T_S,b(t) · Y_b

``X`` maps the JTS-MM world (camera) frame into the Qualisys world frame; ``Y_b`` maps
Qualisys' body frame of ``b`` into JTS-MM's body frame (tag vs marker-cluster origin;
for the calibration tool it also carries the tip offset, since JTS-MM reports the CT
*tip* pose and Qualisys the cluster origin). This is the robot-world / hand-eye problem
``A = X B Y``.

* :func:`init_xy` solves it in closed form from time-paired poses of one moving body
  (relative rotations -> ``R_X`` by Kabsch on rotation vectors, then ``R_Y`` by a
  chordal mean, then the translations by linear least squares).
* :func:`y_given_x` gives the remaining bodies' ``Y_b`` once ``X`` is known.
* :func:`register_paired` refines ``X`` and all ``Y_b`` by robust least squares on
  SE(3) with the time fixed; the joint fit with the clock model lives in
  :func:`biomech.validation.sync.fit_fine` (errors in time and in the rigid transforms
  bias each other, so the study uses the joint fit).
* :func:`relative_pose_pair` is the frame-free comparison: the mandible relative to the
  head in both systems (``X`` cancels).

Poses are (N, 4, 4) arrays in mm; NaN marks an invalid pose.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3


# -- SE(3) helpers ----------------------------------------------------------------------
def make_pose(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    T = np.zeros(R.shape[:-2] + (4, 4))
    T[..., :3, :3] = R
    T[..., :3, 3] = p
    T[..., 3, 3] = 1.0
    return T


def perturb(T: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Left perturbation: rotation ``exp(xi[:3]) R``, translation ``p + xi[3:]`` (rad, mm)."""
    R = Rotation.from_rotvec(xi[:3]).as_matrix() @ T[:3, :3]
    return make_pose(R, T[:3, 3] + xi[3:])


def project_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def mean_pose(T: np.ndarray) -> np.ndarray:
    """Chordal mean rotation and mean translation of the valid poses."""
    T = np.asarray(T, dtype=np.float64)
    ok = se3.valid(T)
    if not ok.any():
        raise ValueError("no valid pose to average")
    return make_pose(project_so3(T[ok, :3, :3].mean(axis=0)), T[ok, :3, 3].mean(axis=0))


def pose_error(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Translation difference ``p_P - p_Q`` (N, 3) and rotation angle of ``R_Q^T R_P``
    in degrees (N,); NaN where either pose is invalid."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    dp = P[:, :3, 3] - Q[:, :3, 3]
    ang = np.full(len(P), np.nan)
    ok = se3.valid(P) & se3.valid(Q)
    if ok.any():
        D = np.einsum("nji,njk->nik", Q[ok, :3, :3], P[ok, :3, :3])
        ang[ok] = np.degrees(np.linalg.norm(Rotation.from_matrix(D).as_rotvec(), axis=1))
    dp[~ok] = np.nan
    return dp, ang


def pose_residuals(P: np.ndarray, Q: np.ndarray, sigma_pos_mm: float,
                   sigma_rot_rad: float) -> np.ndarray:
    """(N, 6) whitened residuals: ``(p_P - p_Q) / σ_p`` and ``log(R_Q^T R_P) / σ_R``;
    NaN rows where either pose is invalid."""
    dp = (P[:, :3, 3] - Q[:, :3, 3]) / sigma_pos_mm
    D = np.einsum("nji,njk->nik", Q[:, :3, :3], P[:, :3, :3])
    dr = np.full((len(P), 3), np.nan)
    ok = np.isfinite(D.reshape(len(D), -1)).all(axis=1) & np.isfinite(dp).all(axis=1)
    if ok.any():
        dr[ok] = Rotation.from_matrix(D[ok]).as_rotvec() / sigma_rot_rad
    dp[~ok] = np.nan
    return np.hstack([dp, dr])


def huber_weights(r: np.ndarray, k: float) -> np.ndarray:
    """Per-frame IRLS weights for the (N, 6) residual: Huber on the norm of the position
    part and of the rotation part separately, returned as (N, 6)."""
    w = np.ones_like(r)
    for sl in (slice(0, 3), slice(3, 6)):
        n = np.linalg.norm(r[:, sl], axis=1)
        wi = np.where(n <= k, 1.0, k / np.maximum(n, 1e-12))
        w[:, sl] = wi[:, None]
    return w


def tukey_weights(r: np.ndarray, c: float = 4.685) -> np.ndarray:
    """Tukey biweight analogue of :func:`huber_weights` (redescending: residual norms
    beyond ``c`` get weight 0)."""
    w = np.ones_like(r)
    for sl in (slice(0, 3), slice(3, 6)):
        n = np.linalg.norm(r[:, sl], axis=1)
        wi = np.where(n < c, (1.0 - (n / c) ** 2) ** 2, 0.0)
        w[:, sl] = wi[:, None]
    return w


def apply(X: np.ndarray, T_S: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """``X · T_S · Y`` for a stack of JTS-MM poses (NaN stays NaN)."""
    return np.matmul(np.matmul(X, np.asarray(T_S, dtype=np.float64)), Y)


# -- closed-form initialisation ---------------------------------------------------------
def _kabsch(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Rotation R minimising Σ |a − R b|²."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _geodesic_deg(R: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """Angle (deg) of ``R0^T R_i`` for a stack ``R``."""
    D = np.einsum("ji,njk->nik", R0, R)
    tr = np.clip((np.trace(D, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def robust_mean_rotation(R: np.ndarray, inlier_deg: float = 3.0, n_iter: int = 5
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Chordal mean of the largest cluster of rotations (seeded at the element-wise median,
    re-centred on the members within ``inlier_deg``); returns ``(mean, inlier mask)``."""
    M = project_so3(np.median(R, axis=0))
    inl = np.ones(len(R), bool)
    for _ in range(n_iter):
        d = _geodesic_deg(R, M)
        tol = max(inlier_deg, 3.0 * np.median(d)) if _ == 0 else inlier_deg
        inl = d <= tol
        if not inl.any():
            break
        M = project_so3(R[inl].mean(axis=0))
    return M, _geodesic_deg(R, M) <= inlier_deg


def init_xy(A: np.ndarray, B: np.ndarray, *, min_angle_deg: float = 5.0,
            max_pairs: int = 4000, inlier_deg: float = 3.0, n_ransac: int = 300,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ``A_i = X B_i Y`` from time-paired poses of one moving body, robust to a
    fraction of frames in a different pose branch (e.g. a 180° flip of one system's body
    estimate).

    ``A_i A_j^{-1} = X (B_i B_j^{-1}) X^{-1}``: the rotation vectors of the space-frame
    relative rotations satisfy ``a_ij = R_X b_ij``. ``R_X`` comes from RANSAC over frame pairs
    (Kabsch on 3-pair samples, inliers within ``inlier_deg``; pairs both close in time and
    random), refined by Kabsch on the inliers. ``R_Y`` is the robust chordal mean of
    ``(R_X R_Bi)^T R_Ai``; the translations follow from
    ``p_Ai = R_X R_Bi p_Y + R_X p_Bi + p_X`` (linear LS on the ``R_Y`` inlier frames, with
    two rounds of 3-SD trimming). The body must rotate about two non-parallel axes.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    ok = se3.valid(A) & se3.valid(B)
    A, B = A[ok], B[ok]
    n = len(A)
    if n < 3:
        raise ValueError("init_xy: fewer than 3 paired poses")
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, max_pairs * 3)
    j = np.where(np.arange(len(i)) % 2 == 0, rng.integers(0, n, len(i)),
                 np.clip(i + rng.integers(1, 12, len(i)), 0, n - 1))
    RA, RB = A[:, :3, :3], B[:, :3, :3]
    a = Rotation.from_matrix(np.einsum("nij,nkj->nik", RA[i], RA[j])).as_rotvec()
    b = Rotation.from_matrix(np.einsum("nij,nkj->nik", RB[i], RB[j])).as_rotvec()
    ang = np.degrees(np.linalg.norm(b, axis=1))
    sel = (ang >= min_angle_deg) & (ang <= 150.0)
    a, b = a[sel][:max_pairs], b[sel][:max_pairs]
    if len(a) < 3:
        raise ValueError("init_xy: the body does not rotate enough for a hand-eye solution")
    tol = np.radians(inlier_deg)
    best, best_n = None, -1
    for _ in range(n_ransac):
        k = rng.choice(len(a), 3, replace=False)
        R = _kabsch(b[k], a[k])
        n_in = int((np.linalg.norm(a - b @ R.T, axis=1) <= tol).sum())
        if n_in > best_n:
            best, best_n = R, n_in
    inl = np.linalg.norm(a - b @ best.T, axis=1) <= tol
    RX = _kabsch(b[inl], a[inl]) if inl.sum() >= 3 else best
    RYi = np.einsum("nji,njk->nik", np.einsum("ij,njk->nik", RX, RB), RA)
    RY, yin = robust_mean_rotation(RYi, inlier_deg)
    keep = yin.copy()
    sol = np.zeros(6)
    for _ in range(3):
        m = int(keep.sum())
        if m < 3:
            raise ValueError("init_xy: too few consistent frames")
        M = np.zeros((3 * m, 6))
        M[:, :3] = np.tile(np.eye(3), (m, 1))
        M[:, 3:] = np.einsum("ij,njk->nik", RX, RB[keep]).reshape(3 * m, 3)
        rhs = (A[keep, :3, 3] - np.einsum("ij,nj->ni", RX, B[keep, :3, 3])).reshape(-1)
        sol, *_ = np.linalg.lstsq(M, rhs, rcond=None)
        pred = (np.einsum("ij,njk,k->ni", RX, RB, sol[3:]) + np.einsum("ij,nj->ni", RX, B[:, :3, 3])
                + sol[:3])
        e = np.linalg.norm(pred - A[:, :3, 3], axis=1)
        s = 1.4826 * np.median(e[keep])
        keep = yin & (e <= max(3.0 * s, 0.5))
    return make_pose(RX, sol[:3]), make_pose(RY, sol[3:])


def y_given_x(A: np.ndarray, B: np.ndarray, X: np.ndarray, inlier_deg: float = 3.0
              ) -> np.ndarray:
    """``Y`` = robust mean of ``(X B_i)^{-1} A_i`` over the valid pairs (rotation: largest
    cluster; translation: median over its members)."""
    ok = se3.valid(A) & se3.valid(B)
    Yi = np.einsum("nij,njk->nik", se3.inv(np.einsum("ij,njk->nik", X, B[ok])), A[ok])
    if len(Yi) == 0:
        raise ValueError("y_given_x: no valid pairs")
    R, inl = robust_mean_rotation(Yi[:, :3, :3], inlier_deg)
    if not inl.any():
        inl = np.ones(len(Yi), bool)
    return make_pose(R, np.median(Yi[inl, :3, 3], axis=0))


# -- robust refinement with the time fixed ----------------------------------------------
@dataclass
class Registration:
    X: np.ndarray
    Y: dict[str, np.ndarray]
    cov: np.ndarray | None = None              # (6 + 6B)^2 in (rad, mm) of the perturbations
    names: list[str] = field(default_factory=list)   # parameter block order: X, then Y bodies
    rms_pos_mm: dict[str, float] = field(default_factory=dict)
    rms_rot_deg: dict[str, float] = field(default_factory=dict)

    def se(self, block: str) -> tuple[float, float]:
        """(rotation SE in degrees, translation SE in mm) of a parameter block: the RMS over
        its three axes of the marginal standard errors."""
        if self.cov is None:
            return float("nan"), float("nan")
        k = self.names.index(block)
        d = np.sqrt(np.clip(np.diag(self.cov)[6 * k: 6 * k + 6], 0, None))
        return (float(np.degrees(np.sqrt(np.mean(d[:3] ** 2)))),
                float(np.sqrt(np.mean(d[3:] ** 2))))


def register_paired(pairs: dict[str, tuple[np.ndarray, np.ndarray]], X0: np.ndarray,
                    Y0: dict[str, np.ndarray], *, sigma_pos_mm: float = 0.3,
                    sigma_rot_deg: float = 0.15, huber_k: float = 2.5,
                    n_irls: int = 4) -> Registration:
    """Refine ``X`` and every ``Y_b`` on time-paired poses ``pairs[b] = (A_Q, B_S)``."""
    bodies = list(pairs)
    sr = np.radians(sigma_rot_deg)
    data = {}
    for b in bodies:
        A, B = pairs[b]
        ok = se3.valid(A) & se3.valid(B)
        data[b] = (np.asarray(A)[ok], np.asarray(B)[ok])

    def unpack(x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        X = perturb(X0, x[:6])
        Y = {b: perturb(Y0[b], x[6 + 6 * k: 12 + 6 * k]) for k, b in enumerate(bodies)}
        return X, Y

    def raw(x: np.ndarray) -> np.ndarray:
        X, Y = unpack(x)
        return np.vstack([pose_residuals(apply(X, data[b][1], Y[b]), data[b][0],
                                         sigma_pos_mm, sr) for b in bodies])

    x = np.zeros(6 + 6 * len(bodies))
    w = np.ones_like(raw(x))
    for _ in range(n_irls):
        sw = np.sqrt(w)
        sol = least_squares(lambda z: (raw(z) * sw).ravel(), x, method="trf", x_scale="jac")
        x = sol.x
        w = huber_weights(raw(x), huber_k)
    sw = np.sqrt(w)
    J = least_squares(lambda z: (raw(z) * sw).ravel(), x, method="trf", x_scale="jac",
                      max_nfev=1).jac
    r = raw(x) * sw
    dof = max(r.size - x.size, 1)
    s2 = float(np.sum(r ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = None
    X, Y = unpack(x)
    reg = Registration(X=X, Y=Y, cov=cov, names=["X", *bodies])
    for b in bodies:
        dp, ang = pose_error(apply(X, data[b][1], Y[b]), data[b][0])
        reg.rms_pos_mm[b] = float(np.sqrt(np.nanmean(np.sum(dp ** 2, axis=1))))
        reg.rms_rot_deg[b] = float(np.sqrt(np.nanmean(ang ** 2)))
    return reg


# -- frame-free comparison --------------------------------------------------------------
def relative_pose_pair(T_S_head: np.ndarray, T_S_jaw: np.ndarray, T_Q_head: np.ndarray,
                       T_Q_jaw: np.ndarray, Y_head: np.ndarray, Y_jaw: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    """The mandible relative to the head in both systems, in Qualisys' body frames.

    JTS-MM: ``(X T_S,head Y_head)^{-1} (X T_S,jaw Y_jaw) = Y_head^{-1} T_S,head^{-1} T_S,jaw Y_jaw``
    (``X`` cancels); Qualisys: ``T_Q,head^{-1} T_Q,jaw``. Returns ``(rel_S, rel_Q)``; compare
    them with :func:`pose_error`.
    """
    rel_S = se3.relative(T_S_head, T_S_jaw)
    rel_S = np.einsum("ij,njk,kl->nil", se3.inv(Y_head[None])[0], rel_S, Y_jaw)
    rel_Q = se3.relative(T_Q_head, T_Q_jaw)
    return rel_S, rel_Q
