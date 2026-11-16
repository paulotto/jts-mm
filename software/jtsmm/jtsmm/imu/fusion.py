# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical + IMU orientation fusion (validation plan V4).

Pure numpy, no I/O. Rotations only: positions stay optical.

Conventions
-----------
* Times are master-clock **seconds** (float64). Optical frame times are the exposure
  centres; the caller converts.
* ``R_opt`` (N, 3, 3): camera ← marker body, NaN where untracked.
* IMU quaternions (w, x, y, z): IMU world ← IMU body (BNO085 game rotation vector: no
  magnetometer, each IMU has its own heading, which drifts).
* Calibration model (per body)::

      R_opt(t) = Exp(w (t − t_ref)) · Z · R_imu(t + lag) · X

  ``Z`` camera ← IMU world, ``X`` IMU body ← marker body (mounting), ``lag`` > 0 when the
  IMU stamps are late, ``w`` (rad/s, camera frame) the slow drift of the IMU world (mostly
  heading, i.e. along ``Z e_z``). Fitted **only** on the training mask supplied by the
  caller (JTS-MM optical, never a reference system).

* **IMU increments.** ``ΔR_k = q_{k−1}⁻¹ q_k`` (:func:`imu_increments`). Their product over an
  interval telescopes to ``q_a⁻¹ q_b``, so the filters propagate between any two times with
  the relative rotation of the (SLERP-sampled, lag-shifted) chip output at the two ends,
  mapped into the marker frame with ``X``: ``R(t_b) = R(t_a) Xᵀ R_imu(t_a)ᵀ R_imu(t_b) X``.
  Nothing is propagated across an IMU gap (:class:`ImuTrack` segments).

Estimators
----------
* **F0** optical only: SLERP between valid optical frames (``"slerp"``, offline) or the
  last valid frame (``"hold"``, causal) — :func:`f0_optical`.
* **F1** SO(3) complementary filter (causal) — :func:`f1_complementary`: propagate with the
  IMU, then ``R ← Exp(α Log(R_opt Rᵀ)) R`` with ``α = 1 − exp(−Δt/τ)``. Innovations larger
  than the predicted spread are rejected (same gate probability as F2).
* **F2** error-state Kalman filter on SO(3) — :func:`f2_kalman`: state = left (camera-
  frame) attitude error ``δθ`` and the drift rate ``b`` (camera frame); because the IMU
  increments are right-multiplied and the drift is a left rotation, the error dynamics are
  exactly linear, ``F = [[I, Δt I], [0, I]]``, ``H = [I, 0]``. Measurement noise
  ``s_k Σ_opt`` with the per-frame quality scale ``s_k`` (:func:`quality_scale`);
  Mahalanobis gating (NIS > χ²₃ quantile) rejects flips/mislabels and logs them; an RTS
  smoother gives the offline estimate.

Settings (:func:`settings_from_sensors`) are derived in advance from sensor
characteristics with documented formulas — never tuned against a reference.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from jtsmm.imu.batch import quat_conj, quat_mul, quat_prepare, quat_resample, quat_to_matrix
from jtsmm.imu.batch import sanitize_stream

__all__ = [
    "Calibration",
    "FusionResult",
    "FusionSettings",
    "ImuTrack",
    "calibrate",
    "chi2_3_ppf",
    "exp_so3",
    "f0_optical",
    "f1_complementary",
    "f2_kalman",
    "imu_increments",
    "log_so3",
    "quality_scale",
    "residual_structure",
    "rot_error_deg",
    "settings_from_sensors",
    "simulate",
    "tau_cost",
    "valid_rotations",
]

DEG = math.pi / 180.0


# ---------------------------------------------------------------------------------------
# SO(3) helpers (vectorized)
# ---------------------------------------------------------------------------------------
def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    z = np.zeros(v.shape[:-1])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return np.stack([np.stack([z, -w, y], -1), np.stack([w, z, -x], -1),
                     np.stack([-y, x, z], -1)], -2)


def exp_so3(v: np.ndarray) -> np.ndarray:
    """Rotation matrices from rotation vectors (..., 3) → (..., 3, 3) (Rodrigues)."""
    v = np.asarray(v, dtype=np.float64)
    th = np.linalg.norm(v, axis=-1)[..., None, None]
    K = _skew(v)
    small = th < 1e-6
    th_s = np.where(small, 1.0, th)
    a = np.where(small, 1.0 - th ** 2 / 6.0, np.sin(th_s) / th_s)
    b = np.where(small, 0.5 - th ** 2 / 24.0, (1.0 - np.cos(th_s)) / th_s ** 2)
    return np.eye(3) + a * K + b * (K @ K)


def _mat2quat(R: np.ndarray) -> np.ndarray:
    """Unit quaternions (w ≥ 0) from rotation matrices (..., 3, 3), vectorized Shepperd."""
    R = np.asarray(R, dtype=np.float64)
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    tr = r00 + r11 + r22
    c = np.stack([
        np.stack([1 + tr, R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0],
                  R[..., 1, 0] - R[..., 0, 1]], -1),
        np.stack([R[..., 2, 1] - R[..., 1, 2], 1 + r00 - r11 - r22,
                  R[..., 0, 1] + R[..., 1, 0], R[..., 0, 2] + R[..., 2, 0]], -1),
        np.stack([R[..., 0, 2] - R[..., 2, 0], R[..., 0, 1] + R[..., 1, 0],
                  1 - r00 + r11 - r22, R[..., 1, 2] + R[..., 2, 1]], -1),
        np.stack([R[..., 1, 0] - R[..., 0, 1], R[..., 0, 2] + R[..., 2, 0],
                  R[..., 1, 2] + R[..., 2, 1], 1 - r00 - r11 + r22], -1),
    ], -2)                                                      # (..., 4 cases, 4)
    piv = np.stack([tr, r00, r11, r22], -1)
    k = np.argmax(np.nan_to_num(piv, nan=-np.inf), axis=-1)
    q = np.take_along_axis(c, k[..., None, None], axis=-2)[..., 0, :]
    d = np.take_along_axis(q, k[..., None], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        q = q / (2.0 * np.sqrt(np.maximum(d, 1e-300)))
        q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return np.where(q[..., :1] < 0, -q, q)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Rotation vectors (..., 3) of rotation matrices (..., 3, 3); NaN propagates."""
    q = _mat2quat(R)
    v = q[..., 1:]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    ang = 2.0 * np.arctan2(s, q[..., :1])
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(s > 1e-12, ang / np.where(s > 1e-12, s, 1.0), 2.0)
    return v * k


def rot_error_deg(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle (deg) between stacks of rotations; NaN where either is NaN."""
    D = np.einsum("...ji,...jk->...ik", Ra, Rb)
    tr = np.clip((np.trace(D, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    return np.degrees(np.arccos(tr))


def _project(M: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(M)
    d = np.sign(np.linalg.det(u @ vt))
    u = u.copy()
    u[..., :, -1] *= d[..., None]
    return u @ vt


def valid_rotations(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(R_proj, ok)``: rotations projected onto SO(3) (NaN where invalid) and the mask
    of finite, right-handed inputs."""
    R = np.asarray(R, dtype=np.float64)
    ok = np.isfinite(R.reshape(len(R), -1)).all(axis=1)
    out = np.full(R.shape, np.nan)
    if ok.any():
        det = np.linalg.det(R[ok])
        good = np.flatnonzero(ok)[det > 0.5]
        ok[:] = False
        ok[good] = True
        if good.size:
            out[good] = _project(R[good])
    return out, ok


def _kabsch(b: np.ndarray, a: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rotation ``R`` minimizing ``Σ w |a_i − R b_i|²``."""
    w = np.ones(len(a)) if w is None else w
    H = (b * w[:, None]).T @ a
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _robust_mean_rotation(R: np.ndarray, inlier_deg: float) -> np.ndarray:
    M = _project(np.median(R, axis=0))
    for it in range(6):
        d = rot_error_deg(np.broadcast_to(M, R.shape), R)
        tol = max(inlier_deg, 3.0 * float(np.median(d))) if it == 0 else inlier_deg
        inl = d <= tol
        if not inl.any():
            break
        M = _project(R[inl].mean(axis=0))
    return M


def _chi2_cdf(x: float, dof: int) -> float:
    """χ² CDF for dof 3 or 5 (closed form)."""
    if x <= 0:
        return 0.0
    base = math.erf(math.sqrt(x / 2.0)) - math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0)
    if dof == 3:
        return base
    if dof == 5:
        return base - math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0) * x / 3.0
    raise ValueError("dof must be 3 or 5")


def chi2_3_ppf(p: float) -> float:
    """Quantile of χ² with 3 degrees of freedom (bisection on the closed-form CDF)."""
    lo, hi = 0.0, 100.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _chi2_cdf(mid, 3) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _robust_cov(d: np.ndarray, p: float = 0.99, n_iter: int = 8) -> np.ndarray:
    """3x3 covariance of ``d`` with iterative Mahalanobis trimming at the χ²₃ ``p``
    quantile, corrected for the truncation (consistent for Gaussian data)."""
    d = np.asarray(d, dtype=np.float64)
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) < 10:
        raise ValueError("robust_cov: fewer than 10 samples")
    q = chi2_3_ppf(p)
    mu = np.median(d, axis=0)
    mad = 1.4826 * np.median(np.abs(d - mu), axis=0)
    C = np.diag(np.maximum(mad, 1e-12) ** 2)
    for _ in range(n_iter):
        e = d - mu
        m2 = np.einsum("ni,ij,nj->n", e, np.linalg.inv(C), e)
        keep = m2 <= q
        if keep.sum() < 10:
            break
        mu = d[keep].mean(axis=0)
        e = d[keep] - mu
        C = (e.T @ e) / max(len(e) - 1, 1) * (p / _chi2_cdf(q, 5))
    return C


# ---------------------------------------------------------------------------------------
# IMU stream
# ---------------------------------------------------------------------------------------
def imu_increments(q: np.ndarray) -> np.ndarray:
    """Body-frame increments ``ΔR_k = q_{k−1}⁻¹ q_k`` (N−1, 4) of a sanitized stream.
    Their ordered product over ``[a, b]`` equals ``q_a⁻¹ q_b`` (telescoping)."""
    qp = quat_prepare(q)
    return quat_mul(quat_conj(qp[:-1]), qp[1:])


@dataclass
class ImuTrack:
    """A sanitized quaternion stream with gap-aware SLERP sampling and a time lag.

    ``t_us`` strictly increasing int64 (master µs), ``q`` normalized and sign-continuous.
    Holes wider than ``max_gap_s`` split the stream into segments; nothing is interpolated
    or propagated across them.
    """

    t_us: np.ndarray
    q: np.ndarray
    max_gap_s: float = 0.05
    n_dropped: int = 0
    _seg: np.ndarray | None = field(default=None, init=False, repr=False)
    _path: np.ndarray | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_stream(cls, t_us: np.ndarray, quat: np.ndarray,
                    accuracy: np.ndarray | None = None, max_gap_s: float = 0.05
                    ) -> "ImuTrack":
        """Drop accuracy-0 and non-finite samples, sort, drop repeated stamps
        (:func:`jtsmm.imu.batch.sanitize_stream`), normalize and make sign-continuous."""
        t = np.asarray(t_us, dtype=np.int64)
        q = np.asarray(quat, dtype=np.float64)
        keep = np.isfinite(q).all(axis=1) & (np.linalg.norm(q, axis=1) > 0.5)
        if accuracy is not None:
            keep &= np.asarray(accuracy) != 0
        ts, qs = sanitize_stream(t[keep], q[keep])
        return cls(ts, quat_prepare(qs), max_gap_s, int(len(t) - len(ts)))

    @property
    def t_s(self) -> np.ndarray:
        return self.t_us * 1e-6

    def rate_hz(self) -> float:
        return 1e6 / float(np.median(np.diff(self.t_us))) if len(self.t_us) > 1 else np.nan

    def _segments(self) -> np.ndarray:
        if self._seg is None:
            gap = np.diff(self.t_us) > self.max_gap_s * 1e6
            self._seg = np.concatenate(([0], np.cumsum(gap)))
        return self._seg

    def quat(self, t_s: np.ndarray, lag_s: float = 0.0) -> np.ndarray:
        """Chip quaternion at true times ``t_s`` (stamp = ``t + lag``); NaN in holes."""
        tq = np.round((np.asarray(t_s, dtype=np.float64) + lag_s) * 1e6).astype(np.int64)
        return quat_resample(self.t_us, self.q, tq, int(self.max_gap_s * 1e6), prepared=True)

    def rot(self, t_s: np.ndarray, lag_s: float = 0.0) -> np.ndarray:
        return quat_to_matrix(self.quat(t_s, lag_s))

    def segment(self, t_s: np.ndarray, lag_s: float = 0.0) -> np.ndarray:
        """Gap-free segment id at each time (−1 where the stream has no sample)."""
        tq = np.round((np.asarray(t_s, dtype=np.float64) + lag_s) * 1e6).astype(np.int64)
        n = len(self.t_us)
        if n < 2:
            return np.full(tq.shape, -1)
        j = np.searchsorted(self.t_us, tq, side="right") - 1
        jc = np.clip(j, 0, n - 2)
        ok = (j >= 0) & (tq <= self.t_us[-1])
        ok &= (np.diff(self.t_us)[jc] <= self.max_gap_s * 1e6) | (tq == self.t_us[jc])
        return np.where(ok, self._segments()[np.clip(j, 0, n - 1)], -1)

    def path_angle(self, t_s: np.ndarray, lag_s: float = 0.0) -> np.ndarray:
        """Cumulative rotation path length (rad) of the chip output at ``t_s``: the sum of
        the increment angles, linearly interpolated (for the scale-error process noise)."""
        if self._path is None:
            inc = imu_increments(self.q)
            ang = 2.0 * np.arctan2(np.linalg.norm(inc[:, 1:], axis=1), np.abs(inc[:, 0]))
            ang[np.diff(self.t_us) > self.max_gap_s * 1e6] = 0.0
            self._path = np.concatenate(([0.0], np.cumsum(ang)))
        return np.interp(np.asarray(t_s, dtype=np.float64) + lag_s, self.t_s, self._path)

    def omega_world(self, t_s: np.ndarray, lag_s: float = 0.0, h_s: float = 0.003
                    ) -> np.ndarray:
        """IMU-world angular velocity (rad/s) by a central difference over ``±h_s``."""
        Ra = self.rot(t_s - h_s, lag_s)
        Rb = self.rot(t_s + h_s, lag_s)
        return log_so3(np.einsum("nij,nkj->nik", Rb, Ra)) / (2.0 * h_s)


# ---------------------------------------------------------------------------------------
# Calibration (Z, X, drift, lag) on training frames only
# ---------------------------------------------------------------------------------------
@dataclass
class Calibration:
    """``R_opt(t) = Exp(w (t − t_ref)) Z R_imu(t + lag) X`` fitted on a training mask."""

    Z: np.ndarray
    X: np.ndarray
    lag_s: float
    drift_rad_s: np.ndarray
    t_ref_s: float
    lag_se_s: float = np.nan
    drift_se_rad_s: float = np.nan
    rms_deg: float = np.nan                 # inlier residual RMS (3-D angle)
    median_deg: float = np.nan
    resid_sd_deg: np.ndarray = field(default_factory=lambda: np.full(3, np.nan))
    inlier_frac: float = np.nan
    n_train: int = 0
    n_used: int = 0
    span_s: float = 0.0
    sf_sigma: float = np.nan                # relative increment error (from training pairs)
    omega_rms_rad_s: float = np.nan         # RMS IMU angular speed over the training frames
    lag_grid_s: np.ndarray = field(default_factory=lambda: np.zeros(0))
    lag_cost: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def drift_heading_deg_min(self) -> float:
        """Drift component about the IMU gravity axis (``Z e_z``), deg/min."""
        return float(np.degrees(self.drift_rad_s @ self.Z[:, 2]) * 60.0)

    @property
    def drift_tilt_deg_min(self) -> float:
        g = self.Z[:, 2]
        tilt = self.drift_rad_s - (self.drift_rad_s @ g) * g
        return float(np.degrees(np.linalg.norm(tilt)) * 60.0)

    def predict(self, t_s: np.ndarray, R_imu_lagged: np.ndarray) -> np.ndarray:
        """Optical-frame rotation predicted from IMU rotations already sampled at
        ``t + lag``."""
        E = exp_so3(np.outer(np.asarray(t_s) - self.t_ref_s, self.drift_rad_s))
        return np.einsum("nij,jk,nkl,lm->nim", E, self.Z, R_imu_lagged, self.X)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        d["drift_heading_deg_min"] = self.drift_heading_deg_min
        d["drift_tilt_deg_min"] = self.drift_tilt_deg_min
        d["Z_rotvec_deg"] = np.degrees(log_so3(self.Z)).tolist()
        d["X_rotvec_deg"] = np.degrees(log_so3(self.X)).tolist()
        d["lag_ms"] = self.lag_s * 1e3
        d["lag_se_ms"] = self.lag_se_s * 1e3
        return d


def _relative_pairs(t: np.ndarray, Ro: np.ndarray, Ri: np.ndarray, rng: np.random.Generator,
                    n: int, max_step: int, max_dt_s: float, min_deg: float, max_pairs: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    i = rng.integers(0, len(t), n)
    j = np.clip(i + rng.integers(1, max_step + 1, len(i)), 0, len(t) - 1)
    sel = (t[j] - t[i] <= max_dt_s) & (j != i)
    i, j = i[sel], j[sel]
    a = log_so3(np.einsum("nji,njk->nik", Ro[i], Ro[j]))          # marker frame
    b = log_so3(np.einsum("nji,njk->nik", Ri[i], Ri[j]))          # IMU body frame
    ang = np.degrees(np.linalg.norm(b, axis=1))
    sel = (ang >= min_deg) & (ang <= 120.0) & np.isfinite(a).all(1) & np.isfinite(b).all(1)
    return a[sel][:max_pairs], b[sel][:max_pairs]


def _init_x(t: np.ndarray, Ro: np.ndarray, Ri: np.ndarray, rng: np.random.Generator,
            max_pairs: int = 4000, inlier_deg: float = 3.0, n_ransac: int = 300
            ) -> np.ndarray:
    """Mounting ``X`` from body-frame relative rotations:
    ``Log(R_opt,iᵀ R_opt,j) = Xᵀ Log(R_imu,iᵀ R_imu,j)``; RANSAC Kabsch (rejects flips).
    Pairs ≤ 2 s apart with ≥ 3° (the heading drift cancels); when a body barely moves
    (e.g. the head), pairs up to 60 s apart with ≥ 1.5° (drift ≲ a few degrees, removed
    by the refinement that follows)."""
    a, b = _relative_pairs(t, Ro, Ri, rng, max_pairs * 4, 40, 2.0, 3.0, max_pairs)
    if len(a) < 50:
        a, b = _relative_pairs(t, Ro, Ri, rng, max_pairs * 8, len(t), 60.0, 1.5, max_pairs)
    if len(a) < 3:
        raise ValueError("calibrate: the body does not rotate enough on the training mask")
    tol = np.radians(inlier_deg)
    best, best_n = np.eye(3), -1
    for _ in range(n_ransac):
        k = rng.choice(len(a), 3, replace=False)
        Rx = _kabsch(a[k], b[k])                                  # b = X a
        n_in = int((np.linalg.norm(b - a @ Rx.T, axis=1) <= tol).sum())
        if n_in > best_n:
            best, best_n = Rx, n_in
    inl = np.linalg.norm(b - a @ best.T, axis=1) <= tol
    return _kabsch(a[inl], b[inl]) if inl.sum() >= 3 else best


def _residuals(cal: Calibration, t: np.ndarray, Ro: np.ndarray, imu: ImuTrack
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    Ri = imu.rot(t, cal.lag_s)
    Rp = cal.predict(t, Ri)
    r = log_so3(np.einsum("nij,nkj->nik", Ro, Rp))
    return r, Rp, Ri


def _robust_weights(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Huber weights on the residual angle; hard rejection beyond max(5 median, 1°)."""
    e = np.linalg.norm(r, axis=1)
    fin = np.isfinite(e)
    med = float(np.median(e[fin])) if fin.any() else 1.0
    inl = fin & (e <= max(5.0 * med, np.radians(1.0)))
    k = max(2.0 * med, np.radians(0.2))
    w = np.where(inl, np.minimum(1.0, k / np.maximum(e, 1e-12)), 0.0)
    return w, inl


def _gn_refine(cal: Calibration, t: np.ndarray, Ro: np.ndarray, imu: ImuTrack, *,
               fit_drift: bool, fit_lag: bool, n_iter: int = 12, fit_x: bool = True
               ) -> tuple[Calibration, np.ndarray | None, np.ndarray]:
    """Robust Gauss–Newton over (δZ[, δX][, δw][, δlag]) with left/right perturbations:
    ``r ≈ r0 − (E δz + R_pred δx + τ δw + E Z ω_w δlag)``; ``fit_x=False`` keeps the
    mounting ``X`` (e.g. from a separate hand-eye calibration)."""
    cov = None
    inl = np.zeros(len(t), bool)
    for _ in range(n_iter):
        r, Rp, _ = _residuals(cal, t, Ro, imu)
        w, inl = _robust_weights(r)
        tau = t - cal.t_ref_s
        E = exp_so3(np.outer(tau, cal.drift_rad_s))
        blocks = [E, Rp] if fit_x else [E]
        if fit_drift:
            blocks.append(np.broadcast_to(np.eye(3), Rp.shape) * tau[:, None, None])
        if fit_lag:
            om = imu.omega_world(t, cal.lag_s)
            blocks.append(np.einsum("nij,jk,nk->ni", E, cal.Z, om)[:, :, None])
        J = -np.concatenate(blocks, axis=2)                        # (n, 3, p)
        ok = inl & np.isfinite(J).all(axis=(1, 2)) & np.isfinite(r).all(axis=1)
        Jo, ro, wo = J[ok], r[ok], w[ok]
        A = np.einsum("n,nip,niq->pq", wo, Jo, Jo)
        g = np.einsum("n,nip,ni->p", wo, Jo, ro)
        try:
            d = -np.linalg.solve(A, g)
        except np.linalg.LinAlgError:
            break
        cal.Z = exp_so3(d[0:3]) @ cal.Z
        p = 3
        if fit_x:
            cal.X = cal.X @ exp_so3(d[3:6])
            p = 6
        if fit_drift:
            cal.drift_rad_s = cal.drift_rad_s + d[p:p + 3]
            p += 3
        if fit_lag:
            cal.lag_s = float(cal.lag_s + np.clip(d[p], -0.01, 0.01))
        sig2 = float(np.sum(wo[:, None] * ro ** 2) / max(3 * ok.sum() - len(d), 1))
        cov = sig2 * np.linalg.inv(A)
        if np.max(np.abs(d[:(6 if fit_x else 3)])) < 1e-7 and (not fit_lag or abs(d[-1]) < 1e-6):
            break
    return cal, cov, inl


def calibrate(t_opt_s: np.ndarray, R_opt: np.ndarray, imu: ImuTrack, train_mask: np.ndarray,
              *, lag_range_s: tuple[float, float] = (-0.05, 0.08), lag_step_s: float = 0.002,
              fit_drift: bool | None = None, min_drift_span_s: float = 60.0,
              lag_s: float | None = None, max_frames: int = 6000, sf_pair_s: float = 0.25,
              seed: int = 0, X_fixed: np.ndarray | None = None) -> Calibration:
    """Fit ``Z``, ``X``, the drift rate ``w`` and the lag on the training frames only.
    ``X_fixed``: the IMU mounting from a separate hand-eye calibration (``R_imu<-body``
    convention of ``X``); it is then kept and only ``Z``, ``w`` and the lag are fitted.

    1. ``X`` from body-frame relative rotations of nearby frame pairs (RANSAC Kabsch; the
       heading drift cancels), at lag 0; ``Z`` = robust chordal mean of
       ``R_opt Xᵀ R_imuᵀ``.
    2. Coarse lag scan over ``lag_range_s`` (truncated mean squared residual).
    3. Robust Gauss–Newton (Huber, hard rejection of flips) jointly over ``Z``, ``X``,
       ``w`` (only when the training frames span ≥ ``min_drift_span_s``) and the lag;
       standard errors from ``σ² (JᵀWJ)⁻¹`` (white-residual approximation). With
       ``lag_s`` given, steps 2-3 keep the lag fixed (e.g. a lag pooled over the IMUs).
    4. Increment-error scale ``sf_sigma`` from training frame pairs ``sf_pair_s`` apart:
       ``e² = c + sf² θ²`` (robust LS), ``e`` = angle between optical and IMU relative
       rotations (marker frame), ``θ`` = IMU path angle of the pair.
    """
    t_all = np.asarray(t_opt_s, dtype=np.float64)
    Rv, ok = valid_rotations(R_opt)
    m = ok & np.asarray(train_mask, bool)
    m &= imu.segment(t_all) >= 0
    idx = np.flatnonzero(m)
    if idx.size < 20:
        raise ValueError(f"calibrate: only {idx.size} usable training frames")
    rng = np.random.default_rng(seed)
    if idx.size > max_frames:
        idx = np.sort(rng.choice(idx, max_frames, replace=False))
    t, Ro = t_all[idx], Rv[idx]
    span = float(t[-1] - t[0])
    if fit_drift is None:
        fit_drift = span >= min_drift_span_s

    Ri0 = imu.rot(t, 0.0)
    fit_x = X_fixed is None
    X0 = _init_x(t, Ro, Ri0, rng) if fit_x else np.asarray(X_fixed, dtype=np.float64)
    Z0 = _robust_mean_rotation(np.einsum("nij,kj,nlk->nil", Ro, X0, Ri0), 3.0)
    cal = Calibration(Z=Z0, X=X0, lag_s=0.0, drift_rad_s=np.zeros(3),
                      t_ref_s=float(np.mean(t)), n_train=int(m.sum()), span_s=span)
    cal, _, _ = _gn_refine(cal, t, Ro, imu, fit_drift=False, fit_lag=False, n_iter=6,
                           fit_x=fit_x)

    grid = np.arange(lag_range_s[0], lag_range_s[1] + 1e-12, lag_step_s)
    cost = np.full(grid.size, np.nan)
    r0, _, _ = _residuals(cal, t, Ro, imu)
    cap2 = (5.0 * float(np.nanmedian(np.linalg.norm(r0, axis=1)))) ** 2
    for k, lag in enumerate(grid):
        cal.lag_s = float(lag)
        r, _, _ = _residuals(cal, t, Ro, imu)
        e2 = np.sum(r ** 2, axis=1)
        e2 = e2[np.isfinite(e2)]
        cost[k] = float(np.mean(np.minimum(e2, cap2))) if e2.size else np.nan
    cal.lag_grid_s, cal.lag_cost = grid, cost
    fit_lag = lag_s is None
    cal.lag_s = float(grid[int(np.nanargmin(cost))]) if fit_lag else float(lag_s)

    cal, cov, inl = _gn_refine(cal, t, Ro, imu, fit_drift=fit_drift, fit_lag=fit_lag,
                               fit_x=fit_x)
    if cov is not None:
        if fit_lag:
            cal.lag_se_s = float(np.sqrt(cov[-1, -1]))
        if fit_drift:
            k = 6 if fit_x else 3
            cal.drift_se_rad_s = float(np.sqrt(np.trace(cov[k:k + 3, k:k + 3]) / 3.0))

    r, _, Ri = _residuals(cal, t, Ro, imu)
    e = np.linalg.norm(r, axis=1)
    cal.inlier_frac = float(inl.mean())
    cal.n_used = int(inl.sum())
    cal.rms_deg = float(np.degrees(np.sqrt(np.mean(e[inl] ** 2))))
    cal.median_deg = float(np.degrees(np.median(e[np.isfinite(e)])))
    cal.resid_sd_deg = np.degrees(np.std(r[inl], axis=0, ddof=1))
    om = imu.omega_world(t, cal.lag_s)
    cal.omega_rms_rad_s = float(np.sqrt(np.nanmean(np.sum(om ** 2, axis=1))))
    cal.sf_sigma = _increment_scale_error(t_all, Rv, m, imu, cal, sf_pair_s)
    return cal


def _increment_scale_error(t_all: np.ndarray, Rv: np.ndarray, m: np.ndarray, imu: ImuTrack,
                           cal: Calibration, pair_s: float) -> float:
    dt = float(np.median(np.diff(t_all)))
    step = max(1, int(round(pair_s / dt)))
    i = np.flatnonzero(m[:-step] & m[step:])
    if i.size < 30:
        return np.nan
    j = i + step
    ti, tj = t_all[i], t_all[j]
    keep = (tj - ti) <= 1.5 * pair_s
    keep &= imu.segment(ti, cal.lag_s) == imu.segment(tj, cal.lag_s)
    i, j, ti, tj = i[keep], j[keep], ti[keep], tj[keep]
    if i.size < 30:
        return np.nan
    Ri, Rj = imu.rot(ti, cal.lag_s), imu.rot(tj, cal.lag_s)
    B = np.einsum("ji,nkj,nkl,lm->nim", cal.X, Ri, Rj, cal.X)     # Xᵀ Riᵀ Rj X
    A = np.einsum("nji,njk->nik", Rv[i], Rv[j])
    e2 = np.sum(log_so3(np.einsum("nij,nkj->nik", A, B)) ** 2, axis=1)
    th = imu.path_angle(tj, cal.lag_s) - imu.path_angle(ti, cal.lag_s)
    ok = np.isfinite(e2) & (e2 <= np.radians(10.0) ** 2)
    e2, th = e2[ok], th[ok]
    if e2.size < 30 or np.ptp(th) < np.radians(2.0):
        return np.nan
    M = np.column_stack([np.ones_like(th), th ** 2])
    w = np.ones_like(e2)
    for _ in range(5):                                            # IRLS (Huber on e²)
        sol = np.linalg.lstsq(M * w[:, None], e2 * w, rcond=None)[0]
        res = e2 - M @ sol
        s = 1.4826 * np.median(np.abs(res)) + 1e-12
        w = np.sqrt(np.minimum(1.0, 1.5 * s / np.maximum(np.abs(res), 1e-12)))
    return float(np.sqrt(max(sol[1], 0.0)))


# ---------------------------------------------------------------------------------------
# Settings from sensor characteristics (fixed before any reference comparison)
# ---------------------------------------------------------------------------------------
@dataclass
class FusionSettings:
    """Filter settings derived by :func:`settings_from_sensors` (all rad, s)."""

    tau_s: float                         # F1 time constant
    sigma_opt: np.ndarray                # (3, 3) optical rotation noise at reference quality
    q_theta: float                       # IMU increment noise density (rad²/s per axis)
    sf_sigma: float                      # relative increment (scale) error
    q_bias: float                        # drift-rate random walk (rad²/s³ per axis)
    bias0: np.ndarray                    # initial drift rate (camera frame, rad/s)
    bias0_sd: float                      # its prior SD (rad/s per axis)
    omega_rms: float                     # RMS angular speed for the τ formula (rad/s)
    dt_opt_s: float                      # nominal optical frame interval
    gate_prob: float = 0.999
    gate_chi2: float = 16.266
    reset_s: float = 1.0                 # re-initialize after this long of consistent gating
    lost_rate: float = 1.0               # rad/s, attitude growth when the IMU is missing
    quality_ref: dict[str, float] = field(default_factory=dict)
    derivation: dict[str, Any] = field(default_factory=dict)

    def q_eff(self) -> float:
        """Per-axis increment error density used for τ: ``q_θ + sf² ω_rms² Δt / 3``."""
        return self.q_theta + self.sf_sigma ** 2 * self.omega_rms ** 2 * self.dt_opt_s / 3.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
        d["sigma_opt_deg_axis"] = np.degrees(np.sqrt(np.diag(self.sigma_opt))).tolist()
        return d


def tau_cost(tau_s: float, dt: float, sigma2: float, q: float, drift: float) -> float:
    """Expected per-axis squared error of the complementary filter at steady state:

    ``J(τ) = σ² α/(2−α) + q Δt (1−α)²/(1−(1−α)²) + (b Δt (1−α)/α)²/3``,
    ``α = 1 − exp(−Δt/τ)``: optical-noise pass-through, IMU random-walk tracking error and
    the lag error of an uncompensated drift rate ``b`` (rad/s, norm; /3 per axis).
    """
    a = 1.0 - math.exp(-dt / tau_s)
    a = min(max(a, 1e-12), 1.0)
    noise = sigma2 * a / (2.0 - a)
    walk = q * dt * (1.0 - a) ** 2 / max(1.0 - (1.0 - a) ** 2, 1e-300)
    bias = (drift * dt * (1.0 - a) / a) ** 2 / 3.0
    return noise + walk + bias


def quality_scale(quality: Mapping[str, np.ndarray] | None, n: int,
                  ref: Mapping[str, float] | None = None) -> np.ndarray:
    """Per-frame measurement-variance scale ``s_k`` (1 at the reference quality):

    ``s = clip((rms/rms_ref)², ¼, 16) · n_tags_ref/max(n_tags, 1) · n_cam_ref/max(n_cam, 1)``,
    clipped to [¼, 50]: variance grows with the squared reprojection RMS and falls with the
    number of independent tags and cameras. Missing fields count as reference quality.
    """
    s = np.ones(n)
    if not quality:
        return s
    ref = dict(ref or {})
    rms = quality.get("reprojection_rms")
    if rms is not None and len(rms) == n:
        r = np.asarray(rms, dtype=np.float64)
        rr = ref.get("rms")
        if not rr:
            rr = float(np.nanmedian(r[r > 0])) if np.any(r > 0) else 1.0
        f = np.clip((r / rr) ** 2, 0.25, 16.0)
        s *= np.where(np.isfinite(f) & (r > 0), f, 1.0)
    for key, rk in (("n_tags", "n_tags"), ("n_cameras", "n_cameras")):
        v = quality.get(key)
        if v is not None and len(v) == n:
            v = np.asarray(v, dtype=np.float64)
            ref_v = ref.get(rk) or (float(np.nanmedian(v[v > 0])) if np.any(v > 0) else 1.0)
            s *= ref_v / np.maximum(np.where(np.isfinite(v), v, ref_v), 1.0)
    return np.clip(s, 0.25, 50.0)


def _in_intervals(t: np.ndarray, intervals: Sequence[tuple[float, float]]) -> np.ndarray:
    k = np.zeros(len(t), bool)
    for a, b in intervals:
        k |= (t >= a) & (t <= b)
    return k


def residual_structure(t_opt_s: np.ndarray, R_opt: np.ndarray, imu: ImuTrack,
                       calib: Calibration, mask: np.ndarray,
                       quality: Mapping[str, np.ndarray] | None = None,
                       quality_ref: Mapping[str, float] | None = None,
                       max_lag_s: float = 2.0) -> dict[str, Any]:
    """Structure function of the calibration residual ``r = Log(R_opt R_predᵀ)`` on
    ``mask`` (JTS-MM optical vs the calibrated IMU; no reference system).

    For frame lags ``Δ`` up to ``max_lag_s``, the per-axis robust variance (MAD²) of
    ``r(t+Δ) − r(t)``, each difference normalized to the reference quality by
    ``sqrt((s_k + s_{k+m})/2)``, is fitted (non-negative LS) by
    ``D(Δ) = 2σ² + q Δ + c Δ²``: ``σ²`` = white optical noise, ``q`` = random-walk density
    of the optical-vs-IMU attitude error (IMU increment and scale errors, pose-dependent
    optical bias), ``c`` = variance of a residual drift rate. Returns per-axis ``sigma2``,
    ``q``, ``c``, the 3x3 correlation of the one-frame differences, and the curve.
    """
    t = np.asarray(t_opt_s, dtype=np.float64)
    Rv, ok = valid_rotations(R_opt)
    m = ok & np.asarray(mask, bool) & (imu.segment(t, calib.lag_s) >= 0)
    idx = np.flatnonzero(m)
    r = np.full((len(t), 3), np.nan)
    if idx.size:
        r[idx] = _residuals(calib, t[idx], Rv[idx], imu)[0]
    e = np.linalg.norm(r, axis=1)
    med = float(np.nanmedian(e[m])) if m.any() else np.nan
    m &= np.isfinite(e) & (e <= max(5.0 * med, np.radians(1.0)))
    sk = quality_scale(quality, len(t), quality_ref)
    dt = float(np.median(np.diff(t)))
    M = max(1, int(round(max_lag_s / dt)))
    lags = np.unique(np.round(np.geomspace(1, M, 14)).astype(int))
    rows, D, corr = [], [], None
    for L in lags:
        i = np.flatnonzero(m[:-L] & m[L:])
        i = i[np.abs(t[i + L] - t[i] - L * dt) <= 0.5 * dt]
        if i.size < 30:
            continue
        d = (r[i + L] - r[i]) / np.sqrt(0.5 * (sk[i] + sk[i + L]))[:, None]
        mad = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0)
        rows.append(L * dt)
        D.append(mad ** 2)
        if corr is None:
            C = _robust_cov(d)
            sd = np.sqrt(np.diag(C))
            corr = C / np.outer(sd, sd)
    if len(rows) < 3:
        raise ValueError("residual_structure: too few consecutive training frames")
    x = np.asarray(rows)
    D = np.asarray(D)
    A = np.column_stack([2.0 * np.ones_like(x), x, x ** 2])
    sol = np.zeros((3, 3))
    for a in range(3):
        cols = [0, 1, 2]
        for _ in range(3):                              # active-set non-negative LS
            co = np.linalg.lstsq(A[:, cols], D[:, a], rcond=None)[0]
            if np.all(co >= 0) or len(cols) == 1:
                break
            cols = [c for c, v in zip(cols, co) if v > 0] or [0]
        sol[a, cols] = np.maximum(np.linalg.lstsq(A[:, cols], D[:, a], rcond=None)[0], 0)
    return {"sigma2": sol[:, 0], "q": sol[:, 1], "c": sol[:, 2], "corr": corr,
            "lags_s": x, "D": D, "n_frames": int(m.sum())}


def settings_from_sensors(t_opt_s: np.ndarray, R_opt: np.ndarray, imu: ImuTrack,
                          calib: Calibration, rest_intervals: Sequence[tuple[float, float]],
                          quality: Mapping[str, np.ndarray] | None = None, *,
                          train_mask: np.ndarray | None = None, gate_prob: float = 0.999,
                          reset_s: float = 1.0, max_lag_s: float = 2.0,
                          sf_default: float = 0.02, drift_floor_deg_min: float = 0.25,
                          tau_bounds_s: tuple[float, float] = (0.02, 5.0)) -> FusionSettings:
    """Filter settings from sensor characteristics only (JTS-MM optical and IMU; no
    reference data, no tuning). The formulas are fixed; only these inputs enter.

    * **Reference quality**: median ``reprojection_rms``, ``n_tags``, ``n_cameras`` of the
      rest frames.
    * **Rest characteristics** (always computed; the noise floor):
      optical ``Σ_rest`` = robust covariance of the second differences of consecutive rest
      frames, ``Log(R_{k+1}R_kᵀ) − Log(R_k R_{k−1}ᵀ)``, / 6 (white noise: var = 6σ²);
      IMU ``q_rest`` = per-axis robust variance of the chip's relative rotations at rest
      over one optical frame interval, / that interval.
    * **With** ``train_mask`` (recommended): the structure function of the calibration
      residual on the training frames (:func:`residual_structure`) gives ``σ²_a``, ``q_a``,
      ``c_a`` per axis. Then ``Σ_opt = diag(σ_a) · corr · diag(σ_a)`` (each ``σ_a²`` at least
      the rest value), ``q_θ = max(mean q_a, q_rest)``, the scale term is contained in ``q``
      (``sf = 0`` in the filter), and the drift prior ``SD(b₀)² = SE_w² + max(mean c_a,
      floor²/3)``. **Without** it: ``Σ_opt = Σ_rest``, ``q_θ = q_rest``, ``sf`` from the
      calibration (else ``sf_default``), ``SD(b₀)² = SE_w² + floor²/3``.
    * Drift: ``b₀`` = calibrated drift rate ``w``; ``q_b = max(|w|, floor)² / (3 T)``
      (``T`` = training span: the rate may wander by its own size over the take).
    * ``τ`` (F1) = argmin :func:`tau_cost` with ``σ² = tr Σ_opt / 3``, ``q = q_eff`` and the
      drift ``sqrt(|w|² + 3 c̄)`` (F1 has no drift state), within ``tau_bounds_s``.
    * Gate: χ²₃ quantile at ``gate_prob``.
    """
    t = np.asarray(t_opt_s, dtype=np.float64)
    Rv, ok = valid_rotations(R_opt)
    rest = _in_intervals(t, rest_intervals)
    dt = float(np.median(np.diff(t)))
    # optical second differences on consecutive rest frames
    k = np.flatnonzero(rest[1:-1] & rest[:-2] & rest[2:] & ok[1:-1] & ok[:-2] & ok[2:]) + 1
    k = k[(t[k + 1] - t[k] <= 1.5 * dt) & (t[k] - t[k - 1] <= 1.5 * dt)]
    if k.size < 20:
        raise ValueError(f"settings_from_sensors: only {k.size} consecutive rest frames")
    inc_a = log_so3(np.einsum("nij,nkj->nik", Rv[k], Rv[k - 1]))
    inc_b = log_so3(np.einsum("nij,nkj->nik", Rv[k + 1], Rv[k]))
    sigma_rest = _robust_cov(inc_b - inc_a) / 6.0
    qref: dict[str, float] = {}
    if quality:
        for key, name in (("reprojection_rms", "rms"), ("n_tags", "n_tags"),
                          ("n_cameras", "n_cameras")):
            v = quality.get(key)
            if v is not None and len(v) == len(t):
                vv = np.asarray(v, dtype=np.float64)[k]
                vv = vv[np.isfinite(vv) & (vv > 0)]
                if vv.size:
                    qref[name] = float(np.median(vv))
    # IMU relative rotations at rest over one optical frame interval (L samples apart)
    ti = imu.t_s
    rest_i = _in_intervals(ti, rest_intervals)
    dt_imu = float(np.median(np.diff(imu.t_us))) * 1e-6
    L = max(1, int(round(dt / dt_imu)))
    seg = imu._segments()
    i0 = np.flatnonzero(rest_i[:-L] & rest_i[L:] & (seg[:-L] == seg[L:]))
    if i0.size < 50:
        raise ValueError("settings_from_sensors: too few IMU samples at rest")
    Rq = quat_to_matrix(imu.q)
    inc = log_so3(np.einsum("nji,njk->nik", Rq[i0], Rq[i0 + L]))
    cov_i = _robust_cov(inc)
    span_i = float(np.median(ti[i0 + L] - ti[i0]))
    q_rest = float(np.trace(cov_i) / 3.0 / span_i)
    floor = drift_floor_deg_min * DEG / 60.0
    w = np.asarray(calib.drift_rad_s, dtype=np.float64)
    wn = float(np.linalg.norm(w))
    se_w = calib.drift_se_rad_s if np.isfinite(calib.drift_se_rad_s) else wn
    struct = None
    if train_mask is not None:
        struct = residual_structure(t, R_opt, imu, calib, train_mask, quality, qref,
                                    max_lag_s)
        s2 = np.maximum(struct["sigma2"], np.diag(sigma_rest))
        sd = np.sqrt(s2)
        sigma_opt = struct["corr"] * np.outer(sd, sd)
        q_theta = max(float(np.mean(struct["q"])), q_rest)
        sf = 0.0
        c_bar = float(np.mean(struct["c"]))
        b_var = se_w ** 2 + max(c_bar, floor ** 2 / 3.0)
    else:
        sigma_opt, q_theta = sigma_rest, q_rest
        sf = calib.sf_sigma if np.isfinite(calib.sf_sigma) else sf_default
        c_bar = 0.0
        b_var = se_w ** 2 + floor ** 2 / 3.0
    T = max(calib.span_s, 60.0)
    q_b = max(wn, floor) ** 2 / (3.0 * T)
    omega = calib.omega_rms_rad_s if np.isfinite(calib.omega_rms_rad_s) else 1.0
    s = FusionSettings(tau_s=np.nan, sigma_opt=sigma_opt, q_theta=q_theta, sf_sigma=float(sf),
                       q_bias=float(q_b), bias0=w.copy(), bias0_sd=float(np.sqrt(b_var)),
                       omega_rms=omega, dt_opt_s=dt, gate_prob=gate_prob,
                       gate_chi2=chi2_3_ppf(gate_prob), reset_s=reset_s,
                       lost_rate=max(2.0 * omega, 0.5), quality_ref=qref)
    sigma2 = float(np.trace(sigma_opt) / 3.0)
    drift_f1 = float(np.sqrt(wn ** 2 + 3.0 * c_bar))
    taus = np.geomspace(*tau_bounds_s, 400)
    J = np.array([tau_cost(x, dt, sigma2, s.q_eff(), drift_f1) for x in taus])
    s.tau_s = float(taus[int(np.argmin(J))])
    s.derivation = {
        "source": "train_structure" if struct is not None else "rest",
        "n_rest_frames": int(k.size), "n_rest_imu": int(i0.size), "dt_imu_s": dt_imu,
        "imu_pair_samples": L,
        "sigma_opt_rest_deg_axis": np.degrees(np.sqrt(np.diag(sigma_rest))).tolist(),
        "q_rest": q_rest,
        "imu_increment_sd_deg": np.degrees(np.sqrt(np.diag(cov_i))).tolist(),
        "sigma_opt_deg_rms": float(np.degrees(np.sqrt(np.trace(sigma_opt)))),
        "calib_sf_sigma": float(calib.sf_sigma),
        "drift_deg_min": float(np.degrees(wn) * 60.0), "span_s": T,
        "drift_f1_deg_min": float(np.degrees(drift_f1) * 60.0),
        "tau_cost_min_deg": float(np.degrees(np.sqrt(J.min()))),
    }
    if struct is not None:
        s.derivation.update({
            "struct_sigma_deg_axis": np.degrees(np.sqrt(struct["sigma2"])).tolist(),
            "struct_q": struct["q"].tolist(),
            "struct_c_deg_s": np.degrees(np.sqrt(struct["c"])).tolist(),
            "struct_n_frames": struct["n_frames"],
            "struct_lags_s": struct["lags_s"].tolist(),
            "struct_D_deg2": np.degrees(np.degrees(struct["D"])).tolist()})
    return s


# ---------------------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------------------
@dataclass
class FusionResult:
    """Estimator output. ``R_query`` causal (forward) estimate at the query times,
    ``R_query_smooth`` the RTS-smoothed one (F2 only). Per optical frame: ``accepted``
    (used as a measurement), ``gated`` (rejected by the gate), ``nis`` (NaN when not
    evaluated), ``innov_deg``."""

    R_query: np.ndarray
    R_query_smooth: np.ndarray | None
    accepted: np.ndarray
    gated: np.ndarray
    nis: np.ndarray
    innov_deg: np.ndarray
    resets: list[int]
    init_frame: int
    diagnostics: dict[str, Any]


def f0_optical(t_opt_s: np.ndarray, R_opt: np.ndarray, t_query_s: np.ndarray,
               mode: str = "slerp", max_gap_s: float | None = None) -> np.ndarray:
    """Optical-only orientation at the query times: SLERP between the bracketing valid
    frames (``"slerp"``) or the last valid frame (``"hold"``, causal). NaN outside the
    valid span or across a gap wider than ``max_gap_s`` (None: any gap is bridged)."""
    t = np.asarray(t_opt_s, dtype=np.float64)
    tq = np.asarray(t_query_s, dtype=np.float64)
    Rv, ok = valid_rotations(R_opt)
    out = np.full((len(tq), 3, 3), np.nan)
    tv, Rk = t[ok], Rv[ok]
    if tv.size == 0:
        return out
    j = np.searchsorted(tv, tq, side="right") - 1
    if mode == "hold":
        m = j >= 0
        if max_gap_s is not None:
            m &= tq - tv[np.clip(j, 0, None)] <= max_gap_s
        out[m] = Rk[j[m]]
        return out
    if mode != "slerp":
        raise ValueError(mode)
    jj = np.clip(j, 0, tv.size - 1)
    exact = (j >= 0) & (tq == tv[jj])
    m = (j >= 0) & (j < tv.size - 1)
    jm = np.clip(j, 0, max(tv.size - 2, 0))
    if tv.size >= 2:
        gap = tv[jm + 1] - tv[jm]
        if max_gap_s is not None:
            m &= gap <= max_gap_s
        f = np.clip((tq - tv[jm]) / np.maximum(gap, 1e-12), 0.0, 1.0)
        D = np.einsum("nji,njk->nik", Rk[jm], Rk[jm + 1])
        Ri = np.einsum("nij,njk->nik", Rk[jm], exp_so3(log_so3(D) * f[:, None]))
        out[m] = Ri[m]
    out[exact] = Rk[jj[exact]]
    return out


class _Prop:
    """Shared IMU propagation between optical frames (lag-shifted, segment-aware)."""

    def __init__(self, t: np.ndarray, imu: ImuTrack, calib: Calibration) -> None:
        self.imu, self.X, self.lag = imu, calib.X, calib.lag_s
        self.Ri = imu.rot(t, self.lag)
        self.seg = imu.segment(t, self.lag)
        self.path = imu.path_angle(t, self.lag)

    def delta(self, a: int, b: int) -> np.ndarray | None:
        """Marker-frame relative rotation from frame a to frame b, None across a gap."""
        if self.seg[a] < 0 or self.seg[a] != self.seg[b]:
            return None
        return self.X.T @ self.Ri[a].T @ self.Ri[b] @ self.X

    def query(self, t_anchor: np.ndarray, R_anchor: np.ndarray, b_anchor: np.ndarray | None,
              Ri_anchor: np.ndarray, seg_anchor: np.ndarray, tq: np.ndarray
              ) -> np.ndarray:
        """``Exp(b (t − t_a)) R_a Xᵀ R_imu,aᵀ R_imu(t) X``; hold where no IMU link."""
        Rq = self.imu.rot(tq, self.lag)
        sq = self.imu.segment(tq, self.lag)
        link = (sq >= 0) & (sq == seg_anchor)
        D = np.einsum("ji,nkj,nkl,lm->nim", self.X, Ri_anchor, Rq, self.X)
        D[~link] = np.eye(3)
        out = np.einsum("nij,njk->nik", R_anchor, D)
        if b_anchor is not None:
            out = np.einsum("nij,njk->nik", exp_so3(b_anchor * (tq - t_anchor)[:, None]), out)
        return out


def _find_init(i0: int, order: np.ndarray, t: np.ndarray, Rv: np.ndarray, prop: _Prop,
               thr_rad: float, n_check: int = 5, win_s: float = 0.5) -> int:
    """First position ``p ≥ i0`` in ``order`` whose optical rotation, propagated with the
    IMU, agrees (median angle ≤ ``thr_rad``) with the next ``n_check`` valid frames
    within ``win_s``: a flipped or mislabelled frame does not start the filter."""
    for p in range(i0, len(order)):
        a = order[p]
        nxt = order[p + 1: p + 1 + n_check]
        nxt = nxt[t[nxt] - t[a] <= win_s]
        if nxt.size < min(3, n_check):
            continue
        d = []
        for b in nxt:
            D = prop.delta(a, b)
            if D is None:
                continue
            d.append(float(np.radians(rot_error_deg(Rv[a] @ D, Rv[b]))))
        if len(d) >= min(3, n_check) and float(np.median(d)) <= thr_rad:
            return p
    return -1


def _frames(t_opt_s: np.ndarray, R_opt: np.ndarray, drop_mask: np.ndarray | None
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(t_opt_s, dtype=np.float64)
    Rv, ok = valid_rotations(R_opt)
    if drop_mask is not None:
        ok &= ~np.asarray(drop_mask, bool)
    return t, Rv, ok, np.flatnonzero(ok)


def f1_complementary(t_opt_s: np.ndarray, R_opt: np.ndarray, imu: ImuTrack,
                     calib: Calibration, settings: FusionSettings, t_query_s: np.ndarray, *,
                     tau_s: float | None = None, quality: Mapping[str, np.ndarray] | None = None,
                     drop_mask: np.ndarray | None = None) -> FusionResult:
    """F1: causal SO(3) complementary filter.

    Propagate with the IMU (marker-frame relative rotation), then pull toward the optical
    rotation, ``R ← Exp(α y) R``, ``y = Log(R_opt Rᵀ)``, ``α = 1 − exp(−Δt/τ)`` (Δt since the
    previous valid frame). Gate: ``|y|² > χ²₃ · v`` with the predicted per-axis spread
    ``v = s_k σ² + J(τ) + q_eff Δt_u + (|b| Δt_u)²/3`` (Δt_u since the last accepted frame);
    consistent gating for ``reset_s`` re-initializes. ``drop_mask`` removes frames.
    """
    tau = settings.tau_s if tau_s is None else float(tau_s)
    t, Rv, ok, order = _frames(t_opt_s, R_opt, drop_mask)
    n = len(t)
    prop = _Prop(t, imu, calib)
    s_k = quality_scale(quality, n, settings.quality_ref)
    sig2 = float(np.trace(settings.sigma_opt) / 3.0)
    wn = float(np.linalg.norm(settings.bias0))
    j_ss = tau_cost(tau, settings.dt_opt_s, sig2, settings.q_eff(), wn)
    thr = max(3.0 * math.sqrt(3.0 * sig2), np.radians(2.0))
    accepted = np.zeros(n, bool)
    gated = np.zeros(n, bool)
    innov = np.full(n, np.nan)
    Rpost = np.full((n, 3, 3), np.nan)
    resets: list[int] = []
    p0 = _find_init(0, order, t, Rv, prop, thr)
    if p0 < 0:
        raise ValueError("f1: no consistent optical start")
    a = order[p0]
    R = Rv[a].copy()
    Rpost[a], accepted[a] = R, True
    t_acc, run, run_y = t[a], [], []
    prev = a
    for b in order[p0 + 1:]:
        D = prop.delta(prev, b)
        dt = t[b] - t[prev]
        R = R @ D if D is not None else R
        y = log_so3(Rv[b] @ R.T)
        dtu = t[b] - t_acc
        lost = 0.0 if D is not None else (settings.lost_rate * dt) ** 2
        v = s_k[b] * sig2 + j_ss + settings.q_eff() * dtu + (wn * dtu) ** 2 / 3.0 + lost
        innov[b] = float(np.degrees(np.linalg.norm(y)))
        if float(y @ y) > settings.gate_chi2 * v:
            gated[b] = True
            run.append(b)
            run_y.append(y)
            if t[run[-1]] - t[run[0]] >= settings.reset_s and len(run) >= 5:
                yy = np.array(run_y)
                spread = float(np.sqrt(np.mean(np.sum((yy - yy.mean(0)) ** 2, axis=1))))
                if spread <= 3.0 * math.sqrt(3.0 * sig2 * float(np.median(s_k[run]))):
                    R = Rv[b].copy()
                    resets.append(int(b))
                    gated[b], accepted[b] = False, True
                    t_acc, run, run_y = t[b], [], []
        else:
            alpha = 1.0 - math.exp(-dt / tau)
            R = exp_so3(alpha * y) @ R
            accepted[b] = True
            t_acc, run, run_y = t[b], [], []
        Rpost[b] = R
        prev = b
    steps = order[p0:]
    Rq = _query_forward(t, Rpost, None, prop, steps, np.asarray(t_query_s, dtype=np.float64))
    diag = {"tau_s": tau, "n_frames": int(ok.sum()), "n_accepted": int(accepted.sum()),
            "n_gated": int(gated.sum()), "n_resets": len(resets),
            "gated_frames": np.flatnonzero(gated).tolist()}
    return FusionResult(Rq, None, accepted, gated, np.full(n, np.nan), innov, resets,
                        int(a), diag)


def _query_forward(t: np.ndarray, Rs: np.ndarray, bs: np.ndarray | None, prop: _Prop,
                   steps: np.ndarray, tq: np.ndarray) -> np.ndarray:
    out = np.full((len(tq), 3, 3), np.nan)
    if steps.size == 0:
        return out
    ts = t[steps]
    j = np.searchsorted(ts, tq, side="right") - 1
    m = j >= 0
    if not m.any():
        return out
    a = steps[j[m]]
    out[m] = prop.query(t[a], Rs[a], None if bs is None else bs[a], prop.Ri[a], prop.seg[a],
                        tq[m])
    return out


def f2_kalman(t_opt_s: np.ndarray, R_opt: np.ndarray, imu: ImuTrack, calib: Calibration,
              settings: FusionSettings, t_query_s: np.ndarray, *,
              quality: Mapping[str, np.ndarray] | None = None,
              drop_mask: np.ndarray | None = None, smooth: bool = True) -> FusionResult:
    """F2: error-state Kalman filter on SO(3) with a drift-rate state, Mahalanobis gating
    and an RTS smoother.

    State: nominal ``(R, b)``; error ``x = [δθ, δb]`` with ``R_true = Exp(δθ) R``.
    Propagation between consecutive valid optical frames: ``R⁻ = Exp(b Δt) R D`` with the
    IMU relative rotation ``D`` (no IMU link: ``D = I`` and the attitude variance grows
    by ``(lost_rate Δt)²``), ``F = [[I, Δt I], [0, I]]``,
    ``Q = [[(q_θ Δt + sf² θ²/3 + q_b Δt³/3) I, q_b Δt²/2 I], [q_b Δt²/2 I, q_b Δt I]]``
    (``θ`` = IMU path angle over Δt). Measurement ``y = Log(R_opt R⁻ᵀ)``,
    ``S = P_θθ + s_k Σ_opt``, NIS = ``yᵀ S⁻¹ y``; frames with NIS > ``gate_chi2`` are gated
    (logged, not used). A run of gated frames longer than ``reset_s`` whose innovations
    agree with each other re-initializes the filter (a new smoother chain).
    The query output interpolates the smoothed states: forward IMU propagation from the
    previous step and backward from the next, blended geodesically by the time fraction.
    """
    t, Rv, ok, order = _frames(t_opt_s, R_opt, drop_mask)
    n = len(t)
    tq = np.asarray(t_query_s, dtype=np.float64)
    prop = _Prop(t, imu, calib)
    s_k = quality_scale(quality, n, settings.quality_ref)
    So = settings.sigma_opt
    sig2 = float(np.trace(So) / 3.0)
    I3 = np.eye(3)
    thr = max(3.0 * math.sqrt(3.0 * sig2), np.radians(2.0))
    p0 = _find_init(0, order, t, Rv, prop, thr)
    if p0 < 0:
        raise ValueError("f2: no consistent optical start")
    steps = order[p0:]
    m = len(steps)
    Rpri = np.full((n, 3, 3), np.nan)
    Rpost = np.full((n, 3, 3), np.nan)
    bpri = np.full((n, 3), np.nan)
    bpost = np.full((n, 3), np.nan)
    Ppri = np.full((m, 6, 6), np.nan)
    Ppost = np.full((m, 6, 6), np.nan)
    Fs = np.full((m, 6, 6), np.nan)
    chain_start = np.zeros(m, bool)
    accepted = np.zeros(n, bool)
    gated = np.zeros(n, bool)
    nis = np.full(n, np.nan)
    innov = np.full(n, np.nan)
    resets: list[int] = []

    def p_init(k: int) -> np.ndarray:
        P = np.zeros((6, 6))
        P[:3, :3] = s_k[k] * So
        P[3:, 3:] = settings.bias0_sd ** 2 * I3
        return P

    a = steps[0]
    R, b, P = Rv[a].copy(), settings.bias0.astype(np.float64).copy(), p_init(a)
    Rpri[a] = Rpost[a] = R
    bpri[a] = bpost[a] = b
    Ppri[0] = Ppost[0] = P
    Fs[0] = np.eye(6)
    chain_start[0] = True
    accepted[a] = True
    run: list[int] = []
    run_y: list[np.ndarray] = []
    qb = settings.q_bias
    for i in range(1, m):
        k, kp = steps[i], steps[i - 1]
        dt = float(t[k] - t[kp])
        D = prop.delta(kp, k)
        F = np.eye(6)
        F[:3, 3:] = dt * I3
        Q = np.zeros((6, 6))
        if D is not None:
            th = float(prop.path[k] - prop.path[kp])
            qa = settings.q_theta * dt + settings.sf_sigma ** 2 * th ** 2 / 3.0
            R = exp_so3(b * dt) @ R @ D
        else:
            qa = settings.q_theta * dt + (settings.lost_rate * dt) ** 2
            R = exp_so3(b * dt) @ R
        Q[:3, :3] = (qa + qb * dt ** 3 / 3.0) * I3
        Q[:3, 3:] = Q[3:, :3] = qb * dt ** 2 / 2.0 * I3
        Q[3:, 3:] = qb * dt * I3
        P = F @ P @ F.T + Q
        Rpri[k], bpri[k], Ppri[i], Fs[i] = R, b, P, F
        y = log_so3(Rv[k] @ R.T)
        S = P[:3, :3] + s_k[k] * So
        Si = np.linalg.inv(S)
        nk = float(y @ Si @ y)
        nis[k] = nk
        innov[k] = float(np.degrees(np.linalg.norm(y)))
        if nk > settings.gate_chi2:
            gated[k] = True
            run.append(k)
            run_y.append(y)
            if t[run[-1]] - t[run[0]] >= settings.reset_s and len(run) >= 5:
                yy = np.array(run_y)
                spread = float(np.sqrt(np.mean(np.sum((yy - yy.mean(0)) ** 2, axis=1))))
                if spread <= 3.0 * math.sqrt(3.0 * sig2 * float(np.median(s_k[run]))):
                    R, P = Rv[k].copy(), p_init(k)
                    P[3:, 3:] = Ppri[i][3:, 3:]
                    resets.append(int(k))
                    chain_start[i] = True
                    gated[k], accepted[k] = False, True
                    run, run_y = [], []
        else:
            K = P[:, :3] @ Si
            dx = K @ y
            R = exp_so3(dx[:3]) @ R
            b = b + dx[3:]
            IKH = np.eye(6)
            IKH[:, :3] -= K
            P = IKH @ P @ IKH.T + K @ (s_k[k] * So) @ K.T          # Joseph form
            accepted[k] = True
            run, run_y = [], []
        Rpost[k], bpost[k], Ppost[i] = R, b, P

    Rq = _query_forward(t, Rpost, bpost, prop, steps, tq)
    Rq_s = None
    Rsm = bsm = None
    if smooth:
        Rsm, bsm = Rpost.copy(), bpost.copy()
        Psm = Ppost.copy()
        for i in range(m - 2, -1, -1):
            if chain_start[i + 1]:
                continue
            k, k1 = steps[i], steps[i + 1]
            G = Ppost[i] @ Fs[i + 1].T @ np.linalg.inv(Ppri[i + 1])
            e = np.concatenate([log_so3(Rsm[k1] @ Rpri[k1].T), bsm[k1] - bpri[k1]])
            dx = G @ e
            Rsm[k] = exp_so3(dx[:3]) @ Rpost[k]
            bsm[k] = bpost[k] + dx[3:]
            Psm[i] = Ppost[i] + G @ (Psm[i + 1] - Ppri[i + 1]) @ G.T
        Rq_s = _query_smooth(t, Rsm, bsm, prop, steps, chain_start, tq)
    fin = np.isfinite(nis)
    nis_v = nis[fin & ~gated]
    q95 = chi2_3_ppf(0.95)
    diag = {"n_frames": int(ok.sum()), "n_accepted": int(accepted.sum()),
            "n_gated": int(gated.sum()), "n_resets": len(resets),
            "gated_frames": np.flatnonzero(gated).tolist(),
            "nis_mean": float(np.mean(nis_v)) if nis_v.size else np.nan,
            "nis_median": float(np.median(nis_v)) if nis_v.size else np.nan,
            "nis_frac95": float(np.mean(nis[fin] <= q95)) if fin.any() else np.nan,
            "nis_frac95_accepted": float(np.mean(nis_v <= q95)) if nis_v.size else np.nan,
            "bias_final_deg_min": (np.degrees(bpost[steps[-1]]) * 60.0).tolist(),
            "R_frames_filter": Rpost, "R_frames_smooth": Rsm}
    return FusionResult(Rq, Rq_s, accepted, gated, nis, innov, resets, int(a), diag)


def _query_smooth(t: np.ndarray, Rs: np.ndarray, bs: np.ndarray, prop: _Prop,
                  steps: np.ndarray, chain_start: np.ndarray, tq: np.ndarray) -> np.ndarray:
    out = np.full((len(tq), 3, 3), np.nan)
    ts = t[steps]
    j = np.searchsorted(ts, tq, side="right") - 1
    m = len(steps)
    # forward from the previous step
    fw = j >= 0
    Ra = np.full((len(tq), 3, 3), np.nan)
    if fw.any():
        a = steps[j[fw]]
        Ra[fw] = prop.query(t[a], Rs[a], bs[a], prop.Ri[a], prop.seg[a], tq[fw])
    # backward from the next step of the same chain
    jn = j + 1
    bw = (jn < m)
    bw[bw] &= ~chain_start[np.clip(jn[bw], 0, m - 1)] | (j[bw] < 0)
    Rb = np.full((len(tq), 3, 3), np.nan)
    if bw.any():
        c = steps[jn[bw]]
        Rb[bw] = prop.query(t[c], Rs[c], bs[c], prop.Ri[c], prop.seg[c], tq[bw])
    both = fw & bw
    only_a = fw & ~bw                              # after the last step / chain end
    out[only_a] = Ra[only_a]                       # (nothing before the first step)
    if both.any():
        ja = j[both]
        f = (tq[both] - ts[ja]) / np.maximum(ts[ja + 1] - ts[ja], 1e-12)
        f = np.clip(f, 0.0, 1.0)
        # links: when the IMU is missing at either end, use the other side only
        D = np.einsum("nij,nkj->nik", Rb[both], Ra[both])
        out[both] = np.einsum("nij,njk->nik", exp_so3(log_so3(D) * f[:, None]), Ra[both])
    return out


# ---------------------------------------------------------------------------------------
# Synthetic scenario (tests, method checks)
# ---------------------------------------------------------------------------------------
def simulate(duration_s: float = 60.0, *, seed: int = 0, opt_rate_hz: float = 40.0,
             imu_rate_hz: float = 350.0, lag_s: float = 0.010,
             heading_drift_deg_min: float = 1.5, imu_noise_deg: float = 0.02,
             scale_err: float = 0.01, opt_noise_deg: Sequence[float] = (0.35, 0.35, 0.7),
             amp_deg: float = 12.0, flip_frac: float = 0.01,
             dropouts: Sequence[tuple[float, float]] = ((20.0, 20.5), (41.0, 42.0)),
             rest: tuple[float, float] = (2.0, 10.0), query_rate_hz: float = 200.0,
             stamp_jitter_us: int = 150) -> dict[str, Any]:
    """Known jaw-like rotation trajectory with simulated BNO085 game-rotation-vector
    output (stamps late by ``lag_s``, heading drift about the IMU gravity axis, scale error
    on the excursion, white noise, jittered and repeated stamps, own heading) and 40-Hz
    optical rotations (anisotropic camera-frame noise scaled by a per-frame quality, flips
    of 20–180°, dropouts). The body is still during ``rest``.

    Returns a dict with ``t_opt_s``, ``R_opt``, ``quality``, ``flips`` (frame indices),
    ``imu_t_us``, ``imu_q``, ``t_query_s``, ``truth(t)`` (callable), ``Z``, ``X``,
    ``lag_s``, ``drift_rad_s`` (camera frame), ``rest``, ``dropouts``.
    """
    rng = np.random.default_rng(seed)
    freqs = np.array([[0.35, 1.1, 1.7], [0.5, 0.9, 2.3], [0.27, 1.3, 1.9]])
    phases = rng.uniform(0, 2 * np.pi, (3, 3))
    amps = np.radians(amp_deg) * np.array([[1.0, 0.5, 0.3], [0.5, 0.4, 0.2], [0.4, 0.3, 0.2]])
    R0 = exp_so3(np.radians([10.0, -20.0, 30.0]))
    Z = exp_so3(rng.normal(size=3) * 1.2)
    X = exp_so3(rng.normal(size=3) * 1.2)

    def envelope(t: np.ndarray) -> np.ndarray:
        e = np.ones_like(t)
        a, b = rest
        e[(t >= a) & (t <= b)] = 0.0
        ramp = 1.0
        up = (t > b) & (t < b + ramp)
        e[up] = 0.5 - 0.5 * np.cos(np.pi * (t[up] - b) / ramp)
        dn = (t > a - ramp) & (t < a)
        e[dn] = 0.5 + 0.5 * np.cos(np.pi * (t[dn] - (a - ramp)) / ramp)
        return e

    def theta(t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=np.float64)
        th = np.einsum("ak,nak->na", amps, np.sin(2 * np.pi * freqs[None] * t[:, None, None]
                                                    + phases[None]))
        return th * envelope(t)[:, None]

    def truth(t: np.ndarray) -> np.ndarray:
        return np.einsum("ij,njk->nik", R0, exp_so3(theta(np.asarray(t, dtype=np.float64))))

    psi_rate = np.radians(heading_drift_deg_min) / 60.0
    ez = np.array([0.0, 0.0, 1.0])
    drift_cam = -psi_rate * (Z @ ez)                     # R_opt = Exp(w t) Z R_rep X

    # IMU: reported = Rz(ψ(t)) · Zᵀ R0 Exp((1+sf) θ) Xᵀ · noise, stamped at t + lag
    t_true = np.arange(0.0, duration_s, 1.0 / imu_rate_hz)
    n_imu = len(t_true)
    R_imu = np.einsum("ji,jk,nkl,ml->nim", Z, R0, exp_so3((1 + scale_err) * theta(t_true)), X)
    R_imu = np.einsum("nij,njk->nik", exp_so3(np.outer(psi_rate * t_true, ez)), R_imu)
    R_imu = np.einsum("nij,njk->nik", R_imu,
                      exp_so3(rng.normal(size=(n_imu, 3)) * np.radians(imu_noise_deg)))
    q = _mat2quat(R_imu)
    q[rng.random(n_imu) < 0.5] *= -1.0                   # sign ambiguity as on the chip
    stamps = np.round((t_true + lag_s) * 1e6).astype(np.int64)
    stamps += rng.integers(-stamp_jitter_us, stamp_jitter_us + 1, n_imu)
    dup = rng.random(n_imu) < 0.002
    stamps[1:][dup[1:]] = stamps[:-1][dup[1:]]           # repeated stamps
    perm = np.arange(n_imu)
    sw = np.flatnonzero(rng.random(n_imu - 1) < 0.002)
    perm[sw], perm[sw + 1] = perm[sw + 1], perm[sw].copy()  # out-of-order reports
    imu_t_us, imu_q = stamps[perm], q[perm]

    # optical
    t_opt = np.arange(0.5, duration_s - 0.5, 1.0 / opt_rate_hz)
    n = len(t_opt)
    n_tags = rng.choice([1, 2, 2, 2, 3], n).astype(np.uint8)
    n_cam = np.where(rng.random(n) < 0.1, 1, 2).astype(np.uint8)
    rms = rng.gamma(9.0, 0.1, n).astype(np.float32)
    quality = {"n_tags": n_tags, "n_cameras": n_cam, "reprojection_rms": rms}
    s = quality_scale(quality, n, {"rms": 0.9, "n_tags": 2.0, "n_cameras": 2.0})
    L = np.diag(np.radians(opt_noise_deg))
    noise = (rng.normal(size=(n, 3)) @ L.T) * np.sqrt(s)[:, None]
    R_opt = np.einsum("nij,njk->nik", exp_so3(noise), truth(t_opt))
    flips = np.flatnonzero(rng.random(n) < flip_frac)
    if flips.size:
        ax = rng.normal(size=(flips.size, 3))
        ax /= np.linalg.norm(ax, axis=1, keepdims=True)
        ang = np.radians(rng.uniform(20.0, 180.0, flips.size))
        R_opt[flips] = np.einsum("nij,njk->nik", R_opt[flips],
                                 exp_so3(ax * ang[:, None]))
    for a, b in dropouts:
        R_opt[(t_opt >= a) & (t_opt <= b)] = np.nan
    t_query = np.arange(1.0, duration_s - 1.0, 1.0 / query_rate_hz)
    return {"t_opt_s": t_opt, "R_opt": R_opt, "quality": quality, "flips": flips,
            "imu_t_us": imu_t_us, "imu_q": imu_q, "t_query_s": t_query, "truth": truth,
            "Z": Z, "X": X, "lag_s": lag_s, "drift_rad_s": drift_cam, "rest": rest,
            "dropouts": list(dropouts), "noise_scale": s}
