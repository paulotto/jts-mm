# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Camera-free jaw pose from the two IMUs (dataset X5, validation VD20).

Every other IMU estimator takes its calibration, heading or position from the optics. Here
the optics enter only as static calibration: the hand-eye IMU mountings (one session), the
landmark frame of the holds (``T_max_hp``, incisor, condyles) and, for E2a only, the closed
pose of the rest phase. During the tasks only the IMUs are used.

**Relative formulation (H4.0).** With the calibration model of :mod:`biomech.imu.calibration`,
``R_max<-mp(t) = A · M(t) · X`` with ``M = R_craᵀ · Rz(ψ) · R_mta``, the rotation since a
closed pose ``c`` is

    R(t) · R_cᵀ = A · M(t) · M_cᵀ · Aᵀ,

so the jaw mounting X cancels. An error δ of the head mounting A only conjugates the result
(the rotation axis tilts: opening error ≤ θ(1 − cos δ), off-axis components leak ≈ θ_off sin δ).
An error δ of ψ does the same only while the head keeps the tilt it had at the reference;
head tilt φ since then adds a rotation of ≈ δ·sin φ (ψ turns about the head's vertical, which
moves with the head). That is why ψ is re-fitted at every anchor (E3) and why E2b needs head
tilt to identify ψ at all.

**Estimators** (ladder H4.2; E0, E1, E4gen pooling, E7 and the statistics live in the stages):

* E2a: ψ from the rest-phase closed pose against the optical ``T_closed`` (:func:`psi_static`,
  uses X), the reference ``M_c`` from the IMUs at rest;
* E2b: ψ from the consistency of the IMU-detected closed poses under head tilt
  (:func:`psi_consistency`; X-free, "not identifiable" below 10° of head-tilt range);
* E3: E2 plus a reset at every IMU-detected closed pose (:func:`detect_closed`,
  :func:`anchored_rotation`): the anchor is the most closed 0.3 s of each detected run
  (causal running minimum of the opening); ψ re-fitted there against the reference closed
  pose (closed-form heading fit), the reference set to the current closed pose; causal
  ``hold`` (primary) or offline ``slerp`` between anchors. The anchor residuals split into a
  heading and a tilt part (:func:`drift_signature`);
* E4hinge / E4own / E4gen: incisor from the anchored rotation plus a condyle-midpoint
  translation (zero, or :class:`TranslationModel`);
* E5: :func:`accel_bridge`, the IMU-site difference integrated from both accelerometers
  (gravity removed with each IMU's own orientation, lever arms) by
  :func:`jtsmm.imu.position.position_kf`, anchored at the E3 anchor windows; E6:
  :func:`fuse_translation`, E5 plus the E4 prediction as pseudo-measurements;
* oracle analyses next to E4own (never camera-free results): X5b
  (:func:`oracle_incisors`, optical rotation + model translation and camera-free rotation +
  optical translation) and X5c (:func:`fit_task_models`, :func:`predict_by_task`: the E4own
  family per movement class, applied with the true task label).

**Fixed values** (set before computing, validated on synthetic data only;
:mod:`biomech.testing.synth_camerafree`, ``tests/test_x5_camerafree_synth.py``):
closed when the opening angle is below the rest-phase P90 + 1° and the relative angular
speed below 5°/s for ≥ 0.3 s; ψ identifiable from a head-tilt range of 10°; ridge λ by inner
blocked CV inside the calibration block; Kalman noise from the rest phase; scores after the
6 Hz zero-phase low-pass of the optical ground truth (order 2, ``se3.lowpass``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as Rot

from jtsmm.imu.batch import angular_velocity, quat_conj, quat_mul, quat_to_matrix
from jtsmm.imu.position import PositionSettings, position_kf
from jtsmm.imu.tilt import G

from biomech import timebase
from biomech.imu.analysis import _theta_y
from biomech.imu.preprocess import ImuStream
from biomech.kinematics import se3

EZ = np.array([0.0, 0.0, 1.0])

# Closed-pose detector (plan H4.2, fixed).
CLOSED: dict[str, float] = {
    "rest_quantile": 0.90,        # theta threshold = rest-phase P90 ...
    "margin_deg": 1.0,            # ... + 1 deg
    "speed_max_dps": 5.0,         # relative angular speed
    "min_s": 0.3,                 # for at least this long
    "speed_smooth_s": 0.1,        # causal moving mean of the relative angular velocity
}
PSI_MIN_TILT_RANGE_DEG = 10.0     # head-tilt range below which psi is not identifiable (H4.8)
FC_HZ = 6.0                       # the ground truth's pose low-pass (biomech.yaml lowpass_hz)
LP_ORDER = 2                      # ... and its order (lowpass_order)
HYST_DPS = 20.0                   # width of the smooth sign in the hysteresis feature
FEATURES = ("rx", "ry", "rz", "rx2", "ry2", "rz2", "hyst")

# accel_bridge: floors keep the Kalman filter regular on noise-free input (r_pos: the 0.05 mm
# floor of position.settings_from_data); the bias random walk is position.py's physical scale
# (20 mm/s³·√s).
BRIDGE: dict[str, float] = {"sigma_acc_floor_mm_s2": 1.0, "r_pos_floor_mm": 0.05,
                            "bias_rw": 20.0}

# Drift signature (H4.3 anchor-residual diagnostic; synthetic-only thresholds).
SIGNATURE: dict[str, float] = {
    "floor_deg": 0.3,             # net anchor residual below this: no drift
    "fusion_frac": 0.5,           # fusion-vs-accelerometer tilt change / tilt residual: "tilt"
    "step_frac": 0.5,             # largest anchor-to-anchor step / net change: "slip"
    "heading_ratio": 2.0,         # heading part / tilt part: "heading"
    "n_end": 3,                   # anchors averaged (median) at each end
}


# -- rotation helpers ---------------------------------------------------------------------------
def _rz(psi: float | np.ndarray) -> np.ndarray:
    """Rz(ψ): (3, 3) for a scalar, (N, 3, 3) for an array."""
    a = np.asarray(psi, float)
    c, s, z, o = np.cos(a), np.sin(a), np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, -s, z], -1), np.stack([s, c, z], -1),
                     np.stack([z, z, o], -1)], -2)


def _wrap(a: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _T(R: np.ndarray) -> np.ndarray:
    return np.swapaxes(R, -1, -2)


def _project(S: np.ndarray) -> np.ndarray:
    """Nearest rotation to ``S`` (Frobenius): the chordal mean of the summed rotations."""
    U, _, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U @ Vt))
    return U @ np.diag([1.0, 1.0, d]) @ Vt


def _chordal_mean(R: np.ndarray) -> np.ndarray:
    ok = np.isfinite(R.reshape(len(R), -1)).all(1)
    return _project(R[ok].sum(0)) if ok.any() else np.full((3, 3), np.nan)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vectors (rad) of (..., 3, 3); NaN where not finite."""
    R = np.asarray(R, float)
    flat = R.reshape(-1, 3, 3)
    out = np.full((len(flat), 3), np.nan)
    ok = np.isfinite(flat.reshape(len(flat), -1)).all(1)
    if ok.any():
        out[ok] = Rot.from_matrix(flat[ok]).as_rotvec()
    return out.reshape(R.shape[:-2] + (3,))


def _M(Rc: np.ndarray, Rm: np.ndarray, psi: float | np.ndarray) -> np.ndarray:
    """``M = R_craᵀ Rz(ψ) R_mta`` (R_ms<-cs up to the mountings), per sample."""
    return _T(Rc) @ _rz(psi) @ Rm


def _rel(Rc: np.ndarray, Rm: np.ndarray, A: np.ndarray, psi: float | np.ndarray,
         M_ref: np.ndarray) -> np.ndarray:
    """``A M(t) M_refᵀ Aᵀ``: rotation since the reference closed pose, maxillary frame."""
    return A @ _M(Rc, Rm, psi) @ _T(np.asarray(M_ref, float)) @ A.T


def _heading_fit(Rc: np.ndarray, Rm: np.ndarray, target: np.ndarray) -> float:
    """ψ minimising Σ‖R_craᵀ Rz(ψ) R_mta − target‖²_F (closed form).

    ‖·‖² = const − 2 tr(Rz(ψ) B) with B = Σ R_mta targetᵀ R_craᵀ, and
    tr(Rz(ψ) B) = cos ψ (B00 + B11) + sin ψ (B01 − B10) + B22."""
    ok = np.isfinite(Rc.reshape(len(Rc), -1)).all(1) & np.isfinite(Rm.reshape(len(Rm), -1)).all(1)
    B = (Rm[ok] @ _T(np.asarray(target, float)) @ _T(Rc[ok])).sum(0)
    return float(np.arctan2(B[0, 1] - B[1, 0], B[0, 0] + B[1, 1]))


def imu_product(q_cra: np.ndarray, q_mta: np.ndarray, psi: float | np.ndarray) -> np.ndarray:
    """``M(t) = R_craᵀ Rz(ψ) R_mta`` (N, 3, 3) from the two game rotation vectors."""
    return _M(quat_to_matrix(q_cra), quat_to_matrix(q_mta), psi)


def relative_rotation(q_cra: np.ndarray, q_mta: np.ndarray, A: np.ndarray,
                      psi: float | np.ndarray, M_ref: np.ndarray) -> np.ndarray:
    """E2: rotation of the mandible since the closed pose ``M_ref`` (maxillary frame),
    ``A M(t) M_refᵀ Aᵀ``; NaN where a quaternion is NaN."""
    return _rel(quat_to_matrix(q_cra), quat_to_matrix(q_mta), A, psi, M_ref)


def opening_deg(R_rel: np.ndarray) -> np.ndarray:
    """Opening angle (deg): the maxillary-y component of the rotation vector of ``R_rel``,
    as ``imu.analysis._theta_y`` (``R_rel`` is already relative to the closed pose)."""
    return _theta_y(np.asarray(R_rel, float), np.eye(3))


def closed_reference(q_cra: np.ndarray, q_mta: np.ndarray, psi: float,
                     mask: np.ndarray) -> np.ndarray:
    """``M_c``: chordal mean of ``M(t)`` over the closed samples ``mask`` (e.g. the rest
    phase): the IMU closed pose every relative rotation refers to."""
    m = np.asarray(mask, bool)
    return _chordal_mean(imu_product(q_cra[m], q_mta[m], psi))


# -- mountings and psi ----------------------------------------------------------------------------
@dataclass
class MountingPrior:
    A: np.ndarray        # R_max<-cra chip
    X: np.ndarray        # R_mta chip<-mp


def mounting_prior(R_mount_cra: np.ndarray, R_mount_mta: np.ndarray,
                   T_max_hp: np.ndarray) -> MountingPrior:
    """A and X from the hand-eye mountings (``R_mount`` = body<-chip, the convention of
    ``validation.fusion.load_mounting``) and the landmark frame of the head plate:
    ``A = T_max_hp[:3, :3] · R_mount,cra``, ``X = R_mount,mtaᵀ``."""
    return MountingPrior(np.asarray(T_max_hp, float)[:3, :3] @ np.asarray(R_mount_cra, float),
                         np.asarray(R_mount_mta, float).T)


def psi_static(q_cra: np.ndarray, q_mta: np.ndarray, A: np.ndarray, X: np.ndarray,
               R_closed: np.ndarray) -> float:
    """E2a: ψ (rad) from closed-pose samples (the rest phase) against the reference
    ``R_closed`` (optical R_max<-mp of the closed pose): the heading fit of
    ``R_craᵀ Rz(ψ) R_mta`` to ``Aᵀ R_closed Xᵀ``. An error of X enters ψ (up to its own
    size, through its component about the vertical)."""
    return _heading_fit(quat_to_matrix(q_cra), quat_to_matrix(q_mta),
                        A.T @ np.asarray(R_closed, float) @ X.T)


@dataclass
class PsiEstimate:
    psi_rad: float                 # NaN when not identifiable
    identifiable: bool
    tilt_range_deg: float          # largest angle between two head verticals over the poses
    n: int                         # closed samples used
    spread_deg: float              # RMS geodesic scatter of the closed poses at the optimum
    psi_best_rad: float            # the optimum even when not identifiable (diagnostic)
    closed: np.ndarray | None = None   # the closed samples used (when detected here)


def _consistency_parts(Rc: np.ndarray, Rm: np.ndarray) -> tuple[np.ndarray, ...]:
    """S(ψ) = Σ M_t(ψ) = S0 + sin ψ S1 + (1 − cos ψ) S2: ``R_craᵀ Rz(ψ) R_cra`` is the
    rotation by ψ about the head vertical ``v = R_craᵀ e_z`` (Rodrigues with K = [v]×)."""
    v = _T(Rc) @ EZ
    K = np.zeros((len(v), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -v[:, 2], v[:, 1], -v[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = v[:, 2], -v[:, 1], v[:, 0]
    M0 = _T(Rc) @ Rm
    KM = K @ M0
    return M0.sum(0), KM.sum(0), (K @ KM).sum(0)


def _nuc_so3(S: np.ndarray) -> np.ndarray:
    """max tr(Rᵀ S) over rotations R, for (..., 3, 3)."""
    U, s, Vt = np.linalg.svd(S)
    d = np.sign(np.linalg.det(U @ Vt))
    return s[..., 0] + s[..., 1] + d * s[..., 2]


def _consistency_fit(Rc: np.ndarray, Rm: np.ndarray, lo: float, hi: float) -> float:
    """ψ in [lo, hi] minimising Σ‖M_t(ψ) − M̄(ψ)‖²_F = 6N − 2 max tr(M̄ᵀ S(ψ))."""
    S0, S1, S2 = _consistency_parts(Rc, Rm)

    def cost(p: float | np.ndarray) -> np.ndarray:
        p = np.asarray(p, float)[..., None, None]
        return -_nuc_so3(S0 + np.sin(p) * S1 + (1 - np.cos(p)) * S2)

    grid = np.linspace(lo, hi, max(9, int(np.degrees(hi - lo)) + 1))
    j = int(np.argmin(cost(grid)))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    if b <= a:
        return float(grid[j])
    return float(minimize_scalar(lambda x: float(cost(x)), bounds=(a, b), method="bounded",
                                 options={"xatol": 1e-7}).x)


def _tilt_range_deg(Rc: np.ndarray) -> float:
    v = _T(Rc) @ EZ
    v = v[:: max(1, len(v) // 600)]
    if len(v) < 2:
        return 0.0
    return float(np.degrees(np.arccos(np.clip((v @ v.T).min(), -1.0, 1.0))))


def _thin(idx: np.ndarray, n_max: int) -> np.ndarray:
    return idx[np.linspace(0, len(idx) - 1, n_max).astype(int)] if len(idx) > n_max else idx


def psi_consistency(t_s: np.ndarray, q_cra: np.ndarray, q_mta: np.ndarray, A: np.ndarray,
                    rest_mask: np.ndarray, *, closed_mask: np.ndarray | None = None,
                    grid_deg: float = 10.0,
                    min_tilt_range_deg: float = PSI_MIN_TILT_RANGE_DEG,
                    params: dict[str, Any] | None = None, max_samples: int = 3000
                    ) -> PsiEstimate:
    """E2b: ψ from the consistency of the closed poses under head tilt (no optics, no X).

    At every closed pose the true ``M`` is the same; with ψ off by δ the poses scatter by
    rotations δ about each pose's head vertical, so ψ minimises their chordal scatter
    (closed form in ψ up to a 1-D search, :func:`_consistency_fit`). Head tilt is what makes
    the verticals differ: below ``min_tilt_range_deg`` (largest angle between two head
    verticals over the poses) ψ is "not identifiable" (``psi_rad`` NaN; the caller uses
    E2a, H4.8).

    ``closed_mask`` None: the closed poses are detected here, with ψ itself unknown. A wrong
    ψ turns head tilt into apparent jaw rotation (≈ δ·Δtilt), so each ψ on a ``grid_deg``
    grid runs the detector (:func:`detect_closed` on E2 with that ψ) with one common
    threshold -- the smallest rest-phase P90 over the grid + 1° (the rest-derived threshold
    of a wrong ψ would grow with its own spurious motion) -- and the ψ with the longest
    detected closed time is kept. The pose of each detected run is its most closed
    ``min_s`` window (lowest mean opening: the run's edges may still be closing, e.g. in
    protrusion, whose small rotation stays under the speed limit); ψ is refined on these
    poses, the poses re-detected with the plan's own threshold, and refined again."""
    p = {**CLOSED, **(params or {})}
    t = np.asarray(t_s, float)
    rest_mask = np.asarray(rest_mask, bool)
    Rc, Rm = quat_to_matrix(q_cra), quat_to_matrix(q_mta)
    fin = np.isfinite(Rc.reshape(len(Rc), -1)).all(1) & np.isfinite(Rm.reshape(len(Rm), -1)).all(1)
    fs = 1.0 / float(np.median(np.diff(t)))
    n_min = max(2, int(np.ceil(p["min_s"] * fs - 1e-9)))

    def e2(psi: float) -> tuple[np.ndarray, np.ndarray]:
        M_ref = _chordal_mean(_M(Rc[rest_mask & fin], Rm[rest_mask & fin], psi))
        return (opening_deg(_rel(Rc, Rm, A, psi, M_ref)),
                _speed(t, q_cra, q_mta, psi, p["speed_smooth_s"]))

    def poses(th: np.ndarray, det: ClosedDetection) -> np.ndarray:
        out = np.zeros(t.size, bool)
        for a, b in det.runs:
            j = _core_start(th[a:b], n_min)
            out[a + j:a + j + n_min] = True
        return out

    if closed_mask is None:
        grid = np.radians(np.arange(-180.0, 180.0, grid_deg))
        sig = [e2(float(g)) for g in grid]
        thr = min(closed_threshold(th, rest_mask, p)[0] for th, _ in sig)
        best = None
        for g, (th, sp) in zip(grid, sig):
            det = detect_closed(t, th, sp, threshold_deg=thr, params=p)
            if best is None or det.mask.sum() > best[0]:
                best = (int(det.mask.sum()), float(g), poses(th, det))
        _, psi, mask = best
        half = np.radians(grid_deg)
        for _ in range(2):
            idx = _thin(np.flatnonzero(mask & fin), max_samples)
            if idx.size < 2:
                break
            psi = _consistency_fit(Rc[idx], Rm[idx], psi - half, psi + half)
            th, sp = e2(psi)
            mask = poses(th, detect_closed(t, th, sp, rest_mask, params=p))
            half = np.radians(2.0)
    else:
        mask = np.asarray(closed_mask, bool)
        idx = _thin(np.flatnonzero(mask & fin), max_samples)
        psi = _consistency_fit(Rc[idx], Rm[idx], -np.pi, np.pi) if idx.size >= 2 else np.nan
    idx = _thin(np.flatnonzero(mask & fin), max_samples)
    if idx.size < 2 or not np.isfinite(psi):
        return PsiEstimate(np.nan, False, 0.0, int(idx.size), np.nan, np.nan, mask)
    psi = float(_wrap(psi))
    Mi = _M(Rc[idx], Rm[idx], psi)
    spread = float(np.degrees(np.sqrt(np.mean(np.sum(
        _rotvec(Mi @ _T(_chordal_mean(Mi))) ** 2, axis=1)))))
    rng = _tilt_range_deg(Rc[idx])
    ok = rng >= min_tilt_range_deg
    return PsiEstimate(psi if ok else np.nan, bool(ok), rng, int(idx.size), spread, psi, mask)


# -- closed-pose detection -------------------------------------------------------------------
def _causal_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Moving mean over the last ``n`` samples (NaN-aware), (N, ...)."""
    if n <= 1:
        return x
    ok = np.isfinite(x)
    cs = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(np.where(ok, x, 0.0), 0)])
    cn = np.concatenate([np.zeros((1,) + x.shape[1:]), np.cumsum(ok, 0)])
    lo = np.maximum(np.arange(1, len(x) + 1) - n, 0)
    k = (cn[1:] - cn[lo])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(k > 0, (cs[1:] - cs[lo]) / k, np.nan)


def _window_means(x: np.ndarray, n: int) -> np.ndarray:
    """Means of the ``n``-sample windows of ``x`` (entry j: window ending at sample j + n − 1)."""
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[n:] - cs[:-n]) / n


def _core_start(theta_run: np.ndarray, n: int) -> int:
    """Start of the ``n``-sample window with the lowest mean opening inside a run: the most
    closed part (teeth in contact are the opening's hard stop)."""
    if len(theta_run) <= n:
        return 0
    return int(np.argmin(_window_means(np.nan_to_num(theta_run, nan=1e6), n)))


def _speed(t_s: np.ndarray, q_cra: np.ndarray, q_mta: np.ndarray, psi: float,
           smooth_s: float) -> np.ndarray:
    t = np.asarray(t_s, float)
    qz = np.array([np.cos(psi / 2), 0.0, 0.0, np.sin(psi / 2)])
    q_rel = quat_mul(quat_mul(quat_conj(q_cra), qz), q_mta)       # quaternion of M(t)
    w = angular_velocity(np.round(t * 1e6), q_rel)
    fs = 1.0 / float(np.median(np.diff(t)))
    w = _causal_mean(w, max(1, int(round(smooth_s * fs))))
    return np.degrees(np.linalg.norm(w, axis=1))


def relative_speed_dps(t_s: np.ndarray, q_cra: np.ndarray, q_mta: np.ndarray, psi: float,
                       smooth_s: float = CLOSED["speed_smooth_s"]) -> np.ndarray:
    """Relative angular speed of the jaw against the head (deg/s): central differences of
    ``M(t)`` (``jtsmm.imu.batch.angular_velocity``) averaged causally over ``smooth_s``
    (the vector, then the norm: white quaternion noise averages out, motion does not)."""
    return _speed(t_s, q_cra, q_mta, psi, smooth_s)


@dataclass
class ClosedDetection:
    mask: np.ndarray                     # (N,) bool, samples of the detected runs
    runs: list[tuple[int, int]]          # half-open index ranges
    threshold_deg: float                 # opening threshold (rest P90 + margin)
    rest_p90_deg: float
    confirm: list[int] = field(default_factory=list)   # causal confirmation sample per run

    def intervals_s(self, t_s: np.ndarray) -> list[tuple[float, float]]:
        t = np.asarray(t_s, float)
        return [(float(t[a]), float(t[b - 1])) for a, b in self.runs]


def closed_threshold(theta_deg: np.ndarray, rest_mask: np.ndarray,
                     params: dict[str, Any] | None = None) -> tuple[float, float]:
    """(threshold, rest P90) of the opening angle: P90 over the rest phase + 1°."""
    p = {**CLOSED, **(params or {})}
    th = np.asarray(theta_deg, float)[np.asarray(rest_mask, bool)]
    th = th[np.isfinite(th)]
    p90 = float(np.quantile(th, p["rest_quantile"])) if th.size else 0.0
    return p90 + float(p["margin_deg"]), p90


def detect_closed(t_s: np.ndarray, theta_deg: np.ndarray, speed_dps: np.ndarray,
                  rest_mask: np.ndarray | None = None, *, threshold_deg: float | None = None,
                  params: dict[str, Any] | None = None) -> ClosedDetection:
    """Closed (intercuspal) poses: opening below the rest-phase P90 + 1° and relative angular
    speed below 5°/s, for at least 0.3 s (plan H4.2, fixed). ``threshold_deg`` overrides the
    rest-derived threshold (then ``rest_mask`` may be None). Each run is confirmed causally
    once it has lasted ``min_s`` (``confirm``)."""
    p = {**CLOSED, **(params or {})}
    t = np.asarray(t_s, float)
    if threshold_deg is None:
        if rest_mask is None:
            raise ValueError("detect_closed needs rest_mask or threshold_deg")
        thr, p90 = closed_threshold(theta_deg, rest_mask, p)
    else:
        thr, p90 = float(threshold_deg), np.nan
    fs = 1.0 / float(np.median(np.diff(t)))
    n_min = max(2, int(np.ceil(p["min_s"] * fs - 1e-9)))
    th, sp = np.asarray(theta_deg, float), np.asarray(speed_dps, float)
    ok = np.isfinite(th) & np.isfinite(sp) & (th < thr) & (sp < p["speed_max_dps"])
    runs = [(a, b) for a, b in timebase.segments(ok) if b - a >= n_min]
    mask = np.zeros(t.size, bool)
    for a, b in runs:
        mask[a:b] = True
    return ClosedDetection(mask, runs, thr, p90, [a + n_min - 1 for a, _ in runs])


# -- anchored rotation (E3) ----------------------------------------------------------------------
@dataclass
class Anchored:
    """E3 output. ``anchors`` (one row per closed run, arrays): ``t0_s``, ``t_confirm_s``,
    ``t1_s``, ``i0``, ``i1`` (the run), ``w0``, ``w1`` (its final anchor window, the most
    closed ``min_s``), ``psi_rad`` (ψ fitted there), ``heading_deg`` (ψ − ψ0: the heading part,
    cumulative), ``tilt_rotvec_deg`` (the closed pose after the heading fit, against the
    reference closed pose: the tilt part, cumulative, maxillary frame), ``resid_rotvec_deg``
    (the previous run's anchor read at this closed pose: what the estimate had drifted to
    before the reset), ``vertical_max`` (head vertical, maxillary frame)."""

    R_rel: np.ndarray                    # (N, 3, 3) rotation since the closed pose
    psi_rad: np.ndarray                  # (N,) heading offset in use
    closed: ClosedDetection
    anchors: dict[str, np.ndarray]
    mode: str
    psi0_rad: float
    M_ref: np.ndarray

    def anchor_mask(self) -> np.ndarray:
        """(N,) bool: the anchor windows (each run's most closed ``min_s``), the samples where
        the jaw is closed with confidence; the run edges may still be sliding (protrusion,
        laterotrusion turn little), so translation anchors (E5) use only these."""
        m = np.zeros(len(self.psi_rad), bool)
        for a, b in zip(self.anchors["w0"], self.anchors["w1"]):
            m[a:b] = True
        return m


def anchored_rotation(t_s: np.ndarray, q_cra: np.ndarray, q_mta: np.ndarray, A: np.ndarray,
                      psi0: float, M_ref: np.ndarray, rest_mask: np.ndarray, *,
                      mode: Literal["hold", "slerp"] = "hold",
                      params: dict[str, Any] | None = None) -> Anchored:
    """E3: the relative rotation reset at every IMU-detected closed pose.

    Causal detection on the anchored estimate itself: from each anchor on, the opening is
    measured against that anchor, so drift since the reference cannot hide a closure. A run
    of the detector (:func:`detect_closed` rules; threshold from E2 over the rest phase) is
    confirmed once it has lasted ``min_s``; at that sample an anchor is set from the window
    that confirmed it, and while the run lasts the anchor moves to every later ``min_s``
    window whose mean opening is a new minimum (a running minimum: still causal) by more than
    the noise of a window mean (``2 σ/√n``: σ the white noise of the E2 opening over the rest
    phase from successive differences, n the samples of a window; 0 on noise-free data).
    The edges of a run may still be closing -- in protrusion and laterotrusion the small
    rotation stays under the speed limit while the jaw slides (synthetic: up to 0.8 mm in
    the run's first 0.3 s) -- and the minimum of the opening is the hard stop of the teeth;
    the margin keeps noise from moving the anchor into the slow onset of the next movement.
    Setting an anchor from a window:

    * ψ is re-fitted against the reference closed pose ``M_ref`` (heading fit: only the
      rotation about the IMU world's vertical changes), so head tilt after the anchor does not
      meet a stale ψ;
    * the reference becomes the chordal mean of ``M(t; ψ_k)`` over the window: the estimate
      reads "closed" there (full reset, like a zero-velocity update; each anchor stands alone,
      so closure-to-closure scatter does not random-walk).

    ``hold`` (causal, primary): each anchor holds until the next one. ``slerp`` (offline):
    each run uses its final (most closed) anchor throughout; between two runs ψ is
    interpolated linearly and the residual left at the next run (``Δ = A M_{k+1} M_kᵀ Aᵀ``)
    is removed in proportion to the elapsed time, ``R = Exp(−u log Δ) · A M(t; ψ(t)) M_kᵀ Aᵀ``.
    Before the first anchor both are E2 (``psi0``, ``M_ref``)."""
    if mode not in ("hold", "slerp"):
        raise ValueError(f"mode must be 'hold' or 'slerp', not {mode!r}")
    p = {**CLOSED, **(params or {})}
    t = np.asarray(t_s, float)
    n = t.size
    rest_mask = np.asarray(rest_mask, bool)
    Rc, Rm = quat_to_matrix(q_cra), quat_to_matrix(q_mta)
    M_ref = np.asarray(M_ref, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    n_min = max(2, int(np.ceil(p["min_s"] * fs - 1e-9)))
    speed = _speed(t, q_cra, q_mta, float(psi0), p["speed_smooth_s"])
    th_e2 = opening_deg(_rel(Rc, Rm, A, psi0, M_ref))
    thr, p90 = closed_threshold(th_e2, rest_mask, p)
    still = np.isfinite(speed) & (speed < p["speed_max_dps"])
    horizon = max(int(30 * fs), 4 * n_min)
    # noise of a window mean: the white part of the rest-phase opening (successive
    # differences; slow changes -- a closure's offset, ψ drift under head tilt -- are not noise)
    margin = 2.0 * _white_sd(th_e2[rest_mask][:, None]) / np.sqrt(n_min)

    def frame(i: int, e: int, ps: float, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        R_ = _rel(Rc[i:e], Rm[i:e], A, ps, M)
        th = opening_deg(R_)
        return R_, th

    def run_end(i: int, ps: float, M: np.ndarray) -> int:
        """First sample at or after ``i`` that is not closed in the frame of (ps, M)."""
        while i < n:
            e = min(n, i + horizon)
            _, th = frame(i, e, ps, M)
            nf = np.flatnonzero(~((th < thr) & still[i:e]))
            if nf.size:
                return i + int(nf[0])
            i = e
        return n

    def anchor(w0: int, w1: int) -> tuple[float, np.ndarray]:
        ps = _heading_fit(Rc[w0:w1], Rm[w0:w1], M_ref)
        return ps, _chordal_mean(_M(Rc[w0:w1], Rm[w0:w1], ps))

    R_hold = np.full((n, 3, 3), np.nan)
    psi_hold = np.full(n, np.nan)
    psi_k, M_k = float(psi0), M_ref
    rows: list[tuple[Any, ...]] = []     # (a, c, b, w0, psi_prev, M_prev, psi, M)
    i = 0
    while i < n:
        e = min(n, i + horizon)
        R_, th = frame(i, e, psi_k, M_k)
        ok = (th < thr) & still[i:e]
        segs = timebase.segments(ok)
        hit = next(((a, b) for a, b in segs if b - a >= n_min), None)
        if hit is None:
            stop = e - i
            if segs and segs[-1][1] == e - i and e < n:
                stop = segs[-1][0]                   # unfinished run: re-examine from its start
            R_hold[i:i + stop], psi_hold[i:i + stop] = R_[:stop], psi_k
            i += stop
            continue
        a = i + hit[0]
        c = a + n_min - 1                            # confirmed: closed for min_s
        R_hold[i:c], psi_hold[i:c] = R_[:c - i], psi_k
        prev = (psi_k, M_k)
        psi_k, M_k = anchor(a, c + 1)
        # The run in the first anchor's frame: its running-minimum windows (new anchors).
        b = run_end(c, psi_k, M_k)
        _, th_run = frame(a, b, psi_k, M_k)
        wm = _window_means(np.nan_to_num(th_run, nan=1e6), n_min)   # wm[j]: window a+j..
        rec, cur = [], wm[0]
        for j in range(1, len(wm)):
            if wm[j] < cur - margin:
                rec.append(j)
                cur = wm[j]
        seg, w0 = c, a
        for j in rec:
            end = a + j + n_min - 1                  # the window's last sample: known now
            R_hold[seg:end] = _rel(Rc[seg:end], Rm[seg:end], A, psi_k, M_k)
            psi_hold[seg:end] = psi_k
            psi_k, M_k = anchor(a + j, end + 1)
            seg, w0 = end, a + j
        b = run_end(seg, psi_k, M_k)                 # the run's end in the final frame
        R_hold[seg:b] = _rel(Rc[seg:b], Rm[seg:b], A, psi_k, M_k)
        psi_hold[seg:b] = psi_k
        rows.append((a, c, b, w0, *prev, psi_k, M_k))
        i = b

    runs = [(r[0], r[2]) for r in rows]
    mask = np.zeros(n, bool)
    for a, b in runs:
        mask[a:b] = True
    closed = ClosedDetection(mask, runs, thr, p90, [r[1] for r in rows])
    table = _anchor_table(t, Rc, Rm, A, psi0, M_ref, rows, n_min)
    if mode == "hold" or not rows:
        return Anchored(R_hold, psi_hold, closed, table, mode, float(psi0), M_ref)

    # offline: each run on its final anchor, interpolation between runs
    R = R_hold.copy()
    ps_out = psi_hold.copy()
    for k, (a, _c, b, _w0, _pp, _Mp, ps, Mk) in enumerate(rows):
        R[a:b], ps_out[a:b] = _rel(Rc[a:b], Rm[a:b], A, ps, Mk), ps
        if k + 1 < len(rows):
            a2, ps2, M2 = rows[k + 1][0], rows[k + 1][6], rows[k + 1][7]
            g = slice(b, a2)
            if a2 > b:
                u = (t[g] - t[b - 1]) / (t[a2] - t[b - 1])
                psi_t = ps + float(_wrap(ps2 - ps)) * u
                F = _rel(Rc[g], Rm[g], A, psi_t, Mk)
                log_d = _rotvec(A @ M2 @ Mk.T @ A.T)
                R[g] = Rot.from_rotvec(-u[:, None] * log_d[None]).as_matrix() @ F
                ps_out[g] = psi_t
        else:
            R[b:], ps_out[b:] = _rel(Rc[b:], Rm[b:], A, ps, Mk), ps
    return Anchored(R, ps_out, closed, table, mode, float(psi0), M_ref)


def _anchor_table(t: np.ndarray, Rc: np.ndarray, Rm: np.ndarray, A: np.ndarray, psi0: float,
                  M_ref: np.ndarray, rows: Sequence[tuple[Any, ...]],
                  n_min: int) -> dict[str, np.ndarray]:
    keys = ("t0_s", "t_confirm_s", "t1_s", "i0", "i1", "w0", "w1", "psi_rad", "heading_deg",
            "tilt_rotvec_deg", "resid_rotvec_deg", "vertical_max")
    out: dict[str, list[Any]] = {k: [] for k in keys}
    for a, c, b, w0, pp, Mp, ps, Mn in rows:
        w = slice(w0, w0 + n_min)
        out["t0_s"].append(t[a])
        out["t_confirm_s"].append(t[c])
        out["t1_s"].append(t[b - 1])
        for k, v in (("i0", a), ("i1", b), ("w0", w0), ("w1", w0 + n_min)):
            out[k].append(v)
        out["psi_rad"].append(ps)
        out["heading_deg"].append(float(np.degrees(_wrap(ps - psi0))))
        out["tilt_rotvec_deg"].append(np.degrees(_rotvec(A @ Mn @ M_ref.T @ A.T)))
        M_prev_here = _chordal_mean(_M(Rc[w], Rm[w], pp))
        out["resid_rotvec_deg"].append(np.degrees(_rotvec(A @ M_prev_here @ Mp.T @ A.T)))
        v = (A @ _T(Rc[w]) @ EZ).mean(0)
        out["vertical_max"].append(v / np.linalg.norm(v))
    ints = ("i0", "i1", "w0", "w1")
    return {k: np.asarray(v, int if k in ints else float) for k, v in out.items()}


def drift_signature(anch: Anchored, t_s: np.ndarray, cra: ImuStream, mta: ImuStream,
                    params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Which drift mechanism the anchor residuals point to (H4.3 diagnostic).

    Per anchor: the heading part (ψ_k − ψ0) and the tilt part (the closed pose after the
    heading fit, against the reference) of the cumulative residual, and per IMU its
    *fusion tilt*: the horizontal part of the mean gravity direction ``R_imu f/|f|`` (deg)
    over the anchor window -- the game rotation vector's tilt error against its own
    accelerometer (zero for a correct fusion, whatever the heading; a constant accelerometer
    bias drops out of its change). Net changes are medians of the last ``n_end`` anchors
    minus the first ``n_end``.

    Signatures (each mechanism alone):

    * **heading drift** (ψ moves): heading part ≫ tilt part, fusion tilt unchanged, the
      change spread over the anchors (a ramp);
    * **tilt drift** (a fusion's tilt drifts, the accelerometer does not): tilt part
      ≈ fusion-tilt change of one IMU (both grow together);
    * **plate slip** (the IMU turns on the jaw): a step -- most of the net change between
      two consecutive anchors -- with the fusion tilt unchanged (quaternion and
      accelerometer turned together).

    Returns the features and ``label`` in {"none", "heading", "tilt", "slip", "unexplained"}
    (rules in that order: floor, fusion, step, heading ratio; thresholds ``SIGNATURE``)."""
    p = {**SIGNATURE, **(params or {})}
    an = anch.anchors
    k = len(an["t0_s"])
    ne = int(p["n_end"])
    if k < 2:
        return {"label": "none", "n_anchors": k}
    t = np.asarray(t_s, float)
    t_us = np.round(t * 1e6).astype(np.int64)
    fus = np.full((k, 2, 2), np.nan)                 # anchor, imu (cra, mta), horizontal xy
    for j, (a, b) in enumerate(zip(an["w0"], an["w1"])):
        for s, st in enumerate((cra, mta)):
            R = quat_to_matrix(st.quat_at(t_us[a:b]))
            f = st.accel_at(t_us[a:b])
            g = np.einsum("nij,nj->ni", R, f / np.linalg.norm(f, axis=1, keepdims=True))
            g = np.nanmean(g, axis=0)
            fus[j, s] = np.degrees(g[:2] / np.linalg.norm(g))

    def end(x: np.ndarray) -> np.ndarray:
        return np.nanmedian(x[-ne:], axis=0) - np.nanmedian(x[:ne], axis=0)

    head = an["heading_deg"]
    tilt = an["tilt_rotvec_deg"]
    d_head = float(abs(end(head)))
    d_tilt = float(np.linalg.norm(end(tilt)))
    d_fus = float(max(np.linalg.norm(end(fus[:, 0])), np.linalg.norm(end(fus[:, 1]))))
    c = np.column_stack([head, tilt])
    steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
    net = float(np.linalg.norm(end(c)))
    step = float(steps.max() / net) if net > 0 else 0.0
    total = float(np.hypot(d_head, d_tilt))
    if total < p["floor_deg"]:
        label = "none"
    elif d_fus >= p["fusion_frac"] * d_tilt and d_fus >= p["floor_deg"]:
        label = "tilt"
    elif step >= p["step_frac"]:
        label = "slip"
    elif d_head >= p["heading_ratio"] * d_tilt:
        label = "heading"
    else:
        label = "unexplained"
    return {"label": label, "n_anchors": k, "heading_deg": d_head, "tilt_deg": d_tilt,
            "fusion_tilt_deg": d_fus, "step_frac": step, "net_deg": total,
            "per_anchor": {"t_s": an["t0_s"], "heading_deg": head, "tilt_rotvec_deg": tilt,
                           "fusion_tilt_deg": fus}}


# -- the rotation ladder ------------------------------------------------------------------------
@dataclass
class RotationLadder:
    t_s: np.ndarray
    estimates: dict[str, np.ndarray]     # name -> (N, 3, 3) rotation since the closed pose
    psi0: dict[str, float]               # "a" / "b": psi of E2a / E2b (rad)
    psi_b: PsiEstimate
    anchored: dict[str, Anchored]        # "E3a", "E3a_slerp", "E3b", "E3b_slerp"
    M_ref: dict[str, np.ndarray]
    flags: dict[str, Any]


def rotation_ladder(t_us: np.ndarray, cra: ImuStream, mta: ImuStream, A: np.ndarray,
                    rest_mask: np.ndarray, *, X: np.ndarray | None = None,
                    R_closed: np.ndarray | None = None,
                    params: dict[str, Any] | None = None) -> RotationLadder:
    """E2a, E2b, E3a/E3b (hold and slerp) on the grid ``t_us`` (µs, the analysis grid).

    ``rest_mask``: grid samples of the rest phase (jaw closed by protocol). E2a needs ``X``
    and the optical closed pose ``R_closed``; without them only the b rungs run. When ψ is
    not identifiable from head tilt, E2b/E3b fall back to E2a (``flags["psi_b_fallback"]``,
    H4.8)."""
    p = {**CLOSED, **(params or {})}
    t_us = np.asarray(t_us, np.int64)
    t = t_us * 1e-6
    rest = np.asarray(rest_mask, bool)
    qc, qm = cra.quat_at(t_us), mta.quat_at(t_us)
    est: dict[str, np.ndarray] = {}
    psi0: dict[str, float] = {}
    refs: dict[str, np.ndarray] = {}
    if X is not None and R_closed is not None:
        psi0["a"] = psi_static(qc[rest], qm[rest], A, X, R_closed)
    pb = psi_consistency(t, qc, qm, A, rest, params=p)
    flags = {"psi_b_identifiable": pb.identifiable, "psi_b_fallback": False}
    if pb.identifiable:
        psi0["b"] = pb.psi_rad
    elif "a" in psi0:
        psi0["b"] = psi0["a"]
        flags["psi_b_fallback"] = True
    anchored: dict[str, Anchored] = {}
    for key, psi in psi0.items():
        refs[key] = closed_reference(qc, qm, psi, rest)
        est[f"E2{key}"] = relative_rotation(qc, qm, A, psi, refs[key])
        for mode in ("hold", "slerp"):
            name = f"E3{key}" + ("" if mode == "hold" else "_slerp")
            anchored[name] = anchored_rotation(t, qc, qm, A, psi, refs[key], rest, mode=mode,
                                               params=p)
            est[name] = anchored[name].R_rel
    return RotationLadder(t, est, psi0, pb, anchored, refs, flags)


# -- translation ---------------------------------------------------------------------------------
def translation_features(rotvec_deg: np.ndarray, theta_dot_dps: np.ndarray,
                         hyst_dps: float = HYST_DPS) -> np.ndarray:
    """(N, 7) regressors of the condyle-midpoint translation (E4, ``FEATURES``): the rotation
    vector (deg, maxillary frame), its squares and the hysteresis term ``sign(θ̇)·θ``
    with θ the opening (``rotvec_y``). The sign is the smooth ``tanh(θ̇ / hyst_dps)``: a hard
    sign would make the translation jump at every turning point (θ̇ = 0 at θ_max)."""
    r = np.asarray(rotvec_deg, float)
    th_d = np.asarray(theta_dot_dps, float)
    return np.column_stack([r, r ** 2, np.tanh(th_d / hyst_dps) * r[:, 1]])


def theta_rate_dps(t_s: np.ndarray, theta_deg: np.ndarray, smooth_s: float = 0.05) -> np.ndarray:
    """θ̇ (deg/s): central differences with a centred ``smooth_s`` moving mean, per finite
    segment (``timebase.gradient``)."""
    t = np.asarray(t_s, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    return timebase.gradient(t * 1e6, theta_deg, smooth_n=timebase.odd_window(smooth_s, fs))


def incisor_position(R_rel: np.ndarray, tau_mm: np.ndarray | None, incisor0: np.ndarray,
                     condyle0: np.ndarray) -> np.ndarray:
    """Incisor (maxillary frame): rotation about the condyle midpoint plus its translation,
    ``R (p0 − c0) + c0 + τ`` (τ None or 0: E4hinge)."""
    p = np.einsum("nij,j->ni", R_rel, np.asarray(incisor0, float) - condyle0) + condyle0
    return p if tau_mm is None else p + tau_mm


def condyle_translation(R_rel: np.ndarray, t_rel: np.ndarray, condyle0: np.ndarray) -> np.ndarray:
    """τ of a relative mandible pose ``[R_rel, t_rel]`` (optics): ``R c0 + t − c0``."""
    c0 = np.asarray(condyle0, float)
    return np.einsum("nij,j->ni", R_rel, c0) + t_rel - c0


@dataclass
class TranslationModel:
    """Ridge model of the condyle-midpoint translation from the rotation (E4own / E4gen).

    ``τ = F(r, θ̇) · coef`` without intercept (τ = 0 at the closed pose, where F = 0); the
    features are scaled to unit RMS for the penalty λ·n·I, λ chosen by blocked K-fold CV
    inside the calibration block (contiguous folds: neighbouring samples are not
    independent). A generic model (``fit_generic``) is fitted on the pooled participants with
    τ divided by each one's incisor–condyle distance and predicts τ times the test
    participant's distance."""

    coef: np.ndarray                     # (7, 3) on the raw features (mm, or mm/mm generic)
    lam: float
    cv_rmse: float                       # inner-CV RMS (mm; mm/mm when generic)
    n: int
    generic: bool = False
    hyst_dps: float = HYST_DPS
    cv_curve: dict[float, float] = field(default_factory=dict)

    @staticmethod
    def _solve(F: np.ndarray, Y: np.ndarray, lam: float, s: np.ndarray) -> np.ndarray:
        Z = F / s
        n = len(Z)
        c = np.linalg.solve(Z.T @ Z + lam * n * np.eye(Z.shape[1]), Z.T @ Y)
        return c / s[:, None]

    @classmethod
    def _fit(cls, F: np.ndarray, Y: np.ndarray, groups: np.ndarray | None,
             lambdas: Sequence[float] | None, n_folds: int, generic: bool,
             hyst_dps: float) -> "TranslationModel":
        ok = np.isfinite(F).all(1) & np.isfinite(Y).all(1)
        F, Y = F[ok], Y[ok]
        if len(F) < 2 * n_folds:
            raise ValueError("too few calibration samples for the translation model")
        lams = list(lambdas) if lambdas is not None else list(np.logspace(-8, 2, 21))
        s = np.sqrt(np.mean(F ** 2, axis=0))
        s[s <= 0] = 1.0
        if groups is not None and len(np.unique(groups[ok])) >= 2:
            g = groups[ok]
            folds = [np.flatnonzero(g == u) for u in np.unique(g)]
        else:
            folds = np.array_split(np.arange(len(F)), n_folds)
        curve = {}
        for lam in lams:
            sse = 0.0
            for te in folds:
                tr = np.setdiff1d(np.arange(len(F)), te)
                c = cls._solve(F[tr], Y[tr], lam, s)
                sse += float(np.sum((F[te] @ c - Y[te]) ** 2))
            curve[float(lam)] = float(np.sqrt(sse / len(F)))
        lam = min(curve, key=lambda k: (curve[k], -k))
        return cls(cls._solve(F, Y, lam, s), lam, curve[lam], len(F), generic, hyst_dps, curve)

    @classmethod
    def fit(cls, rotvec_deg: np.ndarray, theta_dot_dps: np.ndarray, tau_mm: np.ndarray, *,
            lambdas: Sequence[float] | None = None, n_folds: int = 5,
            hyst_dps: float = HYST_DPS) -> "TranslationModel":
        """E4own: one participant's calibration block (optical free_motion)."""
        F = translation_features(rotvec_deg, theta_dot_dps, hyst_dps)
        return cls._fit(F, np.asarray(tau_mm, float), None, lambdas, n_folds, False, hyst_dps)

    @classmethod
    def fit_generic(cls, blocks: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, float]], *,
                    lambdas: Sequence[float] | None = None,
                    hyst_dps: float = HYST_DPS) -> "TranslationModel":
        """E4gen: ``blocks`` = (rotvec_deg, theta_dot_dps, tau_mm, incisor–condyle distance
        mm) of the other participants; folds = participants (leave-one-out inside)."""
        F = np.vstack([translation_features(r, d, hyst_dps) for r, d, _, _ in blocks])
        Y = np.vstack([np.asarray(tau, float) / float(s) for _, _, tau, s in blocks])
        g = np.concatenate([np.full(len(r), j) for j, (r, *_rest) in enumerate(blocks)])
        return cls._fit(F, Y, g, lambdas, max(2, len(blocks)), True, hyst_dps)

    def predict(self, rotvec_deg: np.ndarray, theta_dot_dps: np.ndarray,
                scale_mm: float | None = None) -> np.ndarray:
        """τ (N, 3, mm). A generic model needs the participant's incisor–condyle distance."""
        F = translation_features(rotvec_deg, theta_dot_dps, self.hyst_dps)
        tau = F @ self.coef
        if self.generic:
            if scale_mm is None:
                raise ValueError("a generic translation model needs scale_mm")
            tau = tau * float(scale_mm)
        return tau


# -- oracle decomposition (X5b) and task-conditioned translation (X5c) ---------------------------
def oracle_incisors(R_est: np.ndarray, R_opt: np.ndarray, tau_model_opt_mm: np.ndarray,
                    tau_opt_mm: np.ndarray, incisor0: np.ndarray, condyle0: np.ndarray
                    ) -> dict[str, np.ndarray]:
    """X5b: the incisor (maxillary frame, mm) with one half of E4own replaced by the optics.

    * ``X5b_model``: optical rotation ``R_opt`` + the translation model driven by the optical
      rotation (``tau_model_opt_mm`` = τ̂(r_opt, θ̇_opt)): the translation-model error alone;
    * ``X5b_rotation``: the camera-free rotation ``R_est`` + the optical condyle-midpoint
      translation ``tau_opt_mm``: the rotation-induced error alone, ``(R_est − R_opt)(p0 − c0)``
      against the optics.

    The full E4own is ``incisor_position(R_est, τ̂(r_est, θ̇_est), …)``; its error is the sum of
    the two above plus the model's response to the rotation error, τ̂(r_est) − τ̂(r_opt), so the
    three RMSEs do not add."""
    return {"X5b_model": incisor_position(R_opt, tau_model_opt_mm, incisor0, condyle0),
            "X5b_rotation": incisor_position(R_est, tau_opt_mm, incisor0, condyle0)}


@dataclass
class TaskModel:
    """X5c: the E4own-family model of one scored task's movement class (``model`` None: not
    fitted, ``note`` says why)."""

    label_class: str                     # the label set, e.g. "OPEN+CLOSE"; "all" = E4own's
    model: "TranslationModel | None"
    n_cal: int                           # calibration samples of the class
    note: str = ""


def fit_task_models(rotvec_deg: np.ndarray, theta_dot_dps: np.ndarray, tau_mm: np.ndarray,
                    label_codes: np.ndarray | None,
                    task_labels: dict[str, tuple[str, Sequence[int] | None]], *,
                    min_n: int, lambdas: Sequence[float] | None = None, n_folds: int = 5,
                    hyst_dps: float = HYST_DPS) -> dict[str, TaskModel]:
    """X5c: one :class:`TranslationModel` per movement class, each fitted only on the
    calibration samples whose optical label code is in the class.

    ``rotvec_deg`` (deg), ``theta_dot_dps`` (deg/s), ``tau_mm`` (mm) and ``label_codes``
    (labels-stage codes, < 0 untracked) are the valid calibration samples (E4own's training
    set); ``task_labels`` maps a task to (class name, label codes), codes None = every
    calibration sample (then the fit is E4own's). Tasks with the same codes share one fit.
    A class with fewer than ``min_n`` samples, or no labels, gets no model (``note``)."""
    out: dict[str, TaskModel] = {}
    fits: dict[tuple[int, ...] | None, TaskModel] = {}
    for task, (name, codes) in task_labels.items():
        key = None if codes is None else tuple(sorted(int(c) for c in codes))
        if key not in fits:
            if key is None:
                sel = np.ones(len(tau_mm), bool)
            elif label_codes is None:
                sel = np.zeros(len(tau_mm), bool)
            else:
                sel = np.isin(np.asarray(label_codes), key)
            n = int(sel.sum())
            if key is not None and label_codes is None:
                fits[key] = TaskModel(name, None, 0, "no optical labels of the calibration "
                                                     "block")
            elif n < min_n:
                fits[key] = TaskModel(name, None, n, f"{n} calibration samples of the class "
                                                     f"(< {min_n})")
            else:
                try:
                    m = TranslationModel.fit(rotvec_deg[sel], theta_dot_dps[sel], tau_mm[sel],
                                             lambdas=lambdas, n_folds=n_folds,
                                             hyst_dps=hyst_dps)
                    fits[key] = TaskModel(name, m, n)
                except (ValueError, np.linalg.LinAlgError) as exc:
                    fits[key] = TaskModel(name, None, n, f"fit failed: {exc}")
        out[task] = fits[key]
    return out


def predict_by_task(models: dict[str, TaskModel], masks: dict[str, np.ndarray],
                    rotvec_deg: np.ndarray, theta_dot_dps: np.ndarray) -> np.ndarray:
    """τ (N, 3, mm): inside each task's mask the prediction of that task's class model (the
    oracle class: the true task label); NaN outside every mask and where a task has no
    model."""
    tau = np.full((len(rotvec_deg), 3), np.nan)
    for task, tm in models.items():
        m = np.asarray(masks.get(task, np.zeros(len(tau), bool)), bool)
        if tm.model is not None and m.any():
            tau[m] = tm.model.predict(rotvec_deg[m], theta_dot_dps[m])
    return tau


# -- accelerometer bridge (E5, E6) ----------------------------------------------------------------
@dataclass
class JawGeometry:
    """Static landmarks and IMU sites (maxillary frame, mm) from the holds and hand-eye."""

    incisor0: np.ndarray                 # lower incisor, closed
    condyle0: np.ndarray                 # condyle midpoint (pivot)
    site_cra: np.ndarray                 # head IMU site
    site_mta0: np.ndarray                # jaw IMU site at the closed pose

    @property
    def scale_mm(self) -> float:
        """Incisor–condyle distance (the generic model's scale)."""
        return float(np.linalg.norm(self.incisor0 - self.condyle0))


def jaw_geometry(T_max_hp: np.ndarray, T_max_mp_closed: np.ndarray, lever_cra_mm: np.ndarray,
                 lever_mta_mm: np.ndarray, incisor0: np.ndarray,
                 condyle0: np.ndarray) -> JawGeometry:
    """IMU sites from the plate poses (landmark frame, closed pose) and the hand-eye lever
    arms (IMU site in the plate frame)."""
    Th, Tm = np.asarray(T_max_hp, float), np.asarray(T_max_mp_closed, float)
    return JawGeometry(np.asarray(incisor0, float), np.asarray(condyle0, float),
                       Th[:3, :3] @ np.asarray(lever_cra_mm, float) + Th[:3, 3],
                       Tm[:3, :3] @ np.asarray(lever_mta_mm, float) + Tm[:3, 3])


@dataclass
class Bridge:
    tau_mm: np.ndarray                   # (N, 3) RTS smoother
    incisor_mm: np.ndarray
    tau_fwd_mm: np.ndarray               # (N, 3) forward (causal) filter
    incisor_fwd_mm: np.ndarray
    settings: PositionSettings


def _white_sd(x: np.ndarray) -> float:
    """White-noise SD from successive differences (1.4826 MAD / √2), RMS over axes: smooth
    motion barely enters first differences, sensor noise does."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x).all(axis=1)]
    if len(x) < 10:
        return 0.0
    d = np.diff(x, axis=0)
    sd = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0) / np.sqrt(2.0)
    return float(np.sqrt(np.mean(sd ** 2)))


def accel_bridge(t_s: np.ndarray, R_rel: np.ndarray, psi_rad: float | np.ndarray,
                 cra: ImuStream, mta: ImuStream, A: np.ndarray, geom: JawGeometry,
                 anchor_mask: np.ndarray, rest_mask: np.ndarray, *,
                 pseudo: tuple[np.ndarray, float, float] | None = None,
                 params: dict[str, Any] | None = None) -> Bridge:
    """E5: the jaw-IMU site relative to the head-IMU site from both accelerometers.

    Gravity is removed with each IMU's own orientation (``R f − g e_z`` in each IMU's world,
    the own-orientation rule of ``validation/accel.py``), the jaw IMU's world is turned into
    the head IMU's by ``Rz(ψ)``, and the difference is the acceleration of
    ``e(t) = S_mta − S_cra`` in the head IMU's world: no rotating-frame terms, the lever arms
    are the sites themselves. ``position_kf`` integrates it (state p, v, world bias) with
    closed-pose anchors as position measurements: at every sample of ``anchor_mask``
    (:meth:`Anchored.anchor_mask`, the most closed 0.3 s of each detected run, dense, so the
    velocity is pinned too) ``e = R_cra Aᵀ (site_mta0 − site_cra)``. Whole detected runs
    would be wrong anchors: their edges may still slide by up to a millimetre in protrusion
    and laterotrusion, whose small rotation the detector cannot tell from closed (on
    synthetic data they made the filter diverge). Noise from the rest phase: the white part
    (successive differences) of the relative acceleration (density σ²·dt) and of the anchor
    positions; bias random walk as position.py. Each acceleration sample holds over its own
    interval centred on its stamp (midpoint rule: the zero-order hold of ``position_kf``
    would otherwise delay the integration by half a sample, ≈ 0.3 mm at 150 mm/s).

    Back in the maxillary frame ``d = A R_craᵀ e``: jaw site ``q = site_cra + d``, incisor
    ``q + R_rel (p0 − site_mta0)``, condyle-midpoint translation
    ``q − c0 − R_rel (site_mta0 − c0)``.

    ``pseudo`` (E6, :func:`fuse_translation`): ``(tau_mm (N, 3), sigma_mm, rate_hz)``,
    model-predicted translations as further position measurements."""
    p = {**BRIDGE, **(params or {})}
    t = np.asarray(t_s, float)
    t_us = np.round(t * 1e6).astype(np.int64)
    n = t.size
    R_rel = np.asarray(R_rel, float)
    psi_g = np.broadcast_to(np.asarray(psi_rad, float), (n,))
    sel = (mta.t_a >= t_us[0]) & (mta.t_a <= t_us[-1])
    ta_us = np.asarray(mta.t_a[sel])
    f_m = np.asarray(mta.a[sel], float)
    f_c = cra.accel_at(ta_us)
    R_m = quat_to_matrix(mta.quat_at(ta_us))
    R_c = quat_to_matrix(cra.quat_at(ta_us))
    ta = ta_us * 1e-6
    fin_psi = np.isfinite(psi_g)
    psi_a = np.interp(ta, t[fin_psi], np.unwrap(psi_g[fin_psi]))
    gz = np.array([0.0, 0.0, G])
    a_m = np.einsum("nij,nj->ni", R_m, f_m) - gz
    a_c = np.einsum("nij,nj->ni", R_c, f_c) - gz
    a_rel = (np.einsum("nij,nj->ni", _rz(psi_a), a_m) - a_c) * 1e3          # mm/s²
    ok_a = np.isfinite(a_rel).all(axis=1)
    dt_a = float(np.median(np.diff(ta)))

    W = quat_to_matrix(cra.quat_at(t_us)) @ A.T                             # W_cra <- max
    d_c = geom.site_mta0 - geom.site_cra
    e_anchor = np.einsum("nij,j->ni", W, d_c)
    rest = np.asarray(rest_mask, bool)
    rest_a = np.interp(ta, t, rest.astype(float)) > 0.5
    sig_a = max(_white_sd(a_rel[rest_a & ok_a]), p["sigma_acc_floor_mm_s2"])
    r_anchor = max(_white_sd(e_anchor[rest]), p["r_pos_floor_mm"])
    closed = np.asarray(anchor_mask, bool) & np.isfinite(e_anchor).all(axis=1)
    t_meas, p_meas = t[closed], e_anchor[closed]
    r_pos, deriv = r_anchor, {"r_pos_from": "rest white SD of the anchor position",
                              "q_acc_from": "rest white SD of the relative acceleration",
                              "sigma_acc_mm_s2": sig_a, "dt_acc_s": dt_a,
                              "n_anchor": int(closed.sum()), "n_rest_acc": int(rest_a.sum())}
    r_meas = np.full(len(t_meas), r_anchor)
    if pseudo is not None:
        tau_p, sigma, rate = pseudo
        q_p = np.einsum("nij,j->ni", R_rel, geom.site_mta0 - geom.condyle0) + geom.condyle0 \
            + np.asarray(tau_p, float)
        e_p = np.einsum("nij,nj->ni", W, q_p - geom.site_cra)
        fs = 1.0 / float(np.median(np.diff(t)))
        step = max(1, int(round(fs / rate)))
        keep = np.zeros(n, bool)
        keep[::step] = True
        keep &= ~closed & np.isfinite(e_p).all(axis=1)
        # ``sigma`` is the model's 3-D RMS error (sqrt of the mean squared error norm), the
        # filter's measurement SD is per axis: sigma / sqrt(3). The anchors keep their own
        # SD (as in E5), so each measurement enters with its own variance.
        sig_ax = float(sigma) / np.sqrt(3.0)
        t_meas = np.concatenate([t_meas, t[keep]])
        p_meas = np.concatenate([p_meas, e_p[keep]])
        r_meas = np.concatenate([r_meas, np.full(int(keep.sum()), sig_ax)])
        deriv.update({"pseudo_sigma_3d_mm": float(sigma), "pseudo_sigma_mm": sig_ax,
                      "pseudo_hz": float(rate), "n_pseudo": int(keep.sum())})
    st = PositionSettings(r_pos, sig_a ** 2 * dt_a, float(p["bias_rw"]) ** 2, deriv)
    e_s, e_f = position_kf(t_meas, p_meas, np.ones(len(t_meas), bool), ta[ok_a] - 0.5 * dt_a,
                           a_rel[ok_a], st, t, r_meas_mm=r_meas)

    def to_max(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = geom.site_cra + np.einsum("nji,nj->ni", W, e)
        inc = q + np.einsum("nij,j->ni", R_rel, geom.incisor0 - geom.site_mta0)
        tau = q - geom.condyle0 - np.einsum("nij,j->ni", R_rel, geom.site_mta0 - geom.condyle0)
        return tau, inc

    tau_s, inc_s = to_max(e_s)
    tau_f, inc_f = to_max(e_f)
    return Bridge(tau_s, inc_s, tau_f, inc_f, st)


def fuse_translation(t_s: np.ndarray, R_rel: np.ndarray, psi_rad: float | np.ndarray,
                     cra: ImuStream, mta: ImuStream, A: np.ndarray, geom: JawGeometry,
                     anchor_mask: np.ndarray, rest_mask: np.ndarray, tau_pred_mm: np.ndarray,
                     sigma_mm: float, *, pseudo_hz: float = 5.0,
                     params: dict[str, Any] | None = None) -> Bridge:
    """E6: :func:`accel_bridge` with the E4 translation as pseudo-measurements, thinned to
    ``pseudo_hz`` (the model error is correlated over a movement, so denser samples would
    overweight it). ``sigma_mm``: the model's 3-D RMS error (its inner-CV RMS); each axis
    enters with SD ``sigma_mm / sqrt(3)``, the anchors with their own SD as in E5."""
    return accel_bridge(t_s, R_rel, psi_rad, cra, mta, A, geom, anchor_mask, rest_mask,
                        pseudo=(tau_pred_mm, sigma_mm, pseudo_hz), params=params)


# -- scoring ---------------------------------------------------------------------------------
def _lowpass(R: np.ndarray, p: np.ndarray | None, fs: float, fc: float,
             order: int) -> tuple[np.ndarray, np.ndarray]:
    T = np.zeros((len(R), 4, 4))
    T[:, :3, :3] = R
    T[:, :3, 3] = 0.0 if p is None else p
    T[:, 3, 3] = 1.0
    T = se3.lowpass(T, fs, fc, order)
    return T[:, :3, :3], T[:, :3, 3]


def _rms(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


def score(t_s: np.ndarray, R_est: np.ndarray, R_true: np.ndarray,
          p_est: np.ndarray | None = None, p_true: np.ndarray | None = None,
          mask: np.ndarray | None = None, *, fc_hz: float | None = FC_HZ,
          order: int = LP_ORDER, lowpass_truth: bool = True) -> dict[str, float]:
    """Errors of an estimate against the reference over ``mask``, after the ground truth's
    6 Hz zero-phase low-pass applied to the estimate (and to the truth, as the optics are;
    ``fc_hz`` None: unfiltered). Rotation: geodesic and per-axis (rotation vector of
    ``R_est R_trueᵀ``, maxillary frame), opening angle (maxillary-y component, as
    imu/analysis.py); incisor (or any point): 3-D and per-axis. RMS, mean (bias) and max."""
    t = np.asarray(t_s, float)
    fs = 1.0 / float(np.median(np.diff(t)))
    Re, pe = np.asarray(R_est, float), p_est
    Rt, pt = np.asarray(R_true, float), p_true
    if fc_hz is not None:
        Re, pe_f = _lowpass(Re, pe, fs, fc_hz, order)
        pe = pe_f if p_est is not None else None
        if lowpass_truth:
            Rt, pt_f = _lowpass(Rt, pt, fs, fc_hz, order)
            pt = pt_f if p_true is not None else None
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)
    e = np.degrees(_rotvec(Re @ _T(Rt)))[m]
    d_open = (opening_deg(Re) - opening_deg(Rt))[m]
    geo = np.linalg.norm(e, axis=1)
    out = {"n": int(np.isfinite(geo).sum()), "rmse_open_deg": _rms(d_open),
           "bias_open_deg": float(np.nanmean(d_open)),
           "max_open_deg": float(np.nanmax(np.abs(d_open))),
           "rmse_geo_deg": _rms(geo), "max_geo_deg": float(np.nanmax(geo))}
    for j, ax in enumerate("xyz"):
        out[f"rmse_r{ax}_deg"] = _rms(e[:, j])
    if pe is not None and pt is not None:
        d = (np.asarray(pe, float) - np.asarray(pt, float))[m]
        n3 = np.linalg.norm(d, axis=1)
        out.update({"rmse_inc_mm": _rms(n3), "max_inc_mm": float(np.nanmax(n3))})
        for j, ax in enumerate("xyz"):
            out[f"rmse_inc_{ax}_mm"] = _rms(d[:, j])
            out[f"bias_inc_{ax}_mm"] = float(np.nanmean(d[:, j]))
    return out


__all__ = [
    "Anchored", "Bridge", "CLOSED", "ClosedDetection", "FC_HZ", "FEATURES", "HYST_DPS",
    "JawGeometry", "MountingPrior", "PSI_MIN_TILT_RANGE_DEG", "PsiEstimate", "RotationLadder",
    "SIGNATURE", "TaskModel", "TranslationModel", "accel_bridge", "anchored_rotation",
    "closed_reference", "closed_threshold", "condyle_translation", "detect_closed",
    "drift_signature", "fit_task_models", "fuse_translation", "imu_product",
    "incisor_position", "jaw_geometry", "mounting_prior", "opening_deg", "oracle_incisors",
    "predict_by_task", "psi_consistency", "psi_static", "relative_rotation",
    "relative_speed_dps", "rotation_ladder", "score", "theta_rate_dps", "translation_features",
]
