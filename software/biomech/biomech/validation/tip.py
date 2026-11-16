# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CT tip verification for the Qualisys landmarks (validation plan V7).

Both systems' CT tips were pivot-calibrated (user), but the sync fit's ``Y_CT`` puts the
JTS-MM tip a few mm from the Qualisys CT origin. A tip offset is a constant vector ``d`` in
the CT frame, so it can be re-estimated from the recordings themselves:

* **shared** (option 5): the JTS-MM tip expressed in the Qualisys CT frame through ``Y_CT``
  of the sync fit, ``d = −R_Yᵀ t_Y`` (``T_Q,CT = X · T_S,CT · Y_CT``; JTS-MM's CT origin is
  its tip);
* **hold pivot** (option 6), per system: during each landmark hold the tip rests on a tooth
  or the skin while the handle wobbles, so ``R_i d + t_i = c_h`` (CT pose in the body frame,
  one contact point per hold, one ``d`` for all holds); linear least squares;
* **cross-system regression** (option 7): at the still hold frames the JTS-MM tip mapped by
  ``X`` must equal the Qualisys tip, ``R_Q,i d + t_Q,i = X p_S,i``; the 8 holds give very
  different tool orientations, so this is much better conditioned than option 6. It
  trusts ``X`` (fitted on the sync windows) at the holds;
* **tooth holds** (primary data-based check): the same equation on the hold
  medians with a free world offset ``c``, ``R̄_Q,i d − c = X p̄_S,i − t̄_Q,i``: a residual
  error of ``X`` at the holds no longer leaks into ``d``, and the per-hold residuals show
  which holds disagree (the tip is identifiable because the CT is held in very different
  orientations at the teeth and condyles);
* **frame-free distances** (option 8): the pairwise distances of the landmarks within each
  body must match between the systems; nonlinear least squares in ``d`` (no registration);
* **CT geometry** (option 9): the Qualisys CT body rebuilt from the labelled markers --
  inter-marker distances (same tool on both days?), the QTM body origin relative to the
  marker centroid (same body definition?) and the template RMS. Label swaps are not
  detectable here: the CT triangle is nearly isosceles, so QTM fits swapped labels
  self-consistently (rotated ~180°, template residual ~1 mm). They are corrected when the
  take is read (``io.reference.fix_label_swaps``, erratum EV1) and counted there.

**Decision (V7).** The data-based estimates (6–8) must agree within 2 SE. Their weighted
consensus is the verified Qualisys tip; it is primary ("own_verified") when it is
well-conditioned and lies within 0.5 mm and 2 SE of the shared tip. Otherwise the shared
tip is primary ("shared"), stated as a limitation. Either way the other is the sensitivity
variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import least_squares

from biomech.kinematics import se3


@dataclass
class TipEstimate:
    method: str
    d: np.ndarray              # (3,) mm, tip in the Qualisys CT frame (or correction)
    cov: np.ndarray            # (3, 3)
    cond: float                # condition number of the design (NaN when not linear)
    rms_before_mm: float       # misfit with d = 0 (pivot: hold spread; cross: tip distance)
    rms_after_mm: float
    n: int

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.cov), 0, None))


def shared_tip(Y_ct: np.ndarray, cov_t: np.ndarray | None = None) -> TipEstimate:
    R, t = Y_ct[:3, :3], Y_ct[:3, 3]
    d = -R.T @ t
    cov = R.T @ cov_t @ R if cov_t is not None else np.full((3, 3), np.nan)
    return TipEstimate("shared", d, cov, np.nan, np.nan, np.nan, 0)


def _lstsq(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    x, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ x
    dof = max(len(b) - len(x), 1)
    s2 = float(r @ r / dof)
    AtA = A.T @ A
    try:
        cov = s2 * np.linalg.inv(AtA)
    except np.linalg.LinAlgError:
        cov = np.full((len(x), len(x)), np.nan)
    return x, cov, float(np.linalg.cond(A))


def hold_pivot(rel: Sequence[np.ndarray], method: str = "hold_pivot") -> TipEstimate:
    """``rel[h]``: (n_h, 4, 4) CT poses in the hold's body frame at its still frames."""
    rel = [r[se3.valid(r)] for r in rel]
    rel = [r for r in rel if len(r) >= 3]
    H = len(rel)
    if H == 0:
        raise ValueError("hold_pivot: no hold frames")
    rows = sum(len(r) for r in rel)
    A = np.zeros((3 * rows, 3 + 3 * H))
    b = np.zeros(3 * rows)
    k = 0
    spread0 = []
    for h, r in enumerate(rel):
        n = len(r)
        sl = slice(3 * k, 3 * (k + n))
        A[sl, :3] = r[:, :3, :3].reshape(-1, 3)             # R_1 rows, R_2 rows, ...
        A[sl, 3 + 3 * h: 6 + 3 * h] = np.tile(-np.eye(3), (n, 1))
        b[sl] = -r[:, :3, 3].reshape(-1)
        spread0.append(r[:, :3, 3] - r[:, :3, 3].mean(axis=0))
        k += n
    x, cov, cond = _lstsq(A, b)
    res = (A @ x - b).reshape(-1, 3)
    before = np.concatenate(spread0)
    return TipEstimate(method, x[:3], cov[:3, :3], cond,
                       float(np.sqrt(np.mean(np.sum(before ** 2, axis=1)))),
                       float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), rows)


def cross_system(X: np.ndarray, p_s_world: np.ndarray, T_q_ct: np.ndarray) -> TipEstimate:
    """``R_Q,i d + t_Q,i = X p_S,i`` at the still hold frames (paired, same master time)."""
    ok = np.isfinite(p_s_world).all(axis=1) & se3.valid(T_q_ct)
    p = se3.apply(X, p_s_world[ok])
    R, t = T_q_ct[ok, :3, :3], T_q_ct[ok, :3, 3]
    if len(p) < 5:
        raise ValueError("cross_system: too few paired frames")
    A = np.concatenate(R)
    b = (p - t).reshape(-1)
    d, cov, cond = _lstsq(A, b)
    before = np.linalg.norm(p - t, axis=1)
    after = np.linalg.norm((A @ d - b).reshape(-1, 3), axis=1)
    return TipEstimate("cross_system", d, cov, cond, float(np.sqrt(np.mean(before ** 2))),
                       float(np.sqrt(np.mean(after ** 2))), int(len(p)))


def tooth_holds(holds: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]
                ) -> tuple[TipEstimate, dict[str, np.ndarray]]:
    """``holds[name] = (R_Q (n, 3, 3), t_Q (n, 3), p (n, 3))`` at the common still frames,
    ``p`` the JTS-MM tip in the Qualisys world (``X p_S``). Least squares of
    ``R̄_i d − c = p̄_i − t̄_i`` over the hold medians (``R̄`` the chordal mean). Returns the
    estimate (``rms_before``: hold misfit with ``d = 0`` and the best ``c``) and the per-hold
    residual vectors (mm)."""
    names, A, b = [], [], []
    for n, (R, t, p) in holds.items():
        ok = np.isfinite(R).all(axis=(1, 2)) & np.isfinite(t).all(axis=1) & np.isfinite(p).all(axis=1)
        if ok.sum() < 3:
            continue
        U, _, Vt = np.linalg.svd(R[ok].mean(axis=0))
        Rm = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
        A.append(np.hstack([Rm, -np.eye(3)]))
        b.append(np.median(p[ok] - t[ok], axis=0))
        names.append(n)
    if len(names) < 3:
        raise ValueError("tooth_holds: fewer than 3 holds")
    A_, b_ = np.vstack(A), np.concatenate(b)
    x, cov, cond = _lstsq(A_, b_)
    res = (A_ @ x - b_).reshape(-1, 3)
    bb = b_.reshape(-1, 3)
    before = bb - bb.mean(axis=0)                      # d = 0, c = mean offset
    est = TipEstimate("tooth_holds", x[:3], cov[:3, :3], cond,
                      float(np.sqrt(np.mean(np.sum(before ** 2, axis=1)))),
                      float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), len(names))
    return est, dict(zip(names, res))


def distance_tip(q_terms: dict[str, tuple[np.ndarray, np.ndarray, str]],
                 s_points: dict[str, np.ndarray]) -> TipEstimate:
    """``q_terms[name] = (R̄, t̄, body)``: the Qualisys landmark is ``R̄ d + t̄`` in its body
    frame (mean CT pose in the body over the still frames); ``s_points``: the JTS-MM
    landmarks. Fits ``d`` to the within-body pairwise distances (frame-free)."""
    names = sorted(set(q_terms) & set(s_points))
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if q_terms[a][2] == q_terms[b][2]]
    if len(pairs) < 3:
        raise ValueError("distance_tip: too few landmark pairs")
    target = np.array([np.linalg.norm(s_points[a] - s_points[b]) for a, b in pairs])

    def resid(d):
        P = {n: q_terms[n][0] @ d + q_terms[n][1] for n in names}
        return np.array([np.linalg.norm(P[a] - P[b]) for a, b in pairs]) - target

    sol = least_squares(resid, np.zeros(3), method="lm")
    J = sol.jac
    dof = max(len(pairs) - 3, 1)
    s2 = float(np.sum(sol.fun ** 2) / dof)
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.nan)
    r0 = resid(np.zeros(3))
    return TipEstimate("distances", sol.x, cov, float(np.linalg.cond(J)),
                       float(np.sqrt(np.mean(r0 ** 2))), float(np.sqrt(np.mean(sol.fun ** 2))),
                       len(pairs))


def ct_geometry(markers: dict[str, np.ndarray], T_ct: np.ndarray) -> dict[str, float]:
    """The Qualisys CT body from its labelled markers: inter-marker distances, the marker
    template in the QTM body frame, the body origin relative to the marker centroid and
    the template RMS (label swaps: see the module docstring)."""
    names = sorted(markers)
    P = np.stack([markers[n] for n in names], axis=1)          # (N, M, 3)
    ok = se3.valid(T_ct) & np.isfinite(P).all(axis=(1, 2))
    out: dict[str, float] = {"n_frames": int(ok.sum())}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            dd = np.linalg.norm(P[ok, i] - P[ok, j], axis=1)
            out[f"dist_{i + 1}{j + 1}_mm"] = float(np.median(dd)) if len(dd) else np.nan
            out[f"dist_{i + 1}{j + 1}_sd_mm"] = float(np.std(dd)) if len(dd) else np.nan
    if ok.sum() < 10:
        return out
    Ti = se3.inv(T_ct[ok])
    loc = np.einsum("nij,nmj->nmi", Ti[:, :3, :3], P[ok]) + Ti[:, None, :3, 3]
    tmpl = np.median(loc, axis=0)                               # (M, 3) in the body frame
    cen = tmpl.mean(axis=0)
    out.update({"origin_to_centroid_mm": float(np.linalg.norm(cen)),
                "centroid_x_mm": cen[0], "centroid_y_mm": cen[1], "centroid_z_mm": cen[2]})
    pred = np.einsum("nij,mj->nmi", T_ct[ok, :3, :3], tmpl) + T_ct[ok, None, :3, 3]
    err = np.linalg.norm(pred - P[ok], axis=2).max(axis=1)
    out["template_rms_mm"] = float(np.sqrt(np.mean(err ** 2)))
    return out


def hold_bootstrap(fit, groups: Sequence[Any], *, n_boot: int = 400, seed: int = 0
                   ) -> np.ndarray:
    """Covariance of ``fit(groups) -> d`` from a bootstrap over holds (frames within a hold
    are not independent; the holds are the sampling units)."""
    rng = np.random.default_rng(seed)
    H = len(groups)
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, H, H)
        if len(set(pick.tolist())) < 3:
            continue
        try:
            draws.append(fit([groups[i] for i in pick]))
        except (ValueError, np.linalg.LinAlgError):
            continue
    if len(draws) < 20:
        return np.full((3, 3), np.nan)
    return np.cov(np.array(draws).T)


def verify_own(est: Sequence[TipEstimate], *, max_se_mm: float = 2.0) -> dict[str, object]:
    """V7 (operationalised before the freeze): the system's own tip (``d = 0``) is verified
    when at least one data-based estimate is informative (hold-bootstrap SE < ``max_se_mm``
    on every axis) and every informative estimate contains ``d = 0`` in its 95 % region
    (Mahalanobis, χ²₃)."""
    from scipy.stats import chi2

    info, ok = [], []
    for e in est:
        se = e.se
        if not (np.isfinite(e.cov).all() and np.all(se < max_se_mm)):
            continue
        info.append(e.method)
        try:
            m2 = float(e.d @ np.linalg.solve(e.cov, e.d))
        except np.linalg.LinAlgError:
            continue
        ok.append(bool(chi2.sf(m2, 3) >= 0.05))
    return {"informative": info, "own_consistent": ok,
            "verified": bool(info) and len(ok) == len(info) and all(ok)}


def consensus(est: Sequence[TipEstimate]) -> tuple[np.ndarray, np.ndarray, float]:
    """Inverse-variance weighted mean of the estimates and the chi² p-value of their
    agreement (per axis, pooled)."""
    from scipy.stats import chi2

    ok = [e for e in est if np.isfinite(e.cov).all() and np.all(np.diag(e.cov) > 0)]
    if not ok:
        return np.full(3, np.nan), np.full((3, 3), np.nan), np.nan
    W = [np.linalg.inv(e.cov) for e in ok]
    C = np.linalg.inv(sum(W))
    d = C @ sum(w @ e.d for w, e in zip(W, ok))
    q = float(sum((e.d - d) @ w @ (e.d - d) for w, e in zip(W, ok)))
    dof = 3 * (len(ok) - 1)
    p = float(chi2.sf(q, dof)) if dof > 0 else np.nan
    return d, C, p


def decide(shared: TipEstimate, data_based: Sequence[TipEstimate], *, max_cond: float = 1e3,
           tol_mm: float = 0.5, p_agree: float = 0.05) -> dict[str, object]:
    """V7: own verified tip if the data-based estimates are well-conditioned, agree with
    each other (p ≥ ``p_agree``) and their consensus is within ``tol_mm`` and 2 SE of the
    shared tip; otherwise the shared tip."""
    good = [e for e in data_based if np.isfinite(e.cond) and e.cond < max_cond]
    d, C, p = consensus(good) if good else (np.full(3, np.nan), np.full((3, 3), np.nan), np.nan)
    diff = d - shared.d
    se = np.sqrt(np.clip(np.diag(C), 0, None))
    if np.isfinite(shared.cov).all():
        se = np.sqrt(se ** 2 + np.clip(np.diag(shared.cov), 0, None))
    verified = bool(good and np.isfinite(d).all() and (np.isnan(p) or p >= p_agree))
    accept = bool(verified and np.linalg.norm(diff) < tol_mm
                  and np.all(np.abs(diff) <= 2 * np.maximum(se, 1e-9)))
    return {"primary": "own_verified" if accept else "shared",
            "d_primary": d if accept else shared.d, "d_consensus": d, "cov_consensus": C,
            "p_agree": p, "n_estimates": len(good), "verified": verified,
            "diff_to_shared_mm": float(np.linalg.norm(diff)) if np.isfinite(diff).all()
            else np.nan}
