# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""How far is jaw opening from a fixed hinge? (C1, an02 §6)

Mandibular points (incisor, first molars, skin-palpated condyles) are observed at a
cycle's closed reference pose (``A``, F x K x 3) and at later frames of the
same opening (``B``). Three models predict B from A:

* ``hinge_fit``  rotation about one fixed axis (direction n, through point s),
  both fitted to all cycles; one angle per frame;
* ``hinge_tmj``  rotation about the axis through the two digitized condyle
  points (only the angles are fitted) -- what a hinge exoskeleton aligned to
  palpated condyles would assume;
* ``rot_trans``  the fitted hinge plus a translation along one fitted direction
  u (one angle and one translation per frame) -- a 2-DoF "RP" linkage;
* ``hinge_teeth`` (D25, sensitivity) the fixed axis fitted to the three tooth points only
  (incisor, molars): the fit does not weight the residuals at the skin-palpated condyle
  points. All points are rigidly attached to the mandible, so the poses are the same; only
  where the misalignment is minimized changes. Its residuals are still reported at every
  point.

The residual (mm) of each model at the incisal point and at the condyles is
the misalignment a device built on that model would impose. For a given axis
the optimal angle per frame is closed-form (1-D Wahba problem), so only the
axis (and u) are optimized numerically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _dir(az: float, el: float) -> np.ndarray:
    """Unit vector from azimuth (about z, from +y) and elevation; n=(0,1,0) at (0,0)."""
    return np.array([-np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)])


def rotate(v: np.ndarray, n: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues rotation of v (F, K, 3) about unit n by theta (F,)."""
    c = np.cos(theta)[:, None, None]
    s = np.sin(theta)[:, None, None]
    cross = np.cross(n, v)
    dot = (v @ n)[..., None]
    return v * c + cross * s + n * dot * (1 - c)


def best_angles(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Per-frame angle about n (through the origin) best mapping a -> b (F, K, 3)."""
    ap = a - (a @ n)[..., None] * n
    bp = b - (b @ n)[..., None] * n
    sin = (np.cross(ap, bp) @ n).sum(axis=1)
    cos = np.einsum("fkd,fkd->f", ap, bp)
    return np.arctan2(sin, cos)


@dataclass
class HingeResult:
    model: str
    n: np.ndarray
    s: np.ndarray
    u: np.ndarray | None
    theta_deg: np.ndarray                     # (F,)
    tau_mm: np.ndarray | None                 # (F,)
    residual: np.ndarray                      # (F, K) point error norms, mm
    extra: dict = field(default_factory=dict)

    def rms(self, k: int | slice) -> float:
        r = self.residual[:, k]
        return float(np.sqrt(np.nanmean(r ** 2)))

    def max(self, k: int | slice) -> float:
        return float(np.nanmax(self.residual[:, k]))


def _hinge_predict(A: np.ndarray, B: np.ndarray, n: np.ndarray, s: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    th = best_angles(A - s, B - s, n)
    return rotate(A - s, n, th) + s, th


def _rt_predict(A: np.ndarray, B: np.ndarray, n: np.ndarray, s: np.ndarray, u: np.ndarray,
                max_iter: int = 12, tol_rad: float = 1e-13
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame optimal angle and translation for a given axis (n, s) and direction u.

    For a fixed angle the optimal translation is closed-form,
    tau = mean_k u.(b_k - R a_k), so the frame objective depends on the angle only:
    f(t) = sum_k |R(t) a_k - b_k|^2 - K (u.mean_k(b_k - R(t) a_k))^2, with
    R(t) a = a_par + cos t a_perp + sin t (n x a) (Rodrigues). f is a trigonometric
    function of t; a few vectorized Newton steps from the pure-hinge optimum reach its
    minimum to machine precision. This is the fixed point the earlier angle/translation
    alternation converged to (an02 §6, E10), without its up to 50 iterations.
    """
    a, b = A - s, B - s
    k = a.shape[1]
    a_par = (a @ n)[..., None] * n
    a_perp = a - a_par
    a_x = np.cross(n, a)
    C = np.einsum("fkd,fkd->f", a_perp, b)
    S = np.einsum("fkd,fkd->f", a_x, b)
    al = (b.mean(axis=1) - a_par.mean(axis=1)) @ u
    be = a_perp.mean(axis=1) @ u
    ga = a_x.mean(axis=1) @ u
    th = np.arctan2(S, C)
    for _ in range(max_iter):
        c, si = np.cos(th), np.sin(th)
        r = al - be * c - ga * si
        dr = be * si - ga * c
        g = 2.0 * (C * si - S * c) - 2.0 * k * r * dr
        h = 2.0 * (C * c + S * si) - 2.0 * k * (dr * dr + r * (be * c + ga * si))
        ok = h > 0
        step = np.where(ok, g / np.where(ok, h, 1.0), 0.0)
        th = th - step
        if not np.any(np.abs(step) > tol_rad):
            break
    rot = rotate(a, n, th) + s
    tau = ((B - rot) @ u).mean(axis=1)
    return rot + tau[:, None, None] * u, th, tau


TEETH = (0, 1, 2)          # incisor, left and right molar in the HINGE_POINTS order (D25)


def fit_hinge(A: np.ndarray, B: np.ndarray, s0: np.ndarray, n0: np.ndarray | None = None,
              points: tuple[int, ...] | None = None, model: str = "hinge_fit"
              ) -> HingeResult:
    """Fixed axis fitted to all frames (pure rotation).

    ``points``: indices of the points whose residuals the fit minimizes (None: all; D25
    teeth-only fit: :data:`TEETH`); the per-frame angles then also come from these points.
    The residuals of the result are reported at every point."""
    n0 = _unit(n0) if n0 is not None else np.array([0.0, 1.0, 0.0])
    az0, el0 = float(np.arctan2(-n0[0], n0[1])), float(np.arcsin(np.clip(n0[2], -1, 1)))
    Af, Bf = (A, B) if points is None else (A[:, list(points)], B[:, list(points)])

    def resid(x: np.ndarray) -> np.ndarray:
        n = _dir(x[0], x[1])
        s = x[2:5]
        pred, _ = _hinge_predict(Af, Bf, n, s)
        return np.concatenate([(pred - Bf).ravel(), [1e-3 * (s @ n)]])

    x0 = np.concatenate([[az0, el0], s0])
    sol = least_squares(resid, x0, x_scale=np.array([0.1, 0.1, 10, 10, 10]), method="trf")
    n = _dir(sol.x[0], sol.x[1])
    s = sol.x[2:5]
    th = best_angles(Af - s, Bf - s, n)
    pred = rotate(A - s, n, th) + s
    return HingeResult(model, n, s, None, np.degrees(th), None,
                       np.linalg.norm(pred - B, axis=2), {"cost": float(sol.cost)})


def hinge_through(A: np.ndarray, B: np.ndarray, p_left: np.ndarray, p_right: np.ndarray,
                  model: str = "hinge_tmj") -> HingeResult:
    """Axis fixed through two given points; only the angles are fitted."""
    n = _unit(p_left - p_right)
    s = 0.5 * (p_left + p_right)
    pred, th = _hinge_predict(A, B, n, s)
    return HingeResult(model, n, s, None, np.degrees(th), None, np.linalg.norm(pred - B, axis=2))


def fit_rot_trans(A: np.ndarray, B: np.ndarray, hinge: HingeResult,
                  start: HingeResult | None = None) -> HingeResult:
    """Fitted hinge plus a translation along a fitted direction u (start: anterior-inferior,
    or ``start``'s axis and direction, e.g. the full-data fit for bootstrap resamples)."""
    def resid(x: np.ndarray) -> np.ndarray:
        n = _dir(x[0], x[1])
        s = x[2:5]
        u = _unit(np.array([np.cos(x[5]) * np.cos(x[6]), np.sin(x[5]) * np.cos(x[6]),
                            np.sin(x[6])]))
        pred, _, _ = _rt_predict(A, B, n, s, u)
        return np.concatenate([(pred - B).ravel(), [1e-3 * (s @ n)]])

    h0 = start if start is not None and start.u is not None else hinge
    az = float(np.arctan2(-h0.n[0], h0.n[1]))
    el = float(np.arcsin(np.clip(h0.n[2], -1, 1)))
    if start is not None and start.u is not None:
        uaz = float(np.arctan2(start.u[1], start.u[0]))
        uel = float(np.arcsin(np.clip(start.u[2], -1, 1)))
    else:
        uaz, uel = 0.0, float(np.radians(-30.0))
    x0 = np.concatenate([[az, el], h0.s, [uaz, uel]])
    sol = least_squares(resid, x0, x_scale=np.array([0.1, 0.1, 10, 10, 10, 0.3, 0.3]))
    n = _dir(sol.x[0], sol.x[1])
    s = sol.x[2:5]
    u = _unit(np.array([np.cos(sol.x[5]) * np.cos(sol.x[6]), np.sin(sol.x[5]) * np.cos(sol.x[6]),
                        np.sin(sol.x[6])]))
    pred, th, tau = _rt_predict(A, B, n, s, u)
    return HingeResult("rot_trans", n, s, u, np.degrees(th), tau,
                       np.linalg.norm(pred - B, axis=2), {"cost": float(sol.cost)})
