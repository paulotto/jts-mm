# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Reference (Qualisys) quality and uncertainty (validation plan V8/V9).

Everything here uses Qualisys-internal data only (labelled markers, marker residuals and
type codes, 6DOF poses, the calibration report), never the disagreement with JTS-MM:

* **rigid-body consistency:** the inter-marker distances of a body must be constant; their
  deviation from the median, marker residuals and missing / gap-filled frames;
* **gate (V8):** a frame is flagged when, for a body, a marker is missing or gap-filled, an
  inter-marker distance deviates by more than 0.5 mm, or a marker residual exceeds 1 mm;
* **viewing direction:** each camera's line of sight in the body frame (azimuth,
  elevation), with consistency and residual binned by it -- the signature of the
  angle-dependent centroid bias of flat markers;
* **steps:** jumps of the inter-marker distances between consecutive frames (a proxy for
  cameras starting or stopping to see a marker; only 3 cameras);
* **σ_ref:** Monte-Carlo propagation of the marker noise (σ_m = SD(distance deviation)/√2
  per segment) through Kabsch fits of both bodies to the anatomical-frame outcome
  (incisor, condyle points, rotation);
* **availability:** the fraction of frames with a valid pose, and dropout runs, for both
  systems.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from biomech import timebase
from biomech.kinematics import se3

GATE_DIST_MM = 0.5
GATE_RESID_MM = 1.0
MISSING, MEASURED, GAPFILLED = 0, 1, 2


def consistency(markers: dict[str, np.ndarray], residual: dict[str, np.ndarray] | None = None,
                mtype: dict[str, np.ndarray] | None = None, *,
                dist_mm: float = GATE_DIST_MM, resid_mm: float = GATE_RESID_MM
                ) -> pd.DataFrame:
    """Per-frame quality of one rigid body from its labelled markers (N rows)."""
    names = sorted(markers)
    P = np.stack([markers[n] for n in names], axis=1)                 # (N, M, 3)
    n = len(P)
    pairs = list(combinations(range(len(names)), 2))
    D = np.stack([np.linalg.norm(P[:, i] - P[:, j], axis=1) for i, j in pairs], axis=1)
    med = np.nanmedian(D, axis=0)
    dev = D - med
    out = pd.DataFrame({f"dev_{i + 1}{j + 1}_mm": dev[:, k] for k, (i, j) in enumerate(pairs)})
    with np.errstate(invalid="ignore"):
        out["dev_max_mm"] = np.nanmax(np.abs(dev), axis=1) if dev.size else np.nan
    miss = ~np.isfinite(P).all(axis=2).all(axis=1)
    gap = np.zeros(n, bool)
    if mtype:
        T = np.stack([mtype[k] for k in names if k in mtype], axis=1)
        miss |= (T == MISSING).any(axis=1)
        gap = (T == GAPFILLED).any(axis=1)
    out["missing"] = miss
    out["gapfilled"] = gap
    if residual:
        Rr = np.stack([residual[k] for k in names if k in residual], axis=1)
        with np.errstate(invalid="ignore"):
            out["resid_max_mm"] = np.nanmax(Rr, axis=1)
    else:
        out["resid_max_mm"] = np.nan
    out["gate"] = (miss | gap | (out["dev_max_mm"].to_numpy() > dist_mm)
                   | (out["resid_max_mm"].fillna(0).to_numpy() > resid_mm))
    return out


def template(markers: dict[str, np.ndarray], T: np.ndarray) -> np.ndarray:
    """Marker coordinates in the body frame (M, 3): median of ``T⁻¹ p`` over valid frames."""
    names = sorted(markers)
    P = np.stack([markers[n] for n in names], axis=1)
    ok = se3.valid(T) & np.isfinite(P).all(axis=(1, 2))
    if ok.sum() < 3:
        raise ValueError("template: too few frames with the body and all markers")
    Ti = se3.inv(T[ok])
    loc = np.einsum("nij,nmj->nmi", Ti[:, :3, :3], P[ok]) + Ti[:, None, :3, 3]
    return np.median(loc, axis=0)


def marker_geometry(tmpl: np.ndarray) -> dict[str, float]:
    """Conditioning of a marker cluster (body-frame template, M x 3): side lengths, the
    smallest height of the triangle (how far the markers are from collinear) and the
    singular values of the centred template. A thin cluster determines rotation about its
    long axis poorly (σ_rot ∝ σ_marker / small singular value)."""
    c = tmpl - tmpl.mean(axis=0)
    sv = np.linalg.svd(c, compute_uv=False)
    out = {f"sv{i + 1}_mm": float(v) for i, v in enumerate(sv)}
    if len(tmpl) == 3:
        a, b, d = (float(np.linalg.norm(tmpl[i] - tmpl[j])) for i, j in ((0, 1), (0, 2), (1, 2)))
        area = 0.5 * float(np.linalg.norm(np.cross(tmpl[1] - tmpl[0], tmpl[2] - tmpl[0])))
        out.update(side_12_mm=a, side_13_mm=b, side_23_mm=d, area_mm2=area,
                   min_height_mm=2 * area / max(a, b, d))
    return out


def sigma_marker(dev: pd.DataFrame) -> float:
    """σ per axis of one marker from the inter-marker distance deviations (isotropic,
    independent markers: Var(d_ij) ≈ 2σ² along the pair)."""
    cols = [c for c in dev.columns if c.startswith("dev_") and c != "dev_max_mm"]
    v = dev[cols].to_numpy().ravel()
    v = v[np.isfinite(v)]
    return float(np.std(v) / np.sqrt(2.0)) if len(v) > 2 else float("nan")


def steps(dev: pd.DataFrame, *, k: float = 5.0, floor_mm: float = 0.3) -> np.ndarray:
    """Frames where an inter-marker distance jumps by more than max(k · robust SD of the
    frame-to-frame change, ``floor_mm``)."""
    cols = [c for c in dev.columns if c.startswith("dev_") and c != "dev_max_mm"]
    X = dev[cols].to_numpy()
    dX = np.abs(np.diff(X, axis=0))
    mad = np.nanmedian(np.abs(dX - np.nanmedian(dX, axis=0)), axis=0) * 1.4826
    thr = np.maximum(k * mad, floor_mm)
    with np.errstate(invalid="ignore"):
        jump = (dX > thr).any(axis=1)
    return np.r_[False, jump]


def viewing_directions(T: np.ndarray, cams: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Unit line of sight from the body origin to each camera, in the body frame, as
    (azimuth, elevation) in degrees, (N, 2) per camera; NaN where the body is missing."""
    ok = se3.valid(T)
    out = {}
    for name, c in cams.items():
        v = np.full((len(T), 3), np.nan)
        d = c[None, :] - T[ok, :3, 3]
        v[ok] = np.einsum("nji,nj->ni", T[ok, :3, :3], d)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        az = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        el = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        out[name] = np.column_stack([az, el])
    return out


def viewing_bins(q: pd.DataFrame, views: dict[str, np.ndarray], mask: np.ndarray, *,
                 width_deg: float = 30.0) -> pd.DataFrame:
    """Consistency (|distance deviation| max) and marker residual binned by each camera's
    in-body viewing direction, on the frames in ``mask``."""
    rows = []
    for cam, ae in views.items():
        ok = mask & np.isfinite(ae).all(axis=1)
        if not ok.any():
            continue
        ab = np.floor(ae[ok, 0] / width_deg) * width_deg
        eb = np.floor(ae[ok, 1] / width_deg) * width_deg
        d = pd.DataFrame({"az": ab, "el": eb, "dev": q["dev_max_mm"].to_numpy()[ok],
                          "res": q["resid_max_mm"].to_numpy()[ok]})
        for (a, e), g in d.groupby(["az", "el"]):
            rows.append({"camera": cam, "az_deg": a, "el_deg": e, "n": len(g),
                         "dev_med_mm": float(g.dev.median()),
                         "dev_p95_mm": float(g.dev.quantile(0.95)),
                         "resid_med_mm": float(g.res.median())})
    return pd.DataFrame(rows)


def _kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched rigid fit B ≈ R A + t; A, B (K, M, 3) → (K, 4, 4)."""
    ca, cb = A.mean(axis=1, keepdims=True), B.mean(axis=1, keepdims=True)
    H = np.einsum("kmi,kmj->kij", A - ca, B - cb)
    U, _, Vt = np.linalg.svd(H)
    D = np.ones((len(A), 3))
    D[:, 2] = np.sign(np.linalg.det(U) * np.linalg.det(Vt))      # no reflection
    R = np.einsum("kji,kj,klj->kil", Vt, D, U)                   # V diag(D) Uᵀ
    T = np.repeat(np.eye(4)[None], len(A), axis=0)
    T[:, :3, :3] = R
    T[:, :3, 3] = cb[:, 0] - np.einsum("kij,kj->ki", R, ca[:, 0])
    return T


def sigma_ref(tmpl_head: np.ndarray, tmpl_jaw: np.ndarray, sig_head: float, sig_jaw: float,
              rel: np.ndarray, F_max: np.ndarray, F_mand: np.ndarray,
              tmj_mand: dict[str, np.ndarray], *, draws: int = 200, max_frames: int = 100,
              seed: int = 0) -> dict[str, float]:
    """Monte-Carlo σ of the anatomical-frame outcome from marker noise of both bodies.

    ``rel``: Qualisys ``T_head⁻¹ T_jaw`` at the segment's frames (subsampled to
    ``max_frames``). Each draw perturbs the marker templates (isotropic σ), refits both
    bodies (Kabsch) and propagates the small pose errors ``ΔH``, ``ΔJ`` to
    ``M = F_max⁻¹ ΔH⁻¹ rel ΔJ F_mand``. Returns RMS errors (3-D norm) of the incisor and
    condyle points (mm) and of the rotation (deg)."""
    rng = np.random.default_rng(seed)
    rel = rel[se3.valid(rel)]
    if len(rel) == 0:
        return {"incisor_mm": np.nan, "rot_deg": np.nan, "tmj_mm": np.nan, "n": 0}
    if len(rel) > max_frames:
        rel = rel[np.linspace(0, len(rel) - 1, max_frames).astype(int)]
    K = draws

    def fits(tmpl, s):
        noisy = tmpl[None] + rng.normal(0.0, s, (K,) + tmpl.shape)
        return _kabsch(np.repeat(tmpl[None], K, axis=0), noisy)   # template -> noisy

    dH, dJ = fits(tmpl_head, sig_head), fits(tmpl_jaw, sig_jaw)
    Fi = se3.inv(F_max)
    M0 = np.einsum("ij,njk,kl->nil", Fi, rel, F_mand)                          # (n, 4, 4)
    Mp = np.einsum("ij,kjl,nlm,kmp,pq->kniq", Fi, se3.inv(dH), rel, dJ, F_mand)  # (K,n,4,4)
    e_inc = np.linalg.norm(Mp[..., :3, 3] - M0[None, :, :3, 3], axis=-1)
    D = np.einsum("knij,nlj->knil", Mp[..., :3, :3], M0[:, :3, :3])
    ang = np.degrees(np.arccos(np.clip((np.trace(D, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))
    out = {"incisor_mm": float(np.sqrt(np.mean(e_inc ** 2))),
           "rot_deg": float(np.sqrt(np.mean(ang ** 2))), "n": int(len(rel))}
    tm = []
    for p in tmj_mand.values():
        a = np.einsum("knij,j->kni", Mp[..., :3, :3], p) + Mp[..., :3, 3]
        b = np.einsum("nij,j->ni", M0[:, :3, :3], p) + M0[:, :3, 3]
        tm.append(np.linalg.norm(a - b[None], axis=-1))
    out["tmj_mm"] = float(np.sqrt(np.mean(np.concatenate(tm) ** 2))) if tm else np.nan
    return out


def availability(valid: np.ndarray, t_s: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Valid fraction inside ``mask`` and the dropout runs (count, median and max s)."""
    v = np.asarray(valid, bool) & mask
    runs = [(a, b) for a, b in timebase.segments(mask & ~np.asarray(valid, bool))]
    dur = np.array([t_s[b - 1] - t_s[a] for a, b in runs if b - 1 > a]) if runs else np.zeros(0)
    n = int(mask.sum())
    return {"n": n, "valid_frac": float(v.sum() / n) if n else np.nan,
            "n_dropouts": int(len(runs)),
            "dropout_med_s": float(np.median(dur)) if len(dur) else 0.0,
            "dropout_max_s": float(dur.max()) if len(dur) else 0.0}


def budget(sigma_obs: float, sigma_ref_: float, ratio_min: float = 3.0) -> dict[str, float]:
    """V9: σ_obs² = σ_JTS² + σ_ref²; corrected σ̂_JTS; adequacy σ_obs/σ_ref ≥ 3.

    ``separable``: σ_ref < σ_obs (both finite). Where the reference uncertainty reaches the
    observed error the JTS-MM part is not resolvable: ``sigma_jts_corrected`` is NaN, not 0
    (VD22)."""
    separable = bool(np.isfinite(sigma_obs) and np.isfinite(sigma_ref_)
                     and sigma_ref_ < sigma_obs)
    corr = float(np.sqrt(sigma_obs ** 2 - sigma_ref_ ** 2)) if separable else np.nan
    ratio = sigma_obs / sigma_ref_ if sigma_ref_ and np.isfinite(sigma_ref_) else np.nan
    return {"sigma_obs": sigma_obs, "sigma_ref": sigma_ref_, "sigma_jts_corrected": corr,
            "ratio": ratio, "adequate": bool(np.isfinite(ratio) and ratio >= ratio_min),
            "separable": separable}


def segment_masks(t_s: np.ndarray, intervals: Sequence) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for iv in intervals:
        k = (t_s >= iv.t0) & (t_s <= iv.t1)
        out[iv.name] = out.get(iv.name, np.zeros(len(t_s), bool)) | k
    return out
