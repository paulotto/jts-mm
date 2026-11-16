# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD29: lever arms and pose noise of the two JTS-MM arrays (validation analysis plan
``experiments/jtsmm_validation/analysis_plan.md``, block VD29-VD33, J30; descriptive, post hoc, definitions fixed in the plan before this ran).

Per comparison take (P1 = s10, P2 = s14; never pooled) on the frozen systems store (pin
``paper_systems.yaml``), no stage change, no re-run:

* **Refit.** Every frame's tag corners seen by both cameras are triangulated with the take's
  own camera model and each array (CRA = HP, tags 1/8; MTA = MP, tags 13/17) is refitted by
  Kabsch on the tag-corner model of ``optical.yaml`` (``qc/stereo_cal``: ``corner_pairs``,
  ``triangulate``, ``_BodyFit``) -- per the amendment with the live tracker's trimmed Kabsch
  (``optical.tracking.triangulation.kabsch_trimmed``, same constants, batched here) in every
  refit (reproduction, Monte Carlos, what-ifs, bench); the per-corner model residual is stored
  (tag 17 corner 2). Reproduction: refit against the stored ``optical/pose_*``
  over the movement frames (median rotation / centroid difference; stop above 0.1 deg / 0.1 mm).
* **Lever arms.** At the rest segment (median over its frames; stored poses = the JTS-MM
  own-frame chain of ``ref_kinematics``, reproduced exactly first), the vector from the array's
  corner centroid ``o_b`` to the lower incisor (origin of F_mand) and to the mean of the two
  condyle points, in the array's body axes.
* **Corner noise.** sigma_px per take and array = RMS reprojection residual of the refit (the
  corners the trimmed fit keeps, both cameras, movement frames), each frame's squared residuals x n/(n-6).
* **Pose noise (Monte Carlo).** <= 300 evenly spaced movement frames (both arrays refitted),
  200 draws, seed 0: N(0, sigma_px^2) on every corner coordinate in both cameras,
  re-triangulation, refit of both arrays; sigma_theta per body axis (RMS of the rotation-vector
  component), sigma_eps (3-D RMS displacement of o_b), per frame too (for VD30).
* **Bench check.** Ladder plateaus (``trueness.plateaus``; static window as ``ref_trueness``):
  measured SD of the eval array's orientation (per axis) and corner centroid (3-D) against the
  same Monte Carlo on the plateau frames (stored 3-D corners reprojected through the bench
  calibration; sigma_px = sqrt(2) x the stored per-corner RMS, the literal k beside it);
  k = measured / predicted, median over the plateaus. Precision takes: one plateau, linear
  drift removed; trueness slots: central 80 % (both checked against ``ref_trueness``).
* **Addendum 2** (disclosed, after the values): in-vivo upper bound from the stored-pose
  jitter over the rest segment (and the movement frames beside it), sigma_p and CRA share from
  the exact equation, what-if (b) on that scale; relative reductions of both what-ifs.
* **Propagation** (first-order exact and isotropic) at the incisor and the condyle mean, the
  Monte Carlo jaw-in-head point displacement beside it, CRA share; what-ifs: (a) 50-mm tags
  (Monte Carlo on synthetic corners, with a 30-mm synthetic control), (b) r_CRA to the incisor
  halved (equation only).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/vd29_array_noise.py [--snapshot v14]

Output (``<validation store>/audit/``): ``vd29_array_noise_<snap>.parquet`` (tidy: pid, take,
array, quantity, axis, point, value, unit, source, variant), ``vd29_array_noise_frames_<snap>
.parquet`` (per-frame Monte Carlo, read by VD30), ``vd29_array_noise_bench_<snap>.parquet``
(per plateau), ``vd29_array_noise_corners_<snap>.parquet`` (per-corner model residual) and
``vd29_array_noise_<snap>.json`` (settings, seeds, reproduction checks,
git commit, runtime, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "jtsmm_validation" / "study.yaml"
OPTICAL_YAML = _pins.ROOT / "software" / "optical" / "etc" / "optical.yaml"
BENCH_DIR = "trueness_precision_jtsmm"            # under the study's data_dir
ARRAYS = {"CRA": ("HP", "cra"), "MTA": ("MP", "mta")}   # paper name -> (optical.yaml, pose_*)
EVAL_BODY, EVAL_TAGS = "eval_array", (35, 41)
TAKES = {"s10": "P1", "s14": "P2"}
EXPOSURE_S, TRIM_S = 0.012, 0.5                   # ref_kinematics (study.yaml)
N_FRAMES, N_DRAWS, SEED = 300, 200, 0
REPRO_MAX_DEG, REPRO_MAX_MM = 0.1, 0.1
CHAIN_MAX_MM = 1e-6                               # own-frame chain vs stored ref_kinematics
BENCH_SD_MAX_MM = 1e-9                            # bench centroid SD vs stored ref_trueness
TAG_MM, TAG_WHATIF_MM = 30.0, 50.0
K_BAND = (0.8, 1.25)
FIT_DOF = 6                                       # 6-parameter pose fit (in vivo sigma_px)
# Bench: the stored per-corner reprojection_rms_px of a two-view triangulated corner is
# sqrt(mean over the 2 views of |e|^2) = sqrt(SSR / 2), with 4 residual coordinates and 3
# point parameters. The same dof correction as in vivo (per-coordinate RMS x sqrt(n / (n - p)),
# n = 4, p = 3) gives sigma_px = sqrt(SSR / 1) = sqrt(2) x reprojection_rms_px.
BENCH_DOF_FACTOR = float(np.sqrt(2.0))
AXES = ("x", "y", "z")


# -- geometry helpers ------------------------------------------------------------------------
def rotvec(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    shp = R.shape[:-2]
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(*shp, 3)


def rel_angle_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.degrees(np.linalg.norm(rotvec(np.einsum("fji,fjk->fik", A, B)), axis=1))


def array_tags(model: dict[int, np.ndarray], ranges: dict[str, tuple[int, int]], body: str
               ) -> list[int]:
    lo, hi = ranges[body]
    return sorted(t for t in model if lo <= t <= hi)


def centroid(model: dict[int, np.ndarray], tags: list[int]) -> np.ndarray:
    return np.concatenate([model[t] for t in tags]).mean(axis=0)


def scaled_model(model: dict[int, np.ndarray], factor: float) -> dict[int, np.ndarray]:
    """Tag corners scaled about each tag centre (what-if a)."""
    return {t: c.mean(axis=0) + factor * (c - c.mean(axis=0)) for t, c in model.items()}


def project(cal: Any, X: np.ndarray, cam: int) -> np.ndarray:
    """World (= cam0) points (..., 3) -> pixels (..., 2) of camera ``cam``."""
    import cv2

    shp = X.shape[:-1]
    if cam == 0:
        rv, tv, K, d = np.zeros(3), np.zeros(3), cal.K0, cal.d0
    else:
        rv, tv, K, d = cv2.Rodrigues(cal.R)[0].ravel(), cal.t, cal.K1, cal.d1
    uv, _ = cv2.projectPoints(np.ascontiguousarray(X.reshape(-1, 3), float), rv, tv, K, d)
    return uv.reshape(*shp, 2)


def solve(cal: Any, p0: np.ndarray, p1: np.ndarray, fits: dict[str, Any]
          ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Undistort, triangulate, refit every body: {body: (R, t, ok)} per frame."""
    from biomech.qc import stereo_cal as sc

    n0 = sc._undistort(p0, cal.K0, cal.d0)
    n1 = sc._undistort(p1, cal.K1, cal.d1)
    X = sc.triangulate(cal, n0, n1).reshape(-1, 4, 3)
    return {b: f.fit(X) for b, f in fits.items()}


def _make_trimfit():
    """``qc/stereo_cal._BodyFit`` with the live tracker's fit (amendment VD29 / VD30):
    ``optical.tracking.triangulation.kabsch_trimmed`` batched over frames -- plain Kabsch, then
    up to ``_TRIM_ROUNDS`` rounds dropping the corners whose 3-D residual exceeds
    max(``_TRIM_K`` x median of the kept residuals, ``_TRIM_MIN_MM``); a frame stops trimming
    when nothing changes or fewer than 4 corners would remain (tracker ``min_points``); a pose
    whose final RMS over the kept corners exceeds 5 mm is rejected (``max_residual_mm``)."""
    from biomech.qc import stereo_cal as sc
    from optical.tracking import triangulation as TRI

    k_trim, min_mm, rounds = float(TRI._TRIM_K), float(TRI._TRIM_MIN_MM), int(TRI._TRIM_ROUNDS)

    class TrimFit(sc._BodyFit):
        constants = {"trim_k": k_trim, "trim_min_mm": min_mm, "trim_rounds": rounds,
                     "min_points": 4, "max_residual_mm": 5.0}

        def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            M = len(self.tags) * 4
            Q = np.full((self.F, M, 3), np.nan)
            W = np.zeros((self.F, M))
            for k in range(4):
                Q[self.f, self.c * 4 + k] = X[self.src, k]
                W[self.f, self.c * 4 + k] = 1.0
            ok = W.sum(axis=1) >= 4                         # at least one full tag
            R = np.full((self.F, 3, 3), np.nan)
            t = np.full((self.F, 3), np.nan)
            self.keep = np.zeros((self.F, M), bool)
            if not ok.any():
                return R, t, ok
            Qo, keep = Q[ok], W[ok] > 0
            Ro, to = sc._batch_kabsch(self.P, Qo, keep.astype(float))

            def resid(Rr: np.ndarray, tr: np.ndarray, Qq: np.ndarray) -> np.ndarray:
                return np.linalg.norm(Qq - (np.einsum("fij,mj->fmi", Rr, self.P)
                                            + tr[:, None, :]), axis=2)

            active = np.ones(len(Qo), bool)
            for _ in range(rounds):
                res = resid(Ro, to, Qo)
                med = np.nanmedian(np.where(keep, res, np.nan), axis=1)
                thr = np.maximum(k_trim * med, min_mm)
                new = keep & (np.nan_to_num(res, nan=np.inf) <= thr[:, None])
                ns, ks = new.sum(axis=1), keep.sum(axis=1)
                active = active & (ns != ks) & (ns >= 4)
                if not active.any():
                    break
                keep[active] = new[active]
                Ro[active], to[active] = sc._batch_kabsch(self.P, Qo[active],
                                                          keep[active].astype(float))
            res = resid(Ro, to, Qo)
            rms = np.sqrt(np.nanmean(np.where(keep, res, np.nan) ** 2, axis=1))
            good = rms <= 5.0
            idx = np.flatnonzero(ok)
            ok[idx[~good]] = False
            R[idx[good]], t[idx[good]] = Ro[good], to[good]
            self.keep[idx[good]] = keep[good]
            return R, t, ok

        def pair_keep(self) -> np.ndarray:
            """(len(src), 4) kept-corner mask of the last fit."""
            cols = self.c[:, None] * 4 + np.arange(4)[None, :]
            return self.keep[self.f[:, None], cols]

    return TrimFit


_TRIMFIT = None


def TrimFit(*args: Any, **kw: Any) -> Any:
    global _TRIMFIT
    if _TRIMFIT is None:
        _TRIMFIT = _make_trimfit()
    return _TRIMFIT(*args, **kw)


def fitted_corners(fit: Any, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World corners of the fitted model per pair of ``fit`` (len(fit.src), 4, 3)."""
    P = fit.P.reshape(len(fit.tags), 4, 3)[fit.c]                      # (n, 4, 3)
    return np.einsum("nij,nkj->nki", R[fit.f], P) + t[fit.f][:, None, :]


# -- Monte Carlo ------------------------------------------------------------------------------
def monte_carlo(cal: Any, p0: np.ndarray, p1: np.ndarray, fits: dict[str, Any],
                sig: np.ndarray, cents: dict[str, np.ndarray], n_draws: int,
                rng: np.random.Generator, points: dict[str, np.ndarray] | None = None
                ) -> dict[str, Any]:
    """Noise N(0, sig^2) per pair on every corner coordinate of both cameras; per frame the
    mean over draws of the squared rotation-vector components (body axes), the squared 3-D
    displacement of the centroid ``cents[b]`` and (``points``: jaw-in-head, bodies ``CRA`` and
    ``MTA``) the squared displacement of each point (MTA body frame) in the CRA frame."""
    ref = solve(cal, p0, p1, fits)
    F = next(iter(fits.values())).F
    acc = {b: {"rv2": np.zeros((F, 3)), "eps2": np.zeros(F), "n": np.zeros(F)} for b in fits}
    pt2 = {k: np.zeros(F) for k in (points or {})}
    pt_n = np.zeros(F)

    def jih(sol: dict, p: np.ndarray) -> np.ndarray:
        Rh, th, _ = sol["CRA"]
        Rm, tm, _ = sol["MTA"]
        w = np.einsum("fij,j->fi", Rm, p) + tm
        return np.einsum("fji,fj->fi", Rh, w - th)

    ref_pts = {k: jih(ref, p) for k, p in (points or {}).items()}
    for _ in range(n_draws):
        e = rng.standard_normal((2,) + p0.shape) * sig[None, :, None, None]
        out = solve(cal, p0 + e[0], p1 + e[1], fits)
        for b in fits:
            R0, t0, ok0 = ref[b]
            R, t, ok1 = out[b]
            ok = ok0 & ok1                      # a draw the tracker's fit rejects is skipped
            rv = rotvec(np.einsum("fji,fjk->fik", R0[ok], R[ok]))
            acc[b]["rv2"][ok] += rv ** 2
            o0 = np.einsum("fij,j->fi", R0[ok], cents[b]) + t0[ok]
            o1 = np.einsum("fij,j->fi", R[ok], cents[b]) + t[ok]
            acc[b]["eps2"][ok] += np.sum((o1 - o0) ** 2, axis=1)
            acc[b]["n"][ok] += 1
        if points:
            okp = ref["CRA"][2] & ref["MTA"][2] & out["CRA"][2] & out["MTA"][2]
            pt_n[okp] += 1
            for k, p in points.items():
                d = jih(out, p) - ref_pts[k]
                pt2[k][okp] += np.sum(d[okp] ** 2, axis=1)
    res: dict[str, Any] = {"ref": ref, "n_draws": n_draws}
    for b in fits:
        ok = ref[b][2] & (acc[b]["n"] > 0)
        res[b] = {"ok": ok, "rv2_sum": acc[b]["rv2"], "eps2_sum": acc[b]["eps2"],
                  "n": acc[b]["n"],
                  "n_rejected": int(ref[b][2].sum() * n_draws - acc[b]["n"].sum())}
        with np.errstate(invalid="ignore", divide="ignore"):
            res[b]["rv2"] = acc[b]["rv2"] / acc[b]["n"][:, None]
            res[b]["eps2"] = acc[b]["eps2"] / acc[b]["n"]
    if points:
        okb = ref["CRA"][2] & ref["MTA"][2] & (pt_n > 0)
        res["points"] = {k: {"ok": okb, "d2_sum": v, "n": pt_n} for k, v in pt2.items()}
    return res


def mc_summary(m: dict[str, Any], b: str) -> dict[str, Any]:
    """RMS over all accepted (frame, draw) pairs."""
    ok, n = m[b]["ok"], m[b]["n"]
    th = np.degrees(np.sqrt(m[b]["rv2_sum"][ok].sum(axis=0) / n[ok].sum()))
    return {"theta_deg": th, "theta_rms_deg": float(np.sqrt(np.mean(th ** 2))),
            "theta_3d_deg": float(np.sqrt(np.sum(th ** 2))),
            "eps_mm": float(np.sqrt(m[b]["eps2_sum"][ok].sum() / n[ok].sum())),
            "n_frames": int(ok.sum()), "n_rejected_draws": m[b]["n_rejected"]}


# -- propagation ------------------------------------------------------------------------------
def propagate(noise: dict[str, dict[str, Any]], r: dict[str, np.ndarray]) -> dict[str, float]:
    """sigma_p (3-D RMS, mm) at a point: first order, exact for anisotropic noise, and the
    isotropic form; ``r[b]`` = vector from o_b to the point in the body axes of b."""
    exact, iso = {}, {}
    for b, nb in noise.items():
        th = np.radians(np.asarray(nb["theta_deg"], float))
        rb = np.asarray(r[b], float)
        cross2 = rb @ rb - rb ** 2                          # |e_k x r|^2
        exact[b] = nb["eps_mm"] ** 2 + float(np.sum(th ** 2 * cross2))
        iso[b] = nb["eps_mm"] ** 2 + 2.0 * float(rb @ rb) * float(np.mean(th ** 2))
    se, si = sum(exact.values()), sum(iso.values())
    return {"exact_mm": float(np.sqrt(se)), "isotropic_mm": float(np.sqrt(si)),
            "cra_share_exact": exact["CRA"] / se, "cra_share_isotropic": iso["CRA"] / si}


# -- in-vivo takes ----------------------------------------------------------------------------
def take_context(ctx: Any, pid: str) -> dict[str, Any]:
    """Everything a take needs: calibration, corner pairs (array tags), model, times, held
    intervals, stored poses, the JTS-MM own-frame landmarks (reproduced against ref_kinematics)."""
    from biomech.io.session import SessionData
    from biomech.kinematics import se3
    from biomech.qc import stereo_cal as sc
    from biomech.validation import anatomy as AN
    from biomech.validation import selection as SEL
    from biomech.validation import sync as SC
    from biomech.validation.evaluate import holds_from_table

    folder = ctx.session_path(pid)
    h5 = folder / "session.h5"
    cal = sc.session_calibration(h5)
    if cal is None:
        raise SystemExit(f"{pid}: no stored camera model")
    model, ranges = sc.marker_model(OPTICAL_YAML)
    tags = {a: array_tags(model, ranges, ARRAYS[a][0]) for a in ARRAYS}
    pairs = sc.corner_pairs(h5)
    pairs = pairs.subset(np.flatnonzero(np.isin(pairs.tag, tags["CRA"] + tags["MTA"])))
    with SessionData(folder) as sd:
        opt = sd.optical()
        phases = list(sd.phases)
    tS = SC.frame_times(opt.t_us, EXPOSURE_S)
    ct_ok = se3.valid(opt.poses["ct"])
    held = SEL.movement_intervals(phases, tS, ct_ok, trim_s=TRIM_S)
    mov = np.zeros(len(tS), bool)
    for h in held:
        if h.name in SEL.MOVEMENT_TASKS:
            mov |= (tS >= h.t0) & (tS <= h.t1)
    rest = [h for h in held if h.name in SEL.STATIC_TASKS]
    if not rest:
        raise SystemExit(f"{pid}: no rest segment")
    rest_m = (tS >= rest[0].t0) & (tS <= rest[0].t1)
    stored = {a: opt.poses[ARRAYS[a][1]] for a in ARRAYS}

    # own-frame chain (ref_kinematics): F_max, F_mand from ref_tip:landmarks (jtsmm)
    hs = holds_from_table(ctx.table("ref_tip", "landmarks", [pid]), "jtsmm", "-")
    fr = AN.anatomical_frames(hs)
    pts = AN.points_of(AN.jaw_in_head(stored["CRA"], stored["MTA"], fr), fr)
    arr = ctx.arrays("ref_kinematics", pid)
    chain = {"t_master_s_max_abs_s": float(np.nanmax(np.abs(tS - arr["t_master_s"])))}
    for k, a in (("incisor", "incisor_s_mm"), ("tmj_left", "tmj_left_s_mm"),
                 ("tmj_right", "tmj_right_s_mm")):
        same_nan = bool(np.array_equal(np.isnan(pts[k]), np.isnan(arr[a])))
        chain[f"{k}_max_abs_mm"] = float(np.nanmax(np.abs(pts[k] - arr[a])))
        chain[f"{k}_nan_pattern_equal"] = same_nan
    chain["ok"] = bool(all(v <= CHAIN_MAX_MM for k, v in chain.items() if k.endswith("_mm"))
                       and all(v for k, v in chain.items() if k.endswith("_equal"))
                       and chain["t_master_s_max_abs_s"] == 0.0)
    if not chain["ok"]:
        raise SystemExit(f"{pid}: own-frame chain does not reproduce ref_kinematics: {chain}")
    inc_mp = fr.F_mand[:3, 3].copy()                   # lower incisor = origin of F_mand
    tmj_mp = {n: np.asarray(hs[n].point, float) for n in ("tmj_left", "tmj_right")}
    for n in tmj_mp:                                   # holds in the MP body frame
        if hs[n].body.lower() != "mp" or not np.allclose(
                fr.F_mand[:3, :3] @ fr.tmj_mand[n] + fr.F_mand[:3, 3], tmj_mp[n]):
            raise SystemExit(f"{pid}: condyle point {n} not in the MP body frame")
    points_mp = {"incisor": inc_mp, "condyle": 0.5 * (tmj_mp["tmj_left"] + tmj_mp["tmj_right"])}
    return {"pid": pid, "h5": h5, "cal": cal, "model": model, "ranges": ranges, "tags": tags,
            "pairs": pairs, "tS": tS, "t_us": np.asarray(opt.t_us), "mov": mov,
            "rest": rest_m, "held": held, "stored": stored, "points_mp": points_mp,
            "chain": chain, "phases": phases}


def refit_take(tc: dict[str, Any]) -> dict[str, Any]:
    """Refit of both arrays in every frame with a corner pair; per optical row (T, 4, 4)."""
    from biomech.qc import stereo_cal as sc

    pairs, model = tc["pairs"], tc["model"]
    frames = np.unique(pairs.row)
    fits = {a: TrimFit(pairs, frames, tc["tags"][a], model) for a in ARRAYS}
    sol = solve(tc["cal"], pairs.p0, pairs.p1, fits)
    n0 = sc._undistort(pairs.p0, tc["cal"].K0, tc["cal"].d0)
    n1 = sc._undistort(pairs.p1, tc["cal"].K1, tc["cal"].d1)
    X = sc.triangulate(tc["cal"], n0, n1).reshape(-1, 4, 3)
    T = len(tc["tS"])
    poses = {}
    for a, (R, t, ok) in sol.items():
        P = np.full((T, 4, 4), np.nan)
        P[frames[ok], :3, :3] = R[ok]
        P[frames[ok], :3, 3] = t[ok]
        P[frames[ok], 3] = [0, 0, 0, 1]
        poses[a] = P
    return {"frames": frames, "fits": fits, "sol": sol, "poses": poses, "X": X}


def reproduction(tc: dict[str, Any], rf: dict[str, Any]) -> dict[str, Any]:
    """Refit against the stored poses over the movement frames (both valid)."""
    from biomech.kinematics import se3

    out = {}
    for a in ARRAYS:
        P, S = rf["poses"][a], tc["stored"][a]
        k = tc["mov"] & se3.valid(P) & se3.valid(S)
        c = centroid(tc["model"], tc["tags"][a])
        ang = rel_angle_deg(S[k][:, :3, :3], P[k][:, :3, :3])
        d = np.linalg.norm((np.einsum("fij,j->fi", S[k][:, :3, :3], c) + S[k][:, :3, 3])
                           - (np.einsum("fij,j->fi", P[k][:, :3, :3], c) + P[k][:, :3, 3]),
                           axis=1)
        out[a] = {"n_frames": int(k.sum()), "median_rot_deg": float(np.median(ang)),
                  "p95_rot_deg": float(np.percentile(ang, 95)),
                  "median_centroid_mm": float(np.median(d)),
                  "p95_centroid_mm": float(np.percentile(d, 95)),
                  "n_movement_frames": int(tc["mov"].sum()),
                  "n_refit": int((tc["mov"] & se3.valid(P)).sum()),
                  "n_stored": int((tc["mov"] & se3.valid(S)).sum())}
        out[a]["ok"] = bool(out[a]["median_rot_deg"] <= REPRO_MAX_DEG
                            and out[a]["median_centroid_mm"] <= REPRO_MAX_MM)
    return out


def sigma_px(tc: dict[str, Any], rf: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """RMS reprojection residual of the refit over the movement frames and the corners the
    trimmed fit keeps, each frame's squared residuals x n / (n - 6) (n = residual coordinates
    of the array in the frame: kept corners x 2 cameras x 2)."""
    pairs, frames, cal = tc["pairs"], rf["frames"], tc["cal"]
    out = {}
    for a, fit in rf["fits"].items():
        R, t, ok = rf["sol"][a]
        sel = ok[fit.f] & tc["mov"][frames[fit.f]]
        src, fi = fit.src[sel], fit.f[sel]
        kc = fit.pair_keep()[sel]                                  # (n, 4)
        Xw = fitted_corners(fit, R, t)[sel]
        e0 = project(cal, Xw, 0) - pairs.p0[src]
        e1 = project(cal, Xw, 1) - pairs.p1[src]
        ssr_pair = np.sum((np.sum(e0 ** 2, axis=2) + np.sum(e1 ** 2, axis=2)) * kc, axis=1)
        ssr = np.bincount(fi, ssr_pair, minlength=len(frames))
        n = np.bincount(fi, kc.sum(axis=1), minlength=len(frames)) * 4.0   # x 2 cams x 2
        m = n > FIT_DOF
        corr = ssr[m] * n[m] / (n[m] - FIT_DOF)
        e_all = np.sqrt(np.sum(e0 ** 2, axis=2) + np.sum(e1 ** 2, axis=2))
        out[a] = {"sigma_px": float(np.sqrt(corr.sum() / n[m].sum())),
                  "rms_raw_px": float(np.sqrt(ssr[m].sum() / n[m].sum())),
                  "rms_all_corners_px": float(np.sqrt(np.mean(e_all ** 2) / 4.0)),
                  "kept_corner_share": float(kc.mean()),
                  "n_frames": int(m.sum()), "n_coords": int(n[m].sum()),
                  "mean_dof_factor": float(np.mean(n[m] / (n[m] - FIT_DOF)))}
    return out


def corner_offsets(tc: dict[str, Any], rf: dict[str, Any]) -> pd.DataFrame:
    """Per corner of every array tag: median offset of the triangulated corner from its model
    position in the body frame of the refit (movement frames), and the share of those frames
    in which the trimmed fit dropped it."""
    frames, model, X = rf["frames"], tc["model"], rf["X"]
    rows = []
    for a, fit in rf["fits"].items():
        R, t, ok = rf["sol"][a]
        sel = ok[fit.f] & tc["mov"][frames[fit.f]]
        src, fi, ci = fit.src[sel], fit.f[sel], fit.c[sel]
        kc = fit.pair_keep()[sel]
        loc = np.einsum("nji,nkj->nki", R[fi], X[src] - t[fi][:, None, :])
        for j, tag in enumerate(fit.tags):
            m = ci == j
            for k in range(4):
                d = np.median(loc[m, k] - model[tag][k], axis=0)
                rows.append({"pid": tc["pid"], "take": TAKES[tc["pid"]], "array": a,
                             "tag": int(tag), "corner": k, "dx_mm": float(d[0]),
                             "dy_mm": float(d[1]), "dz_mm": float(d[2]),
                             "offset_mm": float(np.linalg.norm(d)),
                             "trimmed_share": float(1.0 - kc[m, k].mean()),
                             "n_frames": int(m.sum())})
    return pd.DataFrame(rows)


def lever_arms(tc: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    """Body-frame vector o_b -> point, median over the rest frames (stored poses)."""
    from biomech.kinematics import se3

    S = tc["stored"]
    k = tc["rest"] & se3.valid(S["CRA"]) & se3.valid(S["MTA"])
    out: dict[str, dict[str, Any]] = {}
    for a in ARRAYS:
        c = centroid(tc["model"], tc["tags"][a])
        Rb, tb = S[a][k][:, :3, :3], S[a][k][:, :3, 3]
        o = np.einsum("fij,j->fi", Rb, c) + tb
        out[a] = {}
        for name, p in tc["points_mp"].items():
            pw = np.einsum("fij,j->fi", S["MTA"][k][:, :3, :3], p) + S["MTA"][k][:, :3, 3]
            rb = np.einsum("fji,fj->fi", Rb, pw - o)
            out[a][name] = {"r": np.median(rb, axis=0),
                            "norm_median_mm": float(np.median(np.linalg.norm(rb, axis=1))),
                            "n_rest_frames": int(k.sum())}
    return out


def pose_jitter(tc: dict[str, Any], mask: np.ndarray, gap_tol: float = 1.5
                ) -> dict[str, dict[str, Any]]:
    """Addendum 2: per array, the jitter of the stored pose over the frames in ``mask`` (three
    consecutive valid rows, both gaps < ``gap_tol`` x the median frame period, as VD30):
    sigma_theta per body axis = RMS of the rotation vector of Rbar(t)^T R(t) (Rbar the
    geodesic midpoint of R(t-1), R(t+1)) x sqrt(2/3); sigma_eps = 3-D RMS of
    o(t) - (o(t-1) + o(t+1)) / 2 x sqrt(2/3)."""
    from biomech.kinematics import se3
    from scipy.spatial.transform import Rotation

    dt = np.diff(tc["t_us"].astype(float))
    med = np.median(dt)
    f = float(np.sqrt(2.0 / 3.0))
    out = {}
    for a in ARRAYS:
        S = tc["stored"][a]
        ok = se3.valid(S)
        i = np.arange(1, len(S) - 1)
        g = (mask[i] & ok[i] & ok[i - 1] & ok[i + 1] & (dt[i - 1] < gap_tol * med)
             & (dt[i] < gap_tol * med))
        i = i[g]
        Ra, Rb, Rt = S[i - 1, :3, :3], S[i + 1, :3, :3], S[i, :3, :3]
        half = Rotation.from_rotvec(0.5 * rotvec(np.einsum("fji,fjk->fik", Ra, Rb))).as_matrix()
        Rm = np.einsum("fij,fjk->fik", Ra, half)
        rv = rotvec(np.einsum("fji,fjk->fik", Rm, Rt))
        c = centroid(tc["model"], tc["tags"][a])
        o = np.einsum("fij,j->fi", S[:, :3, :3], c) + S[:, :3, 3]
        d = o[i] - 0.5 * (o[i - 1] + o[i + 1])
        out[a] = {"theta_deg": np.degrees(np.sqrt(np.mean(rv ** 2, axis=0))) * f,
                  "eps_mm": float(np.sqrt(np.mean(np.sum(d ** 2, axis=1)))) * f,
                  "n_frames": int(len(i))}
    return out


def mc_frames(tc: dict[str, Any], rf: dict[str, Any]) -> np.ndarray:
    """Up to N_FRAMES evenly spaced movement frames with both arrays refitted (indices into
    ``rf['frames']``)."""
    okb = rf["sol"]["CRA"][2] & rf["sol"]["MTA"][2] & tc["mov"][rf["frames"]]
    cand = np.flatnonzero(okb)
    if len(cand) <= N_FRAMES:
        return cand
    return cand[np.unique(np.round(np.linspace(0, len(cand) - 1, N_FRAMES)).astype(int))]


def subset_fits(tc: dict[str, Any], sel_rows: np.ndarray, model: dict[int, np.ndarray]
                ) -> tuple[Any, dict[str, Any]]:
    from biomech.qc import stereo_cal as sc

    pairs = tc["pairs"].subset(np.flatnonzero(np.isin(tc["pairs"].row, sel_rows)))
    fits = {a: TrimFit(pairs, sel_rows, tc["tags"][a], model) for a in ARRAYS}
    return pairs, fits


def invivo(tc: dict[str, Any], rf: dict[str, Any], spx: dict[str, Any], seed: int
           ) -> dict[str, Any]:
    """Primary Monte Carlo on the observed corners, and the what-if (a) on synthetic corners
    (30-mm control and 50-mm tags) generated from the refit poses."""
    sel = rf["frames"][mc_frames(tc, rf)]
    pairs, fits = subset_fits(tc, sel, tc["model"])
    cents = {a: centroid(tc["model"], tc["tags"][a]) for a in ARRAYS}
    sig = np.where(np.isin(pairs.tag, tc["tags"]["CRA"]), spx["CRA"]["sigma_px"],
                   spx["MTA"]["sigma_px"])
    rng = np.random.default_rng(seed)
    prim = monte_carlo(tc["cal"], pairs.p0, pairs.p1, fits, sig, cents, N_DRAWS, rng,
                       tc["points_mp"])
    # synthetic corners: the refit pose of every frame, model corners of the given tag size,
    # projected into both cameras (the tags detected by both cameras in that frame)
    ref = prim["ref"]
    whatif = {}
    for name, factor in (("whatif_tag30_synthetic", 1.0),
                         ("whatif_tag50", TAG_WHATIF_MM / TAG_MM)):
        mdl = scaled_model(tc["model"], factor)
        _, fits_s = subset_fits(tc, sel, mdl)
        p0 = np.full_like(pairs.p0, np.nan)
        p1 = np.full_like(pairs.p1, np.nan)
        for a, f in fits_s.items():
            R, t, _ = ref[a]
            Xw = fitted_corners(f, R, t)
            p0[f.src], p1[f.src] = project(tc["cal"], Xw, 0), project(tc["cal"], Xw, 1)
        whatif[name] = monte_carlo(tc["cal"], p0, p1, fits_s, sig, cents, N_DRAWS,
                                   np.random.default_rng(seed), tc["points_mp"])
    return {"rows": sel, "primary": prim, **whatif}


# -- bench ------------------------------------------------------------------------------------
def bench_calibration(leaf: Path) -> tuple[Any, dict[str, Any]]:
    """Stereo model of a bench recording: the bundle's intrinsics scaled to the tracked
    stream (meta.json image_size; pixel-centre aligned, ``optical`` scale_K) and extrinsics."""
    import h5py

    from biomech.qc.stereo_cal import StereoCal
    from optical.calibration.geometry import scale_K

    meta = json.loads((leaf / "meta.json").read_text())
    wh = tuple(int(v) for v in meta["image_size"])
    c = {}
    with h5py.File(leaf / meta.get("calibration_bundle", "calibration_bundle.h5"), "r") as f:
        for cam in ("cam0", "cam1"):
            g = f[f"cameras/{cam}"]
            src = tuple(int(v) for v in g["intrinsics/image_size"][()])
            c[cam] = {"K": scale_K(np.asarray(g["intrinsics/K"][()], float), src, wh),
                      "d": np.asarray(g["intrinsics/dist"][()], float).ravel(),
                      "T": np.asarray(g["extrinsics/T_world_cam"][()], float), "src": src}
    T = np.linalg.inv(c["cam0"]["T"]) @ c["cam1"]["T"]
    cal = StereoCal(c["cam0"]["K"], c["cam0"]["d"], c["cam1"]["K"], c["cam1"]["d"], T)
    return cal, {"image_size": list(wh), "bundle_image_size": list(c["cam0"]["src"]),
                 "T_world_cam0_identity": bool(np.allclose(c["cam0"]["T"], np.eye(4))),
                 "stream": meta.get("stream")}


def bench_corners(h5: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eval-array corners per frame (F, 2 tags, 4, 3), their stored reprojection_rms_px
    (F, 2, 4), times, completeness (all 8 corners)."""
    import h5py

    with h5py.File(h5, "r") as f:
        t = np.asarray(f["timestamps/t_master_us"][:], float) * 1e-6
        m = f["markers_3d"]
        fr = np.asarray(m["frame_row"][:], int)
        ids = np.asarray(m["marker_id"][:], int)
        ci = np.asarray(m["corner_index"][:], int)
        pos = np.asarray(m["position_mm"][:], float)
        rp = np.asarray(m["reprojection_rms_px"][:], float)
    X = np.full((len(t), len(EVAL_TAGS), 4, 3), np.nan)
    E = np.full((len(t), len(EVAL_TAGS), 4), np.nan)
    for j, tag in enumerate(EVAL_TAGS):
        k = ids == tag
        X[fr[k], j, ci[k]] = pos[k]
        E[fr[k], j, ci[k]] = rp[k]
    complete = np.isfinite(X).all(axis=(1, 2, 3))
    return X, E, t, complete


def _sd(v: np.ndarray, t: np.ndarray, detrend: bool) -> np.ndarray:
    """Per-column SD: ddof 1, or after a linear trend (ddof 2) as ``precision_metrics``."""
    if not detrend:
        return v.std(axis=0, ddof=1)
    A = np.column_stack([np.ones_like(t), t - t.mean()])
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return (v - A @ coef).std(axis=0, ddof=2)


def bench(data_dir: Path, model: dict[int, np.ndarray], seed: int, stored_slots: pd.DataFrame,
          stored_prec: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    from biomech.qc import stereo_cal as sc
    from biomech.qc.stereo_cal import CornerPairs
    from biomech.validation import trueness as TR

    folder = data_dir / BENCH_DIR
    rows, checks = [], {"sessions": {}}
    rng = np.random.default_rng(seed)
    rng_lit = np.random.default_rng(seed)
    cent = centroid(model, list(EVAL_TAGS))
    for leaf in sorted(q for q in folder.iterdir() if (q / "session.h5").is_file()):
        m_leaf, _ = sc.marker_model(leaf / "optical.yaml") if (leaf / "optical.yaml").is_file() \
            else (None, None)
        same_model = m_leaf is not None and all(np.allclose(m_leaf[t], model[t])
                                                for t in EVAL_TAGS)
        cal, cinfo = bench_calibration(leaf)
        X, E, t, complete = bench_corners(leaf / "session.h5")
        tc, pc = TR.array_centroid(leaf / "session.h5", EVAL_TAGS)
        low = leaf.name.lower().split("_")
        kind = "precision" if "precision" in low else "trueness"
        dist = next(x for x in low if x in TR.DISTANCES)
        ori = next((x for x in low if x in TR.ORIENTATIONS), "")
        if kind == "trueness":
            plats = [(mm, idx[TR.central(len(idx), 0.8)]) for mm, idx in TR.plateaus(tc, pc)]
        else:
            # one static take: the whole take is the plateau (ref_trueness precision_central
            # = 1.0, linear drift removed; plateaus() needs >= 2 groups and returns none here)
            idx = np.flatnonzero(np.isfinite(pc).all(axis=1))
            plats = [(np.nan, idx[TR.central(len(idx), 1.0)])]
        # frames in the camera images (sanity of the bench calibration convention)
        allX = X[complete].reshape(-1, 3)
        u0, u1 = project(cal, allX, 0), project(cal, allX, 1)
        w, h = cinfo["image_size"]
        inside = float(np.mean([((u[:, 0] >= 0) & (u[:, 0] < w) & (u[:, 1] >= 0)
                                 & (u[:, 1] < h)).all() for u in (u0, u1)]))
        zc1 = (allX @ cal.R.T + cal.t)[:, 2]
        checks["sessions"][leaf.name] = {**cinfo, "model_identical": bool(same_model),
                                         "all_reprojected_inside_images": bool(inside == 1.0),
                                         "min_depth_cam1_mm": float(zc1.min()),
                                         "min_depth_cam0_mm": float(allX[:, 2].min()),
                                         "n_plateaus": len(plats)}
        for mm, idx in plats:
            idx = idx[complete[idx]]
            # measured: corner centroid (array_centroid) and Kabsch orientation per frame
            pairs_all = CornerPairs(np.zeros((len(idx) * 2, 4, 2)), np.zeros((len(idx) * 2, 4, 2)),
                                    np.repeat(idx, 2), np.tile(EVAL_TAGS, len(idx)))
            fit = TrimFit(pairs_all, idx, list(EVAL_TAGS), model)
            R, tt, ok = fit.fit(X[idx].reshape(-1, 4, 3))
            from scipy.spatial.transform import Rotation

            Rm = Rotation.from_matrix(R[ok]).mean().as_matrix()
            rv = np.degrees(rotvec(np.einsum("ji,fjk->fik", Rm, R[ok])))
            det = kind == "precision"
            sd_th = _sd(rv, t[idx][ok], det)
            sd_p = _sd(pc[idx], t[idx], det)
            sd_c3 = float(np.sqrt(np.sum(sd_p ** 2)))
            if kind == "trueness":
                st = stored_slots[(stored_slots.source == leaf.name)
                                  & np.isclose(stored_slots.nominal_mm, mm)]
            else:
                st = stored_prec[stored_prec.source == leaf.name]
            stored_sd = float(st["sd_3d_mm"].iloc[0]) if len(st) == 1 else np.nan
            # predicted: the same Monte Carlo on (up to N_FRAMES of) the plateau frames
            sub = idx if len(idx) <= N_FRAMES else idx[np.unique(np.round(
                np.linspace(0, len(idx) - 1, N_FRAMES)).astype(int))]
            Xs = X[sub].reshape(-1, 4, 3)
            pairs = CornerPairs(project(cal, Xs, 0), project(cal, Xs, 1), np.repeat(sub, 2),
                                np.tile(EVAL_TAGS, len(sub)))
            fits = {"eval": TrimFit(pairs, sub, list(EVAL_TAGS), model)}
            rms_store = float(np.sqrt(np.nanmean(E[idx] ** 2)))
            s_px = BENCH_DOF_FACTOR * rms_store
            mc = monte_carlo(cal, pairs.p0, pairs.p1, fits, np.full(len(pairs.row), s_px),
                             {"eval": cent}, N_DRAWS, rng)
            # round trip: noise-free reprojected corners re-triangulated vs stored corners
            Rr, tr_, okr = mc["ref"]["eval"]
            R0s, t0s, _ = fit.fit(X[idx].reshape(-1, 4, 3))
            pos_s = {int(r): i for i, r in enumerate(idx)}
            j = np.array([pos_s[int(r)] for r in sub])
            rt_deg = float(np.max(rel_angle_deg(R0s[j], Rr)))
            pr = mc_summary(mc, "eval")
            # literal: the stored RMS itself as sigma_px (amendment: stored beside the primary)
            prl = mc_summary(monte_carlo(cal, pairs.p0, pairs.p1, fits,
                                         np.full(len(pairs.row), rms_store), {"eval": cent},
                                         N_DRAWS, rng_lit), "eval")
            row = {"session": leaf.name, "kind": kind, "orientation": ori, "distance": dist,
                   "nominal_mm": mm, "n_frames": int(len(idx)), "n_mc_frames": int(len(sub)),
                   "reproj_rms_stored_px": rms_store, "sigma_px": s_px,
                   "meas_centroid_3d_mm": sd_c3, "stored_sd_3d_mm": stored_sd,
                   "pred_centroid_3d_mm": pr["eps_mm"], "k_centroid": sd_c3 / pr["eps_mm"],
                   "roundtrip_max_deg": rt_deg, "n_rejected_draws": pr["n_rejected_draws"],
                   "n_rejected_draws_literal": prl["n_rejected_draws"],
                   "pred_centroid_3d_literal_mm": prl["eps_mm"],
                   "k_centroid_literal": sd_c3 / prl["eps_mm"]}
            for i, ax in enumerate(AXES):
                row[f"pred_theta_{ax}_literal_deg"] = float(prl["theta_deg"][i])
                row[f"k_theta_{ax}_literal"] = float(sd_th[i] / prl["theta_deg"][i])
                row[f"meas_theta_{ax}_deg"] = float(sd_th[i])
                row[f"pred_theta_{ax}_deg"] = float(pr["theta_deg"][i])
                row[f"k_theta_{ax}"] = float(sd_th[i] / pr["theta_deg"][i])
            meas_rms = float(np.sqrt(np.mean(sd_th ** 2)))
            row["meas_theta_rms_deg"], row["pred_theta_rms_deg"] = meas_rms, pr["theta_rms_deg"]
            row["k_theta_rms"] = meas_rms / pr["theta_rms_deg"]
            row["k_theta_rms_literal"] = meas_rms / prl["theta_rms_deg"]
            rows.append(row)
    df = pd.DataFrame(rows)
    d = np.abs(df["meas_centroid_3d_mm"] - df["stored_sd_3d_mm"])
    checks["centroid_sd_vs_ref_trueness"] = {
        "n_plateaus": int(len(df)), "n_matched": int(np.isfinite(d).sum()),
        "max_abs_mm": float(np.nanmax(d)), "ok": bool(np.isfinite(d).all()
                                                      and np.nanmax(d) <= BENCH_SD_MAX_MM)}
    checks["roundtrip_max_deg"] = float(df["roundtrip_max_deg"].max())
    checks["ok"] = bool(checks["centroid_sd_vs_ref_trueness"]["ok"]
                        and all(s["model_identical"] and s["all_reprojected_inside_images"]
                                and s["min_depth_cam1_mm"] > 0
                                for s in checks["sessions"].values()))
    return df, checks


# -- assembly ---------------------------------------------------------------------------------
def _row(pid: str, array: str, quantity: str, axis: str, point: str, value: float,
         unit: str, source: str, variant: str = "computed", **kw: Any) -> dict[str, Any]:
    return {"pid": pid, "take": TAKES.get(pid, pid), "array": array, "quantity": quantity,
            "axis": axis, "point": point, "value": float(value), "unit": unit,
            "source": source, "variant": variant, **kw}


def noise_rows(pid: str, noise: dict[str, dict[str, Any]], source: str, variant: str
               ) -> list[dict[str, Any]]:
    rows = []
    for a, nb in noise.items():
        for i, ax in enumerate(AXES):
            rows.append(_row(pid, a, "sigma_theta", ax, "", nb["theta_deg"][i], "deg", source,
                             variant))
        th = np.asarray(nb["theta_deg"], float)
        rows.append(_row(pid, a, "sigma_theta", "rms", "", np.sqrt(np.mean(th ** 2)), "deg",
                         source, variant))
        rows.append(_row(pid, a, "sigma_theta", "3d", "", np.sqrt(np.sum(th ** 2)), "deg",
                         source, variant))
        rows.append(_row(pid, a, "sigma_eps", "3d", "", nb["eps_mm"], "mm", source, variant))
    return rows


def prop_rows(pid: str, noise: dict[str, dict[str, Any]], r: dict[str, dict[str, np.ndarray]],
              variant: str, points: tuple[str, ...] = ("incisor", "condyle")
              ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, out = [], {}
    for pt in points:
        pr = propagate(noise, {a: r[a][pt] for a in noise})
        out[pt] = pr
        rows += [_row(pid, "both", "sigma_p", "3d", pt, pr["exact_mm"], "mm", "equation_exact",
                      variant),
                 _row(pid, "both", "sigma_p", "3d", pt, pr["isotropic_mm"], "mm",
                      "equation_isotropic", variant),
                 _row(pid, "CRA", "cra_share", "", pt, pr["cra_share_exact"], "1",
                      "equation_exact", variant),
                 _row(pid, "CRA", "cra_share", "", pt, pr["cra_share_isotropic"], "1",
                      "equation_isotropic", variant)]
    return rows, out


def mc_direct_rows(pid: str, m: dict[str, Any], variant: str) -> tuple[list, dict]:
    rows, out = [], {}
    for pt, v in m["points"].items():
        s = float(np.sqrt(v["d2_sum"][v["ok"]].sum() / v["n"][v["ok"]].sum()))
        out[pt] = s
        rows.append(_row(pid, "both", "sigma_p", "3d", pt, s, "mm", "mc_direct", variant))
    return rows, out


def main() -> None:
    global N_DRAWS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    ap.add_argument("--draws", type=int, default=N_DRAWS, help="smoke runs only")
    ap.add_argument("--no-bench", action="store_true", help="smoke runs only")
    a = ap.parse_args()
    N_DRAWS = a.draws
    from biomech.config import load_config
    from biomech.io.context import DataContext
    from biomech.qc import stereo_cal as sc

    t_start = time.perf_counter()
    cfg = load_config(str(STUDY)).with_snapshot(a.snapshot)
    ctx = DataContext(cfg)
    rows: list[dict[str, Any]] = []
    frame_rows: list[pd.DataFrame] = []
    corner_tabs: list[pd.DataFrame] = []
    meta: dict[str, Any] = {"snapshot": a.snapshot, "plan": "analysis_plan.md VD29 + amendment (5658e462) + addendum 2 (a19fbc92)",
                            "settings": {"n_frames_max": N_FRAMES, "n_draws": N_DRAWS,
                                         "seed": SEED, "repro_max_deg": REPRO_MAX_DEG,
                                         "repro_max_mm": REPRO_MAX_MM, "fit_dof": FIT_DOF,
                                         "bench_sigma_px": "sqrt(2) x RMS of the stored "
                                         "per-corner reprojection_rms_px (dof 1 of 4: same "
                                         "correction as in vivo); k_literal from a second "
                                         "Monte Carlo with the stored RMS as sigma_px",
                                         "tag_mm": TAG_MM, "tag_whatif_mm": TAG_WHATIF_MM,
                                         "k_band": list(K_BAND), "exposure_s": EXPOSURE_S,
                                         "trim_s": TRIM_S, "optical_yaml": str(
                                             OPTICAL_YAML.relative_to(_pins.ROOT)),
                                         "fit": "trimmed Kabsch of the live tracker "
                                         "(amendment VD29 / VD30, commit 5658e462)",
                                         "trim_constants": _make_trimfit().constants,
                                         "seeds": "in vivo: default_rng(0) per take for the "
                                         "primary MC and again for each what-if MC; bench: "
                                         "one default_rng(0) over the sessions in sorted "
                                         "order (primary) and a second default_rng(0) for the "
                                         "literal-sigma MC"},
                            "takes": {}, "checks": {}}
    model, _ = sc.marker_model(OPTICAL_YAML)

    # bench first: k ratios
    k_med: dict[str, float] = {}
    if not a.no_bench:
        tb = time.perf_counter()
        slots = ctx.table("ref_trueness", "slots")
        prec = ctx.table("ref_trueness", "precision")
        bdf, bchk = bench(Path(cfg.data_dir), model, SEED, slots[slots.system == "jtsmm"],
                          prec[prec.system == "jtsmm"])
        meta["checks"]["bench"] = bchk
        meta["runtime_bench_s"] = time.perf_counter() - tb
        if not bchk["ok"]:
            raise SystemExit(f"bench reproduction failed: {json.dumps(bchk, indent=1)}")
        for q in ("theta_x", "theta_y", "theta_z", "theta_rms", "centroid"):
            k_med[q] = float(np.median(bdf[f"k_{q}"]))
        for q, axis, unit in (("theta_x", "x", "deg"), ("theta_y", "y", "deg"),
                              ("theta_z", "z", "deg"), ("theta_rms", "rms", "deg"),
                              ("centroid", "3d", "mm")):
            qty = "sigma_theta" if q.startswith("theta") else "sigma_eps"
            rows.append(_row("bench", "eval", f"k_{qty}", axis, "", k_med[q], "1", "bench",
                             n_plateaus=len(bdf),
                             iqr_lo=float(np.percentile(bdf[f"k_{q}"], 25)),
                             iqr_hi=float(np.percentile(bdf[f"k_{q}"], 75))))
            rows.append(_row("bench", "eval", f"k_{qty}_literal", axis, "",
                             float(np.median(bdf[f"k_{q}_literal"])), "1", "bench"))
            rows.append(_row("bench", "eval", f"meas_{qty}", axis, "",
                             float(np.median(bdf[f"meas_{q}_deg" if qty == "sigma_theta"
                                                 else "meas_centroid_3d_mm"])), unit, "bench"))
            rows.append(_row("bench", "eval", f"pred_{qty}", axis, "",
                             float(np.median(bdf[f"pred_{q}_deg" if qty == "sigma_theta"
                                                 else "pred_centroid_3d_mm"])), unit, "bench"))
        rows.append(_row("bench", "eval", "sigma_px", "", "", float(np.median(bdf["sigma_px"])),
                         "px", "bench"))
        meta["bench_k_median"] = k_med
        meta["bench_quote"] = {q: ("bench_calibrated" if not (K_BAND[0] <= v <= K_BAND[1])
                                   else "computed") for q, v in k_med.items()}

    for pid in ctx.pids():
        t0 = time.perf_counter()
        tc = take_context(ctx, pid)
        rf = refit_take(tc)
        rep = reproduction(tc, rf)
        meta["checks"][f"{pid}_own_frame_chain"] = tc["chain"]
        meta["checks"][f"{pid}_refit_vs_stored"] = rep
        if not all(v["ok"] for v in rep.values()):
            raise SystemExit(f"{pid}: refit does not reproduce the stored poses: {rep}")
        spx = sigma_px(tc, rf)
        cdf = corner_offsets(tc, rf)
        corner_tabs.append(cdf)
        for c in cdf.itertuples():
            rows.append(_row(pid, c.array, "corner_offset", "norm", f"tag{c.tag}_c{c.corner}",
                             c.offset_mm, "mm", "refit", n=c.n_frames))
            rows.append(_row(pid, c.array, "corner_trimmed_share", "", f"tag{c.tag}_c{c.corner}",
                             c.trimmed_share, "1", "refit", n=c.n_frames))
        lev = lever_arms(tc)
        for arr_, d in lev.items():
            for pt, v in d.items():
                rows.append(_row(pid, arr_, "lever_arm", "norm", pt, np.linalg.norm(v["r"]),
                                 "mm", "rest", n=v["n_rest_frames"]))
                for i, ax in enumerate(AXES):
                    rows.append(_row(pid, arr_, "lever_arm", ax, pt, v["r"][i], "mm", "rest"))
        for arr_, v in spx.items():
            rows.append(_row(pid, arr_, "sigma_px", "", "", v["sigma_px"], "px", "refit",
                             n=v["n_frames"]))
            rows.append(_row(pid, arr_, "rms_reproj_raw", "", "", v["rms_raw_px"], "px",
                             "refit", n=v["n_frames"]))
        mc = invivo(tc, rf, spx, SEED)
        r = {arr_: {pt: lev[arr_][pt]["r"] for pt in tc["points_mp"]} for arr_ in ARRAYS}
        summ = {}
        for variant, m in (("computed", mc["primary"]),
                           ("whatif_tag30_synthetic", mc["whatif_tag30_synthetic"]),
                           ("whatif_tag50", mc["whatif_tag50"])):
            noise = {arr_: mc_summary(m, arr_) for arr_ in ARRAYS}
            rows += noise_rows(pid, noise, "mc", variant)
            pr_rows, pr = prop_rows(pid, noise, r, variant)
            rows += pr_rows
            md_rows, md = mc_direct_rows(pid, m, variant)
            rows += md_rows
            summ[variant] = {"noise": noise, "prop": pr, "mc_direct": md}
        noise_c = summ["computed"]["noise"]
        # what-if (b): r_CRA to the incisor halved, equation only
        r_half = {"CRA": {"incisor": 0.5 * r["CRA"]["incisor"]}, "MTA": {"incisor":
                                                                        r["MTA"]["incisor"]}}
        wb_rows, wb = prop_rows(pid, noise_c, r_half, "whatif_lever_half", ("incisor",))
        rows += wb_rows
        summ["whatif_lever_half"] = {"prop": wb}
        if k_med:
            cal_noise = {arr_: {"theta_deg": np.asarray(nb["theta_deg"]) * np.array(
                [k_med["theta_x"], k_med["theta_y"], k_med["theta_z"]]),
                "eps_mm": nb["eps_mm"] * k_med["centroid"]} for arr_, nb in noise_c.items()}
            rows += noise_rows(pid, cal_noise, "mc", "bench_calibrated")
            pc_rows, pc = prop_rows(pid, cal_noise, r, "bench_calibrated")
            rows += pc_rows
            wbc_rows, wbc = prop_rows(pid, cal_noise, r_half, "whatif_lever_half_bench_calibrated",
                                      ("incisor",))
            rows += wbc_rows
            summ["bench_calibrated"] = {"noise": cal_noise, "prop": pc}
            summ["whatif_lever_half_bench_calibrated"] = {"prop": wbc}
        # addendum 2: in-vivo upper bound from the stored-pose jitter (rest), movement beside it
        for variant, mask in (("rest_jitter", tc["rest"]), ("movement_jitter", tc["mov"])):
            nj = pose_jitter(tc, mask)
            for row in noise_rows(pid, nj, "jitter", variant):
                row["n"] = nj[row["array"]]["n_frames"]
                rows.append(row)
            pj_rows, pj = prop_rows(pid, nj, r, variant)
            rows += pj_rows
            summ[variant] = {"noise": nj, "prop": pj}
            if variant == "rest_jitter":
                wj_rows, wj = prop_rows(pid, nj, r_half, "rest_jitter_lever_half", ("incisor",))
                rows += wj_rows
                summ["rest_jitter_lever_half"] = {"prop": wj}
        # addendum 2: relative reductions of sigma_p (scale-independent)
        for pt in ("incisor", "condyle"):
            for src, key in (("mc_direct", "mc_direct"), ("equation_exact", "prop")):
                ctl = summ["whatif_tag30_synthetic"][key][pt]
                big = summ["whatif_tag50"][key][pt]
                if key == "prop":
                    ctl, big = ctl["exact_mm"], big["exact_mm"]
                rows.append(_row(pid, "both", "reduction", "3d", pt, 1.0 - big / ctl,
                                 "fraction", src, "whatif_tag50",
                                 note="50-mm tags against the 30-mm synthetic control"))
        for scale, base_key, half_key in (
                ("computed", "computed", "whatif_lever_half"),
                ("bench_calibrated", "bench_calibrated", "whatif_lever_half_bench_calibrated"),
                ("rest_jitter", "rest_jitter", "rest_jitter_lever_half")):
            if base_key in summ and half_key in summ:
                b0 = summ[base_key]["prop"]["incisor"]["exact_mm"]
                b1 = summ[half_key]["prop"]["incisor"]["exact_mm"]
                rows.append(_row(pid, "both", "reduction", "3d", "incisor", 1.0 - b1 / b0,
                                 "fraction", "equation_exact",
                                 "whatif_lever_half" if scale == "computed"
                                 else f"whatif_lever_half_{scale}",
                                 note=f"r_CRA to the incisor halved, noise scale {scale}"))
        # per frame (VD30)
        prim = mc["primary"]
        for arr_ in ARRAYS:
            ok = prim[arr_]["ok"]
            th = np.degrees(np.sqrt(prim[arr_]["rv2"]))
            frame_rows.append(pd.DataFrame({
                "pid": pid, "take": TAKES[pid], "array": arr_, "optical_row": mc["rows"][ok],
                "t_master_s": tc["tS"][mc["rows"][ok]],
                "sigma_theta_x_deg": th[ok, 0], "sigma_theta_y_deg": th[ok, 1],
                "sigma_theta_z_deg": th[ok, 2],
                "sigma_theta_3d_deg": np.sqrt(np.sum(th[ok] ** 2, axis=1)),
                "sigma_eps_mm": np.sqrt(prim[arr_]["eps2"][ok]),
                "n_draws_accepted": prim[arr_]["n"][ok].astype(int)}))
        meta["takes"][pid] = {
            "take": TAKES[pid], "n_movement_frames": int(tc["mov"].sum()),
            "n_rest_frames": int(tc["rest"].sum()), "n_mc_frames": int(len(mc["rows"])),
            "sigma_px": spx, "lever_arms_mm": {arr_: {pt: {"r_body": v["r"].tolist(),
                                                           "norm": float(np.linalg.norm(v["r"])),
                                                           "norm_median": v["norm_median_mm"]}
                                                      for pt, v in d.items()}
                                               for arr_, d in lev.items()},
            "points_mp_mm": {k: v.tolist() for k, v in tc["points_mp"].items()},
            "summary": json.loads(json.dumps(summ, default=lambda o: np.asarray(o).tolist())),
            "runtime_s": time.perf_counter() - t0}
    ctx.close()

    # context: static marking precision of the CT tip (a different object, named as such)
    nj = Path(cfg.out_dir) / "paper_systems" / "numbers.json"
    if nj.is_file():
        nums = json.loads(nj.read_text()).get("numbers", {})
        meta["context_numMarkPrecJtsmm"] = {
            v.get("macro", k): {"value_mm": v.get("value"), "what": v.get("what"),
                                "source": v.get("source")}
            for k, v in nums.items() if k.startswith("markPrecJtsmm")}

    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "software", "scripts"],
                           capture_output=True, text=True, cwd=_pins.ROOT).stdout.strip()
    tag = a.snapshot + ("_smoke" if (a.draws != 200 or a.no_bench) else "")
    files = {"main": out_dir / f"vd29_array_noise_{tag}.parquet",
             "frames": out_dir / f"vd29_array_noise_frames_{tag}.parquet"}
    pd.DataFrame(rows).assign(snapshot=a.snapshot).to_parquet(files["main"])
    pd.concat(frame_rows, ignore_index=True).to_parquet(files["frames"])
    files["corners"] = out_dir / f"vd29_array_noise_corners_{tag}.parquet"
    pd.concat(corner_tabs, ignore_index=True).to_parquet(files["corners"])
    if not a.no_bench:
        files["bench"] = out_dir / f"vd29_array_noise_bench_{tag}.parquet"
        bdf.to_parquet(files["bench"])
    meta.update({"git_sha": sha, "git_dirty_paths": dirty.splitlines(),
                 "runtime_s": time.perf_counter() - t_start,
                 "files": {k: v.name for k, v in files.items()},
                 "sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest()
                            for k, v in files.items()}})
    files["main"].with_suffix(".json").write_text(
        json.dumps(meta, indent=1, default=lambda o: np.asarray(o).tolist()))
    pd.set_option("display.width", 220)
    df = pd.DataFrame(rows)
    print(df[df.variant.isin(["computed"])].to_string(
        index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {files['main']} ({meta['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
