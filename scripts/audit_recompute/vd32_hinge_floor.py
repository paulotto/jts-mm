# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Validation plan VD32 (R5; J30): noise floor of the hinge residuals
on Qualisys (post hoc, descriptive; definition fixed in ``experiments/jtsmm_validation/
analysis_plan.md`` VD32 before this ran).

Per take (P1 = s10, P2 = s14) the ``ref_hinge`` openings of ``open_close`` (same cycles,
frames and three points per frame: incisor, condyle L, condyle R in the Qualisys maxillary
frame). For each model -- ``fit`` (best-fit fixed axis, ``fit_hinge``) and ``tmj`` (axis
through the condyle points, ``hinge_through``) -- the observed Qualisys trajectory is replaced
by its noise-free model reconstruction: the jaw pose of each frame is the opening's closed
(start) pose rotated about the fitted axis by the fitted angle. The Qualisys noise is then
added by the an14 §5.4 Monte Carlo: N(0, σ̂_m²) on each marker of the HP and MP templates
(σ̂_m of the take's ``open_close`` segment, ``ref_quality:consistency``), Kabsch refits giving
ΔH, ΔJ, and M = F_max⁻¹ ΔH⁻¹ (T_HP⁻¹ T_MP) ΔJ F_mand; the three points are recomputed and the
same fit is repeated (initial values and condyle axis from the noisy closed poses, exactly as
``validation/hinge.run_ref_hinge``). 1000 draws, seed 0. Floor = P50 / P95 / P99 of the
incisor RMS residual over the draws; ratio = observed / P99, above = observed > P99 (the
dataset's C1 convention).

Noise per frame: every Qualisys pose that enters the fit (each opening's closed pose and each
picked frame) gets its own independent ΔH, ΔJ in every draw. (A ΔH, ΔJ shared by all frames
of a draw, as in ``quality.sigma_ref`` where only per-frame RMS matters, maps a pure hinge onto
a pure hinge about a moved axis and leaves the residual at zero; the JSON records that check.)

Reproduction first: the stored ``ref_hinge:hinge`` ``hinge_fit_mm`` and ``hinge_tmj_mm``
(qualisys, open_close) of both takes are recomputed from the same openings (to 1e-6 mm), the
stored ``ref_quality:sigma_ref`` of ``open_close`` from the re-derived templates and frames
(to 1e-9 mm), and the jaw poses rebuilt from the three stored points (max point error 1e-9 mm);
the script stops otherwise. Synthetic checks on the noise-free reconstruction: σ̂_m = 0 gives a
refit residual ~0 (< 1e-3 mm); the floor scales linearly with σ̂_m (factors 0.25, 0.5, 2).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/vd32_hinge_floor.py

Output: ``<validation store>/audit/vd32_hinge_floor_<snap>.parquet`` (pid x model) and
``.json`` (settings, seed, reproduction values, synthetic checks, git commit, runtime).
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
TASK, SEGMENT, PIDS = "open_close", "open_close", ("s10", "s14")
MODELS = ("fit", "tmj")
DRAWS, SEED = 1000, 0
SCALES = (0.25, 0.5, 2.0)          # linearity check of the floor in σ̂_m
CHECK_DRAWS = 200                  # draws of the synthetic checks (same seed)
TOL_REPRO_MM, TOL_SIGMA_MM, TOL_POSE_MM, TOL_ZERO_MM = 1e-6, 1e-9, 1e-9, 1e-3


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), None, snapshot)


def fit_models(A: np.ndarray, B: np.ndarray) -> dict[str, Any]:
    """The two fits of ``run_ref_hinge`` (same initial values, same condyle axis)."""
    from biomech.kinematics.hinge_fit import fit_hinge, hinge_through

    tl, tr = A[:, 1].mean(axis=0), A[:, 2].mean(axis=0)
    mid = 0.5 * (tl + tr)
    return {"fit": fit_hinge(A, B, s0=mid, n0=tl - tr), "tmj": hinge_through(A, B, tl, tr)}


def opening_index(P: np.ndarray, cycles: pd.DataFrame, max_frames: int
                  ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Frame indices behind ``hinge.opening_points`` (same rule): per row of (A, B) the closed
    (start) frame and the picked frame; and the start frame of each used opening."""
    starts, picks, used = [], [], []
    for r in cycles[cycles.task == TASK].itertuples():
        idx = np.arange(int(r.start), int(r.peak) + 1)
        ok = idx[np.isfinite(P[idx]).all(axis=(1, 2))]
        if ok.size < 4 or ok[0] != int(r.start):
            continue
        pick = ok[np.unique(np.linspace(1, ok.size - 1, min(max_frames, ok.size - 1)).astype(int))]
        starts.append(np.full(pick.size, ok[0]))
        picks.append(pick)
        used.append(int(ok[0]))
    return np.concatenate(starts), np.concatenate(picks), used


def kabsch1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rigid T (4x4) with b ≈ R a + t for (K, 3) point sets (one fit)."""
    from biomech.validation.quality import _kabsch

    return _kabsch(a[None], b[None])[0]


def axis_rotation(n: np.ndarray, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """(F, 4, 4): x -> R(n, θ)(x - s) + s (Rodrigues, same convention as ``hinge_fit.rotate``)."""
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    c, si = np.cos(theta)[:, None, None], np.sin(theta)[:, None, None]
    R = np.eye(3)[None] * c + K[None] * si + np.outer(n, n)[None] * (1 - c)
    T = np.repeat(np.eye(4)[None], len(theta), axis=0)
    T[:, :3, :3] = R
    T[:, :3, 3] = s - np.einsum("fij,j->fi", R, s)
    return T


class Noise:
    """§5.4 pose errors ΔH, ΔJ (template -> noisy template, Kabsch), drawn per draw and pose."""

    def __init__(self, tmpl_h: np.ndarray, tmpl_j: np.ndarray) -> None:
        self.th, self.tj = tmpl_h, tmpl_j

    def draw(self, rng: np.random.Generator, draws: int, n_pose: int, sig_h: float,
             sig_j: float) -> tuple[np.ndarray, np.ndarray]:
        from biomech.validation.quality import _kabsch

        out = []
        for tm, s in ((self.th, sig_h), (self.tj, sig_j)):
            N = draws * n_pose
            noisy = tm[None] + rng.normal(0.0, s, (N,) + tm.shape)
            out.append(_kabsch(np.repeat(tm[None], N, axis=0), noisy).reshape(draws, n_pose, 4, 4))
        return out[0], out[1]


def perturb(M: np.ndarray, dH: np.ndarray, dJ: np.ndarray, F_max: np.ndarray,
            F_mand: np.ndarray) -> np.ndarray:
    """M (P, 4, 4) jaw in maxillary frame -> (D, P, 4, 4) F_max⁻¹ ΔH⁻¹ G ΔJ F_mand with
    G = F_max M F_mand⁻¹ (= T_HP⁻¹ T_MP)."""
    from biomech.kinematics import se3

    G = np.einsum("ij,pjk,kl->pil", F_max, M, se3.inv(F_mand))
    return np.einsum("ij,dpjk,pkl,dplm,mn->dpin", se3.inv(F_max), se3.inv(dH), G, dJ, F_mand)


def points(M: np.ndarray, pm: np.ndarray) -> np.ndarray:
    """(..., 4, 4) poses, (3, 3) mandibular-frame points -> (..., 3, 3) maxillary-frame points."""
    return np.einsum("...ij,kj->...ki", M[..., :3, :3], pm) + M[..., None, :3, 3]


def floor_run(M_start: np.ndarray, open_of_row: np.ndarray, M_row: dict[str, np.ndarray],
              pm: np.ndarray, noise: Noise, F_max: np.ndarray, F_mand: np.ndarray,
              sig_h: float, sig_j: float, draws: int, seed: int, shared: bool = False
              ) -> dict[str, np.ndarray]:
    """Residual (incisor RMS) of each model's refit per draw; noise shared by both models
    (common random numbers). ``shared``: one ΔH, ΔJ per draw for all poses (check only).
    Also the per-frame incisor displacement RMS of the noise on the model poses (``sigma``)."""
    rng = np.random.default_rng(seed)
    n_open, n_row = len(M_start), len(open_of_row)
    n_pose = 1 if shared else n_open + n_row
    dH, dJ = noise.draw(rng, draws, n_pose, sig_h, sig_j)
    if shared:
        dH = np.repeat(dH, n_open + n_row, axis=1)
        dJ = np.repeat(dJ, n_open + n_row, axis=1)
    out: dict[str, np.ndarray] = {}
    for m, Mr in M_row.items():
        Mp = perturb(np.concatenate([M_start, Mr]), dH, dJ, F_max, F_mand)
        P = points(Mp, pm)                                         # (D, n_open + n_row, 3, 3)
        A = P[:, :n_open][:, open_of_row]
        B = P[:, n_open:]
        out[m] = np.array([fit_models(np.ascontiguousarray(A[d]), B[d])[m].rms(0)
                           for d in range(draws)])
        e = np.linalg.norm(Mp[:, n_open:, :3, 3] - Mr[None, :, :3, 3], axis=-1)
        out[f"sigma_{m}"] = np.sqrt(np.mean(e ** 2))
    return out


def participant(cfg: Any, pid: str, draws: int = DRAWS,
                scale: tuple[float, float] = (1.0, 1.0)) -> dict[str, Any]:
    from biomech.io.context import DataContext
    from biomech.kinematics import se3
    from biomech.kinematics.hinge_fit import rotate
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.validation import anatomy as AN
    from biomech.validation import quality as QY
    from biomech.validation.evaluate import BODIES, holds_from_table, segment_list
    from biomech.validation.hinge import opening_points
    from biomech.validation.stages import _systems

    t0 = time.perf_counter()
    dc = DataContext(cfg)
    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    try:
        ctx = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("ref_quality"),
                                 store=dc.store, participant=part, cache=cache)
        arr = ctx.upstream_arrays("ref_kinematics")
        cycles = ctx.upstream("ref_agreement", "cycles")
        stored = ctx.upstream("ref_hinge", "hinge")
        stored = stored[(stored.task == TASK) & (stored.system == "qualisys")].iloc[0]
        cons = ctx.upstream("ref_quality", "consistency").set_index(["body", "segment"])
        sr = ctx.upstream("ref_quality", "sigma_ref").set_index("segment").loc[SEGMENT]
        mf = int(cfg.params("kinematics")["hinge"]["max_frames_per_cycle"])

        # -- openings and reproduction of the stored fits ------------------------------------
        A, B, used = opening_points(arr, cycles, TASK, "q", mf)
        P = np.stack([arr["incisor_q_mm"], arr["tmj_left_q_mm"], arr["tmj_right_q_mm"]], axis=1)
        i0, i1, starts = opening_index(P, cycles, mf)
        if not (np.array_equal(P[i0], A) and np.array_equal(P[i1], B) and len(starts) == used):
            raise SystemExit(f"{pid}: opening index does not match opening_points")
        obs = fit_models(A, B)
        repro = {"n_openings": used, "n_frames": int(len(A)),
                 "stored_n_openings": int(stored.n_openings),
                 "stored_n_frames": int(stored.n_frames)}
        for m, col in (("fit", "hinge_fit_mm"), ("tmj", "hinge_tmj_mm")):
            repro[f"{m}_mm"] = obs[m].rms(0)
            repro[f"stored_{m}_mm"] = float(stored[col])
            repro[f"{m}_diff_mm"] = abs(obs[m].rms(0) - float(stored[col]))
        repro["equal"] = bool(max(repro["fit_diff_mm"], repro["tmj_diff_mm"]) <= TOL_REPRO_MM
                              and used == repro["stored_n_openings"]
                              and len(A) == repro["stored_n_frames"])

        # -- Qualisys frames, templates, σ̂_m (the ref_quality inputs) ------------------------
        sy, *_ = _systems(ctx)
        fr = AN.anatomical_frames(holds_from_table(ctx.upstream("ref_tip", "landmarks"),
                                                   "qualisys", "primary"))
        tm = {b: QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers(b)},
                             sy.qtm.pose(b)) for b in BODIES}
        sig_h = float(cons.at[("HP", SEGMENT), "sigma_marker_mm"])
        sig_j = float(cons.at[("MP", SEGMENT), "sigma_marker_mm"])
        iv = [iv for seg, _, iv in segment_list(sy) if seg == SEGMENT][0]
        k = (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
        rel = se3.relative(sy.qtm.pose("HP"), sy.qtm.pose("MP"))
        sref = QY.sigma_ref(tm["HP"], tm["MP"], sig_h, sig_j, rel[k], fr.F_max, fr.F_mand,
                            fr.tmj_mand)
        repro["sigma_ref_incisor_mm"] = sref["incisor_mm"]
        repro["stored_sigma_ref_incisor_mm"] = float(sr["sigma_ref_incisor_mm"])
        repro["sigma_ref_equal"] = bool(
            abs(sref["incisor_mm"] - float(sr["sigma_ref_incisor_mm"])) <= TOL_SIGMA_MM
            and np.isclose(sig_h, float(sr["sigma_head_mm"]), rtol=0, atol=1e-12)
            and np.isclose(sig_j, float(sr["sigma_jaw_mm"]), rtol=0, atol=1e-12))

        # -- jaw poses of the closed frames from the three points (incisor = F_mand origin) ---
        pm = np.stack([np.zeros(3), fr.tmj_mand["tmj_left"], fr.tmj_mand["tmj_right"]])
        M_start = np.stack([kabsch1(pm, P[f]) for f in starts])
        pose_err = float(np.abs(points(M_start, pm) - P[starts]).max())
        repro["pose_points_max_err_mm"] = pose_err
        open_of_row = np.searchsorted(np.array(starts), i0)
        if pose_err > TOL_POSE_MM or not np.array_equal(np.array(starts)[open_of_row], i0):
            repro["equal"] = False
        if not (repro["equal"] and repro["sigma_ref_equal"]):
            raise SystemExit(f"{pid}: reproduction failed: {json.dumps(repro, indent=1)}")

        # -- noise-free reconstructions: closed pose rotated about the fitted axis ------------
        M_row = {m: np.einsum("fij,fjk->fik",
                              axis_rotation(obs[m].n, obs[m].s, np.radians(obs[m].theta_deg)),
                              M_start[open_of_row]) for m in MODELS}
        recon = {}
        for m in MODELS:
            # the reconstruction's points are the fit's own prediction (A rotated about the axis)
            Bh = points(M_row[m], pm)
            pred = rotate(A - obs[m].s, obs[m].n, np.radians(obs[m].theta_deg)) + obs[m].s
            recon[m] = {"recon_vs_model_prediction_max_mm": float(np.abs(Bh - pred).max()),
                        "recon_vs_observed_incisor_rms_mm": float(np.sqrt(np.mean(
                            np.linalg.norm(Bh[:, 0] - B[:, 0], axis=-1) ** 2)))}
            if recon[m]["recon_vs_model_prediction_max_mm"] > 1e-9:
                raise SystemExit(f"{pid}: reconstruction of {m} is not the model prediction")
        noise = Noise(tm["HP"], tm["MP"])
        args = (M_start, open_of_row, M_row, pm, noise, fr.F_max, fr.F_mand)

        # -- synthetic checks ----------------------------------------------------------------
        chk: dict[str, Any] = {"recon": recon}
        z = floor_run(*args, 0.0, 0.0, 3, SEED)
        chk["zero_noise_max_mm"] = {m: float(z[m].max()) for m in MODELS}
        base = floor_run(*args, sig_h, sig_j, CHECK_DRAWS, SEED)
        lin = {}
        for c in SCALES:
            r = floor_run(*args, c * sig_h, c * sig_j, CHECK_DRAWS, SEED)
            lin[str(c)] = {m: {"p50_ratio": float(np.median(r[m]) / np.median(base[m])),
                               "p99_ratio": float(np.percentile(r[m], 99)
                                                  / np.percentile(base[m], 99))}
                           for m in MODELS}
        chk["linearity"] = lin
        sh = floor_run(*args, sig_h, sig_j, 20, SEED, shared=True)
        chk["shared_noise_per_draw_p50_mm"] = {m: float(np.median(sh[m])) for m in MODELS}
        chk["zero_ok"] = bool(max(chk["zero_noise_max_mm"].values()) < TOL_ZERO_MM)
        chk["linear_ok"] = bool(all(abs(v[m]["p50_ratio"] / float(c) - 1) < 0.1
                                    for c, v in lin.items() for m in MODELS))
        if not chk["zero_ok"]:
            raise SystemExit(f"{pid}: zero-noise refit residual not ~0: {chk}")

        # -- the floor (VD36: marker noise scaled by the VD34 ratio of each body) -------------
        sig_h, sig_j = sig_h * scale[0], sig_j * scale[1]
        r = floor_run(*args, sig_h, sig_j, draws, SEED)
        rows = []
        for m in MODELS:
            d = r[m]
            p50, p95, p99 = (float(np.percentile(d, q)) for q in (50, 95, 99))
            o = obs[m].rms(0)
            rows.append({"pid": pid, "model": m, "observed_mm": o, "floor_p50_mm": p50,
                         "floor_p95_mm": p95, "floor_p99_mm": p99, "ratio_p99": o / p99,
                         "above": bool(o > p99), "n_openings": used, "n_frames": int(len(A)),
                         "sigma_m_hp_mm": sig_h, "sigma_m_mp_mm": sig_j, "n_draws": int(d.size),
                         "noise_scale_hp": scale[0], "noise_scale_mp": scale[1]})
        extra = {"sigma_incisor_frames_mm": {m: float(r[f"sigma_{m}"]) for m in MODELS},
                 "share_draws_below_observed": {m: float(np.mean(r[m] < obs[m].rms(0)))
                                                for m in MODELS},
                 "sigma_ref_incisor_open_close_mm": float(sr["sigma_ref_incisor_mm"]),
                 "seconds": time.perf_counter() - t0}
        return {"rows": rows, "reproduction": repro, "checks": chk, "extra": extra}
    finally:
        cache.close()
        dc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    ap.add_argument("--pids", nargs="*", default=list(PIDS))
    ap.add_argument("--draws", type=int, default=DRAWS, help="timing tests only (output _partial)")
    ap.add_argument("--noise-scale", choices=["none", "vd34"], default="none",
                    help="vd34: scale each body's marker noise by its VD34 3-D ratio (VD36)")
    a = ap.parse_args()
    from biomech import sensitivity as S

    t0 = time.perf_counter()
    cfg = _config(a.snapshot)
    scales = {pid: (1.0, 1.0) for pid in a.pids}
    out_dir0 = S.primary_out_dir(cfg) / "audit"
    if a.noise_scale == "vd34":
        v = pd.read_parquet(out_dir0 / f"vd34_qualisys_noise_{a.snapshot}.parquet")
        v = v[(v.part == "rho") & (v.task == "median") & (v.axis_group == "3d")]
        scales = {pid: tuple(min(1.0, float(v[(v.pid == pid) & (v.body == b)].rho.iloc[0]))
                             for b in ("HP", "MP")) for pid in a.pids}
    res = {pid: participant(cfg, pid, a.draws, scales[pid]) for pid in a.pids}
    out = pd.DataFrame([r for v in res.values() for r in v["rows"]])
    out_dir = S.primary_out_dir(cfg) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    tag = a.snapshot + ("" if tuple(a.pids) == PIDS and a.draws == DRAWS else "_partial")
    name = "vd36_hinge_floor_scaled" if a.noise_scale == "vd34" else "vd32_hinge_floor"
    f = out_dir / f"{name}_{tag}.parquet"
    out.to_parquet(f)
    meta = {"plan": "experiments/jtsmm_validation/analysis_plan.md "
                    + ("VD36" if a.noise_scale == "vd34" else "VD32"),
            "noise_scale": {p: list(v) for p, v in scales.items()},
            "snapshot": a.snapshot, "git_sha": sha, "task": TASK, "segment": SEGMENT,
            "settings": {"draws": a.draws, "seed": SEED, "percentiles": [50, 95, 99],
                         "noise": "§5.4 Monte Carlo; independent ΔH, ΔJ per pose (each "
                                  "opening's closed pose and each picked frame) and draw; "
                                  "same draws for both models",
                         "max_frames_per_cycle": int(cfg.params("kinematics")["hinge"]
                                                     ["max_frames_per_cycle"]),
                         "check_draws": CHECK_DRAWS, "linearity_scales": list(SCALES)},
            "reproduction": {p: v["reproduction"] for p, v in res.items()},
            "synthetic_checks": {p: v["checks"] for p, v in res.items()},
            "extra": {p: v["extra"] for p, v in res.items()},
            "runtime_s": time.perf_counter() - t0,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    pd.set_option("display.width", 200)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(json.dumps({p: v["checks"] for p, v in res.items()}, indent=1))
    print(f"wrote {f} ({meta['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
