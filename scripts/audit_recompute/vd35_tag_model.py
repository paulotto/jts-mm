# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD35: tag-model sensitivity (validation analysis plan
``experiments/jtsmm_validation/analysis_plan.md``, entry VD34-VD35, J30; descriptive, post hoc, definition fixed in the plan before this ran).

The tag-to-body transforms of ``optical.yaml`` are nominal. Per take (P1 = s10, P2 = s14) and
body (HP = CRA, MP = MTA, CT):

* **Estimated model.** Every tag corner seen by both cameras is triangulated (the take's camera
  model; ``qc/stereo_cal``) and expressed in the body frame of the tracker's trimmed-Kabsch fit
  on the nominal model (``vd29_array_noise.TrimFit``, all frames of the take with a fit); the
  median of each corner over the take is the estimated corner position; one re-centring: the
  estimated corners are rigidly aligned onto the nominal ones (Kabsch), so the body frame stays
  the nominal frame in the least-squares sense (the CT tip calibration, given in that frame,
  stays valid). JTS-MM-internal only: no Qualisys information enters.
* **Refit.** All HP, MP and CT poses refitted with the trimmed fit, once with the nominal and
  once with the estimated model (CT: the tip pose, ``t = R tip + t`` with the take's
  ``ct_tip_body_mm``, as the stored ``pose_ct``); a frame is valid where the stored pose is
  valid and the refit exists (the tracker's frame set).
* **Chain.** The validation stages ``ref_sync`` -> ``ref_tip`` -> ``ref_quality`` ->
  ``ref_kinematics`` run in memory (stage functions unchanged; the session's optical group and
  the upstream tables are substituted in this script) on (``stored``) the stored poses,
  (``nominal``) the nominal-model refit and (``estimated``) the estimated-model refit. From
  each: the sync registration (X, Y_b), VD31 (b) the held-out head-plate difference split about
  u1 (Qualisys template) and about the JTS-MM HP tag baseline (of the model in use, mapped by
  Y_HP) with its constant (per-interval mean) share, VD31 (d) the aligned incisor offset and
  scatter, and the own-frame (``segments`` movement_pooled, without and with the gate) and
  aligned (``segments_aligned`` movement_pooled) pooled incisor RMS.

Reproduction: the ``stored`` chain reproduces the stored v14 tables (registration, held-out,
segments, segments_aligned; and the VD31 (b)/(d) values) to 1e-9, and the ``nominal`` chain
reproduces them within the VD31 tolerances (pooled held-out HP rotation 0.05 deg, aligned RMS
1 %), else the script stops. ``model`` in the output: ``nominal`` = nominal-model refit,
``estimated``; ``stored`` rows are the reproduction.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/vd35_tag_model.py [-j 3]

Output (``<validation store>/audit/``): ``vd35_tag_model_<snap>.parquet`` (tidy: pid, take,
model, quantity, component, value, unit), ``vd35_tag_model_corners_<snap>.parquet`` (estimated
corner model) and ``vd35_tag_model_<snap>.json`` (settings, checks, git commit, runtime,
sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)
import vd29_array_noise as V  # noqa: E402 -- trimmed fit, geometry helpers

STUDY = _pins.EXP / "jtsmm_validation" / "study.yaml"
TAKES = {"s10": "P1", "s14": "P2"}
BODIES = {"HP": "cra", "MP": "mta", "CT": "ct"}      # optical.yaml body -> pose_*
CHAIN = ("ref_sync", "ref_tip", "ref_quality", "ref_kinematics")
EXACT_TOL = 1e-9
POOLED_TOL_DEG = 0.05                                # VD31 checks
ALIGNED_TOL = 0.01
HP_TAGS = (1, 8)


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), None, snapshot)


# -- tag model ------------------------------------------------------------------------------
def kabsch_rigid(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with B ~ R A + t (A, B (N, 3))."""
    ca, cb = A.mean(axis=0), B.mean(axis=0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def take_corners(h5: Path) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    from biomech.qc import stereo_cal as sc

    cal = sc.session_calibration(h5)
    pairs = sc.corner_pairs(h5)
    n0 = sc._undistort(pairs.p0, cal.K0, cal.d0)
    n1 = sc._undistort(pairs.p1, cal.K1, cal.d1)
    X = sc.triangulate(cal, n0, n1).reshape(-1, 4, 3)
    return cal, pairs, X, np.unique(pairs.row)


def body_tags(model: dict[int, np.ndarray], ranges: dict, body: str) -> list[int]:
    return V.array_tags(model, ranges, body)


def corner_medians(pairs: Any, X: np.ndarray, frames: np.ndarray, tags: list[int],
                   model: dict[int, np.ndarray]) -> tuple[dict[int, np.ndarray], dict]:
    """Median of each corner over the take in the trimmed-fit body frame of ``model``."""
    f = V.TrimFit(pairs, frames, tags, model)
    R, t, ok = f.fit(X)
    sel = ok[f.f]
    src, fi, ci = f.src[sel], f.f[sel], f.c[sel]
    loc = np.einsum("nji,nkj->nki", R[fi], X[src] - t[fi][:, None, :])
    out, n = {}, {}
    for j, tag in enumerate(f.tags):
        m = ci == j
        out[tag] = np.median(loc[m], axis=0)
        n[tag] = int(m.sum())
    return out, {"n_frames_fit": int(ok.sum()), "n_obs": n}


def estimate_model(pairs: Any, X: np.ndarray, frames: np.ndarray,
                   nominal: dict[int, np.ndarray], ranges: dict
                   ) -> tuple[dict[int, np.ndarray], list[dict], dict]:
    est: dict[int, np.ndarray] = {}
    rows, info = [], {}
    for body in BODIES:
        tags = body_tags(nominal, ranges, body)
        med, meta = corner_medians(pairs, X, frames, tags, nominal)
        A = np.concatenate([med[t] for t in tags])
        B = np.concatenate([nominal[t] for t in tags])
        R, t = kabsch_rigid(A, B)                         # one re-centring onto nominal
        e = {tg: med[tg] @ R.T + t for tg in tags}
        # diagnostic: a second pass (refit with the estimate, medians, re-centre) -- not used
        med2, _ = corner_medians(pairs, X, frames, tags, {**nominal, **e})
        A2 = np.concatenate([med2[tg] for tg in tags])
        R2, t2 = kabsch_rigid(A2, B)
        e2 = {tg: med2[tg] @ R2.T + t2 for tg in tags}
        est.update(e)
        info[body] = {**meta, "recentre_rot_deg": float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(R).as_rotvec()))), "recentre_trans_mm": float(np.linalg.norm(t)),
            "second_pass_max_change_mm": float(max(np.max(np.linalg.norm(e2[tg] - e[tg], axis=1))
                                                   for tg in tags))}
        for tg in tags:
            for k in range(4):
                d = e[tg][k] - nominal[tg][k]
                rows.append({"body": body, "tag": int(tg), "corner": k,
                             "x_nominal_mm": nominal[tg][k][0], "y_nominal_mm": nominal[tg][k][1],
                             "z_nominal_mm": nominal[tg][k][2], "x_mm": e[tg][k][0],
                             "y_mm": e[tg][k][1], "z_mm": e[tg][k][2], "dx_mm": d[0],
                             "dy_mm": d[1], "dz_mm": d[2], "offset_mm": float(np.linalg.norm(d)),
                             "n_obs": meta["n_obs"][tg]})
    return est, rows, info


def refit_poses(h5: Path, cal: Any, pairs: Any, X: np.ndarray, frames: np.ndarray,
                model: dict[int, np.ndarray], ranges: dict, stored: dict[str, np.ndarray],
                tip: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Trimmed refit of every body; pose_* convention (CT = tip pose); stored validity."""
    from biomech.kinematics import se3

    out, info = {}, {}
    for body, key in BODIES.items():
        f = V.TrimFit(pairs, frames, body_tags(model, ranges, body), model)
        R, t, ok = f.fit(X)
        S = stored[key]
        P = np.full_like(S, np.nan)
        rows = frames[ok]
        tt = t[ok] + (np.einsum("fij,j->fi", R[ok], tip) if body == "CT" else 0.0)
        P[rows, :3, :3] = R[ok]
        P[rows, :3, 3] = tt
        P[rows, 3] = [0, 0, 0, 1]
        vs = se3.valid(S)
        P[~vs] = np.nan
        info[body] = {"stored_valid": int(vs.sum()), "refit_valid": int(se3.valid(P).sum()),
                      "stored_valid_without_refit": int((vs & ~se3.valid(P)).sum())}
        out[key] = P
    return out, info


# -- in-memory chain --------------------------------------------------------------------------
def chain(cfg: Any, dc: Any, pid: str, poses: dict[str, np.ndarray] | None
          ) -> tuple[dict[str, Any], Any]:
    from biomech.io.session import OpticalData
    from biomech.pipeline import ParticipantContext, UnitCache, get_stage

    @dataclass
    class Ctx(ParticipantContext):
        results: dict = field(default_factory=dict)

        def upstream(self, stage_name: str, table: str) -> pd.DataFrame:
            if stage_name in self.results:
                df = self.results[stage_name].tables[table].copy()
                if "unit" not in df.columns:
                    df.insert(0, "unit", self.pid)
                return df
            return super().upstream(stage_name, table)

        def upstream_arrays(self, stage_name: str) -> dict[str, Any]:
            if stage_name in self.results:
                return dict(self.results[stage_name].arrays)
            return super().upstream_arrays(stage_name)

    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    sd = cache.session("motion")
    opt = sd.optical()
    if poses is not None:
        sd._optical = OpticalData(t_us=opt.t_us, poses={**opt.poses, **poses},
                                  quality=opt.quality)
    results: dict[str, Any] = {}
    for st in CHAIN:
        ctx = Ctx(config=cfg, cohort=dc.cohort, params=cfg.params(st), store=dc.store,
                  participant=part, cache=cache, results=results)
        results[st] = get_stage(st).func(ctx)
    ctx = Ctx(config=cfg, cohort=dc.cohort, params=cfg.params("ref_sync"), store=dc.store,
              participant=part, cache=cache, results=results)
    return results, (ctx, cache)


def head_split(ctx: Any, results: dict[str, Any], xhat: np.ndarray) -> dict[str, float]:
    """VD31 (b) on the chain's own clock, X, Y_HP and poses (as vd31_head_cluster)."""
    from biomech.validation import quality as QY
    from biomech.validation import registration as REG
    from biomech.validation import selection as SEL
    from biomech.validation import sync as SC
    from biomech.validation.evaluate import pose_from_row
    from biomech.validation.stages import _systems

    sy, *_ = _systems(ctx)
    tm = QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers("HP")}, sy.qtm.pose("HP"))
    c = tm - tm.mean(axis=0)
    U = np.linalg.svd(c)[2]
    reg = results["ref_sync"].tables["registration"].set_index("transform")
    X, Y = pose_from_row(reg.loc["X"]), pose_from_row(reg.loc["Y_HP"])
    a = Y[:3, :3].T @ xhat
    spl = SC.PoseSpline(sy.qtm.t_s, sy.qtm.pose("HP"))
    per = []
    for iv in sy.held:
        if iv.name not in SEL.MOVEMENT_TASKS:
            continue
        sel = np.flatnonzero((sy.tS >= iv.t0) & (sy.tS <= iv.t1))
        tq = sy.model.to_q(sy.tS[sel])
        Q = spl(tq)
        Q[~spl.covered(tq, 0.0025)] = np.nan
        P = REG.apply(X, sy.S["HP"][sel], Y)
        _, ang = REG.pose_error(P, Q)
        good = np.isfinite(ang)
        D = np.einsum("nji,njk->nik", Q[good, :3, :3], P[good, :3, :3])
        per.append(np.degrees(Rotation.from_matrix(D).as_rotvec()))
    W = np.concatenate(per)
    const = np.concatenate([np.repeat(w.mean(axis=0)[None], len(w), axis=0) for w in per])

    def ms(w: np.ndarray, v: np.ndarray) -> float:
        return float(np.mean((w @ v) ** 2))

    tot = float(np.mean(np.sum(W ** 2, axis=1)))
    ctot = float(np.mean(np.sum(const ** 2, axis=1)))
    ang = float(np.degrees(np.arccos(np.clip(abs(U[0] @ a) / np.linalg.norm(a), 0, 1))))
    return {"total_deg": np.sqrt(tot), "u1_deg": np.sqrt(ms(W, U[0])),
            "u1_share": ms(W, U[0]) / tot, "baseline_deg": np.sqrt(ms(W, a)),
            "baseline_share": ms(W, a) / tot, "const_share": ctot / tot,
            "const_u1_share": ms(const, U[0]) / ctot, "const_baseline_share": ms(const, a) / ctot,
            "axis_angle_u1_baseline_deg": ang, "n": float(len(W))}


def summarize(ctx: Any, results: dict[str, Any], xhat: np.ndarray) -> dict[str, Any]:
    from biomech.validation import selection as SEL

    out: dict[str, Any] = {}
    reg = results["ref_sync"].tables["registration"].set_index("transform")
    for tr in reg.index:
        for c in ("rx_deg", "ry_deg", "rz_deg", "tx_mm", "ty_mm", "tz_mm"):
            if c in reg.columns:
                out[f"registration:{tr}:{c}"] = float(reg.loc[tr, c])
    sy = results["ref_sync"].tables["sync"].iloc[0]
    for c in ("offset_s", "alpha_ppm", "heldout_mp_pos_rms_mm", "heldout_hp_pos_rms_mm",
              "heldout_rel_pos_rms_mm", "heldout_rel_rot_rms_deg"):
        out[f"sync:{c}"] = float(sy[c])
    for k, v in head_split(ctx, results, xhat).items():
        out[f"heldout_hp:{k}"] = float(v)
    rk = results["ref_kinematics"].tables
    sa = rk["segments_aligned"]
    mv = sa[sa["segment"].isin(SEL.MOVEMENT_TASKS) & (sa["kind"] == "movement")]
    n = mv["n"].to_numpy(float)
    b2 = (mv[["incisor_bias_x_mm", "incisor_bias_y_mm", "incisor_bias_z_mm"]].to_numpy()
          ** 2).sum(axis=1)
    out["aligned:offset_mm"] = float(np.sqrt(np.sum(n * b2) / n.sum()))
    out["aligned:scatter_mm"] = float(np.sqrt(np.sum(n * mv["incisor_dyn_rms_mm"].to_numpy() ** 2)
                                              / n.sum()))
    out["aligned:pooled_incisor_rms_mm"] = float(
        sa[sa["segment"] == "movement_pooled"].iloc[0]["incisor_rms_mm"])
    sg = rk["segments"]
    for g in (False, True):
        r = sg[(sg["segment"] == "movement_pooled") & (sg["gated"] == g)].iloc[0]
        tag = "gated" if g else "ungated"
        out[f"own:pooled_incisor_rms_mm_{tag}"] = float(r["incisor_rms_mm"])
        out[f"own:pooled_rot_rms_deg_{tag}"] = float(r["rot_rms_deg"])
    return out


def compare_tables(new: pd.DataFrame, old: pd.DataFrame, keys: list[str]) -> float:
    """Max absolute difference over the shared numeric columns, rows matched on ``keys``."""
    old = old.drop(columns=[c for c in ("unit",) if c in old.columns])
    m = new.merge(old, on=keys, suffixes=("", "__st"))
    if len(m) != len(old) or len(m) != len(new):
        return float("inf")
    d = 0.0
    for c in new.columns:
        if c in keys or f"{c}__st" not in m.columns:
            continue
        a, b = m[c], m[f"{c}__st"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            x, y = a.to_numpy(float), b.to_numpy(float)
            both = np.isnan(x) & np.isnan(y)
            if (np.isnan(x) != np.isnan(y)).any():
                return float("inf")
            d = max(d, float(np.max(np.abs(x[~both] - y[~both]), initial=0.0)))
    return d


def stored_checks(dc: Any, pid: str, results: dict[str, Any]) -> dict[str, float]:
    rd = lambda st, tb: dc.store.read_table(st, tb, [pid])  # noqa: E731
    rk = results["ref_kinematics"].tables
    return {
        "registration": compare_tables(results["ref_sync"].tables["registration"],
                                       rd("ref_sync", "registration"), ["transform"]),
        "heldout": compare_tables(results["ref_sync"].tables["heldout"],
                                  rd("ref_sync", "heldout"), ["interval", "body"]),
        "segments": compare_tables(rk["segments"], rd("ref_kinematics", "segments"),
                                   ["segment", "gated"]),
        "segments_aligned": compare_tables(rk["segments_aligned"],
                                           rd("ref_kinematics", "segments_aligned"),
                                           ["segment"])}


def job(args: tuple[str, str, str]) -> dict[str, Any]:
    """One (take, arm): ``stored`` | ``nominal`` | ``estimated``; single-threaded BLAS as the
    pipeline's workers (``threadpool_limits(1)``)."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _job(args)


def _job(args: tuple[str, str, str]) -> dict[str, Any]:
    import h5py

    from biomech.io.context import DataContext
    from biomech.qc import stereo_cal as sc

    snapshot, pid, arm = args
    t0 = time.perf_counter()
    cfg = _config(snapshot)
    dc = DataContext(cfg)
    try:
        h5 = dc.session_path(pid) / "session.h5"
        nominal, ranges = sc.marker_model(V.OPTICAL_YAML)
        res: dict[str, Any] = {"pid": pid, "arm": arm}
        poses = None
        model = nominal
        if arm != "stored":
            cal, pairs, X, frames = take_corners(h5)
            with h5py.File(h5, "r") as f:
                stored = {k: np.asarray(f[f"optical/pose_{k}"][()], float)
                          for k in BODIES.values()}
                tip = np.array([float(x) for x in
                                str(f["meta"].attrs["ct_tip_body_mm"]).split(",")])
            if arm == "estimated":
                est, crow, info = estimate_model(pairs, X, frames, nominal, ranges)
                model = {**nominal, **est}
                res["corners"] = [{"pid": pid, "take": TAKES[pid], **r} for r in crow]
                res["estimate_info"] = info
            poses, pinfo = refit_poses(h5, cal, pairs, X, frames, model, ranges, stored, tip)
            res["refit_info"] = pinfo
        c = [model[t].mean(axis=0) for t in HP_TAGS]
        xhat = (c[1] - c[0]) / np.linalg.norm(c[1] - c[0])
        results, (ctx, cache) = chain(cfg, dc, pid, poses)
        try:
            res["values"] = summarize(ctx, results, xhat)
            res["xhat"] = xhat.tolist()
            if arm == "stored":
                res["table_checks"] = stored_checks(dc, pid, results)
        finally:
            cache.close()
        res["runtime_s"] = time.perf_counter() - t0
        return res
    finally:
        dc.close()


UNITS = {"_deg": "deg", "_mm": "mm", "_share": "fraction", "_s": "s", "_ppm": "ppm"}


def unit_of(name: str) -> str:
    for suf, u in UNITS.items():
        if name.endswith(suf):
            return u
    if "_mm_" in name:
        return "mm"
    if "_deg_" in name:
        return "deg"
    return "1"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    ap.add_argument("-j", "--jobs", type=int, default=6)
    ap.add_argument("--pids", nargs="*", default=list(TAKES))
    ap.add_argument("--arms", nargs="*", default=["stored", "nominal", "estimated"])
    a = ap.parse_args()
    t_start = time.perf_counter()
    cfg = _config(a.snapshot)
    jobs = [(a.snapshot, p, arm) for p in a.pids for arm in a.arms]
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        res = list(ex.map(job, jobs))
    R = {(r["pid"], r["arm"]): r for r in res}

    # reproduction against the stored v14 values (VD31 parquet for (b) / (d))
    v31 = pd.read_parquet(Path(cfg.out_dir) / "audit" / f"vd31_head_cluster_{a.snapshot}.parquet")
    checks: dict[str, Any] = {}
    for pid in a.pids:
        q = v31[v31.pid == pid].set_index(["part", "component", "task"])["value"]
        ref = {"heldout_hp:total_deg": q[("heldout", "total", "pooled")],
               "heldout_hp:u1_share": q[("heldout", "u1_share", "pooled")],
               "heldout_hp:baseline_share": q[("heldout", "jts_x_share", "pooled")],
               "heldout_hp:const_share": q[("heldout_const", "ms_share", "pooled")],
               "aligned:offset_mm": q[("aligned_rms", "offset", "")],
               "aligned:scatter_mm": q[("aligned_rms", "scatter", "")],
               "aligned:pooled_incisor_rms_mm": q[("aligned_rms", "pooled_stored", "")]}
        ck: dict[str, Any] = {}
        if (pid, "stored") in R:
            s = R[(pid, "stored")]
            dv = {k: abs(s["values"][k] - float(v)) for k, v in ref.items()}
            ck["stored"] = {"tables_max_abs": s["table_checks"], "vd31_max_abs": dv,
                            "ok": bool(all(v <= EXACT_TOL for v in s["table_checks"].values())
                                       and all(v <= 1e-9 for v in dv.values()))}
        if (pid, "nominal") in R:
            n = R[(pid, "nominal")]["values"]
            d_tot = abs(n["heldout_hp:total_deg"] - float(ref["heldout_hp:total_deg"]))
            d_al = abs(n["aligned:pooled_incisor_rms_mm"]
                       / float(ref["aligned:pooled_incisor_rms_mm"]) - 1)
            ck["nominal"] = {"heldout_hp_total_abs_deg": d_tot, "aligned_rel": d_al,
                             "diffs": {k: n[k] - float(v) for k, v in ref.items()},
                             "ok": bool(d_tot <= POOLED_TOL_DEG and d_al <= ALIGNED_TOL)}
        checks[pid] = ck
    if not all(c.get("ok", True) for ck in checks.values() for c in ck.values()):
        raise SystemExit(f"reproduction failed: {json.dumps(checks, indent=1, default=float)}")

    rows = []
    for (pid, arm), r in R.items():
        for k, v in r["values"].items():
            q, comp = k.split(":", 1)
            rows.append({"pid": pid, "take": TAKES[pid],
                         "model": arm, "quantity": q, "component": comp, "value": float(v),
                         "unit": unit_of(comp)})
    out = pd.DataFrame(rows)
    # registration change of the estimated against the nominal model
    for pid in a.pids:
        if (pid, "nominal") in R and (pid, "estimated") in R:
            from biomech.validation.registration import make_pose, pose_error

            def pose(vals: dict, tr: str) -> np.ndarray:
                rv = np.radians([vals[f"registration:{tr}:r{c}_deg"] for c in "xyz"])
                return make_pose(Rotation.from_rotvec(rv).as_matrix(),
                                 np.array([vals[f"registration:{tr}:t{c}_mm"] for c in "xyz"]))

            vn, ve = R[(pid, "nominal")]["values"], R[(pid, "estimated")]["values"]
            trs = sorted({k.split(":")[1] for k in vn if k.startswith("registration:")})
            for tr in trs:
                dp, ang = pose_error(pose(ve, tr)[None], pose(vn, tr)[None])
                out = pd.concat([out, pd.DataFrame([
                    {"pid": pid, "take": TAKES[pid], "model": "estimated",
                     "quantity": "registration_change", "component": f"{tr}:rot_deg",
                     "value": float(ang[0]), "unit": "deg"},
                    {"pid": pid, "take": TAKES[pid], "model": "estimated",
                     "quantity": "registration_change", "component": f"{tr}:trans_mm",
                     "value": float(np.linalg.norm(dp)), "unit": "mm"}])], ignore_index=True)
    corners = pd.DataFrame([c for r in res for c in r.get("corners", [])])
    for c in corners.itertuples():
        out = pd.concat([out, pd.DataFrame([{
            "pid": c.pid, "take": c.take, "model": "estimated", "quantity": "corner_offset",
            "component": f"{c.body}:tag{c.tag}_c{c.corner}", "value": c.offset_mm,
            "unit": "mm"}])], ignore_index=True)

    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "software", "scripts"],
                           capture_output=True, text=True, cwd=_pins.ROOT).stdout.strip()
    partial = set(a.pids) != set(TAKES) or set(a.arms) != {"stored", "nominal", "estimated"}
    tag = a.snapshot + ("_partial" if partial else "")
    files = {"main": out_dir / f"vd35_tag_model_{tag}.parquet",
             "corners": out_dir / f"vd35_tag_model_corners_{tag}.parquet"}
    out.assign(snapshot=a.snapshot).to_parquet(files["main"], index=False)
    corners.to_parquet(files["corners"], index=False)
    meta = {"plan": "experiments/jtsmm_validation/analysis_plan.md VD35 (commit 58ede2a6)",
            "snapshot": a.snapshot, "git_sha": sha, "git_dirty_paths": dirty.splitlines(),
            "settings": {"fit": "trimmed Kabsch of the live tracker (vd29_array_noise.TrimFit)",
                         "estimate": "median of each corner over all frames of the take with a "
                                     "nominal trimmed fit, in that body frame; one re-centring "
                                     "(rigid Kabsch alignment onto the nominal corners)",
                         "validity": "stored pose valid and refit exists",
                         "ct_pose": "tip pose (t = R tip + t, ct_tip_body_mm of the take)",
                         "chain": list(CHAIN), "exact_tol": EXACT_TOL,
                         "pooled_tol_deg": POOLED_TOL_DEG, "aligned_tol": ALIGNED_TOL,
                         "baseline_axis": "HP tag-centre difference of the model in use, "
                                          "mapped into the Qualisys HP frame by that arm's Y_HP"},
            "checks": checks,
            "estimate_info": {r["pid"]: r["estimate_info"] for r in res if "estimate_info" in r},
            "refit_info": {f"{r['pid']}:{r['arm']}": r["refit_info"] for r in res
                           if "refit_info" in r},
            "xhat": {f"{r['pid']}:{r['arm']}": r["xhat"] for r in res},
            "job_runtime_s": {f"{r['pid']}:{r['arm']}": r["runtime_s"] for r in res},
            "runtime_s": time.perf_counter() - t_start,
            "files": {k: v.name for k, v in files.items()},
            "sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}}
    files["main"].with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 300)
    show = out[~out.quantity.isin(["corner_offset", "registration"])]
    print(show.pivot_table(index=["quantity", "component"], columns=["take", "model"],
                           values="value").to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {files['main']} ({meta['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
