# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD20 (validation plan VD20, exploratory): the camera-free ladder of the dataset
study X5 on the comparison takes, against Qualisys. Registered by the import at the end of
:mod:`biomech.validation.stages_ref`.

Per participant (n = 2; the two-participant mean is formed in the views, descriptive):

* **inputs:** the take's IMU streams on a 200 Hz master-clock grid; the hand-eye mounting of
  the take by the VD14 rule (``imu_mounting_files``), the lever arms from the same hand-eye
  session (VD15 method, primary cut-off; none from a mounting embedded in the take); the
  landmark frame from the JTS-MM holds (``ref_tip`` landmarks, variant ``-``); the closed
  pose = the JTS-MM optical mean pose over the rest phase (VD10: "since each system's own
  rest pose");
* **rungs:** F0 (JTS-MM optics), E1 (in-take 7-parameter fit on the take's free_motion
  against the JTS-MM optics), E2a/E2b, E3a/E3b (causal), E4hinge, E4own (ridge model on the
  JTS-MM optical free_motion), E5/E6 (accelerometer bridging; lever arms needed). E4gen is
  not computed here: it would need the dataset participants' optical couplings, i.e. the
  other study's data (``cf_val_summary.note``);
* **scoring:** at the JTS-MM frame times, frames aligned per held-out movement task as in
  VD3/VD10 (``A M B`` fitted on the optical-only jaw-in-head poses of the other movement
  tasks, ``evaluate.run_ref_fusion``'s rule); every trajectory -- estimates, F0 and Qualisys
  -- through the dataset chain (``evaluate.dataset_chain``, 6 Hz), so the comparison has the
  dataset's bandwidth; opening angle and geodesic error of the rotation since each system's
  own closed (rest) pose; incisor after alignment; gated frames (``ref_quality``) left out;
  free_motion (E1 and E4own's calibration block) not scored;
* **non-inferiority to F0:** mean paired per-frame difference of the absolute error (opening;
  incisor) with a 1-s block-bootstrap 95 % CI, non-inferior when the upper CI is below the
  margin (0.5°, 1.0 mm);
* **VD10 drift:** OLS slope of E2's signed opening error against time since calibration (the
  rest-phase reference) and of E3's against time since the last anchor, block-bootstrap CI;
  the anchor-residual diagnostic splits E3's residuals into heading and tilt parts
  (:func:`biomech.imu.camerafree.drift_signature`).

Nothing is tuned on Qualisys. Like the other reference stages, the hand-eye files are read by
path and are not part of the cache key (``--force ref_camerafree`` after replacing one).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as Rot

from biomech.imu import camerafree as cf
from biomech.kinematics import se3
from biomech.pipeline import ParticipantContext, StageResult, stage
from biomech.validation import anatomy as AN
from biomech.validation import registration as REG
from biomech.validation import selection as SEL

DEFAULTS: dict[str, Any] = {
    "reference": "reference.yaml",
    "grid_hz": 200.0,
    "closed": dict(cf.CLOSED),
    "bridge": dict(cf.BRIDGE),
    "pseudo_hz": 5.0,
    "ridge_log10_lambda": [-8.0, 2.0, 21],
    "ridge_folds": 5,
    "calibration_task": "free_motion",
    "margin_open_deg": 0.5,
    "margin_inc_mm": 1.0,
    "block_s": 1.0,
    "n_boot": 1000,
    "primary_e3": "E3b",
}
ROT = ("E1", "E2a", "E2b", "E3a", "E3b")
TRANS = ("E4hinge", "E4own", "E5", "E6")


def _interp(t: np.ndarray, x: np.ndarray, tq: np.ndarray, max_gap: float) -> np.ndarray:
    x = np.asarray(x, float).reshape(len(t), -1)
    ok = np.isfinite(x).all(axis=1)
    out = np.full((len(tq), x.shape[1]), np.nan)
    if ok.sum() < 2:
        return out
    ts = np.asarray(t, float)[ok]
    for j in range(x.shape[1]):
        out[:, j] = np.interp(tq, ts, x[ok, j], left=np.nan, right=np.nan)
    k = np.clip(np.searchsorted(ts, tq), 1, len(ts) - 1)
    out[(ts[k] - ts[k - 1]) > max_gap] = np.nan
    return out


def _rot(rv_deg: np.ndarray) -> np.ndarray:
    out = np.full((len(rv_deg), 3, 3), np.nan)
    ok = np.isfinite(rv_deg).all(axis=1)
    if ok.any():
        out[ok] = Rot.from_rotvec(np.radians(rv_deg[ok])).as_matrix()
    return out


def _rv(R: np.ndarray) -> np.ndarray:
    return np.degrees(cf._rotvec(R))


def _pose(R: np.ndarray, p: np.ndarray | None) -> np.ndarray:
    T = np.full((len(R), 4, 4), np.nan)
    T[:, :3, :3] = R
    T[:, :3, 3] = 0.0 if p is None else p
    T[:, 3] = [0.0, 0.0, 0.0, 1.0]
    T[~np.isfinite(T.reshape(len(T), -1)).all(1)] = np.nan
    return T


def _block_ci(d: np.ndarray, t: np.ndarray, fn: Any = np.mean, block_s: float = 1.0,
              n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % CI of ``fn`` by a bootstrap over ``block_s`` blocks of frames (the block rule of
    ``evaluate``'s paired fusion differences); ``fn(idx)`` on resampled frame indices."""
    ok = np.isfinite(d)
    idx_all = np.flatnonzero(ok)
    if idx_all.size < 10:
        return np.nan, np.nan
    blk = np.floor((t[idx_all] - t[idx_all][0]) / block_s).astype(int)
    groups = [idx_all[blk == b] for b in np.unique(blk)]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(fn(np.concatenate([groups[i] for i in pick])))
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if not v.size:
        return np.nan, np.nan
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


def _rms(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.sqrt(np.mean(x ** 2))) if x.size else np.nan


def _slope(t_min: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
    ok = idx[np.isfinite(y[idx]) & np.isfinite(t_min[idx])]
    if ok.size < 10 or np.ptp(t_min[ok]) <= 0:
        return np.nan
    return float(np.polyfit(t_min[ok], y[ok], 1)[0])


def run_ref_camerafree(t_s: np.ndarray, M_opt: np.ndarray, M_Q: np.ndarray, fr: AN.Frames,
                       held: Sequence[SEL.Interval], gate: np.ndarray, cra: Any, mta: Any,
                       mounting: dict[str, np.ndarray], levers: dict[str, np.ndarray],
                       p: dict[str, Any], pid: str, seed: int = 0,
                       chain: dict[str, Any] | None = None
                       ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """The VD20 work on plain arrays: ``t_s`` the JTS-MM frame times (master s), ``M_opt``
    and ``M_Q`` the JTS-MM and Qualisys jaw-in-head poses there (each in its own anatomical
    frames), ``fr`` the JTS-MM frames, ``held`` the evaluation intervals (rest and movement
    tasks), ``mounting``/``levers`` per IMU slot (``cra``, ``mta``)."""
    from biomech.imu.calibration import fit as fit_e1
    from biomech.validation import kinematics as VK
    from biomech.validation.evaluate import dataset_chain

    p = {**DEFAULTS, **p, "closed": {**cf.CLOSED, **(p.get("closed") or {})},
         "bridge": {**cf.BRIDGE, **(p.get("bridge") or {})}}
    chain = dict(chain or {})
    t_s = np.asarray(t_s, float)
    gate = np.asarray(gate, bool)
    notes = ["E4gen not computed (needs the dataset participants' optics)"]
    rest_iv = [h for h in held if h.name in SEL.STATIC_TASKS]
    mov = [h for h in held if h.name in SEL.MOVEMENT_TASKS]
    if not rest_iv or not mov:
        raise LookupError("ref_camerafree needs a rest phase and movement tasks")
    M_opt_lp = dataset_chain(M_opt, t_s, chain)
    M_Q_lp = dataset_chain(M_Q, t_s, chain)
    k_rest = VK.mask_of(t_s, rest_iv)
    T_c = REG.mean_pose(M_opt_lp[k_rest & se3.valid(M_opt_lp)])       # JTS-MM closed pose
    Q_c = REG.mean_pose(M_Q_lp[k_rest & se3.valid(M_Q_lp)])
    T_max_hp = se3.inv(fr.F_max)
    plate = se3.inv(fr.F_mand)                                         # mandibular -> plate
    R_closed_plate = (T_c @ plate)[:3, :3]
    inc0 = T_c[:3, 3].copy()
    tmj = [T_c[:3, :3] @ v + T_c[:3, 3] for v in fr.tmj_mand.values()]
    c0 = np.mean(tmj, axis=0) if tmj else np.full(3, np.nan)

    # 200 Hz master-clock grid over rest and the movement tasks
    t0, t1 = min(h.t0 for h in held), max(h.t1 for h in held)
    step = 1.0 / float(p["grid_hz"])
    tg_s = np.arange(t0, t1, step)
    tg = np.round(tg_s * 1e6).astype(np.int64)
    rest_g = VK.mask_of(tg_s, rest_iv)
    gap = 2.5 * float(np.median(np.diff(t_s)))
    rv_opt = _interp(t_s, _rv(np.einsum("nij,kj->nik", M_opt_lp[:, :3, :3], T_c[:3, :3])),
                     tg_s, gap)
    cmid_opt = (_interp(t_s, np.mean([se3.apply(M_opt_lp, v) for v in fr.tmj_mand.values()],
                                     axis=0), tg_s, gap)
                if fr.tmj_mand else np.full((len(tg), 3), np.nan))

    mp = cf.mounting_prior(mounting["cra"], mounting["mta"], T_max_hp)
    lad = cf.rotation_ladder(tg, cra, mta, mp.A, rest_g, X=mp.X, R_closed=R_closed_plate,
                             params=p["closed"])
    qc, qm = cra.quat_at(tg), mta.quat_at(tg)
    est: dict[str, np.ndarray] = {k: lad.estimates[k] for k in ("E2a", "E2b", "E3a", "E3b")
                                  if k in lad.estimates}
    cal_g = VK.mask_of(tg_s, [h for h in mov if h.name == p["calibration_task"]])
    R_opt_plate = _rot(rv_opt) @ R_closed_plate
    try:
        cal = fit_e1(qc[cal_g], qm[cal_g], R_opt_plate[cal_g])
        est["E1"] = cal.predict(qc, qm) @ R_closed_plate.T
    except ValueError as exc:
        notes.append(f"E1: {exc}")
    prim = str(p["primary_e3"])
    an = lad.anchored[prim]
    R3 = est[prim]
    rv3 = _rv(R3)
    thd3 = cf.theta_rate_dps(tg_s, rv3[:, 1])
    pos: dict[str, np.ndarray] = {}
    model = None
    if np.isfinite(c0).all():
        pos["E4hinge"] = cf.incisor_position(R3, None, inc0, c0)
        tau_opt = cmid_opt - c0
        thd_opt = cf.theta_rate_dps(tg_s, rv_opt[:, 1])
        ok = cal_g & np.isfinite(rv_opt).all(1) & np.isfinite(tau_opt).all(1) & \
            np.isfinite(thd_opt)
        lo, hi, kk = p["ridge_log10_lambda"]
        try:
            model = cf.TranslationModel.fit(rv_opt[ok], thd_opt[ok], tau_opt[ok],
                                            lambdas=np.logspace(float(lo), float(hi), int(kk)),
                                            n_folds=int(p["ridge_folds"]))
            tau_own = model.predict(rv3, thd3)
            pos["E4own"] = cf.incisor_position(R3, tau_own, inc0, c0)
        except ValueError as exc:
            notes.append(f"E4own: {exc}")
        if "cra" in levers and "mta" in levers:
            geom = cf.jaw_geometry(T_max_hp, T_c @ plate, levers["cra"], levers["mta"], inc0, c0)
            amask = an.anchor_mask()
            br = cf.accel_bridge(tg_s, R3, an.psi_rad, cra, mta, mp.A, geom, amask, rest_g,
                                 params=p["bridge"])
            pos["E5"] = br.incisor_mm
            if model is not None:
                pos["E6"] = cf.fuse_translation(tg_s, R3, an.psi_rad, cra, mta, mp.A, geom,
                                                amask, rest_g, tau_own, model.cv_rmse,
                                                pseudo_hz=float(p["pseudo_hz"]),
                                                params=p["bridge"]).incisor_mm
        else:
            notes.append("E5/E6 not computed: no lever arm from a hand-eye session")
    else:
        notes.append("translation rungs not computed: condyle landmarks missing")

    # every estimate through the dataset chain, then onto the JTS-MM frame times
    fs = float(p["grid_hz"])
    lp_hz = float(chain.get("lowpass_hz", 6.0))
    lp_ord = int(chain.get("lowpass_order", 2))
    M_est: dict[str, np.ndarray] = {e: se3.lowpass(_pose(R, None), fs, lp_hz, lp_ord)
                                    for e, R in est.items()}
    for e, x in pos.items():
        M_est[e] = se3.lowpass(_pose(R3, x), fs, lp_hz, lp_ord)
    at_frames: dict[str, np.ndarray] = {}
    for e, T in M_est.items():
        rv = _interp(tg_s, _rv(T[:, :3, :3]), t_s, 2.5 * step)
        pp = _interp(tg_s, T[:, :3, 3], t_s, 2.5 * step) if e in pos else \
            np.zeros((len(t_s), 3))                                    # rotation-only rung
        at_frames[e] = _pose(_rot(rv) @ T_c[:3, :3], pp)               # back to absolute
    at_frames["F0"] = M_opt_lp

    # per held-out movement task: frames aligned on the other tasks' optical poses (VD3)
    ok_al = se3.valid(M_opt) & se3.valid(M_Q) & ~gate

    def align_fold(h: SEL.Interval) -> tuple[np.ndarray, np.ndarray] | None:
        ktr = VK.mask_of(t_s, [m for m in mov if m is not h]) & ok_al
        if ktr.sum() < 50:
            return None
        try:
            A0, B0 = REG.init_xy(M_Q[ktr], M_opt[ktr])
            rg = REG.register_paired({"jaw": (M_Q[ktr], M_opt[ktr])}, A0, {"jaw": B0})
        except (ValueError, np.linalg.LinAlgError):
            return None
        return rg.X, rg.Y["jaw"]

    th_q = _rv(np.einsum("nij,kj->nik", M_Q_lp[:, :3, :3], Q_c[:3, :3]))
    n = len(t_s)
    err = {e: {"open": np.full(n, np.nan), "geo": np.full(n, np.nan),
               "inc": np.full((n, 3), np.nan)} for e in at_frames}
    scored = [h for h in mov if h.name != p["calibration_task"]]
    for h in scored:
        AB = align_fold(h)
        if AB is None:
            continue
        k = VK.mask_of(t_s, [h]) & ~gate
        A, Bm = AB
        Tc_al = A @ T_c @ Bm
        for e, M in at_frames.items():
            Ma = np.einsum("ij,njk,kl->nil", A, M[k], Bm)
            rel = np.einsum("nij,kj->nik", Ma[:, :3, :3], Tc_al[:3, :3])
            rel_q = np.einsum("nij,kj->nik", M_Q_lp[k, :3, :3], Q_c[:3, :3])
            err[e]["open"][k] = _rv(rel)[:, 1] - th_q[k, 1]
            err[e]["geo"][k] = np.linalg.norm(_rv(np.einsum("nij,nkj->nik", rel, rel_q)), axis=1)
            if e in pos or e == "F0":
                err[e]["inc"][k] = Ma[:, :3, 3] - M_Q_lp[k, :3, 3]
    seg_rows = []
    for name, ivs in [(h.name, [h]) for h in scored] + [("movement_pooled", scored)]:
        k = VK.mask_of(t_s, ivs) & ~gate
        for e, d in err.items():
            o = d["open"][k]
            if not np.isfinite(o).any():
                continue
            inc = d["inc"][k]
            n3 = np.linalg.norm(inc, axis=1)
            seg_rows.append({"pid": pid, "segment": name, "estimator": e,
                             "n": int(np.isfinite(o).sum()), "rmse_open_deg": _rms(o),
                             "bias_open_deg": float(np.nanmean(o)), "rmse_geo_deg": _rms(d["geo"][k]),
                             "rmse_inc_mm": _rms(n3),
                             **{f"rmse_inc_{a}_mm": _rms(inc[:, j]) for j, a in enumerate("xyz")}})
    kp = VK.mask_of(t_s, scored) & ~gate
    ni_rows = []
    for e in err:
        if e == "F0":
            continue
        for oc, a_e, a_0, margin in (
                ("opening_abs_err_deg", np.abs(err[e]["open"]), np.abs(err["F0"]["open"]),
                 float(p["margin_open_deg"])),
                ("incisor_err_mm", np.linalg.norm(err[e]["inc"], axis=1),
                 np.linalg.norm(err["F0"]["inc"], axis=1), float(p["margin_inc_mm"]))):
            if (oc == "incisor_err_mm") != (e in pos):        # each rung on its own outcome
                continue
            d = np.where(kp, a_e - a_0, np.nan)
            if np.isfinite(d).sum() < 10:
                continue
            lo, hi = _block_ci(d, t_s, lambda i, d=d: float(np.mean(d[i])),
                               float(p["block_s"]), int(p["n_boot"]), seed)
            ni_rows.append({"pid": pid, "estimator": e, "outcome": oc,
                            "n": int(np.isfinite(d).sum()), "mean_diff": float(np.nanmean(d)),
                            "ci_lo": lo, "ci_hi": hi, "margin": margin,
                            "noninferior": bool(np.isfinite(hi) and hi < margin)})
    # VD10 drift: E2 against time since calibration, E3 against time since the last anchor
    t_ref = 0.5 * (rest_iv[0].t0 + rest_iv[-1].t1)
    w1 = np.asarray(an.anchors["w1"], int)
    since_anchor = np.full(n, np.nan)
    if w1.size:
        ends = tg_s[w1 - 1]
        j = np.searchsorted(ends, t_s, side="right") - 1
        since_anchor = np.where(j >= 0, t_s - ends[np.clip(j, 0, None)], np.nan)
    dr_rows = []
    idx = np.flatnonzero(kp)
    for e, clock, tt in (("E2a", "calibration", (t_s - t_ref) / 60.0),
                         ("E2b", "calibration", (t_s - t_ref) / 60.0),
                         (prim, "anchor", since_anchor / 60.0)):
        if e not in err:
            continue
        y = err[e]["open"]
        s = _slope(tt, y, idx)
        lo, hi = _block_ci(np.where(kp, y, np.nan), t_s,
                           lambda i, y=y, tt=tt: _slope(tt, y, i), float(p["block_s"]),
                           int(p["n_boot"]), seed)
        dr_rows.append({"pid": pid, "estimator": e, "clock": clock, "n": int(idx.size),
                        "slope_dpm": s, "ci_lo": lo, "ci_hi": hi})
    sig = cf.drift_signature(an, tg_s, cra, mta)
    pb = lad.psi_b
    summary = {"pid": pid, "lever_source": "hand-eye session (lever-arm method of the validation study)" if levers else "",
               **{f"lever_{s}_{a}_mm": float(levers[s][i]) if s in levers else np.nan
                  for s in ("cra", "mta") for i, a in enumerate("xyz")},
               "psi_a_deg": float(np.degrees(lad.psi0.get("a", np.nan))),
               "psi_b_deg": float(np.degrees(pb.psi_best_rad)),
               "psi_b_identifiable": bool(pb.identifiable),
               "psi_b_fallback": bool(lad.flags.get("psi_b_fallback", False)),
               "tilt_range_deg": float(pb.tilt_range_deg), "primary_e3": prim,
               "n_anchors": int(len(an.anchors["t0_s"])),
               "drift_label": str(sig.get("label", "none")),
               **{f"drift_{k}": float(sig.get(k, np.nan)) for k in
                  ("heading_deg", "tilt_deg", "fusion_tilt_deg", "step_frac", "net_deg")},
               "e4own_cv_rmse_mm": float(model.cv_rmse) if model is not None else np.nan,
               "note": "; ".join(notes)}
    return ({"cf_val_summary": pd.DataFrame([summary]),
             "cf_val_segments": pd.DataFrame(seg_rows),
             "cf_val_noninferiority": pd.DataFrame(ni_rows),
             "cf_val_drift": pd.DataFrame(dr_rows)}, {})


@stage("ref_camerafree", depends=["ref_sync", "ref_tip", "ref_quality"], version="1",
       group="validation",
       description="VD20 (exploratory): camera-free ladder (dataset X5) against Qualisys with "
                   "the hand-eye mountings (VD14) and the JTS-MM landmark frame; "
                   "non-inferiority to the optics, VD10 drift test, anchor diagnostic")
def ref_camerafree_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.imu.preprocess import load
    from biomech.validation import lever as LV
    from biomech.validation.evaluate import holds_from_table
    from biomech.validation.fusion import DEFAULTS as FD
    from biomech.validation.fusion import load_mounting
    from biomech.validation.stages import _systems, imu_mounting_files

    sy, p, *_ = _systems(ctx)
    files = imu_mounting_files(ctx, p)
    mounting, levers = {}, {}
    for name, f in files.items():
        slot = FD["imu_slots"][name]
        R = load_mounting(f, slot)
        if R is None:
            continue
        mounting[slot] = R
        d = LV.read_slot(f, slot) if f.suffix == ".h5" else None
        if d is not None:
            M, y, t = LV.design(d, float(LV.DEFAULTS["fc_hz"]), LV.DEFAULTS)
            lv = LV.fit(M, y, t, {**LV.DEFAULTS, "n_boot": 0}, int(ctx.config.seed or 0))
            if "r_x_mm" in lv:
                levers[slot] = np.array([lv[f"r_{a}_mm"] for a in "xyz"])
    if set(mounting) != {"cra", "mta"}:
        raise LookupError(f"no hand-eye mounting for {sorted({'cra', 'mta'} - set(mounting))}")
    sd = ctx.session("motion")
    cra, mta = load(sd, "cra"), load(sd, "mta")
    if cra is None or mta is None:
        raise LookupError("IMU streams missing")
    lm = ctx.upstream("ref_tip", "landmarks")
    hs, hq = holds_from_table(lm, "jtsmm", "-"), holds_from_table(lm, "qualisys", "primary")
    fs_, fq = AN.anatomical_frames(hs), AN.anatomical_frames(hq)
    M_opt = AN.jaw_in_head(sy.S["HP"], sy.S["MP"], fs_)
    M_Q = AN.jaw_in_head(sy.q_at_s("HP"), sy.q_at_s("MP"), fq)
    qa = ctx.upstream_arrays("ref_quality")
    gate = qa.get("gate_at_s")
    gate = np.zeros(len(sy.tS), bool) if gate is None else np.asarray(gate, bool)
    cp = {k: v for k, v in ctx.params.items() if k in DEFAULTS}
    tables, arrays = run_ref_camerafree(
        sy.tS, M_opt, M_Q, fs_, sy.held, gate, cra, mta, mounting, levers, cp, ctx.pid,
        int(ctx.config.seed or 0), chain=dict(ctx.config.signals.get("kinematics") or {}))
    return StageResult(tables=tables, arrays=arrays)
