# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Stages of the system-validation study (``group="validation"``; experiments/jtsmm_validation).

The dataset study (``pipeline.groups: [dataset]``) never runs these: ``resolve_order``
rejects a stage outside the study's groups. Registered by importing this module from
``biomech.stages``.

``ref_sync`` (participant scope): temporal sync of the comparison take with its Qualisys
recording (plan D2) jointly with the spatial registration (D3), plus the validation
diagnostics (held-out movement phases, sliding-window lags). Inputs per participant come
from the study's ``reference.yaml`` (path in ``stages.ref_sync.reference``).

Note: the reference files (QTM ``.mat``, Delsys CSV) are not part of the cache key (only
the JTS-MM ``session.h5`` files are); re-run with ``--force ref_sync`` after replacing one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.pipeline import ParticipantContext, StageResult, stage

DEFAULTS: dict[str, Any] = {
    "reference": "reference.yaml",   # relative to the study file
    "exposure_s": 0.012,             # JTS-MM exposure; trigger = exposure start
    "frame_s": 0.025,                # JTS-MM nominal frame period
    "hint_range_s": 5.0,             # edge vote searched in lag_hint ± this
    "edge_tol_s": 0.05,
    "xcorr_half_range_s": 0.3,
    "sigma_pos_mm": 0.3,             # residual scales of the fine fit (whitening + Huber)
    "sigma_rot_deg": 0.15,
    "huber_k": 2.5,
    "sliding_win_s": 15.0,
    "sliding_step_s": 5.0,
    "speed_baseline_s": 0.1,         # |ω| difference quotient (same in both systems)
    "trim_s": 0.5,                   # D0b: phase ends dropped from the held-out evaluation
    "sync_pad_s": 0.5,
    "fit_bodies": "ct",              # R5b: clock and X from the CT only; "all" = joint fit
    # VD19 sensitivity: extra clock variants with the time model of the sliding-window lag
    # line (offset + drift of the line, registration of the CT-only fit): "omega" (|ω|
    # correlation, frame-free), "pose" (registered poses), "both"; null = none. The variants
    # land in ref_sync:clock_variants / clock_criteria (variant sliding_omega / sliding_pose)
    # and are re-scored in ref_kinematics:robustness (factor clock), own and aligned frames.
    "sliding_variant": None,
}


def load_reference(path: str | Path) -> dict[str, Any]:
    """The validation study's reference manifest (``reference.yaml``)."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "participants" not in doc:
        raise ValueError(f"{path}: no 'participants' block")
    return doc


def reference_path(ctx_config: Any, rel: str) -> Path:
    p = Path(rel)
    if p.is_absolute():
        return p
    base = ctx_config.source.parent if ctx_config.source is not None else Path.cwd()
    return (base / p).resolve()


def participant_reference(doc: dict[str, Any], pid: str, take_dir: Path) -> dict[str, Any]:
    """The participant's entry with file paths resolved (relative to the comparison take)."""
    entry = dict((doc.get("participants") or {}).get(pid) or {})
    if not entry:
        raise LookupError(f"reference manifest has no entry for {pid}")
    out = dict(entry)
    for key in ("qtm_mat", "qtm_raw", "qtm_cal", "delsys_csv"):
        if entry.get(key):
            out[key] = (take_dir / entry[key]).resolve()
    if entry.get("delsys_mvc_csv"):
        out["delsys_mvc_csv"] = [(take_dir / f).resolve() for f in entry["delsys_mvc_csv"]]
    return out


def _pose_diff(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    """(translation mm, rotation deg) between two poses."""
    from biomech.validation import registration as REG

    dp, ang = REG.pose_error(A[None], B[None])
    return float(np.linalg.norm(dp)), float(ang[0])


def _rotvec_deg(T: np.ndarray) -> np.ndarray:
    return np.degrees(Rotation.from_matrix(T[:3, :3]).as_rotvec())


def _assign_drift(session: Any) -> float:
    """cam0 frame-assignment residual drift (µs/s = ppm) from the session's sync report."""
    try:
        rep = getattr(session, "sync_report", None) or {}
        return float(rep["cameras"]["cam0"]["assign_residual"]["drift_us_per_s"])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def run_ref_sync(session: Any, ref: dict[str, Any], bodies: dict[str, str],
                 sync_cfg: dict[str, Any], params: dict[str, Any], pid: str
                 ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """The work of ``ref_sync`` on an open :class:`~biomech.io.session.SessionData` and a
    resolved reference entry; returns ``(tables, arrays)``."""
    from biomech.io.reference import read_qtm_mat
    from biomech.validation import selection as SEL
    from biomech.validation import sync as SC

    p = {**DEFAULTS, **params}
    opt = session.optical()
    if opt is None:
        raise LookupError("comparison take has no optical group")
    tS = SC.frame_times(opt.t_us, p["exposure_s"])
    S = {q: opt.poses[j] for q, j in bodies.items() if j in opt.poses}
    qtm = read_qtm_mat(ref["qtm_mat"])
    Q = {q: qtm.pose(q) for q in bodies if q in qtm.bodies}
    missing = sorted(set(bodies) - set(S) - set(Q))
    if "CT" not in S or "CT" not in Q:
        raise LookupError(f"CT missing in one system (missing bodies: {missing})")
    ct_ok = se3.valid(S["CT"])
    windows = SEL.sync_windows(session.phases, tS, ct_ok,
                               start_phase=sync_cfg.get("start_phase", "mand_point_1"),
                               end_phase=sync_cfg.get("end_phase", "free_motion"),
                               pad_s=p["sync_pad_s"])
    held = SEL.movement_intervals(session.phases, tS, ct_ok, trim_s=p["trim_s"])
    res = SC.synchronise(
        tS, S, qtm.t_s, Q, windows, heldout=held, lag_hint_s=ref.get("lag_hint_s"),
        hint_range_s=p["hint_range_s"], edge_tol_s=p["edge_tol_s"], frame_s=p["frame_s"],
        xcorr_half_range_s=p["xcorr_half_range_s"], sliding_win_s=p["sliding_win_s"],
        sliding_step_s=p["sliding_step_s"], sigma_pos_mm=p["sigma_pos_mm"],
        sigma_rot_deg=p["sigma_rot_deg"], huber_k=p["huber_k"],
        speed_baseline_s=p["speed_baseline_s"], fit_bodies=p["fit_bodies"], variants=True)
    # clock variants and the pre-registered criteria (R5c.3, V6)
    line_w = res.line.get("joint", {})
    sv = p.get("sliding_variant")
    for key in ({"omega": ["omega"], "pose": ["pose"], "both": ["omega", "pose"]}.get(sv, [])
                if sv else []):
        fv = SC.sliding_variant(res.fine, res.line.get("joint" if key == "omega" else "pose", {}))
        if fv is not None:
            res.variants[f"sliding_{key}"] = fv
    crit_rows, var_rows = [], []
    crit: dict[str, dict] = {}
    Qspl_ct = SC.PoseSpline(qtm.t_s, Q["CT"])
    for name, fv in res.variants.items():
        wl = SC.window_lags(tS, S["CT"], Qspl_ct, fv.model, fv.X, fv.Y["CT"], res.windows)
        # VD19: offset and drift SEs from the per-window HAC offset SEs (6-DoF, as the fit)
        dw = SC.drift_from_windows(SC.window_offsets(
            tS, S["CT"], Qspl_ct, fv.model, fv.X, fv.Y["CT"], res.windows,
            scale=fv.scale.get("CT", res.fine.scale.get("CT", (p["sigma_pos_mm"],
                                                               p["sigma_rot_deg"])))),
            fv.model.t_ref_s)
        c = SC.clock_criteria(wl, line_w, fv.model, alpha_se_hac_ppm=dw["se_alpha_ppm_hac"])
        if not name.startswith("sliding_"):      # VD19 sensitivities never choose the clock
            crit[name] = c
        crit_rows.append({"pid": pid, "variant": name, **c})
        dX = _pose_diff(fv.X, res.fine.X)
        var_rows.append({"pid": pid, "variant": name, "offset_s": fv.model.offset_s,
                         "alpha_ppm": fv.model.alpha * 1e6,
                         "se_offset_ms": fv.model.se_offset_s * 1e3,
                         "se_alpha_ppm": fv.model.se_alpha * 1e6,
                         "se_offset_ms_hac": dw["se_offset_ms_hac"],
                         "se_alpha_ppm_hac": dw["se_alpha_ppm_hac"],
                         "d_offset_ms": (fv.model.offset_s - res.fine.model.offset_s) * 1e3,
                         "d_x_trans_mm": dX[0], "d_x_rot_deg": dX[1],
                         "y_ct_tx_mm": fv.Y["CT"][0, 3], "y_ct_ty_mm": fv.Y["CT"][1, 3],
                         "y_ct_tz_mm": fv.Y["CT"][2, 3],
                         "y_ct_trans_mm": float(np.linalg.norm(fv.Y["CT"][:3, 3])),
                         "n_frames": int(sum(fv.n_frames.values())),
                         "window_lags": ";".join(
                             f"{r.window}:{r.lag_ms:.3f}±{r.se_ms_ar1:.3f}"
                             for r in wl.itertuples())})
    primary = SC.choose_clock(crit) if crit else "ct"
    f = res.variants.get(primary, res.fine)
    m = f.model
    infl = f.se_inflation
    # VD19: offset and drift SEs from the per-window HAC offset SEs of the chosen clock
    dwp = SC.drift_from_windows(SC.window_offsets(
        tS, S["CT"], Qspl_ct, m, f.X, f.Y["CT"], res.windows,
        scale=f.scale.get("CT", res.fine.scale.get("CT", (p["sigma_pos_mm"],
                                                          p["sigma_rot_deg"])))), m.t_ref_s)
    if primary != "ct":                       # held-out residuals under the chosen clock
        Qs = {b: SC.PoseSpline(qtm.t_s, Q[b]) for b in Q if b in S}
        res.heldout = SC.evaluate_intervals(tS, {b: S[b] for b in Qs}, Qs, m, f.X, f.Y, held)
    held_mov = res.heldout[res.heldout["interval"].isin(SEL.MOVEMENT_TASKS)]

    def pooled(body: str, col: str) -> float:
        d = held_mov[(held_mov["body"] == body) & (held_mov["n"] > 0)]
        if d.empty:
            return float("nan")
        return float(np.sqrt(np.sum(d["n"] * d[col] ** 2) / d["n"].sum()))

    lw, lp = res.line.get("joint", {}), res.line.get("pose", {})
    sync = pd.DataFrame([{
        "pid": pid, "t_ref_s": res.t_ref_s, "offset_s": m.offset_s, "tau_s": m.tau_s,
        "alpha_ppm": m.alpha * 1e6, "se_offset_ms": m.se_offset_s * 1e3,
        "se_alpha_ppm": m.se_alpha * 1e6,
        "se_offset_ms_hac": dwp["se_offset_ms_hac"], "se_alpha_ppm_hac": dwp["se_alpha_ppm_hac"],
        "se_alpha_ppm_jk": dwp["se_alpha_ppm_jk"], "n_windows_hac": dwp["n_windows"],
        "se_inflation_ar1": infl,       # pre-VD19 SEs: fit covariance x AR(1) factor
        "se_offset_ms_ar1": m.se_offset_s * 1e3 * infl, "se_alpha_ppm_ar1": m.se_alpha * 1e6 * infl,
        "n_frames_fit": int(sum(f.n_frames.values())), "sigma2": f.sigma2,
        "fit_pos_rms_mm": float(np.sqrt(np.mean([v ** 2 for v in f.rms_pos_mm.values()]))),
        "fit_rot_rms_deg": float(np.sqrt(np.mean([v ** 2 for v in f.rms_rot_deg.values()]))),
        "heldout_mp_pos_rms_mm": pooled("MP", "pos_rms_mm"),
        "heldout_hp_pos_rms_mm": pooled("HP", "pos_rms_mm"),
        "heldout_rel_pos_rms_mm": pooled("MP|HP", "pos_rms_mm"),
        "heldout_rel_rot_rms_deg": pooled("MP|HP", "rot_rms_deg"),
        "sliding_omega_slope_ppm": lw.get("slope_ppm", np.nan),
        "sliding_omega_slope_se_ppm": lw.get("slope_se_ppm", np.nan),
        "sliding_omega_resid_sd_ms": lw.get("resid_sd_ms", np.nan),
        "sliding_omega_n": lw.get("n_windows", 0),
        "sliding_pose_slope_ppm": lp.get("slope_ppm", np.nan),
        "sliding_pose_slope_se_ppm": lp.get("slope_se_ppm", np.nan),
        "sliding_pose_resid_sd_ms": lp.get("resid_sd_ms", np.nan),
        "sliding_pose_max_step_ms": lp.get("max_step_ms", np.nan),
        "sliding_pose_n_steps": lp.get("n_steps", 0),
        "sliding_pose_quad_ms_per_100s2": lp.get("quad_ms_per_100s2", np.nan),
        "sliding_pose_quad_se": lp.get("quad_se", np.nan),
        "sliding_pose_n": lp.get("n_windows", 0),
        "qtm_fs_hz": qtm.fs_hz, "missing_bodies": ",".join(missing),
        # JTS-MM's own camera frame-assignment drift (sync report), next to α (plan D2.5)
        "cam_assign_drift_ppm": _assign_drift(session),
    }])

    meth = [{"pid": pid, "method": "edge_vote", "offset_s": res.coarse_lag_s,
             "alpha_ppm": np.nan, "se_offset_ms": np.nan, "se_alpha_ppm": np.nan,
             "n": res.coarse_votes, "resid_sd_ms": np.nan, "delta_ms": np.nan, "rho": np.nan,
             "peak_ratio": np.nan, "t_centre_s": np.nan}]
    if res.edge_fit is not None:
        e = res.edge_fit
        em = e.model.recentred(res.t_ref_s)
        meth.append({"pid": pid, "method": "edges", "offset_s": em.offset_s,
                     "alpha_ppm": em.alpha * 1e6, "se_offset_ms": em.se_offset_s * 1e3,
                     "se_alpha_ppm": em.se_alpha * 1e6, "n": e.n_rise + e.n_fall,
                     "resid_sd_ms": e.resid_sd_s * 1e3, "delta_ms": e.delta_s * 1e3,
                     "rho": np.nan, "peak_ratio": np.nan, "t_centre_s": np.nan})
    for r in res.xcorr:
        meth.append({"pid": pid, "method": f"xcorr_ct_{r['window']}",
                     "offset_s": r["lag_s"], "alpha_ppm": np.nan,
                     "se_offset_ms": r["se_s"] * 1e3, "se_alpha_ppm": np.nan,
                     "n": r["n_grid"], "resid_sd_ms": np.nan, "delta_ms": np.nan,
                     "rho": r["rho"], "peak_ratio": r["peak_ratio"],
                     "t_centre_s": r["t_centre_s"]})
    if res.xcorr_model is not None:
        xm = res.xcorr_model
        meth.append({"pid": pid, "method": "xcorr_ct", "offset_s": xm.offset_s,
                     "alpha_ppm": xm.alpha * 1e6, "se_offset_ms": np.nan,
                     "se_alpha_ppm": np.nan, "n": len(res.xcorr), "resid_sd_ms": np.nan,
                     "delta_ms": np.nan, "rho": np.nan, "peak_ratio": np.nan,
                     "t_centre_s": np.nan})
    meth.append({"pid": pid, "method": "fine", "offset_s": m.offset_s,
                 "alpha_ppm": m.alpha * 1e6, "se_offset_ms": m.se_offset_s * 1e3,
                 "se_alpha_ppm": m.se_alpha * 1e6, "n": int(sum(f.n_frames.values())),
                 "resid_sd_ms": np.nan, "delta_ms": np.nan, "rho": np.nan, "peak_ratio": np.nan,
                 "t_centre_s": np.nan})
    for name, ln in res.line.items():
        meth.append({"pid": pid, "method": f"sliding_{'omega' if name == 'joint' else name}",
                     "offset_s": m.offset_s + ln.get("intercept_ms", np.nan) * 1e-3,
                     "alpha_ppm": ln.get("slope_ppm", np.nan),
                     "se_offset_ms": np.nan, "se_alpha_ppm": ln.get("slope_se_ppm", np.nan),
                     "n": ln.get("n_windows", 0), "resid_sd_ms": ln.get("resid_sd_ms", np.nan),
                     "delta_ms": np.nan, "rho": np.nan, "peak_ratio": np.nan,
                     "t_centre_s": np.nan})
    fj = res.fine_joint
    if fj is not None:
        jm = fj.model
        meth.append({"pid": pid, "method": "fine_joint", "offset_s": jm.offset_s,
                     "alpha_ppm": jm.alpha * 1e6, "se_offset_ms": jm.se_offset_s * 1e3,
                     "se_alpha_ppm": jm.se_alpha * 1e6, "n": int(sum(fj.n_frames.values())),
                     "resid_sd_ms": np.nan, "delta_ms": np.nan, "rho": np.nan,
                     "peak_ratio": np.nan, "t_centre_s": np.nan})
        dX = _pose_diff(fj.X, f.X)
        sync["joint_d_offset_ms"] = (jm.offset_s - m.offset_s) * 1e3
        sync["joint_d_alpha_ppm"] = (jm.alpha - m.alpha) * 1e6
        sync["joint_d_x_rot_deg"], sync["joint_d_x_trans_mm"] = dX[1], dX[0]
        # R5b rule: the CT-only fit is primary; flag a joint fit outside 2 SE of it
        sync["joint_disagrees"] = bool(abs(jm.offset_s - m.offset_s)
                                       > 2 * max(m.se_offset_s * infl, 1e-9))
    methods = pd.DataFrame(meth)

    reg_rows = []
    for name, T in [("X", f.X)] + [(f"Y_{b}", f.Y[b]) for b in f.Y]:
        rv = _rotvec_deg(T)
        se_r, se_t = f.block_se(name)
        body = name[2:] if name.startswith("Y_") else ""
        reg_rows.append({
            "pid": pid, "transform": name, "rx_deg": rv[0], "ry_deg": rv[1], "rz_deg": rv[2],
            "angle_deg": float(np.linalg.norm(rv)), "tx_mm": T[0, 3], "ty_mm": T[1, 3],
            "tz_mm": T[2, 3], "se_rot_deg": se_r, "se_trans_mm": se_t,
            "se_rot_deg_ac": se_r * infl, "se_trans_mm_ac": se_t * infl,
            "fit_pos_rms_mm": f.rms_pos_mm.get(body, np.nan),
            "fit_rot_rms_deg": f.rms_rot_deg.get(body, np.nan),
            "n_frames": f.n_frames.get(body, int(sum(f.n_frames.values())))})
    registration = pd.DataFrame(reg_rows)

    heldout = res.heldout.copy()
    heldout.insert(0, "pid", pid)
    sliding = res.sliding.copy()
    sliding.insert(0, "pid", pid)
    win = pd.DataFrame([{"pid": pid, "window": w.name, "t0_s": w.t0, "t1_s": w.t1,
                         "role": "fit"} for w in res.windows]
                       + [{"pid": pid, "window": w.name, "t0_s": w.t0, "t1_s": w.t1,
                           "role": "heldout"} for w in held])
    arrays = {
        "edges_s_rise": res.edges_S.rise, "edges_s_fall": res.edges_S.fall,
        "edges_q_rise": res.edges_Q.rise, "edges_q_fall": res.edges_Q.fall,
        "edge_pairs": (res.edge_fit.pairs if res.edge_fit is not None
                       else np.zeros((0, 3))),
        "sliding_t_s": sliding["t_centre_s"].to_numpy(),
        "sliding_resid_ms": sliding["resid_ms"].to_numpy(),
        # sync views (R5b): CT visibility per system, CT |ω| (Qualisys on the master clock
        # through the fitted model), the |ω| cross-correlation curve of each window
        "ct_t_s": tS, "ct_visible_s": ct_ok,
        "ct_visible_q": se3.valid(Q["CT"]), "ct_t_q_master_s": m.to_s(qtm.t_s),
    }
    # CT path in the sync windows, both systems in the Qualisys world (chosen clock and X)
    in_w = np.zeros(len(tS), bool)
    for w_ in res.windows:
        in_w |= (tS >= w_.t0) & (tS <= w_.t1)
    iw = np.flatnonzero(in_w & ct_ok)
    if len(iw):
        tq_w = m.to_q(tS[iw])
        Pq = Qspl_ct(tq_w)[:, :3, 3]
        Pq[~Qspl_ct.covered(tq_w, 0.0025)] = np.nan
        arrays.update({"ct_path_t_s": tS[iw],
                       "ct_path_s": np.einsum("ij,njk,kl->nil", f.X, S["CT"][iw],
                                              f.Y["CT"])[:, :3, 3],
                       "ct_path_q": Pq})
    if res.speed_ct is not None:
        sS, sQ = res.speed_ct
        arrays.update({"omega_s_t": sS.t_mid, "omega_s": sS.w,
                       "omega_q_t_master": m.to_s(sQ.t_mid), "omega_q": sQ.w})
    for r in res.xcorr:
        if "lags_s" in r:
            arrays[f"xcorr_{r['window']}_lags_s"] = np.asarray(r["lags_s"])
            arrays[f"xcorr_{r['window']}_rho"] = np.asarray(r["rho_curve"])
    sync["clock_primary"] = primary
    return ({"sync": sync, "sync_methods": methods, "registration": registration,
             "heldout": heldout, "sliding_lag": sliding, "windows": win,
             "clock_variants": pd.DataFrame(var_rows), "clock_criteria": pd.DataFrame(crit_rows)},
            arrays)


@stage("ref_sync", version="3", group="validation",
       description="V-D2/D3: JTS-MM ↔ Qualisys clock model (edges, |ω| xcorr, joint fine "
                   "fit with the registration), held-out residuals, sliding-window lags")
def ref_sync_stage(ctx: ParticipantContext) -> StageResult:
    p = {**DEFAULTS, **ctx.params}
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    ref = participant_reference(doc, ctx.pid, ctx.motion_path())
    bodies = dict(doc.get("bodies") or {"CT": "ct", "MP": "mta", "HP": "cra"})
    tables, arrays = run_ref_sync(ctx.session("motion"), ref, bodies, dict(doc.get("sync") or {}),
                                  p, ctx.pid)
    return StageResult(tables=tables, arrays=arrays)


def _validation_inputs(ctx: ParticipantContext) -> tuple[dict, dict, dict, dict]:
    p = {**DEFAULTS, **ctx.params}
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    ref = participant_reference(doc, ctx.pid, ctx.motion_path())
    bodies = dict(doc.get("bodies") or {"CT": "ct", "MP": "mta", "HP": "cra"})
    sc = dict(doc.get("sync") or {})
    kp = {k: p[k] for k in ("exposure_s", "trim_s", "sync_pad_s")}
    kp.update({k: sc[k] for k in ("start_phase", "end_phase") if k in sc})
    return p, ref, bodies, kp


def _systems(ctx: ParticipantContext):
    from biomech.validation.evaluate import load_systems

    p, ref, bodies, kp = _validation_inputs(ctx)
    sync = ctx.upstream("ref_sync", "sync").iloc[0]
    sy = load_systems(ctx.session("motion"), ref, bodies, sync, kp, ctx.pid)
    return sy, p, ref, bodies, kp, sync


@stage("ref_tip", depends=["ref_sync"], version="2", group="validation",
       description="V7 (R5c.4): both systems' landmarks (same still-hold rule), the CT tip in "
                   "the Qualisys CT frame from shared / hold-pivot / cross-system / distance "
                   "estimates, the decision, CT geometry and the digitization QC")
def ref_tip_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.evaluate import run_ref_tip

    sy, p, ref, bodies, kp, sync = _systems(ctx)
    doc = load_reference(reference_path(ctx.config, p["reference"]))
    piv = {k: (ctx.config.data_dir / str(v)).resolve()
           for k, v in (doc.get("ct_pivot") or {}).items() if v}
    tables, arrays = run_ref_tip(sy, ctx.upstream("ref_sync", "registration"), pivot=piv)
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_quality", depends=["ref_sync", "ref_tip"], version="1", group="validation",
       description="V8/V9 (R5d): reference-internal quality (rigid-body consistency, gate, "
                   "viewing direction, steps, calibration, availability) and σ_ref of the "
                   "anatomical-frame outcome")
def ref_quality_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.evaluate import run_ref_quality

    sy, p, ref, *_ = _systems(ctx)
    tables, arrays = run_ref_quality(sy, ref, ctx.upstream("ref_tip", "landmarks"),
                                     ctx.upstream("ref_tip", "tip_decision").iloc[0])
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_kinematics", depends=["ref_sync", "ref_tip", "ref_quality"], version="4",
       group="validation",
       description="V5 (R5c): jaw-in-head agreement between each system's own anatomical "
                   "frames at the lower incisor, condyle points and rotation; per segment "
                   "with/without the gate; clock and tip robustness; error budget; "
                   "tracking-only decomposition (leave-one-task-out Y); attribution")
def ref_kinematics_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.evaluate import run_ref_kinematics

    sy, p, ref, bodies, kp, sync = _systems(ctx)
    qa = ctx.upstream_arrays("ref_quality")
    tables, arrays = run_ref_kinematics(
        sy, ctx.session("motion"), ref, bodies, sync, ctx.upstream("ref_sync", "registration"),
        ctx.upstream("ref_sync", "clock_variants"), ctx.upstream("ref_tip", "landmarks"),
        ctx.upstream("ref_quality", "sigma_ref"), qa.get("gate_at_s"), qa.get("dev_max_at_s"),
        kp, chain=dict(ctx.config.signals.get("kinematics") or {}))   # VD12: the dataset chain
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_landmarks", depends=["ref_sync", "ref_tip", "ref_quality", "ref_kinematics"],
       version="1", group="validation",
       description="VD4: point-marking accuracy of JTS-MM against Qualisys (trueness with "
                   "CIs, static precision of both systems, frame-free distances, anatomical "
                   "frame errors and their incisor offset at rest, σ_ref per landmark)")
def ref_landmarks_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.landmarks import run_ref_landmarks

    from biomech.io.reference import read_qca_txt
    from biomech.validation.evaluate import _cal_path

    sy, p, ref, *_ = _systems(ctx)
    calp = _cal_path(ref)
    cams = ({k: np.array([v["x_mm"], v["y_mm"], v["z_mm"]])
             for k, v in read_qca_txt(calp).cameras.items()} if calp is not None else {})
    tables, arrays = run_ref_landmarks(
        sy, ctx.upstream("ref_sync", "registration"),
        ctx.upstream("ref_tip", "tip_decision").iloc[0], ctx.upstream("ref_tip", "tip_estimates"),
        ctx.upstream("ref_quality", "consistency"), ctx.upstream("ref_kinematics", "frame_offsets"),
        cams)
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_agreement", depends=["ref_kinematics"], version="3", group="validation",
       description="§5/D4: cycles on the reference signal; per task Bland–Altman, CCC, "
                   "ICC(A,1), r and equivalence of cycle features; waveform SPM (bars), "
                   "mean-difference bands against the metrology and application bounds, CMC")
def ref_agreement_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation import selection as SEL
    from biomech.validation.agreement import run_ref_agreement

    arr = ctx.upstream_arrays("ref_kinematics")
    t = np.asarray(arr["t_master_s"], float)
    ct_ok = np.asarray(arr["ct_visible"], bool)
    p = {**DEFAULTS, **ctx.params}
    held = SEL.movement_intervals(ctx.session("motion").phases, t, ct_ok, trim_s=p["trim_s"])
    seed = int(ctx.config.seed or 0)
    tables, arrays = run_ref_agreement(arr, held, ctx.pid, seed=seed)
    # VD3 (lead result): the same statistics on the frame-aligned JTS-MM trajectories
    if "incisor_s_aligned_mm" in arr:
        al = dict(arr)
        al["incisor_s_mm"] = arr["incisor_s_aligned_mm"]
        al["rot_s_deg"] = arr["rot_s_aligned_deg"]
        al["rot_q_deg"] = arr.get("rot_q_aligned_deg", arr["rot_q_deg"])
        for n in ("tmj_left", "tmj_right"):
            if f"{n}_s_aligned_mm" in arr:
                al[f"{n}_s_mm"] = arr[f"{n}_s_aligned_mm"]
                al[f"{n}_q_mm"] = arr[f"{n}_q_mm"]
        t2, a2 = run_ref_agreement(al, held, ctx.pid, seed=seed)
        tables.update({f"{k}_aligned": v for k, v in t2.items()})
        arrays.update({f"aligned__{k}": v for k, v in a2.items()})
    return StageResult(tables=tables, arrays=arrays)


@stage("ref_sampling", depends=["ref_sync", "ref_tip", "ref_kinematics", "ref_agreement"],
       version="3", group="validation",
       description="VD9: what 40 Hz optics and the dataset filter chain miss -- per-cycle peak "
                   "angular velocity, incisal speed and (VD24) angular acceleration from "
                   "Qualisys at 200 Hz vs the dataset chain on Qualisys at 40 Hz and on "
                   "JTS-MM; motion spectrum at 200 Hz")
def ref_sampling_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.sampling import run_ref_sampling

    sy, *_ = _systems(ctx)
    t = ctx.upstream_arrays("ref_kinematics")["t_master_s"]
    tables, arrays = run_ref_sampling(sy, ctx.upstream("ref_tip", "landmarks"),
                                      ctx.upstream("ref_agreement", "cycles"), t,
                                      seed=int(ctx.config.seed or 0))
    return StageResult(tables=tables, arrays=arrays)


from biomech.validation import stages_ref  # noqa: E402,F401  (ref_emg, ref_trueness)


def imu_mounting_files(ctx: ParticipantContext, p: dict[str, Any]) -> dict[str, Path]:
    """Validation body -> the file holding its IMU mounting for this take: the hand-eye
    calibration in ``reference.yaml`` (``imu_mounting``: one file for both IMUs or
    ``{slot: file}``; a participant's own ``imu_mounting`` overrides it, null = none, VD14),
    else the take itself (sessions recorded since 2026-10-23 embed their mounting)."""
    from biomech.validation.fusion import DEFAULTS as FD

    doc = load_reference(reference_path(ctx.config, p["reference"]))
    entry = (doc.get("participants") or {}).get(ctx.pid) or {}
    im = entry["imu_mounting"] if "imu_mounting" in entry else doc.get("imu_mounting")
    if im:
        files = im if isinstance(im, dict) else {s: im for s in FD["imu_slots"].values()}
        return {n: ctx.config.data_dir / str(files[sl]) for n, sl in FD["imu_slots"].items()
                if files.get(sl) and (ctx.config.data_dir / str(files[sl])).exists()}
    if "imu_mounting" not in entry and (ctx.motion_path() / "session.h5").exists():
        return {n: ctx.motion_path() / "session.h5" for n in FD["imu_slots"]}
    return {}


@stage("ref_fusion", depends=["ref_sync", "ref_tip", "ref_quality"], version="1",
       group="validation",
       description="V4 (secondary, VD8): optical + IMU fusion (F1 complementary, F2 ES-KF with "
                   "smoother) vs optical only on the anatomical-frame metric, leave-one-task-out"
                   " calibration on JTS-MM optical, frames and 200-Hz reference times, "
                   "simulated dropouts, τ curve, diagnostics")
def ref_fusion_stage(ctx: ParticipantContext) -> StageResult:
    from biomech.validation.evaluate import run_ref_fusion

    from biomech.validation.fusion import DEFAULTS as FD
    from biomech.validation.fusion import load_mounting

    sy, p, *_ = _systems(ctx)
    qa = ctx.upstream_arrays("ref_quality")
    fp = dict(ctx.params.get("fusion", {}))
    mounting = {n: load_mounting(f, FD["imu_slots"][n])
                for n, f in imu_mounting_files(ctx, p).items()}
    mounting = {n: R for n, R in mounting.items() if R is not None}
    if mounting:
        fp["imu_mounting"] = mounting
    tables, arrays = run_ref_fusion(sy, ctx.session("motion"), ctx.upstream("ref_tip", "landmarks"),
                                    qa.get("gate_at_s"), fp)
    return StageResult(tables=tables, arrays=arrays)
