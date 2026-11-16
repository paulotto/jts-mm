# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The work of the validation stages ``ref_tip``, ``ref_quality`` and ``ref_kinematics``
(validation plan V5–V9; master plan 4.2, 4.7, 4.9, 4.10).

* ``ref_tip``: both systems' landmarks with the same still-hold rule; the CT tip in the
  Qualisys CT frame from all data-based estimates, the V7 decision, the digitization QC.
* ``ref_quality``: the reference-internal metrics (consistency, gate, viewing direction,
  steps, calibration, availability) and σ_ref of the outcome (V8/V9).
* ``ref_kinematics``: the primary outcome -- each system's jaw-in-head motion between its
  own anatomical frames, compared at the JTS-MM frame times under the chosen clock --
  per segment, with and without the gate, robustness to the clock variants and the tip
  variant, the error budget, the secondary tracking-only analysis (leave-one-task-out Y)
  and the descriptive attribution of the disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from biomech.kinematics import se3
from biomech.validation import anatomy as AN
from biomech.validation import quality as QY
from biomech.validation import registration as REG
from biomech.validation import selection as SEL
from biomech.validation import tip as TP

BODIES = ("HP", "MP")


@dataclass
class Systems:
    pid: str
    tS: np.ndarray                     # JTS-MM exposure-centre times (master s)
    S: dict[str, np.ndarray]           # JTS-MM poses (CT is the tip pose)
    quality_s: dict[str, dict[str, np.ndarray]]
    qtm: Any                           # io.reference.QtmData
    t_qm: np.ndarray                   # Qualisys frame times on the master clock
    model: Any                         # sync.TimeModel (primary clock)
    phases: list
    windows: list
    held: list
    meta: dict[str, Any]

    def q_at_s(self, body: str, model: Any | None = None, margin_s: float = 0.0025
               ) -> np.ndarray:
        """Qualisys pose of ``body`` at the JTS-MM frame times (spline; NaN outside runs)."""
        from biomech.validation.sync import PoseSpline

        m = model or self.model
        spl = PoseSpline(self.qtm.t_s, self.qtm.pose(body))
        tq = m.to_q(self.tS)
        A = spl(tq)
        A[~spl.covered(tq, margin_s)] = np.nan
        return A


def time_model(row: pd.Series | dict[str, Any]):
    from biomech.validation.sync import TimeModel

    return TimeModel(float(row["offset_s"]), float(row["alpha_ppm"]) * 1e-6,
                     float(row["t_ref_s"]))


def load_systems(session: Any, ref: dict[str, Any], bodies: dict[str, str],
                 sync_row: pd.Series, params: dict[str, Any], pid: str) -> Systems:
    from biomech.io.reference import read_qtm_mat
    from biomech.validation import sync as SC

    p = {"exposure_s": 0.012, "trim_s": 0.5, "sync_pad_s": 0.5,
         "start_phase": "mand_point_1", "end_phase": "free_motion", **params}
    opt = session.optical()
    if opt is None:
        raise LookupError("comparison take has no optical group")
    tS = SC.frame_times(opt.t_us, p["exposure_s"])
    S = {q: opt.poses[j] for q, j in bodies.items() if j in opt.poses}
    qs = {q: opt.quality.get(j, {}) for q, j in bodies.items()}
    qtm = read_qtm_mat(ref["qtm_mat"])
    model = time_model(sync_row)
    ct_ok = se3.valid(S["CT"])
    windows = SEL.sync_windows(session.phases, tS, ct_ok, start_phase=p["start_phase"],
                               end_phase=p["end_phase"], pad_s=p["sync_pad_s"])
    held = SEL.movement_intervals(session.phases, tS, ct_ok, trim_s=p["trim_s"])
    return Systems(pid, tS, S, qs, qtm, model.to_s(qtm.t_s), model, list(session.phases),
                   windows, held, dict(getattr(session, "meta", {}) or {}))


def pose_from_row(r: pd.Series) -> np.ndarray:
    return REG.make_pose(Rotation.from_rotvec(np.radians(
        [r["rx_deg"], r["ry_deg"], r["rz_deg"]])).as_matrix(),
        np.array([r["tx_mm"], r["ty_mm"], r["tz_mm"]], float))


# -- ref_tip ---------------------------------------------------------------------------------
def qualisys_holds(sy: Systems, d: np.ndarray,
                   windows: dict[str, tuple[float, float]] | None = None) -> dict[str, AN.Hold]:
    """Qualisys landmarks with the tip ``d`` (mm, Qualisys CT frame): on its own still
    frames, or on the common still ``windows`` (EV2)."""
    T_ct = sy.qtm.pose("CT")
    tip = se3.apply(T_ct, d)
    return AN.digitize(sy.t_qm, tip, sy.qtm.pose("HP"), sy.qtm.pose("MP"), sy.phases,
                       windows=windows)


def landmark_holds(sy: Systems, d: np.ndarray
                   ) -> tuple[dict[str, AN.Hold], dict[str, AN.Hold], dict[str, tuple]]:
    """Both systems' landmarks on the window both were still in (EV2: the same physical
    event): JTS-MM (its CT pose is the tip) and Qualisys with the tip ``d``; the still runs
    are found per system first (Qualisys with its own CT origin). Returns
    ``(holds_jtsmm, holds_qualisys, common_windows)``."""
    holds_s_own = AN.digitize(sy.tS, sy.S["CT"][:, :3, 3], sy.S["HP"], sy.S["MP"], sy.phases)
    common = AN.common_windows(holds_s_own, qualisys_holds(sy, np.zeros(3)))
    holds_s = AN.digitize(sy.tS, sy.S["CT"][:, :3, 3], sy.S["HP"], sy.S["MP"], sy.phases,
                          windows=common)
    return holds_s, qualisys_holds(sy, d, common), common


def _rel_ct(T_body: np.ndarray, T_ct: np.ndarray, idx: np.ndarray) -> np.ndarray:
    return np.einsum("nij,njk->nik", se3.inv(T_body[idx]), T_ct[idx])


def pivot_record(pivot: dict[str, Any], take_geometry: dict[str, float],
                 take_template: Any = None) -> dict[str, Any]:
    """The pivot calibrations (R5c.4 options 2-4): the Qualisys tip in its CT body frame and
    whether the pivot recording used the same QTM body definition as the take (origin
    relative to the marker centroid within 1 mm)."""
    import json

    from biomech.io.reference import read_qtm_mat

    out: dict[str, Any] = {}
    if pivot.get("qualisys") and Path(pivot["qualisys"]).is_file():
        q = json.loads(Path(pivot["qualisys"]).read_text())
        out.update(q_tip=np.asarray(q["tip_body_mm"], float), q_rms_mm=float(q["rms_mm"]),
                   q_n=int(q["n_poses"]), q_cond=float(q.get("cond_A", np.nan)))
    if pivot.get("jtsmm") and Path(pivot["jtsmm"]).is_file():
        j = json.loads(Path(pivot["jtsmm"]).read_text())
        out.update(s_tip=np.asarray(j["tip_body_mm"], float), s_rms_mm=float(j["rms_mm"]),
                   s_n=int(j["n_poses"]))
    if pivot.get("qualisys_mat") and Path(pivot["qualisys_mat"]).is_file():
        qm = read_qtm_mat(pivot["qualisys_mat"])
        g = TP.ct_geometry({k: qm.markers[k] for k in qm.body_markers("CT")}, qm.pose("CT"))
        cen_p = np.array([g.get(f"centroid_{a}_mm", np.nan) for a in "xyz"], float)
        cen_t = np.array([take_geometry.get(f"centroid_{a}_mm", np.nan) for a in "xyz"], float)
        out["body_definition_diff_mm"] = float(np.linalg.norm(cen_p - cen_t))
        out["same_body_definition"] = bool(np.isfinite(out["body_definition_diff_mm"])
                                           and out["body_definition_diff_mm"] < 1.0)
        # the full marker definition (correctly labelled frames, EV1), marker by marker
        tp = qm.label_info.get("CT", {}).get("template_mm")
        if tp is not None and take_template is not None:
            out["template_diff_max_mm"] = float(np.max(np.linalg.norm(
                np.asarray(tp) - np.asarray(take_template), axis=1)))
    return out


def run_ref_tip(sy: Systems, registration: pd.DataFrame,
                pivot: dict[str, Any] | None = None
                ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    pid = sy.pid
    reg = registration.set_index("transform")
    X = pose_from_row(reg.loc["X"])
    Y_ct = pose_from_row(reg.loc["Y_CT"])
    se_t = float(reg.loc["Y_CT"].get("se_trans_mm_ac", np.nan))
    shared = TP.shared_tip(Y_ct, np.eye(3) * se_t ** 2 if np.isfinite(se_t) else None)

    holds_s, holds_q0, common = landmark_holds(sy, np.zeros(3))
    T_qct = sy.qtm.pose("CT")
    Tq = {"hp": sy.qtm.pose("HP"), "mp": sy.qtm.pose("MP")}
    Ts = {"hp": sy.S["HP"], "mp": sy.S["MP"]}
    main = [n for n in AN.HOLDS if n in holds_q0]
    est: list[TP.TipEstimate] = []
    rows = []

    def with_boot(e: TP.TipEstimate, fit, groups) -> TP.TipEstimate:
        e.cov = TP.hold_bootstrap(fit, groups, seed=7)       # holds are the sampling units
        return e

    g_q = [_rel_ct(Tq[AN.BODY_OF[n]], T_qct, holds_q0[n].idx) for n in main]
    try:
        est.append(with_boot(TP.hold_pivot(g_q, "hold_pivot_qualisys"),
                             lambda g: TP.hold_pivot(g).d, g_q))
    except ValueError:
        pass
    g_s = [_rel_ct(Ts[AN.BODY_OF[n]], sy.S["CT"], holds_s[n].idx)
           for n in AN.HOLDS if n in holds_s]
    try:                                    # JTS-MM: residual correction of its own tip
        piv_s = with_boot(TP.hold_pivot(g_s, "hold_pivot_jtsmm"),
                          lambda g: TP.hold_pivot(g).d, g_s)
    except ValueError:
        piv_s = None
    # tooth holds: JTS-MM tip mapped by X vs the Qualisys CT, on the common still
    # frames, with a free world offset c; per-hold residuals
    Tq_ct_s = sy.q_at_s("CT")
    pS = se3.apply(X, sy.S["CT"][:, :3, 3])
    g_t = {n: (Tq_ct_s[holds_s[n].idx, :3, :3], Tq_ct_s[holds_s[n].idx, :3, 3],
               pS[holds_s[n].idx]) for n in AN.HOLDS if n in holds_s}
    hold_res: dict[str, np.ndarray] = {}
    try:
        e_t, hold_res = TP.tooth_holds(g_t)
        est.append(with_boot(e_t, lambda g: TP.tooth_holds({n: g_t[n] for n in g})[0].d,
                             list(g_t)))
    except ValueError:
        pass
    # frame-free distances: Qualisys landmark = R̄ d + t̄ in its body
    q_terms = {}
    for n in main:
        rel = _rel_ct(Tq[AN.BODY_OF[n]], T_qct, holds_q0[n].idx)
        rel = rel[se3.valid(rel)]
        if len(rel):
            q_terms[n] = (rel[:, :3, :3].mean(axis=0), rel[:, :3, 3].mean(axis=0),
                          AN.BODY_OF[n])
    s_pts = {n: h.point for n, h in holds_s.items()}
    names_d = sorted(set(q_terms) & set(s_pts))
    try:
        est.append(with_boot(TP.distance_tip(q_terms, s_pts),
                             lambda g: TP.distance_tip({n: q_terms[n] for n in g},
                                                       {n: s_pts[n] for n in g}).d, names_d))
    except ValueError:
        pass
    ver = TP.verify_own(est)
    dec = TP.decide(shared, est)
    geo_take = TP.ct_geometry({k: sy.qtm.markers[k] for k in sy.qtm.body_markers("CT")},
                              T_qct)
    pv = pivot_record(pivot or {}, geo_take, sy.qtm.label_info.get("CT", {}).get("template_mm"))
    if "q_tip" in pv:
        e = TP.TipEstimate("pivot_qualisys", pv["q_tip"],
                           np.eye(3) * (pv["q_rms_mm"] ** 2 / max(pv["q_n"], 1)), pv["q_cond"],
                           np.nan, pv["q_rms_mm"], pv["q_n"])
        est_pivot = e
    else:
        est_pivot = None
    if est_pivot is not None and pv.get("same_body_definition", False):
        # V7 (a): verified by the pivot recording of the same QTM body definition
        dec.update(primary="own_verified", d_primary=est_pivot.d)
        ver["verified"] = True
        ver["informative"] = ["pivot_qualisys"] + list(ver["informative"])
    elif ver["verified"]:                     # V7: the own (QTM) tip is consistent with data
        dec.update(primary="own_verified", d_primary=np.zeros(3))
    else:
        dec.update(primary="shared", d_primary=shared.d)
    if est_pivot is not None:
        est = [est_pivot] + est
    for e in [shared, *est, *([piv_s] if piv_s is not None else [])]:
        se = e.se
        rows.append({"pid": pid, "method": e.method, "dx_mm": e.d[0], "dy_mm": e.d[1],
                     "dz_mm": e.d[2], "norm_mm": float(np.linalg.norm(e.d)),
                     "se_x_mm": se[0], "se_y_mm": se[1], "se_z_mm": se[2], "cond": e.cond,
                     "rms_before_mm": e.rms_before_mm, "rms_after_mm": e.rms_after_mm,
                     "n": e.n})
    tip_meta = str(sy.meta.get("ct_tip_body_mm", ""))
    decision = pd.DataFrame([{
        "pid": pid, "primary": dec["primary"], "verified": ver["verified"],
        "informative": ",".join(ver["informative"]),
        "own_consistent": ",".join(str(x) for x in ver["own_consistent"]),
        "p_agree": dec["p_agree"], "n_estimates": dec["n_estimates"],
        "diff_to_shared_mm": dec["diff_to_shared_mm"],
        "dx_mm": dec["d_primary"][0], "dy_mm": dec["d_primary"][1], "dz_mm": dec["d_primary"][2],
        "consensus_dx_mm": dec["d_consensus"][0], "consensus_dy_mm": dec["d_consensus"][1],
        "consensus_dz_mm": dec["d_consensus"][2],
        "shared_dx_mm": shared.d[0], "shared_dy_mm": shared.d[1], "shared_dz_mm": shared.d[2],
        "jtsmm_tip_body_mm": tip_meta, "jtsmm_tip_rms_mm": sy.meta.get("ct_tip_rms_mm", ""),
        "pivot_same_body_definition": pv.get("same_body_definition", np.nan),
        "pivot_body_definition_diff_mm": pv.get("body_definition_diff_mm", np.nan),
        "pivot_template_diff_max_mm": pv.get("template_diff_max_mm", np.nan),
        "jtsmm_pivot_vs_applied_mm": (float(np.linalg.norm(pv["s_tip"] - np.array(
            [float(x) for x in tip_meta.split(",")]))) if "s_tip" in pv and tip_meta
            else np.nan),
        "jtsmm_tip_source": Path(str(sy.meta.get("ct_tip_source", ""))).name}])

    # landmarks: JTS-MM, Qualisys with the primary and the shared tip
    variants = {"primary": np.asarray(dec["d_primary"], float), "shared": shared.d,
                "own_origin": np.zeros(3)}
    lm = [AN.landmark_table(holds_s, "jtsmm", pid).assign(variant="-")]
    dist_rows = []
    ds = AN.within_body_distances(holds_s)
    for v, d in variants.items():
        if not np.isfinite(d).all():
            continue
        hq = qualisys_holds(sy, d, common)
        lm.append(AN.landmark_table(hq, "qualisys", pid).assign(variant=v))
        dq = AN.within_body_distances(hq)
        for k in sorted(set(ds) & set(dq)):
            dist_rows.append({"pid": pid, "variant": v, "a": k[0], "b": k[1],
                              "dist_jtsmm_mm": ds[k], "dist_qualisys_mm": dq[k],
                              "diff_mm": ds[k] - dq[k]})
    # frame-free digitization symmetry (incisor to left vs right first molar; both jaws)
    sym_rows = []
    for system, hv in [("jtsmm", holds_s)] + [(f"qualisys:{v}", qualisys_holds(sy, d, common))
                                               for v, d in variants.items()
                                               if np.isfinite(d).all()]:
        for jaw, (i, lft, rgt) in {"mandible": ("mand_point_1", "mand_point_2", "mand_point_3"),
                                   "maxilla": ("max_point_1", "max_point_2", "max_point_3")}.items():
            if all(k in hv for k in (i, lft, rgt)):
                dl = float(np.linalg.norm(hv[i].point - hv[lft].point))
                dr = float(np.linalg.norm(hv[i].point - hv[rgt].point))
                sym_rows.append({"pid": pid, "system": system, "jaw": jaw, "incisor_left_mm": dl,
                                 "incisor_right_mm": dr, "asymmetry_mm": dl - dr,
                                 "intermolar_mm": float(np.linalg.norm(hv[lft].point
                                                                       - hv[rgt].point))})
    info = sy.qtm.label_info.get("CT", {})
    geo = {**geo_take, "label_margin_mm": info.get("margin_mm", np.nan),
           "n_label_swapped": info.get("n_swapped", 0),
           "n_label_undetermined": info.get("n_undetermined", 0),
           "label_runs_by_continuity": info.get("runs_by_continuity", 0)}
    return ({"tip_estimates": pd.DataFrame(rows), "tip_decision": decision,
             "landmarks": (lambda t: t.assign(window=np.where(
                 t["landmark"].str.replace("_hp", "").isin(list(common)), "common", "own")))(
                 pd.concat(lm, ignore_index=True)),
             "landmark_distances": pd.DataFrame(dist_rows),
             "landmark_symmetry": pd.DataFrame(sym_rows),
             "ct_geometry": pd.DataFrame([{"pid": pid, **geo}]),
             "hold_residuals": pd.DataFrame(
                 [{"pid": pid, "landmark": n, "rx_mm": r[0], "ry_mm": r[1], "rz_mm": r[2],
                   "r_mm": float(np.linalg.norm(r)), "window": "common" if n in common else "own"}
                  for n, r in hold_res.items()])}, {})


def holds_from_table(df: pd.DataFrame, system: str, variant: str) -> dict[str, AN.Hold]:
    d = df[(df["system"] == system) & (df["variant"] == variant)]
    return {r.landmark: AN.Hold(r.landmark, r.body, np.array([r.x_mm, r.y_mm, r.z_mm]),
                                r.mad_mm, int(r.n), r.t0_s, r.t1_s) for r in d.itertuples()}


# -- ref_quality -----------------------------------------------------------------------------
def _cal_path(ref: dict[str, Any]) -> Path | None:
    c = ref.get("qtm_cal")
    if not c:
        return None
    p = Path(c)
    if p.is_dir():
        hits = sorted(p.glob("qca_*.txt"))
        return hits[0] if hits else None
    return p if p.exists() else None


def segment_list(sy: Systems) -> list[tuple[str, str, Any]]:
    """(segment, kind, interval-like with t0/t1 on the master clock)."""
    out = []
    for h in sy.held:
        out.append((h.name, "movement" if h.name in SEL.MOVEMENT_TASKS else "static", h))
    for ph in sy.phases:
        if ph.name.startswith(SEL.EXCLUDED_PREFIXES):
            out.append((ph.name, "hold", SEL.Interval(ph.name, ph.t_start_us * 1e-6,
                                                     ph.t_end_us * 1e-6)))
    for w in sy.windows:
        out.append((w.name, "sync", w))
    return out


def run_ref_quality(sy: Systems, ref: dict[str, Any], landmarks: pd.DataFrame,
                    tip_decision: pd.Series
                    ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    from biomech.io.reference import read_qca_txt

    pid = sy.pid
    tables: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {"t_qm_s": sy.t_qm}
    calp = _cal_path(ref)
    cams: dict[str, np.ndarray] = {}
    if calp is not None:
        cal = read_qca_txt(calp)
        cams = {k: np.array([v["x_mm"], v["y_mm"], v["z_mm"]]) for k, v in cal.cameras.items()}
        tables["calibration"] = pd.DataFrame(
            [{"pid": pid, "camera": k, **v, "wand_sd_mm": cal.wand_sd_mm, "when": cal.when}
             for k, v in cal.cameras.items()])
    segs = segment_list(sy)
    seg_rows, view_rows = [], []
    qual: dict[str, pd.DataFrame] = {}
    sig_m: dict[tuple[str, str], float] = {}
    for body in ("HP", "MP", "CT"):
        names = sy.qtm.body_markers(body)
        if len(names) < 3:
            continue
        q = QY.consistency({k: sy.qtm.markers[k] for k in names},
                           {k: sy.qtm.marker_residual[k] for k in names
                            if k in sy.qtm.marker_residual},
                           {k: sy.qtm.marker_type[k] for k in names if k in sy.qtm.marker_type})
        qual[body] = q
        st = QY.steps(q)
        arrays[f"{body}_dev_max_mm"] = q["dev_max_mm"].to_numpy()
        arrays[f"{body}_gate"] = q["gate"].to_numpy()
        zero = np.zeros(len(sy.t_qm), bool)
        swp = sy.qtm.label_swapped.get(body, zero)          # label correction (EV1)
        und = sy.qtm.label_swapped.get(f"{body}__undetermined", zero)
        if sy.qtm.label_info.get(body, {}).get("ambiguous"):
            arrays[f"{body}_label_swapped"] = swp
            arrays[f"{body}_label_undetermined"] = und
        T = sy.qtm.pose(body)
        views = QY.viewing_directions(T, cams) if cams else {}
        for seg, kind, iv in segs:
            k = (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
            if k.sum() < 5:
                continue
            qs = q[k]
            s = QY.sigma_marker(qs[~qs["gate"]]) if (~qs["gate"]).sum() > 5 else np.nan
            sig_m[(body, seg)] = s
            seg_rows.append({"pid": pid, "body": body, "segment": seg, "kind": kind,
                             "n": int(k.sum()), "sigma_marker_mm": s,
                             "dev_sd_mm": float(np.nanstd(qs.filter(like="dev_").drop(
                                 columns="dev_max_mm").to_numpy())),
                             "dev_max_mm": float(qs["dev_max_mm"].max()),
                             "resid_med_mm": float(qs["resid_max_mm"].median()),
                             "frac_missing": float(qs["missing"].mean()),
                             "frac_gapfilled": float(qs["gapfilled"].mean()),
                             "frac_gated": float(qs["gate"].mean()),
                             "n_steps": int(st[k].sum()),
                             "valid_6dof": float(se3.valid(T[k]).mean()),
                             "frac_label_swapped": float(swp[k].mean()),
                             "frac_label_undetermined": float(und[k].mean())})
        if views:
            mov = np.zeros(len(T), bool)
            for seg, kind, iv in segs:
                if kind == "movement":
                    mov |= (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
            vb = QY.viewing_bins(q, views, mov)
            if not vb.empty:
                view_rows.append(vb.assign(pid=pid, body=body))
    tables["consistency"] = pd.DataFrame(seg_rows)
    tables["label_correction"] = pd.DataFrame(
        [{"pid": pid, "body": b, **{k: v for k, v in info.items()
                                    if k not in ("permutations", "template_mm")},
          "permutations": str(info.get("permutations", []))}
         for b, info in sy.qtm.label_info.items()])
    if view_rows:
        tables["viewing"] = pd.concat(view_rows, ignore_index=True)

    # σ_ref of the anatomical-frame outcome per segment (Qualisys frames, primary tip)
    hq = holds_from_table(landmarks, "qualisys", "primary")
    sig_rows = []
    try:
        fr = AN.anatomical_frames(hq)
        tm = {b: QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers(b)},
                             sy.qtm.pose(b)) for b in BODIES}
        from biomech.io.reference import label_ambiguity

        tables["marker_geometry"] = pd.DataFrame(
            [{"pid": pid, "body": b, **QY.marker_geometry(v),
              "label_margin_mm": label_ambiguity(v)[0]} for b, v in tm.items()])
        rel = se3.relative(sy.qtm.pose("HP"), sy.qtm.pose("MP"))
        for seg, kind, iv in segs:
            k = (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
            sh, sj = sig_m.get(("HP", seg), np.nan), sig_m.get(("MP", seg), np.nan)
            if k.sum() < 5 or not (np.isfinite(sh) and np.isfinite(sj)):
                continue
            r = QY.sigma_ref(tm["HP"], tm["MP"], sh, sj, rel[k], fr.F_max, fr.F_mand,
                             fr.tmj_mand)
            sig_rows.append({"pid": pid, "segment": seg, "kind": kind, "sigma_head_mm": sh,
                             "sigma_jaw_mm": sj, **{f"sigma_ref_{a}": v for a, v in r.items()}})
    except (LookupError, ValueError):
        pass
    tables["sigma_ref"] = pd.DataFrame(sig_rows)

    # availability per segment, both systems (bodies HP and MP both valid)
    av_rows = []
    vs = se3.valid(sy.S["HP"]) & se3.valid(sy.S["MP"])
    vq = se3.valid(sy.qtm.pose("HP")) & se3.valid(sy.qtm.pose("MP"))
    for seg, kind, iv in segs:
        for system, t, v in (("jtsmm", sy.tS, vs), ("qualisys", sy.t_qm, vq)):
            k = (t >= iv.t0) & (t <= iv.t1)
            if k.sum():
                av_rows.append({"pid": pid, "system": system, "segment": seg, "kind": kind,
                                **QY.availability(v, t, k)})
    tables["availability"] = pd.DataFrame(av_rows)
    # the gate at the JTS-MM frame times (nearest Qualisys frame; HP or MP gated)
    gate_q = np.zeros(len(sy.t_qm), bool)
    for b in BODIES:
        if b in qual:
            gate_q |= qual[b]["gate"].to_numpy()
    j = np.clip(np.searchsorted(sy.t_qm, sy.tS), 1, len(sy.t_qm) - 1)
    j -= (np.abs(sy.t_qm[j - 1] - sy.tS) < np.abs(sy.t_qm[j] - sy.tS)).astype(int)
    arrays["gate_at_s"] = gate_q[j]
    arrays["dev_max_at_s"] = np.fmax(arrays.get("HP_dev_max_mm", np.full(len(gate_q), np.nan))[j],
                                     arrays.get("MP_dev_max_mm", np.full(len(gate_q), np.nan))[j])
    return tables, arrays


# -- ref_kinematics (V5) ---------------------------------------------------------------------
def _rms(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def outcome(sy: Systems, hs: dict[str, AN.Hold], hq: dict[str, AN.Hold], model: Any = None
            ) -> dict[str, np.ndarray]:
    """Per-frame outcome at the JTS-MM frame times: both systems' jaw-in-head motion between
    their own anatomical frames."""
    fs_, fq = AN.anatomical_frames(hs), AN.anatomical_frames(hq)
    M_S = AN.jaw_in_head(sy.S["HP"], sy.S["MP"], fs_)
    M_Q = AN.jaw_in_head(sy.q_at_s("HP", model), sy.q_at_s("MP", model), fq)
    ps, pq = AN.points_of(M_S, fs_), AN.points_of(M_Q, fq)
    out = {"M_S": M_S, "M_Q": M_Q, "inc_s": ps["incisor"], "inc_q": pq["incisor"],
           "rot_err_deg": AN.rotation_error_deg(M_S, M_Q)}
    for n in ("tmj_left", "tmj_right"):
        if n in ps and n in pq:
            out[f"{n}_s"], out[f"{n}_q"] = ps[n], pq[n]
    return out


DATASET_CHAIN = {"max_gap_s": 0.5, "lowpass_hz": 6.0, "lowpass_order": 2}


def dataset_chain(M: np.ndarray, t_s: np.ndarray, chain: dict[str, Any]) -> np.ndarray:
    """Jaw-in-head poses through the dataset study's processing (``kinematics/pose_chain``):
    SE(3) interpolation of gaps up to ``max_gap_s``, then the zero-phase Butterworth
    low-pass of the rotation vectors and translations (``se3.lowpass``)."""
    c = {**DATASET_CHAIN, **{k: v for k, v in chain.items() if k in DATASET_CHAIN}}
    fs = 1.0 / float(np.median(np.diff(t_s)))
    M, _ = se3.bridge_gaps(M, int(round(float(c["max_gap_s"]) * fs)))
    return se3.lowpass(M, fs, float(c["lowpass_hz"]), int(c["lowpass_order"]))


def filtered_agreement(sy: Systems, o: dict[str, np.ndarray], segs: list, k_pool: np.ndarray, k_rest: np.ndarray, gate: np.ndarray,
                       fold_T: dict[str, tuple[np.ndarray, np.ndarray]], mov: list,
                       chain: dict[str, Any]) -> list[dict[str, Any]]:
    """VD12 rows: per chain (both systems / JTS-MM only filtered), frames (own / aligned with
    the primary analysis's leave-one-task-out transforms) and movement segment, the gated
    agreement of :func:`segment_stats`. The unfiltered rows are ``segments`` and
    ``segments_aligned``."""
    from biomech.validation import kinematics as VK

    MSf = dataset_chain(o["M_S"], sy.tS, chain)
    MQf = dataset_chain(o["M_Q"], sy.tS, chain)
    rows: list[dict[str, Any]] = []
    for name, M_S, M_Q in (("both", MSf, MQf), ("jtsmm_only", MSf, o["M_Q"])):
        for frames in ("own", "aligned"):
            if frames == "own":
                Ms = M_S
            else:                               # the primary analysis's LOTO transforms
                Ms = np.full_like(M_S, np.nan)
                for fold, (A_, B_) in fold_T.items():
                    kh = VK.mask_of(sy.tS, [h for h in mov if h.name == fold])
                    Ms[kh] = np.einsum("ij,njk,kl->nil", A_, M_S[kh], B_)
            # the lower incisor is the origin of each mandibular frame (AN.points_of)
            oo = {"M_S": Ms, "M_Q": M_Q, "inc_s": Ms[:, :3, 3], "inc_q": M_Q[:, :3, 3],
                  "rot_err_deg": AN.rotation_error_deg(Ms, M_Q)}
            okq = k_rest & se3.valid(M_Q)
            if frames == "own":
                rr = rot_rel_rest(oo, k_rest)
            else:
                ref_q = REG.mean_pose(M_Q[okq]) if okq.sum() >= 3 else \
                    REG.mean_pose(M_Q[se3.valid(M_Q)])
                rr = (np.degrees(se3.rotvec_rel(Ms, ref_q)), np.degrees(se3.rotvec_rel(M_Q, ref_q)))
            for seg, kind, iv in segs + [("movement_pooled", "pooled", None)]:
                if kind not in ("movement", "pooled"):
                    continue
                k = (k_pool if iv is None else (sy.tS >= iv.t0) & (sy.tS <= iv.t1)) & ~gate
                st = segment_stats(oo, k, rr)
                rows.append({"pid": sy.pid, "chain": name, "frames": frames, "segment": seg,
                             "kind": kind, "n": st.get("n", 0),
                             **{c: st.get(c, np.nan) for c in FILTERED_COLUMNS}})
    return rows


FILTERED_COLUMNS = ("incisor_rms_mm", "incisor_dyn_rms_mm", "rot_rms_deg", "rot_dyn_rms_deg")


def segment_stats(o: dict[str, np.ndarray], k: np.ndarray, rot_rel: tuple[np.ndarray, np.ndarray]
                  ) -> dict[str, float]:
    e = o["inc_s"][k] - o["inc_q"][k]
    ok = np.isfinite(e).all(axis=1) & np.isfinite(o["rot_err_deg"][k])
    e = e[ok]
    r = {"n": int(ok.sum())}
    if len(e) < 3:
        return r
    bias, sd = e.mean(axis=0), e.std(axis=0, ddof=1)
    r.update({f"incisor_bias_{a}_mm": bias[i] for i, a in enumerate("xyz")})
    r.update({f"incisor_sd_{a}_mm": sd[i] for i, a in enumerate("xyz")})
    r["incisor_rms_mm"] = _rms(np.linalg.norm(e, axis=1))
    r["incisor_dyn_rms_mm"] = float(np.sqrt(np.sum(sd ** 2)))      # bias removed
    r["rot_rms_deg"] = _rms(o["rot_err_deg"][k][ok])
    rs, rq = rot_rel[0][k][ok], rot_rel[1][k][ok]
    dr = rs - rq
    r.update({f"rot_bias_r{a}_deg": float(np.nanmean(dr[:, i])) for i, a in enumerate("xyz")})
    r["rot_dyn_rms_deg"] = float(np.sqrt(np.nansum(np.nanvar(dr, axis=0, ddof=1))))
    rom = np.nanpercentile(o["inc_q"][k][ok], 95, axis=0) - \
        np.nanpercentile(o["inc_q"][k][ok], 5, axis=0)
    r["incisor_rom_mm"] = float(np.linalg.norm(rom))
    r["incisor_rel_pct"] = 100.0 * r["incisor_rms_mm"] / r["incisor_rom_mm"] \
        if r["incisor_rom_mm"] > 0 else np.nan
    for n in ("tmj_left", "tmj_right"):
        if f"{n}_s" in o:
            et = o[f"{n}_s"][k][ok] - o[f"{n}_q"][k][ok]
            r[f"{n}_rms_mm"] = _rms(np.linalg.norm(et, axis=1))
            r[f"{n}_dyn_rms_mm"] = float(np.sqrt(np.nansum(np.nanvar(et, axis=0, ddof=1))))
    return r


def rot_rel_rest(o: dict[str, np.ndarray], k_rest: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation vectors (deg) of each system relative to its own mean rest pose."""
    out = []
    for M in (o["M_S"], o["M_Q"]):
        ok = k_rest & se3.valid(M)
        ref = REG.mean_pose(M[ok]) if ok.sum() >= 3 else REG.mean_pose(M[se3.valid(M)])
        out.append(np.degrees(se3.rotvec_rel(M, ref)))
    return out[0], out[1]


def aligned_pooled(sy: Systems, o: dict[str, np.ndarray], mov: list, gate: np.ndarray
                   ) -> dict[str, float]:
    """Pooled movement-task agreement (gated) of outcome ``o`` after the constant frame
    alignment fitted leave-one-task-out on ``o`` itself (the ``segments_aligned`` rule,
    incisor and rotation only): ``incisor_aligned_rms_mm``, ``rot_aligned_rms_deg``. Used to
    re-score the clock and tip variants (VD19) in frames aligned."""
    from biomech.validation import kinematics as VK

    okM = se3.valid(o["M_S"]) & se3.valid(o["M_Q"]) & ~gate
    M_al = np.full_like(o["M_S"], np.nan)
    for h in mov:
        ktr = VK.mask_of(sy.tS, [m for m in mov if m is not h]) & okM
        if ktr.sum() < 50:
            continue
        try:
            A0, B0 = REG.init_xy(o["M_Q"][ktr], o["M_S"][ktr])
            rg = REG.register_paired({"jaw": (o["M_Q"][ktr], o["M_S"][ktr])}, A0, {"jaw": B0})
        except (ValueError, np.linalg.LinAlgError):
            continue
        kh = VK.mask_of(sy.tS, [h])
        M_al[kh] = np.einsum("ij,njk,kl->nil", rg.X, o["M_S"][kh], rg.Y["jaw"])
    k = VK.mask_of(sy.tS, mov) & ~gate
    e = np.linalg.norm(M_al[k, :3, 3] - o["inc_q"][k], axis=1)
    r = AN.rotation_error_deg(M_al[k], o["M_Q"][k])
    ok = np.isfinite(e) & np.isfinite(r)
    return {"incisor_aligned_rms_mm": _rms(e[ok]) if ok.sum() >= 3 else np.nan,
            "rot_aligned_rms_deg": _rms(r[ok]) if ok.sum() >= 3 else np.nan}


def run_ref_kinematics(sy: Systems, session: Any, ref: dict[str, Any], bodies: dict[str, str],
                       sync_row: pd.Series, registration: pd.DataFrame,
                       clock_variants: pd.DataFrame, landmarks: pd.DataFrame,
                       sigma_ref: pd.DataFrame, gate_at_s: np.ndarray | None,
                       dev_at_s: np.ndarray | None, params: dict[str, Any],
                       chain: dict[str, Any] | None = None
                       ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """... ``chain``: the dataset study's pose processing (``signals.kinematics``: gap
    bridging and zero-phase low-pass) for the VD12 sensitivity ``segments_filtered``."""
    from biomech.validation import kinematics as VK

    pid = sy.pid
    hs = holds_from_table(landmarks, "jtsmm", "-")
    hq = holds_from_table(landmarks, "qualisys", "primary")
    o = outcome(sy, hs, hq)
    segs = segment_list(sy)
    mov = [h for h in sy.held if h.name in SEL.MOVEMENT_TASKS]
    rest = [h for h in sy.held if h.name in SEL.STATIC_TASKS]
    k_rest = VK.mask_of(sy.tS, rest) if rest else VK.mask_of(sy.tS, mov[:1])
    rr = rot_rel_rest(o, k_rest)
    gate = np.zeros(len(sy.tS), bool) if gate_at_s is None else np.asarray(gate_at_s, bool)
    k_pool = VK.mask_of(sy.tS, mov)

    rows = []
    for gated in (False, True):
        keep = ~gate if gated else np.ones(len(sy.tS), bool)
        for seg, kind, iv in segs + [("movement_pooled", "pooled", None)]:
            k = (k_pool if iv is None else (sy.tS >= iv.t0) & (sy.tS <= iv.t1)) & keep
            rows.append({"pid": pid, "segment": seg, "kind": kind, "gated": gated,
                         **segment_stats(o, k, rr)})
    seg_tab = pd.DataFrame(rows)

    # robustness: the pooled primary outcome under every clock variant and tip variant
    rob = []
    for r in clock_variants.itertuples():
        m = time_model({"offset_s": r.offset_s, "alpha_ppm": r.alpha_ppm,
                        "t_ref_s": sync_row["t_ref_s"]})
        ov = outcome(sy, hs, hq, m)
        st = segment_stats(ov, k_pool & ~gate, rot_rel_rest(ov, k_rest))
        rob.append({"pid": pid, "factor": "clock", "variant": r.variant,
                    "incisor_rms_mm": st.get("incisor_rms_mm", np.nan),
                    "incisor_dyn_rms_mm": st.get("incisor_dyn_rms_mm", np.nan),
                    "rot_rms_deg": st.get("rot_rms_deg", np.nan),
                    **aligned_pooled(sy, ov, mov, gate)})     # VD19: also frames aligned
    for v in ("shared", "own_origin"):
        hv = holds_from_table(landmarks, "qualisys", v)
        if len(hv) >= 6:
            ov = outcome(sy, hs, hv)
            st = segment_stats(ov, k_pool & ~gate, rot_rel_rest(ov, k_rest))
            rob.append({"pid": pid, "factor": "tip", "variant": v,
                        "incisor_rms_mm": st.get("incisor_rms_mm", np.nan),
                        "incisor_dyn_rms_mm": st.get("incisor_dyn_rms_mm", np.nan),
                        "rot_rms_deg": st.get("rot_rms_deg", np.nan),
                        **aligned_pooled(sy, ov, mov, gate)})

    # error budget (V9): σ_obs = bias-removed RMS of the gated segment vs σ_ref
    bud = []
    sr = sigma_ref.set_index("segment") if not sigma_ref.empty else pd.DataFrame()
    g = seg_tab[seg_tab["gated"]].set_index("segment").copy()
    # condyle: σ_ref pools both condyles (quality.sigma_ref), so σ_obs does too (RMS of the
    # left and right bias-removed RMS); a missing side leaves the other one
    tm = [c for c in ("tmj_left_dyn_rms_mm", "tmj_right_dyn_rms_mm") if c in g.columns]
    if tm:
        g["tmj_dyn_rms_mm"] = np.sqrt(np.nanmean(g[tm].to_numpy(float) ** 2, axis=1))
    for seg in g.index:
        if seg not in sr.index:
            continue
        for oc, obs_col, ref_col in (("incisor", "incisor_dyn_rms_mm", "sigma_ref_incisor_mm"),
                                     ("rotation", "rot_dyn_rms_deg", "sigma_ref_rot_deg"),
                                     ("condyle", "tmj_dyn_rms_mm", "sigma_ref_tmj_mm")):
            if obs_col in g.columns and ref_col in sr.columns:
                b = QY.budget(float(g.loc[seg, obs_col]), float(sr.loc[seg, ref_col]))
                bud.append({"pid": pid, "segment": seg, "kind": g.loc[seg, "kind"],
                            "outcome": oc, **b})

    # secondary (VD3): frame-aligned agreement. The two systems' anatomical frames differ by
    # constant rigid transforms (their digitizations of the same teeth differ): M_Q ≈ A M_S B
    # with A (maxillary) and B (mandibular) fitted on the *other* movement tasks
    # (leave-one-task-out; the scored task never enters), robust closed form + refinement.
    al_rows, off_rows = [], []
    al = {"inc_s": np.full((len(sy.tS), 3), np.nan), "M_S": np.full_like(o["M_S"], np.nan)}
    for t_ in ("tmj_left", "tmj_right"):
        if f"{t_}_s" in o:
            al[f"{t_}_s"] = np.full((len(sy.tS), 3), np.nan)
    okM = se3.valid(o["M_S"]) & se3.valid(o["M_Q"]) & ~gate
    fr_s = AN.anatomical_frames(hs)
    fold_T: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name_, train in [(h.name, [m for m in mov if m is not h]) for h in mov] + \
            [("insample", list(mov))]:
        ktr = VK.mask_of(sy.tS, train) & okM
        if ktr.sum() < 50:
            continue
        try:
            A0, B0 = REG.init_xy(o["M_Q"][ktr], o["M_S"][ktr])
            rg = REG.register_paired({"jaw": (o["M_Q"][ktr], o["M_S"][ktr])}, A0, {"jaw": B0})
            A_, B_ = rg.X, rg.Y["jaw"]
        except (ValueError, np.linalg.LinAlgError):
            continue
        for nm_, T_ in (("A_maxillary", A_), ("B_mandibular", B_)):
            dp_, ang_ = REG.pose_error(T_[None], np.eye(4)[None])
            off_rows.append({"pid": pid, "fold": name_, "transform": nm_,
                             "rot_deg": float(ang_[0]), "trans_mm": float(np.linalg.norm(dp_))})
        if name_ == "insample":
            continue
        fold_T[name_] = (A_, B_)
        kh = VK.mask_of(sy.tS, [h for h in mov if h.name == name_])
        Mal = np.einsum("ij,njk,kl->nil", A_, o["M_S"][kh], B_)
        al["M_S"][kh] = Mal
        al["inc_s"][kh] = Mal[:, :3, 3]
        for t_ in ("tmj_left", "tmj_right"):
            if f"{t_}_s" in al and t_ in fr_s.tmj_mand:
                # the condyle point of the *Qualisys* mandibular frame, carried by both
                al[f"{t_}_s"][kh] = se3.apply(Mal, AN.anatomical_frames(hq).tmj_mand[t_])
    o_al = {"M_S": al["M_S"], "M_Q": o["M_Q"], "inc_s": al["inc_s"], "inc_q": o["inc_q"],
            "rot_err_deg": AN.rotation_error_deg(al["M_S"], o["M_Q"])}
    for t_ in ("tmj_left", "tmj_right"):
        if f"{t_}_s" in al:
            o_al[f"{t_}_s"], o_al[f"{t_}_q"] = al[f"{t_}_s"], o[f"{t_}_q"]
    # after alignment both systems live in the Qualisys anatomical frames: one shared rest
    # reference (Qualisys rest pose; rest itself is in no alignment fold)
    okq = k_rest & se3.valid(o["M_Q"])
    ref_q = REG.mean_pose(o["M_Q"][okq]) if okq.sum() >= 3 else \
        REG.mean_pose(o["M_Q"][se3.valid(o["M_Q"])])
    rr_al = (np.degrees(se3.rotvec_rel(al["M_S"], ref_q)),
             np.degrees(se3.rotvec_rel(o["M_Q"], ref_q)))
    for seg, kind, iv in segs + [("movement_pooled", "pooled", None)]:
        if kind not in ("movement", "pooled"):
            continue
        k = (k_pool if iv is None else (sy.tS >= iv.t0) & (sy.tS <= iv.t1)) & ~gate
        al_rows.append({"pid": pid, "segment": seg, "kind": kind,
                        **segment_stats(o_al, k, rr_al)})

    # sensitivity (VD7): the frame offsets from one fixed calibration segment instead of
    # leave-one-task-out -- fitted on free_motion alone (never scored), applied to the four
    # protocol tasks
    fm_rows = []
    fm = [h for h in mov if h.name == "free_motion"]
    ktr = (VK.mask_of(sy.tS, fm) & okM) if fm else np.zeros(len(sy.tS), bool)
    if ktr.sum() >= 50:
        try:
            A0, B0 = REG.init_xy(o["M_Q"][ktr], o["M_S"][ktr])
            rg = REG.register_paired({"jaw": (o["M_Q"][ktr], o["M_S"][ktr])}, A0, {"jaw": B0})
            A_, B_ = rg.X, rg.Y["jaw"]
        except (ValueError, np.linalg.LinAlgError):
            A_ = None
        if A_ is not None:
            for nm_, T_ in (("A_maxillary", A_), ("B_mandibular", B_)):
                dp_, ang_ = REG.pose_error(T_[None], np.eye(4)[None])
                off_rows.append({"pid": pid, "fold": "free_motion_only", "transform": nm_,
                                 "rot_deg": float(ang_[0]), "trans_mm": float(np.linalg.norm(dp_))})
            Mfm = np.einsum("ij,njk,kl->nil", A_, o["M_S"], B_)
            o_fm = {"M_S": Mfm, "M_Q": o["M_Q"], "inc_s": Mfm[:, :3, 3], "inc_q": o["inc_q"],
                    "rot_err_deg": AN.rotation_error_deg(Mfm, o["M_Q"])}
            for t_ in ("tmj_left", "tmj_right"):
                if f"{t_}_q" in o and t_ in AN.anatomical_frames(hq).tmj_mand:
                    o_fm[f"{t_}_s"] = se3.apply(Mfm, AN.anatomical_frames(hq).tmj_mand[t_])
                    o_fm[f"{t_}_q"] = o[f"{t_}_q"]
            rr_fm = (np.degrees(se3.rotvec_rel(Mfm, ref_q)), rr_al[1])
            proto = [h for h in mov if h.name != "free_motion"]
            for seg, kind, iv in segs + [("protocol_pooled", "pooled", None)]:
                if kind not in ("movement", "pooled") or seg == "free_motion":
                    continue
                k = (VK.mask_of(sy.tS, proto) if iv is None
                     else (sy.tS >= iv.t0) & (sy.tS <= iv.t1)) & ~gate
                fm_rows.append({"pid": pid, "segment": seg, "kind": kind,
                                **segment_stats(o_fm, k, rr_fm)})

    # sensitivity (VD12): the same agreement after the dataset study's pose chain (gap
    # bridging, zero-phase low-pass), applied to both systems ("both": the measurement error
    # of the processed signal) and to JTS-MM alone ("jtsmm_only": the processed signal
    # against the unfiltered reference, i.e. noise reduction and filter attenuation together).
    filt_rows = filtered_agreement(sy, o, segs, k_pool, k_rest, gate, fold_T, mov,
                                   chain or {}) if chain is not None else []

    # secondary: tracking-only (leave-one-task-out Y between the body frames; R5b)
    reg_tab = registration
    try:
        sec_t, sec_a = VK.run_ref_kinematics(session, ref, bodies, sync_row, reg_tab, params, pid)
    except (LookupError, ValueError):
        sec_t, sec_a = {}, {}
    decomp = []
    if "segments" in sec_t:
        lo = sec_t["segments"]
        lo = lo[lo["variant"] == "loto"].set_index("segment")
        p0 = seg_tab[~seg_tab["gated"]].set_index("segment")
        for seg in p0.index.intersection(lo.index):
            decomp.append({"pid": pid, "segment": seg,
                           "end_to_end_incisor_rms_mm": p0.loc[seg, "incisor_rms_mm"],
                           "tracking_only_incisor_rms_mm": lo.loc[seg, "incisor_rms_mm"],
                           "end_to_end_rot_rms_deg": p0.loc[seg, "rot_rms_deg"],
                           "tracking_only_rot_rms_deg": lo.loc[seg, "rot_rms_deg"]})

    # attribution (descriptive): the per-frame incisor error against quality covariates
    err = np.linalg.norm(o["inc_s"] - o["inc_q"], axis=1)
    cov = {"q_dev_max_mm": dev_at_s if dev_at_s is not None else np.full(len(err), np.nan),
           "ct_visible": se3.valid(sy.S["CT"]).astype(float)}
    for b in BODIES:
        for f in ("n_tags", "reprojection_rms"):
            v = sy.quality_s.get(b, {}).get(f)
            if v is not None:
                cov[f"jts_{b}_{f}"] = np.asarray(v, float)
    sp = np.full(len(err), np.nan)
    sp[1:] = np.linalg.norm(np.diff(o["inc_q"], axis=0), axis=1) / np.diff(sy.tS)
    cov["speed_mm_s"] = sp
    att = []
    km = k_pool
    for name, v in cov.items():
        ok = km & np.isfinite(err) & np.isfinite(v)
        if ok.sum() > 20 and np.nanstd(v[ok]) > 0:
            rho = float(pd.Series(err[ok]).corr(pd.Series(v[ok]), method="spearman"))
            att.append({"pid": pid, "covariate": name, "spearman_rho": rho, "n": int(ok.sum())})

    labels = np.full(len(sy.tS), "", dtype=object)
    for ph in sy.phases:
        labels[(sy.tS >= ph.t_start_us * 1e-6) & (sy.tS <= ph.t_end_us * 1e-6)] = ph.name
    arrays = {"t_s": sy.tS - sy.tS[0], "t_master_s": sy.tS, "phase": labels.astype(str),
              "incisor_s_aligned_mm": al["inc_s"],
              "err_incisor_aligned_mm": np.linalg.norm(al["inc_s"] - o["inc_q"], axis=1),
              "err_rot_aligned_deg": o_al["rot_err_deg"],
              "rot_s_aligned_deg": rr_al[0], "rot_q_aligned_deg": rr_al[1],
              "incisor_s_mm": o["inc_s"], "incisor_q_mm": o["inc_q"], "rot_s_deg": rr[0],
              "rot_q_deg": rr[1], "err_incisor_mm": err, "err_rot_deg": o["rot_err_deg"],
              "gate": gate, "ct_visible": se3.valid(sy.S["CT"]),
              "in_sync_window": VK.mask_of(sy.tS, sy.windows),
              "evaluated": VK.mask_of(sy.tS, sy.held)}
    for n in ("tmj_left", "tmj_right"):
        if f"{n}_s" in o:
            arrays[f"{n}_s_mm"], arrays[f"{n}_q_mm"] = o[f"{n}_s"], o[f"{n}_q"]
        if f"{n}_s" in al:
            arrays[f"{n}_s_aligned_mm"] = al[f"{n}_s"]
    tables = {"segments": seg_tab, "segments_aligned": pd.DataFrame(al_rows),
              "segments_aligned_freemotion": pd.DataFrame(fm_rows),
              "segments_filtered": pd.DataFrame(filt_rows),
              "frame_offsets": pd.DataFrame(off_rows),
              "robustness": pd.DataFrame(rob), "budget": pd.DataFrame(bud),
              "decomposition": pd.DataFrame(decomp), "attribution": pd.DataFrame(att)}
    for k_, v_ in sec_t.items():
        tables[f"tracking_only_{k_}"] = v_
    for k_, v_ in sec_a.items():
        if k_ in ("err_incisor_mm", "err_rot_deg"):
            arrays[f"tracking_only_{k_}"] = v_
    return tables, arrays


# -- ref_fusion (V4; secondary, VD8) ----------------------------------------------------------
def _block_ci(d: np.ndarray, t: np.ndarray, *, block_s: float = 1.0, n_boot: int = 1000,
              seed: int = 0) -> tuple[float, float]:
    """95 % CI of the mean of ``d`` by a bootstrap over ``block_s`` blocks (frames within a
    block are not independent)."""
    ok = np.isfinite(d)
    d, t = d[ok], t[ok]
    if len(d) < 10:
        return np.nan, np.nan
    blk = np.floor((t - t[0]) / block_s).astype(int)
    ids = np.unique(blk)
    sums = np.bincount(np.searchsorted(ids, blk), weights=d, minlength=len(ids))
    cnts = np.bincount(np.searchsorted(ids, blk), minlength=len(ids))
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ids), (n_boot, len(ids)))
    m = sums[pick].sum(axis=1) / np.maximum(cnts[pick].sum(axis=1), 1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


SMOOTH_HZ = 6.0      # VD16: jitter = content above the dataset's pose low-pass (motion f95 below)


def _finite_runs(x: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    ok = np.isfinite(x).all(axis=1)
    i = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    return [(a, b) for a, b in zip(i[::2], i[1::2]) if b - a >= min_len]


def _lowpass_runs(x: np.ndarray, fs: float, fc: float) -> np.ndarray:
    """Zero-phase 4th-order Butterworth low-pass of each finite run (NaN elsewhere)."""
    from scipy.signal import butter, sosfiltfilt

    sos = butter(4, fc, fs=fs, output="sos")
    out = np.full_like(x, np.nan)
    for a, b in _finite_runs(x, int(fs)):
        out[a:b] = sosfiltfilt(sos, x[a:b], axis=0)
    return out


def _jitter_rms(x: np.ndarray, fs: float, fc: float = SMOOTH_HZ) -> float:
    """RMS (3-D norm) of the content of ``x`` above ``fc`` over its finite runs of ≥ 1 s."""
    hp = x - _lowpass_runs(x, fs, fc)
    n = np.linalg.norm(hp, axis=1)
    n = n[np.isfinite(n)]
    return float(np.sqrt(np.mean(n ** 2))) if n.size else np.nan


def smoothness_row(x: np.ndarray, ref: np.ndarray, fs: float, *, is_ref: bool = False
                   ) -> dict[str, float]:
    """VD16/VD18 columns of one estimator's incisor trajectory ``x`` against the reference
    ``ref`` (both (n, 3) mm on the same frames): own content above 6 Hz (smoothness,
    ``jitter_rms_mm``), the content above 6 Hz of the **error** ``x − ref``
    (``jitter_err_rms_mm``, the reported jitter since VD18) and the RMS error; the
    reference itself (``is_ref``) gets 0 for the error columns."""
    d = x - ref
    return {"n": int(np.isfinite(x).all(axis=1).sum()),
            "jitter_rms_mm": _jitter_rms(x, fs),
            "jitter_err_rms_mm": 0.0 if is_ref else _jitter_rms(d, fs),
            "err_rms_mm": 0.0 if is_ref else float(np.sqrt(np.nanmean(np.sum(d ** 2, 1))))}


def _opening_deg(M: np.ndarray, k_rest: np.ndarray) -> np.ndarray:
    """Opening angle (deg): the y component (maxillary lateral axis) of the jaw-in-head
    rotation vector since the mean rest pose of the same system (the dataset convention)."""
    ok = k_rest & se3.valid(M)
    if ok.sum() < 3:
        return np.full(len(M), np.nan)
    return np.degrees(se3.rotvec_rel(M, REG.mean_pose(M[ok])))[:, 1]


def run_ref_fusion(sy: Systems, session: Any, landmarks: pd.DataFrame,
                   gate_at_s: np.ndarray | None, params: dict[str, Any] | None = None
                   ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Fused (F1, F2) vs optical-only (F0), one leave-one-task-out fold per movement task:
    calibration and lag on the sync windows plus the other tasks (JTS-MM optical only),
    settings from the sensors, Qualisys only to evaluate -- at the optical frame times and
    at the 200-Hz reference frames between them; simulated dropouts; the descriptive τ curve
    of F1. Both outcomes of equal value (user, 2026-10-23): each system's own anatomical
    frames (end-to-end) and frame-aligned (VD3: ``A M_S B`` fitted on the *optical-only*
    jaw-in-head poses of the other movement tasks, so fusion never shapes the alignment)."""
    from dataclasses import replace

    from scipy.stats import chi2

    from jtsmm.imu import fusion as FU
    from biomech.validation import fusion as VF
    from biomech.validation import kinematics as VK

    pid = sy.pid
    fp = dict(params or {})
    # IMU mounting from a separate hand-eye calibration (``imu_mounting``: name -> R_mount,
    # body<-chip): F1/F2 are then also run with it fixed ("_precal"), next to the in-take
    # calibration
    mounting = fp.pop("imu_mounting", None) or {}
    hs = holds_from_table(landmarks, "jtsmm", "-")
    hq = holds_from_table(landmarks, "qualisys", "primary")
    fs_, fq = AN.anatomical_frames(hs), AN.anatomical_frames(hq)
    Q_hp, Q_mp = sy.q_at_s("HP"), sy.q_at_s("MP")
    M_Q = AN.jaw_in_head(Q_hp, Q_mp, fq)
    gate = np.zeros(len(sy.tS), bool) if gate_at_s is None else np.asarray(gate_at_s, bool)
    mov = [h for h in sy.held if h.name in SEL.MOVEMENT_TASKS]
    # "I": the IMU alone -- each body's IMU orientation through the fold's
    # calibration, no optical correction: the stand-alone IMU validated against Qualisys.
    est = ("F0", "F0_hold", "I", "F1", "F2", "F2_causal", "F2P", "F2P_causal") + (
        ("F1_precal", "F2_precal") if mounting else ())
    seg_rows, drop_rows, tau_rows, cal_rows, diag_rows, pair_rows = [], [], [], [], [], []
    open_rows: list[dict[str, Any]] = []
    smooth_rows: list[dict[str, Any]] = []
    n = len(sy.tS)
    # VD16: aligned incisor trajectories in the Qualisys maxillary frame, at the optical
    # frames (each fold's held-out frames) and at the 200-Hz reference frames of the folds
    TRAJ = ("F0", "F0_lp6", "F2", "F2P")
    inc40 = {e: np.full((n, 3), np.nan) for e in ("F0", "F0_lp6", "F2P")}
    q200: dict[str, list[np.ndarray]] = {k_: [] for k_ in ("t", "qualisys", *TRAJ)}
    fs_opt = 1.0 / float(np.median(np.diff(sy.tS)))
    err = {e: {oc: np.full(n, np.nan) for oc in ("inc", "rot", "dyn", "inc_al", "rot_al")}
           for e in est}
    M_opt = AN.jaw_in_head(sy.S["HP"], sy.S["MP"], fs_)
    ok_al = se3.valid(M_opt) & se3.valid(M_Q) & ~gate

    def align_fold(h) -> tuple[np.ndarray, np.ndarray] | None:
        """VD3 alignment for fold ``h`` from the optical-only poses of the other tasks."""
        ktr = VK.mask_of(sy.tS, [m for m in mov if m is not h]) & ok_al
        if ktr.sum() < 50:
            return None
        try:
            A0, B0 = REG.init_xy(M_Q[ktr], M_opt[ktr])
            rg = REG.register_paired({"jaw": (M_Q[ktr], M_opt[ktr])}, A0, {"jaw": B0})
        except (ValueError, np.linalg.LinAlgError):
            return None
        return rg.X, rg.Y["jaw"]

    def aligned(M_S: np.ndarray, AB) -> np.ndarray:
        return (np.einsum("ij,njk,kl->nil", AB[0], M_S, AB[1]) if AB is not None
                else np.full_like(M_S, np.nan))
    rest = [h for h in sy.held if h.name in SEL.STATIC_TASKS]
    k_rest = VK.mask_of(sy.tS, rest) if rest else VK.mask_of(sy.tS, mov[:1])
    rq_rest = REG.mean_pose(M_Q[k_rest & se3.valid(M_Q)])
    rv_q = np.degrees(se3.rotvec_rel(M_Q, rq_rest))

    def dyn_err(M_S: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Bias-removed rotation error: the rotation vectors since each system's own rest
        pose, minus their mean difference over the evaluated frames (the constant offset of
        the frame definitions is not what fusion changes)."""
        ok = k_rest & se3.valid(M_S)
        ref = REG.mean_pose(M_S[ok]) if ok.sum() >= 3 else REG.mean_pose(M_S[se3.valid(M_S)])
        d = np.degrees(se3.rotvec_rel(M_S, ref)) - rv_q
        out = np.full(len(d), np.nan)
        if k.any():
            dk = d[k] - np.nanmean(d[k], axis=0)
            out[k] = np.linalg.norm(dk, axis=1)
        return out
    tq_hp, tq_mp = sy.qtm.pose("HP"), sy.qtm.pose("MP")
    for h in mov:
        others = [m for m in mov if m is not h]
        train = list(sy.windows) + others
        try:                                  # training never touches the evaluated task
            ctx = VF.prepare(session, fp,
                             lambda t: VK.mask_of(t, train) & ~VK.mask_of(t, [h]),
                             lambda t: VK.mask_of(t, [h]))
        except (ValueError, LookupError) as exc:
            diag_rows.append({"pid": pid, "fold": h.name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        k = VK.mask_of(sy.tS, [h]) & ~gate
        AB = align_fold(h)
        win = (h.t0 - 5.0, h.t1 + 5.0)          # the position filter (F2P) runs around the fold
        out = VF.fuse(ctx, pos_window=win)

        def rms(x: np.ndarray) -> float:
            return float(np.sqrt(np.nanmean(x ** 2))) if np.isfinite(x).any() else np.nan

        outs = {e: (out["HP"]["T"][e], out["MP"]["T"][e]) for e in est if e in out["HP"]["T"]}
        T_imu = {}
        for name, b in ctx.bodies.items():          # IMU only, at the optical frame times
            T = out[name]["T"]["F0"].copy()
            T[:, :3, :3] = b.calibration.predict(ctx.t_frames_s,
                                                 b.imu.rot(ctx.t_frames_s, b.calibration.lag_s))
            T_imu[name] = T
        outs["I"] = (T_imu["HP"], T_imu["MP"])
        if mounting:
            try:
                ctx_p = VF.prepare(session, {**fp, "imu_mounting": mounting},
                                   lambda t: VK.mask_of(t, train) & ~VK.mask_of(t, [h]),
                                   lambda t: VK.mask_of(t, [h]))
                out_p = VF.fuse(ctx_p, estimators=("F1", "F2"))
                for e in ("F1", "F2"):
                    outs[f"{e}_precal"] = (out_p["HP"]["T"][e], out_p["MP"]["T"][e])
                for name in ("HP", "MP"):
                    c_free, c_pre = ctx.bodies[name].calibration, ctx_p.bodies[name].calibration
                    cal_rows.append({"pid": pid, "fold": h.name, "body": name,
                                     "variant": "precal", "lag_ms": c_pre.lag_s * 1e3,
                                     "rms_deg": c_pre.rms_deg,
                                     "mounting_vs_intake_deg": float(np.degrees(np.linalg.norm(
                                         Rotation.from_matrix(c_free.X.T @ c_pre.X).as_rotvec())))})
            except (ValueError, LookupError) as exc:
                diag_rows.append({"pid": pid, "fold": h.name,
                                  "error": f"precal: {type(exc).__name__}: {exc}"})
        for e, (T_hp_e, T_mp_e) in outs.items():
            M_S = AN.jaw_in_head(T_hp_e, T_mp_e, fs_)
            ei = np.linalg.norm(M_S[:, :3, 3] - M_Q[:, :3, 3], axis=1)
            er = AN.rotation_error_deg(M_S, M_Q)
            ed = dyn_err(M_S, k)
            Ma = aligned(M_S, AB)
            ea = np.linalg.norm(Ma[:, :3, 3] - M_Q[:, :3, 3], axis=1)
            ra = AN.rotation_error_deg(Ma, M_Q)
            for oc, v in (("inc", ei), ("rot", er), ("dyn", ed), ("inc_al", ea), ("rot_al", ra)):
                err[e][oc][k] = v[k]
            seg_rows.append({"pid": pid, "segment": h.name, "estimator": e, "query": "frames",
                             "n": int(np.isfinite(ei[k]).sum()),
                             "incisor_rms_mm": rms(ei[k]), "rot_rms_deg": rms(er[k]),
                             "rot_dyn_rms_deg": rms(ed[k]),
                             "incisor_aligned_rms_mm": rms(ea[k]),
                             "rot_aligned_rms_deg": rms(ra[k])})
        # Opening angle (the rotation about the maxillary lateral axis since each system's
        # own rest pose, the dataset convention): IMU alone vs Qualisys, JTS-MM optical vs
        # Qualisys, and IMU alone vs JTS-MM optical (the reference of the dataset's C3a).
        theta = {"imu": _opening_deg(AN.jaw_in_head(*outs["I"], fs_), k_rest),
                 "optical": _opening_deg(AN.jaw_in_head(*outs["F0"], fs_), k_rest),
                 "qualisys": _opening_deg(M_Q, k_rest)}
        for a_, b_ in (("imu", "qualisys"), ("optical", "qualisys"), ("imu", "optical")):
            d = (theta[a_] - theta[b_])[k]
            d = d[np.isfinite(d)]
            open_rows.append({"pid": pid, "segment": h.name, "comparison": f"{a_}_vs_{b_}",
                              "n": int(d.size),
                              "rmse_deg": float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan,
                              "bias_deg": float(np.mean(d)) if d.size else np.nan,
                              "sd_deg": float(np.std(d, ddof=1)) if d.size > 1 else np.nan,
                              "range_ref_deg": float(np.nanmax(theta[b_][k]) -
                                                     np.nanmin(theta[b_][k]))
                              if np.isfinite(theta[b_][k]).any() else np.nan})
        # the 200-Hz Qualisys frames inside the fold (exact reference, no spline)
        kq = (sy.t_qm >= h.t0) & (sy.t_qm <= h.t1)
        if kq.sum() > 10:
            outq = VF.fuse(ctx, sy.t_qm[kq], estimators=("F0", "F1", "F2", "F2P"),
                           pos_window=win)
            Mq = AN.jaw_in_head(tq_hp[kq], tq_mp[kq], fq)
            # VD16: how much each estimator jitters above the motion band, against Qualisys
            fs_q = 1.0 / float(np.median(np.diff(sy.t_qm[kq])))
            Ma0 = aligned(AN.jaw_in_head(*outs["F0"], fs_), AB)[:, :3, 3]
            lp = _lowpass_runs(np.where(k[:, None], Ma0, np.nan), fs_opt, SMOOTH_HZ)
            inc40["F0"][k], inc40["F0_lp6"][k] = Ma0[k], lp[k]
            tr = {"F0_lp6": np.column_stack([np.interp(sy.t_qm[kq], sy.tS[k], lp[k][:, j],
                                                       left=np.nan, right=np.nan)
                                             for j in range(3)]) if k.sum() > 1
                  else np.full((int(kq.sum()), 3), np.nan)}
            for e in ("F0", "F2", "F2P"):
                if e in outq["HP"]["T"]:
                    tr[e] = aligned(AN.jaw_in_head(outq["HP"]["T"][e], outq["MP"]["T"][e],
                                                   fs_), AB)[:, :3, 3]
            if "F2P" in outs:
                inc40["F2P"][k] = aligned(AN.jaw_in_head(*outs["F2P"], fs_), AB)[k, :3, 3]
            pq = Mq[:, :3, 3]
            q200["t"].append(sy.t_qm[kq])
            q200["qualisys"].append(pq)
            for e in TRAJ:
                q200[e].append(tr.get(e, np.full_like(pq, np.nan)))
            for e, x in (("qualisys", pq), *tr.items()):
                smooth_rows.append({"pid": pid, "segment": h.name, "estimator": e,
                                    **smoothness_row(x, pq, fs_q, is_ref=e == "qualisys")})
            for e in ("F0", "F1", "F2", "F2P"):
                if e not in outq["HP"]["T"]:
                    continue
                M_S = AN.jaw_in_head(outq["HP"]["T"][e], outq["MP"]["T"][e], fs_)
                ei = np.linalg.norm(M_S[:, :3, 3] - Mq[:, :3, 3], axis=1)
                er = AN.rotation_error_deg(M_S, Mq)
                Ma = aligned(M_S, AB)
                seg_rows.append({"pid": pid, "segment": h.name, "estimator": e,
                                 "query": "ref200", "n": int(np.isfinite(ei).sum()),
                                 "incisor_rms_mm": rms(ei), "rot_rms_deg": rms(er),
                                 "incisor_aligned_rms_mm": rms(np.linalg.norm(
                                     Ma[:, :3, 3] - Mq[:, :3, 3], axis=1)),
                                 "rot_aligned_rms_deg": rms(AN.rotation_error_deg(Ma, Mq))})
        # simulated dropouts (scored on the dropped frames only)
        for L, dm in VF.dropout_masks(sy.tS, [h]).items():
            kd = dm & ~gate
            if kd.sum() < 5:
                continue
            od = VF.fuse(ctx, drop_mask=dm, pos_window=win,
                         estimators=("F0", "F0_hold", "F1", "F2", "F2P", "F2P_causal"))
            for e in ("F0", "F0_hold", "F1", "F2", "F2P", "F2P_causal"):
                if e not in od["HP"]["T"]:
                    continue
                M_S = AN.jaw_in_head(od["HP"]["T"][e], od["MP"]["T"][e], fs_)
                er = AN.rotation_error_deg(M_S, M_Q)
                ed = dyn_err(M_S, k)
                ei = np.linalg.norm(M_S[:, :3, 3] - M_Q[:, :3, 3], axis=1)
                Ma = aligned(M_S, AB)
                drop_rows.append({"pid": pid, "segment": h.name, "dropout_s": L, "estimator": e,
                                  "n": int(kd.sum()), "rot_rms_deg": rms(er[kd]),
                                  "rot_dyn_rms_deg": rms(ed[kd]), "incisor_rms_mm": rms(ei[kd]),
                                  "incisor_aligned_rms_mm": rms(np.linalg.norm(
                                      Ma[:, :3, 3] - M_Q[:, :3, 3], axis=1)[kd]),
                                  "rot_aligned_rms_deg": rms(AN.rotation_error_deg(Ma, M_Q)[kd])})
        # τ curve of F1 (descriptive, never used for selection)
        for tau in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0):
            Rs = {}
            for name, b in ctx.bodies.items():
                st = replace(b.settings, tau_s=tau)
                Rs[name] = FU.f1_complementary(ctx.t_frames_s, b.R_opt, b.imu, b.calibration,
                                               st, ctx.t_frames_s, quality=b.quality).R_query
            T_hp = out["HP"]["T"]["F0"].copy()
            T_mp = out["MP"]["T"]["F0"].copy()
            T_hp[:, :3, :3], T_mp[:, :3, :3] = Rs["HP"], Rs["MP"]
            M_t = AN.jaw_in_head(T_hp, T_mp, fs_)
            er, ed = AN.rotation_error_deg(M_t, M_Q), dyn_err(M_t, k)
            tau_rows.append({"pid": pid, "segment": h.name, "tau_s": tau,
                             "rot_rms_deg": float(np.sqrt(np.nanmean(er[k] ** 2))),
                             "rot_dyn_rms_deg": float(np.sqrt(np.nanmean(ed[k] ** 2)))})
        for name in ("HP", "MP"):
            c, s_ = out[name]["calibration"], out[name]["settings"]
            cal_rows.append({"pid": pid, "fold": h.name, "body": name,
                             "lag_ms": c.get("lag_ms"), "lag_se_ms": c.get("lag_se_ms"),
                             "rms_deg": c.get("rms_deg"), "inlier_frac": c.get("inlier_frac"),
                             "drift_heading_deg_min": c.get("drift_heading_deg_min"),
                             "tau_s": s_.get("tau_s"), "n_train": c.get("n_train"),
                             "pos_r_mm": (out[name].get("pos_settings") or {}).get("r_pos_mm"),
                             "pos_sigma_acc_mm_s2": ((out[name].get("pos_settings") or {})
                                                     .get("derivation", {})
                                                     .get("sigma_acc_mm_s2"))})
            d2 = out[name]["diagnostics"].get("F2", {})
            nis = np.asarray(d2.get("nis", []), float)
            nis = nis[np.isfinite(nis)]
            diag_rows.append({"pid": pid, "fold": h.name, "body": name,
                              "nis_mean": float(nis.mean()) if len(nis) else np.nan,
                              "nis_in_95": float(np.mean(nis <= chi2.ppf(0.95, 3)))
                              if len(nis) else np.nan,
                              "n_gated": int(np.asarray(d2.get("gated", []), bool).sum())})
    # paired differences fused − optical (pooled over the folds, block bootstrap)
    for e in ("I", "F1", "F2", "F2_causal", "F2P", "F2P_causal") + (
            ("F1_precal", "F2_precal") if mounting else ()):
        for oc in ("dyn", "rot", "inc", "rot_al", "inc_al"):
            d = err[e][oc] - err["F0"][oc]
            lo, hi = _block_ci(d, sy.tS)
            pair_rows.append({"pid": pid, "estimator": e, "outcome": oc,
                              "mean_diff": float(np.nanmean(d)), "ci_lo": lo, "ci_hi": hi,
                              "n": int(np.isfinite(d).sum())})
    arrays = {"t_master_s": sy.tS}
    for e in est:
        arrays[f"err_rot_{e}_deg"] = err[e]["rot"]
        arrays[f"err_rot_dyn_{e}_deg"] = err[e]["dyn"]
        arrays[f"err_incisor_{e}_mm"] = err[e]["inc"]
        arrays[f"err_rot_aligned_{e}_deg"] = err[e]["rot_al"]
        arrays[f"err_incisor_aligned_{e}_mm"] = err[e]["inc_al"]
    arrays["incisor_q_mm"] = M_Q[:, :3, 3]
    for e, x in inc40.items():
        arrays[f"incisor_aligned_{e}_mm"] = x
    if q200["t"]:
        arrays["t_q200_s"] = np.concatenate(q200["t"])
        for e in ("qualisys", *TRAJ):
            arrays[f"incisor_q200_{e}_mm"] = np.concatenate(q200[e])
    return ({"segments": pd.DataFrame(seg_rows), "paired": pd.DataFrame(pair_rows),
             "smoothness": pd.DataFrame(smooth_rows),
             "dropout": pd.DataFrame(drop_rows), "tau": pd.DataFrame(tau_rows),
             "calibration": pd.DataFrame(cal_rows), "diagnostics": pd.DataFrame(diag_rows),
             "imu_opening": pd.DataFrame(open_rows)},
            arrays)
