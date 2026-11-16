# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Point-marking (digitization) accuracy of JTS-MM against Qualisys (validation plan VD4).

The systems paper verifies two things separately:

* **(I) point marking** (this module): how accurately JTS-MM digitizes the tooth and condyle
  points, and the anatomical frames built from them, with Qualisys (CT label swaps
  corrected, EV1; common still window, EV2; pivot-verified tip, V7) as the reference;
* **(II) motion** (``ref_kinematics`` / ``ref_agreement``): the frame-aligned trajectory
  agreement (VD3), which removes the marking difference on purpose.

Per landmark the JTS-MM point is mapped into the Qualisys body frame through the body
registration ``Y`` (``T_Q,body = X T_S,body Y`` → ``p_Q = Y⁻¹ p_S``) and compared with the
Qualisys point. The registration must not carry a world error into the points: the sync
fit's ``X`` trades translation against ``Y_CT`` (the CT is waved about a point near its
tip), and ``Y_HP``/``Y_MP`` inherit that offset. So ``X`` is refit here from the two
pivot-calibrated tips over the sync-window motion (point correspondences, robust), and the
body ``Y`` given that ``X`` from the body poses outside the sync windows -- independent of
the landmarks:

* **trueness:** the error vector with a 95 % CI (moving-block bootstrap of each hold's
  frames, combined with a block bootstrap of ``Y`` over the per-frame estimates);
* **accuracy:** RMS of |e| over the teeth and over the condyle points;
* **static precision** of each system: the within-hold SD and MAD of the tip in the body;
* **frame-free:** the within-body landmark distances (no registration);
* **frames:** the angle errors of the maxillary and mandibular frames, the incisor offset
  they cause at rest (the marking part of the end-to-end V5 result), and the VD3 offsets
  estimated from motion for comparison;
* **σ_ref:** Monte Carlo of the Qualisys landmark from the CT and body marker noise of the
  hold and the pivot tip SE (conservative: the noise is not averaged over the hold frames).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.kinematics import se3
from biomech.validation import anatomy as AN
from biomech.validation import quality as QY
from biomech.validation import registration as REG

AXES = ("anterior", "left", "superior")       # maxillary / mandibular frame x, y, z
FRAME_OF = {"hp": "F_max", "mp": "F_mand"}
TEETH = ("mand_point_1", "mand_point_2", "mand_point_3", "max_point_1", "max_point_2",
         "max_point_3")


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _block_idx(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block bootstrap indices of length ``n``."""
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def x_from_tips(sy: Any, d: np.ndarray, *, k_mad: float = 4.0, n_iter: int = 5
                ) -> tuple[np.ndarray, float, int]:
    """World registration from the CT tips: ``X`` with ``tip_Q ≈ X tip_S`` over the sync-window
    frames (both systems' tips pivot-calibrated; Qualisys tip = ``T_Q,CT d``), robust
    Kabsch (frames beyond ``k_mad`` MADs of the residual dropped). Returns ``(X, RMS mm, n)``."""
    Tq = sy.q_at_s("CT")
    ok = se3.valid(Tq) & se3.valid(sy.S["CT"])
    win = np.zeros(len(sy.tS), bool)
    for w in sy.windows:
        win |= (sy.tS >= w.t0) & (sy.tS <= w.t1)
    ok &= win
    if ok.sum() < 20:
        raise ValueError("x_from_tips: too few sync-window frames")
    a, b = sy.S["CT"][ok, :3, 3], se3.apply(Tq[ok], d)
    keep = np.ones(len(a), bool)
    X = np.eye(4)
    for _ in range(n_iter):
        R = REG._kabsch(a[keep] - a[keep].mean(axis=0), b[keep] - b[keep].mean(axis=0))
        t = b[keep].mean(axis=0) - R @ a[keep].mean(axis=0)
        X = REG.make_pose(R, t)
        r = np.linalg.norm(a @ R.T + t - b, axis=1)
        med = np.median(r)
        keep = r <= med + k_mad * 1.4826 * np.median(np.abs(r - med))
    return X, float(np.sqrt(np.mean(r[keep] ** 2))), int(keep.sum())


def y_from_bodies(sy: Any, X: np.ndarray, body: str) -> np.ndarray:
    """Body registration given ``X``: the robust mean of ``(X T_S,i)⁻¹ T_Q,i`` outside the
    sync windows (``registration.y_given_x``)."""
    ok = np.ones(len(sy.tS), bool)
    for w in sy.windows:
        ok &= ~((sy.tS >= w.t0) & (sy.tS <= w.t1))
    return REG.y_given_x(sy.q_at_s(body)[ok], sy.S[body][ok], X)


def _hold_locals(T_body: np.ndarray, tip_world: np.ndarray, idx: np.ndarray) -> np.ndarray:
    loc = AN.local_tip(T_body[idx], tip_world[idx])
    return loc[np.isfinite(loc).all(axis=1)]


def y_bootstrap(sy: Any, X: np.ndarray, body: str, Y0: np.ndarray, *, n_boot: int,
                rng: np.random.Generator, block_s: float = 1.0, step: int = 3
                ) -> np.ndarray:
    """Bootstrap draws of the body registration ``Y`` (``n_boot`` x 4 x 4): per-frame
    ``Y_i = (X T_S,i)⁻¹ T_Q,i`` outside the sync windows (the CT moves next to the face
    there), chordal means of moving-block resamples, applied as perturbations of ``Y0``."""
    Ts, Tq = sy.S[body], sy.q_at_s(body)
    ok = se3.valid(Ts) & se3.valid(Tq)
    for w in sy.windows:
        ok &= ~((sy.tS >= w.t0) & (sy.tS <= w.t1))
    idx = np.flatnonzero(ok)[::step]
    if len(idx) < 20:
        return np.repeat(Y0[None], n_boot, axis=0)
    Yi = np.einsum("nij,njk->nik", se3.inv(np.einsum("ij,njk->nik", X, Ts[idx])), Tq[idx])
    Ym = REG.mean_pose(Yi)
    dt = float(np.median(np.diff(sy.tS[idx])))
    block = max(int(round(block_s / dt)), 1)
    out = np.empty((n_boot, 4, 4))
    for k in range(n_boot):
        out[k] = Y0 @ se3.inv(Ym) @ REG.mean_pose(Yi[_block_idx(len(Yi), block, rng)])
    return out


def _sigma_ref_hold(L_ct: np.ndarray, tm_body: np.ndarray, T_ct: np.ndarray,
                    T_body: np.ndarray, d: np.ndarray, d_cov: np.ndarray, s_ct: float,
                    s_body: float, rng: np.random.Generator, draws: int = 300) -> float:
    """σ (3-D RMS, mm) of one Qualisys landmark from marker noise of the CT and the body
    (isotropic σ per marker, refit by Kabsch) and the tip SE, at the hold's mean poses."""
    def fits(tmpl: np.ndarray, s: float) -> np.ndarray:
        noisy = tmpl[None] + rng.normal(0.0, s, (draws,) + tmpl.shape)
        return QY._kabsch(np.repeat(tmpl[None], draws, axis=0), noisy)

    p0 = se3.apply(se3.inv(T_body), se3.apply(T_ct, d))
    dC, dB = fits(L_ct, s_ct), fits(tm_body, s_body)
    dd = rng.multivariate_normal(d, d_cov, draws) if np.isfinite(d_cov).all() else \
        np.repeat(d[None], draws, axis=0)
    Tc = np.einsum("ij,kjl->kil", T_ct, dC)
    Tb = np.einsum("ij,kjl->kil", T_body, dB)
    tip = np.einsum("kij,kj->ki", Tc[:, :3, :3], dd) + Tc[:, :3, 3]
    Tbi = se3.inv(Tb)
    p = np.einsum("kij,kj->ki", Tbi[:, :3, :3], tip) + Tbi[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum((p - p0) ** 2, axis=1))))


def hold_attribution(sy: Any, X: np.ndarray, Y_ct: np.ndarray, hq: dict, hs: dict,
                     consistency: pd.DataFrame, cams: dict[str, np.ndarray],
                     L_ct: np.ndarray) -> pd.DataFrame:
    """Per hold the covariates that could explain a marking error (descriptive):
    the CT orientation disagreement between the systems (× the lever = tip disagreement),
    JTS-MM tag count / reprojection RMS / cameras of the CT and of the hold's body, the
    Qualisys CT cluster deviation, and the smallest angle between a Qualisys camera's line
    of sight and the CT marker-plate normal (flat markers seen obliquely, R5d)."""
    Tq_s = sy.q_at_s("CT")
    n_plate = np.cross(L_ct[1] - L_ct[0], L_ct[2] - L_ct[0])
    n_plate /= np.linalg.norm(n_plate)
    lever = float(np.linalg.norm(L_ct.mean(axis=0)))
    cons = consistency.set_index(["body", "segment"]) if not consistency.empty else None
    rows = []
    for n, h in hs.items():
        if n.endswith("_hp") or n not in hq:
            continue
        idx = h.idx
        ok = se3.valid(Tq_s[idx]) & se3.valid(sy.S["CT"][idx])
        Rs = np.einsum("ij,njk,kl->nil", X[:3, :3], sy.S["CT"][idx][ok][:, :3, :3], Y_ct[:3, :3])
        dR = np.einsum("nji,njk->nik", Rs, Tq_s[idx][ok][:, :3, :3])
        ang = float(np.degrees(np.linalg.norm(_rotvec_mean(dR)))) if ok.any() else np.nan
        row = {"pid": sy.pid, "landmark": n, "ct_rot_disagreement_deg": ang,
               "predicted_tip_mm": float(np.radians(ang) * lever) if np.isfinite(ang) else np.nan}
        for body in ("CT", h.body.upper()):
            q = sy.quality_s.get(body, {})
            for k in ("n_tags", "reprojection_rms", "n_cameras"):
                if k in q:
                    row[f"jtsmm_{body.lower()}_{k}"] = float(np.nanmedian(np.asarray(q[k],
                                                                                    float)[idx]))
        if cons is not None and ("CT", n) in cons.index:
            row["qualisys_ct_dev_max_mm"] = float(cons.loc[("CT", n), "dev_max_mm"])
        if cams:
            iq = hq[n].idx
            T = sy.qtm.pose("CT")[iq]
            T = T[se3.valid(T)]
            if len(T):
                obl = []
                for c in cams.values():
                    v = c[None] - T[:, :3, 3]
                    v = np.einsum("nji,nj->ni", T[:, :3, :3], v)
                    v /= np.linalg.norm(v, axis=1, keepdims=True)
                    obl.append(np.degrees(np.arccos(np.clip(np.abs(v @ n_plate), 0, 1))))
                obl = np.median(np.stack(obl, axis=1), axis=0)
                row["qualisys_min_obliquity_deg"] = float(obl.min())
                row["qualisys_n_cams_below_60deg"] = int((obl < 60).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _rotvec_mean(dR: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(dR).mean().as_rotvec()


def run_ref_landmarks(sy: Any, registration: pd.DataFrame, tip_decision: pd.Series,
                      tip_estimates: pd.DataFrame, consistency: pd.DataFrame,
                      frame_offsets: pd.DataFrame, cams: dict[str, np.ndarray] | None = None,
                      *, n_boot: int = 400, seed: int = 11
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    from biomech.validation.evaluate import landmark_holds, pose_from_row

    pid = sy.pid
    rng = np.random.default_rng(seed)
    reg = registration.set_index("transform")
    X_sync = pose_from_row(reg.loc["X"])
    d = np.array([tip_decision["dx_mm"], tip_decision["dy_mm"], tip_decision["dz_mm"]], float)
    try:                                   # registration free of the X/Y_CT trade-off
        X, x_rms, x_n = x_from_tips(sy, d)
        Y0 = {b: y_from_bodies(sy, X, b.upper()) for b in ("hp", "mp")}
        reg_src = "tips"
    except ValueError:
        X, x_rms, x_n = X_sync, np.nan, 0
        Y0 = {"hp": pose_from_row(reg.loc["Y_HP"]), "mp": pose_from_row(reg.loc["Y_MP"])}
        reg_src = "sync_fit"
    reg_rows = [{"pid": pid, "transform": "X", "source": reg_src, "tip_rms_mm": x_rms,
                 "n_frames": x_n,
                 "rot_vs_sync_deg": _angle_deg(X[:3, :3], X_sync[:3, :3]),
                 "trans_vs_sync_mm": float(np.linalg.norm(X[:3, 3] - X_sync[:3, 3]))}]
    for b, nm in (("hp", "Y_HP"), ("mp", "Y_MP")):
        Ys = pose_from_row(reg.loc[nm])
        reg_rows.append({"pid": pid, "transform": nm, "source": reg_src,
                         "rot_vs_sync_deg": _angle_deg(Y0[b][:3, :3], Ys[:3, :3]),
                         "trans_vs_sync_mm": float(np.linalg.norm(Y0[b][:3, 3] - Ys[:3, 3]))})
    piv = tip_estimates[tip_estimates["method"] == "pivot_qualisys"]
    d_cov = (np.diag(piv[["se_x_mm", "se_y_mm", "se_z_mm"]].iloc[0].to_numpy(float) ** 2)
             if not piv.empty else np.full((3, 3), np.nan))
    hs, hq, common = landmark_holds(sy, d)
    names = [n for n in hs if n in hq]
    Ts_body = {"hp": sy.S["HP"], "mp": sy.S["MP"]}
    Tq_body = {"hp": sy.qtm.pose("HP"), "mp": sy.qtm.pose("MP")}
    tip_s = sy.S["CT"][:, :3, 3]
    T_qct = sy.qtm.pose("CT")
    tip_q = se3.apply(T_qct, d)
    loc_s = {n: _hold_locals(Ts_body[hs[n].body], tip_s, hs[n].idx) for n in names}
    loc_q = {n: _hold_locals(Tq_body[hq[n].body], tip_q, hq[n].idx) for n in names}
    Yb = {b: y_bootstrap(sy, X, b.upper(), Y0[b], n_boot=n_boot, rng=rng) for b in Y0}
    fr_q = AN.anatomical_frames(hq)
    fr_s = AN.anatomical_frames(hs)

    def mapped(p_s: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return se3.apply(se3.inv(Y), p_s)

    # bootstrap draws of every landmark (both systems' hold frames, Y)
    dt_s = float(np.median(np.diff(sy.tS)))
    dt_q = float(np.median(np.diff(sy.t_qm)))
    boot_s = {n: np.empty((n_boot, 3)) for n in names}
    boot_q = {n: np.empty((n_boot, 3)) for n in names}
    for n in names:
        bs, bq = max(int(round(0.25 / dt_s)), 1), max(int(round(0.25 / dt_q)), 1)
        for k in range(n_boot):
            boot_s[n][k] = np.median(loc_s[n][_block_idx(len(loc_s[n]), bs, rng)], axis=0)
            boot_q[n][k] = np.median(loc_q[n][_block_idx(len(loc_q[n]), bq, rng)], axis=0)

    # σ_ref per landmark (Qualisys marker noise + tip SE)
    L_ct = np.asarray(sy.qtm.label_info.get("CT", {}).get("template_mm") or
                      QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers("CT")},
                                  T_qct), float)
    tm = {b: QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers(b.upper())},
                         Tq_body[b]) for b in Tq_body}
    cons = consistency.set_index(["body", "segment"]) if not consistency.empty else None

    def sig(body: str, seg: str) -> float:
        v = np.nan
        if cons is not None and (body, seg) in cons.index:
            v = float(cons.loc[(body, seg), "sigma_marker_mm"])
        if not np.isfinite(v) and cons is not None:
            v = float(consistency[consistency["body"] == body]["sigma_marker_mm"].median())
        return v if np.isfinite(v) else 0.3

    rows, prec, sref = [], [], []
    for n in names:
        b = hq[n].body
        F = getattr(fr_q, FRAME_OF[b])[:3, :3]                   # Qualisys anatomical axes
        e0 = mapped(hs[n].point, Y0[b]) - hq[n].point
        eb = np.einsum("kij,kj->ki", se3.inv(Yb[b])[:, :3, :3], boot_s[n]) \
            + se3.inv(Yb[b])[:, :3, 3] - boot_q[n]
        ax0, axb = F.T @ e0, eb @ F
        lo, hi = np.percentile(axb, [2.5, 97.5], axis=0)
        nb = np.linalg.norm(eb, axis=1)
        seg = n[:-3] if n.endswith("_hp") else n
        idx = hq[n].idx
        s_ref = _sigma_ref_hold(L_ct, tm[b], REG.mean_pose(T_qct[idx]),
                                REG.mean_pose(Tq_body[b][idx]), d, d_cov, sig("CT", seg),
                                sig(b.upper(), seg), rng)
        rows.append({"pid": pid, "landmark": n, "body": b,
                     "kind": "tooth" if n in TEETH else "condyle",
                     "window": "common" if seg in common else "own",
                     **{f"e_{a}_mm": float(v) for a, v in zip(AXES, ax0)},
                     **{f"e_{a}_lo_mm": float(v) for a, v in zip(AXES, lo)},
                     **{f"e_{a}_hi_mm": float(v) for a, v in zip(AXES, hi)},
                     "e_mm": float(np.linalg.norm(e0)),
                     "e_lo_mm": float(np.percentile(nb, 2.5)),
                     "e_hi_mm": float(np.percentile(nb, 97.5)),
                     "sigma_ref_mm": s_ref, "ratio": float(np.linalg.norm(e0) / s_ref)
                     if s_ref > 0 else np.nan,
                     "n_jtsmm": len(loc_s[n]), "n_qualisys": len(loc_q[n])})
        for system, loc in (("jtsmm", loc_s[n]), ("qualisys", loc_q[n])):
            med = np.median(loc, axis=0)
            r = np.linalg.norm(loc - med, axis=1)
            prec.append({"pid": pid, "landmark": n, "system": system, "n": len(loc),
                         "sd_mm": float(np.sqrt(np.mean(np.sum((loc - loc.mean(axis=0)) ** 2,
                                                               axis=1)))),
                         "mad_mm": float(np.median(r))})
        sref.append({"pid": pid, "landmark": n, "sigma_ct_marker_mm": sig("CT", seg),
                     "sigma_body_marker_mm": sig(b.upper(), seg), "sigma_ref_mm": s_ref})
    err = pd.DataFrame(rows)

    # frame-free distances
    ds, dq = AN.within_body_distances(hs), AN.within_body_distances(hq)
    dist = pd.DataFrame([{"pid": pid, "a": k[0], "b": k[1], "dist_jtsmm_mm": ds[k],
                          "dist_qualisys_mm": dq[k], "diff_mm": ds[k] - dq[k]}
                         for k in sorted(set(ds) & set(dq))])

    # anatomical frames: angle errors (bootstrap CI), marking-only offset at rest, VD3
    def frames_of(pts_s: dict[str, np.ndarray], pts_q: dict[str, np.ndarray], Y: dict
                  ) -> dict[str, tuple[float, float]]:
        Hs = {k: AN.Hold(k, hs[k].body, v, 0, 0, 0, 0) for k, v in pts_s.items()}
        Hq = {k: AN.Hold(k, hq[k].body, v, 0, 0, 0, 0) for k, v in pts_q.items()}
        fs, fq = AN.anatomical_frames(Hs), AN.anatomical_frames(Hq)
        out = {}
        for nm, b in (("F_max", "hp"), ("F_mand", "mp")):
            Fm = se3.inv(Y[b]) @ getattr(fs, nm)
            Fq = getattr(fq, nm)
            out[nm] = (_angle_deg(Fq[:3, :3], Fm[:3, :3]),
                       float(np.linalg.norm(Fq[:3, 3] - Fm[:3, 3])))
        return out

    f0 = frames_of({n: hs[n].point for n in names}, {n: hq[n].point for n in names}, Y0)
    fb = [frames_of({n: boot_s[n][k] for n in names}, {n: boot_q[n][k] for n in names},
                    {b: Yb[b][k] for b in Yb}) for k in range(min(n_boot, 200))]
    # The offsets from motion (VD3) are the leave-one-task-out folds; the in-sample and the
    # free-motion-only fits stored next to them are sensitivities (EV4).
    loto = (frame_offsets[~frame_offsets["fold"].isin(("insample", "free_motion_only"))]
            if not frame_offsets.empty and "fold" in frame_offsets else frame_offsets)
    vd3 = (loto.groupby("transform")["rot_deg"].median()
           if not loto.empty else pd.Series(dtype=float))
    rest = [iv for seg, kind, iv in _segments(sy) if seg == "rest"]
    rel = se3.relative(Tq_body["hp"], Tq_body["mp"])
    k_rest = np.zeros(len(sy.t_qm), bool)
    for iv in rest:
        k_rest |= (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
    k_rest &= se3.valid(rel)
    Fm_max = se3.inv(Y0["hp"]) @ fr_s.F_max
    Fm_mand = se3.inv(Y0["mp"]) @ fr_s.F_mand
    inc_off = rot_off = np.nan
    if k_rest.sum() >= 5:
        R0 = REG.mean_pose(rel[k_rest])
        Mq = se3.inv(fr_q.F_max) @ R0 @ fr_q.F_mand
        Mm = se3.inv(Fm_max) @ R0 @ Fm_mand
        inc_off = float(np.linalg.norm(Mq[:3, 3] - Mm[:3, 3]))
        rot_off = _angle_deg(Mq[:3, :3], Mm[:3, :3])
    frame_rows = []
    for nm, vd in (("F_max", "A_maxillary"), ("F_mand", "B_mandibular")):
        a = np.array([x[nm][0] for x in fb])
        t = np.array([x[nm][1] for x in fb])
        frame_rows.append({"pid": pid, "frame": nm, "angle_deg": f0[nm][0],
                           "angle_lo_deg": float(np.percentile(a, 2.5)),
                           "angle_hi_deg": float(np.percentile(a, 97.5)),
                           "origin_mm": f0[nm][1],
                           "origin_lo_mm": float(np.percentile(t, 2.5)),
                           "origin_hi_mm": float(np.percentile(t, 97.5)),
                           "vd3_rot_deg": float(vd3.get(vd, np.nan))})
    frames = pd.DataFrame(frame_rows)

    teeth, cond = err[err["kind"] == "tooth"], err[err["kind"] == "condyle"]
    summary = pd.DataFrame([{
        "pid": pid,
        "rms_teeth_mm": float(np.sqrt(np.mean(teeth["e_mm"] ** 2))) if len(teeth) else np.nan,
        "rms_condyle_mm": float(np.sqrt(np.mean(cond["e_mm"] ** 2))) if len(cond) else np.nan,
        "sigma_ref_teeth_mm": float(np.sqrt(np.mean(teeth["sigma_ref_mm"] ** 2)))
        if len(teeth) else np.nan,
        "sigma_ref_condyle_mm": float(np.sqrt(np.mean(cond["sigma_ref_mm"] ** 2)))
        if len(cond) else np.nan,
        "rms_dist_diff_teeth_mm": float(np.sqrt(np.mean(dist[dist["a"].isin(TEETH)
                                                              & dist["b"].isin(TEETH)]
                                                         ["diff_mm"] ** 2)))
        if not dist.empty else np.nan,
        "frame_max_deg": f0["F_max"][0], "frame_mand_deg": f0["F_mand"][0],
        "rest_incisor_offset_mm": inc_off, "rest_rotation_offset_deg": rot_off,
        "precision_jtsmm_mad_mm": float(np.median([p["mad_mm"] for p in prec
                                                   if p["system"] == "jtsmm"])),
        "precision_qualisys_mad_mm": float(np.median([p["mad_mm"] for p in prec
                                                      if p["system"] == "qualisys"])),
        "n_common_windows": len(common)}])
    summary["adequacy_teeth"] = summary["rms_teeth_mm"] / summary["sigma_ref_teeth_mm"]
    attr = hold_attribution(sy, X_sync, pose_from_row(reg.loc["Y_CT"]), hq, hs, consistency,
                            cams or {}, L_ct)
    if not attr.empty:
        attr = attr.merge(err[["landmark", "e_mm"]], on="landmark", how="left")
    return ({"landmark_errors": err, "landmark_precision": pd.DataFrame(prec),
             "hold_attribution": attr, "registration": pd.DataFrame(reg_rows),
             "landmark_distances": dist, "frame_errors": frames,
             "landmark_sigma_ref": pd.DataFrame(sref), "marking_summary": summary}, {})


def _segments(sy: Any):
    from biomech.validation.evaluate import segment_list

    return segment_list(sy)
