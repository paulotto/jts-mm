# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Jaw-in-head agreement JTS-MM vs Qualisys at the lower incisor (validation plan V2/V3, D4).

* **Clock and X:** from ``ref_sync`` (CT alone, V1).
* **Y_MP / Y_HP:** leave-one-task-out (V2). Each evaluated segment uses a ``Y`` fitted on
  the *other* movement tasks (robust closed form, :func:`registration.y_given_x`). The
  variants ``rest``, ``sync`` (from ``ref_sync``) and ``insample`` (all movement tasks)
  are sensitivity results.
* **Point:** the lower incisor ``mand_point_1``, always the incisor, from its still hold
  (:func:`frames.landmarks.digitize` with ``hold="still"``). It is expressed in the
  maxillary frame of the comparison take (``max_point_1..3`` by convention; x anterior,
  y left, z up). The condyle points are secondary; the body-origin error is a diagnostic.
* **Orientation:** the rotation vector of the mandible in the maxillary frame relative to
  the JTS-MM rest pose (:func:`kinematics.se3.rotvec_rel`, the dataset convention), the
  same reference for both systems.

The Qualisys jaw-in-head pose is mapped into the JTS-MM body frames,
``rel_Q→S = Y_HP · T_Q,HP⁻¹ T_Q,MP · Y_MP⁻¹``, so both systems share the maxillary frame
and the landmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from biomech.frames.maxillary import maxillary_frame
from biomech.kinematics import se3
from biomech.validation import registration as REG
from biomech.validation import selection as SEL
from biomech.validation.selection import Interval

VARIANTS: tuple[str, ...] = ("loto", "rest", "sync", "insample")


def mask_of(t_s: np.ndarray, intervals: Sequence[Interval]) -> np.ndarray:
    k = np.zeros(len(t_s), bool)
    for iv in intervals:
        k |= (t_s >= iv.t0) & (t_s <= iv.t1)
    return k


def loto_folds(held: Sequence[Interval]) -> dict[str, list[Interval]]:
    """Evaluated segment -> the movement tasks its ``Y`` is fitted on (never itself)."""
    mov = [h for h in held if h.name in SEL.MOVEMENT_TASKS]
    return {h.name: [m for m in mov if m is not h] for h in held}


def fit_y(t_s: np.ndarray, S: dict[str, np.ndarray], Qa: dict[str, np.ndarray],
          X: np.ndarray, intervals: Sequence[Interval], bodies: Sequence[str] = ("MP", "HP")
          ) -> dict[str, np.ndarray]:
    k = mask_of(t_s, intervals)
    return {b: REG.y_given_x(Qa[b][k], S[b][k], X) for b in bodies}


def jaw_in_head(S_head: np.ndarray, S_jaw: np.ndarray, Q_head: np.ndarray, Q_jaw: np.ndarray,
                Y_head: np.ndarray, Y_jaw: np.ndarray, T_max_hp: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """``(T_S, T_Q)``: the mandible (JTS-MM ``mta`` frame) in the maxillary frame, measured by
    JTS-MM and by Qualisys (mapped into the JTS-MM body frames)."""
    rel_S = se3.relative(S_head, S_jaw)
    rel_Q = se3.relative(Q_head, Q_jaw)
    rel_Q = np.einsum("ij,njk,kl->nil", Y_head, rel_Q, se3.inv(Y_jaw[None])[0])
    T_S = se3.compose(T_max_hp[None], rel_S)
    T_Q = se3.compose(T_max_hp[None], rel_Q)
    return T_S, T_Q


@dataclass
class Landmarks3:
    T_max_hp: np.ndarray
    incisor: np.ndarray        # mta frame
    tmj_left: np.ndarray
    tmj_right: np.ndarray
    incisor_mad_mm: float


def landmarks(opt: Any, phases: Sequence) -> Landmarks3:
    from biomech.frames.landmarks import digitize

    lm = digitize(opt, list(phases), hold="still")
    need = ("mand_point_1", "max_point_1", "max_point_2", "max_point_3", "tmj_left",
            "tmj_right")
    missing = [n for n in need if not lm.has(n)]
    if missing:
        raise LookupError(f"landmarks missing: {missing}")
    T_hp_max = maxillary_frame(lm.get("max_point_2"), lm.get("max_point_3"),
                               lm.get("max_point_1"))
    return Landmarks3(se3.inv(T_hp_max), lm.get("mand_point_1"), lm.get("tmj_left"),
                      lm.get("tmj_right"), float(lm.items["mand_point_1"].mad_mm))


def _rms(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if len(a) else float("nan")


def segment_rows(pid: str, variant: str, names: Sequence[tuple[str, str, np.ndarray]],
                 e_inc: np.ndarray, e_rot: np.ndarray, e_tl: np.ndarray, e_tr: np.ndarray,
                 e_org: np.ndarray, d_rot: np.ndarray) -> list[dict[str, Any]]:
    """One row per (segment, kind, mask): incisor error (3-D RMS, per-axis bias and SD),
    rotation error, condyle points and the body-origin diagnostic."""
    rows = []
    for seg, kind, k in names:
        n_inc = np.linalg.norm(e_inc[k], axis=1)
        ok = np.isfinite(n_inc)
        row: dict[str, Any] = {"pid": pid, "variant": variant, "segment": seg, "kind": kind,
                               "n": int(ok.sum()), "incisor_rms_mm": _rms(n_inc),
                               "rot_rms_deg": _rms(e_rot[k]),
                               "tmj_left_rms_mm": _rms(np.linalg.norm(e_tl[k], axis=1)),
                               "tmj_right_rms_mm": _rms(np.linalg.norm(e_tr[k], axis=1)),
                               "origin_rms_mm": _rms(e_org[k])}
        for i, ax in enumerate("xyz"):
            v = e_inc[k][ok, i]
            row[f"incisor_bias_{ax}_mm"] = float(np.mean(v)) if len(v) else np.nan
            row[f"incisor_sd_{ax}_mm"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
            r = d_rot[k][ok, i]
            row[f"rot_bias_r{ax}_deg"] = float(np.mean(r)) if len(r) else np.nan
        rows.append(row)
    return rows


def run_ref_kinematics(session: Any, ref: dict[str, Any], bodies: dict[str, str],
                       sync_row: pd.Series, registration: pd.DataFrame,
                       params: dict[str, Any], pid: str
                       ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    from scipy.spatial.transform import Rotation

    from biomech.io.reference import read_qtm_mat
    from biomech.validation import sync as SC

    p = {"exposure_s": 0.012, "trim_s": 0.5, "sync_pad_s": 0.5, "margin_s": 0.0025,
         "start_phase": "mand_point_1", "end_phase": "free_motion", **params}
    opt = session.optical()
    tS = SC.frame_times(opt.t_us, p["exposure_s"])
    S = {q: opt.poses[j] for q, j in bodies.items() if j in opt.poses}
    qtm = read_qtm_mat(ref["qtm_mat"])
    model = SC.TimeModel(float(sync_row["offset_s"]), float(sync_row["alpha_ppm"]) * 1e-6,
                         float(sync_row["t_ref_s"]))
    reg = registration.set_index("transform")

    def pose_of(name: str) -> np.ndarray:
        r = reg.loc[name]
        return REG.make_pose(Rotation.from_rotvec(np.radians(
            [r["rx_deg"], r["ry_deg"], r["rz_deg"]])).as_matrix(),
            np.array([r["tx_mm"], r["ty_mm"], r["tz_mm"]]))

    X = pose_of("X")
    tq = model.to_q(tS)
    Qa = {}
    for b in ("MP", "HP"):
        spl = SC.PoseSpline(qtm.t_s, qtm.pose(b))
        A = spl(tq)
        A[~spl.covered(tq, p["margin_s"])] = np.nan
        Qa[b] = A
    ct_ok = se3.valid(S["CT"])
    windows = SEL.sync_windows(session.phases, tS, ct_ok, start_phase=p["start_phase"],
                               end_phase=p["end_phase"], pad_s=p["sync_pad_s"])
    held = SEL.movement_intervals(session.phases, tS, ct_ok, trim_s=p["trim_s"])
    lm = landmarks(opt, session.phases)

    # Y per variant and evaluated segment
    mov = [h for h in held if h.name in SEL.MOVEMENT_TASKS]
    rest = [h for h in held if h.name in SEL.STATIC_TASKS]
    Y_ins = fit_y(tS, S, Qa, X, mov)
    Y_fix = {"insample": Y_ins, "sync": {b: pose_of(f"Y_{b}") for b in ("MP", "HP")}}
    if rest:
        Y_fix["rest"] = fit_y(tS, S, Qa, X, rest)
    folds = loto_folds(held)
    Y_loto = {name: fit_y(tS, S, Qa, X, tr) for name, tr in folds.items()}

    seg_masks: list[tuple[str, str, np.ndarray]] = []
    for h in held:
        seg_masks.append((h.name, "movement" if h.name in SEL.MOVEMENT_TASKS else "static",
                          mask_of(tS, [h])))
    seg_masks.append(("movement_pooled", "pooled", mask_of(tS, mov)))
    for ph in session.phases:
        if ph.name.startswith(SEL.EXCLUDED_PREFIXES):
            seg_masks.append((ph.name, "hold", (tS >= ph.t_start_us * 1e-6)
                              & (tS <= ph.t_end_us * 1e-6)))
    for w in windows:
        seg_masks.append((w.name, "sync", mask_of(tS, [w])))

    def errors(Y: dict[str, np.ndarray], k: np.ndarray | None = None):
        idx = np.arange(len(tS)) if k is None else np.flatnonzero(k)
        T_S, T_Q = jaw_in_head(S["HP"][idx], S["MP"][idx], Qa["HP"][idx], Qa["MP"][idx],
                               Y["HP"], Y["MP"], lm.T_max_hp)
        return idx, T_S, T_Q

    n = len(tS)
    out = {v: {"T_S": np.full((n, 4, 4), np.nan), "T_Q": np.full((n, 4, 4), np.nan)}
           for v in VARIANTS if v == "loto" or v in Y_fix}
    for v in out:
        if v == "loto":
            # each held segment with its own fold; everything else (holds, sync) with the
            # in-sample Y (plot only: those segments are never pooled)
            idx, T_S, T_Q = errors(Y_ins)
            out[v]["T_S"][idx], out[v]["T_Q"][idx] = T_S, T_Q
            for h in held:
                idx, T_S, T_Q = errors(Y_loto[h.name], mask_of(tS, [h]))
                out[v]["T_S"][idx], out[v]["T_Q"][idx] = T_S, T_Q
        else:
            idx, T_S, T_Q = errors(Y_fix[v])
            out[v]["T_S"][idx], out[v]["T_Q"][idx] = T_S, T_Q

    k_rest = mask_of(tS, rest) if rest else mask_of(tS, mov[:1])
    T_ref = REG.mean_pose(out["loto"]["T_S"][k_rest & se3.valid(out["loto"]["T_S"])])
    rows = []
    arrays: dict[str, np.ndarray] = {"t_s": tS - tS[0]}
    for v, d in out.items():
        T_S, T_Q = d["T_S"], d["T_Q"]
        p_S = se3.apply(T_S, lm.incisor)
        p_Q = se3.apply(T_Q, lm.incisor)
        e_inc = p_S - p_Q
        e_tl = se3.apply(T_S, lm.tmj_left) - se3.apply(T_Q, lm.tmj_left)
        e_tr = se3.apply(T_S, lm.tmj_right) - se3.apply(T_Q, lm.tmj_right)
        dp, e_rot = REG.pose_error(T_S, T_Q)
        r_S = np.degrees(se3.rotvec_rel(T_S, T_ref))
        r_Q = np.degrees(se3.rotvec_rel(T_Q, T_ref))
        rows += segment_rows(pid, v, seg_masks, e_inc, e_rot, e_tl, e_tr,
                             np.linalg.norm(dp, axis=1), r_S - r_Q)
        if v == "loto":
            arrays.update({"incisor_s_mm": p_S, "incisor_q_mm": p_Q, "rot_s_deg": r_S,
                           "rot_q_deg": r_Q, "err_incisor_mm": np.linalg.norm(e_inc, axis=1),
                           "err_rot_deg": e_rot})
    labels = np.full(n, "", dtype=object)
    for ph in session.phases:
        labels[(tS >= ph.t_start_us * 1e-6) & (tS <= ph.t_end_us * 1e-6)] = ph.name
    arrays["phase"] = labels.astype(str)
    arrays["ct_visible"] = ct_ok
    arrays["in_sync_window"] = mask_of(tS, windows)
    arrays["evaluated"] = mask_of(tS, held)

    ys = []
    for name, Y in [*((f"loto:{k}", y) for k, y in Y_loto.items()),
                    *((k, y) for k, y in Y_fix.items())]:
        for b in ("MP", "HP"):
            d_ins = REG.pose_error(Y[b][None], Y_ins[b][None])
            rv = np.degrees(Rotation.from_matrix(Y[b][:3, :3]).as_rotvec())
            ys.append({"pid": pid, "y": name, "body": b, "rx_deg": rv[0], "ry_deg": rv[1],
                       "rz_deg": rv[2], "tx_mm": Y[b][0, 3], "ty_mm": Y[b][1, 3],
                       "tz_mm": Y[b][2, 3], "d_insample_rot_deg": float(d_ins[1][0]),
                       "d_insample_trans_mm": float(np.linalg.norm(d_ins[0]))})
    meta = pd.DataFrame([{"pid": pid, "incisor_mta_x_mm": lm.incisor[0],
                          "incisor_mta_y_mm": lm.incisor[1], "incisor_mta_z_mm": lm.incisor[2],
                          "incisor_mta_dist_mm": float(np.linalg.norm(lm.incisor)),
                          "incisor_hold_mad_mm": lm.incisor_mad_mm,
                          "n_folds": len(folds)}])
    return ({"segments": pd.DataFrame(rows), "y_variants": pd.DataFrame(ys),
             "landmarks": meta}, arrays)
