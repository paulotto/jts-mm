# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD31 (analysis_plan.md of jtsmm_validation, VD29-VD33 block; J30,
R4): the nearly collinear Qualisys head cluster (descriptive, post hoc; definitions fixed in the
plan before this ran). Frozen store v14, no stage change, no re-run; per take (P1 = s10,
P2 = s14), never pooled across takes.

u1, u2, u3 = principal axes of the centred Qualisys HP template (``quality.template``, right
singular vectors; u1 the long axis; singular values s1 >= s2 >= s3).

(a) ``mc``: the §5.4 Monte Carlo of ``ref_quality:sigma_ref`` (same σ̂_m per movement segment,
    200 draws, seed 0) for the HP body alone: rotation vector of ΔH (Kabsch template -> noisy
    template, i.e. in the template frame) on u1 and on (u2, u3); RMS over the draws per
    movement task, median over the tasks (``task = "median"``); the analytic first-order
    σ̂_m/sqrt(s2² + s3²) (u1) and the (u2, u3) analogue beside it. ΔH does not depend on the
    frames, so the ≤ 100-frame subsample does not enter the HP-alone split. Reproduction: the
    stage's ``sigma_ref`` is called on the stage's inputs and must reproduce the stored
    ``sigma_ref`` row of every movement task, and its ΔH draws (captured) must equal the draws
    used here bit for bit.
(b) ``heldout``: per held-out movement interval and frame the rotation vector of
    R(T̂_Q,HP)ᵀ R(X T_S,HP Y_HP) (Qualisys HP body frame = template frame; the quantity of
    ``ref_sync:heldout`` ``rot_rms_deg``, body HP, recomputed with ``sync.PoseSpline`` and the
    stored clock, X and Y_HP exactly as ``sync.evaluate_intervals``), projected on u1, (u2, u3)
    and on the JTS-MM HP weak axis (tag baseline x of the marker model, mapped into the
    Qualisys HP frame by Y_HP: a = R_Yᵀ x̂). Pooled over the intervals (n-weighted, as §3.8 /
    ``val_table_methods``); ``heldout_const`` = the per-interval mean rotation vector (per
    interval rows: signed projections, axis directions in the ``.json``),
    ``heldout_varying`` = the remainder (the two mean squares add up to the pooled one).
    Reproduction: every interval's n and ``rot_rms_deg`` of the stored table (|Δ| < 1e-6°) and
    the pooled total against the view's number (``_heldout_rot``) within 0.05°.
    Rule: "dominated by the long axis" if the u1 component carries >= 50 % of the pooled mean
    squared difference.
(c) ``marking_frame``: rotation vector of the maxillary frame error of the marking
    (``ref_landmarks:frame_errors`` F_max, R(F_m) R(F_q)ᵀ in the Qualisys HP frame, F_m the
    JTS-MM frame mapped by the marking's Y_HP) on the same axes. Reproduction: its angle equals
    the stored ``frame_max_deg`` (|Δ| < 1e-6°).
(d) ``aligned_rms``: from ``ref_kinematics:segments_aligned`` per movement task k:
    offset = sqrt(Σ n_k |b_k|² / Σ n_k), scatter = sqrt(Σ n_k dyn_k² / Σ n_k); reproduction:
    sqrt(offset² + scatter²) equals the pooled ``incisor_rms_mm`` within 1 %.

Values: ``*_share`` components are fractions of the mean square (of the part, for
``heldout_const`` / ``heldout_varying`` the ``ms_share`` is the fraction of the pooled total);
the other angle components are RMS (deg) of that projection. The script stops on any failed
reproduction check.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/vd31_head_cluster.py [--snapshot v14]

Output: ``<validation store>/audit/vd31_head_cluster_<snap>.parquet`` (tidy: pid, part,
component, task, value, unit) and ``.json`` (settings, seeds, reproduction checks, axis angles,
git commit, runtime, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "jtsmm_validation" / "study.yaml"
PIDS = ("s10", "s14")                    # P1, P2
DRAWS, SEED, MAX_FRAMES = 200, 0, 100    # §5.4 Monte Carlo (quality.sigma_ref defaults)
POOLED_TOL_DEG = 0.05                    # plan: pooled total reproduces §3.8 within 0.05°
INTERVAL_TOL_DEG = 1e-6
FRAME_TOL_DEG = 1e-6
ALIGNED_TOL = 0.01                       # plan: within 1 %
DOMINATED = 0.5                          # plan rule: u1 share of the mean square >= 50 %
HP_TAGS = (1, 8)                         # JTS-MM head array (CRA) tag ids


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), None, snapshot)


def principal_axes(tmpl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows u1, u2, u3 and singular values of the centred template (as marker_geometry)."""
    c = tmpl - tmpl.mean(axis=0)
    _, sv, Vt = np.linalg.svd(c)
    return Vt, sv


def jts_baseline_axis() -> np.ndarray:
    """Unit tag-baseline direction of the JTS-MM HP array in its body frame (marker model)."""
    from biomech.qc.stereo_cal import marker_model

    model, _ = marker_model()
    c = [model[t].mean(axis=0) for t in HP_TAGS]
    v = c[1] - c[0]
    return v / np.linalg.norm(v)


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between two axes (sign-free), deg."""
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / np.linalg.norm(a)
                                               / np.linalg.norm(b), 0, 1))))


def split(w: np.ndarray, U: np.ndarray, a: np.ndarray) -> dict[str, float]:
    """Mean squares (deg²) of the rotation vectors w (N, 3) on u1, u2, u3, (u2,u3), a, total."""
    p = w @ U.T
    out = {f"u{k + 1}": float(np.mean(p[:, k] ** 2)) for k in range(3)}
    out["u23"] = out["u2"] + out["u3"]
    out["total"] = float(np.mean(np.sum(w ** 2, axis=1)))
    out["jts_x"] = float(np.mean((w @ a) ** 2))
    return out


def take(cfg: Any, dc: Any, pid: str, xhat: np.ndarray) -> tuple[list[dict], dict]:
    from biomech.kinematics import se3
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.validation import anatomy as AN
    from biomech.validation import quality as QY
    from biomech.validation import registration as REG
    from biomech.validation import selection as SEL
    from biomech.validation import sync as SC
    from biomech.validation.evaluate import holds_from_table, landmark_holds, pose_from_row, \
        segment_list
    from biomech.validation.landmarks import _angle_deg, x_from_tips, y_from_bodies
    from biomech.validation.stages import _systems
    from biomech.viz.views.validation_gallery import _heldout_rot

    rows: list[dict] = []
    chk: dict[str, Any] = {}

    def add(part: str, comp: str, value: float, unit: str, task: str = "") -> None:
        rows.append({"pid": pid, "part": part, "component": comp, "task": task,
                     "value": float(value), "unit": unit})

    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    try:
        ctx = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("ref_sync"),
                                 store=dc.store, participant=part, cache=cache)
        sy, *_ = _systems(ctx)
        read = lambda st, tb: dc.store.read_table(st, tb, [pid])  # noqa: E731

        # -- template and its principal axes (reproduces ref_quality:marker_geometry) ------
        tm = {b: QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers(b)},
                             sy.qtm.pose(b)) for b in ("HP", "MP")}
        U, sv = principal_axes(tm["HP"])
        mg = QY.marker_geometry(tm["HP"])
        mg_st = read("ref_quality", "marker_geometry").set_index("body").loc["HP"]
        keys = ["sv1_mm", "sv2_mm", "min_height_mm"]
        chk["marker_geometry"] = {k: {"recomputed": mg[k], "stored": float(mg_st[k])}
                                  for k in keys}
        if not all(np.isclose(mg[k], mg_st[k], rtol=1e-9, atol=1e-9) for k in keys):
            raise SystemExit(f"{pid}: marker geometry not reproduced: {chk['marker_geometry']}")
        for k, s in enumerate(sv):
            add("mc", f"s{k + 1}", s, "mm", "template")
        add("mc", "min_height", mg["min_height_mm"], "mm", "template")

        # -- (a) Monte Carlo split ------------------------------------------------------------
        sref = read("ref_quality", "sigma_ref").set_index("segment")
        hq = holds_from_table(read("ref_tip", "landmarks"), "qualisys", "primary")
        fr = AN.anatomical_frames(hq)
        rel = se3.relative(sy.qtm.pose("HP"), sy.qtm.pose("MP"))
        segs = {seg: iv for seg, kind, iv in segment_list(sy) if kind == "movement"}
        mc_rows, mc_chk = {}, {}
        for seg in SEL.MOVEMENT_TASKS:
            if seg not in segs or seg not in sref.index:
                continue
            st = sref.loc[seg]
            sh, sj = float(st["sigma_head_mm"]), float(st["sigma_jaw_mm"])
            k = (sy.t_qm >= segs[seg].t0) & (sy.t_qm <= segs[seg].t1)
            captured: list[np.ndarray] = []
            orig = QY._kabsch

            def spy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
                T = orig(A, B)
                captured.append(T)
                return T

            QY._kabsch = spy                               # in-memory only: capture ΔH
            try:
                r = QY.sigma_ref(tm["HP"], tm["MP"], sh, sj, rel[k], fr.F_max, fr.F_mand,
                                 fr.tmj_mand, draws=DRAWS, max_frames=MAX_FRAMES, seed=SEED)
            finally:
                QY._kabsch = orig
            rng = np.random.default_rng(SEED)              # the stage's head draws (first)
            noisy = tm["HP"][None] + rng.normal(0.0, sh, (DRAWS,) + tm["HP"].shape)
            dH = QY._kabsch(np.repeat(tm["HP"][None], DRAWS, axis=0), noisy)
            same = bool(captured and np.array_equal(captured[0], dH))
            rep = {a: (float(r[a]), float(st[f"sigma_ref_{a}"]))
                   for a in ("incisor_mm", "rot_deg", "tmj_mm")}
            ok = same and all(np.isclose(x, y, rtol=1e-9, atol=1e-12) for x, y in rep.values())
            mc_chk[seg] = {"draws_identical": same,
                           **{a: {"recomputed": x, "stored": y} for a, (x, y) in rep.items()},
                           "ok": ok}
            if not ok:
                raise SystemExit(f"{pid} {seg}: sigma_ref not reproduced: {mc_chk[seg]}")
            w = np.degrees(Rotation.from_matrix(dH[:, :3, :3]).as_rotvec())
            ms = split(w, U, np.zeros(3))
            an = {"u1": sh / np.hypot(sv[1], sv[2]),
                  "u2": sh / np.hypot(sv[0], sv[2]), "u3": sh / np.hypot(sv[0], sv[1])}
            vals = {"u1": np.sqrt(ms["u1"]), "u23": np.sqrt(ms["u23"]),
                    "u2": np.sqrt(ms["u2"]), "u3": np.sqrt(ms["u3"]),
                    "total": np.sqrt(ms["total"]),
                    "analytic_u1": np.degrees(an["u1"]),
                    "analytic_u23": np.degrees(np.hypot(an["u2"], an["u3"])),
                    "sigma_m": sh}
            mc_rows[seg] = vals
            for c, v in vals.items():
                add("mc", c, v, "mm" if c == "sigma_m" else "deg", seg)
        chk["sigma_ref"] = mc_chk
        med = pd.DataFrame(mc_rows).T.median()
        for c, v in med.items():
            add("mc", c, v, "mm" if c == "sigma_m" else "deg", "median")

        # -- (b) held-out head-plate difference split ----------------------------------------
        sync = read("ref_sync", "sync").iloc[0]
        reg = read("ref_sync", "registration").set_index("transform")
        X, Y = pose_from_row(reg.loc["X"]), pose_from_row(reg.loc["Y_HP"])
        a_hp = Y[:3, :3].T @ xhat                        # JTS-MM baseline in the Qualisys HP frame
        chk["axis_angle_u1_jts_x_deg"] = axis_angle_deg(U[0], a_hp)
        chk["jts_x_in_qualisys_hp"] = a_hp.tolist()
        chk["u_axes"] = U.tolist()
        model = sy.model
        spl = SC.PoseSpline(sy.qtm.t_s, sy.qtm.pose("HP"))
        ho = read("ref_sync", "heldout")
        st_hp = ho[ho["body"] == "HP"].set_index("interval")
        W, per, iv_chk = [], [], {}
        for iv in sy.held:
            if iv.name not in SEL.MOVEMENT_TASKS:
                continue
            sel = np.flatnonzero((sy.tS >= iv.t0) & (sy.tS <= iv.t1))
            tq = model.to_q(sy.tS[sel])
            Q = spl(tq)
            Q[~spl.covered(tq, 0.0025)] = np.nan          # evaluate_intervals margin
            P = REG.apply(X, sy.S["HP"][sel], Y)
            _, ang = REG.pose_error(P, Q)
            good = np.isfinite(ang)
            D = np.einsum("nji,njk->nik", Q[good, :3, :3], P[good, :3, :3])
            w = np.degrees(Rotation.from_matrix(D).as_rotvec())
            rms = float(np.sqrt(np.mean(ang[good] ** 2)))
            s = st_hp.loc[iv.name]
            iv_chk[iv.name] = {"n": int(good.sum()), "n_stored": int(s["n"]),
                               "rot_rms_deg": rms, "stored": float(s["rot_rms_deg"]),
                               "t0_equal": bool(np.isclose(iv.t0, s["t0_s"])),
                               "t1_equal": bool(np.isclose(iv.t1, s["t1_s"]))}
            if (good.sum() != int(s["n"]) or abs(rms - float(s["rot_rms_deg"])) > INTERVAL_TOL_DEG
                    or not (iv_chk[iv.name]["t0_equal"] and iv_chk[iv.name]["t1_equal"])):
                raise SystemExit(f"{pid} {iv.name}: held-out HP not reproduced: "
                                 f"{iv_chk[iv.name]}")
            W.append(w)
            per.append((iv.name, w))
            sp = split(w, U, a_hp)
            for c in ("u1", "u2", "u3", "u23", "jts_x", "total"):
                add("heldout", c, np.sqrt(sp[c]), "deg", iv.name)
            add("heldout", "n", len(w), "frames", iv.name)
        chk["heldout_intervals"] = iv_chk
        Wall = np.concatenate(W)
        tot = split(Wall, U, a_hp)
        view = _heldout_rot(ho, pid, "HP")
        chk["heldout_pooled"] = {"recomputed_deg": float(np.sqrt(tot["total"])),
                                 "view_deg": view, "tol_deg": POOLED_TOL_DEG}
        if not abs(np.sqrt(tot["total"]) - view) <= POOLED_TOL_DEG:
            raise SystemExit(f"{pid}: pooled HP rotation not reproduced: {chk['heldout_pooled']}")
        for c in ("u1", "u2", "u3", "u23", "jts_x", "total"):
            add("heldout", c, np.sqrt(tot[c]), "deg", "pooled")
        for c in ("u1", "u23", "jts_x"):
            add("heldout", f"{c}_share", tot[c] / tot["total"], "fraction", "pooled")
        add("heldout", "n", len(Wall), "frames", "pooled")
        add("heldout", "axis_angle_u1_jts_x", chk["axis_angle_u1_jts_x_deg"], "deg", "pooled")
        add("heldout", "dominated_by_u1", float(tot["u1"] / tot["total"] >= DOMINATED), "bool",
            "pooled")
        const = np.concatenate([np.repeat(w.mean(axis=0)[None], len(w), axis=0)
                                for _, w in per])
        vary = np.concatenate([w - w.mean(axis=0) for _, w in per])
        for name, arr in (("heldout_const", const), ("heldout_varying", vary)):
            sp = split(arr, U, a_hp)
            for c in ("u1", "u2", "u3", "u23", "jts_x", "total"):
                add(name, c, np.sqrt(sp[c]), "deg", "pooled")
            for c in ("u1", "u23", "jts_x"):
                add(name, f"{c}_share", sp[c] / sp["total"], "fraction", "pooled")
            add(name, "ms_share", sp["total"] / tot["total"], "fraction", "pooled")
        for nm, w in per:              # per-interval constant vector, signed (axes in .json)
            m = w.mean(axis=0)
            for k in range(3):
                add("heldout_const", f"u{k + 1}", m @ U[k], "deg", nm)
            add("heldout_const", "u23", float(np.hypot(m @ U[1], m @ U[2])), "deg", nm)
            add("heldout_const", "jts_x", m @ a_hp, "deg", nm)
            add("heldout_const", "total", float(np.linalg.norm(m)), "deg", nm)

        # -- (c) marking frame error ---------------------------------------------------------
        td = read("ref_tip", "tip_decision").iloc[0]
        d = np.array([td["dx_mm"], td["dy_mm"], td["dz_mm"]], float)
        try:
            Xm, *_ = x_from_tips(sy, d)
            Y0 = y_from_bodies(sy, Xm, "HP")
            src = "tips"
        except ValueError:
            Y0 = pose_from_row(reg.loc["Y_HP"])
            src = "sync_fit"
        src_st = read("ref_landmarks", "registration").set_index("transform").loc["Y_HP", "source"]
        hs, hql, _ = landmark_holds(sy, d)
        names = [n for n in hs if n in hql]
        Hs = {k: AN.Hold(k, hs[k].body, hs[k].point, 0, 0, 0, 0) for k in names}
        Hq = {k: AN.Hold(k, hql[k].body, hql[k].point, 0, 0, 0, 0) for k in names}
        Fm = se3.inv(Y0) @ AN.anatomical_frames(Hs).F_max
        Fq = AN.anatomical_frames(Hq).F_max
        ang = _angle_deg(Fq[:3, :3], Fm[:3, :3])
        stored = float(read("ref_landmarks", "marking_summary").iloc[0]["frame_max_deg"])
        chk["frame_max"] = {"recomputed_deg": ang, "stored_deg": stored, "y_source": src,
                            "y_source_stored": str(src_st)}
        if abs(ang - stored) > FRAME_TOL_DEG or src != src_st:
            raise SystemExit(f"{pid}: frame_max_deg not reproduced: {chk['frame_max']}")
        w = np.degrees(Rotation.from_matrix(Fm[:3, :3] @ Fq[:3, :3].T).as_rotvec())[None]
        a_m = Y0[:3, :3].T @ xhat
        sp = split(w, U, a_m)
        for c in ("u1", "u2", "u3", "u23", "jts_x", "total"):
            add("marking_frame", c, np.sqrt(sp[c]), "deg")
        for c in ("u1", "u23", "jts_x"):
            add("marking_frame", f"{c}_share", sp[c] / sp["total"], "fraction")
        add("marking_frame", "axis_angle_u1_jts_x", axis_angle_deg(U[0], a_m), "deg")

        # -- (d) offset and scatter of the aligned incisor RMS -------------------------------
        sa = read("ref_kinematics", "segments_aligned")
        mv = sa[sa["segment"].isin(SEL.MOVEMENT_TASKS) & (sa["kind"] == "movement")]
        n = mv["n"].to_numpy(float)
        b2 = (mv[["incisor_bias_x_mm", "incisor_bias_y_mm", "incisor_bias_z_mm"]].to_numpy()
              ** 2).sum(axis=1)
        off = float(np.sqrt(np.sum(n * b2) / n.sum()))
        sca = float(np.sqrt(np.sum(n * mv["incisor_dyn_rms_mm"].to_numpy() ** 2) / n.sum()))
        pooled = float(sa[sa["segment"] == "movement_pooled"].iloc[0]["incisor_rms_mm"])
        comb = float(np.hypot(off, sca))
        chk["aligned_rms"] = {"offset_mm": off, "scatter_mm": sca, "combined_mm": comb,
                              "pooled_stored_mm": pooled, "rel_diff": comb / pooled - 1,
                              "n_tasks": int(len(mv)),
                              "n_sum": int(n.sum()),
                              "n_pooled": int(sa[sa["segment"] == "movement_pooled"]
                                              .iloc[0]["n"])}
        if abs(comb / pooled - 1) > ALIGNED_TOL:
            raise SystemExit(f"{pid}: aligned incisor RMS not reproduced: {chk['aligned_rms']}")
        add("aligned_rms", "offset", off, "mm")
        add("aligned_rms", "scatter", sca, "mm")
        add("aligned_rms", "combined", comb, "mm")
        add("aligned_rms", "pooled_stored", pooled, "mm")
        return rows, chk
    finally:
        cache.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    a = ap.parse_args()
    from biomech.io.context import DataContext

    t0 = time.perf_counter()
    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    xhat = jts_baseline_axis()
    rows, checks = [], {}
    try:
        for pid in PIDS:
            r, c = take(cfg, dc, pid, xhat)
            rows += r
            checks[pid] = c
    finally:
        dc.close()
    out = pd.DataFrame(rows)
    out_dir = cfg.out_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"vd31_head_cluster_{a.snapshot}.parquet"
    out.to_parquet(f, index=False)
    sh = out[(out.part == "heldout") & (out.component == "u1_share")].set_index("pid")["value"]
    meta = {"plan": "experiments/jtsmm_validation/analysis_plan.md VD31 (a-d)",
            "snapshot": a.snapshot, "git_sha": sha, "pids": list(PIDS),
            "settings": {"draws": DRAWS, "seed": SEED, "max_frames": MAX_FRAMES,
                         "pooled_tol_deg": POOLED_TOL_DEG,
                         "interval_tol_deg": INTERVAL_TOL_DEG, "frame_tol_deg": FRAME_TOL_DEG,
                         "aligned_tol": ALIGNED_TOL, "dominated_share": DOMINATED,
                         "jts_hp_tags": list(HP_TAGS), "jts_x_body": xhat.tolist(),
                         "heldout_margin_s": 0.0025},
            "rule_dominated_by_u1": {p: bool(v >= DOMINATED) for p, v in sh.items()},
            "reproduction": checks,
            "runtime_s": round(time.perf_counter() - t0, 1),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 400)
    show = out[out.task.isin(["median", "pooled", "", "template"])]
    print(show.pivot_table(index=["part", "component"], columns="pid", values="value")
          .to_string(float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
