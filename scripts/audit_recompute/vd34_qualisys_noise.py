# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD34 (analysis_plan.md of jtsmm_validation, entry VD34-VD35; J30):
is the Qualisys noise model too large? Sensitivity of σ_Q and of the VD22 budget (descriptive,
post hoc; definitions fixed in the plan before this ran). Frozen store v14, no stage change,
no re-run; per take (P1 = s10, P2 = s14).

Observed high-frequency orientation scatter, per take, body (HP, MP) and movement task, over
the body's consistent Qualisys frames (``quality.consistency`` gate of that body off, valid
pose) at the native 200 Hz: RMS of the rotation vector of R_lp(t)ᵀ R(t) (body frame = template
frame), R_lp the zero-phase 4th-order Butterworth 6-Hz low-pass of the orientation
(``kinematics.se3.lowpass``: rotation vector relative to the mean orientation of each
contiguous run of consistent frames, ``sosfiltfilt`` of a 4th-order section; runs shorter than
its ``min_len`` = 15 frames are left out), on u1, (u2, u3) and 3-D (principal axes of the
centred template of that body, ``quality.template``).

Predicted: the §5.4 Monte Carlo rotation of the same body alone (the ΔH resp. ΔJ draws of
``quality.sigma_ref`` for that segment: same σ̂_m, 200 draws, seed 0; white per frame) on the
same axis groups, × sqrt((f_N − 6)/f_N), f_N = 100 Hz. ρ = observed / predicted per task; per
body and axis group the median over the tasks.

Sensitivity (not a replacement of the primary budget): σ̂_m of each body (every segment of
``ref_quality:sigma_ref``) × its median 3-D ρ when ρ < 1 (unchanged otherwise), the §5.4 Monte
Carlo rerun (``quality.sigma_ref``, same seed and frames), the motion rows of the
``ref_kinematics`` budget recomputed with that σ_Q (``quality.budget`` on the stored σ_obs) and
the VD22 table rebuilt with ``validation.budget.error_budget`` (marking rows unchanged: their
σ_ref is the landmark σ_ref, outside VD34). Medians over the movement tasks as the macros
``budget{Jts,Share}{Incisor,Condyle,Rotation}Tasks…`` (part motion, kind movement; the median
of ``sigma_jts_corrected`` skips the not-resolvable tasks like the macros).

Reproduction first: with ρ = 1 the script reproduces the stored ``ref_quality:sigma_ref`` of
both takes (every segment, rtol 1e-12) and the stored ``ref_budget:budget`` (whole table) or it
stops; the template singular values reproduce ``ref_quality:marker_geometry``.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/vd34_qualisys_noise.py [--snapshot v14]

Output: ``<validation store>/audit/vd34_qualisys_noise_<snap>.parquet`` (tidy; ``part``:
``rho`` -- body, task, axis_group, observed_deg, predicted_deg, rho, n frames; ``sigma_m`` --
body, task, stored / sensitivity; ``sigma_q`` -- task, outcome, stored / sensitivity;
``budget`` -- task, outcome, quantity (sigma_obs, sigma_ref, sigma_jts_corrected,
ref_var_share, separable), stored / sensitivity; ``task = "median"`` rows = median over the
movement tasks) and ``.json`` (settings, scale factors, reproduction checks, git commit,
runtime, sha256).
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
BODIES = ("HP", "MP")
DRAWS, SEED, MAX_FRAMES = 200, 0, 100    # §5.4 Monte Carlo (quality.sigma_ref defaults)
FC_HZ, ORDER, MIN_LEN = 6.0, 4, 15       # low-pass; se3.lowpass min_len
F_NYQ = 100.0                            # plan: f_N = 100 Hz
WHITE = np.sqrt((F_NYQ - FC_HZ) / F_NYQ)
OUTCOMES = (("incisor", "sigma_ref_incisor_mm", "mm"), ("condyle", "sigma_ref_tmj_mm", "mm"),
            ("rotation", "sigma_ref_rot_deg", "deg"))
QUANTITIES = ("sigma_obs", "sigma_ref", "sigma_jts_corrected", "ref_var_share", "separable")


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), None, snapshot)


def principal_axes(tmpl: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows u1, u2, u3 and singular values of the centred template."""
    _, sv, Vt = np.linalg.svd(tmpl - tmpl.mean(axis=0))
    return Vt, sv


def groups(w: np.ndarray, U: np.ndarray) -> dict[str, float]:
    """RMS (deg) of rotation vectors w (N, 3, deg) on u1, (u2, u3) and 3-D."""
    p = w @ U.T
    return {"u1": float(np.sqrt(np.mean(p[:, 0] ** 2))),
            "u23": float(np.sqrt(np.mean(p[:, 1] ** 2 + p[:, 2] ** 2))),
            "3d": float(np.sqrt(np.mean(np.sum(w ** 2, axis=1))))}


def sigma_ref_with_draws(QY: Any, *args: Any, **kw: Any) -> tuple[dict, list[np.ndarray]]:
    """``quality.sigma_ref`` and the ΔH, ΔJ draws it used (``_kabsch`` captured in memory)."""
    captured: list[np.ndarray] = []
    orig = QY._kabsch

    def spy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        T = orig(A, B)
        captured.append(T)
        return T

    QY._kabsch = spy
    try:
        r = QY.sigma_ref(*args, **kw)
    finally:
        QY._kabsch = orig
    return r, captured


def sigma_ref_table(QY: Any, sy: Any, segs: list, tm: dict, fr: Any, sig: dict,
                    scale: dict[str, float], pid: str) -> pd.DataFrame:
    """The ``sigma_ref`` loop of ``evaluate.run_ref_quality`` with σ̂_m × ``scale[body]``."""
    from biomech.kinematics import se3

    rel = se3.relative(sy.qtm.pose("HP"), sy.qtm.pose("MP"))
    rows = []
    for seg, kind, iv in segs:
        k = (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
        sh, sj = sig.get(("HP", seg), np.nan), sig.get(("MP", seg), np.nan)
        if k.sum() < 5 or not (np.isfinite(sh) and np.isfinite(sj)):
            continue
        sh, sj = sh * scale["HP"], sj * scale["MP"]
        r = QY.sigma_ref(tm["HP"], tm["MP"], sh, sj, rel[k], fr.F_max, fr.F_mand, fr.tmj_mand,
                         draws=DRAWS, max_frames=MAX_FRAMES, seed=SEED)
        rows.append({"pid": pid, "segment": seg, "kind": kind, "sigma_head_mm": sh,
                     "sigma_jaw_mm": sj, **{f"sigma_ref_{a}": v for a, v in r.items()}})
    return pd.DataFrame(rows)


def kin_budget(QY: Any, stored_kin: pd.DataFrame, sref: pd.DataFrame) -> pd.DataFrame:
    """The ``ref_kinematics`` budget rows with σ_ref from ``sref`` (σ_obs as stored)."""
    s = sref.set_index("segment")
    col = {o: c for o, c, _ in OUTCOMES}
    rows = []
    for r in stored_kin.to_dict("records"):
        b = QY.budget(float(r["sigma_obs"]), float(s.loc[r["segment"], col[r["outcome"]]]))
        rows.append({"pid": r["pid"], "segment": r["segment"], "kind": r["kind"],
                     "outcome": r["outcome"], **b})
    return pd.DataFrame(rows)[[c for c in stored_kin.columns if c != "unit"]]


def take(cfg: Any, dc: Any, pid: str) -> dict[str, Any]:
    from biomech.kinematics import se3
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.validation import quality as QY
    from biomech.validation import selection as SEL
    from biomech.validation.evaluate import holds_from_table, segment_list
    from biomech.validation import anatomy as AN
    from biomech.validation.stages import _systems
    from biomech import timebase

    read = lambda st, tb: dc.store.read_table(st, tb, [pid])  # noqa: E731
    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    try:
        ctx = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("ref_sync"),
                                 store=dc.store, participant=part, cache=cache)
        sy, *_ = _systems(ctx)
        fs = float(sy.qtm.fs_hz)
        if not np.isclose(fs, 2 * F_NYQ):
            raise SystemExit(f"{pid}: Qualisys rate {fs} Hz, plan assumes {2 * F_NYQ} Hz")
        tm = {b: QY.template({k: sy.qtm.markers[k] for k in sy.qtm.body_markers(b)},
                             sy.qtm.pose(b)) for b in BODIES}
        ax = {b: principal_axes(tm[b]) for b in BODIES}
        mg = read("ref_quality", "marker_geometry").set_index("body")
        geo_ok = all(np.isclose(ax[b][1][i], mg.loc[b, f"sv{i + 1}_mm"], rtol=1e-9, atol=1e-9)
                     for b in BODIES for i in range(2))
        if not geo_ok:
            raise SystemExit(f"{pid}: template singular values not reproduced")
        cons = read("ref_quality", "consistency")
        sig = {(r["body"], r["segment"]): float(r["sigma_marker_mm"])
               for r in cons.to_dict("records")}
        hq = holds_from_table(read("ref_tip", "landmarks"), "qualisys", "primary")
        fr = AN.anatomical_frames(hq)
        segs = segment_list(sy)
        stored_sref = read("ref_quality", "sigma_ref")

        # -- reproduction: ρ = 1 ---------------------------------------------------------------
        rep = sigma_ref_table(QY, sy, segs, tm, fr, sig, {"HP": 1.0, "MP": 1.0}, pid)
        cols = [c for c in stored_sref.columns if c.startswith("sigma_")]
        same = (len(rep) == len(stored_sref)
                and (rep["segment"].to_numpy() == stored_sref["segment"].to_numpy()).all()
                and np.allclose(rep[cols].to_numpy(float), stored_sref[cols].to_numpy(float),
                                rtol=1e-12, atol=0))
        if not same:
            raise SystemExit(f"{pid}: sigma_ref not reproduced with rho = 1")

        # -- observed and predicted scatter per body and movement task ------------------------
        gate = {b: QY.consistency({k: sy.qtm.markers[k] for k in sy.qtm.body_markers(b)},
                                  {k: sy.qtm.marker_residual[k] for k in sy.qtm.body_markers(b)
                                   if k in sy.qtm.marker_residual},
                                  {k: sy.qtm.marker_type[k] for k in sy.qtm.body_markers(b)
                                   if k in sy.qtm.marker_type})["gate"].to_numpy()
                for b in BODIES}
        rel = se3.relative(sy.qtm.pose("HP"), sy.qtm.pose("MP"))
        srs = stored_sref.set_index("segment")
        rho_rows = []
        for seg, kind, iv in segs:
            if kind != "movement" or seg not in SEL.MOVEMENT_TASKS:
                continue
            k = (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1)
            st = srs.loc[seg]
            _, draws = sigma_ref_with_draws(
                QY, tm["HP"], tm["MP"], float(st["sigma_head_mm"]), float(st["sigma_jaw_mm"]),
                rel[k], fr.F_max, fr.F_mand, fr.tmj_mand, draws=DRAWS, max_frames=MAX_FRAMES,
                seed=SEED)
            for b, dT in zip(BODIES, draws[:2]):
                U = ax[b][0]
                T = sy.qtm.pose(b).copy()
                use = k & se3.valid(T) & ~gate[b]
                T[~use] = np.nan
                Tlp = se3.lowpass(T, fs, FC_HZ, order=ORDER, min_len=MIN_LEN)
                keep = np.zeros(len(T), bool)
                for a, e in timebase.segments(use):
                    if e - a >= MIN_LEN:
                        keep[a:e] = True
                D = np.einsum("nji,njk->nik", Tlp[keep, :3, :3], T[keep, :3, :3])
                w = np.degrees(Rotation.from_matrix(D).as_rotvec())
                obs = groups(w, U)
                pred = groups(np.degrees(Rotation.from_matrix(dT[:, :3, :3]).as_rotvec()), U)
                for g in ("u1", "u23", "3d"):
                    p = pred[g] * WHITE
                    rho_rows.append({"pid": pid, "part": "rho", "body": b, "task": seg,
                                     "axis_group": g, "observed_deg": obs[g],
                                     "predicted_deg": p, "rho": obs[g] / p,
                                     "n": int(keep.sum()), "n_task": int(k.sum())})
        rho = pd.DataFrame(rho_rows)
        med = rho.groupby(["body", "axis_group"])[["observed_deg", "predicted_deg", "rho", "n"]] \
            .median().reset_index().assign(pid=pid, part="rho", task="median")
        rho = pd.concat([rho, med], ignore_index=True)
        r3 = med[med["axis_group"] == "3d"].set_index("body")["rho"]
        scale = {b: float(min(r3[b], 1.0)) for b in BODIES}

        # -- sensitivity: scaled σ̂_m, Monte Carlo rerun -----------------------------------
        sens = sigma_ref_table(QY, sy, segs, tm, fr, sig, scale, pid)
        return {"pid": pid, "rho": rho, "scale": scale, "rho3d": r3.to_dict(),
                "stored_sref": stored_sref, "sens_sref": sens, "rep_sref": rep,
                "stored_kin": read("ref_kinematics", "budget"),
                "marking": read("ref_landmarks", "marking_summary"),
                "sv": {b: ax[b][1].tolist() for b in BODIES},
                "u": {b: ax[b][0].tolist() for b in BODIES},
                "gated_frac_movement": {b: float(gate[b][np.any([
                    (sy.t_qm >= iv.t0) & (sy.t_qm <= iv.t1) for s_, kd, iv in segs
                    if kd == "movement"], axis=0)].mean()) for b in BODIES}}
    finally:
        cache.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    a = ap.parse_args()
    from biomech.io.context import DataContext
    from biomech.validation import quality as QY
    from biomech.validation.budget import DEFAULTS as BD
    from biomech.validation.budget import error_budget

    t0 = time.perf_counter()
    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    try:
        res = [take(cfg, dc, pid) for pid in PIDS]
        stored_budget = dc.store.read_table("ref_budget", "budget", None)
    finally:
        dc.close()
    marking = pd.concat([r["marking"] for r in res], ignore_index=True)
    ratio = float(BD["adequacy_ratio"])

    # reproduction of the whole VD22 table with ρ = 1
    kin_rep = pd.concat([kin_budget(QY, r["stored_kin"], r["rep_sref"]) for r in res],
                        ignore_index=True)
    rep_budget = error_budget(kin_rep, marking, ratio)
    try:
        pd.testing.assert_frame_equal(rep_budget.reset_index(drop=True),
                                      stored_budget[rep_budget.columns].reset_index(drop=True),
                                      check_exact=False, rtol=1e-12, atol=0, check_dtype=False)
        budget_same = True
    except AssertionError as e:
        raise SystemExit(f"ref_budget:budget not reproduced with rho = 1: {e}")
    kin_sens = pd.concat([kin_budget(QY, r["stored_kin"], r["sens_sref"]) for r in res],
                         ignore_index=True)
    sens_budget = error_budget(kin_sens, marking, ratio)

    rows: list[dict] = []
    for r in res:
        rows += r["rho"].to_dict("records")
        pid = r["pid"]
        for b, col in (("HP", "sigma_head_mm"), ("MP", "sigma_jaw_mm")):
            for (seg, kind, v0), v1 in zip(r["stored_sref"][["segment", "kind", col]]
                                           .itertuples(index=False),
                                           r["sens_sref"][col]):
                rows.append({"pid": pid, "part": "sigma_m", "body": b, "task": seg,
                             "kind": kind, "stored": v0, "sensitivity": v1, "unit": "mm"})
        for oc, col, unit in OUTCOMES:
            for (seg, kind, v0), v1 in zip(r["stored_sref"][["segment", "kind", col]]
                                           .itertuples(index=False), r["sens_sref"][col]):
                rows.append({"pid": pid, "part": "sigma_q", "task": seg, "kind": kind,
                             "outcome": oc, "stored": v0, "sensitivity": v1, "unit": unit})
    key = ["pid", "part", "segment", "kind", "outcome"]
    m = stored_budget.merge(sens_budget, on=key, suffixes=("_st", "_se"))
    m = m[m["part"] == "motion"]
    for q in QUANTITIES:
        for x in m.to_dict("records"):
            rows.append({"pid": x["pid"], "part": "budget", "task": x["segment"],
                         "kind": x["kind"], "outcome": x["outcome"], "quantity": q,
                         "stored": float(x[f"{q}_st"]), "sensitivity": float(x[f"{q}_se"]),
                         "unit": "" if q in ("ref_var_share", "separable") else x["unit_st"]})
    out = pd.DataFrame(rows)
    # medians over the movement tasks (macro filters: part motion, kind movement)
    mv = out[out["part"].isin(["sigma_q", "budget"]) & (out["kind"] == "movement")]
    med = mv.groupby(["pid", "part", "outcome", "quantity"], dropna=False)[
        ["stored", "sensitivity"]].median().reset_index()
    med = med.merge(mv[["pid", "part", "outcome", "quantity", "unit"]].drop_duplicates(),
                    on=["pid", "part", "outcome", "quantity"], how="left")
    out = pd.concat([out, med.assign(task="median", kind="movement")], ignore_index=True)
    cols = ["pid", "part", "body", "task", "kind", "axis_group", "outcome", "quantity",
            "observed_deg", "predicted_deg", "rho", "n", "n_task", "stored", "sensitivity",
            "unit"]
    out = out.reindex(columns=cols)

    out_dir = cfg.out_dir / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    f = out_dir / f"vd34_qualisys_noise_{a.snapshot}.parquet"
    out.to_parquet(f, index=False)
    meta = {"plan": "experiments/jtsmm_validation/analysis_plan.md VD34 (commit 58ede2a6)",
            "snapshot": a.snapshot, "git_sha": sha, "pids": list(PIDS),
            "settings": {"draws": DRAWS, "seed": SEED, "max_frames": MAX_FRAMES,
                         "lowpass_hz": FC_HZ, "butter_order": ORDER, "zero_phase": True,
                         "min_run_frames": MIN_LEN, "f_nyquist_hz": F_NYQ,
                         "white_factor": float(WHITE), "adequacy_ratio": ratio},
            "rho3d_median": {r["pid"]: r["rho3d"] for r in res},
            "sigma_m_scale": {r["pid"]: r["scale"] for r in res},
            "template_sv_mm": {r["pid"]: r["sv"] for r in res},
            "template_u": {r["pid"]: r["u"] for r in res},
            "gated_frac_movement": {r["pid"]: r["gated_frac_movement"] for r in res},
            "reproduction": {"sigma_ref_rho1_rtol": 1e-12, "sigma_ref_equal": True,
                             "budget_rho1_equal": budget_same,
                             "budget_rows": int(len(rep_budget))},
            "runtime_s": round(time.perf_counter() - t0, 1),
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1, default=float))
    pd.set_option("display.width", 220)
    pd.set_option("display.max_rows", 400)
    rr = out[(out.part == "rho")]
    print(rr.pivot_table(index=["pid", "body", "task"], columns="axis_group",
                         values="rho").round(3).to_string())
    print(json.dumps({k: meta[k] for k in ("rho3d_median", "sigma_m_scale")}, indent=1))
    md = out[(out.task == "median") & out.part.isin(["sigma_q", "budget"])]
    print(md[["pid", "part", "outcome", "quantity", "stored", "sensitivity", "unit"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
