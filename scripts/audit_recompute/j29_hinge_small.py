# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J29: the hinge at the joint over the first part of the opening (exploratory, post
hoc; definitions fixed in J29 before this ran).

Per participant of the paper store (pin r16a, reporting_population) the ``uncertainty`` stage's C1
quantities of the hinge at the joint (the fixed axis through the condyle points, D33) are
recomputed on truncated open-close openings: each opening kept up to the last frame before the
incisal opening (drop of the incisor below its position at the opening start) first exceeds
the level, or before the opening angle exceeds 12°. The observed incisor RMS residual and its
in-vivo null ``invivo_tmj`` (1000 draws, the participant's rest residuals, seed = participant
seed + 6) are both computed on the truncated openings with the stage's own functions.

Reproduction first: without truncation every participant's stored ``obs_hinge_tmj_mm`` and
``null_invivo_tmj_p99_mm`` are reproduced exactly (the script stops otherwise).

Levels: ``primary`` 10 mm, 20 mm, 12°; ``detect`` 2, 4, ..., 30 mm in ascending order up to the
first grid level at or above the participant's largest opening, stopping at the first level
whose residual exceeds its P99.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j29_hinge_small.py [-j 8]

Output: ``<adults store>/audit/j29_hinge_small_<snap>.parquet`` (participant x level) and
``.json`` (settings, reproduction, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
PRIMARY = (("mm", 10.0), ("mm", 20.0), ("deg", 12.0))
DETECT = tuple(float(x) for x in range(2, 31, 2))
MIN_OPENINGS, MIN_FRAMES, MIN_VALID = 2, 10, 5      # as the uncertainty stage / openings


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def truncate(jm: Any, ops: list, kind: str, level: float) -> list:
    """Each opening up to the last frame before it first exceeds ``level`` (incisal opening in
    mm or opening angle in deg); openings with fewer than 5 valid frames left are dropped."""
    from biomech.kinematics import se3
    from biomech.kinematics.coupling import Opening

    if kind == "full":
        return list(ops)
    out = []
    for op in ops:
        if kind == "mm":
            inc = se3.apply(op.T, jm.points_mp["incisor"])
            v = -(inc[:, 2] - inc[0, 2])
        else:
            v = np.asarray(op.theta, float)
        over = np.flatnonzero(np.nan_to_num(v, nan=-np.inf) > level)
        k = int(over[0]) if over.size else len(v)
        T = op.T[:k]
        if k == 0 or se3.valid(T).sum() < MIN_VALID or not se3.valid(T[:1])[0]:
            continue
        out.append(Opening(op.cycle, T, op.theta[:k]))
    return out


def largest_opening(jm: Any, ops: list) -> float:
    from biomech.kinematics import se3

    m = [np.nanmax(-(se3.apply(op.T, jm.points_mp["incisor"])[:, 2]
                     - se3.apply(op.T[:1], jm.points_mp["incisor"])[0, 2])) for op in ops]
    return float(np.nanmax(m)) if m else float("nan")


def evaluate(jm: Any, ops: list, T_rest: Any, floor_t: np.ndarray, floor_r: np.ndarray,
             p: dict, kp: Any, mf: int, seed: int) -> dict[str, Any]:
    """Observed hinge-at-the-joint residual and its invivo_tmj null on ``ops`` (stage code)."""
    from biomech.kinematics.clinical import REST_HP_S
    from biomech.kinematics.coupling import hinge_data, hinge_through
    from biomech.uncertainty import montecarlo as mc

    A, B, _ = hinge_data(jm, ops, mf)
    row: dict[str, Any] = {"n_openings": len(ops), "n_frames": int(A.shape[0])}
    if len(ops) < MIN_OPENINGS or A.shape[0] < MIN_FRAMES:
        return {**row, "tested": False, "why": "too few openings or frames"}
    obs = float(hinge_through(A, B, A[:, 3].mean(axis=0), A[:, 4].mean(axis=0)).rms(0))
    ax = mc.condyle_axis(jm, ops, mf)
    d = mc.null_distributions(jm, ops, ax[0], ax[1], floor_t, floor_r, int(p["n_sims"]),
                              kp.lowpass_hz, mf, seed + 6, kp.lowpass_order,
                              noise=mc.rest_noise(T_rest, jm.fs, REST_HP_S),
                              through_condyles=True)["hinge_tmj"]
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {**row, "obs_mm": obs, "tested": False, "why": "no null"}
    p99 = float(np.percentile(d, 99))
    return {**row, "obs_mm": obs, "p50_mm": float(np.median(d)), "p99_mm": p99,
            "ratio": obs / p99 if p99 > 0 else np.nan, "above": bool(obs > p99),
            "p_mc": mc.mc_pvalue(obs, d), "n_sims": int(d.size), "tested": True, "why": ""}


def participant(args: tuple[str, str]) -> list[dict[str, Any]]:
    snapshot, pid = args
    from biomech.io.context import DataContext
    from biomech.kinematics.clinical import find_task_cycles
    from biomech.kinematics.coupling import openings
    from biomech.kinematics.pose_chain import KinParams, build
    from biomech.pipeline import ParticipantContext, UnitCache

    cfg = _config(snapshot)
    dc = DataContext(cfg)
    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    t0 = time.perf_counter()
    try:
        ctx = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("uncertainty"),
                                 store=dc.store, participant=part, cache=cache)
        p = ctx.params
        kp = KinParams.from_dict(ctx.signals.get("kinematics"))
        kparams = cfg.params("kinematics")
        anat = ctx.upstream("kinematics", "anatomy").iloc[0]
        sd = ctx.session("motion")
        jm = build(sd, kp)
        ph = sd.phase("open_close")
        tc = find_task_cycles(jm, "open_close", ph.t_start_us, ph.t_end_us,
                              kparams["cycles"])[0]
        ops = openings(jm, tc)
        mf = kparams["hinge"]["max_frames_per_cycle"]
        rest = sd.phase("rest")
        T_rest = jm.T_raw[jm.window(rest.t_start_us, rest.t_end_us)] if rest is not None \
            else None
        floor_t = np.array([float(anat.get(f"floor_trans_sd_mm_raw_{a}", np.nan)) for a in "xyz"])
        floor_r = np.array([float(anat.get(f"floor_rot_sd_deg_raw_{a}", np.nan)) for a in "xyz"])
        seed = ctx.seed

        def run(kind: str, level: float, purpose: str) -> dict[str, Any]:
            r = evaluate(jm, truncate(jm, ops, kind, level), T_rest, floor_t, floor_r, p, kp,
                         mf, seed)
            return {"pid": pid, "purpose": purpose, "kind": kind, "level": level, **r}

        rows = [run("full", np.inf, "reproduction")]
        rows += [run(k, lv, "primary") for k, lv in PRIMARY]
        top = largest_opening(jm, ops)
        rows[0]["largest_opening_mm"] = top
        for lv in DETECT:
            r = run("mm", lv, "detect")
            rows.append(r)
            if r.get("above") or not np.isfinite(top) or lv >= top:
                break
        for r in rows:
            r["seconds"] = time.perf_counter() - t0
        return rows
    finally:
        cache.close()
        dc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    ap.add_argument("-j", "--jobs", type=int, default=8)
    ap.add_argument("--pids", nargs="*", help="only these participants (timing test)")
    a = ap.parse_args()
    from biomech.io.context import DataContext

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    try:
        pids = list(dc.pids())
        stored = dc.table("uncertainty", "hinge_noise").set_index("pid")
    finally:
        dc.close()
    if a.pids:
        pids = [q for q in pids if q in a.pids]
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        out = pd.DataFrame([r for rows in ex.map(participant, [(a.snapshot, q) for q in pids])
                            for r in rows])
    rep = out[out.purpose == "reproduction"].set_index("pid")
    repro = {q: {"obs": float(rep.at[q, "obs_mm"]), "stored_obs": float(stored.at[q, "obs_hinge_tmj_mm"]),
                 "p99": float(rep.at[q, "p99_mm"]),
                 "stored_p99": float(stored.at[q, "null_invivo_tmj_p99_mm"])} for q in rep.index}
    for v in repro.values():
        v["equal"] = bool(np.isclose(v["obs"], v["stored_obs"], rtol=1e-9, atol=1e-12)
                          and np.isclose(v["p99"], v["stored_p99"], rtol=1e-9, atol=1e-12))
    if not all(v["equal"] for v in repro.values()):
        raise SystemExit(f"reproduction failed: {json.dumps(repro, indent=1)}")
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    tag = a.snapshot + ("_partial" if a.pids else "")
    f = out_dir / f"j29_hinge_small_{tag}.parquet"
    out.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(f)
    meta = {"snapshot": a.snapshot, "git_sha": sha, "variant": VARIANT,
            "primary": [list(x) for x in PRIMARY], "detect_mm": list(DETECT),
            "n_participants": len(pids), "reproduction": repro,
            "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    f.with_suffix(".json").write_text(json.dumps(meta, indent=1))
    pd.set_option("display.width", 200)
    cols = ["pid", "purpose", "kind", "level", "n_openings", "n_frames", "obs_mm", "p99_mm",
            "ratio", "above", "tested", "seconds"]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(f"wrote {f}")


if __name__ == "__main__":
    main()
