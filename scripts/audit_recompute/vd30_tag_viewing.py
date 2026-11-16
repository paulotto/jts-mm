# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD30: viewing geometry of the JTS-MM tags (validation analysis plan
``experiments/jtsmm_validation/analysis_plan.md``, block VD29-VD33 and the VD29 / VD30
amendment; J30; descriptive, post hoc, definitions fixed before this
ran).

Per comparison take (P1 = s10, P2 = s14; never pooled), array (HP = CRA, MP = MTA) and camera,
over the held-out movement frames, per detected tag (every tag the tracker detected in that
camera, ``optical/corners``):

* incidence angle between the tag normal and the line from the tag centre to the camera centre
  (deg, 0 = frontal), from the VD29 refit pose (trimmed Kabsch, as the tracker);
* tag side in pixels (mean of the four edges of the detected quadrilateral);
* reprojection residual of the tag (RMS over its four corners of the corner distance, px,
  refit pose);
* detection rate per tag and per array with >= 1 tag: share of the optical frames of the
  movement intervals.

Median and quartiles. Binned by incidence (15 deg bins, cells >= 50): the median (and RMS)
reprojection residual per detection; the orientation jitter per frame (stored poses, three
consecutive valid frames: angle between R(t) and the geodesic midpoint of R(t-1), R(t+1),
x sqrt(2/3) = the 3-D rotation noise for white noise) and the VD29 per-frame Monte Carlo 3-D
sigma_theta, both binned by the frame's mean incidence of the array's tags over both cameras.

Rule for "the head tags are seen less frontally and tracked worse": supported only if, in both
takes, (i) HP median incidence > MP in both cameras, (ii) HP median jitter > MP, (iii) within
each array the binned jitter does not decrease with incidence over the 15 deg bins overlapping
[min Q1, max Q3] of the per-frame mean incidence of the two arrays (amendment); with one bin
per array (iii) is not evaluable and read as not met (addendum 2).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/vd30_tag_viewing.py [--snapshot v14]

Needs ``vd29_array_noise_frames_<snap>.parquet`` (run vd29_array_noise.py first). Output
(``<validation store>/audit/``): ``vd30_tag_viewing_<snap>.parquet`` (summaries),
``vd30_tag_viewing_bins_<snap>.parquet`` (binned) and ``vd30_tag_viewing_<snap>.json``
(settings, rule, reproduction checks, git commit, runtime, sha256).
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
import vd29_array_noise as V  # noqa: E402 -- shared refit (same directory)

BIN_DEG = 15.0
MIN_CELL = 50
JITTER_FACTOR = float(np.sqrt(2.0 / 3.0))
GAP_TOL = 1.5                       # consecutive frames: both gaps < 1.5 x median frame period
PAPER = {"CRA": "HP", "MTA": "MP"}


def detections(tc: dict[str, Any]) -> pd.DataFrame:
    """Every detected array tag per (optical row, camera), first detection kept."""
    import h5py

    with h5py.File(tc["h5"], "r") as f:
        g = f["optical/corners"]
        cam = np.asarray(g["cam"][()], int)
        row = np.asarray(g["optical_row"][()], np.int64)
        tag = np.asarray(g["tag_id"][()], int)
        cp = np.asarray(g["corners_px"][()], float)
        ids = [c.decode() if isinstance(c, bytes) else str(c)
               for c in g.attrs.get("cam_ids", [b"cam0", b"cam1"])]
        Tw = {c: np.asarray(f[f"optical/corners/calibration/{c}/T_world_cam"][()], float)
              for c in ("cam0", "cam1")}
    arr_of = {t: a for a in V.ARRAYS for t in tc["tags"][a]}
    keep = np.isin(tag, list(arr_of)) & tc["mov"][row]
    cam, row, tag, cp = cam[keep], row[keep], tag[keep], cp[keep]
    _, first = np.unique(row * 64 + cam * 32 + tag, return_index=True)
    first = np.sort(first)
    d = pd.DataFrame({"row": row[first], "camera": [ids[c] for c in cam[first]],
                      "tag": tag[first]})
    d["array"] = d["tag"].map(arr_of)
    return d, cp[first], Tw, ids


def per_detection(tc: dict[str, Any], rf: dict[str, Any], d: pd.DataFrame, cp: np.ndarray,
                  Tw: dict[str, np.ndarray]) -> pd.DataFrame:
    """Incidence (refit pose), tag side and reprojection residual per detection."""
    model, cal = tc["model"], tc["cal"]
    edges = np.linalg.norm(np.roll(cp, -1, axis=1) - cp, axis=2)
    d = d.copy()
    d["side_px"] = edges.mean(axis=1)
    inc = np.full(len(d), np.nan)
    res = np.full(len(d), np.nan)
    tags = d["tag"].to_numpy()
    rows = d["row"].to_numpy()
    camv = d["camera"].to_numpy()
    for a in V.ARRAYS:
        P = rf["poses"][a]
        for t in tc["tags"][a]:
            c = model[t]
            n_b = np.cross(c[1] - c[0], c[0] - c[3])
            n_b /= np.linalg.norm(n_b)
            for cam in ("cam0", "cam1"):
                m = (tags == t) & (camv == cam)
                m &= np.isfinite(P[rows]).all(axis=(1, 2))
                if not m.any():
                    continue
                R, tt = P[rows[m], :3, :3], P[rows[m], :3, 3]
                ctr = np.einsum("fij,j->fi", R, c.mean(axis=0)) + tt
                nw = np.einsum("fij,j->fi", R, n_b)
                v = Tw[cam][:3, 3][None] - ctr
                cosang = np.sum(nw * v, axis=1) / np.linalg.norm(v, axis=1)
                inc[m] = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                Xw = np.einsum("fij,kj->fki", R, c) + tt[:, None, :]
                uv = V.project(cal, Xw, 0 if cam == "cam0" else 1)
                res[m] = np.sqrt(np.mean(np.sum((uv - cp[m]) ** 2, axis=2), axis=1))
    d["incidence_deg"] = inc
    d["residual_px"] = res
    return d


def jitter(tc: dict[str, Any], a: str) -> pd.DataFrame:
    """Orientation jitter per movement frame from the stored poses."""
    from biomech.kinematics import se3
    from scipy.spatial.transform import Rotation

    S = tc["stored"][a]
    ok = se3.valid(S)
    t = tc["t_us"].astype(float)
    dt = np.diff(t)
    med = np.median(dt)
    T = len(S)
    i = np.arange(1, T - 1)
    good = (tc["mov"][i] & ok[i] & ok[i - 1] & ok[i + 1]
            & (dt[i - 1] < GAP_TOL * med) & (dt[i] < GAP_TOL * med))
    i = i[good]
    Ra, Rb, Rt = S[i - 1, :3, :3], S[i + 1, :3, :3], S[i, :3, :3]
    half = Rotation.from_rotvec(0.5 * V.rotvec(np.einsum("fji,fjk->fik", Ra, Rb))).as_matrix()
    Rm = np.einsum("fij,fjk->fik", Ra, half)
    ang = np.degrees(np.linalg.norm(V.rotvec(np.einsum("fji,fjk->fik", Rm, Rt)), axis=1))
    return pd.DataFrame({"row": i, "array": a, "jitter_deg": ang * JITTER_FACTOR})


def qstats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan}
    q1, m, q3 = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "median": float(m), "q1": float(q1), "q3": float(q3)}


def binned(x: np.ndarray, y: np.ndarray) -> list[dict[str, Any]]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    out = []
    hi_edge = max(90.0, BIN_DEG * np.ceil(x.max() / BIN_DEG)) if len(x) else 90.0
    for lo in np.arange(0.0, hi_edge, BIN_DEG):
        m = (x >= lo) & (x < lo + BIN_DEG)
        n = int(m.sum())
        out.append({"bin_lo_deg": lo, "bin_hi_deg": lo + BIN_DEG, "n": n,
                    "median": float(np.median(y[m])) if n else np.nan,
                    "rms": float(np.sqrt(np.mean(y[m] ** 2))) if n else np.nan,
                    "valid": n >= MIN_CELL})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.systems())
    ap.add_argument("--frames-tag", default=None,
                    help="suffix of the VD29 per-frame file (default: the snapshot)")
    a = ap.parse_args()
    from biomech.config import load_config
    from biomech.io.context import DataContext

    t_start = time.perf_counter()
    cfg = load_config(str(V.STUDY)).with_snapshot(a.snapshot)
    out_dir = Path(cfg.out_dir) / "audit"
    ftag = a.frames_tag or a.snapshot
    fpath = out_dir / f"vd29_array_noise_frames_{ftag}.parquet"
    if not fpath.is_file():
        raise SystemExit(f"{fpath} missing: run vd29_array_noise.py first")
    mcf = pd.read_parquet(fpath)
    ctx = DataContext(cfg)
    summ, bins, rule, checks, takes = [], [], {}, {}, {}
    for pid in ctx.pids():
        t0 = time.perf_counter()
        take = V.TAKES[pid]
        tc = V.take_context(ctx, pid)
        rf = V.refit_take(tc)
        rep = V.reproduction(tc, rf)
        checks[f"{pid}_refit_vs_stored"] = rep
        if not all(v["ok"] for v in rep.values()):
            raise SystemExit(f"{pid}: refit does not reproduce the stored poses: {rep}")
        d0, cp, Tw, ids = detections(tc)
        d = per_detection(tc, rf, d0, cp, Tw)
        checks[f"{pid}_incidence_beyond_90_share"] = float(np.mean(d["incidence_deg"] > 90))
        checks[f"{pid}_detections_without_refit_pose"] = int(d["incidence_deg"].isna().sum())
        n_mov = int(tc["mov"].sum())
        base = {"pid": pid, "take": take}
        for arr_ in V.ARRAYS:
            for cam in ("cam0", "cam1"):
                da = d[(d.array == arr_) & (d.camera == cam)]
                for tag in ["all"] + list(tc["tags"][arr_]):
                    dt_ = da if tag == "all" else da[da.tag == tag]
                    for q, col, unit in (("incidence", "incidence_deg", "deg"),
                                         ("tag_side", "side_px", "px"),
                                         ("reproj_residual", "residual_px", "px")):
                        summ.append({**base, "array": arr_, "paper_array": PAPER[arr_],
                                     "camera": cam, "tag": str(tag), "quantity": q,
                                     "unit": unit, **qstats(dt_[col])})
                    rate = dt_["row"].nunique() / n_mov
                    summ.append({**base, "array": arr_, "paper_array": PAPER[arr_],
                                 "camera": cam, "tag": str(tag), "quantity": "detection_rate",
                                 "unit": "1", "n": n_mov, "median": float(rate),
                                 "q1": np.nan, "q3": np.nan})
            # residual binned by the detection's incidence (per camera and both cameras)
            for cam in ("cam0", "cam1", "both"):
                da = d[(d.array == arr_) & ((d.camera == cam) if cam != "both" else True)]
                for b in binned(da["incidence_deg"].to_numpy(), da["residual_px"].to_numpy()):
                    bins.append({**base, "array": arr_, "paper_array": PAPER[arr_],
                                 "camera": cam, "quantity": "reproj_residual_px",
                                 "by": "detection_incidence", **b})
        # per-frame mean incidence over both cameras
        fm = (d.dropna(subset=["incidence_deg"]).groupby(["array", "row"])["incidence_deg"]
              .mean().rename("mean_incidence_deg").reset_index())
        jit = pd.concat([jitter(tc, arr_) for arr_ in V.ARRAYS], ignore_index=True)
        jit = jit.merge(fm, on=["array", "row"], how="left")
        mc = mcf[mcf.pid == pid].rename(columns={"optical_row": "row"}).merge(
            fm, on=["array", "row"], how="left")
        for arr_ in V.ARRAYS:
            ja = jit[jit.array == arr_]
            st = qstats(ja["jitter_deg"])
            summ.append({**base, "array": arr_, "paper_array": PAPER[arr_], "camera": "both",
                         "tag": "all", "quantity": "jitter", "unit": "deg", **st,
                         "rms": float(np.sqrt(np.mean(ja["jitter_deg"] ** 2)))})
            stf = qstats(fm[fm.array == arr_]["mean_incidence_deg"])
            summ.append({**base, "array": arr_, "paper_array": PAPER[arr_], "camera": "both",
                         "tag": "all", "quantity": "frame_mean_incidence", "unit": "deg", **stf})
            ma = mc[mc.array == arr_]
            summ.append({**base, "array": arr_, "paper_array": PAPER[arr_], "camera": "both",
                         "tag": "all", "quantity": "mc_sigma_theta_3d", "unit": "deg",
                         **qstats(ma["sigma_theta_3d_deg"]),
                         "rms": float(np.sqrt(np.mean(ma["sigma_theta_3d_deg"] ** 2)))})
            for b in binned(ja["mean_incidence_deg"].to_numpy(), ja["jitter_deg"].to_numpy()):
                bins.append({**base, "array": arr_, "paper_array": PAPER[arr_], "camera": "both",
                             "quantity": "jitter_deg", "by": "frame_mean_incidence", **b})
            for b in binned(ma["mean_incidence_deg"].to_numpy(),
                            ma["sigma_theta_3d_deg"].to_numpy()):
                bins.append({**base, "array": arr_, "paper_array": PAPER[arr_], "camera": "both",
                             "quantity": "mc_sigma_theta_3d_deg", "by": "frame_mean_incidence",
                             **b})
            # per-array detection rate on any camera and refit share (context)
            summ.append({**base, "array": arr_, "paper_array": PAPER[arr_], "camera": "refit",
                         "tag": "all", "quantity": "detection_rate", "unit": "1", "n": n_mov,
                         "median": float(np.isfinite(rf["poses"][arr_][tc["mov"]])
                                         .all(axis=(1, 2)).mean()), "q1": np.nan, "q3": np.nan})

        # rule
        S = pd.DataFrame(summ)
        S = S[S.pid == pid]

        def med(arr_: str, q: str, cam: str) -> float:
            r = S[(S.array == arr_) & (S.quantity == q) & (S.camera == cam) & (S.tag == "all")]
            return float(r["median"].iloc[0])

        i_cam = {cam: {"HP": med("CRA", "incidence", cam), "MP": med("MTA", "incidence", cam),
                       "holds": med("CRA", "incidence", cam) > med("MTA", "incidence", cam)}
                 for cam in ("cam0", "cam1")}
        ii = {"HP": med("CRA", "jitter", "both"), "MP": med("MTA", "jitter", "both")}
        ii["holds"] = ii["HP"] > ii["MP"]
        fq = {arr_: np.percentile(jit[(jit.array == arr_)]["mean_incidence_deg"].dropna(),
                                  [25, 75]) for arr_ in V.ARRAYS}
        lo = float(min(fq["CRA"][0], fq["MTA"][0]))
        hi = float(max(fq["CRA"][1], fq["MTA"][1]))
        iii = {"range_deg": [lo, hi]}
        B = pd.DataFrame(bins)
        for arr_ in V.ARRAYS:
            jb = B[(B.pid == pid) & (B.array == arr_) & (B.quantity == "jitter_deg")]
            jb = jb[(jb.bin_lo_deg < hi) & (jb.bin_hi_deg > lo) & jb.valid].sort_values(
                "bin_lo_deg")
            medians = jb["median"].to_numpy()
            nondec = bool(np.all(np.diff(medians) >= 0)) if len(medians) else False
            iii[PAPER[arr_]] = {"bins": [[float(x), float(y)] for x, y in
                                         zip(jb.bin_lo_deg, jb.bin_hi_deg)],
                                "medians": medians.tolist(), "n": jb["n"].tolist(),
                                "evaluable": bool(len(medians) >= 2),
                                "does_not_decrease": nondec}
        iii["holds"] = all(iii[p]["does_not_decrease"] and iii[p]["evaluable"]
                           for p in ("HP", "MP"))
        rule[pid] = {"take": take, "i": i_cam,
                     "i_holds": all(v["holds"] for v in i_cam.values()),
                     "ii": ii, "iii": iii}
        rule[pid]["all_hold"] = bool(rule[pid]["i_holds"] and ii["holds"] and iii["holds"])
        takes[pid] = {"take": take, "n_movement_frames": n_mov, "n_detections": int(len(d)),
                      "n_jitter_frames": {PAPER[k]: int((jit.array == k).sum())
                                          for k in V.ARRAYS},
                      "runtime_s": time.perf_counter() - t0}
    ctx.close()
    verdict = all(r["all_hold"] for r in rule.values())
    rule["supported"] = bool(verdict)
    rule["verdict"] = ("supported" if verdict else
                       "not supported (paper drops the hypothesis; geometric alternative: VD29)")
    not_eval = [r["take"] for k, r in rule.items() if isinstance(r, dict) and "iii" in r
                and not all(r["iii"][p]["evaluable"] for p in ("HP", "MP"))]
    reasons = []
    if not_eval:
        reasons.append("(iii) not evaluable: one bin per array (" + ", ".join(not_eval)
                       + "); read as not met (addendum 2)")
    for k, r in rule.items():
        if isinstance(r, dict) and "i_holds" in r:
            if not r["i_holds"]:
                reasons.append(f"(i) not met in {r['take']}")
            if not r["ii"]["holds"]:
                reasons.append(f"(ii) not met in {r['take']}")
    rule["reason"] = "; ".join(reasons)
    rule["descriptive"] = ("(i) and (ii) hold in both takes" if all(
        r["i_holds"] and r["ii"]["holds"] for r in rule.values()
        if isinstance(r, dict) and "i_holds" in r) else "(i)/(ii) do not hold in every take") \
        + "; incidence is confounded with the tag's image size (see tag_side)"

    out_dir.mkdir(parents=True, exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "software", "scripts"],
                           capture_output=True, text=True, cwd=_pins.ROOT).stdout.strip()
    files = {"main": out_dir / f"vd30_tag_viewing_{a.snapshot}.parquet",
             "bins": out_dir / f"vd30_tag_viewing_bins_{a.snapshot}.parquet"}
    pd.DataFrame(summ).assign(snapshot=a.snapshot).to_parquet(files["main"])
    pd.DataFrame(bins).assign(snapshot=a.snapshot).to_parquet(files["bins"])
    meta = {"snapshot": a.snapshot, "plan": "analysis_plan.md VD30 + VD29 / VD30 amendment + addendum 2",
            "settings": {"bin_deg": BIN_DEG, "min_cell": MIN_CELL,
                         "jitter_factor": JITTER_FACTOR, "gap_tol": GAP_TOL,
                         "fit": "trimmed Kabsch of the live tracker (VD29 refit)",
                         "residual": "per detection: sqrt(mean over the 4 corners of the "
                                     "squared corner distance), px",
                         "incidence": "angle between the tag normal (refit pose) and the line "
                                      "tag centre -> camera centre",
                         "rule_iii_range": "per-frame mean incidence of the frames with a "
                                           "jitter value, [min Q1, max Q3] of the two arrays",
                         "mc_frames_file": fpath.name},
            "checks": checks, "rule": rule, "takes": takes, "git_sha": sha,
            "git_dirty_paths": dirty.splitlines(), "runtime_s": time.perf_counter() - t_start,
            "files": {k: v.name for k, v in files.items()},
            "sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in files.items()}}
    files["main"].with_suffix(".json").write_text(
        json.dumps(meta, indent=1, default=lambda o: np.asarray(o).tolist()))
    pd.set_option("display.width", 220)
    S = pd.DataFrame(summ)
    print(S[S.tag == "all"].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print(json.dumps(rule, indent=1, default=lambda o: np.asarray(o).tolist()))
    print(f"wrote {files['main']} ({meta['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
