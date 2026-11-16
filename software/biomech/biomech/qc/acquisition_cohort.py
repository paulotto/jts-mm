# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Acquisition quality over the cohort (X11, J14, an20): the per-recording tables
of :mod:`biomech.qc.acquisition` pooled, the recording days, IMU-lag outliers, the stereo
calibration cross test and the trajectory sensitivity to the extrinsic calibration
(:mod:`biomech.qc.stereo_cal`).

The dataset was recorded on six days, with the rig recalibrated between
sessions (also on the same day); ``days`` and ``calibrations`` say which recording used which
calibration where it was stored (s10 on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULTS: dict[str, Any] = {
    "lag_outlier_ms": 5.0,     # a participant's IMU lag this far above the median is flagged
    # calibration bundle (path relative to the repository) for the sensitivity to the
    # extrinsic calibration: set in the study file (stages.acquisition_cohort); empty = skipped
    "calibration_bundle": "",
    "cross_pairs": 20000,
}


def _rms_pooled(c: pd.DataFrame) -> float:
    n = c["resid_n"].astype(float)
    return float(np.sqrt((c["resid_rms_us"] ** 2 * n).sum() / n.sum()))


def per_session(cams: pd.DataFrame, lag: pd.DataFrame, rec: pd.DataFrame,
                cal: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: day, frames, drop rates, longest burst, time-base RMS per
    camera, IMU lag, calibration in use (where stored)."""
    w = cams.pivot_table(index="pid", columns="cam",
                         values=["n_frames", "drop_rate_pct", "burst_max", "resid_rms_us"])
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    out = w.reset_index()
    keep = [c for c in ("pid", "lag_ms", "r_peak", "task_lag_range_ms") if c in lag]
    out = out.merge(lag[keep], on="pid", how="left")
    out = out.merge(rec[["pid", "date", "day_index", "recording_order", "recording_utc"]],
                    on="pid", how="left")
    ccols = [c for c in ("pid", "stored", "epipolar_median_px", "epipolar_p95_px",
                         "reprojection_median_px", "baseline_mm", "cal_key") if c in cal]
    out = out.merge(cal[ccols], on="pid", how="left")
    return out.sort_values("recording_utc").reset_index(drop=True)


def summary(cams: pd.DataFrame, cnt: pd.DataFrame, lag: pd.DataFrame, rec: pd.DataFrame,
            p: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The pooled numbers (one row) and the per-participant counter check."""
    d = pd.to_datetime(rec["date"])
    s: dict[str, Any] = {"n_recordings": int(cams["pid"].nunique()),
                         "n_days": int(rec["date"].nunique()),
                         "span_days": int((d.max() - d.min()).days) + 1}
    s["frames_total"] = int(cams["n_frames"].sum())
    s["n_camera_streams"] = int(len(cams))
    reg = cams["regular_triggers"].sum()
    s["dropped_total"] = int(cams["dropped"].sum())
    s["drop_rate_pooled_pct"] = float(100.0 * s["dropped_total"] / reg) if reg else np.nan
    for c in sorted(cams["cam"].unique()):
        x = cams[cams["cam"] == c]["drop_rate_pct"]
        s[f"drop_rate_min_{c}_pct"], s[f"drop_rate_max_{c}_pct"] = float(x.min()), float(x.max())
        r = cams[cams["cam"] == c]["resid_rms_us"]
        s[f"resid_rms_min_{c}_us"], s[f"resid_rms_max_{c}_us"] = float(r.min()), float(r.max())
    s["drop_rate_min_pct"] = float(cams["drop_rate_pct"].min())
    s["drop_rate_max_pct"] = float(cams["drop_rate_pct"].max())
    s["resid_rms_min_us"] = float(cams["resid_rms_us"].min())
    s["resid_rms_max_us"] = float(cams["resid_rms_us"].max())
    s["resid_rms_median_us"] = float(cams["resid_rms_us"].median())
    s["burst_max"] = int(cams["burst_max"].max())
    nb = cams["n_bursts"].sum()
    s["burst_single_share_pct"] = float(100.0 * (cams["burst_single_frac"] * cams["n_bursts"])
                                        .sum() / nb) if nb else np.nan
    s["startup_triggers_median"] = float(cams["startup_triggers"].median())
    s["resid_rms_pooled_us"] = _rms_pooled(cams)
    s["resid_p99_max_us"] = float(cams["resid_p99_us"].max())
    s["resid_p99_median_us"] = float(cams["resid_p99_us"].median())
    s["resid_max_us"] = float(cams["resid_max_us"].max())
    s["resid_max_pid"] = str(cams.loc[cams["resid_max_us"].idxmax(), "pid"])
    s["resid_n_above_spike"] = int(cams["resid_n_above_spike"].sum())
    s["resid_consecutive_spikes"] = int(cams["resid_consecutive_spikes"].sum())
    s["resid_spike_share_pct"] = float(100.0 * s["resid_n_above_spike"] / cams["resid_n"].sum())
    s["one_period_us"] = 25000.0
    if "resid_tasks_rms_us" in cams:
        n = cams["resid_tasks_n"].astype(float)
        s["resid_tasks_rms_pooled_us"] = float(np.sqrt((cams["resid_tasks_rms_us"] ** 2 * n)
                                                       .sum() / n.sum()))
        s["resid_tasks_max_us"] = float(cams["resid_tasks_max_us"].max())
    # every drop resolved by the cadence step: cadence_advance = drops + start-up ids after
    # the camera's first frame; every other association path unused
    chk = []
    ca = cnt[cnt["counter"] == "cadence_advance"]
    for _, r in cams.iterrows():
        v = ca[(ca["pid"] == r["pid"]) & (ca["cam"] == r["cam"])]["value"]
        exp = r["dropped"] + r["startup_triggers"] - r["pre_first_triggers"]
        chk.append({"pid": r["pid"], "cam": r["cam"], "cadence_advance":
                    float(v.iloc[0]) if len(v) else np.nan, "expected": float(exp),
                    "match": bool(len(v) and float(v.iloc[0]) == float(exp))})
    chk = pd.DataFrame(chk)
    other = cnt[~cnt["counter"].isin(["cadence_advance", "camera_drop_recovered"])]
    s["all_drops_by_cadence"] = bool(chk["match"].all())
    s["other_path_count_total"] = float(other["value"].fillna(0).sum())
    lg = lag.dropna(subset=["lag_ms"])
    if len(lg):
        med = float(lg["lag_ms"].median())
        s.update({"imu_lag_n": int(len(lg)), "imu_lag_median_ms": med,
                  "imu_lag_q1_ms": float(lg["lag_ms"].quantile(.25)),
                  "imu_lag_q3_ms": float(lg["lag_ms"].quantile(.75)),
                  "imu_lag_min_ms": float(lg["lag_ms"].min()),
                  "imu_lag_max_ms": float(lg["lag_ms"].max()),
                  "imu_lag_r_min": float(lg["r_peak"].min()),
                  "imu_lag_r_max": float(lg["r_peak"].max()),
                  "imu_lag_task_range_max_ms": float(lg["task_lag_range_ms"].max())})
        out = lg[lg["lag_ms"] > med + float(p["lag_outlier_ms"])]
        rest = lg[lg["lag_ms"] <= med + float(p["lag_outlier_ms"])]
        s.update({"imu_lag_outliers_n": int(len(out)),
                  "imu_lag_outliers": ", ".join(sorted(out["pid"])),
                  "imu_lag_outlier_min_ms": float(out["lag_ms"].min()) if len(out) else np.nan,
                  "imu_lag_outlier_max_ms": float(out["lag_ms"].max()) if len(out) else np.nan,
                  "imu_lag_regular_median_ms": float(rest["lag_ms"].median()),
                  "imu_lag_regular_min_ms": float(rest["lag_ms"].min()),
                  "imu_lag_regular_max_ms": float(rest["lag_ms"].max()),
                  "imu_lag_outlier_excess_ms": float(out["lag_ms"].median()
                                                     - rest["lag_ms"].median()) if len(out)
                  else np.nan})
        # are the outliers a recording-day effect? (the other recordings of their days)
        days_out = set(rec[rec["pid"].isin(out["pid"])]["day_index"])
        same_day = lg.merge(rec[["pid", "day_index"]], on="pid")
        same_day = same_day[same_day["day_index"].isin(days_out)
                            & ~same_day["pid"].isin(out["pid"])]
        s["imu_lag_outlier_n_days"] = int(len(days_out))
        s["imu_lag_outlier_day_index"] = ", ".join(str(int(x)) for x in sorted(days_out))
        s["imu_lag_sameday_regular_n"] = int(len(same_day))
        s["imu_lag_sameday_regular"] = ", ".join(sorted(same_day["pid"]))
    return pd.DataFrame([s]), chk


def days(rec: pd.DataFrame, cal: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    d = rec.merge(cal[["pid", "stored"]], on="pid", how="left")
    rows = []
    for (day, idx), g in d.sort_values("recording_utc").groupby(["date", "day_index"]):
        rows.append({"date": day, "day_index": int(idx), "n": int(len(g)),
                     "participants": ", ".join(g["pid"]),
                     "n_calibration_stored": int(g["stored"].fillna(False).sum()),
                     "calibrations": ", ".join(sorted({labels[p] for p in g["pid"]
                                                       if p in labels}))})
    return pd.DataFrame(rows)


def calibrations(cal: pd.DataFrame, rec: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Distinct stored calibrations, labelled A, B, ... in order of first use."""
    c = cal[cal["stored"].fillna(False)].merge(rec[["pid", "date", "day_index",
                                                     "recording_utc"]], on="pid")
    c = c.sort_values("recording_utc")
    labels: dict[str, str] = {}
    by_key: dict[str, str] = {}
    for _, r in c.iterrows():
        if r["cal_key"] not in by_key:
            by_key[r["cal_key"]] = chr(ord("A") + len(by_key))
        labels[r["pid"]] = by_key[r["cal_key"]]
    rows = []
    for key, lab in by_key.items():
        g = c[c["cal_key"] == key]
        rows.append({"label": lab, "cal_key": key,
                     "day_index": ", ".join(str(int(x)) for x in sorted(set(g["day_index"]))),
                     "participants": ", ".join(g["pid"]),
                     "baseline_mm": float(g["baseline_mm"].iloc[0]),
                     "convergence_deg": float(g["convergence_deg"].iloc[0]),
                     "f0_px": float(g["f0_px"].iloc[0]), "f1_px": float(g["f1_px"].iloc[0]),
                     "extrinsics_rms0_px": float(g["extrinsics_rms0_px"].iloc[0]),
                     "extrinsics_rms1_px": float(g["extrinsics_rms1_px"].iloc[0]),
                     "epipolar_median_min_px": float(g["epipolar_median_px"].min()),
                     "epipolar_median_max_px": float(g["epipolar_median_px"].max())})
    return pd.DataFrame(rows), labels


def analyse(cams: pd.DataFrame, cnt: pd.DataFrame, lag: pd.DataFrame, lagw: pd.DataFrame,
            rec: pd.DataFrame, cal: pd.DataFrame, paths: dict[str, Path],
            p: dict[str, Any] | None = None, seed: int = 0) -> dict[str, pd.DataFrame]:
    from biomech.config import REPO_ROOT
    from biomech.io.session import SessionData
    from biomech.qc import stereo_cal as sc

    p = {**DEFAULTS, **(p or {})}
    sp = {**sc.DEFAULTS, **(p.get("stereo") or {})}
    rec = rec.copy()                            # day 1, 2, ... and the order of recording
    order = sorted(rec["date"].dropna().unique())
    rec["day_index"] = rec["date"].map({d: i + 1 for i, d in enumerate(order)})
    rec["recording_order"] = rec["recording_utc"].rank(method="first").astype(int)
    summ, chk = summary(cams, cnt, lag, rec, p)
    cal_tab, labels = calibrations(cal, rec)
    out = {"summary": summ, "counter_check": chk, "per_session": per_session(cams, lag, rec, cal),
           "days": days(rec, cal, labels), "calibrations": cal_tab}
    stored = [pid for pid in cal_tab["participants"].str.split(", ").explode()] \
        if len(cal_tab) else []
    if not stored:
        return out
    cals = {pid: sc.session_calibration(Path(paths[pid]) / "session.h5") for pid in stored}
    pairs = {pid: sc.corner_pairs(Path(paths[pid]) / "session.h5") for pid in stored}
    rows = []
    for rpid in stored:
        sub = sc.thin(pairs[rpid], int(p["cross_pairs"]), seed)
        for lab in cal_tab["label"]:
            cpid = next(k for k, v in labels.items() if v == lab)
            rows.append({"pid": rpid, "recording_label": labels[rpid], "calibration": lab,
                         "own": labels[rpid] == lab,
                         "epipolar_median_px": sc.epipolar_median(cals[cpid], sub)})
    cross = pd.DataFrame(rows)
    out["cross_test"] = cross
    own, foreign = cross[cross["own"]], cross[~cross["own"]]
    summ.loc[0, "cross_own_max_px"] = float(own["epipolar_median_px"].max())
    summ.loc[0, "cross_foreign_min_px"] = float(foreign["epipolar_median_px"].min()) \
        if len(foreign) else np.nan
    summ.loc[0, "cross_foreign_max_px"] = float(foreign["epipolar_median_px"].max()) \
        if len(foreign) else np.nan
    summ.loc[0, "n_calibrations"] = int(len(cal_tab))
    summ.loc[0, "n_calibration_stored"] = int(len(stored))
    # sensitivity to the extrinsic calibration
    bpath = Path(REPO_ROOT) / str(p["calibration_bundle"])
    if not p["calibration_bundle"] or not bpath.is_file():
        out["sensitivity_notes"] = pd.DataFrame(
            [{"note": f"{bpath if p['calibration_bundle'] else 'no calibration bundle set'}"
                      f"{' not found' if p['calibration_bundle'] else ''}: skipped"}])
        return out
    snap = sc.bundle_snapshots(bpath)
    sig = sc.sigma_model(snap)
    model, ranges = sc.marker_model()
    D, R = [], []
    for k, pid in enumerate(stored):
        sd = SessionData(paths[pid])
        try:
            import h5py

            with h5py.File(Path(paths[pid]) / "session.h5", "r") as f:
                trow = np.asarray(f["optical/t_master_us"][:])
            d, r = sc.sensitivity(cals[pid], pairs[pid], trow, sd.phases, sig, model, ranges,
                                  sp, seed=seed + k)
        finally:
            sd.close()
        d.insert(0, "pid", pid)
        r.insert(0, "pid", pid)
        D.append(d)
        R.append(r)
    D, R = pd.concat(D, ignore_index=True), pd.concat(R, ignore_index=True)
    out["sensitivity_draws"], out["sensitivity_rom"] = D, R
    srows = []
    for scen, g in D.groupby("scenario"):
        rg = R[R["scenario"] == scen]
        row = {"scenario": scen, "n_snapshots": snap["n"],
               "snapshot_rms_rot_deg": snap["rms_rot_deg"],
               "snapshot_rms_trans_mm": snap["rms_trans_mm"],
               "sigma_rot_deg": sig[scen]["sigma_rot_deg"],
               "sigma_trans_mm": sig[scen]["sigma_trans_mm"],
               "n_recordings": int(g["pid"].nunique()), "n_draws": int(len(g)),
               "traj_rms_median_mm": float(g["traj_rms_mm"].median()),
               "traj_rms_p95_mm": float(g["traj_rms_mm"].quantile(.95)),
               "rot_rms_median_deg": float(g["rot_rms_deg"].median()),
               "rot_rms_p95_deg": float(g["rot_rms_deg"].quantile(.95)),
               "d_rom_abs_median_mm": float(rg["d_rom_mm"].abs().median()),
               "d_rom_abs_p95_mm": float(rg["d_rom_mm"].abs().quantile(.95)),
               "d_rom_rel_p95_pct": float((100.0 * rg["d_rom_mm"].abs() / rg["rom_mm"])
                                          .quantile(.95))}
        for task, tg in rg.groupby("task"):
            row[f"d_rom_abs_median_{task}_mm"] = float(tg["d_rom_mm"].abs().median())
            row[f"d_rom_abs_p95_{task}_mm"] = float(tg["d_rom_mm"].abs().quantile(.95))
        srows.append(row)
    out["sensitivity_summary"] = pd.DataFrame(srows)
    return out
