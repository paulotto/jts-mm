# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cohort statistics for kinematics: S1 reference values and C1 (an07 §5, §8, §14)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from scipy import stats as sps

from biomech.stats.core import describe, nan_quiet, hedges_g, icc_sem_mdc
from biomech.stats.families import wilcoxon_greater


# Subject-level metric -> (label, unit, extractor over the kinematics cycles table)


def _cyc(task: str, col: str, agg: str = "median", side: str | None = None,
         sign: float = 1.0) -> Callable[[pd.DataFrame], float]:
    def f(c: pd.DataFrame) -> float:
        g = c[(c.task == task) & c.ok]
        if side is not None:
            g = g[g.side == side]
        v = sign * g[col].to_numpy(float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return np.nan
        return float({"median": np.median, "max": np.max}[agg](v))
    return f


METRICS: dict[str, tuple[str, str, Callable[[pd.DataFrame], float]]] = {
    "opening_selfpaced_mm": ("Self-paced opening", "mm", _cyc("open_close", "opening_mm")),
    "opening_angle_deg": ("Opening angle", "deg", _cyc("open_close", "theta_y_deg")),
    "protrusion_mm": ("Protrusion", "mm", _cyc("protrusion_retrusion", "protrusion_mm", "max")),
    "retrusion_mm": ("Retrusion", "mm", _cyc("protrusion_retrusion", "retrusion_mm", "max")),
    "laterotrusion_left_mm": ("Laterotrusion left", "mm",
                              _cyc("left_right", "lateral_mm", "max", "left")),
    "laterotrusion_right_mm": ("Laterotrusion right", "mm",
                               _cyc("left_right", "lateral_mm", "max", "right", -1.0)),
    "opening_deviation_mm": ("Opening deviation", "mm", _cyc("open_close", "deviation_mm")),
    "condyle_left_disp_mm": ("Condylar path left (opening)", "mm",
                             _cyc("open_close", "condyle_left_disp_mm")),
    "condyle_right_disp_mm": ("Condylar path right (opening)", "mm",
                              _cyc("open_close", "condyle_right_disp_mm")),
    "peak_open_vel_dps": ("Peak opening velocity", "deg/s",
                          _cyc("open_close", "peak_open_vel_dps")),
    "peak_close_vel_dps": ("Peak closing velocity", "deg/s",
                           _cyc("open_close", "peak_close_vel_dps")),
    "peak_open_vel_max_dps": ("Peak opening velocity (max)", "deg/s",
                              _cyc("open_close", "peak_open_vel_dps", "max")),
    "peak_close_vel_max_dps": ("Peak closing velocity (max)", "deg/s",
                               _cyc("open_close", "peak_close_vel_dps", "max")),
    "chewing_amplitude_mm": ("Chewing amplitude", "mm", _cyc("chewing", "opening_mm")),
    "chewing_angle_deg": ("Chewing opening angle", "deg", _cyc("chewing", "theta_y_deg")),
    "chewing_peak_vel_dps": ("Chewing peak velocity", "deg/s",
                             _cyc("chewing", "peak_open_vel_dps", "max")),
}

RELIABILITY = [("open_close", "opening_mm"), ("open_close", "theta_y_deg"),
               ("open_close", "peak_open_vel_dps"), ("chewing", "opening_mm"),
               ("protrusion_retrusion", "protrusion_mm"), ("left_right", "lateral_mm")]


@nan_quiet
def subject_table(cycles: pd.DataFrame, anatomy: pd.DataFrame, demo: pd.DataFrame
                  ) -> pd.DataFrame:
    rows = []
    for pid, c in cycles.groupby("pid"):
        r: dict[str, Any] = {"pid": pid}
        for k, (_, _, f) in METRICS.items():
            r[k] = f(c)
        d = c[(c.task == "chewing") & c.ok]["duration_s"]
        r["chewing_freq_hz"] = float(1 / d.median()) if len(d) else np.nan
        oc = c[(c.task == "open_close") & c.ok]["opening_mm"]
        r["opening_cv"] = float(oc.std(ddof=1) / oc.mean()) if len(oc) > 1 else np.nan
        rows.append(r)
    t = pd.DataFrame(rows).merge(anatomy, on="pid", how="left").merge(demo, on="pid", how="left")
    lat = t[["laterotrusion_left_mm", "laterotrusion_right_mm"]]
    t["lateral_symmetry_pct"] = 100 * (lat.iloc[:, 0] - lat.iloc[:, 1]) / lat.sum(axis=1)
    t["opening_per_intermolar"] = t["opening_selfpaced_mm"] / t["upper_intermolar_mm"]
    return t


ENVELOPE_TASKS = ("open_close", "chewing", "free_motion")


@nan_quiet
def dynamics_envelope(arrays: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """X7 (post hoc description): per participant the largest opening and closing
    angular velocity of the free-motion take and the largest |angular acceleration| over
    open-close, chewing and free motion, from the stored pose chain (``<task>__theta_vel``,
    ``__theta_acc``: 6 Hz low-pass, Savitzky-Golay derivatives; opening positive). The 99th
    percentile of |acceleration| is kept as a robustness column (a single sample decides the
    maximum). ``arrays``: pid -> the participant's kinematics arrays."""
    rows = []
    for pid, a in arrays.items():
        r: dict[str, Any] = {"pid": pid}
        v = np.asarray(a.get("free_motion__theta_vel", []), float)
        v = v[np.isfinite(v)]
        r["free_peak_open_vel_dps"] = float(v.max()) if v.size and v.max() > 0 else np.nan
        r["free_peak_close_vel_dps"] = float(-v.min()) if v.size and v.min() < 0 else np.nan
        best, task, p99 = np.nan, "", []
        for t in ENVELOPE_TASKS:
            acc = np.abs(np.asarray(a.get(f"{t}__theta_acc", []), float))
            acc = acc[np.isfinite(acc)]
            if not acc.size:
                continue
            p99.append(float(np.percentile(acc, 99)))
            if not np.isfinite(best) or acc.max() > best:
                best, task = float(acc.max()), t
        r["peak_acc_dps2"] = best
        r["peak_acc_task"] = task
        r["peak_acc_p99_dps2"] = max(p99) if p99 else np.nan
        rows.append(r)
    return pd.DataFrame(rows, columns=["pid", "free_peak_open_vel_dps", "free_peak_close_vel_dps",
                                       "peak_acc_dps2", "peak_acc_task", "peak_acc_p99_dps2"])


WORKSPACE_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion")
POSTERIOR_Q = 99.0          # the posterior extent: 99th percentile of the samples, not one sample


@nan_quiet
def posterior_excursion(arrays: dict[str, dict[str, np.ndarray]]) -> pd.DataFrame:
    """X7 (post hoc description, author decision 2026-11-02): per participant how far
    the incisal point moves behind its rest position (median of the rest phase) in any of the
    five movement tasks, along the maxillary x axis (anterior positive): the
    :data:`POSTERIOR_Q` percentile of the posterior displacement over all task samples
    (``posterior_excursion_mm``, robust to a single tracking glitch) and its maximum
    (``posterior_excursion_max_mm``). The opening rotation carries the incisor backward, which
    no other workspace row bounds."""
    rows = []
    for pid, a in arrays.items():
        R = np.asarray(a.get("rest__incisor", np.empty((0, 3))), float)
        R = R[np.isfinite(R).all(axis=1)] if R.ndim == 2 else np.empty((0, 3))
        parts = []
        for t in WORKSPACE_TASKS:
            P = np.asarray(a.get(f"{t}__incisor", np.empty((0, 3))), float)
            if P.ndim == 2 and P.shape[1] == 3:
                parts.append(P[np.isfinite(P).all(axis=1)])
        P = np.vstack(parts) if parts else np.empty((0, 3))
        if not len(R) or not len(P):
            rows.append({"pid": pid, "posterior_excursion_mm": np.nan,
                         "posterior_excursion_max_mm": np.nan})
            continue
        back = np.clip(np.median(R[:, 0]) - P[:, 0], 0.0, None)    # mm behind rest
        rows.append({"pid": pid,
                     "posterior_excursion_mm": float(np.percentile(back, POSTERIOR_Q)),
                     "posterior_excursion_max_mm": float(back.max())})
    return pd.DataFrame(rows, columns=["pid", "posterior_excursion_mm",
                                       "posterior_excursion_max_mm"])


@nan_quiet
def reference_values(subj: pd.DataFrame, metrics: list[str], seed: int) -> pd.DataFrame:
    rows = []
    for m in metrics:
        if m not in subj:
            continue
        base = {"metric": m, "label": METRICS.get(m, (m,))[0]}
        rows.append({**base, "group": "all", **describe(subj[m].to_numpy(), seed=seed)})
        for sex in ("f", "m"):
            rows.append({**base, "group": sex, **describe(subj.loc[subj.sex == sex, m].to_numpy(),
                                                         seed=seed)})
        g = hedges_g(subj.loc[subj.sex == "f", m].to_numpy(), subj.loc[subj.sex == "m", m]
                     .to_numpy(), seed=seed)
        v = subj[[m, "age"]].dropna()
        rho = sps.spearmanr(v[m], v["age"]) if len(v) > 3 else None
        rows.append({**base, "group": "f_minus_m", "g": g["g"], "g_lo": g["g_lo"],
                     "g_hi": g["g_hi"],
                     "spearman_age": float(rho.statistic) if rho is not None else np.nan,
                     "spearman_age_p": float(rho.pvalue) if rho is not None else np.nan})
    return pd.DataFrame(rows)


@nan_quiet
def influence(subj: pd.DataFrame, metrics: list[str], pids: list[str]) -> pd.DataFrame:
    """Change of the cohort mean when one participant is left out (older participants)."""
    rows = []
    for m in metrics:
        if m not in subj:
            continue
        full = subj[m].mean()
        for pid in pids:
            rest = subj.loc[subj.pid != pid, m].mean()
            rows.append({"metric": m, "left_out": pid, "mean_all": full, "mean_without": rest,
                         "change_pct": 100 * (rest - full) / full if full else np.nan})
    return pd.DataFrame(rows)


@nan_quiet
def reliability(cycles: pd.DataFrame) -> pd.DataFrame:
    """ICC/SEM/MDC across cycles. Laterotrusion is split by side: its cycle numbers
    restart per side, and pooling would treat left and right cycles as raters."""
    rows = []
    ok = cycles[cycles.ok].copy()
    ok["rep"] = ok["cycle"]
    if "side" not in ok:
        ok["side"] = ""
    for task, col in RELIABILITY:
        if col not in ok:
            continue
        g = ok[ok.task == task]
        sides = ["left", "right"] if task == "left_right" else [""]
        for side in sides:
            gs = g[g.side.fillna("") == side] if task == "left_right" else g
            if task == "left_right":
                gs = gs.assign(**{col: gs[col].abs()})
            out = icc_sem_mdc(gs, col, rep="rep", k=8)
            rows.append({"task": task, "side": side, "metric": col, **out})
    return pd.DataFrame(rows)


@nan_quiet

def hinge_noise_counts(noise: pd.DataFrame, tag: str = "invivo") -> tuple[int, int]:
    """(above, tested) for the noise level ``tag`` of ``uncertainty.hinge_noise``: a
    participant is tested only when its null exists (finite ``ratio_<tag>``; a row written
    because there were too few cycles, or a null without draws, is not), so "above noise in
    k/n" never counts an untested participant as above, nor in the denominator (E30)."""
    if noise is None or noise.empty or f"above_{tag}" not in noise:
        return 0, 0
    ratio = noise.get(f"ratio_{tag}")
    tested = (pd.to_numeric(ratio, errors="coerce").notna() if ratio is not None
              else noise[f"above_{tag}"].notna())
    above = noise[f"above_{tag}"].map(lambda v: v is True or v == 1.0 or v is np.True_)
    return int((above & tested).sum()), int(tested.sum())


NOISE_TAGS = ("bench", "invivo", "invivo_slow", "invivo_motion", "invivo_teeth",
              "bench_tmj", "invivo_tmj")


@nan_quiet
def hinge_noise_tests(noise: pd.DataFrame, plausible: set[str] | None, seed: int
                      ) -> pd.DataFrame:
    """C1 against each noise null, per null ``tag`` and participant ``group`` (D24, D25).

    Per null (``noise``; the tags of ``uncertainty.hinge_noise``: bench, the primary invivo, and the D24/D25
    variants invivo_slow, invivo_motion, invivo_teeth, and the D33 condyle-hinge nulls
    bench_tmj, invivo_tmj, where computed) and group (``all``;
    ``plausible_condyles``: D18-plausible condyle points, D25): participants tested and above
    the null's P99 (the primary count, :func:`hinge_noise_counts`), the median ratio
    observed / P99 with its BCa CI, the cohort test -- one-sided exact Wilcoxon signed-rank
    of log(observed / P99) > 0 over the tested participants -- and the median and largest
    per-participant Monte-Carlo p ((k + 1)/(N + 1))."""
    rows: list[dict[str, Any]] = []
    if noise is None or noise.empty:
        return pd.DataFrame(rows)
    groups = [("all", noise.pid.notna())]
    if plausible is not None:
        groups.append(("plausible_condyles", noise.pid.isin(plausible)))
    for tag in NOISE_TAGS:
        if f"ratio_{tag}" not in noise:
            continue
        for group, sel in groups:
            g = noise[sel]
            above, tested = hinge_noise_counts(g, tag)
            ratio = pd.to_numeric(g[f"ratio_{tag}"], errors="coerce").to_numpy(float)
            ok = np.isfinite(ratio) & (ratio > 0)
            w, pw, exact = wilcoxon_greater(np.log(ratio[ok]))
            d = describe(ratio, seed=seed, stat="median")
            pmc = pd.to_numeric(g.get(f"p_mc_{tag}", pd.Series(dtype=float)),
                                errors="coerce").to_numpy(float)
            pmc = pmc[np.isfinite(pmc)]
            rows.append({
                "noise": tag, "group": group, "n_tested": tested, "above_n": above,
                "noise_ratio_median": d["median"], "noise_ratio_ci_lo": d["ci_lo"],
                "noise_ratio_ci_hi": d["ci_hi"],
                "log_noise_ratio_median": float(np.median(np.log(ratio[ok]))) if ok.any()
                else np.nan,
                "wilcoxon_log_ratio_w": w, "wilcoxon_log_ratio_p": pw,
                "wilcoxon_log_ratio_exact": exact,
                "p_mc_median": float(np.median(pmc)) if pmc.size else np.nan,
                "p_mc_max": float(pmc.max()) if pmc.size else np.nan,
            })
    return pd.DataFrame(rows)


def hinge_tables(hinge: pd.DataFrame, noise: pd.DataFrame, fha: pd.DataFrame, seed: int,
                 plausible: set[str] | None = None) -> dict[str, pd.DataFrame]:
    rows = []
    for model, g in hinge.groupby("model"):
        for col in ("rms_incisor_mm", "max_incisor_mm", "rms_condyles_mm", "translation_range_mm"):
            if g[col].notna().any():
                rows.append({"model": model, "metric": col,
                             **describe(g[col].to_numpy(), seed=seed, stat="median")})
    out = {"hinge_summary": pd.DataFrame(rows)}
    if not noise.empty:
        nb, tb = hinge_noise_counts(noise, "bench")
        ni, ti = hinge_noise_counts(noise, "invivo")
        out["hinge_noise_summary"] = pd.DataFrame([{
            "n": int(len(noise)), "n_tested_bench": tb, "n_tested_invivo": ti,
            "above_bench_n": nb, "above_invivo_n": ni,
            **{f"ratio_invivo_{k}": v for k, v in describe(noise["ratio_invivo"].to_numpy(),
                                                            seed=seed, stat="median").items()},
        }])
        out["hinge_noise_tests"] = hinge_noise_tests(noise, plausible, seed)
    if not fha.empty:
        per = (fha.groupby(["pid", "cycle"])
               .agg(pierce_x_range=("pierce_x_mm", lambda s: s.max() - s.min()),
                    pierce_z_range=("pierce_z_mm", lambda s: s.max() - s.min()),
                    dist_tmj=("dist_tmj_mid_mm", "mean"))
               .groupby("pid").median().reset_index())
        out["fha_subject"] = per
        out["fha_summary"] = pd.DataFrame([
            {"metric": c, **describe(per[c].to_numpy(), seed=seed, stat="median")}
            for c in ("pierce_x_range", "pierce_z_range", "dist_tmj")])
    return out


@nan_quiet
def coupling_band(curves: dict[str, np.ndarray], grid: np.ndarray, min_cycles: int = 3
                  ) -> dict[str, np.ndarray]:
    """Per-participant median curves -> cohort median and IQR on the angle grid."""
    out: dict[str, np.ndarray] = {"grid_deg": grid}
    for name, per_pid in curves.items():
        meds = []
        for arr in per_pid:
            arr = np.asarray(arr if arr is not None else np.empty((0, 0)), float)
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != grid.size:
                meds.append(np.full(grid.size, np.nan))     # participant without curves
                continue
            ok = np.isfinite(arr).sum(axis=0) >= min_cycles
            meds.append(np.where(ok, np.nanmedian(arr, axis=0), np.nan))
        M = np.array(meds)
        out[f"{name}__subjects"] = M
        enough = np.isfinite(M).sum(axis=0) >= max(3, len(M) // 2)
        for q, tag in ((50, "median"), (25, "q1"), (75, "q3")):
            out[f"{name}__{tag}"] = np.where(enough, np.nanpercentile(M, q, axis=0), np.nan)
    return out


COUPLING_SLOPE_DEG = (5.0, 15.0)     # X9: angle range of the per-participant slope


@nan_quiet
def coupling_summary(curves: dict[str, list], grid: np.ndarray, pids: list[str],
                     plausible: set[str] | None = None, min_cycles: int = 3) -> pd.DataFrame:
    """X9 (post hoc description): per participant the Theil-Sen slope of the
    anterior condyle-midpoint translation on the opening angle over
    :data:`COUPLING_SLOPE_DEG`, and the RMS difference of the participant's median curve from
    the leave-one-subject-out cohort median curve over the angles both define (how much a
    generic coupling model misses). ``curves``: ``tmj_mid_ant_mm`` -> per participant the
    cycle curves on ``grid`` (as for :func:`coupling_band`)."""
    from scipy.stats import theilslopes

    per = curves.get("tmj_mid_ant_mm") or []
    meds = []
    for arr in per:
        arr = np.asarray(arr if arr is not None else np.empty((0, 0)), float)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] != grid.size:
            meds.append(np.full(grid.size, np.nan))
            continue
        ok = np.isfinite(arr).sum(axis=0) >= min_cycles
        meds.append(np.where(ok, np.nanmedian(arr, axis=0), np.nan))
    M = np.array(meds) if meds else np.empty((0, grid.size))
    lo, hi = COUPLING_SLOPE_DEG
    win = (grid >= lo) & (grid <= hi)
    rows = []
    for i, pid in enumerate(pids[:len(M)]):
        m = M[i]
        sel = win & np.isfinite(m)
        slope = float(theilslopes(m[sel], grid[sel])[0]) if sel.sum() >= 3 else np.nan
        others = np.delete(M, i, axis=0)
        enough = np.isfinite(others).sum(axis=0) >= max(3, len(others) // 2)
        gen = np.where(enough, np.nanmedian(others, axis=0), np.nan) if len(others) else \
            np.full(grid.size, np.nan)
        both = np.isfinite(m) & np.isfinite(gen)
        rms = float(np.sqrt(np.mean((m[both] - gen[both]) ** 2))) if both.sum() >= 3 else np.nan
        top = float(grid[np.isfinite(m)].max()) if np.isfinite(m).any() else np.nan
        rows.append({"pid": pid, "slope_mm_per_deg": slope, "generic_rms_mm": rms,
                     "angle_max_deg": top, "n_angles": int(both.sum()),
                     "condyles_plausible": bool(plausible is not None and pid in plausible)})
    return pd.DataFrame(rows, columns=["pid", "slope_mm_per_deg", "generic_rms_mm",
                                       "angle_max_deg", "n_angles", "condyles_plausible"])


# Jaw geometry (kinematics:geometry, D21); all but the arch depths and the hinge radius
# rest on the palpated condyles.
GEOMETRY_METRICS = ("bonwill_side_mm", "bonwill_ratio", "balkwill_deg", "lower_arch_depth_mm",
                    "upper_arch_depth_mm", "condyle_height_mm", "hinge_radius_mm",
                    "hinge_axis_dx_mm", "hinge_axis_dz_mm", "hinge_axis_offset_mm",
                    "hinge_axis_angle_deg")
CONDYLE_METRICS = ("condyle_left_disp_mm", "condyle_right_disp_mm", "intercondylar_mm",
                   "condyle_incisor_left_mm", "condyle_incisor_right_mm",
                   *(m for m in GEOMETRY_METRICS
                     if m not in ("lower_arch_depth_mm", "upper_arch_depth_mm",
                                  "hinge_radius_mm")))


# D29: fixed neighbours of the D18 depth threshold (signals.kinematics.min_condyle_posterior_mm,
# 40 mm), never tuned; evaluated here from the stored condyle depth, no re-run needed.
CONDYLE_DEPTH_NEIGHBOURS_MM = (30.0, 50.0)


def condyle_depth_mm(subj: pd.DataFrame) -> pd.Series:
    """D18 side-mean depth of the two condyle points behind the upper incisor (maxillary
    frame, x anterior): −(x_left + x_right)/2 from ``kinematics:anatomy`` (``frame_tmj_*_x``),
    as ``pose_chain.condyle_depth_check`` computes it."""
    if not {"frame_tmj_left_x", "frame_tmj_right_x"} <= set(subj.columns):
        return pd.Series(np.nan, index=subj.index)
    return -0.5 * (pd.to_numeric(subj.frame_tmj_left_x, errors="coerce")
                   + pd.to_numeric(subj.frame_tmj_right_x, errors="coerce"))


def plausible_at_depth(subj: pd.DataFrame, min_depth_mm: float) -> pd.Series:
    """Condyle plausibility with the D18 depth threshold set to ``min_depth_mm`` (D29): the
    other checks (3-D distance, level) must pass as stored -- plausible, or implausible
    only for being too shallow -- and the side-mean depth must be ``>= min_depth_mm`` (the
    code's comparison). Unknown (no anatomy) is not plausible. At the configured 40 mm it
    reproduces the stored ``condyles_plausible``."""
    cp = pd.to_numeric(subj.condyles_plausible, errors="coerce")
    reason = subj.get("condyles_reason", pd.Series("", index=subj.index)).fillna("")
    others_ok = cp.eq(1) | reason.eq("too_shallow")
    return others_ok & condyle_depth_mm(subj).ge(min_depth_mm)


@nan_quiet
def condyle_sensitivity(subj: pd.DataFrame, hinge: pd.DataFrame, seed: int,
                        fha_subject: pd.DataFrame | None = None) -> pd.DataFrame:
    """Condyle-based outcomes for all participants vs. only those whose skin-palpated
    condyles pass the plausibility check (analysis_plan A4, D25; flag, don't exclude).

    Rows (``metric``): the condyle metrics, the incisal residual of every hinge model
    (``hinge_rms_incisor_<model>``, the D25 teeth-only fit included), the translation three
    ways (D25): ``translation_max_mm`` (the larger side's condyle-point excursion, the
    requirement's model-free value), ``translation_mean_mm`` (the mean of both sides: the
    side difference from rotation about an off-centre axis cancels to first order) and the
    model-based τ range of the rotation-plus-translation fit (``translation_range_mm``; D10:
    not identifiable, a bracket only), the condyle residual of that fit and the helical-axis
    distance from the condyle midpoint (``dist_tmj``, from ``fha_subject``).

    Groups: ``all``; ``plausible_condyles`` (the stored D18 flag); ``condyles_unknown``
    (no anatomy); D29: ``plausible_condyles_<d>mm`` with the depth threshold at its fixed
    neighbours :data:`CONDYLE_DEPTH_NEIGHBOURS_MM`."""
    cp = pd.to_numeric(subj.condyles_plausible, errors="coerce")
    ok = set(subj.loc[cp.eq(1), "pid"])          # NaN (no anatomy) is unknown, not plausible
    unknown = set(subj.loc[cp.isna(), "pid"])
    alt = {f"plausible_condyles_{d:g}mm": set(subj.loc[plausible_at_depth(subj, d), "pid"])
           for d in CONDYLE_DEPTH_NEIGHBOURS_MM}
    h = hinge.pivot(index="pid", columns="model", values="rms_incisor_mm") \
        .add_prefix("hinge_rms_incisor_").reset_index()
    tr = hinge[hinge.model == "rot_trans"][["pid", "translation_range_mm", "rms_condyles_mm"]]
    t = subj.merge(h, on="pid", how="left").merge(tr, on="pid", how="left")
    sides = [c for c in ("condyle_left_disp_mm", "condyle_right_disp_mm") if c in t]
    if len(sides) == 2:
        t["translation_max_mm"] = t[sides].max(axis=1, skipna=False)
        t["translation_mean_mm"] = t[sides].mean(axis=1, skipna=False)
    if fha_subject is not None and not fha_subject.empty and "dist_tmj" in fha_subject:
        t = t.merge(fha_subject[["pid", "dist_tmj"]], on="pid", how="left")
    rows = []
    cols = list(CONDYLE_METRICS) + [c for c in t.columns if c.startswith("hinge_rms_incisor_")] \
        + ["translation_max_mm", "translation_mean_mm", "translation_range_mm",
           "rms_condyles_mm", "dist_tmj"]
    for c in cols:
        if c not in t:
            continue
        groups = [("all", t.pid.notna()), ("plausible_condyles", t.pid.isin(ok))]
        if unknown:
            groups.append(("condyles_unknown", t.pid.isin(unknown)))
        groups += [(g, t.pid.isin(v)) for g, v in alt.items()]
        for group, sel in groups:
            rows.append({"metric": c, "group": group, **describe(t.loc[sel, c].to_numpy(),
                                                                seed=seed, stat="median")})
    return pd.DataFrame(rows)
