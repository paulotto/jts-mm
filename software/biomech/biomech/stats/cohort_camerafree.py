# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cohort part of the camera-free study X5 (``camerafree_cohort``).

* **E4gen**: the ridge translation model (:class:`biomech.imu.camerafree.TranslationModel`)
  trained leave-one-subject-out on the optical calibration blocks (free_motion) of every
  other participant of the store with optics -- a participant without jaw IMU trains too
  (E32): every participant with optics in the analysis of record (14 participants, 13 training
  blocks per test participant); the 13-participant paper store has 12 training blocks --
  scaled by each one's incisor–condyle distance, driven by the test participant's primary
  E3 rotation and scored like the participant-stage estimators;
* **per participant**: the median over the scored tasks of each task's RMSE (opening,
  geodesic; incisor over all scored, the rotational and the translational tasks);
* **cohort**: median with 95 % BCa CI (``stats.core.describe``) and the verdict against
  the bound (within: upper CI < bound; compatible: median < bound; exceeds: lower CI > bound;
  otherwise inconclusive);
* **paired ladder contrasts** (``stats.families.paired_contrast``, the shared test path):
  exact Wilcoxon signed-rank (normal approximation with tied differences), Hodges–Lehmann
  estimate with its exact 95 % CI, Holm within the two families (rotation: E2a−E1, E2b−E2a,
  E3−E2a on the opening RMSE; translation: E4own−E4gen, E4gen−E4hinge, E5−E4own, E6−E4own on
  the incisor RMSE); H2 (E3's drift slope against E2a's, |slope|) as its own family;
* **drift**: per participant the OLS slope of the signed opening error against time since
  the reference (rest phase) and since the last anchor (°/min), cohort medians, and a
  random-intercept mixed model per estimator and clock (``stats.core.mixed_model``; samples
  at 1 Hz are autocorrelated, so its SE is optimistic: descriptive);
* **cycle amplitudes**: repeated-measures Bland–Altman (``bland_altman_rm`` with the
  participant-cluster bootstrap CIs) per estimator, quantity and task group;
* **X5b / X5c** (oracle analyses next to E4own, never camera-free results): per participant
  the same task-RMSE medians (and each movement task's own RMSE) of the decomposition rows
  (``camerafree:cf_decomposition``: optical rotation + E4own model, E3 + optical
  translation) and of the task-conditioned model (``camerafree:cf_task_model``), with E4own
  from ``cf_translation``; cohort medians with BCa CI (no bound verdict: descriptive); paired
  contrasts X5b_rotation − X5b_model (which half dominates) and X5c − E4own, Holm within each;
* **E7** (movement-class model choice from the ml LOSO classifiers) is not computed: the
  rule mapping a predicted class to a translation model is not fixed in the plan (H4.2 gives
  one line), and ``ml_eval`` is a plugin stage that the D22 sensitivity variants skip, so a
  dependency on it would drop this stage from those runs. Its rows say so (NaN).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.stats.core import bland_altman_rm, bland_altman_rm_ci, describe, holm, mixed_model
from biomech.stats.families import paired_contrast

BOUNDS = {"rmse_open_deg": 1.2, "rmse_geo_deg": np.nan, "rmse_inc_mm": 2.0,
          "rmse_inc_mm_rot": 2.0, "rmse_inc_mm_trans": 2.0}
FAMILIES = {"rotation": ("rmse_open_deg", (("E2a", "E1"), ("E2b", "E2a"), ("E3", "E2a"))),
            "translation": ("rmse_inc_mm", (("E4own", "E4gen"), ("E4gen", "E4hinge"),
                                            ("E5", "E4own"), ("E6", "E4own")))}
ORACLE_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing")
ORACLE_ANALYSIS = {"X5b_model": "X5b", "X5b_rotation": "X5b", "X5c": "X5c",
                   "E4own": "E4own (reference)"}
ORACLE_FAMILIES = {
    "X5b": (("rmse_inc_mm_rot", "rmse_inc_mm_trans"), (("X5b_rotation", "X5b_model"),)),
    "X5c": (("rmse_inc_mm", "rmse_inc_mm_rot", "rmse_inc_mm_trans"), (("X5c", "E4own"),)),
}
# Reader wording (stored cells are shown as they are): no plan codes.
E7_NOTE = ("movement-class model not computed: the rule from movement class to translation "
           "model is not fixed in the analysis plan, and the decoder stage is skipped by the "
           "pose low-pass sensitivity variants")
DEFAULTS: dict[str, Any] = {"exclude_generic": [], "n_boot": 4000}


# -- small statistics ------------------------------------------------------------------------
def verdict(ci_lo: float, med: float, ci_hi: float, bound: float) -> str:
    if not np.isfinite(bound) or not np.isfinite(med):
        return ""
    if np.isfinite(ci_hi) and ci_hi < bound:
        return "within"
    if np.isfinite(ci_lo) and ci_lo > bound:
        return "exceeds"
    return "compatible" if med < bound else "inconclusive"


# -- E4gen -----------------------------------------------------------------------------------
def generic(ctx: Any, p: dict[str, Any], cp: dict[str, Any], summary: pd.DataFrame
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """E4gen per participant with camera-free arrays: (task rows, cycle rows, model rows)."""
    from biomech.imu import camerafree as cf
    from biomech.imu import stage_camerafree as S
    from biomech.kinematics.pose_chain import KinParams

    blocks: dict[str, Any] = {}
    for q in ctx.participants:
        try:
            b = S.optical_block(ctx.upstream_arrays("kinematics", q.pid),
                                str(p["calibration_phase"]), float(p["grid_hz"]))
        except (KeyError, FileNotFoundError, LookupError):
            b = None
        if b is not None:
            blocks[q.pid] = b
    lo, hi, k = p["ridge_log10_lambda"]
    lams = np.logspace(float(lo), float(hi), int(k))
    kp = KinParams.from_dict(ctx.config.signals.get("kinematics"))
    excl = {str(x) for x in cp.get("exclude_generic") or []}
    cyc_all = ctx.upstream("kinematics", "cycles")
    rows, cyc_rows, mrows = [], [], []
    conf = summary[summary["configured"].astype(bool)] if "configured" in summary else summary
    for s in conf.itertuples():
        pid = s.pid
        try:
            a = ctx.upstream_arrays("camerafree", pid)
        except (KeyError, FileNotFoundError, LookupError):
            continue
        if "rotvec_deg__E3" not in a:
            continue
        train = [b for q, b in blocks.items() if q != pid and q not in excl]
        if not train:
            continue
        model = cf.TranslationModel.fit_generic(train, lambdas=lams)
        mrows.append({"pid": pid, "n_train": len(train), "lambda": model.lam,
                      "cv_rmse_rel": model.cv_rmse, "n_samples": model.n})
        t_s = np.asarray(a["t_s"], float)
        rv3 = np.asarray(a["rotvec_deg__E3"], float)
        R3 = S.rot_of(rv3)
        inc0 = np.array([getattr(s, f"incisor0_{x}_mm") for x in "xyz"], float)
        c0 = np.array([getattr(s, f"condyle0_{x}_mm") for x in "xyz"], float)
        tau = model.predict(rv3, cf.theta_rate_dps(t_s, rv3[:, 1]), scale_mm=float(s.scale_mm))
        inc = cf.incisor_position(R3, tau, inc0, c0)
        fs = 1.0 / float(np.median(np.diff(t_s)))
        R_lp, inc_lp = S.lowpass(R3, inc, fs, kp.lowpass_hz, kp.lowpass_order)
        R_opt = S.rot_of(np.asarray(a["rotvec_deg__opt"], float))
        inc_opt = np.asarray(a["incisor_mm__opt"], float)
        phase = np.asarray(a["phase"])
        names = [str(x) for x in a["phase_names"]]
        excl_sc = set(p["exclude_scored"])
        masks = {n: phase == j for j, n in enumerate(names) if n != "rest"}
        masks["scored"] = np.isin(phase, [j for j, n in enumerate(names) if n not in excl_sc])
        rr = S.score_rows({"pid": pid, "estimator": "E4gen"}, t_s, R_lp, R_opt, masks,
                          inc_lp, inc_opt)
        for r in rr:
            r["scored"] = r["task"] == "scored" or r["task"] not in excl_sc
            r["task_group"] = S.TASK_GROUP.get(r["task"], "all" if r["task"] == "scored"
                                               else "other")
        rows += rr
        cyc = cyc_all[cyc_all["pid"] == pid] if "pid" in cyc_all else cyc_all
        cyc_rows += S.cycle_rows(pid, cyc, np.round(t_s * 1e6).astype(np.int64), {},
                                 {"opt": inc_opt, "E4gen": inc_lp})
    return rows, cyc_rows, mrows


# -- participant values, cohort, contrasts ---------------------------------------------------
def participant_values(rot: pd.DataFrame, tr: pd.DataFrame, prim: str,
                       tasks: tuple[str, ...] = ()) -> pd.DataFrame:
    """Per participant and estimator: the median over the scored tasks of the task RMSEs
    (with ``tasks``: also each named task's own incisor RMSE, outcome
    ``rmse_inc_mm__<task>``). The primary E3 also appears as ``E3``."""
    rows = []

    def add(df: pd.DataFrame, col: str, outcome: str, groups: tuple[str, ...] | None) -> None:
        if df.empty or col not in df:
            return
        d = df[df["scored"].astype(bool) & (df["task"] != "scored")]
        if groups is not None:
            d = d[d["task_group"].isin(groups)]
        for (pid, est), g in d.groupby(["pid", "estimator"]):
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size:
                rows.append({"pid": pid, "estimator": est, "outcome": outcome,
                             "value": float(np.median(v)), "n_tasks": int(v.size)})

    add(rot, "rmse_open_deg", "rmse_open_deg", None)
    add(rot, "rmse_geo_deg", "rmse_geo_deg", None)
    add(tr, "rmse_inc_mm", "rmse_inc_mm", None)
    add(tr, "rmse_inc_mm", "rmse_inc_mm_rot", ("rotational",))
    add(tr, "rmse_inc_mm", "rmse_inc_mm_trans", ("translational",))
    for t in tasks:
        if not tr.empty and "task" in tr:
            add(tr[tr["task"] == t], "rmse_inc_mm", f"rmse_inc_mm__{t}", None)
    out = pd.DataFrame(rows, columns=["pid", "estimator", "outcome", "value", "n_tasks"])
    alias = out[out["estimator"] == prim].assign(estimator="E3")
    return pd.concat([out, alias], ignore_index=True)


def describe_rows(part: pd.DataFrame, seed: int, n_boot: int,
                  bounds: dict[str, float]) -> list[dict[str, Any]]:
    """Per estimator and outcome: ``describe`` (median, 95 % BCa CI) of the participant
    values, the bound (NaN when the outcome has none) and the verdict."""
    rows = []
    for (est, oc), g in part.groupby(["estimator", "outcome"], sort=False):
        d = describe(g["value"].to_numpy(float), n_boot=n_boot, seed=seed, stat="median")
        b = bounds.get(oc, np.nan)
        rows.append({"estimator": est, "outcome": oc, **d, "bound": b,
                     "verdict": verdict(d["ci_lo"], d["median"], d["ci_hi"], b), "note": ""})
    return rows


def cohort_table(part: pd.DataFrame, seed: int, n_boot: int) -> pd.DataFrame:
    rows = describe_rows(part, seed, n_boot, BOUNDS)
    for oc in ("rmse_inc_mm", "rmse_inc_mm_rot", "rmse_inc_mm_trans"):
        rows.append({"estimator": "E7", "outcome": oc, **describe(np.array([])),
                     "bound": BOUNDS[oc], "verdict": "", "note": E7_NOTE})
    return pd.DataFrame(rows)


def contrasts(part: pd.DataFrame, slopes: pd.DataFrame,
              families: dict[str, tuple[Any, tuple[tuple[str, str], ...]]] = FAMILIES
              ) -> pd.DataFrame:
    """Paired contrasts a − b of the participant values per family (outcome, or a tuple of
    outcomes, and the pairs), Holm within the family; the drift family from ``slopes``."""
    rows = []

    def one(family: str, a: str, b: str, wa: pd.Series, wb: pd.Series, outcome: str) -> None:
        j = pd.concat([wa.rename("a"), wb.rename("b")], axis=1).dropna()
        d = (j["a"] - j["b"]).to_numpy(float)
        c = paired_contrast(d)
        rows.append({"family": family, "contrast": f"{a}-{b}", "a": a, "b": b,
                     "outcome": outcome, "n": int(d.size), "hl": c["hl"],
                     "hl_ci_lo": c["hl_ci_lo"], "hl_ci_hi": c["hl_ci_hi"],
                     "median_a": float(j["a"].median()) if len(j) else np.nan,
                     "median_b": float(j["b"].median()) if len(j) else np.nan,
                     "n_a_better": int((d < 0).sum()), "p": c["p"], "method": c["method"]})

    for fam, (ocs, pairs) in families.items():
        for oc in ((ocs,) if isinstance(ocs, str) else ocs):
            w = part[part["outcome"] == oc].pivot_table(index="pid", columns="estimator",
                                                        values="value")
            for a, b in pairs:
                if a in w and b in w:
                    one(fam, a, b, w[a], w[b], oc)
                else:
                    rows.append({"family": fam, "contrast": f"{a}-{b}", "a": a, "b": b,
                                 "outcome": oc, "n": 0, "p": np.nan, "method": "",
                                 "note": E7_NOTE if "E7" in (a, b)
                                 else f"{a} or {b} missing"})
    if not slopes.empty:
        s = slopes.assign(abs_slope=slopes["slope_dpm"].abs())
        e3 = s[(s.estimator == "E3") & (s.clock == "anchor")].set_index("pid")["abs_slope"]
        e2 = s[(s.estimator == "E2a") & (s.clock == "ref")].set_index("pid")["abs_slope"]
        if len(e3) and len(e2):
            one("drift", "E3", "E2a", e3, e2, "abs_drift_slope_dpm")
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for fam, g in out.groupby("family"):
        out.loc[g.index, "p_holm"] = holm(g["p"].tolist())
    return out


def drift(dr: pd.DataFrame, prim: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per-participant slopes, mixed models) of the signed opening error (°/min)."""
    if dr.empty:
        return pd.DataFrame(), pd.DataFrame()
    dr = pd.concat([dr, dr[dr["estimator"] == prim].assign(estimator="E3")], ignore_index=True)
    srows, mrows = [], []
    for clock, col in (("ref", "t_since_ref_s"), ("anchor", "t_since_anchor_s")):
        d = dr[np.isfinite(dr[col]) & np.isfinite(dr["err_open_deg"])]
        for (pid, est), g in d.groupby(["pid", "estimator"]):
            if len(g) >= 10 and np.ptp(g[col]) > 0:
                slope = float(np.polyfit(g[col] / 60.0, g["err_open_deg"], 1)[0])
                srows.append({"pid": pid, "estimator": est, "clock": clock,
                              "slope_dpm": slope, "n": int(len(g)),
                              "span_min": float(np.ptp(g[col]) / 60.0)})
        for est, g in d.groupby("estimator"):
            base = {"estimator": est, "clock": clock, "n_pid": int(g["pid"].nunique()),
                    "n": int(len(g))}
            if g["pid"].nunique() < 2:
                mrows.append({**base, "note": "fewer than 2 participants"})
                continue
            try:
                t = mixed_model(g.assign(t_min=g[col] / 60.0), "err_open_deg ~ t_min")
                r = t[t["term"] == "t_min"].iloc[0]
                mrows.append({**base, "slope_dpm": float(r["coef"]), "se": float(r["se"]),
                              "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
                              "p": float(r["p"]), "optimizer": str(r["optimizer"]),
                              "note": ""})
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                mrows.append({**base, "note": f"mixed model failed: {exc}"})
    return pd.DataFrame(srows), pd.DataFrame(mrows)


def oracle(dec: pd.DataFrame, x5c: pd.DataFrame, tr: pd.DataFrame, prim: str, seed: int,
           n_boot: int) -> dict[str, pd.DataFrame]:
    """X5b / X5c next to E4own: per-participant values (task-RMSE medians and each movement
    task's RMSE), cohort description (no bound: oracle analyses are descriptive) and the
    paired contrasts (``ORACLE_FAMILIES``)."""
    frames = [d[d["estimator"].isin(["X5b_model", "X5b_rotation", "X5c"])]
              for d in (dec, x5c) if not d.empty and "rmse_inc_mm" in d]
    if not frames or all(f.empty for f in frames):
        note = pd.DataFrame([{"note": "no X5b / X5c rows (no E4own model or no labels)"}])
        return {"cf_oracle_cohort": note}
    if not tr.empty and "estimator" in tr:
        frames.append(tr[tr["estimator"] == "E4own"])
    rows = pd.concat(frames, ignore_index=True)
    part = participant_values(pd.DataFrame(), rows, prim, ORACLE_TASKS)
    part = part[part["estimator"] != "E3"].copy()
    part.insert(1, "analysis", part["estimator"].map(ORACLE_ANALYSIS))
    coh = pd.DataFrame(describe_rows(part, seed, n_boot, {}))
    if not coh.empty:
        coh.insert(0, "analysis", coh["estimator"].map(ORACLE_ANALYSIS))
    return {"cf_oracle_participant": part, "cf_oracle_cohort": coh,
            "cf_oracle_contrasts": contrasts(part, pd.DataFrame(), ORACLE_FAMILIES)}


def cycles_ba(cyc: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    if cyc.empty:
        return pd.DataFrame(rows)
    cyc = cyc.assign(task_group=cyc["task"].map(lambda t: {
        "open_close": "rotational", "chewing": "rotational",
        "protrusion_retrusion": "translational", "left_right": "translational"}.get(t, "other")))
    for (est, q, tg), g in cyc.groupby(["estimator", "quantity", "task_group"]):
        g = g[np.isfinite(g["amp_est"]) & np.isfinite(g["amp_opt"])]
        if len(g) < 3:
            continue
        ba = bland_altman_rm(g["amp_est"], g["amp_opt"], g["pid"])
        ci = bland_altman_rm_ci(g["amp_est"], g["amp_opt"], g["pid"], seed)
        bound = 1.2 if q == "opening_deg" else 2.0
        rows.append({"estimator": est, "quantity": q, "task_group": tg, **ba, **ci,
                     "bound": bound, "loa_within_bound": bool(
                         np.isfinite(ba["loa_lo"]) and max(abs(ba["loa_lo"]),
                                                           abs(ba["loa_hi"])) <= bound)})
    return pd.DataFrame(rows)


def run(ctx: Any, p: dict[str, Any], cp: dict[str, Any] | None) -> dict[str, pd.DataFrame]:
    cp = {**DEFAULTS, **(cp or {})}
    seed = int(ctx.config.seed or 0)
    prim = str(p["primary_e3"])

    def up(name: str) -> pd.DataFrame:
        try:
            return ctx.upstream("camerafree", name)
        except (KeyError, FileNotFoundError, LookupError):
            return pd.DataFrame()

    summary = up("cf_summary")
    rot, tr, cyc, dr = up("cf_rotation"), up("cf_translation"), up("cf_cycles"), up("cf_drift")
    configured = not summary.empty and "configured" in summary and \
        summary["configured"].astype(bool).any()
    if not configured:
        note = pd.DataFrame([{"note": "no participant with camera-free results (no hand-eye "
                                      "mounting configured or no IMU data)"}])
        return {"cf_cohort": note}
    g_rows, g_cyc, g_models = generic(ctx, p, cp, summary)
    gen = pd.DataFrame(g_rows)
    tr_all = pd.concat([tr, gen], ignore_index=True) if not gen.empty else tr
    cyc_all = pd.concat([cyc, pd.DataFrame(g_cyc)], ignore_index=True) if g_cyc else cyc
    part = participant_values(rot, tr_all, prim)
    for col, oc in (("det_sensitivity", "det_sensitivity"), ("det_precision", "det_precision")):
        if col in summary:
            v = summary[summary["configured"].astype(bool)][["pid", col]].dropna()
            part = pd.concat([part, pd.DataFrame({"pid": v["pid"], "estimator": prim,
                                                  "outcome": oc, "value": v[col],
                                                  "n_tasks": np.nan})], ignore_index=True)
    slopes, lmm = drift(dr, prim)
    if not slopes.empty:
        sl = slopes.rename(columns={"slope_dpm": "value"})
        part = pd.concat([part, pd.DataFrame({
            "pid": sl["pid"], "estimator": sl["estimator"],
            "outcome": "drift_slope_" + sl["clock"] + "_dpm", "value": sl["value"],
            "n_tasks": np.nan})], ignore_index=True)
    tables = oracle(up("cf_decomposition"), up("cf_task_model"), tr, prim, seed,
                    int(cp["n_boot"]))
    return {**tables, "cf_participant": part,
            "cf_cohort": cohort_table(part, seed, int(cp["n_boot"])),
            "cf_contrasts": contrasts(part, slopes),
            "cf_generic": gen, "cf_generic_models": pd.DataFrame(g_models),
            "cf_drift_slopes": slopes, "cf_drift_cohort": lmm,
            "cf_cycles_ba": cycles_ba(cyc_all, seed)}
