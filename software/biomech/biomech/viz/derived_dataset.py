# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Derived tables of the dataset paper's numbers (``source: derived:<name>``; 2026-11-05).

Registered with :mod:`biomech.viz.derived`; import this module (``derived.table`` does) before
resolving the dataset numbers:

* ``svm_search`` -- the hyper-parameter grid of the primary decoder (RBF SVM) as the ml package
  defines it (``ml.models.registry.REGISTRY["svm"].grid``): one row per parameter with its
  values as text (``C``; ``gamma_mult``, the multiple of the training fold's "scale" kernel
  width). The nested search (inner folds, training subsample) is in the store's frozen
  configuration (``config:stages.ml_eval.*``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.viz.derived import _table


def _fmt_value(v: Any) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


@_table("svm_search")
def svm_search(ctx: Any) -> pd.DataFrame:
    from ml.models.registry import REGISTRY

    grid = REGISTRY["svm"].grid
    return pd.DataFrame([{"param": k, "grid_values": ", ".join(_fmt_value(v) for v in vals),
                          "grid_n": len(vals)} for k, vals in grid.items()])


@_table("cohort_flow")
def cohort_flow(ctx: Any) -> pd.DataFrame:
    """Participant flow (STROBE item 13; 2026-11-06) from the cohort manifest of
    the store the paper reads (the reporting_population snapshot keeps every recorded participant and
    marks the flagged participant ``include: false``): recorded, reported, excluded by the
    flag, and among those the ones whose jaw IMU did not record. One row."""
    import yaml

    doc = yaml.safe_load(ctx.config.cohort_path.read_text(encoding="utf-8")) or {}
    rows = list(doc.get("participants") or [])
    exc = [r for r in rows if not r.get("include", True)]
    flagged = [r for r in exc if "outside_protocol" in (r.get("flags") or [])]
    return pd.DataFrame([{
        "recorded": len(rows), "reported": len(rows) - len(exc), "excluded": len(exc),
        "excluded_outside_protocol": len(flagged),
        "excluded_outside_protocol_no_jaw_imu": sum(1 for r in flagged
                                         if "imu" in (r.get("unavailable") or {}))}])


@_table("p95_upper_bound")
def p95_upper_bound(ctx: Any) -> pd.DataFrame:
    """Why a P95 has no upper bound at this sample size (E31; 2026-11-08), for the
    workspace rows' n (``requirements:requirements``, key ``opening_mm``): the smallest n for
    which a one-sided upper ``CONF`` bound of the P95 exists (the sample maximum qualifies once
    1 - p^n >= CONF, i.e. n >= ln(1 - CONF) / ln p; found with the stage's own
    ``order_stat_upper``), and the confidence 1 - p^n with which the sample maximum of the
    present n lies above the P95. One row."""
    import numpy as np

    from biomech.stats.requirements import CONF, P_DESIGN, order_stat_upper

    req = ctx.table("requirements", "requirements")
    row = req[req["key"] == "opening_mm"]
    n = int(row["n"].iloc[0]) if not row.empty else 0
    p = P_DESIGN / 100
    n_min = next(m for m in range(1, 10_000) if order_stat_upper(m, p, CONF)[0] > 0)
    return pd.DataFrame([{"n": n, "n_min_upper": n_min,
                          "n_min_upper_exact": float(np.log(1 - CONF) / np.log(p)),
                          "max_above_p95_conf": 1 - p ** n if n else np.nan}])


# X5d (J15, exploratory): estimators scored on free motion. Leakage-free = not using
# the participant's free-motion optics; in-sample = fitted on it (E1, E4own) or using a model
# fitted on it (E6), reported as optimistic references only.
CF_FREE = (("rmse_open_deg", "E2a", "leakage_free"), ("rmse_open_deg", "E2b", "leakage_free"),
           ("rmse_open_deg", "E3", "leakage_free"), ("rmse_open_deg", "E1", "in_sample"),
           ("rmse_inc_mm", "E4hinge", "leakage_free"), ("rmse_inc_mm", "E4gen", "leakage_free"),
           ("rmse_inc_mm", "E5", "leakage_free"), ("rmse_inc_mm", "E4own", "in_sample"),
           ("rmse_inc_mm", "E6", "in_sample"))


@_table("cf_free_motion")
def cf_free_motion(ctx: Any) -> pd.DataFrame:
    """X5d (J15; exploratory, descriptive): per estimator the cohort median with its
    95 % BCa CI (``stats.core.describe``, 4000 resamples, the study seed, as ``cf_cohort``) of
    the participants' RMSE on the free-motion phase, from the stored per-task rows
    (``camerafree:cf_rotation`` / ``cf_translation``, ``camerafree_cohort:cf_generic`` for
    E4gen; same 6 Hz low-pass as every X5 score). ``E3`` is the primary E3
    (``cf_summary.primary_e3``). ``role``: leakage_free, or in_sample (E1, E4own, E6: fitted
    on free motion or using a model fitted on it; optimistic). No bound, no test."""
    import numpy as np

    from biomech.stats.core import describe

    def tab(stage: str, name: str) -> pd.DataFrame:
        try:
            return ctx.table(stage, name)
        except (KeyError, FileNotFoundError, LookupError):
            return pd.DataFrame()

    summ = tab("camerafree", "cf_summary")
    prim = (str(summ["primary_e3"].dropna().iloc[0])
            if "primary_e3" in summ and summ["primary_e3"].notna().any() else "E3b")
    rows = pd.concat([tab("camerafree", "cf_rotation"), tab("camerafree", "cf_translation"),
                      tab("camerafree_cohort", "cf_generic")], ignore_index=True)
    if rows.empty or "task" not in rows:
        return pd.DataFrame()
    fm = rows[rows["task"] == "free_motion"]
    seed = int(ctx.config.seed or 0)
    out = []
    for oc, est, role in CF_FREE:
        src = prim if est == "E3" else est
        v = fm.loc[fm["estimator"] == src, ["pid", oc]].dropna().drop_duplicates("pid")[oc]
        d = describe(v.to_numpy(float), n_boot=4000, seed=seed, stat="median")
        out.append({"estimator": est, "source_estimator": src, "outcome": oc, "role": role,
                    "n": d["n"], "median": d["median"], "ci_lo": d["ci_lo"],
                    "ci_hi": d["ci_hi"], "q1": d["q1"], "q3": d["q3"],
                    "ci_percentile": d["ci_percentile"]})
    return pd.DataFrame(out).astype({"n": int}).replace({np.inf: np.nan})


ANCHOR_TASKS = ("open_close", "chewing", "protrusion_retrusion", "left_right")


@_table("cf_heading_identifiable")
def cf_heading_identifiable(ctx: Any) -> pd.DataFrame:
    """J32.1 (descriptive, post hoc): the heading from the IMUs alone (E2b) against the
    optical heading (E2a) in the participants whose E2b heading was identifiable
    (``camerafree:cf_summary.psi_b_identifiable``). The pre-specified contrast E2b - E2a runs
    over all participants, and those without an identifiable heading fall back to E2a
    (difference exactly 0). Per participant the difference of the opening-angle RMSE
    (``camerafree_cohort:cf_participant``, outcome ``rmse_open_deg``); one row: n, median,
    smallest, largest, and the number of participants with an increase. No test, no CI."""
    def tab(stage: str, name: str) -> pd.DataFrame:
        try:
            return ctx.table(stage, name)
        except (KeyError, FileNotFoundError, LookupError):
            return pd.DataFrame()

    summ = tab("camerafree", "cf_summary")
    part = tab("camerafree_cohort", "cf_participant")
    if summ.empty or part.empty or "psi_b_identifiable" not in summ:
        return pd.DataFrame()
    ident = set(summ.loc[summ["psi_b_identifiable"].fillna(False).astype(bool), "pid"])
    w = part[part["outcome"] == "rmse_open_deg"].pivot_table(
        index="pid", columns="estimator", values="value")
    if not {"E2a", "E2b"} <= set(w.columns):
        return pd.DataFrame()
    d = (w["E2b"] - w["E2a"]).dropna()
    d = d[d.index.isin(ident)]
    if d.empty:
        return pd.DataFrame()
    return pd.DataFrame([{"outcome": "rmse_open_deg", "n": int(d.size),
                          "median": float(d.median()), "min": float(d.min()),
                          "max": float(d.max()), "n_up": int((d > 0).sum())}])


def _j35_file(ctx: Any) -> pd.DataFrame:
    """``<store>/audit/j35_cf_opening_<snap>.parquet`` (J35), with its sha256."""
    import hashlib
    from pathlib import Path

    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"j35_cf_opening_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/j35_cf_opening.py")
    return pd.read_parquet(f).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


def cf_opening_table(d: pd.DataFrame) -> pd.DataFrame:
    """The J35 statistics from the cycle rows of the audit file (one row per task and
    estimator): the repeated-measures limits of agreement of C3a (bias, limits, the larger
    absolute limit with its MOVER interval), participants and cycles, and for E2a and the
    resets the contrast to E1 of the participants' mean absolute errors (Hodges-Lehmann with
    exact 95 % CI, two-sided signed-rank p, Holm over the two contrasts of the task)."""
    import numpy as np

    from biomech.stats.cohort_misc import max_abs_loa_mover
    from biomech.stats.core import bland_altman_rm, holm
    from biomech.stats.families import paired_contrast

    rows = []
    for task, gt in d.groupby("task"):
        mae = (gt.assign(ae=(gt.opening_est_mm - gt.opening_opt_mm).abs())
               .groupby(["pid", "estimator"]).ae.mean().unstack())
        con = {}
        for est in ("E2a", "E3"):
            if {est, "E1"} <= set(mae.columns):
                con[est] = paired_contrast((mae[est] - mae["E1"]).to_numpy(float))
        ph = dict(zip(con, holm([c["p"] for c in con.values()]))) if con else {}
        for est, g in gt.groupby("estimator"):
            e, r = g.opening_est_mm.to_numpy(float), g.opening_opt_mm.to_numpy(float)
            pid = g.pid.to_numpy()
            ok = np.isfinite(e) & np.isfinite(r)
            ba = bland_altman_rm(e, r, pid)
            m, lo, hi = max_abs_loa_mover((e - r)[ok], pid[ok])
            c = con.get(est, {})
            rows.append({"task": task, "estimator": est, "n_cycles": ba["n"],
                         "n": ba["n_subjects"], "bias": ba["bias"], "loa_lo": ba["loa_lo"],
                         "loa_hi": ba["loa_hi"], "max_abs_loa": m, "max_abs_loa_ci_lo": lo,
                         "max_abs_loa_ci_hi": hi, "hl": c.get("hl", np.nan),
                         "hl_ci_lo": c.get("hl_ci_lo", np.nan),
                         "hl_ci_hi": c.get("hl_ci_hi", np.nan), "p": c.get("p", np.nan),
                         "p_holm": ph.get(est, np.nan)})
    return pd.DataFrame(rows)


@_table("cf_opening_cycles")
def cf_opening_cycles(ctx: Any) -> pd.DataFrame:
    """J35 (exploratory, post hoc): the incisal opening per cycle from the camera-free
    estimators (E1 in-take fit, E2a mountings calibrated once, E3 the primary resets) with the
    generic opening model, against the optical opening, in open-close and chewing; see
    :func:`cf_opening_table`. From the audit file of ``j35_cf_opening.py``."""
    d = _j35_file(ctx)
    return cf_opening_table(d).assign(sha256=d["sha256"].iloc[0])


@_table("cf_anchor_offsets")
def cf_anchor_offsets(ctx: Any) -> pd.DataFrame:
    """Rest-pose offsets at the camera-free anchors (J16; descriptive): how far the
    jaw, at the rest poses the IMUs detect, lies from the rest reference pose -- the
    repeatable-rest-pose assumption of the resets. From ``camerafree:cf_anchors``,
    the primary E3's rows (``cf_summary.primary_e3``): per anchor the optical offset over the
    anchor window, ``|opt_open_deg|`` (opening angle) and ``opt_inc_mm`` (3-D incisor);
    per participant the median over its anchors in each scored task; per task the cohort
    median with its 95 % BCa CI (``stats.core.describe``, 4000 resamples, the study seed).
    n per task: the participants with at least one anchor there."""
    from biomech.stats.core import describe

    def tab(stage: str, name: str) -> pd.DataFrame:
        try:
            return ctx.table(stage, name)
        except (KeyError, FileNotFoundError, LookupError):
            return pd.DataFrame()

    summ = tab("camerafree", "cf_summary")
    prim = (str(summ["primary_e3"].dropna().iloc[0])
            if "primary_e3" in summ and summ["primary_e3"].notna().any() else "E3b")
    a = tab("camerafree", "cf_anchors")
    if a.empty or "estimator" not in a:
        return pd.DataFrame()
    a = a[a["estimator"] == prim].assign(open_abs=lambda d: d["opt_open_deg"].abs())
    seed = int(ctx.config.seed or 0)
    out = []
    for task in ANCHOR_TASKS:
        g = a[a["task"] == task]
        for col, oc in (("open_abs", "offset_open_deg"), ("opt_inc_mm", "offset_inc_mm")):
            v = g.groupby("pid")[col].median().dropna()
            d = describe(v.to_numpy(float), n_boot=4000, seed=seed, stat="median")
            out.append({"task": task, "outcome": oc, "estimator": prim, "n": d["n"],
                        "n_anchors": int(g[col].notna().sum()), "median": d["median"],
                        "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"], "q1": d["q1"],
                        "q3": d["q3"], "ci_percentile": d["ci_percentile"]})
    return pd.DataFrame(out)


def _audit_file(ctx: Any, suffix: str = "") -> Any:
    """``<store>/audit/c2_null_period_<snapshot><suffix>.parquet`` of the store ``ctx`` reads
    (written by ``scripts/audit_recompute/c2_null_period.py``; J17)."""
    from pathlib import Path

    snap = getattr(ctx.config, "snapshot", None) or "live"
    return Path(ctx.config.out_dir) / "audit" / f"c2_null_period_{snap}{suffix}.parquet"


@_table("c2_null_period")
def c2_null_period(ctx: Any) -> pd.DataFrame:
    """C2 shift null with and without period-multiple shifts (J17, robustness of
    D23a): one row per ``band`` (0 = the primary null, reproduced exactly; 0.1 = shifts within
    ±10 % of a multiple of the participant's median cycle period redrawn), task, direction,
    detector and rule: the cohort null median ``lead_null_ms``, the median observed − null
    ``mn_median`` with its 95 % BCa CI (``mn_ci_lo``, ``mn_ci_hi``; n ``mn_n``), the one-sided
    signed-rank ``p_null`` and sign-flip ``p_null_perm`` (10 000) with their Holm versions.
    Read from the audit script's output; ``sha256`` identifies the file."""
    import hashlib

    f = _audit_file(ctx)
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/c2_null_period.py")
    d = pd.read_parquet(f)
    return d.assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


@_table("c2_null_period_desc")
def c2_null_period_desc(ctx: Any) -> pd.DataFrame:
    """Descriptives of the primary shift null (J17 amendment), per task: the cohort
    median over participants (n = participants with a value) of the share of surrogate windows
    wrapping the phase end (``junction_share``), the share of the 200 shifts within ±10 % of a
    multiple of the median cycle period (``near_share``) and the SD of the start-to-start
    intervals over their median (``interval_sd_rel``)."""
    import hashlib

    import numpy as np

    f = _audit_file(ctx, "_desc")
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/c2_null_period.py")
    d = pd.read_parquet(f)
    d = d[d["band"] == 0]
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    rows = []
    for task, g in d.groupby("task"):
        for col in ("junction_share", "near_share", "interval_sd_rel", "period_s"):
            v = g[col].to_numpy(float)
            v = v[np.isfinite(v)]
            rows.append({"task": task, "metric": col, "n": int(v.size),
                         "median": float(np.median(v)) if v.size else np.nan, "sha256": sha})
    return pd.DataFrame(rows)


# -- J18: C2 timing, post hoc / exploratory additions (J18) -------------------------
def _keyparts(key: str) -> dict[str, str]:
    return dict(p.split("=", 1) for p in str(key).split("|") if "=" in p)


@_table("j18_symmetric")
def j18_symmetric(ctx: Any) -> pd.DataFrame:
    """J18.1: the C2 lead with the symmetric onset window (variant ``c2_symmetric_window``,
    ±0.4 s) from the comparison frozen with the store (``variant`` column): per task,
    direction and rule (TKEO) the observed lead (median, CI, n), its shift null and the median
    observed − null, plus the 95 % BCa CI of observed − null over the variant's per-participant
    ``timing_null`` pairs (study seed)."""
    import numpy as np

    from biomech.sensitivity import comparison_path
    from biomech.stats.core import describe

    d = pd.read_parquet(comparison_path(ctx.config, "c2_symmetric_window"))
    tc = d[d["table"] == "timing_cohort"]
    tn = d[(d["table"] == "timing_null") & (d["column"] == "lead_minus_null_ms")]
    seed = int(ctx.config.seed or 0)
    rows = []
    for task in ("open_close", "chewing"):
        for direction in ("opening", "closing"):
            for rule in ("first", "drive"):
                want = {"task": task, "direction": direction, "detector": "tkeo",
                        "rule": rule}
                sel = tc[tc["key"].map(lambda k: (lambda kp: all(kp.get(a) == b for a, b in
                                                     {**want, "metric": "lead_ms"}.items()))
                                       (_keyparts(k)))]
                v = {c: float(g["variant"].iloc[0]) for c, g in sel.groupby("column")}
                pairs = tn[tn["key"].map(lambda k: all(_keyparts(k).get(a) == b
                                                       for a, b in want.items()))]
                mn = describe(pairs["variant"].to_numpy(float), seed=seed, stat="median")
                rows.append({**want, "n": v.get("n", np.nan), "median": v.get("median", np.nan),
                             "ci_lo": v.get("ci_lo", np.nan), "ci_hi": v.get("ci_hi", np.nan),
                             "lead_null_ms": v.get("lead_null_ms", np.nan),
                             "lead_minus_null_ms": v.get("lead_minus_null_ms", np.nan),
                             "p_null": v.get("p_null", np.nan),
                             "mn_median": mn["median"], "mn_ci_lo": mn["ci_lo"],
                             "mn_ci_hi": mn["ci_hi"], "mn_n": mn["n"]})
    return pd.DataFrame(rows)


@_table("j18_release")
def j18_release(ctx: Any) -> pd.DataFrame:
    """J18.3: closer release relative to the opening onset, per task and anchor (``omega``:
    t_ω from ``timing:c2e_release``; ``alpha``: the opening cycle's t_α from
    ``timing:c2e_cycles``, joined on pid, task, cycle): the cohort median of the
    per-participant median release lead (anchor − release; CI, n) and the cohort median of the
    per-participant share of cycles whose release precedes the anchor."""
    import numpy as np

    from biomech.stats.core import describe

    rel = ctx.table("timing", "c2e_release")
    cyc = ctx.table("timing", "c2e_cycles")
    seed = int(ctx.config.seed or 0)
    a = (cyc[cyc["direction"] == "opening"].dropna(subset=["t_alpha_us"])
         .drop_duplicates(["pid", "task", "cycle"])[["pid", "task", "cycle", "t_alpha_us"]])
    r = rel.dropna(subset=["t_release_us"]).merge(a, on=["pid", "task", "cycle"], how="left")
    r["lead_omega_ms"] = r["lead_release_ms"]
    r["lead_alpha_ms"] = (r["t_alpha_us"] - r["t_release_us"]) / 1e3
    rows = []
    for task, g in r.groupby("task"):
        for anchor in ("omega", "alpha"):
            col = f"lead_{anchor}_ms"
            gg = g.dropna(subset=[col])
            per = gg.groupby("pid")[col].median()
            share = gg.groupby("pid")[col].apply(lambda x: float((x > 0).mean()))
            d = describe(per.to_numpy(float), seed=seed, stat="median")
            s = describe(share.to_numpy(float), seed=seed, stat="median")
            rows.append({"task": task, "anchor": anchor, "n": d["n"], "median": d["median"],
                         "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                         "share_before": s["median"], "share_ci_lo": s["ci_lo"],
                         "share_ci_hi": s["ci_hi"], "n_cycles": int(len(gg))})
    return pd.DataFrame(rows)


@_table("j18_activation")
def j18_activation(ctx: Any) -> pd.DataFrame:
    """J18.7: digastric lead against activation strength (stored ``timing:onsets``, TKEO,
    openers, opening): per participant the Spearman ρ of ``burst_db`` with ``lead_ms`` over
    the usable channel-cycles (≥ 5) and of ``burst_db`` with ``usable`` over all; per task and
    rule the cohort median ρ (CI, n). Descriptive."""
    import numpy as np
    from scipy.stats import spearmanr

    from biomech.io.session import OPENERS
    from biomech.stats.core import describe

    o = ctx.table("timing", "onsets")
    o = o[(o["detector"] == "tkeo") & (o["direction"] == "opening")
          & o["muscle"].isin(OPENERS)]
    seed = int(ctx.config.seed or 0)
    rows = []
    for (task, rule), g in o.groupby(["task", "rule"]):
        if rule not in ("first", "drive"):
            continue
        for kind in ("lead", "usable"):
            rho = []
            for _, p in g.groupby("pid"):
                if kind == "lead":
                    q = p[p["usable"].astype(bool)].dropna(subset=["burst_db", "lead_ms"])
                    if len(q) >= 5 and q["burst_db"].nunique() > 1 and q["lead_ms"].nunique() > 1:
                        rho.append(spearmanr(q["burst_db"], q["lead_ms"])[0])
                else:
                    q = p.dropna(subset=["burst_db"])
                    u = q["usable"].astype(float)
                    if len(q) >= 5 and u.nunique() > 1:
                        rho.append(spearmanr(q["burst_db"], u)[0])
            d = describe(np.asarray(rho, float), seed=seed, stat="median")
            rows.append({"task": task, "rule": rule, "relation": kind, "n": d["n"],
                         "median": d["median"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]})
    return pd.DataFrame(rows)


def _j18_file(ctx: Any, name: str) -> pd.DataFrame:
    """``<store>/audit/j18_<name>_<snapshot>.parquet`` (scripts/audit_recompute/j18_*.py)."""
    import hashlib
    from pathlib import Path

    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"j18_{name}_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/j18_*.py")
    return pd.read_parquet(f).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


def _cohort(v: Any, seed: int) -> dict[str, float]:
    import numpy as np

    from biomech.stats.core import describe

    d = describe(np.asarray(v, float), seed=seed, stat="median")
    return {"n": d["n"], "median": d["median"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"]}


def _wilcoxon2(v: Any) -> float:
    import numpy as np

    from biomech.stats.families import wilcoxon_signed

    x = np.asarray(v, float)
    x = x[np.isfinite(x) & (x != 0)]
    return float(wilcoxon_signed(x, "two-sided")[1])


def _j18_rows(ctx: Any) -> pd.DataFrame:
    """Every J18 cohort row (J18.2, .4, .5) with its test, Holm over the family
    "timing, post hoc" (the 22 tests named in the plan)."""
    import numpy as np

    from biomech.stats.core import holm

    seed = int(ctx.config.seed or 0)
    rows = []
    cp = _j18_file(ctx, "coupling")
    for (task, direction), g in cp.groupby(["task", "direction"]):
        lag = g["lag_ms"].to_numpy(float)
        d = g["lag_minus_null_ms"].to_numpy(float)
        rows.append({"item": "coupling", "task": task, "key": direction, "stat": "lag",
                     **_cohort(lag, seed), "p": np.nan})
        rows.append({"item": "coupling", "task": task, "key": direction,
                     "stat": "lag_minus_null", **_cohort(d, seed), "p": _wilcoxon2(d)})
    rs = _j18_file(ctx, "rise")
    for (task, cue), g in rs.groupby(["task", "cue"]):
        ld = g["lead_ms"].to_numpy(float)
        mn = (g["lead_ms"] - g["null_lead_ms"]).to_numpy(float)
        rows.append({"item": "rise", "task": task, "key": cue, "stat": "lead",
                     **_cohort(ld, seed), "p": _wilcoxon2(ld)})
        rows.append({"item": "rise", "task": task, "key": cue, "stat": "lead_minus_null",
                     **_cohort(mn, seed), "p": _wilcoxon2(mn)})
        rows.append({"item": "rise", "task": task, "key": cue, "stat": "null_lead",
                     **_cohort(g["null_lead_ms"].to_numpy(float), seed), "p": np.nan})
    dv = _j18_file(ctx, "device")
    for (task, det), g in dv.groupby(["task", "detector"]):
        for k in ("3", "5"):
            per = g.groupby("pid")[f"delta_k{k}_ms"].median()
            rows.append({"item": "device", "task": task, "key": det, "stat": f"delta_k{k}",
                         **_cohort(per.to_numpy(float), seed),
                         "p": _wilcoxon2(per.to_numpy(float)) if k == "3" else np.nan})
            share = g.groupby("pid")[f"t_imu_k{k}_us"].apply(lambda x: float(x.notna().mean()))
            rows.append({"item": "device", "task": task, "key": det,
                         "stat": f"imu_detected_k{k}", **_cohort(share.to_numpy(float), seed),
                         "p": np.nan})
        share = g.groupby("pid")["t_emg_us"].apply(lambda x: float(x.notna().mean()))
        rows.append({"item": "device", "task": task, "key": det, "stat": "emg_detected",
                     **_cohort(share.to_numpy(float), seed), "p": np.nan})
    out = pd.DataFrame(rows)
    fam = out["p"].notna()
    out["p_holm"] = np.nan
    out.loc[fam, "p_holm"] = holm(out.loc[fam, "p"].tolist())
    out["family_size"] = int(fam.sum())
    return out


@_table("j18_timing")
def j18_timing(ctx: Any) -> pd.DataFrame:
    """J18.2 / .4 / .5 cohort rows (``item`` coupling | rise | device; ``key`` direction, cue
    or detector; ``stat``): median with 95 % BCa CI, n; two-sided signed-rank ``p`` where the
    plan names a test, ``p_holm`` over the family "timing, post hoc" (``family_size``)."""
    return _j18_rows(ctx)


@_table("j18_device_rest")
def j18_device_rest(ctx: Any) -> pd.DataFrame:
    """J18.5 false alarms per minute of rest, per detector: cohort median (CI, n) and the
    rest minutes (median)."""
    seed = int(ctx.config.seed or 0)
    d = _j18_file(ctx, "device_rest")
    return pd.DataFrame([{"detector": det, **_cohort(g["false_per_min"], seed),
                          "minutes_median": float(g["minutes"].median()),
                          "n_false_total": int(g["n_false"].sum())}
                         for det, g in d.groupby("detector")])


@_table("j18_synth")
def j18_synth(ctx: Any) -> pd.DataFrame:
    """J18.6 per level, lag, task and rule (TKEO), every quantity per donor first (J22.3):
    usable share of the opener channel-cycles (median over donors of the donor shares); bias =
    cohort median (CI, n donors) of the per-donor median error over usable cycles; robust SD
    (1.4826 MAD) of the per-cycle errors (median over donors of the donor values); the pooled
    share, robust SD and SD over all cycles for comparison; level in dB."""
    import numpy as np

    seed = int(ctx.config.seed or 0)
    d = _j18_file(ctx, "synth")
    d = d[d["detector"] == "tkeo"]
    rows = []
    for (lv, lag, task, rule), g in d.groupby(["level", "lag_ms", "task", "rule"]):
        u = g[g["usable"].astype(bool)]
        per = u.groupby("donor")["error_ms"].median()
        e = u["error_ms"].to_numpy(float)
        e = e[np.isfinite(e)]
        def rsd(x: Any) -> float:
            x = np.asarray(x, float)
            x = x[np.isfinite(x)]
            return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else np.nan

        sd_donor = u.groupby("donor")["error_ms"].apply(rsd)
        rows.append({"level": lv, "level_db": float(g["level_db"].iloc[0]), "lag_ms": lag,
                     "task": task, "rule": rule,
                     # J22.3: per donor first, then the median over donors
                     "usable_share": float(g.groupby("donor")["usable"].mean().median()),
                     "usable_share_pooled": float(g["usable"].mean()),
                     **_cohort(per.to_numpy(float), seed),
                     "sd_robust": float(sd_donor.median()) if len(sd_donor) else np.nan,
                     "sd_robust_pooled": rsd(e),
                     "sd": float(np.std(e, ddof=1)) if e.size > 1 else np.nan,
                     "n_cycles": int(e.size)})
    return pd.DataFrame(rows)


# -- J19: counts for the Results prose (J19) ------------------------------------------
@_table("mains_share_counts")
def mains_share_counts(ctx: Any) -> pd.DataFrame:
    """Mains share at rest (``qc:emg``, an01): per participant the largest channel value of the
    raw and the cleaned share; counts of participants above 0.5 raw (dominated by mains) and
    above 0.1 raw (not clean), and the largest cleaned share over all participants and
    channels. The mains fit itself runs on every participant (D8). One row."""
    q = ctx.table("qc", "emg")
    per = q.groupby("pid")[["mains_share_raw", "mains_share_clean"]].max()
    return pd.DataFrame([{"n": int(len(per)),
                          "n_raw_above_half": int((per["mains_share_raw"] > 0.5).sum()),
                          "n_raw_above_tenth": int((per["mains_share_raw"] > 0.1).sum()),
                          "raw_max": float(per["mains_share_raw"].max()),
                          "clean_max": float(per["mains_share_clean"].max())}])


@_table("cal_check")
def cal_check(ctx: Any) -> pd.DataFrame:
    """Stereo calibration storage and the indirect check of J21 (one row).

    From ``acquisition_cohort``: the first recording day with a stored calibration
    (``stored_from_day``), the takes that stored one, the distinct calibrations and the takes
    sharing a calibration with another take (``n_shared``). From the audit file
    ``cal_fit_residual_<snap>.parquet`` per body (cra, mta): the range of the take medians of
    the stored rigid-fit residual (mm) for the takes without (``early``) and with (``late``)
    a stored calibration; the cross test on the latter (largest median with the own
    calibration ``own_max``, smallest with a foreign one ``foreign_min``); the largest
    deviation of the recomputed own median from the stored one (``repro_dev_pct``);
    ``separates`` (foreign_min > own_max); and, where it separates, the early takes at or below
    ``own_max`` (``consistent_n``) and above ``foreign_min`` (``flagged_n``)."""
    import hashlib
    from pathlib import Path

    import numpy as np

    ps = ctx.table("acquisition_cohort", "per_session")
    cal = ctx.table("acquisition_cohort", "calibrations")
    st = ps[ps["stored"].fillna(False).astype(bool)]
    shared = cal[cal["participants"].str.contains(",")]
    row: dict[str, Any] = {
        "stored_from_day": int(st["day_index"].min()) if len(st) else np.nan,
        "n_stored": int(len(st)), "n_calibrations": int(len(cal)),
        "n_shared": int(sum(len(x.split(", ")) for x in shared["participants"])),
        "n_early": int(len(ps) - len(st))}
    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"cal_fit_residual_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/cal_fit_residual.py")
    d = pd.read_parquet(f)
    late = set(st["pid"])
    for b in ("cra", "mta"):
        s_ = d[(d.part == "stored") & (d.body == b)].set_index("pid")["median_mm"]
        c = d[(d.part == "cross") & (d.body == b)]
        e = s_[~s_.index.isin(late)]
        l_ = s_[s_.index.isin(late)]
        own = c[c["own"]].set_index("pid")["median_mm"]
        own_max = float(own.max())
        foreign_min = float(c[~c["own"]]["median_mm"].min())
        sep = foreign_min > own_max
        row.update({f"early_min_{b}": float(e.min()), f"early_max_{b}": float(e.max()),
                    f"late_min_{b}": float(l_.min()), f"late_max_{b}": float(l_.max()),
                    f"own_max_{b}": own_max, f"foreign_min_{b}": foreign_min,
                    f"repro_dev_pct_{b}": float((100 * (own / s_[own.index] - 1)).abs().max()),
                    f"separates_{b}": bool(sep),
                    f"consistent_n_{b}": int((e <= own_max).sum()) if sep else np.nan,
                    f"flagged_n_{b}": int((e > foreign_min).sum()) if sep else np.nan})
    return pd.DataFrame([row]).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


@_table("incisal_speed")
def incisal_speed(ctx: Any) -> pd.DataFrame:
    """J22.1: peak incisal speed per cycle (``kinematics:cycles``, ``ok`` cycles), per task
    (open-close, chewing). Per participant the median and the largest cycle peak; (a) cohort
    median with 95 % BCa CI of the participant medians (``median``, ``ci_lo``, ``ci_hi``,
    ``n``); (b) the S1 design block of the participant maxima (``design`` = Harrell-Davis P95,
    ``design_lower`` = exact one-sided lower 95 % bound X(``lower_k``), ``design_max``);
    ``pooled_median`` over all cycles for comparison."""
    import numpy as np

    from biomech.stats.requirements import p95_design

    seed = int(ctx.config.seed or 0)
    c = ctx.table("kinematics", "cycles")
    c = c[c["ok"].astype(bool)]
    rows = []
    for task in ("open_close", "chewing"):
        g = c[c["task"] == task].dropna(subset=["peak_incisal_speed_mm_s"])
        per = g.groupby("pid")["peak_incisal_speed_mm_s"]
        d = p95_design(per.max().to_numpy(float))
        rows.append({"task": task, **_cohort(per.median().to_numpy(float), seed),
                     "design": d["value"], "design_lower": d["ci_lo"],
                     "lower_k": d["lower_k"], "design_max": d["sample_max"],
                     "design_n": d["n"], "n_cycles": int(len(g)),
                     "cycles_min": int(per.size().min()), "cycles_max": int(per.size().max()),
                     "pooled_median": float(g["peak_incisal_speed_mm_s"].median())})
    return pd.DataFrame(rows)


@_table("emg_sides")
def emg_sides(ctx: Any) -> pd.DataFrame:
    """J22.2: per muscle (masseter, temporalis, digastric) the participant value = mean of the
    left and right channel (``qc:emg``) of the MVC level over the largest task activity
    (``mvc_over_task``) and of the SNR (``snr_db``); median over participants (n) and, for
    comparison, the median pooled over the channels."""
    q = ctx.table("qc", "emg")
    rows = []
    for m in ("masseter", "temporalis", "digastric"):
        g = q[q["muscle"].isin([f"{m}_l", f"{m}_r"])]
        row: dict[str, Any] = {"muscle": m}
        for col in ("mvc_over_task", "snr_db"):
            per = g.groupby("pid")[col].mean().dropna()
            row.update({f"{col}_median": float(per.median()), f"{col}_n": int(per.size),
                        f"{col}_pooled_median": float(g[col].median())})
        rows.append(row)
    return pd.DataFrame(rows)


@_table("label_kappa_participants")
def label_kappa_participants(ctx: Any) -> pd.DataFrame:
    """J22.4: label agreement per participant first (``labels_check:kappa``, fixed thresholds):
    per phase the median, smallest and largest of the participants' kappas (level
    ``participant_phase``), and ``phase = without_rest``: the participants' kappas without the
    rest phase (level ``participant``)."""
    k = ctx.table("labels_check", "kappa")
    k = k[k["variant"] == "fixed"]
    rows = []
    pp = k[k["level"] == "participant_phase"]
    for ph, g in pp.groupby("phase"):
        v = g["kappa"].dropna()
        rows.append({"phase": ph, "median": float(v.median()), "min": float(v.min()),
                     "max": float(v.max()), "n": int(v.size)})
    v = k[(k["level"] == "participant") & (k["scope"] == "without_rest")]["kappa"].dropna()
    rows.append({"phase": "without_rest", "median": float(v.median()), "min": float(v.min()),
                 "max": float(v.max()), "n": int(v.size)})
    return pd.DataFrame(rows)


@_table("dyncal_participants")
def dyncal_participants(ctx: Any) -> pd.DataFrame:
    """J23: the IMU recalibration check per participant first. From
    ``imu_dyncal:dyncal_transitions`` per participant the median |residual jump| (deg) of its
    phase transitions after >= ``STILL_MIN_S`` of stillness (``still``) and of the others
    (``other``); participants with both enter. Paired difference still - other: median over
    participants, Hodges-Lehmann estimate with exact 95 % CI and two-sided signed-rank p
    (``stats.families.paired_contrast``); the medians of the two participant values; transition
    counts per participant (range). One row."""
    import numpy as np

    from biomech.imu.stage_dyncal import STILL_MIN_S
    from biomech.stats.families import paired_contrast

    tr = ctx.table("imu_dyncal", "dyncal_transitions")
    tr = tr.assign(r=tr["residual_deg"].abs(), still=tr["still_in_gap_s"] >= STILL_MIN_S)
    tr = tr.dropna(subset=["r"])
    g = tr.groupby(["pid", "still"])["r"]
    med = g.median().unstack("still")
    cnt = g.size().unstack("still")
    both = med.dropna(subset=[True, False]) if {True, False} <= set(med.columns) else med.iloc[:0]
    d = (both[True] - both[False]).to_numpy(float)
    c = paired_contrast(d)
    return pd.DataFrame([{
        "still_min_s": STILL_MIN_S, "n": int(len(both)), "n_recorded": int(tr["pid"].nunique()),
        "still_median_deg": float(both[True].median()) if len(both) else np.nan,
        "other_median_deg": float(both[False].median()) if len(both) else np.nan,
        "diff_median_deg": float(np.median(d)) if d.size else np.nan,
        "hl_deg": c["hl"], "hl_ci_lo": c["hl_ci_lo"], "hl_ci_hi": c["hl_ci_hi"],
        "p": c["p"], "method": c["method"],
        "n_still_min": int(cnt.loc[both.index, True].min()) if len(both) else 0,
        "n_still_max": int(cnt.loc[both.index, True].max()) if len(both) else 0,
        "n_other_min": int(cnt.loc[both.index, False].min()) if len(both) else 0,
        "n_other_max": int(cnt.loc[both.index, False].max()) if len(both) else 0}])


@_table("ml_contrasts")
def ml_contrasts(ctx: Any) -> pd.DataFrame:
    """J24: effect sizes of the C3b paired tests. Per participant the difference of the
    SVM balanced accuracy EMG + IMU - ``other``: leave-one-subject-out (``ml_eval:loso``,
    other = emg, emg_nodig, emg_nodig_imu, imu; ``protocol`` loso) and after calibration
    (``ml_eval:variants`` few_shot, other = imu, per ``k``; ``protocol`` few_shot);
    Hodges-Lehmann estimate with exact 95 % CI and the two-sided signed-rank p
    (``stats.families.paired_contrast``), the stored test's ``p`` and ``p_holm``
    (``ml_eval:tests``), n and the participants better with EMG + IMU."""
    import numpy as np

    from biomech.stats.families import paired_contrast

    loso = ctx.table("ml_eval", "loso")
    tests = ctx.table("ml_eval", "tests").set_index("test")
    rows = []

    def add(proto: str, other: str, k: Any, d: np.ndarray, test: str) -> None:
        c = paired_contrast(d)
        t = tests.loc[test] if test in tests.index else None
        rows.append({"protocol": proto, "other": other, "k": k, "n": int(np.isfinite(d).sum()),
                     "n_better": int((d > 0).sum()), "median_diff": float(np.nanmedian(d)),
                     "hl": c["hl"], "hl_ci_lo": c["hl_ci_lo"], "hl_ci_hi": c["hl_ci_hi"],
                     "p_recomputed": c["p"],
                     "p": float(t["p"]) if t is not None else np.nan,
                     "p_holm": float(t["p_holm"]) if t is not None else np.nan})

    w = loso[loso["model"] == "svm"].pivot(index="pid", columns="feature_set",
                                           values="balanced_accuracy")
    for other in ("emg", "emg_nodig", "emg_nodig_imu", "imu"):
        if "emg_imu" in w and other in w:
            d = (w["emg_imu"] - w[other]).dropna().to_numpy(float)
            add("loso", other, np.nan, d, f"wilcoxon|svm|emg_imu_vs_{other}")
    v = ctx.table("ml_eval", "variants")
    f = v[v["variant"] == "few_shot"].pivot_table(index="pid", columns=["feature_set", "k"],
                                                  values="balanced_accuracy")
    for k in sorted({int(c[1]) for c in f.columns}):
        if ("emg_imu", k) in f and ("imu", k) in f:
            d = (f[("emg_imu", k)] - f[("imu", k)]).dropna().to_numpy(float)
            add("few_shot", "imu", k, d, f"fewshot|svm|emg_imu_vs_imu|k={k}")
    return pd.DataFrame(rows)


def _j27_file(ctx: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
    """``<store>/audit/j27_h2_power_<snap>.parquet`` and its json (J27)."""
    import hashlib
    import json
    from pathlib import Path

    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"j27_h2_power_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/j27_h2_power.py")
    meta = json.loads(f.with_suffix(".json").read_text())
    return (pd.read_parquet(f).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest()),
            meta)


@_table("h2_power")
def h2_power(ctx: Any) -> pd.DataFrame:
    """J27 / J27a: sample size of a future confirmatory test of the closer release
    (open-close H2 at alpha 0.05; chewing H3 under Holm over the two tasks), by resampling the
    current usable participants. One row per task, variant (``scale``: the pre-specified shrink,
    invariant for the signed-rank test; ``shift_loc``; ``shift_both``), shrink and test
    (combined, locking, location): ``n80`` / ``n90`` = smallest simulated n (usable
    participants, grid 6-30) with power >= 0.80 / 0.90 (NaN: none up to 30; ``at_grid_min``:
    reached at the smallest n simulated, so n may be smaller), the recruitment targets
    ``recruit80`` / ``recruit90`` = ceil(n / usable share), the power at the current usable n
    (``power_now``), the usable participants (``n_usable_pool``) of ``n_participants``, and the
    minimum usable channel-cycles per participant (``min_usable``)."""
    import math

    import numpy as np

    d, meta = _j27_file(ctx)
    rows = []
    for (task, var, s, test), g in d.groupby(["task", "variant", "shrink", "test"]):
        g = g.sort_values("n")
        pool = int(meta["pools"][task]["n"])
        share = pool / float(meta["n_participants"])

        def first(thr: float) -> float:
            hit = g[g["power"] >= thr]
            return float(hit["n"].iloc[0]) if len(hit) else np.nan

        n80, n90 = first(0.8), first(0.9)
        now = g[g["n"] == pool]["power"]
        rows.append({"task": task, "variant": var, "shrink": float(s), "test": test,
                     "n80": n80, "n90": n90,
                     "at_grid_min80": bool(n80 == g["n"].min()),
                     "at_grid_min90": bool(n90 == g["n"].min()),
                     "recruit80": float(math.ceil(n80 / share)) if np.isfinite(n80) else np.nan,
                     "recruit90": float(math.ceil(n90 / share)) if np.isfinite(n90) else np.nan,
                     "power_now": float(now.iloc[0]) if len(now) else np.nan,
                     "n_usable_pool": pool, "n_participants": int(meta["n_participants"]),
                     "usable_share": share, "min_usable": int(meta["min_usable"]),
                     "n_usable_cycles_min": int(meta["pools"][task]["n_usable_min"]),
                     "n_usable_cycles_median": float(meta["pools"][task]["n_usable_median"]),
                     "n_rep": int(meta["n_rep"]), "delta_ms": float(meta["delta_ms"])})
    return pd.DataFrame(rows)


# -- J28: numbers requested in the JNER revision (J28) --------------------------------
@_table("data_glance")
def data_glance(ctx: Any) -> pd.DataFrame:
    """J28.1 the data set at a glance (one row): motion and MVC takes of the participants,
    recording minutes of the motion takes (span of the optical time base: sum, median, range),
    valid cycles per task (``ok`` cycles of ``kinematics:cycles``, per participant; median,
    range), usable digastric opening channel-cycles of C2 (TKEO, first-onset rule, both tasks;
    count, total, share), samples (optical frames over cameras and takes from
    ``acquisition_cohort:summary``; IMU orientation and EMG rows read from the session files
    of both takes), the size of those session folders (GB) and of the store's result bundle
    (MB)."""
    from pathlib import Path

    import h5py
    import numpy as np

    from biomech.io.session import OPENERS

    pids = list(ctx.pids())
    row: dict[str, Any] = {"n_participants": len(pids)}
    takes, minutes, imu_q, emg_rows, size = 0, [], 0, 0, 0
    for p in pids:
        part = ctx.cohort.get(p)
        for kind in ("motion", "mvc"):
            name = getattr(part, kind, None)
            if not name:
                continue
            d = Path(ctx.cohort.session_path(name))
            if not (d / "session.h5").exists():
                continue
            takes += 1
            size += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            with h5py.File(d / "session.h5", "r") as f:
                if "emg/t_master_us" in f:
                    emg_rows += int(f["emg/t_master_us"].shape[0])
                for slot in ("imu_mta", "imu_cra"):
                    if f"{slot}/quat" in f:
                        imu_q += int(f[f"{slot}/quat"].shape[0])
                if kind == "motion" and "optical/t_master_us" in f:
                    t = np.asarray(f["optical/t_master_us"][()], float)
                    minutes.append((t[-1] - t[0]) / 6e7 if t.size > 1 else np.nan)
    m = np.asarray(minutes, float)
    row.update({"n_takes": takes, "minutes_total": float(np.nansum(m)),
                "minutes_median": float(np.nanmedian(m)), "minutes_min": float(np.nanmin(m)),
                "minutes_max": float(np.nanmax(m)), "imu_quat_samples": imu_q,
                "emg_samples": emg_rows, "session_gb": size / 1e9})
    acq = ctx.table("acquisition_cohort", "summary")
    row["optical_frames"] = int(acq.frames_total.iloc[0]) if "frames_total" in acq else np.nan
    cyc = ctx.table("kinematics", "cycles")
    cyc = cyc[cyc["ok"].astype(bool)]
    for task in ("open_close", "protrusion_retrusion", "left_right", "chewing"):
        per = cyc[cyc.task == task].groupby("pid").size().reindex(pids).fillna(0)
        row.update({f"cycles_{task}_median": float(per.median()),
                    f"cycles_{task}_min": int(per.min()), f"cycles_{task}_max": int(per.max())})
    on = ctx.table("timing", "onsets")
    o = on[(on.detector == "tkeo") & (on.rule == "first") & (on.direction == "opening")
           & on.muscle.isin(OPENERS)]
    row.update({"c2_channel_cycles": int(len(o)), "c2_usable": int(o.usable.astype(bool).sum()),
                "c2_usable_share": float(o.usable.astype(bool).mean())})
    snap = getattr(ctx.config, "snapshot", None)
    b = Path(ctx.config.out_dir) / "snapshots" / str(snap) / f"bundle_{snap}.h5" if snap else None
    if b is None or not b.exists():
        b = Path(ctx.config.out_dir) / f"bundle_{snap}.h5"
    row["bundle_mb"] = b.stat().st_size / 1e6 if b.exists() else np.nan
    return pd.DataFrame([row])


@_table("spm_summary")
def spm_summary(ctx: Any) -> pd.DataFrame:
    """J28.2 the two pre-specified SPM tests (``stats_emg:spm``, family confirmatory, Holm over
    both): significant clusters, their extent in % of the cycle (sum of end - start) and the
    Holm p. One row per test."""
    import ast

    sp = ctx.table("stats_emg", "spm")
    sp = sp[sp.family == "confirmatory"]
    rows = []
    for _, r in sp.iterrows():
        cl = r.clusters
        cl = ast.literal_eval(cl) if isinstance(cl, str) else list(cl if cl is not None else [])
        cl = [tuple(c) for c in cl]
        rows.append({"test": r.test, "n": int(r.n), "reject": bool(r.h0reject),
                     "n_clusters": len(cl),
                     "extent_pct": float(sum(float(c[1]) - float(c[0]) for c in cl)),
                     "clusters_text": ", ".join(f"{float(c[0]):.0f} to {float(c[1]):.0f} %"
                                                for c in cl) or "none",
                     "p_holm": float(r.p_holm)})
    return pd.DataFrame(rows)


@_table("condyle_lr")
def condyle_lr(ctx: Any) -> pd.DataFrame:
    """J28.4 left against right condylar translation: per participant |left - right| of the
    condyle-point excursion in opening (``stats_kinematics:subjects``); median with 95 % BCa
    CI and the S1 design block (Harrell-Davis P95, exact lower bound, largest), for all
    participants and those with plausible condyle points."""
    import numpy as np

    from biomech.stats.requirements import p95_design

    seed = int(ctx.config.seed or 0)
    s = ctx.table("stats_kinematics", "subjects").drop_duplicates("pid")
    d = (pd.to_numeric(s.condyle_left_disp_mm, errors="coerce")
         - pd.to_numeric(s.condyle_right_disp_mm, errors="coerce")).abs()
    plaus = pd.to_numeric(s.condyles_plausible, errors="coerce").eq(1)
    rows = []
    for grp, sel in (("all", np.ones(len(s), bool)), ("plausible", plaus.to_numpy())):
        v = d[sel].dropna().to_numpy(float)
        dd = p95_design(v)
        rows.append({"group": grp, **_cohort(v, seed), "p95": dd["value"],
                     "p95_lower": dd["ci_lo"], "max": dd["sample_max"]})
    return pd.DataFrame(rows)


@_table("s2_cycles")
def s2_cycles(ctx: Any) -> pd.DataFrame:
    """J28.6 cycles needed for the target CI half-width (``variability:cycles_needed``): per
    task, metric and side the median and largest over participants, with the participants'
    available cycles (median)."""
    c = ctx.table("variability", "cycles_needed")
    g = c.groupby(["task", "metric", "side"], dropna=False)
    return g.agg(n=("pid", "nunique"), needed_median=("n_needed", "median"),
                 needed_max=("n_needed", "max"),
                 available_median=("n_available", "median"),
                 target_halfwidth=("target_halfwidth", "first")).reset_index()


@_table("axis_location_counts")
def axis_location_counts(ctx: Any) -> pd.DataFrame:
    """Where the best-fit (compromise) axis lies relative to the palpated condyle midpoint
    (``stats_kinematics:subjects``; maxillary frame, x anterior, z up): participants behind
    (dx < 0) and below (dz < 0), all and with plausible condyle points. One row per group."""
    s = ctx.table("stats_kinematics", "subjects").drop_duplicates("pid")
    s = s.dropna(subset=["hinge_axis_dx_mm", "hinge_axis_dz_mm"])
    rows = []
    for group, g in (("all", s), ("plausible_condyles",
                                  s[s["condyles_plausible"].astype(float) > 0])):
        rows.append({"group": group, "n": int(len(g)),
                     "n_behind": int((g["hinge_axis_dx_mm"] < 0).sum()),
                     "n_below": int((g["hinge_axis_dz_mm"] < 0).sum())})
    return pd.DataFrame(rows)


@_table("robustness_counts")
def robustness_counts(ctx: Any) -> pd.DataFrame:
    """From the robustness table itself (``jnl_table_robustness``, run_matrix): the number of
    sensitivity runs and of runs whose decision changed in at least one of the four C2 lead
    columns (a cell marked "*"), and in any column. One row."""
    from biomech.viz.views import paper_journal as PJ
    from biomech.viz.views.registry import ViewRequest

    t = PJ.jnl_table_robustness(ViewRequest(ctx=ctx, options={"form": "run_matrix"}))
    runs = [r for r in t.rows if not r.get("_rule") and "run" in r
            and not str(r["run"]).startswith("Primary")]
    lead = ("o0", "o1", "o2", "o3")
    marked = lambda r, keys: any(str(r.get(k, "")).rstrip().endswith("*") for k in keys)
    allk = [c.key for c in t.columns if c.key != "run"]
    return pd.DataFrame([{"n_runs": len(runs),
                          "n_lead_changed": sum(marked(r, lead) for r in runs),
                          "n_any_changed": sum(marked(r, allk) for r in runs)}])


def _j29_file(ctx: Any) -> pd.DataFrame:
    """``<store>/audit/j29_hinge_small_<snap>.parquet`` (J29), with its sha256."""
    import hashlib
    from pathlib import Path

    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"j29_hinge_small_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/j29_hinge_small.py")
    return pd.read_parquet(f).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


@_table("hinge_small")
def hinge_small(ctx: Any) -> pd.DataFrame:
    """J29 (exploratory, post hoc): the hinge at the joint over the first part of the
    opening. One row per primary level (``level_key`` mm10, mm20, deg12): participants tested
    and above the P99 of their in-vivo null on the same truncated openings, the median ratio
    residual / P99 with its 95 % BCa CI (study seed), the one-sided exact Wilcoxon of the log
    ratio > 0 and its Holm p over the three levels, the median residual and P99 (mm). One row
    ``detect``: the first incisal opening (2-mm grid) at which a participant's residual
    exceeds its P99 -- median over participants with censored participants (no exceedance up
    to their largest opening) counted as larger than every detected value (NaN when half or
    more are censored), the range of the detected values, the counts detected and censored."""
    import numpy as np

    from biomech.stats.core import describe, holm
    from biomech.stats.families import wilcoxon_greater

    d = _j29_file(ctx)
    seed = ctx.config.seed
    rows = []
    for kind, lv, key in (("mm", 10.0, "mm10"), ("mm", 20.0, "mm20"), ("deg", 12.0, "deg12")):
        g = d[(d.purpose == "primary") & (d.kind == kind) & (d.level == lv)]
        t = g[g.tested.astype(bool)]
        ratio = pd.to_numeric(t.ratio, errors="coerce").to_numpy(float)
        ok = np.isfinite(ratio) & (ratio > 0)
        w, pw, exact = wilcoxon_greater(np.log(ratio[ok]))
        ds = describe(ratio, seed=seed, stat="median")
        rows.append({"level_key": key, "kind": kind, "level": lv, "n_tested": int(len(t)),
                     "n_untested": int(len(g) - len(t)),
                     "above_n": int(t.above.astype(bool).sum()),
                     "ratio_median": ds["median"], "ratio_ci_lo": ds["ci_lo"],
                     "ratio_ci_hi": ds["ci_hi"], "wilcoxon_w": w, "p": pw,
                     "exact": bool(exact),
                     "obs_median_mm": float(np.median(t.obs_mm)) if len(t) else np.nan,
                     "p99_median_mm": float(np.median(t.p99_mm)) if len(t) else np.nan,
                     "n_openings_median": float(np.median(t.n_openings)) if len(t)
                     else np.nan})
    ph = holm([r["p"] for r in rows])
    for r, q in zip(rows, ph):
        r["p_holm"] = q
    det = d[d.purpose == "detect"]
    first, cens = [], []
    for pid, g in det.groupby("pid"):
        g = g.sort_values("level")
        hit = g[g.above.fillna(False).astype(bool)]
        (first if len(hit) else cens).append(float(hit.level.iloc[0]) if len(hit)
                                             else float(g.level.max()))
    n = len(first) + len(cens)
    vals = np.sort(np.r_[first, np.full(len(cens), np.inf)])
    med = float(np.median(vals)) if n and len(cens) < n / 2 else np.nan
    rows.append({"level_key": "detect", "kind": "mm", "level": np.nan, "n_tested": n,
                 "above_n": len(first), "n_censored": len(cens), "detect_median_mm": med,
                 "detect_min_mm": float(min(first)) if first else np.nan,
                 "detect_max_mm": float(max(first)) if first else np.nan,
                 "censored_at_min_mm": float(min(cens)) if cens else np.nan})
    return pd.DataFrame(rows).assign(sha256=d.sha256.iloc[0])


@_table("cf_lever")
def cf_lever(ctx: Any) -> pd.DataFrame:
    """Per participant (plan "X5 lever", 2026-11-12, post hoc, descriptive): the closed-pose
    distance from the lower incisor to the condyle midpoint (``cf_summary.scale_mm``), the
    rotation that alone uses up the incisor bound (bound / distance, small-angle, degrees) and
    the rotation RMSE of the primary step about each maxillary axis (median over the scored
    tasks of the per-task RMSE)."""
    from biomech.stats.cohort_camerafree import BOUNDS

    s = ctx.table("camerafree", "cf_summary")
    r = ctx.table("camerafree", "cf_rotation")
    out = []
    for _, row in s.iterrows():
        pid, ell = row["pid"], float(row["scale_mm"])
        prim = str(row.get("primary_e3") or "E3b")
        d = r[(r["pid"] == pid) & (r["estimator"] == prim) & r["scored"].astype(bool)
              & (r["task"] != "scored")]
        out.append({"pid": pid, "primary_e3": prim, "lever_mm": ell,
                    "bound_inc_mm": float(BOUNDS["rmse_inc_mm"]),
                    "bound_angle_deg": float(np.degrees(BOUNDS["rmse_inc_mm"] / ell))
                    if np.isfinite(ell) and ell > 0 else np.nan,
                    **{f"rmse_{a}_deg": float(d[f"rmse_{a}_deg"].median()) if len(d) else np.nan
                       for a in ("rx", "ry", "rz")}})
    return pd.DataFrame(out)



# -- J34: the C2 start gate and the decoding chance level ----------
def _j34_file(ctx: Any, name: str) -> pd.DataFrame:
    """``<store>/audit/j34_<name>_<snap>.parquet`` (scripts/audit_recompute/j34_*.py), with its
    sha256."""
    import hashlib
    from pathlib import Path

    snap = getattr(ctx.config, "snapshot", None) or "live"
    f = Path(ctx.config.out_dir) / "audit" / f"j34_{name}_{snap}.parquet"
    if not f.exists():
        raise FileNotFoundError(f"{f} missing: run scripts/audit_recompute/j34_*.py")
    return pd.read_parquet(f).assign(sha256=hashlib.sha256(f.read_bytes()).hexdigest())


@_table("c2_gate_flow")
def c2_gate_flow(ctx: Any) -> pd.DataFrame:
    """J34.1: the C2 channel-cycles (TKEO, rule first, opening, openers, both tasks)
    through the timing gates in the order of ``emg_usability_flow`` (active at the window
    start, burst below ``burst_min_db``, no onset, masked samples), each counted once at the
    first gate it fails. One row: the total and, per gate, the count and its share of the
    total."""
    from biomech.viz.views.emg_r8 import _onset_rows, _reason, _tp

    rows = _onset_rows(ctx, None)
    rows = rows[(rows.direction == "opening")]
    if rows.empty:
        return pd.DataFrame()
    burst_min = float(_tp(ctx, "burst_min_db", default=6.0))
    why = [_reason(r, "first", burst_min) for r in rows.itertuples()]
    why = ["burst below" if w.startswith("burst below") else w for w in why]
    n = len(why)
    out: dict[str, Any] = {"n": n}
    for key, label in (("usable", "usable"), ("active_start", "active at window start"),
                       ("burst", "burst below"), ("no_onset", "no onset found"),
                       ("masked", "masked samples in the window")):
        k = sum(w == label for w in why)
        out[key] = k
        out[f"{key}_share"] = k / n
    return pd.DataFrame([out])


@_table("c2_baseline")
def c2_baseline(ctx: Any) -> pd.DataFrame:
    """J34.2: the openers' in-task baseline against the rest threshold
    (``audit/j34_c2_baseline``): per participant the median over the opener channel-cycles of
    10·log10(median TKEO detector signal over the window's first 0.3 s / rest threshold); one
    row with the median and range over participants (n participants, channel-cycles)."""
    d = _j34_file(ctx, "c2_baseline")
    per = d.groupby("pid").taskbase_db.median().dropna()
    return pd.DataFrame([{"n": int(per.size), "n_channel_cycles": int(len(d)),
                          "median_db": float(per.median()), "min_db": float(per.min()),
                          "max_db": float(per.max()), "sha256": d.sha256.iloc[0]}])


@_table("c2_taskbase_leads")
def c2_taskbase_leads(ctx: Any) -> pd.DataFrame:
    """J34.3: the opening leads of the in-task threshold rule (``drive_taskbase``) over
    the detector × mechanical-source variants of ``stats_timing:timing_spec`` (the values of
    ``timing_detectors``): per task the cohort median of each variant; one row per task with
    the number of variants, the largest and smallest median, the number above zero and the
    smallest and largest number of participants per variant."""
    t = ctx.table("stats_timing", "timing_spec")
    rows = []
    for task in ("open_close", "chewing"):
        x = t[(t.outcome == f"{task}|opening|lead")
              & t.variant.str.endswith("onset=drive_taskbase")]
        if x.empty:
            continue
        g = x.dropna(subset=["value"]).groupby("variant").agg(
            n=("pid", "nunique"), med=("value", "median"))
        rows.append({"task": task, "n_variants": int(len(g)), "median_max": float(g.med.max()),
                     "median_min": float(g.med.min()), "above_zero": int((g.med > 0).sum()),
                     "n_min": int(g.n.min()), "n_max": int(g.n.max())})
    return pd.DataFrame(rows)


@_table("label_shift")
def label_shift(ctx: Any) -> pd.DataFrame:
    """J34.4 / J34.4b / J34.4c: why the label-shift null of the C3b permutation test
    lies above 1/7 (``audit/j34_label_shift`` and ``audit/j34_shift_ceiling``). One row: the
    mean over the permutations of the shifted labels' balanced accuracy against the true labels
    (``align_mean``) and of the position decoder (``pos_mean``), their Spearman correlation
    with the stored null value, and the mean and range over participants of the expected
    balanced accuracy of the best class per window under the restricted shift
    (``ceiling_*``)."""
    from scipy import stats as sps

    d = _j34_file(ctx, "label_shift")
    c = _j34_file(ctx, "shift_ceiling")
    out: dict[str, Any] = {"n_perm": int(len(d)), "null_mean": float(d["null"].mean())}
    for col, key in (("align_ba", "align"), ("pos_ba", "pos")):
        r = sps.spearmanr(d[col], d["null"])
        out.update({f"{key}_mean": float(d[col].mean()), f"{key}_rho": float(r.statistic),
                    f"{key}_p": float(r.pvalue)})
    out.update({"n": int(len(c)), "ceiling_mean": float(c.ceiling_ba.mean()),
                "ceiling_min": float(c.ceiling_ba.min()),
                "ceiling_max": float(c.ceiling_ba.max()), "sha256": d.sha256.iloc[0],
                "ceiling_sha256": c.sha256.iloc[0]})
    return pd.DataFrame([out])
