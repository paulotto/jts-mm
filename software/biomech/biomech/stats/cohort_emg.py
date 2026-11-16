# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cohort statistics for sEMG: activation, indices, co-activation, SPM (an07 §5, §6, §10)."""

from __future__ import annotations

import zlib

from typing import Any

import numpy as np
import pandas as pd


from biomech.io.session import CLOSERS, MUSCLES, OPENERS  # MUSCLES: CH1-6 of the montage
from biomech.stats.core import describe, nan_quiet, holm, mixed_model
from biomech.stats.waveforms import (anova1_participants, cmc, paired_t, paired_t_nonparam,
                                     variance_ratio)


@nan_quiet
def activation_tables(act: pd.DataFrame, refs: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    a = act.merge(refs[["pid", "muscle", "usable_amplitude"]], on=["pid", "muscle"], how="left")
    use = a[a.usable_amplitude.fillna(False)]
    rows = []
    for (task, m), g in use.groupby(["task", "muscle"]):
        rows.append({"task": task, "muscle": m, **describe(g["median_pct"].to_numpy(), seed=seed,
                                                           stat="median")})
    out = {"activation_cohort": pd.DataFrame(rows)}
    lmm_df = use[use.task != "rest"].dropna(subset=["median_pct"])
    if lmm_df.pid.nunique() >= 5:
        formula = "log_pct ~ C(task) * C(muscle)"
        try:
            out["activation_lmm"] = mixed_model(
                lmm_df.assign(log_pct=np.log(lmm_df.median_pct.clip(lower=0.1))),
                formula).assign(status="converged")
        except RuntimeError as exc:
            # A non-converging model must not take every other EMG cohort table with it
            # (E25): one row saying so instead of the fixed-effect table.
            out["activation_lmm"] = pd.DataFrame([{
                "status": "not converged", "formula": formula,
                "n_participants": int(lmm_df.pid.nunique()), "n_obs": int(len(lmm_df)),
                "note": str(exc)[:500]}])
    rest = a[a.task == "rest"]
    rows = []
    for m, g in rest.groupby("muscle"):
        rows.append({"muscle": m, "metric": "rest_rms_uv", **describe(g.rest_rms_uv.to_numpy(),
                                                                      seed=seed, stat="median")})
        ok = g[g.usable_amplitude.fillna(False)]
        rows.append({"muscle": m, "metric": "rest_pct", **describe(ok.median_pct.to_numpy(),
                                                                   seed=seed, stat="median")})
    out["rest_activity"] = pd.DataFrame(rows)
    return out


@nan_quiet
def indices_table(ind: pd.DataFrame, seed: int) -> pd.DataFrame:
    use = ind[ind.usable]
    return pd.DataFrame([{"metric": c, **describe(use[c].to_numpy(), seed=seed)}
                         for c in ("asym_masseter_pct", "asym_temporalis_pct",
                                   "activity_index_pct")])


@nan_quiet
def cci_table(cyc: pd.DataFrame, seed: int) -> pd.DataFrame:
    per = cyc.groupby(["pid", "task"])[["cci_opening", "cci_closing"]].median().reset_index()
    rows = []
    for task, g in per.groupby("task"):
        for c in ("cci_opening", "cci_closing"):
            rows.append({"task": task, "metric": c, **describe(g[c].to_numpy(), seed=seed)})
    return pd.DataFrame(rows)


def _peak_norm(Y: np.ndarray) -> np.ndarray:
    mx = np.nanmax(Y, axis=-1, keepdims=True)
    return Y / np.where(mx > 0, mx, np.nan)


@nan_quiet
def spm_tests(profiles: dict[str, dict[str, np.ndarray]], shape_ok: dict[str, set[str]]
              ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Confirmatory SPM (analysis_plan §5.2) + S2 SPM (§5.3.1-2).

    ``profiles[pid]`` holds ``{task}__pct`` (cycles, 101, 6) and ``{task}__theta_deg``.
    """
    rows: list[dict[str, Any]] = []

    def col(pr: dict[str, np.ndarray], m: str) -> int | None:
        """Index of muscle ``m`` in this participant's profile arrays (by name)."""
        names = [str(n) for n in pr.get("names", MUSCLES)]
        return names.index(m) if m in names else None
    arrays: dict[str, np.ndarray] = {}

    # Test 1: L vs R masseter over the chewing cycle (subject means).
    A, B = [], []
    for pid, pr in profiles.items():
        P = pr.get("chewing__pct")
        il, ir = col(pr, "masseter_l"), col(pr, "masseter_r")
        if P is None or il is None or ir is None \
                or not {"masseter_l", "masseter_r"} <= shape_ok.get(pid, set()):
            continue
        A.append(np.nanmean(P[:, :, il], 0))
        B.append(np.nanmean(P[:, :, ir], 0))
    t1 = paired_t(np.array(A), np.array(B)) if A else {"n": 0, "h0reject": None, "clusters": []}
    arrays["spm1__z"] = np.array(t1.get("z", []))
    arrays["spm1__left"], arrays["spm1__right"] = np.array(A), np.array(B)

    # Test 2: digastric vs elevator timing over the open-close cycle (peak-normalized).
    A, B = [], []
    for pid, pr in profiles.items():
        P = pr.get("open_close__pct")
        ok = shape_ok.get(pid, set())
        op = [i for m in OPENERS if m in ok and (i := col(pr, m)) is not None]
        cl = [i for m in CLOSERS if m in ok and (i := col(pr, m)) is not None]
        if P is None or not op or not cl:
            continue
        A.append(_peak_norm(np.nanmean(np.nanmean(P[:, :, op], 2), 0)))
        B.append(_peak_norm(np.nanmean(np.nanmean(P[:, :, cl], 2), 0)))
    t2 = paired_t(np.array(A), np.array(B)) if A else {"n": 0, "h0reject": None, "clusters": []}
    arrays["spm2__z"] = np.array(t2.get("z", []))
    arrays["spm2__openers"], arrays["spm2__closers"] = np.array(A), np.array(B)

    pmin = [min([c["p"] for c in t["clusters"]], default=1.0) if t.get("h0reject") is not None
            else np.nan for t in (t1, t2)]
    adj = holm(pmin)
    for name, t, pa in (("masseter_L_vs_R_chewing", t1, adj[0]),
                        ("openers_vs_closers_open_close", t2, adj[1])):
        raw = t.get("h0reject")
        # The confirmatory decision uses the Holm-adjusted p across the two tests (E25);
        # the per-test (unadjusted) decision is kept as h0reject_raw.
        rows.append({"family": "confirmatory", "test": name, "n": t["n"],
                     "h0reject": (bool(pa < 0.05) if raw is not None and np.isfinite(pa)
                                  else (None if raw is None else False)),
                     "h0reject_raw": raw, "zstar": t.get("zstar", np.nan),
                     "clusters": str([(round(c["start_pct"]), round(c["end_pct"]),
                                       round(c["p"], 4)) for c in t["clusters"]]),
                     "p_min_cluster": pmin[0 if name.startswith("mass") else 1],
                     "p_holm": pa})

    # S2: inter-participant ANOVA (cycles as repetitions) and single-participant SPM.
    theta = {pid: pr["open_close__theta_deg"] for pid, pr in profiles.items()
             if "open_close__theta_deg" in pr}
    an = anova1_participants(theta)
    rows.append({"family": "variability", "test": "anova1_opening_angle_open_close",
                 "n": an.get("n_groups"), "h0reject": an.get("h0reject"),
                 "zstar": an.get("zstar", np.nan), "supra_frac": an.get("supra_frac", np.nan),
                 # The cluster ranges themselves (not only their count): views draw them
                 # as significance bars.
                 "clusters": str(an.get("clusters", [])),
                 "n_clusters": len(an.get("clusters", []))})
    mas = {pid: pr["chewing__pct"][:, :, col(pr, "masseter_r")] for pid, pr in profiles.items()
           if "chewing__pct" in pr and col(pr, "masseter_r") is not None
           and "masseter_r" in shape_ok.get(pid, set())}
    an2 = anova1_participants(mas)
    rows.append({"family": "variability", "test": "anova1_masseter_r_chewing",
                 "n": an2.get("n_groups"), "h0reject": an2.get("h0reject"),
                 "zstar": an2.get("zstar", np.nan), "supra_frac": an2.get("supra_frac", np.nan),
                 # The cluster ranges themselves (not only their count): views draw them
                 # as significance bars.
                 "clusters": str(an2.get("clusters", [])),
                 "n_clusters": len(an2.get("clusters", []))})
    n_sig = 0
    n_tested = 0
    for pid, pr in profiles.items():
        P = pr.get("chewing__pct")
        il, ir = col(pr, "masseter_l"), col(pr, "masseter_r")
        if P is None or il is None or ir is None \
                or not {"masseter_l", "masseter_r"} <= shape_ok.get(pid, set()):
            continue
        r = paired_t_nonparam(P[:, :, il], P[:, :, ir],
                              seed=zlib.crc32(f"masseter_L_vs_R|{pid}".encode()))
        if r["h0reject"] is None:
            continue
        n_tested += 1
        n_sig += int(r["h0reject"])
        rows.append({"family": "single_participant", "test": f"masseter_L_vs_R_chewing|{pid}",
                     "n": r["n"], "h0reject": r["h0reject"], "zstar": r.get("zstar", np.nan),
                     "clusters": str([(round(c["start_pct"]), round(c["end_pct"]),
                                       round(c["p"], 4)) for c in r["clusters"]])})
    rows.append({"family": "single_participant", "test": "masseter_L_vs_R_chewing|summary",
                 "n": n_tested, "n_significant": n_sig})

    # Variance decomposition and CMC.
    vr = variance_ratio(theta)
    for k, v in vr.items():
        arrays[f"varratio__theta__{k}"] = v
    vr2 = variance_ratio(mas)
    for k, v in vr2.items():
        arrays[f"varratio__masseter_r_chewing__{k}"] = v
    within = [cmc(P) for P in theta.values()]
    between = cmc(np.array([np.nanmean(P, 0) for P in theta.values()])) if theta else np.nan
    rows.append({"family": "variability", "test": "cmc_opening_angle",
                 "cmc_within_median": float(np.nanmedian(within)) if within else np.nan,
                 "cmc_between": between})
    return pd.DataFrame(rows), arrays


@nan_quiet
def spectral_tables(spec: pd.DataFrame, windows: dict[str, dict[str, np.ndarray]],
                    names: dict[str, list[str]], shape_ok: dict[str, set[str]],
                    grid_hz: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Frequency-domain descriptors across the cohort (X1 (change log), exploratory).

    * ``spectral_cohort``: per task x muscle, MNF across shape-usable participants and its
      between-participant SD vs the pooled within-participant SD across 0.5-s windows;
    * ``spectral_spm``: SPM along the frequency axis on the per-window spectral shape
      (log power share per 10-Hz bin): masseter L vs R in chewing (paired t on participant
      means) and the participants ANOVA (where in frequency participants differ);
    * arrays: participant mean shapes and the cohort mean/SD for the views.
    """
    rows = []
    if not spec.empty:
        use = spec[spec.apply(lambda r: r.muscle in shape_ok.get(r.pid, set()), axis=1)]
        for (task, m), g in use.groupby(["task", "muscle"]):
            v = g.mnf_hz.to_numpy(float)
            w = g.mnf_sd_within_hz.to_numpy(float)
            sd_b = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else np.nan
            sd_w = float(np.sqrt(np.nanmean(w ** 2))) if np.isfinite(w).any() else np.nan
            rows.append({"task": task, "muscle": m, "n": int(np.isfinite(v).sum()),
                         "mnf_mean_hz": float(np.nanmean(v)), "mnf_sd_between_hz": sd_b,
                         "mnf_sd_within_hz": sd_w,
                         "between_over_within": sd_b / sd_w if sd_w and sd_w > 0 else np.nan,
                         "mdf_mean_hz": float(np.nanmean(g.mdf_hz))})
    arrays: dict[str, np.ndarray] = {"spectral__grid_hz": np.asarray(grid_hz, float)}
    spm_rows = []

    def col(pid: str, m: str) -> int | None:
        nm = names.get(pid, list(MUSCLES))
        return nm.index(m) if m in nm else None

    for task in ("chewing", "open_close", "mvc_closing"):
        for m in MUSCLES:
            means = []
            for pid, w in windows.items():
                W = w.get(task)
                i = col(pid, m)
                if W is None or i is None or m not in shape_ok.get(pid, set()):
                    continue
                means.append(np.nanmean(W[i], axis=0))
            if means:
                M = np.array(means)
                arrays[f"spectral__{task}__{m}__subjects"] = M
                arrays[f"spectral__{task}__{m}__mean"] = np.nanmean(M, axis=0)
                arrays[f"spectral__{task}__{m}__sd"] = np.nanstd(M, axis=0, ddof=1) \
                    if len(M) > 1 else np.full(M.shape[1], np.nan)
    # L vs R masseter, chewing (paired, participant means).
    A, B = [], []
    for pid, w in windows.items():
        W = w.get("chewing")
        il, ir = col(pid, "masseter_l"), col(pid, "masseter_r")
        if W is None or il is None or ir is None or not {"masseter_l", "masseter_r"} <= \
                shape_ok.get(pid, set()):
            continue
        A.append(np.nanmean(W[il], axis=0))
        B.append(np.nanmean(W[ir], axis=0))
    t1 = paired_t(np.array(A), np.array(B)) if A else {"n": 0, "h0reject": None, "clusters": []}
    spm_rows.append(_spec_row("masseter_L_vs_R_chewing_spectrum", t1, grid_hz))
    # Participants differ? ANOVA with windows as repetitions (masseter R, chewing).
    reps = {}
    for pid, w in windows.items():
        W = w.get("chewing")
        i = col(pid, "masseter_r")
        if W is not None and i is not None and "masseter_r" in shape_ok.get(pid, set()):
            reps[pid] = W[i]
    an = anova1_participants(reps)
    spm_rows.append(_spec_row("anova1_masseter_r_chewing_spectrum", an, grid_hz))
    if "z" in t1:
        arrays["spectral_spm1__z"] = np.asarray(t1["z"], float)
    return pd.DataFrame(rows), pd.DataFrame(spm_rows), arrays


def _spec_row(name: str, res: dict[str, Any], grid_hz: np.ndarray) -> dict[str, Any]:
    """Clusters from node indices to Hz (bin centres)."""
    def hz(i: float) -> float:
        return float(np.interp(i, np.arange(len(grid_hz)), grid_hz))

    cl = [(round(hz(c["start_pct"])), round(hz(c["end_pct"])), round(c["p"], 4))
          for c in res.get("clusters", [])]
    return {"family": "exploratory_spectral", "test": name,
            "n": res.get("n", res.get("n_groups")), "h0reject": res.get("h0reject"),
            "zstar": res.get("zstar", np.nan), "supra_frac": res.get("supra_frac", np.nan),
            "clusters_hz": str(cl)}


def mask_validation_table(mv: pd.DataFrame) -> pd.DataFrame:
    """Cohort view of the low-frequency mask check (W1.3, exploratory): per muscle, the
    share of channel-tasks where masked windows look like clean ones, and the median
    masked − clean difference of the 20-40 Hz share and of the mean frequency."""
    if mv is None or mv.empty:
        return pd.DataFrame()
    d = mv.assign(d_share=mv.share_20_40_masked - mv.share_20_40_clean,
                  d_mnf=mv.mnf_masked_hz - mv.mnf_clean_hz)
    rows = [{"muscle": "all", "n": int(len(d)), "conservative_frac": float(d.conservative.mean()),
             "d_share_median": float(d.d_share.median()), "d_mnf_median_hz": float(d.d_mnf.median())}]
    for m, g in d.groupby("muscle"):
        rows.append({"muscle": m, "n": int(len(g)), "conservative_frac": float(g.conservative.mean()),
                     "d_share_median": float(g.d_share.median()),
                     "d_mnf_median_hz": float(g.d_mnf.median())})
    return pd.DataFrame(rows)
