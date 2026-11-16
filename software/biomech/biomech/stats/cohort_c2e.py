# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""C2-E cohort tests and decision (X6 (change log), post hoc).

Input: the per-participant tables of :mod:`biomech.timing.c2e` (``c2e_null``, ``coupling``),
the reporting population is whatever the store holds (the paper reads the reporting-population
variant store, D35). Every test is a :mod:`biomech.stats.families` building block.

* **H1 (primary)**, closers lead closing, task-pooled (scope ``pooled``, direction closing,
  source omega): intersection-union (B5) of
  A, locking -- one-sided signed-rank test of log(IQR_obs / median IQR_S1) < 0, and
  B, location -- one-sided signed-rank test of mu - delta > 0 (mu = the mixture location;
  delta = the bias margin fixed by the synthetic gate G4), BCa CI of the median mu.
  Participants with < ``min_usable`` usable channel-cycles are left out of the tests and
  counted (``n_flagged``: those with ``flag_usable`` ... ``min_usable`` - 1).
* **H2 / H3** (only if H1 is rejected; Holm at alpha): closer release before opening in
  open_close / chewing (P3): IUT of the release depth against S1 (signed-rank of log(depth /
  median S1 depth) > 0) and the release lead - delta > 0.
* **Post hoc multiplicity sensitivity** (author decision 2026-10-31, after the results):
  ``p_holm_family`` on the H1, H2 and H3 rows = Holm over the three combined p (``p``),
  always computed, also when the gatekeeping stops; :data:`HOLM_FAMILY_NOTE`. The
  pre-specified gatekeeping (``p_adj``, ``rejected``, the decision) is unchanged.
* **Supporting block** (Holm within the block; no claim rests on it): P4 alpha-referenced
  location, P5 coupling lag > 0 (per task), P6 changepoint location, the optical cross-check,
  the symmetric window (median - S1 null > 0; the fallback statistic) and the digastric
  re-test (locking; expected locked with a negative lead).
* **Structure rows** (descriptive): S2 locking of H1 and the release slope b (S2).
* **Decision** ("shown (post hoc)" | "not shown" | "inconclusive (underpowered)" |
  "delta not fixed"): H1 rejected, the BCa lower bound of the median mu > delta, no veto
  row (``supporting_veto``) with a CI entirely below 0, and the synthetic gates passed;
  "inconclusive" when H1 is not rejected and the gate power at 40 ms is below 0.5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from biomech.stats import families as F
from biomech.stats.core import describe, holm, nan_quiet

COLUMNS = ["key", "family", "hypothesis", "scope", "direction", "source", "n", "n_flagged",
           "statistic", "median", "ci_lo", "ci_hi", "p_lock", "p_loc", "p", "p_adj",
           "p_holm_family", "rejected", "veto", "bias_margin_ms", "note"]
# bias_margin_ms: the location margin delta of the C2-E tests (``stages.timing.c2e.delta_ms``;
# column ``delta_ms`` before 2026-11-03, renamed so it no longer shares the edge-bias name)


@dataclass(frozen=True)
class CohortParams:
    """The cohort part of ``stages.timing.c2e``."""

    alpha: float = 0.05
    delta_ms: float | None = None
    min_usable: int = 5
    flag_usable: int = 3
    gates_passed: bool = False
    gate_power_40ms: float | None = None
    s2_min_sd_ms: float | None = None
    supporting_veto: tuple[str, ...] = ("p4_alpha", "p5_open_close", "p5_chewing",
                                        "p6_changepoint", "optical", "symmetric_window")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "CohortParams":
        d = dict(d or {})
        keep = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        if "supporting_veto" in keep:
            keep["supporting_veto"] = tuple(keep["supporting_veto"])
        return cls(**keep)


def _pick(null: pd.DataFrame, scope: str, direction: str, source: str) -> pd.DataFrame:
    if null is None or null.empty or "source" not in null:
        return pd.DataFrame()
    return null[(null.scope == scope) & (null.direction == direction)
                & (null.source == source)]


TASK_WORD = {"open_close": "open-close", "chewing": "chewing"}
HOLM_FAMILY_NOTE = "post hoc sensitivity: Holm over H1–H3 (not the pre-specified procedure)"
# Reader wording (stored cells are shown as they are): no plan codes (S1 / S2 / G2 / G6).
S2_COHORT_NOTE = ("descriptive; cohort p withheld (the cycle-shuffle surrogate is not "
                  "calibrated at cohort level: false-positive rate 0.095 at nominal 0.05 in the "
                  "synthetic check; no power for back-to-back cycles)")


def _row(key: str, family: str, hyp: str, scope: str, direction: str, source: str,
         **kw: Any) -> dict[str, Any]:
    r = {c: np.nan for c in COLUMNS}
    r.update(key=key, family=family, hypothesis=hyp, scope=scope, direction=direction,
             source=source, note="", veto=False, rejected=False)
    r.update(kw)
    return r


def _usable(g: pd.DataFrame, cp: CohortParams) -> tuple[pd.DataFrame, int]:
    n = g.n_usable.fillna(0) if "n_usable" in g else pd.Series(0, index=g.index)
    return g[n >= cp.min_usable], int(((n >= cp.flag_usable) & (n < cp.min_usable)).sum())


def _iut_row(key: str, family: str, hyp: str, g: pd.DataFrame, cp: CohortParams, seed: int,
             lock_col: str, lock_alt: str, loc_col: str, scope: str, direction: str,
             source: str) -> dict[str, Any]:
    """Locking (log ratio, ``lock_alt``) + location (``loc_col`` - delta > 0), IUT."""
    use, nf = _usable(g, cp)
    lock = F.log_ratio_test(use[lock_col].to_numpy(float) if lock_col in use else [],
                            lock_alt, seed, cp.min_usable)
    loc = F.location_test(use[loc_col].to_numpy(float) if loc_col in use else [],
                          cp.delta_ms, seed, cp.min_usable)
    p = F.iut(lock["p"], loc["p"])
    return _row(key, family, hyp, scope, direction, source, n=int(len(use)), n_flagged=nf,
                statistic=f"{loc_col} (median, BCa CI)", median=loc["median"],
                ci_lo=loc["ci_lo"], ci_hi=loc["ci_hi"], p_lock=lock["p"], p_loc=loc["p"],
                p=p, bias_margin_ms=cp.delta_ms,
                note=f"locking: median {lock_col} {lock['lr_median']:.3f} "
                     f"[{lock['lr_ci_lo']:.3f}, {lock['lr_ci_hi']:.3f}]")


@nan_quiet
def c2e_cohort(null: pd.DataFrame, coupling: pd.DataFrame | None, params: dict[str, Any],
               seed: int) -> pd.DataFrame:
    """The ``c2e_cohort`` table (module docstring). ``params`` = ``stages.timing.c2e``."""
    cp = CohortParams.from_dict(params)
    rows: list[dict[str, Any]] = []
    if null is None or null.empty:
        return pd.DataFrame(rows, columns=COLUMNS)
    # -- H1 ---------------------------------------------------------------------------------
    h1 = _iut_row("h1", "primary", "closers lead closing (task-pooled)",
                  _pick(null, "pooled", "closing", "omega"), cp, seed, "log_ratio", "less",
                  "mu_ms", "pooled", "closing", "omega")
    h1_rej = bool(np.isfinite(h1["p"]) and h1["p"] <= cp.alpha)
    h1.update(p_adj=h1["p"], rejected=h1_rej)
    rows.append(h1)
    # -- H2 / H3 (serial gatekeeping) -------------------------------------------------------
    hs = []
    for key, task in (("h2", "open_close"), ("h3", "chewing")):
        g = _pick(null, task, "opening", "release")
        hs.append(_iut_row(key, "secondary", f"closer release before opening ({task})", g, cp,
                           seed, "log_depth_ratio", "greater", "median_ms", task, "opening",
                           "release"))
    _, adj = F.gatekeeping(h1["p"], [r["p"] for r in hs], cp.alpha)
    for r, pa in zip(hs, adj):
        r.update(p_adj=pa, rejected=bool(np.isfinite(pa) and pa <= cp.alpha))
        if not h1_rej:
            r["note"] += "; not tested (H1 not rejected, serial gatekeeping)"
    holm_family([h1, *hs])
    rows += hs
    # -- supporting block -------------------------------------------------------------------
    sup = []
    for key, src, hyp in (("p4_alpha", "alpha15", "closers lead the alpha onset (P4)"),
                          ("p6_changepoint", "cp", "changepoint onsets lead closing (P6)"),
                          ("optical", "opt", "closers lead the optical onset")):
        g, _ = _usable(_pick(null, "pooled", "closing", src), cp)
        loc = F.location_test(g.mu_ms.to_numpy(float) if "mu_ms" in g else [], cp.delta_ms,
                              seed, cp.min_usable)
        sup.append(_row(key, "supporting", hyp, "pooled", "closing", src, n=loc["n"],
                        statistic="mu_ms (median, BCa CI)", median=loc["median"],
                        ci_lo=loc["ci_lo"], ci_hi=loc["ci_hi"], p_loc=loc["p"], p=loc["p"],
                        bias_margin_ms=cp.delta_ms))
    g, _ = _usable(_pick(null, "pooled", "closing", "sym"), cp)
    loc = F.location_test(g.median_minus_null_ms.to_numpy(float)
                          if "median_minus_null_ms" in g else [], 0.0, seed, cp.min_usable)
    sup.append(_row("symmetric_window", "supporting",
                    "symmetric window: median - S1 null > 0 (fallback statistic)", "pooled",
                    "closing", "sym", n=loc["n"], statistic="median_minus_null_ms",
                    median=loc["median"], ci_lo=loc["ci_lo"], ci_hi=loc["ci_hi"],
                    p_loc=loc["p"], p=loc["p"]))
    if coupling is not None and not coupling.empty and "emd_lag_ms" in coupling:
        for task, g in coupling.groupby("task"):
            loc = F.location_test(g.emd_lag_ms.to_numpy(float), 0.0, seed, cp.min_usable)
            sup.append(_row(f"p5_{task}", "supporting", f"EMG-to-alpha lag > 0 ({task}, P5)",
                            task, "closing", "coupling", n=loc["n"],
                            statistic="emd_lag_ms (residualised)", median=loc["median"],
                            ci_lo=loc["ci_lo"], ci_hi=loc["ci_hi"], p_loc=loc["p"],
                            p=loc["p"], note=f"{int((g.p_coupling <= cp.alpha).sum())} with "
                                             f"circular-shift coupling p <= {cp.alpha:g}"))
    g, nf = _usable(_pick(null, "pooled", "opening", "omega"), cp)
    lock = F.log_ratio_test(g.log_ratio.to_numpy(float) if "log_ratio" in g else [], "less",
                            seed, cp.min_usable)
    mu = describe(g.mu_ms.to_numpy(float) if "mu_ms" in g else np.empty(0), seed=seed,
                  stat="median")
    sup.append(_row("digastric", "supporting", "digastric locked to opening (expected "
                    "negative lead)", "pooled", "opening", "omega", n=lock["n"], n_flagged=nf,
                    statistic="mu_ms (median, BCa CI)", median=mu["median"],
                    ci_lo=mu["ci_lo"], ci_hi=mu["ci_hi"], p_lock=lock["p"], p=lock["p"]))
    for r, pa in zip(sup, F.block_holm([r["p"] for r in sup])):
        r["p_adj"] = pa
        r["rejected"] = bool(np.isfinite(pa) and pa <= cp.alpha)
        r["veto"] = bool(r["key"] in cp.supporting_veto and np.isfinite(r["ci_hi"])
                         and r["ci_hi"] < 0)
    rows += sup
    # -- structure (descriptive) ------------------------------------------------------------
    lr = s2_log_ratios(null, cp)
    if lr is not None:
        t = F.log_ratio_test(lr, "less", seed, cp.min_usable)
        # Effect size only: the cohort p is withheld. The synthetic gate showed it is not
        # calibrated (G2: 0.095 at nominal 0.05 over 201 cohorts) and S2 has no power for
        # back-to-back cycles (G6); per-participant S2 values stay in c2e_null.
        rows.append(_row("s2_h1", "structure", "closer locking beyond the rhythm (cycle shuffle)",
                         "per task", "closing", "omega", n=t["n"],
                         statistic="log(IQR / cycle-shuffle IQR)", median=t["lr_median"],
                         ci_lo=t["lr_ci_lo"], ci_hi=t["lr_ci_hi"], p=np.nan,
                         note=S2_COHORT_NOTE))
    for task in ("open_close", "chewing"):
        g = _pick(null, task, "opening", "release")
        if "b_s2" not in g or g.empty:
            continue
        b = describe(g.b_s2.to_numpy(float), seed=seed, stat="median")
        sd = float(np.nanmedian(g.x_sd_ms)) if g.x_sd_ms.notna().any() else np.nan
        ident = (bool(sd >= cp.s2_min_sd_ms) if cp.s2_min_sd_ms is not None
                 and np.isfinite(sd) else np.nan)
        note = f"median SD of the mechanical event minus the anchor {sd:.0f} ms"
        if ident is False:
            note += ("; the cycle shuffle has no power: 'precedes, not distinguishable from "
                     "the end of closing'")
        rows.append(_row(f"s2_release_{task}", "structure",
                         f"release slope b ({TASK_WORD[task]}, cycle shuffle)",
                         task, "opening", "release", n=b["n"], statistic="Theil-Sen b",
                         median=b["median"], ci_lo=b["ci_lo"], ci_hi=b["ci_hi"], p=np.nan,
                         note=note + "; " + S2_COHORT_NOTE))
    rows.append(_row("decision", "decision", decide(h1, sup, cp), "", "", "",
                     bias_margin_ms=cp.delta_ms))
    return pd.DataFrame(rows, columns=COLUMNS)


def holm_family(hyps: list[dict[str, Any]]) -> None:
    """Post hoc sensitivity, in place: ``p_holm_family`` = Holm over the combined ``p`` of
    the H1-H3 rows (NaN kept out of the family), whatever the gatekeeping decided."""
    for r, pa in zip(hyps, holm([r["p"] for r in hyps])):
        r["p_holm_family"] = pa
        r["note"] = f"{r['note']}; {HOLM_FAMILY_NOTE}" if r["note"] else HOLM_FAMILY_NOTE


def s2_log_ratios(null: pd.DataFrame, cp: CohortParams) -> np.ndarray | None:
    """Per participant, the mean over its closing tasks of log(IQR / S2 IQR) (locking beyond the
    rhythm; usable participants only); None without S2 columns. The cohort row reports their
    median and CI only (``S2_COHORT_NOTE``); the gate test checks the withheld cohort p."""
    s2 = pd.concat([_pick(null, "open_close", "closing", "omega"),
                    _pick(null, "chewing", "closing", "omega")])
    if "iqr_s2_median" not in s2 or s2.empty:
        return None
    s2, _ = _usable(s2, cp)
    return (np.log(s2.iqr_obs.astype(float) / s2.iqr_s2_median.astype(float))
            .groupby(s2.pid).mean().to_numpy(float))


def decide(h1: dict[str, Any], supporting: list[dict[str, Any]], cp: CohortParams) -> str:
    """The X6 decision rule (module docstring)."""
    if cp.delta_ms is None or not np.isfinite(cp.delta_ms):
        return "delta not fixed"
    veto = any(r["veto"] for r in supporting)
    if (h1["rejected"] and np.isfinite(h1["ci_lo"]) and h1["ci_lo"] > cp.delta_ms
            and not veto and cp.gates_passed):
        return "shown (post hoc)"
    if not h1["rejected"] and cp.gate_power_40ms is not None and cp.gate_power_40ms < 0.5:
        return "inconclusive (underpowered)"
    return "not shown"
