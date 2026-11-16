# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""C2 shift null without period-multiple shifts (J17; robustness of D23a).

The D23a chance null of the lead (``timing.lead.pseudo_onset_null``) shifts the EMG
circularly by 200 shifts uniform on [1 s, T - 1 s]. A shift near a multiple of the cycle
period can put the bursts back next to the movements. Here the same null is recomputed with
those shifts rejected: a shift s is excluded when k = round(s / P) >= 1 and
|s / P - k| <= ``--band`` (P = the participant's median start-to-start interval of the ok
kinematics cycles of the task); the same seeded generator draws until 200 are accepted.
Everything else is the unchanged pipeline (``timing.lead.analyse`` on the stored inputs of
the pinned store; the X2 ensemble and X3 release parts are skipped, they do not touch the
null). The cohort rows come from ``stats.cohort_misc.timing_tables`` on the stored
``timing:summary`` / ``onsets`` with the recomputed null.

Validation: ``--band 0`` must reproduce the stored ``timing:lead_null`` and the null columns
of ``stats_timing:timing_cohort`` exactly (the script checks it and stops otherwise).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/c2_null_period.py [-j 4]

Output: ``<reporting_population store>/audit/c2_null_period_<snapshot>.parquet`` (one row per band,
task, direction, detector, rule of ``timing_cohort``'s ``lead_ms`` rows: null columns, plus
the median observed − null with its 95 % BCa CI ``mn_*`` from the per-participant pairs),
``..._participants.parquet`` (the recomputed per-participant null rows) and ``..._desc.parquet``
(per participant, task and band: median period, interval SD / median, the share of the shifts
within ±10 % of a period multiple, the share of surrogate windows wrapping the phase end).
Excluded shifts are redrawn: the first batch equals the primary's 200 shifts, excluded ones
are dropped and the same generator draws on until 200 are accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import _pins  # noqa: E402 -- the pinned snapshots (script directory on sys.path)

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
TASKS = ("open_close", "chewing")
NULL_COLS = ["lead_null_ms", "lead_minus_null_ms", "n_null_pairs", "W_null", "p_null",
             "r_null", "p_null_perm", "p_null_holm", "p_null_perm_holm"]
KEYS = ["task", "direction", "detector", "rule"]


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


def median_period_s(cycles: pd.DataFrame, task: str) -> float:
    """Median start-to-start interval (s) of the ok cycles of ``task``."""
    c = cycles[(cycles.task == task) & cycles.ok.astype(bool)]
    st = np.sort(c.t_start_us.to_numpy(float)) / 1e6
    return float(np.median(np.diff(st))) if st.size > 2 else float("nan")


class BandShifts:
    """Stands in for ``biomech.timing.surrogates`` inside ``timing.lead``: identical except
    ``uniform_shifts``, which rejects shifts within ``band`` of a period multiple (k >= 1)."""

    def __init__(self, real: Any, band: float) -> None:
        self._real, self.band, self.period = real, float(band), float("nan")
        self.task = ""
        self.shifts: dict[str, np.ndarray] = {}           # task -> the shifts used (s)
        self.wraps: dict[str, list[float]] = {}           # task -> wrap flags of the windows

    def shifted_windows(self, i_ref, shift, off, n):
        i_e, sel = self._real.shifted_windows(i_ref, shift, off, n)
        o = np.asarray(off, np.int64)
        wrap = (np.asarray(i_e) + o.min() < 0) | (np.asarray(i_e) + o.max() >= n)
        self.wraps.setdefault(self.task, []).extend(wrap.astype(float).tolist())
        return i_e, sel

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)

    def uniform_shifts(self, rng: np.random.Generator, span: float, n: int, gap: float
                       ) -> np.ndarray:
        if self.band <= 0 or not np.isfinite(self.period):
            s0 = self._real.uniform_shifts(rng, span, n, gap)
            self.shifts[self.task] = s0
            return s0
        out: list[float] = []
        while len(out) < n:
            s = self._real.uniform_shifts(rng, span, n, gap)
            r = s / self.period
            k = np.round(r)
            keep = ~((k >= 1) & (np.abs(r - k) <= self.band))
            out.extend(s[keep].tolist())
        self.shifts[self.task] = np.asarray(out[:n])
        return np.asarray(out[:n])


def interval_sd_rel(cycles: pd.DataFrame, task: str) -> float:
    """SD of the start-to-start intervals of the ok cycles of ``task`` / their median."""
    c = cycles[(cycles.task == task) & cycles.ok.astype(bool)]
    d = np.diff(np.sort(c.t_start_us.to_numpy(float)) / 1e6)
    return float(np.std(d, ddof=1) / np.median(d)) if d.size > 2 else float("nan")


def participant_null(args: tuple[str, str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The recomputed ``lead_null`` rows of one participant for one band, and its
    descriptive row per task (period, interval SD, near-period and junction shares)."""
    snapshot, pid, band = args
    from biomech.io.context import DataContext
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.timing import lead, surrogates

    # Patched per job and restored in ``finally`` (a worker process may run several jobs;
    # the pool also uses one task per child, so no patch can leak into the next job).
    saved = (lead.surrogates, lead.pseudo_onset_null, lead.analyse_ensemble,
             lead.analyse_release)
    cfg = _config(snapshot)
    dc = DataContext(cfg)
    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    try:
        pc = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("timing"),
                                store=dc.store, participant=part, cache=cache)
        cycles = pc.upstream("kinematics", "cycles")
        shim = BandShifts(surrogates, band)                     # the real module
        periods = {t: median_period_s(cycles, t) for t in TASKS}
        orig = lead.pseudo_onset_null

        def wrapped(det, pid_, task, *a, **k):
            shim.period, shim.task = periods.get(task, float("nan")), task
            return orig(det, pid_, task, *a, **k)

        lead.surrogates, lead.pseudo_onset_null = shim, wrapped
        lead.analyse_ensemble = lambda *a, **k: ([], {})       # X2: not needed for the null
        lead.analyse_release = lambda *a, **k: ([], [])        # X3: not needed for the null
        tables, _ = lead.analyse(pc.emg_prep(), pid, pc.upstream_arrays("imu"), cycles,
                                 pc.params, pc.seed)
        out = tables["lead_null"].copy()
        out["band"] = band
        out["period_s"] = out["task"].map(periods)
        desc = []
        for t in TASKS:
            sh = shim.shifts.get(t, np.empty(0))
            P = periods.get(t, float("nan"))
            r = sh / P if sh.size and np.isfinite(P) else np.empty(0)
            k = np.round(r)
            near = float(np.mean((k >= 1) & (np.abs(r - k) <= 0.10))) if r.size else np.nan
            w = shim.wraps.get(t, [])
            desc.append({"pid": pid, "task": t, "band": band, "period_s": P,
                         "interval_sd_rel": interval_sd_rel(cycles, t),
                         "near_share": near, "n_shifts": int(sh.size),
                         "junction_share": float(np.mean(w)) if w else np.nan,
                         "n_windows": len(w)})
        return out, pd.DataFrame(desc)
    finally:
        (lead.surrogates, lead.pseudo_onset_null, lead.analyse_ensemble,
         lead.analyse_release) = saved
        cache.close()
        dc.close()


def cohort_rows(cfg: Any, lead_null: pd.DataFrame) -> pd.DataFrame:
    """``timing_cohort`` lead rows with the given null (``stats_timing`` call, unchanged)."""
    from biomech.io.context import DataContext
    from biomech.stats.cohort_misc import timing_tables

    dc = DataContext(cfg)
    try:
        t = timing_tables(dc.table("timing", "summary"), dc.table("timing", "onsets"),
                          cfg.seed, None, None, params=cfg.params("stats_timing"),
                          onset_rule=cfg.params("timing").get("onset_rule", "drive"),
                          lead_null=lead_null.drop(columns=["band", "period_s"],
                                                   errors="ignore"))
        from biomech.stats.core import describe

        tc = t["timing_cohort"]
        out = tc[tc.metric == "lead_ms"][KEYS + NULL_COLS].reset_index(drop=True)
        pairs = t.get("timing_null", pd.DataFrame())
        ci = []
        for _, r in out.iterrows():
            g = pairs[(pairs.task == r.task) & (pairs.direction == r.direction)
                      & (pairs.detector == r.detector) & (pairs.rule == r.rule)] \
                if len(pairs) else pairs
            v = g["lead_minus_null_ms"].to_numpy(float) if len(g) else np.empty(0)
            d = describe(v, seed=int(cfg.seed or 0), stat="median")
            ci.append({"mn_median": d["median"], "mn_ci_lo": d["ci_lo"],
                       "mn_ci_hi": d["ci_hi"], "mn_n": d["n"]})
        return pd.concat([out, pd.DataFrame(ci)], axis=1)
    finally:
        dc.close()


def _same(a: pd.DataFrame, b: pd.DataFrame, keys: list[str], cols: list[str]) -> list[str]:
    m = a.merge(b, on=keys, suffixes=("_a", "_b"), how="outer", indicator=True)
    bad = [] if (m["_merge"] == "both").all() else ["rows differ"]
    for c in cols:
        x, y = m[f"{c}_a"].to_numpy(float), m[f"{c}_b"].to_numpy(float)
        if not np.allclose(x, y, rtol=1e-9, atol=1e-9, equal_nan=True):
            bad.append(c)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    ap.add_argument("--band", type=float, default=0.10)
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--pids", nargs="*")
    a = ap.parse_args()
    from biomech.io.context import DataContext

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    stored_null = dc.table("timing", "lead_null")
    stored_cohort = dc.table("stats_timing", "timing_cohort")
    pids = a.pids or sorted(stored_null.pid.unique())
    dc.close()
    jobs = [(a.snapshot, p, b) for b in (0.0, a.band) for p in pids]
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        parts = list(ex.map(participant_null, jobs))
    null = pd.concat([p[0] for p in parts], ignore_index=True)
    desc = pd.concat([p[1] for p in parts], ignore_index=True)

    # Validation: band 0 reproduces the stored null and cohort null columns.
    nk = ["pid", "task", "direction", "detector", "rule"]
    ncols = ["lead_null_ms", "lead_null_muscle_mean_ms", "n_null", "n_null_usable",
             "usable_null_frac"]
    zero = null[null.band == 0].drop(columns=["band", "period_s"])
    bad = _same(zero, stored_null[stored_null.pid.isin(pids)], nk, ncols)
    if bad:
        raise SystemExit(f"band 0 does not reproduce timing:lead_null: {bad}")
    rows = []
    for b, g in null.groupby("band"):
        rows.append(cohort_rows(cfg, g).assign(band=b))
    coh = pd.concat(rows, ignore_index=True)
    if not a.pids:
        sc = stored_cohort[stored_cohort.metric == "lead_ms"][KEYS + NULL_COLS]
        bad = _same(coh[coh.band == 0].drop(columns="band"), sc, KEYS, NULL_COLS)
        if bad:
            raise SystemExit(f"band 0 does not reproduce stats_timing:timing_cohort: {bad}")
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"c2_null_period_{a.snapshot}" + ("" if not a.pids else "_partial")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    coh.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(out_dir / f"{stem}.parquet")
    null.to_parquet(out_dir / f"{stem}_participants.parquet")
    desc.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(out_dir / f"{stem}_desc.parquet")
    per = null.drop_duplicates(["pid", "task"])[["pid", "task", "period_s"]]
    meta = {"snapshot": a.snapshot, "band": a.band, "git_sha": sha, "pids": pids,
            "validation": "band 0 reproduces timing:lead_null"
                          + ("" if a.pids else " and stats_timing:timing_cohort"),
            "median_period_s": per.groupby("task").period_s.median().to_dict(),
            "sha256": hashlib.sha256((out_dir / f"{stem}.parquet").read_bytes()).hexdigest(),
            "sha256_desc": hashlib.sha256((out_dir / f"{stem}_desc.parquet").read_bytes()
                                          ).hexdigest()}
    (out_dir / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    show = coh[(coh.detector == "tkeo") & coh.rule.isin(["first", "drive"])]
    print(show[["band"] + KEYS + ["lead_null_ms", "lead_minus_null_ms", "p_null_holm",
                                  "p_null_perm_holm"]].to_string(index=False))
    d0 = desc[desc.band == 0]
    print(d0.groupby("task")[["interval_sd_rel", "near_share", "junction_share"]].median())
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
