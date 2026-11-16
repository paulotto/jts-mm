# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J18.2 / J18.4 / J18.5: C2 timing additions that need the EMG (post hoc,
exploratory; every definition is fixed in J18 before this ran).

Per participant of the paper store (adults, pin r16a) the unchanged pipeline is re-run
(``timing.lead.analyse`` with its ensemble call recorded, ``n_null`` 0; ``timing.c2e._coupling``)
and checked against the store first:

* the X2 median curves (``timing`` arrays ``ensemble__*``) and the onset table
  (``t_decision_us``, ``lead_ms``) must be identical;
* P5 of the closers (``timing:coupling`` ``emd_lag_ms``, ``emd_r_null_median``) must be
  identical, also through this script's own copy of ``coupling.coupling`` that additionally
  keeps the peak lag of every circular shift.

Then:

* **J18.2** P5 of the openers against +alpha (same code path and parameters; seed
  ``null_seed(seed, "j18:p5o:<task>")``) and, for both directions, the centred lag null (the
  peak lag of each of the 200 shifted EMG features; participant null = median);
* **J18.4** rise / release times of the X2 median curves with the fixed rule (baseline
  -1000 ... -600 ms, mu +- max(3 sd, 1 dB), held >= 25 ms, searched -600 ... +600 ms) and the
  same rule on 200 S1-shifted curves (``null_seed(seed, "j18:rise:<task>")``);
* **J18.5** per cycle the causal EMG decisions (stored ``onsets``: TKEO, rule first, the
  earliest ``t_decision_us`` of the group; stored ``release``: X3 ``t_release_us``) and the
  causal IMU detector (opening-angle rate of ``theta_quat``, causal 20-ms mean, rest mean
  +- k SD, k = 3 and 5, held >= 25 ms), all within -600 ... +200 ms of t_mech; and the false
  alarms per minute of rest of each detector (decisions >= 0.5 s apart).

    software/jtsmm/.venv/bin/python scripts/audit_recompute/j18_emg.py [-j 4] [--pids s01]

Output in ``<adults store>/audit/``: ``j18_coupling_<snap>.parquet``,
``j18_rise_<snap>.parquet``, ``j18_device_<snap>.parquet`` (per cycle),
``j18_device_rest_<snap>.parquet`` and ``j18_emg_<snap>.json`` (git sha, checks, sha256).
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

import _pins  # noqa: E402

STUDY = _pins.EXP / "icorr2027" / "study.yaml"
VARIANT = "reporting_population"
TASKS = ("open_close", "chewing")
N_SHIFTS = 200
GAP_S = 1.0
# J18.4 rule
BASE_MS = (-1000.0, -600.0)
SEARCH_MS = (-600.0, 600.0)
K_SD, FLOOR_DB, HOLD_MS = 3.0, 1.0, 25.0
CUES = (("digastric_rise", "opening", "agonist", "rise"),
        ("closer_rise", "closing", "agonist", "rise"),
        ("closer_release", "opening", "antagonist", "release"))
# J18.5
WIN_PRE_MS, WIN_POST_MS = 600.0, 200.0
IMU_SMOOTH_S, IMU_HOLD_S, REFRACTORY_S = 0.02, 0.025, 0.5
IMU_K = (3.0, 5.0)


def _config(snapshot: str):
    from biomech import sensitivity as S
    from biomech.config import load_config

    return S.store_config(load_config(str(STUDY)), VARIANT, snapshot)


# -- J18.4 rule ---------------------------------------------------------------------------
def rule_time(curve: np.ndarray, g: np.ndarray, kind: str) -> float:
    """First t in SEARCH_MS where the curve stays beyond mu +- max(K_SD sd, FLOOR_DB) for
    HOLD_MS (consecutive grid points spanning it); NaN if none."""
    c = np.asarray(curve, float)
    b = c[(g >= BASE_MS[0]) & (g <= BASE_MS[1])]
    b = b[np.isfinite(b)]
    if b.size < 3:
        return np.nan
    mu, sd = float(b.mean()), float(b.std(ddof=1))
    d = max(K_SD * sd, FLOOR_DB)
    hit = (c > mu + d) if kind == "rise" else (c < mu - d)
    hit &= np.isfinite(c)
    step = float(g[1] - g[0])
    need = int(round(HOLD_MS / step)) + 1                     # points spanning HOLD_MS
    idx = np.flatnonzero((g >= SEARCH_MS[0]) & (g <= SEARCH_MS[1]))
    run = 0
    for i in idx:
        run = run + 1 if hit[i] else 0
        if run >= need:
            return float(g[i - need + 1])
    return np.nan


# -- J18.2 coupling with the lag null (a copy of coupling.coupling that keeps the lags) ------
def coupling_lags(t_us, emg, alpha_arg, starts_us, ends_us, cutoff_hz, order, cp, rng,
                  occlusions_us) -> dict[str, Any]:
    from biomech.timing import coupling as C
    from biomech.timing import surrogates

    out = {"lag_ms": np.nan, "r": np.nan, "r_null_median": np.nan, "null_lag_median_ms": np.nan,
           "n_null": 0, "null_lags_ms": np.empty(0)}
    t_us = np.asarray(t_us)
    if t_us.size < 4 or len(starts_us) < 3:
        return out
    dt_ms = float(np.median(np.diff(t_us))) / 1e3
    e = C.lowpass(t_us, np.asarray(emg, float), cutoff_hz, order)
    a = -np.asarray(alpha_arg, float)
    for t_occ in occlusions_us:
        a[np.abs(t_us - t_occ) <= cp.occlusion_mask_s * 1e6] = np.nan
    a = C.lowpass(t_us, a, cutoff_hz, order)
    k = int(round(cp.max_lag_ms / dt_ms))
    er = C.residualise(t_us, e, starts_us, ends_us, cp.n_phase_bins)
    ar = C.residualise(t_us, a, starts_us, ends_us, cp.n_phase_bins)
    lag, r = C.peak_lag(*C.lagged_corr(er, ar, k, cp.min_pairs, center=True))
    out.update(lag_ms=lag * dt_ms, r=r)
    if not np.isfinite(r):
        return out
    shifts = surrogates.s1_sample_shifts(rng, len(t_us), 1e3 / dt_ms, cp.n_null, cp.gap_s)
    nr, nl = np.full(shifts.size, np.nan), np.full(shifts.size, np.nan)
    for j, s in enumerate(shifts):
        es = C.residualise(t_us, np.roll(e, int(s)), starts_us, ends_us, cp.n_phase_bins)
        nl[j], nr[j] = C.peak_lag(*C.lagged_corr(es, ar, k, cp.min_pairs, center=True))
    fin = np.isfinite(nr)
    out.update(r_null_median=float(np.median(nr[fin])) if fin.any() else np.nan,
               null_lag_median_ms=float(np.median(nl[fin]) * dt_ms) if fin.any() else np.nan,
               n_null=int(fin.sum()), null_lags_ms=nl[fin] * dt_ms)
    return out


# -- J18.5 IMU detector ----------------------------------------------------------------------
def imu_rate(t_us: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Causal opening-angle rate (deg/s): backward difference, causal IMU_SMOOTH_S mean."""
    t = np.asarray(t_us, float) / 1e6
    th = np.asarray(theta, float)
    r = np.full(th.size, np.nan)
    r[1:] = np.diff(th) / np.diff(t)
    fs = 1.0 / float(np.median(np.diff(t)))
    n = max(int(round(IMU_SMOOTH_S * fs)), 1)
    out = np.full(r.size, np.nan)
    for i in range(n, r.size):
        w = r[i - n + 1:i + 1]
        if np.isfinite(w).all():
            out[i] = w.mean()
    return out


def imu_decisions(t_us, rate, mu, sd, k, direction, refractory_s) -> np.ndarray:
    from biomech.timing.release import _decisions

    fs = 1e6 / float(np.median(np.diff(t_us)))
    hold = max(int(round(IMU_HOLD_S * fs)), 1)
    with np.errstate(invalid="ignore"):
        hit = (rate > mu + k * sd) if direction == "opening" else (rate < mu - k * sd)
    hit &= np.isfinite(rate)
    return np.asarray(t_us)[_decisions(hit, hold, int(round(refractory_s * fs)))]


def _merge(times: np.ndarray, refractory_us: float) -> np.ndarray:
    out, last = [], -np.inf
    for t in np.sort(np.asarray(times, float)):
        if t - last >= refractory_us:
            out.append(t)
            last = t
    return np.asarray(out)


def participant(args: tuple[str, str]) -> dict[str, Any]:
    """One participant, single-threaded BLAS as the pipeline runs its units
    (``pipeline._exec_participant``): multi-threaded dot products differ in the last bits."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        return _participant(args)


def _participant(args: tuple[str, str]) -> dict[str, Any]:
    snapshot, pid = args
    from biomech.io.context import DataContext
    from biomech.io.session import CLOSERS, OPENERS
    from biomech.pipeline import ParticipantContext, UnitCache
    from biomech.timing import c2e, ensemble, lead, release, surrogates
    from jtsmm.emg.onset import detect_onsets

    saved = (lead.analyse_ensemble, lead.analyse_release)
    cfg = _config(snapshot)
    dc = DataContext(cfg)
    part = dc.cohort.get(pid)
    cache = UnitCache(dc.cohort, part)
    checks: dict[str, Any] = {}
    try:
        pc = ParticipantContext(config=cfg, cohort=dc.cohort, params=cfg.params("timing"),
                                store=dc.store, participant=part, cache=cache)
        p, seed = pc.params, pc.seed
        prep = pc.emg_prep()
        imu = pc.upstream_arrays("imu")
        cycles = pc.upstream("kinematics", "cycles")
        rec: dict[str, Any] = {}

        def ens(names, phases, rest_level, events, pid_, ep, seed_):
            rec.update(names=names, phases=phases, rest_level=rest_level, events=events, ep=ep)
            return saved[0](names, phases, rest_level, events, pid_, ep, seed_)

        lead.analyse_ensemble = ens
        lead.analyse_release = lambda *a, **k: ([], [])
        tables, arrays = lead.analyse(prep, pid, imu, cycles, p, seed, n_null=0)
        stored_arr = dc.store.read_arrays("timing", pid)
        stored_on = pc.upstream("timing", "onsets")
        ens_keys = [k for k in stored_arr if k.startswith("ensemble__")]
        checks["ensemble_identical"] = all(
            np.array_equal(stored_arr[k], arrays[k], equal_nan=True) for k in ens_keys)
        on = tables["onsets"]
        checks["onsets_identical"] = bool(
            len(on) == len(stored_on) and np.array_equal(
                on["t_decision_us"].to_numpy(float), stored_on["t_decision_us"].to_numpy(float),
                equal_nan=True) and np.array_equal(on["lead_ms"].to_numpy(float),
                                                    stored_on["lead_ms"].to_numpy(float),
                                                    equal_nan=True))
        names = list(rec["names"])
        ep = rec["ep"]
        g = ensemble.grid_ms(ep)

        # ---- J18.4 ------------------------------------------------------------------------
        rise_rows = []
        for task, (t_us, cleaned, mask) in rec["phases"].items():
            fs = 1e6 / np.median(np.diff(t_us))
            db = ensemble.envelope_db(cleaned, mask, fs, rec["rest_level"], ep.envelope_s)
            gs = np.round(g * fs / 1e3).astype(np.int64)
            N = len(t_us)
            shifts = surrogates.s1_sample_shifts(surrogates.null_seed(seed, f"j18:rise:{task}"),
                                                 N, fs, N_SHIFTS, GAP_S)
            for cue, direction, role, kind in CUES:
                ev = np.asarray(rec["events"].get((task, direction), []), float)
                ev = ev[np.isfinite(ev)]
                key = f"ensemble__{task}__{direction}__{role}"
                if ev.size == 0 or key not in arrays:
                    continue
                agon, anta = ensemble.GROUPS[direction]
                group = agon if role == "agonist" else anta
                idx = [names.index(m) for m in group if m in names]
                i_ref = np.clip(np.searchsorted(t_us, ev), 0, N - 1)
                t_obs = rule_time(arrays[key], g, kind)

                def curve(s: int) -> np.ndarray:
                    import warnings

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        y = np.nanmean(ensemble.aligned_db_circular(db[:, idx], i_ref, gs, s),
                                       axis=2)
                        return np.nanmedian(y, axis=0)

                c0 = curve(0)
                nt = np.array([rule_time(curve(int(s)), g, kind) for s in shifts])
                fin = nt[np.isfinite(nt)]
                rise_rows.append({
                    "pid": pid, "task": task, "cue": cue, "direction": direction, "role": role,
                    "n_cycles": int(ev.size), "t_ms": t_obs, "lead_ms": -t_obs,
                    "null_lead_ms": float(-np.median(fin)) if fin.size else np.nan,
                    "null_found_share": float(fin.size / max(nt.size, 1)),
                    "n_shifts": int(nt.size),
                    "shift0_max_abs_diff_db": float(np.nanmax(np.abs(c0 - arrays[key])))
                    if np.isfinite(c0 - arrays[key]).any() else np.nan})

        # ---- J18.2 ------------------------------------------------------------------------
        cp_par = c2e.C2EParams.from_dict(p.get("c2e"))
        stored_cp = pc.upstream("timing", "coupling")
        coup_rows = []
        for task in TASKS:
            allc = cycles[cycles.task == task].sort_values("t_start_us")
            allc = allc.assign(prev_end_us=allc.t_end_us.shift(1))
            cyc = allc[allc.ok.astype(bool)]
            pe = prep.phase(task)
            x = np.where(pe.mask, np.nan, pe.cleaned)
            ref = c2e._coupling(pid, task, imu, pe, x, names, cyc, cp_par, seed)
            srow = stored_cp[stored_cp.task == task]
            same = (not srow.empty and np.allclose(
                [ref.get("emd_lag_ms", np.nan), ref.get("emd_r_null_median", np.nan)],
                [srow.emd_lag_ms.iloc[0], srow.emd_r_null_median.iloc[0]], equal_nan=True,
                rtol=0, atol=0))
            checks[f"coupling_identical_{task}"] = bool(same)
            t = imu[f"{task}__t_us"].astype(np.int64)
            thq = imu.get(f"{task}__theta_quat")
            if thq is None:
                continue
            from biomech import timebase
            from biomech.timing import coupling as C

            alpha = timebase.gradient(t, timebase.gradient(t, np.asarray(thq, np.float64)))
            st = cyc.t_start_us.to_numpy(float)
            en = np.r_[st[1:], cyc.t_end_us.to_numpy(float)[-1:]]
            occl = cyc.t_end_us.to_numpy(float)
            for direction, group, alpha_arg, key in (
                    ("closing", CLOSERS, alpha, f"c2e:p5:{task}"),
                    ("opening", OPENERS, -alpha, f"j18:p5o:{task}")):
                cols = [names.index(m) for m in group if m in names]
                emg = C.emg_feature(np.asarray(pe.t_us), x[:, cols], t, float(pe.fs),
                                    cp_par.coupling.smooth_s)
                res = coupling_lags(t, emg, alpha_arg, st, en, cp_par.alpha_cutoff_hz,
                                    cp_par.alpha_order, cp_par.coupling,
                                    surrogates.null_seed(seed, key), occl)
                if direction == "closing":
                    checks[f"coupling_copy_identical_{task}"] = bool(np.allclose(
                        [res["lag_ms"], res["r_null_median"]],
                        [ref.get("emd_lag_ms", np.nan), ref.get("emd_r_null_median", np.nan)],
                        equal_nan=True, rtol=0, atol=0))
                coup_rows.append({"pid": pid, "task": task, "direction": direction,
                                  "lag_ms": res["lag_ms"], "r": res["r"],
                                  "r_null_median": res["r_null_median"],
                                  "null_lag_median_ms": res["null_lag_median_ms"],
                                  "lag_minus_null_ms": res["lag_ms"] - res["null_lag_median_ms"],
                                  "n_null": res["n_null"]})

        # ---- J18.5 ------------------------------------------------------------------------
        rest_t = imu["rest__t_us"].astype(np.int64)
        rest_rate = imu_rate(rest_t, imu["rest__theta_quat"])
        mu, sd = float(np.nanmean(rest_rate)), float(np.nanstd(rest_rate, ddof=1))
        on = stored_on[(stored_on.detector == "tkeo") & (stored_on.rule == "first")]
        rel = pc.upstream("timing", "release")
        dev_rows = []
        for task in TASKS:
            t = imu[f"{task}__t_us"].astype(np.int64)
            rate = imu_rate(t, imu[f"{task}__theta_quat"])
            dec = {(d, k): imu_decisions(t, rate, mu, sd, k, d, 0.0)
                   for d in ("opening", "closing") for k in IMU_K}

            def first_in(arr, tm):
                w = arr[(arr >= tm - WIN_PRE_MS * 1e3) & (arr <= tm + WIN_POST_MS * 1e3)]
                return float(w[0]) if w.size else np.nan

            def add(det_name, direction, tm, t_emg, cycle):
                row = {"pid": pid, "task": task, "detector": det_name, "direction": direction,
                       "cycle": cycle, "t_mech_us": tm, "t_emg_us": t_emg}
                for k in IMU_K:
                    ti = first_in(dec[(direction, k)], tm)
                    row[f"t_imu_k{k:g}_us"] = ti
                    row[f"delta_k{k:g}_ms"] = (ti - t_emg) / 1e3
                dev_rows.append(row)

            for det_name, direction, group in (("digastric_onset", "opening", OPENERS),
                                               ("closer_onset", "closing", CLOSERS)):
                g_on = on[(on.task == task) & (on.direction == direction)
                          & on.muscle.isin(group)]
                for (cyc_k, tm), gg in g_on.groupby(["cycle", "t_mech_us"]):
                    te = gg.t_decision_us.to_numpy(float)
                    te = te[np.isfinite(te)]
                    add(det_name, direction, float(tm), float(te.min()) if te.size else np.nan,
                        int(cyc_k))
            for _, r in rel[rel.task == task].iterrows():
                add("closer_release", "opening", float(r.t_mech_us),
                    float(r.t_release_us) if bool(r.detected) else np.nan, int(r.cycle))

        # rest false alarms
        rest = prep.phase("rest")
        rest_c = np.asarray(prep.causal("rest"))
        rmask = np.asarray(rest.mask, bool)
        minutes_emg = float((~rmask.all(axis=1)).sum() / rest.fs / 60.0)
        det, _, _ = lead.build_detector(prep, p)
        rest_rows = []
        for det_name, group in (("digastric_onset", OPENERS), ("closer_onset", CLOSERS)):
            times = []
            for m in group:
                if m not in names:
                    continue
                i = names.index(m)
                st = det.rest_thr("tkeo", i, True)
                xc = np.where(rmask[:, i], np.nan, rest_c[:, i])
                d_idx = detect_onsets(xc, rest.fs, None, det.params["tkeo"], causal=True,
                                      baseline_stats=st)
                times += list(np.asarray(rest.t_us)[d_idx])
            n = len(_merge(np.asarray(times), REFRACTORY_S * 1e6))
            rest_rows.append({"pid": pid, "detector": det_name, "n_false": n,
                              "minutes": minutes_emg,
                              "false_per_min": n / minutes_emg if minutes_emg else np.nan})
        rp = release.ReleaseParams.from_dict(p.get("release"))
        rest_level_c = lead._rest_level(rest_c, rest.mask, rest.fs, ep.envelope_s)
        e_db = release.causal_db(np.asarray(rest.t_us), rest_c, rest.mask, rest_level_c, names,
                                 rp.window_s)
        n_rel = len(release.detect(np.asarray(rest.t_us), e_db, rp))
        rest_rows.append({"pid": pid, "detector": "closer_release", "n_false": n_rel,
                          "minutes": minutes_emg,
                          "false_per_min": n_rel / minutes_emg if minutes_emg else np.nan})
        minutes_imu = float(np.isfinite(rest_rate).sum() / (1e6 / np.median(np.diff(rest_t)))
                            / 60.0)
        for k in IMU_K:
            for direction in ("opening", "closing"):
                n = len(imu_decisions(rest_t, rest_rate, mu, sd, k, direction, REFRACTORY_S))
                rest_rows.append({"pid": pid, "detector": f"imu_{direction}_k{k:g}",
                                  "n_false": n, "minutes": minutes_imu,
                                  "false_per_min": n / minutes_imu if minutes_imu else np.nan})
        return {"pid": pid, "checks": checks, "rise": pd.DataFrame(rise_rows),
                "coupling": pd.DataFrame(coup_rows), "device": pd.DataFrame(dev_rows),
                "rest": pd.DataFrame(rest_rows), "imu_rest_mu_sd": (mu, sd)}
    finally:
        lead.analyse_ensemble, lead.analyse_release = saved
        cache.close()
        dc.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--snapshot", default=_pins.dataset_paper())
    ap.add_argument("-j", "--jobs", type=int, default=4)
    ap.add_argument("--pids", nargs="*")
    a = ap.parse_args()
    from biomech.io.context import DataContext

    cfg = _config(a.snapshot)
    dc = DataContext(cfg)
    pids = a.pids or sorted(dc.table("timing", "summary").pid.unique())
    dc.close()
    with ProcessPoolExecutor(max_workers=a.jobs, max_tasks_per_child=1) as ex:
        res = list(ex.map(participant, [(a.snapshot, p) for p in pids]))
    checks = {r["pid"]: r["checks"] for r in res}
    bad = {p: [k for k, v in c.items() if not v] for p, c in checks.items()}
    bad = {p: v for p, v in bad.items() if v}
    if bad:
        raise SystemExit(f"validation failed: {bad}")
    out_dir = Path(cfg.out_dir) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = a.snapshot + ("" if not a.pids else "_partial")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=_pins.ROOT).stdout.strip()
    meta: dict[str, Any] = {"snapshot": a.snapshot, "git_sha": sha, "pids": pids,
                            "checks": "all identical: X2 curves, onsets, P5 (stage and copy)"}
    for name in ("rise", "coupling", "device", "rest"):
        df = pd.concat([r[name] for r in res], ignore_index=True)
        stem = f"j18_{'device_rest' if name == 'rest' else name}_{suffix}"
        df.assign(snapshot=a.snapshot, git_sha=sha).to_parquet(out_dir / f"{stem}.parquet")
        meta[f"sha256_{name}"] = hashlib.sha256((out_dir / f"{stem}.parquet").read_bytes()
                                                ).hexdigest()
    meta["imu_rest_mu_sd"] = {r["pid"]: r["imu_rest_mu_sd"] for r in res}
    (out_dir / f"j18_emg_{suffix}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "imu_rest_mu_sd"}, indent=1))


if __name__ == "__main__":
    main()
