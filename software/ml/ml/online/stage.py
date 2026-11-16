# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cohort stage ``ml_online``: pseudo-online replay of every participant (X4, ml08).

Registered through ``ml.biomech_stages`` (entry point ``biomech.stages``). Per participant:

1. the fold model(s) of the held-out participant, trained on the other participants' stored
   ``ml_windows`` windows with ``ml_eval``'s fold hyper-parameters (in parallel; loaded from
   the model cache ``ml_eval`` filled when present, :mod:`ml.models.cache`, same model);
2. the take replayed chunk by chunk (:func:`ml.online.stream.replay`, one at a time, one
   thread): causal chain, window features, in-loop decoding with the primary feature set's
   fold model, the X3 release detector; per-chunk compute time. With ``replay_overlap``
   (default; environment ``ML_ONLINE_REPLAY_OVERLAP=0|1`` overrides) the replays run in
   their own process pinned to one reserved core while the other cores fit the
   calibration models of the participants already replayed;
3. calibration refits (:mod:`ml.online.decoder`), decoding of the stored stream windows with
   the model in force at each window, smoothing, onset events and the metrics of
   :mod:`ml.online.metrics`; X3 against the mechanical opening onsets of ``timing.onsets``;
4. the deviation of the stream features from the stored offline ``ml_windows`` features.

The stream starts ``warmup_s`` before the first windowed phase (the rest phase); the mains
frequency comes from up to 60 s *before* that point (``line_hz: preroll``), so nothing after
the stream start is used to set it (``offline`` = the take's offline value, which is measured
over the first 60 s of the take's phases).
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from biomech.emg.pipeline import take_line_hz
from biomech.emg.prepare import EmgParams, _line_hz
from biomech.imu.preprocess import load as load_imu
from biomech.io.session import SessionData
from biomech.config import ConfigError
from biomech.pipeline import CohortContext, StageResult, participant_seed, stage
from biomech.timing.release import ReleaseParams

from ml.config.classes import FREE_PHASE, PHASES, REST_PHASE
from ml.core.dataset import Imu, WindowConfig, feature_set_blocks
from ml.online import decoder as dc
from ml.online import metrics as mt
from ml.online.stream import StreamConfig, StreamInput, replay
from ml.protocols import evaluate as ev
from ml.protocols import pool as pl
from ml.protocols import variants as va
from ml.protocols.folds import make_folds

log = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "chunk_s": 0.02,
    "model": "svm",
    "feature_sets": ["emg_imu", "imu"],
    "primary_feature_set": "emg_imu",
    "line_hz": "preroll",
    "filter_mode": "continuous",
    "imu_max_wait_s": 0.1,
    "smoothing": {"n": 3, "enter": 2, "hold": 1},
    "calibration": {"modes": ["none", "first", "per_phase"], "first_k": 100, "per_phase_k": 20,
                    "guard_s": 0.35, "weight": "balanced"},
    "latency": {"pre_s": 0.5, "budget_ms": 150.0},
    "fa_guard_s": 0.5,
    "pids": [],
    "n_jobs": 0,
    # Replays (compute time per chunk) in their own process on one reserved core while the
    # calibration refits use the other cores; false: one after another in this process,
    # the refits afterwards (decisions identical either way).
    "replay_overlap": True,
}
_STAMP = re.compile(r"_(\d{8}T\d{6}Z)_")


def config(p: dict[str, Any] | None) -> dict[str, Any]:
    p = dict(p or {})
    if "max_train" in p:
        raise ConfigError("stages.ml_online.max_train was removed: the fold models are "
                          "ml_eval's, fitted with stages.ml_eval.max_train and inner_folds")
    bad = sorted(set(p) - set(DEFAULTS))
    if bad:
        raise ValueError(f"unknown ml_online keys {bad}; known: {sorted(DEFAULTS)}")
    out = {**DEFAULTS, **p}
    for k in ("smoothing", "calibration", "latency"):
        out[k] = {**DEFAULTS[k], **dict(p.get(k) or {})}
    if out["line_hz"] not in ("preroll", "offline"):
        raise ValueError("ml_online.line_hz must be preroll or offline")
    if out["primary_feature_set"] not in out["feature_sets"]:
        raise ValueError("ml_online.primary_feature_set must be one of feature_sets")
    return out


def take_stamp(folder: str) -> str | None:
    m = _STAMP.search(str(folder))
    return m.group(1) if m else None


def windowed_phases(sd: SessionData, lab: dict[str, Any]) -> list[tuple[str, int, int]]:
    """The phases ``ml_windows`` windows: labelled ones of PHASES plus the rest phase."""
    out = [(ph.name, int(ph.t_start_us), int(ph.t_end_us)) for ph in sd.phases
           if ph.name in PHASES and (f"{ph.name}__label" in lab or ph.name == REST_PHASE)]
    return sorted(out, key=lambda r: r[1])


def load_stream(sd: SessionData, channels: list[str], phases: list[tuple[str, int, int]],
                emg_signals: dict[str, Any], warmup_s: float, line: str = "preroll"
                ) -> tuple[StreamInput, dict[str, Any]]:
    """The take from ``warmup_s`` before its first windowed phase to its last one."""
    fs = sd.emg_fs
    start = max(int(phases[0][1] - warmup_s * 1e6), 0)
    stop = max(t1 for _, _, t1 in phases)
    nominal = float(emg_signals.get("nominal_line_hz", 50.0))
    pre_t, pre_uv, _ = sd.emg_window(max(start - 60_000_000, 0), start - 1, channels)
    if pre_t.size >= int(fs):
        # biomech's take estimator (nominal grid first, then the other of 50/60 Hz)
        f_pre, src = take_line_hz(pre_uv, fs, pre_t, nominal), "preroll"
    else:
        f_pre, src = nominal, "nominal"
    f_off = float(_line_hz(sd, EmgParams.from_dict(emg_signals)))
    lf = f_pre if line == "preroll" else f_off
    if line == "offline":
        src = "offline"
    t, uv, _ = sd.emg_window(start, stop, channels)
    lead = sd.leadoff_window(start, stop, channels)
    mta, cra = load_imu(sd, "mta"), load_imu(sd, "cra")
    imu = None if mta is None or cra is None else Imu(
        (mta.t_q, mta.q), (cra.t_q, cra.q), (mta.t_a, mta.a), (cra.t_a, cra.a))
    inp = StreamInput(t_us=np.asarray(t, np.int64), uv=uv, fs=fs, channels=list(channels),
                      phases=phases, line_hz=lf, full_scale_uv=sd.emg_full_scale_uv(channels),
                      leadoff=lead if lead.shape == uv.shape else None, imu=imu)
    info = {"line_hz": lf, "line_source": src, "line_hz_preroll": f_pre,
            "line_hz_offline": f_off, "stream_start_us": int(t[0]) if t.size else start,
            "stream_s": float((t[-1] - t[0]) / 1e6) if t.size > 1 else 0.0}
    return inp, info


def _channels(feats: pd.DataFrame, pid: str) -> list[str]:
    unit = "unit" if "unit" in feats.columns else "pid"
    f = feats[(feats[unit] == pid) & (feats.block == "emg")].feature
    return list(dict.fromkeys(n.split("|", 1)[0] for n in f))


# The fit tasks get the shared (memory-mapped) pooled matrix plus row indices instead of a
# training copy per task (ml.protocols.pool).
def _fit_fold(xref, yref, idx, g, model, params, slices, seed, max_train, inner,
              cache_dir=None):
    X, y = xref.get(), yref.get()
    t0 = time.perf_counter()
    est, info = dc.fit_fold(X[idx], y[idx], g, model=model, params=params, slices=slices, seed=seed,
                            max_train=max_train, inner=inner, cache_dir=cache_dir)
    dt = time.perf_counter() - t0
    if getattr(est, "fit_cache_hit_", False):        # ml_eval's model: its own fit time
        dt = float(est.fit_time_s_)
    return est, info, dt


def _fit_cal(xref, yref, idx, g, Xc, yc, unit, model, params, slices, seed, max_train, weight):
    X, y = xref.get(), yref.get()
    t0 = time.perf_counter()
    est = dc.fit_calibrated(X[idx], y[idx], g, Xc, yc, unit=unit, model=model, params=params,
                            slices=slices, seed=seed, max_train=max_train, weight=weight)
    return est, time.perf_counter() - t0


def _replay_job(path, names, phases, emg_sig, warmup_s, line, wcfg, scfg, rp, est0, cols):
    """Load the take and replay it with in-loop decoding (``est0``, one window at a time);
    returns (result, stream info, replay wall time)."""
    sd = SessionData(path)
    try:
        inp, info = load_stream(sd, names, phases, emg_sig, warmup_s, line)
    finally:
        sd.close()

    def predict(xe, xi, est0=est0, cols=cols):
        row = np.r_[xe, xi if xi is not None else np.zeros(0)][cols]
        return int(est0.predict(row[None, :])[0])

    t_rep = time.perf_counter()
    res = replay(inp, wcfg, scfg, predict=predict, release_params=rp)
    res.meta["phases"] = phases
    return res, info, time.perf_counter() - t_rep


def _opening_onsets(onsets: pd.DataFrame, pid: str) -> dict[str, np.ndarray]:
    if onsets.empty:
        return {}
    o = onsets[(onsets.pid == pid) & (onsets.direction == "opening")]
    o = o.groupby(["task", "cycle"], as_index=False).t_mech_us.first()
    return {t: g.t_mech_us.to_numpy(float) for t, g in o.groupby("task")}


def _c2_lead(summary: pd.DataFrame, pid: str, task: str, rule: str) -> float:
    """The participant's C2 opening lead (median over the openers; TKEO, primary rule)."""
    if summary.empty or "lead_ms" not in summary:
        return np.nan
    s = summary[(summary.pid == pid) & (summary.task == task) & (summary.direction == "opening")
                & (summary.detector == "tkeo")]
    if "rule" in s:
        s = s[s.rule == rule]
    v = s.lead_ms.to_numpy(float)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan


def feature_deviation(res, a: dict[str, np.ndarray], pid: str, late_s: float) -> list[dict]:
    """Stream vs stored offline features of the same windows (|difference| / offline SD)."""
    if not a or "t_end_us" not in a:
        return []
    to = a["t_end_us"].astype(np.int64)
    ts = res.windows["t_end_us"]
    pos = np.searchsorted(ts, to)
    pos = np.clip(pos, 0, max(ts.size - 1, 0))
    ok = (ts.size > 0) & (ts[pos] == to) if ts.size else np.zeros(to.size, bool)
    phase_start = {n: t0 for n, t0, _ in res.meta.get("phases", [])}
    rows = []
    for b, Xs in (("emg", res.X_emg), ("imu", res.X_imu)):
        if f"X_{b}" not in a:
            continue
        Xo = a[f"X_{b}"].astype(np.float64)
        sd = np.maximum(Xo.std(axis=0), 1e-9)
        m = np.flatnonzero(ok)
        z = np.abs(Xs[pos[m]] - Xo[m]) / sd
        fin = np.isfinite(z).all(axis=1)
        z = z[fin]
        ph = res.windows["phase"][pos[m]][fin]
        age = (to[m][fin] - np.array([phase_start.get(p, 0) for p in ph])) / 1e6
        late = age >= late_s

        def q(v, f):
            return float(f(v)) if v.size else np.nan
        rows.append({"pid": pid, "block": b, "n_offline": int(to.size), "n_matched": int(m.size),
                     "n_compared": int(z.shape[0]),
                     "median_abs_z": q(z, np.median), "p99_abs_z": q(z, lambda v:
                                                                     np.percentile(v, 99)),
                     "max_abs_z": q(z, np.max),
                     "frac_windows_gt_0p1": q(z.max(axis=1) > 0.1 if z.size else z, np.mean),
                     "p99_abs_z_late": q(z[late], lambda v: np.percentile(v, 99)),
                     "max_abs_z_late": q(z[late], np.max), "late_after_s": late_s})
    return rows


@stage("ml_online", scope="cohort", depends=("ml_windows", "ml_eval", "timing", "labels"),
       version="1", description="Pseudo-online replay: causal decoder + release detector (X4)")
def ml_online_stage(ctx: CohortContext) -> StageResult:
    from ml.biomech_stages import _cancel_fn, _pool, check_cancelled, model_cache_dir

    pr = config(ctx.params)
    seed = ctx.config.seed
    wcfg = WindowConfig.from_dict(ctx.config.params("ml_windows").get("window", {}))
    emg_sig = ctx.signals["emg"]
    mask_p = emg_sig.get("mask", {})
    scfg = StreamConfig.from_dict({"chunk_s": float(pr["chunk_s"]),
                                   "dilate_s": float(mask_p.get("dilate_s", 0.1)),
                                   "rail_frac": float(mask_p.get("rail_frac", 0.99)),
                                   "filter_mode": str(pr["filter_mode"]),
                                   "imu_max_wait_s": float(pr["imu_max_wait_s"])})
    sm = dc.Smoothing.from_dict(pr["smoothing"])
    cp = dc.CalibrationPlan.from_dict(pr["calibration"])
    tparams = ctx.config.params("timing")
    rp = ReleaseParams.from_dict(tparams.get("release"))
    tasks = list(tparams.get("tasks", ["open_close", "chewing"]))
    model, fsets, prim = str(pr["model"]), list(pr["feature_sets"]), pr["primary_feature_set"]
    pre_s, budget = float(pr["latency"]["pre_s"]), float(pr["latency"]["budget_ms"])
    # The uncalibrated fold model must be ml_eval's: its subsample size and inner folds.
    ev_p = ctx.config.params("ml_eval")
    inner, max_train = int(ev_p["inner_folds"]), ev_p["max_train"]
    max_train = None if max_train is None else int(max_train)

    ds = _pool(ctx)
    loso = ctx.upstream("ml_eval", "loso")
    main = va.main_params(loso.to_dict("records"), model) if not loso.empty else {}
    folds = {f.unit: f for f in make_folds(ds, "loso_subject")}
    want = {str(p).lower() for p in pr["pids"]}
    parts = [p for p in ctx.participants if p.pid in folds and (not want or p.pid in want)]
    feats = ctx.upstream("ml_windows", "features")
    onsets_df = ctx.upstream("timing", "onsets")
    rel_off = ctx.upstream("timing", "release_summary")
    lead_df = ctx.upstream("timing", "summary")
    mats = {f: ds.matrix(f) for f in fsets}
    for f in fsets:
        if "imu" in feature_set_blocks(f) and "imu" not in ds.blocks:
            raise ValueError(f"feature set {f!r} needs IMU windows")

    cache_dir = model_cache_dir(ctx)
    keys = [(p.pid, f) for p in parts for f in fsets]
    base: dict[tuple[str, str], tuple] = {}
    summary, decisions, release_rows, cal_tasks, streams = [], [], [], [], {}
    dev: dict[str, list[dict]] = {}
    arrays: dict[str, np.ndarray] = {}
    n_jobs = ev.resolve_jobs(pr["n_jobs"])
    overlap = os.environ.get("ML_ONLINE_REPLAY_OVERLAP", str(pr["replay_overlap"]))
    overlap = overlap.strip().lower() not in ("0", "false", "off", "no")
    reserve = pl.reserve_core() if overlap and n_jobs > 1 else None
    cancel = _cancel_fn(ctx)
    t_all = time.perf_counter()
    with pl.Shared(prefix="ml_online_") as sh:
        yref = sh.put(np.asarray(ds.y))
        refs = {f: sh.put(mats[f][0]) for f in fsets}

        def fold_task(pid: str, f: str) -> pl.Task:
            fo = folds[pid]
            given = main.get(f, {}).get(pid)
            return pl.Task(("fold", pid, f), _fit_fold,
                           (refs[f], yref, fo.train, fo.groups, model, given, mats[f][1], seed,
                            max_train, inner, cache_dir),
                           cost=ev.fit_cost(model, min(fo.train.size, max_train or
                                                       fo.train.size), mats[f][0].shape[1],
                                            given) * (36 if given is None else 1))

        def take_fold(pid: str, f: str, r) -> None:
            est, info, dt = r
            given = main.get(f, {}).get(pid)
            base[(pid, f)] = (est, dict(given) if given is not None else info, dt,
                              "main_loso" if given is not None else "inner_search")

        prep: dict[str, dict] = {}

        def prepare(p) -> dict | None:
            """Labels, windowed phases and channels of a participant (None: skipped)."""
            lab = ctx.upstream_arrays("labels", p.pid)
            sd = SessionData(ctx.cohort.session_path(p.motion))
            try:
                phases = windowed_phases(sd, lab)
            finally:
                sd.close()
            if not any(n == REST_PHASE for n, _, _ in phases):
                log.warning("ml_online: %s has no rest phase; skipped", p.pid)
                return None
            return {"p": p, "lab": lab, "phases": phases, "names": _channels(feats, p.pid),
                    "path": ctx.cohort.session_path(p.motion)}

        def replay_args(pid: str) -> tuple:
            q = prep[pid]
            cols, _ = dc.fset_columns({"emg": ds.names["emg"], "imu": ds.names.get("imu", [])},
                                      prim)
            return (q["path"], q["names"], q["phases"], emg_sig, wcfg.warmup_s, pr["line_hz"],
                    wcfg, scfg, rp, base[(pid, prim)][0], cols)

        def after_replay(pid: str, out) -> list[tuple]:
            """Labels, stream info, calibration refit tasks and feature deviation of one
            replayed participant."""
            res, info, wall = out
            q = prep[pid]
            p, phases, lab = q["p"], q["phases"], q["lab"]
            labels = {n: (lab[f"{n}__t_us"], lab[f"{n}__label"]) for n, _, _ in phases
                      if f"{n}__label" in lab}
            if res.names["emg"] != ds.names["emg"]:
                raise ValueError(f"{pid}: stream EMG feature names differ from ml_windows")
            y = res.attach_labels(labels, wcfg)
            streams[pid] = (res, y, labels, info, wall)
            mvc_first = None
            if p.mvc and take_stamp(p.mvc) and take_stamp(p.motion):
                mvc_first = take_stamp(p.mvc) < take_stamp(p.motion)
            info.update({"mvc_precedes_motion": mvc_first, "ref_source": "rest_rms",
                         "rest_precedes_tasks": res.calibration.rest_precedes_tasks,
                         "calibration_error": res.calibration.error,
                         "calibrated_after_rest_ms":
                             (res.calibration.t_us - dict((n, t1) for n, _, t1 in phases)
                              [REST_PHASE]) / 1e3 if res.calibration.t_us >= 0 else np.nan})
            # Calibration refits for the other modes.
            new = []
            dec = res.decodable()
            for f in fsets:
                Xs, _ = dc.stream_matrix(res.X_emg, res.X_imu, res.names, f)
                for mode in cp.modes:
                    mp = dc.plan(mode, res.windows["t_end_us"], res.windows["phase"], y, dec, cp)
                    params = base[(pid, f)][1]
                    for i, cal in enumerate(mp.cal):
                        fo = folds[pid]
                        new.append(((pid, f, mode, i), (
                            _fit_cal, (refs[f], yref, fo.train, fo.groups, Xs[cal],
                                       y[cal], pid, model, params, mats[f][1], seed,
                                       max_train, cp.weight)),
                            ev.fit_cost(model, min(fo.train.size, max_train or fo.train.size)
                                        + cal.size, mats[f][0].shape[1], params)))
            dev[pid] = feature_deviation(res, ctx.upstream_arrays("ml_windows", pid), pid,
                                         late_s=wcfg.warmup_s + wcfg.window_s)
            return new

        if reserve is None:
            # 1) fold models, 2) replays one after another here, 3) calibration refits.
            res0 = pl.run_tasks([fold_task(pid, f) for pid, f in keys], n_jobs,
                                cancelled=cancel)
            for pid, f in keys:
                take_fold(pid, f, res0[("fold", pid, f)])
            log.info("ml_online: %d fold models in %.0f s", len(keys),
                     time.perf_counter() - t_all)
            for p in parts:
                check_cancelled(ctx)
                q = prepare(p)
                if q is None:
                    continue
                prep[p.pid] = q
                cal_tasks += after_replay(p.pid, _replay_job(*replay_args(p.pid)))
            t_cal = time.perf_counter()
            out = pl.run_tasks([pl.Task(k, fn, a, cost=c) for k, (fn, a), c in cal_tasks],
                               n_jobs, cancelled=cancel)
            cal_res = {k: out[k] for k, _, _ in cal_tasks}
            log.info("ml_online: %d calibration refits in %.0f s", len(cal_tasks),
                     time.perf_counter() - t_cal)
        else:
            # Fold models and calibration refits on every core but one; the replays, one at
            # a time, on that reserved core as soon as their fold model is in. A replayed
            # participant's refits join the pool at once (longest first, no barriers).
            replay_cpu, rest = reserve
            lanes = {"main": pl.Lane(max(1, min(n_jobs, len(rest))), rest),
                     "replay": pl.Lane(1, replay_cpu)}
            for p in parts:
                q = prepare(p)
                if q is not None:
                    prep[p.pid] = q
            order = [p.pid for p in parts if p.pid in prep]

            def on_result(key, r):
                if key[0] == "fold":
                    take_fold(key[1], key[2], r)
                    pid = key[1]
                    # the replay's calibration refits (after_replay) read the fold model of
                    # every feature set: queue it once all of them are in, not only prim's
                    # (otherwise a race: KeyError for a feature set still fitting)
                    if pid in prep and all((pid, f) in base for f in fsets):
                        # replays queue in participant order (cost = position)
                        return [pl.Task(("replay", pid), _replay_job, replay_args(pid),
                                        lane="replay", cost=-order.index(pid))]
                    return None
                if key[0] == "replay":
                    new = after_replay(key[1], r)
                    cal_tasks.extend(new)
                    return [pl.Task(k, fn, a, cost=c) for k, (fn, a), c in new]
                return None

            out = pl.run_tasks([fold_task(pid, f) for pid, f in keys], lanes,
                               on_result=on_result, cancelled=cancel)
            cal_res = {k: out[k] for k, _, _ in cal_tasks}
            log.info("ml_online: %d fold models, %d replays (core %s) and %d calibration "
                     "refits in %.0f s", len(keys), len(prep), replay_cpu[0], len(cal_res),
                     time.perf_counter() - t_all)
    dev_rows = [r for p in parts for r in dev.get(p.pid, [])]

    stride_s = wcfg.stride_s
    check_cancelled(ctx)
    for p in parts:
        if p.pid not in streams:
            continue
        pid = p.pid
        res, y, labels, info, wall = streams[pid]
        w = res.windows
        t_end, phase = w["t_end_us"], w["phase"]
        dec = res.decodable()
        on = mt.label_onsets(labels)
        ns, cc = res.chunk_ns, res.calibration.chunk
        # The calibration chunk (rest statistics, all rest windows) is a one-off; reported apart.
        comp = mt.compute_stats(np.delete(ns, cc) if 0 <= cc < ns.size else ns, scfg.chunk_s)
        comp["calibration_ms"] = float(ns[cc] / 1e6) if 0 <= cc < ns.size else np.nan
        comp["c2_lead_ms"] = _c2_lead(lead_df, pid, tasks[0], tparams.get("onset_rule", "drive"))
        task_win = ~w["retro"]
        delay = (w["t_dec_us"] - t_end)[task_win] / 1e3
        common = {"pid": pid, **info, **comp, "replay_wall_s": wall,
                  "n_windows": int(len(res)), "n_decodable": int(dec.sum()),
                  "n_imu_late": int(w["imu_late"].sum()),
                  "n_uncalibrated_task": int((~w["calibrated"] & task_win).sum()),
                  "emit_delay_ms_median": float(np.median(delay)) if delay.size else np.nan,
                  "emit_delay_ms_max": float(np.max(delay)) if delay.size else np.nan}
        # X3 on the causal chain.
        ph_t = {n: (a, b) for n, a, b in res.meta["phases"] if n in tasks}
        # Chance level with the participant seed, as the offline X3 (timing stage).
        rel = mt.release_metrics(res.release_us, ph_t, _opening_onsets(onsets_df, pid), rp,
                                 seed=participant_seed(seed, pid))
        for r in rel:
            off = rel_off[(rel_off.pid == pid) & (rel_off.task == r["task"])
                          & rel_off.primary] if not rel_off.empty else rel_off
            for k in ("detected_frac", "chance_detected_frac", "anticipation_ms",
                      "false_per_min"):
                r[f"{k}_offline"] = float(off[k].iloc[0]) if len(off) else np.nan
            release_rows.append({"pid": pid, **r})
        x3 = next((r for r in rel if r["task"] == tasks[0]), {})
        x3cols = {f"x3_{k}": x3.get(k, np.nan) for k in (
            "detected_frac", "chance_detected_frac", "anticipation_ms", "false_per_min",
            "detected_frac_offline", "anticipation_ms_offline")}
        lo = loso[(loso.pid == pid) & (loso.model == model)] if not loso.empty else loso
        for f in fsets:
            Xs, _ = dc.stream_matrix(res.X_emg, res.X_imu, res.names, f)
            est0, params, fit0, hyper = base[(pid, f)]
            t_dec = w["t_dec_us"] if "imu" in feature_set_blocks(f) else w["t_dec_emg_us"]
            off = lo[lo.feature_set == f]
            offd = {k: float(off[k].iloc[0]) if len(off) and k in off else np.nan
                    for k in ("balanced_accuracy", "latency_ms_median", "fa_per_min",
                              "free_balanced_accuracy")}
            base_raw = None
            for mode in cp.modes:
                mp = dc.plan(mode, t_end, phase, y, dec, cp)
                row = {**common, "feature_set": f, "model": model, "mode": mode,
                       "applicable": mp.applicable, "hyper": hyper, "params": str(params),
                       "fit0_s": fit0,
                       "n_cal": int(mp.cal[-1].size) if mp.cal else 0,
                       "n_models": 1 + len(mp.cal),
                       "cal_fit_s": float(sum(cal_res[(pid, f, mode, i)][1]
                                              for i in range(len(mp.cal))))}
                if not mp.applicable:
                    summary.append(row)
                    continue
                models = [est0] + [cal_res[(pid, f, mode, i)][0] for i in range(len(mp.cal))]
                raw = dc.decode(models, mp, Xs, t_end, dec)
                if mode == "none":
                    base_raw = raw
                s = dc.smooth(raw, raw >= 0, phase, sm)
                rows = np.flatnonzero(mp.portion & dec & (y >= 0))
                free = np.flatnonzero((phase == FREE_PHASE) & dec & (y >= 0))
                lat = mt.decision_latency(on, t_end, t_dec, s, phase,
                                          mp.portion & w["calibrated"], pre_s)
                lv = np.array([r["latency_ms"] for r in lat], float)
                wl = np.array(mt.window_latency(t_end, y, raw, phase, rows, pre_s), float)
                rest_rows = np.flatnonzero((phase == REST_PHASE) & dec
                                           & (mp.portion if mode != "none" else True))
                ev_idx = dc.onsets(s, phase)
                fa = mt.false_activations_free(ev_idx, t_end, y, phase,
                                               np.ones(len(res), bool), w["calibrated"], on,
                                               stride_s, float(pr["fa_guard_s"]))
                fin = lv[np.isfinite(lv)]
                row.update({
                    "n_eval": int(rows.size),
                    "ba_raw": mt.scores(y, raw, rows), "ba": mt.scores(y, s, rows),
                    "ba_free_raw": mt.scores(y, raw, free), "ba_free": mt.scores(y, s, free),
                    "ba_k0_same_rows": mt.scores(y, base_raw, rows) if base_raw is not None
                    else np.nan,
                    "n_onsets": int(len(lat)),
                    "detected_frac": float(np.mean(np.isfinite(lv))) if lv.size else np.nan,
                    "latency_ms_median": float(np.median(fin)) if fin.size else np.nan,
                    "latency_ms_q1": float(np.percentile(fin, 25)) if fin.size else np.nan,
                    "latency_ms_q3": float(np.percentile(fin, 75)) if fin.size else np.nan,
                    f"frac_le_{budget:g}ms": float(np.mean(fin <= budget)) if fin.size
                    else np.nan,
                    "frac_anticipatory": float(np.mean(fin <= 0)) if fin.size else np.nan,
                    "latency_win_ms_median": float(np.nanmedian(wl)) if np.isfinite(wl).any()
                    else np.nan,
                    "fa_per_min_rest": mt.rest_false_activations(t_end, s, rest_rows)
                    ["fa_per_min"],
                    "fa_per_min_rest_raw": mt.rest_false_activations(t_end, raw, rest_rows)
                    ["fa_per_min"],
                    **fa,
                    **{f"{k}_offline": v for k, v in offd.items()},
                    **x3cols})
                if mode == "none" and f == prim:
                    il = w["yhat_inloop"]
                    row["inloop_equal_batch"] = bool(np.array_equal(il, raw))
                summary.append(row)
                hits = {r["hit"] for r in lat if r["detected"]}
                kinds = mt.classify_events(ev_idx, hits, t_end, y, on, float(pr["fa_guard_s"]))
                lat_by_hit = {r["hit"]: r["latency_ms"] for r in lat if r["detected"]}
                for j, kind in zip(ev_idx, kinds):
                    decisions.append({"pid": pid, "feature_set": f, "mode": mode,
                                      "t_us": int(t_dec[j]), "t_end_us": int(t_end[j]),
                                      "cls": int(s[j]), "phase": str(phase[j]),
                                      "y_true": int(y[j]), "kind": kind,
                                      "scored": bool(mp.portion[j] or phase[j] == FREE_PHASE),
                                      "latency_ms": lat_by_hit.get(int(j), np.nan)})
                if mode == "none" and f == prim:
                    arrays[f"online__{pid}__t_us"] = w["t_dec_us"].astype(np.int64)
                    arrays[f"online__{pid}__t_end_us"] = t_end.astype(np.int64)
                    arrays[f"online__{pid}__y"] = y.astype(np.int8)
                    arrays[f"online__{pid}__yhat"] = s.astype(np.int8)
                    arrays[f"online__{pid}__yhat_raw"] = raw.astype(np.int8)
                    arrays[f"online__{pid}__phase"] = np.array(
                        [PHASES.index(x) for x in phase], dtype=np.int8)
        arrays[f"online__{pid}__release_us"] = res.release_us.astype(np.int64)

    return StageResult(tables={"summary": pd.DataFrame(summary),
                               "decisions": pd.DataFrame(decisions),
                               "release": pd.DataFrame(release_rows),
                               "feature_deviation": pd.DataFrame(dev_rows)},
                       arrays=arrays)

