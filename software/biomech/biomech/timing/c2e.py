# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""C2-E: does EMG precede jaw motion? The post hoc family X6, per
participant. The pre-specified C2 (``timing.lead``) is unchanged.

Per task (open_close, chewing) and direction (closing: the closers, masseter + temporalis, the
primary; opening: the digastrics, the supporting re-test):

* **Mechanical events (B1):** t_omega, the relative-quaternion angular-velocity onset of
  :func:`biomech.timing.lead.mechanical_onsets` (no |a_rel|); t_alpha, the signed
  angular-acceleration onset (15 Hz primary, 10 / 20 Hz sensitivity); t_opt, the optical onset
  (cross-check). Without IMU arrays the windows are anchored at t_opt (``event_source``
  opt) and only the optical row exists. This is a defensive path: the only participant
  without a jaw IMU, never reaches it -- its ``imu`` unit fails, the pipeline skips its
  ``timing`` unit, and that participant has no C2-E rows.
* **Onsets:** TKEO, drive_taskbase, offline (:meth:`WindowOnsets.restricted`), in the window
  [t_event - ``emg_pre_s``, t_event + ``emg_post_s``] with the same gates as C2; the variance
  changepoint (:mod:`biomech.timing.changepoint`, P6) in the same window; the symmetric window
  +-``symmetric_window_s`` (the fallback statistic).
* **S1 (B2):** ``n_surrogates`` whole-phase circular shifts per task
  (:mod:`biomech.timing.surrogates`, seed ``null_seed(seed, "c2e:<task>")``); every real cycle
  is re-evaluated in every surrogate with the same windows and gates (index arithmetic,
  :meth:`WindowOnsets.batch`). One shift serves both directions, every source and the release.
* **Per-participant statistics** (:mod:`biomech.timing.locking`): locking (IQR vs S1), location
  (mixture mu), the median and its S1 null (fallback), S2 locking for the primary source; per
  scope (each task and ``pooled`` = both tasks, H1).
* **P3 release** (:func:`biomech.timing.ensemble.release_s1`): the X2 closer release re-aligned
  to the opening t_omega, its depth against S1, the per-cycle release IQR against S1, and the
  Theil-Sen slope against S2 (anchor = end of the previous closing).
* **P5 coupling** (:mod:`biomech.timing.coupling`).

Tables: ``c2e_cycles`` (observed channel-cycles), ``c2e_null`` (per-participant statistics
with their surrogate nulls), ``c2e_release`` (per-cycle release), ``coupling`` (P5). Units: ms,
µs for absolute times; lead > 0 = EMG first. Every row that could not be computed says why
(``reason`` / ``flag``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

import numpy as np
import pandas as pd

from biomech import timebase
from biomech.emg.prepare import EmgPrep
from biomech.io.session import CLOSERS, OPENERS
from biomech.timing import changepoint, coupling, ensemble, locking, surrogates
from biomech.timing import lead as L

GROUPS = {"closing": ("closers", CLOSERS), "opening": ("openers", OPENERS)}
# Keys of the same yaml block read by the cohort stage (stats.cohort_c2e.CohortParams).
COHORT_KEYS = ("alpha", "gate_power_40ms", "s2_min_sd_ms", "gates_passed", "supporting_veto")
SCOPE_POOLED = "pooled"


@dataclass(frozen=True)
class C2EParams:
    """``stages.timing.c2e`` (etc/biomech.yaml; every value documented there)."""

    enabled: bool = True
    tasks: tuple[str, ...] = ("open_close", "chewing")
    n_surrogates: int = 200
    gap_s: float = 1.0
    min_usable: int = 5
    flag_usable: int = 3
    alpha_cutoff_hz: float = L.ALPHA_CUTOFF_HZ
    alpha_order: int = L.ALPHA_ORDER
    alpha_frac: float = L.ALPHA_FRAC
    alpha_guard_s: float = L.ALPHA_GUARD_S
    alpha_sensitivity_hz: tuple[float, ...] = (10.0, 20.0)
    symmetric_window_s: float | None = 0.4
    changepoint: bool = True
    cp_min_seg_s: float = 0.05
    n_s2: int = 200
    delta_ms: float | None = None
    # What is computed (all on for the analysis; the synthetic gates switch off what a gate
    # does not read, for speed): directions, the re-referenced rows (alpha, optical), P3, P5.
    directions: tuple[str, ...] = ("closing", "opening")
    reref: bool = True
    release: bool = True
    p5: bool = True
    mixture: locking.MixtureParams = field(default_factory=locking.MixtureParams)
    coupling: coupling.CouplingParams = field(default_factory=coupling.CouplingParams)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "C2EParams":
        d = {k: v for k, v in dict(d or {}).items() if k not in COHORT_KEYS}
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown stages.timing.c2e keys {sorted(unknown)}")
        for k in ("tasks", "alpha_sensitivity_hz", "directions"):
            if k in d:
                d[k] = tuple(d[k])
        d["mixture"] = locking.MixtureParams.from_dict(d.get("mixture"))
        d["coupling"] = coupling.CouplingParams.from_dict(d.get("coupling"))
        return cls(**d)


def _alpha_key(hz: float) -> str:
    return f"alpha{hz:g}"


# -- mechanical events per cycle -----------------------------------------------------------
def cycle_events(imu_arrays: dict[str, Any], kin_arrays: dict[str, Any] | None,
                 cyc: pd.DataFrame, task: str, p: dict[str, Any], c: C2EParams,
                 base: dict[str, float]) -> tuple[pd.DataFrame, str]:
    """Per kinematic cycle (``ok`` cycles of the task, in time order) the mechanical events
    of both directions (µs; NaN = none): ``{dir}_t_omega``, ``{dir}_t_alpha`` (+ the
    sensitivity cutoffs), ``{dir}_t_opt``, and the S2 anchors ``opening_anchor`` (end of the
    previous closing: ``prev_end_us`` of ``cyc``, the cycle just before including rejected
    ones; without that column the previous row) / ``closing_anchor`` (the opening onset). Returns (table, event
    source): ``omega`` with IMU arrays, ``opt`` from the optical angle without them, ``""``
    with neither."""
    rows: list[dict[str, Any]] = []
    have_imu = f"{task}__t_us" in imu_arrays
    have_opt = kin_arrays is not None and f"{task}__t_us" in kin_arrays
    if not (have_imu or have_opt):
        return pd.DataFrame(rows), ""
    if have_imu:
        t = imu_arrays[f"{task}__t_us"].astype(np.int64)
        om = imu_arrays[f"{task}__omega_dps"].astype(np.float64)
        ac = imu_arrays[f"{task}__acc_rel_ms2"].astype(np.float64)
        tho = imu_arrays[f"{task}__theta_opt"].astype(np.float64)
        thq = imu_arrays.get(f"{task}__theta_quat")
        alphas = {}
        if thq is not None:
            for hz in (c.alpha_cutoff_hz, *c.alpha_sensitivity_hz):
                alphas[_alpha_key(hz)] = L.alpha_signal(t, np.asarray(thq, np.float64), hz,
                                                        c.alpha_order)
    else:
        t = kin_arrays[f"{task}__t_us"].astype(np.int64)
        tho = kin_arrays[f"{task}__theta_deg"].astype(np.float64)
        om = ac = np.full(t.size, np.nan)
        alphas = {}
    pa = {**p, "c2e": {"alpha_frac": c.alpha_frac, "alpha_guard_s": c.alpha_guard_s}}
    prev_end = np.nan
    for _, cy in cyc.iterrows():
        r: dict[str, Any] = {"cycle": int(cy.cycle), "t_start_us": float(cy.t_start_us),
                             "t_end_us": float(cy.t_end_us),
                             "opening_anchor": float(cy.prev_end_us) if "prev_end_us" in cy
                             else prev_end}
        t_open = np.nan
        for direction, (a0, a1) in (("opening", (cy.t_start_us - p["pre_s"] * 1e6,
                                                 cy.t_peak_us)),
                                    ("closing", (cy.t_peak_us - 0.1e6, cy.t_end_us))):
            mech = {}
            for key, al in (alphas.items() or [("", None)]):
                mk = L.mechanical_onsets(t, om, ac, tho, int(a0), int(a1), direction, base, pa,
                                         alpha=al, alpha_t0=t_open)
                if not mech:
                    mech = mk
                if key:
                    r[f"{direction}_t_{key}"] = mk["t_alpha"]
            r[f"{direction}_t_omega"] = mech["t_omega"] if have_imu else np.nan
            r[f"{direction}_t_opt"] = mech["t_opt"]
            if direction == "opening":
                t_open = mech["t_omega"] if have_imu else mech["t_opt"]
        r["closing_anchor"] = t_open
        prev_end = float(cy.t_end_us)
        rows.append(r)
    keys = [_alpha_key(h) for h in (c.alpha_cutoff_hz, *c.alpha_sensitivity_hz)]
    cols = [f"{d}_t_{k}" for d in ("opening", "closing") for k in ("omega", "opt", *keys)]
    out = pd.DataFrame(rows)
    for col in cols:                                   # absent sources: NaN columns
        if col not in out:
            out[col] = np.nan
    return out, ("omega" if have_imu else "opt")


# -- windows and surrogate onsets ------------------------------------------------------------
@dataclass
class _Eval:
    """Leads (ms) of one task x direction x window: arrays (S + 1, cycles, channels); row 0
    is the observed EMG, rows 1.. the S1 surrogates."""

    lead: np.ndarray
    usable: np.ndarray
    k_on: np.ndarray
    burst_db: np.ndarray
    active: np.ndarray


def _evaluate(det: L.WindowOnsets, x: np.ndarray, cleaned: np.ndarray, mask: np.ndarray,
              t_us: np.ndarray, fs: float, t_ev: np.ndarray, shifts: np.ndarray,
              pre_s: float, post_s: float, muscles: list[str]) -> _Eval:
    """Onsets of every (shift, cycle) window for the ``muscles`` (columns, in order)."""
    N = len(t_us)
    n_pre, n_post = int(round(pre_s * fs)), int(round(post_s * fs))
    off = np.arange(-n_pre, n_post + 1)
    i_ref = np.searchsorted(t_us, t_ev)
    S, C, K = len(shifts), len(t_ev), len(muscles)
    lead = np.full((S, C, K), np.nan)
    usable = np.zeros((S, C, K), bool)
    k_on = np.full((S, C, K), -1, np.int64)
    bdb = np.full((S, C, K), np.nan)
    act = np.zeros((S, C, K), bool)
    if C == 0 or S == 0:
        return _Eval(lead, usable, k_on, bdb, act)
    _, sel = surrogates.shifted_windows(np.tile(i_ref, S), np.repeat(shifts, C), off, N)
    res = det.batch(x, cleaned, mask, sel, n_pre, tuple(muscles))
    frac = (t_ev - t_us[np.clip(i_ref, 0, N - 1)]) / 1e3           # t_event - window ref (ms)
    for j, m in enumerate(muscles):
        if m not in res:
            continue
        r = res[m]
        k = r["k_on"].reshape(S, C)
        k_on[:, :, j] = k
        usable[:, :, j] = r["usable"].reshape(S, C)
        bdb[:, :, j] = r["burst_db"].reshape(S, C)
        act[:, :, j] = r["active"].reshape(S, C)
        lead[:, :, j] = np.where(k >= 0, frac[None, :] + (n_pre - k) * 1e3 / fs, np.nan)
    return _Eval(lead, usable, k_on, bdb, act)


def _changepoints(x: np.ndarray, t_us: np.ndarray, fs: float, t_ev: np.ndarray,
                  shifts: np.ndarray, pre_s: float, post_s: float, cols: list[int],
                  min_seg_s: float, min_db: float, max_cells: int = 2_000_000
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P6 leads (ms), usability and ratio (dB), (S + 1, cycles, channels), in the windows of
    :func:`_evaluate`. Usable: a variance increase of >= ``min_db`` and > 90 % valid
    samples."""
    N = len(t_us)
    n_pre, n_post = int(round(pre_s * fs)), int(round(post_s * fs))
    off = np.arange(-n_pre, n_post + 1)
    i_ref = np.searchsorted(t_us, t_ev)
    S, C, K = len(shifts), len(t_ev), len(cols)
    lead = np.full((S, C, K), np.nan)
    use = np.zeros((S, C, K), bool)
    rdb = np.full((S, C, K), np.nan)
    if C == 0 or S == 0 or K == 0:
        return lead, use, rdb
    _, sel = surrogates.shifted_windows(np.tile(i_ref, S), np.repeat(shifts, C), off, N)
    frac = (t_ev - t_us[np.clip(i_ref, 0, N - 1)]) / 1e3
    step = max(max_cells // off.size, 1)
    for j, ci in enumerate(cols):
        k = np.full(len(sel), -1, np.int64)
        db = np.full(len(sel), np.nan)
        vf = np.zeros(len(sel))
        for a in range(0, len(sel), step):
            X = x[sel[a:a + step], ci]
            k[a:a + step], _, db[a:a + step] = changepoint.variance_changepoints(X, fs,
                                                                                min_seg_s)
            vf[a:a + step] = np.isfinite(X).mean(axis=1)
        k, db, vf = k.reshape(S, C), db.reshape(S, C), vf.reshape(S, C)
        lead[:, :, j] = np.where(k >= 0, frac[None, :] + (n_pre - k) * 1e3 / fs, np.nan)
        rdb[:, :, j] = db
        use[:, :, j] = (k >= 0) & (db >= min_db) & (vf > 0.9)
    return lead, use, rdb


# -- per-participant statistics --------------------------------------------------------------
def _flag(n: int, c: C2EParams) -> str:
    return "ok" if n >= c.min_usable else ("few_usable" if n >= c.flag_usable else "too_few")


def _stats(lead: np.ndarray, usable: np.ndarray, lo: float, hi: float, c: C2EParams
           ) -> dict[str, Any]:
    """Locking, location (mixture) and median vs S1 of leads (S + 1, cycles, channels)."""
    obs = lead[0][usable[0]]
    null = [lead[s][usable[s]] for s in range(1, lead.shape[0])]
    pooled = np.concatenate(null) if null else np.empty(0)
    n_cc = int(lead[0].size)
    out: dict[str, Any] = {"n_channel_cycles": n_cc, "n_usable": int(obs.size),
                           "usable_frac": float(obs.size / n_cc) if n_cc else np.nan,
                           "usable_frac_null": float(usable[1:].mean())
                           if usable[1:].size else np.nan,
                           "flag": _flag(int(obs.size), c)}
    out.update(locking.locking(obs, null, c.min_usable))
    mix = locking.mixture_location(obs, pooled, lo, hi, c.mixture, c.min_usable)
    out.update(mu_ms=mix["mu_ms"], sigma_ms=mix["sigma_ms"], pi=mix["pi"],
               loglik_gain=mix["loglik_gain"], mixture_reason=mix["reason"])
    med = float(np.median(obs)) if obs.size else np.nan
    mn = float(np.median(pooled)) if pooled.size else np.nan
    out.update(median_ms=med, median_null_ms=mn, median_minus_null_ms=med - mn)
    return out


def _null_row(pid: str, scope: str, direction: str, group: str, source: str, ev_src: str,
              n_cycles: int, st: dict[str, Any]) -> dict[str, Any]:
    return {"pid": pid, "scope": scope, "direction": direction, "group": group,
            "source": source, "event_source": ev_src, "n_cycles": n_cycles, **st}


def analyse(prep: EmgPrep, pid: str, imu_arrays: dict[str, Any], kin_cycles: pd.DataFrame,
            p: dict[str, Any], seed: int = 0, kin_arrays: dict[str, Any] | None = None
            ) -> dict[str, pd.DataFrame]:
    """The C2-E tables of one participant (module docstring). ``p`` = the ``stages.timing``
    block (``emg_pre_s``, ``emg_post_s``, ``burst_min_db``, ``ensemble`` and ``c2e``)."""
    c = C2EParams.from_dict(p.get("c2e"))
    det = L.build_detector(prep, p)[0].restricted()
    base = L.mechanical_baselines(imu_arrays, p)
    pre, post = float(p["emg_pre_s"]), float(p["emg_post_s"])
    lo, hi = -post * 1e3, pre * 1e3                      # lead range of the window (ms)
    ep = ensemble.EnsembleParams.from_dict(p.get("ensemble"))
    names = list(prep.names)
    rest = prep.phase("rest")
    rest_x = np.where(rest.mask, np.nan, rest.cleaned)
    rest_level = L._rest_level(rest_x, rest.mask, rest.fs, ep.envelope_s)
    cyc_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    coup_rows: list[dict[str, Any]] = []
    pooled: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray, float, float]]] = {}
    pooled_n: dict[tuple[str, str], int] = {}
    ev_sources: set[str] = set()
    for task in c.tasks:
        # the S2 opening anchor is the end of the cycle just before (rejected or not), so
        # every cycle of the task gives its end before the ok cycles are kept
        if not kin_cycles.empty:
            allc = kin_cycles[kin_cycles.task == task].sort_values("t_start_us")
            allc = allc.assign(prev_end_us=allc.t_end_us.shift(1))
            cyc = allc[allc.ok.astype(bool)]
        else:
            cyc = kin_cycles
        try:
            pe = prep.phase(task)
        except (KeyError, LookupError):
            pe = None
        if pe is None or cyc.empty or len(pe.t_us) < 10:
            null_rows.append(_null_row(pid, task, "", "", "", "", 0,
                                       {"flag": "no_phase_or_cycles"}))
            continue
        ev, ev_src = cycle_events(imu_arrays, kin_arrays, cyc, task, p, c, base)
        if not ev_src:
            null_rows.append(_null_row(pid, task, "", "", "", "", len(cyc),
                                       {"flag": "no_mechanical_signal"}))
            continue
        ev_sources.add(ev_src)
        fs = float(pe.fs)
        t_us = np.asarray(pe.t_us)
        x = np.where(pe.mask, np.nan, pe.cleaned)
        N = len(t_us)
        rng = surrogates.null_seed(seed, f"c2e:{task}")
        shifts = np.concatenate([[0], surrogates.s1_sample_shifts(rng, N, fs,
                                                                   c.n_surrogates, c.gap_s)])
        for direction in c.directions:
            gname, group = GROUPS[direction]
            muscles = [m for m in group if m in names]
            t_ev = ev[f"{direction}_t_{ev_src}"].to_numpy(float)
            i_ref = np.searchsorted(t_us, t_ev)
            # The observed windows (asymmetric and symmetric) must lie inside the phase; the
            # surrogates wrap around it (S1).
            w_pre = max(pre, c.symmetric_window_s or 0.0)
            w_post = max(post, c.symmetric_window_s or 0.0)
            inside = (np.isfinite(t_ev) & (i_ref - int(round(w_pre * fs)) >= 0)
                      & (i_ref + int(round(w_post * fs)) < N))
            e = ev[inside].reset_index(drop=True)
            te = t_ev[inside]
            r = _evaluate(det, x, pe.cleaned, pe.mask, t_us, fs, te, shifts, pre, post,
                          muscles)
            sources = {ev_src: (r.lead, r.usable, lo, hi)}
            # Re-referenced leads (P4 alpha, the optical cross-check): the same onsets.
            ref_cols = ([f"{direction}_t_{_alpha_key(h)}" for h in
                         (c.alpha_cutoff_hz, *c.alpha_sensitivity_hz)]
                        + [f"{direction}_t_opt"]) if ev_src == "omega" and c.reref else []
            for col in ref_cols:
                d = (e[col].to_numpy(float) - te) / 1e3
                key = col.split("_t_")[1]
                dl = d[None, :, None]
                fin = np.isfinite(d)
                lo_r = lo + (np.min(d[fin]) if fin.any() else 0.0)
                hi_r = hi + (np.max(d[fin]) if fin.any() else 0.0)
                sources[key] = (r.lead + dl, r.usable & np.isfinite(dl), lo_r, hi_r)
            if c.changepoint:
                cl, cu, _ = _changepoints(x, t_us, fs, te, shifts, pre, post,
                                            [names.index(m) for m in muscles],
                                            c.cp_min_seg_s, float(p["burst_min_db"]))
                sources["cp"] = (cl, cu, lo, hi)
            if c.symmetric_window_s:
                w = float(c.symmetric_window_s)
                rs = _evaluate(det, x, pe.cleaned, pe.mask, t_us, fs, te, shifts, w, w, muscles)
                sources["sym"] = (rs.lead, rs.usable, -w * 1e3, w * 1e3)
            # Observed channel-cycles.
            anchor = e[f"{direction}_anchor"].to_numpy(float)
            t_om = e[f"{direction}_t_omega"].to_numpy(float)
            t_al = e[f"{direction}_t_{_alpha_key(c.alpha_cutoff_hz)}"].to_numpy(float)
            t_op = e[f"{direction}_t_opt"].to_numpy(float)
            for k in range(len(te)):
                for j, m in enumerate(muscles):
                    row = {"pid": pid, "task": task, "direction": direction,
                           "cycle": int(e.cycle[k]), "muscle": m, "event_source": ev_src,
                           "t_event_us": te[k], "anchor_us": anchor[k],
                           "t_omega_us": t_om[k], "t_alpha_us": t_al[k], "t_opt_us": t_op[k],
                           "t_onset_us": te[k] - r.lead[0, k, j] * 1e3,
                           "burst_db": r.burst_db[0, k, j],
                           "active_at_start": bool(r.active[0, k, j]),
                           "usable": bool(r.usable[0, k, j]),
                           "reason": ("" if r.usable[0, k, j] else
                                      "no_onset" if r.k_on[0, k, j] < 0 else
                                      "active_at_start" if r.active[0, k, j] else
                                      "burst_below_min_or_masked")}
                    for key, (ld, us, _, _) in sources.items():
                        name = "lead_omega_ms" if key == "omega" else f"lead_{key}_ms"
                        row[name] = ld[0, k, j]
                        if key in ("cp", "sym"):
                            row[f"usable_{key}"] = bool(us[0, k, j])
                    cyc_rows.append(row)
            for key, (ld, us, a_lo, a_hi) in sources.items():
                st = _stats(ld, us, a_lo, a_hi, c)
                if key == ev_src and len(te) >= 3:
                    on_abs = te[:, None] - r.lead[0] * 1e3
                    st.update({k_: v for k_, v in locking.s2_iqr(
                        anchor, te, on_abs, r.usable[0],
                        surrogates.null_seed(seed, f"c2e:s2:{task}:{direction}"), c.n_s2,
                        c.min_usable).items() if k_ != "iqr_obs"})
                null_rows.append(_null_row(pid, task, direction, gname, key, ev_src, len(te),
                                           st))
                pooled.setdefault((direction, key), []).append((ld, us, a_lo, a_hi))
                pooled_n[(direction, key)] = pooled_n.get((direction, key), 0) + len(te)
            if direction == "opening" and c.release:
                rel_rows_t, rel_null = _release(pid, task, pe, t_us, fs, names, rest_level, ep,
                                                e, te, shifts, seed, c, ev_src)
                rel_rows += rel_rows_t
                null_rows += rel_null
        if ev_src == "omega" and c.p5:
            coup_rows.append(_coupling(pid, task, imu_arrays, pe, x, names, cyc, c, seed))
    # Pooled over tasks (H1): surrogate s of each task joined (independent shifts per task).
    src_pooled = "/".join(sorted(ev_sources))
    for (direction, key), parts in pooled.items():
        S = min(pt[0].shape[0] for pt in parts)
        ld = np.concatenate([pt[0][:S].reshape(S, -1) for pt in parts], axis=1)[:, :, None]
        us = np.concatenate([pt[1][:S].reshape(S, -1) for pt in parts], axis=1)[:, :, None]
        a_lo, a_hi = min(pt[2] for pt in parts), max(pt[3] for pt in parts)
        st = _stats(ld, us, a_lo, a_hi, c)
        null_rows.append(_null_row(pid, SCOPE_POOLED, direction, GROUPS[direction][0], key,
                                   src_pooled, pooled_n[(direction, key)], st))
    return {"c2e_cycles": pd.DataFrame(cyc_rows), "c2e_null": pd.DataFrame(null_rows),
            "c2e_release": pd.DataFrame(rel_rows), "coupling": pd.DataFrame(coup_rows)}


def _release(pid: str, task: str, pe: Any, t_us: np.ndarray, fs: float, names: list[str],
             rest_level: np.ndarray, ep: ensemble.EnsembleParams, e: pd.DataFrame,
             te: np.ndarray, shifts: np.ndarray, seed: int, c: C2EParams, ev_src: str
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """P3 rows: per-cycle release (``c2e_release``) and the participant rows (``c2e_null``,
    source ``release``) of one task, aligned to the window events ``te`` (t_omega; t_opt
    without IMU)."""
    db = ensemble.envelope_db(pe.cleaned, pe.mask, fs, rest_level, ep.envelope_s)
    i_ref = np.searchsorted(t_us, te)
    r = ensemble.release_s1(db, names, i_ref, fs, ep, shifts[1:])
    t_rel = te + r["cycle_t_ms"] * 1e3 if len(te) else np.empty(0)
    anchor = e["opening_anchor"].to_numpy(float) if len(e) else np.empty(0)
    rows = [{"pid": pid, "task": task, "cycle": int(e.cycle[k]), "event_source": ev_src,
             "t_event_us": te[k],
             "anchor_us": anchor[k], "t_release_us": t_rel[k],
             "lead_release_ms": -r["cycle_t_ms"][k], "change_db": r["cycle_change_db"][k]}
            for k in range(len(te))]
    lead_c = -np.asarray(r["cycle_t_ms"], float)
    null_c = [-np.asarray(v, float) for v in r["null_cycle_t_ms"]]
    st: dict[str, Any] = {"n_channel_cycles": len(te),
                          "n_usable": int(np.isfinite(lead_c).sum()),
                          "flag": _flag(int(np.isfinite(lead_c).sum()), c)}
    st.update(locking.locking(lead_c, null_c, c.min_usable))
    nd = np.asarray(r["null_depth_db"], float)
    fin = nd[np.isfinite(nd)]
    depth = float(r["depth_db"])
    dmed = float(np.median(fin)) if fin.size else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        ldr = float(np.log(depth / dmed)) if depth > 0 and dmed > 0 else np.nan
    st.update(median_ms=-float(r["t_ms"]), depth_db=depth, depth_null_median_db=dmed,
              log_depth_ratio=ldr, p_depth=surrogates.mc_p(fin, depth, "greater"))
    s2 = locking.s2_slope(anchor, te, t_rel, surrogates.null_seed(seed, f"c2e:s2rel:{task}"),
                          c.n_s2)
    st.update(b_s2=s2["b"], b_null_median=s2["b_null_median"], p_b_s2=s2["p_s2"],
              x_sd_ms=s2["x_sd_ms"])
    return rows, [_null_row(pid, task, "opening", "closers", "release", ev_src, len(te), st)]


def _coupling(pid: str, task: str, imu_arrays: dict[str, Any], pe: Any, x: np.ndarray,
              names: list[str], cyc: pd.DataFrame, c: C2EParams, seed: int) -> dict[str, Any]:
    """P5 row of one task (IMU grid; closers' rectified EMG against -alpha)."""
    row: dict[str, Any] = {"pid": pid, "task": task}
    t = imu_arrays[f"{task}__t_us"].astype(np.int64)
    thq = imu_arrays.get(f"{task}__theta_quat")
    cols = [names.index(m) for m in CLOSERS if m in names]
    if thq is None or not cols:
        return {**row, "reason": "no_theta_quat_or_closers"}
    alpha = timebase.gradient(t, timebase.gradient(t, np.asarray(thq, np.float64)))
    emg = coupling.emg_feature(np.asarray(pe.t_us), x[:, cols], t, float(pe.fs),
                               c.coupling.smooth_s)
    # Residualisation cycles run start to next start (the last to its end): truncating them
    # at the cycle end dropped the high-acceleration edges unevenly across lags and biased
    # the lag in the synthetic gate (G5); the real cycles are back to back anyway.
    st = cyc.t_start_us.to_numpy(float)
    en = np.r_[st[1:], cyc.t_end_us.to_numpy(float)[-1:]]
    res = coupling.coupling(t, emg, alpha, st, en, c.alpha_cutoff_hz, c.alpha_order,
                            c.coupling, surrogates.null_seed(seed, f"c2e:p5:{task}"),
                            occlusions_us=cyc.t_end_us.to_numpy(float))
    return {**row, **res}
