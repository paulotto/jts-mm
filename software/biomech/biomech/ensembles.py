# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trial ensembles and their aggregation: the data behind every "perspective".

An :class:`Ensemble` holds trials (cycles, events, ...) on a common axis --

* ``cycle``: 0-100 % of the movement cycle (101 points),
* ``angle``: the opening-angle grid (coupling curves),
* ``time``: ms from the cycle start (absolute time, NaN-padded),
* ``event``: ms around an onset (mechanical or EMG),
* ``path``: a cycle's 3-D landmark path, cycle-normalized (101 x 3),

as ``data`` (trials, samples, channels) with one row per trial in ``trials``
(pid, task, side, cycle, ok, flags, ...). :func:`aggregate` turns it into the
summaries the views draw, at one of three levels:

* ``trials``: no aggregation (the raw ensemble),
* ``participant``: per participant over its *ok* trials,
* ``cohort``: over the participant summaries -- never over pooled trials, so
  every participant weighs the same; the pooled within-participant SD is
  returned alongside (between- vs within-participant variability, S2).

Statistics: ``mean_sd``, ``median_iqr`` and ``mean_ci`` (bootstrap 95 % CI of
the mean over participants or trials).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

Stat = Literal["mean_sd", "median_iqr", "mean_ci"]
Level = Literal["trials", "participant", "cohort"]


@dataclass
class Ensemble:
    name: str
    grid: np.ndarray                      # (S,)
    data: np.ndarray                      # (K, S, C)
    trials: pd.DataFrame                  # K rows: pid, task, side, cycle, ok, ...
    channels: list[str]
    align: str = "cycle"
    unit: str = ""                        # siunitx unit of the values
    grid_unit: str = "\\percent"

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, float)
        if self.data.ndim == 2:
            self.data = self.data[:, :, None]
        if len(self.trials) != self.data.shape[0]:
            raise ValueError(f"{self.name}: {len(self.trials)} trial rows for "
                             f"{self.data.shape[0]} trials")
        self.trials = self.trials.reset_index(drop=True)
        if "ok" not in self.trials:
            self.trials["ok"] = True

    @property
    def n_trials(self) -> int:
        return int(self.data.shape[0])

    def select(self, mask: np.ndarray | pd.Series) -> "Ensemble":
        m = np.asarray(mask, bool)
        return Ensemble(self.name, self.grid, self.data[m], self.trials[m].copy(),
                        self.channels, self.align, self.unit, self.grid_unit)

    def channel(self, name: str) -> "Ensemble":
        i = self.channels.index(name)
        return Ensemble(self.name, self.grid, self.data[:, :, i:i + 1], self.trials.copy(),
                        [name], self.align, self.unit, self.grid_unit)

    def ok(self) -> "Ensemble":
        return self.select(self.trials.ok.astype(bool).to_numpy())

    def pids(self) -> list[str]:
        return sorted(self.trials.pid.unique())


@dataclass
class Summary:
    """``center``/``lo``/``hi`` are (G, S, C) for G groups (participants, or 1 for the cohort)."""

    level: str
    stat: str
    grid: np.ndarray
    groups: list[str]
    center: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: np.ndarray                          # (G,) trials (participant) or participants (cohort)
    n_trials: np.ndarray                   # (G,) trials behind each group
    members: np.ndarray | None = None      # cohort: (P, S, C) participant centres
    within_sd: np.ndarray | None = None    # cohort: (S, C) pooled within-participant SD
    channels: list[str] = field(default_factory=list)


def _center(x: np.ndarray, stat: str, rng: np.random.Generator, n_boot: int
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and spread over axis 0, NaN-aware."""
    with np.errstate(invalid="ignore"), _quiet():
        if stat == "median_iqr":
            c = np.nanmedian(x, axis=0)
            lo, hi = np.nanpercentile(x, [25, 75], axis=0)
            return c, lo, hi
        c = np.nanmean(x, axis=0)
        if stat == "mean_sd":
            sd = np.nanstd(x, axis=0, ddof=1) if x.shape[0] > 1 else np.full(c.shape, np.nan)
            return c, c - sd, c + sd
        if x.shape[0] < 2:
            return c, np.full(c.shape, np.nan), np.full(c.shape, np.nan)
        idx = rng.integers(0, x.shape[0], (n_boot, x.shape[0]))
        boots = np.stack([np.nanmean(x[i], axis=0) for i in idx])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        return c, lo, hi


class _quiet:
    def __enter__(self):
        import warnings

        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
        return self

    def __exit__(self, *exc):
        return self._w.__exit__(*exc)


def aggregate(ens: Ensemble, level: Level = "cohort", stat: Stat = "mean_sd", *,
              ok_only: bool = True, seed: int = 0, n_boot: int = 1000,
              participant_stat: Stat | None = None, min_frac: float = 0.5) -> Summary:
    """Summaries at ``level``; cohort statistics run over participant centres.

    ``participant_stat`` sets the per-participant centre for the cohort level
    (default: mean for mean_* statistics, median for median_iqr).
    """
    e = ens.ok() if ok_only else ens
    rng = np.random.default_rng(seed)
    pids = e.pids()
    S, C = e.data.shape[1], e.data.shape[2]
    if level == "trials":
        return Summary("trials", "none", e.grid, [str(i) for i in range(e.n_trials)], e.data,
                       e.data, e.data, np.ones(e.n_trials, int), np.ones(e.n_trials, int),
                       channels=e.channels)
    pc, plo, phi, pn = [], [], [], []
    pstat = participant_stat or ("median_iqr" if stat == "median_iqr" else "mean_sd")
    var = []
    for pid in pids:
        x = e.data[(e.trials.pid == pid).to_numpy()]
        c, lo, hi = _center(x, pstat if level == "cohort" else stat, rng, n_boot)
        pc.append(c)
        plo.append(lo)
        phi.append(hi)
        pn.append(x.shape[0])
        with _quiet():
            var.append(np.nanvar(x, axis=0, ddof=1) if x.shape[0] > 1 else np.full((S, C), np.nan))
    if not pids:
        empty = np.full((0, S, C), np.nan)
        return Summary(level, stat, e.grid, [], empty, empty, empty, np.zeros(0, int),
                       np.zeros(0, int), channels=e.channels)
    PC = np.stack(pc)
    if level == "participant":
        return Summary("participant", stat, e.grid, pids, PC, np.stack(plo), np.stack(phi),
                       np.array(pn), np.array(pn), channels=e.channels)
    c, lo, hi = _center(PC, stat, rng, n_boot)
    # A cohort value needs at least half of the participants (never one person's curve).
    n_at = np.isfinite(PC).sum(axis=0)
    thin = n_at < max(2, int(np.ceil(min_frac * len(pids))))
    c, lo, hi = (np.where(thin, np.nan, v) for v in (c, lo, hi))
    n_part = np.sum(np.isfinite(PC).any(axis=(1, 2)))
    with _quiet():
        within = np.sqrt(np.nanmean(np.stack(var), axis=0))
    return Summary("cohort", stat, e.grid, ["cohort"], c[None], lo[None], hi[None],
                   np.array([n_part]), np.array([int(np.sum(pn))]), members=PC,
                   within_sd=within, channels=e.channels)


def variance_decomposition(ens: Ensemble) -> dict[str, np.ndarray]:
    """Pointwise between-participant variance of participant means vs pooled within."""
    s = aggregate(ens, "cohort", "mean_sd")
    with _quiet():
        between = np.nanvar(s.members, axis=0, ddof=1)
    within = s.within_sd ** 2
    return {"between": between, "within": within, "ratio": between / within}


def describe_perspective(level: str, align: str, stat: str, n_part: int, n_trials: int) -> str:
    """Caption fragment, e.g. 'participant means, mean ± SD (n = 14 participants, 160 cycles)'."""
    what = {"trials": "individual cycles", "trials+summary": "individual cycles with",
            "participant": "per-participant", "cohort": "cohort (participant means)",
            "compare": "small multiples"}.get(level, level)
    st = {"mean_sd": "mean ± SD", "median_iqr": "median and IQR",
          "mean_ci": "mean and 95 % bootstrap CI"}.get(stat, "")
    ax = {"cycle": "cycle-normalized", "angle": "on the opening-angle grid", "time":
          "absolute time", "event": "event-aligned", "path": "cycle paths"}.get(align, align)
    return f"{what} {st}, {ax} (n = {n_part} participants, {n_trials} cycles)".replace("  ", " ")


# -- builders (from the ResultStore through a DataContext) ------------------------------------
KIN_SIGNALS: dict[str, tuple[str, list[str], str]] = {
    # name -> (array suffix, channels, unit)
    "theta": ("theta_deg", ["opening angle"], "\\degree"),
    "theta_vel": ("theta_vel", ["angular velocity"], "\\degree\\per\\second"),
    "theta_acc": ("theta_acc", ["angular acceleration"], "\\degree\\per\\second\\squared"),
    "incisor": ("incisor", ["x", "y", "z"], "\\milli\\metre"),
    "incisor_disp": ("incisor", ["x", "y", "z"], "\\milli\\metre"),
    "incisor_vel": ("incisor_vel", ["x", "y", "z"], "\\milli\\metre\\per\\second"),
    "incisor_speed": ("incisor_vel", ["speed"], "\\milli\\metre\\per\\second"),
    "incisor_acc": ("incisor_acc", ["magnitude"], "\\milli\\metre\\per\\second\\squared"),
    "tmj_left": ("tmj_left", ["x", "y", "z"], "\\milli\\metre"),
    "tmj_right": ("tmj_right", ["x", "y", "z"], "\\milli\\metre"),
}
N_CYCLE = 101


def _kin_values(a: dict[str, np.ndarray], task: str, signal: str) -> np.ndarray | None:
    suffix = KIN_SIGNALS[signal][0]
    v = a.get(f"{task}__{suffix}")
    if v is None:
        return None
    v = np.asarray(v, float)
    if v.ndim == 1:
        v = v[:, None]
    if signal in ("incisor_speed", "incisor_acc"):
        v = np.linalg.norm(v, axis=1)[:, None]
    return v


def _time_normalize(seg: np.ndarray, n: int) -> np.ndarray:
    from biomech.kinematics.cycles import time_normalize

    return time_normalize(seg, n)


def kinematic(ctx, task: str, signal: str = "theta", align: str = "cycle", *,
              max_deg: float = 40.0, step_deg: float = 1.0, dt_ms: float = 25.0) -> Ensemble:
    """Kinematic ``signal`` of every detected cycle of ``task`` (ok and flagged).

    ``align``: cycle (0-100 %), time (ms from start), angle (opening half on a
    deg grid), path (alias of cycle for 3-D landmark signals).
    """
    cyc = ctx.table("kinematics", "cycles")
    cyc = cyc[cyc.task == task] if not cyc.empty else cyc
    rows, data = [], []
    max_len = 0
    segs: list[tuple[pd.Series, np.ndarray, np.ndarray, np.ndarray | None]] = []
    for pid in ctx.pids():
        a = ctx.arrays("kinematics", pid)
        t = a.get(f"{task}__t_us")
        v = _kin_values(a, task, signal)
        th = a.get(f"{task}__theta_deg")
        if t is None or v is None:
            continue
        for _, c in cyc[cyc.pid == pid].iterrows():
            # Movement onset..offset when known (cycle start/end lie in the flat rest).
            t0 = c.get("t_on_us", c.t_start_us) if align in ("cycle", "path", "time") else \
                c.t_start_us
            t1 = c.get("t_off_us", c.t_end_us) if align in ("cycle", "path", "time") else \
                c.t_end_us
            t0 = c.t_start_us if not np.isfinite(t0) else t0
            t1 = c.t_end_us if not np.isfinite(t1) else t1
            sel = (t >= t0) & (t <= t1)
            if sel.sum() < 3:
                continue
            seg = v[sel]
            if signal == "incisor_disp":
                seg = seg - seg[0]
            segs.append((c, t[sel], seg, None if th is None else np.asarray(th, float)[sel]))
            max_len = max(max_len, int(sel.sum()))
    if align in ("cycle", "path"):
        grid = np.linspace(0, 100, N_CYCLE)
        for c, _, seg, _ in segs:
            data.append(_time_normalize(seg, N_CYCLE))
            rows.append(c)
        gunit = "\\percent"
    elif align == "time":
        grid = np.arange(max_len) * dt_ms
        for c, tt, seg, _ in segs:
            g = (tt - tt[0]) * 1e-3
            out = np.full((grid.size, seg.shape[1]), np.nan)
            for j in range(seg.shape[1]):
                out[:, j] = np.interp(grid, g, seg[:, j], left=np.nan, right=np.nan)
            data.append(out)
            rows.append(c)
        gunit = "\\milli\\second"
    elif align == "angle":
        grid = np.arange(0.0, max_deg + 1e-9, step_deg)
        for c, tt, seg, th in segs:
            if th is None:
                continue
            k = int(np.searchsorted(tt, c.t_peak_us))
            th_o, s_o = th[:k + 1], seg[:k + 1]
            ok = np.isfinite(th_o)
            out = np.full((grid.size, seg.shape[1]), np.nan)
            if ok.sum() >= 3:
                order = np.argsort(th_o[ok])
                for j in range(seg.shape[1]):
                    out[:, j] = np.interp(grid, th_o[ok][order], s_o[ok][:, j][order],
                                          left=np.nan, right=np.nan)
            data.append(out)
            rows.append(c)
        gunit = "\\degree"
    else:
        raise ValueError(f"unknown align {align!r}")
    _, chans, unit = KIN_SIGNALS[signal]
    trials = pd.DataFrame([{"pid": r.pid, "task": r.task, "side": r.get("side", ""),
                            "cycle": int(r.cycle), "ok": bool(r.ok),
                            "flags": r.get("flags", ""), "t_start_us": int(r.get("t_on_us", r.t_start_us)),
                            "duration_s": float(r.get("duration_s", np.nan))} for r in rows])
    D = np.stack(data) if data else np.zeros((0, grid.size, len(chans)))
    if trials.empty:
        trials = pd.DataFrame(columns=["pid", "task", "side", "cycle", "ok"])
    return Ensemble(f"{task}/{signal}", grid, D, trials, list(chans), align, unit, gunit)


def emg_profiles(ctx, task: str) -> Ensemble:
    """Cycle-normalized %MVC/%RVC profiles of every muscle (emg stage arrays)."""
    cyc = ctx.table("kinematics", "cycles")
    rows, data, names = [], [], None
    for pid in ctx.pids():
        a = ctx.arrays("emg", pid)
        P = a.get(f"profile__{task}__pct")
        if P is None:
            continue
        nm = [str(n) for n in a.get("profile__names", [])] or None
        names = names or nm
        if nm is not None and names is not None and nm != names:   # align channels by name
            idx = [nm.index(m) if m in nm else -1 for m in names]
            P = np.stack([P[:, :, i] if i >= 0 else np.full(P.shape[:2], np.nan)
                          for i in idx], axis=2)
        ids = a.get(f"profile__{task}__cycle", np.arange(len(P)))
        c = cyc[(cyc.pid == pid) & (cyc.task == task)].set_index("cycle")
        for k, cid in enumerate(ids):
            r = c.loc[int(cid)] if int(cid) in c.index else None
            rows.append({"pid": pid, "task": task, "cycle": int(cid),
                         "ok": bool(r.ok) if r is not None else True,
                         "side": r.side if r is not None else ""})
            data.append(P[k])
    names = names or []
    D = np.stack(data) if data else np.zeros((0, N_CYCLE, len(names)))
    trials = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["pid", "task", "cycle", "ok"])
    return Ensemble(f"{task}/emg", np.linspace(0, 100, N_CYCLE), D, trials, names, "cycle",
                    "\\percent")


def event_aligned(ctx, task: str, direction: str = "opening", source: str = "theta",
                  window_ms: tuple[float, float] = (-500.0, 500.0), step_ms: float = 10.0,
                  muscles: list[str] | None = None) -> Ensemble:
    """Signals around the mechanical onsets of the timing stage (one per cycle).

    ``source``: ``theta`` / ``theta_vel`` (kinematics) or ``emg`` (%MVC envelopes).
    """
    on = ctx.table("timing", "onsets")
    grid = np.arange(window_ms[0], window_ms[1] + 1e-9, step_ms)
    rows, data, chans = [], [], None
    if on.empty:
        return Ensemble(f"{task}/{direction}/{source}", grid, np.zeros((0, grid.size, 1)),
                        pd.DataFrame(columns=["pid", "ok"]), [source], "event")
    ev = (on[(on.task == task) & (on.direction == direction)]
          .drop_duplicates(["pid", "cycle"])[["pid", "cycle", "side", "t_mech_us"]])
    for pid, g in ev.groupby("pid"):
        if source == "emg":
            a = ctx.arrays("emg", pid)
            t = a.get(f"{task}__t_us")
            v = a.get(f"{task}__norm_pct")
            refs = ctx.table("emg", "refs", [pid])
            names = list(refs.muscle) if not refs.empty else []
            if muscles:
                idx = [names.index(m) for m in muscles if m in names]
                v = None if v is None else np.asarray(v, float)[:, idx]
                names = [names[i] for i in idx]
            chans = chans or names
        else:
            a = ctx.arrays("kinematics", pid)
            t = a.get(f"{task}__t_us")
            v = _kin_values(a, task, source)
            chans = chans or KIN_SIGNALS[source][1]
        if t is None or v is None:
            continue
        v = np.asarray(v, float)
        if v.ndim == 1:
            v = v[:, None]
        for _, r in g.iterrows():
            tt = (np.asarray(t, float) - r.t_mech_us) * 1e-3
            out = np.column_stack([np.interp(grid, tt, v[:, j], left=np.nan, right=np.nan)
                                   for j in range(v.shape[1])])
            data.append(out)
            rows.append({"pid": pid, "task": task, "cycle": int(r.cycle), "side": r.side,
                         "ok": True})
    chans = chans or [source]
    D = np.stack(data) if data else np.zeros((0, grid.size, len(chans)))
    unit = "\\percent" if source == "emg" else KIN_SIGNALS[source][2]
    trials = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["pid", "ok"])
    return Ensemble(f"{task}/{direction}/{source}", grid, D, trials, list(chans), "event", unit,
                    "\\milli\\second")


def profile_ensembles(ctx, task: str) -> tuple[Ensemble, Ensemble]:
    """EMG cycle profiles and the opening angle stored with them (identical cycles).

    The profiles are cut at the cycle start..end (the pre-specified SPM cycles); plot
    kinematics against them with this angle, not with ``kinematic(..., "cycle")``, which
    cuts at movement onset..offset.
    """
    emg = emg_profiles(ctx, task)
    data = []
    for pid in ctx.pids():
        a = ctx.arrays("emg", pid)
        if f"profile__{task}__pct" not in a:
            continue
        th = a.get(f"profile__{task}__theta_deg")
        n = len(a[f"profile__{task}__pct"])
        data += list(np.asarray(th, float)) if th is not None else [np.full(N_CYCLE, np.nan)] * n
    D = np.stack(data)[:, :, None] if data else np.zeros((0, N_CYCLE, 1))
    theta = Ensemble(f"{task}/theta", emg.grid, D, emg.trials.copy(), ["theta"], "cycle", "\\degree")
    return emg, theta
