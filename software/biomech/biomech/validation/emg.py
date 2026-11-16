# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG comparability of JTS-MM and Delsys Trigno (plan D5, validation plan §2.4).

**Design (cross-session).** The comparison takes carry no usable JTS-MM EMG (only the
Delsys electrodes were on the skin). The JTS-MM side is therefore the *dataset take* of the
same subject and its MVC take (read by path from ``reference.yaml``, never from the
icorr2027 store); the Delsys side is the comparison take and its MVC CSVs. This is method
comparability, not sample-wise agreement:

* different days (s10: JTS-MM 10-13, Delsys 10-20, Delsys MVC 10-19; s14: 10-20 / 10-21 /
  10-20, the Delsys MVC about 1.5 h before the JTS-MM takes of that day, EV5);
* different electrode types (JTS-MM gelled snap electrodes, Delsys Avanti dry bars),
  sites (placed separately each day) and reference electrodes;
* different hardware chains (24-bit ADC at 4 kHz vs Trigno at 4370 Hz / 2148 Hz).

Agreement is judged on normalized activation *shapes* and *levels* relative to each
system's own maximum, on onset timing distributions and on signal quality, never per
sample or per cycle.

**Same chain for both systems.** The Delsys channels are mapped by label, converted to µV
(``io.reference.read_delsys_csv``), resampled onto one uniform grid at the JTS-MM EMG rate
(:func:`delsys_take`) and wrapped in :class:`EmgTake`, a stand-in for the parts of
:class:`~biomech.io.session.SessionData` that the dataset EMG code reads. Both systems then
go through the unchanged dataset code: :class:`~biomech.emg.prepare.EmgPrep` (mains fit,
20-450 Hz zero-phase band-pass, analysis masks, masked 150-ms RMS envelope, D11 reference =
largest activity recorded over the MVC phases and the tasks), ``qc.report.emg_table``
(rest noise, SNR, mains share), ``spectral`` (MNF/MDF, band shares) and
``timing.lead.analyse`` (TKEO/Hodges/envelope onsets, ``first``/``drive`` rules, X2
ensemble, X3 release). Delsys has no lead-off bits; its rail rule uses the Avanti range
(``full_scale_uv``, ±11 mV), which facial EMG never approaches.

**MVC.** Each Delsys MVC CSV holds an opening then a closing effort; :func:`segment_mvc`
splits it where the active muscle group changes (openers vs closers, each channel scaled to
its own file maximum) and labels the halves ``mvc_opening`` / ``mvc_closing``. With several
MVC files the per-channel reference comes from the file with the largest MVC activity
(:class:`MultiMvcPrep`; the D11 largest-activity rule across files).

**Delsys timeline.** The Trigno base triggers QTM: ``t_QTM = t_Delsys − δ`` (δ = QTM start
latency, nominally 0; a hidden QTM trigger event sits at ≈ 0.74 s). QTM time maps to the
JTS-MM master clock through the ``ref_sync`` model (``TimeModel.to_s``). δ is chosen by the
pre-stated plausibility test :func:`delta_check`: the Delsys opener/closer envelopes are
cross-correlated with the Qualisys jaw speed (and angle) in ``open_close`` and ``chewing``
over ±1.5 s; the candidate δ whose implied EMG lead (``peak lag + δ``) is physiological
(0-250 ms) in most speed correlations wins. A clock-rate difference between Trigno and
QTM is not modelled (not identifiable from these data); the δ check is repeated per task
and reported.

**Cycles.** JTS-MM profiles use the dataset take's own kinematic cycles; Delsys profiles
use the comparison take's cycles from its JTS-MM optics (the same cycle detection,
:func:`cycle_kinematics`; the comparison take's ``mand_point_1`` landmark from the still
hold, V3). Optical times are shifted to the exposure centre (+6 ms) in both takes so the
EMG-to-motion timing is on one footing. Mechanical onsets for the timing come from the
optical opening angle (no IMU-based onset, because the Delsys side has none); for Delsys
the Qualisys opening angle is a second source.

**Selection (D0b).** Task activation: the five movement tasks (``free_motion`` cut 0.5 s
before the first CT appearance, the end sync); profiles: the cycle tasks; rest noise and
SNR: ``rest``; MVC: the MVC phases/segments only; spectra: task phases, mask-aware.

Caveat: the D11 reference makes normalized levels ≤ 100 % per system; ``ref_source =
task`` marks a submaximal MVC (reported, not excluded). Low Delsys masseter R RMS (6-8 µV
in s10/s14) points at the electrode and is reported in the quality table.
"""

from __future__ import annotations

import types
import warnings
from dataclasses import dataclass, field
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from biomech.emg.prepare import EmgParams, EmgPrep
from biomech.io.session import CLOSERS, MUSCLES, OPENERS

DEFAULTS: dict[str, Any] = {
    "reference": "reference.yaml",     # relative to the study file
    "exposure_s": 0.012,               # JTS-MM exposure; optical time -> exposure centre
    "fs_hz": None,                     # Delsys resampling rate; None = JTS-MM EMG rate
    "full_scale_uv": 11000.0,          # Trigno Avanti input range (rail rule)
    "delta_s": "auto",                 # δ (s): "auto" = plausibility test, or a number
    "delta_candidates_s": [0.0, 0.74],  # nominal 0 and the hidden QTM trigger event
    "xcorr_half_range_s": 1.5,
    "xcorr_tasks": ["open_close", "chewing"],
    "xcorr_grid_hz": 100.0,
    "lead_window_s": [0.0, 0.25],      # physiological EMG-to-motion lead for the δ test
    "speed_baseline_s": 0.05,          # Qualisys jaw speed: central difference over ±25 ms
    "ct_guard_s": 0.5,                 # free_motion cut before the first CT appearance
    "profile_points": 101,
    "spm_alpha": 0.05,
    "n_boot": 2000,
    "mvc_smooth_s": 1.0,               # MVC segmentation: group activity smoothing
}

SYSTEMS = ("jtsmm", "delsys")
MOVEMENT_TASKS = ("open_close", "protrusion_retrusion", "left_right", "chewing",
                  "free_motion")


# -- a SessionData stand-in for EMG held in memory ---------------------------------------
@dataclass
class Phase:
    """Same fields as ``jtsmm.tools.plot_mp_trajectory.PhaseInterval``."""
    name: str
    kind: str
    t_start_us: int
    t_end_us: int


@dataclass
class EmgTake:
    """EMG of one take in memory with the SessionData methods the EMG code reads
    (``EmgPrep``, ``causal_clean``, ``qc.report.emg_table``).

    ``t_us`` is the sample time on the analysis clock (JTS-MM master µs for a mapped Delsys
    comparison take, the Delsys clock for an MVC file). No lead-off information (all False);
    the rail rule uses ``full_scale_uv``."""
    name: str
    t_us: np.ndarray                 # (N,) int64, increasing
    uv: np.ndarray                   # (N, C) µV
    names: list[str]
    fs: float
    phases: list[Any] = field(default_factory=list)
    full_scale_uv: float = np.inf
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def folder(self) -> Path:
        return Path(self.name)

    @property
    def emg_fs(self) -> float:
        return float(self.fs)

    def muscle_names(self) -> list[str]:
        return list(self.names)

    def phase(self, name: str) -> Any:
        return next((ph for ph in self.phases if ph.name == name), None)

    def phase_names(self) -> list[str]:
        return [ph.name for ph in self.phases]

    def _cols(self, channels: list[str] | None) -> list[int]:
        return [self.names.index(c) for c in (channels or self.names)]

    def _slice(self, t0_us: int, t1_us: int) -> slice:
        return slice(int(np.searchsorted(self.t_us, t0_us, "left")),
                     int(np.searchsorted(self.t_us, t1_us + 1, "left")))

    def emg_full_scale_uv(self, channels: list[str] | None = None) -> np.ndarray:
        return np.full(len(self._cols(channels)), float(self.full_scale_uv))

    def emg_window(self, t0_us: int, t1_us: int, channels: list[str] | None = None
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        s, cols = self._slice(t0_us, t1_us), self._cols(channels)
        return self.t_us[s], self.uv[s][:, cols].astype(np.float64), list(channels or self.names)

    def leadoff_window(self, t0_us: int, t1_us: int, channels: list[str] | None = None
                       ) -> np.ndarray:
        s = self._slice(t0_us, t1_us)
        return np.zeros((s.stop - s.start, len(self._cols(channels))), bool)

    def emg_phase(self, phase: str, channels: list[str] | None = None
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        ph = self.phase(phase)
        if ph is None:
            raise LookupError(f"no phase {phase!r} in {self.name}")
        return self.emg_window(ph.t_start_us, ph.t_end_us, channels)


class PhasedSession:
    """A SessionData with another phase list (the D0b-selected phases); everything else is
    delegated. ``emg_phase`` is redefined so it uses the replaced phases."""

    def __init__(self, sd: Any, phases: list[Any]) -> None:
        self._sd = sd
        self.phases = list(phases)

    def __getattr__(self, k: str) -> Any:
        return getattr(self._sd, k)

    def phase(self, name: str) -> Any:
        return next((ph for ph in self.phases if ph.name == name), None)

    def phase_names(self) -> list[str]:
        return [ph.name for ph in self.phases]

    def emg_phase(self, phase: str, channels: list[str] | None = None):
        ph = self.phase(phase)
        if ph is None:
            raise LookupError(f"no phase {phase!r}")
        return self._sd.emg_window(ph.t_start_us, ph.t_end_us, channels)


def select_phases(phases: Sequence[Any], ct_t_us: np.ndarray | None = None,
                  ct_ok: np.ndarray | None = None, guard_s: float = 0.5,
                  min_run: int = 3, min_s: float = 1.0) -> list[Phase]:
    """D0b for the EMG: a rest or movement-task phase in which the calibration tool appears
    (a visibility run of ``min_run`` frames: the end sync inside ``free_motion``) ends
    ``guard_s`` before that appearance (dropped when shorter than ``min_s``). The other
    phases are kept unchanged (no EMG outcome reads the holds; the take's grid frequency
    is measured from the first phase on, as in the dataset pipeline)."""
    from biomech.validation.selection import ct_runs_within

    out = []
    for ph in phases:
        t0, t1 = int(ph.t_start_us), int(ph.t_end_us)
        if ph.name in ("rest",) + MOVEMENT_TASKS and ct_t_us is not None and ct_ok is not None:
            r = ct_runs_within(np.asarray(ct_t_us, float) * 1e-6, ct_ok, t0 * 1e-6, t1 * 1e-6,
                               min_run=min_run)
            if r is not None:
                t1 = min(t1, int(round((r[0] - guard_s) * 1e6)))
            if t1 - t0 < min_s * 1e6:
                continue
        out.append(Phase(ph.name, getattr(ph, "kind", "experiment"), t0, t1))
    return out


def clip_phases(phases: Sequence[Any], t0_us: int, t1_us: int, min_s: float = 1.0
                ) -> list[Phase]:
    """Phases clipped to a recording's coverage; shorter than ``min_s`` afterwards: dropped."""
    out = []
    for ph in phases:
        a, b = max(int(ph.t_start_us), int(t0_us)), min(int(ph.t_end_us), int(t1_us))
        if b - a >= min_s * 1e6:
            out.append(Phase(ph.name, getattr(ph, "kind", "experiment"), a, b))
    return out


# -- Delsys -> EmgTake ---------------------------------------------------------------------
def resample_channel(t_s: np.ndarray, uv: np.ndarray, fs_in: float, t_out_s: np.ndarray,
                     fs_out: float) -> np.ndarray:
    """One channel onto ``t_out_s``: zero-phase anti-alias low-pass (8th order at 0.45 x
    ``fs_out``) when downsampling, then cubic interpolation (flat to ≪ 1 % up to 450 Hz;
    linear interpolation of the 2148-Hz sensor would cost 1.3 dB at 450 Hz)."""
    from scipy.interpolate import CubicSpline
    from scipy.signal import butter, sosfiltfilt

    x = np.asarray(uv, dtype=np.float64)
    t = np.asarray(t_s, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(t)
    t, x = t[ok], x[ok]
    if fs_in > 1.001 * fs_out:
        sos = butter(8, 0.45 * fs_out / (0.5 * fs_in), output="sos")
        x = sosfiltfilt(sos, x)
    y = CubicSpline(t, x, extrapolate=False)(t_out_s)
    return y


def delsys_take(d: Any, names: Sequence[str], fs_out: float,
                time_map: Callable[[np.ndarray], np.ndarray] | None = None,
                phases: Sequence[Any] = (), full_scale_uv: float = 11000.0,
                name: str = "delsys") -> EmgTake:
    """A Delsys recording (``DelsysData``) as an :class:`EmgTake` on one uniform grid.

    The grid runs at ``fs_out`` on the Delsys clock over the span all channels cover;
    ``time_map`` (Delsys seconds -> analysis seconds; default identity) gives the sample
    times ``t_us`` (e.g. the JTS-MM master clock through δ and the ``ref_sync`` model)."""
    names = [n for n in names if n in d.channels]
    if not names:
        raise LookupError(f"{name}: no Delsys channel of {list(d.channels)} matches")
    t0 = max(float(d.channels[n].t_s[0]) for n in names)
    t1 = min(float(d.channels[n].t_s[-1]) for n in names)
    grid = t0 + np.arange(int(np.floor((t1 - t0) * fs_out)) + 1) / fs_out
    uv = np.column_stack([resample_channel(d.channels[n].t_s, d.channels[n].uv,
                                           d.channels[n].fs_hz, grid, fs_out) for n in names])
    tm = grid if time_map is None else np.asarray(time_map(grid), dtype=np.float64)
    t_us = np.round(tm * 1e6).astype(np.int64)
    meta = {"t0_delsys_s": float(grid[0]), "t1_delsys_s": float(grid[-1]),
            "rates_hz": {n: float(d.channels[n].fs_hz) for n in names},
            "labels": {n: d.channels[n].label for n in names},
            "header": dict(getattr(d, "header", {}) or {}), "path": str(getattr(d, "path", ""))}
    return EmgTake(name=name, t_us=t_us, uv=uv, names=list(names), fs=float(fs_out),
                   phases=list(phases), full_scale_uv=float(full_scale_uv), meta=meta)


def segment_mvc(take: EmgTake, smooth_s: float = 1.0, band_hz=(20.0, 450.0)
                ) -> tuple[list[Phase], dict[str, Any]]:
    """Split a Delsys MVC file into ``mvc_opening`` / ``mvc_closing`` by the active group.

    Per channel the 20-450 Hz RMS envelope (smoothed over ``smooth_s``) is scaled to its own
    file maximum; ``d(t)`` = mean opener level − mean closer level. The split maximizes the
    opener dominance before it and the closer dominance after it, i.e. the extremum of
    ``cumsum(d)``: the maximum for opening-first, the minimum for closing-first (the larger
    in magnitude decides the order). Returns the two phases and a summary (split time,
    order, dominance in each half)."""
    from scipy.ndimage import uniform_filter1d

    from biomech.emg.pipeline import bandpass, rms_envelope

    x = bandpass(take.uv - np.nanmedian(take.uv, axis=0), take.fs, band_hz)
    env = uniform_filter1d(rms_envelope(x, take.fs, 0.15), size=max(int(smooth_s * take.fs), 1),
                           axis=0, mode="nearest")
    env = env / np.maximum(np.nanmax(env, axis=0), 1e-12)
    op = [i for i, n in enumerate(take.names) if n in OPENERS]
    cl = [i for i, n in enumerate(take.names) if n in CLOSERS]
    if not op or not cl:
        raise LookupError(f"{take.name}: MVC segmentation needs openers and closers")
    dd = env[:, op].mean(axis=1) - env[:, cl].mean(axis=1)
    cs = np.cumsum(dd)
    i_max, i_min = int(np.argmax(cs)), int(np.argmin(cs))
    opening_first = abs(cs[i_max]) >= abs(cs[i_min])
    k = i_max if opening_first else i_min
    k = int(np.clip(k, 1, len(dd) - 2))
    t = take.t_us
    a = Phase("mvc_opening" if opening_first else "mvc_closing", "mvc", int(t[0]), int(t[k]))
    b = Phase("mvc_closing" if opening_first else "mvc_opening", "mvc", int(t[k + 1]),
              int(t[-1]))
    info = {"split_s": float((t[k] - t[0]) * 1e-6), "duration_s": float((t[-1] - t[0]) * 1e-6),
            "order": "opening_first" if opening_first else "closing_first",
            "dominance_first": float(np.mean(dd[:k + 1])),
            "dominance_second": float(np.mean(dd[k + 1:]))}
    return [a, b], info


# -- EmgPrep over several MVC recordings ---------------------------------------------------
class MultiMvcPrep(EmgPrep):
    """:class:`EmgPrep` with several MVC recordings (Delsys: two MVC CSVs).

    Each recording's references are computed by the unchanged ``EmgPrep.refs`` code; the
    reference of a channel comes from the recording with the largest MVC activity
    (``max_uv``; the D11 largest-activity rule across recordings). With one recording it is
    identical to :class:`EmgPrep`."""

    def __init__(self, motion: Any, mvcs: Sequence[Any], p: EmgParams) -> None:
        mvcs = [m for m in mvcs if m is not None]
        super().__init__(motion, mvcs[0] if mvcs else None, p)
        self.mvcs = mvcs

    @cached_property
    def refs_per_file(self) -> list[dict[str, np.ndarray]]:
        out = []
        try:
            for sd in self.mvcs:
                self.mvc = sd
                out.append(EmgPrep.refs.func(self))
        finally:
            self.mvc = self.mvcs[0] if self.mvcs else None
        return out

    @cached_property
    def refs(self) -> dict[str, np.ndarray]:
        per = self.refs_per_file
        if len(per) <= 1:
            out = per[0] if per else EmgPrep.refs.func(self)
            return {**out, "file": np.zeros(len(self.names), int)}
        mx = np.stack([np.nan_to_num(r["max_uv"], nan=-np.inf) for r in per])
        best = np.argmax(mx, axis=0)
        out: dict[str, np.ndarray] = {"phase": per[0]["phase"], "file": best}
        for k in per[0]:
            if k == "phase":
                continue
            out[k] = np.array([per[best[i]][k][i] for i in range(len(self.names))])
        return out

    def mvc_phase(self, k: int, phase: str) -> Any:
        """The signals (``PhaseEmg``) of one MVC phase of recording ``k`` (not memoized)."""
        try:
            self.mvc = self.mvcs[k]
            return self._phase_uncached(phase, "mvc", None)
        finally:
            self.mvc = self.mvcs[0] if self.mvcs else None


# -- kinematic cycles (dataset code) ---------------------------------------------------------
def jaw_motion(sd: Any, kp: Any, hold: str = "phase") -> Any:
    """``kinematics.pose_chain.build`` with the landmark hold rule ``hold``.

    ``"still"`` (comparison takes, V3) runs the unchanged ``build`` with ``digitize`` bound
    to ``hold="still"`` (a copy of the function with that one global replaced), so the pose
    chain stays the dataset code."""
    from biomech.frames.landmarks import digitize
    from biomech.kinematics import pose_chain

    if hold == "phase":
        return pose_chain.build(sd, kp)
    g = dict(pose_chain.build.__globals__)
    g["digitize"] = partial(digitize, hold=hold)
    fn = types.FunctionType(pose_chain.build.__code__, g, "build_" + hold,
                            pose_chain.build.__defaults__)
    return fn(sd, kp)


def cycle_kinematics(sd: Any, pid: str, kp: Any, cycles_p: dict[str, Any],
                     hold: str = "phase", t_offset_us: int = 0
                     ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Cycles (``cycle_metrics`` rows) and per-phase opening angle as the ``kinematics``
    stage computes them (``find_task_cycles``; angle about +y from the median closed
    open-close start). All times shifted by ``t_offset_us`` (exposure centre)."""
    from biomech.kinematics import se3
    from biomech.kinematics.clinical import TASKS, cycle_metrics, find_task_cycles, theta_deg

    jm = jaw_motion(sd, kp, hold)
    task_cycles = {}
    for task in TASKS:
        ph = sd.phase(task)
        if ph is None:
            continue
        for tc in find_task_cycles(jm, task, ph.t_start_us, ph.t_end_us, cycles_p):
            task_cycles[(task, tc.side)] = tc
    closed_T = None
    oc = task_cycles.get(("open_close", ""))
    if oc is not None:
        idx = np.flatnonzero(oc.sel)
        starts = np.array([idx[c.start] for c in oc.cycles if c.ok], dtype=int)
        starts = starts[se3.valid(jm.T[starts])] if starts.size else starts
        if starts.size:
            z = se3.apply(jm.T[starts], jm.points_mp["incisor"])[:, 2]
            closed_T = jm.T[starts[np.argsort(z)[len(z) // 2]]]
    rows: list[dict[str, Any]] = []
    for tc in task_cycles.values():
        rows += cycle_metrics(jm, tc, pid, closed_T)
    cyc = pd.DataFrame(rows)
    for c in ("t_start_us", "t_peak_us", "t_end_us", "t_on_us", "t_off_us"):
        if c in cyc:
            cyc[c] = cyc[c].astype(np.int64) + int(t_offset_us)
    arrays: dict[str, np.ndarray] = {}
    for ph in sd.phases:
        if getattr(ph, "kind", "") != "experiment" or closed_T is None:
            continue
        sel = jm.window(ph.t_start_us, ph.t_end_us)
        arrays[f"{ph.name}__t_us"] = jm.t_us[sel].astype(np.int64) + int(t_offset_us)
        arrays[f"{ph.name}__theta_deg"] = theta_deg(jm.T[sel], closed_T)[0]
    return cyc, arrays


# -- Qualisys jaw signals --------------------------------------------------------------------
def qualisys_jaw(qtm: Any, mp: str = "MP", hp: str = "HP", baseline_s: float = 0.05,
                 ref_window_s: tuple[float, float] | None = None,
                 axis_window_s: tuple[float, float] | None = None
                 ) -> dict[str, np.ndarray]:
    """Jaw-in-head signals of a QTM take on its own clock: angular speed (deg/s, central
    difference over ±``baseline_s/2``), rotation angle from the reference orientation and
    the signed opening angle (rotation vector projected on the principal axis of the
    ``axis_window_s`` motion, positive away from the reference). Reference = mean
    orientation in ``ref_window_s`` (default: the whole take)."""
    from scipy.spatial.transform import Rotation

    t = np.asarray(qtm.t_s, dtype=np.float64)
    Th, Tm = qtm.pose(hp), qtm.pose(mp)
    ok = np.isfinite(Th.reshape(len(t), -1)).all(1) & np.isfinite(Tm.reshape(len(t), -1)).all(1)
    R = np.full((len(t), 3, 3), np.nan)
    R[ok] = np.einsum("nji,njk->nik", Th[ok, :3, :3], Tm[ok, :3, :3])
    k = max(int(round(0.5 * baseline_s * qtm.fs_hz)), 1)
    speed = np.full(len(t), np.nan)
    a, b = np.arange(0, len(t) - 2 * k), np.arange(2 * k, len(t))
    good = ok[a] & ok[b]
    if good.any():
        Rd = np.einsum("nji,njk->nik", R[a[good]], R[b[good]])
        ang = np.degrees(np.linalg.norm(Rotation.from_matrix(Rd).as_rotvec(), axis=1))
        speed[a[good] + k] = ang / ((b[good] - a[good]) / qtm.fs_hz)

    def win(w: tuple[float, float] | None) -> np.ndarray:
        return ok if w is None else ok & (t >= w[0]) & (t <= w[1])

    angle = np.full(len(t), np.nan)
    theta = np.full(len(t), np.nan)
    sel = win(ref_window_s)
    if sel.sum() >= 3:
        Rref = Rotation.from_matrix(R[sel]).mean().as_matrix()
        rv = np.full((len(t), 3), np.nan)
        rv[ok] = Rotation.from_matrix(np.einsum("ji,njk->nik", Rref, R[ok])).as_rotvec()
        angle = np.degrees(np.linalg.norm(rv, axis=1))
        ax_sel = win(axis_window_s)
        if ax_sel.sum() >= 10:
            X = rv[ax_sel]
            _, _, vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
            u = vt[0]
            proj = rv @ u
            if np.nanpercentile(proj[ax_sel], 95) < -np.nanpercentile(proj[ax_sel], 5):
                proj = -proj
            theta = np.degrees(proj)
    return {"t_s": t, "speed_dps": speed, "angle_deg": angle, "theta_deg": theta, "ok": ok}


# -- δ: Delsys -> QTM start latency ---------------------------------------------------------
def _xcorr(e: np.ndarray, m_ext: np.ndarray, n_lag: int) -> np.ndarray:
    """Pearson r(L) between ``e`` (n) and ``m_ext[n_lag + L : n_lag + L + n]``."""
    n = len(e)
    out = np.full(2 * n_lag + 1, np.nan)
    for j, L in enumerate(range(-n_lag, n_lag + 1)):
        m = m_ext[n_lag + L: n_lag + L + n]
        ok = np.isfinite(e) & np.isfinite(m)
        if ok.sum() < 20:
            continue
        a, b = e[ok] - e[ok].mean(), m[ok] - m[ok].mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        out[j] = (a * b).sum() / den if den > 0 else np.nan
    return out


def group_envelopes(d: Any, env_s: float = 0.15, band_hz=(20.0, 450.0)
                    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per Delsys channel (own clock, own rate): 20-450 Hz band-passed RMS envelope."""
    from biomech.emg.pipeline import bandpass, rms_envelope

    out = {}
    for mus, ch in d.channels.items():
        x = np.asarray(ch.uv, float)
        x = bandpass(x - np.median(x), ch.fs_hz, band_hz)
        out[mus] = (np.asarray(ch.t_s, float), rms_envelope(x, ch.fs_hz, env_s))
    return out


def delta_check(envs: dict[str, tuple[np.ndarray, np.ndarray]], jaw: dict[str, np.ndarray],
                windows_q: dict[str, tuple[float, float]], candidates: Sequence[float],
                half_range_s: float = 1.5, grid_hz: float = 100.0,
                lead_window_s: Sequence[float] = (0.0, 0.25)
                ) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    """The plausibility test that tells the QTM start latency δ apart (plan D5).

    For each task window (QTM seconds) and muscle group (openers, closers; each channel's
    envelope scaled by its window median), the envelope sampled at Delsys time = QTM time
    (δ = 0 assumed) is correlated with the Qualisys jaw speed and angle at lags L in
    ±``half_range_s`` (L > 0: EMG first). A true latency δ shifts the peak to
    ``L = lead − δ``, so each candidate implies ``lead = L_peak + δ``. Decision: over the
    *opener* correlations (speed and angle; the openers are the agonists of opening, so
    their envelope maps onto the opening angle and speed, whereas the closers fire at
    closing and occlusion and peak at negative lags), the candidate whose implied lead lies
    in ``lead_window_s`` most often; ties go to the higher mean windowed peak r; no vote at
    all = inconclusive, nominal δ = 0. The closer rows are reported. (The first draft voted
    on the speed rows only; the rectified speed peaks at opening *and* closing, which left
    single tasks ambiguous in s10, so the openers' signed angle was added before δ was used
    for any outcome.)"""
    lo, hi = float(lead_window_s[0]), float(lead_window_s[1])
    step = 1.0 / grid_hz
    n_lag = int(round(half_range_s * grid_hz))
    lags = np.arange(-n_lag, n_lag + 1) * step
    groups = {"openers": [m for m in OPENERS if m in envs],
              "closers": [m for m in CLOSERS if m in envs]}
    jt = jaw["t_s"]
    rows, arrays = [], {"delta_lags_s": lags}
    for task, (q0, q1) in windows_q.items():
        g = q0 + np.arange(int(np.floor((q1 - q0) / step))) * step
        g_ext = q0 + (np.arange(len(g) + 2 * n_lag) - n_lag) * step     # g[i] + L at n_lag+L+i
        mot = {}
        for sig in ("speed_dps", "angle_deg"):
            y = jaw[sig]
            ok = np.isfinite(y)
            if ok.sum() < 10:
                continue
            v = np.interp(g_ext, jt[ok], y[ok], left=np.nan, right=np.nan)
            near = np.clip(np.searchsorted(jt, g_ext), 0, len(jt) - 1)
            v[~ok[near]] = np.nan
            mot[sig] = v
        for grp, mus in groups.items():
            if not mus:
                continue
            parts = []
            for m in mus:
                te, ee = envs[m]
                e = np.interp(g, te, ee, left=np.nan, right=np.nan)
                med = np.nanmedian(e)
                parts.append(e / med if med > 0 else e)
            e = np.nanmean(np.vstack(parts), axis=0)
            for sig, v in mot.items():
                r = _xcorr(e, v, n_lag)
                arrays[f"delta__{task}__{grp}__{sig}"] = r
                if not np.isfinite(r).any():
                    continue
                jp = int(np.nanargmax(r))
                row = {"task": task, "group": grp, "signal": sig, "n": len(g),
                       "lag_peak_s": float(lags[jp]), "r_peak": float(r[jp])}
                for dlt in candidates:
                    lead = float(lags[jp] + dlt)
                    inw = (lags + dlt >= lo) & (lags + dlt <= hi)
                    row[f"lead_ms_d{dlt:g}"] = lead * 1e3
                    row[f"plausible_d{dlt:g}"] = bool(lo <= lead <= hi)
                    row[f"r_window_d{dlt:g}"] = float(np.nanmax(r[inw])) if inw.any() and \
                        np.isfinite(r[inw]).any() else np.nan
                rows.append(row)
    tab = pd.DataFrame(rows)
    votes = {float(d): 0 for d in candidates}
    rwin = {float(d): np.nan for d in candidates}
    if not tab.empty:
        sp = tab[tab.group == "openers"]
        for dlt in candidates:
            votes[float(dlt)] = int(sp[f"plausible_d{dlt:g}"].sum())
            rwin[float(dlt)] = float(sp[f"r_window_d{dlt:g}"].mean())
    best = max(votes.values()) if votes else 0
    if best == 0:
        chosen, status = 0.0, "inconclusive"
    else:
        tied = [d for d, v in votes.items() if v == best]
        chosen = max(tied, key=lambda d: np.nan_to_num(rwin[d], nan=-np.inf))
        status = "decided" if len(tied) == 1 else "tie_broken_by_r"
    info = {"delta_s": float(chosen), "status": status,
            **{f"votes_d{d:g}": v for d, v in votes.items()},
            **{f"r_window_mean_d{d:g}": v for d, v in rwin.items()},
            "n_tests": int((tab.group == "openers").sum()) if not tab.empty else 0}
    return tab, arrays, info


# -- outcomes --------------------------------------------------------------------------------
def profile_key(task: str, side: str) -> str:
    """``open_close`` or ``left_right__left`` (task and side as separate key fields)."""
    return f"{task}__{side}" if side else task


def split_profile_key(key: str) -> tuple[str, str]:
    """(task, side) of a :func:`profile_key` (side ``""`` for the symmetric tasks)."""
    task, _, side = key.partition("__")
    return task, side


PROFILE_PREFIX, PROFILE_SUFFIX = "profile__", "__pct"


def profile_array_key(system: str, key: str) -> str:
    """Name of the stored ``ref_emg`` array of the cycle profiles of one system and profile
    key: ``profile__<system>__<key>__pct``, shape (cycles, points, channels), % of the
    system's own D11 reference."""
    return f"{PROFILE_PREFIX}{system}__{key}{PROFILE_SUFFIX}"


def finite_cycles(Y: np.ndarray) -> np.ndarray:
    """The cycles (rows) of a (cycles, points) profile stack with every point finite (a
    channel-cycle masked by ``cycle_mask_max`` is NaN and drops out)."""
    return Y[np.isfinite(Y).all(axis=1)]


def profile_r(a: np.ndarray, b: np.ndarray) -> float:
    """Shape correlation of two mean profiles on the same normalized-time grid: Pearson r,
    i.e. the mean product of the two profiles z-scored over their points (population SD), so
    level and gain differences do not enter. NaN when either profile is constant."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(a, b)[0, 1])


def cycle_profiles(prep: EmgPrep, cycles: pd.DataFrame, cycle_mask_max: float,
                   n_points: int = 101, min_cycle_s: float = 0.1
                   ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Normalized envelope (% of the D11 reference) per ``ok`` cycle, time-normalized to
    ``n_points``; a channel-cycle masked above ``cycle_mask_max`` is NaN (the ``emg``
    stage's rule). Returns ``key -> (profiles (n, n_points, C), cycle ids)``."""
    from biomech.kinematics.cycles import time_normalize

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if cycles is None or cycles.empty:
        return out
    ok = cycles[cycles["ok"].astype(bool)]
    for (task, side), g in ok.groupby(["task", "side"], sort=False):
        if prep.motion.phase(task) is None:
            continue
        pe = prep.phase(task)
        prof, ids = [], []
        for c in g.itertuples():
            sel = (pe.t_us >= c.t_start_us) & (pe.t_us <= c.t_end_us)
            if sel.sum() < max(int(round(min_cycle_s * pe.fs)), 2):
                continue
            seg = np.array(pe.norm[sel], dtype=np.float64)
            seg[:, pe.mask[sel].mean(axis=0) > cycle_mask_max] = np.nan
            prof.append(time_normalize(seg, n_points))
            ids.append(int(c.cycle))
        if prof:
            out[profile_key(task, side)] = (np.stack(prof), np.array(ids))
    return out


def ttest2_spm(YA: np.ndarray, YB: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
    """SPM two-sample t-test (independent sessions; unequal variances), cycles as
    repetitions. NaN rows dropped; fewer than 3 per group: not run."""
    import spm1d

    from biomech.stats.waveforms import _clusters

    YA = YA[np.isfinite(YA).all(axis=1)]
    YB = YB[np.isfinite(YB).all(axis=1)]
    if len(YA) < 3 or len(YB) < 3:
        return {"n_a": len(YA), "n_b": len(YB), "h0reject": None, "zstar": np.nan,
                "z": None, "clusters": []}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ti = spm1d.stats.ttest2(YA, YB, equal_var=False).inference(alpha, two_tailed=True)
    except Exception as exc:                     # degenerate data (zero variance, ...)
        return {"n_a": len(YA), "n_b": len(YB), "h0reject": None, "zstar": np.nan,
                "z": None, "clusters": [], "error": str(exc)}
    return {"n_a": len(YA), "n_b": len(YB), "h0reject": bool(ti.h0reject),
            "zstar": float(ti.zstar), "z": np.asarray(ti.z, float), "clusters": _clusters(ti)}


def compare_profiles(pa: dict[str, tuple[np.ndarray, np.ndarray]],
                     pb: dict[str, tuple[np.ndarray, np.ndarray]], names: Sequence[str],
                     pid: str, alpha: float = 0.05
                     ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Per profile key and muscle: Pearson r and CMC of the two ensemble means, RMS and mean
    difference (percentage points, B − A), SPM two-sample t (``spm`` rows; ``clusters``
    rows for the significance bars, E0). ``pa`` = JTS-MM, ``pb`` = Delsys."""
    from biomech.stats.waveforms import cmc

    rows, clus, arrays = [], [], {}
    for key in sorted(set(pa) & set(pb)):
        A, B = pa[key][0], pb[key][0]
        for i, m in enumerate(names):
            ya, yb = finite_cycles(A[:, :, i]), finite_cycles(B[:, :, i])
            task, side = split_profile_key(key)
            row: dict[str, Any] = {"pid": pid, "profile": key, "task": task, "side": side,
                                   "muscle": m,
                                   "n_jtsmm": len(ya), "n_delsys": len(yb)}
            if len(ya) >= 2 and len(yb) >= 2:
                ma, mb = ya.mean(axis=0), yb.mean(axis=0)
                d = mb - ma
                row.update({"r": profile_r(ma, mb),
                            "cmc": cmc(np.vstack([ma, mb])),
                            "rms_diff_pct": float(np.sqrt(np.mean(d ** 2))),
                            "mean_diff_pct": float(d.mean()),
                            "mean_jtsmm_pct": float(ma.mean()),
                            "mean_delsys_pct": float(mb.mean()),
                            "cmc_within_jtsmm": cmc(ya), "cmc_within_delsys": cmc(yb)})
                # Kadaba's CMC has no real value when the level difference of the two
                # curves exceeds their shared shape: say so, never "n/a".
                row["cmc_defined"] = bool(np.isfinite(row["cmc"]))
            s = ttest2_spm(ya, yb, alpha) if len(ya) and len(yb) else {"h0reject": None,
                                                                         "zstar": np.nan,
                                                                         "clusters": []}
            row.update({"spm_h0reject": s["h0reject"], "spm_zstar": s["zstar"],
                        "spm_n_clusters": len(s["clusters"])})
            for c in s["clusters"]:
                clus.append({"pid": pid, "profile": key, "muscle": m, **c})
            rows.append(row)
    return (pd.DataFrame(rows),
            pd.DataFrame(clus, columns=["pid", "profile", "muscle", "start_pct", "end_pct",
                                        "p"]), arrays)


def activation_rows(prep: EmgPrep, system: str, pid: str) -> list[dict[str, Any]]:
    """Median and p95 of the normalized and of the µV envelope per task phase and muscle."""
    rows = []
    for ph in prep.motion.phases:
        if ph.name not in ("rest",) + MOVEMENT_TASKS:
            continue
        pe = prep.phase(ph.name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i, m in enumerate(prep.names):
                rows.append({"pid": pid, "system": system, "task": ph.name, "muscle": m,
                             "median_pct": float(np.nanmedian(pe.norm[:, i])),
                             "p95_pct": float(np.nanpercentile(pe.norm[:, i], 95)),
                             "median_uv": float(np.nanmedian(pe.env[:, i])),
                             "p95_uv": float(np.nanpercentile(pe.env[:, i], 95)),
                             "mask_frac": float(pe.mask[:, i].mean()),
                             "duration_s": float((ph.t_end_us - ph.t_start_us) * 1e-6)})
    return rows


def mvc_rows(prep: MultiMvcPrep, system: str, pid: str, files: Sequence[str]
             ) -> list[dict[str, Any]]:
    """MVC levels (µV) per muscle, the reference actually used (D11) and MVC/task-max."""
    r, ref, tp = prep.refs, prep.reference, prep.task_peaks
    rows = []
    for i, m in enumerate(prep.names):
        k = int(r["file"][i]) if "file" in r else 0
        rows.append({"pid": pid, "system": system, "muscle": m, "mvc_phase": str(r["phase"][i]),
                     "mvc_file": files[k] if k < len(files) else "",
                     "mvc_max_uv": float(r["max_uv"][i]), "mvc_p95_uv": float(r["p95_uv"][i]),
                     "mvc_max1s_uv": float(r["max1s_uv"][i]),
                     "mvc_mask_frac": float(r["mask_frac"][i]),
                     "task_max_uv": float(tp["max_uv"][i]), "task_max_phase": str(tp["phase"][i]),
                     "ref_uv": float(prep.ref_uv()[i]), "ref_source": str(ref["source"][i]),
                     "mvc_over_task": float(ref["mvc_over_task"][i])})
        for j, rr in enumerate(getattr(prep, "refs_per_file", [])):
            rows[-1][f"mvc_max_uv_file{j}"] = float(rr["max_uv"][i])
    return rows


def spectra_rows(prep: MultiMvcPrep, system: str, pid: str) -> list[dict[str, Any]]:
    """Welch spectra (0.5-s mask-free segments) per task phase and MVC phase: MNF, MDF and
    band shares (the ``emg`` stage's X1 quantities)."""
    from biomech.spectral import band_shares, mean_median_freq, welch_masked

    todo: list[tuple[str, str, Any]] = [(ph.name, "motion", None) for ph in prep.motion.phases
                                        if ph.name in ("rest",) + MOVEMENT_TASKS]
    for k, sd in enumerate(prep.mvcs):
        todo += [(ph.name, f"mvc{k}", k) for ph in sd.phases if ph.kind == "mvc"]
    rows = []
    for ph, take, k in todo:
        pe = prep.phase(ph) if k is None else prep.mvc_phase(k, ph)
        f, psd, per = welch_masked(pe.cleaned, pe.fs, pe.mask, seg_s=0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mnf, mdf = mean_median_freq(f, psd)
            shares = band_shares(f, psd)
        for i, m in enumerate(pe.names):
            rows.append({"pid": pid, "system": system, "take": take, "phase": ph, "muscle": m,
                         "mnf_hz": float(mnf[i]), "mdf_hz": float(mdf[i]),
                         "n_windows": int(np.isfinite(per[i, :, 0]).sum()) if per.size else 0,
                         **{f"share_{int(lo)}_{int(hi)}": float(shares[i, j])
                            for j, (lo, hi) in enumerate(((20, 50), (50, 100), (100, 200),
                                                          (200, 450)))}})
    return rows


def mech_arrays(kin: dict[str, np.ndarray], phases: Sequence[str], grid_hz: float = 200.0,
                max_gap_s: float = 0.1) -> dict[str, np.ndarray]:
    """The arrays ``timing.lead.analyse`` reads from the ``imu`` stage, built from an
    optical opening angle: ``omega_dps`` = its derivative on a ``grid_hz`` grid,
    ``theta_opt`` = the angle, ``acc_rel_ms2`` = NaN (no accelerometer source), so the
    mechanical onset is the optical velocity onset (same rule for both systems)."""
    from biomech import timebase

    out: dict[str, np.ndarray] = {}
    for ph in phases:
        t, th = kin.get(f"{ph}__t_us"), kin.get(f"{ph}__theta_deg")
        if t is None or th is None or len(t) < 4:
            continue
        g = np.arange(int(t[0]), int(t[-1]), int(round(1e6 / grid_hz)), dtype=np.int64)
        thg = timebase.resample(np.asarray(t, np.int64), np.asarray(th, float), g,
                                max_gap_s=max_gap_s)
        out[f"{ph}__t_us"] = g
        out[f"{ph}__theta_opt"] = thg
        out[f"{ph}__omega_dps"] = timebase.gradient(g, thg)
        out[f"{ph}__acc_rel_ms2"] = np.full(len(g), np.nan)
    return out


def timing_tables(prep: EmgPrep, pid: str, kin: dict[str, np.ndarray], cycles: pd.DataFrame,
                  p_timing: dict[str, Any], seed: int, system: str, mech: str
                  ) -> dict[str, pd.DataFrame]:
    """``timing.lead.analyse`` (onsets with every detector and rule, X2 ensemble, X3
    release) with the optical mechanical onset; rows tagged with system and source."""
    from biomech.timing.lead import analyse

    imu = mech_arrays(kin, list(p_timing.get("tasks", ["open_close", "chewing"])) + ["rest"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tabs, _ = analyse(prep, pid, imu, cycles, p_timing, seed, n_null=0)
    out = {}
    for k in ("onsets", "summary", "ensemble", "release_summary"):
        df = tabs.get(k, pd.DataFrame()).copy()
        if not df.empty:
            df.insert(1, "system", system)
            df.insert(2, "mech_source", mech)
        out[k] = df
    return out


def lead_comparison(onsets: pd.DataFrame, pid: str, seed: int, n_boot: int = 2000,
                    detector: str = "tkeo", rule: str = "drive") -> pd.DataFrame:
    """Distribution agreement of the cycle leads (cross-session, so no per-cycle pairing):
    per task, direction and muscle group the median lead per system and source, the
    difference Delsys − JTS-MM of the medians with a percentile bootstrap CI (cycles
    resampled within each system), and the IQRs. Usable channel-cycles only."""
    if onsets.empty:
        return pd.DataFrame()
    u = onsets[onsets.usable & (onsets.detector == detector) & (onsets.rule == rule)]
    rng = np.random.default_rng(seed)
    rows = []
    for (task, direction), g in u.groupby(["task", "direction"]):
        a = g[(g.system == "jtsmm") & (g.mech_source == "jtsmm_optical")]["lead_ms"].to_numpy()
        for src, gb in g[g.system == "delsys"].groupby("mech_source"):
            b = gb["lead_ms"].to_numpy()
            row = {"pid": pid, "task": task, "direction": direction, "detector": detector,
                   "rule": rule, "delsys_mech_source": src, "n_jtsmm": len(a), "n_delsys": len(b),
                   "median_jtsmm_ms": float(np.median(a)) if len(a) else np.nan,
                   "median_delsys_ms": float(np.median(b)) if len(b) else np.nan,
                   "iqr_jtsmm_ms": float(np.subtract(*np.percentile(a, [75, 25])))
                   if len(a) else np.nan,
                   "iqr_delsys_ms": float(np.subtract(*np.percentile(b, [75, 25])))
                   if len(b) else np.nan}
            if len(a) >= 3 and len(b) >= 3:
                bs = (np.median(rng.choice(b, (n_boot, len(b))), axis=1)
                      - np.median(rng.choice(a, (n_boot, len(a))), axis=1))
                row["diff_ms"] = row["median_delsys_ms"] - row["median_jtsmm_ms"]
                row["diff_ci_lo_ms"], row["diff_ci_hi_ms"] = (float(v) for v in
                                                              np.percentile(bs, [2.5, 97.5]))
            rows.append(row)
    return pd.DataFrame(rows)


def quality_rows(prep: EmgPrep, system: str, pid: str, usability: dict[str, Any]
                 ) -> pd.DataFrame:
    """``qc.report.emg_table`` (rest noise, SNR = MVC p95 / rest envelope, mains share,
    masks, usability flags) per muscle, tagged with the system."""
    from biomech.qc.report import emg_table

    u = {"snr_min_db": 6.0, "rest_noise_uv": 20.0, "ref_mask_max": 0.10,
         "lowfreq_flag_frac": 0.10, **(usability or {})}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        df = emg_table(prep, pid, u)
    df.insert(1, "system", system)
    return df


# -- orchestration ---------------------------------------------------------------------------
@dataclass
class EmgSides:
    """Both systems prepared for the comparison (exposed for tests and the real-data check)."""
    jtsmm: MultiMvcPrep
    delsys: MultiMvcPrep
    cyc_jtsmm: pd.DataFrame
    cyc_delsys: pd.DataFrame
    kin_jtsmm: dict[str, np.ndarray]
    kin_delsys: dict[str, np.ndarray]
    kin_qualisys: dict[str, np.ndarray] | None = None
    mvc_files: list[str] = field(default_factory=list)
    mvc_info: list[dict[str, Any]] = field(default_factory=list)


def compare(sides: EmgSides, pid: str, dp: dict[str, Any], p: dict[str, Any], seed: int
            ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Every D5 outcome from two prepared sides (no file access)."""
    names = list(sides.jtsmm.names)
    if list(sides.delsys.names) != names:
        raise ValueError(f"channel order differs: {names} vs {sides.delsys.names}")
    tables: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {"names": np.array(names)}
    cmax = float(dp.get("cycle_mask_max", 0.10))
    preps = {"jtsmm": sides.jtsmm, "delsys": sides.delsys}
    cycs = {"jtsmm": sides.cyc_jtsmm, "delsys": sides.cyc_delsys}

    tables["mvc"] = pd.DataFrame(
        mvc_rows(sides.jtsmm, "jtsmm", pid, ["dataset_mvc_take"])
        + mvc_rows(sides.delsys, "delsys", pid, sides.mvc_files))
    tables["activation"] = pd.DataFrame(activation_rows(sides.jtsmm, "jtsmm", pid)
                                        + activation_rows(sides.delsys, "delsys", pid))
    tables["quality"] = pd.concat([quality_rows(preps[s], s, pid, dp.get("usability", {}))
                                   for s in SYSTEMS], ignore_index=True)
    # Activity relevance per system (VD11, the dataset's D19 rule with the dataset's
    # ``stages.emg_relevance`` parameters: active when the task p95 reaches the rest mean +
    # k_sd SD and min_pct % of the largest activity recorded; EV11).
    from biomech.emg.relevance import relevance

    act = tables["activation"]
    rp = dp.get("relevance") or {}
    rkw = {"k_sd": float(rp.get("k_sd", 3.0)), "min_pct": float(rp.get("min_pct", 5.0))}
    tables["relevance"] = pd.concat(
        [relevance(preps[s], act[act.system == s], pid, **rkw).assign(system=s)
         for s in SYSTEMS], ignore_index=True)
    tables["spectra"] = pd.DataFrame(spectra_rows(sides.jtsmm, "jtsmm", pid)
                                     + spectra_rows(sides.delsys, "delsys", pid))
    if sides.mvc_info:
        tables["mvc_segments"] = pd.DataFrame([{"pid": pid, **i} for i in sides.mvc_info])

    prof = {s: cycle_profiles(preps[s], cycs[s], cmax, int(p.get("profile_points", 101)))
            for s in SYSTEMS}
    for s in SYSTEMS:
        for key, (Y, ids) in prof[s].items():
            arrays[profile_array_key(s, key)] = Y.astype(np.float32)
            arrays[f"{PROFILE_PREFIX}{s}__{key}__cycle"] = ids
    tables["profiles"], tables["spm_clusters"], parr = compare_profiles(
        prof["jtsmm"], prof["delsys"], names, pid, float(p.get("spm_alpha", 0.05)))
    arrays.update(parr)

    tt = dict(dp.get("timing") or {})
    if tt:
        parts = [timing_tables(sides.jtsmm, pid, sides.kin_jtsmm, sides.cyc_jtsmm, tt, seed,
                               "jtsmm", "jtsmm_optical"),
                 timing_tables(sides.delsys, pid, sides.kin_delsys, sides.cyc_delsys, tt, seed,
                               "delsys", "jtsmm_optical")]
        if sides.kin_qualisys:
            parts.append(timing_tables(sides.delsys, pid, sides.kin_qualisys, sides.cyc_delsys,
                                       tt, seed, "delsys", "qualisys"))
        for k in ("onsets", "summary", "ensemble", "release_summary"):
            dfs = [x[k] for x in parts if not x[k].empty]
            tables[f"timing_{k}"] = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        tables["timing_compare"] = lead_comparison(tables["timing_onsets"], pid, seed,
                                                   int(p.get("n_boot", 2000)),
                                                   rule=str(tt.get("onset_rule", "drive")))
    return tables, arrays


def prepare_sides(cmp_session: Any, ref: dict[str, Any], dataset_root: Path,
                  sync_row: Any, p: dict[str, Any], dp: dict[str, Any], pid: str
                  ) -> tuple[EmgSides, dict[str, pd.DataFrame], dict[str, np.ndarray],
                             list[Any]]:
    """Read and prepare both systems. Returns the sides, the timeline tables (δ check,
    Delsys metadata), their arrays, and the opened dataset sessions (caller closes)."""
    from biomech.io.reference import read_delsys_csv, read_qtm_mat
    from biomech.io.session import SessionData
    from biomech.kinematics import se3
    from biomech.kinematics.pose_chain import KinParams
    from biomech.validation import sync as SC

    p = {**DEFAULTS, **(p or {})}
    emgp = EmgParams.from_dict(dict(dp.get("emg") or {}))
    kp = KinParams.from_dict(dp.get("kinematics"))
    cyc_p = dict(dp.get("cycles") or {})
    t_off = int(round(0.5 * float(p["exposure_s"]) * 1e6))

    # JTS-MM: dataset take + its MVC take (by path), the dataset EMG code unchanged
    opened: list[Any] = []
    ds = SessionData(Path(dataset_root) / ref["dataset_take"])
    opened.append(ds)
    dmvc = None
    if ref.get("dataset_mvc_take"):
        dmvc = SessionData(Path(dataset_root) / ref["dataset_mvc_take"])
        opened.append(dmvc)
    ds_opt = ds.optical()
    ds_ct = (ds_opt.t_us, se3.valid(ds_opt.poses["ct"])) if ds_opt is not None and \
        "ct" in ds_opt.poses else (None, None)
    ds_view = PhasedSession(ds, select_phases(ds.phases, *ds_ct, guard_s=p["ct_guard_s"]))
    names = [m for m in ds.muscle_names() if m in MUSCLES]
    prep_j = MultiMvcPrep(ds_view, [dmvc], emgp)
    cyc_j, kin_j = cycle_kinematics(ds, pid, kp, cyc_p, hold="phase", t_offset_us=t_off)

    # Delsys timeline: δ, then QTM -> master
    model = SC.TimeModel(float(sync_row["offset_s"]), float(sync_row["alpha_ppm"]) * 1e-6,
                         float(sync_row["t_ref_s"]))
    qtm = read_qtm_mat(ref["qtm_mat"])
    dcsv = read_delsys_csv(ref["delsys_csv"])
    opt = cmp_session.optical()
    ct = (opt.t_us, se3.valid(opt.poses["ct"])) if opt is not None and "ct" in opt.poses \
        else (None, None)
    cphases = select_phases(cmp_session.phases, *ct, guard_s=p["ct_guard_s"])
    rest = next((ph for ph in cphases if ph.name == "rest"), None)
    oc = next((ph for ph in cphases if ph.name == "open_close"), None)

    def q_win(ph: Any) -> tuple[float, float]:
        return (float(model.to_q(ph.t_start_us * 1e-6)), float(model.to_q(ph.t_end_us * 1e-6)))

    jaw = qualisys_jaw(qtm, baseline_s=float(p["speed_baseline_s"]),
                       ref_window_s=q_win(rest) if rest else None,
                       axis_window_s=q_win(oc) if oc else None)
    wins = {ph.name: q_win(ph) for ph in cphases if ph.name in p["xcorr_tasks"]}
    envs = group_envelopes(dcsv)
    cands = [float(x) for x in p["delta_candidates_s"]]
    dtab, darr, dinfo = delta_check(envs, jaw, wins, cands, float(p["xcorr_half_range_s"]),
                                    float(p["xcorr_grid_hz"]), p["lead_window_s"])
    if str(p["delta_s"]).lower() != "auto":
        dinfo = {**dinfo, "delta_s": float(p["delta_s"]), "status": "fixed"}
    delta = float(dinfo["delta_s"])
    dtab.insert(0, "pid", pid)

    fs = float(p["fs_hz"] or ds.emg_fs)
    take = delsys_take(dcsv, names, fs, time_map=lambda td: model.to_s(td - delta),
                       full_scale_uv=float(p["full_scale_uv"]), name=f"{pid}_delsys")
    take.phases = clip_phases(cphases, int(take.t_us[0]), int(take.t_us[-1]))
    # Recording file names and dates carry initials and dates: never stored. MVC files are
    # "mvc1", "mvc2", ... (manifest order); their recording day only relative to the
    # comparison take.
    mvcs, files, info = [], [], []
    for k, f in enumerate(ref.get("delsys_mvc_csv") or []):
        dm = read_delsys_csv(f)
        label = f"mvc{k + 1}"
        mt = delsys_take(dm, names, fs, full_scale_uv=float(p["full_scale_uv"]), name=label)
        ph, inf = segment_mvc(mt, float(p["mvc_smooth_s"]))
        mt.phases = ph
        mvcs.append(mt)
        files.append(label)
        info.append({"file": label,
                     "days_before_comparison": _days_between(dm.header.get("Date/Time", ""),
                                                             dcsv.header.get("Date/Time", "")),
                     **inf})
    prep_d = MultiMvcPrep(take, mvcs, emgp)
    cyc_d, kin_d = cycle_kinematics(cmp_session, pid, kp, cyc_p, hold="still", t_offset_us=t_off)

    # Qualisys opening angle on the master clock (second mechanical source for Delsys)
    kin_q: dict[str, np.ndarray] = {}
    tq_us = np.round(np.asarray(model.to_s(jaw["t_s"])) * 1e6).astype(np.int64)
    for ph in cphases:
        sel = (tq_us >= ph.t_start_us) & (tq_us <= ph.t_end_us)
        if sel.sum() > 10 and np.isfinite(jaw["theta_deg"][sel]).any():
            kin_q[f"{ph.name}__t_us"] = tq_us[sel]
            kin_q[f"{ph.name}__theta_deg"] = jaw["theta_deg"][sel]

    meta = pd.DataFrame([{
        "pid": pid, "delta_s": delta, "delta_status": dinfo["status"],
        **{k: v for k, v in dinfo.items() if k not in ("delta_s", "status")},
        "delsys_length_s": dcsv.length_s, "qtm_duration_s": float(qtm.n_frames / qtm.fs_hz),
        "duration_diff_s": float(dcsv.length_s - qtm.n_frames / qtm.fs_hz),
        "qtm_minus_delsys_start_s": _seconds_between(dcsv.header.get("Date/Time", ""),
                                                     qtm.wall_start),
        "qtm_n_events": len(qtm.events),
        "qtm_events": ";".join(f"{e.get('Label', '')}@{e.get('Time', '')}" for e in qtm.events),
        "delsys_header_keys": ";".join(sorted(dcsv.header)),
        "delsys_start_mode": dcsv.header.get("Start Mode", dcsv.header.get("Trigger", "")),
        "delsys_rates_hz": ";".join(f"{m}={c.fs_hz:g}" for m, c in dcsv.channels.items()),
        "delsys_unmapped": ";".join(dcsv.unmapped), "resample_hz": fs,
        "offset_s": float(sync_row["offset_s"]), "alpha_ppm": float(sync_row["alpha_ppm"]),
        "t_ref_s": float(sync_row["t_ref_s"]),
    }])
    sides = EmgSides(prep_j, prep_d, cyc_j, cyc_d, kin_j, kin_d, kin_q or None, files, info)
    return sides, {"delta_check": dtab, "timeline": meta}, darr, opened


def _parse_wall(s: str) -> Any:
    from datetime import datetime

    s = str(s).replace(",", "").strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _days_between(a: str, b: str) -> float:
    """Whole days from wall-clock ``a`` to ``b`` (NaN when unparsable); dates themselves are
    never stored."""
    da, db = _parse_wall(a), _parse_wall(b)
    return float((db.date() - da.date()).days) if da and db else float("nan")


def _seconds_between(a: str, b: str) -> float:
    da, db = _parse_wall(a), _parse_wall(b)
    return float((db - da).total_seconds()) if da and db else float("nan")


def run_ref_emg(cmp_session: Any, ref: dict[str, Any], dataset_root: Path, sync_row: Any,
                p: dict[str, Any], dp: dict[str, Any], pid: str, seed: int
                ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """The work of ``ref_emg``: prepare both systems (:func:`prepare_sides`) and compute
    every D5 outcome (:func:`compare`). ``dp`` carries the dataset pipeline's parameters
    (``emg`` signals, ``kinematics`` signals, ``cycles``, ``timing``, ``cycle_mask_max``,
    ``usability``)."""
    sides, ttabs, tarr, opened = prepare_sides(cmp_session, ref, dataset_root, sync_row, p, dp,
                                               pid)
    try:
        tables, arrays = compare(sides, pid, dp, {**DEFAULTS, **(p or {})}, seed)
    finally:
        for sd in opened:
            sd.close()
    tables.update(ttabs)
    arrays.update(tarr)
    for s, cyc in (("jtsmm", sides.cyc_jtsmm), ("delsys", sides.cyc_delsys)):
        if not cyc.empty:
            c = cyc[["task", "side", "cycle", "ok", "t_start_us", "t_peak_us", "t_end_us"]].copy()
            c.insert(0, "system", s)
            c.insert(0, "pid", pid)
            tables.setdefault("cycles", pd.DataFrame())
            tables["cycles"] = pd.concat([tables["cycles"], c], ignore_index=True)
    return tables, arrays
