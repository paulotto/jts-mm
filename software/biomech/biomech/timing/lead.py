# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Does muscle activity precede jaw motion, and by how much? (C2, an05)

Per kinematic cycle of ``open_close`` (primary) and ``chewing``:

Mechanical onset, three independent sources on the IMU 200 Hz grid:
* ``omega``  opening angular velocity (calibrated IMU orientation): backward
             from the peak velocity to the last sample below ``vel_frac`` of it;
* ``acc``    mandible-minus-head linear acceleration magnitude (raw accel,
             gravity removed): backward from its peak to the last sample below
             the rest baseline (median + ``acc_k`` x robust SD);
* ``opt``    the optical opening angle's velocity, as for omega (cross-check,
             40 Hz source).
The mechanical onset is the earliest of omega and acc (analysis_plan §4.3);
opt is reported alongside.

EMG onset: openers (digastric) for opening, closers (masseter, temporalis) for
closing, in a window around the mechanical onset, thresholds from the same
channel's rest phase; three detectors offline plus their causal twins
(:mod:`jtsmm.emg.onset`). A channel-cycle counts only if its burst rises at
least ``burst_min_db`` above rest, the channel was not already active at
the window start (tonic activity has no onset) and more than ``min_finite_share`` of
the window's samples are unmasked (default 0.9; a literal before 2026-11-03).

lead = t_mechanical - t_EMG (> 0: EMG first). anticipation = t_mechanical -
t_causal_decision: the margin a real-time controller would actually have.

Signals (E27): offline onsets use the zero-phase cleaned EMG; the causal twins
(``t_decision_us``, ``causal_delay_ms``, ``anticipation_ms``) and the X3 release detector use
the causally cleaned EMG (:func:`biomech.emg.causal.causal_clean`), with their thresholds from
the causally cleaned rest phase. Rest thresholds pool the detector signal over the NaN-free
runs of the masked rest phase (no concatenation across mask gaps, E18); a channel with less
than ``MIN_REST_S`` of valid rest gets no threshold and is skipped (A3).

Chance null (analysis_plan D23, table ``lead_null``): the window is asymmetric (−600 … +200 ms),
so a burst unrelated to the movement still gives a positive "first" lead. The same per-window
analysis (:class:`WindowOnsets`) is repeated at 200 pseudo-onsets per task
(:func:`pseudo_onset_null`): in open_close and chewing (:data:`NULL_SHIFT_TASKS`, D23a) each
pseudo-onset is a real onset with the EMG circularly shifted against the motion (S1 of
:mod:`biomech.timing.surrogates`); random times ≥ 1 s from every real onset only for tasks with
pauses (none in the study). ``stats_timing`` tests the per-participant observed − null lead
next to the frozen test against 0. What it tests (D34): the null median is the window bias of
an unrelated burst (≈ 200–330 ms), so observed − null > 0 has no power for physiological leads;
it quantifies the window bias, not the order. The coupling test of the post hoc family C2-E
(X6) is :mod:`biomech.timing.c2e`.

Mechanical-onset sources (E34): ``acc`` is the unsigned, unfiltered jaw-minus-head |a_rel| (no
lever-arm or rotation correction); it is the earliest source in most cycles and pulls the
primary onset min(omega, acc) early. ``alpha`` (C2-E, X6) is the signed angular-acceleration
onset (:func:`alpha_signal`), computed when the IMU arrays carry ``theta_quat``; it never
enters ``t_mech``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from jtsmm.emg.onset import (OnsetParams, baseline_stats, baseline_threshold, detect_onsets,
                             detector_signal, odd_window)

from biomech import timebase
from biomech.emg.artifacts import masked_envelope
from biomech.emg.prepare import EmgPrep
from biomech.io.session import CLOSERS, OPENERS
from biomech.timing.ensemble import EnsembleParams
from biomech.timing.ensemble import analyse as analyse_ensemble
from biomech.timing.release import ReleaseParams
from biomech.timing.release import analyse as analyse_release
from biomech.timing import surrogates
from biomech.timing.surrogates import null_seed

DETECTORS = ("tkeo", "hodges_bui", "envelope")


def _backward_onset(t: np.ndarray, x: np.ndarray, thr: float, i_peak: int) -> int | None:
    below = np.flatnonzero(~(x[: i_peak + 1] >= thr))
    if below.size == 0:
        return None
    k = int(below[-1]) + 1
    return k if k <= i_peak else None



def select_onset(onsets: np.ndarray, active: np.ndarray, i_mech: int, n: int,
                 min_active: float = 0.5) -> int | None:
    """The onset of the burst that drives the movement (analysis_plan D6).

    Candidates are the detected onsets up to the end of the window (``n``). The
    selected one is the *last* onset at or before the mechanical onset whose
    burst is still on: the detector signal is above threshold for at least
    ``min_active`` of the samples from that onset to ``i_mech``. Without such an
    onset, the first onset after ``i_mech`` (EMG after motion, a negative lead)
    is used. An earlier, unrelated burst that has ended before the movement is
    never chosen; the first-onset-in-window rule chose it whenever it existed.
    """
    on = np.asarray(onsets, dtype=np.int64)
    on = on[(on >= 0) & (on < n)]
    before = on[on <= i_mech]
    for k in before[::-1]:
        seg = active[k:max(i_mech, k + 1)]
        if seg.size and seg.mean() >= min_active:
            return int(k)
    after = on[on > i_mech]
    return int(after[0]) if after.size else None

# -- signed angular acceleration (C2-E, X6 (change log) B1) --------------------------------
ALPHA_CUTOFF_HZ = 15.0        # defaults of stages.timing.c2e (alpha_cutoff_hz, alpha_frac,
ALPHA_ORDER = 4               # alpha_guard_s); the stage passes the configured values
ALPHA_FRAC = 0.05
ALPHA_GUARD_S = 0.05


def alpha_signal(t_us: np.ndarray, theta_deg: np.ndarray, cutoff_hz: float = ALPHA_CUTOFF_HZ,
                 order: int = ALPHA_ORDER) -> np.ndarray:
    """Signed angular acceleration along the opening axis (deg/s², > 0 = accelerating the
    opening): the opening angle through an ``order``-th order Butterworth low-pass at
    ``cutoff_hz``, zero-phase, per finite segment (shorter segments than the filter's padding
    stay NaN), then the gap-aware second derivative (:func:`biomech.timebase.gradient`)."""
    from scipy.signal import butter, sosfiltfilt

    t_us = np.asarray(t_us)
    th = np.asarray(theta_deg, dtype=np.float64)
    out = np.full(th.shape, np.nan)
    if th.size < 4:
        return out
    fs = 1e6 / float(np.median(np.diff(t_us)))
    if not 0 < cutoff_hz < 0.5 * fs:
        raise ValueError(f"alpha cutoff {cutoff_hz} Hz outside (0, {0.5 * fs}) Hz")
    sos = butter(order, cutoff_hz, fs=fs, output="sos")
    pad = 3 * (2 * len(sos) + 1)
    filt = np.full(th.shape, np.nan)
    for a, b in timebase.segments(np.isfinite(th)):
        if b - a > pad:
            filt[a:b] = sosfiltfilt(sos, th[a:b])
    v = timebase.gradient(t_us, filt)
    out[:] = timebase.gradient(t_us, v)
    return out


def _alpha_onset(t: np.ndarray, omega: np.ndarray, alpha: np.ndarray, sel: np.ndarray, ip: int,
                 direction: str, base: dict[str, float], alpha_t0: float | None,
                 frac: float, guard_s: float, w_thr: float) -> float:
    """t_alpha (us; NaN when none) of :func:`mechanical_onsets` (its docstring); ``w_thr`` =
    the omega onset threshold (the closing counts as on while -omega exceeds it)."""
    tt = t[sel]
    if direction == "opening":
        w = omega[sel]
        closing = np.flatnonzero(w[:ip] < -w_thr)
        i0 = 0
        if closing.size:
            dt = float(np.median(np.diff(tt))) if tt.size > 1 else 1.0
            i0 = int(closing[-1]) + 1 + int(round(guard_s * 1e6 / dt))
        ta, da = tt[i0:ip + 1], alpha[sel][i0:ip + 1]
    else:
        t_end = tt[ip]
        a_sel = (t >= (alpha_t0 if alpha_t0 is not None and np.isfinite(alpha_t0)
                       else tt[0])) & (t <= t_end)
        ta, da = t[a_sel], -alpha[a_sel]
    if ta.size < 3 or not np.isfinite(da).any():
        return np.nan
    ia = int(np.nanargmax(da))
    if not da[ia] > 0:
        return np.nan
    thr = max(frac * da[ia], base.get("alpha_noise", 0.0))
    k = _backward_onset(ta, np.nan_to_num(da, nan=-np.inf), thr, ia)
    return float(ta[k]) if k is not None else np.nan


def mechanical_onsets(t: np.ndarray, omega: np.ndarray, acc: np.ndarray, theta_opt: np.ndarray,
                      t0: int, t1: int, direction: str, base: dict[str, float],
                      p: dict[str, Any], alpha: np.ndarray | None = None,
                      alpha_t0: float | None = None) -> dict[str, float]:
    """Onset times (us) of opening (+omega) or closing (-omega) within [t0, t1].

    ``t_omega``, ``t_acc``, ``t_opt`` as in the module docstring; ``t_peak`` = time of the
    directional omega peak. With ``alpha`` (:func:`alpha_signal` on the same grid; C2-E, X6
    B1) also ``t_alpha``, the signed angular-acceleration onset: backward from the directional
    alpha peak to the last sample below max(``alpha_frac`` x peak, ``base['alpha_noise']``)
    (``p['c2e']``; defaults :data:`ALPHA_FRAC`, :data:`ALPHA_GUARD_S`), searched
    * opening (+alpha): from ``alpha_guard_s`` after the previous closing omega has ended (the
      last sample before the omega peak whose closing velocity exceeds the omega onset
      threshold), so the tooth impact at occlusion is excluded, to the omega peak;
    * closing (-alpha, the braking of the opening): from ``alpha_t0`` (the opening onset of
      the same cycle; the window start without it) to the closing omega peak.
    """
    sel = (t >= t0) & (t <= t1)
    out = {"t_omega": np.nan, "t_acc": np.nan, "t_opt": np.nan, "peak_vel_dps": np.nan,
           "t_alpha": np.nan, "t_peak": np.nan}
    if sel.sum() < 10:
        return out
    tt = t[sel]
    sgn = 1.0 if direction == "opening" else -1.0
    w = sgn * omega[sel]
    if np.isfinite(w).any():
        ip = int(np.nanargmax(w))
        out["peak_vel_dps"] = float(w[ip])
        out["t_peak"] = float(tt[ip])
        thr = max(p["vel_frac"] * w[ip], base["omega_noise"])
        k = _backward_onset(tt, np.nan_to_num(w, nan=-np.inf), thr, ip)
        if k is not None:
            out["t_omega"] = float(tt[k])
        if alpha is not None:
            c = p.get("c2e") or {}
            out["t_alpha"] = _alpha_onset(t, omega, alpha, sel, ip, direction, base, alpha_t0,
                                          float(c.get("alpha_frac", ALPHA_FRAC)),
                                          float(c.get("alpha_guard_s", ALPHA_GUARD_S)), thr)
        a = np.where(np.arange(acc[sel].size) <= ip, acc[sel], np.nan)
        if np.isfinite(a).any():
            ia = int(np.nanargmax(a))
            k = _backward_onset(tt, np.nan_to_num(a, nan=-np.inf), base["acc_thr"], ia)
            if k is not None:
                out["t_acc"] = float(tt[k])
    th = theta_opt[sel]
    if np.isfinite(th).sum() > 5:
        # Gap-aware derivative: NaN stays NaN, no zero-fill spike; all-NaN guarded (A3).
        v = sgn * timebase.gradient(tt, th)
        if np.isfinite(v).any():
            ip = int(np.nanargmax(v))
            k = _backward_onset(tt, np.nan_to_num(v, nan=-np.inf), p["vel_frac"] * v[ip], ip)
            if k is not None:
                out["t_opt"] = float(tt[k])
    return out


MIN_REST_S = 0.5             # valid rest needed for a channel's threshold (A3)


def _threshold(rest_x: np.ndarray, fs: float, op: OnsetParams, *, causal: bool = False,
               min_valid_s: float = MIN_REST_S) -> float:
    """Rest threshold of one channel: detector signal pooled over the NaN-free runs of the
    masked rest signal (E18); NaN with less than ``min_valid_s`` of valid rest (A3)."""
    return baseline_threshold(rest_x, fs, op, causal=causal, min_valid_s=min_valid_s)


# Onset selection rules, both always computed (analysis_plan D6): ``first`` = the first
# onset in the window (frozen plan), ``drive`` = the burst that drives the movement.
ONSET_RULES = ("first", "drive", "drive_taskbase")
TASKBASE_S = 0.3              # drive_taskbase: threshold from the first 0.3 s of the window
KEYS = ["pid", "task", "direction", "muscle", "detector", "rule"]


def _row(pid, task, c, direction, m, d, rule, mech, t_mech, t_on, t_dec, burst_db, active0,
         usable) -> dict[str, Any]:
    return {
        "pid": pid, "task": task, "cycle": int(c.cycle), "side": c.side,
        "direction": direction, "muscle": m, "detector": d, "rule": rule,
        "t_mech_us": t_mech, "t_omega_us": mech["t_omega"],
        "t_acc_us": mech["t_acc"], "t_opt_us": mech["t_opt"],
        "peak_vel_dps": mech["peak_vel_dps"],
        "t_onset_us": t_on, "t_decision_us": t_dec,
        "lead_ms": (t_mech - t_on) / 1e3,
        "lead_omega_ms": (mech["t_omega"] - t_on) / 1e3,
        "lead_acc_ms": (mech["t_acc"] - t_on) / 1e3,
        "lead_opt_ms": (mech["t_opt"] - t_on) / 1e3,
        "causal_delay_ms": (t_dec - t_on) / 1e3,
        "anticipation_ms": (t_mech - t_dec) / 1e3,
        "burst_db": burst_db, "active_at_start": active0,
        "usable": bool(usable),
        # C2-E (X6 P4): the same EMG onset re-referenced to the signed-alpha onset.
        "t_alpha_us": mech["t_alpha"], "lead_alpha_ms": (mech["t_alpha"] - t_on) / 1e3,
    }


def primary(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Rows of the configured (primary) onset rule; tables without a rule column pass."""
    return df[df["rule"] == rule] if "rule" in df.columns and not df.empty else df


def _rest_level(x: np.ndarray, mask: np.ndarray, fs: float, window_s: float) -> np.ndarray:
    """Median masked envelope (uV) per channel of a rest signal (NaN = masked)."""
    m = np.asarray(mask, bool) | ~np.isfinite(x)
    env = masked_envelope(np.nan_to_num(x), m, fs, window_s)
    env[m] = np.nan
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(env, axis=0)


_NONE = np.empty(0, dtype=np.int64)


class WindowOnsets:
    """Onsets of one EMG window under every detector and onset rule, with the usability
    criteria (analysis_plan §3.1, D6). Shared by the per-cycle analysis, the chance null
    (:func:`pseudo_onset_null`, D23) and the C2-E surrogates (X6), so all apply the same
    thresholds, detectors, rules and usability criteria; built once per participant by
    :func:`build_detector`.

    ``detectors``, ``rules`` and ``causal`` restrict what is evaluated (the C2-E surrogates
    need TKEO, drive_taskbase and no causal twin only; values of what is evaluated do not
    change); :meth:`batch` is the vectorised evaluation of that restricted case."""

    def __init__(self, names: list[str], fs: float, params: dict[str, OnsetParams],
                 thr: dict[tuple[str, int], float], rest_ok: dict[int, bool], rest_thr: Any,
                 rest_env_med: np.ndarray, burst_min_db: float, *,
                 detectors: tuple[str, ...] = DETECTORS,
                 rules: tuple[str, ...] = ONSET_RULES, causal: bool = True,
                 min_finite_share: float = 0.9) -> None:
        self.names, self.fs, self.params, self.thr = list(names), fs, params, thr
        self.rest_ok, self.rest_thr, self.rest_env_med = rest_ok, rest_thr, rest_env_med
        self.burst_min_db = burst_min_db
        self.min_finite_share = float(min_finite_share)
        unknown = (set(detectors) - set(DETECTORS)) | (set(rules) - set(ONSET_RULES))
        if unknown:
            raise ValueError(f"unknown detectors/rules {sorted(unknown)}")
        self.detectors, self.rules, self.causal = tuple(detectors), tuple(rules), bool(causal)

    def restricted(self, *, detectors: tuple[str, ...] = ("tkeo",),
                   rules: tuple[str, ...] = ("drive_taskbase",), causal: bool = False
                   ) -> "WindowOnsets":
        """The same thresholds with fewer detectors / rules / no causal twin (C2-E)."""
        return WindowOnsets(self.names, self.fs, self.params, self.thr, self.rest_ok,
                            self.rest_thr, self.rest_env_med, self.burst_min_db,
                            detectors=detectors, rules=rules, causal=causal,
                            min_finite_share=self.min_finite_share)

    def __call__(self, t_mech: float, tw: np.ndarray, sel: np.ndarray, x: np.ndarray,
                 xc_all: np.ndarray, cleaned: np.ndarray, mask: np.ndarray,
                 group: tuple[str, ...]):
        """Yield ``(muscle, detector, rule, t_on, t_dec, burst_db, active, usable)`` for the
        window ``sel`` (boolean or index selection of the phase samples) with sample times
        ``tw``; ``x`` = the masked zero-phase EMG (NaN = masked), ``xc_all`` = the causal
        chain, ``cleaned``/``mask`` = the cleaned EMG and its analysis mask (phase arrays)."""
        names, fs, thr = self.names, self.fs, self.thr
        for m in group:
            if m not in names:
                continue
            i = names.index(m)
            if not self.rest_ok[i]:
                continue
            xi = x[sel, i]
            xci = xc_all[sel, i]
            # Masked samples left out of the RMS, not zero-filled (E18).
            env = masked_envelope(cleaned[sel, i], mask[sel, i], fs, 0.05)
            peak = float(np.nanmax(env)) if np.isfinite(env).any() else np.nan
            rem = self.rest_env_med[i]
            burst_db = float(20 * np.log10(peak / rem)) if rem > 0 and peak > 0 else np.nan
            rest_rules = bool({"first", "drive"} & set(self.rules))
            for d in self.detectors:
                op = self.params[d]
                if not np.isfinite(thr[(d, i)]):
                    continue
                sig = detector_signal(np.nan_to_num(xi), fs, op)
                n0 = max(int(0.05 * fs), 1)
                active0 = bool(np.median(sig[:n0]) > thr[(d, i)])
                on = (detect_onsets(xi, fs, None, op, baseline_stats=self.rest_thr(d, i, False))
                      if rest_rules else _NONE)
                # Causal twin on the causal chain, threshold from the causal rest.
                dec = (detect_onsets(xci, fs, None, op, causal=True,
                                     baseline_stats=self.rest_thr(d, i, True))
                       if self.causal and rest_rules else _NONE)
                i_mech = int(np.searchsorted(tw, t_mech))
                # Sensitivity rule (X3 plan W3.3): threshold from this cycle's own
                # pre-movement samples (-600..-300 ms) instead of the rest phase.
                n_tb = max(int(TASKBASE_S * fs), 1)
                pre, pre_c = xi[:n_tb], xci[:n_tb]
                tb = None
                if "drive_taskbase" in self.rules and np.isfinite(pre).sum() >= int(0.2 * fs):
                    thr_tb = _threshold(pre, fs, op, min_valid_s=0.0)
                    if np.isfinite(thr_tb):
                        tb = (detect_onsets(xi, fs, None, op, baseline_signal=pre),
                              detect_onsets(xci, fs, None, op, causal=True,
                                            baseline_signal=pre_c) if self.causal else _NONE,
                              bool(np.median(sig[:n0]) > thr_tb), sig > thr_tb)
                for rule in self.rules:
                    on_r, dec_r, act_r = on, dec, active0
                    if rule == "first":                         # frozen plan (pre-D6)
                        k = int(on[0]) if on.size else None
                    elif rule == "drive":
                        k = select_onset(on, sig > thr[(d, i)], i_mech, len(tw))
                    else:                                       # drive_taskbase
                        if tb is None:
                            continue
                        on_r, dec_r, act_r, above = tb
                        k = select_onset(on_r, above, i_mech, len(tw))
                    t_on = float(tw[k]) if k is not None else np.nan
                    # The causal decision must belong to the same burst: the first
                    # decision at or after the selected offline onset (an earlier
                    # decision would be a different burst).
                    dk = dec_r[dec_r >= k] if k is not None else dec_r[:0]
                    t_dec = float(tw[dk[0]]) if dk.size else np.nan
                    usable = (np.isfinite(t_on) and not act_r
                              and np.isfinite(burst_db) and burst_db >= self.burst_min_db
                              and np.isfinite(xi).mean() > self.min_finite_share)
                    yield m, d, rule, t_on, t_dec, burst_db, act_r, usable

    def batch(self, x: np.ndarray, cleaned: np.ndarray, mask: np.ndarray, sel: np.ndarray,
              i_mech: int, group: tuple[str, ...], chunk: int = 2048
              ) -> dict[str, dict[str, np.ndarray]]:
        """Vectorised :meth:`__call__` for TKEO, drive_taskbase, offline (the C2-E surrogate
        case; :meth:`restricted`) over many windows of the same length.

        ``sel`` (M, Lw) holds the phase sample indices of M windows (e.g. from
        :func:`biomech.timing.surrogates.shifted_windows`), ``i_mech`` the window sample of
        the mechanical onset; ``x``, ``cleaned``, ``mask`` are the phase arrays of
        :meth:`__call__`. Per muscle of ``group`` that has a threshold: ``k_on`` (window
        sample of the selected onset, -1 = none), ``burst_db``, ``active`` (active at the
        window start) and ``usable``, each (M,). Windows with a masked sample go through
        :meth:`__call__` itself; the others are computed row-wise in arrays with the same
        operations (identical values, test_r11_c2e_units)."""
        if self.detectors != ("tkeo",) or self.rules != ("drive_taskbase",) or self.causal:
            raise ValueError("batch evaluates TKEO / drive_taskbase / offline only; "
                             "use WindowOnsets.restricted()")
        fs, op = self.fs, self.params["tkeo"]
        sel = np.asarray(sel, np.int64)
        M, Lw = sel.shape
        out: dict[str, dict[str, np.ndarray]] = {}
        tw = np.arange(Lw) * (1e6 / fs)
        for m in group:
            if m not in self.names:
                continue
            i = self.names.index(m)
            if not self.rest_ok[i] or not np.isfinite(self.thr[("tkeo", i)]):
                continue
            r = {"k_on": np.full(M, -1, np.int64), "burst_db": np.full(M, np.nan),
                 "active": np.zeros(M, bool), "usable": np.zeros(M, bool)}
            for a in range(0, M, chunk):
                s_ = sel[a:a + chunk]
                X = x[s_, i]
                fin = np.isfinite(X)
                env = masked_envelope(cleaned[s_, i].T, mask[s_, i].T, fs, 0.05)
                ok = np.isfinite(env).any(axis=0)
                peak = np.full(len(s_), np.nan)
                peak[ok] = np.nanmax(env[:, ok], axis=0)
                rem = self.rest_env_med[i]
                with np.errstate(divide="ignore", invalid="ignore"):
                    bdb = (20 * np.log10(peak / rem) if rem > 0 else np.full(len(s_), np.nan))
                bdb[~(peak > 0)] = np.nan
                fast = fin.all(axis=1)
                k = np.full(len(s_), -1, np.int64)
                act = np.zeros(len(s_), bool)
                if fast.any():
                    k[fast], act[fast] = _tkeo_taskbase_rows(X[fast], fs, op, i_mech)
                use = ((k >= 0) & ~act & np.isfinite(bdb) & (bdb >= self.burst_min_db)
                       & (fin.mean(axis=1) > self.min_finite_share))
                for j in np.flatnonzero(~fast):          # masked samples: the scalar path
                    k[j], act[j], use[j] = -1, False, False
                    for _, _, _, t_on, _, b_, a_, u_ in self(tw[i_mech], tw, s_[j], x, x,
                                                            cleaned, mask, (m,)):
                        k[j] = int(np.searchsorted(tw, t_on)) if np.isfinite(t_on) else -1
                        act[j], use[j], bdb[j] = bool(a_), bool(u_), b_
                r["k_on"][a:a + chunk], r["active"][a:a + chunk] = k, act
                r["usable"][a:a + chunk], r["burst_db"][a:a + chunk] = use, bdb
            out[m] = r
        return out


def _tkeo2(a: np.ndarray) -> np.ndarray:
    """Row-wise :func:`jtsmm.emg.onset.tkeo` of (R, L)."""
    out = np.empty_like(a)
    out[:, 1:-1] = a[:, 1:-1] ** 2 - a[:, :-2] * a[:, 2:]
    out[:, 0], out[:, -1] = out[:, 1], out[:, -2]
    return out


def _tkeo_taskbase_rows(X: np.ndarray, fs: float, op: OnsetParams, i_mech: int
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Rows of NaN-free windows (R, L): the drive_taskbase TKEO onset (window sample, -1 =
    none) and active-at-start, by the operations of ``jtsmm.emg.onset.detect_onsets``
    (offline TKEO, threshold and raw threshold from the first ``TASKBASE_S`` of the row) and
    :func:`select_onset`, row-wise."""
    from scipy.ndimage import uniform_filter1d

    R, n = X.shape
    w = odd_window(op.smooth_s, fs)
    raw = np.abs(_tkeo2(X))
    sig = uniform_filter1d(raw, size=w, axis=1, mode="nearest")
    n_tb = max(int(TASKBASE_S * fs), 1)
    rp = np.abs(_tkeo2(X[:, :n_tb]))
    sp = uniform_filter1d(rp, size=w, axis=1, mode="nearest")
    thr = sp.mean(axis=1) + op.k_sd * sp.std(axis=1)
    raw_thr = rp.mean(axis=1) + op.k_sd * rp.std(axis=1)
    n0 = max(int(0.05 * fs), 1)
    active = np.median(sig[:, :n0], axis=1) > thr
    above = sig > thr[:, None]
    # Runs of `above` per row (detect_onsets), in row-major order.
    d = np.diff(np.pad(above.astype(np.int8), ((0, 0), (1, 1))), axis=1)
    rs, a = np.nonzero(d == 1)
    _, b = np.nonzero(d == -1)
    k_on = np.full(R, -1, np.int64)
    if rs.size == 0:
        return k_on, active
    n_on = max(int(round(op.min_on_s * fs)), 1)
    n_off = max(int(round(op.min_off_s * fs)), 1)
    edge = int(round(op.edge_s * fs))
    h = (w - 1) // 2
    ln = b - a
    rejected = (ln < n_on) | (a < edge) | ((b > n - edge) & (ln < 2 * n_on))
    upd = ~rejected | (a < edge)                  # runs that move "last_end"
    idx = np.arange(rs.size)
    cm = np.maximum.accumulate(np.where(upd, idx, -1))
    prev = np.concatenate([[-1], cm[:-1]])
    has = (prev >= 0) & (rs[np.maximum(prev, 0)] == rs)
    last_end = np.where(has, b[np.maximum(prev, 0)], -n_off - 1)
    is_on = ~rejected & (a - last_end >= n_off)
    ro, ao = rs[is_on], a[is_on]
    # TKEO refinement: the first sample within +-h whose raw |TKEO| exceeds the raw threshold.
    lo = np.maximum(ao - h, 0)
    hi = np.minimum(ao + h + 1, n)
    cols = lo[:, None] + np.arange(2 * h + 1)[None, :]
    inside = cols < hi[:, None]
    hit = (raw[ro[:, None], np.minimum(cols, n - 1)] > raw_thr[ro][:, None]) & inside
    onset = np.where(hit.any(axis=1), lo + hit.argmax(axis=1), ao)
    # select_onset: the last onset <= i_mech whose burst is on for >= half of [k, i_mech),
    # else the first onset after i_mech.
    c = np.zeros((R, n + 1), np.int64)
    np.cumsum(above, axis=1, out=c[:, 1:])
    end = np.maximum(i_mech, onset + 1)
    frac = (c[ro, end] - c[ro, onset]) / (end - onset)
    good = (onset <= i_mech) & (frac >= 0.5)
    k_before = np.full(R, -1, np.int64)
    np.maximum.at(k_before, ro[good], onset[good])
    after = onset > i_mech
    k_after = np.full(R, np.iinfo(np.int64).max)
    np.minimum.at(k_after, ro[after], onset[after])
    k_on = np.where(k_before >= 0, k_before,
                    np.where(k_after < np.iinfo(np.int64).max, k_after, -1))
    return k_on, active


# -- chance null of the lead (D23) ---------------------------------------------------------
N_NULL = 200                  # pseudo-onsets per participant and task (D23)
NULL_GAP_S = 1.0              # pseudo-onsets >= 1 s from every real onset and the phase edges
# Continuous rhythm: circular shift of the EMG instead of pseudo-onsets. open_close added by
# D23a (2026-10-29): its cycles leave no span >= NULL_GAP_S from every real onset (r10: 0 s
# in 12 of 13 participants), so pseudo-onsets could not be drawn.
NULL_SHIFT_TASKS = ("chewing", "open_close")
DIRECTION_GROUPS = {"opening": OPENERS, "closing": CLOSERS}
NULL_KEYS = ["pid", "task", "direction", "detector", "rule"]


def allowed_intervals(t0: float, t1: float, exclude: np.ndarray, gap: float
                      ) -> list[tuple[float, float]]:
    """Sub-intervals of [t0 + gap, t1 - gap] at least ``gap`` from every ``exclude`` time."""
    iv = [(t0 + gap, t1 - gap)] if t1 - t0 > 2 * gap else []
    for e in np.sort(np.asarray(exclude, float)[np.isfinite(exclude)]):
        out = []
        for a, b in iv:
            if e + gap <= a or e - gap >= b:
                out.append((a, b))
                continue
            if e - gap > a:
                out.append((a, e - gap))
            if e + gap < b:
                out.append((e + gap, b))
        iv = out
    return [(a, b) for a, b in iv if b > a]


def draw_uniform(iv: list[tuple[float, float]], n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` times uniform on the union of the intervals ``iv`` (empty if it has no length)."""
    if not iv:
        return np.empty(0)
    a = np.array([v[0] for v in iv])
    ln = np.array([v[1] - v[0] for v in iv])
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    u = rng.uniform(0.0, cum[-1], n)
    j = np.clip(np.searchsorted(cum, u, side="right") - 1, 0, len(iv) - 1)
    return a[j] + (u - cum[j])


def pseudo_onset_null(det: WindowOnsets, pid: str, task: str,
                      phase: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                      onsets: dict[str, np.ndarray], exclude: np.ndarray, p: dict[str, Any],
                      seed: int = 0, n: int = N_NULL) -> list[dict[str, Any]]:
    """Chance level of the lead (analysis_plan D23): the per-cycle analysis repeated at
    ``n`` pseudo mechanical onsets per task, with the same window (``emg_pre_s``,
    ``emg_post_s``), detectors, onset rules and usability criteria (:class:`WindowOnsets`).

    * every task not in :data:`NULL_SHIFT_TASKS` (none in the study since D23a): pseudo-onsets drawn
      uniformly in the phase, at least :data:`NULL_GAP_S` from every real onset
      (``exclude``: the mechanical onsets of both directions plus the start and peak of the
      task's cycles flagged not ok) and from the phase edges; the same pseudo-onsets serve
      both directions;
    * ``chewing``, ``open_close`` (continuous rhythm, :data:`NULL_SHIFT_TASKS`): pseudo-onset
      k is the real onset k mod n_real of the
      direction (``onsets[direction]``) with the EMG circularly shifted within the phase by
      a uniform 1 s ... T - 1 s (T = phase length); one shift per pseudo-onset, shared by
      both directions.

    ``phase = (t_us, x, xc_all, cleaned, mask)`` of the task (as in :func:`analyse`).
    Returns one row per direction x detector x rule: ``lead_null_ms`` (median lead over the
    usable pseudo channel-cycles of the direction's muscle group, pooled over channels as
    the primary D15 value), ``lead_null_muscle_mean_ms`` (mean of the per-muscle medians,
    the ``muscle_mean`` aggregation), ``anticipation_null_ms``, ``usable_null_frac`` (usable
    share of the pseudo channel-cycles), ``n_null`` (pseudo-onsets evaluated),
    ``n_null_usable`` and ``null_span_s`` (time the pseudo-onsets were drawn from).
    """
    t_us, x, xc_all, cleaned, mask = phase
    rows: list[dict[str, Any]] = []
    if len(t_us) < 10 or n <= 0:
        return rows
    fs = det.fs
    rng = null_seed(seed, task)
    t0, t1 = float(t_us[0]), float(t_us[-1])
    gap = NULL_GAP_S * 1e6
    shift = task in NULL_SHIFT_TASKS
    # (direction) -> list of (t_mech, tw, sel) windows
    windows: dict[str, list[tuple[float, np.ndarray, np.ndarray]]] = {}
    if not shift:
        iv = allowed_intervals(t0, t1, exclude, gap)
        span = float(sum(b - a for a, b in iv)) / 1e6
        pseudo = draw_uniform(iv, n, rng)
        for direction in DIRECTION_GROUPS:
            ws = windows.setdefault(direction, [])
            for tm in pseudo:
                sel = (t_us >= tm - p["emg_pre_s"] * 1e6) & (t_us <= tm + p["emg_post_s"] * 1e6)
                if sel.sum() < int(0.2 * fs):
                    continue
                ws.append((float(tm), t_us[sel], sel))
    else:
        N = len(t_us)
        T = N / fs                                       # circular length (s)
        span = max(T - 2 * NULL_GAP_S, 0.0)
        if span > 0:
            # S1, one shift per pseudo-onset (surrogates.shifted_windows).
            shifts = np.round(surrogates.uniform_shifts(rng, T, n, NULL_GAP_S) * fs
                              ).astype(np.int64)
            n_pre = int(round(p["emg_pre_s"] * fs))
            n_post = int(round(p["emg_post_s"] * fs))
            off = np.arange(-n_pre, n_post + 1)
            for direction in DIRECTION_GROUPS:
                real = np.sort(np.asarray(onsets.get(direction, []), float))
                real = real[np.isfinite(real) & (real >= t0) & (real <= t1)]
                ws = windows.setdefault(direction, [])
                if real.size == 0:
                    continue
                i_r = np.clip(np.searchsorted(t_us, real[np.arange(n) % real.size]), 0, N - 1)
                i_e, sels = surrogates.shifted_windows(i_r, shifts, off, N)
                for ie, sel in zip(i_e.tolist(), sels):
                    tw = float(t_us[ie]) + off * (1e6 / fs)
                    ws.append((float(t_us[ie]), tw, sel))
    for direction, ws in windows.items():
        group = DIRECTION_GROUPS[direction]
        acc: dict[tuple[str, str], dict[str, Any]] = {}
        for tm, tw, sel in ws:
            for m, d, rule, t_on, t_dec, _, _, usable in det(tm, tw, sel, x, xc_all, cleaned,
                                                             mask, group):
                a = acc.setdefault((d, rule), {"n": 0, "lead": [], "ant": [], "mus": {}})
                a["n"] += 1
                if usable:
                    a["lead"].append((tm - t_on) / 1e3)
                    a["ant"].append((tm - t_dec) / 1e3)
                    a["mus"].setdefault(m, []).append((tm - t_on) / 1e3)
        for (d, rule), a in acc.items():
            lead = np.asarray(a["lead"], float)
            ant = np.asarray(a["ant"], float)
            mm = [float(np.median(v)) for v in a["mus"].values() if v]
            rows.append({
                "pid": pid, "task": task, "direction": direction, "detector": d, "rule": rule,
                "null_kind": "circular_shift" if shift else "pseudo_onset",
                "n_null": int(len(ws)), "n_null_rows": int(a["n"]),
                "n_null_usable": int(lead.size),
                "usable_null_frac": float(lead.size / a["n"]) if a["n"] else np.nan,
                "lead_null_ms": float(np.median(lead)) if lead.size else np.nan,
                "lead_null_muscle_mean_ms": float(np.mean(mm)) if mm else np.nan,
                "anticipation_null_ms": float(np.nanmedian(ant))
                if np.isfinite(ant).any() else np.nan,
                "null_span_s": span})
    return rows


def build_detector(prep: EmgPrep, p: dict[str, Any]
                   ) -> tuple[WindowOnsets, np.ndarray, np.ndarray]:
    """The participant's :class:`WindowOnsets` (rest thresholds of every detector and channel,
    rest-valid channels, rest envelope medians; all detectors, rules and the causal twin) and
    the masked rest signals ``(rest_x, rest_c)`` (zero-phase, causal; NaN = masked)."""
    names = prep.names
    rest = prep.phase("rest")
    rest_x = np.where(rest.mask, np.nan, rest.cleaned)
    rest_c = np.asarray(prep.causal("rest"))                 # causal chain, NaN = masked
    fs = rest.fs
    with np.errstate(invalid="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rest_env_med = np.nanmedian(np.where(rest.mask, np.nan, rest.env), axis=0)
    params = {d: OnsetParams(method=d, k_sd=p["k_sd"]) for d in DETECTORS}
    thr = {(d, i): _threshold(rest_x[:, i], fs, params[d]) for d in DETECTORS
           for i in range(len(names))}
    # Channels with too little valid rest (A3): no threshold, skipped below.
    rest_ok = {i: bool(np.isfinite(rest_x[:, i]).sum() >= MIN_REST_S * fs
                       and np.isfinite(rest_c[:, i]).sum() >= MIN_REST_S * fs)
               for i in range(len(names))}
    # Rest-phase detector thresholds, once per (detector, channel, chain) instead of once
    # per cycle (identical to passing the rest signal to every detect_onsets call).
    rest_stats: dict[tuple[str, int, bool], tuple[float, float] | None] = {}

    def rest_thr(d: str, i: int, causal: bool) -> tuple[float, float] | None:
        k = (d, i, causal)
        if k not in rest_stats:
            rest_stats[k] = baseline_stats(rest_c[:, i] if causal else rest_x[:, i], fs,
                                           params[d], causal=causal)
        return rest_stats[k]

    det = WindowOnsets(names, fs, params, thr, rest_ok, rest_thr, rest_env_med,
                       p["burst_min_db"], min_finite_share=float(p.get("min_finite_share", 0.9)))
    return det, rest_x, rest_c


def alpha_params(p: dict[str, Any]) -> tuple[float, int]:
    """(cutoff Hz, order) of the alpha filter from ``p['c2e']`` (defaults when absent)."""
    c = p.get("c2e") or {}
    return float(c.get("alpha_cutoff_hz", ALPHA_CUTOFF_HZ)), int(c.get("alpha_order",
                                                                         ALPHA_ORDER))


def mechanical_baselines(imu_arrays: dict[str, Any], p: dict[str, Any]) -> dict[str, float]:
    """Rest-phase noise levels of the mechanical sources: ``omega_noise`` (3 SD of omega),
    ``acc_thr`` (median + ``acc_k`` x robust SD of |a_rel|) and ``alpha_noise`` (3 SD of
    alpha, C2-E; 0 without the rest opening angle)."""
    om_r = imu_arrays.get("rest__omega_dps")
    ac_r = imu_arrays.get("rest__acc_rel_ms2")
    th_r = imu_arrays.get("rest__theta_quat")
    base = {
        "omega_noise": float(3 * np.nanstd(om_r)) if om_r is not None else 0.0,
        "acc_thr": float(np.nanmedian(ac_r) + p["acc_k"] * 1.4826
                         * np.nanmedian(np.abs(ac_r - np.nanmedian(ac_r))))
        if ac_r is not None else np.inf,
        "alpha_noise": 0.0,
    }
    if th_r is not None and "rest__t_us" in imu_arrays:
        al = alpha_signal(imu_arrays["rest__t_us"], th_r, *alpha_params(p))
        if np.isfinite(al).sum() > 1:
            base["alpha_noise"] = float(3 * np.nanstd(al))
    return base


def analyse(prep: EmgPrep, pid: str, imu_arrays: dict[str, Any], kin_cycles: pd.DataFrame,
            p: dict[str, Any], seed: int = 0, n_null: int = N_NULL
            ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """Single-cycle onsets (all rules) plus the event-aligned ensemble timing (X2), the
    causal release detector (X3) and the chance null of the lead (D23, table ``lead_null``;
    ``n_null`` = 0 skips it). ``seed`` (the configured study seed) drives the bootstrap
    (X2), the chance-level pseudo-onsets (X3) and the lead null (D23)."""
    names = prep.names
    rest = prep.phase("rest")
    fs = rest.fs
    det, rest_x, rest_c = build_detector(prep, p)
    base = mechanical_baselines(imu_arrays, p)
    rows = []
    events: dict[tuple[str, str], list[float]] = {}
    ens_phases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rel_phases: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    null_phases: dict[str, tuple[np.ndarray, ...]] = {}
    for task in p["tasks"]:
        if f"{task}__t_us" not in imu_arrays or kin_cycles.empty:
            continue
        t = imu_arrays[f"{task}__t_us"].astype(np.int64)
        om = imu_arrays[f"{task}__omega_dps"].astype(np.float64)
        ac = imu_arrays[f"{task}__acc_rel_ms2"].astype(np.float64)
        tho = imu_arrays[f"{task}__theta_opt"].astype(np.float64)
        thq = imu_arrays.get(f"{task}__theta_quat")
        al = (alpha_signal(t, np.asarray(thq, np.float64), *alpha_params(p))
              if thq is not None else None)
        pe = prep.phase(task)
        xc_all = np.asarray(prep.causal(task))
        ens_phases[task] = (pe.t_us, pe.cleaned, pe.mask)
        rel_phases[task] = (pe.t_us, xc_all, pe.mask)
        x = np.where(pe.mask, np.nan, pe.cleaned)
        null_phases[task] = (pe.t_us, x, xc_all, pe.cleaned, pe.mask)
        cyc = kin_cycles[(kin_cycles.task == task) & kin_cycles.ok]
        for _, c in cyc.iterrows():
            t_open = np.nan                      # the cycle's opening onset (closing alpha)
            for direction, (a0, a1), group in (
                    ("opening", (c.t_start_us - p["pre_s"] * 1e6, c.t_peak_us), OPENERS),
                    ("closing", (c.t_peak_us - 0.1e6, c.t_end_us), CLOSERS)):
                mech = mechanical_onsets(t, om, ac, tho, int(a0), int(a1), direction, base, p,
                                         alpha=al, alpha_t0=t_open)
                if direction == "opening":
                    t_open = mech["t_omega"]
                cands = [v for v in (mech["t_omega"], mech["t_acc"]) if np.isfinite(v)]
                if not cands:
                    continue
                t_mech = min(cands)
                events.setdefault((task, direction), []).append(t_mech)
                w0 = t_mech - p["emg_pre_s"] * 1e6
                w1 = t_mech + p["emg_post_s"] * 1e6
                sel = (pe.t_us >= w0) & (pe.t_us <= w1)
                if sel.sum() < int(0.2 * fs):
                    continue
                tw = pe.t_us[sel]
                for m, d, rule, t_on, t_dec, burst_db, act_r, usable in det(
                        t_mech, tw, sel, x, xc_all, pe.cleaned, pe.mask, group):
                    rows.append(_row(pid, task, c, direction, m, d, rule, mech, t_mech,
                                     t_on, t_dec, burst_db, act_r, usable))
    onsets = pd.DataFrame(rows)
    summary = pd.DataFrame()
    if not onsets.empty:
        u = onsets[onsets.usable]
        summary = (u.groupby(KEYS)
                   .agg(n=("lead_ms", "size"), lead_ms=("lead_ms", "median"),
                        lead_omega_ms=("lead_omega_ms", "median"),
                        lead_acc_ms=("lead_acc_ms", "median"),
                        lead_opt_ms=("lead_opt_ms", "median"),
                        causal_delay_ms=("causal_delay_ms", "median"),
                        anticipation_ms=("anticipation_ms", "median"))
                   .reset_index())
        summary["n_total"] = summary.merge(
            onsets.groupby(KEYS).size().rename("nt").reset_index(), on=KEYS
        )["nt"]
    tables = {"onsets": onsets, "summary": summary}

    # D23: the chance null of the lead, at pseudo-onsets (open_close) or with the EMG
    # circularly shifted against the motion (chewing); same window, detectors, rules and
    # usability as above. Real onsets: the mechanical onsets of both directions, plus the
    # start and peak of every cycle flagged not ok (a movement without a computed onset).
    null_rows: list[dict[str, Any]] = []
    if n_null > 0:
        for task, ph in null_phases.items():
            ons = {d: np.asarray(events.get((task, d), []), float) for d in DIRECTION_GROUPS}
            bad = kin_cycles[(kin_cycles.task == task) & ~kin_cycles.ok.astype(bool)]
            excl = np.concatenate([*ons.values(), bad.t_start_us.to_numpy(float),
                                   bad.t_peak_us.to_numpy(float)])
            null_rows += pseudo_onset_null(det, pid, task, ph, ons, excl, p, seed, n_null)
    tables["lead_null"] = pd.DataFrame(null_rows)

    # X2: detector-free timing of agonist rise and antagonist release (every cycle counts),
    # on the zero-phase signal.
    ep = EnsembleParams.from_dict(p.get("ensemble"))
    rest_level = _rest_level(rest_x, rest.mask, fs, ep.envelope_s)
    ens_rows, arrays = analyse_ensemble(list(names), ens_phases, rest_level,
                                        {k: np.asarray(v) for k, v in events.items()}, pid, ep,
                                        seed)
    tables["ensemble"] = pd.DataFrame(ens_rows)

    # X3: causal closer-release detector (what a real-time controller could use), on the
    # causal chain with the rest level of the same chain (E27).
    rp = ReleaseParams.from_dict(p.get("release"))
    rest_level_c = _rest_level(rest_c, rest.mask, fs, ep.envelope_s)
    opening = {task: np.asarray(v) for (task, d), v in events.items() if d == "opening"}
    rel_cyc, rel_task = analyse_release(list(names), rel_phases, rest_level_c, opening, pid, rp,
                                        seed, shift_tasks=NULL_SHIFT_TASKS)
    tables["release"] = pd.DataFrame(rel_cyc)
    tables["release_summary"] = pd.DataFrame(rel_task)
    return tables, arrays
