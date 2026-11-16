# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sampling rate and filter effects on the dynamic requirements (VD9).

The dataset paper derives its velocity requirements from the 40 Hz JTS-MM optics, low-passed
at 6 Hz and differentiated by a Savitzky-Golay filter (an02 §5). The 200 Hz Qualisys take
shows what that chain misses. On the same cycles, three pipelines compute the per-cycle peaks:

* ``ref200``: Qualisys at its native 200 Hz, low-passed at 20 Hz (well above the jaw's
  movement bandwidth) and differentiated with a 35 ms Savitzky-Golay window: the reference;
* ``ref40``: Qualisys at the JTS-MM frame times (40 Hz) through the dataset chain (6 Hz
  low-pass, 0.15 s Savitzky-Golay window): the dataset method applied to the reference;
* ``jtsmm``: JTS-MM through the dataset chain: the dataset as recorded.

Peaks per cycle: opening and closing angular velocity of the opening angle (deg/s) and the
incisal speed (mm/s). ``ref40 / ref200`` is the bias of the dataset chain alone,
``jtsmm / ref200`` the bias of the dataset as recorded; the reported ratio (EV7) is the
dataset chain's for the opening angular velocity in open_close, per participant
(``peak_ratio_open``); the other ratios stay in ``bias``.

Acceleration (VD24; the lower bound of the dataset's X7 peak |θ̈| row): the second
derivative of the same Savitzky-Golay fit that gives the opening rate (same chain, same
window), per cycle the peak |θ̈| of the opening angle (``peak_abs_acc_dps2``, open_close and
chewing as the angular velocities). Its ratios to the 200 Hz reference are in ``bias``
(quantity ``peak_abs_acc_dps2``, bootstrapped with their own generator so that the velocity
CIs keep their draws) and, for open_close per participant and pipeline, in
``peak_ratio_acc``.

Movement bandwidth (EV7, primary; ``spectrum_unfiltered``): Welch spectrum of the opening
rate from the **unfiltered** 200 Hz Qualisys opening angle by a central difference, minus
the noise floor measured the same way on the rest phase (negative differences clipped to 0);
f50/f95/f99 and the share of power above 20 Hz (the Nyquist frequency of 40 Hz optics) from
that. ``spectrum`` keeps the pre-EV7 values, computed on the ``ref200`` chain (20 Hz
low-pass and Savitzky-Golay derivative), which removes most power above 20 Hz by itself.
n = 2 participants: reported as a caveat next to the dataset requirement, never used to
correct it.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.kinematics import se3
from biomech.kinematics.derivatives import derivatives

PIPELINES = {"ref200": (20.0, 0.035), "ref40": (6.0, 0.15), "jtsmm": (6.0, 0.15)}
TASKS_ANGLE = ("open_close", "chewing")
VEL_QUANTITIES = ("peak_open_vel_dps", "peak_close_vel_dps", "peak_incisal_speed_mm_s")
ACC_QUANTITIES = ("peak_abs_acc_dps2",)


def _chain(t_s: np.ndarray, M: np.ndarray, cutoff_hz: float, window_s: float,
           rest_mask: np.ndarray, accel: bool = False) -> tuple[np.ndarray, ...]:
    """(opening-angle rate deg/s, incisal speed mm/s) of jaw-in-head poses ``M`` through
    a zero-phase low-pass and a Savitzky-Golay derivative; ``accel``: also the opening-angle
    acceleration (deg/s², the second derivative of the same fit, VD24) as a third value."""
    fs = 1.0 / float(np.median(np.diff(t_s)))
    Mf = se3.lowpass(M, fs, cutoff_hz, 2)
    ok = rest_mask & se3.valid(Mf)
    ref = Mf[ok][0] if ok.any() else Mf[se3.valid(Mf)][0]
    theta = np.degrees(se3.rotvec_rel(Mf, ref))[:, 1]
    t_us = t_s * 1e6
    dth, ddth = derivatives(t_us, theta, window_s=window_s)
    dp, _ = derivatives(t_us, Mf[:, :3, 3], window_s=window_s)
    return (dth, np.linalg.norm(dp, axis=1), ddth) if accel else (dth, np.linalg.norm(dp, axis=1))


def cycle_peaks(pid: str, signals: dict[str, tuple[np.ndarray, ...]],
                cycles: pd.DataFrame, t_frames_s: np.ndarray) -> pd.DataFrame:
    """Per cycle and pipeline the peak opening / closing angular velocity and incisal
    speed, and the peak |angular acceleration| of the opening angle (VD24). ``signals``:
    pipeline -> (t_s, opening rate, incisal speed[, opening acceleration]); ``cycles`` holds
    frame indices into ``t_frames_s`` (``ref_agreement:cycles``)."""
    rows = []
    for c in cycles.itertuples():
        t0, t1 = float(t_frames_s[int(c.start)]), float(t_frames_s[int(c.end)])
        for name, sig in signals.items():
            t, dth, sp = sig[:3]
            acc = sig[3] if len(sig) > 3 else None
            k = (t >= t0) & (t <= t1)
            if k.sum() < 3:
                continue
            ang = c.task in TASKS_ANGLE
            a_ok = ang and acc is not None and bool(np.isfinite(acc[k]).any())
            rows.append({"pid": pid, "task": c.task, "side": c.side, "cycle": int(c.cycle),
                         "pipeline": name,
                         "peak_open_vel_dps": float(np.nanmax(dth[k])) if ang else np.nan,
                         "peak_close_vel_dps": float(np.nanmax(-dth[k])) if ang else np.nan,
                         "peak_incisal_speed_mm_s": float(np.nanmax(sp[k])),
                         "peak_abs_acc_dps2": float(np.nanmax(np.abs(acc[k]))) if a_ok
                         else np.nan})
    return pd.DataFrame(rows)


def bias(peaks: pd.DataFrame, n_boot: int = 2000, seed: int = 0,
         quantities: tuple[str, ...] = VEL_QUANTITIES) -> pd.DataFrame:
    """Median ratio of each pipeline's peaks to the 200 Hz reference over the cycles of a
    task, with a cycle-bootstrap 95 % CI (one generator per call, drawn in the order of the
    groups and ``quantities``)."""
    rng = np.random.default_rng(seed)
    rows = []
    if peaks.empty:
        return pd.DataFrame(rows)
    key = ["pid", "task", "side", "cycle"]
    ref = peaks[peaks.pipeline == "ref200"].set_index(key)
    for (pid, task, name), g in peaks[peaks.pipeline != "ref200"].groupby(
            ["pid", "task", "pipeline"]):
        g = g.set_index(key)
        for q in quantities:
            if q not in g.columns:
                continue
            r = (g[q] / ref[q].reindex(g.index)).to_numpy(float)
            r = r[np.isfinite(r) & (r > 0)]
            if r.size < 3:
                continue
            boot = np.median(rng.choice(r, (n_boot, r.size)), axis=1)
            rows.append({"pid": pid, "task": task, "pipeline": name, "quantity": q,
                         "n": int(r.size), "ratio_median": float(np.median(r)),
                         "ratio_ci_lo": float(np.percentile(boot, 2.5)),
                         "ratio_ci_hi": float(np.percentile(boot, 97.5))})
    return pd.DataFrame(rows)


def spectrum(pid: str, t_s: np.ndarray, dth: np.ndarray, intervals: list[tuple[str, float, float]]
             ) -> pd.DataFrame:
    """Movement bandwidth of the ``ref200`` chain (20 Hz low-pass, Savitzky-Golay derivative;
    the pre-EV7 values, kept in ``spectrum``): f50/f95/f99 of the opening-rate power per
    task and the share of power above 20 Hz. The low-pass removes most of that power by
    itself; the primary is :func:`spectrum_unfiltered`."""
    from scipy.signal import welch

    fs = 1.0 / float(np.median(np.diff(t_s)))
    rows = []
    for task, a, b in intervals:
        k = (t_s >= a) & (t_s <= b) & np.isfinite(dth)
        x = dth[k]
        if x.size < 256:
            continue
        f, P = welch(x - x.mean(), fs, nperseg=min(512, x.size))
        c = np.cumsum(P) / np.sum(P)
        rows.append({"pid": pid, "task": task, "fs_hz": fs,
                     "f50_hz": float(f[np.searchsorted(c, 0.50)]),
                     "f95_hz": float(f[np.searchsorted(c, 0.95)]),
                     "f99_hz": float(f[min(np.searchsorted(c, 0.99), len(f) - 1)]),
                     "share_above_20hz": float(P[f > 20.0].sum() / P.sum())})
    return pd.DataFrame(rows)


def peak_ratio_open(b: pd.DataFrame, pipeline: str = "ref40") -> pd.DataFrame:
    """EV7: the reported peak ratio -- the dataset chain on the reference (``ref40``) for
    the opening angular velocity in open_close (the quantity of the dataset requirement),
    one row per participant (rows of ``bias``)."""
    if b.empty:
        return pd.DataFrame(columns=["pid", "task", "pipeline", "quantity", "n",
                                     "ratio_median", "ratio_ci_lo", "ratio_ci_hi"])
    k = ((b["task"] == "open_close") & (b["quantity"] == "peak_open_vel_dps")
         & (b["pipeline"] == pipeline))
    return b[k].reset_index(drop=True)


def peak_ratio_acc(b: pd.DataFrame) -> pd.DataFrame:
    """VD24: the acceleration counterpart of :func:`peak_ratio_open` -- the peak |θ̈| in
    open_close, per participant and pipeline (``ref40``: the dataset chain on the reference,
    the X7 lower bound; ``jtsmm``: the dataset as recorded), rows of ``bias``."""
    cols = ["pid", "task", "pipeline", "quantity", "n", "ratio_median", "ratio_ci_lo",
            "ratio_ci_hi"]
    if b.empty:
        return pd.DataFrame(columns=cols)
    k = (b["task"] == "open_close") & (b["quantity"] == "peak_abs_acc_dps2")
    return b.loc[k, cols].reset_index(drop=True)


def opening_angle(M: np.ndarray, rest_mask: np.ndarray) -> np.ndarray:
    """Opening angle (deg, y of the jaw-in-head rotation vector) relative to the first valid
    rest pose, without any filtering (NaN where the pose is missing)."""
    ok = rest_mask & se3.valid(M)
    v = se3.valid(M)
    if not v.any():
        return np.full(len(M), np.nan)
    ref = M[ok][0] if ok.any() else M[v][0]
    return np.degrees(se3.rotvec_rel(M, ref))[:, 1]


def central_rate(t_s: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Central difference ``(x[i+1] − x[i−1]) / (t[i+1] − t[i−1])`` (NaN at the ends and next
    to missing samples)."""
    out = np.full(len(x), np.nan)
    if len(x) >= 3:
        out[1:-1] = (x[2:] - x[:-2]) / (t_s[2:] - t_s[:-2])
    return out


def _welch_runs(x: np.ndarray, fs: float, nperseg: int
                ) -> tuple[np.ndarray, np.ndarray, int]:
    """Welch PSD (density, Hann, 50 % overlap, mean removed per run) averaged over the finite
    runs of ``x`` of at least ``nperseg`` samples, weighted by each run's segment count.
    Returns ``(f, P, n_segments)``; ``P`` is NaN when no run is long enough."""
    from scipy.signal import welch

    ok = np.isfinite(x)
    i = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    acc, nseg = np.zeros(len(f)), 0
    for a, b in zip(i[::2], i[1::2]):
        if b - a < nperseg:
            continue
        seg = x[a:b]
        k = 1 + (len(seg) - nperseg) // (nperseg // 2)
        _, P = welch(seg - seg.mean(), fs, nperseg=nperseg)
        acc += k * P
        nseg += k
    return f, (acc / nseg if nseg else np.full(len(f), np.nan)), nseg


def spectrum_unfiltered(pid: str, t_s: np.ndarray, theta_deg: np.ndarray,
                        intervals: list[tuple[str, float, float]],
                        rest: list[tuple[float, float]], *, nperseg: int = 512,
                        f_split_hz: float = 20.0) -> pd.DataFrame:
    """EV7 (primary movement bandwidth): the opening-rate spectrum of the **unfiltered**
    opening angle ``theta_deg`` (central difference on the native frames), per task, minus
    the noise floor -- the same spectrum on the rest phase(s) ``rest`` -- with the power
    below the floor clipped to 0. From the floor-subtracted power: f50/f95/f99 and the
    share above ``f_split_hz``. Welch: Hann, ``nperseg`` samples (512 = 2.56 s at 200 Hz),
    50 % overlap, over the finite runs (no gap bridging). The floor uses the same
    ``nperseg`` when a rest run is long enough, else the largest power of two that fits,
    interpolated onto the task frequencies (``noise_nperseg``). No rest phase: no
    subtraction (``noise_subtracted`` false)."""
    t_s = np.asarray(t_s, float)
    fs = 1.0 / float(np.median(np.diff(t_s)))
    rate = central_rate(t_s, np.asarray(theta_deg, float))
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    # noise floor on the rest phase(s)
    kr = np.zeros(len(t_s), bool)
    for a, b in rest:
        kr |= (t_s >= a) & (t_s <= b)
    xr = np.where(kr, rate, np.nan)
    noise, n_noise, nps_noise = np.full(len(f), np.nan), 0, nperseg
    fr, Pr, n_noise = _welch_runs(xr, fs, nperseg)
    if n_noise:
        noise = Pr
    else:
        ok = np.isfinite(xr)
        i = np.flatnonzero(np.diff(np.r_[0, ok.astype(int), 0]))
        longest = int(max((b - a for a, b in zip(i[::2], i[1::2])), default=0))
        if longest >= 32:
            nps_noise = 2 ** int(np.floor(np.log2(longest)))
            fr, Pr, n_noise = _welch_runs(xr, fs, nps_noise)
            noise = np.interp(f, fr, Pr)
    rest_s = float(np.sum(np.isfinite(xr)) / fs)
    rows = []
    for task, a, b in intervals:
        k = (t_s >= a) & (t_s <= b)
        _, P, nseg = _welch_runs(np.where(k, rate, np.nan), fs, nperseg)
        if not nseg:
            continue
        sub = bool(np.isfinite(noise).all())
        Pc = np.clip(P - noise, 0.0, None) if sub else P
        tot = float(Pc.sum())
        row = {"pid": pid, "task": task, "fs_hz": fs, "n_segments": int(nseg),
               "noise_subtracted": sub, "noise_rest_s": rest_s,
               "noise_nperseg": int(nps_noise) if sub else 0,
               "noise_power_frac": float(np.sum(np.minimum(noise, P)) / P.sum())
               if sub and P.sum() > 0 else np.nan,
               "f50_unfilt_hz": np.nan, "f95_unfilt_hz": np.nan, "f99_unfilt_hz": np.nan,
               "share_above_20hz_unfilt": np.nan}
        if tot > 0:
            c = np.cumsum(Pc) / tot
            row.update(f50_unfilt_hz=float(f[np.searchsorted(c, 0.50)]),
                       f95_unfilt_hz=float(f[min(np.searchsorted(c, 0.95), len(f) - 1)]),
                       f99_unfilt_hz=float(f[min(np.searchsorted(c, 0.99), len(f) - 1)]),
                       share_above_20hz_unfilt=float(Pc[f > f_split_hz].sum() / tot))
        rows.append(row)
    return pd.DataFrame(rows)


def run_ref_sampling(sy: Any, landmarks: pd.DataFrame, cycles: pd.DataFrame,
                     t_cycles_s: np.ndarray, seed: int = 0
                     ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    """The three pipelines on the participant's cycles (``cycles``: frame indices into
    ``t_cycles_s``, from ``ref_agreement``), their bias and the spectrum."""
    from biomech.validation import anatomy as AN
    from biomech.validation import selection as SEL
    from biomech.validation.evaluate import holds_from_table

    fs_, fq = (AN.anatomical_frames(holds_from_table(landmarks, s, v))
               for s, v in (("jtsmm", "-"), ("qualisys", "primary")))
    M_Q200 = AN.jaw_in_head(sy.qtm.pose("HP"), sy.qtm.pose("MP"), fq)
    M_Q40 = AN.jaw_in_head(sy.q_at_s("HP"), sy.q_at_s("MP"), fq)
    M_S = AN.jaw_in_head(sy.S["HP"], sy.S["MP"], fs_)
    rest = [h for h in sy.held if h.name in SEL.STATIC_TASKS]

    def rest_mask(t: np.ndarray) -> np.ndarray:
        m = np.zeros(len(t), bool)
        for h in rest:
            m |= (t >= h.t0) & (t <= h.t1)
        return m

    sig = {}
    for name, (t, M) in (("ref200", (sy.t_qm, M_Q200)), ("ref40", (sy.tS, M_Q40)),
                         ("jtsmm", (sy.tS, M_S))):
        lp, win = PIPELINES[name]
        dth, sp, ddth = _chain(np.asarray(t, float), M, lp, win,
                               rest_mask(np.asarray(t, float)), accel=True)
        sig[name] = (np.asarray(t, float), dth, sp, ddth)
    peaks = cycle_peaks(sy.pid, sig, cycles, np.asarray(t_cycles_s, float))
    mov = [(h.name, h.t0, h.t1) for h in sy.held if h.name in SEL.MOVEMENT_TASKS]
    b = bias(peaks, seed=seed)
    # VD24: the acceleration ratios with their own generator (the velocity rows keep theirs)
    b_acc = bias(peaks, seed=seed, quantities=ACC_QUANTITIES)
    b = pd.concat([b, b_acc], ignore_index=True) if not b_acc.empty else b
    # EV7: the unfiltered 200 Hz opening angle (no low-pass), same rest reference pose
    t_q = np.asarray(sy.t_qm, float)
    th_raw = opening_angle(M_Q200, rest_mask(t_q))
    tables = {"peaks": peaks, "bias": b, "peak_ratio_open": peak_ratio_open(b),
              "peak_ratio_acc": peak_ratio_acc(b),
              "spectrum": spectrum(sy.pid, *sig["ref200"][:2], mov),
              "spectrum_unfiltered": spectrum_unfiltered(
                  sy.pid, t_q, th_raw, mov, [(h.t0, h.t1) for h in rest])}
    return tables, {}
