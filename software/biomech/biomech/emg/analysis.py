# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One participant's sEMG outcomes for the ``emg`` stage (an03).

* activation map: %MVC / %RVC per task x muscle (median and p95 over the phase);
* rest activity: cleaned rest RMS (uV) and rest level in %MVC/%RVC;
* clench indices from ``mvc_closing`` (uV): masseter and temporalis asymmetry
  ``(R - L) / (R + L)`` and activity index ``(M - T) / (M + T)``;
* cycle-normalized profiles (101 points) aligned to the kinematic cycles of
  ``open_close`` and ``chewing``, with the opening angle on the same grid;
* co-activation per open-close half-cycle (Winter overlap of opener vs closer).

A channel-cycle with more than ``cycle_mask_max`` masked samples is NaN.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from biomech.emg.prepare import EmgPrep, PhaseEmg
from biomech.io.session import CLOSERS, OPENERS
from biomech.kinematics.cycles import time_normalize

PROFILE_TASKS = ("open_close", "chewing")
GRID_HZ = 100.0
# A cycle needs this much EMG to be time-normalized (formerly 20 samples, i.e. 10 ms at 2 kHz
# but 20 ms at 1 kHz: seconds-based so the rule does not depend on the sampling rate).
MIN_CYCLE_EMG_S = 0.1


def to_grid(pe: PhaseEmg, values: np.ndarray, hz: float = GRID_HZ) -> tuple[np.ndarray, np.ndarray]:
    """Sample a smooth (envelope) signal onto a regular grid inside the phase."""
    t = pe.t_us
    grid = np.arange(t[0], t[-1], int(1e6 / hz), dtype=np.int64)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    return grid, values[idx]


def cci(a: np.ndarray, b: np.ndarray) -> float:
    """Winter co-contraction index (%): 2 * sum(min) / sum(a + b)."""
    ok = np.isfinite(a) & np.isfinite(b)
    den = np.sum(a[ok] + b[ok])
    return float(200.0 * np.sum(np.minimum(a[ok], b[ok])) / den) if den > 0 else np.nan


SPECTRAL_WINDOW_TASKS = ("open_close", "chewing", "mvc_closing")
COARSE_EDGES = np.arange(20.0, 461.0, 10.0)


def coarse_log_shape(f: np.ndarray, per: np.ndarray) -> np.ndarray:
    """(C, S, B) log10 share of in-band power per 10-Hz bin (spectral shape per window)."""
    bins = []
    for lo, hi in zip(COARSE_EDGES[:-1], COARSE_EDGES[1:]):
        sel = (f >= lo) & (f < hi)
        bins.append(np.nanmean(per[..., sel], axis=-1))
    B = np.stack(bins, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.log10(B / np.nansum(B, axis=-1, keepdims=True))



MASK_WIN_S = 0.25              # mask validation (W1.3): window length


def mask_validation(prep: EmgPrep, pid: str, phases: list[str]) -> pd.DataFrame:
    """Do samples masked by the low-frequency rule carry contaminated EMG? (A1 check)

    Per channel and task, 0.25-s windows of the cleaned (20-450 Hz) signal: "masked"
    windows have > 50 % of their samples hit by the low-frequency rule, "clean" windows no
    masked sample at all. Per window: the 20-40 Hz share of the 20-450 Hz power and the
    mean frequency (Hann, rfft). Pre-stated decision rule: the rule is *conservative* for a
    channel-task when the masked windows' medians of both lie inside the clean windows'
    IQR (no detectable contamination left after the 20 Hz high-pass), otherwise
    *contaminated*.
    """
    rows = []
    for ph in phases:
        if ph == "rest":
            continue
        pe = prep.phase(ph)
        lf = pe.masks.hits.get("lowfreq")
        if lf is None or not lf.any():
            continue
        anyhit = pe.masks.any_hit()          # "clean" = no rule at all (D14: lowfreq is flag-only)
        n = max(int(MASK_WIN_S * pe.fs), 16)
        k = pe.cleaned.shape[0] // n
        if k < 4:
            continue
        f = np.fft.rfftfreq(n, 1.0 / pe.fs)
        band, low = (f >= 20) & (f <= 450), (f >= 20) & (f < 40)
        win = np.hanning(n)
        for i, m in enumerate(pe.names):
            x = pe.cleaned[: k * n, i].reshape(k, n)
            hit = lf[: k * n, i].reshape(k, n).mean(axis=1)
            anym = anyhit[: k * n, i].reshape(k, n).any(axis=1)
            masked, clean = hit > 0.5, ~anym
            if masked.sum() < 2 or clean.sum() < 4:
                continue
            P = np.abs(np.fft.rfft((x - x.mean(axis=1, keepdims=True)) * win, axis=1)) ** 2
            tot = P[:, band].sum(axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                share = P[:, low].sum(axis=1) / tot
                mnf = (P[:, band] * f[band]).sum(axis=1) / tot
            s1, s3 = np.nanpercentile(share[clean], [25, 75])
            m1, m3 = np.nanpercentile(mnf[clean], [25, 75])
            sm, mm = float(np.nanmedian(share[masked])), float(np.nanmedian(mnf[masked]))
            rows.append({"pid": pid, "task": ph, "muscle": m, "n_masked": int(masked.sum()),
                         "n_clean": int(clean.sum()), "share_20_40_masked": sm,
                         "share_20_40_clean": float(np.nanmedian(share[clean])),
                         "share_q1": float(s1), "share_q3": float(s3),
                         "mnf_masked_hz": mm, "mnf_clean_hz": float(np.nanmedian(mnf[clean])),
                         "mnf_q1": float(m1), "mnf_q3": float(m3),
                         "conservative": bool(s1 <= sm <= s3 and m1 <= mm <= m3)})
    return pd.DataFrame(rows)


def alt_mask_rules(rules: tuple[str, ...]) -> tuple[str, ...]:
    """The D14 sensitivity counterpart: the low-frequency rule switched in or out."""
    return (tuple(r for r in rules if r != "lowfreq") if "lowfreq" in rules
            else tuple(rules) + ("lowfreq",))


def mask_sensitivity(prep: EmgPrep, pid: str, phases: list[str], kin_cycles: pd.DataFrame,
                     cycle_mask_max: float) -> pd.DataFrame:
    """Masked share, cycles kept and median activation per channel and task under the
    primary analysis mask rules and their D14 counterpart (lowfreq in/out).

    Each variant is normalized with its own reference (the reference itself depends on
    the mask, D11), so ``median_pct_*`` is what each variant would report.
    """
    rules = tuple(prep.p.mask.analysis_rules)
    alt = alt_mask_rules(rules)
    other = prep.with_mask_rules(alt)
    rows = []
    for ph in phases:
        if ph == "rest":
            continue
        a, b = prep.phase(ph), other.phase(ph)
        cyc = (kin_cycles[(kin_cycles.task == ph) & kin_cycles.ok]
               if not kin_cycles.empty and ph in PROFILE_TASKS else kin_cycles.iloc[:0])
        sels = [(a.t_us >= c.t_start_us) & (a.t_us <= c.t_end_us) for c in cyc.itertuples()]
        sels = [s for s in sels if s.sum() >= max(int(round(MIN_CYCLE_EMG_S * a.fs)), 2)]
        for i, n in enumerate(prep.names):
            row = {"pid": pid, "task": ph, "muscle": n, "rules": ",".join(rules),
                   "alt_rules": ",".join(alt),
                   "mask_frac": float(a.mask[:, i].mean()),
                   "mask_frac_alt": float(b.mask[:, i].mean()),
                   "median_pct": float(np.nanmedian(a.norm[:, i])),
                   "median_pct_alt": float(np.nanmedian(b.norm[:, i])),
                   "n_cycles": len(sels)}
            row["kept"] = int(sum(a.mask[s, i].mean() <= cycle_mask_max for s in sels))
            row["kept_alt"] = int(sum(b.mask[s, i].mean() <= cycle_mask_max for s in sels))
            rows.append(row)
    return pd.DataFrame(rows)


def _spectra(prep: EmgPrep, pid: str, phases: list[str], shape_ok
             ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Welch spectra of the cleaned EMG per task and MVC phase (X1, exploratory)."""
    from biomech.spectral import band_shares, mean_median_freq, welch_masked

    rows, arrays = [], {}
    todo = [(ph, "motion") for ph in phases]
    if prep.mvc is not None:
        todo += [(ph.name, "mvc") for ph in prep.mvc.phases if ph.kind == "mvc"]
    for ph, take in todo:
        pe = prep.phase(ph, take=take)
        f, psd, per = welch_masked(pe.cleaned, pe.fs, pe.mask, seg_s=0.5)
        keep = (f >= 10) & (f <= 500)
        arrays["spectral__freq"] = f[keep].astype(np.float32)
        arrays[f"spectral__{ph}__psd"] = psd[:, keep].astype(np.float32)
        mnf, mdf = mean_median_freq(f, psd)
        mnf_seg, _ = mean_median_freq(f, per)                  # (C, S) per-window MNF
        if ph in SPECTRAL_WINDOW_TASKS:
            # Per-window spectral shape on 10-Hz bins (20-460 Hz), log of the power share:
            # repetitions for the frequency-domain SPM across participants.
            arrays[f"spectral__{ph}__win"] = coarse_log_shape(f, per).astype(np.float32)
        shares = band_shares(f, psd)
        for i, n in enumerate(pe.names):
            v = mnf_seg[i][np.isfinite(mnf_seg[i])]
            rows.append({"pid": pid, "task": ph, "muscle": n, "take": take,
                         "mnf_hz": float(mnf[i]), "mdf_hz": float(mdf[i]),
                         "mnf_sd_within_hz": float(v.std(ddof=1)) if v.size > 1 else np.nan,
                         "n_windows": int(v.size), "usable_shape": shape_ok(n),
                         **{f"share_{int(lo)}_{int(hi)}": float(shares[i, j])
                            for j, (lo, hi) in enumerate(((20, 50), (50, 100), (100, 200),
                                                          (200, 450)))}})
    return pd.DataFrame(rows), arrays


CHAIN_WINDOW_S = 2.0          # stored example of the processing chain per phase
CHAIN_LEAD_S = 0.4            # ... starting this long before the first movement onset
CHAIN_PHASES = ("rest", "open_close", "chewing")   # stored chain examples (4 kHz, ~0.8 MB each)
MAINS_FMAX_HZ = 500.0


def chain_window(pe: PhaseEmg, kin_cycles: pd.DataFrame, phase: str) -> tuple[float, float]:
    """Default example window (s from the phase start): from ``CHAIN_LEAD_S`` before the
    first movement onset of the phase (1 s into the phase without cycles), ``CHAIN_WINDOW_S``
    long. Shared by the stored example and the ``data_emg_chain`` view."""
    first = None
    if not kin_cycles.empty and "task" in kin_cycles:
        cyc = kin_cycles[kin_cycles.task == phase]
        if not cyc.empty:
            t_on = cyc.t_on_us if "t_on_us" in cyc else cyc.t_start_us
            t_on = np.where(np.isfinite(t_on.to_numpy(float)), t_on, cyc.t_start_us)
            first = float(np.min(t_on))
    w0 = max(0.0, (first - pe.t_us[0]) * 1e-6 - CHAIN_LEAD_S) if first is not None else 1.0
    return w0, w0 + CHAIN_WINDOW_S


def chain_products(prep: EmgPrep, phases: list[str], kin_cycles: pd.DataFrame
                   ) -> dict[str, np.ndarray]:
    """What the processing-chain and spectra views show, stored so that they render from
    the result store alone (G2): per phase the Welch PSD (1-s segments, <= 500 Hz) of the raw
    signal and after mains removal with the grid frequency used, and a ``CHAIN_WINDOW_S``
    example of every channel at each step (raw, mains removed, band-passed, envelope in %
    of max recorded, analysis mask) for the ``CHAIN_PHASES``."""
    from scipy.signal import welch

    from jtsmm.emg.mains import remove_mains

    out: dict[str, np.ndarray] = {}
    for ph in phases:
        pe = prep.phase(ph)
        if pe.raw.shape[0] < 4:
            continue
        raw = pe.raw - np.median(pe.raw, axis=0)
        f0 = prep.phase_line_hz(prep.motion, ph)
        dem, f0 = remove_mains(raw, pe.fs, t_us=pe.t_us, line_hz=f0,
                               robust=prep.p.robust_mains)
        nper = int(min(pe.fs, raw.shape[0]))
        f, p_raw = welch(raw, pe.fs, nperseg=nper, axis=0)
        _, p_dem = welch(dem, pe.fs, nperseg=nper, axis=0)
        keep = f <= min(MAINS_FMAX_HZ, pe.fs / 2)
        out["mains__freq"] = f[keep].astype(np.float32)
        out[f"mains__{ph}__raw_psd"] = p_raw[keep].T.astype(np.float32)
        out[f"mains__{ph}__demained_psd"] = p_dem[keep].T.astype(np.float32)
        out[f"mains__{ph}__line_hz"] = np.array([np.nan if f0 is None else f0], np.float64)
        if ph not in CHAIN_PHASES:
            continue
        w0, w1 = chain_window(pe, kin_cycles, ph)
        t = (pe.t_us - pe.t_us[0]) * 1e-6
        sel = (t >= w0) & (t <= w1)
        if not sel.any():
            continue
        out[f"chain__{ph}__t_us"] = pe.t_us[sel]
        out[f"chain__{ph}__t0_us"] = np.array([pe.t_us[0]], np.int64)
        out[f"chain__{ph}__raw"] = raw[sel].astype(np.float32)
        out[f"chain__{ph}__demained"] = dem[sel].astype(np.float32)
        out[f"chain__{ph}__cleaned"] = pe.cleaned[sel].astype(np.float32)
        out[f"chain__{ph}__norm"] = pe.norm[sel].astype(np.float32)
        out[f"chain__{ph}__masked"] = pe.mask[sel].astype(np.uint8)
    return out


def analyse(prep: EmgPrep, *args: Any, **kwargs: Any) -> tuple[dict[str, pd.DataFrame],
                                                              dict[str, np.ndarray]]:
    """See module docstring. All-NaN (fully masked) channel-cycles are expected."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
        return _analyse(prep, *args, **kwargs)


def _analyse(prep: EmgPrep, pid: str, kin_cycles: pd.DataFrame, kin_arrays: dict[str, Any],
            usability: pd.DataFrame, p: dict[str, Any]) -> tuple[dict[str, pd.DataFrame],
                                                                  dict[str, np.ndarray]]:
    names = prep.names
    refs = prep.refs
    tables: dict[str, pd.DataFrame] = {}
    arrays: dict[str, np.ndarray] = {}
    use = usability.set_index("muscle") if not usability.empty else None

    def shape_ok(m: str) -> bool:
        return use is None or m not in use.index or bool(use.loc[m, "usable_shape"])

    ref = prep.reference
    tables["refs"] = pd.DataFrame([{
        "pid": pid, "muscle": n, "ref_phase": refs["phase"][i], "p95_uv": refs["p95_uv"][i],
        "max1s_uv": refs["max1s_uv"][i], "mask_frac": refs["mask_frac"][i],
        # D11: the normalization reference is the largest activity recorded for the channel.
        "ref_uv": float(prep.ref_uv()[i]), "ref_source": str(ref["source"][i]),
        "mvc_max_uv": float(refs["max_uv"][i]),
        "task_max_uv": float(prep.task_peaks["max_uv"][i]),
        "task_max_phase": str(ref["task_phase"][i]),
        "mvc_over_task": float(ref["mvc_over_task"][i]),
        "usable_amplitude": bool(use.loc[n, "usable_amplitude"]) if use is not None else True,
        "usable_shape": bool(use.loc[n, "usable_shape"]) if use is not None else True,
    } for i, n in enumerate(names)])

    rest = prep.rest
    act_rows, cyc_rows, spec_rows = [], [], []
    rest_pe = prep.phase("rest")
    for i, n in enumerate(names):
        act_rows.append({"pid": pid, "task": "rest", "muscle": n,
                         "median_pct": float(np.nanmedian(rest_pe.norm[:, i])),
                         "p95_pct": float(np.nanpercentile(rest_pe.norm[:, i], 95)),
                         "rest_rms_uv": float(rest["rms_uv"][i]),
                         "mask_frac": float(rest_pe.mask[:, i].mean())})

    phases = [ph.name for ph in prep.motion.phases if ph.kind == "experiment"]
    for ph in phases:
        pe = rest_pe if ph == "rest" else prep.phase(ph)
        grid, norm = to_grid(pe, pe.norm)
        _, env = to_grid(pe, pe.env)
        _, msk = to_grid(pe, pe.mask.astype(np.float32))
        arrays[f"{ph}__t_us"] = grid
        arrays[f"{ph}__norm_pct"] = norm.astype(np.float32)
        arrays[f"{ph}__env_uv"] = env.astype(np.float32)
        arrays[f"{ph}__masked"] = msk.astype(np.uint8)
        if ph != "rest":
            for i, n in enumerate(names):
                act_rows.append({"pid": pid, "task": ph, "muscle": n,
                                 "median_pct": float(np.nanmedian(pe.norm[:, i])),
                                 "p95_pct": float(np.nanpercentile(pe.norm[:, i], 95)),
                                 "rest_rms_uv": np.nan,
                                 "mask_frac": float(pe.mask[:, i].mean())})
        # Specification curve: the reference (analysis_plan §5.3, D11): max recorded
        # (primary), the frozen MVC/RVC p95, and the largest 1-s mean.
        if ph in ("open_close", "chewing"):
            variants = {v: 100.0 * pe.env / prep.ref_uv(v)[None, :]
                        for v in ("max_recorded", "p95", "max1s")}
            for i, n in enumerate(names):
                for var, arr in variants.items():
                    spec_rows.append({"pid": pid, "outcome": f"median_pct|{ph}|{n}",
                                      "variant": f"norm={var}",
                                      "value": float(np.nanmedian(arr[:, i]))})

        if ph not in PROFILE_TASKS or kin_cycles.empty:
            continue
        cyc = kin_cycles[(kin_cycles.task == ph) & kin_cycles.ok]
        th_t = kin_arrays.get(f"{ph}__t_us")
        th = kin_arrays.get(f"{ph}__theta_deg")
        prof, prof_th, ids = [], [], []
        for _, c in cyc.iterrows():
            sel = (pe.t_us >= c.t_start_us) & (pe.t_us <= c.t_end_us)
            if sel.sum() < max(int(round(MIN_CYCLE_EMG_S * pe.fs)), 2):
                continue
            seg = pe.norm[sel].copy()
            mfrac = pe.mask[sel].mean(axis=0)
            seg[:, mfrac > p["cycle_mask_max"]] = np.nan
            prof.append(time_normalize(seg, 101))
            ids.append(int(c.cycle))
            if th is not None:
                k = (th_t >= c.t_start_us) & (th_t <= c.t_end_us)
                prof_th.append(time_normalize(th[k].astype(float), 101))
            # per-cycle means and co-activation (open/close halves)
            pk = (pe.t_us[sel] <= c.t_peak_us)
            row = {"pid": pid, "task": ph, "cycle": int(c.cycle)}
            for i, n in enumerate(names):
                row[f"mean_pct_{n}"] = float(np.nanmean(seg[:, i]))
                row[f"masked_{n}"] = float(mfrac[i])
            # Co-activation from shape-usable channels only (an01 §3); each muscle is
            # relative to its own reference contraction (openers: mvc_opening,
            # closers: mvc_closing), so the CCI compares normalized activity.
            op_i = [names.index(m) for m in OPENERS if m in names and shape_ok(m)]
            cl_i = [names.index(m) for m in CLOSERS if m in names and shape_ok(m)]
            if op_i and cl_i:
                op = np.nanmean(seg[:, op_i], axis=1)
                cl = np.nanmean(seg[:, cl_i], axis=1)
                row["cci_opening"] = cci(op[pk], cl[pk])
                row["cci_closing"] = cci(op[~pk], cl[~pk])
            else:
                row["cci_opening"] = row["cci_closing"] = np.nan
            cyc_rows.append(row)
        if prof:
            arrays["profile__names"] = np.array(names)          # channel order of the profiles
            arrays[f"profile__{ph}__pct"] = np.stack(prof).astype(np.float32)
            arrays[f"profile__{ph}__cycle"] = np.array(ids)
            if prof_th:
                arrays[f"profile__{ph}__theta_deg"] = np.stack(prof_th).astype(np.float32)

    tables["activation"] = pd.DataFrame(act_rows)
    tables["cycles"] = pd.DataFrame(cyc_rows)
    tables["spec"] = pd.DataFrame(spec_rows)
    tables["spectral"], spec_arrays = _spectra(prep, pid, phases, shape_ok)
    tables["mask_validation"] = mask_validation(prep, pid, phases)
    tables["mask_sensitivity"] = mask_sensitivity(prep, pid, phases, kin_cycles,
                                                  p["cycle_mask_max"])
    arrays.update(spec_arrays)
    arrays.update(chain_products(prep, phases, kin_cycles))

    p95 = dict(zip(names, refs["p95_uv"]))
    ind: dict[str, Any] = {"pid": pid}
    for m in ("masseter", "temporalis"):
        r, lft = p95.get(f"{m}_r", np.nan), p95.get(f"{m}_l", np.nan)
        ind[f"asym_{m}_pct"] = float(100 * (r - lft) / (r + lft))
    mas = np.nanmean([p95.get("masseter_r", np.nan), p95.get("masseter_l", np.nan)])
    tem = np.nanmean([p95.get("temporalis_r", np.nan), p95.get("temporalis_l", np.nan)])
    ind["activity_index_pct"] = float(100 * (mas - tem) / (mas + tem))
    closers = ("masseter_r", "masseter_l", "temporalis_r", "temporalis_l")
    # The indices compare the clench contractions themselves, so they need every closer's
    # clench to be its maximum (D11: reference source "mvc"); otherwise flagged unusable.
    src = dict(zip(names, ref["source"]))
    ind["usable"] = bool((use is None or all(use.loc[m, "usable_amplitude"] for m in closers
                                             if m in use.index))
                         and all(src.get(m) == "mvc" for m in closers if m in src))
    tables["indices"] = pd.DataFrame([ind])
    return tables, arrays
