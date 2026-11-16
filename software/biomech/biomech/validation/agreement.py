# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Cycle-level agreement JTS-MM vs Qualisys on the anatomical-frame outcome (validation
plan §5, V5, V8; master plan 4.9, D4).

* **Cycles** are detected on the *reference* (Qualisys) signal of each movement task --
  opening −z (open-close, chewing), protrusion x, laterotrusion ±y -- with the dataset's
  cycle rules (``kinematics.cycles.detect``), and the same windows are cut from both
  systems. Only cycles complete in both systems and outside the reference gate are used.
* **Features per cycle** (both systems): task amplitude, incisor excursion (3-D, from the
  cycle start), peak speed, rotation excursion, condyle excursion.
* **Feature agreement** per participant, task and feature: Bland–Altman (bias, LoA; the
  bias CI a t-interval with the effective n of the autocorrelated cycle differences, EV6,
  the cycle and moving-block bootstraps as sensitivities; LoA CIs cycle-bootstrap), Lin's
  CCC (1/n moments as in Lin 1989, EV9), ICC(A,1) (absolute agreement), Pearson r, and
  equivalence of the bias against the metrology (±0.5 mm / ±0.5°) and application
  (±2.0 mm / ±1.2°) bounds: equivalent when the bias CI lies inside the bound; and the
  bound-free equivalence margin, the smallest symmetric bound the CI lies inside (VD13), so
  a reader can apply any tolerance.
* **Waveforms** (time-normalised cycles, 101 samples): paired SPM t-test JTS-MM vs
  Qualisys with cluster ranges (significance bars), the mean difference with a
  cycle-bootstrap 95 % band, the share of the cycle inside each bound, the equivalence
  margin of the whole band (VD13) and the between-system CMC per cycle.

n = 2 participants: participant is a fixed factor; nothing is pooled across participants.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from biomech.kinematics.cycles import detect

TOL = {"metrology": {"mm": 0.5, "deg": 0.5}, "application": {"mm": 2.0, "deg": 1.2}}
TASK_SIGNAL = {"open_close": [("", 2, -1.0)], "chewing": [("", 2, -1.0)],
               "protrusion_retrusion": [("", 0, 1.0)],
               "left_right": [("left", 1, 1.0), ("right", 1, -1.0)]}
CHANNELS = {"incisor_x": ("incisor", 0, "mm"), "incisor_y": ("incisor", 1, "mm"),
            "incisor_z": ("incisor", 2, "mm"), "rot_x": ("rot", 0, "deg"),
            "rot_y": ("rot", 1, "deg"), "rot_z": ("rot", 2, "deg")}


def ccc(a: np.ndarray, b: np.ndarray) -> float:
    """Lin's (1989) concordance correlation coefficient with 1/n (population) moments,
    ρ_c = 2 s_ab / (s_a² + s_b² + (ā − b̄)²) (EV9; pairs with a NaN dropped, NaN below 3
    pairs)."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    sab = np.mean((a - a.mean()) * (b - b.mean()))
    return float(2 * sab / (a.var() + b.var() + (a.mean() - b.mean()) ** 2))


def icc_a1(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(A,1): two-way, absolute agreement, single measurement (McGraw & Wong)."""
    ok = np.isfinite(a) & np.isfinite(b)
    Y = np.c_[a[ok], b[ok]]
    n, k = Y.shape
    if n < 3:
        return np.nan
    gm = Y.mean()
    msr = k * np.sum((Y.mean(axis=1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((Y.mean(axis=0) - gm) ** 2) / (k - 1)
    sse = np.sum((Y - Y.mean(axis=1, keepdims=True) - Y.mean(axis=0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    den = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / den) if den > 0 else np.nan


def lag1_autocorr(d: np.ndarray) -> float:
    """Lag-1 autocorrelation of a series in time order, ``Σ (dᵢ − d̄)(dᵢ₊₁ − d̄) / Σ (dᵢ − d̄)²``
    (NaN below 3 values)."""
    d = np.asarray(d, float)
    if d.size < 3:
        return np.nan
    x = d - d.mean()
    den = float(np.sum(x * x))
    return float(np.sum(x[1:] * x[:-1]) / den) if den > 0 else 0.0


def n_effective(d: np.ndarray) -> tuple[float, float]:
    """(ρ₁, n_eff) of cycle differences in time order (EV6): ρ₁ is the lag-1 autocorrelation
    clipped to [0, 0.99] (≥ 0: a negative estimate never narrows the interval; < 1 keeps
    n_eff positive) and ``n_eff = n (1 − ρ₁)/(1 + ρ₁)``, the sample size of independent
    cycles with the same variance of the mean under AR(1) errors."""
    n = int(np.asarray(d).size)
    r = lag1_autocorr(d)
    r = float(np.clip(r, 0.0, 0.99)) if np.isfinite(r) else 0.0
    return r, n * (1.0 - r) / (1.0 + r)


def bias_t_interval(d: np.ndarray, level: float = 0.95) -> dict[str, float]:
    """Primary bias CI of EV6: ``d̄ ± t_{(1+level)/2, df} · s / √n_eff`` with ``s`` the SD of
    the differences, ``n_eff`` from :func:`n_effective` and ``df = max(n_eff − 1, 1)``."""
    from scipy.stats import t as t_dist

    d = np.asarray(d, float)
    rho, ne = n_effective(d)
    df = max(ne - 1.0, 1.0)
    half = float(t_dist.ppf(0.5 + level / 2, df) * d.std(ddof=1) / np.sqrt(ne))
    m = float(d.mean())
    return {"bias_ci_lo": m - half, "bias_ci_hi": m + half, "rho1": rho, "n_eff": float(ne),
            "t_df": float(df)}


def pointwise_t_band(D: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """Waveform mean-difference band (EV14): at every node the :func:`bias_t_interval` of the
    cycle differences ``D`` (cycles in time order x nodes), i.e. the same effective-n
    t-interval as the cycle-feature bias; a node without spread gets the mean itself."""
    D = np.asarray(D, float)
    lo, hi = np.empty(D.shape[1]), np.empty(D.shape[1])
    for j in range(D.shape[1]):
        d = D[:, j]
        if np.nanstd(d) > 0:
            b = bias_t_interval(d, level)
            lo[j], hi[j] = b["bias_ci_lo"], b["bias_ci_hi"]
        else:
            lo[j] = hi[j] = float(np.nanmean(d))
    return lo, hi


def moving_block_means(d: np.ndarray, rng: np.random.Generator, n_boot: int,
                       block: int = 2) -> np.ndarray:
    """Bootstrap means of ``d`` (time order) by the moving-block bootstrap: overlapping blocks
    of ``block`` successive cycles drawn with replacement, concatenated, cut to ``n``."""
    n = len(d)
    block = max(1, min(block, n))
    starts = np.arange(n - block + 1)
    k = int(np.ceil(n / block))
    pick = rng.choice(starts, (n_boot, k))
    idx = (pick[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return d[idx].mean(axis=1)


def ba_boot(est: np.ndarray, ref: np.ndarray, *, n_boot: int = 2000, seed: int = 0,
            order: np.ndarray | None = None, block: int = 2) -> dict[str, float]:
    """Bland–Altman within one participant over its cycles.

    Bias CI (EV6, primary; ``bias_ci_lo/hi``): t-interval with the effective sample size of
    the cycle differences in time order (``order``: e.g. the cycle start frames; given
    order otherwise), see :func:`bias_t_interval`. Sensitivities: the cycle percentile
    bootstrap that treated cycles as independent (``bias_ci_lo/hi_boot``, the CI before
    EV6) and the moving-block bootstrap with blocks of ``block`` successive cycles
    (``bias_ci_lo/hi_mbb``). LoA CIs: cycle percentile bootstrap (unchanged)."""
    d = np.asarray(est, float) - np.asarray(ref, float)
    if order is not None:
        d = d[np.argsort(np.asarray(order, float), kind="stable")]
    d = d[np.isfinite(d)]
    out = {"n": int(len(d)), "bias": np.nan, "loa_lo": np.nan, "loa_hi": np.nan}
    if len(d) < 3:
        return out
    sd = d.std(ddof=1)
    out.update(bias=float(d.mean()), loa_lo=float(d.mean() - 1.96 * sd),
               loa_hi=float(d.mean() + 1.96 * sd), sd=float(sd))
    rng = np.random.default_rng(seed)
    B = d[rng.integers(0, len(d), (n_boot, len(d)))]
    m, s = B.mean(axis=1), B.std(axis=1, ddof=1)
    for key, v in (("bias", m), ("loa_lo", m - 1.96 * s), ("loa_hi", m + 1.96 * s)):
        out[f"{key}_ci_lo"], out[f"{key}_ci_hi"] = (float(x) for x in np.percentile(v, [2.5, 97.5]))
    out["bias_ci_lo_boot"] = out.pop("bias_ci_lo")
    out["bias_ci_hi_boot"] = out.pop("bias_ci_hi")
    mb = moving_block_means(d, rng, n_boot, block)
    out["bias_ci_lo_mbb"], out["bias_ci_hi_mbb"] = (float(x) for x in np.percentile(mb, [2.5, 97.5]))
    out.update(bias_t_interval(d))
    return out


def margin(lo: float, hi: float) -> float:
    """Equivalence margin (VD13): the smallest symmetric bound the interval lies inside."""
    return float(max(abs(lo), abs(hi)))


def normalise(x: np.ndarray, a: int, b: int, n: int = 101) -> np.ndarray:
    seg = x[a:b + 1]
    t = np.linspace(0.0, 1.0, len(seg))
    g = np.linspace(0.0, 1.0, n)
    if seg.ndim == 1:
        return np.interp(g, t, seg)
    return np.column_stack([np.interp(g, t, seg[:, j]) for j in range(seg.shape[1])])


def find_cycles(arr: dict[str, np.ndarray], task: str, k: np.ndarray, fs: float,
                params: dict[str, Any] | None = None) -> list[tuple[str, int, int, int]]:
    """(side, start, peak, end) frame indices of the ok cycles on the reference signal, kept
    only when both systems are finite over the cycle and the reference gate is clear."""
    p = {"min_period_s": 0.4, "prominence_frac": 0.3, "max_nan_frac": 0.1, **(params or {})}
    idx = np.flatnonzero(k)
    if len(idx) < 20:
        return []
    out = []
    for side, ax, sign in TASK_SIGNAL.get(task, []):
        x = sign * arr["incisor_q_mm"][idx, ax]
        kw = dict(p)
        if task in ("open_close", "chewing"):
            kw["return_frac"] = p.get("return_frac", 0.2)
        for c in detect(x, fs, **kw):
            if not c.ok:
                continue
            a, b = idx[c.start], idx[c.end]
            sl = slice(a, b + 1)
            if not (np.isfinite(arr["incisor_s_mm"][sl]).all()
                    and np.isfinite(arr["incisor_q_mm"][sl]).all()):
                continue
            if "gate" in arr and np.asarray(arr["gate"][sl], bool).any():
                continue
            out.append((side, a, idx[c.peak], b))
    return out


def cycle_features(arr: dict[str, np.ndarray], a: int, pk: int, b: int, ax: int, sign: float,
                   t: np.ndarray) -> dict[str, float]:
    out = {}
    for sys_, s in (("s", "s"), ("q", "q")):
        inc = arr[f"incisor_{s}_mm"][a:b + 1]
        rot = arr[f"rot_{s}_deg"][a:b + 1]
        sig = sign * inc[:, ax]
        out[f"amp_mm_{sys_}"] = float(sig.max() - sig[0])
        out[f"excursion_mm_{sys_}"] = float(np.linalg.norm(inc - inc[0], axis=1).max())
        v = np.linalg.norm(np.diff(inc, axis=0), axis=1) / np.diff(t[a:b + 1])
        out[f"peak_speed_mm_s_{sys_}"] = float(np.nanmax(v)) if len(v) else np.nan
        out[f"rot_excursion_deg_{sys_}"] = float(np.linalg.norm(rot - rot[0], axis=1).max())
        tm = [np.linalg.norm(arr[f"{n}_{s}_mm"][a:b + 1] - arr[f"{n}_{s}_mm"][a], axis=1).max()
              for n in ("tmj_left", "tmj_right") if f"{n}_{s}_mm" in arr]
        out[f"tmj_excursion_mm_{sys_}"] = float(max(tm)) if tm else np.nan
    return out


FEATURES = {"amp_mm": "mm", "excursion_mm": "mm", "peak_speed_mm_s": "mm/s",
            "rot_excursion_deg": "deg", "tmj_excursion_mm": "mm"}


def run_ref_agreement(arr: dict[str, np.ndarray], held: list, pid: str, *, seed: int = 0,
                      params: dict[str, Any] | None = None
                      ) -> tuple[dict[str, pd.DataFrame], dict[str, np.ndarray]]:
    from biomech.stats.waveforms import paired_t

    t = np.asarray(arr["t_master_s"], float)
    fs = 1.0 / float(np.median(np.diff(t)))
    cyc_rows, feat_rows, wave_rows = [], [], []
    out_arrays: dict[str, np.ndarray] = {}
    for h in held:
        if h.name not in TASK_SIGNAL:
            continue
        k = (t >= h.t0) & (t <= h.t1)
        cyc = find_cycles(arr, h.name, k, fs, params)
        sig_of = {side: (ax, sign) for side, ax, sign in TASK_SIGNAL[h.name]}
        rows = []
        for i, (side, a, pk, b) in enumerate(cyc):
            ax, sign = sig_of[side]
            rows.append({"pid": pid, "task": h.name, "side": side, "cycle": i, "start": a,
                         "peak": pk, "end": b, "duration_s": float(t[b] - t[a]),
                         **cycle_features(arr, a, pk, b, ax, sign, t)})
        cyc_rows += rows
        if not rows:
            continue
        cdf = pd.DataFrame(rows)
        for f, unit in FEATURES.items():
            es, rf = cdf[f"{f}_s"].to_numpy(float), cdf[f"{f}_q"].to_numpy(float)
            if np.isfinite(es).sum() < 3 or np.isfinite(rf).sum() < 3:
                continue
            ba = ba_boot(es, rf, seed=seed, order=cdf["start"].to_numpy(float))
            tol_unit = "deg" if unit == "deg" else "mm"
            row = {"pid": pid, "task": h.name, "feature": f, "unit": unit, **ba,
                   "ccc": ccc(es, rf), "icc_a1": icc_a1(es, rf),
                   "pearson_r": float(pd.Series(es).corr(pd.Series(rf)))
                   if np.isfinite(es).sum() > 2 else np.nan,
                   "mean_q": float(np.nanmean(rf))}
            if unit in ("mm", "deg") and np.isfinite(ba.get("bias_ci_lo", np.nan)):
                for bound, tol in TOL.items():
                    row[f"equivalent_{bound}"] = bool(-tol[tol_unit] < ba["bias_ci_lo"]
                                                      and ba["bias_ci_hi"] < tol[tol_unit])
                row["equiv_margin"] = margin(ba["bias_ci_lo"], ba["bias_ci_hi"])
                # EV6 sensitivities: the margin of the independent-cycle percentile
                # bootstrap (the pre-EV6 value) and of the moving-block bootstrap
                row["equiv_margin_boot"] = margin(ba["bias_ci_lo_boot"], ba["bias_ci_hi_boot"])
                row["equiv_margin_mbb"] = margin(ba["bias_ci_lo_mbb"], ba["bias_ci_hi_mbb"])
            feat_rows.append(row)
        # waveforms: absolute (incl. the frame-definition offset) and displacement from the
        # cycle start (offset removed; consistent with the cycle features)
        for (ch, (src, j, unit)), mode in [(c, m) for c in CHANNELS.items()
                                           for m in ("abs", "disp")]:
            A = np.array([normalise(arr[f"{src}_s_{'mm' if src == 'incisor' else 'deg'}"][:, j],
                                    r["start"], r["end"]) for r in rows])
            B = np.array([normalise(arr[f"{src}_q_{'mm' if src == 'incisor' else 'deg'}"][:, j],
                                    r["start"], r["end"]) for r in rows])
            if len(A) < 3:
                continue
            if mode == "disp":
                A, B = A - A[:, :1], B - B[:, :1]
            D = A - B
            # EV14: the band is the effective-n t-interval per node (cycles in time order,
            # as the bias CI of the cycle features, EV6); the independent-cycle bootstrap
            # band of before stays as a sensitivity (``equiv_margin_boot``)
            lo, hi = pointwise_t_band(D[np.argsort([r["start"] for r in rows], kind="stable")])
            rng = np.random.default_rng(seed)
            boot = D[rng.integers(0, len(D), (2000, len(D)))].mean(axis=1)
            lo_b, hi_b = np.percentile(boot, [2.5, 97.5], axis=0)
            spm = {"n": len(A), "h0reject": None, "zstar": np.nan, "clusters": []}
            j1 = 1 if mode == "disp" else 0
            sd_ok = A.std(axis=0)[j1:].min() > 1e-9 and B.std(axis=0)[j1:].min() > 1e-9
            if len(A) >= 5 and sd_ok and np.all(D.std(axis=0)[j1:] > 1e-9):
                j0 = 1 if mode == "disp" else 0     # displacement: node 0 is 0 by construction
                try:
                    spm = paired_t(A[:, j0:], B[:, j0:])
                    for c in spm.get("clusters", []):
                        c["start_pct"] += j0
                        c["end_pct"] += j0
                except Exception as exc:          # spm1d data checks (e.g. zero variance)
                    spm["error"] = type(exc).__name__
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cm = [cmc_pair(A[i], B[i]) for i in range(len(A))]
            wr = {"pid": pid, "task": h.name, "channel": ch, "mode": mode, "unit": unit,
                  "n_cycles": len(A),
                  "spm_h0reject": spm.get("h0reject"), "spm_zstar": spm.get("zstar"),
                  # stored as the dataset does ("[(start %, end %, p), ...]"; blocks.parse_clusters)
                  "clusters": str([(round(c["start_pct"], 1), round(c["end_pct"], 1),
                                    float(c["p"])) for c in spm.get("clusters", [])]),
                  "n_clusters": len(spm.get("clusters", [])), "spm_error": spm.get("error", ""),
                  "mean_diff_rms": float(np.sqrt(np.mean(D.mean(axis=0) ** 2))),
                  "cmc_median": float(np.nanmedian(cm)) if cm else np.nan}
            for bound, tol in TOL.items():
                tv = tol[unit]
                wr[f"frac_inside_{bound}"] = float(np.mean((lo > -tv) & (hi < tv)))
            # displacement: node 0 is 0 by construction and does not count
            j_m = 1 if mode == "disp" else 0
            wr["equiv_margin"] = float(np.max(np.maximum(np.abs(lo[j_m:]), np.abs(hi[j_m:]))))
            wr["equiv_margin_boot"] = float(np.max(np.maximum(np.abs(lo_b[j_m:]),
                                                              np.abs(hi_b[j_m:]))))
            wave_rows.append(wr)
            key = f"{h.name}__{ch}" + ("" if mode == "abs" else "__disp")
            out_arrays[f"{key}__mean_s"] = A.mean(axis=0)
            out_arrays[f"{key}__sd_s"] = A.std(axis=0, ddof=1)
            out_arrays[f"{key}__mean_q"] = B.mean(axis=0)
            out_arrays[f"{key}__sd_q"] = B.std(axis=0, ddof=1)
            out_arrays[f"{key}__diff_mean"] = D.mean(axis=0)
            out_arrays[f"{key}__diff_lo"], out_arrays[f"{key}__diff_hi"] = lo, hi
    return ({"cycles": pd.DataFrame(cyc_rows), "features": pd.DataFrame(feat_rows),
             "waveforms": pd.DataFrame(wave_rows)}, out_arrays)


def cmc_pair(a: np.ndarray, b: np.ndarray) -> float:
    from biomech.stats.waveforms import cmc

    return cmc(np.vstack([a, b]))
