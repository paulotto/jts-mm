# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Electrode impedance and the pre-recording EMG check (an01 §3.4, D21).

The recorder ran ``jtsmm emg check`` before every take and stored the result in the
session (``/meta/emg_check_json``). Per muscle channel: the electrode–skin |Z| from the
ADS1299 AC lead-off tone at 31.25 Hz (|Z| = V_tone / I_lead-off − R_protect,
``jtsmm.emg.impedance``), the mains-removed 20–450 Hz noise RMS and the line peak of a
capture of a few seconds, each against the check's limits. The reference (DRL) electrode
is not measurable on this board (``jtsmm.emg.impedance.DRL_IMPEDANCE_NOTE``).

Nothing is excluded on these values (flag, don't exclude): the check judged the contact
before the take, ``qc:emg`` judges the recorded data. :func:`associations` asks whether
|Z| shows up in the recorded rest signal, with participant and muscle held fixed: both
differ in |Z| (beard, skin, electrode site) and in the recorded quantities for reasons of
their own (a weak opener has a low SNR at any |Z|).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from jtsmm.emg.impedance import CONTACT_Z_MAX_KOHM
from scipy import stats as sps

from biomech.stats.core import holm
from biomech.stats.families import wilcoxon_signed

# |Z| against the recorded rest quantities of qc:emg (the D21 relations), and the check's
# own noise and line readings against their recorded counterparts. Holm runs within each
# family (E30): "check vs recorded signal" and "check vs check".
RELATIONS = (("z_kohm", "snr_db"), ("z_kohm", "rest_rms_uv"), ("z_kohm", "rest_lf_uv"),
             ("z_kohm", "mains_share_raw"), ("noise_rms_uv", "rest_rms_uv"),
             ("line_peak_uv", "mains_share_raw"))
HOLM_FAMILY = {"z_kohm": "check_vs_recorded", "noise_rms_uv": "check_vs_check",
               "line_peak_uv": "check_vs_check"}


def _check(meta: dict[str, Any]) -> dict[str, Any] | None:
    raw = meta.get("emg_check_json")
    if raw is None:
        return None
    return json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))


def _muscle(c: dict[str, Any], names: list[str]) -> str:
    """Muscle of one channel of the check: by its channel number (1-based) in the session's EMG
    names, which are the montage as cabled (corrected on read for files whose stored names
    swap the digastrics, an03 §1); the check's own stored name only as a fallback."""
    ch = c.get("channel")
    if isinstance(ch, int) and 1 <= ch <= len(names):
        return names[ch - 1]
    return str(c.get("name", ""))


def check_table(takes: list[tuple[str, Any]], pid: str) -> pd.DataFrame:
    """One row per take and muscle channel. ``takes``: (take, SessionData) pairs. The takes
    are ordered by their start (``order``; ``minutes`` since the first take's check):
    electrode–skin impedance falls while the gel wets the skin."""
    found = []
    for take, sd in takes:
        d = _check(sd.meta) if sd is not None else None
        if d is not None:
            found.append((float(sd.meta.get("created_unix_utc", np.nan)), take, d))
    if not found:
        return pd.DataFrame()
    t0 = min(t for t, _, _ in found)
    rows = []
    names = {take: sd.muscle_names() for take, sd in takes if hasattr(sd, "muscle_names")}
    for k, (t, take, d) in enumerate(sorted(found, key=lambda f: f[0])):
        for c in d.get("channels", []):
            z = c.get("z_kohm")
            rows.append({
                "pid": pid, "take": take, "order": k + 1, "minutes": (t - t0) / 60.0,
                "muscle": _muscle(c, names.get(take, [])),
                "z_kohm": float(z) if z is not None else np.nan,
                "z_quality": c.get("z_quality", ""),
                "contact_ok": bool(z is not None and z <= CONTACT_Z_MAX_KOHM),
                "noise_rms_uv": float(c.get("noise_rms_uv") or np.nan),
                "noise_total_uv": float(c.get("noise_total_uv") or np.nan),
                "noise_ok": bool(c.get("noise_ok")),
                "line_peak_uv": float(c.get("line_peak_uv") or np.nan),
                "line_ok": bool(c.get("line_ok")),
                "check_passed": bool(d.get("passed")),
                "line_hz_check": float(d.get("line_hz_measured") or np.nan),
            })
    return pd.DataFrame(rows)


def rescale(chk: pd.DataFrame, scale: dict[Any, float] | None) -> pd.DataFrame:
    """|Z| divided by the bench scale k of each channel (bench check B1, an erratum only if
    the check asks for it; ``scale``: channel number 1-6 -> k, None or empty = unchanged).
    The channel of a row is its muscle's place in the montage (``io.session.MUSCLES``, the
    names as cabled); ``z_kohm_recorded`` keeps the recorded value, ``contact_ok`` follows the
    rescaled one."""
    if chk.empty or not scale:
        return chk
    from biomech.io.session import MUSCLES

    k = {MUSCLES[int(c) - 1]: float(v) for c, v in scale.items()}
    if any(not v > 0 for v in k.values()):
        raise ValueError(f"impedance_scale: every k must be > 0, got {scale}")
    out = chk.copy()
    out["z_kohm_recorded"] = out["z_kohm"]
    out["z_kohm"] = out["z_kohm"] / out["muscle"].map(k).fillna(1.0)
    out["contact_ok"] = out["z_kohm"].le(CONTACT_Z_MAX_KOHM)
    return out


def summary(chk: pd.DataFrame) -> pd.DataFrame:
    """Per take and muscle (and over all muscles): |Z| median, quartiles and range, the
    channels above the contact limit, and the check's noise and line verdicts.

    Second take vs first (same electrodes; E30): the unit is the **participant**. Row
    ``second_minus_first`` / ``all``: per participant the median paired change over their
    channels; ``n`` participants, median/quartiles/range over them, ``n_negative`` of them
    negative, and the exact two-sided Wilcoxon signed-rank test over the participants
    (``wilcoxon_p``; ``n_channels`` pairs behind them). Row ``second_minus_first`` /
    ``all_channels``: the participant x muscle pairs, descriptive only (no test: the six
    channels of a participant are not independent)."""
    if chk.empty:
        return pd.DataFrame()
    rows = []
    for take, g in chk.groupby("take", sort=False):
        for muscle, h in [*g.groupby("muscle"), ("all", g)]:
            z = h.z_kohm.dropna()
            rows.append({
                "take": take, "muscle": muscle, "n": int(len(z)),
                "z_median_kohm": float(z.median()), "z_q1_kohm": float(z.quantile(0.25)),
                "z_q3_kohm": float(z.quantile(0.75)), "z_min_kohm": float(z.min()),
                "z_max_kohm": float(z.max()), "z_limit_kohm": CONTACT_Z_MAX_KOHM,
                "above_limit_n": int((h.z_kohm > CONTACT_Z_MAX_KOHM).sum()),   # measured only
                "noise_fail_n": int((~h.noise_ok).sum()),
                "line_fail_n": int((~h.line_ok).sum()),
                "takes_passed_n": int(h.groupby("pid").check_passed.first().sum()),
                "takes_n": int(h.pid.nunique())})
    w = chk.pivot_table(index=["pid", "muscle"], columns="order", values="z_kohm")
    if {1, 2} <= set(w.columns):
        d = (w[2] - w[1]).dropna()
        mins = chk[chk["order"] == 2].groupby("pid").minutes.first()
        per = d.groupby(level="pid").median()
        base = {"take": "second_minus_first", "minutes_median": float(mins.median())}
        rows.append({**base, "muscle": "all", **_spread(per), "n_channels": int(len(d)),
                     "n_negative": int((per < 0).sum()), "wilcoxon_p": wilcoxon_exact(per)})
        rows.append({**base, "muscle": "all_channels", **_spread(d), "n_channels": int(len(d)),
                     "n_negative": int((d < 0).sum()), "wilcoxon_p": np.nan})
    return pd.DataFrame(rows)


def _spread(v: pd.Series) -> dict[str, Any]:
    return {"n": int(len(v)), "z_median_kohm": float(v.median()),
            "z_q1_kohm": float(v.quantile(0.25)), "z_q3_kohm": float(v.quantile(0.75)),
            "z_min_kohm": float(v.min()), "z_max_kohm": float(v.max())}


def wilcoxon_exact(d: pd.Series | np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p of the finite values (zeros dropped; exact without
    ties, the normal approximation with ties, :func:`families.wilcoxon_signed`); NaN
    below 6 values (the smallest attainable p, 1/16 at n = 5, could not reach 0.05)."""
    v = np.asarray(d, float)
    v = v[np.isfinite(v) & (v != 0)]
    return wilcoxon_signed(v, "two-sided", min_n=6)[1]


def _centred_ranks(v: pd.Series, groups: list[pd.Series]) -> np.ndarray:
    r = v.rank().to_numpy(float)
    for g in groups:
        r = r - pd.Series(r).groupby(g.to_numpy()).transform("mean").to_numpy()
    return r


def _adjusted(d: pd.DataFrame, x: str, y: str) -> tuple[float, float]:
    """Rank regression y ~ x + participant + muscle (fixed effects): slope and p."""
    X = pd.get_dummies(d[["pid", "muscle"]], drop_first=True, dtype=float)
    X.insert(0, "x", d[x].rank().to_numpy(float))
    X.insert(0, "c", 1.0)
    yy = d[y].rank().to_numpy(float)
    A = X.to_numpy(float)
    beta, *_ = np.linalg.lstsq(A, yy, rcond=None)
    dof = len(yy) - np.linalg.matrix_rank(A)
    if dof <= 0:
        return float(beta[1]), np.nan
    s2 = float(((yy - A @ beta) ** 2).sum() / dof)
    se = float(np.sqrt(s2 * np.linalg.pinv(A.T @ A)[1, 1]))
    return float(beta[1]), float(2 * sps.t.sf(abs(beta[1] / se), dof)) if se > 0 else np.nan


def associations(chk: pd.DataFrame, emg: pd.DataFrame, take: str = "motion",
                 n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Does the check predict the recorded rest signal? Per relation (:data:`RELATIONS`) over
    the channels of ``take`` joined with ``qc:emg``: the pooled Spearman rho with a 95 %
    percentile CI from resampling participants, the rank correlation within participants
    and within muscles (ranks centred per group), and the rank-regression slope with both
    held fixed (``rho_adj``: the OLS slope of the ranks of y on the ranks of x with
    participant and muscle dummies, a **rank slope**, not a correlation coefficient; p,
    Holm within its family) -- the one to read; the pooled rho mixes in the differences
    between participants and between muscles.

    Holm families (E30, ``holm_family``): ``check_vs_recorded`` (|Z| against the recorded
    rest signal, the D21 relations) and ``check_vs_check`` (the check's own noise and line
    readings against their recorded counterparts)."""
    if chk.empty or emg.empty:
        return pd.DataFrame()
    m = chk[chk["take"] == take].drop(columns="unit", errors="ignore").merge(
        emg.drop(columns="unit", errors="ignore"), on=["pid", "muscle"], how="inner")
    rng = np.random.default_rng(seed)
    rows = []
    for x, y in RELATIONS:
        if x not in m or y not in m:
            continue
        d = m[["pid", "muscle", x, y]].dropna().reset_index(drop=True)
        if len(d) < 6 or d[x].nunique() < 2 or d[y].nunique() < 2:
            continue                           # no ranks to correlate
        rho = float(sps.spearmanr(d[x], d[y]).statistic)
        pids = d.pid.unique()
        by = {p: d.index[d.pid == p].to_numpy() for p in pids}
        boot = []
        for _ in range(n_boot):
            i = np.concatenate([by[p] for p in rng.choice(pids, len(pids))])
            b = d.loc[i]
            if b[x].nunique() > 1 and b[y].nunique() > 1:
                boot.append(sps.spearmanr(b[x], b[y]).statistic)
        within = {}
        for k, gs in (("pid", [d.pid]), ("muscle", [d.muscle])):
            within[k] = float(np.corrcoef(_centred_ranks(d[x], gs),
                                          _centred_ranks(d[y], gs))[0, 1])
        beta, p = _adjusted(d, x, y)
        rows.append({"take": take, "check_quantity": x, "recorded_quantity": y,
                     "n": int(len(d)), "n_pid": int(len(pids)), "rho_pooled": rho,
                     "rho_pooled_lo": float(np.percentile(boot, 2.5)) if boot else np.nan,
                     "rho_pooled_hi": float(np.percentile(boot, 97.5)) if boot else np.nan,
                     "rho_within_pid": within["pid"], "rho_within_muscle": within["muscle"],
                     "rho_adj": beta, "p_adj": p})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["holm_family"] = out.check_quantity.map(HOLM_FAMILY).fillna("check_vs_check")
        out["p_holm"] = np.nan
        for _, idx in out.groupby("holm_family").groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_adj"].tolist())
    return out
