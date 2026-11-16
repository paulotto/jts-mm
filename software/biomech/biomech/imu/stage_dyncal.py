# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU dynamic-calibration audit of the dataset takes (stored by J4.6).

The BNO085 motion engine recalibrated itself during every take (an04 §1). The audit asks
whether its corrections show up as steps or drift-rate changes in the camera-free opening
error, aligned with still periods (when the engine recalibrates). Error trace: the E2a
opening angle minus the optics at the closed-pose samples (``camerafree`` arrays). Still
periods, steps and phase-transition jumps come from the same helpers as the validation part
(:func:`biomech.imu.dyncal.dyncal_trace`), so both studies use one definition.
The cohort stage compares the residual jumps at phase transitions after at least
``still_min_s`` of stillness with the others (two-sided Mann-Whitney test; descriptive).
``scripts/audit_recompute/imu_dyncal.py`` stays the independent re-computation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from biomech.pipeline import CohortContext, ParticipantContext, StageResult, stage

STILL_MIN_S = 2.0        # transitions after at least this much stillness (the audit's rule)


def dyncal_take(session: Any, pid: str, cf: dict[str, Any],
                params: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """Summary, still periods and transitions of one dataset take (empty summary with a note
    when the take has no camera-free result or no IMU streams)."""
    from biomech.imu import dyncal as D

    p = {**D.DEFAULTS, **dict(params or {})}
    if not cf or "open_deg__E2a" not in cf:
        return {"dyncal_summary": pd.DataFrame([{"pid": pid,
                                                 "note": "no camera-free result (no jaw IMU)"}])}
    g = D.imu_grid(session, float(p["grid_hz"]))
    if g is None:
        return {"dyncal_summary": pd.DataFrame([{"pid": pid, "note": "IMU streams missing"}])}
    t = np.asarray(cf["t_s"], float)
    e = np.asarray(cf["open_deg__E2a"], float) - np.asarray(cf["open_deg__opt"], float)
    e[~np.asarray(cf["closed"], bool)] = np.nan
    first: dict[str, Any] = {}
    for ph in session.phases:
        first.setdefault(ph.name, ph)
    phases = [(n, float(ph.t_start_us), float(ph.t_end_us)) for n, ph in first.items()]
    s, per, tr = D.dyncal_trace(pid, *g, phases, t, e, "E2a opening error at closed samples", p)
    return {"dyncal_summary": s.assign(note=""), "dyncal_periods": per,
            "dyncal_transitions": tr}


def transition_test(tr: pd.DataFrame, still_min_s: float = STILL_MIN_S) -> pd.DataFrame:
    """|Residual jump| at phase transitions after >= ``still_min_s`` stillness vs the others:
    medians, counts and a two-sided Mann-Whitney p (one row)."""
    from scipy.stats import mannwhitneyu

    if tr.empty or "residual_deg" not in tr:
        return pd.DataFrame()
    r = tr.residual_deg.abs()
    st = tr.still_in_gap_s >= still_min_s
    a, b = r[st].dropna().to_numpy(), r[~st].dropna().to_numpy()
    p = float(mannwhitneyu(a, b, alternative="two-sided").pvalue) if a.size and b.size \
        else np.nan
    return pd.DataFrame([{"still_min_s": still_min_s, "n_still": int(a.size),
                          "n_other": int(b.size),
                          "jump_still_median_deg": float(np.median(a)) if a.size else np.nan,
                          "jump_other_median_deg": float(np.median(b)) if b.size else np.nan,
                          "p_mannwhitney": p, "n_participants": int(tr.pid.nunique())}])


@stage("imu_dyncal", depends=("camerafree",), version="1",
       description="IMU dynamic-calibration audit (I3c, stored by J4.6): steps and drift "
                   "of the camera-free opening error around still periods and transitions")
def imu_dyncal_stage(ctx: ParticipantContext) -> StageResult:
    try:
        cf = ctx.upstream_arrays("camerafree")
    except Exception:                       # no camera-free unit (no jaw IMU)
        cf = {}
    return StageResult(tables=dyncal_take(ctx.session("motion"), ctx.pid, cf))


@stage("imu_dyncal_cohort", scope="cohort", depends=("imu_dyncal",), partial=("imu_dyncal",),
       version="1", description="IMU dynamic-calibration audit: transitions after stillness "
                                "vs the others over the cohort")
def imu_dyncal_cohort_stage(ctx: CohortContext) -> StageResult:
    tr = ctx.upstream("imu_dyncal", "dyncal_transitions")
    summ = ctx.upstream("imu_dyncal", "dyncal_summary")
    return StageResult(tables={"dyncal_transition_test": transition_test(tr),
                               "dyncal_summary": summ})
