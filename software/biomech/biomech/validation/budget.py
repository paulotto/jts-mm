# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""VD22 (descriptive): error budget per participant, and the marking gap.

Built from tables the participant stages already store; nothing is recomputed from the
recordings:

* ``budget``: per participant, part and segment the observed JTS-MM vs Qualisys error split
  in quadrature into the reference part and the JTS-MM part, σ_obs² = σ_JTS² + σ_ref² (V9,
  :func:`biomech.validation.quality.budget`). Motion rows are the ``ref_kinematics`` budget
  (bias-removed RMS per gated segment: incisor and condyle in mm, rotation in deg); marking
  rows are the VD4 marking summary (RMS point error of the teeth and of the condyle points
  against Qualisys, mm, with the reference uncertainty σ_ref of those landmarks).
  ``ref_var_share`` = σ_ref²/σ_obs², the share of the observed variance the reference alone
  accounts for, capped at 1. ``separable`` = σ_ref < σ_obs: only then can the JTS-MM part be
  resolved. Where the reference uncertainty reaches the observed error, ``sigma_jts_corrected``
  is NaN (not 0), the share is 1 and ``reason`` says why (:data:`NOT_SEPARABLE`).
* ``marking_holds``: per participant and hold the marking error, the CT orientation
  disagreement and the tip error it predicts, and whether the hold's CT labels were swapped.
* ``marking_causes``: per participant the frame errors of the marking (VD4) next to their
  candidate causes: the CT orientation disagreement of the two systems during the holds and
  the tip error it predicts over the CT lever (``ref_landmarks:hold_attribution``), the
  geometry of the Qualisys head cluster (smallest triangle height, ratio of the second to the
  first singular value: near 0 = nearly collinear; ``ref_quality:marker_geometry``), the holds
  whose CT labels were swapped (EV1; ``ref_quality:consistency``) and σ_ref of the teeth.
* ``marking_gap``: each of those quantities for the participant with the smaller and the
  one with the larger maxillary frame error, their difference (larger − smaller) and ratio.
  Descriptive (n = 2), no test; the two are separate reference measurements, no ranking.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from biomech.validation import quality as QY

DEFAULTS: dict[str, Any] = {
    "adequacy_ratio": 3.0,     # V9: reference adequate when σ_obs/σ_ref ≥ 3 (as ref_kinematics)
    "swap_frac_min": 0.5,      # a hold counts as label-swapped when ≥ this share of its CT
                               # frames was relabelled (EV1)
    "head_body": "HP",         # Qualisys head cluster (marker_geometry body)
}
# Holds per jaw (the VD4 landmarks marked with the CT tip)
HOLD_GROUPS: dict[str, tuple[str, ...]] = {
    "maxillary": ("max_point_1", "max_point_2", "max_point_3"),
    "mandibular": ("mand_point_1", "mand_point_2", "mand_point_3"),
    "condyle": ("tmj_left", "tmj_right"),
}
MARKING = (("teeth", "rms_teeth_mm", "sigma_ref_teeth_mm"),
           ("condyle", "rms_condyle_mm", "sigma_ref_condyle_mm"))
BUDGET_COLUMNS = ["pid", "part", "segment", "kind", "outcome", "unit", "sigma_obs", "sigma_ref",
                  "sigma_jts_corrected", "ratio", "adequate", "ref_var_share", "separable",
                  "reason"]
# reasons of a budget row without a JTS-MM part
NOT_SEPARABLE = "reference noise >= observed difference: JTS-MM part not resolvable"
MISSING = "sigma_obs or sigma_ref missing"
GAP_COLUMNS = ["quantity", "pid_smaller", "pid_larger", "value_smaller", "value_larger",
               "larger_minus_smaller", "larger_over_smaller"]
# marking_gap quantities in reading order (the marking_causes columns)
GAP_QUANTITIES = ("frame_max_deg", "frame_mand_deg", "rms_teeth_mm", "sigma_ref_teeth_mm",
                  "ct_rot_maxillary_deg", "ct_tip_maxillary_mm", "ct_rot_mandibular_deg",
                  "ct_tip_mandibular_mm", "ct_lever_mm", "head_min_height_mm",
                  "head_sv_ratio", "n_swapped_maxillary", "n_swapped_mandibular",
                  "n_swapped_condyle")


def _share(sigma_ref: Any, sigma_obs: Any) -> np.ndarray:
    """σ_ref²/σ_obs² capped at 1 (NaN where σ_obs is 0 or missing)."""
    so = np.asarray(sigma_obs, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(so > 0, np.minimum(np.asarray(sigma_ref, float) ** 2 / so ** 2, 1.0),
                        np.nan)


def _separability(out: pd.DataFrame) -> pd.DataFrame:
    """``separable`` (σ_ref < σ_obs, both finite) and ``reason``; ``sigma_jts_corrected`` NaN
    where the JTS-MM part is not resolvable (:func:`QY.budget` already returns NaN there)."""
    so, sr = out["sigma_obs"].to_numpy(float), out["sigma_ref"].to_numpy(float)
    finite = np.isfinite(so) & np.isfinite(sr)
    sep = finite & (sr < so)
    out["separable"] = sep
    out["reason"] = np.where(sep, "", np.where(finite, NOT_SEPARABLE, MISSING))
    out["sigma_jts_corrected"] = np.where(sep, out["sigma_jts_corrected"].to_numpy(float),
                                          np.nan)
    return out


def error_budget(kin_budget: pd.DataFrame, marking: pd.DataFrame,
                 adequacy_ratio: float = DEFAULTS["adequacy_ratio"]) -> pd.DataFrame:
    """Table ``budget`` (columns :data:`BUDGET_COLUMNS`): marking rows from the VD4 summary
    (one per participant and landmark group), then the motion rows of ``ref_kinematics``."""
    rows = []
    for r in marking.to_dict("records"):
        for outcome, obs, ref in MARKING:
            if obs in r and ref in r:
                rows.append({"pid": r["pid"], "part": "marking", "segment": "marking",
                             "kind": "marking", "outcome": outcome, "unit": "mm",
                             **QY.budget(float(r[obs]), float(r[ref]), adequacy_ratio)})
    mot = kin_budget.copy()
    if not mot.empty:
        mot.insert(1, "part", "motion")
        mot["unit"] = np.where(mot["outcome"] == "rotation", "deg", "mm")
    out = pd.concat([pd.DataFrame(rows), mot], ignore_index=True)
    if out.empty:
        return pd.DataFrame(columns=BUDGET_COLUMNS)
    out["ref_var_share"] = _share(out["sigma_ref"], out["sigma_obs"])
    out = _separability(out)
    return out[[c for c in BUDGET_COLUMNS if c in out.columns]]


def hold_group(landmark: str) -> str | None:
    """``maxillary`` | ``mandibular`` | ``condyle`` of a hold (None: not a VD4 hold)."""
    return next((g for g, names in HOLD_GROUPS.items() if landmark in names), None)


HOLD_COLUMNS = ["pid", "landmark", "group", "e_mm", "ct_rot_disagreement_deg",
                "predicted_tip_mm", "frac_label_swapped", "label_swapped"]


def marking_holds(holds: pd.DataFrame, consistency: pd.DataFrame,
                  swap_frac_min: float = DEFAULTS["swap_frac_min"]) -> pd.DataFrame:
    """Table ``marking_holds``: per participant and VD4 hold the marking error (``e_mm``,
    registered point error against Qualisys), the CT orientation disagreement of the two
    systems and the tip error it predicts over the CT lever (``ref_landmarks:
    hold_attribution``), and the share of the hold's CT frames whose labels the EV1
    correction swapped (``ref_quality:consistency``); ``label_swapped`` = share ≥
    ``swap_frac_min``."""
    if holds.empty:
        return pd.DataFrame(columns=HOLD_COLUMNS)
    h = holds.assign(group=holds["landmark"].map(hold_group)).dropna(subset=["group"])
    ct = consistency[(consistency["body"] == "CT") & (consistency["kind"] == "hold")] \
        if not consistency.empty else pd.DataFrame(columns=["pid", "segment"])
    sw = ct.rename(columns={"segment": "landmark"})[["pid", "landmark", "frac_label_swapped"]] \
        if "frac_label_swapped" in ct.columns else pd.DataFrame(columns=["pid", "landmark"])
    out = h.merge(sw, on=["pid", "landmark"], how="left")
    out["label_swapped"] = out["frac_label_swapped"].fillna(0.0) >= float(swap_frac_min)
    return out[[c for c in HOLD_COLUMNS if c in out.columns]].reset_index(drop=True)


def _median(d: pd.DataFrame, col: str) -> float:
    return float(d[col].median()) if not d.empty and col in d else np.nan


def marking_causes(marking: pd.DataFrame, holds: pd.DataFrame, geometry: pd.DataFrame,
                   head_body: str = DEFAULTS["head_body"]) -> pd.DataFrame:
    """Table ``marking_causes``: one row per participant of ``marking`` (VD4 summary), from
    the per-hold table ``holds`` (:func:`marking_holds`).

    ``ct_rot_<group>_deg`` / ``ct_tip_<group>_mm``: median over the group's holds of the CT
    orientation disagreement (deg) and of the tip error it predicts (mm, angle in rad × lever);
    ``ct_lever_mm``: the lever (tip distance from the CT marker centroid) behind that
    prediction; ``head_min_height_mm`` / ``head_sv_ratio``: Qualisys head-cluster geometry;
    ``n_swapped_<group>`` of ``n_holds_<group>``: label-swapped holds (EV1)."""
    geo = geometry[geometry["body"] == head_body].set_index("pid") \
        if not geometry.empty else pd.DataFrame()
    rows = []
    for m in marking.to_dict("records"):
        pid = m["pid"]
        row: dict[str, Any] = {"pid": pid, **{c: m.get(c, np.nan) for c in (
            "frame_max_deg", "frame_mand_deg", "rms_teeth_mm", "sigma_ref_teeth_mm")}}
        hp = holds[holds["pid"] == pid]
        for g in ("maxillary", "mandibular"):
            hg = hp[hp["group"] == g]
            row[f"ct_rot_{g}_deg"] = _median(hg, "ct_rot_disagreement_deg")
            row[f"ct_tip_{g}_mm"] = _median(hg, "predicted_tip_mm")
        ang = np.radians(hp["ct_rot_disagreement_deg"].to_numpy(float))
        ok = np.isfinite(ang) & (ang > 0)
        row["ct_lever_mm"] = float(np.median(hp["predicted_tip_mm"].to_numpy(float)[ok]
                                             / ang[ok])) if ok.any() else np.nan
        gr = geo.loc[pid] if pid in geo.index else None
        row["head_min_height_mm"] = float(gr["min_height_mm"]) if gr is not None else np.nan
        row["head_sv_ratio"] = (float(gr["sv2_mm"] / gr["sv1_mm"])
                                if gr is not None and gr["sv1_mm"] > 0 else np.nan)
        for g in HOLD_GROUPS:
            hg = hp[hp["group"] == g]
            row[f"n_holds_{g}"] = int(len(hg))
            row[f"n_swapped_{g}"] = int(hg["label_swapped"].sum()) if len(hg) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def marking_gap(causes: pd.DataFrame) -> pd.DataFrame:
    """Table ``marking_gap`` (columns :data:`GAP_COLUMNS`): per quantity of
    :data:`GAP_QUANTITIES` the value of the participant with the smaller (``pid_smaller``) and
    the one with the larger (``pid_larger``) maxillary frame error, their difference
    ``larger_minus_smaller`` and ratio ``larger_over_smaller`` (NaN when the value of the
    smaller is 0). Empty with fewer than two participants."""
    c = causes.dropna(subset=["frame_max_deg"]) if not causes.empty else causes
    if len(c) < 2:
        return pd.DataFrame(columns=GAP_COLUMNS)
    lo, hi = c.loc[c["frame_max_deg"].idxmin()], c.loc[c["frame_max_deg"].idxmax()]
    rows = []
    for q in GAP_QUANTITIES:
        if q not in c.columns:
            continue
        a, b = float(lo[q]), float(hi[q])
        rows.append({"quantity": q, "pid_smaller": lo["pid"], "pid_larger": hi["pid"],
                     "value_smaller": a, "value_larger": b, "larger_minus_smaller": b - a,
                     "larger_over_smaller": b / a if a != 0 and np.isfinite(a) else np.nan})
    return pd.DataFrame(rows, columns=GAP_COLUMNS)


def run_ref_budget(kin_budget: pd.DataFrame, marking: pd.DataFrame, holds: pd.DataFrame,
                   consistency: pd.DataFrame, geometry: pd.DataFrame,
                   params: Mapping[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    """The tables of ``ref_budget`` from the cohort-wide upstream tables."""
    p = {**DEFAULTS, **(params or {})}
    per_hold = marking_holds(holds, consistency, float(p["swap_frac_min"]))
    causes = marking_causes(marking, per_hold, geometry, str(p["head_body"]))
    return {"budget": error_budget(kin_budget, marking, float(p["adequacy_ratio"])),
            "marking_holds": per_hold, "marking_causes": causes,
            "marking_gap": marking_gap(causes)}
