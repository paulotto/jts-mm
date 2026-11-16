# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Evidence level of every requirement row (J5): one map, read by
:mod:`biomech.stats.requirements` when it stores the table and checked against the evidence
register of the analysis plan (``experiments/icorr2027/analysis_plan.md``, "Evidence register").

Levels (reader wording in ``etc/vocab.yaml`` domain ``evidence``):

* ``confirmatory`` -- a pre-specified test (C1 best-fit hinge);
* ``not_confirmed`` / ``not_confirmed_posthoc_rule`` -- the pre-specified C2 lead, not
  confirmed because the result depends on the onset rule (author decision 2026-10-15; the
  paper view refuses the label once both rules have Holm p < 0.05); the driving-burst rule
  itself was defined post hoc (D6);
* ``secondary_post_hoc`` -- a secondary, post hoc test (the hinge through the palpated
  condyles, A4 / D33);
* ``primary`` -- the pre-specified primary outcome of a description (C3a generic model);
* ``descriptive`` -- a pre-specified description (S1, C3a);
* ``exploratory`` -- X1-X3;
* ``post_hoc`` -- the C2-E family X6, specified before computing;
* ``post_hoc_descriptive`` -- rows added after the results were seen (J4).
"""

from __future__ import annotations

LEVELS = ("confirmatory", "not_confirmed", "not_confirmed_posthoc_rule", "secondary_post_hoc",
          "primary", "descriptive", "exploratory", "post_hoc", "post_hoc_descriptive")

REQUIREMENT_EVIDENCE: dict[str, str] = {
    "opening_mm": "descriptive",
    "protrusion_mm": "descriptive",
    "laterotrusion_mm": "descriptive",
    "opening_angle_deg": "descriptive",
    "translation_mm": "descriptive",
    "peak_vel_dps": "descriptive",
    "misalign_hinge_fit_mm": "confirmatory",
    "misalign_hinge_tmj_mm": "secondary_post_hoc",
    "bandwidth_hz": "exploratory",
    "lead_first_ms": "not_confirmed",
    "lead_drive_ms": "not_confirmed_posthoc_rule",
    "closer_release_ms": "exploratory",
    "causal_release_ms": "exploratory",
    "c2e_h1_ms": "post_hoc",
    "c2e_h2_ms": "post_hoc",
    "imu_angle_rmse_deg": "descriptive",
    "imu_opening_loa_loso_mm": "primary",
    "imu_opening_loa_subject_mm": "descriptive",
    "close_vel_dps": "post_hoc_descriptive",
    "free_peak_open_vel_dps": "post_hoc_descriptive",
    "peak_acc_dps2": "post_hoc_descriptive",
    "posterior_excursion_mm": "post_hoc_descriptive",
}


def requirement(key: str) -> str:
    """The stored evidence level of requirement ``key`` (KeyError for an unlisted key: every
    row must be registered)."""
    return REQUIREMENT_EVIDENCE[key]
