<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# VD33 wording pass: rename list

Change-log entry J30, VD33 of the validation study (2026-11-13). Paths relative to `software/biomech/biomech/`.
`n` = number of replacements of that exact string. Entries in (a) that also rename σ_ref are listed once, under (a).

## (a) reference → Qualisys / Delsys (105 entries, 125 replacements)

| old phrase | new phrase | where (item / file) | n |
|---|---|---|---|
| `"Reference quality over the take"` | `"Qualisys data quality over the take"` | val_ref_quality title — viz/views/validation.py | 2 |
| `"Qualisys reference quality over the take: largest` | `"quality of the Qualisys data over the take: largest` | val_ref_quality caption — viz/views/validation.py | 1 |
| `"Reference consistency vs viewing direction"` | `"Qualisys consistency vs viewing direction"` | val_ref_angle title (dashboard) — viz/views/validation.py | 2 |
| `"Error budget and reference adequacy"` | `"Error budget and Qualisys adequacy"` | val_error_budget title — viz/views/validation.py, viz/views/validation_r8.py | 3 |
| `JTS-MM and Qualisys, the propagated reference uncertainty σ_ref and the corrected ` | `JTS-MM and Qualisys, the propagated Qualisys uncertainty σ_Q and the corrected ` | val_error_budget caption — viz/views/validation.py | 1 |
| `("✗", "reference not adequate (σ_obs/σ_ref < 3)")` | `("✗", "Qualisys not adequate (σ_obs/σ_Q < 3)")` | val_error_budget caption — viz/views/validation.py | 1 |
| `"Table: reference quality"` | `"Table: Qualisys data quality"` | val_table_quality title (dashboard) — viz/views/validation.py | 1 |
| `"the values at the 200-Hz reference frames"` | `"the values at the 200-Hz Qualisys frames"` | val_fusion_segments note — viz/views/validation.py | 1 |
| `"sigma_ref": ("reference part (σ_ref)", "system:qualisys", "d")}` | `"sigma_ref": ("Qualisys part (σ_Q)", "system:qualisys", "d")}` | val_budget / val_marking_combined / val_motion_agreement legend — viz/views/validation_part_i.py | 1 |
| `"quadrature into the reference part and the JTS-MM part (σ_obs² = σ_JTS² + σ_ref²) "` | `"quadrature into the Qualisys part and the JTS-MM part (σ_obs² = σ_JTS² + σ_Q²) "` | val_budget caption — viz/views/validation_part_i.py | 1 |
| `("black marker", "reference part"), pshape_text` | `("black marker", "Qualisys part"), pshape_text` | val_budget caption — viz/views/validation_part_i.py | 1 |
| `"Where the reference part reaches the observed error (σ_ref ≥ σ_obs) the "` | `"Where the Qualisys part reaches the observed difference (σ_Q ≥ σ_obs) the "` | val_budget note — viz/views/validation_part_i.py | 1 |
| `Column("sigma_ref", "Reference part", "", 2)` | `Column("sigma_ref", "Qualisys part", "", 2)` | val_table_budget column head — viz/views/validation_part_i.py | 1 |
| `Column("ref_var_share", "Reference share", "", 2)` | `Column("ref_var_share", "Qualisys share", "", 2)` | val_table_budget column head — viz/views/validation_part_i.py | 1 |
| `"error budget per participant: observed error, its reference part (propagated "` | `"error budget per participant: observed difference, its Qualisys part (propagated "` | val_table_budget caption — viz/views/validation_part_i.py | 1 |
| `"reference share = σ_ref²/σ_obs², the share of the observed variance the reference "` | `"Qualisys share = σ_Q²/σ_obs², the share of the observed variance the Qualisys part "` | val_table_budget caption — viz/views/validation_part_i.py | 1 |
| `"part cannot be separated from the reference noise and is left empty",` | `"part cannot be separated from the Qualisys noise and is left empty",` | val_table_budget caption — viz/views/validation_part_i.py | 1 |
| `"calibration-tool marker labels (counts) and the reference uncertainty of the teeth "` | `"calibration-tool marker labels (counts) and the Qualisys uncertainty of the teeth "` | val_table_marking_gap caption — viz/views/validation_part_i.py | 1 |
| `label="reference part (σ_ref²)"` | `label="Qualisys part (σ_Q²)"` | val_budget form stacked legend — viz/views/validation_r8.py | 1 |
| `"+ σ_ref², the reference part σ_ref²/σ_obs² (capped at 100 %) and the JTS-MM part "` | `"+ σ_Q², the Qualisys part σ_Q²/σ_obs² (capped at 100 %) and the JTS-MM part "` | val_budget form stacked caption — viz/views/validation_r8.py | 1 |
| `[("black bar", "reference part"), ("blue bar", "JTS-MM part (the rest)"),` | `[("black bar", "Qualisys part"), ("blue bar", "JTS-MM part (the rest)"),` | val_budget form stacked caption — viz/views/validation_r8.py | 1 |
| `"JTS-MM part not resolvable (σ_ref ≥ σ_obs, all reference)"` | `"JTS-MM part not resolvable (σ_Q ≥ σ_obs, all Qualisys)"` | val_budget form stacked caption — viz/views/validation_r8.py | 1 |
| `"the reference share",` | `"the Qualisys share",` | val_budget form stacked caption — viz/views/validation_r8.py | 1 |
| `"Qualisys error σ_obs, its reference part σ_ref and the JTS-MM part σ_JTS = "` | `"Qualisys difference σ_obs, its Qualisys part σ_Q and the JTS-MM part σ_JTS = "` | val_budget form dot caption — viz/views/validation_r8.py | 1 |
| `("black", "reference part"), pshape_text` | `("black", "Qualisys part"), pshape_text` | val_budget form dot caption — viz/views/validation_r8.py | 1 |
| `"estimate and the reference through the dataset's low-pass): "` | `"estimate and Qualisys through the dataset's low-pass): "` | val_camerafree forms caption — viz/views/validation_r8.py | 1 |
| `f"derivative; reference: Qualisys at its own rate, {lp_r:g} Hz low-pass, "` | `f"derivative; Qualisys at its own rate: {lp_r:g} Hz low-pass, "` | val_sampling form quantities caption — viz/views/validation_r8.py | 1 |
| `Axis("Ratio to the reference", "")` | `Axis("Ratio to Qualisys at 200 Hz", "")` | val_sampling form quantities x axis — viz/views/validation_r8.py | 1 |
| `f"(JTS-MM frames at {fr:.0f} Hz; {_chain_text()}) relative to the reference, for the "` | `f"(JTS-MM frames at {fr:.0f} Hz; {_chain_text()}) relative to Qualisys at 200 Hz, for the "` | val_sampling form quantities caption — viz/views/validation_r8.py | 1 |
| `"ratio 1, equal to the reference"` | `"ratio 1, equal to Qualisys at 200 Hz"` | val_sampling caption — viz/views/validation_r8.py, viz/views/validation_r7.py | 2 |
| `"with the reference low-pass"` | `"with the Qualisys low-pass"` | val_sampling form bandwidth legend — viz/views/validation_r8.py | 2 |
| `"movement bandwidth of the 200 Hz Qualisys reference per participant and task: (a) the "` | `"movement bandwidth of the 200 Hz Qualisys data per participant and task: (a) the "` | val_sampling form bandwidth caption — viz/views/validation_r8.py | 1 |
| `"power above the Nyquist frequency of the JTS-MM optics, computed with the reference "` | `"power above the Nyquist frequency of the JTS-MM optics, computed with the Qualisys "` | val_sampling form bandwidth caption — viz/views/validation_r8.py | 1 |
| `"spectrum reflects reference noise."` | `"spectrum reflects Qualisys noise."` | val_sampling form bandwidth note — viz/views/validation_r8.py | 1 |
| `"how well the Qualisys reference resolves the JTS-MM error per participant, segment "` | `"how well Qualisys resolves the JTS-MM − Qualisys difference per participant, segment "` | val_error_budget forms ratio/heatmap caption — viz/views/validation_r8.py | 1 |
| `"the propagated reference uncertainty σ_ref",` | `"the propagated Qualisys uncertainty σ_Q",` | val_error_budget forms ratio/heatmap caption — viz/views/validation_r8.py | 1 |
| `notes=[f"Reference adequate where the ratio reaches {adeq:g}."]` | `notes=[f"Qualisys adequate where the ratio reaches {adeq:g}."]` | val_error_budget forms note — viz/views/validation_r8.py | 1 |
| `"per participant and task the RMS over the task's reference frames; the "` | `"per participant and task the RMS over the task's 200-Hz Qualisys frames; the "` | val_fusion_accuracy statistic — viz/views/validation_r8.py | 1 |
| `"reference": "Qualisys reference",` | `"reference": "Qualisys side",` | val_marking_causes legend (group key unchanged) — viz/views/validation_r8.py | 1 |
| `"the marking results and their candidate causes on the reference side (Qualisys tooth "` | `"the marking results and their candidate causes on the Qualisys side (tooth "` | val_marking_causes caption — viz/views/validation_r8.py | 1 |
| `"every movement-task frame that passes the reference gate, thinned "` | `"every movement-task frame that passes the Qualisys quality gate, thinned "` | val_paths3d_error caption — viz/views/validation_r8.py | 1 |
| `"opening-rate bandwidth the reference shows: "` | `"opening-rate bandwidth Qualisys shows: "` | val_sampling caption — viz/views/validation_r7.py | 1 |
| `"below which 95 % of the opening-rate power of the 200 Hz reference lies, per task, "` | `"below which 95 % of the opening-rate power of Qualisys at 200 Hz lies, per task, "` | val_sampling caption — viz/views/validation_r7.py | 1 |
| `"reference noise and is an upper bound of the opening-rate spectrum, not the "` | `"Qualisys noise and is an upper bound of the opening-rate spectrum, not the "` | val_sampling caption — viz/views/validation_r7.py | 1 |
| `"reflects reference noise",` | `"reflects Qualisys noise",` | val_sampling caption — viz/views/validation_r7.py | 1 |
| `Panel("(a) Peak relative to the 200 Hz reference"` | `Panel("(a) Peak relative to Qualisys at 200 Hz"` | val_sampling panel (a) title — viz/views/validation_r7.py | 1 |
| `("qualisys", "Qualisys, 200 Hz (reference)", "system:qualisys")` | `("qualisys", "Qualisys, 200 Hz", "system:qualisys")` | val_fusion_smoothing legend — viz/views/validation_r7.py | 1 |
| `"(c) RMS of the error against Qualisys above 6 Hz at the 200-Hz reference frames. "` | `"(c) RMS of the difference to Qualisys above 6 Hz at the 200-Hz Qualisys frames. "` | val_fusion_smoothing caption — viz/views/validation_r7.py | 1 |
| `("sensor", "accelerometer, reference orientation", "system:jtsmm", "o")` | `("sensor", "accelerometer, Qualisys orientation", "system:jtsmm", "o")` | val_imu_accel legend — viz/views/validation_r7.py | 1 |
| `label="reference acceleration (RMS)"` | `label="Qualisys acceleration (RMS)"` | val_imu_accel legend — viz/views/validation_r7.py | 1 |
| `"(a) a few seconds of one task; (b) RMS error per task for the accelerometer with the " ⏎ "reference orientation, the same without the lever arm, and with gravity removed by "` | `"(a) a few seconds of one task; (b) RMS difference per task for the accelerometer with the " ⏎ "Qualisys orientation, the same without the lever arm, and with gravity removed by "` | val_imu_accel caption — viz/views/validation_r7.py | 1 |
| `"reference acceleration (bars). The P2 head-IMU reference is limited by its nearly " ⏎ "collinear Qualisys head cluster",` | `"Qualisys acceleration (bars). At the P2 head IMU the Qualisys acceleration is limited " ⏎ "by the nearly collinear head cluster",` | val_imu_accel caption — viz/views/validation_r7.py | 1 |
| `[("lines (a)", "Qualisys at the IMU site, accelerometer with the reference "` | `[("lines (a)", "Qualisys at the IMU site, accelerometer with the Qualisys "` | val_imu_accel caption — viz/views/validation_r7.py | 1 |
| `("bars (b)", "reference acceleration (RMS)"),` | `("bars (b)", "Qualisys acceleration (RMS)"),` | val_imu_accel caption — viz/views/validation_r7.py | 1 |
| `Column("rms_ref_mm_s2", "Reference RMS", MS2, 0, "S")` | `Column("rms_ref_mm_s2", "Qualisys RMS", MS2, 0, "S")` | val_table_imu_accel column head — viz/views/validation_r7.py | 1 |
| `"held-out tasks (6 Hz): RMS of the reference dynamic acceleration, RMS error with the " ⏎ "reference orientation, without the lever arm and with the IMU's own orientation, "` | `"held-out tasks (6 Hz): RMS of the Qualisys dynamic acceleration, RMS difference with the " ⏎ "Qualisys orientation, without the lever arm and with the IMU's own orientation, "` | val_table_imu_accel caption — viz/views/validation_r7.py | 1 |
| `notes=["R² below 0: the error variance exceeds the variance of the reference " ⏎ "acceleration. At the head the dynamic acceleration is small and the Qualisys " ⏎ "reference depends on the head-cluster geometry, which is nearly collinear for "` | `notes=["R² below 0: the variance of the difference exceeds the variance of the Qualisys " ⏎ "acceleration. At the head the dynamic acceleration is small and the Qualisys " ⏎ "acceleration depends on the head-cluster geometry, which is nearly collinear for "` | val_table_imu_accel note — viz/views/validation_r7.py | 1 |
| `"jaw-in-head agreement over the pooled movement tasks (reference quality gate "` | `"jaw-in-head agreement over the pooled movement tasks (Qualisys quality gate "` | val_table_filtered caption — viz/views/validation_r7.py | 1 |
| `reference-quality` | `Qualisys quality` | every 'reference-quality gate' (val_kin_timeseries, val_robustness, val_decomposition, val_error_budget, val_table_segments*, val_kin_agreement*, val_table_features*, val_motion_agreement, val_hinge, val_table_hinge, val_table_filtered, val_table_segments_aligned_freemotion) — viz/views/validation.py, viz/views/validation_part_i.py, viz/views/validation_r7.py, viz/views/validation_gallery.py | 13 |
| `"reference part "` | `"Qualisys part "` | val_marking_combined caption — viz/views/validation_condensed.py | 1 |
| `"reference part, at rest and in the "` | `"Qualisys part, at rest and in the "` | val_motion_agreement caption — viz/views/validation_condensed.py | 1 |
| `"budget of the incisor and condyle motion split into the reference and the JTS-MM part "` | `"budget of the incisor and condyle motion split into the Qualisys and the JTS-MM part "` | val_motion_agreement caption — viz/views/validation_condensed.py | 1 |
| `"peak opening velocity over the 200 Hz reference (median per-cycle ratio), and the "` | `"peak opening velocity over Qualisys at 200 Hz (median per-cycle ratio), and the "` | val_table_fitness note — viz/views/validation_condensed.py | 1 |
| `"(reference gate passed); the RMS of the drawn frames equals the table value."` | `"(Qualisys quality gate passed); the RMS of the drawn frames equals the table value."` | val_kin_error_ecdf note (also val_motion_agreement) — viz/views/validation_gallery.py | 1 |
| `"for the IMU alone, every estimate and the reference through the dataset's 6 Hz "` | `"for the IMU alone, every estimate and Qualisys through the dataset's 6 Hz "` | val_camerafree caption — viz/views/camerafree.py | 1 |
| `"keeps more of the peak, so the primary requirement is a lower bound; the 200 Hz " ⏎ "reference comparison of the companion validation study bounds the remaining loss and "` | `"keeps more of the peak, so the primary requirement is a lower bound; the comparison with " ⏎ "Qualisys at 200 Hz in the companion validation study bounds the remaining loss and "` | JNER req_lowpass_bound caption (suppLowpassBound) — viz/views/kin_r8.py | 1 |
| `"One compromise axis, chewing (reference system)"` | `"One compromise axis, chewing (Qualisys)"` | JNER table_compatibility form with_chewing row (not exported; form own) — viz/views/kin_r8.py | 1 |
| `(". Chewing: best-fit hinge on the Qualisys reference of the "` | `(". Chewing: best-fit hinge on the Qualisys data of the "` | JNER table_compatibility form with_chewing note (not exported) — viz/views/kin_r8.py | 1 |
| `" (chewing reference: n = 1)"` | `" (chewing row: n = 1)"` | JNER table_compatibility form with_chewing n (not exported) — viz/views/kin_r8.py | 1 |
| `qualisys: {label: "Qualisys (reference)", short: "Qualisys"}` | `qualisys: {label: "Qualisys", short: "Qualisys"}` | vocab estimator.qualisys — etc/vocab.yaml | 1 |
| `ref200: {label: "Qualisys 200 Hz (reference)", short: Qualisys 200 Hz}` | `ref200: {label: "Qualisys 200 Hz", short: Qualisys 200 Hz}` | vocab pipeline.ref200 — etc/vocab.yaml | 1 |
| `sensor: {label: "accelerometer (reference orientation)", short: "sensor"}` | `sensor: {label: "accelerometer (Qualisys orientation)", short: "sensor"}` | vocab accel_variant.sensor — etc/vocab.yaml | 1 |
| `validation: {label: "systems paper: JTS-MM against the reference"}` | `validation: {label: "systems paper: JTS-MM against Qualisys"}` | vocab video layout.validation — etc/vocab.yaml | 1 |
| `jtsmm_only: {label: "JTS-MM filtered, reference unfiltered", short: "JTS-MM filtered"}` | `jtsmm_only: {label: "JTS-MM filtered, Qualisys unfiltered", short: "JTS-MM filtered"}` | vocab filter_chain.jtsmm_only (val_table_filtered rows) — etc/vocab.yaml | 1 |
| `validation: reference-system agreement` | `validation: agreement with Qualisys and Delsys` | vocab domain.validation — etc/vocab.yaml | 1 |
| `ref_quality: reference-internal quality` | `ref_quality: Qualisys-internal quality` | vocab stage.ref_quality — etc/vocab.yaml | 1 |
| `label: "Reference uncertainty, teeth"` | `label: "Qualisys uncertainty, teeth"` | catalog sigma_ref_teeth_mm — etc/catalog.yaml | 1 |
| `label: "Reference uncertainty, condyles"` | `label: "Qualisys uncertainty, condyles"` | catalog sigma_ref_condyle_mm, sigma_ref_tmj_mm — etc/catalog.yaml | 2 |
| `label: "Reference uncertainty, incisor"` | `label: "Qualisys uncertainty, incisor"` | catalog sigma_ref_incisor_mm — etc/catalog.yaml | 1 |
| `label: "Reference uncertainty, rotation"` | `label: "Qualisys uncertainty, rotation"` | catalog sigma_ref_rot_deg — etc/catalog.yaml | 1 |
| `label: "Reference uncertainty"` | `label: "Qualisys uncertainty"` | catalog sigma_ref, sigma_ref_mm — etc/catalog.yaml | 2 |
| `adequate: {label: "Reference adequate"` | `adequate: {label: "Qualisys adequate"` | catalog adequate — etc/catalog.yaml | 1 |
| `range_ref_deg: {label: "Reference range"` | `range_ref_deg: {label: "Qualisys range"` | catalog range_ref_deg — etc/catalog.yaml | 1 |
| `range_ref: {label: "Reference range"` | `range_ref: {label: "Qualisys range"` | catalog range_ref — etc/catalog.yaml | 1 |
| `lag_ref_ms: {label: "IMU lag vs reference"` | `lag_ref_ms: {label: "IMU lag vs Qualisys"` | catalog lag_ref_ms — etc/catalog.yaml | 1 |
| `label: "Reference acceleration"` | `label: "Qualisys acceleration"` | catalog rms_ref_mm_s2 — etc/catalog.yaml | 1 |
| `ref_var_share: {label: "Reference share of the variance", symbol: "\\sigma_{\\mathrm{ref}}^2/` | `ref_var_share: {label: "Qualisys share of the variance", symbol: "\\sigma_{\\mathrm{Q}}^2/` | catalog ref_var_share label + symbol (b: \sigma_{\mathrm{ref}} -> \sigma_{\mathrm{Q}}) — etc/catalog.yaml | 1 |
| `title: "Reference uncertainty", caption: "Propagated reference uncertainty of the outcomes` | `title: "Qualisys uncertainty", caption: "Propagated Qualisys uncertainty of the outcomes` | tables ref_quality:sigma_ref — etc/tables.yaml | 1 |
| `title: "Landmark reference uncertainty", caption: "Reference uncertainty of the Qualisys landmarks` | `title: "Landmark Qualisys uncertainty", caption: "Uncertainty of the Qualisys landmarks` | tables ref_landmarks:landmark_sigma_ref — etc/tables.yaml | 1 |
| `registered point errors of the teeth and condyles, reference uncertainty, tooth distance difference, frame errors,` | `registered point differences of the teeth and condyles, Qualisys uncertainty, tooth distance difference, frame differences,` | tables ref_landmarks:marking_summary — etc/tables.yaml | 1 |
| `observed dispersion, reference uncertainty, corrected JTS-MM dispersion, their ratio and whether the reference is adequate."` | `observed dispersion, Qualisys uncertainty, corrected JTS-MM dispersion, their ratio and whether Qualisys is adequate."` | tables ref_kinematics:budget — etc/tables.yaml | 1 |
| `caption: "Cycles found on the reference signal with` | `caption: "Cycles found on the Qualisys signal with` | tables ref_agreement:cycles — etc/tables.yaml | 1 |
| `from Qualisys at 200 Hz (reference), from Qualisys at 40 Hz` | `from Qualisys at 200 Hz, from Qualisys at 40 Hz` | tables ref_sampling:peaks — etc/tables.yaml | 1 |
| `median ratio of the peak to the 200 Hz Qualisys reference over the cycles` | `median ratio of the peak to Qualisys at 200 Hz over the cycles` | tables ref_sampling:bias — etc/tables.yaml | 1 |
| `The reference acceleration carries more noise` | `The Qualisys acceleration carries more noise` | tables ref_sampling:bias — etc/tables.yaml | 1 |
| `title: "Movement bandwidth, filtered reference", caption: "Per task: frequencies below which 50, 95, and 99 % of the opening-rate power of the 200 Hz Qualisys reference lie` | `title: "Movement bandwidth, filtered Qualisys", caption: "Per task: frequencies below which 50, 95, and 99 % of the opening-rate power of Qualisys at 200 Hz lie` | tables ref_sampling:spectrum — etc/tables.yaml | 1 |
| `Savitzky-Golay derivative) to the 200 Hz Qualisys reference, with a cycle-bootstrap 95 % CI."` | `Savitzky-Golay derivative) to Qualisys at 200 Hz, with a cycle-bootstrap 95 % CI."` | tables ref_sampling:peak_ratio_open — etc/tables.yaml | 1 |
| `title: "IMU opening angle against the references"` | `title: "IMU opening angle against Qualisys and the optics"` | tables ref_fusion:imu_opening — etc/tables.yaml | 1 |
| `with the range of the reference angle.` | `with the range of the Qualisys angle.` | tables ref_fusion:imu_opening — etc/tables.yaml | 1 |
| `and the RMS difference, at the 200-Hz reference frames; Qualisys itself` | `and the RMS difference, at the 200-Hz Qualisys frames; Qualisys itself` | tables ref_fusion:smoothness — etc/tables.yaml | 1 |
| `RMS error of the dynamic acceleration at the IMU site, the reference RMS,` | `RMS difference of the dynamic acceleration at the IMU site, the Qualisys RMS,` | tables ref_imu_accel:accel — etc/tables.yaml | 1 |
| `and outcome (VD22): observed error, reference uncertainty, the JTS-MM part corrected in quadrature, the adequacy ratio and the share of the observed variance the reference alone accounts for."` | `and outcome (VD22): observed difference, Qualisys uncertainty, the JTS-MM part corrected in quadrature, the adequacy ratio and the share of the observed variance the Qualisys part alone accounts for."` | tables ref_budget:budget — etc/tables.yaml | 1 |
| `the holds with swapped CT labels (EV1) and the reference uncertainty of the teeth."` | `the holds with swapped CT labels (EV1) and the Qualisys uncertainty of the teeth."` | tables ref_budget:marking_causes — etc/tables.yaml | 1 |
| `to the 200 Hz Qualisys reference (20 Hz low-pass` | `to Qualisys at 200 Hz (20 Hz low-pass` | tables ref_sampling:peak_ratio_acc — etc/tables.yaml | 1 |
| `"(b) Reference spectrum, f95 (unfiltered)"` | `"(b) Qualisys spectrum, f95 (unfiltered)"` | val_sampling panel (b) title — viz/views/validation_r7.py | 1 |

## (b) σ_ref → σ_Q (2 entries, 24 replacements)

| old phrase | new phrase | where (item / file) | n |
|---|---|---|---|
| `"σ_obs orange, σ_ref of Qualisys grey, the corrected σ_JTS blue "` | `"σ_obs orange, σ_Q (Qualisys) grey, the corrected σ_JTS blue "` | val_error_budget caption — viz/views/validation.py | 1 |
| `σ_ref` | `σ_Q` | every remaining σ_ref in labels, legends and captions (val_error_budget, val_marking_accuracy, val_marking_combined, val_budget, val_table_budget, docstrings) — viz/views/validation.py, viz/views/validation_r8.py, viz/views/validation_r7.py, viz/views/validation_part_i.py, viz/views/validation_condensed.py, viz/views/validation_gallery.py | 23 |

Also renamed σ_ref → σ_Q inside 13 entries listed under (a)/(c) (14 occurrences): val_error_budget caption (2), val_table_marking note, the budget legend `Qualisys part (σ_Q)` (val_budget / val_marking_combined / val_motion_agreement), val_budget caption + note, val_table_budget caption, val_budget forms stacked/dot, val_error_budget forms ratio/heatmap, and the LaTeX symbol of `catalog ref_var_share` (`\sigma_{\mathrm{ref}}^2/…` → `\sigma_{\mathrm{Q}}^2/…`). After the pass the systems export contains no `σ\textsubscript{ref}` / `\mathrm{ref}` σ anywhere (captions, figures, tables).

## (c) error → difference (119 entries, 135 replacements)

| old phrase | new phrase | where (item / file) | n |
|---|---|---|---|
| `"Fusion: rotation error over the take"` | `"Fusion: rotation difference over the take"` | val_fusion_timeseries title (dashboard) — viz/views/validation.py | 2 |
| `"Rotation error, own frames"` | `"Rotation difference, own frames"` | val_fusion_timeseries panel — viz/views/validation.py | 1 |
| `"Rotation error, frames aligned"` | `"Rotation difference, frames aligned"` | val_fusion_timeseries panel — viz/views/validation.py | 1 |
| `"Incisor error, own frames"` | `"Incisor difference, own frames"` | val_fusion_timeseries panel — viz/views/validation.py | 1 |
| `"Incisor error, frames aligned"` | `"Incisor difference, frames aligned"` | val_fusion_timeseries panel — viz/views/validation.py | 1 |
| `"Fusion: per-segment error, optical vs fused"` | `"Fusion: per-segment difference, optical vs fused"` | val_fusion_segments title — viz/views/validation.py, viz/views/validation_r8.py | 3 |
| `"Fusion: error spectrum"` | `"Fusion: difference spectrum"` | val_fusion_spectrum title (dashboard) — viz/views/validation.py | 2 |
| `"Rotation error PSD"` | `"Rotation difference PSD"` | val_fusion_spectrum panel — viz/views/validation.py | 1 |
| `"Welch spectrum of the rotation error (held-out tasks)."` | `"Welch spectrum of the rotation difference to Qualisys (held-out tasks)."` | val_fusion_spectrum caption — viz/views/validation.py | 1 |
| `"Fusion: complementary-filter error against τ"` | `"Fusion: complementary-filter difference against τ"` | val_fusion_tau title (dashboard) — viz/views/validation.py | 2 |
| `"how well each estimator bridges simulated optical dropouts: RMS error against "` | `"how well each estimator bridges simulated optical dropouts: RMS difference to "` | val_fusion_dropout caption — viz/views/validation.py | 1 |
| `"its incisor error grows with the gap length (c, d); a secondary result."` | `"its incisor difference to Qualisys grows with the gap length (c, d); a secondary result."` | val_fusion_dropout note — viz/views/validation.py | 1 |
| `"Point marking: error per landmark"` | `"Point marking: difference per landmark"` | val_marking_accuracy title — viz/views/validation.py, viz/views/validation_gallery.py | 3 |
| `"point-marking error of JTS-MM against Qualisys: the JTS-MM landmark mapped into the "` | `"point-marking difference of JTS-MM from Qualisys: the JTS-MM landmark mapped into the "` | val_marking_accuracy caption — viz/views/validation.py | 1 |
| `("black markers with whiskers", "error with its 95 % CI")` | `("black markers with whiskers", "difference with its 95 % CI")` | val_marking_accuracy caption — viz/views/validation.py | 1 |
| `"Point marking: what explains the hold errors"` | `"Point marking: what explains the hold differences"` | val_tip_holds title — viz/views/validation.py | 2 |
| `Axis("Marking error \|e\|", MM)` | `Axis("Marking difference \|e\|", MM)` | val_tip_holds y axis — viz/views/validation.py | 1 |
| `"what explains the point-marking errors, per landmark hold: (a) the residual of the "` | `"what explains the point-marking differences, per landmark hold: (a) the residual of the "` | val_tip_holds caption — viz/views/validation.py | 1 |
| `"joint tip fit on the tooth holds; (b–d) the marking error against candidate causes: "` | `"joint tip fit on the tooth holds; (b–d) the marking difference against candidate causes: "` | val_tip_holds caption — viz/views/validation.py | 1 |
| `"maxillary and mandibular anatomical-frame angle errors, the incisor offset at "` | `"maxillary and mandibular anatomical-frame angle differences, the incisor offset at "` | val_table_marking caption — viz/views/validation.py | 1 |
| `"Registered point errors and σ_ref in the supplementary source data."` | `"Registered point differences and σ_Q in the supplementary source data."` | val_table_marking note — viz/views/validation.py | 1 |
| `p = Panel(title, Axis("Error", unit), Axis(""), [], legend="none")` | `p = Panel(title, Axis("σ", unit), Axis(""), [], legend="none")` | val_budget x axis (also in val_marking_combined (b), val_motion_agreement (c)) — viz/views/validation_part_i.py | 1 |
| `"error budget per participant: the observed JTS-MM vs Qualisys error split in "` | `"error budget per participant: the observed JTS-MM − Qualisys difference split in "` | val_budget caption — viz/views/validation_part_i.py | 1 |
| `"for (a) the point marking (registered point error of the teeth and condyle points) "` | `"for (a) the point marking (registered point difference of the teeth and condyle points) "` | val_budget caption — viz/views/validation_part_i.py | 1 |
| `[("wide bar", "observed error"), ("narrow bar", "JTS-MM part"),` | `[("wide bar", "observed difference"), ("narrow bar", "JTS-MM part"),` | val_budget caption — viz/views/validation_part_i.py | 1 |
| `marking = RMS point error` | `marking = RMS point difference` | val_budget, val_table_budget statistic (also val_marking_combined, val_motion_agreement) — viz/views/validation_part_i.py, viz/views/validation_r8.py | 3 |
| `"The marking error per hold against the tip error the calibration tool's "` | `"The marking difference per hold against the tip difference the calibration tool's "` | val_budget note — viz/views/validation_part_i.py | 1 |
| `Column("what", "Error", align="l")` | `Column("what", "Difference", align="l")` | val_table_budget column head — viz/views/validation_part_i.py | 1 |
| `"the frame errors of the point marking and their candidate causes for the two "` | `"the frame differences of the point marking and their candidate causes for the two "` | val_table_marking_gap caption — viz/views/validation_part_i.py | 1 |
| `"the holds and the tip error it predicts over the calibration-tool lever (median over "` | `"the holds and the tip difference it predicts over the calibration-tool lever (median over "` | val_table_marking_gap caption — viz/views/validation_part_i.py | 1 |
| `"anatomical non-hinge motion; the JTS-MM minus Qualisys difference is what tracking " ⏎ "error adds",` | `"anatomical non-hinge motion; the JTS-MM minus Qualisys difference is what tracking " ⏎ "with JTS-MM adds",` | val_hinge caption — viz/views/validation_part_i.py | 1 |
| `"Qualisys (what tracking error adds), and the distance of the best-fit axis from the "` | `"Qualisys (what tracking with JTS-MM adds), and the distance of the best-fit axis from the "` | val_table_hinge caption — viz/views/validation_part_i.py | 1 |
| `Axis("Error", unit, log=True)` | `Axis("σ", unit, log=True)` | val_budget form dot x axis — viz/views/validation_r8.py | 1 |
| `[("grey", "observed error"), ("blue", "JTS-MM part"),` | `[("grey", "observed difference"), ("blue", "JTS-MM part"),` | val_budget form dot caption — viz/views/validation_r8.py | 1 |
| `"motion, the gap to JTS-MM is what tracking error adds",` | `"motion, the gap to JTS-MM is what its tracking adds",` | val_hinge form dumbbell caption — viz/views/validation_r8.py | 1 |
| `"what tracking error adds to the dataset's fixed-hinge residuals: the incisor RMS "` | `"what tracking with JTS-MM adds to the dataset's fixed-hinge residuals: the incisor RMS "` | val_hinge form difference caption — viz/views/validation_r8.py | 1 |
| `"per-frame difference of the jaw-in-head error (fused minus optical) over the "` | `"per-frame change of the jaw-in-head difference to Qualisys (fused minus optical) over the "` | val_fusion_segments form paired caption — viz/views/validation_r8.py | 1 |
| `label=("95 % CI of the error vector"` | `label=("95 % CI of the difference vector"` | val_marking_accuracy form ellipsoids3d legend — viz/views/validation_r8.py | 1 |
| `"point-marking error of JTS-MM against Qualisys at each landmark with its uncertainty, "` | `"point-marking difference of JTS-MM from Qualisys at each landmark with its uncertainty, "` | val_marking_accuracy form ellipsoids3d caption — viz/views/validation_r8.py | 1 |
| `("arrows", "error vector JTS-MM − Qualisys at true scale"),` | `("arrows", "difference vector JTS-MM − Qualisys at true scale"),` | val_marking_accuracy form ellipsoids3d caption — viz/views/validation_r8.py | 1 |
| `"95 % CI of the error along the anterior, left and "` | `"95 % CI of the difference along the anterior, left and "` | val_marking_accuracy form ellipsoids3d caption — viz/views/validation_r8.py | 1 |
| `"one error vector per landmark` | `"one difference vector per landmark` | val_marking_accuracy forms ellipsoids3d / arrows3d statistic — viz/views/validation_r8.py, viz/views/validation_gallery.py | 2 |
| `"Systems errors next to the dataset effects"` | `"System differences next to the dataset effects"` | val_errors_vs_effects title — viz/views/validation_r8.py | 2 |
| `Panel("Error and effect",` | `Panel("Margin and effect",` | val_errors_vs_effects panel (a) title — viz/views/validation_r8.py | 1 |
| `Panel("Effect / error",` | `Panel("Effect / margin",` | val_errors_vs_effects panel (b) title — viz/views/validation_r8.py | 1 |
| `"whether the JTS-MM errors are small next to the effects the companion data-set study "` | `"whether the JTS-MM − Qualisys differences are small next to the effects the companion data-set study "` | val_errors_vs_effects caption — viz/views/validation_r8.py | 1 |
| `("vertical line (b)", "1, effect as large as the error")` | `("vertical line (b)", "1, effect as large as the margin")` | val_errors_vs_effects caption — viz/views/validation_r8.py | 1 |
| `("err_rms_mm", "RMS error against Qualisys")` | `("err_rms_mm", "RMS difference to Qualisys")` | val_fusion_accuracy panel (a) title — viz/views/validation_r8.py | 1 |
| `f"Error content above {SMOOTH_HZ:g} Hz"` | `f"Difference content above {SMOOTH_HZ:g} Hz"` | val_fusion_accuracy panel (b) title — viz/views/validation_r8.py | 1 |
| `"the IMU fusion of the rotation and of the rotation and position; (a) RMS error, (b) "` | `"the IMU fusion of the rotation and of the rotation and position; (a) RMS difference, (b) "` | val_fusion_accuracy caption — viz/views/validation_r8.py | 1 |
| `f"the part of the error above {SMOOTH_HZ:g} Hz (frame-to-frame noise)",` | `f"the part of the difference above {SMOOTH_HZ:g} Hz (frame-to-frame noise)",` | val_fusion_accuracy caption — viz/views/validation_r8.py | 1 |
| `pb = Panel("Marking error per hold", Axis("Predicted tip error", MM),` | `pb = Panel("Marking difference per hold", Axis("Predicted tip difference", MM),` | val_marking_causes panel (b) title, x axis — viz/views/validation_r8.py | 1 |
| `Axis("Marking error", MM), [],` | `Axis("Marking difference", MM), [],` | val_marking_causes panel (b) y axis — viz/views/validation_r8.py | 1 |
| `"orientation disagreement and the tip error it predicts); (b) per marking hold the "` | `"orientation disagreement and the tip difference it predicts); (b) per marking hold the "` | val_marking_causes caption — viz/views/validation_r8.py | 1 |
| `"point error against the tip error the calibration-tool orientation disagreement "` | `"point difference against the tip difference the calibration-tool orientation disagreement "` | val_marking_causes caption — viz/views/validation_r8.py | 1 |
| `("diagonal (b)", "error equal to the prediction")` | `("diagonal (b)", "difference equal to the prediction")` | val_marking_causes caption — viz/views/validation_r8.py | 1 |
| `"Incisor paths in 3-D colored by the error"` | `"Incisor paths in 3-D colored by the difference"` | val_paths3d_error title — viz/views/validation_r8.py | 2 |
| `" (RMS error, agreement)",` | `" (bench: RMS error; in vivo: RMS difference)",` | val_table_related_work, JTS-MM row (kind) — viz/views/validation_r8.py | 1 |
| `Panel("(c) Error jitter above 6 Hz"` | `Panel("(c) Difference jitter above 6 Hz"` | val_fusion_smoothing panel (c) title — viz/views/validation_r7.py | 1 |
| `Panel("(b) RMS error per held-out task"` | `Panel("(b) RMS difference per held-out task"` | val_imu_accel panel (b) title — viz/views/validation_r7.py | 1 |
| `("markers (b)", "RMS error per variant, color = variant (legend), circle / square "` | `("markers (b)", "RMS difference per variant, color = variant (legend), circle / square "` | val_imu_accel caption — viz/views/validation_r7.py | 1 |
| `Column("rms_err_mm_s2", "Error", MS2, 0, "S")` | `Column("rms_err_mm_s2", "Difference", MS2, 0, "S")` | val_table_imu_accel column head — viz/views/validation_r7.py | 1 |
| `Column("no_lever_mm_s2", "Error, no lever arm", MS2, 0, "S")` | `Column("no_lever_mm_s2", "Difference, no lever arm", MS2, 0, "S")` | val_table_imu_accel column head — viz/views/validation_r7.py | 1 |
| `Column("own_mm_s2", "Error, own orientation", MS2, 0, "S")` | `Column("own_mm_s2", "Difference, own orientation", MS2, 0, "S")` | val_table_imu_accel column head — viz/views/validation_r7.py | 1 |
| `a.title = "Landmark distance error"` | `a.title = "Landmark difference \|e\|"` | val_marking_combined panel (a) title — viz/views/validation_condensed.py | 1 |
| `"observed error / JTS-MM part / "` | `"observed difference / JTS-MM part / "` | val_marking_combined (b), val_motion_agreement (c) caption — viz/views/validation_condensed.py | 2 |
| `("black / grey markers (c, d)", "frame error from the marked points (with its 95 % "` | `("black / grey markers (c, d)", "frame difference from the marked points (with its 95 % "` | val_marking_combined caption — viz/views/validation_condensed.py | 1 |
| `"frame offset found independently from the motion data; the errors along the "` | `"frame offset found independently from the motion data; the differences along the "` | val_marking_combined caption — viz/views/validation_condensed.py | 1 |
| `label="JTS-MM − Qualisys (error)"` | `label="JTS-MM − Qualisys (difference)"` | val_marking_accuracy form arrows3d legend — viz/views/validation_gallery.py | 1 |
| `"point-marking error of JTS-MM against Qualisys at each landmark, in the Qualisys "` | `"point-marking difference of JTS-MM from Qualisys at each landmark, in the Qualisys "` | val_marking_accuracy form arrows3d caption — viz/views/validation_gallery.py | 1 |
| `("arrows", "error vector JTS-MM − Qualisys at true scale (the components and their "` | `("arrows", "difference vector JTS-MM − Qualisys at true scale (the components and their "` | val_marking_accuracy form arrows3d caption — viz/views/validation_gallery.py | 1 |
| `"error on the dropped frames, frames aligned, at the shortest and the longest simulated "` | `"difference to Qualisys on the dropped frames, frames aligned, at the shortest and the longest simulated "` | val_fusion_dropout gallery form caption — viz/views/validation_gallery.py | 1 |
| `NI_OUTCOME = {"opening_abs_err_deg": "opening angle \|error\| (°)", ⏎ "incisor_err_mm": "incisor error (mm)"}` | `NI_OUTCOME = {"opening_abs_err_deg": "opening angle \|difference\| (°)", ⏎ "incisor_err_mm": "incisor difference (mm)"}` | val_table_camerafree outcome column — viz/views/camerafree.py | 1 |
| `"camera-free rungs against Qualisys: the paired per-frame difference of the absolute " ⏎ "error to the JTS-MM optics, pooled over the movement tasks (rung minus optics, in the " ⏎ "outcome's unit; negative = the rung is closer to Qualisys)",` | `"camera-free rungs against Qualisys: per frame the absolute difference to Qualisys, " ⏎ "paired against the JTS-MM optics and pooled over the movement tasks (rung minus optics, " ⏎ "in the outcome's unit; negative = the rung is closer to Qualisys)",` | val_table_camerafree caption — viz/views/camerafree.py | 1 |
| `"drift of the camera-free opening angle against Qualisys: the signed slope of the " ⏎ "opening error of the mountings alone against the time since calibration, and of the "` | `"drift of the camera-free opening angle against Qualisys: the signed slope of the " ⏎ "opening difference of the mountings alone against the time since calibration, and of the "` | val_table_camerafree_drift caption — viz/views/camerafree.py | 1 |
| `components: {label: "error components along the anatomical axes", short: "components"}` | `components: {label: "difference components along the anatomical axes", short: "components"}` | vocab form.components (val_marking_accuracy gallery) — etc/vocab.yaml | 1 |
| `arrows3d: {label: "3-D error arrows on the dental arch", short: "3-D arrows"}` | `arrows3d: {label: "3-D difference arrows on the dental arch", short: "3-D arrows"}` | vocab form.arrows3d (gallery) — etc/vocab.yaml | 1 |
| `ellipsoids3d: {label: "3-D error arrows with 95 % CI loops", short: "3-D CI loops"}` | `ellipsoids3d: {label: "3-D difference arrows with 95 % CI loops", short: "3-D CI loops"}` | vocab form.ellipsoids3d (gallery) — etc/vocab.yaml | 1 |
| `label: "Maxillary frame error"` | `label: "Maxillary frame difference"` | catalog frame_max_deg (val_table_marking, val_table_marking_gap, val_marking_causes) — etc/catalog.yaml | 1 |
| `label: "Mandibular frame error"` | `label: "Mandibular frame difference"` | catalog frame_mand_deg — etc/catalog.yaml | 1 |
| `label: "Tooth point error (RMS)"` | `label: "Tooth point difference (RMS)"` | catalog rms_teeth_mm — etc/catalog.yaml | 1 |
| `label: "Condyle point error (RMS)"` | `label: "Condyle point difference (RMS)"` | catalog rms_condyle_mm — etc/catalog.yaml | 1 |
| `label: "Error anterior, CI high"` | `label: "Difference anterior, CI high"` | catalog e_anterior_hi_mm — etc/catalog.yaml | 1 |
| `label: "Error anterior, CI low"` | `label: "Difference anterior, CI low"` | catalog e_anterior_lo_mm — etc/catalog.yaml | 1 |
| `label: "Error anterior"` | `label: "Difference anterior"` | catalog e_anterior_mm — etc/catalog.yaml | 1 |
| `label: "Error left, CI high"` | `label: "Difference left, CI high"` | catalog e_left_hi_mm — etc/catalog.yaml | 1 |
| `label: "Error left, CI low"` | `label: "Difference left, CI low"` | catalog e_left_lo_mm — etc/catalog.yaml | 1 |
| `label: "Error left"` | `label: "Difference left"` | catalog e_left_mm — etc/catalog.yaml | 1 |
| `label: "Error superior, CI high"` | `label: "Difference superior, CI high"` | catalog e_superior_hi_mm — etc/catalog.yaml | 1 |
| `label: "Error superior, CI low"` | `label: "Difference superior, CI low"` | catalog e_superior_lo_mm — etc/catalog.yaml | 1 |
| `label: "Error superior"` | `label: "Difference superior"` | catalog e_superior_mm — etc/catalog.yaml | 1 |
| `label: "Landmark error, CI high"` | `label: "Landmark difference, CI high"` | catalog e_hi_mm — etc/catalog.yaml | 1 |
| `label: "Landmark error, CI low"` | `label: "Landmark difference, CI low"` | catalog e_lo_mm — etc/catalog.yaml | 1 |
| `label: "Landmark error"` | `label: "Landmark difference"` | catalog e_mm — etc/catalog.yaml | 1 |
| `label: "Error vs speed slope"` | `label: "Difference vs speed slope"` | catalog err_vs_speed_mm_per_mps — etc/catalog.yaml | 1 |
| `label: "Position error p95"` | `label: "Position difference p95"` | catalog pos_p95_mm — etc/catalog.yaml | 1 |
| `label: "Jitter error"` | `label: "Jitter of the difference"` | catalog jitter_err_rms_mm — etc/catalog.yaml | 1 |
| `label: "Incisor RMS error"` | `label: "Incisor RMS difference"` | catalog err_rms_mm — etc/catalog.yaml | 1 |
| `label: "Acceleration error"` | `label: "Acceleration difference"` | catalog rms_err_mm_s2 — etc/catalog.yaml | 1 |
| `label: "Predicted tip error, maxillary holds"` | `label: "Predicted tip difference, maxillary holds"` | catalog ct_tip_maxillary_mm (val_table_marking_gap, val_marking_causes) — etc/catalog.yaml | 1 |
| `label: "Predicted tip error, mandibular holds"` | `label: "Predicted tip difference, mandibular holds"` | catalog ct_tip_mandibular_mm — etc/catalog.yaml | 1 |
| `label: "Smaller frame error"` | `label: "Smaller frame difference"` | catalog pid_smaller, value_smaller — etc/catalog.yaml | 2 |
| `label: "Larger frame error"` | `label: "Larger frame difference"` | catalog pid_larger, value_larger — etc/catalog.yaml | 2 |
| `title: "Point-marking errors", caption: "Point-marking error per landmark: JTS-MM − Qualisys in the Qualisys anatomical axes with 95 % CIs, the reference uncertainty and the adequacy ratio."` | `title: "Point-marking differences", caption: "Point-marking difference per landmark: JTS-MM − Qualisys in the Qualisys anatomical axes with 95 % CIs, the Qualisys uncertainty and the adequacy ratio."` | tables ref_landmarks:landmark_errors — etc/tables.yaml | 1 |
| `title: "Anatomical-frame errors", caption: "Anatomical-frame angle and origin errors from the point marking` | `title: "Anatomical-frame differences", caption: "Anatomical-frame angle and origin differences from the point marking` | tables ref_landmarks:frame_errors — etc/tables.yaml | 1 |
| `caption: "Per-hold covariates of the marking error (descriptive)."` | `caption: "Per-hold covariates of the marking difference (descriptive)."` | tables ref_landmarks:hold_attribution — etc/tables.yaml | 1 |
| `title: "Error attribution", caption: "Spearman` | `title: "Disagreement attribution", caption: "Spearman` | tables ref_kinematics:attribution — etc/tables.yaml | 1 |
| `incisor RMS residual of both hinge models: the part tracking error adds to the residual."` | `incisor RMS residual of both hinge models: the part tracking with JTS-MM adds to the residual."` | tables ref_hinge:hinge_diff — etc/tables.yaml | 1 |
| `(marking: registered point error of the teeth and condyle points;` | `(marking: registered point difference of the teeth and condyle points;` | tables ref_budget:budget — etc/tables.yaml | 1 |
| `title: "Marking error per hold and its candidate causes", caption: "Per participant and marking hold (VD22): the registered point error against Qualisys, the CT orientation disagreement of the two systems and the tip error it predicts` | `title: "Marking difference per hold and its candidate causes", caption: "Per participant and marking hold (VD22): the registered point difference to Qualisys, the CT orientation disagreement of the two systems and the tip difference it predicts` | tables ref_budget:marking_holds — etc/tables.yaml | 1 |
| `title: "Marking frame errors and their candidate causes", caption: "Per participant (VD22): the frame errors of the marking next to the CT orientation disagreement during the holds and the tip error it predicts` | `title: "Marking frame differences and their candidate causes", caption: "Per participant (VD22): the frame differences of the marking next to the CT orientation disagreement during the holds and the tip difference it predicts` | tables ref_budget:marking_causes — etc/tables.yaml | 1 |
| `the reference case (smallest maxillary frame error) and the stress case` | `the reference case (smallest maxillary frame difference) and the stress case` | tables ref_budget:marking_gap — etc/tables.yaml | 1 |
| `mean paired per-frame difference of the absolute error to F0 (opening angle, incisor)` | `mean paired per-frame difference of the absolute difference to Qualisys, rung minus F0 (opening angle, incisor)` | tables ref_camerafree:cf_val_noninferiority — etc/tables.yaml | 1 |
| `slope of the signed opening error against Qualisys` | `slope of the signed opening difference to Qualisys` | tables ref_camerafree:cf_val_drift — etc/tables.yaml | 1 |
| `title: "IMU dynamic calibration: error-trace steps"` | `title: "IMU dynamic calibration: difference-trace steps"` | tables ref_diagnostics:dyncal_summary — etc/tables.yaml | 1 |
| `steps of the IMU-alone rotation error against Qualisys (0.25 s` | `steps of the IMU-alone rotation difference to Qualisys (0.25 s` | tables ref_diagnostics:dyncal_summary — etc/tables.yaml | 1 |
| `the change of the IMU-alone error over the period."` | `the change of the IMU-alone difference to Qualisys over the period."` | tables ref_diagnostics:dyncal_periods — etc/tables.yaml | 1 |
| `title: "Error jumps across phase boundaries", caption: "Per participant and pair of consecutive phases (VD24): jump of the IMU-alone rotation error across` | `title: "Difference jumps across phase boundaries", caption: "Per participant and pair of consecutive phases (VD24): jump of the IMU-alone rotation difference across` | tables ref_diagnostics:dyncal_transitions — etc/tables.yaml | 1 |
| `residual lag and the error per speed."` | `residual lag and the difference per speed."` | tables ref_sync:heldout — etc/tables.yaml | 1 |

## (d) kept occurrences (with reason)

### Still rendered in the systems export (export after the pass, every hit classified)

| phrase kept | where | reason |
|---|---|---|
| `step error`, `mean step error`, axis `Error (mm)`, `0, no error`, `Trueness (mean step error)` | val_bench (both forms), val_trueness, val_table_ladder, val_bench--form-merged | bench ladder: true error against the hand-measured nominal steps |
| `scale error` (column `Scale error`) | val_table_ladder; tables.yaml ref_trueness:scale | bench: free-scale fit against the ruler (fixed metric name) |
| `standard error`, `HAC standard error(s)`, `Offset SE` | val_setup, val_sync_combined, val_sync_lag, val_table_methods | statistical standard error |
| `it is a frame diagnostic, not a timing error` | val_table_methods notes | clock/sync model error term |
| `error budget` (titles `Error budget per participant`, `Error budget of the points`, `Error budget, motion`, `Error budget and Qualisys adequacy`; captions `(b) The error budget …`, `(c) The error budget …`) | val_budget, val_table_budget, val_error_budget, val_marking_combined, val_motion_agreement | BORDERLINE: established metrology term and the name of VD22; its terms were renamed (observed difference, Qualisys part σ_Q, JTS-MM part) |
| `root-mean-square error (RMSE)`; axis `RMSE vs Qualisys` | val_table_fitness, val_camerafree (+ its forms) | BORDERLINE: fixed metric name/abbreviation (glossary RMSE); the free-text `RMS error against Qualisys` was renamed elsewhere |
| `max \|error\|`, `largest relative error of a single capture`, `largest errors` | val_table_emg_bench | sEMG front end on the bench against the known test input: true error |
| `normalization reference` | val_emg_mvc caption | EMG normalization reference (D11) |
| `(RMS error)`, `worst error below 0.5 mm`, `(no kinematic reference)` | val_table_related_work, val_table_related_work_other (literature rows) | the literature's own metric names / wording, from related_work.yaml (not ours); the JTS-MM row now reads `(bench: RMS error; in vivo: RMS difference)` |
| `\|e\|` symbol | val_marking_accuracy, val_marking_combined, val_tip_holds | BORDERLINE: symbol kept (e = landmark difference vector), only the words around it renamed |
| pgf style `jtsmm/role/reference`, `error bars` / `y error plus` options | every fig.tex | code identifiers (plot role, pgfplots syntax), not text |
| item ids / file names `val_error_budget`, `val_errors_vs_effects`, `val_paths3d_error`, `val_kin_error_ecdf`, labels `valErrorBudget`, `valErrorsVsEffects`, `valPathsThreedError`, `valKinErrorEcdf` | export ids, \label keys | identifiers; also macro names `numRef…`, `numBudgetRef…`, columns `sigma_ref`, `ref_var_share`, `err_rms_mm`, … unchanged |

### Kept in source (not Qualisys/Delsys, or not rendered)

| phrase kept | file | reason |
|---|---|---|
| `normalization reference`, `reference contraction`, `Reference envelope`, `Reference (p95)`, `Normalization reference`, `Reference source`, `95th percentile of the reference`, `RVC: reference voluntary contraction` | catalog.yaml (ref_p95_uv, ref_uv, ref_source, p95_uv, max1s_uv, ref_mask_frac …), vocab.yaml emg_norm, glossary.yaml, tables.yaml emg:refs / ref_emg:mvc, validation_emg.py, captions_jner.yaml (activation, tabActivation) | EMG normalization reference / reference contraction |
| `Clock reference time` (t_ref_s), `Clock rate error` (alpha_ppm) | catalog.yaml | clock model reference / error term |
| `Displacement reference`, `cycle-start reference`, `closed-pose reference` | catalog.yaml `reference`, vocab.yaml | reference pose of the kinematic displacement |
| `reference chain`, `Reference definition`, `Reference phase` | catalog.yaml ref_chain/ref_mode/ref_phase, vocab ref_chain | decoder / EMG reference chain |
| `Time since reference`, `Drift since the reference`, `one reference closed pose at rest`, `time since the rest-phase reference` | catalog.yaml t_since_ref_s, drift_slope_ref_dpm; camerafree.py cf_drift; tables.yaml camerafree:* | rest-phase reference of the camera-free calibration |
| `offset from the reference tip` | tables.yaml ref_tip:tip_estimates | reference tip estimate of the CT tip check |
| `the reference case (smallest maxillary frame difference)` | tables.yaml ref_budget:marking_gap | baseline participant, not Qualisys ("frame error" → "frame difference" changed) |
| `(the reference of the dataset paper's IMU result)` | tables.yaml ref_fusion:imu_opening | the JTS-MM optics as the dataset's reference for the IMU; title changed to `… against Qualisys and the optics`, `range of the reference angle` → `Qualisys angle` |
| `optics (reference)` (E0), camera-free `error against the optics`, `translation-model error`, `rotation-induced error`, `Incisor error`, `Opening angle \|error\|`, `Opening error`, `inner-CV error`, `Error component`, `Error trace` | vocab.yaml (E0, cf_component), catalog.yaml (incisor_err_mm, opening_abs_err_deg, err_open_deg, max_*_deg, e4own_cv_rmse_mm, cv_rmse_*, component, trace), camerafree.py cf_* views, tables.yaml camerafree*/imu_dyncal* | dataset camera-free analyses against the JTS-MM optics (not a commercial system); the systems table val_table_camerafree uses its own outcome names (renamed) |
| `Error change over the period` (err_change_deg) | catalog.yaml | BORDERLINE: shared by the dataset dyncal audit (vs optics) and ref_diagnostics (vs Qualisys); label kept, the validation tables.yaml captions renamed |
| `scatter against the reference with the identity line` | vocab.yaml form.scatter | ICORR IMU-vs-optics form (gallery label) |
| `reference_subset`, `no_reference_data`, `Reference validation status`, `Whether reference-system data exist`, `Reference-system validation (Qualisys/Delsys subset)` | vocab.yaml, tables.yaml validation:status, views/results_qc.py qc_validation | BORDERLINE: dataset QC dashboard only, not in the JNER or systems export; left unchanged |
| `tabReference`, `tabReferenceLiterature`, `Literature values depend on the condylar reference point`, literature `reference` | captions_jner.yaml, JNER views | reference values / literature references |
| `ref_budget: error budget` | vocab.yaml stage names | error budget (see above) |
| `RMSE: root-mean-square error`, `SEM: standard error of measurement` | glossary.yaml | fixed metric names |
| `frame of reference`, `\ref`, `references` (literature), `reference implementation`, `reference electrode`, `reference slot` | various | not the commercial systems (none of these needed a change) |
| docstrings / comments with `reference` / `error` (e.g. `Qualisys = reference` in validation.py val_marking_accuracy docstring, validation_part_i.py module docstring, validation_r7.py docstrings, validation_both.py module docstring) | viz/views/*.py | not rendered; σ_ref in docstrings was renamed by the generic σ_Q pass |
| `desc:` texts of catalog entries (e.g. `Monte Carlo RMS error of the Qualisys incisor`, `200-Hz reference frames`) | catalog.yaml | not rendered in the exports (schema / data-dictionary text); labels and symbols changed |
| `text:` of numbers_*.yaml macros | experiments/*/numbers_*.yaml | not rendered (decision of the authors) |

### ICORR conference export

No string used by the ICORR conference export was changed (grep of `data/analysis/icorr2027/paper_dataset/conference` for every changed phrase: no hit). The JNER export changes: `suppLowpassBound` caption (`the 200 Hz reference comparison of the companion validation study` → `the comparison with Qualisys at 200 Hz in the companion validation study`); the `with_chewing` form of `table_compatibility` (not exported, form `own`) renamed for consistency. captions_jner.yaml names neither Qualisys nor Delsys and was left unchanged.

### Test change

`software/biomech/tests/test_r8_validation_views.py::test_budget_stacked_draws_the_stored_reference_share` picked the series by `"reference" in s.label`; now `"Qualisys part" in s.label`.
