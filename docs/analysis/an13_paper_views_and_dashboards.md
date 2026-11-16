<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# an13 — Paper figures, tables and dashboards

What every paper item and dashboard shows, which stored tables it reads, and the exact rule
behind each number. Code: `software/biomech/biomech/viz/views/paper.py` (paper),
`dashboards.py` (dashboards); the data atlas is in an10. Every view is a registered spec
(an08 §4), so the viewer, `biomech export` and `check-outputs` render the same thing.

**Paper sets** (`biomech/papers.py`; one YAML file per paper next to the study file):

- **Keys.** `snapshot` (the frozen store every number comes from), `variant` (optional: the
  reporting population), `numbers` (the in-text numbers file), `targets` (venue
  profile + output folder each) and `default_target`, then `items`: `id` (view), `label`
  (LaTeX name, letters only), `part` (core / supporting / supplement), `in` (targets),
  `target_params` (per-target view parameters), every other key a view parameter; an entry
  `- {section: …}` starts a storyboard section. `status: legacy` keeps a set as history.
- **Reporting population (D35).** `variant: reporting_population` makes every consumer — `export`,
  `numbers`, `verify-export`, `check-outputs`, `check-paper` and the viewer's gallery — read
  the `reporting_population` store (13 adults) instead of the primary store (14, the analysis of
  record). `snapshot` then names a snapshot *of that store* (`biomech snapshot --variant
  reporting_population <name>`). Both are resolved in one place (`PaperSet.store_config` →
  `sensitivity.store_config`; `papers.paper_context`). The n = 14 values enter only through
  the robustness items ("including the participant recorded outside the protocol").
- **Dataset paper** (`experiments/icorr2027/paper_dataset.yaml`; JNER; targets `journal`,
  Springer profile, default, and `conference`, IEEE profile; `variant: reporting_population`). Three
  blocks, then robustness and the supplement; every earlier item is kept
  (sections renamed, items moved; the forms are picked in the gallery):
  - **A — Requirements:** A1 setup and data (`icorr_setup`, `table_methods`); A2 requirements
    (`jnl_requirements`, `table_requirements`; E31: P95, lower bound, maximum); A3 not a hinge
    at the TMJ (`icorr_hinge`, `kin_hinge_axis_location`; C1 led by the condyle-hinge null,
    D33; the Qualisys check VD21 cited); A4 geometry and reference values (`table_reference`,
    `table_geometry`).
  - **B — Sensing:** B1 the IMU recovers opening (`icorr_imu`, C3a); B2 camera-free rotation
    (`cf_opening_ladder`, `cf_table_results`, supplement `cf_decomposition`,
    `cf_table_oracle`; X5 path 2, D28; VD20 cited); B3 muscle activation and signal quality
    (`emg_landscape`, `table_emg_activation`, `activation_map`); B4 electrode impedance
    (`emg_impedance`, `table_impedance`).
  - **C — Intent:** C1 EMG and motion timing (`c2e_sequence`, `c2e_table`: the X6 event order,
    post hoc and gated, §6; `jnl_c2`: C2 not confirmed, with the window lesson, D34;
    supplement `c2e_locking`); C2 closer
    release (`jnl_x3`: detection above chance, not anticipation); C3 intent decoding
    (`jnl_decoding`, `jnl_table_decoding`); C4 online decoding (`jnl_online`).
  - **Robustness and variability** (`jnl_robustness`, `jnl_table_robustness`; the n = 14
    values) and the **supplement**, ordered by block.
- **Systems paper** (`experiments/jtsmm_validation/paper_systems.yaml`; IEEE Sensors J;
  validation views only). Metrology order: bench → synchronisation → (I) point
  marking → (II) motion with fusion, IMU and the error budget (VD22) → fitness for purpose
  (VD21 hinge, VD20 camera-free opening, and the dataset bridges VD9 / VD10 / VD12) → (III) EMG
  with the side check (VD23) → supplement. n = 2 and the reference limits named upfront; per
  participant plus the mean everywhere (EV8). Views: an14 §12.
- **Pins (2026-11-16).** Dataset `r17a` (the `reporting_population` store of the re-run under the
  released code, change-log entry J36, values equal to `r16a`; 13 adults; analysis of record `r17`,
  n = 14) and systems `v15` (also pinned
  in the dataset `cite:` block). `--snapshot live` reads the live store.
- The former sets `paper_icorr.yaml` and `paper_journal.yaml` are kept as history
  (`status: legacy`); §4b and §5 describe their item lists, which the dataset set reuses.
- The item views come from `paper_icorr.py` (setup, hinge, IMU), `paper_journal.py`
  (`jnl_*`), `paper.py` (tables), `paper_dataset.py` (the methods table, activation landscape,
  impedance, geometry, axis location, activation table), `camerafree.py` (`cf_*`) and
  `paper_c2e.py` (`c2e_*`, §6).
- Panel letters ("(a)" in the tables below) are drawn as bold tags (an08 §4); exported captions
  use reader wording instead of the plan codes quoted here (D6 -> post hoc, X2 -> exploratory,
  C1 -> pre-specified).

Conventions: "participant value" = the participant-level number from the stats stages
(cycles are never pooled across participants); dots are participants, a diamond is the
median, grey is context.

**Uncertainty conventions (2026-10-22).**
- **Cohort curves:** the SD/IQR band (light) plus a pointwise bootstrap 95 % CI of the
  centre curve (darker; 2000 resamples of participants, seed = `config.seed`;
  `blocks.add_curves_ci`).
- **Per-participant summaries:** strips of all participants with the median and its BCa
  95 % CI (`blocks.strip_ci`; the same routine and seed as the stats stages,
  `stats.core.describe`).
- **CMC values:** carry cycle-bootstrap CIs.
- **Variance-ratio and CMC CIs:** drawn only when the view reproduces the stored value
  exactly, so a stale store never gets a mismatched band.
- **Captions:** take their numbers and method settings from the stored results or the config.
- **Layout:** figure heights scale with n (`blocks.rows_height`).
- **Validation study:** its views are in an14 §12.

## 1. Figures

### Fig. 1 `fig_kinematics` — "Jaw opening is not a hinge" (C1)

| Panel | Content | Source and rule |
|---|---|---|
| (a) Incisal opening path | sagittal (x–z) path of the incisor relative to the cycle start, time-normalized (`align=path`, movement onset..offset); grey: participant means, accent: cohort centre (`cohort` perspective) or the cycles and the participant mean (`participant`/`trials`) | `ensembles.kinematic(…, "incisor_disp", "path")`, `ensembles.aggregate` (an10) |
| (b) Helical-axis migration | per participant an arrow from the median FHA piercing point of the early steps (θ_to ≤ 10°) to that of the later steps (θ_to > 10°), x and z relative to the condyle midpoint in the closed pose | `kinematics.fha` (an02 §5), `kinematics.landmarks`, `T_closed` |
| (c) Misalignment at the incisor | RMS incisal residual of `hinge_tmj`, `hinge_fit`, `rot_trans` (log axis); grey band: range over participants of the in-vivo noise-only P99 (`null_invivo_p99_mm`) | `kinematics.hinge` (an02 §6), `uncertainty.hinge_noise` (an06) |
| (d) Condylar translation with opening | anterior displacement of the condyle midpoint vs opening angle (opening half, 1° grid, `align=angle`), relative to its first grid value; summary per the perspective | `ensembles.kinematic(…, "tmj_left"/"tmj_right", "angle")` |

### Fig. 2 `fig_emg_timing` — "Muscle activity and motion onset" (C2; D6, X2)

| Panel | Content | Source and rule |
|---|---|---|
| (a) Opening angle | θ over the open-close cycle (0–100 %), cut at cycle start..end — the same cycles as (b) and the SPM test | `ensembles.profile_ensembles` (the angle stored with the EMG profiles) |
| (b) Opener and closer activation | per cycle, mean of the shape-usable channels of each group (digastrics; masseters + temporales), divided by the cycle's own maximum (peak = 1); summary band per the perspective; grey bars: SPM supra-threshold clusters of the confirmatory openers-vs-closers test | `emg` profiles (an03), `stats_emg.spm` (an07) |
| (c) EMG event before motion onset | rows, each a strip over participants (> 0: EMG first): *first onset* and *driving burst* = per-participant single-cycle digastric lead (TKEO) under the frozen and the D6 rule; *opener rise* and *closer release* = ensemble events (X2, exploratory, `lead_ms = −t`); *causal release* = per-participant median anticipation of the causal closer-release detector (X3, exploratory, primary setting); *causal margin* = `anticipation_ms` under the primary rule | `timing.summary` (rule column), `timing.ensemble`, `timing.release_summary` (an05 §3, §3.1, §3.2) |

### Fig. 3 `fig_wearable` — "Wearable sensing recovers opening and intent" (C3)

| Panel | Content | Source and rule |
|---|---|---|
| (a) IMU vs optical opening | Bland–Altman per open-close cycle: x = (IMU + optical)/2, y = IMU − optical, IMU opening from the participant's own quaternion model; lines: bias and repeated-measures LoA (D7) | `stats_imu.imu_opening_cycles`, `imu_opening_ba` (an04, an07) |
| (b) Intent decoding | balanced accuracy of the 7-class SVM per feature set: unseen subject (LOSO) vs same subject (blocked CV, D4), per-participant dots, medians joined; chance = 1/7 | `ml_eval.loso`, `ml_eval.blocked` (an05 §5) |
| (c) Confusion, unseen subject | row-normalized confusion (%) of the SVM with EMG + IMU features, summed over LOSO folds | `ml_eval` array `confusion__emg_imu\|svm` |

### Fig. 4 `fig_variability` — "Variability and processing sensitivity" (S2)

| Panel | Content | Source and rule |
|---|---|---|
| (a) Processing spread / between-subject SD | `spread_over_sd` of the specification curve for the outcomes listed in `SPEC_NAMES` (largest opening, opening/closing/chewing lead, digastric and masseter activation), sorted; > 1: the processing choice matters more than who was measured | `variability.spec_curve` (an07) |
| (b) Between / within-subject variance | pointwise variance ratio along the cycle for the opening angle and masseter R in chewing (log axis; 1 = equal) | `stats_emg` arrays `varratio__*__ratio` (an07) |

## 2. Tables

### Table I `table_reference` — reference values

One row per metric in `REF_ROWS` (self-paced opening, largest observed opening, opening angle,
protrusion, laterotrusion L and R, opening deviation, condylar path L, peak opening velocity,
chewing frequency; definitions in an02 §4):

| Column | Rule |
|---|---|
| Mean, SD, 95 % CI | `stats_kinematics.reference_values`, group `all`: mean and SD (ddof 1) of the participant values; BCa bootstrap CI of the mean |
| F, M | the same mean per sex |
| g | Hedges' g (F − M), exploratory (an07) |
| ICC, MDC | from `stats_kinematics.reliability`, matched by `REL_KEY` (task, metric, side): ICC(3,1) and MDC95 across the first 8 cycles; laterotrusion per side |

### Table II `table_requirements` — derived requirements for a jaw exoskeleton

Formatted from the `requirements` stage (`stats/requirements.py`, an07): one row per
requirement, grouped (workspace, kinematics, dynamics, intent detection, wearable sensing),
columns *Requirement*, *Value [95 % CI]*, *Statistic*, *Derivation* (plan codes stripped,
notes such as the C1 count above the in-vivo noise or the Holm p appended) and *Evidence*
(pre-specified test / pre-specified description / exploratory / post hoc).
- **Capacity rows** (E31): the Harrell–Davis P95 across participants with its exact
  one-sided lower 95 % bound (an order statistic) and the sample maximum; no finite upper
  bound exists at this n, which the table says ("P95 (lower 95 % bound; maximum)").
- **Performance rows:** cohort medians with BCa CIs; both C2 onset rules with the D15 minimum
  of 3 usable channel-cycles and their chance-null columns (the window bias, D34); the IMU
  opening agreement as the largest absolute repeated-measures limit of agreement with its
  MOVER interval (D26).
- **Post hoc rows** (C2-E, X6): closer activation before closing and the null-calibrated
  closer release before opening, labelled post hoc.
- **Decoding rows** (unseen user: BA with its CI; after a 5-s-per-movement calibration: median
  and IQR) come from `ml_eval.summary` / `variants_summary`.
- Without the `requirements` stage the table says so in one row. Option `scope: conference`
  shows the shorter conference table (`target_params` of the paper set).

## 3. Dashboards (viewer *Results* page, exportable like paper items)

| Id | Shows | Source |
|---|---|---|
| `landmark_qc` | hold MAD of the 8 landmarks per participant (heatmap) and condyle–incisor distances against the 70 mm plausibility limit | `kinematics.landmarks`, `anatomy` |
| `qc_emg` | heatmaps participant × muscle: SNR (reference vs rest, dB) and masked task samples (%) | `qc.emg` (an01) |
| `activation_map` | task × muscle heatmap of the cohort median activation (% of max recorded, D11) | `stats_emg.activation_cohort` |
| `imu_agreement` | per estimator (quaternion, complementary, tilt) a strip per task of the opening-angle RMSE vs optical (log axis) | `imu.agreement` (an04) |
| `timing_detectors` | opening lead per detector × mechanical source × onset rule (36 variants: first, driving burst, driving burst with in-task threshold), per-participant dots | `stats_timing.timing_spec` (an05, an07) |
| `ml_overview` | unseen-subject balanced accuracy per model × feature set, chance line | `ml_eval.loso` |
| `emg_spectra` | normalized log PSD per muscle: participants grey, cohort mean ± SD, SPM bars along frequency (X1) | `emg.spectral`, `stats_emg.spectral_spm` (an03 §5) |
| `spectral_variability` | mean frequency per participant and muscle, and between- vs within-participant SD of MNF (X1) | `emg.spectral`, `stats_emg.spectral_cohort` |
| `movement_bandwidth` | per task, per-participant raw f95 of the IMU angular-velocity power (dots) and the median optical-confirmed f95 over non-noise-limited participants (square) (X1, D12) | `imu.bandwidth` (an04 §6) |
| `timing_ensemble` | per phase (open-close, chewing), opening and closing: per-participant median envelopes of agonists and antagonists around the mechanical onset (dB re −600…−300 ms), median of participants; event medians in the panel note (X2) | `timing.ensemble` + arrays (an05 §3.1) |

`emg_spectra` and `spectral_variability` (chewing, open-close, MVC closing) and
`timing_ensemble` (open-close, chewing) take a `phase` parameter; the rest are cohort-level.

## 4. Result views by domain (Results page)

Every stored analysis output is shown by at least one view; `tests/test_view_coverage.py`
builds all views (every declared phase, perspective and picker value) on the synthetic study
and fails on a stored table or array family that no view reads (inputs of other stages are
listed with their reason in `check/coverage.py:INTERNAL`). Modules are discovered
automatically (an08 §7); the ids below are the export names.

| Domain (Results section) | Module | Views (pickers / phases in brackets) |
|---|---|---|
| Paper | `paper.py`, `paper_icorr.py`, `paper_journal.py`, `paper_dataset.py`, `paper_c2e.py` | `fig_kinematics`, `fig_emg_timing` (perspective, stat, pid), `fig_wearable`, `fig_variability`, `table_reference`, `table_requirements` (§1, §2); `icorr_setup` (pid), `icorr_hinge`, `icorr_imu`; the journal set `jnl_*` (§5); `table_methods`, `emg_landscape` (form), `table_impedance`, `table_geometry`, `table_emg_activation`; `c2e_sequence` (form), `c2e_locking` (form), `c2e_table` (§6) |
| Kinematics | `results_kinematics.py`, `frequency.py` | `kin_summary` (metric), `kin_anatomy` (column) + `_table`, `kin_opening_model`, `kin_coupling` (pid), `kin_coupling_cohort`, `kin_hinge_frames` (pid), `kin_theta_raw` (pid, phase), `kin_fha_cohort`, `kin_condyle_sensitivity`, `kin_norms` + `_table`, `kin_influence`, `kin_subjects` (metric), `kin_reliability`, `kin_hinge_summary`, `chewing_rate` |
| EMG — time domain | `results_emg.py`, dashboards | `emg_activation` (phase), `emg_cycles` (phase: per-cycle means, profiles, CCI), `emg_refs` + `_table` (D11 reference provenance), `emg_mask_sensitivity` (phase: cycles kept and activation under the D14 rules vs with the low-frequency rule), `emg_envelope` (pid, phase, units `uv` = µV / `norm` = % of max recorded), `emg_indices`, `emg_rest_activity`, `emg_cci`, `emg_activation_lmm`, `emg_spm` (test) + `emg_spm_table`, `emg_varratio`; dashboard `activation_map` |
| EMG — frequency domain | `frequency.py`, dashboards | `emg_psd` (pid, phase: stored Welch PSD per muscle, mains harmonics marked), `emg_spectral_descriptors` (phase: MNF/MDF strips, band-share heatmap), `emg_spectral_spm` (L-vs-R SPM{t} along frequency, participants ANOVA), `emg_spectral_cohort` (+ `_table`), `emg_mask_validation` (low-frequency-flagged vs clean windows, D14), `emg_mains` (line frequency, mains share raw → clean, rest RMS); dashboards `emg_spectra` (phase), `spectral_variability` (phase) |
| Camera-free (IMU domain) | `camerafree.py` | `cf_ladder` (camera use per rung), `cf_opening_ladder`, `cf_drift` (pid), `cf_incisor_axes`, `cf_traces` (pid), `cf_table_results`, `cf_decomposition`, `cf_table_oracle`, `cf_oracle_tasks` (estimator) (§6) |
| IMU | `frequency.py`, `results_qc.py`, dashboards | `imu_spectra` (pid, phase: angular-velocity PSD with f50/f95/f99/f95_coh, coherence C′ with the 0.5 threshold and the coherent band), `imu_calibration`, `imu_cycles` (task), `imu_quasi_static` (pid, phase); dashboards `movement_bandwidth` (raw f50/f95/f99, optical-confirmed f95, coherent band, noise-limited marked), `imu_agreement` |
| Timing and intent cues | `results_timing.py`, `results_online.py`, `paper_c2e.py`, dashboards | `c2e_cycles` (phase, lead source), `c2e_release` (phase) (§6); `timing_onsets` (phase, detector, rule), `timing_release` (phase: X3 per cycle), `timing_release_sweep` (phase: X3 drop × hold), `timing_usability`, `timing_cohort` (phase) + `timing_cohort_table` (phase), `online_release` (X3 on the causal stream vs offline, per participant; X4, ml08); dashboards `timing_detectors`, `timing_ensemble` (phase) |
| Uncertainty | `results_kinematics.py` | `unc_null` (pid: H0 histograms with the observed residual), `unc_ci` (bootstrap CIs), `unc_hinge_noise` |
| Variability and sensitivity | `results_kinematics.py`, `results_sensitivity.py` | `var_spec_values` (outcome), `var_cycles_needed`, `var_protocol`, `var_spec_curve`; `sensitivity_changes` (variant, table, column: the up to 25 values with the largest relative change), `sensitivity_table` (variant, table: every compared value, first 400) — both read `sensitivity/<variant>/comparison.parquet` (an08 §1) and show pseudonyms only |
| Quality control | `results_qc.py`, `results_pipeline.py`, `paper_dataset.py`, dashboards | `pipeline_coverage` (participants per cohort statistic), `emg_opener_detectability`, `emg_impedance`, `kin_hinge_axis_location`; `qc_heatmap_emg\|imu\|optical\|phases\|sessions` (column), `qc_overview`, `qc_emg_flags`, `qc_demographics` (aggregate only), `qc_validation`; dashboards `landmark_qc`, `qc_emg` |
| Intent decoding | `results_ml.py`, `results_qc.py`, `results_online.py`, dashboards | `ml_confusion` (config), `ml_timeline` (pid, fset: true vs decoded class over time), `ml_summary` (metric), `ml_permutation`, `ml_variants` (few-shot, late fusion, reduced EMG), `ml_fusion`, `ml_tests`, `ml_generalization`, `ml_windows_table`, `labels_timeline` (pid, phase), `labels_counts` (phase), `labels_rule_agreement`; simulated real time (X4, ml08): `online_accuracy` (stream BA, latency, false activations per feature set × calibration mode), `online_timeline` (pid), `online_compute` (time per 20-ms chunk, real-time factor), `online_feature_deviation` (stream vs offline window features); dashboard `ml_overview` |
| Recorded data | `data.py` | the 22 `data_*` views of the data atlas (an10) |
| Result tables | `tables.py` | one complete table view per stored stage table declared in `etc/tables.yaml` (`tab_<stage>__<table>`: every column, reader headers and cells, redaction; an12 *Completeness*), e.g. `tab_labels_check__kappa` for the label validity check (an05 §4.1), which no other view draws |

Phases in brackets are expanded by export and check-outputs over the view's declared
`phases`; pickers (`choices`) over every value (`registry.expand_phases`, an08 §7).

Labels and units come from `etc/catalog.yaml`; columns without an entry get a label from
their words and the unit from their suffix (`catalog.guess`: `_mm`, `_deg`, `_hz`, `_ms`,
`_uv`, `_dps`, `_pct`, …).

## 4b. ICORR set (IEEE profile; `paper_icorr.py`)

`paper_icorr.yaml` (requirements + C1 + C3a, 6 pages):

| Item | View | Shows |
|---|---|---|
| Fig. 1 | `icorr_setup` (pid; default: the participant with all EMG channels usable whose self-paced opening is closest to the cohort median) | (a) protocol timeline of the motion take; (b–d) the first 8 s of open-close: optical and IMU opening angle, opener and closer activation (entity colours), intent label. The setup photo is placed in LaTeX. |
| Fig. 2 | `icorr_hinge` | (a, b) incisal path and helical-axis migration (as `fig_kinematics`); (c) per participant the best-fit-hinge incisal RMS with its bootstrap CI against that participant's in-vivo and bench noise-only P99 (log axis; count above in-vivo noise from `hinge_noise_summary` in the caption); (d) condyle-point excursion per participant with the P95 design value and its CI from the `requirements` table. |
| Fig. 3 | `icorr_imu` | Bland–Altman of IMU-estimated vs optical incisal opening per cycle (grey) and participant mean (black), generic (LOSO) and own model stacked on one x range; bias and repeated-measures LoA with participant-cluster bootstrap CI bands and the proportional-bias line (slope [CI] in the panel note); the angle RMSE (`imu_agreement_cohort`) in the caption. |
| Table I | `table_requirements` | §2 |

## 5. Journal set (Springer profile; `paper_journal.py`)

`paper_journal.yaml` lists the main figures in this order, then Tables I-IV; Fig. 1, 3 and 4 are
the ICORR views unchanged. All views are thin: they read stored tables, compute medians/IQRs
of participant values and nothing heavier (the one exception: the few-shot EMG + IMU vs IMU
test, an exact Wilcoxon signed-rank test with Holm over the calibration amounts, which ml_eval
does not store). Blocks (`views.blocks`): `strip_mark`, `ecdf` (+ `ecdf_steps`), `slopegraph`,
`paired_forest`, `spec_curve` (curve + indicator matrix in one column, `sharex`); tests in
`tests/test_r4_journal_blocks.py` (blocks fast; every `jnl_*` view on the synthetic study for
the IEEE and Springer venues, slow).

| Fig. / Table | Id | Panels and rule (source) |
|---|---|---|
| Fig. 1 | `icorr_setup` | protocol and synchronized modalities (ICORR set, `paper_icorr.py`) |
| Fig. 2 | `jnl_requirements` | (a) largest incisal opening / protrusion / laterotrusion per participant, bar: P95 [CI] (`stats_kinematics.subjects`, `requirements` keys `opening_mm`, `protrusion_mm`, `laterotrusion_mm`); (b) ECDF of `peak_open_vel_max_dps` with P95 [CI] (`peak_vel_dps`); (c) `imu.bandwidth.f95_coh_hz` per opening task and the per-participant maximum over non-noise-limited tasks with P95 [CI] (`bandwidth_hz`; noise-limited shown light, not used, D12); (d) one ms axis relative to the opening onset: X2 closer release (`ensemble_cohort`), X3 causal decision (`release_cohort`, primary), C2 lead under both rules (`timing_cohort`, TKEO, n, excluded by the 3-cycle minimum, Holm p in the caption), offline decoder latency (`ml_eval.summary` loso/svm/emg_imu) and stream latency (`ml_online.summary`, median and IQR). In a store without the `requirements` stage (older snapshots) the P95 is the pandas-linear percentile of the participant values and the caption says the CIs are missing |
| Fig. 3 | `icorr_hinge` | C1 (ICORR set, `paper_icorr.py`) |
| Fig. 4 | `icorr_imu` | C3a (ICORR set, `paper_icorr.py`) |
| Fig. 5 | `jnl_c2` | (a) open-close opening envelopes (`timing` arrays `ensemble__open_close__opening__{agonist,antagonist}`, dB re −600…−300 ms), median and IQR across participants (X2); (b) slopegraph of the per-participant lead (`timing_spec`, `detector=tkeo\|mech=lead_ms\|onset={first,drive}`), diamonds: `timing_cohort` median [BCa CI], annotations n, excluded (`n_excluded_min_cycles`), Holm p (`p_holm`, confirmatory family of four); caption: both rules, chewing values, "not confirmed"; (c) specification curve of the 36 detector × mechanical source × rule medians [CI] (`timing_cohort`, metrics `lead_ms`, `lead_omega_ms`, `lead_acc_ms`, `lead_opt_ms`) over the indicator matrix; the pre-specified and the driving-burst TKEO specifications highlighted |
| Fig. 6 | `jnl_x3` | X3, exploratory: (a) per participant detected vs chance (`timing.release_summary`, primary); (b, c) anticipation and false releases, offline replay of the causal chain vs stream (`ml_online.release`), median [IQR]; (d) detected − chance vs false releases over the drop × hold sweep (`release_cohort`) |
| Fig. 7 | `jnl_decoding` | (a) per participant BA(EMG + IMU) − BA(IMU): LOSO (`ml_eval.loso`, Holm p from `tests`) and few-shot k = 20/50/100 (`ml_eval.variants`, exact Wilcoxon, Holm over k); (b) few-shot BA vs calibration seconds (k × 0.05 s), median and IQR (`variants_summary`); (c) per-class recall of the pooled LOSO confusion (`confusion__{emg_imu,imu,emg}\|svm`); (d) block permutation importance (`permutation_importance`) |
| Fig. 8 | `jnl_online` | X4, exploratory (`ml_online`): (a) stream `ba_raw` per participant per feature set × calibration mode, square: offline median of the same decoder; (b) ECDF of the latency of every detected onset (`decisions`, kind hit, no calibration); (c) false activations at rest; (d) compute per 20-ms chunk (median, P99, largest), real-time factor in the caption |
| Fig. 9 | `jnl_robustness` | (a) standardized specification curve: every non-primary variant of `mio_mm`, the open-close opening lead and the masseter R (chewing) / digastric R (open-close) activation: mean paired change vs the primary variant / `sd_between` with a t interval (`variability.spec_values`, `spec_curve`), matrix: outcome and the choice that differs; (b) all vs plausible condyles (`stats_kinematics.condyle_sensitivity`), C1 count per subset in the caption (`uncertainty.hinge_noise`, `subjects.condyles_plausible`); (c) primary vs sensitivity run for the four C2 leads and the X2 closer release (`sensitivity/<name>/comparison.parquet`; † = the comparison's primary differs from the current analysis); the D16 label floor (identical by construction) from `labels.label_agreement` |
| Table I | `table_reference` | §2 |
| Table II | `table_requirements` | §2 |
| Table III | `jnl_table_decoding` | rows: unseen (LOSO; all five feature sets), same participant (blocked), calibrated (few-shot, largest k), simulated real time (feature set × mode). Columns: BA median [BCa CI] (`summary`) or median (IQR); recall range of the pooled LOSO confusion; latency and false activations (LOSO: `summary`; stream: medians of `ml_online.summary`); "> IMU": participants with higher BA than IMU alone in the same protocol; Holm p of EMG + IMU vs IMU (LOSO: stored; few-shot: computed); evidence level |
| Table IV | `jnl_table_robustness` | per sensitivity run (header row, † as in Fig. 9c): the four C2 leads, X2 closer release, X3 detection vs chance, masseter R chewing activation, LOSO BA, C1 above-noise count: primary and run value from the comparison file, "same conclusion" = same decision at p < 0.05 (Holm where stored; detection: above chance; noise count: at most one participant below); the `labels_p95` row states the D16 identity |

Supplement candidates (existing dashboards, listed as comments in the yaml): `timing_ensemble`
(chewing), `timing_detectors`, `timing_release_sweep`, `online_release`, `movement_bandwidth`,
`imu_agreement`, `activation_map`, `emg_spectra`, `ml_permutation`, `ml_variants`,
`online_compute`.

## 6. C2-E and camera-free views

**C2-E** (`viz/views/paper_c2e.py`; post hoc family X6, an05 §3.3). Stored results only
(`timing:c2e_null`, `c2e_cycles`, `c2e_release`, `coupling`, `stats_timing:c2e_cohort`); a
participant enters a cohort row with ≥ `min_usable` usable channel-cycles, as in the tests;
the motion events are the angular-velocity onsets t_ω; on a time axis an EMG event lies at
−lead (negative = before the motion event). The cohort median CI is the stats stages' BCa
routine; no test is computed in a view.

| Id | Kind | Shows |
|---|---|---|
| `c2e_sequence` | paper | the event sequence per task — closer release before the opening onset, the opening onset (0), the digastric onset, the closing onset and the closer onset before it — every participant as a dot, the cohort median with its BCa CI; forms `timeline` (one axis from the opening onset; the closing onset at each participant's median opening duration), `aligned` (each EMG event against its own motion onset), `raincloud` |
| `c2e_locking` | paper | (a) locking of the closers before closing (both tasks pooled): each participant's lead IQR against the median IQR of its S1 surrogates; (b) the EMG-to-acceleration lag per task (P5), residualised and raw; forms `dumbbell`, `slopegraph` |
| `c2e_table` | paper table | the rows of `stats_timing:c2e_cohort`: H1, the two gated release hypotheses, the supporting block, the structure rows (descriptive, cohort p withheld) and the decision |
| `c2e_cycles` | dashboard | every channel-cycle lead per participant (task; lead source picker) |
| `c2e_release` | dashboard | every cycle's closer release and the release depth against the surrogates |

Captions state the temporal-order wording ("precedes … within the cycle", an05 §3.3) and the
post hoc evidence level.

**Camera-free** (`viz/views/camerafree.py`; X5, X5b, X5c; dataset group, domain IMU). Tables
`camerafree:*` and `camerafree_cohort:*`.

| Id | Shows |
|---|---|
| `cf_ladder` | which camera data each rung of the estimator ladder uses |
| `cf_opening_ladder` | opening-angle RMSE per rung (E1 … E3b) per participant, cohort median with BCa CI, the 1.2° bound (paper item, block B2) |
| `cf_drift` | E2 vs E3 opening error over the take (pid): drift since the reference and resets at the anchors |
| `cf_incisor_axes` | translation: incisor error per axis and rung |
| `cf_traces` | estimates against the optics over the take (pid) |
| `cf_table_results` | cohort table: medians [BCa CI], bound verdicts, paired ladder contrasts (paper item, block B2) |
| `cf_decomposition` | X5b: rotation-induced vs translation-model incisor error per task (supplement) |
| `cf_table_oracle` | X5b / X5c cohort table (supplement) |
| `cf_oracle_tasks` | the per-participant rows behind them: participant × task heatmap of one oracle combination (estimator picker) |

The systems-paper views of the same period — `val_camerafree`, `val_table_camerafree` (VD20),
`val_hinge`, `val_table_hinge` (VD21), `val_budget`, `val_table_budget`,
`val_table_marking_gap` (VD22), `val_emg_side`, `val_table_emg_side` (VD23) — are described in
an14 §12.
