<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Data-set study (JNER and ICORR papers)

The analysis behind two papers on the same recordings: the **data-set paper** (jaw exoskeleton
requirements from synchronized kinematics, surface EMG, and inertial data) and the
**camera-free paper** (jaw tracking with two IMUs against the optical reference, analysis X5
of the plan). The folder name is historical.

- **Plan:** `analysis_plan.md` was frozen before any outcome was computed (v1.0). Every later
  change has an id, a date, a reason, and a note on whether results had been seen.
  `change_log.md` lists all of them in one table per kind.
- **Cohort:** the analysis of record keeps every recorded participant. The papers report the
  13 participants of the reporting population: the paper sets read the `reporting_population` variant
  store.
- **Snapshots:** the papers are exported from frozen snapshots of the processed data, pinned
  in the paper sets (`r17a` for both papers, analysis of record `r17`), each with a verified
  bundle. `supplementary_results/README.md` gives the checksums.

| File | Role |
|---|---|
| `analysis_plan.md` | Pre-specified outcomes, data handling, processing, and statistics, as frozen. |
| `change_log.md` | Concise log of every amendment, erratum, deviation, and added analysis. |
| `cohort.yaml` | Participants, takes, and exclusions. Belongs to the data and is not in the public release. |
| `study.yaml` | `biomech` configuration (paths, stage parameters, sensitivity variants). |
| `paper_dataset.yaml`, `paper_camerafree.yaml` | The paper sets: every figure and table of the data-set paper and of the camera-free paper, with the store and snapshot they read. |
| `numbers_dataset.yaml` | The in-text numbers of both papers (`\numKey` macros). |
| `captions_jner.yaml`, `captions_icorr_cf.yaml` | Written captions of the exported items. |
| `supplementary_results/` | Supplementary figures and tables of the data-set paper with their data. |
| `Makefile` | `check`, `qc`, `all`, `status`, `view`, `paper-journal`, `paper-camerafree`, `verify`, `export-all`, `check-all`, `video`. |

All logic lives in `software/biomech` and `software/ml`. This folder only holds
configuration. Definitions are in [`docs/analysis/`](../../docs/analysis/README.md).

## Setup

```bash
# repo root; the Makefile finds biomech in this venv without activating it
python -m venv software/jtsmm/.venv
software/jtsmm/.venv/bin/pip install -e "software/jtsmm[analysis,dev]" -e software/jaw_tracking \
    -e "software/biomech[all,dev]" -e "software/ml[dev]"
```

The sessions are expected in `data/pilot_study/jtsmm_data/` (untar `jtsmm_data.tar.gz`).
`--data-dir` or `$JTSMM_DATA_DIR` override that location. Results go to
`data/analysis/icorr2027/`. Both directories are git-ignored, because they contain subject data.

## Run

```bash
make -C experiments/icorr2027 check     # validate the cohort manifest
make -C experiments/icorr2027 qc        # data quality (outcome-independent)
make -C experiments/icorr2027 status    # stage x participant status
biomech show --config experiments/icorr2027/study.yaml qc emg     # any result table
```

Stages are cached per participant. A cache key covers the stage version, its parameters (and
the shared `signals` block), the analysis code, the input files and all upstream keys, so a
re-run recomputes only what changed. Each run records git SHA, config hash and package versions
in `data/analysis/icorr2027/runs/<id>/manifest.json`.

## Stages

All stages are implemented; `biomech stages` lists them with their dependencies, and
[`docs/analysis/README.md`](../../docs/analysis/README.md) §2 points to each stage's method doc.

| Stage | Scope | Main outputs |
|---|---|---|
| `qc`, `qc_cohort` | participant, cohort | QC tables, EMG usability, overview |
| `kinematics` | participant | cycles, clinical summary, anatomy/noise floor, finite helical axes (FHA), hinge models, coupling, signals |
| `emg` | participant | references (maximum / reference voluntary contraction, MVC / RVC; largest activity recorded, D11), activation, indices, profiles, co-activation |
| `imu` | participant | calibration, angle agreement, opening via IMU, onset signals |
| `timing` | participant | onsets, EMG lead and its chance null (D23 / D34), causal anticipation; the post hoc family C2-E (X6) |
| `uncertainty` | participant | Monte Carlo noise-only residuals, CI of the observed residual |
| `labels` | participant | intent labels from kinematics |
| `labels_check` | cohort | label validity: Cohen's κ against a second, independent rule (macros `labelKappa…`) |
| `ml_windows`, `ml_eval` | participant, cohort | decoding features; leave-one-subject-out (LOSO) / blocked / permutation (`ml` package) |
| `ml_online` | cohort | pseudo-online (simulated real-time) replay of decoding and the release detector (X4) |
| `emg_relevance` | participant | EMG activity relevance per task and muscle (D19) |
| `camerafree`, `camerafree_cohort` | participant, cohort | camera-free rotation ladder E1–E3 and translation models, anchors, drift (X5); X5b oracle decomposition, X5c task-conditioned model; cohort statistics and contrasts |
| `requirements` | cohort | derived exoskeleton requirements with statistic, lower bound, maximum and evidence level |
| `stats_kinematics`, `stats_emg`, `stats_timing`, `stats_imu` | cohort | reference values and reliability, C1 summaries and nulls, activation (linear mixed model) and statistical parametric mapping (SPM), C2 and C2-E cohort tests, IMU agreement (Bland–Altman) |
| `variability`, `validation` | cohort | specification curve, cycles needed, protocol heterogeneity (S2); reference-validation status (G) |

## Outputs for the paper

```bash
make -C experiments/icorr2027 all       # run everything (cached; ~40 min cold, ML is most of it)
make -C experiments/icorr2027 paper     # TikZ/pgfplots figures + booktabs tables + preview.pdf (journal + conference targets)
make -C experiments/icorr2027 export-all  # every view, perspective and participant (TikZ + PNG/PDF)
make -C experiments/icorr2027 view      # viewer (control center, Results, ...) on http://127.0.0.1:8060
make -C experiments/icorr2027 check-all # biomech check-outputs: every analysis, figure and video
make -C experiments/icorr2027 video     # synchronized video (PID=, LAYOUT=, PHASE=, SPEED=)
```

The targets use `biomech` from the activated venv, or else `software/jtsmm/.venv/bin/biomech`;
`make BIOMECH=/path/to/biomech ...` overrides both.

**Store of the paper outputs.**
- `make paper` reads the store the paper set names: the `reporting_population` variant at the pinned
  snapshot.
- The paper store is the frozen `r17a` (`biomech snapshot --config experiments/icorr2027/study.yaml
  --variant reporting_population r17a`), pinned in the set; `--snapshot live` reads the live store.
- `make all` runs the primary analysis (every recorded participant).
- The `reporting_population` store is a sensitivity variant (`biomech sensitivity`).
