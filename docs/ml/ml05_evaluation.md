<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml05 — Evaluation

Metrics, run artifacts, `ml eval` and the comparison table.

> **Status:** implemented — `ml/evaluation/` (metrics, evaluate, compare, plots),
> `ml/training/artifacts.py`.

## 1. Metrics

| Metric | Role |
|--------|------|
| **Balanced accuracy** | headline per fold / subject (mean per-class recall) |
| **Macro-F1** | over the classes **present** in the true labels; the one scorer for search and early stopping |
| Accuracy, weighted F1 | reported, never used for selection (REST dominates) |
| Cohen's κ, MCC | agreement beyond the class prior |
| Macro AUROC, per-class average precision | need `predict_proba` (`train.probability: true`) |
| Per-class precision / recall / F1 / support | which motions fail |
| Detection latency | per REST→class onset: first time the prediction holds the class for 2 windows, from 0.5 s before the onset; median ms, negative = anticipatory; share detected. Onset, search range and the 2-window run stay inside one segment (same take and phase, no step between window ends > 2 strides), so a phase boundary or dropped windows never create or confirm an onset (`evaluate.latency`) |
| False activations | rest-phase windows predicted as movement: share and events per minute |
| `free_*` | balanced accuracy / macro-F1 on the held-out unit's `free_motion` |

Across folds: mean and **sample SD (ddof = 1)**; with a single fold the SD is undefined (null).

## 2. Confusion matrices

Rows = true, columns = predicted, classes in id order; counts and row-normalized panels
(`confusion_matrix.png`) plus `confusion_matrix.csv`. Patterns worth reading: OPEN↔CLOSE
(amplitude, not direction), LEFT↔RIGHT (side channels not separated), collapse into REST.

## 3. Run directory

```
<stamp>_<model>_<feature set>_<protocol>_<hash>/
├── config.effective.yaml   source config file + effective train settings
├── seeds.json, env.json, git_commit.txt
├── dataset_ref.json        dataset path, summary, build config hash
├── metrics.json            pooled metrics, per-fold metrics, mean, std (ddof = 1)
├── classification_report.txt, confusion_matrix.{png,csv}, per_class_f1.png
├── summary.json            model, feature_set, protocol, test units, best params, aggregate
└── per_fold/fold_<k>/
    ├── metrics.json        fold metrics + latency, false activations, free_*, fit info
    ├── predictions.npz     indices, y_true, y_pred, y_proba, t_end_us, groups
    │                       [free_indices, free_pred]
    └── model.joblib        fitted model (all models, the GRU included)
```

`indices` are row indices into the training `dataset.npz`. Predictions are never matched by
timestamp: pooled takes share the master-clock range, and the former timestamp lookup mapped rows
of one take onto another.

## 4. `ml eval`

```bash
ml eval --run RUN                       # pooled metrics from saved fold predictions
ml eval --run RUN --dataset DS          # reload every fold model, re-score its rows (by index)
ml eval --model-path M --dataset DS --feature-set emg+imu   # one model on every window
ml eval --compare RUNS                  # same as ml train --compare
```

`--run --dataset` checks that the indices exist and that the labels at those rows equal the saved
`y_true`; otherwise it fails (wrong dataset). Runs of earlier versions have no indices and
are refused with a retrain message. `--model-path` also scores the model's own training windows,
so it is not a cross-validated score. Output: `<run>/eval/` or `paths.evals_dir/<stamp>_<name>/`.

## 5. Comparison

`ml train --compare RUNS` writes `comparison.md` / `.csv` (model, feature set, protocol, fold
mean ± SD of every scalar metric) and `per_class_f1.png` (best run per model). Rules: rank by
macro-F1 or balanced accuracy within one protocol only; prefer the simpler model when the
difference is within the fold SD; mark bidirectional GRU results as offline-only.

## 6. Exploratory diagnostics (biomech `ml_eval`, D13)

Supporting outputs, not headline results; switched by `stages.ml_eval.diagnostics`
([`ml07`](ml07_array_pipeline.md) §2). Both are cheap. The decoding variants (§6.1) are switched
by `stages.ml_eval.variants`.

**`fusion_diagnostics`** (`ml.evaluation.diagnostics`), one row per EMG feature column, on *all*
pooled windows (all phases, free motion included): `var_between` = variance of the
per-participant means (participants weighted equally), `var_within` = mean of the
per-participant variances, `ratio` = between / within, `icc` = between / (between + within),
each `_before` and `_after` D13 ([`ml03`](ml03_feature_extraction.md) §4.1). "Before" is on the
scale D13 centres (log for linear amplitude features; for the default features this equals the
`emg_norm: mvc` features): for `rest_relative` windows it is reconstructed exactly from the stored
rest statistics; for `mvc` windows "after" is computed from the participant's rest-phase windows
in the pool. A large ratio means the feature mainly encodes *who* the participant is, which LOSO
cannot learn away. Columns also give `channel`, `kind`, `n_windows`, `n_participants`.

**`permutation_importance`** (`evaluate.block_importance`): for the configured LOSO model ×
feature set (default `svm`, `emg_imu`), the fitted fold model predicts its test fold again with
one block's columns permuted jointly across the test rows (within-block structure kept, link to
the label and to the other block broken; no refit), `n_repeats` times (default 5). One row per
participant × block: `balanced_accuracy`, `ba_permuted` (mean), `ba_drop`. The model and its
search are the LOSO fit of the `loso` table (no extra training).

### 6.1 Decoding variants (`variants`, `variants_summary`)

Exploratory follow-ups to the poor cross-participant transfer of EMG; switched by
`stages.ml_eval.variants` (`null` = all off, a variant set to `null` = that one off; default all
on). Code: `ml.protocols.variants`. Common to all:

- **Model** SVM (RBF, `class_weight="balanced"`, `BlockScaler` per fold), **folds** the
  `loso_subject` folds of the main run (structured windows only; free motion not scored), the
  **training subsample** of the main run (`max_train` windows drawn with `default_rng(seed)`,
  `evaluate.subsample`).
- **Hyper-parameters** (`hyper: main`, default): the `C`, `gamma_mult` the main nested LOSO chose
  for that fold and feature set (`best_params` of the `loso` row; column `hyper` =
  `main_loso:<fset>`). A fold without such a row, or `hyper: search`, gets one grouped inner
  search on the fold's training subsample (`hyper` = `inner_search`). No other search is run.
- **Scores** balanced accuracy and macro-F1 of the participant's evaluated windows (§1).

| Variant | Definition | Evaluated on |
|---------|------------|--------------|
| `late_fusion` (`emg+imu`) | SVM on the `emg` block (params of `svm\|emg`) and SVM on the `imu` block (params of `svm\|imu`), each with grouped sigmoid calibration (`SvmClassifier`, `probability=True`); fused posterior `w·P_emg + (1−w)·P_imu`, prediction = argmax. `w` (column `w_emg`) ∈ {0, 0.25, 0.5, 0.75, 1} (`w_step: 0.25`) maximizes the macro-F1 of the *out-of-fold* posteriors over `GroupKFold(inner_folds)` by participant inside the training fold; ties go to the smaller `w` (`ml.fusion.late.LateFusionClassifier`, `method="weighted"`, `block_params`). `w = 0` is the IMU-only model exactly | whole held-out fold |
| `reduced_emg` (`emg[log_rms,mnf]+imu`) | `emg_imu` with the EMG block restricted to the per-channel `log_rms` and `mnf` columns (12 of 66; `features`), IMU block unchanged, per-block weighting as for `emg_imu`; params of `svm\|emg_imu` (`gamma_mult` is relative to the fold's `"scale"`, so it adapts to the dimension) | whole held-out fold |
| `few_shot` (`imu`, `emg_imu`; k ∈ {0, 20, 50, 100} windows per phase) | a guided calibration routine (`mode: per_phase`, default): the held-out participant's windows per (session, phase) in recording order; calibration set = the **first k windows of every phase** (each movement once: k = 20/50/100 windows = 1/2.5/5 s per phase at the 50 ms stride; a phase shorter than k contributes all its windows). Evaluation set, the same for every k: in each phase the windows ending more than `guard_s` = 0.35 s after its K-th window, K = max(k) (the 75 %-overlapping windows then share no samples). `mode: first` takes the first k windows of the recording instead (only the first task's classes). The SVM (params of `svm\|<fset>`) is refit on the fold's training subsample (`n_src` windows, weight 1) plus the calibration windows (n_cal = k × phases), each with sample weight `w_user` = `n_src / (n_classes_src · n_cal)` (`weight: balanced`: the calibration windows together weigh as much as one average training class) or a fixed multiplier; `class_weight="balanced"` multiplies both. k = 0 is the plain LOSO model, identical to the main run's fold model, scored on the evaluation set | evaluation set (`test_portion = after_calibration`) |

`variants`: one row per variant × feature set × k × participant: `variant`, `feature_set`, `k`
(0 for the non-calibrated variants), `pid`, `balanced_accuracy`, `macro_f1`, `n_test`,
`test_portion` (`fold` | `after_calibration`), `params`, `hyper`, `fit_time_s` (fit of the variant
model(s) only, single core), and per variant `w_emg` (late fusion), `n_emg_features` (reduced),
`n_cal`, `cal_classes` (distinct classes among the k windows), `w_user` (few-shot).

`variants_summary`: per variant × feature set × k and metric (`balanced_accuracy`, `macro_f1`,
`fit_time_s`): `n` participants, `median`, `q1`, `q3`, `iqr`, `mean`; for few-shot also the paired
per-participant differences `ba_gain_vs_k0` = BA(k) − BA(0) (same evaluation set) and
`ba_emg_imu_minus_imu` (feature set `emg_imu-imu`, same k), so the EMG benefit with calibration
is read directly. The `few_shot` k = 0 rows are *not* comparable to the `loso` table (smaller
evaluation set); `late_fusion` and `reduced_emg` are (same rows).

Cost per fold (default config): late fusion 32 single-block SVM fits (2 blocks × (3 out-of-fold
+ 1 final) × 4 for the grouped calibration), reduced EMG 1 fit, few-shot 2 × 4 fits; ≈ 41 fits
against ≈ 185 of the main SVM LOSO (5 feature sets × (36 search + 1)), i.e. about +20 % of the
SVM part of `ml_eval` (late fusion ≈ 3/4 of it). Parallel over (variant, fold) with `n_jobs`.

## 7. Parallel execution (biomech `ml_eval`)

| Setting / part | Behaviour |
|----------------|-----------|
| `stages.ml_eval.n_jobs` (12) | worker processes (loky, one BLAS thread each); `0`, `-1`, `null` or `"auto"` = every core (`evaluate.resolve_jobs`). Nested LOSO (`evaluate.loso_many`): every configuration's inner fits (configuration × fold × grid point × inner split) and refits in one pool (`ml.protocols.pool`), longest first, no barriers. Inner splits of different outer folds that train on the very same rows (same subject set and subsample) with the same grid point are one fit, scored on each fold's inner test rows (2026-10-31; the same computation done once) |
| `stages.ml_eval.flat_search` (`false`) | `true` runs the inner grid fits of *all* folds (fold × grid point × inner split, e.g. 12 × 12 × 3 SVM fits) as one flat pool, then one refit per fold (`evaluate._flat_search`); meant for machines with more cores than folds |
| `blocked` | both runs of every configuration (LOSO choice and defaults) in one pool over all fold fits (`evaluate.blocked_many`, 2026-10-31; before, one run after the other); a fold fit both runs share (a model without a grid: LOSO choice `{}` = defaults) runs once |
| permutation test | parallel over the `n_perm` permutations (`n_jobs` workers, in batches so a cancelled run stops between them); each permuted LOSO run is serial with the observed run's `inner_folds` and `max_train`; skipped (no row) when the observed `lda\|emg_imu` mean is missing or non-finite |
| variants (§6.1) | parallel over (variant, fold) tasks |
| `ml train --n-jobs` | outer folds as for `ml_eval`; no flat search ([`ml04`](ml04_models_and_training.md) §5) |

The results do not depend on `n_jobs` or `flat_search`: the same subsample, splits and choice
(first best point in grid order) are used, and the flat search reports the same `best_params`
(with `inner_macro_f1`) as the per-fold search. Few-shot k = 0 reuses the main run's
predictions of the same fold model instead of refitting it (its `fit_time_s` is NaN).

Pilot benchmark (LOSO SVM `emg_imu`, other jobs on the machine): 6 workers 398 s, 12 workers 380 s,
24 workers with threaded inner search 590 s; flat search 483 s (12 workers). Hence the default
of 12 workers, one per fold, flat search off; re-benchmark on an idle machine before changing it.
The whole `ml_eval` stage took about 34 min on the pilot (12 participants).

## 8. Synthetic data caveat

The tests use synthetic sessions whose EMG and IMU depend on the class, so models must learn
them (a check of the plumbing, not of the method). Near-chance scores on data whose signals are
independent of the labels are expected; high scores there indicate a leak.
