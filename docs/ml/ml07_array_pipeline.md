<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml07 — The biomech stages `ml_windows` and `ml_eval` (ICORR 2027)

The ml package registers its stages in the biomech pipeline (entry point `biomech.stages` →
`ml.biomech_stages`); they are cached, appear in the status matrix and can be started from the
viewer. `ml_windows` and `ml_eval` (this page) are thin adapters over the same core as the `ml`
CLI; the third stage, `ml_online` (pseudo-online replay, registered on import of
`ml.online.stage`), is described in [`ml08`](ml08_online.md).

> **Status:** implemented; `ml_windows` v3, `ml_eval` v4, `ml_online` v1. Since D14
> (2026-10-16) the causal reference segments exclude rail and lead-off samples only.

```bash
biomech run --config experiments/icorr2027/study.yaml --stage ml_eval   # + ml_windows if stale
biomech show --config experiments/icorr2027/study.yaml ml_eval summary  # or the viewer's Results page
```

## 1. Data flow

`session.h5` + biomech stages `labels`, `qc` (and the MVC take) → `ml_windows` (per participant)
→ `ml_eval` (cohort) → `ml_online` (cohort; also reads `timing`, `labels` and the recordings).

| Stage | What it does |
|-------|--------------|
| `ml_windows` | biomech provides the measured mains frequency, its artifact masks (decoding uses rail + lead-off, dilated) and the IMU streams. The normalization reference has biomech's definition (`signals.emg.reference`: `p95` of the MVC envelope, `max_recorded` = largest envelope over the MVC phase and every task (D11), `max1s`) but is computed on the **causal chain of the windows** (`_causal_reference`; samples hit by biomech's analysis mask rules except the envelope rule are excluded (`ref_rules`), i.e. since D14 rail and lead-off — the low-frequency rule only flags; warm-up as for the windows), so reference and features have the same gain. Without a reference and with `emg_norm: rest_relative` the causal rest RMS is used (D13 features do not depend on it). The `rest` phase is always windowed (thresholds, D13 statistics). `ml.core.dataset.build_windows` does the rest ([`ml02`](ml02_emg_preprocessing.md), [`ml03`](ml03_feature_extraction.md)) |
| `ml_eval` | pools the stored windows into a `WindowDataset`, each participant with its **own** feature names (a mismatch, e.g. a different channel order, is an error), and runs `ml.protocols.evaluate`: nested `loso_subject` for every configured model × feature set (parallel workers, [`ml05`](ml05_evaluation.md) §7), `blocked`, the permutation test (LDA, `emg_imu`, circular label shifts of each participant's label sequence; `n_perm` runs fitted like the observed LOSO: its `inner_folds` and `max_train`), Friedman + Holm-Wilcoxon against `emg_imu`, age/sex descriptives, the exploratory `fusion_diagnostics` and `permutation_importance` ([`ml05`](ml05_evaluation.md) §6), and the decoding variants late fusion, reduced EMG and few-shot calibration on the same LOSO folds (`ml.protocols.variants`, [`ml05`](ml05_evaluation.md) §6.1) |

Outputs:

| Stage | Tables | Arrays |
|-------|--------|--------|
| `ml_windows` | `meta` (`n_windows`, rejected `masked`, `label`, `gap`, `nonfinite`, `emg_norm`, `n_rest_windows`, `ref_chain`, `ref_mode`), `features` (`block`, `feature`; all four blocks), `refs` (`muscle`, `ref_uv`, `source` mvc/task/rest, ZC/SSC/WAMP `thr`) | `X_emg`, `X_imu`, `X_seq`, `X_seq_imu`, `y`, `t_end_us`, `phase` (index into the phase list), `norm_{emg,seq}_{center,scale}` (D13) |
| `ml_eval` | `loso`, `blocked`, `summary`, `tests`, `generalization`, `fusion_diagnostics`, `permutation_importance`, `variants`, `variants_summary` | `confusion__<fset>\|<model>` (summed over folds), `permutation_null`, `pred__<pid>__<fset>__{t_us,y,yhat}` |

`pred__<pid>__<fset>__*` is the window-level decoder output of the **primary model**
(`primary_model`, SVM) for every feature set it was run with: window end time (`t_us`), true
(`y`) and predicted (`yhat`) class of the participant's held-out LOSO windows
(e.g. `pred__s01__emg_imu__yhat`). The viewer's decision timeline (`ml_timeline`, data explorer
label row) and the video overlay read them.

`loso` columns: `pid`, `feature_set`, `model`, `best_params` (the fold's grid choice),
`balanced_accuracy`, `macro_f1`, `n`, `latency_ms_median`, `detected_frac`, `fa_frac`,
`fa_per_min`, `free_balanced_accuracy`, `free_macro_f1`. `summary`: per protocol × feature set × model and
metric, `n`, mean, SD, median, quartiles, range and the 95 % BCa bootstrap CI of the median
(`ci_lo`, `ci_hi`). `tests`: Friedman over the primary model's feature sets and
Holm-corrected Wilcoxon tests against `emg_imu` (with ≥ 3 feature sets and ≥ 5 participants), the
permutation test. `generalization`: Spearman age vs balanced accuracy and median per sex for the
primary model with `emg_imu` (≥ 5 participants).

Version 4 adds the `variants` and `variants_summary` tables (exploratory decoding
variants, [`ml05`](ml05_evaluation.md) §6.1); the existing tables are unchanged.

Version 3 changes numbers: ZC/SSC/WAMP thresholds on the sample-difference scale,
D13 rest-relative EMG features (default), causal-chain reference, SVM grid
`C ∈ {0.01, 0.1, 1, 10}` × `gamma_mult ∈ {0.25, 1, 4}`, LDA uniform priors
([`ml03`](ml03_feature_extraction.md) §4, [`ml04`](ml04_models_and_training.md) §1).

Version 2 added `X_seq`, `X_seq_imu` and the `nonfinite` counter; all other columns are unchanged.
Method changes that move numbers: inner search scored with macro-F1 (was balanced accuracy), MLP
early stopping on a group-split validation set (was a random window split), per-block weighting
for multi-block feature sets, labels older than 0.1 s at the window end ignored.

## 2. Configuration

`software/biomech/biomech/etc/biomech.yaml`, `stages.ml_windows.window` (any `WindowConfig` key,
[`ml03`](ml03_feature_extraction.md) §2, incl. `emg_norm`; unknown keys are errors) and
`stages.ml_eval` (`configs` (model → feature sets: `lda`, `svm` × `emg`, `imu`, `emg_imu`,
`emg_nodig`, `emg_nodig_imu`; `mlp` × `emg`, `emg_imu`), `blocked` (LDA and SVM × `emg`, `imu`,
`emg_imu`), `primary_model: svm`, `inner_folds: 3`, `max_train: 12000`, `n_perm: 100`,
`n_jobs: 12` (0 = every core), `flat_search: false` ([`ml05`](ml05_evaluation.md) §7), `diagnostics:
{fusion_variance: true, importance: {model: svm, feature_set: emg_imu, n_repeats: 5}}`,
`importance: null` = off; `variants: {hyper: main, late_fusion: {w_step: 0.25}, reduced_emg:
{feature_set: emg_imu, features: [log_rms, mnf]}, few_shot: {feature_sets: [imu, emg_imu],
mode: per_phase, k: [0, 20, 50, 100], guard_s: 0.35, weight: balanced}}` (`variants: null` = all off, a variant
`null` = off; definitions [`ml05`](ml05_evaluation.md) §6.1), overridable in the study YAML. The
model × feature set of `importance` is added to `configs` if missing. The reference definition
follows `signals.emg.reference` (default `max_recorded`, D11: EMG in % of max recorded); the
decoding masks and the reference rules follow `signals.emg.mask` (`dilate_s`, `analysis_rules`).
The `ml` CLI's `ml.yaml` is not read by these stages.
Method definitions: [`an05`](../analysis/an05_timing_and_onsets.md) §5.

## 3. Equivalence with the CLI

`ml labels from-biomech` copies the `labels` stage, the MVC take and biomech's MVC reference into
sidecars; `ml dataset --labels-dir` on those sidecars computes the causal p95 reference from the
MVC take (the copied zero-phase reference is only a fallback without one). With the default
`emg_norm: rest_relative` the features do not depend on the reference, so both entry points give
the same windows, labels and features on equal settings (verified on s01-s03;
with `emg_norm: mvc` they agree when `signals.emg.reference` is `p95`). `ml train` on that dataset runs the same folds and
models as `ml_eval` (set `train.max_train: 12000` to match its training subsample).

## 4. Extending

- **Model:** register it in `ml/models/registry.py`; list it under `stages.ml_eval.configs`.
- **Feature:** add it to the registry in `ml/core/features.py`; name it in `window.emg_features`.
- **Feature set:** combine blocks with `+` (`emg+imu`); no code change.
