<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml04 — Models, protocols and training

The model registry, the evaluation protocols and what `ml train` does. The biomech stage
`ml_eval` runs the same functions ([`ml07`](ml07_array_pipeline.md)).

> **Status:** implemented — `ml/models/registry.py`, `ml/models/gru.py`, `ml/protocols/`,
> `ml/training/run.py`.

## 1. Model registry

`ml/models/registry.py` is the only place that names models; the CLI choices, `ml eval` and the
comparison table read it.

| Name | Model | Default inner grid |
|------|-------|--------------------|
| `lda` | shrinkage LDA (`lsqr`, `shrinkage="auto"`), **uniform priors** over the training classes (`priors: uniform`; `empirical` = class frequencies) | — |
| `svm` | RBF SVC, `class_weight="balanced"` | `C ∈ {0.01, 0.1, 1, 10}` × `gamma_mult ∈ {0.25, 1, 4}` (same in `ml.yaml`) |
| `mlp` | scikit-learn MLP (64, 32) | `alpha ∈ {1e-4, 1e-2}` (`ml.yaml` adds `learning_rate_init ∈ {1e-3}`) |
| `gru` | PyTorch GRU on `(n, S, d)` sub-frame sequences | none (`ml.yaml`: `hidden_size ∈ {32, 64}`) |
| `late_mean:<m>`, `late_weighted:<m>`, `late_stack:<m>` | late fusion of one `<m>` per block ([`ml06`](ml06_sensor_fusion.md) §4) | — |

Common contract: `fit(X, y, groups)`, `predict`, `predict_proba` on the 7-class axis; the first
step is a per-block scaler fit on the training rows. `groups` drive every internal split:

- **SVM probabilities** (only when requested, `train.probability`): sigmoid calibration on grouped
  out-of-fold decision values (`GroupKFold`); with fewer than two groups a softmax of the decision
  function is used and recorded (`proba_method`). The former calibration used window-level
  folds and leaked overlapping windows.
- **MLP / GRU early stopping:** validation rows are whole groups; the score is validation
  macro-F1, the best weights are restored. scikit-learn's own `early_stopping` (random window
  split) is not used.
- **GRU:** AdamW, class-weighted cross-entropy, gradient clipping, mean-pooled (or last) hidden
  state, unidirectional by default (bidirectional is offline-only). It outputs the classes present
  in training. Without PyTorch the registry reports it as unavailable and `make` raises with the
  install hint.

- **SVM gamma:** `gamma_ = gamma_mult / (n_features · Var(X_scaled))`, computed on the fold's
  training rows after the block scaler, so `gamma_mult = 1` is scikit-learn's `"scale"`; with a
  numeric `gamma` the multiplier scales that value.
- **LDA priors:** windows are REST-dominated; empirical priors shift every decision towards REST.
  Uniform priors are the LDA counterpart of the SVM's balanced class weights.

Erratum (2026-10-16): the former SVM grid `C ∈ {1, 10}` chose its lowest value in 59/60 LOSO
folds and never tuned gamma; LDA used empirical priors.

Parameters and grids per model come from `train.models.<name>.{params, grid}` in `ml.yaml`
(`ml train`); a model without a `grid` there uses the registry grid. The biomech stage `ml_eval`
does not read `ml.yaml` and always uses the registry grids and default parameters.

## 2. Inner search

`GroupKFold` (`inner_folds`, default 3) over the inner groups of the training rows; every grid
point is scored with the **one** scorer, macro-F1 over the classes present in the true labels;
the best point is refit on all training rows with the same groups; the first best point in grid
order wins. `--no-search` uses the defaults. `max_train` subsamples training windows per fold for
speed (`default_rng(seed)`, sorted; `evaluate.subsample`, shared by `ml_eval`, its variants and
`ml_online`).

## 3. Protocols

| Protocol | Outer folds | Inner groups | Claim |
|----------|-------------|--------------|-------|
| `loso_subject` | one per subject | subjects | inter-subject (headline) |
| `loso_session` | one per take | sessions | inter-session; with several subjects their other takes stay in training (warning) |
| `blocked` | per subject, `k` (5) contiguous time blocks cut **inside every phase**; block b of all phases tested together | blocks | within-subject |

- Training uses the structured phases; `free_motion` windows of the held-out unit are predicted
  and scored separately (`free_*` metrics).
- `blocked` drops training windows closer than `guard_s` (0.35 s) to a test window of the same
  take, because 75 %-overlapping windows share samples. Cutting the session into global blocks
  would remove whole classes from training (classes are confined to their task). It replaces the
  former bout-grouped `groupkfold`, which had no guard.
- There is **no hidden hold-out**: every unit is tested exactly once by its outer fold. The former
  `holdout_fraction` (which originally removed about 20 % of the subjects from LOSO) is gone.
- Names: the old `loso` was ambiguous (the legacy CLI meant sessions, the biomech stage subjects)
  and is rejected with a message naming `loso_subject` and `loso_session`.
  `leave_one_subject_out` / `losubo` and `leave_one_session_out` are accepted aliases;
  `groupkfold` is rejected with a pointer to `blocked`.

## 4. Leakage rules

- Scaling, search, calibration and early stopping only see the fold's training rows, split by
  groups.
- Feature thresholds and normalization references are label-free and fixed per take
  ([`ml02`](ml02_emg_preprocessing.md) §6); there is no build-time standardization.
- Predictions are stored with their dataset **row indices**.

## 5. `ml train`

```bash
ml train --dataset DIR_OR_NPZ [--model NAME] [--feature-set SET] [--protocol P]
         [--search/--no-search] [--max-train N] [--n-jobs N] [--runs-dir DIR] [--config FILE]
ml train --compare RUNS_DIR
```

Defaults come from `train:` in `ml.yaml` (`lda`, `emg`, `loso_subject`). `--n-jobs` (default
`train.n_jobs` = 1) runs the outer folds in parallel worker processes, one fold per task (one BLAS
thread each); 0 means every core. Results do not depend on it. The run directory is
`<runs_dir>/<UTC stamp>_<model>_<feature set>_<protocol>_<config hash>/`
([`ml05`](ml05_evaluation.md) §3). The console shows fold mean ± SD (ddof = 1).

## 6. Reproducibility

One `seed` drives NumPy, Python, scikit-learn and PyTorch. Each run writes the effective config
and its source file, `seeds.json`, the dataset reference (path, summary, build config hash),
package versions and the git commit (with a dirty flag).
