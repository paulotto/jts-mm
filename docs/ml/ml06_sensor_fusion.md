<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml06 — IMU path and sensor fusion

EMG + IMU decoding: the IMU signals and features, and the early and late fusion strategies.

> **Status:** implemented — IMU signals in `ml/core/signals.py`, features in
> `ml/core/features.py`, early fusion `ml/fusion/early.py`, late fusion `ml/fusion/late.py`;
> available in `ml dataset` / `ml train` and in the biomech stages.

## 1. Question

Does adding the mandibular (MTA) and cranial (CRA) IMUs improve decoding over EMG alone, and
does early or late fusion do it better? Everything else is held fixed: same windows, labels,
folds, scorer and metrics, so the comparisons are paired.

## 2. IMU signals and features

Recorded: `/imu_mta|imu_cra/quat` (game rotation vector) and `accel` (**raw, including
gravity**), each with its own time vector; there is no `linear_accel` and no gyro. Samples with
accuracy 0 are dropped and each stream is sorted with repeated stamps removed.

| Signal (12) | Definition |
|-------------|------------|
| `w_mta_*`, `w_cra_*` | body-frame angular velocity (deg/s) by **backward** differences q[k−1]⁻¹ q[k] on a 200 Hz grid (SLERP resampling) |
| `a_mta_*`, `a_cra_*` | gravity-free acceleration: accel − Rᵀ g with R from the same quaternion |

Features per window (25): mean and SD of the 12 signals plus the mean mandible angular speed.
GRU sub-frames: the mean of each signal per sub-frame (`seq_imu`). The features are
calibration-free (no optical setup needed on a device).

The IMU measures the same kinematics the labels are derived from, so an IMU advantage is
expected; the EMG-only and IMU-only rows are needed to interpret any fusion gain.

## 3. Early fusion

A multi-block feature set (`emg+imu`, `seq+seq_imu`) is concatenated. `BlockScaler` standardizes
every column on the training rows and weights block b by `sqrt(D / (B · d_b))`, so each block
carries the same total variance; otherwise the 66 EMG columns dominate the 25 IMU columns in an
RBF distance by count alone. Single blocks are plain standardization.

## 4. Late fusion

One base model per block, posteriors combined (`late_<method>:<model>`, e.g. `late_stack:lda`):

| Method | Combiner |
|--------|----------|
| `mean` | mean of the block posteriors |
| `weighted` | convex weights chosen on grouped out-of-fold posteriors (simplex grid, step `step`, default 0.1; ties to the first grid point = most weight on the last block) by macro-F1; `block_params` gives each block model its own hyper-parameters (used by the `late_fusion` variant, [`ml05`](ml05_evaluation.md) §6.1) |
| `stack` | multinomial logistic regression on grouped out-of-fold posteriors |

Out-of-fold posteriors use `GroupKFold` over the fold's inner groups (at most `inner` = 3
splits; `weighted` and `stack` need ≥ 2 groups), so the meta level never sees in-fold
predictions; every base model is built with `probability=True` (SVM: grouped sigmoid
calibration, ml04 §1). Late
fusion could in principle fall back to one modality when the other drops out; the implementation
has no such fallback (every block is required at prediction).

## 5. Comparison table

| Configuration | Feature set / model |
|---------------|---------------------|
| EMG only | `emg` |
| IMU only | `imu` |
| Early fusion | `emg+imu` (also `emg_nodig+imu`; biomech names `emg_imu`, `emg_nodig_imu`) |
| Early fusion, sequences | `seq+seq_imu` with `gru` |
| Late fusion | `emg+imu` with `late_mean:`, `late_weighted:`, `late_stack:` |

Report deltas against EMG only under the same protocol and seed. The biomech stage `ml_eval`
runs the early-fusion rows (LDA, SVM, MLP) and, as an exploratory variant, an SVM late fusion
with a weight chosen on inner folds ([`ml05`](ml05_evaluation.md) §6.1); its
`fusion_diagnostics` table quantifies how much each EMG feature encodes the participant
([`ml05`](ml05_evaluation.md) §6).

## 6. Open questions

- Should IMU drift be corrected against optical before feature extraction ([`s05`](../software/s05_imu_processing.md) §5)?
- Is a shared-encoder (intermediate) fusion worth a fourth strategy?
- Optical data stays out of the model inputs: it is the label source.
