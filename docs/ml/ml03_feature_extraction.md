<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml03 — Windows, labels and features

How a take becomes rows of a `WindowDataset` (`ml/core/windows.py`, `ml/core/features.py`,
`ml/core/dataset.py`). `ml dataset` and the biomech stage `ml_windows` both call
`build_windows`.

> **Status:** implemented and vectorized (one `sliding_window_view`, one FFT per window; about 7 s
> for a pilot take including file I/O and the causal filter, against 15 s for the former
> per-window loop).

## 1. Inputs

`ml dataset` reads the take (`ml/io/session.py`) and its sidecar, keeps the experiment phases that
contain labelled samples plus the `rest` phase (always: unlabelled it gives no training windows
but sets the ZC/SSC/WAMP thresholds and the D13 statistics), and takes the label samples inside
each phase (255 → −1). A take without `/phases` is one segment over the labelled span
(`full_session`). The biomech stage `ml_windows` selects the same phases (labelled phases of the
protocol plus `rest`). The subject
id is, in order: the sidecar's `subject_id`, `/meta/subject_id`, the participant id parsed from the
folder name (`session_<pid>_<sex>_<age>_...`), `/meta/experiment_name`; if none exists the build
fails (a constant "unknown" would make leave-one-subject-out impossible).

```bash
ml dataset [SESSIONS...] [--labels FILE | --labels-dir DIR] [--data-dir DIR] [--mvc FILE]
           [--label-at end|majority] [--skip-errors] [--config FILE] [--output DIR]
```

Without `SESSIONS`, `--labels-dir` pools every sidecar in it (`<session folder>_labels.h5`);
with neither, `session_*/` folders under `--data-dir` (default `paths.session_data_dir`) are
used. The output defaults to `paths.processed_dir/<session_id or pool_...>`.

## 2. Windows

| Setting (`dataset.window`) | Default | Meaning |
|---|---|---|
| `window_s` / `stride_s` | 0.2 / 0.05 | trailing window `(t − W, t]`, 75 % overlap |
| `warmup_s` | 2.0 | causal filter warm-up before each phase |
| `gap_factor` | 1.1 | a window whose first and last sample are more than 1.1 W apart (recording gap) is dropped |
| `max_masked` | 0.2 | a window with more than 20 % masked samples on any channel is dropped |
| `label_at` | `end` | `end` or `majority` (§3) |
| `purity_min` | 0.6 | `majority` only |
| `label_max_age_s` | 0.1 | `end`: a label sample older than this is no label |
| `thr_sigma` | 1.0 | ZC/SSC/WAMP threshold = 1 SD of the rest-phase **sample differences** (§4) |
| `emg_norm` | `rest_relative` | EMG features relative to the rest phase (D13, §4.1) \| `mvc` (% of the reference only) |
| `imu_hz` | 200 | IMU resampling grid |
| `seq_subframes` | 10 | GRU sub-frames per window |

Windows are generated per phase, so no window crosses a phase edge, and they are placed by sample
count but admitted by duration (`gap_factor`). The window's time is its last sample's
`t_master_us` (`t_end_us`).

## 3. Window labels

- `end` (default): the label in force at `t_end`, i.e. the last label sample at or before the
  window end. This is the question an online decoder answers at that instant.
- `majority`: the label in force for the longest time inside the window (labels hold from one
  label sample to the next); dropped below `purity_min` or when any part is unlabelled.
- `purity` (share of the window's duration whose label equals the assigned one) is stored per row
  in both modes as a diagnostic.

Rejection counters per take: `masked`, `label` (−1 / unlabelled / too old), `gap`, `nonfinite`.

## 4. Feature registry

One registry (`ml.core.features.REGISTRY`) is the dispatch: configs name registry keys and an
unknown key is an error. Every function works on `(..., L)` blocks, so the same code computes
whole windows and GRU sub-frames.

| Key | Definition |
|-----|------------|
| `mav`, `log_mav` | mean \|x\| (% ref), its log |
| `rms`, `log_rms` | root mean square, its log |
| `wl`, `log_wl` | mean \|Δx\| (waveform length per sample), its log |
| `dasdv` | RMS of Δx |
| `log_ssi` | log Σx² |
| `zc` | rate of sign changes `x[k]·x[k+1] < 0` with \|Δx\| ≥ thr (Hudgins) |
| `ssc` | rate of slope-sign changes `Δx[k]·Δx[k+1] < 0` with either \|Δx\| ≥ thr |
| `wamp` | rate of \|Δx\| ≥ thr (Willison amplitude) |
| `band_<lo>_<hi>` | relative power in [lo, hi) Hz of the 20-450 Hz power (Hann, one FFT) |
| `mnf`, `mdf` | mean / median frequency in 20-450 Hz (Hz/100) |
| `log_ttp` | log total 20-450 Hz power |

Defaults: EMG `log_mav, log_rms, log_wl, zc, ssc, wamp, band_20_50, band_50_100, band_100_200,
band_200_450, mnf` (11 × 6 = 66 columns, names `<muscle>|<feature>`); sub-frames
`log_mav, log_rms, log_wl, zc`.

All three count features threshold sample differences, so `thr` (per channel) is `thr_sigma` ×
the SD of `np.diff` of the filtered rest phase (`features.rest_thresholds`; first phase if a take
has no rest phase), masked samples ignored. Erratum (2026-10-16): the threshold was the
*amplitude* SD; at 4 kHz the difference SD of 20-450 Hz EMG is ~0.4 × the amplitude SD, so the
threshold hardly ever fired at rest and the three features were near-constant.

IMU (25 columns, [`ml06`](ml06_sensor_fusion.md) §2): mean and SD of 12 signals plus the mean
mandible angular speed.

### 4.1 Label-free rest-relative normalization (D13, `emg_norm: rest_relative`)

Every EMG feature is expressed relative to the take's own unlabelled `rest` phase (one take per
participant in the biomech stage). The statistics use every rest-phase window that is not
over-masked, gap-spanning or non-finite, **whatever its label**; labels are never read.

| Kind (`features.feature_kind`) | Features | Transform |
|---|---|---|
| amplitude, already log | `log_mav`, `log_rms`, `log_wl`, `log_ssi`, `log_ttp` | `f − median_rest(f)` |
| amplitude, linear | `mav`, `rms`, `wl`, `dasdv` | `log(f) − median_rest(log f)` |
| count / spectral | `zc`, `ssc`, `wamp`, `mnf`, `mdf`, `band_*` | `(f − mean_rest f) / SD_rest f` (SD floored at 1e-3) |

Amplitude features become invariant to a per-participant gain (electrode impedance, skin,
reference), count and spectral features to a participant's rest noise level; rest windows have
median 0 (amplitude) or mean 0 (count / spectral). The same transform, with statistics over the
rest sub-frames, applies to the `seq` block. Centres and scales are stored per take
(`metadata.json` / `meta["norm"]`; biomech arrays `norm_<block>_{center,scale}`). A take without
a rest phase is an error under `rest_relative`; `emg_norm: mvc` keeps the features in % of the
reference (comparison). IMU features are unchanged.

## 5. Blocks and feature sets

| Block | Shape | Content |
|-------|-------|---------|
| `emg` | (n, 66) | EMG window features |
| `imu` | (n, 25) | IMU window features |
| `seq` | (n, 10, 24) | EMG sub-frame features (GRU) |
| `seq_imu` | (n, 10, 12) | IMU sub-frame means (GRU early fusion) |

A feature set joins blocks with `+` (`emg+imu`, `seq+seq_imu`); `emg_nodig` is the EMG block
without the digastrics. Historical names `emg_imu`, `emg_nodig_imu`, `seq_emg_imu` are aliases
(the biomech stages use them).

## 6. Dataset file

`ml dataset` writes `<out>/dataset.npz` and `metadata.json`:

| Array | Content |
|-------|---------|
| `X_emg`, `X_imu`, `X_seq`, `X_seq_imu` | blocks (float32) |
| `y` | class 0..6 |
| `pid`, `session_id`, `phase` | subject, take, phase name per row |
| `t_end_us` | window end on the master clock |
| `groups` | label-run id (consecutive windows with the same label and phase), unique across takes |
| `purity` | §3 |

`metadata.json`: feature names per block, class counts, subjects, sessions, rejection counters,
reference, its source and chain, ZC/SSC/WAMP thresholds, `emg_norm` and the D13 statistics
(`norm`, `n_rest_windows`), measured mains frequency, window settings, config hash, source paths
and git commit. Datasets from the former builder (`X_feat`) are refused with a rebuild
message.
