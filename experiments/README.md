<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# Experiments

Study configurations, thin evaluation scripts and their READMEs. The analysis code lives in
the packages (`software/biomech`, `software/jtsmm`, `software/optical`, `software/ml`); a
study folder only says what to run on which sessions. Results are written outside the
repository (see [below](#running-an-evaluation-script)).

## Studies

| Folder | Content |
|---|---|
| [`icorr2027/`](icorr2027/README.md) | Dataset study: the JNER data-set paper (analysis plan, paper set, in-text numbers) |
| [`jtsmm_validation/`](jtsmm_validation/README.md) | System validation study: the IEEE Sensors systems paper |
| [`data/`](data/README.md) | Data checks and pilot figures |

## Evaluations of the recording system

| Folder | Topic |
|---|---|
| [`5_1_1_pnp_vs_triangulation/`](5_1_1_pnp_vs_triangulation/README.md) | Precision and trueness of the two-marker array on a ladder |
| [`5_1_2_marker_size_tradeoff/`](5_1_2_marker_size_tradeoff/README.md) | Marker size trade-off |
| [`5_2_1_pivot_calibration_repeatability/`](5_2_1_pivot_calibration_repeatability/README.md) | CT pivot calibration repeatability |
| [`5_2_2_imu_optical_handeye/`](5_2_2_imu_optical_handeye/README.md) | IMU–optical hand-eye calibration repeatability |
| [`5_3_1_trigger_frame_assoc_stats/`](5_3_1_trigger_frame_assoc_stats/README.md) | Trigger–frame association statistics |
| [`5_3_2_timing_residuals/`](5_3_2_timing_residuals/README.md) | Timing residuals of the camera time base |
| [`5_4_1_emg_properties/`](5_4_1_emg_properties/README.md) | EMG signal properties |
| [`5_4_2_motion_trajectory_analysis/`](5_4_2_motion_trajectory_analysis/README.md) | Motion amplitude and cycle consistency |
| [`5_4_3_inter_subject_variability/`](5_4_3_inter_subject_variability/README.md) | Inter-subject variability (SPM) |
| [`5_4_4_imu_only_motion_quality/`](5_4_4_imu_only_motion_quality/README.md) | IMU-only motion estimation quality |

The READMEs of the folders `5_3_*` and `5_4_*` share the conventions below.

What the papers use of these evaluations was ported into `biomech` after a review of each
method (change-log entry J14 of the dataset study): the acquisition quality
of the dataset (X11, an20) and, in the validation study, the association under injected faults,
the calibration repeatability and single camera vs stereo (VD26–VD28, an14 §11e–§11h). The
folders here are the earlier versions and are kept as they are.

## Running an evaluation script

The `eval_*.py` scripts of the folders `5_3_*` and `5_4_*` run from any working directory with the Python of
the venv that has `jtsmm`, `jaw_tracking` and `biomech` installed (on the Pi:
`scripts/install_pi.sh`, where `--analysis` is the default). No `cd`, no `sys.path`.

- **Sessions:** the scripts need recorded sessions, which are not part of the repository.
  They are searched in `--data-dir` > `$JTSMM_DATA_DIR` > `<repo>/data/pilot_study/jtsmm_data` >
  `~/jtsmm_data` (`biomech.paths.resolve_data_dir`).
- **Output:** `--out-dir` > `$BIOMECH_RESULTS_DIR/<study>/results` >
  `~/experiments/<study>/results`.
- **`--subject`** filters by participant **exactly** (`s03` or `s03_m_33`, several separated
  by commas); `s1` does not match `s10`.
- **Provenance:** `derived/provenance.json` holds the git SHA (with a dirty flag), the package
  versions, `argv` and the takes.
