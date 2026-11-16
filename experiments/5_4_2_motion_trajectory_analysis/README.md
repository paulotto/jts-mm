<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# Motion Trajectory Analysis

**Goal:** Movement amplitude and rep-to-rep consistency from the kinematics.

## Trajectory

1. An existing `jaw_motion.h5` next to the session (or under `<out-dir>/derived/<session>/`
   after `--run-jts`; runs with the same Python, `sys.executable`).
2. Fallback: optical relative pose, mandible origin in the cranial marker frame
   (`R_craᵀ (t_mta − t_cra)`, mm).

JTS is **not** written into the raw file with `--append-session`.

Opening = `‖p − p_rest‖` (`biomech.kinematics.traces`). Cycles:
**peak-to-peak** (`biomech.kinematics.peak_cycles.peak_to_peak_cycles`, the
definition of the earlier evaluation, kept on purpose for comparability; not
`biomech.kinematics.cycles.detect` of the dataset-study pipeline), time-normalised to
0–100 % (101 samples, `cycles.time_normalize`). Metrics: ROM, RMS of the cycles
against the mean cycle, CV of the cycle amplitudes.

**NaN (new):** Gaps are bridged linearly for the peak search; a peak on
a gap does not count. The time normalisation interpolates over inner gaps.
`n_cycles_gappy` (CSV) and `cycle_nan_frac` (JSON) mark cycles with > 10 %
missing samples instead of discarding them. Pilot: only s13 changes
(chewing 32 → 34 cycles); all other takes are identical.

## Evaluation

```bash
python experiments/5_4_2_motion_trajectory_analysis/eval_motion_trajectories.py
# optional, slow:
python .../eval_motion_trajectories.py --run-jts
```

The first path is relative to the repository root.
Call, data paths (`--data-dir`, `--out-dir`), `--subject` and `provenance.json`: see
[Experiments › Running an evaluation script](../README.md#running-an-evaluation-script).


`~/experiments/5_4_2_motion_trajectory_analysis/results/trajectory_metrics.csv`, `amplitude_bars.png`, `cycle_overlays.png`,
`cycle_xy_paths.png` (2D path, cycles coloured; plane = the two axes with the
largest ROM), `cycle_xy_example.png` (one session, 2×2),
`~/experiments/5_4_2_motion_trajectory_analysis/results/derived/cycles_<session>.json` (input for `5_4_3_inter_subject_variability`).
