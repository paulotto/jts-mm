<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# Inter-Subject Variability (SPM)

**Goal:** Time-point-specific differences between participants on
time-normalised cycles (no box plot per sample).

## Input

Kinematics: `cycles_*.json` from `5_4_2_motion_trajectory_analysis` (`--cycles-dir`, default
`<results root>/5_4_2_motion_trajectory_analysis/results/derived`). EMG:
the same optical cycle windows; %MVC directly from `biomech.emg.prepare.EmgPrep`
(masked envelope / p95 MVC of the first MVC take), so the same chain as
`5_4_1_emg_properties` — `mvc_scales.json` is no longer read (`--mvc-json` is dropped).
An EMG cycle wave with masked edges (not finite after warping) cannot
enter the SPM; it is counted in `emg_incomplete_waves`, not silently
discarded.

## Three evaluations

1. **Cycle-as-rep SPM** (`spm_fields.png`) — `spm1d.stats.anova1` over subjects,
   α = 0.05, every cycle as an independent repetition. The pilot n is small;
   this overestimates the significance. Beside it **amplitude-normalised**
   (`spm_fields_ampnorm.png`): every cycle $(y-\min)/(\max-\min)$, so that
   the shape is tested separately from the ROM.

2. **Participant as the unit** (`spm_subject_unit.png`, `_ampnorm`) — one
   mean cycle per participant. Left: mean ± SD of the cycles *within* the
   participant; right: **between-subject SD** of the three means (no F test
   on n=1 per group).

3. **EMG %MVC** — the same optical peak-to-peak windows, 150 ms RMS / MVC.
   Cycle-as-rep like the kinematics: `spm_emg_bars_closers.png` /
   `spm_emg_bars_openers.png` (grey bars) plus F panels
   (`spm_emg_fields_*`). Amplitude-normalised: `*_ampnorm.png`.
   Subject-unit stays `spm_emg_mvc.png`.

**Cycle:** Within a named phase (`open_close`, …) on the
1-D opening $\|p-p_\mathrm{rest}\|$ (`biomech.kinematics.peak_cycles`, as in `5_4_2_motion_trajectory_analysis`). Peaks with `scipy.find_peaks`
(minimum distance 0.4 s, prominence $0.25\,\sigma$). One cycle is
**peak-to-peak** (maximum → next maximum), linearly on 101 samples
(0–100 %). EMG is cut to the same time windows.

**Grey bars** (Pataky style): intervals with SPM-$F > F^*$ (α=0.05),
cycle-as-rep ANOVA. 1×4 paper layout: `spm_bars_ampnorm_row.png`,
`spm_emg_bars_closers_ampnorm_row.png`.

```bash
python experiments/5_4_3_inter_subject_variability/eval_spm.py
python .../eval_spm.py --cycles-dir /tmp/542/derived --subject s01,s02,s03
```

The first path is relative to the repository root.
`spm1d` comes with `biomech[stats]` (installed by `scripts/install_pi.sh`); no
separate `pip install spm1d`. When spm1d aborts (e.g. zero variance at a
node), the reason is in the block (`ok: false`, `reason`).

Call, data paths (`--data-dir`, `--out-dir`), `--subject` and `provenance.json`: see
[Experiments › Running an evaluation script](../README.md#running-an-evaluation-script).


`~/experiments/5_4_3_inter_subject_variability/results/spm_summary.json` contains all blocks (`cycle_reps_mm`,
`subject_unit_*`, `emg_mvc_*`).
