<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# EMG Signal Properties

**Goal:** SNR and MVC-normalised amplitude of the six muscles.

## Pipeline

`biomech.emg.prepare.EmgPrep` (the same chain as the dataset-study pipeline):
`remove_mains` at the mains frequency measured per take (`take_line_hz`) →
band-pass 20–450 Hz (zero-phase) → RMS envelope 150 ms. Artifacts (rail,
lead-off, envelope > 3× MVC, movement low-frequency) are **masked** (NaN),
not zeroed. Phases from `/phases`. Every motion take uses the **first**
MVC take of the same participant (an01 §2).

| Metric | Definition |
|--------|------------|
| SNR | `std(activation) / std(rest)` after cleaning (s09), without samples of the physical rules (rail, lead-off, low-frequency) |
| mains_share | before/after removal, per phase |
| MVC | 95th percentile of the masked envelope; digastric ← `mvc_opening`, masseter/temporalis ← `mvc_closing` |
| %MVC | masked envelope / MVC in motion phases (`pct_mvc_p50/p95`); sensitivity with the physical rules only: `pct_mvc_*_physmask` |
| Flags | `mvc_below_recording_max` + `recording_max_uv`; `mask_frac`, `mask_frac_<rule>` |

### Changes in behaviour against the earlier `_lib` version

- **Mains fix:** mains frequency measured once per take and the corrected
  final block of `remove_mains` — both through `biomech.emg`.
- **Masks instead of spike zeroing:** `suppress_envelope_outliers` (spikes > 150 µV
  to 0) is dropped; masked samples count in no statistic.
- **MVC fallback as a flag:** When the MVC is below the channel maximum in the
  motion take, it stays the denominator. The row gets
  `mvc_below_recording_max = True` and `recording_max_uv`; before, the maximum
  was substituted silently (`mvc_source = recording_max`). Pilot s01: digastric
  MVC 25/30 µV instead of 117 µV → %MVC p50 ≈ 30–100 % instead of 9–23 %.
- `mvc_scales.json` now contains only real MVC values (`null` when missing).

**Plots:** The main statement is `pct_mvc.png` (MVC µV scales at the top, %MVC
closers vs. openers at the bottom). In addition per muscle (`pct_mvc_by_muscle.png`).
The old box over all channels is in `pct_mvc_box_pooled.png`
(appendix — closers and openers mixed).

## Commercial system

The old files of the commercial system (only ML features and EMG windows, no raw EMG)
are not part of the repository.
`--legacy` is skipped as long as no session-like raw data are passed.

## Evaluation

```bash
python experiments/5_4_1_emg_properties/eval_emg_properties.py
python .../eval_emg_properties.py --data-dir ~/jtsmm_data --subject s01,s03
```

The first path is relative to the repository root.
Call, data paths (`--data-dir`, `--out-dir`), `--subject` and `provenance.json`: see
[Experiments › Running an evaluation script](../README.md#running-an-evaluation-script).


`~/experiments/5_4_1_emg_properties/results/emg_properties.csv`, `snr_heatmap.png`, `pct_mvc.png`,
`mvc_uv_bars.png`, `pct_mvc_by_group.png`, `pct_mvc_by_muscle.png`,
`pct_mvc_box_pooled.png`, `~/experiments/5_4_1_emg_properties/results/derived/mvc_scales.json`.
