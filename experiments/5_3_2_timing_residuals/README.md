<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# Timing Residuals

**Goal:** Accuracy of the reconstructed camera time base.

## Residual

After an accepted trigger association, identical to `ClockAligner._update_fit`
(window 30):

$$
r_f = T_{\mathrm{trigger}} - (a \cdot t_{\mathrm{Sensor},µs} + b)
\qquad
\sigma_c(t) = \mathrm{RMS}(r)\ \mathrm{in\ the\ running\ window}
$$

Only rows with `t_master_us > 0`. Source: `/cameras/<id>/{sensor_timestamp_ns,t_master_us}`.

Two evaluations:

- **full take** — all associated frames (including the camera start; the RMS can come
  from a few millisecond spikes).
- **named phases** — only samples whose `t_master` lies in a named
  phase interval (calibration points, rest, open_close, …). The startup drops
  out.

## Evaluation

```bash
python experiments/5_3_2_timing_residuals/eval_timing_residuals.py
python .../eval_timing_residuals.py --example session_s03_m_33_20250306T110000Z_00000001
```

The first path is relative to the repository root.
Call, data paths (`--data-dir`, `--out-dir`), `--subject` and `provenance.json`: see
[Experiments › Running an evaluation script](../README.md#running-an-evaluation-script).

Takes: all motion takes with cameras (folder name without `_mvc`).

`~/experiments/5_3_2_timing_residuals/results/derived/residuals_<session>_<cam>.npz` stores `t_master_us`,
`residual_us`, `sigma_c_us`, `in_phase`. Plots: `residual_hist.png`,
`sigma_c_example.png` (full take) and `residual_hist_phases.png`,
`sigma_c_example_phases.png` (named phases). `residual_summary.json` has
`pool` and `pool_phases`.
