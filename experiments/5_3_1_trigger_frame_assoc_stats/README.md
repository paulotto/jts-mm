<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# Trigger–Frame Association Statistics

**Goal:** Show the robustness of the cadence association on real sessions. The cursor
jumps by as many trigger periods as have passed in the SensorTimestamps of the
camera. Predict-then-match remains only the fallback when this
interval is not a plausible drop burst.

## Data

All motion takes with cameras and a sync report in the data directory
(`biomech.io.inventory.scan`; the old manifest `_lib/session_index.json` has been
removed). The evaluation **does not write** into `session.h5`.

Live counters exist only in the end snapshot `session.h5:/sync.attrs.report_json`
(`ClockAligner.sync_report`). There are no per-frame labels.

## Metric definitions

| Mode | Operationalisation |
|------|---------------------|
| cadence | `cadence_advance / n_written` (trigger periods that the camera skipped) |
| missing trigger | `cadence_missing_trigger / n_written` (predicted id whose TRIGGER_EVENT is missing) |
| predict-then-match | `(camera_drop_recovered − cadence_advance) / n_written`, at least 0. Cadence writes into the same counter; the rest is the fit fallback |
| slip | `slip_count / n_written` (id jumps, as in the aligner) |
| hold | `waiting_for_trigger / n_ingest_attempts` with `attempts ≈ written + drops + leftover` (poll events, no frame label) |
| nominal | `clip(1 − cadence − predict − slip, 0, 1)` on written frames (lower bound) |
| reject / fit-reset | `validation_fail_count`, `validation_relock_count` (`validation_relock_after = 8`) |
| unstamped | `drops / (written + drops)`; uint64 zero stamps separately; optical NaN only when `t_master_us` is float |

MVC takes are left out.

## Evaluation

```bash
python experiments/5_3_1_trigger_frame_assoc_stats/eval_assoc_stats.py
python .../eval_assoc_stats.py --data-dir ~/jtsmm_data --subject s03 --no-plot
```

The first path is relative to the repository root.
Call, data paths (`--data-dir`, `--out-dir`), `--subject` and `provenance.json`: see
[Experiments › Running an evaluation script](../README.md#running-an-evaluation-script).


Output in `~/experiments/5_3_1_trigger_frame_assoc_stats/results/`: `assoc_sessions.csv`, `assoc_summary.json`,
`assoc_shares.png`, plus `~/experiments/5_3_1_trigger_frame_assoc_stats/results/derived/provenance.json`.
