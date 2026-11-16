<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# Marker Size Trade-off

## Goal

Empirical justification for the 30 mm tag choice: triangulation detection
rate and pose stability across marker sizes at near / mid / far working
distances. PnP is out of scope here (`5_1_1_pnp_vs_triangulation` already compared methods).

## Recordings

```yaml
experiment:
  name: marker_plate_run1
  folder: 5_1_2_marker_size_tradeoff
  condition: near
```

Use a plate with multiple ArUco sizes (IDs/sizes in `study.yaml`). Capture:

```bash
optical track --persist-frames                    # uses experiment.condition
optical track --persist-frames --condition mid
optical track --persist-frames --condition far
```

Each leaf includes `calibration_bundle.h5` + `optical.yaml` snapshots from capture
time (used automatically by `eval_marker_size.py`).

## Eval

```bash
python experiments/5_1_2_marker_size_tradeoff/eval_marker_size.py
```

The path above is relative to the repository root.
Runs from any directory with the venv that has `software/optical` installed
(helpers: `optical.evaluation.recordings`, tracker:
`optical.tracking.tracker.load_tracker_for_recording`).

Report: `~/experiments/5_1_2_marker_size_tradeoff/results/marker_size.json` (triangulation detection rate +
translation/rotation jitter per size × distance).
