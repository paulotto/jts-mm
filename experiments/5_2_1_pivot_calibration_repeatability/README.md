<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# CT Pivot Calibration Repeatability

**Goal:** Stability and repeatability of the pivot calibration (tip offset $t_{\text{tip}}$ and OLS residuals over independent repetitions).

## Data

Persisted runs (default of `session.ct_calibration_recordings_dir`):

```text
~/experiments/5_2_1_pivot_calibration_repeatability/recordings/{slug}_{timestamp}/
  ct_tip_calibration.json
  poses.json
  meta.json
  calibration_bundle.h5   # snapshot of the optical calibration at capture time
  optical.yaml            # snapshot of the optical config (mounts/markers)
  frames/
```

Created with:

```bash
jtsmm ct-calibrate --persist-ct-calibration
# optional: session.experiment.name = ct_tip_calibration_N
```

## Analysis / Eval

From any directory (the script finds `analyze_repeatability.py` next to itself); the path
of the `cd` is relative to the repository root:

```bash
cd experiments/5_2_1_pivot_calibration_repeatability   # optional
python eval_repeatability.py                 # stored tip JSONs
python eval_repeatability.py --from-poses    # OLS again from poses.json
python eval_repeatability.py --from-frames   # re-track with the leaf bundle, then OLS
```

(`analyze_repeatability.py` stays as an alias for the default path.)

Output in `~/experiments/5_2_1_pivot_calibration_repeatability/results/`:

| File | Contents |
|-------|--------|
| `repeatability_report.txt` | Summary (tip scatter + OLS residuals) |
| `repeatability_summary.json` | Machine-readable metrics |
| `repeatability_runs.csv` | One row per calibration run |
| `repeatability.png` | Scatter / Δmean / rms bars |

Options: `--input`, `--out`, `--no-plot`, `--optical-config`, `--from-poses`, `--from-frames`.

`--from-frames` needs `calibration_bundle.h5` in every leaf (new persist runs). Old
takes without a snapshot can only be evaluated through the tip JSON / `--from-poses`.
