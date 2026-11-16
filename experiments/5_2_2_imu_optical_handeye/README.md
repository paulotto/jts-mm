<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# IMU–Optical Hand-Eye Calibration Repeatability

**Goal:** Stability and repeatability of the mounting rotation $R_{\mathrm{mount}}$
over independent repetitions of the same rigid IMU–array pair (analogous to the
tip-offset scatter + OLS residuals of `5_2_1_pivot_calibration_repeatability`).

In operation, **one** calibration after every remount (glued again / array exchanged) is enough.
The takes here are the repetition study, not one calibration per participant.

## Data

Persisted runs (default of `session.imu_calibration_recordings_dir`):

```text
~/experiments/5_2_2_imu_optical_handeye/recordings/imu_mounting_{N}_{timestamp}/
  imu_mounting.json       # R_mount, n_pairs, rms_deg, run
  session.h5              # optical poses + IMU quats (for --from-session)
  meta.json
  calibration_bundle.h5   # optional snapshot of the optical calibration
  optical.yaml            # optional snapshot of the optical config
```

Created with:

```bash
jtsmm imu calibrate --mode record --persist-imu-calibration
# optional: --run N   (otherwise existing leaves + 1)
```

## Analysis / Eval

From any directory (`eval_repeatability.py` imports `eval_handeye.py` next to itself); the path
of the `cd` is relative to the repository root:

```bash
cd experiments/5_2_2_imu_optical_handeye   # optional
python eval_handeye.py                 # stored imu_mounting.json
python eval_handeye.py --from-session  # Park–Martin again on session.h5
```

(`eval_repeatability.py` is the same script.)

Output in `~/experiments/5_2_2_imu_optical_handeye/results/`:

| File | Contents |
|-------|--------|
| `repeatability_report.txt` | $R_{\mathrm{mount}}$ scatter + fit residuals |
| `repeatability_summary.json` | Machine-readable metrics |
| `repeatability_runs.csv` | One row per slot × run |
| `repeatability.png` | Rotvec scatter + grouped MTA/CRA bars (Δmean, RMS) |

Options: `--input`, `--out`, `--no-plot`, `--from-session`.
