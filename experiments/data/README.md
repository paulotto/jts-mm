<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# Data checks and pilot figures

Per-participant plausibility checks and exploratory figures over recorded
takes. Nothing here writes into a session folder.

| Script | What it does | Output (default) |
|---|---|---|
| `check_subject.py <sNN[_sex_age]>` | PASS/WARN/FAIL checks of impedance, EMG (`EmgPrep`, masked), trajectory and IMU for every motion take of one participant, each against the MVC take before it. Exit 0 = no FAIL, 1 = a FAIL, 2 = no take | stdout |
| `eval_trajectory_quality.py` | rest scatter, valid-frame share, pose-fit RMS and cycle repeatability of the JTS landmark, also with the IMU-fused cranial rotation | `<results>/data/trajectory_quality/trajectory_quality.{json,md}` |
| `preview_imu_rotation.py` | optical vs IMU-fused cranial rotation, 3D jaw path per participant | `<results>/data/imu_rotation_preview/<subject>.png` |
| `plot_mp_in_hp_all.py` | mandibular-marker origin in model axes, one panel per participant | `<results>/data/trajectory_quality/mp_in_hp_all_subjects.png` |
| `plot_tmj_subjects.py [session ...]` | both condyles per participant | `<results>/data/<subject>/tmj/tmj_left_right.png` |
| `run_jts_plots.py <session>` | EMG plots, JTS analysis and the 3D plotly path of one take, each step run with this interpreter (`sys.executable`) | `<results>/data/<subject>/jts/` |

`<results>` is `$BIOMECH_RESULTS_DIR`, else `~/experiments`.

## Running

From any directory, with the Python of the venv that has `jtsmm`,
`jaw_tracking` and `biomech` installed (on the Pi: `scripts/install_pi.sh`,
which passes `--analysis` by default). The paths below are relative to the
repository root:

```bash
python experiments/data/check_subject.py s03
python experiments/data/eval_trajectory_quality.py --subject s01,s02 --out-dir /tmp/tq
```

- Sessions: the scripts need recorded sessions, which are not part of the repository.
  They are searched in `--data-dir` > `$JTSMM_DATA_DIR` > `<repo>/data/pilot_study/jtsmm_data` >
  `~/jtsmm_data` (`biomech.paths.resolve_data_dir`). `check_subject.py` also looks in
  `<data-dir>/tmp` and one level of sub-folders.
- `--subject` matches participants **exactly** (`s03` or `s03_m_33`, comma-separated);
  `s1` no longer matches `s10`.
- One take per participant: the **latest** motion take, as before (`--take first` for
  the pre-specified first take of an01 §2).

## Changes against the earlier version

- Session discovery is `biomech.cohort.discover_sessions` (well-named folders only);
  the hard-coded session list of `preview_imu_rotation.py` is gone.
- Cycle time normalisation is `biomech.kinematics.cycles.time_normalize`: a cycle whose
  first or last sample is untracked now gets NaN edges instead of the clamped edge
  value, so `eval_trajectory_quality.py` drops it from the curve average (pilot
  results unchanged).
- `plot_mp_in_hp_all.py` grows the grid with the participant count (the old fixed
  2 x 4 grid failed with more than eight participants).
- `plot_tmj_subjects.py` no longer deletes `trajectory_quality/tmj_left_right_all_subjects.png`.
- `run_jts_plots.py` needs `plotly` and `kaleido` for its last step
  (`jts.plotly_visualization`); neither is part of `biomech[plot,stats]`.
