<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# IMU-Only Motion Estimation Quality

**Goal:** Assess raw IMU data for position vs. orientation.

Most pilot takes have only the **cranial** IMU (`imu_mta` empty). Then:
CRA acceleration / orientation against `pose_cra`. When both slots are present
(s01 take 1): relative kinematics `q_rel = conj(q_cra) ⊗ q_mta`.

`R_mount` comes from the newest `imu_mounting.json` of `5_2_2_imu_optical_handeye` (also when the run
does not meet the 2° criterion — that is the calibration that exists), searched
with `jtsmm.imu.persist.discover_runs`/`load_run` under `--mounting-dir`
(default: `session.imu_calibration_recordings_dir` of the operator config, otherwise
`~/experiments/5_2_2_imu_optical_handeye/recordings`). The CSV columns
`R_mount_mta`/`R_mount_cra` say whether a calibration was found;
`provenance.json` names the file.

## Position (naive)

Trapezoidal double integration of the linear acceleration (the raw value
`accel` was recorded **including** gravity; it is removed with the game-rotation-vector
orientation, `jtsmm.imu.gravity_lin_accel.linear_accel_from_raw`) **without** ZUPT and without
`auto_reset`. Only the **displacement magnitudes** are compared (yaw-invariant,
IMU world and optical world need not agree):

$$
e(t)=\bigl|\,\|p_{\mathrm{IMU}}(t)-p_{\mathrm{IMU}}(t_0)\|
-\|p_{\mathrm{opt}}(t)-p_{\mathrm{opt}}(t_0)\|\,\bigr|
$$

Windows 0.5 / 1 / 2 s: $\bigl|\,\|Δp_{\mathrm{IMU}}\|-\|Δp_{\mathrm{opt}}\|\,\bigr|$.
Fit $e\approx c t^{2}$ on the `open_close` phase.

## Plot (`imu_only_*.png`)

Four panels, no longer only the error over the whole take (the log scale
made the first seconds invisible):

- **top:** first 8 s, IMU and optical **on top of each other** (tracking).
  Position as $|p-p_0|$ in mm; orientation as $|ΔR|$ since $t_0$
  plus the geodesic error (dashed).
- **bottom:** whole take. The optical position stays cm-sized, the IMU position
  runs away; the orientation stays in the range of degrees, the error grows
  more slowly than the position.

CSV in addition: `pos_err_{1,2,5}s_m`, `orient_rms_{5,10}s_deg`.

## Orientation

Not the absolute angle of IMU world vs. optical world (the game RV has no
common yaw). Metric: geodesic angle between the
**changes since t0**

$$
q_{\Delta}(t)=\overline{q(t_0)}\otimes q(t)
\qquad
\theta(t)=\angle\!\bigl(q_{\Delta,\mathrm{IMU}}(t),\,q_{\Delta,\mathrm{opt}}(t)\bigr)
$$

after the hand-eye ``R_mount``. Constant frame/yaw offsets drop out.

```bash
python experiments/5_4_4_imu_only_motion_quality/eval_imu_only.py
python .../eval_imu_only.py --subject s01 --mounting-dir /path/to/5_2_2/recordings
```

The first path is relative to the repository root.
Call, data paths (`--data-dir`, `--out-dir`), `--subject` and `provenance.json`: see
[Experiments › Running an evaluation script](../README.md#running-an-evaluation-script).


`~/experiments/5_4_4_imu_only_motion_quality/results/imu_only_metrics.csv`, `imu_only_*.png`, NPZ under `~/experiments/5_4_4_imu_only_motion_quality/results/derived/`.
