<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: MIT
-->

# Precision and trueness (two-marker array on a ladder)

## Goal

Measure how precisely and how truly the rig localises a two-marker array,
and compare single-camera PnP with multi-camera triangulation.

Precision is the spread of a static pose about its own mean. Trueness is how
closely the reconstructed slot spacing matches a hand-measured ladder. The
array is its own rigid body (`study.yaml`: `array_body`). The CRA (HP) is not
used.

World frame is cam0: +x right, +y down, +z depth.

## Hardware

- A rigid array of two markers, defined in `optical.yaml` under
  `markers.rigid_bodies` / `id_ranges` / `mounts_T_body_tag` as `array_body`
  (name `eval_array`).
- A ladder the array plugs into at hand-measured spacings. Enter those
  ruler values in `study.yaml` `ladder.slots` (`section` + `mm`). The
  `mm: 40` style entries shipped here are placeholders.

## Precision

Hold the array still for ~30 s at near / mid / far. Same metric as before
(`r_mean±σ_r` about the temporal mean, first 30 s), on the array pose.

```yaml
experiment:
  name: precision_test_1
  folder: 5_1_1_pnp_vs_triangulation
  condition: near          # mid / far
  use_experiment_plan: false
```

```bash
optical track --persist-frames --condition near
optical track --persist-frames --condition mid
optical track --persist-frames --condition far
```

## Trueness

One recording per ladder slot. Each take walks an experiment plan whose
measurement section is that slot (`d0`, `d40`, …). Landmark phases that the
plan inserts ahead of the section are ignored; only the slot section is
scored.

Place the ladder once **lateral** to the camera (along depth, world z) and
once **frontal** (left–right, world x). Repeat each placement at **near**,
**mid**, and **far**. The condition name carries all three:

`{near|mid|far}_{lateral|frontal}_{section}` → e.g. `near_lateral_d40`,
`far_frontal_d120`.

```yaml
experiment:
  name: trueness_test_1
  folder: 5_1_1_pnp_vs_triangulation
  condition: near_lateral_d40
  use_experiment_plan: true
  phases:
    - d40
```

```bash
optical track --persist-frames --condition near_lateral_d0
optical track --persist-frames --condition near_lateral_d40
# …remaining slots, then mid and far, then the same with the ladder frontal:
optical track --persist-frames --condition far_frontal_d0
```

Change `phases` to the slot you are recording (`d0`, `d40`, …) so the
section name matches `study.yaml`. The leaf must contain `trueness` in
`experiment.name`.

The ladder does not have to lie on a camera axis. Within one range and one
orientation, the slot positions are fit to a straight line whose parameter is
the hand-measured distance in millimetres. The tilt of that line is free, so
a crooked setup does not become part of the error. What remains is the
residual in world x, y and z, plotted per axis. Separately, the distance from
the reference slot (`ladder.reference_mm`) is compared with
``||position − reference position||``. Near is not compared with mid or far.

## Eval

```bash
python experiments/5_1_1_pnp_vs_triangulation/eval_precision.py  # latest precision_* leaf per near|mid|far
python experiments/5_1_1_pnp_vs_triangulation/eval_trueness.py   # every trueness_* leaf, one per slot
python experiments/5_1_1_pnp_vs_triangulation/eval_accuracy.py   # combines the two reports
```

The paths above are relative to the repository root.
Run from any directory with the venv that has `software/optical` installed;
`array_pose.py` is found next to the scripts. Leaves are read with
`optical.evaluation.recordings`, replayed with
`optical.tracking.tracker.load_tracker_for_recording` (leaf-local calibration
and config snapshots), and scored with `optical.evaluation.precision`
(chordal mean rotation and atan2 geodesic angle from
`optical.calibration.geometry`). `--recordings` / `--output` override the
`~/experiments/5_1_1_pnp_vs_triangulation/{recordings,results}` defaults.

Reports:

| File | Contents |
|---|---|
| `~/experiments/5_1_1_pnp_vs_triangulation/results/precision.json` | radial r_mean±σ_r, rotation, per-axis σ |
| `~/experiments/5_1_1_pnp_vs_triangulation/results/precision_axes.png` | σ of x, y, z about the mean |
| `~/experiments/5_1_1_pnp_vs_triangulation/results/trueness.json` | per-slot x/y/z error, both orientations, near/mid/far |
| `~/experiments/5_1_1_pnp_vs_triangulation/results/trueness_lateral.png` | depth placement; rows near/mid/far, columns x/y/z |
| `~/experiments/5_1_1_pnp_vs_triangulation/results/trueness_frontal.png` | left–right placement; rows near/mid/far, columns x/y/z |

Error bars on the trueness plots are the within-section σ of that axis.
Each leaf stores `session.h5`, `meta.json`, `frames/`, plus snapshots of
`calibration_bundle.h5` and `optical.yaml`. Trueness leaves also store
`*_jts_config.json` (the section frame interval).
