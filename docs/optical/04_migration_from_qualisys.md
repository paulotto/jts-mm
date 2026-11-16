<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# 04 — Migration from Qualisys to the ArUco optical pipeline

JTS (Jaw Tracking System) lives in this repo at:

- `software/jaw_tracking/jts/`

That package currently assumes Qualisys as the only **offline** data source.
This document explains what must change so ArUco camera tracking can replace Qualisys.

The optical pipeline already produces HDF5 sessions and a matching JTS config
(`data/optical/tracking/<experiment>_jts_config.json`). The JTS loader for that HDF5 format is implemented in [`software/jaw_tracking/jts/aruco_camera.py`](../../software/jaw_tracking/jts/aruco_camera.py).

## 1. What JTS currently expects from Qualisys

In `software/jaw_tracking/jts/core.py`:

- `JawMotionAnalysis._create_motion_data_handler()` only supports:
  - `data_source.type == "qualisys"`
- The pipeline then computes relative motion using:
  - `motion_data.compute_relative_transform(reference_body, moving_body, interval)`

With the default config (`software/jaw_tracking/config/config.json`):

- **reference_body**: `"HP"` (headpiece / CRA)
- **moving_body**: `"MP"` (mouthpiece / MTA)
- calibration uses `"CT"` (calibration tool / digitizing pointer)

So, to “replace Qualisys,” you must provide **per-frame 6-DoF poses** for at least:

- `HP` (CRA)
- `MP` (MTA)
- `CT` (digitizing pointer) — required for the landmark calibration workflow in JTS

If you skip `CT`, you also need to redesign the calibration step in JTS.

## 2. What the optical tracker already provides

After `optical track` (standalone) **or** `jtsmm record --optical-track` (multimodal `session.h5` `/optical/`):

| Artifact | Contents |
|----------|----------|
| `data/optical/tracking/<experiment>.h5` | Per-frame HP/MP/CT poses, timestamps, quality (`quality/{HP,MP,CT}/`) |
| `session.h5` `/optical/` (multimodal) | `pose_mta` / `pose_cra` / `pose_ct` + `quality/{mta,cra,ct}/` on `t_master_us` |
| `data/optical/tracking/<experiment>_jts_config.json` | `data_source.filename`, calibration frame intervals, motion sub-experiments |

The generated config uses `data_source.type = "aruco_camera"` and an offline-friendly
`calibration` block (no `mode` / `online_config` keys — those crash offline `perform_calibration()`).

## 3. Two integration strategies (pick one)

### Strategy A (quick, hacky): export a Qualisys-like `.mat`

You can make JTS work without code changes by writing a `.mat` file that matches the key fields JTS reads in `jts/qualisys.py`.

Pros: minimal edits in JTS. Cons: brittle schema.

### Strategy B (recommended): add a new offline data source to JTS — **implemented**

`software/jaw_tracking/jts/aruco_camera.py` implements
`ArucoCameraMotionCaptureData(MotionCaptureData)`, which reads our tracking session HDF5
(see doc 03 §3 and `tracking/io.py` layout) via the `ArucoCameraData` loader.

`_create_motion_data_handler()` (in `jts/core.py`) now accepts
`data_source.type == "aruco_camera"` and routes to this handler.

Pros: stable contract, matches the configs `optical track` already generates.

Implementation notes:

- Each body (`HP`, `MP`, `CT`) is loaded into a `jts.qualisys.RigidBody`
  (positions in mm, `(N, 3, 3)` rotation matrices), so the existing calibration
  and relative-motion machinery (`QualysisData.compute_average_transformations`,
  `compute_relative_transformations`) works unchanged.
- Untracked frames are kept as `NaN` poses (Qualisys-compatible).
- Bodies are read from `bodies/<name>/position_mm` + `rotation_matrix`, falling
  back to decomposing `bodies/<name>/pose_4x4` if those datasets are absent.
- Tests: `tests/test_aruco_camera.py`.

## 4. What the camera tracker must output (offline file contract)

Implemented in `tracking/io.py` (`write_tracking_session`):

- `meta/`: `frame_rate_hz`, `experiment_name`, `rig_id`, …
- `timestamps/frame_idx`, `t_master_us` (MCU time on picamera2 live track; NaN if a frame missed trigger match)
- `bodies/{HP,MP,CT}/`: `pose_4x4`, `position_mm`, `rotation_matrix`, `tracking_valid`
- `quality/{HP,MP,CT}/`: tags used, reprojection RMS, camera count

Positions are in **millimeters** (JTS internal unit).

## 5. Landmark calibration without Qualisys

The digitizing pointer (`CT`) must be camera-trackable (ArUco on the pointer; see doc 01).
During live capture, `run.py` always runs six calibration phases before motion phases;
JTS uses the recorded frame intervals to compute `T_marker_landmark` offline.

## 6. Status

| Step | Status |
|------|--------|
| Recorder pose pipeline + HDF5 output | **Done** |
| Guided live phases + JTS config generation | **Done** |
| HAT-backed MCU `t_master_us` in standalone optical HDF5 (`optical track`) | **Done** |
| JTS `aruco_camera` loader | **Done** (`jts/aruco_camera.py`) |
| End-to-end JTS analysis on optical tracking data | **Unblocked** (loader wired into `core._create_motion_data_handler`) |
