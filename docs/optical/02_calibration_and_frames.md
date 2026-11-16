<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# 02 — Calibration and frames (what you must implement to replace Qualisys)

Qualisys gives you calibrated 3D marker trajectories “for free.” With cameras, you must explicitly manage:

- intrinsics per camera
- extrinsics between cameras (and to a world frame) for **2…N cameras**
- timestamping / alignment across cameras and other modalities

This repo has a demo in `calibration.py`, but integration requires a stable calibration workflow and reusable calibration artifacts.

## Decision: multi-camera world-frame calibration (scales to N cameras)

For a scalable setup (2…N cameras), we will **not** rely on “stereo-only” calibration as the primary artifact.
Instead, we define a world frame and compute **one extrinsic per camera**:

- $T_{\text{world}\leftarrow\text{cam}_i}$ for every camera `cam_i`

This makes it easy to:

- add a third/fourth camera without re-deriving chained transforms
- run triangulation / multi-view fusion with any subset of cameras

World frame definition (stored in the calibration bundle):

- **Multi-camera:** **world = cam0** — ``T_world_cam0 = I``; other cameras store ``T_world_cam_i`` relative to cam0, fused from paired ChArUco board snapshots.
- **Single camera:** **world = cam0** (identity ``T_world_cam``); skip ``--mode extrinsics``.

## 1. Calibration artifacts you need

### 1.1 Per-camera intrinsics

For each camera `cam_i` you need:

- `K_i` (3×3 camera matrix)
- `dist_i` (distortion coefficients; pick a model and keep it consistent)
- `image_size_i`

### 1.2 Cross-camera extrinsics

You need **one** of these (pick based on your hardware):

- **Single camera**: *no extrinsics needed.* Define `world = cam` (identity `T_world_cam`); see `03_pose_estimation_and_output.md` §1.5. Skip `--mode extrinsics` and go straight to `--mode bundle`, which fills the identity transform automatically.
- **Stereo-only** (2 cameras only): `R_01`, `T_01` between cam0 and cam1
- **Multi-camera world frame** (2–3 cameras): `T_world_cam_i` for each camera

For jaw tracking, “multi-camera world” is usually the better long-term choice because it scales to 3 cameras and avoids accumulating transforms.

### 1.3 Versioned bundle file

Write a single calibration bundle that your tracker can load at runtime.

#### Decision: calibration bundle format

Use a single **HDF5** file (stable, self-describing, easy to extend) stored at:

- `data/optical/calibrations/<rig_id>/calibration_bundle.h5`

Minimum contents (schema v1):

- camera intrinsics for each camera
- camera extrinsics for each camera (world-frame, one per camera)
- fiducial configuration (dictionary name, tag size, ID ranges)
- schema version + capture date

Store it in a stable location (e.g. `data/optical/calibrations/<rig_id>/calibration_bundle.*`).

#### HDF5 group layout (recommended)

```
calibration_bundle.h5
├── meta/
│   ├── attr: schema_version              "1.0"
│   ├── attr: created_utc                 ISO string
│   ├── attr: rig_id                      string
│   └── attr: notes                       string
│
├── cameras/
│   ├── cam0/
│   │   ├── intrinsics/
│   │   │   ├── K                         (3,3) float64
│   │   │   ├── dist                      (N,)  float64  (OpenCV distortion vector)
│   │   │   ├── image_size                (2,)  int32    [w,h]
│   │   │   └── rms_px                    scalar float64
│   │   └── extrinsics/
│   │       ├── T_world_cam               (4,4) float64  world = cam0
│   │       ├── cross_view_rms_px         scalar float64 extrinsics quality (§2.2)
│   │       ├── rms_px                    scalar float64 mean board PnP residual (not a quality metric)
│   │       └── quality/                  per-snapshot cross-view px, inlier/paired masks, Δrot, Δt
│   ├── cam1/ ...
│   └── camN/ ...
│
└── fiducials/
    ├── attr: detector                    "aruco"
    ├── attr: dictionary                  "DICT_4X4_50"
    ├── attr: tag_size_mm                 25.0
    └── attr: marker_config_path          "software/optical/etc/optical.yaml"
```

## 2. Calibration procedure (recommended)

### 2.1 Intrinsics

Use a checkerboard or (better) ChArUco board:

- capture ≥ ``capture.min_images_intrinsics`` (25) good views per camera
- cover full image area (corners too), varied tilt
- keep lighting constant; `optical calibrate` does not lock exposure (the
  `capture.lock_exposure` / `lock_focus` keys are ignored) and the IMX296 lens is manual-focus

Acceptance check:

- RMS reprojection error $\lesssim 1.0$ px

### 2.2 Extrinsics

For 2…N cameras, use a **board-based multi-view extrinsics** solve (implemented in ``multicam.py``):

- Place the board so it is visible in all cameras (not necessarily every frame, but ensure overlap).
- Capture synchronized snapshots (SPACE per pose); frames are paired by ``SensorTimestamp``
  (same trigger) and must have the intrinsics' image size.
- For each snapshot where **cam0** and **cam_i** both detect the board:
  - solve PnP → ``T_cam←board`` per camera; views with PnP RMS > ``capture.max_pnp_rms_px``
    (1.5 px) are dropped; collinear ChArUco detections are never accepted
  - accumulate ``T_world←cam_i = T_cam0←board @ T_board←cam_i``
- Fuse robustly (``calibration/extrinsics.py``): rotation medoid + median translation as
  reference, reject outlier snapshots, chordal-mean / mean of the inliers. No bundle adjustment.
- Store **``T_world_cam0 = I``** and fused ``T_world_cam_i`` in the bundle.

Acceptance gates (bundle written automatically only if all pass; otherwise the cache is
kept and ``--mode bundle`` accepts it explicitly, recorded as ``meta/extrinsics_gates_passed``):

- **cross-view error** ≤ ``capture.max_cross_view_rms_px`` (4.0 px): cam_i's detected
  corners vs. cam0's board pose projected through ``T_cam_i←cam0``
- ≥ ``capture.min_inlier_fraction`` (0.6) of paired snapshots agree

Caches are ``<bundle_stem>.intrinsics_cache.npz`` / ``.extrinsics_cache.npz`` (per camera
profile); ``--mode bundle`` refuses extrinsics solved with different intrinsics.

Operator checks:

- baseline and camera poses are physically plausible
- during ``multicam.py --mode extrinsics``, each SPACE snapshot logs **baseline mm**
  between ``cam0`` and ``cam1`` when both detect the board; compare to
  ``expected_baseline_mm`` in the calibration config. The **bundle baseline**
  (``|T_world_cam1|`` with ``T_world_cam0 = I``) should match the per-snapshot mean.

Optional config field:

- ``expected_baseline_mm`` — physical distance between camera centers (mm) for log comparison

### 2.3 “N camera” capture protocol (repeatable)

Use this operator workflow:

- **Step A (intrinsics)**: per camera, 20–30 captures with strong perspective variation.
- **Step B (extrinsics)**: with cameras fixed, move the board through the shared volume:
  - ≥ ``capture.min_snapshots_extrinsics`` (40) multi-camera “snapshots”
  - each snapshot: one frame per camera (as close in time as possible)
  - keep exposure fixed across cameras for consistent detection

If you have hardware trigger: capture simultaneously. If not: do rapid sequential reads and keep only snapshots where board is stable (low motion blur).

## 3. Frame acquisition requirements for a real tracker

The current `Camera` class in `camera.py` is a preview utility.
A production jaw tracker needs additional per-frame metadata:

- **frame index** (monotonic integer)
- **camera timestamp** (monotonic clock, ideally hardware timestamp if available; store in ns)
- exposure / gain (optional but useful for debugging)

If the rest of your system uses a master clock (e.g. `t_master_us`), you must also implement:

- a mapping from camera timestamps → master timestamps, or
- hardware triggering so the master timestamp is known at capture time

### Decision: per-camera timestamps + optional master alignment

For a scalable multi-camera system, use a two-layer timestamping contract:

- **Always** record `t_cam_ns` per camera frame (monotonic).
- **Optionally** add `t_master_us` later via an alignment step (if/when you integrate the MCU trigger clock).

This allows:

- multi-camera triangulation using **per-snapshot synchronization** (group frames by nearest time)
- future integration with your master clock without redesigning the file format

## 4. Where marker detection belongs

For performance and clean data flow:

- Run marker detection **in the camera capture process** (per camera), not downstream.

Output per frame should include only compact results:

- detected IDs
- corners (4×2 per tag)
- optional confidence / reprojection residual

Avoid pushing full pixel frames into the alignment / triangulation stage unless you explicitly need them.

## 5. What to change in this repo (concrete)

Even if you keep the scripts as-is, integration requires these additions:

- A reusable calibration loader/writer (don’t rely on ad-hoc `.npz` filenames in the working directory).
- A multi-camera capture abstraction (2…N cameras) that yields `(cam_id, frame_idx, t_cam_ns, frame)` records.
- A stable calibration bundle (`calibration_bundle.h5`) that the tracker loads at startup.
- A clear per-frame metadata contract (units, monotonicity, epoch).

Related code today:

- `software/optical/optical/calibration/multicam.py`: interactive intrinsics (robust
  `calibrateCameraExtended`), extrinsics capture and bundle export (`optical calibrate`).
- `software/optical/optical/calibration/extrinsics.py`: robust fusion, cross-view error, gates.
- `software/optical/optical/calibration/io.py`: bundle read/write, consumed by the tracker.

