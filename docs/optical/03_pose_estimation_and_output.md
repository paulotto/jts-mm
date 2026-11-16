<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# 03 — Pose estimation and output contract (the Qualisys replacement)

This is the core of replacing Qualisys: convert per-camera 2D tag detections into the **same optical outputs** your downstream jaw-kinematics expects.

There are two valid approaches:

- **A. PnP pose per camera (recommended MVP)**: compute 6-DoF pose for each tag directly from a single camera image using the known tag size + intrinsics.
- **B. Multi-view triangulation (recommended for best accuracy)**: triangulate 3D points from ≥2 cameras, then rigid-fit the CRA/MTA arrays.

You can implement A first, then add B later.

## 1. Approach A — single-camera PnP (tag pose)

### 1.1 Inputs per camera per frame

- image (or grayscale)
- `K`, `dist` from intrinsics calibration
- detected `corners` and `ids` from `cv2.aruco.ArucoDetector.detectMarkers`
- `tag_size_mm`

### 1.2 Compute tag pose

For each detected tag:

1. Construct 3D object points (tag corners in tag frame), e.g.:
   - $(\pm s/2, \pm s/2, 0)$ where `s = tag_size_mm`
2. Solve PnP:
   - `cv2.solvePnP(obj_points, img_points, K, dist, flags=SOLVEPNP_IPPE_SQUARE)` is a good starting point for planar squares.
3. Convert `(rvec, tvec)` to a 4×4 transform $T_{\text{cam}\leftarrow\text{tag}}$.

### 1.3 Convert tag poses into CRA/MTA array poses

You must know the fixed transform $T_{\text{array}\leftarrow\text{tag}}$ for each tag ID (from `01_marker_setup.md`).

For each array (CRA, MTA), for each visible tag ID:

$$
T_{\text{cam}\leftarrow\text{array}} =
T_{\text{cam}\leftarrow\text{tag}} \; T_{\text{tag}\leftarrow\text{array}}
$$

### 1.4 Fuse multiple tags (robustness)

Implemented (`pose_estimation.py`): **one PnP per body and camera** over the body-frame corners of all visible tags (IPPE for planar tag sets, SQPnP otherwise, each solution refined with LM). Per-tag PnP + fusion is not used: a single 30 mm tag at 600 mm has two IPPE solutions with nearly equal error (the planar flip), which showed up as 20–60° rotation outliers (p95 ≈ 30° at 20° tilt / 0.5 px noise; body PnP over two tags: p95 < 5°).

When a view still has two plausible solutions (RMS within 1 px, rotations > 5° apart), the tracker picks the one consistent with, in order: an unambiguous view of the same body, the other cameras (agreement within 20°), the previous pose (`tracker.pnp.prior_max_frames`, default 15 frames after the last unambiguous solve), else the lowest RMS. Cameras are then fused (quaternion average + median translation; `use_reprojection_weights` optional).

Gates: a tag whose RMS under the joint solve exceeds `tracker.pnp.max_reprojection_rms_px` (default **8.0 px**) is dropped and the body re-solved; Kabsch bodies above `max_residual_mm` (**5.0**) or with fewer than `min_views` (**2**) are rejected; NaN unless the fused `T_world←body` is rigid SE(3). See §3.3.

### 1.5 Convert camera frame pose to world frame

If you have extrinsics $T_{\text{world}\leftarrow\text{cam}}$:

$$
T_{\text{world}\leftarrow\text{array}} =
T_{\text{world}\leftarrow\text{cam}} \; T_{\text{cam}\leftarrow\text{array}}
$$

If you only have stereo baseline, you can define cam0 as world for an MVP, but you should document that “world” equals cam0.

## 2. Approach B — multi-view triangulation (markers → rigid fit)

This approach is conceptually closer to Qualisys.

### 2.1 What you triangulate

Options:

- triangulate **tag centers**
- triangulate all **4 tag corners** (more constraints; better)

### 2.2 Triangulation inputs

For each camera `i`:

- projection matrix $P_i = K_i [R_i|t_i]$ using calibrated extrinsics
- undistorted 2D measurements $(u, v)$ per tag (or per corner)

### 2.3 Output: 3D points

For each tag (or corner) visible in ≥2 cameras, triangulate a 3D point in the chosen world frame.

### 2.4 Rigid-body fit for CRA/MTA

Given a set of observed 3D points $\mathbf{x}_j$ and the known model points $\mathbf{X}_j$ in the array frame, solve:

$$
\min_{R,t} \sum_j \| (R \mathbf{X}_j + t) - \mathbf{x}_j \|^2
$$

This is the Kabsch / Procrustes solve and yields $T_{\text{world}\leftarrow\text{array}}$.

Store the per-frame residual as a quality metric (drop frames with large residuals).

Implemented (`triangulation.py`): a corner view whose reprojection error exceeds `tracker.triangulation.max_corner_reprojection_px` (**2 px**) is dropped and the corner re-triangulated (a 2-view corner over the limit is discarded); Kabsch then runs with up to two rounds of residual trimming (drop points above max(3·median, 1 mm), keep ≥ 4). Quality keeps pixels and millimetres apart: `reprojection_rms_px` = corner reprojection RMS, `fit_rms_mm` = Kabsch residual; `n_cameras` = cameras that saw a tag of the body.

## 3. Output contract (what replaces Qualisys)

For JTS (`software/jaw_tracking/jts/`), the most direct replacement for Qualisys is:

- write per-frame 6-DoF poses for the three rigid bodies JTS expects:
  - `HP` (headpiece / CRA)
  - `MP` (mouthpiece / MTA)
  - `CT` (calibration tool / digitizing pointer)

Define a single per-frame record emitted by the tracker (per time `t`):

- timestamp (ideally `t_master_us`)
- `pose_HP`: 4×4 float matrix (or `nan` if not available)
- `pose_MP`: 4×4 float matrix (or `nan`)
- `pose_CT`: 4×4 float matrix (or `nan`, but then JTS calibration will not work)
- optional: `markers_3d`: list/array of triangulated points with IDs
- quality metrics:
  - number of tags per array used
  - reprojection / rigid-fit residual

This record (or an equivalent file containing these trajectories) is what JTS must consume instead of Qualisys exports/streams.

### 3.1 Tracking session HDF5 layout

Written by `tracking/run.py` to:

- `data/optical/tracking/<experiment>.h5`

Schema version **1.0** (see `tracking/io.py`).

```
<experiment>.h5
├── meta/
│   ├── attr: schema_version              "1.0"
│   ├── attr: created_utc                 ISO string
│   ├── attr: rig_id                      string (from calibration bundle)
│   ├── attr: experiment_name             string (from experiment plan)
│   ├── attr: frame_rate_hz               float64  (inferred from timestamps; MCU t_master_us or t_cam_ns)
│   ├── attr: n_frames                    int
│   ├── attr: method                      "pnp" | "triangulation"
│   ├── attr: notes                       string
│   ├── attr: units_position              "mm"
│   └── attr: bodies                      ["HP", "MP", "CT"]
│
├── timestamps/
│   ├── frame_idx                         (N,) int64     0 … N−1
│   ├── t_master_us                       (N,) float64   MCU master time (picamera2 live track; NaN if a frame did not match a TRIGGER_EVENT)
│   └── t_cam_ns/
│       └── <cam_id>                      (N,) float64   camera `SensorTimestamp` / host capture ns
│
├── bodies/
│   ├── HP/
│   │   ├── pose_4x4                      (N, 4, 4) float64   T_world←HP (NaN if untracked)
│   │   ├── position_mm                   (N, 3) float64        translation column of pose
│   │   ├── rotation_matrix               (N, 3, 3) float64
│   │   └── tracking_valid                (N,) bool
│   ├── MP/   … same layout …
│   └── CT/   … same layout …
│
├── quality/
│   ├── HP/
│   │   ├── tags_used                     (N,) int32
│   │   ├── reprojection_rms_px           (N,) float64   corner / PnP reprojection RMS (px)
│   │   ├── fit_rms_mm                    (N,) float64   Kabsch residual (triangulation; NaN for PnP)
│   │   └── n_cameras                     (N,) int32     cameras that saw the body
│   ├── MP/   …
│   └── CT/   …
│
└── markers_3d/                           optional; present when method = "triangulation"
    ├── attr: layout                      "flat_v1"
    ├── frame_row                         (M,) int32     row in timestamps/ of each point
    ├── marker_id                         (M,) int16
    ├── corner_index                      (M,) int8
    ├── position_mm                       (M, 3) float64 triangulated corner in world frame
    ├── n_views                           (M,) int8
    └── reprojection_rms_px               (M,) float32
```

Older files stored `markers_3d/frame_<row>/point_<j>/` groups (~1.4 GB per 20 min); `optical.tracking.io.read_markers_3d` reads both. `meta/` may carry extra attrs from live capture (`sync_report`, `pair_*` counters, `ct_tip_*`, `trigger_rate_hz`; dicts as JSON strings). The writer appends rows incrementally, so an aborted take keeps what was captured. With `tracker.ct_tip_calibration` set (default: `software/optical/etc/ct_tip_calibration.json`), `bodies/CT` is the pointer **tip** pose, as `/optical/pose_ct` in jtsmm; `null` keeps the CT body origin.

**Convention:** all stored body poses are **`T_world←body`** (homogeneous 4×4). Positions are in **millimeters**. See §4 for how “world” is defined and which intermediate transforms are *not* stored in this file.

### 3.2 Multimodal `session.h5` `/optical/` (jtsmm record)

`jtsmm record --optical-track` writes the same three bodies onto the MCU master clock, using JTS-MM names:

```
session.h5
├── meta/  attr: optical_tracking = "true"
└── optical/
    ├── t_master_us          (T,)
    ├── pose_mta             (T, 4, 4)   MP
    ├── pose_cra             (T, 4, 4)   HP
    ├── pose_ct              (T, 4, 4)   CT
    └── quality/
        ├── attr: rms_kind   "reprojection_px" | "kabsch_mm"
        ├── mta/  n_tags, n_cameras, reprojection_rms
        ├── cra/  …
        └── ct/   …
```

Missing lock stays a **NaN pose** so `/optical` stays gap-free with `/cameras`. Canonical field list and integrity budgets: [`s02` §8.5](../software/s02_data_pipeline.md).

### 3.3 Live rejection (same tracker for standalone and multimodal)

Per frame in `ArucoJawTracker.process_frame`:

1. Drop tags above `tracker.pnp.max_reprojection_rms_px` (PnP) or bodies above `tracker.triangulation.max_residual_mm` (Kabsch).
2. Require `pose.valid` **and** `is_rigid_se3(T)` (rotation in SO(3), last row `[0,0,0,1]`).
3. Store quality (`n_tags`, `n_cameras`, RMS) even when the pose is NaN.

A tracker exception in `OpticalTrackBridge` still appends a timestamped NaN row so the optical stream does not stall.

### 3.4 Post-session checks

```bash
jtsmm integrity check --strict SESSION.h5
python -m jtsmm.tools.eval_optical_session SESSION.h5 --no-show --strict
```

`--strict` on eval applies `check_optical(..., tracked=True)` and exits **2** on errors **or** warnings (plots still print). `jtsmm integrity --strict` is the full multimodal pass (EMG/IMU/cameras **and** optical) and treats warnings as failures (exit **1**). Phase co-visibility (MP+CT on `mand_*`, HP+CT on `max_*`, HP+MP on motion) is enforced here, not in the live SPACE loop.

## 4. Coordinate systems and transforms

This section ties together the frames used across marker setup, calibration, tracking, and JTS analysis, and states **where each transform is defined vs persisted**.

### 4.1 Frames in the pipeline

| Frame | Meaning | Typical definition in this repo |
|-------|---------|--------------------------------|
| **Tag** | ArUco marker plane (+Z out of tag) | OpenCV corner order; see doc 01 §3 |
| **Body** (`HP`, `MP`, `CT`) | Rigid frame on headpiece, mouthpiece, or pointer | Origin/axes on the physical object; what JTS calls a “rigid body” |
| **Camera** | OpenCV camera frame (+Z forward, +Y down) | Per `cam0`, `cam1`, … |
| **World** | Shared reference for a session | **Multi-cam and single-cam:** `world = cam0` (`T_world←cam0 = I`; other cameras relative to cam0) |
| **Landmark** | Anatomical CS fitted to 3 digitized points | Created by **JTS** from CT pointer captures (not by the optical tracker) |
| **Model** | Dental / CAD model origin | Defined by `model_points` in the JTS config |

Naming convention in code and docs: **`T_A←B`** is the 4×4 map that takes a point from frame B to frame A:
`p_A = T_A←B @ p_B`.

### 4.2 Transform chain (optical tracker, per frame)

For Approach A (PnP), the tracker computes:

```
T_world←body = T_world←cam @ T_cam←tag @ T_tag←body
```

| Transform | How it is obtained | Saved? |
|-----------|-------------------|--------|
| `T_cam←tag` | `cv2.solvePnP` on detected corners + `tag_size_mm` | **No** (runtime only) |
| `T_body←tag` | Fixed mount; `optical.yaml` → `markers.mounts_T_body_tag` | **Config file** (+ dictionary/size copied into calibration bundle `fiducials/` attrs) |
| `T_world←cam` | Extrinsics calibration | **`calibration_bundle.h5`** → `cameras/<id>/extrinsics/T_world_cam` |
| `T_world←body` | Fuse tags → fuse cameras → one pose per body | **`<experiment>.h5`** → `bodies/<HP\|MP\|CT>/pose_4x4` (and derived `position_mm`, `rotation_matrix`) |

For Approach B (triangulation), tag corners are triangulated to 3D in world space, then Kabsch fits `T_world←body` using model points derived from the same `mounts_T_body_tag`. The result is still stored as `T_world←body` in the session file; optional corner clouds go under `markers_3d/`.

**Multi-tag / multi-camera fusion** (median translation, quaternion rotation average) happens inside `pose_estimation.py` / `pose.py` and is **not** written out separately—only the fused `T_world←body` per frame is saved.

### 4.3 Transforms used downstream (JTS, not in tracking HDF5)

JTS reads the three body trajectories from `<experiment>.h5` and applies further **static** and **time-varying** transforms. None of these are written by `tracking/run.py`; they are computed at analysis time (and stored in JTS output HDF5 if configured).

| Transform | Role | Defined / computed from |
|-----------|------|-------------------------|
| `T_HP←MP(t)` | Mandible motion relative to headpiece | `inv(T_world←HP) @ T_world←MP` per frame (JTS `compute_relative_transform`) |
| `T_MP←landmark`, `T_HP←landmark` | Map marker frame to anatomical landmark CS | JTS `perform_calibration()` from CT digitization intervals in `_jts_config.json` |
| `T_model←maxilla` (registration) | Align maxilla to CAD model | JTS `register_to_model_coordinates()` using `coordinate_transform.model_points` in JTS config |
| `T_model←mand_landmark(t)` | Final jaw trajectory in model coordinates | JTS pipeline output (e.g. `T_model_origin_mand_landmark_t` in JTS result HDF5) |

The optical tracker’s job is only to supply **`T_world←HP`**, **`T_world←MP`**, and **`T_world←CT`** over time, plus frame intervals for landmark digitization (via the generated JTS config). JTS owns everything anatomical and model-related.

### 4.4 Quick reference: what lives in which file

| File | Transforms / geometry stored |
|------|------------------------------|
| `software/optical/etc/optical.yaml` (`markers:`) | `mounts_T_body_tag` (`T_body←tag` per marker ID), ID ranges, tag size |
| `data/optical/calibrations/.../calibration_bundle.h5` | `T_world←cam` per camera, intrinsics `K`/`dist`, fiducial metadata |
| `data/optical/tracking/<experiment>.h5` | **`T_world←HP`**, **`T_world←MP`**, **`T_world←CT`** per frame; `quality/{HP,MP,CT}/` |
| `session.h5` `/optical/` (multimodal) | **`T_world←MP/HP/CT`** as `pose_mta` / `pose_cra` / `pose_ct`; `quality/{mta,cra,ct}/` |
| `data/optical/tracking/<experiment>_jts_config.json` | Frame intervals for landmark + motion phases; `model_points` for registration (points, not full 4×4 matrices) |
| JTS `analysis_results/*.h5` | Landmark calibrations, relative motion, model-registered trajectories (after analysis) |

## 5. Implementation in `software/optical/optical/tracking/` (code) and `data/optical/` (artifacts)

The tracking subpackage (doc 03) implements the output contract:

- `tracking/aruco.py` — marker detection (IDs + corners)
- `tracking/pose.py` — SE(3) math, tag corner model, pose fusion
- `tracking/pose_estimation.py` — Approach A: per-camera PnP + tag/body fusion
- `tracking/triangulation.py` — Approach B: multi-view DLT + Kabsch fit
- `tracking/tracker.py` — `ArucoJawTracker` orchestration
- `tracking/io.py` — HDF5 session write/read (HP/MP/CT trajectories)
- `tracking/experiment.py` — experiment plan, built-in calibration phases, JTS config builder
- `tracking/run.py` — live capture CLI (SPACE-controlled phases)

### 5.1 Live capture workflow (`run.py`)

1. Operator configures **motion phases** (manual entry or `experiment:` in `optical.yaml`).
2. **Eight calibration phases** (`mand_point_1`–`3`, `max_point_1`–`3`, `tmj_left`, `tmj_right`) are inserted before the first motion phase. In each jaw, point 1 is the central incisor, point 2 the participant's left second molar and point 3 the right second molar; the `tmj_*` phases are the skin-palpated condyles (`LANDMARK_HINTS` in `tracking/experiment.py`; the prompt shows what to touch). EMG MVC is a separate `jtsmm record --mvc` take; `optical track` has no EMG path.
3. Preview runs continuously; SPACE starts/stops each phase; frames are saved only while a phase is active, so the HDF5 `frame_idx` is contiguous across the recorded intervals.
4. Outputs under `data/optical/tracking/`:
   - `<experiment>.h5` — pose trajectories (`meta.experiment_name`, `bodies/HP|MP|CT`, …)
   - `<experiment>_jts_config.json` — JTS config with captured frame intervals
5. Live `optical track` (picamera2) arms the HAT trigger, captures the **half-res lores Y** stream (same as `jtsmm record --optical-track`), scales intrinsics to that size, aligns each frame bundle to MCU `t_master_us` via `TRIGGER_EVENT`, and stores that stamp under `timestamps/t_master_us`. `t_cam_ns/<cam_id>` is still written for offline debugging. `--backend v4l2` does not use the HAT; `t_master_us` stays NaN.

Configs:

- `software/optical/etc/optical.yaml` — auto 1-cam (PnP) or 2-cam (triangulation) profile
- `experiment:` in those files — experiment name + motion phase names only

### 5.2 Remaining gaps / related entry points

- **Unified multimodal session:** `jtsmm record --optical-track` writes `/optical/` (poses + `quality/`) into `session.h5` on the MCU master clock; see [`s02`](../software/s02_data_pipeline.md) / [`s03`](../software/s03_camera_pipeline.md).
- **Experiment phases on multimodal record:** `jtsmm record --use-experiment-plan` walks motion + landmarks. `jtsmm record --mvc` is a separate EMG-only take (`mvc_opening` / `mvc_closing`, no `/optical` rows). Optional JTS sidecar from optical frame intervals (MVC omitted).
- `camera_test.py` is a preview-only utility (not wired into the pipeline).
- Standalone `optical track` remains the experiment-phase tool that **gates** frame writes between SPACE presses (contiguous optical HDF5).
- Marker size / lores tradeoff: `optical eval` (live) and `optical eval-quality` (offline degrade ladder).
- JTS `aruco_camera` data source (doc 04) is implemented in `software/jaw_tracking/jts/aruco_camera.py` — generated configs point at our HDF5 and JTS loads them through `core._create_motion_data_handler`.
- Calibration-phase co-visibility (MP+CT / HP+CT) is **not** gated in the live SPACE loop; `jtsmm integrity check` / `eval_optical_session --strict` flag phases that miss the lock floors in §3.4 / s02 §8.5.

