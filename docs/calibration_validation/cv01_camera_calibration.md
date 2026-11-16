<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# cv01 — Camera Calibration

This document describes the camera-calibration procedure for both intrinsic (per-camera) and extrinsic (multi-camera) parameters. Calibration is repeated whenever the camera setup is moved, focus is changed, or every ~30 days as a routine refresh.

## 1. Targets

| Calibration | Goal | Required for |
|-------------|------|--------------|
| Intrinsic | `K`, distortion coefficients per camera | Reprojection, undistortion |
| Multi-camera extrinsic | Per-camera `T_world ← cam` | 3D triangulation |
| Anatomical / world reference | World frame definition + maxillary anatomical frame | Coupling with the existing JTS pipeline |

Successful calibration must produce a **single HDF5 calibration bundle** per camera profile that `optical calibrate` writes under `data/optical/calibrations/<rig_id>/` and that the tracker loads.

## 2. Calibration target

We use a **ChArUco board** (chess + ArUco hybrid) because:
- Robust corner detection even with partial occlusion.
- Each detected corner has a unique ID, so missing or misordered corners do not corrupt the fit.
- Standard OpenCV implementation, no custom code.

The board is defined in `calibration.board` of [`optical.yaml`](../../software/optical/etc/optical.yaml) and the same board is used for intrinsics and extrinsics:

| Field | Value |
|-------|-------|
| `squares_x × squares_y` | 7 × 5 (→ 6 × 4 inner corners) |
| `square_length_mm` | 38.52 |
| `marker_length_mm` | 28.1 |
| `dictionary` | `DICT_4X4_50` |
| `min_corners` | 6 (fewer detected corners → view ignored) |

Print it on a rigid, flat backing (foamcore / aluminium composite) and **measure** the printed square length; enter the measured value, since it sets the metric scale of every extrinsic.

Detections whose corners lie on one line (one row / one column / a diagonal) are rejected by the detector: PnP and `calibrateCamera` return garbage or assert on them.

## 3. Intrinsic calibration

### 3.1 Procedure

```bash
optical calibrate --mode intrinsics      # per camera: SPACE capture, Q solve, ESC abort
```

1. Capture at least `capture.min_images_intrinsics` (default 25) views per camera at the full sensor resolution (`calibration.cameras[].resolution`).
2. Coverage: board in all four image corners and the centre, tilts of ±30°, distances around the working distance.
3. Corners come from `cv2.aruco.CharucoDetector`; the solve is `cv2.calibrateCameraExtended` (5-coefficient Brown–Conrady `k1 k2 p1 p2 k3`).
4. Bad views are dropped instead of failing the session: degenerate (collinear) views up front; a view that makes OpenCV raise is left out; views whose RMS exceeds max(1 px, 3 × median) are dropped with one re-solve (while ≥ half and ≥ 5 views remain). Dropped views and the 1-σ of fx / fy / cx / cy are printed.

### 3.2 Acceptance criteria

- Overall reprojection RMS < **0.5 px** (printed, also in px → mm on the board plane).
- 1-σ of fx, fy < 0.5 % of the value; ≥ 20 views kept.
- Focal length within 2 % of the previous calibration of the same camera/lens.

These are operator checks; the code does not refuse a solve on them.

### 3.3 Output

Written to `<bundle_stem>.intrinsics_cache.npz` next to the bundle (per camera `K`, `dist`, `image_size`, `rms_px`), later merged into the bundle (§4.4). Re-running intrinsics invalidates the extrinsics cache (§4.3).

## 4. Multi-camera extrinsic calibration

### 4.1 Procedure

```bash
optical calibrate --mode extrinsics      # SPACE snapshot, Q finish (≥ min_snapshots_extrinsics paired)
```

1. Frames of all cameras come from the same hardware trigger (checked by `SensorTimestamp`, ≤ 4 ms apart) and must have the intrinsics' image size.
2. Per SPACE snapshot, `cv2.solvePnP` per camera gives `T_cam←board`. Views with PnP RMS > `capture.max_pnp_rms_px` (1.5 px) are dropped.
3. **World frame = cam0** (`T_world←cam0 = I`). For every snapshot seen by cam0 and cam_i: `T_world←cam_i = T_cam0←board · T_board←cam_i`.
4. Robust fusion: rotation medoid + median translation as reference, reject samples deviating more than max(floor, 3 × median deviation), average the inliers (chordal mean rotation, mean translation).
5. There is **no joint least-squares / bundle adjustment**; each snapshot contributes one relative pose.

### 4.2 Quality metric

**Cross-view error**: for each paired snapshot, the corners cam_i detected are predicted from cam0's board pose through the fused `T_cam_i←cam0` and compared to cam_i's detections (px RMS, per snapshot and overall). Unlike the per-camera PnP residual (`rms_px`, kept for reference), it measures the camera-to-camera transform itself. The per-snapshot baseline (cam0–cam1 distance) is logged against `expected_baseline_mm`.

### 4.3 Acceptance gates

The bundle is written automatically only when all gates pass (keys in `calibration.capture`, defaults in brackets):

- overall cross-view RMS ≤ `max_cross_view_rms_px` [4.0 px];
- ≥ `min_inlier_fraction` [0.6] of the paired snapshots (and ≥ 3) agree.

Otherwise the extrinsics cache is kept and the problems are printed. `optical calibrate --mode bundle` writes that cache without recapture, but re-checks the stored cross-view against the current gates: under the new threshold `meta/extrinsics_gates_passed` is true, and a result that still fails is written with the flag false. `--mode bundle` refuses an extrinsics cache solved with different intrinsics (SHA-256 of K, dist, image size).

### 4.4 Output

One bundle per camera profile (`paths` / `profiles.<n>.calibration_bundle` in `optical.yaml`, e.g. `calibration_bundle.h5` for 1 camera, `calibration_bundle_2cams.h5` for 2), written atomically. Caches are named after the bundle so profiles never mix. Layout: [`02_calibration_and_frames.md`](../optical/02_calibration_and_frames.md) §1.3; per camera:

```
cameras/<cam_id>/extrinsics/
├── T_world_cam          (4, 4) float64   world = cam0
├── cross_view_rms_px    scalar           extrinsics quality (NaN for 1 camera)
├── rms_px               scalar           mean board PnP residual (not a quality metric)
└── quality/             per-snapshot cross-view px, inlier / paired masks, Δrot, Δt
```

A single-camera rig skips `--mode extrinsics`; `--mode bundle` writes the identity.

## 5. World coordinate convention

The world frame is defined as the maxillary anatomical frame `CSAnatMax` from the IEEE SENSORS 2025 paper (§II.B). After the camera-only extrinsic calibration, the world frame is **cam0**; it is brought into alignment with `CSAnatMax` downstream (not by `optical calibrate`) by:

1. Recording a digitization session with the existing DP (Digitizing Pointer): three maxillary tooth landmarks.
2. Solving for `T_world_csanatmax` via Kabsch using the digitized points (in cam0 frame) and their known positions in `CSAnatMax`.
3. Composing all camera extrinsics with this transform.

After this step, all kinematic outputs are in `CSAnatMax`, exactly as the existing JTS pipeline expects.

## 6. Per-session calibration check (planned, not implemented)

Before each recording session:

```bash
jtsmm cam check-calibration --duration 5
```

This places the (stationary) A2 ChArUco board in the FOV for 5 seconds, triangulates its known corners, and reports the per-corner residual. If the residual exceeds 1 mm RMS, the operator is warned to re-run the extrinsic calibration.

This is a cheap sanity check that catches the common failure mode of "the camera tripod got bumped".

## 7. Calibration drift sources

| Source | Effect | Mitigation |
|--------|--------|-----------|
| Tripod bumped | Translation drift | Per-session check (§6) |
| Lens temperature change | Focus / focal length drift, especially with M12 lenses | Warm-up cameras 5 minutes before calibration; re-calibrate if room temperature changes >5 °C |
| Lens loosened | Focus shift | Lock with thread-lock after focus adjustment |
| Sensor packaging stress | Distortion shift | Avoid mounting cameras under high mechanical preload |
| Time | Slow long-term drift | Quarterly re-calibration recommended |

## 8. Calibration file lifecycle

- One calibration file per build, named `calibration_<build>_<YYYY-MM-DD>.h5`.
- The optical pipeline stores a **`calibration_ref`** (path + hash) in each tracking session HDF5 and in the multimodal session `/meta/calibration_ref` once unified capture lands, so it is impossible to lose track of which calibration produced a given session.
- The `/meta/calibration_ref` attribute in each session HDF5 stores the SHA-256 of the calibration file at the moment of recording, so even if the file is later modified the original is uniquely identified.

## 9. References

- Hartley R., Zisserman A. — *"Multiple View Geometry in Computer Vision"*, Cambridge 2003 (Chapter 5–7).
- OpenCV ChArUco tutorial: <https://docs.opencv.org/4.x/da/d13/tutorial_aruco_calibration.html>.
- Zhang Z. — *"A flexible new technique for camera calibration"*, IEEE TPAMI 22(11) 2000.
- Olson E. — *"AprilTag: a robust and flexible visual fiducial system"*, ICRA 2011.
