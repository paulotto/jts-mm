<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# 01 — Marker setup (ArUco) for jaw tracking

This document defines what you must decide and encode **before** writing the tracker:

- which fiducial dictionary and IDs you use
- how the tags are mounted on the CRA and MTA
- how detections map to rigid-body poses

## 1. Pick a dictionary and tag size

### Decision (for this integration)

- **Dictionary**: `cv2.aruco.DICT_4X4_50`
- **Tag side length**: **25.0 mm** (physical printed edge length)

Rationale:

- `DICT_4X4_50` is already used by `software/optical/optical/tracking/aruco.py`.
- 25 mm is a practical size for stable detection at typical lab camera distances without making the arrays bulky.

You must standardize (and encode in config + calibration bundle metadata):

- **dictionary**: e.g. `DICT_4X4_50` (classic ArUco) or an AprilTag dictionary if you prefer robustness
- **physical tag side length** $mm$: required for PnP pose estimation

Store both in a config (YAML/JSON) and inside the calibration bundle as metadata.

For choosing size vs tracking resolution (e.g. half-res lores in `jtsmm record`), run `optical eval-quality` (see `eval_quality:` in [`optical.yaml`](../../software/optical/etc/optical.yaml)). Lab measurements with 30 mm tags remained reliable down to ~VGA/half; quarter-res is marginal.

## 2. Assign IDs to CRA vs MTA vs CT (and to individual tags)

For integration with JTS (`software/jaw_tracking/jts/`), you must track three rigid bodies:

- `HP` (headpiece / CRA)
- `MP` (mouthpiece / MTA)
- `CT` (calibration tool / digitizing pointer)

### Decision (for this integration)

- **HP tag IDs**: **0–9**
- **MP tag IDs**: **10–19**
- **CT tag IDs**: **20–29**

Recommended minimum set (start here):

- `HP`: IDs **0, 1, 2**
- `MP`: IDs **10, 11, 12**
- `CT`: IDs **20** (single tag is acceptable for the pointer MVP)

Then the runtime logic becomes deterministic:

- detected `id ∈ HP_RANGE` → contributes to `HP` pose
- detected `id ∈ MP_RANGE` → contributes to `MP` pose
- detected `id ∈ CT_RANGE` → contributes to `CT` pose

Keep the mapping in one place (`markers:` in `optical.yaml`) so it’s not hardcoded across files.

## 3. Define the rigid-body frames

You need a frame hierarchy to avoid ambiguity:

- **Tag frame**: defined from the ArUco corner ordering. For this integration use:
  - tag corners in tag frame (z=0 plane): $(-\frac{s}{2}, +\frac{s}{2}, 0), (+\frac{s}{2}, +\frac{s}{2}, 0), (+\frac{s}{2}, -\frac{s}{2}, 0), (-\frac{s}{2}, -\frac{s}{2}, 0)$ corresponding to OpenCV’s corner order `[0,1,2,3]` returned by `detectMarkers`.
  - tag +Z points out of the tag plane (right-hand rule); consistent with `solvePnP` object points definition.
- **Rigid-body frame** (`HP`, `MP`, `CT`): a rigid coordinate system attached to the physical object (what JTS consumes as “rigid body pose”).
- **World frame**: a global reference for the session (often “CRA at t=0” or “board frame” during calibration).
- **Camera frame**: OpenCV uses +Z forward from camera, +X right, +Y down (image coordinates).

You must define and document:

- $T_{\text{cam}\leftarrow\text{tag}}$ from `solvePnP` / `estimatePoseSingleMarkers`
- $T_{\text{array}\leftarrow\text{tag}}$ from your CAD / measurement (fixed per tag mount)
- $T_{\text{world}\leftarrow\text{cam}}$ from extrinsics calibration

Then per-frame array pose is:

$$
T_{\text{world}\leftarrow\text{array}} =
T_{\text{world}\leftarrow\text{cam}}
\; T_{\text{cam}\leftarrow\text{tag}}
\; T_{\text{tag}\leftarrow\text{array}}
$$

where $T_{\text{tag}\leftarrow\text{array}} = (T_{\text{array}\leftarrow\text{tag}})^{-1}$.

## 4. How many tags per array?

Minimum viable:

- `HP`: 1 tag
- `MP`: 1 tag
- `CT`: 1 tag

Recommended for robustness to occlusion:

- `HP`: **≥2 tags** (3 is better)
- `MP`: **≥2 tags** (3 is better)
- `CT`: 1 tag is fine (pointer is usually held in view during calibration)

With multiple tags, you fuse them (see `03_pose_estimation_and_output.md`).

## 5. How to get $T_{\text{array}\leftarrow\text{tag}}$

You need a fixed transform from each tag to the rigid-body frame (`HP`/`MP`/`CT`).
Options:

- **From CAD** (best if the tag pads are designed into the array): export the tag pose(s) relative to the array origin.
- **From measurement** (acceptable for prototypes): measure tag corner coordinates in the array frame (3D) and solve a rigid transform.
- **From one-time calibration jig**: place the array in a known fixture and solve the transforms by observing it with the calibrated cameras.

Deliverable of this step:

- a small file (JSON/YAML) listing, per tag ID:
  - which rigid body (`HP`, `MP`, or `CT`)
  - tag size (mm) if not global
  - $T_{\text{body}\leftarrow\text{tag}}$ as a 4×4 matrix

Template file to fill in:

- `software/optical/etc/optical.yaml` (`markers:` section) — edit mounts / IDs / tag size in place (auto 1/2-cam profile at runtime)

## 6. Print + mounting guidelines (practical)

- Print on matte stock to avoid glare.
- Ensure the tag is planar and rigidly mounted (no bending).
- Keep tags away from reflective surfaces; specular highlights break detection.
- Use consistent tag size across the system (don’t mix sizes unless your config supports it).

