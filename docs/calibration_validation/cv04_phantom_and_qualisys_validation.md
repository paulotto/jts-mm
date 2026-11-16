<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# cv04 — Phantom and Qualisys Validation

> **Superseded for the Qualisys part:** the executable protocol for the reference validation (Qualisys Oqus and Delsys Trigno registered to the JTS-MM clock **offline** from shared impacts and motion, no wiring to JTS-MM; 3D-printed step ladder as trueness ground truth; full list of what to record) is [`cv05_reference_validation_protocol.md`](cv05_reference_validation_protocol.md). §2 below is kept for history only; its TTL/LED synchronization is not used.

This document describes the two validation experiments planned for JTS-MM: a (future-work) **phantom rig** and a **side-by-side comparison with the existing Qualisys-based JTS**. This project delivers only the Qualisys comparison; the phantom rig is fully specified here for follow-on work.

## 1. Goals

| Validation | Purpose | Phase |
|------------|---------|-------|
| Qualisys side-by-side | Estimate JTS-MM optical precision vs. the published 182 ± 47 µm baseline; check EMG/IMU sanity in a real recording | This project |
| Phantom rig | Quantify accuracy (not just precision) against a ground-truth geometry; characterize systematic errors | Future work |

Precision = the spread of measurements around their own mean (i.e. noise). Accuracy = how close the mean is to the truth (i.e. bias). The Qualisys comparison gives us a relative precision number; the phantom rig is needed for accuracy.

## 2. Qualisys side-by-side experiment

### 2.1 Setup

- Both systems running simultaneously in the same lab.
- JTS-MM cameras (Track B, 2 Pi GS) on tripods on one side; Qualisys Oqus 3-camera rig on the other side.
- The MTA and CRA carry **AprilTag 36h11** pads for JTS-MM (default). For Qualisys side-by-side, temporarily add **retroreflective spheres** on the same arrays (optional kit in [`../04_component_catalog.md`](../04_component_catalog.md#9-fiducials--scene-lighting)) so the Oqus rig sees its native markers; JTS-MM software still uses AprilTag on the Pi cameras.
- Diffuse visible lighting for JTS-MM; Qualisys uses its own IR strobes — avoid pointing Qualisys strobes directly into the Pi camera lenses (glare on tags).
- All electrodes and IMUs as for a normal session.
- The Qualisys rig records its native data at 200 Hz; JTS-MM records at 40 Hz (cameras; template trigger rate), 390/250 Hz (IMU game-rotation-vector quaternion / accelerometer incl. gravity) and 4 kSPS (EMG).

### 2.2 Time alignment between the two systems

The Qualisys system has its own clock; we synchronize it to JTS-MM via:

1. **TTL trigger output** from JTS-MM's MCU goes into the Qualisys "external trigger" input (Qualisys Oqus supports this).
2. Qualisys then records its frames triggered by the same edges as JTS-MM, putting both systems in lock-step.
3. The JTS-MM `TRIGGER_EVENT` records and the Qualisys frame timestamps share a common edge for every frame.

Fallback (if the Qualisys trigger input is busy or wired for a different experiment):

1. Drive the HAT **visible `SYNC_LED`** (or another co-visible white LED) with a known temporal pattern (e.g. 1 Hz pulses) — **not** an IR ring for retroreflective tracking.
2. Both systems detect the flash in their image streams (Qualisys may see it as a bright blob if in FOV).
3. Post-hoc alignment via cross-correlation of LED-on detections.

### 2.3 Co-registration of coordinate frames

Both systems should report the MTA pose in some shared frame. The simplest co-registration:

1. Place a calibration object (e.g. a printed cube with **AprilTag** corners or a ChArUco board) in both systems' FOV; for Qualisys-only alignment, a cube with retroreflective spheres is acceptable.
2. Each system computes the cube's pose in its world frame.
3. The transform between the two world frames is one Procrustes solve.

### 2.4 Motion protocol

The same motion battery as the IEEE SENSORS 2025 paper §III:

- Mouth opening + closing.
- Right + left lateral excursion.
- Forward (protrusion) + backward (retrusion).
- Cyclic motions.

Each motion: 30 s sequence repeated 3 times.

### 2.5 Comparison metrics

For each frame `t` where both systems report a mandibular pose:

- **Translation RMS error**: `‖t_jtsmm(t) - t_qualisys(t)‖` averaged over the session.
- **Rotation RMS error**: angle of `R_jtsmm · R_qualisys^(-1)`, in degrees.
- **Per-axis bias**: mean of the difference vector, for translation and Euler angles.

A plot of (JTS-MM − Qualisys) over time per axis reveals systematic drifts versus random noise.

Expected results (based on first-principles error propagation from camera intrinsic noise and triangulation geometry):

- Translation RMS error: 0.3–0.8 mm.
- Rotation RMS error: 0.3–0.6°.
- Per-axis bias <0.2 mm, <0.1°.

If results are significantly worse, the typical culprits are: (a) miscalibrated extrinsics on the JTS-MM side; (b) marker labeling errors mapping JTS-MM markers to anatomical positions; (c) too few cameras (2 vs. 3 for Qualisys).

### 2.6 Outputs

- A notebook `notebooks/qualisys_side_by_side.ipynb` containing the analysis and plots.
- A summary table in `report.md`.
- The dual-system HDF5 (JTS-MM + a converted Qualisys HDF5) committed as sample data.

### 2.7 Risks

| Risk | Mitigation |
|------|-----------|
| Qualisys IR strobes glare on Pi cameras / wash out AprilTags | Baffle or angle strobes away from Pi lenses; test in a brief pre-experiment |
| Lab schedule conflict for the Qualisys rig | Book early; have a fallback slot |
| Time alignment poor | Use both the TTL trigger and the LED method redundantly; record both for offline analysis |

## 3. Phantom rig (future work)

### 3.1 Concept

A motorized robotic arm (or a 3-axis stepper rig) holds the MTA + CRA assembly and moves them through a programmed trajectory of known position vs. time. The "true" pose comes from the encoder of the rig; the measured pose comes from JTS-MM. The difference is the absolute measurement error.

This is the gold-standard validation. We do not build it in this project for two reasons: (a) the rig itself is a 1–2 month engineering project; (b) the Qualisys side-by-side gives us most of the same information for the precision question.

### 3.2 Recommended rig

- A small Cartesian XYZ stage (CNC-style; ~30 cm working volume) with a tilt-rotate stage on top.
- Repeatability: <50 µm and <0.05° per axis.
- Open-source: <https://github.com/sienci/sienci-lab-mill> or a generic Marlin CNC chassis.
- Total cost: ~1500 EUR.

The rig's controller (e.g. GRBL or smoothieware) runs a script that emits a TTL pulse every time it reaches a programmed waypoint. JTS-MM captures that pulse as a special "rig event" record and timestamps it on the master clock; the dataset thus contains paired (rig position, JTS-MM measured pose) entries.

### 3.3 Trajectory design

- A **3D grid** of static poses (~100 points covering ±50 mm × ±50 mm × ±20 mm + ±30° tilt + ±30° rotate): measures bias and noise as a function of position.
- A **cyclic motion** at speeds matching jaw opening (~50 mm/s): characterizes dynamic accuracy.
- A **plateau-and-jump** trajectory: measures settling time / latency.

### 3.4 Outputs

- Calibration of any residual JTS-MM systematic error (e.g. a known translation bias).
- Look-up table for accuracy-as-a-function-of-position.
- A characterization paper-ready figure (RMS error vs. true position, RMS error vs. velocity).

### 3.5 Status

The phantom rig is **explicitly out of scope** for this project. It is documented here so that follow-on work has a head start.

## 4. Acceptance criteria for the Qualisys comparison

The Qualisys side-by-side experiment is accepted when:

- [ ] At least one subject's data is recorded with both systems simultaneously.
- [ ] All four primary motions (opening/closing, lateral, protrusion, cyclic) are present and have valid optical pose in both systems for >95 % of frames.
- [ ] The analysis notebook reproducibly produces a translation-RMS-error and rotation-RMS-error number.
- [ ] These numbers are reported in the project's summary report.

## 5. References

- IEEE SENSORS 2025 paper (Müller et al.) — for the baseline precision number (182 ± 47 µm) and the motion protocol.
- Farronato M. et al., 2021 — workflow for mandibular kinematic acquisition with portable systems (validation rig design).
- Nagy Z. et al., 2023 — *"In-vitro accuracy of a novel jaw-tracking technology"* — a phantom-rig validation paper to model our methodology on.
- Abdi A. H. et al., 2018 — *"Minimizing fiducial localization error using sphere-based registration in jaw tracking"* — for the phantom-rig pose-registration approach.
