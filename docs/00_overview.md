<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# 00 — System Overview

## 1. The idea in one paragraph

JTS-MM is an **open-source, low-cost, multimodal jaw acquisition platform** that drops in for the commercial Qualisys-based jaw tracking system described in our IEEE SENSORS 2025 paper, adds **six-channel facial sEMG** and **two IMUs**, and runs entirely on a Raspberry Pi 5 with a custom HAT. The platform is the hardware foundation for research on multimodal sensor fusion and user-intent recognition for adaptive jaw exoskeleton control, and is intended to be replicated by other labs at <1000 EUR per build.

## 2. Motivation

### 2.1 Why extend the existing JTS?

The first-generation jaw tracking system (JTS) achieves ~182 µm precision at 200 Hz using a Qualisys Oqus rig. It is the right tool for **kinematic** characterization of mandibular motion, but it has three limitations for the next step — **intent recognition** for jaw exoskeleton control:

1. **Kinematics alone is insufficient for intent recognition.** EMG carries muscle-activation information that precedes motion onset by 50–150 ms. Without EMG we cannot detect intent before motion starts, which is exactly what an adaptive exoskeleton needs.
2. **The optical motion-capture (Qualisys) rig is not portable, not low-cost, and not open.** A Qualisys Oqus rig is in the 50 kEUR+ class and proprietary. This blocks replication, clinical deployment, and the open-source promise of the original paper.
3. **No ground-truth-free operation.** The current pipeline relies entirely on optical pose. A second, independent modality (IMU) is needed to cross-validate the optical signal, detect marker occlusion, and operate during brief optical drop-outs.

### 2.2 Why these three modalities?

The platform combines kinematic, physiological, and inertial information:

- **Optical kinematics** (cameras): the ground-truth modality. End-effector pose of the mandible.
- **sEMG** (6 channels: digastric, masseter, temporalis × L/R): muscle activation, precedes motion onset, distinguishes "intended" from "compensatory" or "passive" motion.
- **IMU** (jaw + head; optional pointer): orientation, angular velocity and linear acceleration; redundant kinematics that fills optical drop-outs and validates the camera pose. An optional third IMU on the **digitizing pointer** logs pointer attitude during landmark registration (original JTS paper §II.A).

Combined, these three modalities cover the chain *"intent → activation → motion"* and let an intent classifier exploit cross-modal redundancy.

### 2.3 Why a custom HAT instead of OpenBCI?

OpenBCI Cyton-GS is the closest commercial option, but it costs ~$999 USD per board and bundles features we do not need (Bluetooth radio, SD card, plastic case). The KiCAD-open hardware equivalent ([Couby/OpenBCI_hardware_KiCAD](https://github.com/Couby/OpenBCI_hardware_KiCAD)) is a faithful clone of the same board.

The chip that does the actual work — the TI **ADS1299** — costs ~35–40 EUR by itself. A purpose-built HAT around it, with **only** the features we need (6 EMG inputs, 2 IMU interfaces, camera trigger output, USB-CDC to a Pi 5), comes in at roughly **~99 EUR populated-board BOM** (≈121 EUR including the board's share of a 5-up JLC panel; see [`02_components_bom.md`](02_components_bom.md) §2). The cost reduction is ~7–10× per unit and we own the design, schematic, and firmware end to end. See [`04_component_catalog.md`](04_component_catalog.md) for line items and [`02_components_bom.md`](02_components_bom.md) for totals.

## 3. Explicit deltas vs. the IEEE SENSORS 2025 system

The published JTS system is **kept intact and extended** rather than replaced. Concretely:

| Aspect | Original JTS (IEEE SENSORS 2025) | JTS-MM (this project) |
|--------|----------------------------------|------------------------|
| Optical capture | Qualisys Oqus, 3 cameras, ~50 kEUR | 2–3 low-cost cameras, ~150–290 EUR (two tracks: webcams or Pi GS, incl. lenses + tripods); **AprilTag 36h11** on MTA/CRA (not IR retroreflective spheres) |
| Optical precision | (182 ± 47) µm | Track B target (300–500) µm; Track A target 1–2 mm |
| Optical sample rate | 200 Hz | Track B: 40 Hz default (up to 100 Hz, `imx296_high_fps`; HW-triggered); Track A: 30–60 Hz |
| EMG | none | 6 channels, 24-bit ADS1299, 4 kHz, gain 1–24 |
| IMU | none | 2× BNO085 on MTA + CRA (required); **optional 3rd** on digitizing pointer for landmark-digitization QA ([`h04`](hardware/h04_imu_subsystem.md) §3.4) |
| Compute platform | Host PC + Qualisys QTM | Raspberry Pi 5 + custom HAT |
| Time base | OMoCap system clock | MCU master clock, every sample carries `t_master_us` |
| Storage | HDF5 | HDF5 (strict **superset** of the original schema) |
| Data processing | Python pipeline at [paulotto/jaw_tracking_system](https://github.com/paulotto/jaw_tracking_system) | Same pipeline; new modalities slot in as extra HDF5 groups |
| Hardware openness | MTA / CRA / DP printable + CAD published | Same MTA / CRA / DP **plus** open KiCad HAT and open RP2350A firmware |

The original system's mandibular tracking array (MTA), cranial reference array (CRA), and digitizing pointer (DP) are **re-used** with minor mechanical changes: each of MTA and CRA gains an IMU mounting platform and **three flat AprilTag pads** (replacing retroreflective sphere posts in the default build). See [`04_component_catalog.md`](04_component_catalog.md#9-fiducials--scene-lighting). The existing Kabsch-based registration to anatomical landmarks (`CSAnatMand`, `CSAnatMax`) is unchanged. The HDF5 group `/optical/` of the original schema is preserved bit-for-bit.

## 4. Scope and non-goals

> **Default build (the configuration this project validates):** Raspberry Pi 5 (8 GB) + custom analog HAT, **2× Pi Global-Shutter cameras (Track B, hardware-triggered via XTR+ / XTR−)** with an optional 3rd USB3 GS camera, **6-channel ADS1299 EMG**, **2× BNO085 IMUs (I2C)**, and **AprilTag 36h11** fiducials under visible light. **Optional HAT blocks:** USB opto **`TRIG_USB`** (3rd camera) and visible **sync LED** (Track A webcams only). Track A (USB webcams) is the documented low-budget alternative. Everywhere the docs say "default", they mean this configuration.

### 4.1 In scope (delivered by this project)

- Functional 2–3-camera optical tracking with the Pi 5 + libcamera or OpenCV.
- Functional 6-channel EMG acquisition with the custom HAT.
- Functional 2-IMU acquisition with the custom HAT.
- A single shared time base, validated to <100 µs (Track B) or <2 ms (Track A).
- Software pipeline that records to an extended JTS HDF5 file.
- Documentation, BOM, KiCad sources, firmware sources, and a side-by-side comparison study against the existing Qualisys system on at least 1 subject.

### 4.2 Out of scope (deferred / future work)

- **Wireless / wearable operation.** Tethered benchtop only. The HAT is meant to sit next to the subject, not on the subject.
- **Medical-device certification.** The system is a research instrument, not a clinical device. IEC 60601 isolation is *documented* in [`hardware/h07_power_grounding_safety.md`](hardware/h07_power_grounding_safety.md) but not engineered into the prototype.
- **Real-time exoskeleton control loop.** This project delivers the acquisition layer; the closed-loop control demo is left as follow-on work.
- **Pressure / force sensors.** Not part of this hardware iteration. They will live on the exoskeleton itself.
- **Phantom-rig validation.** Replaced by side-by-side Qualisys comparison; phantom rig is described as future work in [`calibration_validation/cv04_phantom_and_qualisys_validation.md`](calibration_validation/cv04_phantom_and_qualisys_validation.md).

## 5. Target users

- **The simulation/robotics group at TU Darmstadt**: uses the system to record data sets for jaw exoskeleton research.
- **External labs**: replicate the system to gather their own TMD-related data without buying a Qualisys rig.

## 6. Key design principles

Every decision in the docs below is justified against these five principles. When in doubt, read them and pick the option that scores best on the union of the five.

1. **Open**. Every artifact (schematics, layout, firmware, software, docs) is released under a permissive license. Components are off-the-shelf and replaceable by alternatives that are also documented in the BOM.
2. **Low-cost**. Target BOM <1000 EUR for a complete benchtop system (Track B ≈ 880–1000 EUR, Track A ≈ 830–910 EUR; see [`02_components_bom.md`](02_components_bom.md) §1). The **optical subsystem alone** is ~150 EUR (Track A) to ~290 EUR (Track B); the cost floor for a full system is set by the Pi 5, HAT, and EMG accessories, not the cameras.
3. **Replicable**. A second lab with a competent builder must be able to build a working copy from this docs tree in ≤2 months, including PCB fab.
4. **Time-coherent**. There is exactly **one** master clock (the MCU). Every datum, every frame, every quaternion carries an MCU timestamp. Cross-modal analysis must never have to guess about alignment.
5. **Backwards-compatible**. The existing JTS pipeline at [paulotto/jaw_tracking_system](https://github.com/paulotto/jaw_tracking_system) keeps running unchanged on any HDF5 file produced by JTS-MM. New modalities live in new groups; existing groups are byte-identical in semantics to the published schema.

## 7. Document conventions

- **Acronyms.** JTS = (first-generation) Jaw Tracking System. JTS-MM = this project (Multimodal). MTA = Mandibular Tracking Array. CRA = Cranial Reference Array. DP = Digitizing Pointer. AFE = Analog Front-End. HAT = Hardware Attached on Top (Raspberry Pi nomenclature). DRDY = Data Ready. DRL = Driven Right Leg (bias drive electrode).
- **References.** Citations to the IEEE SENSORS 2025 paper are by section number. Citations to TI / Bosch / RPi documentation are by datasheet section.
- **Code citations.** Where existing code is referenced, the path is given as `<owner>/<repo>:<path>`.
