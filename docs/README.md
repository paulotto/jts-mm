<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-FileCopyrightText: 2026 David Stebich
SPDX-License-Identifier: CC-BY-4.0
-->

# Open-Source Multimodal Jaw Tracking System (JTS-MM)

This directory contains the complete specification for the **second-generation, multimodal extension** of the open-source jaw tracking system originally published as:

> P.-O. Müller, S. Suppelt, M. Kupnik, O. von Stryk — *"An Optical Measurement System for Open-Source Tracking of Jaw Motions"*, IEEE SENSORS 2025.
> Original repository: [paulotto/jaw_tracking_system](https://github.com/paulotto/jaw_tracking_system).

The first-generation system uses a commercial Qualisys Oqus motion-capture rig (3 cameras, ~200 µm precision, 200 Hz). The **JTS-MM** extension keeps the spirit of the original (open-source, low-cost, biocompatible, modular) but replaces the commercial back-end with consumer hardware and adds two new sensing modalities for jaw exoskeleton research and TMD rehabilitation:

- **2–3 low-cost cameras** for optical kinematics (two tracks documented, see [`hardware/h01_camera_subsystem.md`](hardware/h01_camera_subsystem.md)); fiducials on MTA/CRA under visible lab light (design: **AprilTag 36h11**; the recordings use **ArUco `DICT_4X4_50`**, 30 mm tags, `software/optical/etc/optical.yaml`; no IR rings in the default BOM)
- **6 surface EMG channels** on the masticatory muscles (digastric, masseter, temporalis — left/right)
- **2 IMUs** (mandibular and cranial) for orientation cross-validation
- **A single shared time-base** that ties every sample, frame, and quaternion to one master clock — target jitter **<100 µs for hardware-triggered Track B cameras** and **<2 ms for software-synced Track A webcams** (see [`00_overview.md`](00_overview.md) §4.1).

The **EMG + IMU + timing front-end** is consolidated on a **custom Raspberry Pi 5 HAT** that undercuts the OpenBCI **Cyton-GS** in cost (chip-level BOM vs. ~$999 board) while using the same ADS1299 silicon (so OpenBCI register sequences and example code largely carry over). The IMUs themselves are **body-mounted** on the MTA/CRA and connect to the HAT over shielded cables; only their connectors and pull-ups live on the board.

## Naming and the hardware stack

One project, a few names — used consistently throughout:

| Name | Refers to |
|------|-----------|
| **JTS-MM** | the system: the second-generation **M**ulti**M**odal Jaw Tracking System |
| **JTS** | the first-generation (Qualisys) system from the IEEE SENSORS 2025 paper |
| **`jtsmm`** | the Python package + the `jtsmm` CLI on the Pi 5 |
| **the HAT** | the custom analog/timing board (ADS1299 + RP2350A + IMU/camera I/O) |

**Physical stack (bottom to top):** Raspberry Pi 5 → **M.2 HAT+** (NVMe SSD for session storage) → **custom analog HAT** (this project). "HAT+" is the Raspberry Pi mechanical/electrical spec the custom board conforms to; the M.2 HAT+ underneath is an off-the-shelf carrier for the NVMe drive, and the stacking header must clear it (see [`hardware/h02_custom_hat_shield.md`](hardware/h02_custom_hat_shield.md) §11.5).

## How to read these docs

The docs are organized **top-down**, from system-level intent to concrete component choices. New readers should read them in numbered order:

| Section | Purpose | Audience |
|---------|---------|----------|
| [`00_overview.md`](00_overview.md) | The idea, motivation, what changes vs. the published system | Everyone |
| [`01_architecture.md`](01_architecture.md) | Block diagram, data flow, system modes | Architects, integrators |
| [`02_components_bom.md`](02_components_bom.md) | Budget, cost scenarios, ordering checklist, fab notes | Procurement |
| [`04_component_catalog.md`](04_component_catalog.md) | **Line-item parts only:** MPN, specs, cost, alternatives | PCB assembly, Mouser orders |
| `hardware/` | Per-subsystem hardware design docs | Electronics designers |
| `software/` | Pi 5 / RP2350A software architecture | Software developers |
| `calibration_validation/` | Calibration protocols + validation experiments | Experimentalists |
| `operations/` | Per-session checklists + troubleshooting | Operators |
| `optical/` | ArUco multi-camera tracking (Qualisys replacement) | Camera / kinematics developers |
| [`analysis/`](analysis/README.md) | Offline study analysis (`biomech`): cohort, QC, kinematics, EMG, timing, statistics, export | Analysts, paper authors |
| [`ml/`](ml/README.md) | Jaw-motion intent decoding (`ml`): labels, causal EMG + IMU windows, models, protocols, fusion | ML developers, analysts |
| [`../experiments/`](../experiments/README.md) | Study configurations and evaluation scripts: the dataset and validation studies, the evaluations of the recording system (`experiments/5_*`) | Analysts, paper authors |

### Optical tracking (`optical/`)

| File | Topic |
|------|-------|
| [`01_marker_setup.md`](optical/01_marker_setup.md) | Tag dictionary, ID plan, printable fiducials |
| [`02_calibration_and_frames.md`](optical/02_calibration_and_frames.md) | Multi-cam intrinsics/extrinsics, calibration bundle |
| [`03_pose_estimation_and_output.md`](optical/03_pose_estimation_and_output.md) | PnP/triangulation, HDF5 output (standalone + `/optical/quality`), coordinate frames, integrity |
| [`04_migration_from_qualisys.md`](optical/04_migration_from_qualisys.md) | JTS integration via `aruco_camera` data source |

Implementation: [`software/optical/README.md`](../software/optical/README.md). Configs: [`software/optical/etc/`](../software/optical/etc/).

### Machine learning (`ml/`)

| File | Topic |
|------|-------|
| [`README.md`](ml/README.md) | Architecture (one core, CLI + biomech stages), classes, data contract, quick start |
| [`ml01_ground_truth_labeling.md`](ml/ml01_ground_truth_labeling.md) | Label sources (`ml label`, `ml labels from-biomech`), sidecar format |
| [`ml02_emg_preprocessing.md`](ml/ml02_emg_preprocessing.md) | Causal EMG chain, masks, normalization, leakage rules |
| [`ml03_feature_extraction.md`](ml/ml03_feature_extraction.md) | Windows, window labels, feature registry, dataset file |
| [`ml04_models_and_training.md`](ml/ml04_models_and_training.md) | Model registry, `loso_subject` / `loso_session` / `blocked`, `ml train` |
| [`ml05_evaluation.md`](ml/ml05_evaluation.md) | Metrics, run artifacts, `ml eval`, comparison |
| [`ml06_sensor_fusion.md`](ml/ml06_sensor_fusion.md) | IMU features, early and late fusion |
| [`ml07_array_pipeline.md`](ml/ml07_array_pipeline.md) | biomech stages `ml_windows` / `ml_eval` |

Implementation: [`software/ml/README.md`](../software/ml/README.md). Config: [`software/ml/etc/ml.yaml`](../software/ml/etc/ml.yaml).

## Subsystem index

### Hardware (`hardware/`)

| File | Topic |
|------|-------|
| [`h01_camera_subsystem.md`](hardware/h01_camera_subsystem.md) | Camera Track A (USB webcams) and Track B (Pi global-shutter CSI), with trade-offs |
| [`h02_custom_hat_shield.md`](hardware/h02_custom_hat_shield.md) | HAT mechanical/electrical design philosophy and KiCad layout rules |
| [`h03_emg_frontend.md`](hardware/h03_emg_frontend.md) | ADS1299 schematic, input protection, electrodes, SENIAM placement |
| [`h04_imu_subsystem.md`](hardware/h04_imu_subsystem.md) | BNO085 vs ICM-42688-P, mounting on MTA/CRA, cabling |
| [`h05_microcontroller_firmware.md`](hardware/h05_microcontroller_firmware.md) | RP2350A pin map, task model, PIO, USB protocol, ADS1299 registers |
| [`h05_sensor_pipeline_architecture.md`](hardware/h05_sensor_pipeline_architecture.md) | End-to-end sample flow: dual-core split, SPI/I2C DMA, USB vendor bulk vs CDC, SPSC rings, back-pressure |
| [`h06_synchronization.md`](hardware/h06_synchronization.md) | HW camera trigger + LED-flash fallback + clock disciplining |
| [`h07_power_grounding_safety.md`](hardware/h07_power_grounding_safety.md) | Power tree, star ground, IEC 60601 awareness |
| [`h08_rp2350_flashing.md`](hardware/h08_rp2350_flashing.md) | RP2350A SWD (3-wire Debug Probe), power, BOOTSEL/UF2, optional 10-pin header |
| [`h09_footprint_master_table.md`](hardware/h09_footprint_master_table.md) | Cross-sheet footprint index and BOM field conventions |
| [`h09_component_field_conventions.md`](hardware/h09_component_field_conventions.md) | Component field conventions (BOM / procurement fields in the schematic) |
| [`h10_schematic_logical_review.md`](hardware/h10_schematic_logical_review.md) | Record of the schematic review before the PCB layout (per sheet, ERC summary, findings and changes) |
| [`h10_kicad_pcb_layout_guide.md`](hardware/h10_kicad_pcb_layout_guide.md) | KiCad 10 PCB layout: 4-layer stack-up, AGND/DGND star tie |
| [`h11_mechanical_parts.md`](hardware/h11_mechanical_parts.md) | 3D-printed parts (teeth attachment, markers, tools): versions, design analyses, CAD-vs-configuration check; files in [`../hardware/mechanical/`](../hardware/mechanical/README.md) |

**KiCad schematic capture walkthroughs** (ASCII wiring diagrams, step-by-step; pair with the spec docs above):

| Build guide | Sheet |
|-------------|--------|
| [`h02_hat_plus_schematic_build_guide.md`](hardware/h02_hat_plus_schematic_build_guide.md) | `hat_plus` + top-level buses |
| [`h02_hat_plus_eeprom_flashing.md`](hardware/h02_hat_plus_eeprom_flashing.md) | HAT+ ID EEPROM (`eepmake` / `eepflash`, AT24CS64) |
| [`h07_power_schematic_build_guide.md`](hardware/h07_power_schematic_build_guide.md) | `power` |
| [`h05_mcu_schematic_build_guide.md`](hardware/h05_mcu_schematic_build_guide.md) | `mcu` |
| [`h04_imu_schematic_build_guide.md`](hardware/h04_imu_schematic_build_guide.md) | `imu_connectors` |
| [`h04_imu_cable_assembly.md`](hardware/h04_imu_cable_assembly.md) | IMU harness: 7-pin SH/GH, shielding, crimp |
| [`h04_imu_kicad_implementation.md`](hardware/h04_imu_kicad_implementation.md) | KiCad checklist: SM07B + `IMU_SHLD` → AGND |
| [`h03_emg_schematic_build_guide.md`](hardware/h03_emg_schematic_build_guide.md) | `emg` |
| [`h06_trigger_io_schematic_build_guide.md`](hardware/h06_trigger_io_schematic_build_guide.md) | `trigger_io` |

Conventions: [`hardware/_schematic_build_guide_conventions.md`](hardware/_schematic_build_guide_conventions.md).

### Firmware

| Resource | Topic |
|----------|-------|
| **[`firmware/jtsmm/README.md`](../firmware/jtsmm/README.md)** | **Production** — dual-core polling/DMA, `0x55 0xAA` framed records over USB vendor bulk + CDC |
| [`hardware/h05_microcontroller_firmware.md`](hardware/h05_microcontroller_firmware.md) §7 | Build flow summarized in the MCU spec |
| [`hardware/h08_rp2350_flashing.md`](hardware/h08_rp2350_flashing.md) §6 | SWD wiring, OpenOCD, UF2 |

### Software

| File | Topic |
|------|-------|
| **[`software/jtsmm/README.md`](../software/jtsmm/README.md)** | **Production** — multimodal capture, YAML config, HDF5 sessions |
| **[`software/optical/README.md`](../software/optical/README.md)** | **Optical tracking** — ArUco calibration, pose estimation, experiment phases |
| **[`software/jaw_tracking/README.md`](../software/jaw_tracking/README.md)** | **JTS analysis** — jaw kinematics (vendored IEEE 2025 code) |
| [`schema/README.md`](../schema/README.md) | Shared config contract (JSON Schema v1) |
| [`s01_pi5_software_stack.md`](software/s01_pi5_software_stack.md) | OS, dependencies, real-time scheduling, services |
| [`s02_data_pipeline.md`](software/s02_data_pipeline.md) | Multi-process capture + **canonical HDF5 schema** (strict superset of JTS) |
| [`s03_camera_pipeline.md`](software/s03_camera_pipeline.md) | Cameras + ArUco in `jtsmm record --optical-track` (lores) and standalone [`software/optical/`](../software/optical/README.md) |
| [`s04_emg_signal_processing.md`](software/s04_emg_signal_processing.md) | Filter chain, envelope, features, onset detection |
| [`s05_imu_processing.md`](software/s05_imu_processing.md) | Fusion, drift correction, mandible-vs-cranium quaternion |
| [`s06_time_synchronization.md`](software/s06_time_synchronization.md) | Software clock alignment, PTS handling, drift model |
| [`s07_visualization_storage.md`](software/s07_visualization_storage.md) | Live UI, replay, on-disk file layout (schema itself is in `s02`) |
| [`s08_runtime_configuration.md`](software/s08_runtime_configuration.md) | Runtime configuration (schema v1): every key, defaults, integrity gates |
| [`s09_c_cpp_api_comments.md`](software/s09_c_cpp_api_comments.md) | C/C++ API comment conventions and layout (firmware) |
| [`s09_mains_removal.md`](software/s09_mains_removal.md) | Mains-interference removal in EMG (harmonic fit and subtraction) |

### Calibration & validation (`calibration_validation/`)

| File | Topic |
|------|-------|
| [`cv01_camera_calibration.md`](calibration_validation/cv01_camera_calibration.md) | Intrinsics (ChArUco) and multi-camera extrinsics |
| [`cv02_imu_calibration.md`](calibration_validation/cv02_imu_calibration.md) | 6-position static, Allan variance, magnetometer ellipsoid |
| [`cv03_emg_setup_mvc.md`](calibration_validation/cv03_emg_setup_mvc.md) | Skin preparation, MVC normalization protocol |
| [`cv04_phantom_and_qualisys_validation.md`](calibration_validation/cv04_phantom_and_qualisys_validation.md) | Phantom rig (future work) + side-by-side comparison with Qualisys |
| [`cv05_reference_validation_protocol.md`](calibration_validation/cv05_reference_validation_protocol.md) | Executable reference-validation protocol: step-ladder trueness, Qualisys Oqus frame lock, Delsys Trigno sEMG, what to record for offline evaluation |

### Operations (`operations/`)

| File | Topic |
|------|-------|
| [`op01_setup_checklist.md`](operations/op01_setup_checklist.md) | Per-session setup checklist |
| [`op02_subject_preparation.md`](operations/op02_subject_preparation.md) | Electrode placement, MTA/CRA fit, skin prep |
| [`op03_troubleshooting.md`](operations/op03_troubleshooting.md) | Common failure modes + recovery |
| [`op04_emg_study_safety_sop.md`](operations/op04_emg_study_safety_sop.md) | One-page EMG safety SOP for ethics / operator training |
| [`op05_remote_viewer.md`](operations/op05_remote_viewer.md) | Remote viewer over the LAN |
| [`op06_config_hardware_validation.md`](operations/op06_config_hardware_validation.md) | Validating a configuration on the hardware |

## Quick links

- Original system code: <https://github.com/paulotto/jaw_tracking_system>
- Reference open-hardware EMG board (the design we deliberately undercut on cost): [Couby/OpenBCI_hardware_KiCAD](https://github.com/Couby/OpenBCI_hardware_KiCAD)
- OpenBCI Cyton-GS (the commercial reference): <https://docs.openbci.com/GettingStarted/Boards/CytonGS/>

## Reading the docs as PDF

A built copy of the whole documentation as one book is [`pdf/jtsmm_docs.pdf`](pdf/jtsmm_docs.pdf).
It is built from the public repository (`build_docs_pdf.py` in a checkout of it).

`scripts/docs_pdf/build_docs_pdf.py` typesets every markdown file of the repository (these
docs and the READMEs and reports next to the code; not `data/`) as one book,
`build/docs_pdf/jtsmm_docs.pdf`, with live links between documents. The book follows the
system from the idea to the papers, in parts that each open with a short introduction:
overview and architecture, hardware, firmware, software and data, optical tracking,
calibration and operations, analysis and papers, machine learning. The parts
and their reading order are `PARTS` in the script; a new document that matches no part is
reported. Mermaid diagrams are rendered with mermaid-cli (cached in `build/docs_pdf/mermaid/`).

The build also checks the sources (links and anchors, table rows, code fences, heading
levels), their consistency (`\num` macros, file references, stale snapshot names, the indexes
above), their age against the code they cite (a doc checked without an edit can carry
`<!-- docs-reviewed: YYYY-MM-DD -->`), and the typeset result (overfull lines and tables,
missing glyphs, unresolved links, text outside the type area); the findings are in
`build/docs_pdf/report.md`.

```bash
software/jtsmm/.venv/bin/python scripts/docs_pdf/build_docs_pdf.py                 # the book
software/jtsmm/.venv/bin/python scripts/docs_pdf/build_docs_pdf.py --checks-only
software/jtsmm/.venv/bin/python scripts/docs_pdf/build_docs_pdf.py --volumes       # one PDF per part
software/jtsmm/.venv/bin/python scripts/docs_pdf/build_docs_pdf.py --part hardware
```

Requirements: pandoc ≥ 3.1, TeX Live with LuaLaTeX (Libertinus, KOMA-Script, tcolorbox, listings,
fancyvrb; xurl if present), poppler-utils; for the diagrams Node.js and `@mermaid-js/mermaid-cli`
(`mmdc` on the `PATH`, or `DOCS_PDF_MMDC`; the script also looks in `~/.local/opt/`).
Without `mmdc` a diagram appears as its source.

## Convention

- **Optical markers:** design default **AprilTag 36h11**; the recordings use **ArUco `DICT_4X4_50`** (30 mm tags, two per body; `software/optical/etc/optical.yaml`), on MTA/CRA under visible diffuse lighting. IR retroreflective spheres and per-camera IR rings are an **optional** Qualisys-comparison kit only — not powered from the HAT in the default build ([`hardware/h07_power_grounding_safety.md`](hardware/h07_power_grounding_safety.md) §1).
- Time stamps in this project are always referred to as `t_master_us` and expressed in microseconds since MCU power-on. Every record in every stream carries one. This single rule is what makes the multimodal fusion tractable; respect it ruthlessly.
- Coordinate frames follow the conventions of the original JTS paper: `CSMTA`, `CSCRA`, `CSAnatMand`, `CSAnatMax`, `OOMoCap`, `OVM`.
- All physical units are SI unless explicitly stated. Angles are radians inside code, degrees in plots and docs where convenient.
