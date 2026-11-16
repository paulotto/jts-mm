<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# JTS-MM: open-source multimodal jaw tracking system

[![DOI](https://zenodo.org/badge/1415493323.svg)](https://doi.org/10.5281/zenodo.24275718)

JTS-MM records jaw motion, masticatory muscle activity, and inertial signals on one hardware
clock. Two low-cost global-shutter cameras track fiducial markers on the lower teeth and the
head (six degrees of freedom), a custom Raspberry Pi 5 HAT samples six channels of surface
electromyography (sEMG) and two inertial measurement units (IMUs), and one microcontroller
triggers the cameras and stamps every sample.

This repository holds the hardware, the firmware, the recording software, and the analysis
code of the studies that use the system. It extends the optical jaw tracking system of

> P.-O. Müller, S. Suppelt, M. Kupnik, O. von Stryk, "An Optical Measurement System for
> Open-Source Tracking of Jaw Motions", 2025 IEEE SENSORS,
> doi: [10.1109/SENSORS59705.2025.11330651](https://doi.org/10.1109/SENSORS59705.2025.11330651)
> ([paulotto/jaw_tracking_system](https://github.com/paulotto/jaw_tracking_system)).

## What you can do with it

| Goal | Start here |
|---|---|
| Build the system | [`docs/README.md`](docs/README.md) (reading order), `hardware/jtsmm_hat/` (KiCad), `hardware/mechanical/` (printed parts), `firmware/jtsmm/` |
| Record | [`software/jtsmm/README.md`](software/jtsmm/README.md), [`software/optical/README.md`](software/optical/README.md), `docs/operations/` |
| Analyse recordings | [`software/biomech/README.md`](software/biomech/README.md), [`docs/analysis/README.md`](docs/analysis/README.md) |
| Decode movement intent | [`software/ml/README.md`](software/ml/README.md), `docs/ml/` |
| Check a number of a paper | the paper-to-code map below |

## Papers and where their analysis lives

Three manuscripts use this repository. Each analysis has a study folder with its
configuration, its analysis plan, and the declarations of every figure, table, and number.

| Paper | Study folder | Plan and change log | Outputs |
|---|---|---|---|
| Data set: jaw exoskeleton requirements from kinematics, wearable sensing, and muscle timing | `experiments/icorr2027/` | `analysis_plan.md` (as frozen), `change_log.md` | `paper_dataset.yaml`, `numbers_dataset.yaml`, `supplementary_results/` |
| Camera-free: jaw tracking with two IMUs against the optical reference | `experiments/icorr2027/` | the same plan, analysis X5 | `paper_camerafree.yaml`, `numbers_dataset.yaml` |
| System: JTS-MM and its validation against commercial reference systems | `experiments/jtsmm_validation/` | `analysis_plan.md` (as frozen), `change_log.md` | `paper_systems.yaml`, `numbers_systems.yaml` |

Every figure, table, and in-text number of the papers is generated from a declaration in
these files by `biomech export`. The methods are documented per pipeline stage in
`docs/analysis/`.

Each analysis plan is published as it was frozen, before any outcome was computed. Its
`change_log.md` lists every later change in one row: amendments, corrected errors, deviations,
and added analyses, each with its reason and with whether results had been seen. Analyses that
were added after the freeze have a script in `scripts/audit_recompute/` and an entry in the
change log.

`experiments/icorr2027/supplementary_results/` holds the supplementary figures and tables of
the data-set paper with the data behind them, and names the frozen snapshot of the processed
data with its checksum.

**Versions.** This repository has one commit per release and no development history. The
papers and the items in `supplementary_results/` name the commit of the working repository
from which they were exported: `865fa319`. This release was built from commit
`e7d71426` of the working repository, and the build checks that the analysis code
is identical in both.

**Hash of the analysis code:** `555904d8898fc557`. Every result carries the hash of the analysis
source that computed it. The frozen snapshots of the papers (dataset `r17a`, analysis of record `r17`, validation `v15`) carry this hash,
and this repository computes the same one:

```bash
python -c "from biomech import pipeline; print(pipeline.code_hash())"
```

## Data

The recordings are not part of this repository. They are available from the authors on
request, and a data descriptor is planned.

## What this release leaves out

- **Participant files.** The files that describe individual participants (cohort files,
  session names, recording dates) belong to the data. Passages of the documentation that
  named them are shortened and marked.
- **Working material.** Internal work plans, review notes, the bench bring-up code that the
  firmware replaced, concept studies, and third-party datasheets and manufacturer 3D models
  stay in the working repository. Some documents still mention such files.
- **Paper targets of the Makefiles.** Targets that export or check the papers
  (`paper-*`, `verify`, `sync-paper*`) need the processed data and the manuscripts.

Without the recordings you can still run the complete pipeline on a synthetic study with a
known truth:

```bash
biomech synth --out /tmp/jtsmm_synth        # writes sessions, cohort, study.yaml, and runs every stage
```

## Installation and tests

Python 3.10 or newer. From the repository root:

```bash
# the Makefiles and the script tests of the recorder look for the environment at this path
python -m venv software/jtsmm/.venv && . software/jtsmm/.venv/bin/activate
pip install -e "software/jtsmm[analysis,dev]" -e software/jaw_tracking \
    -e "software/biomech[all,dev]" -e "software/ml[dev]" -e "software/optical[dev]"
pytest software/biomech                      # fast suite; add -m "slow or ml or not slow" for all
pytest software/optical/tests
pytest software/ml
pytest software/jtsmm
```

The firmware has its own build and host tests (`firmware/jtsmm/README.md`). The recorder runs
on a Raspberry Pi 5 (`scripts/install_pi.sh`).

## Contents

| Path | Contents | Papers | Licence |
|---|---|---|---|
| `firmware/jtsmm/third_party/cjson` | cJSON parser used by the firmware (third party). | systems | MIT |
| `firmware/jtsmm/pio/ws2812.pio` | WS2812 PIO program from the Raspberry Pi Pico examples (third party). | systems | BSD-3-Clause |
| `software/jtsmm/viewer_web/vendor/three.module.js` | three.js for the live viewer (third party). | systems | MIT |
| `software/jtsmm/viewer_web/vendor/uPlot*` | uPlot for the live viewer (third party). | systems | MIT |
| `software/jaw_tracking` | Jaw-kinematics pipeline of the first-generation optical system (IEEE SENSORS 2025), vendored; keeps its licence CC BY-NC-SA 4.0. | systems, data set, camera-free | CC-BY-NC-SA-4.0 |
| `pyproject.toml`<br>`requirements.txt`<br>`requirements_pi.txt`<br>`…` | Repository-wide configuration and the documentation link test. | all | MIT |
| `docs/README.md`<br>`docs/pdf/jtsmm_docs.pdf`<br>`docs/00_overview.md`<br>`…` | Documentation index, system overview, architecture, and components. | all | CC-BY-4.0 |
| `firmware/jtsmm` | RP2350 firmware of the HAT: acquisition, master clock, USB framing. | systems | MIT |
| `hardware/jtsmm_hat`<br>`hardware/jtsmm_hat_schematics.pdf` | KiCad project of the Raspberry Pi 5 HAT: schematic, PCB, libraries, production files, EEPROM. | systems | CERN-OHL-P-2.0 |
| `hardware/mechanical/viewer/vendor` | OrbitControls of three.js for the 3D viewer of the parts (third party). | systems | MIT |
| `hardware/mechanical/analyses/**/scripts`<br>`hardware/mechanical/viewer/index.html`<br>`hardware/mechanical/viewer/viewer.js`<br>`…` | Scripts of the design and concept studies of the tooth attachment, and the 3D viewer of the parts. | systems | MIT |
| `hardware/mechanical/analyses` | Design and concept studies of the tooth attachment (report, figures, CAD exports). | systems | CC-BY-4.0 |
| `hardware/mechanical/tools`<br>`hardware/mechanical/Makefile`<br>`hardware/mechanical/.gitignore` | Scripts that build the parts tables, renders and variants from the FreeCAD files. | systems, data set | MIT |
| `hardware/mechanical/*.md`<br>`hardware/mechanical/tables`<br>`hardware/mechanical/renders`<br>`…` | Parts tables, renders and figures of the 3D-printed parts. | systems, data set | CC-BY-4.0 |
| `hardware/mechanical` | 3D-printed parts: teeth attachment, markers, calibration tool (FreeCAD, STL, print files, parts.yaml). | systems, data set | CERN-OHL-P-2.0 |
| `scripts/bench` | EMG bench measurements (noise, impedance). | systems | MIT |
| `hardware/bench` | Results of the EMG bench measurements. | systems | CC-BY-4.0 |
| `software/jtsmm` | Raspberry Pi 5 host software: multimodal capture, configuration, HDF5 sessions, live viewer. | systems, data set, camera-free | MIT |
| `software/optical` | Optical tracking: multi-camera ArUco calibration and pose estimation. | systems, data set, camera-free | MIT |
| `schema` | Configuration schema shared by firmware and host. | systems | MIT |
| `aruco_markers` | ArUco marker images. | systems | CC-BY-4.0 |
| `scripts/build_install.sh`<br>`scripts/check_session_integrity.sh`<br>`scripts/flash_swd.sh`<br>`…` | Installation, flashing and network scripts of the recorder. | systems | MIT |
| `docs/calibration_validation/*.py` | Simulation scripts of the synchronisation estimate. | systems | MIT |
| `docs/hardware`<br>`docs/software`<br>`docs/optical`<br>`…` | Hardware, software, optical, calibration and operation documentation. | systems | CC-BY-4.0 |
| `experiments/5_*`<br>`experiments/data` | Evaluation scripts of the system experiments (accuracy, marker size, pivot, hand-eye, timing, EMG, IMU); smoke-tested by the biomech tests. | systems | MIT |
| `experiments/jtsmm_validation` | Validation study against Qualisys and Delsys: study configuration, analysis plan, paper numbers. | systems | MIT |
| `experiments/jtsmm_validation/*.md` | Analysis plan of the validation study. | systems | CC-BY-4.0 |
| `software/biomech` | Analysis pipeline: kinematics, EMG, IMU, statistics, figures and paper export. | data set, camera-free, systems | MIT |
| `software/ml` | Intent decoding: features, models, offline and simulated real-time evaluation. | data set, camera-free | MIT |
| `experiments/icorr2027/study.yaml`<br>`experiments/icorr2027/Makefile`<br>`experiments/icorr2027/numbers_dataset.yaml`<br>`…` | Data-set study: configuration, paper sets, numbers and captions of both papers. | data set, camera-free | MIT |
| `experiments/icorr2027/README.md`<br>`experiments/icorr2027/analysis_plan.md`<br>`experiments/icorr2027/change_log.md`<br>`…` | Analysis plan of the data-set study and the supplementary results of the JNER paper. | data set, camera-free | CC-BY-4.0 |
| `scripts/audit_recompute` | Scripts of the analyses added after the plan was frozen; they write the result files that the derived tables of the paper exports read. | data set, camera-free, systems | MIT |
| `docs/analysis`<br>`docs/ml` | Method documentation of the analysis and the decoder. | data set, camera-free, systems | CC-BY-4.0 |

Papers: *systems* = the JTS-MM system and its validation against commercial reference
systems. *data set* = the multimodal jaw-motion study. *camera-free* = jaw tracking from the
IMUs alone.

## How to cite

Please cite the article that describes what you use, and this repository for the system and
the code (see [`CITATION.cff`](CITATION.cff)). Every release is archived on Zenodo:
[10.5281/zenodo.24275718](https://doi.org/10.5281/zenodo.24275718) resolves to the latest
version, and each version has its own DOI (version 1.0.2, cited by the papers:
[10.5281/zenodo.24283858](https://doi.org/10.5281/zenodo.24283858); version 1.0.1:
[10.5281/zenodo.24281877](https://doi.org/10.5281/zenodo.24281877); version 1.0.0:
[10.5281/zenodo.24275719](https://doi.org/10.5281/zenodo.24275719)). Cite the version you used.

## Licence

Every file states its copyright and licence in an SPDX header. Files that cannot carry a
header (binary, CSV, JSON, CAD) are listed in [`REUSE.toml`](REUSE.toml). The licence texts
are in [`LICENSES/`](LICENSES/).

| Licence | Applies to |
|---|---|
| MIT | software, firmware, scripts, tests and configuration |
| CERN-OHL-P-2.0 | hardware design files (KiCad, FreeCAD, STL, print and production files) |
| CC-BY-4.0 | documentation, figures, renders and result tables |
| MIT | cJSON parser used by the firmware (third party). © 2009-2017 Dave Gamble and cJSON contributors |
| BSD-3-Clause | WS2812 PIO program from the Raspberry Pi Pico examples (third party). © 2020 Raspberry Pi (Trading) Ltd. |
| MIT | three.js for the live viewer (third party). © 2010-2023 Three.js Authors |
| MIT | uPlot for the live viewer (third party). © 2016-2024 Leon Sorokin |
| CC-BY-NC-SA-4.0 | Jaw-kinematics pipeline of the first-generation optical system (IEEE SENSORS 2025), vendored; keeps its licence CC BY-NC-SA 4.0. |
| MIT | OrbitControls of three.js for the 3D viewer of the parts (third party). © 2010-2023 Three.js Authors |

## Acknowledgement

Partially funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation),
project number 450821862 (Research Training Group 2761 LokoAssist).
