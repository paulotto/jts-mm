<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# biomech

Study-level analysis of recorded JTS-MM sessions (the open-source multimodal jaw tracking
system): jaw kinematics, surface EMG (sEMG), inertial (IMU) data, timing, uncertainty and
statistics. Method definitions are in [`docs/analysis/`](../../docs/analysis/README.md) (its §2
lists every pipeline stage with its method doc); the studies using it are the dataset study
[`experiments/icorr2027`](../../experiments/icorr2027/README.md) and the system-validation study
[`experiments/jtsmm_validation`](../../experiments/jtsmm_validation/README.md).

```bash
pip install -e "software/jtsmm[analysis,dev]" -e software/jaw_tracking -e "software/biomech[all,dev]" -e "software/ml[dev]"
biomech stages
biomech run --config experiments/icorr2027/study.yaml -j 6
biomech status --config experiments/icorr2027/study.yaml
pytest software/biomech
```

| Module | Role |
|---|---|
| `config`, `cohort` | defaults + study YAML (strict); manifest load/validate/draft, aggregate demographics |
| `pipeline`, `stages` | stage registry (+ plugins via the `biomech.stages` entry point, e.g. the `ml` stages), chained cache keys incl. analysis code hash, `run()` with events and cancel; `stages.py` holds the dataset stages, `imu/stage_camerafree.py` and `validation/stages*.py` the others |
| `sensitivity`, `papers` | sensitivity variants and their comparison; paper sets (items, targets, reporting-population variant, pinned snapshot) |
| `results`, `catalog` | `ResultStore` (the only read/write path), snapshots, data dictionary, dataset release; labels, units, rounding of every column |
| `paths`, `io` | data-dir resolution; phase-windowed session access (`SessionData`), session inventory, provenance; `io.context.DataContext` (cached per-participant data for viewer, export and animation) |
| `timebase`, `ensembles` | gap-aware segments, resampling (linear/hold/nearest/slerp), windows; trial ensembles and their aggregation per perspective (trials / participant / cohort) |
| `emg`, `spectral` | cleaning chain (robust mains fit, per-phase line frequency), artifact masks, `EmgPrep` (rest, references, per-phase signals), outcomes; Welch spectra, mean and median frequency (MNF/MDF), movement bandwidth (exploratory X1) |
| `frames`, `kinematics` | landmark roles, maxillary frame; pose chain, cycles, clinical metrics, derivatives (Savitzky–Golay, angular velocity), helical axes, hinge models, coupling |
| `qc` | QC metrics, usability flags, stream products, electrode impedance |
| `imu` | sanitized streams, one-time calibration, opening estimators, onset signals; the camera-free study (X5) |
| `timing` | mechanical/EMG onsets, lead and its chance null, ensemble timing, causal release, the post hoc family C2-E |
| `labels` | intent labels from kinematics; their validity check against a second rule (stage `labels_check`, `labels/kappa.py`) |
| `uncertainty` | Monte Carlo noise floor for the hinge residual |
| `stats` | bootstrap, effect sizes, linear mixed models (LMM), reliability (ICC, SEM, MDC), Bland–Altman, statistical parametric mapping (SPM), variability, requirements, cohort stages (docs/analysis/an07) |
| `viz` | figure/table specs, one layout engine for TikZ/pgfplots + booktabs, matplotlib and Plotly; view registry with paper views, dashboards and the data atlas (`views/data.py`); `report.py` writes per-participant reports |
| `anim` | synchronized videos (ffmpeg), Plotly previews and LaTeX filmstrips from the same specs |
| `check` | `biomech check-outputs`: coverage, numbers, ranges, figure layout/size, reader text, participant counts, export integrity, videos, privacy |
| `validation` | the system-validation study (`ref_*` stages; docs/analysis/an14) |
| `testing` | synthetic ground-truth study generator (`biomech synth`, end-to-end tests) |
| `viewer` | Dash viewer: Overview (run control), Signals (data explorer), Results, Paper, Videos |

**Rules for adding analysis:**
- Write a pure function in a subpackage: arrays or DataFrames in, a tidy DataFrame out.
- Wire it as a stage in `stages.py` and give it a parameter block in `biomech/etc/biomech.yaml`.
- Add its output columns to `biomech/etc/catalog.yaml`.
- Test it on synthetic data in `tests/`.
- Bump the stage `version` when its results change.
