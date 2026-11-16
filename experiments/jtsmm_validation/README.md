<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# JTS-MM system validation study

Validation of JTS-MM against commercial reference systems for the **systems paper**:
Qualisys (optical motion capture, 200 Hz) for the kinematics and Delsys Trigno for the surface
EMG (sEMG).
This is a separate study from `experiments/icorr2027`. It shares the code base
(`software/biomech`), but it has its own cohort, result store, analysis plan and paper set.
No table, figure or claim from here enters the icorr2027 store or papers, and the reverse
holds too. The dataset paper only *cites* systems-paper numbers (VD9, VD10, VD20, VD21) through
the `cite:` block of its numbers file.

**Paper.** A two-participant feasibility validation (P1 and P2), with every result per
participant plus the descriptive mean. P1 and P2 are reported as two separate reference
measurements, without ranking. The error budget is the neutral account of their differences.
The paper is exported from the frozen snapshot `v15`, pinned in `paper_systems.yaml` and in
the `cite:` block of the data-set numbers.

| File | Role |
|---|---|
| `study.yaml` | `biomech` configuration. Runs only the `validation` stage group (`pipeline.groups`). |
| `cohort.yaml` | Participants and their **comparison takes**, i.e. simultaneous JTS-MM + Qualisys + Delsys recordings. Belongs to the data and is not in the public release. |
| `reference.yaml` | Per participant: the QTM (Qualisys Track Manager) export, the Delsys CSVs (take and MVC), the JTS-MM dataset take and MVC take of the same subject (for the cross-session EMG comparison), body correspondences and sync phases.. Belongs to the data and is not in the public release. |
| `analysis_plan.md` | Which data each metric uses (D0b), the tolerances, and the stages, as frozen before any agreement result was looked at. |
| `change_log.md` | Concise log of every deviation and added analysis (VD) and every erratum (EV) after the freeze. |
| `paper_systems.yaml` | The systems paper set, in metrology order: bench (D6); synchronisation (D2/D3, VD19); (I) point marking (VD4); (II) motion, in each system's own frames and frame-aligned with equal value (VD3, VD5), with the error budget (VD22) and the optical + IMU fusion as a secondary result (V4, VD6, VD8); fitness for purpose (VD21 hinge, VD20 camera-free opening, the dataset bridges VD9 / VD10 / VD12); (III) EMG with the side check (D5, VD23); the supplement. It pins `snapshot: v15`. |
| `numbers_systems.yaml` | The in-text numbers of the systems paper (`\numKey` macros → `numbers.tex`): per participant (`…POne`, `…PTwo`) plus the descriptive mean (`…Mean`). All 394 keys resolve from `v11`, including the VD22 / VD23 keys (`budget…`, `gap…`, `cause…`, `sideIndex…`) and the EMG profile agreement added 2026-11-01 (`emgProfileR…`, `emgProfileCmc…`, `emgProfileCmcDefined…`). |

The code is in `software/biomech/biomech/validation/` (sync, registration, sample selection,
stages) and `software/biomech/biomech/io/reference.py` (the QTM `.mat` and Trigno CSV readers).

## Data

The comparison takes are in `data/jtsmm_qualisys_delsys_eval/`, which is git-ignored because it holds subject data.
Each take folder has a `qualisys_emg/` subfolder that holds the QTM export (`.mat`, `.qtm`, `cal/`)
and the Delsys CSVs. The JTS-MM dataset takes are read by path from
`data/pilot_study/jtsmm_data/` (`reference.yaml: dataset_dir`). Results go to
`data/analysis/jtsmm_validation/`.

## Run

```bash
biomech cohort check --config experiments/jtsmm_validation/study.yaml
biomech run --config experiments/jtsmm_validation/study.yaml          # stage ref_sync
biomech show --config experiments/jtsmm_validation/study.yaml ref_sync sync
```

The reference files (QTM, Delsys) are **not** part of the cache key. Only the JTS-MM
`session.h5` files are. After you replace an export, re-run with `--force ref_sync`.

## Stages

- `ref_sync` (per participant, validation plan D2 + D3) estimates the clock model
  `t_Q = t_S + c + α (t_S − t_ref)` together with the spatial registration
  `T_Q,b = X · T_S,b · Y_b` (t_S, t_Q: JTS-MM and Qualisys time; b ∈ CT calibration tool, MP
  mouthpiece, HP head plate). The fit uses the CT sync windows only.
  - It writes the tables `sync`, `sync_methods`, `registration`, `heldout`, `sliding_lag` and `windows`.
  - It writes the edges and the sliding lags as arrays.
  - `sync_methods` compares the coarse methods (CT visibility edges, CT `|ω|` cross-correlation)
    with the joint fine fit and with two sliding-window checks:
    - `|ω|²` correlation (frame-free);
    - registered-pose time shift (precise, but it uses the registration).

Further stages:
- `ref_tip`, `ref_quality`, `ref_kinematics`, `ref_landmarks`, `ref_agreement`,
  `ref_sampling` (VD9), `ref_emg`, `ref_trueness`, `ref_fusion`;
- `ref_imu_lever` (VD15), `ref_imu_accel` (VD17);
- `ref_hinge` (VD21), `ref_camerafree` (VD20);
- `ref_budget` (VD22), `ref_emg_side` (VD23).

`biomech stages` lists them with their dependencies. Every rule is in
[`docs/analysis/an14_system_validation.md`](../../docs/analysis/an14_system_validation.md)
(§2: every stage with what it reads and stores).

**QTM label swaps (EV1).**
- The CT marker triangle is nearly isosceles, so QTM swaps two of its labels and still fits
  the body, rotated ≈ 180°.
- The reader corrects this when the take is read (`io.reference.read_qtm_mat`, on by
  default, Qualisys data only).
- **Recommendation for future recordings:** a CT marker layout whose side lengths differ
  by ≥ 10 mm, or a fourth marker.
