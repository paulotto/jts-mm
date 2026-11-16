<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# ml01 — Ground truth labeling

Where the class labels come from, how they are tied to the master clock, and the label sidecar
that every downstream step reads.

> **Status:** implemented — `ml label` (Dash app), `ml labels from-biomech` (exporter),
> `ml/io/labels.py` (sidecar), `ml/labeling/` (app, trajectory loaders).

## 1. Two label sources, one format

| Source | Command | Use |
|--------|---------|-----|
| biomech `labels` stage: incisal velocity in the maxillary frame with task-specific axes and hysteresis ([`an05`](../analysis/an05_timing_and_onsets.md) §4) | `ml labels from-biomech` | ICORR 2027 analysis, all participants |
| manual painting of the JTS landmark trajectory | `ml label` | validation of the automatic labels, new protocols |

Both write the same sidecar (§4) with the same class ids, so `ml dataset` does not care where a
label came from (`source` attribute).

## 2. `ml labels from-biomech`

```bash
ml labels from-biomech --store data/analysis/icorr2027 \
    --cohort experiments/icorr2027/cohort.yaml [--out DIR] [--data-dir DIR] [--pid s01 ...]
```

- One sidecar per **included** participant, for the **primary** motion take of the cohort
  manifest (secondary takes are never pooled). Output default: `<store>/ml_labels/`
  (`<session folder>_labels.h5`), which is git-ignored because it is subject-derived data.
- The stage stores per phase `<phase>__t_us` and `<phase>__label` (REST = 0 … RIGHT = 6,
  −1 = untracked). −1 becomes 255 (unlabelled).
- `optical_frame` is the nearest frame on `/optical/t_master_us` (the label times are optical
  frame times); an offset above half a frame is an error, the largest one is stored. Phases whose
  label times overlap or are unsorted are an error.
- `subject_id` is set to the participant id, `session_id` to the take's id; `session_path`,
  `mvc_session` and `mvc_ref_uv` (biomech's per-muscle MVC p95 reference from the `emg` stage,
  table `refs`, column `p95_uv`; zero-phase chain) let `ml dataset` find the take and its MVC
  take. `ml dataset` recomputes the reference from the MVC take on the causal chain; the copied
  `mvc_ref_uv` is only a fallback without one ([`ml02`](ml02_emg_preprocessing.md) §5).
- Participants without an `ok` `labels` result in the store, or without their `session.h5`, are
  skipped and reported.
- Session folders are found via `--data-dir`, `$JTSMM_DATA_DIR`, or `paths.data_dir` of the
  `study.yaml` next to the manifest.

## 3. `ml label` (manual)

`ml label <session.h5> [--jts-output jaw_motion.h5] [--jts-config cfg.json] [--labels OUT]
[--host 0.0.0.0] [--port 8050] [--local-only] [--velocity/--no-velocity] [--invert-axis x|y|z]
[--config FILE]` (defaults from `label:` in `ml.yaml`; `--local-only` binds to 127.0.0.1; the
sidecar defaults to `<session_stem>_labels.h5`)

- The operator labels the JTS mandibular landmark trajectory (`T_model_origin_mand_landmark_t`,
  embedded `/jts/` or a `jaw_motion.h5`); without JTS output the app falls back to the
  CRA-relative MTA translation from `/optical`.
- One experiment phase at a time: select a span (box/lasso or slider) and paint a class.
  **Unpainted frames are unlabelled (255) and ignored**; REST exists only where it was painted.
  The *IGNORE* button and *Reset phase* return a span to unlabelled.
- Alignment of JTS frames to optical frames: either the trajectory covers every optical frame, or
  exactly the config's `analysis.experiment.frame_interval` (inclusive). Any other length is an
  error — earlier versions truncated or anchored at frame 0 and silently shifted labels in time.

Operator rules: label the motion, not the hold (a mouth held open is REST); leave ambiguous
transitions unpainted; label complete repetitions.

## 4. Sidecar format (schema 1.1)

```
/labels/
├── t_master_us    (F,) uint64   optical frame times of the labelled span
├── optical_frame  (F,) int64    absolute optical frame index
├── y              (F,) uint8    0..6, 255 = unlabelled
├── bout_id        (F,) int32    contiguous same-value run id, from 1 (REST and 255 included)
└── attrs: class_names, unlabelled_value (255), source, labeler, trajectory_source,
           jts_output, jts_config, jts_group, session_id, subject_id, created_utc,
           schema_version, n_labeled, n_unlabelled, n_rest, class_counts
           [+ from-biomech: session_path, mvc_session, mvc_ref_uv, mvc_ref_source,
              biomech_store, labels_key, labels_run_id, phases, max_frame_offset_us]
```

Labels are stored on the optical grid with explicit `t_master_us`, never on an EMG-rate grid, so
one file serves EMG, IMU and fusion. Schema 1.0 files (no 255; unpainted frames were REST) are
still read. `bout_id` values below 0 (older writers gave all REST the id −1, which made REST one
group that collided across pooled takes) are recomputed on read. Window groups are recomputed
downstream from the window labels anyway ([`ml03`](ml03_feature_extraction.md) §6).

## 5. Validation on load

`read_labels` rejects unknown schemas, non-increasing `t_master_us`, unequal array lengths and
values outside 0..6 / 255. `ml dataset` additionally rejects a sidecar whose `session_id` names
another take and a take without a resolvable subject id ([`ml03`](ml03_feature_extraction.md) §1).
