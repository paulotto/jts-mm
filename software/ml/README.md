<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# jtsmm-ml

Jaw-motion intent decoding on JTS-MM `session.h5` recordings: labels, causal EMG + IMU windows,
models, evaluation protocols and fusion. One core serves the `ml` CLI and the biomech stages
`ml_windows` / `ml_eval` and the pseudo-online replay stage `ml_online` (registered through the
`biomech.stages` entry point in `pyproject.toml`; stage overview:
[`docs/analysis/README.md`](../../docs/analysis/README.md) §2).

Specification: [`docs/ml/`](../../docs/ml/README.md). Mathematics of the features, the support vector machine (SVM),
the protocols, the metrics, the permutation and few-shot tests and the pseudo-online replay,
with Methods wording: [`docs/analysis/an05`](../../docs/analysis/an05_timing_and_onsets.md) §5.

## Install

```bash
# from the repository root; jtsmm (and biomech for the stages / from-biomech) are in-repo packages
pip install -e software/jtsmm -e software/biomech -e './software/ml[label,torch,dev]'
```

| Extra | Purpose |
|-------|---------|
| `label` | Dash labeling app (`ml label`) |
| `torch` | GRU (gated recurrent unit) model (the registry lists it as unavailable without torch) |
| `dev` | pytest, ruff |

## Config

The packaged [`etc/ml.yaml`](etc/ml.yaml) is the default; another file is used only with
`--config` (`~/ml.yaml` is not read implicitly). Every command prints the file it used. Sections:
`paths`, `label`, `dataset` (`emg`, `imu`, `window`), `train` (`model`, `feature_set`,
`protocol`, `models`). Parsing is strict.

Data roots: `~/ml_data` (`ML_DATA_DIR`) with `processed/`, `ml_runs/`, `ml_evals/`;
recordings in `~/jtsmm_data` (`JTSMM_DATA_DIR`).

## CLI

```bash
ml label path/to/session.h5                                   # Dash: paint labels (unpainted = ignore)
ml labels from-biomech --store data/analysis/icorr2027 \
    --cohort experiments/icorr2027/cohort.yaml                # sidecars from the biomech labels stage
ml dataset --labels-dir data/analysis/icorr2027/ml_labels     # pool every sidecar
ml dataset path/to/session_dir [--labels FILE] [--label-at end|majority] [--mvc MVC.h5]
ml train --dataset DS --model lda|svm|mlp|gru|late_stack:lda --feature-set emg+imu \
    --protocol loso_subject|loso_session|blocked
ml eval --run RUN [--dataset DS]
ml train --compare RUNS
```

## Layout

```
software/ml/
├── etc/ml.yaml
├── ml/
│   ├── config/      packaged-config loader, schema, class list, data paths
│   ├── core/        signals, windows, features (registry), dataset (WindowDataset, build_windows)
│   ├── models/      registry (linear discriminant analysis LDA, SVM, multilayer perceptron MLP, late fusion), gru (PyTorch), base
│   ├── fusion/      early (BlockScaler), late (mean / weighted / stacking)
│   ├── protocols/   folds (leave-one-subject-out loso_subject, leave-one-session-out loso_session, blocked), evaluate (fit, scores, permutation)
│   ├── datasets/    ml dataset: session.h5 + sidecar -> core
│   ├── io/          session.h5, sidecars, JTS output, from-biomech exporter
│   ├── labeling/    Dash app
│   ├── training/    run directories
│   ├── evaluation/  metrics, ml eval, comparison, plots
│   ├── online/      pseudo-online replay (X4, docs/ml/ml08_online.md): stream, decoder, release, metrics, stage (ml_online)
│   ├── cli/         label, labels, dataset, train, eval
│   └── biomech_stages.py   ml_windows / ml_eval
└── tests/           synthetic sessions only (tests/synth.py); never reads data/
```

## Tests

```bash
cd software/ml && ../jtsmm/.venv/bin/python -m pytest -q && ../jtsmm/.venv/bin/python -m ruff check ml tests
```
