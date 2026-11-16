# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``ml train``: run one protocol on a dataset and write a run directory.

The folds, inner search, fitting and scoring are :mod:`ml.protocols.evaluate` --
the same code the biomech ``ml_eval`` stage runs. This module only writes the
artifacts (ml05 §4): per fold ``metrics.json``, ``predictions.npz`` (row indices
into the dataset) and ``model.joblib``; the aggregate ``metrics.json`` with the
fold mean and the sample SD (ddof = 1); confusion matrix and per-class F1 plots;
``summary.json``; and the effective config, seeds, dataset reference, environment
and git commit.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from ml.config.classes import CLASS_COUNT
from ml.config.loader import config_hash
from ml.config.paths import expand_path
from ml.config.schema import TrainConfig
from ml.core.dataset import WindowDataset
from ml.evaluation.metrics import compute_metrics
from ml.protocols import evaluate as ev
from ml.training.artifacts import (new_run_dir, write_aggregate_artifacts, write_env,
                                   write_fold_artifacts, write_git_commit, write_json, write_yaml)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def train(dataset_path: str | Path, cfg: TrainConfig, *, runs_dir: str | Path | None = None,
          config_source: str = "") -> Path:
    """Run ``cfg.protocol`` for ``cfg.model`` on ``cfg.feature_set``; return the run dir."""
    set_global_seed(cfg.seed)
    ds = WindowDataset.load(dataset_path)
    root = expand_path(runs_dir or cfg.output.runs_dir)
    label = re.sub(r"[^A-Za-z0-9.+-]+", "-", f"{cfg.model}_{cfg.feature_set}_{cfg.protocol}")
    run_dir = new_run_dir(root, model=label, config_hash=config_hash(cfg))

    write_yaml(run_dir / "config.effective.yaml",
               {"source": config_source, "train": _jsonable(asdict(cfg))})
    write_json(run_dir / "seeds.json", {"seed": cfg.seed})
    write_json(run_dir / "dataset_ref.json", {
        "dataset_path": str(Path(ds.meta.get("path", dataset_path))),
        "summary": ds.summary(),
        "build": {k: ds.meta.get(k) for k in ("config_hash", "normalization", "git_commit")
                  if k in ds.meta}})
    write_env(run_dir / "env.json")
    write_git_commit(run_dir / "git_commit.txt", cwd=Path(dataset_path).resolve().parent)

    results = ev.run(ds, cfg.feature_set, cfg.model, cfg.protocol, seed=cfg.seed,
                     inner=cfg.inner_folds, max_train=cfg.max_train, search=cfg.search,
                     grid=cfg.grid_for(cfg.model), params=cfg.params_for(cfg.model),
                     probability=cfg.probability, keep_models=True, n_jobs=cfg.n_jobs,
                     k=cfg.blocked_k, guard_s=cfg.guard_s)
    if not results:
        raise RuntimeError("no fold could be trained (every training split had < 2 classes)")

    fold_metrics, yt, yp, pr = [], [], [], []
    for i, r in enumerate(results):
        te = r.fold.test
        proba = r.proba if r.proba is not None else np.zeros((te.size, CLASS_COUNT))
        m = {k: v for k, v in compute_metrics(ds.y[te], r.y_pred, y_proba=r.proba).items()
             if k != "classification_report"}
        s = ev.fold_scores(ds, r)
        m.update({k: s[k] for k in ("latency_ms_median", "detected_frac", "fa_frac", "fa_per_min",
                                    "free_balanced_accuracy", "free_macro_f1") if k in s})
        m.update({"fold": i, "test_unit": r.fold.unit, "n_train": int(r.fold.train.size),
                  "n_test": int(te.size), "n_free": int(r.fold.free.size),
                  "fit_time_s": r.fit_time_s, "best_params": r.best_params, **r.extra})
        fold_metrics.append(m)
        fd = run_dir / "per_fold" / f"fold_{i}"
        write_fold_artifacts(fd, metrics=m, y_true=ds.y[te], y_pred=r.y_pred, y_proba=proba,
                             t_end_us=ds.t_end_us[te], groups=ds.groups[te], indices=te,
                             free_indices=r.fold.free, free_pred=r.free_pred)
        joblib.dump(r.model, fd / "model.joblib")
        yt.append(ds.y[te])
        yp.append(r.y_pred)
        pr.append(proba)

    agg = write_aggregate_artifacts(run_dir, fold_metrics=fold_metrics,
                                    y_true_all=np.concatenate(yt), y_pred_all=np.concatenate(yp),
                                    y_proba_all=np.concatenate(pr) if cfg.probability else None)
    write_json(run_dir / "summary.json", {
        "model": cfg.model, "feature_set": cfg.feature_set, "protocol": cfg.protocol,
        "n_folds": len(results), "test_units": [r.fold.unit for r in results],
        "best_params_per_fold": [r.best_params for r in results],
        "aggregate": {"mean": agg.get("mean"), "std": agg.get("std")},
        "std_ddof": 1})
    return run_dir


def _jsonable(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o
