# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ml train: run an evaluation protocol for one model and feature set."""

from __future__ import annotations

import json
from pathlib import Path

import click

from ml.config.loader import describe_source, load_train_config
from ml.models.registry import LATE_METHODS, names
from ml.protocols.folds import PROTOCOLS


@click.command("train")
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), default=None,
              help="dataset.npz (or its directory) from `ml dataset`")
@click.option("--model", default=None,
              help=f"{' | '.join(names())} | {'|'.join(LATE_METHODS)}:<model> "
              "(default: train.model)")
@click.option("--feature-set", default=None,
              help="emg | imu | emg+imu | emg_nodig | seq | seq+seq_imu ... (default: "
              "train.feature_set)")
@click.option("--protocol", default=None,
              help=f"{' | '.join(PROTOCOLS)} (default: train.protocol)")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="ml.yaml (default: the packaged software/ml/etc/ml.yaml)")
@click.option("--runs-dir", type=click.Path(path_type=Path), default=None,
              help="Override paths.runs_dir")
@click.option("--search/--no-search", default=None, help="Grouped inner hyper-parameter search")
@click.option("--max-train", type=int, default=None, help="Subsample training windows per fold")
@click.option("--n-jobs", type=int, default=None, help="Parallel outer folds")
@click.option("--compare", type=click.Path(exists=True, path_type=Path), default=None,
              help="Aggregate run directories into comparison.md / .csv instead of training")
def train_cmd(dataset: Path | None, model: str | None, feature_set: str | None,
              protocol: str | None, config_path: Path | None, runs_dir: Path | None,
              search: bool | None, max_train: int | None, n_jobs: int | None,
              compare: Path | None) -> None:
    """Train and cross-validate a classifier (writes a run directory)."""
    if compare is not None:
        _compare(compare, runs_dir)
        return
    if dataset is None:
        raise click.UsageError("--dataset is required (or use --compare)")
    click.echo(describe_source(config_path))
    try:
        cfg = load_train_config(config_path, model=model, feature_set=feature_set,
                                protocol=protocol, search=search, max_train=max_train,
                                n_jobs=n_jobs)
    except (ValueError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    from ml.training.run import train

    click.echo(f"  model={cfg.model} feature_set={cfg.feature_set} protocol={cfg.protocol}")
    try:
        run_dir = train(dataset, cfg, runs_dir=runs_dir, config_source=describe_source(config_path))
    except (ValueError, KeyError, ImportError, RuntimeError) as exc:
        raise click.ClickException(f"{type(exc).__name__}: {exc}") from exc
    agg = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))["aggregate"]
    mean, std = agg.get("mean") or {}, agg.get("std") or {}
    click.echo(f"Wrote run -> {run_dir}")
    for k in ("balanced_accuracy", "macro_f1", "accuracy"):
        if k in mean:
            sd = std.get(k)
            click.echo(f"  {k}={mean[k]:.3f}" + (f" ± {sd:.3f} (SD over folds, ddof=1)"
                                                   if sd is not None else ""))


def _compare(root: Path, dest: Path | None) -> None:
    from ml.evaluation.compare import compare_runs, write_comparison_table

    rows = compare_runs(root)
    if not rows:
        raise click.ClickException(f"no runs with metrics.json under {root}")
    md, csv = write_comparison_table(rows, dest or root)
    click.echo(f"Wrote comparison -> {md}\n                -> {csv}")
