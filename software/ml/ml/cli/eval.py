# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ml eval: re-score a training run or one fitted model with the shared metric suite."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click

from ml.config.loader import describe_source, load_ml_config


@click.command("eval")
@click.option("--run", "run_dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Run directory from `ml train`")
@click.option("--dataset", type=click.Path(exists=True, path_type=Path), default=None,
              help="dataset.npz: with --run reload the fold models; required with --model-path")
@click.option("--model-path", type=click.Path(exists=True, path_type=Path), default=None,
              help="One fitted model.joblib (scored on every window it is given)")
@click.option("--feature-set", default=None, help="Feature set of --model-path")
@click.option("--out-dir", type=click.Path(path_type=Path), default=None,
              help="Output (default: <run>/eval or paths.evals_dir/<stamp>)")
@click.option("--compare", type=click.Path(exists=True, path_type=Path), default=None,
              help="Aggregate run directories into comparison.md / .csv")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="ml.yaml (default: the packaged software/ml/etc/ml.yaml)")
def eval_cmd(run_dir: Path | None, dataset: Path | None, model_path: Path | None,
             feature_set: str | None, out_dir: Path | None, compare: Path | None,
             config_path: Path | None) -> None:
    """Evaluate fold predictions (by dataset row index) or a fitted model."""
    from ml.core.dataset import WindowDataset
    from ml.evaluation import evaluate as E

    if compare is not None:
        from ml.cli.train import _compare

        _compare(compare, out_dir)
        return
    if (run_dir is None) == (model_path is None):
        raise click.UsageError("pass exactly one of --run or --model-path (or --compare)")
    click.echo(describe_source(config_path))
    try:
        if run_dir is not None:
            res = (E.evaluate_run_models(run_dir, WindowDataset.load(dataset)) if dataset
                   else E.evaluate_run_predictions(run_dir))
            dest = out_dir or run_dir / "eval"
        else:
            if dataset is None or feature_set is None:
                raise click.UsageError("--model-path needs --dataset and --feature-set")
            res = E.evaluate_model_path(model_path, WindowDataset.load(dataset), feature_set)
            from ml.config.paths import evals_dir

            paths = dict(load_ml_config(config_path).get("paths") or {})
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            dest = out_dir or evals_dir(paths) / f"{stamp}_{model_path.parent.name}"
    except click.UsageError:
        raise
    except (ValueError, KeyError, FileNotFoundError) as exc:
        raise click.ClickException(f"{type(exc).__name__}: {exc}") from exc
    E.write_evaluation(dest, res)
    m = res.metrics
    click.echo(f"Wrote eval -> {dest}")
    click.echo(f"  balanced_acc={m['balanced_accuracy']:.3f}  macro_f1={m['macro_f1']:.3f}  "
               f"accuracy={m['accuracy']:.3f}  n={len(res.predictions.y_true)}")
