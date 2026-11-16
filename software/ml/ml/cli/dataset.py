# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ml dataset: build a windowed dataset from session.h5 + label sidecars."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import click

from ml.config.loader import describe_source, load_dataset_config, load_ml_config
from ml.config.paths import expand_path, session_data_dir


@click.command("dataset")
@click.argument("sessions", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--labels", "label_path", type=click.Path(exists=True, path_type=Path),
              default=None,
              help="Sidecar for a single session (default: <stem>_labels.h5 beside it)")
@click.option("--labels-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Directory of sidecars (<session folder>_labels.h5), e.g. from "
              "`ml labels from-biomech`; without SESSIONS every sidecar in it is used")
@click.option("--data-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Search session_*/ here when neither SESSIONS nor --labels-dir is given")
@click.option("--mvc", "mvc_path", type=click.Path(exists=True, path_type=Path), default=None,
              help="MVC take for normalization: mvc (single session)")
@click.option("--label-at", type=click.Choice(["end", "majority"]), default=None,
              help="Override window.label_at (majority = legacy purity rule)")
@click.option("--skip-errors", is_flag=True, help="Skip takes that fail instead of aborting")
@click.option("--config", "config_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="ml.yaml (default: the packaged software/ml/etc/ml.yaml)")
@click.option("--output", type=click.Path(path_type=Path), default=None,
              help="Output directory (default: paths.processed_dir/<session_id or pool_...>)")
def dataset_cmd(sessions: tuple[Path, ...], label_path: Path | None, labels_dir: Path | None,
                data_dir: Path | None, mvc_path: Path | None, label_at: str | None,
                skip_errors: bool, config_path: Path | None, output: Path | None) -> None:
    """Windows + features (EMG, IMU, GRU sub-frames) from session.h5 and label sidecars.

    \b
    ml dataset path/to/session_dir                      # one take, sidecar beside it
    ml dataset --labels-dir data/analysis/icorr2027/ml_labels            # pool all
    ml dataset session_a session_b --labels-dir DIR     # selected takes
    """
    from dataclasses import replace

    from ml.datasets.builder import build_dataset, build_pool
    from ml.datasets.discover import resolve_session_inputs

    click.echo(describe_source(config_path))
    cfg = load_dataset_config(config_path)
    if label_at:
        cfg = replace(cfg, window=replace(cfg.window, label_at=label_at))
    paths = dict(load_ml_config(config_path).get("paths") or {})
    try:
        if len(sessions) == 1 and label_path is not None:
            refs = [(_h5(sessions[0]), label_path)]
        else:
            if label_path is not None:
                raise click.UsageError("--labels is only valid with exactly one session")
            refs = [(r.session_path, r.label_path) for r in resolve_session_inputs(
                list(sessions) or None, data_dir=data_dir or session_data_dir(paths),
                labels_dir=labels_dir)]
        if len(refs) == 1:
            ds = build_dataset(refs[0][0], label_path=refs[0][1], config=cfg, mvc_path=mvc_path)
            name = str(ds.session[0])
        else:
            if mvc_path is not None:
                raise click.UsageError("--mvc is only valid for a single session")
            ds = build_pool(refs, config=cfg, on_error="skip" if skip_errors else "raise")
            name = (f"pool_{len(set(ds.session.tolist()))}sess_"
                    f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    except click.UsageError:
        raise
    except Exception as exc:
        raise click.ClickException(f"{type(exc).__name__}: {exc}") from exc
    out = expand_path(output or Path(cfg.output.processed_dir) / name)
    path = ds.save(out)
    s = ds.summary()
    click.echo(f"Wrote {path}")
    click.echo(f"  windows={s['n_windows']} subjects={s['subjects']} groups={s['n_groups']}")
    click.echo(f"  blocks={s['blocks']}")
    click.echo(f"  class_counts={s['class_counts']}")
    rej = [m.get("rejected") for m in ds.meta.get("sources", [ds.meta])]
    click.echo(f"  rejected={rej}")
    for f in ds.meta.get("failures") or []:
        click.echo(f"  failed: {f['session_path']}: {f['error']}", err=True)


def _h5(p: Path) -> Path:
    return p / "session.h5" if p.is_dir() else p
