# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""ml labels: label sidecars from other sources."""

from __future__ import annotations

from pathlib import Path

import click


@click.group("labels")
def labels_cmd() -> None:
    """Create label sidecars without the labeling GUI."""


@labels_cmd.command("from-biomech")
@click.option("--store", "store_dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              required=True, help="biomech output directory (holds cache/labels/<pid>/)")
@click.option("--cohort", "cohort_path", type=click.Path(exists=True, dir_okay=False,
                                                          path_type=Path),
              required=True, help="Cohort manifest (primary motion take per participant)")
@click.option("--out", "out_dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Output directory (default: <store>/ml_labels, git-ignored)")
@click.option("--data-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=None, help="Session folders (default: $JTSMM_DATA_DIR or paths.data_dir "
              "of the study.yaml next to the cohort manifest)")
@click.option("--pid", "pids", multiple=True, help="Only these participants (repeatable)")
def from_biomech_cmd(store_dir: Path, cohort_path: Path, out_dir: Path | None,
                     data_dir: Path | None, pids: tuple[str, ...]) -> None:
    """Write sidecars from the biomech `labels` stage (-1 untracked -> 255 unlabelled)."""
    from ml.io.biomech_labels import export_from_biomech

    out = out_dir or store_dir / "ml_labels"
    try:
        rows = export_from_biomech(store_dir, cohort_path, out, data_dir=data_dir,
                                   pids=list(pids) or None)
    except (ValueError, FileNotFoundError, KeyError) as exc:
        raise click.ClickException(str(exc)) from exc
    for r in rows:
        if r["status"] == "ok":
            click.echo(f"  {r['pid']}: {r['n']} frames ({r['n_unlabelled']} unlabelled) -> "
                       f"{r['path']}")
        else:
            click.echo(f"  {r['pid']}: skipped ({r['reason']})", err=True)
    click.echo(f"Wrote {sum(r['status'] == 'ok' for r in rows)} sidecar(s) to {out}")
