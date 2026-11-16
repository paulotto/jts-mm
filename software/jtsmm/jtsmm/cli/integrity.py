# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm integrity — post-session validation (HDF5 + session summary)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from jtsmm.integrity import (
    IntegrityReport,
    check_session,
    format_report,
    format_session_summary,
    resolve_session_h5,
)
from jtsmm.integrity.summary import (
    SessionIntegritySummary,
    load_emg_from_session_h5,
    set_emg_check,
    set_hdf5,
    set_preflight,
    write_session_integrity,
)


def _load_config(config_path: Path | None) -> dict[str, Any] | None:
    if config_path is None:
        return None
    from jtsmm.config import load

    return load(config_path)


def _echo_hdf5_report(report: IntegrityReport, *, plain: bool) -> None:
    if plain:
        for w in report.warnings:
            click.echo(f"warning: {w}")
        for err in report.issues:
            click.echo(f"error: {err}", err=True)
        if report.ok and not report.warnings:
            click.echo(f"integrity: OK — {report.path}")
        elif report.ok:
            click.echo(f"integrity: OK with warnings — {report.path}")
    else:
        click.echo(format_report(report))


def run_integrity_check(
    session_path: Path,
    *,
    config_path: Path | None = None,
    plain: bool = False,
    write_summary: bool = False,
    strict_warn: bool = False,
    verify_config_hash: bool = False,
) -> IntegrityReport:
    """Shared entry for CLI + tests: resolve path, check, optionally write summary.

    ``config_path`` supplies the ``host.integrity`` gates (instead of the ones
    recorded in the session); ``verify_config_hash`` additionally fails when the
    session was not recorded with that config.
    """
    h5_path = resolve_session_h5(session_path)
    cfg = _load_config(config_path)
    report = check_session(
        h5_path,
        expected_config=cfg if verify_config_hash else None,
        resolved_config=cfg,
        config_source=str(Path(config_path).expanduser().resolve()) if config_path else None,
    )

    if write_summary:
        summary = SessionIntegritySummary(session_dir=h5_path.parent)
        set_preflight(
            summary,
            skipped=True,
            skip_reason="not available when re-checking offline (run during `jtsmm record`)",
        )
        emg_detail, emg_passed = load_emg_from_session_h5(h5_path)
        if emg_detail is not None:
            set_emg_check(summary, result=emg_detail, passed=emg_passed)
        else:
            set_emg_check(
                summary,
                skipped=True,
                skip_reason="no /meta/emg_check_json in session",
            )
        set_hdf5(summary, report)
        written = write_session_integrity(h5_path.parent, summary)
        click.echo(format_session_summary(summary))
        click.echo(f"Wrote integrity summary -> {written['integrity.json']}")
    else:
        _echo_hdf5_report(report, plain=plain)

    if not report.ok:
        raise SystemExit(1)
    if strict_warn and report.warnings:
        raise SystemExit(1)
    return report


@click.group("integrity", invoke_without_command=True)
@click.pass_context
def integrity_group(ctx: click.Context) -> None:
    """Validate recorded sessions (HDF5 structure, timing, sync).

    \b
    Examples:
      jtsmm integrity check SESSION.h5
      jtsmm integrity check SESSION_DIR/
      jtsmm integrity check --strict --write-summary SESSION_DIR/
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        raise SystemExit(0)


@integrity_group.command("check")
@click.argument("session_path", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help=(
        "YAML whose host.integrity thresholds / errors_as_warnings apply "
        "(default: the config recorded in the session, then built-in defaults)"
    ),
)
@click.option(
    "--verify-config-hash",
    is_flag=True,
    help="Also fail unless the session's config_hash matches --config",
)
@click.option(
    "--strict-warn",
    "--strict",
    is_flag=True,
    help="Treat warnings as errors",
)
@click.option(
    "--plain",
    is_flag=True,
    help="Print flat warning:/error: lines instead of the summary table",
)
@click.option(
    "--write-summary/--no-write-summary",
    default=False,
    help=(
        "Write integrity.json + integrity.txt into the session directory "
        "(refreshes the HDF5 section; pulls EMG check from /meta when present)"
    ),
)
def integrity_check_cmd(
    session_path: Path,
    config_path: Path | None,
    verify_config_hash: bool,
    strict_warn: bool,
    plain: bool,
    write_summary: bool,
) -> None:
    """Validate a recorded session HDF5 file (or its parent session directory)."""
    try:
        run_integrity_check(
            session_path,
            config_path=config_path,
            plain=plain,
            write_summary=write_summary,
            strict_warn=strict_warn,
            verify_config_hash=verify_config_hash,
        )
    except FileNotFoundError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(1) from exc
    raise SystemExit(0)


# Export name expected by ``cli.main``.
integrity_cmd = integrity_group


if __name__ == "__main__":
    integrity_group.main(standalone_mode=True)
