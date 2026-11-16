# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm config — inspect, validate, audit, apply."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import click
import yaml

from jtsmm.config import (
    apply_mcu,
    build_mcu_apply_payload,
    default_config_path,
    format_apply_command,
    load,
    load_yaml,
    open_hat_serial,
    raise_on_errors,
    validate_strict,
)
from jtsmm.config.coverage import audit_to_json, format_audit_table, run_audit
from jtsmm.config.loader import template_path, user_config_path
from jtsmm.capture.serial_util import find_hat_port


@click.group("config")
def config_group() -> None:
    """Configuration utilities."""


@config_group.command("show")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--mcu-json", is_flag=True, help="Print MCU apply payload JSON only")
def config_show(config_path: Path | None, mcu_json: bool) -> None:
    """Print loaded YAML (validated)."""
    resolved = load(config_path)
    if mcu_json:
        click.echo(json.dumps(build_mcu_apply_payload(resolved), indent=2))
        return
    click.echo(yaml.safe_dump(resolved, sort_keys=False))


@config_group.command("validate")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--no-strict", is_flag=True, help="Skip strict completeness / unknown-key checks")
def config_validate(config_path: Path | None, no_strict: bool) -> None:
    """Exit 1 on hard validation errors."""
    cfg_path = Path(config_path) if config_path else default_config_path()
    doc = load_yaml(cfg_path)
    issues = validate_strict(doc, path=cfg_path, strict=not no_strict)
    for note in (i for i in issues if i.startswith(("warn:", "info:"))):
        click.echo(note, err=True)
    try:
        raise_on_errors(issues)
    except ValueError as exc:
        click.echo(str(exc), err=True)
        raise SystemExit(1) from exc
    click.echo(f"OK: {cfg_path}")


@config_group.command("audit")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--strict", "strict_mode", is_flag=True, help="Exit 1 on any gap")
@click.option("--json", "as_json", is_flag=True, help="Machine-readable output")
def config_audit(config_path: Path | None, strict_mode: bool, as_json: bool) -> None:
    """Parameter coverage report for the config `jtsmm` would load (or --config).

    The file is read without validation, so an incomplete file shows up as
    MISSING_YAML rows; run `jtsmm config validate` for the full check.
    """
    cfg_path = Path(config_path) if config_path else default_config_path()
    try:
        doc = load_yaml(cfg_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        click.echo(f"error: cannot read {cfg_path}: {exc}", err=True)
        raise SystemExit(1) from exc
    report = run_audit(doc)
    if as_json:
        click.echo(audit_to_json(report))
    else:
        click.echo(f"file: {cfg_path}")
        click.echo(format_audit_table(report))
        if not report.ok():
            click.echo("FAIL (strict)" if strict_mode else "WARN")
    if strict_mode and not report.ok():
        raise SystemExit(1)


@config_group.command("init")
@click.option(
    "--dest",
    type=click.Path(path_type=Path),
    default=None,
    help="Destination path (default: ~/jtsmm.yaml)",
)
@click.option("--force", is_flag=True, help="Overwrite existing file")
def config_init(dest: Path | None, force: bool) -> None:
    """Write your operator YAML (~/jtsmm.yaml, or $JTSMM_CONFIG) from the shipped template."""
    dst = (dest or user_config_path()).expanduser()
    src = template_path()
    if dst.exists() and dst.resolve() == src.resolve():
        raise click.ClickException(f"{dst} is the shipped template itself; pick another --dest")
    if dst.is_file() and not force:
        click.echo(f"exists: {dst} (use --force to overwrite)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    click.echo(f"wrote {dst} from {src}")
    click.echo("next: set session.experiment / subject_id and check optical_track before recording")


@config_group.command("apply")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", default=None)
@click.option("--dry-run", is_flag=True, help="Print JSON line only")
@click.option("--persist", is_flag=True, help="Also run config save on MCU after apply")
def config_apply(config_path: Path | None, port: str | None, dry_run: bool, persist: bool) -> None:
    """Apply resolved MCU configuration over USB."""
    resolved = load(config_path)
    cmd = format_apply_command(build_mcu_apply_payload(resolved))
    if dry_run:
        click.echo(cmd.rstrip())
        return

    hat = resolved.get("hat") or {}
    serial_port = find_hat_port(port or hat.get("serial_port"))
    ser = open_hat_serial(serial_port)
    try:
        ok = apply_mcu(ser, resolved, persist=persist)
    finally:
        ser.close()

    if not ok:
        click.echo("MCU did not acknowledge config apply", err=True)
        raise SystemExit(1)
    click.echo(f"config apply OK on {serial_port}")
