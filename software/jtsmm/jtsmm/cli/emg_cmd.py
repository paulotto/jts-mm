# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm emg — electrode checks and impedance measurement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from jtsmm.config.catalog import EMG_GAINS
from jtsmm.emg.check import format_check_table, run_emg_check, run_impedance
from jtsmm.emg.impedance import format_impedance_table


def _emg_override_options(fn: Any) -> Any:
    """Same EMG overrides as `jtsmm record`, so a check tests the record config."""
    opts = [
        click.option("--bias-drive/--no-bias-drive", "bias_drive", default=None,
                     help="Enable/disable the DRL/bias drive (overrides config)"),
        click.option("--bias-channels", default=None, metavar="CH",
                     help="CH1–CH6 that drive the DRL average (overrides config)"),
        click.option("--active-channels", default=None, metavar="CH",
                     help="Only these CH1–CH6 stay powered (overrides config)"),
        click.option("--leadoff/--no-leadoff", "leadoff", default=None,
                     help="DC lead-off of the recording config restored after the check "
                          "(the check itself sets lead-off per sub-session)"),
        click.option("--gain", "default_gain", type=click.Choice([str(g) for g in EMG_GAINS]),
                     default=None, help="PGA gain for CH1-6 (overrides config)"),
    ]
    for opt in reversed(opts):
        fn = opt(fn)
    return fn


def _resolve(config_path: Path | None, port: str | None, overrides: dict[str, Any]) -> dict[str, Any]:
    from jtsmm.config import apply_emg_overrides, load

    resolved = load(config_path)
    gain = overrides.get("default_gain")
    try:
        apply_emg_overrides(
            resolved,
            bias_drive=overrides.get("bias_drive"),
            bias_channels=overrides.get("bias_channels"),
            active_channels=overrides.get("active_channels"),
            leadoff=overrides.get("leadoff"),
            default_gain=int(gain) if gain is not None else None,
            monitoring=False,
        )
    except ValueError as exc:
        # e.g. --no-bias-drive --bias-channels 1, or --active-channels 7 (as in record)
        raise click.BadParameter(str(exc)) from exc
    if port:
        resolved.setdefault("hat", {})["serial_port"] = port
    return resolved


def _restore_mcu(resolved: dict[str, Any], port: str | None) -> None:
    """Re-apply the resolved config: the check leaves the MCU in its last
    sub-session's setup (DRL probe with bias off, or the AC tone on)."""
    from jtsmm.emg.check import restore_mcu_config

    err = restore_mcu_config(resolved, port=port)
    if err:
        click.echo(f"warning: could not restore the MCU config after the check: {err}", err=True)


@click.group("emg")
def emg_group() -> None:
    """EMG electrode and signal-quality tools."""


@emg_group.command("check")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", default=None, help="J_USB1 serial port override")
@click.option("--duration", type=float, default=None,
              help="Lead-off capture seconds (default: host.emg_check.leadoff.duration_s)")
@click.option("--line-hz", type=float, default=None,
              help="Nominal mains frequency (default: host.emg_check.line_hz)")
@click.option("--impedance/--no-impedance", "with_impedance", default=None,
              help="Include the AC |Z| session (default: host.emg_check.impedance.enabled)")
@_emg_override_options
def emg_check_cmd(
    config_path: Path | None,
    port: str | None,
    duration: float | None,
    line_hz: float | None,
    with_impedance: bool | None,
    **overrides: Any,
) -> None:
    """Run lead-off, |Z|, baseline-noise, and line-frequency checks (cv03 §5)."""
    resolved = _resolve(config_path, port, overrides)
    click.echo(
        "EMG check: lead-off + |Z| (leadoff on) → DRL isolation (CH1-6 muted) "
        "→ noise/line (leadoff off)…"
    )
    # None falls back to host.emg_check.* inside run_emg_check (it used to be
    # shadowed by hard-coded click defaults, so the YAML never applied).
    try:
        result = run_emg_check(
            resolved,
            port=port,
            duration_s=duration,
            line_hz=line_hz,
            with_impedance=with_impedance,
        )
    finally:
        _restore_mcu(resolved, port)
    click.echo(format_check_table(result))
    raise SystemExit(0 if result.passed else 1)


@emg_group.command("impedance")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", default=None, help="J_USB1 serial port override")
@click.option("--duration", type=float, default=None,
              help="Capture seconds (default: host.emg_check.impedance.duration_s)")
@click.option("--current-na", type=click.Choice(["6", "24"]), default=None,
              help="Lead-off current (ADS1299 ILEAD_OFF; default: from config)")
@click.option("--export", "export_path", type=click.Path(path_type=Path), default=None,
              help="Write JSON results to this path")
@_emg_override_options
def impedance_cmd(
    config_path: Path | None,
    port: str | None,
    duration: float | None,
    current_na: str | None,
    export_path: Path | None,
    **overrides: Any,
) -> None:
    """Estimate electrode |Z| from the ADS1299 lead-off tone (host-side analysis)."""
    from jtsmm.config.catalog import FULL_CHANNEL_NAMES, DRL_CHANNEL, DRL_NAME
    from jtsmm.config.registry import emg_check_impedance_duration_s

    resolved = _resolve(config_path, port, overrides)
    if duration is None:
        duration = emg_check_impedance_duration_s(resolved)
    i_na = int(current_na) if current_na else None
    click.echo(f"EMG impedance: capturing {duration:.1f} s with lead-off enabled…")
    try:
        results, meta = run_impedance(resolved, port=port, duration_s=duration, current_na=i_na)
    finally:
        _restore_mcu(resolved, port)
    click.echo(
        format_impedance_table(results, channel_names=FULL_CHANNEL_NAMES)
    )
    click.echo(
        f"\n(tone {meta['tone_hz']:.2f} Hz fixed @ {meta['fs_hz']:.0f} SPS, "
        f"I={meta['current_na']} nA, n={meta['n_samples']})"
    )
    if export_path is not None:
        payload: dict[str, Any] = {
            "meta": meta,
            "channels": [
                {
                    "channel": r.channel,
                    "name": (
                        DRL_NAME
                        if r.channel == DRL_CHANNEL
                        else (
                            FULL_CHANNEL_NAMES[r.channel - 1]
                            if r.channel - 1 < len(FULL_CHANNEL_NAMES)
                            else ""
                        )
                    ),
                    "z_kohm": r.z_kohm,
                    "v_tone_uv": r.v_tone_uv,
                    "quality": r.quality,
                    "note": r.note,
                }
                for r in results
            ],
        }
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        click.echo(f"exported: {export_path}")
    bad = [r for r in results if not r.ok]
    raise SystemExit(0 if not bad else 1)
