# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""jtsmm doctor — host pre-flight checks and hardware IMU rate validation."""

from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path
from typing import Any

import click

from jtsmm.doctor import run_preflight
from jtsmm.doctor.preflight_issues import partition_preflight_issues


@click.group("doctor", invoke_without_command=True)
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--port", default=None, help="J_USB1 serial port override")
@click.option("--no-serial", is_flag=True, help="Skip MCU serial probe")
@click.option("--mcu-selftest", is_flag=True, help="Run firmware selftest over USB")
@click.pass_context
def doctor_cmd(
    ctx: click.Context,
    config_path: Path | None,
    port: str | None,
    no_serial: bool,
    mcu_selftest: bool,
) -> None:
    """Run host pre-flight checks (default) or a subcommand like `imu-rates`."""
    ctx.ensure_object(dict)
    ctx.obj["port"] = port
    ctx.obj["config_path"] = config_path
    if ctx.invoked_subcommand is not None:
        return

    from jtsmm.config import load

    resolved = None
    if config_path is not None:
        resolved = load(config_path, validate=False)
    elif port is not None:
        resolved = load(None, validate=False)
    if resolved is not None and port:
        resolved.setdefault("hat", {})["serial_port"] = port
    issues = run_preflight(
        resolved=resolved,
        config_path=config_path if resolved is None else None,
        check_serial=not no_serial,
        check_mcu_selftest=mcu_selftest,
        # Exit non-zero when the HAT is missing: a bare `jtsmm doctor` used to
        # warn and exit 0 with nothing connected.
        require_hat=not no_serial,
    )
    if not issues:
        click.echo("doctor: OK")
        raise SystemExit(0)

    errors, warnings, infos = partition_preflight_issues(issues)
    for w in warnings:
        click.echo(f"warning: {w}")
    for info in infos:
        click.echo(info)
    exit_code = 0
    for err in errors:
        click.echo(f"error: {err}", err=True)
        exit_code = 1
    raise SystemExit(exit_code)


# Presets exercised by --all-presets (plan Phase 4 matrix A/C/E + raw_only).
_MATRIX_PRESETS = (
    "fused_only", "kinematics_accel", "kinematics_default", "datasheet_motion", "raw_only",
)


def _build_config(base: Path | None, preset: str | None, slot: str) -> dict[str, Any]:
    """Resolve a config with ``preset`` applied to the requested IMU slot(s)."""
    from jtsmm.config.loader import default_config_path, load_yaml
    from jtsmm.config.profiles import resolve_profiles

    raw = load_yaml(Path(base) if base else default_config_path())
    if preset is not None:
        mcu = raw.setdefault("mcu", {})
        slots = ("imu_mta", "imu_cra") if slot == "both" else (f"imu_{slot}",)
        for s in slots:
            block = copy.deepcopy(mcu.get(s) or {})
            block["profile"] = preset
            block.pop("reports", None)
            mcu[s] = block
    return resolve_profiles(raw)


def _print_case(result: Any) -> bool:
    from jtsmm.doctor.imu_rates import CaseResult

    assert isinstance(result, CaseResult)
    click.echo(f"\n=== {result.label} ({result.duration_s:.0f} s) ===")
    if result.note:
        click.echo(f"  ! {result.note}")
    click.echo(f"{'slot':4}  {'report':<28} {'n':>6} {'meas':>8} {'med':>7} {'granted':>8} "
               f"{'cfg':>7} {'drop%':>6}  status")
    for s in result.streams:
        status = "PASS" if s.ok else "FAIL"
        note = f"  {s.note}" if s.note else ""
        click.echo(
            f"{s.slot:4}  {s.report:<28} {s.n:>6} {s.measured_hz:>8.1f} {s.median_hz:>7.1f} "
            f"{_fmt_hz(s.granted_hz):>8} {s.expected_hz:>7.0f} {100 * s.drop_rate:>6.2f}  "
            f"{status}{note}"
        )
    for slot in sorted(set(result.link) | set(result.window_gaps)):
        win = (f"seq_gaps in window={result.window_gaps[slot]} ({result.window_span_s:.1f} s)  "
               if slot in result.window_gaps else "")
        grew = " ".join(f"{k}={v}" for k, v in (result.link.get(slot) or {}).items())
        click.echo(f"{slot:4}  lost before the ring: {win}since apply: {grew}")
    click.echo(f"case: {'PASS' if result.passed else 'FAIL'}  "
               "(meas = count/time, med = 1/median interval, granted = BNO085's answer)")
    return result.passed


def _fmt_hz(hz: float | None) -> str:
    return "-" if hz is None else f"{hz:.1f}"


def _parse_rates(text: str) -> list[float]:
    try:
        return [float(x) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise click.BadParameter(f"not a comma-separated list of numbers: {text!r}",
                                 param_hint="--sweep-rates") from exc


def _run_case_or_note(
    port: Any, resolved: dict[str, Any], duration: float, label: str, emg: bool
) -> Any:
    """run_case, with a config the validator refuses reported as that case's note."""
    from jtsmm.doctor.imu_rates import CaseResult, run_case

    try:
        return run_case(port=port, resolved=resolved, duration_s=duration, label=label, emg=emg)
    except ValueError as exc:  # validate_strict: e.g. the bus budget
        return CaseResult(label=label, duration_s=duration,
                          note=f"config rejected: {str(exc).splitlines()[-1].strip(' -')}")


def _run_sweep(
    port: Any, base: Path | None, preset: str | None, slot: str, duration: float,
    report: str | None, rates_text: str | None, min_fusion_hz: float | None,
    report_path: Path | None, emg: bool,
) -> None:
    from jtsmm.doctor.imu_rates import (
        default_min_fusion_hz, summarize_sweep, sweep_configs, sweep_rates_for, sweep_report_for,
    )

    resolved = _build_config(base, preset, slot)
    try:
        report = report or sweep_report_for(resolved, slot)
        rates = sweep_rates_for(report, _parse_rates(rates_text) if rates_text else None)
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    click.echo(f"sweep: {report} at {', '.join(f'{r:g}' for r in rates)} Hz "
               f"({duration:g} s each, EMG {'on' if emg else 'off'}). Per-case PASS/FAIL compares with the request; "
               "read `meas` and the summary.")
    results = []
    for label, cfg in sweep_configs(resolved, report, rates, slot):
        result = _run_case_or_note(port, cfg, duration, label, emg)
        _print_case(result)
        results.append(result)

    if min_fusion_hz is None:
        min_fusion_hz = default_min_fusion_hz(resolved, slot)
    rows = summarize_sweep(results, report, rates, slot, min_fusion_hz)
    lines = _sweep_table(rows, report, slot, min_fusion_hz)
    click.echo("")
    for line in lines:
        click.echo(line)
    if report_path is not None:
        _write_report(Path(report_path), results, extra=["## Sweep summary", "", "```", *lines,
                                                         "```", ""])
        click.echo(f"\nreport written: {report_path}")
    raise SystemExit(0 if all(r.complete for r in rows) else 1)


def _sweep_table(
    rows: list[Any], report: str, slot: str, min_fusion_hz: float | None
) -> list[str]:
    slots = ("mta", "cra") if slot == "both" else (slot,)
    pair = "/".join(slots)
    out = [f"=== sweep summary: {report}, Hz ({pair}) ===",
           f"{'request':>8}  {'fusion':>11}  {report[:13]:>11}  {'granted':>11}  "
           f"{'med':>11}  {'lost%':>6}  {'drop%':>6}"]
    for r in rows:
        def pair_of(d: dict[str, Any], fmt: str = "{:.0f}") -> str:
            return "/".join("-" if d.get(s) is None else fmt.format(d[s]) for s in slots)
        tag = "  <- best" if r.best else ("  (incomplete)" if not r.complete else "")
        out.append(f"{r.requested_hz:>8.0f}  {pair_of(r.fusion_hz):>11}  "
                   f"{pair_of(r.swept_hz):>11}  {pair_of(r.swept_granted_hz):>11}  "
                   f"{pair_of(r.swept_median_hz):>11}  {100 * r.loss_frac:>6.2f}  "
                   f"{100 * r.max_drop_rate:>6.2f}{tag}")
    out.append("granted = the BNO085's own answer; med = 1/median interval (equals the "
               "count rate only for evenly spaced reports); lost = SH-2 sequence gaps "
               "(BNO085 -> MCU) inside the window; drop = MCU -> USB")
    rule = "complete, no losses or drops"
    if min_fusion_hz is not None:
        rule += f", fusion >= {min_fusion_hz:.0f} Hz on both slots (--min-fusion-hz)"
    best = next((r for r in rows if r.best), None)
    if best is None:
        out.append(f"no request meets: {rule}")
    else:
        out.append(f"best: request {report} @ {best.requested_hz:.0f} Hz "
                   f"(mcu.imu_*.reports.{report}.rate_hz) = the highest {report} rate "
                   f"on the slower slot with {rule}")
    return out


def _write_report(path: Path, results: list[Any], extra: list[str] | None = None) -> None:
    lines = [f"# IMU sample-rate validation — {_dt.date.today().isoformat()}", ""]
    for r in results:
        lines.append(f"## {r.label} ({r.duration_s:.0f} s) — {'PASS' if r.passed else 'FAIL'}")
        lines.append("")
        lines.append("| slot | report | n | measured Hz | median-interval Hz | granted Hz "
                     "| configured Hz | drop % | status |")
        lines.append("|------|--------|---|-------------|-----|-----|---------------|--------|--------|")
        for s in r.streams:
            lines.append(
                f"| {s.slot} | {s.report} | {s.n} | {s.measured_hz:.1f} | {s.median_hz:.1f} | "
                f"{_fmt_hz(s.granted_hz)} | {s.expected_hz:.0f} | {100 * s.drop_rate:.2f} | "
                f"{'PASS' if s.ok else 'FAIL'}{(' — ' + s.note) if s.note else ''} |"
            )
        for slot in sorted(set(r.link) | set(r.window_gaps)):
            win = (f"seq_gaps in window={r.window_gaps[slot]} ({r.window_span_s:.1f} s); "
                   if slot in r.window_gaps else "")
            lines.append(f"\n{slot} lost before the ring: {win}since apply: "
                         + ", ".join(f"{k}={v}" for k, v in (r.link.get(slot) or {}).items()))
        if getattr(r, "status_text", ""):
            lines += ["", "<details><summary>imu_status after the case</summary>", "",
                      "```", r.status_text.strip(), "```", "</details>"]
        lines.append("")
    lines.extend(extra or [])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


@doctor_cmd.command("imu-rates")
@click.option("--preset", default=None, help="IMU preset to apply to both slots")
@click.option("--all-presets", is_flag=True, help="Run the fused/kinematics (accel, lin)/datasheet/raw matrix")
@click.option("--slot", type=click.Choice(["mta", "cra", "both"]), default="both")
@click.option("--duration", type=float, default=10.0, help="Capture seconds per case")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--report", "report_path", type=click.Path(path_type=Path), default=None,
              help="Write a markdown matrix report to this path")
@click.option("--emg/--no-emg", default=True,
              help="Stream EMG during each case, as a recording does (default); "
                   "--no-emg shows whether EMG load costs IMU reports")
@click.option("--sweep", is_flag=True,
              help="Request one report at several rates (rest of the config or --preset kept) "
                   "and show what each slot delivers; marks the best request")
@click.option("--sweep-report", default=None,
              help="Report to sweep (default: the one enabled non-fusion report, e.g. accelerometer)")
@click.option("--sweep-rates", default=None,
              help="Comma-separated request rates in Hz (default: 100,125,200,250,333,400,500, "
                   "up to the report's maximum)")
@click.option("--min-fusion-hz", type=float, default=None,
              help="Sweep: the best request must leave the fusion report at least this rate "
                   "(default: 85 %% of the configured fusion rate)")
@click.pass_context
def imu_rates_cmd(
    ctx: click.Context,
    preset: str | None,
    all_presets: bool,
    slot: str,
    duration: float,
    config_path: Path | None,
    report_path: Path | None,
    emg: bool,
    sweep: bool,
    sweep_report: str | None,
    sweep_rates: str | None,
    min_fusion_hz: float | None,
) -> None:
    """Measure achieved IMU stream rates on the connected HAT (J_USB1).

    PASS means the delivered rate matches the request. The BNO085 picks its own
    intervals around the fusion report, so the request is not always what you
    get; --sweep shows which request gives the most.
    """
    from jtsmm.doctor.imu_rates import run_case

    port = ctx.obj.get("port")
    base = config_path or ctx.obj.get("config_path")

    if not sweep and (sweep_report or sweep_rates or min_fusion_hz is not None):
        raise click.UsageError("--sweep-report / --sweep-rates / --min-fusion-hz need --sweep")
    if sweep:
        if all_presets:
            raise click.UsageError("--sweep and --all-presets do not combine; use --preset")
        _run_sweep(port, base, preset, slot, duration, sweep_report, sweep_rates,
                   min_fusion_hz, report_path, emg)
        return

    cases: list[tuple[str, str | None]] = []
    if all_presets:
        cases = [(p, p) for p in _MATRIX_PRESETS]
    elif preset is not None:
        cases = [(preset, preset)]
    else:
        cases = [(str(base) if base else "config", None)]

    results = []
    all_ok = True
    for label, preset_name in cases:
        resolved = _build_config(base, preset_name, slot)
        result = run_case(port=port, resolved=resolved, duration_s=duration, label=label, emg=emg)
        all_ok = _print_case(result) and all_ok
        results.append(result)

    if report_path is not None:
        _write_report(Path(report_path), results)
        click.echo(f"\nreport written: {report_path}")

    raise SystemExit(0 if all_ok else 1)
