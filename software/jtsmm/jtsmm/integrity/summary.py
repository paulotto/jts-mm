# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Per-session integrity summary (preflight + EMG check + HDF5)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from jtsmm.integrity.checker import IntegrityReport, format_report

INTEGRITY_JSON_NAME = "integrity.json"
INTEGRITY_TXT_NAME = "integrity.txt"
SUMMARY_SCHEMA_VERSION = 1


def _json_safe(obj: Any) -> Any:
    """Convert numpy scalars / paths into plain Python JSON types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    return obj


def resolve_session_h5(path: Path | str) -> Path:
    """Accept a ``session.h5`` file or a session directory containing one."""
    path = Path(path)
    if path.is_dir():
        candidate = path / "session.h5"
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"no session.h5 in directory: {path}")
    if path.is_file():
        return path
    raise FileNotFoundError(f"session path not found: {path}")


def report_to_dict(report: IntegrityReport) -> dict[str, Any]:
    """JSON-safe snapshot of an :class:`IntegrityReport`."""
    return _json_safe(
        {
            "path": str(report.path),
            "ok": report.ok,
            "errors": list(report.issues),
            "warnings": list(report.warnings),
            "summary": report.summary,
        }
    )


@dataclass
class CheckSection:
    """One named integrity stage (preflight / emg / hdf5)."""

    name: str
    ran: bool = False
    ok: bool = True
    skipped: bool = False
    skip_reason: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    infos: list[str] = field(default_factory=list)
    detail: dict[str, Any] | None = None

    @property
    def status(self) -> str:
        if self.skipped or not self.ran:
            return "skipped"
        if not self.ok:
            return "fail"
        if self.warnings:
            return "ok_with_warnings"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "ran": self.ran,
            "ok": self.ok,
            "skipped": self.skipped,
            "status": self.status,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "infos": list(self.infos),
        }
        if self.skip_reason is not None:
            out["skip_reason"] = self.skip_reason
        if self.detail is not None:
            out["detail"] = self.detail
        return out


@dataclass
class SessionIntegritySummary:
    """Aggregated integrity results written next to ``session.h5``."""

    session_dir: Path
    session_h5: str = "session.h5"
    preflight: CheckSection = field(default_factory=lambda: CheckSection("preflight"))
    emg: CheckSection = field(default_factory=lambda: CheckSection("emg"))
    hdf5: CheckSection = field(default_factory=lambda: CheckSection("hdf5"))
    created_unix_utc: float | None = None

    @property
    def ok(self) -> bool:
        for section in (self.preflight, self.emg, self.hdf5):
            if section.ran and not section.ok:
                return False
        return True

    @property
    def overall(self) -> str:
        if not self.ok:
            return "fail"
        warnings = 0
        ran_any = False
        for section in (self.preflight, self.emg, self.hdf5):
            if section.ran and not section.skipped:
                ran_any = True
                warnings += len(section.warnings)
        if not ran_any:
            return "skipped"
        if warnings:
            return "ok_with_warnings"
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        created = self.created_unix_utc
        if created is None:
            created = datetime.now(timezone.utc).timestamp()
        return {
            "schema_version": SUMMARY_SCHEMA_VERSION,
            "created_unix_utc": created,
            "session_dir": str(self.session_dir),
            "session_h5": self.session_h5,
            "overall": self.overall,
            "ok": self.ok,
            "checks": {
                "preflight": self.preflight.to_dict(),
                "emg": self.emg.to_dict(),
                "hdf5": self.hdf5.to_dict(),
            },
        }


def set_preflight(
    summary: SessionIntegritySummary,
    *,
    errors: list[str] | None = None,
    warnings: list[str] | None = None,
    infos: list[str] | None = None,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> None:
    section = summary.preflight
    section.ran = True
    section.skipped = skipped
    section.skip_reason = skip_reason
    section.errors = list(errors or [])
    section.warnings = list(warnings or [])
    section.infos = list(infos or [])
    section.ok = not section.errors


def _emg_channel_failed(row: dict[str, Any]) -> bool:
    """True when a ChannelCheck row failed.

    Rows come from ``dataclasses.asdict(ChannelCheck)``, which carries the three
    ``*_ok`` flags but NOT the derived ``ok`` property. Reading ``row["ok"]``
    therefore always defaulted to True, which is why a failing EMG check used to
    report only the generic DRL note instead of the channels that actually
    failed.
    """
    return not all(
        bool(row.get(key, True)) for key in ("leadoff_ok", "noise_ok", "line_ok")
    )


def _emg_failure_reason(detail: dict[str, Any]) -> str | None:
    """One-line headline naming the failing channels (fallback when no notes)."""
    rows = detail.get("channels")
    if not isinstance(rows, list):
        return None
    bad = [
        str(r.get("name") or r.get("channel"))
        for r in rows
        if isinstance(r, dict) and _emg_channel_failed(r)
    ]
    drl = detail.get("drl")
    drl_bad = isinstance(drl, dict) and _emg_channel_failed(drl)
    sat = detail.get("drl_saturation")
    sat_bad = isinstance(sat, dict) and not sat.get("ok", True)
    if not bad and not drl_bad and not sat_bad:
        return None
    parts: list[str] = []
    if bad:
        head = ", ".join(bad[:3])
        more = f" +{len(bad) - 3} more" if len(bad) > 3 else ""
        parts.append(f"{len(bad)} channel(s): {head}{more}")
    if drl_bad:
        parts.append("DRL contact")
    if sat_bad:
        parts.append("a disturbed DRL/bias node")
    return "EMG check failed on " + "; ".join(parts)


def set_emg_check(
    summary: SessionIntegritySummary,
    *,
    result: dict[str, Any] | None = None,
    passed: bool | None = None,
    skipped: bool = False,
    skip_reason: str | None = None,
) -> None:
    section = summary.emg
    section.ran = True
    section.skipped = skipped
    section.skip_reason = skip_reason
    if skipped:
        section.ok = True
        section.detail = None
        return
    detail = dict(result or {})
    section.detail = detail
    if passed is None:
        passed = bool(detail.get("passed", True))
    section.ok = bool(passed)
    if not section.ok:
        note = _emg_failure_reason(detail) or detail.get("note") or "EMG check failed"
        section.errors = [str(note)]
        failed = []
        for ch in detail.get("channels") or []:
            if isinstance(ch, dict) and _emg_channel_failed(ch):
                name = ch.get("name") or ch.get("channel")
                notes = "; ".join(ch.get("notes") or []) or "FAIL"
                failed.append(f"CH{ch.get('channel', '?')} ({name}): {notes}")
        drl = detail.get("drl")
        if isinstance(drl, dict) and _emg_channel_failed(drl):
            notes = "; ".join(drl.get("notes") or []) or "FAIL"
            failed.append(f"DRL: {notes}")
        sat = detail.get("drl_saturation")
        if isinstance(sat, dict) and not sat.get("ok", True):
            notes = "; ".join(sat.get("notes") or []) or (
                f"bias node swinging {float(sat.get('ac_mv', 0)):.0f} mV RMS"
            )
            failed.append(f"bias amp (DRL): {notes}")
        if failed:
            section.errors = failed
    else:
        section.errors = []


def set_hdf5(
    summary: SessionIntegritySummary,
    report: IntegrityReport,
) -> None:
    section = summary.hdf5
    section.ran = True
    section.skipped = False
    section.ok = report.ok
    section.errors = list(report.issues)
    section.warnings = list(report.warnings)
    section.detail = report_to_dict(report)
    summary.session_h5 = Path(report.path).name


def format_session_summary(summary: SessionIntegritySummary, *, width: int = 88) -> str:
    """Human-readable combined integrity report (all stages)."""
    rule = "=" * width
    thin = "-" * width
    lines: list[str] = [
        rule,
        f" jtsmm session integrity — {summary.session_dir}",
        rule,
        f" overall     {summary.overall.upper()}",
        thin,
        f" {'check':<14}{'status':<18}  errors  warnings",
        thin,
    ]
    for section in (summary.preflight, summary.emg, summary.hdf5):
        status = section.status
        lines.append(
            f" {section.name:<14}{status:<18}  "
            f"{len(section.errors):>6}  {len(section.warnings):>8}"
        )
        if section.skipped and section.skip_reason:
            lines.append(f"   skip: {section.skip_reason}")
        for err in section.errors:
            lines.append(f"   error: {err}")
        for w in section.warnings:
            lines.append(f"   warning: {w}")
        for info in section.infos[:5]:
            lines.append(f"   info: {info}")
        if len(section.infos) > 5:
            lines.append(f"   info: … +{len(section.infos) - 5} more")

    lines.append(thin)
    if summary.hdf5.ran and not summary.hdf5.skipped and summary.hdf5.detail:
        # Re-render the HDF5 table from the stored report fields when possible.
        detail = summary.hdf5.detail
        fake = IntegrityReport(path=Path(detail.get("path", summary.session_h5)))
        fake.issues = list(detail.get("errors") or [])
        fake.warnings = list(detail.get("warnings") or [])
        fake.summary = dict(detail.get("summary") or {})
        lines.append(format_report(fake, width=width))
    else:
        lines.append(rule)
        lines.append(f" RESULT: {summary.overall.upper()}")
        lines.append(rule)
    return "\n".join(lines)


def write_session_integrity(
    session_directory: Path,
    summary: SessionIntegritySummary,
) -> dict[str, Path]:
    """Write ``integrity.json`` + ``integrity.txt`` into the session directory."""
    session_directory = Path(session_directory)
    session_directory.mkdir(parents=True, exist_ok=True)
    summary.session_dir = session_directory
    if summary.created_unix_utc is None:
        summary.created_unix_utc = datetime.now(timezone.utc).timestamp()

    json_path = session_directory / INTEGRITY_JSON_NAME
    txt_path = session_directory / INTEGRITY_TXT_NAME
    json_path.write_text(
        json.dumps(_json_safe(summary.to_dict()), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    txt_path.write_text(format_session_summary(summary) + "\n", encoding="utf-8")
    return {
        INTEGRITY_JSON_NAME: json_path,
        INTEGRITY_TXT_NAME: txt_path,
    }


def load_emg_from_session_h5(h5_path: Path) -> tuple[dict[str, Any] | None, bool | None]:
    """Pull ``/meta/emg_check_*`` from a recorded session, if present."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        meta = f.get("meta")
        if meta is None:
            return None, None
        raw = meta.attrs.get("emg_check_json")
        passed_raw = meta.attrs.get("emg_check_passed")
        if raw is None:
            return None, None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        try:
            detail = json.loads(str(raw))
        except json.JSONDecodeError:
            return None, None
        passed: bool | None = None
        if passed_raw is not None:
            if isinstance(passed_raw, bytes):
                passed_raw = passed_raw.decode("utf-8")
            passed = str(passed_raw).lower() in ("1", "true", "yes")
        elif isinstance(detail, dict) and "passed" in detail:
            passed = bool(detail["passed"])
        return detail if isinstance(detail, dict) else None, passed
