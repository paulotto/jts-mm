# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Parameter coverage audit — schema, YAML template, registry, MCU apply, firmware."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import yaml

from . import registry as reg
from .loader import default_config_path, load_json_schema, load_yaml, schema_dir, template_path
from .mcu_apply import build_mcu_apply_payload

__all__ = [
    "FIRMWARE_JSON_KEYS",
    "AuditEntry",
    "AuditReport",
    "audit_to_json",
    "default_config_path",
    "firmware_parses",
    "format_audit_table",
    "load_coverage_manifest",
    "run_audit",
    "schema_leaf_paths",
    "template_path",
]

_REGISTRY_ACCESSORS = {
    "host.integrity.thresholds": reg.integrity_thresholds,
    "host.integrity.errors_as_warnings": reg.integrity_errors_as_warnings,
    "host.record.abort_on_error": reg.record_abort_on_error,
    "host.capture.settle_ms": reg.capture_settle_ms,
    "host.capture.stream_restart": reg.capture_stream_restart,
    "host.capture.native_prefer": reg.capture_native_prefer,
    "host.capture.trigger_after_cameras": reg.capture_trigger_after_cameras,
    "host.capture.pps_samples": reg.capture_pps_samples,
    "host.capture.camera_pps_anchor": reg.capture_camera_pps_anchor,
    "host.capture.camera_delta_tracking": reg.capture_camera_delta_tracking,
    "host.paths.data_dir": reg.paths_data_dir,
    "host.record.emg_check.enabled": reg.record_emg_check_enabled,
    "host.record.emg_check.block_on_contact_failure": reg.record_emg_check_block_on_contact,
    "host.emg_filter.mains_mode": reg.emg_filter_mains_mode,
    "host.record.preflight.enabled": reg.record_preflight_enabled,
    "host.record.integrity.enabled": reg.record_integrity_enabled,
    "host.record.mcu_apply.enabled": reg.record_mcu_apply_enabled,
    "host.emg_check.line_hz": reg.emg_check_line_hz,
    "host.emg_check.settle_ms": reg.emg_check_settle_ms,
    "host.emg_check.leadoff.duration_s": reg.emg_check_leadoff_duration_s,
    "host.emg_check.noise.duration_s": reg.emg_check_noise_duration_s,
    "host.emg_check.impedance.enabled": reg.emg_check_impedance_enabled,
    "host.emg_check.impedance.duration_s": reg.emg_check_impedance_duration_s,
    "host.emg_check.drl_isolation.duration_s": reg.emg_check_drl_isolation_duration_s,
    "host.viewer.enabled": reg.viewer_enabled,
    "host.viewer.bind_host": reg.viewer_bind_host,
    "host.viewer.port": reg.viewer_port,
    "session.mode": reg.session_mode,
    "session.duration_s": reg.session_duration_s,
    "session.default_duration_s": reg.session_default_duration_s,
    "session.subject_id": reg.session_subject_id,
    "session.protocol": reg.session_protocol,
    "session.meta": reg.session_meta,
    "session.ct_calibration": reg.session_ct_calibration,
    "session.ct_calibration_n_poses": reg.session_ct_calibration_n_poses,
    "session.ct_tip_calibration": reg.session_ct_tip_calibration,
    "session.persist_ct_calibration": reg.session_persist_ct_calibration,
    "session.ct_calibration_recordings_dir": reg.session_ct_calibration_recordings_dir,
    "session.persist_imu_calibration": reg.session_persist_imu_calibration,
    "session.imu_mounting_calibration": reg.session_imu_mounting_calibration,
    "session.imu_calibration_recordings_dir": reg.session_imu_calibration_recordings_dir,
    "session.optical_track": reg.session_optical_track,
    "session.optical_track_hz": reg.session_optical_track_hz,
    "session.optical_config": reg.session_optical_config,
    "session.mvc": reg.session_mvc,
    "session.use_experiment_plan": reg.session_use_experiment_plan,
    "session.experiment": reg.session_experiment,
    "session.experiment_plan": reg.session_experiment_plan,
    "session.write_jts_config": reg.session_write_jts_config,
}


@dataclass
class AuditEntry:
    path: str
    status: str
    layers: list[str] = field(default_factory=list)
    consumer: str = ""


# Statuses that do not fail `config audit --strict`. DEFAULT = an optional key
# absent from the file (its code default applies).
PASSING_STATUSES = frozenset({"OK", "DEFAULT"})


@dataclass
class AuditReport:
    entries: list[AuditEntry] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"OK": 0}
        for entry in self.entries:
            out[entry.status] = out.get(entry.status, 0) + 1
        return out

    def ok(self) -> bool:
        return all(e.status in PASSING_STATUSES for e in self.entries)


# Every key firmware/jtsmm/src/runtime/config_json.c reads from `config apply`
# JSON, as wire paths (both IMU slots share parse_imu_slot -> "imu"; report
# entries are "imu.reports.*"). Static on purpose: the audit must work without
# the firmware tree. tests/test_fix_H_audit.py re-extracts it from the C source.
FIRMWARE_JSON_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "platform.profile",
        "platform.system.cpu_mhz",
        "platform.system.xosc_hz",
        "platform.spi0.hz",
        "platform.spi0.cfg_hz",
        "platform.i2c.hz",
        "platform.pio.trigger.clock_hz",
        "platform.pio.trigger_clock_hz",
        "platform.buffers.emg_ring",
        "platform.buffers.imu_ring",
        "platform.buffers.event_ring",
        "platform.irq.emg",
        "platform.irq.imu",
        "platform.irq.trigger",
        "emg.enabled",
        "emg.profile",
        "emg.sample_rate_sps",
        "emg.default_gain",
        "emg.stream_channels",
        "emg.pd_channels",
        "emg.bias.enabled",
        "emg.bias.channels",
        "emg.bias.ch7_readback",
        "emg.leadoff.enabled",
        "emg.leadoff.during_recording",
        "emg.leadoff.ac_tone",
        "emg.leadoff.probe_drl",
        "emg.leadoff.current_na",
        "emg.test_signal.enabled",
        "emg.test_signal.amplitude_mv",
        "imu.enabled",
        "imu.profile",
        "imu.fusion.primary",
        "imu.fusion.rate_hz",
        "imu.sync.mode",
        "imu.sync.decimation_factor",
        "imu.dynamic_calibration.accel",
        "imu.dynamic_calibration.gyro",
        "imu.dynamic_calibration.mag",
        "imu.reports.*.enabled",
        "imu.reports.*.rate_hz",
        "trigger.enabled",
        "trigger.profile",
        "trigger.rate_hz",
        "trigger.pulse.width_us",
        "trigger.pulse.polarity",
        "trigger.outputs.csi1",
        "trigger.outputs.csi2",
        "trigger.outputs.scope",
        "trigger.outputs.sync_led",
    }
)

# YAML key -> wire key where mcu_apply.py translates (1-based list -> bitmask).
_WIRE_RENAMES = {"emg.active_channels": "emg.pd_channels"}


def firmware_key(path: str) -> str | None:
    """``mcu.imu_cra.fusion.rate_hz`` -> ``imu.fusion.rate_hz`` (None if not mcu.*)."""
    if not path.startswith("mcu."):
        return None
    parts = path[len("mcu.") :].split(".")
    if parts[0] in ("imu_mta", "imu_cra"):
        parts[0] = "imu"
        if len(parts) > 3 and parts[1] == "reports":
            parts[2] = "*"  # every report entry is parsed the same way
    key = ".".join(parts)
    return _WIRE_RENAMES.get(key, key)


def firmware_parses(path: str) -> bool:
    key = firmware_key(path)
    if key is None:
        return False
    return key in FIRMWARE_JSON_KEYS or any(k.startswith(key + ".") for k in FIRMWARE_JSON_KEYS)


def _resolve_ref(root: dict[str, Any], node: Any) -> Any:
    while isinstance(node, dict) and "$ref" in node:
        target: Any = root
        for part in node["$ref"].lstrip("#/").split("/"):
            target = target[part]
        node = {**target, **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def schema_leaf_paths() -> set[str]:
    """Dotted paths of every leaf key in jtsmm.schema.json (arrays count as leaves)."""
    root = load_json_schema()
    out: set[str] = set()

    def walk(node: Any, prefix: str) -> None:
        node = _resolve_ref(root, node)
        props = node.get("properties") if isinstance(node, dict) else None
        if props:
            for key, sub in props.items():
                walk(sub, f"{prefix}.{key}" if prefix else key)
        elif prefix:
            out.add(prefix)

    walk(root, "")
    return out


def load_coverage_manifest() -> list[dict[str, Any]]:
    path = schema_dir() / "config_coverage.yaml"
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return list(data.get("parameters") or [])


def _has_path(doc: Any, dot_path: str) -> bool:
    cur: Any = doc
    for part in dot_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def run_audit(doc: dict[str, Any] | None = None) -> AuditReport:
    """Audit ``doc`` (default: the config `jtsmm` would load) against the manifest.

    Works on an unvalidated document so an incomplete file yields MISSING_YAML
    rows instead of an exception.
    """
    doc = doc if doc is not None else load_yaml(default_config_path())
    try:
        payload = build_mcu_apply_payload(doc)
    except Exception:  # noqa: BLE001 - e.g. a malformed channel list; audited below
        payload = {}
    report = AuditReport()

    for entry in load_coverage_manifest():
        path = str(entry["path"])
        layers = list(entry.get("layer") or [])
        consumer = str(entry.get("consumer", ""))
        required = bool(entry.get("required"))
        status = "OK"

        if "yaml" in layers and not _has_path(doc, path):
            status = "MISSING_YAML" if required else "DEFAULT"
        elif path in _REGISTRY_ACCESSORS:
            try:
                _REGISTRY_ACCESSORS[path](doc)
            except Exception:  # noqa: BLE001 - the accessor cannot read this value
                status = "MISSING_WIRE"
        elif path.startswith(("host.", "session.")):
            if (
                "host_code" in layers
                and consumer != "schema_only"
                and not path.startswith("host.cameras")
                and path != "host.schema.relaxed_unknown_keys"
            ):
                status = "MISSING_WIRE"
        elif path.startswith("mcu.") and "host_apply" in layers and _has_path(doc, path):
            wire = firmware_key(path)
            if wire is not None and wire == path[len("mcu.") :] and not _has_path(payload, wire):
                status = "MISSING_WIRE"

        if status in PASSING_STATUSES and "firmware" in layers and not firmware_parses(path):
            status = "MISSING_FW"

        report.entries.append(
            AuditEntry(path=path, status=status, layers=layers, consumer=consumer)
        )
    return report


def format_audit_table(report: AuditReport) -> str:
    counts = report.counts
    total = len(report.entries)
    lines = [
        f"config audit: {total} parameters",
        f"  OK:              {counts.get('OK', 0)}",
    ]
    for key in sorted(k for k in counts if k != "OK"):
        lines.append(f"  {key}:    {counts[key]}")
    problems = [e for e in report.entries if e.status not in PASSING_STATUSES]
    if problems:
        lines.append("Issues:")
        for entry in problems[:20]:
            lines.append(f"  {entry.status}: {entry.path}")
        if len(problems) > 20:
            lines.append(f"  ... and {len(problems) - 20} more")
    return "\n".join(lines)


def audit_to_json(report: AuditReport) -> str:
    return json.dumps(
        {
            "total": len(report.entries),
            "counts": report.counts,
            "ok": report.ok(),
            "entries": [
                {
                    "path": e.path,
                    "status": e.status,
                    "layers": e.layers,
                    "consumer": e.consumer,
                }
                for e in report.entries
            ],
        },
        indent=2,
    )
