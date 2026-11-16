# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""MCU config apply JSON round-trip."""

from __future__ import annotations

import json

from jtsmm.config import (
    build_mcu_apply_payload,
    format_apply_command,
    start_mcu_streams,
    start_mcu_trigger,
)


def test_build_mcu_apply_payload_keys(resolved_config):
    payload = build_mcu_apply_payload(resolved_config)
    assert payload["schema_version"] == 1
    for key in ("platform", "emg", "imu_mta", "imu_cra", "trigger"):
        assert key in payload
    assert "host" not in payload


def test_apply_command_is_single_line(resolved_config):
    cmd = format_apply_command(build_mcu_apply_payload(resolved_config))
    assert cmd.startswith("config apply ")
    assert cmd.endswith("\n")
    json_part = cmd[len("config apply ") :].strip()
    parsed = json.loads(json_part)
    assert parsed["emg"]["sample_rate_sps"] == 4000


def test_apply_command_fits_firmware_line_limit(resolved_config):
    cmd = format_apply_command(build_mcu_apply_payload(resolved_config))
    # firmware cmd_parser line_buf (JTSMM_CMD_LINE_MAX) includes trailing NUL
    assert len(cmd.rstrip("\n")) < 4096
    assert len(cmd.rstrip("\n")) < 2048, "must fit legacy 2048 B CDC buffer on unflashed rigs"


def test_apply_json_roundtrip_stable(resolved_config):
    payload = build_mcu_apply_payload(resolved_config)
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    decoded = json.loads(encoded)
    assert decoded == json.loads(json.dumps(payload, sort_keys=True))


def test_start_mcu_streams_skips_ads_when_emg_false(monkeypatch):
    monkeypatch.setattr("jtsmm.config.mcu_apply.time.sleep", lambda *_a, **_k: None)
    writes: list[bytes] = []

    class _Ser:
        def write(self, data: bytes) -> None:
            writes.append(data)

        def flush(self) -> None:
            return None

    start_mcu_streams(_Ser(), emg=False)  # type: ignore[arg-type]
    blob = b"".join(writes)
    assert b"ads_start" not in blob
    assert b"imu_stream mta on" in blob
    assert b"imu_stream cra on" in blob
    writes.clear()
    start_mcu_streams(_Ser(), emg=True)  # type: ignore[arg-type]
    assert b"ads_start" in b"".join(writes)


def test_start_mcu_streams_can_defer_the_camera_trigger(monkeypatch):
    """trig_start must be separable: a trigger with no camera to consume it
    biases every later frame timestamp by a constant (sequence consume)."""
    monkeypatch.setattr("jtsmm.config.mcu_apply.time.sleep", lambda *_a, **_k: None)
    writes: list[bytes] = []

    class _Ser:
        def write(self, data: bytes) -> None:
            writes.append(data)

        def flush(self) -> None:
            return None

    start_mcu_streams(_Ser(), trigger=False)  # type: ignore[arg-type]
    blob = b"".join(writes)
    assert b"trig_start" not in blob
    assert b"ads_start" in blob and b"imu_stream mta on" in blob

    writes.clear()
    start_mcu_trigger(_Ser())  # type: ignore[arg-type]
    assert b"".join(writes) == b"trig_start\r\n"
