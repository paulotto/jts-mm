# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""YAML → MCU apply JSON → firmware config_json_parse round-trip."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from jtsmm.config import build_mcu_apply_payload
from jtsmm.config.loader import repo_root
from tests.helpers import load_operator_template

GOLDEN_JSON = repo_root() / "schema" / "v1" / "golden" / "mcu_apply_template.json"
FW_TEST_BIN = (
    repo_root() / "firmware" / "jtsmm" / "tests" / "build" / "test_config_json"
)


def _canonical_json(doc: dict) -> str:
    return json.dumps(doc, separators=(",", ":"), sort_keys=True)


def test_yaml_build_mcu_apply_matches_golden():
    payload = build_mcu_apply_payload(load_operator_template())
    with GOLDEN_JSON.open("r", encoding="utf-8") as fh:
        golden = json.load(fh)
    assert payload == golden


def test_yaml_to_json_roundtrip_stable():
    payload = build_mcu_apply_payload(load_operator_template())
    encoded = _canonical_json(payload)
    decoded = json.loads(encoded)
    assert decoded == payload


@pytest.mark.parametrize(
    "path,value",
    [
        ("platform.spi0.hz", 4_000_000),
        ("platform.buffers.emg_ring", 98304),
        ("emg.sample_rate_sps", 4000),
        ("emg.default_gain", 6),
        ("imu_mta.fusion.rate_hz", 390),
        ("imu_mta.reports.game_rotation_vector.rate_hz", 390),
        # 40 fps cameras: 25 ms period, 12 ms IMX296 exposure (low phase).
        ("trigger.rate_hz", 40),
        ("trigger.pulse.width_us", 12000),
    ],
)
def test_golden_mcu_json_key_values(path: str, value):
    with GOLDEN_JSON.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    cur = payload
    for part in path.split("."):
        cur = cur[part]
    assert cur == value


def test_firmware_config_json_host_binary():
    if not FW_TEST_BIN.is_file():
        pytest.skip("firmware host test binary not built (run cmake in firmware/jtsmm/tests)")
    proc = subprocess.run(
        [str(FW_TEST_BIN), str(GOLDEN_JSON)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_sync_mcu_apply_golden_up_to_date():
    """Guard: regenerate with scripts/sync_config_template.py --sync-mcu-golden."""
    payload = build_mcu_apply_payload(load_operator_template())
    with GOLDEN_JSON.open("r", encoding="utf-8") as fh:
        golden = json.load(fh)
    assert payload == golden
