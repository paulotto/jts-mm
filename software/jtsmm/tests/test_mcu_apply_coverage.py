# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""MCU apply payload and strict re-validation tests."""

from __future__ import annotations

import copy

import pytest

from jtsmm.config import build_mcu_apply_payload, validate_strict
from jtsmm.config.mcu_apply import apply_mcu
from tests.helpers import load_operator_template


def test_build_mcu_apply_includes_platform_irq():
    doc = load_operator_template()
    payload = build_mcu_apply_payload(doc)
    assert payload["platform"]["irq"]["emg"] == 0


def test_build_mcu_apply_emg_stream_channels():
    doc = load_operator_template()
    payload = build_mcu_apply_payload(doc)
    assert payload["emg"]["stream_channels"] == 6


def test_build_mcu_apply_imu_reports_present():
    doc = load_operator_template()
    mta = build_mcu_apply_payload(doc)["imu_mta"]
    assert mta["reports"]["game_rotation_vector"]["enabled"] is True


def test_apply_mcu_strict_rejects_bad_doc():
    doc = copy.deepcopy(load_operator_template())
    del doc["mcu"]["emg"]["sample_rate_sps"]

    class FakeSer:
        def reset_input_buffer(self): ...
        def write(self, *_): ...
        def flush(self): ...

    with pytest.raises(ValueError):
        apply_mcu(FakeSer(), doc, strict=True)


@pytest.mark.parametrize(
    "path",
    [
        "mcu.platform.buffers.event_ring",
        "mcu.emg.leadoff.ac_tone",
        "mcu.trigger.start_on",
        "host.emg_check.impedance.enabled",
        "session.default_duration_s",
    ],
)
def test_required_paths_present_in_template(path):
    doc = load_operator_template()
    issues = [i for i in validate_strict(doc) if i.startswith("error:")]
    assert issues == [], issues
    parts = path.split(".")
    cur = doc
    for p in parts:
        assert p in cur
        cur = cur[p]
