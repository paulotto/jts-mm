# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Trigger profiles and IMX296 exposure constraint."""

from __future__ import annotations

import copy

import pytest

from jtsmm.config import resolve_profiles
from jtsmm.config.catalog import IMX296_EXPOSURE_MARGIN_US


def _base():
    from tests.helpers import load_operator_template

    return load_operator_template()


def test_imx296_dual_csi_defaults():
    """Operator template resolved: 40 fps / 12 ms exposure reaches the MCU.

    Tracks software/jtsmm/etc/jtsmm.yaml.
    """
    trig = resolve_profiles(_base())["mcu"]["trigger"]
    assert trig["rate_hz"] == 40
    assert trig["outputs"]["csi1"] is True
    assert trig["pulse"]["width_us"] == 12000


def test_imx296_dual_csi_profile_alone_matches_firmware():
    """The bare profile resolves to the firmware profile/default (trigger_config.c).

    The profile file used to carry a 60 Hz / 4 ms stub, so a config naming only
    the profile disagreed with what the firmware resolves for the same name.
    """
    doc = copy.deepcopy(_base())
    doc["mcu"]["trigger"] = {"profile": "imx296_dual_csi"}
    trig = resolve_profiles(doc)["mcu"]["trigger"]
    assert trig["rate_hz"] == 30
    assert trig["pulse"]["width_us"] == 12000
    assert trig["pulse"]["polarity"] == "imx296_fast"


def test_track_a_led_profile_matches_firmware():
    """track_a_led: 30 Hz, 200 us (firmware trigger_config.c), not the CSI base values."""
    doc = copy.deepcopy(_base())
    doc["mcu"]["trigger"] = {"profile": "track_a_led"}
    trig = resolve_profiles(doc)["mcu"]["trigger"]
    assert trig["rate_hz"] == 30
    assert trig["pulse"]["width_us"] == 200


def test_high_fps_profile():
    doc = copy.deepcopy(_base())
    doc["mcu"]["trigger"] = {"profile": "imx296_high_fps"}
    trig = resolve_profiles(doc)["mcu"]["trigger"]
    assert trig["rate_hz"] == 100
    assert trig["pulse"]["width_us"] == 2000


def test_bench_scope_outputs():
    doc = copy.deepcopy(_base())
    doc["mcu"]["trigger"] = {"profile": "bench_scope"}
    trig = resolve_profiles(doc)["mcu"]["trigger"]
    assert trig["outputs"]["scope"] is True
    assert trig["outputs"]["csi1"] is False


def test_imx296_exposure_margin_constant():
    assert IMX296_EXPOSURE_MARGIN_US == pytest.approx(14.26)


def test_custom_trigger_rate(tmp_path):
    doc = copy.deepcopy(_base())
    doc["mcu"]["trigger"] = {
        "profile": "custom",
        "custom": {"rate_hz": 30, "pulse": {"width_us": 3000}},
    }
    trig = resolve_profiles(doc)["mcu"]["trigger"]
    assert trig["rate_hz"] == 30
