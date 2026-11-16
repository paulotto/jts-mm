# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm view --gain` overrides `LiveSource`'s ADS1299 PGA gain."""

from __future__ import annotations

from pathlib import Path

from jtsmm.viewer.sources import LiveSource


def test_live_source_default_gain_override(tmp_config_path: Path):
    src = LiveSource(config_path=tmp_config_path, default_gain=12)
    assert src._gain == 12
    assert src._emg["default_gain"] == 12


def test_live_source_default_gain_unset_keeps_config(tmp_config_path: Path):
    src = LiveSource(config_path=tmp_config_path)
    assert src._gain == 6  # etc/jtsmm.yaml template default
