# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Mock record session smoke test."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest
import yaml

from jtsmm.capture.session import run_session
from jtsmm.integrity.checker import check_session


@pytest.fixture
def mock_config_path(tmp_config_path: Path) -> Path:
    """Plain timed mock recording: no SPACE-driven plan, no optical worker.

    The shipped template walks an experiment plan with optical tracking, which
    (correctly) needs an interactive TTY and cv2; this smoke test is about the
    recording path itself.
    """
    doc = yaml.safe_load(tmp_config_path.read_text(encoding="utf-8"))
    session = doc.setdefault("session", {})
    session["use_experiment_plan"] = False
    session["optical_track"] = False
    tmp_config_path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return tmp_config_path


def test_mock_session_smoke(mock_config_path: Path):
    out = run_session(
        config_path=mock_config_path,
        duration_s=0.15,
        mock_only=True,
        skip_mcu_apply=True,
    )
    assert out.is_file()
    with h5py.File(out, "r") as f:
        assert "meta" in f
        assert "cameras" in f
        assert "recording_date" in f["meta"].attrs
    report = check_session(out)
    assert report.ok


def test_mock_session_gain_and_meta(mock_config_path: Path):
    """--gain and --meta/--subject/--protocol (via run_session's default_gain/meta) land in the HDF5."""
    out = run_session(
        config_path=mock_config_path,
        duration_s=0.1,
        mock_only=True,
        skip_mcu_apply=True,
        default_gain=12,
        meta={"subject_id": "P07", "protocol": "motion_battery_v1", "operator": "jd"},
    )
    with h5py.File(out, "r") as f:
        mcu_cfg = json.loads(f["meta"].attrs["mcu_config_json"])
        assert int(mcu_cfg["emg"]["default_gain"]) == 12
        assert f["meta"].attrs["subject_id"] == "P07"
        assert f["meta"].attrs["protocol"] == "motion_battery_v1"
        assert f["meta"].attrs["operator"] == "jd"
