# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Registry accessor tests."""

from __future__ import annotations

import copy

import pytest

from jtsmm.config import registry as reg
from tests.helpers import load_operator_template


@pytest.fixture
def cfg():
    return load_operator_template()


def test_capture_settle_ms_from_yaml(cfg):
    assert reg.capture_settle_ms(cfg) == 500


def test_capture_settle_ms_env_override(cfg, monkeypatch):
    monkeypatch.setenv("JTSMM_SETTLE_MS", "123")
    assert reg.capture_settle_ms(cfg) == 123


def test_capture_stream_restart_env(cfg, monkeypatch):
    monkeypatch.setenv("JTSMM_STREAM_RESTART", "0")
    assert reg.capture_stream_restart(cfg) is False


def test_paths_data_dir_yaml(cfg):
    assert "jtsmm_data" in reg.paths_data_dir(cfg)


def test_paths_data_dir_env(cfg, monkeypatch):
    monkeypatch.setenv("JTSMM_DATA_DIR", "/tmp/custom")
    assert reg.paths_data_dir(cfg) == "/tmp/custom"


def test_viewer_settings(cfg):
    assert reg.viewer_enabled(cfg) is True
    assert reg.viewer_port(cfg) == 8770


def test_record_flags_default_on(cfg):
    assert reg.record_emg_check_enabled(cfg) is True
    assert reg.record_preflight_enabled(cfg) is True


def test_emg_check_durations(cfg):
    assert reg.emg_check_leadoff_duration_s(cfg) == 5.0
    assert reg.emg_check_noise_duration_s(cfg) == 5.0


def test_session_meta_from_config_empty(cfg):
    assert reg.session_meta_from_config(cfg) == {}


def test_session_meta_from_config_subject_protocol(cfg):
    doc = copy.deepcopy(cfg)
    doc["session"]["subject_id"] = "anon-001"
    doc["session"]["protocol"] = "motion_v1"
    doc["session"]["meta"] = {"operator": "jd"}
    meta = reg.session_meta_from_config(doc)
    assert meta["subject_id"] == "anon-001"
    assert meta["protocol"] == "motion_v1"
    assert meta["operator"] == "jd"


def test_effective_duration_cli_override(cfg):
    assert reg.effective_duration_s(cfg, cli_duration=3.0) == 3.0


def test_effective_duration_default(cfg):
    doc = copy.deepcopy(cfg)
    doc["session"]["duration_s"] = None
    assert reg.effective_duration_s(doc) == 10.0


def test_capture_duration_plan_ignores_default(cfg):
    doc = copy.deepcopy(cfg)
    doc["session"]["duration_s"] = None
    doc["session"]["default_duration_s"] = 10.0
    assert reg.capture_duration_s(doc, plan_active=True) is None
    assert reg.capture_duration_s(doc, plan_active=False) == 10.0


def test_capture_duration_plan_honors_explicit(cfg):
    doc = copy.deepcopy(cfg)
    doc["session"]["duration_s"] = None
    assert reg.capture_duration_s(doc, cli_duration=1800.0, plan_active=True) == 1800.0
    doc["session"]["duration_s"] = 600.0
    assert reg.capture_duration_s(doc, plan_active=True) == 600.0


def test_session_mvc_default_off(cfg):
    assert reg.session_mvc(cfg) is False


def test_session_persist_ct_calibration_defaults(cfg):
    assert reg.session_persist_ct_calibration(cfg) is False
    path = reg.session_ct_calibration_recordings_dir(cfg)
    assert path is not None
    assert "5_2_1_pivot_calibration_repeatability" in path
    assert path.rstrip("/").endswith("recordings") or path.endswith("recordings")


def test_session_persist_imu_calibration_defaults(cfg):
    assert reg.session_persist_imu_calibration(cfg) is True        # archived by default
    path = reg.session_imu_calibration_recordings_dir(cfg)
    assert path is not None
    assert "5_2_2_imu_optical_handeye" in path
    assert path.rstrip("/").endswith("recordings") or path.endswith("recordings")


def test_session_use_experiment_plan_from_yaml(cfg):
    assert reg.session_use_experiment_plan(cfg) is True
    exp = reg.session_experiment(cfg)
    assert isinstance(exp, dict)
    assert exp["phases"]
