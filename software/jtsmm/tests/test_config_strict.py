# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Strict load-time validation tests."""

from __future__ import annotations

import copy

import pytest
import yaml

from jtsmm.config import load, load_yaml, validate_strict
from jtsmm.config.strict import raise_on_errors
from tests.helpers import load_operator_template


def test_monolithic_template_strict_load(tmp_config_path):
    cfg = load(tmp_config_path)
    assert cfg["schema_version"] == 1
    assert cfg["host"]["capture"]["settle_ms"] == 500


def test_missing_required_key_fails(tmp_path):
    doc = copy.deepcopy(load_operator_template())
    del doc["host"]["capture"]["settle_ms"]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    issues = validate_strict(load_yaml(path), path=path)
    assert any("missing required key" in i and "settle_ms" in i for i in issues)


def test_unknown_key_fails(tmp_path):
    doc = copy.deepcopy(load_operator_template())
    doc["host"]["unknown_key"] = 1
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    issues = validate_strict(load_yaml(path), path=path)
    assert any("unknown key" in i for i in issues)


def test_relaxed_unknown_keys_allows_extra(tmp_path):
    doc = copy.deepcopy(load_operator_template())
    doc["host"]["schema"]["relaxed_unknown_keys"] = True
    doc["host"]["debug_only"] = True
    path = tmp_path / "relaxed.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    issues = [i for i in validate_strict(load_yaml(path), path=path) if i.startswith("error:")]
    assert not any("unknown key" in i for i in issues)


def test_invalid_yaml_root_type(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_yaml(path)


def test_load_strict_false_skips_raise(tmp_path, operator_config):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]["settle_ms"]
    path = tmp_path / "incomplete.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path, strict=True)
    # strict=False still returns doc (doctor debug path uses validate=False)
    loaded = load(path, strict=False, validate=False)
    assert "capture" in loaded["host"]


def test_reserved_session_meta_key_rejected(tmp_path):
    doc = copy.deepcopy(load_operator_template())
    doc["session"]["meta"] = {"config_hash": "bad"}
    path = tmp_path / "meta.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    issues = validate_strict(load_yaml(path), path=path)
    assert any("reserved" in i for i in issues)


def test_invalid_emg_sample_rate_schema_error(tmp_path):
    doc = copy.deepcopy(load_operator_template())
    doc["mcu"]["emg"]["sample_rate_sps"] = 333
    path = tmp_path / "rate.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_raise_on_errors_helper():
    with pytest.raises(ValueError):
        raise_on_errors(["error: boom"])


def test_validate_strict_warn_prefix_preserved(operator_config):
    doc = copy.deepcopy(operator_config)
    doc["mcu"]["emg"]["profile"] = "bench"
    doc["mcu"]["emg"]["test_signal"] = {"enabled": False}
    issues = validate_strict(doc)
    assert any(i.startswith("warn:") for i in issues)
