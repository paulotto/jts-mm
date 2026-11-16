# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Strict/lenient loading: every knob does what it says, and bad types never crash."""

from __future__ import annotations

import copy
from collections import Counter

import pytest
import yaml
from click.testing import CliRunner

from jtsmm.cli.main import main as cli_main
from jtsmm.config import load, validate_config, validate_strict


def _write(tmp_path, doc, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def _errors(issues):
    return [i for i in issues if i.startswith("error:")]


def test_relaxed_unknown_keys_really_relaxes(operator_config, tmp_path):
    doc = copy.deepcopy(operator_config)
    doc["host"]["schema"]["relaxed_unknown_keys"] = True
    doc["host"]["debug_only"] = True
    doc["mcu"]["emg"]["experimental"] = 1
    issues = validate_strict(doc)
    assert _errors(issues) == []
    assert any(i.startswith("warn:") and "host.debug_only" in i for i in issues)
    assert load(_write(tmp_path, doc))["host"]["debug_only"] is True


def test_unknown_key_reported_once(operator_config):
    doc = copy.deepcopy(operator_config)
    doc["host"]["debug_only"] = True
    errs = _errors(validate_strict(doc))
    assert len(errs) == 1 and "unknown key not in schema: host.debug_only" in errs[0]


def test_missing_key_reported_once(operator_config):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]["settle_ms"]
    errs = _errors(validate_strict(doc))
    assert errs == ["error: missing required key: host.capture.settle_ms"]


def test_load_validate_false_is_honoured_under_strict(operator_config, tmp_path):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]
    doc["mcu"]["emg"]["sample_rate_sps"] = "2k"
    path = _write(tmp_path, doc)
    with pytest.raises(ValueError):
        load(path)
    assert "capture" not in load(path, validate=False)["host"]


def test_load_strict_false_skips_completeness_only(operator_config, tmp_path):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]["settle_ms"]
    assert load(_write(tmp_path, doc), strict=False)
    doc["mcu"]["emg"]["sample_rate_sps"] = 333
    with pytest.raises(ValueError, match="sample_rate_sps"):
        load(_write(tmp_path, doc, "bad.yaml"), strict=False)


def test_config_validate_no_strict_flag_is_read(operator_config, tmp_path):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]["settle_ms"]
    path = _write(tmp_path, doc)
    runner = CliRunner()
    assert runner.invoke(cli_main, ["config", "validate", "--config", str(path)]).exit_code == 1
    res = runner.invoke(cli_main, ["config", "validate", "--config", str(path), "--no-strict"])
    assert res.exit_code == 0, res.output


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["mcu"]["emg"].__setitem__("sample_rate_sps", "2k"),
        lambda d: d["mcu"]["imu_mta"]["reports"].__setitem__("gyroscope", None),
        lambda d: d["host"]["cameras"]["list"][0].__setitem__("gain", "high"),
        lambda d: d["mcu"]["trigger"].__setitem__("rate_hz", "thirty"),
    ],
)
def test_wrong_types_reported_not_raised(operator_config, mutate):
    doc = copy.deepcopy(operator_config)
    mutate(doc)
    issues = validate_strict(doc)  # must not raise
    errs = _errors(issues)
    assert errs, issues
    assert all(n == 1 for n in Counter(errs).values()), errs
    assert any(i.startswith("info:") and "skipped" in i for i in issues)
    assert validate_config(doc)  # the public helper returns schema errors, no crash


def test_schema_errors_carry_their_location(operator_config):
    doc = copy.deepcopy(operator_config)
    doc["mcu"]["emg"]["sample_rate_sps"] = 333
    errs = _errors(validate_strict(doc))
    assert len(errs) == 1 and "mcu.emg.sample_rate_sps:" in errs[0]
