# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Template drift: sync_config_template.py fails on every kind of gap; dead keys warn."""

from __future__ import annotations

import copy
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

from jtsmm.config import validate_config
from jtsmm.config.loader import template_path

REPO = Path(__file__).resolve().parents[3]


def _script():
    spec = importlib.util.spec_from_file_location(
        "sync_config_template", REPO / "scripts" / "sync_config_template.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_template_passes(tmp_path):
    assert _script().main([]) == 0


def _variant(tmp_path, text: str) -> list[str]:
    path = tmp_path / "t.yaml"
    path.write_text(text, encoding="utf-8")
    return ["--template", str(path)]


def test_missing_required_key_fails_without_check(tmp_path):
    text = template_path().read_text(encoding="utf-8").replace("    settle_ms: 500\n", "", 1)
    assert _script().main(_variant(tmp_path, text)) == 1


def test_schema_error_fails(tmp_path):
    text = template_path().read_text(encoding="utf-8").replace(
        "sample_rate_sps: 4000", "sample_rate_sps: 333", 1
    )
    assert _script().main(_variant(tmp_path, text)) == 1


def test_unlisted_optional_key_fails(tmp_path):
    text = re.sub(r"^\s*# active_channels:.*\n", "", template_path().read_text(), count=1, flags=re.M)
    mod = _script()
    assert mod.main(_variant(tmp_path, text)) == 1
    assert "mcu.emg.active_channels" in mod.unlisted_schema_keys(yaml.safe_load(text), text)


def test_template_documents_its_real_trigger_timing():
    doc = yaml.safe_load(template_path().read_text(encoding="utf-8"))
    ref = yaml.safe_load((REPO / "schema" / "v1" / "jtsmm.schema.yaml").read_text())
    trig = doc["mcu"]["trigger"]
    assert ref["mcu"]["trigger"]["rate_hz"] == trig["rate_hz"] == 40
    assert ref["mcu"]["trigger"]["pulse"]["width_us"] == trig["pulse"]["width_us"]
    for cam in doc["host"]["cameras"]["list"]:
        assert cam["trigger"]["expect_rate_hz"] == trig["rate_hz"]


@pytest.mark.parametrize(
    "block,warns",
    [({"mains_mode": "track"}, False), ({"enabled": False, "mains_mode": "track"}, False),
     ({"enabled": True}, True), ({"notch_hz": 50}, True)],
)
def test_dead_emg_filter_keys_rejected(operator_config, block, warns):
    doc = copy.deepcopy(operator_config)
    doc["host"]["emg_filter"] = block
    hard = [m for m in validate_config(doc) if not m.startswith("warn:")]
    assert any("has no effect" in m for m in hard) is warns
