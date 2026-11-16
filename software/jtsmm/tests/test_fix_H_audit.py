# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm config audit` must be able to fail, and must check real firmware keys."""

from __future__ import annotations

import copy
import re

import pytest
import yaml
from click.testing import CliRunner

from jtsmm.cli.main import main as cli_main
from jtsmm.config.coverage import (
    FIRMWARE_JSON_KEYS,
    load_coverage_manifest,
    run_audit,
    schema_leaf_paths,
)
from jtsmm.config.loader import checkout_root, load_yaml, template_path

_CALL = re.compile(
    r"\b(GET_SECTION|GET_BOOL|GET_UINT|GET_STR|json_bool|json_uint|json_str|"
    r"required_section)\(\s*(\w+),\s*\"(\w+)\"([^;]*)"
)


def _extract_firmware_keys(src: str) -> set[str]:
    """Every key config_json.c reads, as wire paths (see FIRMWARE_JSON_KEYS).

    Walks each parse_* body in order: GET_SECTION(parent, "key", var) binds var
    to a nested path (a var may be rebound to the next section), every field
    read under a bound var yields a key.
    """
    prefixes = {
        "config_json_parse": "",
        "parse_platform": "platform",
        "parse_emg": "emg",
        "parse_imu_slot": "imu",
        "parse_imu_reports": "imu.reports.*",
        "parse_trigger": "trigger",
    }
    out: set[str] = set()
    for m in re.finditer(r"^(?:static )?bool (\w+)\(([^)]*)\) \{\n(.*?)^\}\n", src, re.S | re.M):
        name, args, body = m.groups()
        if name not in prefixes:
            continue
        # cJSON variable -> wire path prefix: the function's cJSON argument, the
        # parsed root in config_json_parse, the per-report `item` loop variable.
        scopes: dict[str, str] = {}
        first = re.match(r"\s*cJSON \*(\w+)", args)
        if first:
            scopes[first.group(1)] = prefixes[name]
        if name == "config_json_parse":
            scopes["root"] = ""
        if name == "parse_imu_reports":
            scopes["item"] = prefixes[name]
        for kind, parent, key, rest in _CALL.findall(body):
            if parent not in scopes:
                continue
            path = f"{scopes[parent]}.{key}" if scopes[parent] else key
            if kind == "GET_SECTION":
                scopes[re.findall(r"\w+", rest)[-1]] = path
            elif kind != "required_section":
                out.add(path)
    return out


def test_firmware_key_list_matches_config_json_c():
    root = checkout_root()
    if root is None:
        pytest.skip("needs the repository checkout")
    src = (root / "firmware" / "jtsmm" / "src" / "runtime" / "config_json.c").read_text()
    assert _extract_firmware_keys(src) == set(FIRMWARE_JSON_KEYS)


def test_every_schema_key_is_in_the_manifest():
    manifest = [str(e["path"]) for e in load_coverage_manifest()]
    missing = sorted(
        p for p in schema_leaf_paths()
        if not any(p == m or p.startswith(m + ".") for m in manifest)
    )
    assert missing == []


def test_every_manifest_path_exists_in_the_schema():
    leaves = schema_leaf_paths()
    stale = sorted(
        str(e["path"]) for e in load_coverage_manifest()
        if not any(p == e["path"] or p.startswith(f"{e['path']}.") for p in leaves)
    )
    assert stale == []


def test_template_audit_is_clean():
    report = run_audit(load_yaml(template_path()))
    bad = {e.path: e.status for e in report.entries if e.status not in ("OK", "DEFAULT")}
    assert bad == {}


def test_audit_flags_a_key_the_firmware_does_not_parse(monkeypatch):
    import jtsmm.config.coverage as cov

    keys = set(cov.FIRMWARE_JSON_KEYS) - {"platform.pio.trigger.clock_hz"}
    monkeypatch.setattr(cov, "FIRMWARE_JSON_KEYS", frozenset(keys))
    statuses = {e.path: e.status for e in run_audit(load_yaml(template_path())).entries}
    assert statuses["mcu.platform.pio.trigger.clock_hz"] == "MISSING_FW"


def test_config_audit_strict_passes_on_template():
    res = CliRunner().invoke(cli_main, ["config", "audit", "--strict", "--config", str(template_path())])
    assert res.exit_code == 0, res.output


def test_config_audit_strict_fails_on_broken_file(tmp_path, operator_config):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]
    del doc["mcu"]["trigger"]["rate_hz"]
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    res = CliRunner().invoke(cli_main, ["config", "audit", "--strict", "--config", str(path)])
    assert res.exit_code == 1, res.output
    assert "MISSING_YAML: host.capture.settle_ms" in res.output
    assert res.exception is None or isinstance(res.exception, SystemExit)


def test_config_audit_unreadable_file_exits_1(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- not a mapping\n", encoding="utf-8")
    res = CliRunner().invoke(cli_main, ["config", "audit", "--config", str(path)])
    assert res.exit_code == 1 and "cannot read" in res.output


def test_config_audit_defaults_to_the_loaded_config(_isolate_home, operator_config, monkeypatch):
    # In a checkout the template beats ~/jtsmm.yaml; $JTSMM_CONFIG beats both.
    doc = copy.deepcopy(operator_config)
    del doc["session"]["protocol"]
    path = _isolate_home / "jtsmm.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setenv("JTSMM_CONFIG", str(path))
    res = CliRunner().invoke(cli_main, ["config", "audit", "--strict"])
    assert f"file: {path}" in res.output
    assert "MISSING_YAML: session.protocol" in res.output
    assert res.exit_code == 1


def test_optional_key_absent_is_default_not_failure(operator_config):
    doc = copy.deepcopy(operator_config)
    del doc["host"]["capture"]["pps_samples"]
    statuses = {e.path: e.status for e in run_audit(doc).entries}
    assert statuses["host.capture.pps_samples"] == "DEFAULT"
