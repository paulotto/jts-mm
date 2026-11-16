# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One data-dir key, an installer that sets it, and tests that never touch ~/jtsmm_data."""

from __future__ import annotations

import copy
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import yaml

from jtsmm.config import load, validate_config
from jtsmm.config import registry as reg
from jtsmm.config.loader import template_path
from jtsmm.io.session_layout import resolve_storage_base

from .conftest import REAL_DATA_DIRS, REAL_HOME

REPO = Path(__file__).resolve().parents[3]


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "set_config_value", REPO / "scripts" / "set_config_value.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- test isolation -------------------------------------------------------------------


def test_default_data_dir_is_not_the_real_one(operator_config, tmp_path_factory):
    base = tmp_path_factory.getbasetemp()
    assert Path(os.path.expanduser("~")) != REAL_HOME
    for path in (reg.paths_data_dir(operator_config), str(resolve_storage_base(operator_config))):
        assert Path(path).is_relative_to(base), path
        assert not any(Path(path).is_relative_to(d) for d in REAL_DATA_DIRS)


def test_isolation_reaches_subprocesses(tmp_path_factory):
    code = (
        "from jtsmm.config.loader import load_yaml, template_path;"
        "from jtsmm.config.registry import paths_data_dir;"
        "print(paths_data_dir(load_yaml(template_path())))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert Path(out).is_relative_to(tmp_path_factory.getbasetemp()), out


def test_tmp_config_path_redirects_every_data_dir(tmp_config_path, tmp_path):
    cfg = load(tmp_config_path)
    assert reg.paths_data_dir(cfg) == str(tmp_path / "data")


# --- one key, deprecated alias ----------------------------------------------------------


def test_data_dir_wins_and_differing_alias_is_rejected(operator_config):
    cfg = copy.deepcopy(operator_config)
    cfg["host"]["paths"]["data_dir"] = "/data/a"
    cfg["host"]["storage"] = {"base_dir": "/data/b"}
    assert reg.paths_data_dir(cfg) == "/data/a"
    msgs = validate_config(cfg)
    assert any(not m.startswith("warn:") and "IGNORED" in m for m in msgs)
    cfg["host"]["storage"]["base_dir"] = "/data/a"
    assert not any("storage.base_dir" in m for m in validate_config(cfg))


def test_alias_alone_is_still_honoured(operator_config):
    cfg = copy.deepcopy(operator_config)
    del cfg["host"]["paths"]
    cfg["host"]["storage"] = {"base_dir": "/data/legacy"}
    assert reg.paths_data_dir(cfg) == "/data/legacy"


def test_template_has_no_deprecated_key():
    doc = yaml.safe_load(template_path().read_text(encoding="utf-8"))
    assert "storage" not in doc["host"]
    assert doc["host"]["paths"]["data_dir"] == "~/jtsmm_data"


# --- installer patch keeps comments -----------------------------------------------------


def test_set_config_value_patches_data_dir_and_keeps_comments(tmp_path):
    mod = _load_script()
    dst = tmp_path / "jtsmm.yaml"
    dst.write_text(template_path().read_text(encoding="utf-8"), encoding="utf-8")
    before = dst.read_text(encoding="utf-8")
    assert mod.main([str(dst), "host.paths.data_dir", "/home/pi/my data"]) == 0
    after = dst.read_text(encoding="utf-8")
    assert yaml.safe_load(after)["host"]["paths"]["data_dir"] == "/home/pi/my data"
    changed = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert len(changed) == 1 and "data_dir" in changed[0][1]
    assert before.count("#") == after.count("#")
    # Absent key: error, unless --if-present.
    assert mod.main([str(dst), "host.storage.base_dir", "/x"]) == 1
    assert mod.main([str(dst), "host.storage.base_dir", "/x", "--if-present"]) == 0


def test_set_config_value_nested_and_inline_comment(tmp_path):
    mod = _load_script()
    text = (
        "host:\n"
        "  cameras:\n"
        "    list:\n"
        "      - id: cam0\n"
        "        data_dir: nope\n"
        "  storage:\n"
        "    base_dir: ~/old   # legacy\n"
        "  paths:\n"
        "    data_dir: ~/jtsmm_data\n"
        "session:\n"
        "  data_dir: nope\n"
    )
    new, found = mod.set_scalar(text, "host.storage.base_dir", "/d")
    assert found and '    base_dir: "/d"   # legacy\n' in new
    new, found = mod.set_scalar(new, "host.paths.data_dir", "/d")
    doc = yaml.safe_load(new)
    assert doc["host"]["paths"]["data_dir"] == "/d"
    assert doc["session"]["data_dir"] == "nope"
    assert doc["host"]["cameras"]["list"][0]["data_dir"] == "nope"
