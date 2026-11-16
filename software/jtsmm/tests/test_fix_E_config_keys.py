# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every integrity gate is a documented, schema-valid config key."""

from __future__ import annotations

import dataclasses
import json

from jtsmm.config.loader import repo_root
from jtsmm.integrity.checker import IntegrityThresholds

from tests.helpers import load_operator_template


def _fields() -> set[str]:
    return {f.name for f in dataclasses.fields(IntegrityThresholds)}


def test_schema_lists_every_threshold():
    schema = json.loads((repo_root() / "schema" / "v1" / "jtsmm.schema.json").read_text())
    blob = json.dumps(schema)
    missing = [k for k in _fields() if f'"{k}"' not in blob]
    assert not missing, missing


def test_template_sets_every_threshold_to_its_default():
    block = load_operator_template()["host"]["integrity"]["thresholds"]
    defaults = IntegrityThresholds()
    assert set(block) == _fields()
    for k, v in block.items():
        assert float(v) == getattr(defaults, k), k


def test_s08_documents_every_threshold():
    text = (repo_root() / "docs" / "software" / "s08_runtime_configuration.md").read_text()
    # The table writes an err partner as "`warn_key` / `..._err...`".
    partner = {e: w for w, e in IntegrityThresholds._WARN_ERR_PAIRS}
    missing = [k for k in _fields()
               if f"`{k}`" not in text and f"`{partner.get(k, k)}`" not in text]
    assert not missing, missing
