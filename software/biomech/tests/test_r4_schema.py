# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every stored array family has a data-dictionary entry (results/schema.py)."""

from __future__ import annotations

import pytest

from biomech.results.schema import array_entry, describe

pytestmark = pytest.mark.slow


def test_every_array_family_is_described(synth_study):
    _, store, _, _ = synth_study
    d = describe(store)
    missing = [(s, f) for s, v in d.items() for f, e in v["arrays"].items()
               if not e.get("pattern")]
    assert not missing, f"describe these array families in results/schema.ARRAYS: {missing}"
    assert array_entry("kinematics", "{task}__incisor")["unit"] == "mm"
