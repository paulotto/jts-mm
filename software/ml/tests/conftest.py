# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every ML test carries the ``ml`` marker: deselected by default (``addopts``), run at the
end of a change set with ``pytest -m "slow or ml or not slow"``."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        item.add_marker(pytest.mark.ml)
