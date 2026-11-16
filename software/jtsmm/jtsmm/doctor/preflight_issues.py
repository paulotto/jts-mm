# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Classify preflight issue strings (error / warn / info)."""

from __future__ import annotations


def partition_preflight_issues(
    issues: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Split issues into (errors, warnings, infos)."""
    errors: list[str] = []
    warnings: list[str] = []
    infos: list[str] = []
    for item in issues:
        if item.startswith("warn:"):
            warnings.append(item[6:])
        elif item.startswith("info:"):
            infos.append(item[6:])
        else:
            errors.append(item)
    return errors, warnings, infos
