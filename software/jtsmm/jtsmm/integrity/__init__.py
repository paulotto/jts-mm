# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Integrity checks — post-session HDF5 validation and session summaries."""

from __future__ import annotations

from .checker import (
    IntegrityReport,
    check_session,
    check_session_issues,
    format_report,
)
from .optical import check_optical
from .summary import (
    INTEGRITY_JSON_NAME,
    INTEGRITY_TXT_NAME,
    SessionIntegritySummary,
    format_session_summary,
    report_to_dict,
    resolve_session_h5,
    write_session_integrity,
)

__all__ = [
    "INTEGRITY_JSON_NAME",
    "INTEGRITY_TXT_NAME",
    "IntegrityReport",
    "SessionIntegritySummary",
    "check_optical",
    "check_session",
    "check_session_issues",
    "format_report",
    "format_session_summary",
    "report_to_dict",
    "resolve_session_h5",
    "write_session_integrity",
]
