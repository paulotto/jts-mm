# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The checker grades how a take ended (supervisor sync_report fields)."""

from __future__ import annotations

from pathlib import Path

import pytest

from jtsmm.integrity.checker import IntegrityReport, _check_sync_report


def _run(sync: dict) -> IntegrityReport:
    rep = IntegrityReport(path=Path("x.h5"))
    _check_sync_report(sync, rep)
    return rep


@pytest.mark.parametrize("reason", ["duration", "phases_complete", "ct_calibration_only"])
def test_planned_ends_are_clean(reason):
    rep = _run({"end_reason": reason, "mcu_stats_end_fresh": True})
    assert not rep.issues and not rep.warnings
    assert rep.summary["sync"]["end_reason"] == reason


@pytest.mark.parametrize("reason", ["operator_stop", "signal", "operator_abort", "plan_timeout"])
def test_early_operator_or_signal_end_warns(reason):
    rep = _run({"end_reason": reason})
    assert not rep.issues
    assert any(reason in w for w in rep.warnings)


@pytest.mark.parametrize("reason", ["watchdog", "setup", "exception: OSError: disk full"])
def test_watchdog_or_exception_end_is_an_error(reason):
    rep = _run({"end_reason": reason})
    assert any(reason in e for e in rep.issues)


def test_stale_stats_teardown_and_hold_overflow_warn():
    rep = _run({
        "end_reason": "duration",
        "mcu_stats_end_fresh": False,
        "teardown_errors": ["optical.shutdown"],
        "cam_hold_overflow": 9,
    })
    text = " | ".join(rep.warnings)
    assert "mcu_stats_end predates" in text
    assert "optical.shutdown" in text
    assert "cam_hold_overflow=9" in text
    assert not rep.issues


def test_old_sessions_without_the_fields_are_silent():
    rep = _run({})
    assert not rep.issues and not rep.warnings
