# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Preflight issue classification tests."""

from __future__ import annotations

from jtsmm.doctor.preflight_issues import partition_preflight_issues


def test_partition_preflight_issues():
    issues = [
        "warn: camera sysfs missing",
        "info: viewer local URL http://127.0.0.1:8770/",
        "info: viewer LAN URL http://192.168.1.2:8770/",
        "serial port not found",
    ]
    errors, warnings, infos = partition_preflight_issues(issues)
    assert errors == ["serial port not found"]
    assert warnings == ["camera sysfs missing"]
    assert len(infos) == 2
    assert infos[0].startswith("viewer local URL")
