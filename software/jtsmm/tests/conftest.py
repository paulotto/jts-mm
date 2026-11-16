# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

from .helpers import load_operator_template

# The operator's real home, captured before any test redirects HOME. Tests must
# never write under it: the template's data dir is ~/jtsmm_data and a leaked test
# session lands next to real recordings (331 session_test_emg_* dirs did).
REAL_HOME = Path(os.path.expanduser("~"))
REAL_DATA_DIRS = (REAL_HOME / "jtsmm_data",)


def _listing(path: Path) -> set[str]:
    try:
        return set(os.listdir(path))
    except OSError:
        return set()


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config._jtsmm_real_data = {p: _listing(p) for p in REAL_DATA_DIRS}  # type: ignore[attr-defined]


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Fail the run if anything appeared in the real data dir while tests ran.

    Another process writing there at the same time (a real recording, or a
    second test run on an unfixed checkout) would also trip this; the message
    lists the entries so it is easy to tell.
    """
    before = getattr(session.config, "_jtsmm_real_data", None)
    if not before:
        return
    leaked = {p: sorted(_listing(p) - names) for p, names in before.items()}
    leaked = {p: names for p, names in leaked.items() if names}
    if leaked:
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        for path, names in leaked.items():
            msg = f"TEST ISOLATION FAILURE: {len(names)} new entries in {path}: {names[:10]}"
            if reporter is not None:
                reporter.write_line(msg, red=True)
            else:
                print(msg)
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture(autouse=True)
def _isolate_home(tmp_path_factory, monkeypatch):
    """Point HOME (and every data-dir override) at a temp dir for every test.

    Covers ~/jtsmm_data, ~/jtsmm.yaml, ~/experiments/... and the XDG fallback,
    in-process and in any subprocess a test starts.
    """
    home = tmp_path_factory.mktemp("home")  # fresh per test: no ~/jtsmm.yaml carried over
    monkeypatch.setenv("HOME", str(home))
    for var in ("JTSMM_DATA_DIR", "JTSMM_CONFIG", "XDG_DATA_HOME"):
        monkeypatch.delenv(var, raising=False)
    # The stable IMU mounting file is a repo file (like the CT tip): a test that fits a
    # mounting with the default config must not overwrite it -- redirect the default path.
    from jtsmm.config import registry as reg

    orig = reg.session_imu_mounting_calibration
    repo_default = Path(__file__).resolve().parents[1] / "etc" / "imu_mounting_calibration.json"

    def _mounting(resolved):
        p = orig(resolved)
        if p and Path(p).resolve() == repo_default:
            return str(home / "imu_mounting_calibration.json")
        return p

    _mounting.__module__ = reg.__name__        # still a registry accessor for the plumbing tests
    _mounting.__name__ = orig.__name__
    monkeypatch.setattr(reg, "session_imu_mounting_calibration", _mounting)
    yield home


@pytest.fixture
def operator_config() -> dict:
    return load_operator_template()


@pytest.fixture
def resolved_config(operator_config) -> dict:
    return operator_config


@pytest.fixture
def tmp_config_path(tmp_path, operator_config) -> Path:
    """Operator template with every data dir under ``tmp_path/data``."""
    data = str(tmp_path / "data")
    host = operator_config.setdefault("host", {})
    host.setdefault("paths", {})["data_dir"] = data
    if "storage" in host:
        host["storage"]["base_dir"] = data
    path = tmp_path / "jtsmm.yaml"
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(operator_config, fh)
    return path
