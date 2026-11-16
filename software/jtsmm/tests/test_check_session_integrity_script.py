# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""End-to-end test for scripts/check_session_integrity.sh against a real
(mock) recorded session, so the reusable integrity script is proven to work
without anyone having to hand-write one-off python each time a session needs
auditing."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from jtsmm.cli.record import record_cmd

from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL

from .test_session_reproduce import _force_tmp_storage, _venv_bin

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "check_session_integrity.sh"


def _record_session(tmp_config_path: Path, data_root: Path) -> Path:
    _force_tmp_storage(tmp_config_path, data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            *MOCK_RECORD_DISABLE_OPTICAL,
            "--duration",
            "0.1",
            "--no-doctor",
            "--no-emg-check",
        ],
        env={"JTSMM_DATA_DIR": str(data_root)},
    )
    assert result.exit_code == 0, result.output
    out_line = next(
        line for line in result.output.splitlines() if line.startswith("Wrote session")
    )
    return Path(out_line.split("-> ", 1)[1].strip())


@pytest.fixture()
def _bin_env() -> dict[str, str]:
    env = dict(os.environ)
    venv_bin = _venv_bin()
    env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def test_script_exists_and_is_executable():
    assert SCRIPT.is_file()
    assert os.access(SCRIPT, os.X_OK)


def test_script_checks_a_healthy_session(tmp_config_path: Path, tmp_path: Path, _bin_env):
    h5_path = _record_session(tmp_config_path, tmp_path / "data")
    proc = subprocess.run(
        [str(SCRIPT), str(h5_path)],
        capture_output=True, text=True, env=_bin_env, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "jtsmm integrity report" in proc.stdout or "jtsmm session integrity" in proc.stdout
    assert "RESULT: OK" in proc.stdout or "overall" in proc.stdout.lower()


def test_script_expands_a_directory_argument(tmp_config_path: Path, tmp_path: Path, _bin_env):
    h5_path = _record_session(tmp_config_path, tmp_path / "data")
    proc = subprocess.run(
        [str(SCRIPT), str(h5_path.parent)],
        capture_output=True, text=True, env=_bin_env, timeout=60,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (
        proc.stdout.count("jtsmm integrity report")
        + proc.stdout.count("jtsmm session integrity")
    ) >= 1


def test_script_fails_on_missing_file(_bin_env, tmp_path: Path):
    proc = subprocess.run(
        [str(SCRIPT), str(tmp_path / "does_not_exist.h5")],
        capture_output=True, text=True, env=_bin_env, timeout=60,
    )
    assert proc.returncode != 0


def test_script_defaults_to_data_session_h5_in_repo_cwd(tmp_path: Path, _bin_env):
    fake_root = tmp_path / "fake_repo"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "data").mkdir()
    shutil.copy(SCRIPT, fake_root / "scripts" / SCRIPT.name)
    os.chmod(fake_root / "scripts" / SCRIPT.name, 0o755)
    shutil.copy(REPO_ROOT / "data" / "session.h5", fake_root / "data" / "session.h5") \
        if (REPO_ROOT / "data" / "session.h5").is_file() else None
    if not (fake_root / "data" / "session.h5").is_file():
        pytest.skip("no data/session.h5 fixture available to copy")
    proc = subprocess.run(
        [str(fake_root / "scripts" / SCRIPT.name)],
        capture_output=True, text=True, env=_bin_env, cwd=fake_root, timeout=60,
    )
    assert "jtsmm integrity report" in proc.stdout or "jtsmm session integrity" in proc.stdout, (
        proc.stdout + proc.stderr
    )
