# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`jtsmm record` names the phase an abort happened in (not always "setup")."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from jtsmm.cli.record import record_cmd
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL


@pytest.mark.parametrize(
    ("phase", "text"),
    [("take", "Ctrl-C during the take"), ("finish", "Ctrl-C while finishing the take"),
     (None, "Ctrl-C during setup")],
)
def test_ctrl_c_message_names_the_phase(tmp_config_path: Path, tmp_path: Path, monkeypatch,
                                        phase, text):
    from jtsmm.integrity import IntegrityReport

    partial = tmp_path / "aborted" / "session.h5"
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"")

    def interrupted_run_session(*_a, **_kw):
        exc = KeyboardInterrupt()
        exc.jtsmm_session_h5 = partial  # type: ignore[attr-defined]
        if phase is not None:
            exc.jtsmm_phase = phase  # type: ignore[attr-defined]
        raise exc

    monkeypatch.setattr("jtsmm.cli.record.run_session", interrupted_run_session)
    monkeypatch.setattr(
        "jtsmm.integrity.check_session", lambda path, **_kw: IntegrityReport(path=path)
    )
    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", *MOCK_RECORD_DISABLE_OPTICAL,
         "--duration", "0.1", "--no-doctor"],
    )
    assert result.exit_code == 130, result.output
    assert f"record aborted ({text})" in result.output
    doc = json.loads((partial.parent / "integrity.json").read_text(encoding="utf-8"))
    assert f"recording aborted: {text}" in doc["checks"]["hdf5"]["errors"]
