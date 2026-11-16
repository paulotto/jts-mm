# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Ctrl-C during a mock take is a stop request: file + integrity.json still written."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
from click.testing import CliRunner

from jtsmm.cli.record import record_cmd
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL


def test_keyboard_interrupt_in_loop_still_writes_integrity(tmp_config_path: Path, monkeypatch):
    from jtsmm.capture import capture_cam

    calls = {"n": 0}
    real = capture_cam.CameraCapture.drain_frames

    def drain(self, **kw):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return real(self, **kw)

    monkeypatch.setattr(capture_cam.CameraCapture, "drain_frames", drain)
    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", *MOCK_RECORD_DISABLE_OPTICAL,
         "--duration", "30", "--no-doctor", "--no-emg-check", "--no-abort"],
    )
    assert result.exit_code == 0, result.output
    out_line = next(line for line in result.output.splitlines() if line.startswith("Wrote session"))
    h5 = Path(out_line.split("-> ", 1)[1].strip())
    assert (h5.parent / "integrity.json").is_file()
    with h5py.File(h5, "r") as f:
        assert json.loads(f["sync"].attrs["report_json"])["end_reason"] == "operator_stop"
