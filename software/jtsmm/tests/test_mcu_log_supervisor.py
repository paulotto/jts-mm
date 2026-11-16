# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Firmware log lines (REC_LOG) reach the operator and /events on both writer paths.

They carry the firmware's own diagnosis (IMU NOT FOUND, IMU WATCHDOG, imu recover
... FAIL); both pump paths used to drop them.
"""

from __future__ import annotations

import json

import h5py
import pytest

from jtsmm import native
from jtsmm.io.session_reader import Session
from jtsmm.protocol.types import RecordType
from tests.test_fix_F_supervisor import _rec, env  # noqa: F401 - pytest fixture

# a real (non-mock) supervisor run needs the native capture extension on both writer paths
pytestmark = pytest.mark.skipif(not native.is_available(), reason="native extension not built")

LINES = [
    "imu_mta: NOT FOUND on I2C (probe failed) -- slot stays configured; ...",
    "IMU WATCHDOG: mta stalled 3s, recovery attempt 1/6",
]


@pytest.mark.parametrize("python_writer", [False, True])
def test_firmware_log_printed_and_stored(env, monkeypatch, capsys, python_writer):  # noqa: F811
    from jtsmm.capture.supervisor import SessionSupervisor

    if python_writer:
        monkeypatch.delenv("JTSMM_NATIVE", raising=False)
        env.resolved["host"].setdefault("capture", {})["native_prefer"] = False
    env.mcu.q[10:10] = [_rec(RecordType.LOG, 6_000, {"text": LINES[0] + "\r\n"})]
    env.mcu.q[200:200] = [_rec(RecordType.LOG, 101_000, {"text": LINES[1]})]
    monkeypatch.setattr(env.sup_mod, "_FINAL_STATS_TIMEOUT_S", 0.05)

    h5 = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.4)

    err = capsys.readouterr().err
    assert ("HDF5 writer: Python" if python_writer else "HDF5 writer: native") in err
    for line in LINES:
        assert f"mcu: {line}\n" in err
    with Session(h5) as s:
        logs = s.mcu_logs()
        assert [e["text"] for e in logs] == LINES  # trailing CR/LF stripped
        assert [e["t_master_us"] for e in logs] == [6_000, 101_000]  # MCU stamp, unmapped
        assert {e["map_quality"] for e in logs} == {"mcu_clock"}
        assert s.notes() == []
    with h5py.File(h5, "r") as f:
        sync = json.loads(f["sync"].attrs["report_json"])
    assert sync["mcu_logs"] == {"n": 2, "n_failed": 0, "stored_in": "/events (kind=mcu_log)"}
