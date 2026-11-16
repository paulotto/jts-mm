# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Supervisor stores a start REC_STATS snapshot (sync.mcu_stats_start)."""

from __future__ import annotations

import h5py
import pytest

from jtsmm import native
from jtsmm.protocol.types import RecordType
from tests.test_fix_F_supervisor import _rec, _sync, env  # noqa: F401 - fixture

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)


def _stats_replies(env, monkeypatch, start: dict | None, end: dict) -> None:
    def send(_ser) -> None:
        if "streams_off" not in env.log:
            env.log.append("stats_start")
            if start is not None:  # arrives behind the records already queued
                env.mcu.q.append(_rec(RecordType.STATS, 600_000, dict(start)))
            return
        env.log.append("stats")
        env.mcu.q.extend(_rec(RecordType.EMG_BLOCK, 900_000 + i, {"channels": [0] * 8})
                         for i in range(5))
        env.mcu.q.append(_rec(RecordType.STATS, 999_000, dict(end)))

    monkeypatch.setattr(env.sup_mod, "_send_stats_request", send)


@pytest.mark.parametrize("python_writer", [False, True])
def test_start_snapshot_is_stored_and_nothing_is_discarded(env, monkeypatch, python_writer):
    from jtsmm.capture.supervisor import SessionSupervisor

    if python_writer:
        monkeypatch.delenv("JTSMM_NATIVE", raising=False)
        env.resolved["host"].setdefault("capture", {})["native_prefer"] = False
    _stats_replies(env, monkeypatch, {"emg_push": 1000, "stats_seq": 3},
                   {"emg_push": 1405, "stats_seq": 9})
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.2)
    assert env.log.index("streams_on") < env.log.index("stats_start") < env.log.index("streams_off")
    sync = _sync(h5_path)
    start = sync["mcu_stats_start"]
    assert start["emg_push"] == 1000 and start["t_master_us"] == 600_000
    # the 400 records queued before the snapshot went into the file, counted
    assert start["host_emg_rows_before"] == 400
    assert sync["mcu_stats_end"]["stats_seq"] == 9 and sync["mcu_stats_end_fresh"] is True
    with h5py.File(h5_path, "r") as f:
        assert f["emg"]["samples"].shape[0] == 405


def test_no_start_reply_is_bounded_and_keeps_every_record(env, monkeypatch):
    from jtsmm.capture.supervisor import SessionSupervisor

    monkeypatch.setattr(env.sup_mod, "_START_STATS_TIMEOUT_S", 0.1)
    _stats_replies(env, monkeypatch, None, {"emg_push": 5, "stats_seq": 1})
    h5_path = SessionSupervisor(env.resolved, port="/dev/null").run(duration_s=0.2)
    sync = _sync(h5_path)
    assert "mcu_stats_start" not in sync
    with h5py.File(h5_path, "r") as f:
        assert f["emg"]["samples"].shape[0] == 405
