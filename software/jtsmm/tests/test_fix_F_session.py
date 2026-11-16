# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""run_session: stop-before-apply, CLI overrides reach the MCU payload, skip trace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from jtsmm.config import applied_config_hash, build_mcu_apply_payload


class _Ser:
    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


class _FakeSupervisor:
    last: "_FakeSupervisor | None" = None

    def __init__(self, resolved: dict, port: str, **kw: Any) -> None:
        self.resolved = resolved
        self.port = port
        self.kw = kw
        _FakeSupervisor.last = self

    def run(self, duration, **_kw) -> Path:
        return Path("/nonexistent/session.h5")


@pytest.fixture
def patched_session(monkeypatch):
    from jtsmm.capture import session as sess

    calls: list[Any] = []
    monkeypatch.setattr(sess, "find_hat_port", lambda _p: "/dev/fake_hat")
    monkeypatch.setattr(sess, "open_hat_serial", lambda _p: _Ser())
    monkeypatch.setattr(sess, "stop_mcu_streams", lambda _ser: calls.append(("stop",)))
    monkeypatch.setattr(sess, "stop_mcu_streams_robust", lambda _p, **_k: calls.append(("teardown",)))

    def fake_apply(_ser, resolved, **kw):
        calls.append(("apply", build_mcu_apply_payload(resolved)))
        rep = kw.get("report")
        if rep is not None:
            rep.update(result="ok", config_dump={"emg": {"default_gain": 12}}, verify_problems=[])
        return True

    monkeypatch.setattr(sess, "apply_mcu", fake_apply)
    monkeypatch.setattr(sess, "read_config_dump", lambda _ser: {"emg": {"default_gain": 6}})
    monkeypatch.setattr(sess, "SessionSupervisor", _FakeSupervisor)
    return sess, calls


_NO_OPTICAL = dict(optical_track=False, ct_calibration=False, use_experiment_plan=False, mvc=False)


def test_overrides_reach_applied_payload_and_session_config(patched_session, tmp_config_path):
    sess, calls = patched_session
    sess.run_session(
        tmp_config_path,
        duration_s=0.1,
        default_gain=12,
        bias_drive=True,
        bias_channels="1,2",
        active_channels="1-3",
        leadoff=False,
        **_NO_OPTICAL,
    )
    kinds = [c[0] for c in calls]
    assert kinds.index("stop") < kinds.index("apply"), kinds
    payload = next(c[1] for c in calls if c[0] == "apply")
    emg = payload["emg"]
    assert emg["default_gain"] == 12
    assert emg["bias"]["enabled"] is True and emg["bias"]["channels"] == 0b000011
    assert emg["pd_channels"] == 0b111000
    assert emg["leadoff"]["during_recording"] is False

    sup = _FakeSupervisor.last
    assert sup is not None
    # The supervisor (→ /meta mcu_config_json + config_hash) sees the same config.
    assert build_mcu_apply_payload(sup.resolved)["emg"] == emg
    assert applied_config_hash(sup.resolved) != ""
    meta = sup.kw["extra_meta"]
    assert meta["mcu_apply_skipped"] == "false"
    assert json.loads(meta["mcu_config_dump_json"]) == {"emg": {"default_gain": 12}}
    assert kinds[-1] == "teardown"


def test_skipped_apply_is_recorded_with_a_dump_snapshot(patched_session, tmp_config_path):
    sess, calls = patched_session
    sess.run_session(tmp_config_path, duration_s=0.1, skip_mcu_apply=True, **_NO_OPTICAL)
    assert not [c for c in calls if c[0] == "apply"]
    meta = _FakeSupervisor.last.kw["extra_meta"]
    assert meta["mcu_apply_skipped"] == "true"
    assert meta["mcu_apply_skip_reason"] == "--skip-apply"
    assert json.loads(meta["mcu_config_dump_json"]) == {"emg": {"default_gain": 6}}


def test_apply_disabled_in_config_is_recorded(patched_session, tmp_config_path):
    import yaml

    doc = yaml.safe_load(tmp_config_path.read_text())
    doc["host"]["record"]["mcu_apply"]["enabled"] = False
    tmp_config_path.write_text(yaml.safe_dump(doc))
    sess, calls = patched_session
    sess.run_session(tmp_config_path, duration_s=0.1, **_NO_OPTICAL)
    meta = _FakeSupervisor.last.kw["extra_meta"]
    assert meta["mcu_apply_skip_reason"] == "host.record.mcu_apply.enabled=false"


def test_failed_apply_raises_with_result(patched_session, tmp_config_path, monkeypatch):
    sess, _calls = patched_session

    def failing(_ser, _resolved, **kw):
        kw["report"]["result"] = "busy"
        return False

    monkeypatch.setattr(sess, "apply_mcu", failing)
    with pytest.raises(RuntimeError, match="busy"):
        sess.run_session(tmp_config_path, duration_s=0.1, **_NO_OPTICAL)


def test_teardown_stop_failure_is_warned(monkeypatch, capsys):
    from jtsmm.capture import session as sess

    def boom(_port, **_kw):
        raise OSError("port gone")

    monkeypatch.setattr(sess, "stop_mcu_streams_robust", boom)
    sess._teardown_mcu_streams("/dev/fake_hat")
    err = capsys.readouterr().err
    assert "port gone" in err and "/dev/fake_hat" in err
