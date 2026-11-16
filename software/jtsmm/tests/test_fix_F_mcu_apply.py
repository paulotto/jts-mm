# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`config apply` result handling, BUSY retry, verification and stream-start replies."""

from __future__ import annotations

import copy
import json
from typing import Callable

import pytest

from jtsmm.config import mcu_apply
from jtsmm.config.mcu_apply import (
    apply_mcu,
    build_mcu_apply_payload,
    start_mcu_streams,
    start_mcu_trigger,
    verify_mcu_config,
    wait_config_result,
)


class ScriptedSerial:
    """CDC stand-in: ``responder(line)`` returns the text the MCU prints for a command."""

    def __init__(self, responder: Callable[[str], str]) -> None:
        self.responder = responder
        self.sent: list[str] = []
        self._rx = bytearray()

    @property
    def in_waiting(self) -> int:
        return len(self._rx)

    def read(self, n: int) -> bytes:
        out = bytes(self._rx[:n])
        del self._rx[:n]
        return out

    def write(self, data: bytes) -> None:
        line = data.decode("utf-8").strip()
        self.sent.append(line)
        self._rx.extend(self.responder(line).encode("utf-8"))

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        self._rx.clear()


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(mcu_apply.time, "sleep", lambda *_a, **_k: None)


def _dump_for(resolved: dict) -> dict:
    """What a compliant firmware reports after applying ``resolved``."""
    p = build_mcu_apply_payload(resolved)
    dump = {"emg": copy.deepcopy(p["emg"]), "trigger": {"rate_hz": p["trigger"]["rate_hz"],
                                                      "pulse": {"width_us": p["trigger"]["pulse"]["width_us"]}}}
    for slot in ("imu_mta", "imu_cra"):
        dump[slot] = {"reports": {n: {"rate_hz": c["rate_hz"]} for n, c in p[slot]["reports"].items()}}
    return dump


def _mcu(dump: dict, *, apply_replies: list[str], save_reply: str | None = "config save OK\r\n"):
    replies = list(apply_replies)

    def respond(line: str) -> str:
        if line.startswith("config apply"):
            return replies.pop(0) if replies else "config apply OK\r\n"
        if line == "config dump":
            return json.dumps(dump) + "\r\n"
        if line == "config save":
            return save_reply or ""
        return ""

    return respond


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("config apply started\r\nconfig apply OK\r\n", "ok"),
        ("config apply BUSY\r\n", "busy"),
        ("config apply parse FAIL\r\n", "fail"),
        ("config apply re-parse FAIL\r\n", "fail"),
        ("config apply: host overrides profile defaults: fs 1000->2000\r\nconfig apply ERR\r\n", "fail"),
        ("config init OK\r\n", "ok"),
    ],
)
def test_wait_config_result_classifies_whole_lines(text, want):
    ser = ScriptedSerial(lambda _l: "")
    ser._rx.extend(text.encode())
    assert wait_config_result(ser, timeout_s=0.5) == want


def test_wait_config_result_times_out_on_silence():
    assert wait_config_result(ScriptedSerial(lambda _l: ""), timeout_s=0.05) == "timeout"


def test_apply_busy_stops_streams_and_retries_once(resolved_config):
    ser = ScriptedSerial(_mcu(_dump_for(resolved_config), apply_replies=["config apply BUSY\r\n"]))
    report: dict = {}
    assert apply_mcu(ser, resolved_config, strict=False, report=report, busy_wait_s=0.05)
    applies = [i for i, s in enumerate(ser.sent) if s.startswith("config apply")]
    assert len(applies) == 2
    between = ser.sent[applies[0] + 1 : applies[1]]
    assert "ads_stop" in between and "trig_stop" in between
    assert report["result"] == "ok" and report["verify_problems"] == []
    assert report["config_dump"]["emg"]["default_gain"] == build_mcu_apply_payload(resolved_config)["emg"]["default_gain"]


def test_apply_busy_twice_fails_loudly(resolved_config, capsys):
    ser = ScriptedSerial(
        _mcu({}, apply_replies=["config apply BUSY\r\n", "config apply BUSY\r\n"])
    )
    assert not apply_mcu(ser, resolved_config, strict=False, busy_wait_s=0.05)
    assert "NOT applied" in capsys.readouterr().err


def test_apply_fail_line_returns_false(resolved_config):
    ser = ScriptedSerial(_mcu({}, apply_replies=["config apply parse FAIL\r\n"]))
    assert not apply_mcu(ser, resolved_config, strict=False, timeout_s=0.5)


def test_apply_fail_reason_is_reported(resolved_config, capsys):
    reply = "config apply started\r\nconfig apply FAIL (trigger.pulse.width_us >= period)\r\n"
    ser = ScriptedSerial(_mcu({}, apply_replies=[reply]))
    report: dict = {}
    assert not apply_mcu(ser, resolved_config, strict=False, timeout_s=0.5, report=report)
    assert report["result"] == "fail"
    assert report["line"] == "config apply FAIL (trigger.pulse.width_us >= period)"
    assert "width_us >= period" in capsys.readouterr().err


def test_persist_without_save_answer_is_a_failure(resolved_config, capsys):
    ser = ScriptedSerial(_mcu(_dump_for(resolved_config), apply_replies=[], save_reply=None))
    orig = mcu_apply.time.monotonic
    # Skip the 5 s save wait: jump the clock once `config save` has been sent.
    t0 = [0.0]

    def fake_monotonic() -> float:
        if "config save" in ser.sent:
            t0[0] += 10.0
        return orig() + t0[0]

    mcu_apply.time.monotonic = fake_monotonic  # type: ignore[assignment]
    try:
        assert not apply_mcu(ser, resolved_config, strict=False, persist=True)
    finally:
        mcu_apply.time.monotonic = orig  # type: ignore[assignment]
    assert "never answered `config save`" in capsys.readouterr().err


def test_verify_flags_trigger_and_imu_mismatches(resolved_config):
    dump = _dump_for(resolved_config)
    dump["trigger"]["rate_hz"] = int(dump["trigger"]["rate_hz"]) * 2
    dump["trigger"]["pulse"]["width_us"] = 1
    name = next(iter(dump["imu_mta"]["reports"]))
    dump["imu_mta"]["reports"][name]["rate_hz"] = 7
    dump["imu_cra"]["reports"] = {"gyroscope": {"rate_hz": 100}}
    problems = verify_mcu_config(ScriptedSerial(lambda _l: ""), resolved_config, dump=dump)
    joined = "\n".join(problems)
    assert "trigger.rate_hz" in joined
    assert "trigger.pulse.width_us" in joined
    assert f"imu_mta.reports.{name}.rate_hz" in joined
    assert "imu_cra.reports: requested" in joined and "MCU also runs ['gyroscope']" in joined


def test_verify_clean_dump_has_no_problems(resolved_config):
    dump = _dump_for(resolved_config)
    assert verify_mcu_config(ScriptedSerial(lambda _l: ""), resolved_config, dump=dump) == []


def test_stream_start_reports_firmware_refusals(capsys):
    ser = ScriptedSerial(
        lambda line: "trig_start FAIL\r\n" if line == "trig_start" else (
            "unknown: imu_stream cra on\r\n" if line == "imu_stream cra on" else ""
        )
    )
    problems = start_mcu_streams(ser, trigger=False)
    assert problems == ["`imu_stream cra on`: unknown: imu_stream cra on"]
    assert start_mcu_trigger(ser) == ["`trig_start`: trig_start FAIL"]
    err = capsys.readouterr().err
    assert "trig_start FAIL" in err and "unknown: imu_stream" in err


def test_stream_start_busy_while_config_apply_runs_is_a_failure(capsys):
    """Firmware refuses stream starts mid-apply with `<cmd> BUSY` (cmd_parser.c)."""
    ser = ScriptedSerial(lambda line: f"{line.split()[0]} BUSY\r\n")
    problems = start_mcu_streams(ser)
    assert problems == [
        "`ads_start`: ads_start BUSY",
        "`imu_stream mta on`: imu_stream BUSY",
        "`imu_stream cra on`: imu_stream BUSY",
        "`trig_start`: trig_start BUSY",
    ]
    assert "imu_stream BUSY" in capsys.readouterr().err


def test_stream_start_quiet_firmware_is_ok():
    ser = ScriptedSerial(lambda _l: "")
    assert start_mcu_streams(ser) == []
    assert ser.sent == ["ads_start", "imu_stream mta on", "imu_stream cra on", "trig_start"]
