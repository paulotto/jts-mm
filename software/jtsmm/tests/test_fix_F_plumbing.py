# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Config plumbing (package F rows): every flag/key reaches the payload or the stream start."""

from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from jtsmm import native
from jtsmm.cli.record import record_cmd
from jtsmm.config import mcu_apply
from jtsmm.config.mcu_apply import (
    _emg_to_wire,
    build_mcu_apply_payload,
    mcu_stream_enables,
    start_mcu_streams,
    verify_mcu_config,
)
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL


class _Ser:
    def __init__(self) -> None:
        self.sent: list[str] = []

    def write(self, data: bytes) -> None:
        self.sent.append(data.decode().strip())

    def flush(self) -> None:
        pass


def test_disabled_imu_slot_is_not_started(monkeypatch):
    monkeypatch.setattr(mcu_apply.time, "sleep", lambda *_a: None)
    ser = _Ser()
    start_mcu_streams(ser, emg=False, trigger=False, imu_cra=False)
    assert ser.sent == ["imu_stream mta on"]


def test_stream_enables_follow_mcu_enabled_flags(resolved_config):
    cfg = copy.deepcopy(resolved_config)
    assert mcu_stream_enables(cfg) == {"emg": True, "imu_mta": True, "imu_cra": True, "trigger": True}
    cfg["mcu"]["imu_cra"]["enabled"] = False
    cfg["mcu"]["trigger"]["enabled"] = False
    cfg["mcu"]["emg"]["enabled"] = False
    assert mcu_stream_enables(cfg) == {"emg": False, "imu_mta": True, "imu_cra": False, "trigger": False}


def test_explicit_pd_channels_pass_through_without_masking_bias():
    wire = _emg_to_wire({"pd_channels": 0x3F, "bias": {"enabled": True, "channels": [1, 2]}})
    assert wire["pd_channels"] == 0x3F
    assert wire["bias"]["channels"] == 0b11  # not masked to 0 by the power-down
    # active_channels keeps deriving both when no explicit mask is given
    wire = _emg_to_wire({"active_channels": [1, 2, 3], "bias": {"enabled": True, "channels": [1, 4]}})
    assert wire["pd_channels"] == 0b111000 and wire["bias"]["channels"] == 0b1
    # an empty list is (still) "no selection"
    assert "pd_channels" not in _emg_to_wire({"active_channels": []})


def _dump(resolved: dict) -> dict:
    p = build_mcu_apply_payload(resolved)
    d = copy.deepcopy(p)
    d["emg"]["profile"] = "custom"  # expected after host overrides
    d["platform"]["buffers"]["emg_ring"] = 65475  # rounded to whole records
    return d


def test_verify_compares_every_common_leaf(resolved_config):
    dump = _dump(resolved_config)
    assert verify_mcu_config(None, resolved_config, dump=dump) == []
    dump["platform"]["spi0"]["hz"] = 8_000_000
    dump["emg"]["leadoff"]["ac_tone"] = True
    dump["emg"]["test_signal"] = {"enabled": True}
    joined = "\n".join(verify_mcu_config(None, resolved_config, dump=dump))
    assert "platform.spi0.hz" in joined
    assert "emg.leadoff.ac_tone" in joined
    assert "emg.test_signal.enabled" in joined
    assert "profile" not in joined and "buffers" not in joined


# --- supervisor honours mcu.<block>.enabled --------------------------------


@pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native extension not built",
)
def test_supervisor_skips_disabled_streams(monkeypatch, tmp_path, resolved_config):
    from jtsmm.capture import supervisor as sup_mod
    from tests.test_fix_F_supervisor import FakeCam, FakeMcu

    starts: list[dict] = []
    trig: list[str] = []
    for key, val in (("JTSMM_NATIVE", "1"), ("JTSMM_SETTLE_MS", "0"),
                     ("JTSMM_STREAM_RESTART", "0"), ("JTSMM_PPS_SAMPLES", "0")):
        monkeypatch.setenv(key, val)
    monkeypatch.setattr(native, "NativeMcuCapture", lambda *a, **k: FakeMcu())
    monkeypatch.setattr(sup_mod, "start_mcu_streams", lambda _ser, **kw: starts.append(kw))
    monkeypatch.setattr(sup_mod, "start_mcu_trigger", lambda _ser: trig.append("trig_start"))
    for name in ("stop_mcu_streams", "stop_mcu_streams_robust", "_send_stats_request"):
        monkeypatch.setattr(sup_mod, name, lambda *_a, **_k: None)
    monkeypatch.setattr(sup_mod, "_FINAL_STATS_TIMEOUT_S", 0.05)
    monkeypatch.setattr(sup_mod, "open_hat_serial", lambda _p: contextlib.nullcontext())
    monkeypatch.setattr(sup_mod, "cameras_from_config", lambda _c, **_k: [FakeCam()])
    resolved_config["host"].setdefault("storage", {})["base_dir"] = str(tmp_path / "data")
    resolved_config["mcu"]["trigger"]["enabled"] = False
    resolved_config["mcu"]["imu_cra"]["enabled"] = False
    resolved_config["mcu"]["emg"]["enabled"] = False
    sup_mod.SessionSupervisor(resolved_config, port="/dev/null").run(duration_s=0.1)
    assert starts and starts[0]["emg"] is False and starts[0]["trigger"] is False
    assert starts[0]["imu_mta"] is True and starts[0]["imu_cra"] is False
    assert trig == []


# --- record CLI ----------------------------------------------------------------


def test_skip_apply_skips_the_emg_check(tmp_config_path: Path, tmp_path: Path, monkeypatch):
    called: list = []
    monkeypatch.setattr("jtsmm.emg.check.run_emg_check", lambda *a, **k: called.append(1))

    def fake_run(config_path, **kw):
        out = tmp_path / "s" / "session.h5"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")
        return out

    monkeypatch.setattr("jtsmm.cli.record.run_session", fake_run)
    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), *MOCK_RECORD_DISABLE_OPTICAL, "--duration", "1",
         "--no-doctor", "--no-integrity", "--skip-apply"],
    )
    assert result.exit_code == 0, result.output
    assert called == []
    assert "skipping the pre-recording EMG check because of --skip-apply" in result.output


@pytest.mark.parametrize(
    "flags",
    [["--no-bias-drive", "--bias-channels", "1"], ["--active-channels", "7"]],
)
def test_bad_emg_overrides_are_usage_errors(tmp_config_path: Path, flags):
    result = CliRunner().invoke(record_cmd, ["--config", str(tmp_config_path), "--mock", *flags])
    assert result.exit_code == 2, result.output  # click usage error, not a traceback
    assert "Invalid value" in result.output
    assert not isinstance(result.exception, ValueError)


def test_preflight_probes_the_port_given_on_the_cli(tmp_config_path: Path, monkeypatch):
    seen: dict = {}

    def fake_preflight(**kw):
        seen.update(kw)
        return []

    monkeypatch.setattr("jtsmm.cli.record.run_preflight", fake_preflight)
    result = CliRunner().invoke(
        record_cmd,
        ["--config", str(tmp_config_path), "--mock", *MOCK_RECORD_DISABLE_OPTICAL,
         "--duration", "0.1", "--no-integrity", "--no-emg-check", "--port", "/dev/ttyTEST9",
         "--gain", "12"],
    )
    assert result.exit_code == 0, result.output
    assert seen["resolved"]["hat"]["serial_port"] == "/dev/ttyTEST9"
    assert seen["resolved"]["mcu"]["emg"]["default_gain"] == 12
