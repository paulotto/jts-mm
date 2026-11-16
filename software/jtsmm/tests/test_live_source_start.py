# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""`LiveSource.start()` must force MCU streams off before `config apply`.

A prior `jtsmm view`/`record` that didn't shut down cleanly (killed, crashed, or
whose own stop silently failed -- see `LiveSource.stop()`) can leave the ADS1299
mid-stream. Sending `config apply` into that state races the DRDY-driven reads against
register writes on the same SPI bus and can wedge the MCU until power-cycled.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from jtsmm.viewer.hub import Hub
from jtsmm.viewer.sources import LiveSource


class _FakeSerial:
    def close(self) -> None:
        pass


class _FakeNativeCapture:
    calls: list[str] | None = None

    def __init__(self, port: str | None = None, **_kw) -> None:
        self.port = port

    def start(self) -> None:
        if _FakeNativeCapture.calls is not None:
            _FakeNativeCapture.calls.append("usb_open")

    def settle(self, *, quiet_ms: int = 250) -> None:
        pass

    def drain_records(self) -> list:
        return []

    def stop(self) -> None:
        pass


def test_live_source_start_stops_streams_before_apply(monkeypatch, tmp_config_path: Path):
    calls: list[str] = []

    monkeypatch.setattr("jtsmm.native.use_native", lambda: True)
    monkeypatch.setattr("jtsmm.capture.serial_util.find_hat_port", lambda _explicit: "/dev/fake_hat")
    monkeypatch.setattr("jtsmm.config.open_hat_serial", lambda _port: _FakeSerial())
    monkeypatch.setattr("jtsmm.config.stop_mcu_streams", lambda _ser: calls.append("stop"))
    monkeypatch.setattr("jtsmm.config.apply_mcu", lambda _ser, _resolved, **_kw: calls.append("apply") or True)
    monkeypatch.setattr("jtsmm.config.start_mcu_streams", lambda _ser, **_kw: calls.append("start"))
    monkeypatch.setattr("jtsmm.capture.native_backend.NativeCapture", _FakeNativeCapture)
    monkeypatch.setattr(_FakeNativeCapture, "calls", calls)
    monkeypatch.setattr("time.sleep", lambda _s: None)

    src = LiveSource(config_path=tmp_config_path)
    hub = src.make_hub()
    src.start(hub)
    try:
        # USB reader before the streams: libusb needs ~1.5 s, and EMG produced
        # meanwhile overflowed the MCU ring (~1000 emg_drops per viewer start).
        assert calls == ["stop", "apply", "usb_open", "start"]
    finally:
        src.stop()


def test_live_source_stop_warns_instead_of_swallowing(monkeypatch, tmp_config_path: Path, capsys):
    """A failed `stop_mcu_streams_robust` must be visible, not silently dropped."""
    src = LiveSource(config_path=tmp_config_path)
    src._serial_port = "/dev/fake_hat"
    src._cap = None
    src._thread = None

    def _boom(_port: str, **_kw) -> None:
        raise RuntimeError("port busy")

    monkeypatch.setattr("jtsmm.config.stop_mcu_streams_robust", _boom)
    src.stop()

    err = capsys.readouterr().err
    assert "/dev/fake_hat" in err
    assert "port busy" in err


def _order_env(monkeypatch, calls: list[str]) -> None:
    monkeypatch.setattr("jtsmm.native.use_native", lambda: True)
    monkeypatch.setattr("jtsmm.capture.serial_util.find_hat_port", lambda _explicit: "/dev/fake_hat")
    monkeypatch.setattr("jtsmm.config.open_hat_serial", lambda _port: _FakeSerial())
    monkeypatch.setattr("jtsmm.config.stop_mcu_streams", lambda _ser: None)
    monkeypatch.setattr("jtsmm.config.stop_mcu_streams_robust", lambda *_a, **_k: None)
    monkeypatch.setattr("jtsmm.config.apply_mcu", lambda _ser, _cfg, **_kw: True)
    monkeypatch.setattr("jtsmm.config.start_mcu_streams",
                        lambda _ser, **_kw: calls.append("start") or [])
    monkeypatch.setattr("jtsmm.capture.native_backend.NativeCapture", _FakeNativeCapture)
    monkeypatch.setattr(_FakeNativeCapture, "calls", calls)
    monkeypatch.setattr("time.sleep", lambda _s: None)


def test_emg_check_opens_usb_before_streams(monkeypatch, tmp_config_path: Path):
    import numpy as np

    from jtsmm.config import load
    from jtsmm.emg import check

    calls: list[str] = []
    _order_env(monkeypatch, calls)
    monkeypatch.setattr(check, "capture_emg_uv", lambda *_a, **_k: (
        np.zeros((10, 8)), [], np.arange(10, dtype=np.int64)))
    check._hardware_session(load(tmp_config_path), port=None, leadoff_on=False, duration_s=0.1)
    assert calls == ["usb_open", "start"]


def test_imu_rates_opens_usb_before_streams(monkeypatch, tmp_config_path: Path):
    from jtsmm.config import load
    from jtsmm.doctor import imu_rates

    calls: list[str] = []
    _order_env(monkeypatch, calls)
    monkeypatch.setattr(imu_rates, "measure_capture", lambda *_a, **_k: ({}, {}))
    imu_rates.run_case(port="/dev/fake_hat", resolved=load(tmp_config_path), duration_s=0.1)
    assert calls == ["usb_open", "start"]


def test_live_source_starts_only_enabled_streams(monkeypatch, tmp_config_path: Path, capsys):
    """Disabled IMU slots are not started (the firmware would refuse them with a warning)."""
    import yaml

    doc = yaml.safe_load(tmp_config_path.read_text())
    doc.setdefault("mcu", {}).setdefault("imu_mta", {})["enabled"] = False
    doc["mcu"].setdefault("imu_cra", {})["enabled"] = False
    tmp_config_path.write_text(yaml.safe_dump(doc))
    starts: list[dict] = []
    _order_env(monkeypatch, [])
    monkeypatch.setattr("jtsmm.config.start_mcu_streams", lambda _ser, **kw: starts.append(kw) or [])

    src = LiveSource(config_path=tmp_config_path)
    src.start(src.make_hub())
    try:
        assert starts == [{"emg": True, "trigger": True, "imu_mta": False, "imu_cra": False}]
        # The browser shows "MTA off" instead of a green chip, and the terminal says why.
        assert src.meta()["imu_enabled"] == {"mta": False, "cra": False}
        err = capsys.readouterr().err
        assert "IMU MTA disabled in the config (mcu.imu_mta.enabled: false)" in err
        assert "IMU CRA disabled" in err
    finally:
        src.stop()


def test_tap_meta_carries_imu_enables():
    from jtsmm.viewer.sources import TapSource

    tap = TapSource(object(), emg_fs=2000, emg_gain=6, imu_enabled={"mta": True, "cra": False})
    assert tap.meta()["imu_enabled"] == {"mta": True, "cra": False}
    assert "imu_enabled" not in TapSource(object(), emg_fs=2000, emg_gain=6).meta()
