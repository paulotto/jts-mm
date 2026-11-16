# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CLI > YAML > default for `jtsmm emg`, `jtsmm view/replay`; check-config wiring."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import yaml
from click.testing import CliRunner

import jtsmm.cli.emg_cmd as emg_cmd
import jtsmm.emg.check as check_mod
from jtsmm.config.mcu_apply import build_mcu_apply_payload


def _cfg_file(tmp_path: Path, tmp_config_path: Path, **emg_check) -> Path:
    doc = yaml.safe_load(tmp_config_path.read_text())
    doc.setdefault("host", {}).setdefault("emg_check", {}).update(emg_check)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def _capture_calls(monkeypatch):
    calls: dict = {}

    def fake_check(resolved, **kw):
        calls["check"] = (resolved, kw)
        return check_mod.CheckResult([], True)

    def fake_imp(resolved, **kw):
        calls["imp"] = (resolved, kw)
        return [], {"tone_hz": 31.25, "fs_hz": 250.0, "current_na": 6, "n_samples": 0}

    monkeypatch.setattr(emg_cmd, "run_emg_check", fake_check)
    monkeypatch.setattr(emg_cmd, "run_impedance", fake_imp)
    monkeypatch.setattr(check_mod, "restore_mcu_config",
                        lambda r, port=None: calls.setdefault("restored", r) and None)
    return calls


def test_emg_check_leaves_unset_options_to_the_yaml(monkeypatch, tmp_path, tmp_config_path):
    calls = _capture_calls(monkeypatch)
    cfg = _cfg_file(tmp_path, tmp_config_path, line_hz=60.0)
    r = CliRunner().invoke(emg_cmd.emg_group, ["check", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    resolved, kw = calls["check"]
    assert kw["line_hz"] is None and kw["duration_s"] is None and kw["with_impedance"] is None
    assert resolved["host"]["emg_check"]["line_hz"] == 60.0     # run_emg_check reads it
    assert "restored" in calls                                  # MCU config restored


def test_emg_check_cli_overrides_win(monkeypatch, tmp_path, tmp_config_path):
    calls = _capture_calls(monkeypatch)
    cfg = _cfg_file(tmp_path, tmp_config_path, line_hz=60.0)
    r = CliRunner().invoke(emg_cmd.emg_group, [
        "check", "--config", str(cfg), "--line-hz", "50", "--duration", "3",
        "--no-impedance", "--gain", "12", "--no-bias-drive",
    ])
    assert r.exit_code == 0, r.output
    resolved, kw = calls["check"]
    assert kw["line_hz"] == 50.0 and kw["duration_s"] == 3.0 and kw["with_impedance"] is False
    assert resolved["mcu"]["emg"]["default_gain"] == 12
    assert resolved["mcu"]["emg"]["bias"]["enabled"] is False


def test_emg_impedance_duration_defaults_to_yaml(monkeypatch, tmp_path, tmp_config_path):
    calls = _capture_calls(monkeypatch)
    cfg = _cfg_file(tmp_path, tmp_config_path, impedance={"enabled": True, "duration_s": 4.5})
    r = CliRunner().invoke(emg_cmd.emg_group, ["impedance", "--config", str(cfg)])
    assert r.exit_code == 0, r.output
    assert calls["imp"][1]["duration_s"] == 4.5


def test_run_emg_check_uses_yaml_line_hz_when_not_given(monkeypatch):
    def _session(*_a, **_k):
        return check_mod._Capture(np.zeros((1000, 8), np.float32), [],
                                  {"sample_rate_sps": 2000.0}, None)

    monkeypatch.setattr(check_mod, "_hardware_session", _session)
    res = check_mod.run_emg_check(
        {"host": {"emg_check": {"line_hz": 60.0}}, "mcu": {"emg": {"bias": {"enabled": False}}}},
        port="/dev/null", with_impedance=False,
    )
    assert res.line_hz_measured == 60.0


def test_drl_probe_really_powers_down_ch1_to_ch6():
    """`active_channels: []` meant 'no selection' to the payload builder."""
    cfg = check_mod._check_config({"mcu": {"emg": {"active_channels": [1, 2]}}},
                                  leadoff_on=True, mute_muscle=True, probe_drl=True)
    wire = build_mcu_apply_payload(cfg)["emg"]
    assert wire["pd_channels"] == 0x3F


def test_viewer_address_precedence(tmp_path, tmp_config_path):
    from jtsmm.cli.view import _viewer_address

    doc = yaml.safe_load(tmp_config_path.read_text())
    doc.setdefault("host", {})["viewer"] = {"enabled": False, "bind_host": "127.0.0.1", "port": 9000}
    cfg = tmp_path / "v.yaml"
    cfg.write_text(yaml.safe_dump(doc))
    assert _viewer_address(cfg, None, None) == ("127.0.0.1", 9000)
    assert _viewer_address(cfg, "0.0.0.0", 8770) == ("0.0.0.0", 8770)   # explicit default wins
    assert _viewer_address(cfg, None, 8771) == ("127.0.0.1", 8771)


def test_replay_uses_the_sessions_mains_mode(tmp_path):
    from jtsmm.viewer.sources import ReplaySource
    from tests.test_viewer import _write_session

    p = tmp_path / "s.h5"
    _write_session(p)
    with h5py.File(p, "a") as f:
        f["meta"].attrs["host_config_json"] = json.dumps(
            {"host": {"emg_filter": {"mains_mode": "notch"}}}
        )
    src = ReplaySource(p)
    try:
        assert src._mains_mode == "notch"
    finally:
        src.stop()
