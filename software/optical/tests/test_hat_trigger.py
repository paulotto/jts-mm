# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""HAT trigger arming for live optical commands (no hardware)."""

from __future__ import annotations

import importlib.util
from unittest.mock import patch

import pytest
from click.testing import CliRunner

import optical.sync.hat_trigger as ht

# The CLI imports calibration.multicam, which needs OpenCV.
needs_cv2 = pytest.mark.skipif(
    importlib.util.find_spec("cv2") is None, reason="optical CLI needs cv2"
)


@pytest.fixture(autouse=True)
def _reset_hat_trigger(monkeypatch):
    ht.reset_hat_trigger_state()
    # Default rate comes from jtsmm.yaml (mcu.trigger.rate_hz); pin it here so
    # the test does not depend on the operator's ~/jtsmm.yaml.
    monkeypatch.setattr(ht, "configured_trigger_rate_hz", lambda *a, **k: ht.DEFAULT_RATE_HZ)
    yield
    ht.reset_hat_trigger_state()


def test_invoke_jtsmm_trigger_runs_cli():
    fake = type("P", (), {"stdout": "trigger started (30 Hz)\n", "returncode": 0})()
    with (
        patch.object(ht, "_jtsmm_executable", return_value="/usr/bin/jtsmm"),
        patch("optical.sync.hat_trigger.subprocess.run", return_value=fake) as run,
    ):
        ht.invoke_jtsmm_trigger("--rate", "30")
    run.assert_called_once()
    assert run.call_args.args[0] == ["/usr/bin/jtsmm", "trigger", "--rate", "30"]


def test_hat_trigger_starts_configured_rate_and_stops():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake):
        with ht.hat_trigger():
            assert calls == [("--rate", "30")]
        assert calls == [("--rate", "30"), ("--stop",)]


def test_hat_trigger_stops_on_exception():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake):
        with pytest.raises(RuntimeError, match="boom"):
            with ht.hat_trigger():
                raise RuntimeError("boom")
    assert calls == [("--rate", "30"), ("--stop",)]


def test_hat_trigger_is_reentrant():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake):
        with ht.hat_trigger():
            with ht.hat_trigger():
                assert calls == [("--rate", "30")]
            assert calls == [("--rate", "30")]
        assert calls == [("--rate", "30"), ("--stop",)]


def test_hat_trigger_skip_env(monkeypatch):
    monkeypatch.setenv(ht.SKIP_ENV, "1")
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake):
        with ht.hat_trigger():
            pass
    assert calls == []


@needs_cv2
def test_cli_calibrate_uses_30hz_even_when_config_is_faster(monkeypatch):
    """Full-res calibrate overruns a 60 Hz period; the pulse stays at 30 Hz."""
    monkeypatch.setattr(ht, "configured_trigger_rate_hz", lambda *a, **k: 60)
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    from optical.cli.main import main

    runner = CliRunner()
    with (
        patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake),
        patch("optical.cli.calibrate.run_calibration") as run,
    ):
        result = runner.invoke(main, ["calibrate", "--mode", "extrinsics"])
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert calls == [("--rate", "30"), ("--stop",)]


@needs_cv2
def test_cli_calibrate_runs_trigger_around_command():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    from optical.cli.main import main

    runner = CliRunner()
    with (
        patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake),
        patch("optical.cli.calibrate.run_calibration") as run,
    ):
        result = runner.invoke(main, ["calibrate", "--mode", "intrinsics"])
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert calls == [("--rate", "30"), ("--stop",)]


@needs_cv2
def test_cli_calibrate_bundle_skips_trigger():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    from optical.cli.main import main

    runner = CliRunner()
    with (
        patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake),
        patch("optical.cli.calibrate.run_calibration") as run,
    ):
        result = runner.invoke(main, ["calibrate", "--mode", "bundle"])
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert calls == []


@needs_cv2
def test_cli_calibrate_stops_trigger_on_failure():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    from optical.cli.main import main

    runner = CliRunner()
    with (
        patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake),
        patch("optical.cli.calibrate.run_calibration", side_effect=RuntimeError("nope")),
    ):
        result = runner.invoke(main, ["calibrate", "--mode", "intrinsics"])
    assert result.exit_code != 0
    assert calls == [("--rate", "30"), ("--stop",)]


@needs_cv2
def test_cli_track_does_not_pre_arm_trigger():
    """track arms the pulse inside run_live, not from the CLI wrapper."""
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    from optical.cli.main import main

    runner = CliRunner()
    with (
        patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake),
        patch("optical.cli.track.run_live") as run,
    ):
        result = runner.invoke(main, ["track"])
    assert result.exit_code == 0, result.output
    run.assert_called_once()
    assert "sync_hat" not in run.call_args.kwargs
    assert calls == []


@needs_cv2
def test_cli_track_rejects_removed_sync_hat_flag():
    from optical.cli.main import main

    runner = CliRunner()
    with patch.object(ht, "invoke_jtsmm_trigger"):
        result = runner.invoke(main, ["track", "--sync-hat"])
    assert result.exit_code != 0
    combined = (result.output + str(result.exception)).lower()
    assert "no such option" in combined


@needs_cv2
def test_cli_markers_does_not_start_trigger():
    calls: list[tuple[str, ...]] = []

    def _fake(*args: str) -> None:
        calls.append(args)

    from optical.cli.main import main

    runner = CliRunner()
    with patch.object(ht, "invoke_jtsmm_trigger", side_effect=_fake):
        result = runner.invoke(main, ["markers", "--help"])
    assert result.exit_code == 0, result.output
    assert calls == []
