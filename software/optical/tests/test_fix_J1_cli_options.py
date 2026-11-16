# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Every `optical` CLI option reaches its consumer (effects table).

For each command the consumer (run_live, run_calibration, ...) is replaced by
a recorder, the CLI is invoked with one option set to a non-default value, and
the recorded call must carry that value. A new option without a row in EFFECTS
fails ``test_every_option_has_an_effect_row``. Without cv2 the modules that
need OpenCV at import time are stubbed, so this runs in the jtsmm venv too.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import optical.sync.hat_trigger as ht

_CV2_MODULES = (
    "optical.calibration.multicam",
    "optical.evaluation.run_marker_size_eval",
    "optical.evaluation.run_quality_size_eval",
)


class _AnyModule(types.ModuleType):
    """Stand-in for a cv2-dependent module: every attribute is a dummy."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return type(name, (), {"__init__": lambda self, *a, **k: None})


@pytest.fixture
def cli(monkeypatch):
    if importlib.util.find_spec("cv2") is None:
        for name in _CV2_MODULES:
            monkeypatch.setitem(sys.modules, name, _AnyModule(name))
        for name in [m for m in sys.modules if m.startswith("optical.cli") or m == "optical.tracking.run"]:
            monkeypatch.delitem(sys.modules, name)
    main = importlib.import_module("optical.cli.main").main
    monkeypatch.setattr(ht, "invoke_jtsmm_trigger", lambda *a: None)
    return main


@pytest.fixture
def files(tmp_path):
    cfg = tmp_path / "optical.yaml"
    cfg.write_text("tracker: {pose_method: pnp}\n")
    plan = tmp_path / "plan.json"
    plan.write_text("{}")
    frames = tmp_path / "frames"
    frames.mkdir()
    return {"cfg": cfg, "plan": plan, "frames": frames, "tmp": tmp_path}


# command -> (consumer attribute(s) on the command module, base argv)
COMMANDS = {
    "track": ("optical.cli.track", ["run_live"], []),
    "calibrate": ("optical.cli.calibrate", ["run_calibration"], ["--mode", "intrinsics"]),
    "replay": ("optical.cli.replay", ["run_replay", "run_compare"], ["--frames", "{frames}"]),
    "markers": ("optical.cli.markers", ["run_markers", "run_charuco"], ["--out-dir", "{tmp}"]),
    "eval": ("optical.cli.eval_cmd", ["run_marker_size_eval"], []),
    "eval-quality": ("optical.cli.eval_cmd", ["run_quality_size_eval"], []),
}

# command -> {param name: (argv, consumer, kwarg, expected)}
EFFECTS = {
    "track": {
        "config_path": (["--config", "{cfg}"], "run_live", "tracker_config_path", "{cfg}"),
        "experiment_plan_path": (["--experiment-plan", "{plan}"], "run_live", "experiment_plan_path", "{plan}"),
        "use_experiment_plan": (["--no-use-experiment-plan"], "run_live", "use_experiment_plan", False),
        "backend": (["--backend", "V4L2"], "run_live", "camera_backend", "v4l2"),
        "output_path": (["--output", "o.h5"], "run_live", "output_path", Path("o.h5")),
        "jts_config_path": (["--jts-config", "j.json"], "run_live", "jts_config_path", Path("j.json")),
        "record_frames_dir": (["--record-frames", "rf"], "run_live", "record_frames_dir", Path("rf")),
        "persist_frames": (["--persist-frames"], "run_live", "persist_frames", True),
        "condition": (["--condition", "near"], "run_live", "condition", "near"),
        "preview_host": (["--preview-host", "127.0.0.1"], "run_live", "preview_host", "127.0.0.1"),
        "preview_port": (["--preview-port", "5123"], "run_live", "preview_port", 5123),
        "hat_port": (["--hat-port", "/dev/ttyX"], "run_live", "hat_port", "/dev/ttyX"),
        "trigger_rate": (["--trigger-rate", "45"], "run_live", "trigger_rate_hz", 45),
    },
    "calibrate": {
        "config_path": (["--config", "{cfg}"], "run_calibration", "config_path", "{cfg}"),
        "mode": (["--mode", "EXTRINSICS"], "run_calibration", "mode", "extrinsics"),
        "backend": (["--backend", "v4l2"], "run_calibration", "backend", "v4l2"),
        "preview_host": (["--preview-host", "127.0.0.1"], "run_calibration", "preview_host", "127.0.0.1"),
        "preview_port": (["--preview-port", "5123"], "run_calibration", "preview_port", 5123),
    },
    "replay": {
        "config_path": (["--config", "{cfg}"], "run_replay", "tracker_config_path", "{cfg}"),
        "frames_dir": (["--frames", "{frames}"], "run_replay", "frames_dir", "{frames}"),
        "output_path": (["--output", "o.h5"], "run_replay", "output_path", Path("o.h5")),
        "experiment_name": (["--experiment-name", "x"], "run_replay", "experiment_name", "x"),
        "print_every": (["--print-every", "7"], "run_replay", "print_every", 7),
        "verbose": (["--verbose"], "run_replay", "verbose", True),
        "max_frames": (["--max-frames", "9"], "run_replay", "max_frames", 9),
        "compare_methods": (["--compare-methods"], "run_compare", "frames_dir", "{frames}"),
        "debug_only": (["--debug-only"], "run_replay", "debug_only", True),
    },
    "markers": {
        "charuco": (["--charuco", "--out", "b.png"], "run_charuco", "out", Path("b.png")),
        "marker_config": (["--marker-config", "{cfg}"], "run_markers", "marker_config", "{cfg}"),
        "out_dir": (["--out-dir", "od"], "run_markers", "out_dir", Path("od")),
        "side_px": (["--side-px", "300"], "run_markers", "side_px", 300),
        "ids": (["--ids", "3", "--ids", "5"], "run_markers", "ids", [3, 5]),
        "out_path": (["--charuco", "--out", "c.png"], "run_charuco", "out", Path("c.png")),
        "squares_x": (["--charuco", "--out", "c.png", "--squares-x", "9"], "run_charuco", "squares_x", 9),
        "squares_y": (["--charuco", "--out", "c.png", "--squares-y", "6"], "run_charuco", "squares_y", 6),
        "square_mm": (["--charuco", "--out", "c.png", "--square-mm", "25"], "run_charuco", "square_mm", 25.0),
        "marker_mm": (["--charuco", "--out", "c.png", "--marker-mm", "18"], "run_charuco", "marker_mm", 18.0),
        "dictionary": (["--charuco", "--out", "c.png", "--dictionary", "DICT_5X5_50"], "run_charuco", "dictionary", "DICT_5X5_50"),
        "dpi": (["--charuco", "--out", "c.png", "--dpi", "150"], "run_charuco", "dpi", 150),
        "margin_mm": (["--charuco", "--out", "c.png", "--margin-mm", "8"], "run_charuco", "margin_mm", 8.0),
    },
    "eval": {
        "config_path": (["--config", "{cfg}"], "run_marker_size_eval", "config", "{cfg}"),
        "backend": (["--backend", "v4l2"], "run_marker_size_eval", "backend", "v4l2"),
        "preview_port": (["--preview-port", "0"], "run_marker_size_eval", "preview_port", 0),
        "preview_host": (["--preview-host", "127.0.0.1"], "run_marker_size_eval", "preview_host", "127.0.0.1"),
        "output_path": (["--output", "r.json"], "run_marker_size_eval", "output", Path("r.json")),
        "skip_motion": (["--skip-motion"], "run_marker_size_eval", "skip_motion", True),
        "resolution": (["--resolution", "728x544"], "run_marker_size_eval", "resolution", "728x544"),
        "cameras": (["--cameras", "cam0,cam1"], "run_marker_size_eval", "cameras", "cam0,cam1"),
        "n_cameras": (["--n-cameras", "2"], "run_marker_size_eval", "n_cameras", 2),
    },
    "eval-quality": {
        "config_path": (["--config", "{cfg}"], "run_quality_size_eval", "config", "{cfg}"),
        "backend": (["--backend", "v4l2"], "run_quality_size_eval", "backend", "v4l2"),
        "preview_port": (["--preview-port", "0"], "run_quality_size_eval", "preview_port", 0),
        "preview_host": (["--preview-host", "127.0.0.1"], "run_quality_size_eval", "preview_host", "127.0.0.1"),
        "output_path": (["--output", "r.json"], "run_quality_size_eval", "output", Path("r.json")),
        "frames_dir": (["--frames-dir", "{frames}"], "run_quality_size_eval", "frames_dir", "{frames}"),
        "save_frames": (["--save-frames"], "run_quality_size_eval", "save_frames", True),
    },
}


def _fmt(value, files):
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        return files[value[1:-1]]
    return value


def test_every_option_has_an_effect_row(cli):
    for name, cmd in cli.commands.items():
        opts = {p.name for p in cmd.params if isinstance(p, click.Option) and p.name != "help"}
        assert name in EFFECTS, f"new command {name!r}: add it to EFFECTS"
        assert opts == set(EFFECTS[name]), f"{name}: options without a row: {opts ^ set(EFFECTS[name])}"


@pytest.mark.parametrize(
    "command,param", [(c, p) for c, rows in EFFECTS.items() for p in rows]
)
def test_option_reaches_consumer(cli, files, monkeypatch, command, param):
    mod_name, consumers, base = COMMANDS[command]
    mod = importlib.import_module(mod_name)
    calls: dict = {}
    for consumer in consumers:
        monkeypatch.setattr(mod, consumer, lambda _c=consumer, **kw: calls.setdefault(_c, kw))
    argv, consumer, kwarg, expected = EFFECTS[command][param]
    base_argv = [str(_fmt(a, files)) for a in base]
    if "--charuco" in argv or "--marker-config" in argv or "--out-dir" in argv:
        base_argv = []  # markers: the row sets up its own mode
    if command == "replay" and "--frames" in argv:
        base_argv = []
    args = [command, *base_argv, *[str(_fmt(a, files)) for a in argv]]
    if command == "markers" and consumer == "run_markers" and "--out-dir" not in argv:
        args += ["--out-dir", str(files["tmp"])]
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 0, (args, result.output, result.exception)
    assert consumer in calls, (args, calls)
    want = _fmt(expected, files)
    assert calls[consumer][kwarg] == want, (args, calls[consumer])


def test_track_rejects_hat_options_with_v4l2(cli, monkeypatch):
    import optical.cli.track as track

    monkeypatch.setattr(track, "run_live", lambda **kw: None)
    for extra in (["--hat-port", "/dev/x"], ["--trigger-rate", "30"]):
        result = CliRunner().invoke(cli, ["track", "--backend", "v4l2", *extra])
        assert result.exit_code != 0 and "picamera2" in result.output


def test_markers_rejects_options_of_the_other_mode(cli, monkeypatch, tmp_path):
    import optical.cli.markers as markers

    monkeypatch.setattr(markers, "run_markers", lambda **kw: None)
    monkeypatch.setattr(markers, "run_charuco", lambda **kw: None)
    r = CliRunner().invoke(cli, ["markers", "--charuco", "--out", "b.png", "--side-px", "10"])
    assert r.exit_code != 0 and "--side-px" in r.output
    r = CliRunner().invoke(cli, ["markers", "--out-dir", str(tmp_path), "--dpi", "150"])
    assert r.exit_code != 0 and "--dpi" in r.output


def test_replay_compare_rejects_experiment_name(cli, monkeypatch, tmp_path):
    import optical.cli.replay as replay

    monkeypatch.setattr(replay, "run_compare", lambda **kw: None)
    r = CliRunner().invoke(
        cli, ["replay", "--frames", str(tmp_path), "--compare-methods", "--experiment-name", "x"]
    )
    assert r.exit_code != 0


def test_track_trigger_context_forwards_port_and_rate(monkeypatch):
    """run_live's HAT trigger gets --hat-port and --trigger-rate (picamera2 only)."""
    if importlib.util.find_spec("cv2") is None:
        for name in _CV2_MODULES:
            monkeypatch.setitem(sys.modules, name, _AnyModule(name))
        monkeypatch.delitem(sys.modules, "optical.tracking.run", raising=False)
    run = importlib.import_module("optical.tracking.run")
    calls = []
    monkeypatch.setattr(ht, "invoke_jtsmm_trigger", lambda *a: calls.append(a))
    ht.reset_hat_trigger_state()
    with run._trigger_context("picamera2", hat_port="/dev/ttyACM7", rate_hz=45):
        assert calls == []  # deferred until the cameras run
        ht.arm_hat_trigger()
    assert calls == [("--rate", "45", "--port", "/dev/ttyACM7"), ("--stop", "--port", "/dev/ttyACM7")]
    calls.clear()
    with run._trigger_context("v4l2", hat_port="/dev/x", rate_hz=45):
        pass
    assert calls == []
