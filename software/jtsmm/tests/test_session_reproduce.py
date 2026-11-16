# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Session reproduce bundle (config YAML + reproduce.sh)."""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import h5py
import yaml
from click.testing import CliRunner

from jtsmm.cli.record import record_cmd
from jtsmm.io.session_reproduce import (
    CONFIG_NAME,
    OPERATOR_CONFIG_NAME,
    REPRODUCE_JSON_NAME,
    REPRODUCE_SCRIPT_NAME,
    RecordReproduceSpec,
    build_record_argv,
    write_session_reproduce,
)
from tests.helpers import MOCK_RECORD_DISABLE_OPTICAL


def _session_dir_from_record_output(output: str) -> Path:
    out_line = next(line for line in output.splitlines() if line.startswith("Wrote session"))
    return Path(out_line.split("-> ", 1)[1].strip()).parent


def _venv_bin() -> Path:
    """Resolve the package venv bin/ even when ``sys.executable`` is a conda symlink."""
    local = Path(__file__).resolve().parents[1] / ".venv" / "bin"
    if (local / "jtsmm").is_file():
        return local
    # Pi install layout used by this rig.
    installed = Path("/home/pi/jtsmm/venv/bin")
    if (installed / "jtsmm").is_file():
        return installed
    # any other environment: the one running the tests (without resolving symlinks)
    running = Path(sys.executable).parent
    if (running / "jtsmm").is_file():
        return running
    return local


def _force_tmp_storage(tmp_config_path: Path, data_root: Path) -> None:
    """Point both storage knobs at ``data_root`` (template also sets paths.data_dir)."""
    doc = yaml.safe_load(tmp_config_path.read_text(encoding="utf-8"))
    doc.setdefault("host", {}).setdefault("storage", {})["base_dir"] = str(data_root)
    doc.setdefault("host", {}).setdefault("paths", {})["data_dir"] = str(data_root)
    tmp_config_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")


def _record_mock(
    tmp_config_path: Path,
    data_root: Path,
    *,
    duration: str = "0.1",
    gain: str = "12",
    subject: str = "P07",
    protocol: str = "motion_battery_v1",
    meta: str = "operator=jd",
) -> Path:
    _force_tmp_storage(tmp_config_path, data_root)
    data_root.mkdir(parents=True, exist_ok=True)
    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        [
            "--config",
            str(tmp_config_path),
            "--mock",
            "--duration",
            duration,
            "--no-doctor",
            "--no-integrity",
            "--no-emg-check",
            "--no-viewer",
            *MOCK_RECORD_DISABLE_OPTICAL,
            "--subject",
            subject,
            "--protocol",
            protocol,
            "--meta",
            meta,
            "--gain",
            gain,
            "--leadoff",
        ],
        env={**os.environ, "JTSMM_DATA_DIR": str(data_root)},
    )
    assert result.exit_code == 0, result.output
    session_dir = _session_dir_from_record_output(result.output)
    assert data_root in session_dir.parents or session_dir.parent == data_root
    return session_dir


def test_build_record_argv_includes_flags():
    spec = RecordReproduceSpec(
        duration_s=30.0,
        port="/dev/ttyACM0",
        mock=True,
        skip_preflight=True,
        no_integrity=True,
        viewer=False,
        leadoff=True,
        default_gain=12,
        subject="s01",
        protocol="p01",
        meta_kv=("operator=pm",),
        emg_check=False,
    )
    argv = build_record_argv(spec, config_path="/tmp/jtsmm.yaml")
    assert argv[:4] == ["jtsmm", "record", "--config", "/tmp/jtsmm.yaml"]
    assert "--duration" in argv and "30.0" in argv
    assert "--port" in argv and "/dev/ttyACM0" in argv
    assert "--mock" in argv
    assert "--skip-preflight" in argv
    assert "--no-integrity" in argv
    assert "--no-viewer" in argv
    assert "--leadoff" in argv
    assert "--gain" in argv and "12" in argv
    assert "--subject" in argv and "s01" in argv
    assert "--protocol" in argv and "p01" in argv
    assert "--meta" in argv and "operator=pm" in argv
    assert "--no-emg-check" in argv
    assert "--no-use-experiment-plan" in argv


def test_build_record_argv_imu_calibrate():
    spec = RecordReproduceSpec(
        imu_calibrate=True,
        mock=True,
        skip_preflight=True,
        optical_track=True,
        optical_config="/tmp/optical.yaml",
        use_experiment_plan=False,
        persist_imu_calibration=True,
        imu_calibration_recordings_dir="~/experiments/5_2_2_imu_optical_handeye/recordings",
    )
    argv = build_record_argv(spec, config_path="/tmp/jtsmm.yaml")
    assert argv[:6] == ["jtsmm", "imu", "calibrate", "--mode", "record", "--config"]
    assert "--optical-track" not in argv
    assert "--use-experiment-plan" not in argv
    assert "--optical-config" in argv
    assert "--mock" in argv
    assert "--skip-preflight" in argv
    assert "--persist-imu-calibration" in argv
    assert "--imu-calibration-recordings-dir" in argv


def test_build_record_argv_ct_calibrate():
    spec = RecordReproduceSpec(
        ct_calibrate=True,
        ct_calibration=True,
        ct_calibration_n_poses=12,
        ct_tip_calibration="/tmp/tip.json",
        persist_ct_calibration=True,
        ct_calibration_recordings_dir="~/experiments/5_2_1_pivot_calibration_repeatability/recordings",
        optical_config="/tmp/optical.yaml",
        mock=True,
    )
    argv = build_record_argv(spec, config_path="/tmp/jtsmm.yaml")
    assert argv[:3] == ["jtsmm", "ct-calibrate", "--config"]
    assert "--ct-calibration-n-poses" in argv and "12" in argv
    assert "--ct-tip-calibration" in argv
    assert "--persist-ct-calibration" in argv
    assert "--ct-calibration-recordings-dir" in argv
    assert "--optical-config" in argv
    assert "--duration" not in argv


def test_imu_calibrate_reproduce_strips_operator_experiment(tmp_path: Path):
    src = tmp_path / "operator.yaml"
    src.write_text("session:\n  mode: acquisition\n", encoding="utf-8")
    resolved = {
        "session": {
            "mode": "acquisition",
            "use_experiment_plan": True,
            "experiment": {"name": "ct_tip_calibration_23", "phases": ["rest", "open_close"]},
            "mvc": False,
            "ct_calibration": True,
        }
    }
    session_dir = tmp_path / "session_imu"
    spec = RecordReproduceSpec(
        argv=["jtsmm", "imu", "calibrate", "--mode", "record"],
        config_path=str(src),
        imu_calibrate=True,
        optical_track=True,
        use_experiment_plan=False,
        mvc=False,
        ct_calibration=False,
        write_jts_config=False,
    )
    written = write_session_reproduce(session_dir, resolved, spec)
    doc = yaml.safe_load(written["config"].read_text(encoding="utf-8"))
    assert doc["session"]["use_experiment_plan"] is False
    assert doc["session"]["experiment"] is None
    assert doc["session"]["experiment_plan"] is None
    assert doc["session"]["mvc"] is False
    assert doc["session"]["ct_calibration"] is False


def test_build_record_argv_mvc_mode():
    spec = RecordReproduceSpec(mvc=True, optical_track=True, use_experiment_plan=True)
    argv = build_record_argv(spec, config_path="/tmp/jtsmm.yaml")
    assert "--mvc" in argv
    assert "--no-use-experiment-plan" in argv
    assert "--use-experiment-plan" not in argv
    assert "--optical-track" not in argv


def test_build_record_argv_omits_duration_when_unbounded():
    spec = RecordReproduceSpec(use_experiment_plan=True)
    argv = build_record_argv(spec, config_path="/tmp/jtsmm.yaml")
    assert "--duration" not in argv
    assert "--use-experiment-plan" in argv


def test_write_session_reproduce_bundle(tmp_path: Path):
    src = tmp_path / "operator.yaml"
    src.write_text("session:\n  mode: acquisition\n  duration_s: null\n", encoding="utf-8")
    resolved = {
        "session": {"mode": "acquisition", "duration_s": None, "default_duration_s": 10.0, "meta": {}},
        "mcu": {"emg": {"default_gain": 6}},
        "host": {"storage": {"base_dir": str(tmp_path / "data")}},
    }
    session_dir = tmp_path / "session_x"
    spec = RecordReproduceSpec(
        argv=["jtsmm", "record", "--config", str(src), "--duration", "12.5", "--subject", "s01"],
        config_path=str(src),
        duration_s=12.5,
        subject="s01",
        meta_kv=("operator=jd",),
        mock=True,
        skip_preflight=True,
        no_integrity=True,
        emg_check=False,
    )
    written = write_session_reproduce(session_dir, resolved, spec)

    assert written["config"].name == CONFIG_NAME
    assert written["operator_config"].name == OPERATOR_CONFIG_NAME
    assert written["script"].name == REPRODUCE_SCRIPT_NAME
    assert written["json"].name == REPRODUCE_JSON_NAME

    cfg = yaml.safe_load(written["config"].read_text(encoding="utf-8"))
    assert cfg["session"]["duration_s"] == 12.5
    assert cfg["session"]["subject_id"] == "s01"
    assert cfg["session"]["meta"]["operator"] == "jd"
    assert written["operator_config"].read_text(encoding="utf-8") == src.read_text(encoding="utf-8")

    script = written["script"].read_text(encoding="utf-8")
    assert script.startswith("#!/usr/bin/env bash")
    assert f'--config "$HERE/{CONFIG_NAME}"' in script
    assert "--duration" in script and "12.5" in script
    assert "--subject" in script and "s01" in script
    assert os.stat(written["script"]).st_mode & stat.S_IXUSR

    meta = json.loads(written["json"].read_text(encoding="utf-8"))
    assert meta["original_argv"][0] == "jtsmm"
    assert meta["spec"]["duration_s"] == 12.5


def test_record_cmd_writes_reproduce_bundle(tmp_config_path: Path, tmp_path: Path):
    session_dir = _record_mock(tmp_config_path, tmp_path / "data")
    assert (session_dir / CONFIG_NAME).is_file()
    assert (session_dir / OPERATOR_CONFIG_NAME).is_file()
    assert (session_dir / REPRODUCE_SCRIPT_NAME).is_file()
    assert (session_dir / REPRODUCE_JSON_NAME).is_file()

    cfg = yaml.safe_load((session_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    assert cfg["session"]["duration_s"] == 0.1
    assert cfg["session"]["subject_id"] == "P07"
    assert int(cfg["mcu"]["emg"]["default_gain"]) == 12

    script = (session_dir / REPRODUCE_SCRIPT_NAME).read_text(encoding="utf-8")
    assert "--mock" in script
    assert "--gain" in script
    assert "12" in script


def test_reproduce_script_passes_bash_syntax_check(tmp_config_path: Path, tmp_path: Path):
    session_dir = _record_mock(tmp_config_path, tmp_path / "data")
    script = session_dir / REPRODUCE_SCRIPT_NAME
    proc = subprocess.run(
        ["bash", "-n", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_reproduce_json_argv_reinokes_matching_session(tmp_config_path: Path, tmp_path: Path):
    """Re-invoke via reproduce.json argv through CliRunner; settings must match."""
    data_root = tmp_path / "data"
    session_dir = _record_mock(
        tmp_config_path,
        data_root,
        duration="0.15",
        gain="24",
        subject="S99",
        protocol="proto_x",
        meta="operator=qa",
    )
    meta = json.loads((session_dir / REPRODUCE_JSON_NAME).read_text(encoding="utf-8"))
    reproduce_argv = meta["reproduce_argv"]
    assert reproduce_argv[0] == "jtsmm"
    assert reproduce_argv[1] == "record"

    # Drop the program name; CliRunner invokes the record subcommand directly.
    cli_args = reproduce_argv[2:]
    cfg_path = Path(cli_args[cli_args.index("--config") + 1])
    assert cfg_path.is_file()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert cfg["session"]["duration_s"] == 0.15
    assert cfg["session"]["subject_id"] == "S99"
    assert int(cfg["mcu"]["emg"]["default_gain"]) == 24

    runner = CliRunner()
    result = runner.invoke(
        record_cmd,
        cli_args,
        env={**os.environ, "JTSMM_DATA_DIR": str(data_root)},
    )
    assert result.exit_code == 0, result.output
    replay_dir = _session_dir_from_record_output(result.output)
    assert replay_dir != session_dir
    assert data_root in replay_dir.parents or replay_dir.parent == data_root

    with h5py.File(replay_dir / "session.h5", "r") as f:
        mcu_cfg = json.loads(f["meta"].attrs["mcu_config_json"])
        assert int(mcu_cfg["emg"]["default_gain"]) == 24
        assert f["meta"].attrs["subject_id"] == "S99"
        assert f["meta"].attrs["protocol"] == "proto_x"
        assert f["meta"].attrs["operator"] == "qa"
        assert abs(float(f["meta"].attrs["session_duration_s"]) - 0.15) < 2.0

    replay_cfg = yaml.safe_load((replay_dir / CONFIG_NAME).read_text(encoding="utf-8"))
    assert replay_cfg["session"]["duration_s"] == 0.15
    assert int(replay_cfg["mcu"]["emg"]["default_gain"]) == 24


def test_reproduce_sh_executes_and_creates_matching_session(tmp_config_path: Path, tmp_path: Path):
    """End-to-end: run the generated bash script with the venv ``jtsmm`` on PATH."""
    data_root = tmp_path / "data"
    session_dir = _record_mock(
        tmp_config_path,
        data_root,
        duration="0.12",
        gain="6",
        subject="REPRO",
        protocol="e2e",
        meta="operator=script",
    )
    script = session_dir / REPRODUCE_SCRIPT_NAME
    assert script.is_file()

    before = {p.name for p in data_root.glob("session_*") if p.is_dir()}

    jtsmm_bin = _venv_bin() / "jtsmm"
    assert jtsmm_bin.is_file(), f"missing venv jtsmm entry point: {jtsmm_bin}"
    env = os.environ.copy()
    env["PATH"] = str(_venv_bin()) + os.pathsep + env.get("PATH", "")
    env["JTSMM_DATA_DIR"] = str(data_root)

    proc = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert proc.returncode == 0, (
        f"reproduce.sh failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "Wrote session" in proc.stdout

    after = {p.name for p in data_root.glob("session_*") if p.is_dir()}
    created = after - before
    assert len(created) == 1, f"expected one new session, got {created}"
    replay_dir = data_root / next(iter(created))
    assert replay_dir != session_dir
    assert (replay_dir / "session.h5").is_file()
    assert (replay_dir / REPRODUCE_SCRIPT_NAME).is_file()

    with h5py.File(replay_dir / "session.h5", "r") as f:
        mcu_cfg = json.loads(f["meta"].attrs["mcu_config_json"])
        assert int(mcu_cfg["emg"]["default_gain"]) == 6
        assert f["meta"].attrs["subject_id"] == "REPRO"
        assert f["meta"].attrs["protocol"] == "e2e"
        assert f["meta"].attrs["operator"] == "script"


def test_reproduce_sh_uses_session_local_config_not_operator_path(
    tmp_config_path: Path, tmp_path: Path
):
    session_dir = _record_mock(tmp_config_path, tmp_path / "data")
    script = (session_dir / REPRODUCE_SCRIPT_NAME).read_text(encoding="utf-8")
    # Must bind to $HERE/jtsmm.yaml, not the original tmp_config_path.
    assert f'--config "$HERE/{CONFIG_NAME}"' in script
    assert str(tmp_config_path) not in script

    # Collapse continuations and substitute HERE so shlex can validate quoting.
    flat = script.replace("\\\n", " ")
    flat = re.sub(r'\$HERE', str(session_dir), flat)
    cmd = None
    for line in flat.splitlines():
        stripped = line.strip()
        if stripped.startswith("exec"):
            stripped = stripped[len("exec") :].strip()
        if stripped.startswith("jtsmm "):
            cmd = stripped
            break
    assert cmd is not None, script
    tokens = shlex.split(cmd)
    assert tokens[0] == "jtsmm"
    assert tokens[1] == "record"
    cfg_idx = tokens.index("--config")
    assert Path(tokens[cfg_idx + 1]) == session_dir / CONFIG_NAME
