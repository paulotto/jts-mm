# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Write per-session reproduce bundle (config YAML + shell script)."""

from __future__ import annotations

import json
import os
import shutil
import stat
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAME = "jtsmm.yaml"
OPERATOR_CONFIG_NAME = "jtsmm.operator.yaml"
REPRODUCE_SCRIPT_NAME = "reproduce.sh"
REPRODUCE_JSON_NAME = "reproduce.json"


@dataclass
class RecordReproduceSpec:
    """CLI / runtime settings needed to re-run ``jtsmm record`` for a session."""

    argv: list[str] = field(default_factory=list)
    config_path: str | None = None
    duration_s: float | None = None
    port: str | None = None
    mock: bool = False
    skip_apply: bool = False
    skip_preflight: bool = False
    no_integrity: bool = False
    viewer: bool | None = None
    viewer_host: str | None = None
    viewer_port: int | None = None
    bias_drive: bool | None = None
    bias_channels: str | None = None
    active_channels: str | None = None
    leadoff: bool | None = None
    default_gain: int | None = None
    subject: str | None = None
    protocol: str | None = None
    meta_kv: tuple[str, ...] = ()
    emg_check: bool = True
    optical_track: bool = False
    optical_config: str | None = None
    ct_calibration: bool = False
    ct_calibration_n_poses: int = 20
    ct_tip_calibration: str | None = None
    persist_ct_calibration: bool = False
    ct_calibration_recordings_dir: str | None = None
    experiment_plan: str | None = None
    use_experiment_plan: bool = False
    mvc: bool = False
    write_jts_config: bool = True
    jts_config_path: str | None = None
    imu_calibrate: bool = False
    persist_imu_calibration: bool = False
    imu_calibration_recordings_dir: str | None = None
    ct_calibrate: bool = False
    no_abort: bool = False
    force_emg_check: bool = False


def _shell_quote(value: str) -> str:
    """POSIX single-quote escaping for reproduce.sh."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _argv_before_config(argv: list[str]) -> list[str]:
    for i, token in enumerate(argv):
        if token == "--config":
            return argv[:i]
    return argv[:2]


def _argv_after_config(argv: list[str]) -> list[str]:
    for i, token in enumerate(argv):
        if token == "--config" and i + 1 < len(argv):
            return argv[i + 2 :]
    return argv[2:]


def _append_runtime_flags(
    args: list[str], spec: RecordReproduceSpec, *, integrity_flag: bool = True
) -> None:
    if spec.port:
        args.extend(["--port", str(spec.port)])
    if spec.mock:
        args.append("--mock")
    if spec.skip_apply:
        args.append("--skip-apply")
    if spec.skip_preflight:
        args.append("--skip-preflight")
    if spec.no_integrity and integrity_flag:
        args.append("--no-integrity")
    if spec.viewer is True:
        args.append("--viewer")
    elif spec.viewer is False:
        args.append("--no-viewer")
    if spec.viewer_host is not None:
        args.extend(["--viewer-host", str(spec.viewer_host)])
    if spec.viewer_port is not None:
        args.extend(["--viewer-port", str(int(spec.viewer_port))])


def build_record_argv(spec: RecordReproduceSpec, *, config_path: str) -> list[str]:
    """Build a CLI argv that recreates the capture settings."""
    if spec.imu_calibrate:
        args = ["jtsmm", "imu", "calibrate", "--mode", "record", "--config", config_path]
        _append_runtime_flags(args, spec)
        if spec.optical_config:
            args.extend(["--optical-config", str(spec.optical_config)])
        if spec.persist_imu_calibration:
            args.append("--persist-imu-calibration")
            if spec.imu_calibration_recordings_dir:
                args.extend(
                    [
                        "--imu-calibration-recordings-dir",
                        str(spec.imu_calibration_recordings_dir),
                    ]
                )
        return args
    if spec.ct_calibrate:
        args = ["jtsmm", "ct-calibrate", "--config", config_path]
        # ct-calibrate records no session to check, so it has no --no-integrity.
        _append_runtime_flags(args, spec, integrity_flag=False)
        if spec.optical_config:
            args.extend(["--optical-config", str(spec.optical_config)])
        if spec.ct_calibration_n_poses != 20:
            args.extend(["--ct-calibration-n-poses", str(int(spec.ct_calibration_n_poses))])
        if spec.ct_tip_calibration:
            args.extend(["--ct-tip-calibration", str(spec.ct_tip_calibration)])
        if spec.persist_ct_calibration:
            args.append("--persist-ct-calibration")
            if spec.ct_calibration_recordings_dir:
                args.extend(
                    ["--ct-calibration-recordings-dir", str(spec.ct_calibration_recordings_dir)]
                )
        return args
    args = ["jtsmm", "record", "--config", config_path]
    if spec.duration_s is not None:
        args.extend(["--duration", str(spec.duration_s)])
    _append_runtime_flags(args, spec)
    if spec.bias_drive is True:
        args.append("--bias-drive")
    elif spec.bias_drive is False:
        args.append("--no-bias-drive")
    if spec.bias_channels is not None:
        args.extend(["--bias-channels", str(spec.bias_channels)])
    if spec.active_channels is not None:
        args.extend(["--active-channels", str(spec.active_channels)])
    if spec.leadoff is True:
        args.append("--leadoff")
    elif spec.leadoff is False:
        args.append("--no-leadoff")
    if spec.default_gain is not None:
        args.extend(["--gain", str(int(spec.default_gain))])
    if spec.subject is not None:
        args.extend(["--subject", str(spec.subject)])
    if spec.protocol is not None:
        args.extend(["--protocol", str(spec.protocol)])
    for pair in spec.meta_kv:
        args.extend(["--meta", str(pair)])
    if not spec.emg_check:
        args.append("--no-emg-check")
    if spec.force_emg_check:
        args.append("--force-emg-check")
    if spec.no_abort:
        args.append("--no-abort")
    if spec.mvc:
        args.append("--mvc")
        args.append("--no-use-experiment-plan")
        return args
    if spec.optical_track:
        args.append("--optical-track")
    if spec.optical_config:
        args.extend(["--optical-config", str(spec.optical_config)])
    if spec.ct_calibration:
        args.append("--ct-calibration")
        if spec.ct_calibration_n_poses != 20:
            args.extend(["--ct-calibration-n-poses", str(int(spec.ct_calibration_n_poses))])
    if spec.ct_tip_calibration:
        args.extend(["--ct-tip-calibration", str(spec.ct_tip_calibration)])
    if spec.persist_ct_calibration:
        args.append("--persist-ct-calibration")
        if spec.ct_calibration_recordings_dir:
            args.extend(
                ["--ct-calibration-recordings-dir", str(spec.ct_calibration_recordings_dir)]
            )
    if spec.mvc:
        args.append("--mvc")
        args.append("--no-use-experiment-plan")
    elif spec.use_experiment_plan:
        args.append("--use-experiment-plan")
        if spec.experiment_plan:
            args.extend(["--experiment-plan", str(spec.experiment_plan)])
        if not spec.write_jts_config:
            args.append("--no-write-jts-config")
        if spec.jts_config_path:
            args.extend(["--jts-config", str(spec.jts_config_path)])
    else:
        args.append("--no-use-experiment-plan")
    return args


def _prepare_config_doc(resolved: dict[str, Any], spec: RecordReproduceSpec) -> dict[str, Any]:
    """Deep-copy resolved operator config and bake effective session fields."""
    doc = deepcopy(resolved)
    session = doc.setdefault("session", {})
    session["duration_s"] = None if spec.duration_s is None else float(spec.duration_s)
    if spec.subject is not None:
        session["subject_id"] = str(spec.subject)
    if spec.protocol is not None:
        session["protocol"] = str(spec.protocol)
    if spec.meta_kv:
        meta = session.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            session["meta"] = meta
        for pair in spec.meta_kv:
            if "=" not in pair:
                continue
            key, _, value = pair.partition("=")
            key = key.strip()
            if key:
                meta[key] = value
    session["optical_track"] = bool(spec.optical_track)
    session["optical_config"] = spec.optical_config
    session["ct_calibration"] = bool(spec.ct_calibration)
    session["ct_calibration_n_poses"] = int(spec.ct_calibration_n_poses)
    session["ct_tip_calibration"] = spec.ct_tip_calibration
    session["persist_ct_calibration"] = bool(spec.persist_ct_calibration)
    session["ct_calibration_recordings_dir"] = spec.ct_calibration_recordings_dir
    session["persist_imu_calibration"] = bool(spec.persist_imu_calibration)
    session["imu_calibration_recordings_dir"] = spec.imu_calibration_recordings_dir
    session["experiment_plan"] = spec.experiment_plan
    session["use_experiment_plan"] = bool(spec.use_experiment_plan)
    session["mvc"] = bool(spec.mvc)
    session["write_jts_config"] = bool(spec.write_jts_config)
    if spec.imu_calibrate or spec.ct_calibrate:
        # Dedicated cal commands must not replay the operator motion battery.
        session["experiment"] = None
        session["experiment_plan"] = None
        session["use_experiment_plan"] = False
        session["mvc"] = False
    return doc


def write_session_reproduce(
    session_directory: Path,
    resolved: dict[str, Any],
    spec: RecordReproduceSpec,
) -> dict[str, Path]:
    """Write ``jtsmm.yaml``, optional operator copy, ``reproduce.sh``, ``reproduce.json``.

    Returns paths of written files. Failures are the caller's to handle — this
    function raises on I/O errors.
    """
    session_directory = Path(session_directory)
    session_directory.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    config_path = session_directory / CONFIG_NAME
    doc = _prepare_config_doc(resolved, spec)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)
    written["config"] = config_path

    src = Path(spec.config_path).expanduser() if spec.config_path else None
    if src is not None and src.is_file():
        operator_path = session_directory / OPERATOR_CONFIG_NAME
        # Avoid copying onto itself when recording already used the session copy.
        if src.resolve() != operator_path.resolve() and src.resolve() != config_path.resolve():
            shutil.copy2(src, operator_path)
            written["operator_config"] = operator_path

    replay_argv = build_record_argv(spec, config_path=f"$HERE/{CONFIG_NAME}")
    # For the shell script, keep $HERE unquoted-expansion friendly.
    script_comment = (
        "jtsmm imu calibrate --mode record"
        if spec.imu_calibrate
        else "jtsmm ct-calibrate"
        if spec.ct_calibrate
        else "jtsmm record --mvc"
        if spec.mvc
        else "jtsmm record"
    )
    script_lines = [
        "#!/usr/bin/env bash",
        f"# Auto-generated by {script_comment} — re-run this session with the same settings.",
        "set -euo pipefail",
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        "",
    ]
    if spec.argv:
        script_lines.append("# Original invocation:")
        script_lines.append("# " + " ".join(_shell_quote(a) for a in spec.argv))
        script_lines.append("")
    script_lines.append("exec \\")
    # The command words are whatever build_record_argv put before --config
    # (jtsmm record / jtsmm ct-calibrate / jtsmm imu calibrate ...), so the
    # script can never run a different command than reproduce.json records.
    for word in _argv_before_config(replay_argv):
        script_lines.append(f"  {word} \\")
    rest = _argv_after_config(replay_argv)
    cfg_end = " \\" if rest else ""
    script_lines.append(f'  --config "$HERE/{CONFIG_NAME}"{cfg_end}')
    for i, arg in enumerate(rest):
        end = " \\" if i < len(rest) - 1 else ""
        script_lines.append(f"  {_shell_quote(arg)}{end}")
    script_lines.append("")

    script_path = session_directory / REPRODUCE_SCRIPT_NAME
    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    mode = script_path.stat().st_mode
    script_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    written["script"] = script_path

    meta = {
        "created_unix_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "original_argv": list(spec.argv),
        "reproduce_argv": build_record_argv(spec, config_path=str(config_path)),
        "spec": asdict(spec),
        "files": {k: v.name for k, v in written.items()},
        "cwd": os.getcwd(),
    }
    json_path = session_directory / REPRODUCE_JSON_NAME
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    written["json"] = json_path
    return written
