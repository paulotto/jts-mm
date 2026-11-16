# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""reproduce.sh runs the same command reproduce.json records."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from jtsmm.io.session_reproduce import RecordReproduceSpec, write_session_reproduce


def _script(tmp_path: Path, spec: RecordReproduceSpec) -> str:
    written = write_session_reproduce(tmp_path / "sess", {"session": {}}, spec)
    text = written["script"].read_text(encoding="utf-8")
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(written["script"])], check=True)
    return text


def _command_words(script: str) -> list[str]:
    lines = script.splitlines()
    i = lines.index("exec \\")
    words = []
    for line in lines[i + 1:]:
        tok = line.strip().rstrip("\\").strip()
        if tok.startswith("--config"):
            break
        words.append(tok)
    return words


def test_ct_calibrate_script_runs_ct_calibrate(tmp_path: Path):
    spec = RecordReproduceSpec(ct_calibrate=True, mock=True, no_integrity=True,
                               ct_calibration_n_poses=12)
    script = _script(tmp_path, spec)
    assert _command_words(script) == ["jtsmm", "ct-calibrate"]
    assert "record" not in _command_words(script)
    assert "--ct-calibration-n-poses" in script
    assert "--no-integrity" not in script  # ct-calibrate has no such option


def test_imu_calibrate_script_keeps_mode_record(tmp_path: Path):
    script = _script(tmp_path, RecordReproduceSpec(imu_calibrate=True, mock=True))
    assert _command_words(script) == ["jtsmm", "imu", "calibrate", "--mode", "record"]


def test_record_script_unchanged(tmp_path: Path):
    script = _script(tmp_path, RecordReproduceSpec(duration_s=3.0, mock=True))
    assert _command_words(script) == ["jtsmm", "record"]
    assert "--duration" in script


@pytest.mark.parametrize("flag", ["--no-integrity"])
def test_record_keeps_no_integrity(tmp_path: Path, flag: str):
    script = _script(tmp_path, RecordReproduceSpec(no_integrity=True))
    assert flag in script


# --- every `jtsmm record` option survives into reproduce.sh -----------------

# Options that intentionally have no RecordReproduceSpec field of their own.
_NOT_REPRODUCED = {
    "no_doctor": "alias of --skip-preflight (folded into skip_preflight)",
}
# Options only emitted together with another setting.
_CONTEXT = {
    "experiment_plan": {"use_experiment_plan": True},
    "write_jts_config": {"use_experiment_plan": True},
    "jts_config_path": {"use_experiment_plan": True},
    "ct_calibration_n_poses": {"ct_calibration": True},
    "ct_calibration_recordings_dir": {"persist_ct_calibration": True},
}
_SAMPLE = {
    "duration_s": 3.0, "port": "/dev/ttyACM0",
    "optical_config": "/x/o.yaml", "ct_calibration_n_poses": 7,
    "ct_tip_calibration": "/x/tip.json", "ct_calibration_recordings_dir": "/x/rec",
    "experiment_plan": "/x/plan.yaml", "jts_config_path": "/x/jts.yaml",
    "viewer_host": "0.0.0.0", "viewer_port": 9999, "bias_channels": "1-6",
    "active_channels": "1,2", "default_gain": 12, "subject": "s01", "protocol": "p",
    "meta_kv": ("a=b",),
}


def _record_params():
    from jtsmm.cli.record import record_cmd

    return [p for p in record_cmd.params if p.name]


def test_every_record_option_maps_to_a_reproduce_field():
    import dataclasses

    fields = {f.name for f in dataclasses.fields(RecordReproduceSpec)}
    missing = [p.name for p in _record_params()
               if p.name not in fields and p.name not in _NOT_REPRODUCED]
    assert not missing, f"add these to RecordReproduceSpec / build_record_argv: {missing}"


def test_every_reproducible_record_option_reaches_the_argv():
    import dataclasses

    from jtsmm.io.session_reproduce import build_record_argv

    defaults = {f.name: f.default for f in dataclasses.fields(RecordReproduceSpec)
                if f.default is not dataclasses.MISSING}
    for p in _record_params():
        if p.name in _NOT_REPRODUCED or p.name == "config_path":
            continue
        kw = dict(_CONTEXT.get(p.name, {}))
        if p.name in _SAMPLE:
            kw[p.name] = _SAMPLE[p.name]
        elif isinstance(defaults.get(p.name), bool):
            kw[p.name] = not defaults[p.name]
        else:  # tri-state bool flags default to None
            kw[p.name] = True
        argv = build_record_argv(RecordReproduceSpec(**kw), config_path="/x/jtsmm.yaml")
        opts = set(p.opts) | set(p.secondary_opts)
        assert opts & set(argv), f"{p.name}: none of {sorted(opts)} in {argv}"


def test_no_abort_and_force_emg_check_are_reproduced(tmp_path: Path):
    script = _script(tmp_path, RecordReproduceSpec(no_abort=True, force_emg_check=True))
    assert "--no-abort" in script and "--force-emg-check" in script
