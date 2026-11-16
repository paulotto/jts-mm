# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Hdf5Writer: survives SIGKILL, always writes frame_idx, stores /emg/seq (K1)."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm.integrity.checker import check_session
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.sync.clock_align import AlignedRecord

_CHILD = """
import sys, time
from pathlib import Path
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.sync.clock_align import AlignedRecord
w = Hdf5Writer(path=Path(sys.argv[1]), config_hash='x',
               mcu_config={'emg': {'default_gain': 6}}, host_config={},
               file_flush_interval_s=0.2)
w.open()
i = 0
t0 = time.monotonic()
while True:
    for _ in range(100):
        w.append(AlignedRecord('emg', i * 500, {'channels': [i % 1000] * 8,
                                                'status': b'\\xc0\\0\\0'}))
        i += 1
    w.flush()
    if time.monotonic() - t0 > 1.0:
        print('ready', i, flush=True)
    time.sleep(0.01)
"""


def _killed_recording(path: Path) -> int:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]
    )
    p = subprocess.Popen([sys.executable, "-c", _CHILD, str(path)],
                         stdout=subprocess.PIPE, text=True, env=env)
    try:
        line = p.stdout.readline()
    finally:
        os.kill(p.pid, signal.SIGKILL)
        p.wait()
    assert line.startswith("ready"), line
    return int(line.split()[1])


@pytest.mark.skipif(shutil.which("h5clear") is None, reason="needs h5clear (hdf5-tools)")
def test_sigkilled_recorder_leaves_a_recoverable_file(tmp_path: Path):
    path = tmp_path / "killed.h5"
    written = _killed_recording(path)
    subprocess.run(["h5clear", "-s", str(path)], check=True)
    with h5py.File(path, "r") as f:
        n = f["emg"]["t_master_us"].shape[0]
        assert f["emg"]["samples"].shape[0] == n
        # everything up to roughly the last file flush (0.2 s) survives
        assert n >= written - 3000, (n, written)
        assert np.all(np.diff(np.asarray(f["emg"]["t_master_us"], dtype=np.int64)) > 0)


def test_checker_explains_an_unclosed_file(tmp_path: Path):
    path = tmp_path / "killed.h5"
    _killed_recording(path)
    report = check_session(path)
    assert not report.ok
    assert any("h5clear -s" in e for e in report.issues), report.issues


def _writer(path: Path) -> Hdf5Writer:
    w = Hdf5Writer(path=path, config_hash="x", mcu_config={"emg": {"default_gain": 6}},
                   host_config={})
    w.open()
    return w


def test_frame_idx_exists_even_when_the_first_block_is_all_zero(tmp_path: Path):
    path = tmp_path / "cam.h5"
    w = _writer(path)
    w.append(AlignedRecord("camera", 100, {"cam_id": "cam0", "sensor_timestamp_ns": 1,
                                           "trigger_id": 1, "frame_idx": 0}))
    w.flush()
    w.append(AlignedRecord("camera", 200, {"cam_id": "cam0", "sensor_timestamp_ns": 2,
                                           "trigger_id": 2, "frame_idx": 1}))
    w.close()
    with h5py.File(path, "r") as f:
        assert list(f["cameras"]["cam0"]["frame_idx"][:]) == [0, 1]


def test_emg_seq_written_when_present(tmp_path: Path):
    path = tmp_path / "seq.h5"
    w = _writer(path)
    for i in range(5):
        w.append(AlignedRecord("emg", 500 * i, {"channels": [0] * 8, "seq": 0xFFFFFFFE + i}))
    w.flush()
    for i in range(5, 8):
        w.append(AlignedRecord("emg", 500 * i, {"channels": [0] * 8, "seq": 0xFFFFFFFE + i}))
    w.close()
    with h5py.File(path, "r") as f:
        seq = np.asarray(f["emg"]["seq"])
        assert seq.dtype == np.uint32 and seq.size == f["emg"]["t_master_us"].shape[0]
        assert list(seq[:4]) == [0xFFFFFFFE, 0xFFFFFFFF, 0, 1]


def test_no_emg_seq_without_it(tmp_path: Path):
    path = tmp_path / "noseq.h5"
    w = _writer(path)
    w.append(AlignedRecord("emg", 0, {"channels": [0] * 8}))
    w.close()
    with h5py.File(path, "r") as f:
        assert "seq" not in f["emg"]
