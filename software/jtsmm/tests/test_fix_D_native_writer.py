# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Native HDF5 writer robustness: crash recovery, disk-full, retry without duplicates."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm import native

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native HDF5 writer not built",
)

_PKG_ROOT = str(Path(native.__file__).resolve().parents[2])
_EMG_DS = ("samples", "samples_uv", "t_master_us", "leadoff_status")


def _run_py(code: str, *, timeout: float = 60.0, **kw) -> subprocess.Popen:
    env = dict(os.environ, PYTHONPATH=_PKG_ROOT)
    return subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **kw,
    )


def _emg_lengths(path: Path) -> dict[str, int]:
    with h5py.File(path, "r") as f:
        return {k: int(f["emg"][k].shape[0]) for k in _EMG_DS if k in f["emg"]}


def test_sigkilled_writer_leaves_recoverable_file(tmp_path: Path):
    """A SIGKILL after a few flushes must leave every flushed row recoverable.

    Before H5Fflush was called the file was unreadable even after `h5clear -s`.
    """
    h5clear = shutil.which("h5clear")
    if h5clear is None:
        pytest.skip("h5clear (hdf5-tools) not installed")
    path = tmp_path / "killed.h5"
    proc = _run_py(
        f"""
        import sys, time
        from jtsmm import native
        w = native.NativeHdf5Writer(path={str(path)!r}, config_hash="x", disk_flush_interval_s=0.2)
        w.open()
        n = 0
        for block in range(4):
            for i in range(500):
                w.append_emg(n, [n, -n, 1, 2, 3, 4, 5, 6])
                w.append_trigger(n, n)
                n += 1
            time.sleep(0.25)      # > disk_flush_interval_s: this flush reaches the disk
            w.flush()
        print("flushed", n, flush=True)
        time.sleep(60)
        """
    )
    try:
        line = proc.stdout.readline()
        assert line.startswith("flushed"), proc.stderr.read()
        n = int(line.split()[1])
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    subprocess.run([h5clear, "-s", str(path)], check=True, capture_output=True)
    with h5py.File(path, "r") as f:
        assert f["emg"]["samples"].shape == (n, 8)
        assert np.array_equal(f["emg"]["t_master_us"][:], np.arange(n, dtype=np.uint64))
        assert f["triggers"]["trigger_id"].shape == (n,)
    assert set(_emg_lengths(path).values()) == {n}


def test_partial_flush_retry_does_not_duplicate_rows(tmp_path: Path):
    """samples written, samples_uv failed -> the retry must not re-append samples."""
    path = tmp_path / "retry.h5"
    w = native.NativeHdf5Writer(path=str(path), config_hash="x")
    w.open()
    for i in range(100):
        w.append_emg(i, [i] * 8)
    w._debug_fail_nth_append(2)  # 1st append (samples) lands, 2nd (samples_uv) fails
    with pytest.raises(RuntimeError, match="injected"):
        w.flush()
    for i in range(100, 150):  # rows keep arriving between the failure and the retry
        w.append_emg(i, [i] * 8)
    w.flush()
    w.close()
    assert set(_emg_lengths(path).values()) == {150}
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["emg"]["t_master_us"][:], np.arange(150, dtype=np.uint64))
        assert np.array_equal(f["emg"]["samples"][:, 0], np.arange(150, dtype=np.int32))


def test_failed_close_still_releases_file_and_raises(tmp_path: Path):
    path = tmp_path / "close_fail.h5"
    w = native.NativeHdf5Writer(path=str(path), config_hash="x")
    w.open()
    w.append_emg(1, [1] * 8)
    w._debug_fail_nth_append(1)
    with pytest.raises(RuntimeError):
        w.close()
    w.close()  # already closed: no-op, no second error
    del w
    with h5py.File(path, "r") as f:  # released and structurally valid
        assert "meta" in f and "session_duration_s" in f["meta"].attrs


def test_attrs_are_replaced_not_silently_dropped(tmp_path: Path):
    path = tmp_path / "attrs.h5"
    w = native.NativeHdf5Writer(
        path=str(path), config_hash="real", extra_meta={"config_hash": "override", "subject": "P1"}
    )
    w.open()
    w.set_sync_report('{"n": 1}')
    w.set_sync_report('{"n": 2}')  # H5Acreate2 used to fail here and keep the first one
    w.close()
    with h5py.File(path, "r") as f:
        assert f["sync"].attrs["report_json"] == '{"n": 2}'
        # extra_meta is written after the built-ins and wins, as in the Python writer.
        assert f["meta"].attrs["config_hash"] == "override"
        assert f["meta"].attrs["subject"] == "P1"


def test_disk_full_does_not_terminate_and_keeps_datasets_aligned(tmp_path: Path):
    """RLIMIT_FSIZE makes writes fail with EFBIG like a full disk. The writer must
    raise (not std::terminate from the destructor), keep rows for retry, and after
    space returns finish with equal-length datasets and no duplicated rows."""
    path = tmp_path / "full.h5"
    proc = _run_py(
        f"""
        import resource, signal, sys
        from jtsmm import native
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)   # get EFBIG instead of a kill
        soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
        w = native.NativeHdf5Writer(path={str(path)!r}, config_hash="x", disk_flush_interval_s=0.0)
        w.open()
        resource.setrlimit(resource.RLIMIT_FSIZE, (256 * 1024, hard))
        n, failures = 0, 0
        for block in range(200):
            for i in range(1000):
                w.append_emg(n, [n] * 8)
                n += 1
            try:
                w.flush()
            except RuntimeError:
                failures += 1
                break
        # "Disk" frees up again: the retry must write each row exactly once.
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        retry_ok = True
        try:
            w.flush()
            w.close()
        except RuntimeError:
            retry_ok = False
        print("result", n, failures, retry_ok, flush=True)
        # Destructor of a failed/closed writer must never throw.
        w2 = native.NativeHdf5Writer(path={str(tmp_path / "full2.h5")!r}, config_hash="x")
        w2.open()
        w2.append_emg(0, [0] * 8)
        resource.setrlimit(resource.RLIMIT_FSIZE, (4096, hard))
        del w2
        print("done", flush=True)
        """
    )
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, f"rc={proc.returncode} (SIGABRT=-6 means std::terminate)\n{err}"
    assert "done" in out, err
    _, n, failures, retry_ok = next(l for l in out.splitlines() if l.startswith("result")).split()
    assert int(failures) == 1, "RLIMIT_FSIZE never hit; raise the row count"
    if retry_ok == "True":
        assert set(_emg_lengths(path).values()) == {int(n)}
        with h5py.File(path, "r") as f:
            t = f["emg"]["t_master_us"][:]
            assert np.array_equal(t, np.arange(int(n), dtype=np.uint64))


def _native_writer(path: Path, mcu_config: dict):
    from jtsmm.capture.native_backend import NativeWriter

    return NativeWriter(path=path, config_hash="x", mcu_config=mcu_config, host_config={})


def test_float_trigger_rate_seq_per_channel_lsb_and_imu_attrs(tmp_path: Path):
    from jtsmm.config.catalog import channel_lsb_uv, lsb_uv

    path = tmp_path / "features.h5"
    mcu = {
        "emg": {"default_gain": 12, "sample_rate_sps": 2000},
        "trigger": {"rate_hz": 29.97},  # cast<int>() used to throw in the constructor
        "imu_mta": {"reports": {"accelerometer": {"rate_hz": 100},
                                "gyroscope": {"rate_hz": 200, "enabled": False}}},
    }
    w = _native_writer(path, mcu)
    w.open()
    raw = w._writer
    for i in range(10):
        raw.append_emg(i, [1000 * (c + 1) for c in range(8)], seq=0xFFFFFFFB + i if i < 5 else i - 5)
    raw.append_quat(False, 5, [1.0, 0.0, 0.0, 0.0], 0x08)
    raw.append_imu_motion(False, 0x01, 6, [0.0, 0.0, 9.81])
    raw.append_imu_motion(False, 0x02, 7, [0.1, 0.0, 0.0])
    raw.append_trigger(8, 1)
    w.close()
    with h5py.File(path, "r") as f:
        rate = f["triggers"].attrs["rate_hz"]
        assert isinstance(rate, np.floating) and float(rate) == pytest.approx(29.97)
        seq = f["emg"]["seq"]
        assert seq.dtype == np.uint32 and seq.shape == f["emg"]["t_master_us"].shape
        assert list(seq[:6]) == [0xFFFFFFFB, 0xFFFFFFFC, 0xFFFFFFFD, 0xFFFFFFFE, 0xFFFFFFFF, 0]
        lsb = channel_lsb_uv(12)
        assert np.allclose(f["emg"].attrs["lsb_uv_per_channel"], lsb)
        assert float(f["emg"].attrs["lsb_uv"]) == pytest.approx(lsb_uv(12))
        uv = f["emg"]["samples_uv"][0]
        assert uv[0] == np.float32(1000) * np.float32(lsb[0])
        assert uv[7] == np.float32(8000) * np.float32(lsb[7])  # CH8 at gain 24, not 12
        assert f["imu_mta"].attrs["quat_fusion"] == "game_rotation_vector"
        assert int(f["imu_mta"]["accel"].attrs["rate_hz"]) == 100
        assert "rate_hz" not in f["imu_mta"]["gyro"].attrs  # disabled report


def test_native_writer_wrapper_is_closed_after_a_failing_close(tmp_path: Path):
    w = _native_writer(tmp_path / "wrap.h5", {})
    w.open()
    w._writer.append_emg(1, [0] * 8)
    w._writer._debug_fail_nth_append(1)
    with pytest.raises(RuntimeError):
        w.close()
    assert not w.has_open_file
    w.close()  # no second attempt, no second error


def test_int_trigger_rate_stays_int_and_no_seq_without_seq(tmp_path: Path):
    path = tmp_path / "legacy.h5"
    w = _native_writer(path, {"trigger": {"rate_hz": 30}})
    w.open()
    w._writer.append_emg(1, [0] * 8)
    w._writer.append_trigger(1, 1)
    w.close()
    with h5py.File(path, "r") as f:
        assert isinstance(f["triggers"].attrs["rate_hz"], np.integer)
        assert int(f["triggers"].attrs["rate_hz"]) == 30
        assert "seq" not in f["emg"]
