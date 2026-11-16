# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""/events: the native writer stores the same datasets/dtypes/values as the Python one.

Viewer notes land in /events on both writer paths, are written incrementally
(a killed recorder keeps them up to the last flush) and can be re-stamped in
place at close (``set_event_time``), whether the row is on disk or buffered.
"""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import h5py
import pytest

from jtsmm import native
from jtsmm.io.hdf5_writer import Hdf5Writer

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None
         and hasattr(native.NativeHdf5Writer, "append_event")),
    reason="native HDF5 writer (with /events) not built",
)

EVENT_DS = ("t_master_us", "kind", "text", "phase_kind", "t_host_ns", "map_quality", "viewer_t_s")
_PKG_ROOT = str(Path(native.__file__).resolve().parents[2])

# (t_master_us, kind, text, phase_kind, t_host_ns, map_quality, viewer_t_s)
_EVENTS = [
    (1_000_000, "note", "jaw click left", "", 5_000_000_000, "pi_fit", 1.25),
    (1_500_000, "phase_start", "open_close", "experiment", 5_500_000_000, "pi_fit", None),
    (2_000_000, "note", "Kiefer knackt – ü ß 😬", "", 6_000_000_000, "mcu_arrival", 2.5),
    (2_600_000, "phase_end", "open_close", "experiment", 6_600_000_000, "pi_fit", None),
    (3_000_000, "note", "x" * 300, "", 7_000_000_000, "pi_fit", 0.0),
]


def _python(path: Path) -> Hdf5Writer:
    w = Hdf5Writer(path=path, config_hash="x", mcu_config={}, host_config={})
    w.open()
    return w


def _native(path: Path):
    from jtsmm.capture.native_backend import NativeWriter

    w = NativeWriter(path=path, config_hash="x", mcu_config={}, host_config={})
    w.open()
    return w


def _feed(w) -> list[int]:
    rows = []
    for i, (t, kind, text, pk, host, q, vt) in enumerate(_EVENTS):
        rows.append(w.append_event(t_master_us=t, kind=kind, text=text, phase_kind=pk,
                                   t_host_ns=host, map_quality=q, viewer_t_s=vt))
        if i == 2:
            w.flush()  # rows 0-2 on disk, 3-4 buffered
    # close-time re-map: one row on disk, one still buffered
    w.set_event_time(0, t_master_us=1_000_123, map_quality="pi_fit_final")
    w.set_event_time(4, t_master_us=3_000_456, map_quality="pi_fit_final")
    with pytest.raises(IndexError):
        w.set_event_time(len(_EVENTS), t_master_us=1, map_quality="x")
    w.close()
    return rows


def _read(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        g = f["events"]
        return {
            "names": sorted(g.keys()),
            "attrs": dict(g.attrs),
            "types": {k: g[k].id.get_type() for k in g},
            "maxshape": {k: g[k].maxshape for k in g},
            "values": {k: list(g[k].asstr()[()]) if h5py.check_string_dtype(g[k].dtype)
                       else g[k][()].tolist() for k in g},
        }


def test_events_parity_python_vs_native(tmp_path: Path):
    rows_py = _feed(_python(tmp_path / "py.h5"))
    rows_c = _feed(_native(tmp_path / "c.h5"))
    assert rows_py == rows_c == list(range(len(_EVENTS)))
    py, c = _read(tmp_path / "py.h5"), _read(tmp_path / "c.h5")
    assert py["names"] == c["names"] == sorted(EVENT_DS)
    assert py["attrs"] == c["attrs"]
    for k in EVENT_DS:
        assert py["types"][k] == c["types"][k], k  # H5Tequal: same dtype incl. vlen UTF-8
        assert py["maxshape"][k] == c["maxshape"][k] == (None,)
        if k != "viewer_t_s":
            assert py["values"][k] == c["values"][k], k
    for a, b in zip(py["values"]["viewer_t_s"], c["values"]["viewer_t_s"]):
        assert (math.isnan(a) and math.isnan(b)) or a == b
    v = py["values"]
    assert v["t_master_us"] == [1_000_123, 1_500_000, 2_000_000, 2_600_000, 3_000_456]
    assert v["map_quality"][0] == v["map_quality"][4] == "pi_fit_final"
    assert v["text"][2] == "Kiefer knackt – ü ß 😬"
    assert math.isnan(v["viewer_t_s"][1]) and v["viewer_t_s"][2] == 2.5
    with h5py.File(tmp_path / "c.h5", "r") as f:
        assert isinstance(f["events"]["kind"][0], bytes)  # vlen str, as h5py stores str


def test_native_set_event_time_after_partial_flush(tmp_path: Path):
    """A row a failed flush already landed must be re-stamped on disk too."""
    path = tmp_path / "partial.h5"
    w = native.NativeHdf5Writer(path=str(path), config_hash="x")
    w.open()
    for i in range(3):
        w.append_event(t_master_us=100 + i, kind="note", text=f"n{i}", t_host_ns=i)
    # appends: t_master_us(1) kind(2) text(3) phase_kind(4) t_host_ns(5)
    # map_quality(6) viewer_t_s(7) -> t_master_us and map_quality land, then fail
    w._debug_fail_nth_append(7)
    with pytest.raises(RuntimeError, match="injected"):
        w.flush()
    w.set_event_time(1, 999, "pi_fit")
    w.flush()
    w.close()
    with h5py.File(path, "r") as f:
        g = f["events"]
        assert {g[k].shape[0] for k in EVENT_DS} == {3}
        assert g["t_master_us"][()].tolist() == [100, 999, 102]
        assert list(g["map_quality"].asstr()[()]) == ["", "pi_fit", ""]
        assert all(math.isnan(x) for x in g["viewer_t_s"][()])


@pytest.mark.parametrize("which", ["python", "native"])
def test_killed_recorder_keeps_flushed_notes(tmp_path: Path, which: str):
    h5clear = shutil.which("h5clear")
    if h5clear is None:
        pytest.skip("h5clear (hdf5-tools) not installed")
    path = tmp_path / f"killed_{which}.h5"
    make = (
        f"from jtsmm.io.hdf5_writer import Hdf5Writer\n"
        f"w = Hdf5Writer(path=Path({str(path)!r}), config_hash='x', mcu_config={{}}, "
        f"host_config={{}}, file_flush_interval_s=0.0)"
        if which == "python" else
        f"from jtsmm import native\n"
        f"w = native.NativeHdf5Writer(path={str(path)!r}, config_hash='x', "
        f"disk_flush_interval_s=0.0)"
    )
    code = textwrap.dedent(
        """
        import time
        from pathlib import Path
        {make}
        w.open()
        for i in range(5):
            w.append_event(t_master_us=1000 * i, kind="note", text=f"note {{i}}", t_host_ns=i)
            w.flush()
        print("flushed", flush=True)
        time.sleep(60)
        """
    ).format(make=make)
    proc = subprocess.Popen([sys.executable, "-c", code], env=dict(os.environ, PYTHONPATH=_PKG_ROOT),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        line = proc.stdout.readline()
        assert line.startswith("flushed"), proc.stderr.read()
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:
            proc.kill()
    subprocess.run([h5clear, "-s", str(path)], check=True, capture_output=True)
    with h5py.File(path, "r") as f:
        assert list(f["events"]["text"].asstr()[()]) == [f"note {i}" for i in range(5)]
        assert f["events"]["t_master_us"][()].tolist() == [0, 1000, 2000, 3000, 4000]
