# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Python and native HDF5 writers store bit-identical /emg/samples_uv."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm import native
from jtsmm.config.catalog import channel_lsb_uv
from jtsmm.sync.clock_align import AlignedRecord

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native HDF5 writer not built",
)


def _write_both(tmp_path: Path, gain: int, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from jtsmm.capture.native_backend import NativeWriter
    from jtsmm.io.hdf5_writer import Hdf5Writer

    mcu = {"emg": {"default_gain": gain, "sample_rate_sps": 2000}}
    out = []
    for cls, name in ((Hdf5Writer, "py.h5"), (NativeWriter, "native.h5")):
        w = cls(path=tmp_path / name, config_hash="x", mcu_config=mcu, host_config={})
        w.open()
        for i, row in enumerate(counts):
            ch = [int(v) for v in row]
            if cls is Hdf5Writer:
                w.append(AlignedRecord("emg", 1000 + i, {"channels": ch}))
            else:
                w._writer.append_emg(1000 + i, ch)  # noqa: SLF001
        w.flush()
        w.close()
        with h5py.File(tmp_path / name, "r") as f:
            out.append(np.asarray(f["emg"]["samples_uv"]))
    return out[0], out[1]


def _counts(n: int = 4000) -> np.ndarray:
    rng = np.random.default_rng(7)
    c = rng.integers(-(1 << 23), 1 << 23, size=(n, 8), dtype=np.int64)
    c[:8] = [[-(1 << 23), (1 << 23) - 1, 0, 1, -1, 12345, -54321, 7]] * 8
    return c


@pytest.mark.parametrize("gain", [1, 6, 24])
def test_catalog_gains_are_bit_identical(tmp_path: Path, gain: int):
    py, nat = _write_both(tmp_path, gain, _counts())
    assert py.dtype == nat.dtype == np.float32
    assert np.array_equal(py.view(np.uint32), nat.view(np.uint32))


def test_lsb_not_exact_in_float32_is_still_bit_identical(tmp_path: Path, monkeypatch):
    """An LSB with more than 24 significant bits (e.g. a 2**23 - 1 denominator)
    made float64-then-cast differ from the native float32 product."""
    odd = [v * (1 << 23) / ((1 << 23) - 1) for v in channel_lsb_uv(6, 8)]
    assert any(np.float64(np.float32(v)) != v for v in odd)
    monkeypatch.setattr("jtsmm.io.hdf5_writer.channel_lsb_uv", lambda g, n=8: odd[:n])
    monkeypatch.setattr("jtsmm.capture.native_backend.channel_lsb_uv", lambda g, n=8: odd[:n])
    py, nat = _write_both(tmp_path, 6, _counts())
    assert np.array_equal(py.view(np.uint32), nat.view(np.uint32))
