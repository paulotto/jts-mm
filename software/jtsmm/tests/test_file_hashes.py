# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""file_hashes and info CLI tests."""

from __future__ import annotations

import json
from pathlib import Path

import h5py

from jtsmm.config import applied_config_hash, split_host_mcu
from jtsmm.io.file_hashes import compute_file_hashes
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.sync.clock_align import AlignedRecord


def test_file_hashes_on_close(tmp_path: Path, resolved_config):
    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    out = tmp_path / "session.h5"
    side = tmp_path / "cam0.mkv"
    side.write_bytes(b"mockvideo")
    writer = Hdf5Writer(
        path=out,
        config_hash=applied_config_hash(resolved_config),
        mcu_config=mcu_cfg,
        host_config=host_bundle,
        firmware_version="1.0.0-prod",
        sidecar_paths={"cam0.mkv": side},
    )
    writer.open()
    writer.append(AlignedRecord("trigger", 100, {"trigger_id": 1}))
    writer.close()

    with h5py.File(out, "r") as f:
        raw = f["meta"].attrs.get("file_hashes_json")
        assert raw is not None
        table = json.loads(raw)
        assert "cam0.mkv" in table
        assert len(table["cam0.mkv"]) == 64


def test_compute_file_hashes(tmp_path: Path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"hello")
    js = compute_file_hashes({"a.bin": p})
    table = json.loads(js)
    assert "a.bin" in table
