# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""HDF5 writer round-trip."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jtsmm.config import applied_config_hash, split_host_mcu
from jtsmm.io.hdf5_writer import Hdf5Writer
from jtsmm.sync.clock_align import AlignedRecord


def test_hdf5_write_read_roundtrip(tmp_path: Path, resolved_config):
    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    out = tmp_path / "session.h5"
    writer = Hdf5Writer(
        path=out,
        config_hash=applied_config_hash(resolved_config),
        mcu_config=mcu_cfg,
        host_config=host_bundle,
    )
    writer.open()
    writer.append(
        AlignedRecord(
            "emg",
            1000,
            {"channels": [1, 2, 3, 4, 5, 6, 7, 8]},
        )
    )
    writer.append(
        AlignedRecord("trigger", 2000, {"trigger_id": 1, "flags": (0, 0, 0, 0)}),
    )
    writer.close()

    with h5py.File(out, "r") as f:
        assert "meta" in f
        assert f["meta"].attrs["config_hash"]
        assert f["emg"]["samples"].shape == (1, 8)
        assert f["triggers"]["trigger_id"][0] == 1
        assert "file_hashes_json" in f["meta"].attrs


def test_hdf5_write_optical_group(tmp_path: Path, resolved_config):
    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    out = tmp_path / "session_optical.h5"
    writer = Hdf5Writer(
        path=out,
        config_hash=applied_config_hash(resolved_config),
        mcu_config=mcu_cfg,
        host_config=host_bundle,
    )
    writer.open()
    pose = np.eye(4, dtype=np.float32)
    writer.append_optical(
        t_master_us=1234,
        pose_mta=pose,
        pose_cra=pose * 2.0,
        pose_ct=pose * 3.0,
    )
    writer.close()

    with h5py.File(out, "r") as f:
        assert f["optical"]["t_master_us"][0] == 1234
        assert f["optical"]["pose_mta"].shape == (1, 4, 4)
        assert float(f["optical"]["pose_cra"][0, 0, 0]) == 2.0
        assert float(f["optical"]["pose_ct"][0, 0, 0]) == 3.0
        q = f["optical"]["quality"]
        assert q["mta"]["n_tags"][0] == 0
        assert q["cra"]["n_cameras"].shape == (1,)


def test_hdf5_write_sync_group(tmp_path: Path, resolved_config):
    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    out = tmp_path / "session_sync.h5"
    writer = Hdf5Writer(
        path=out,
        config_hash=applied_config_hash(resolved_config),
        mcu_config=mcu_cfg,
        host_config=host_bundle,
    )
    writer.open()
    writer.write_sync_report({"pi_to_master": {"n": 0}})
    writer.write_sync_drops({"cam0": 3})
    writer.close()

    with h5py.File(out, "r") as f:
        assert "sync" in f
        raw = f["sync"].attrs["report_json"]
        if isinstance(raw, bytes):
            raw = raw.decode()
        assert '"n":0' in raw.replace(" ", "")
        assert f["sync"]["drops/cam0"][0, 1] == 3
