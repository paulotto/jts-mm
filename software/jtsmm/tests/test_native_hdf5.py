# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""M8b gate: the native HDF5 writer must reproduce the Python writer's schema.

Feeds identical EMG/IMU/trigger data through ``Hdf5Writer`` (Python) and
``NativeHdf5Writer`` (C++/libhdf5) and compares the key datasets and meta attrs.
Skips when the native extension is not built.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm import __version__, native
from jtsmm.config import applied_config_hash, split_host_mcu
from jtsmm.config.catalog import FULL_CHANNEL_NAMES, channel_lsb_uv, lsb_uv
from jtsmm.io.hdf5_writer import Hdf5Writer, _muscle_leadoff_mask
from jtsmm.sync.clock_align import AlignedRecord

pytestmark = pytest.mark.skipif(
    not (native.is_available() and native.NativeHdf5Writer is not None),
    reason="native HDF5 writer not built",
)


def _synthetic_status(i: int) -> bytes:
    """Deterministic, varying ADS1299 status word so leadoff_status exercises
    non-zero bits in both writers (bits[23:20] are the fixed 0b1100 preamble)."""
    b0 = 0xC0 | ((i >> 4) & 0x0F)
    b1 = ((i & 0x0F) << 4) | ((i >> 8) & 0x0F)
    b2 = (i >> 2) & 0xF0
    return bytes([b0, b1, b2])


def _make_records(n_emg: int = 2500, n_imu: int = 900, n_trig: int = 60):
    emg = []
    rng = np.random.default_rng(0)
    for i in range(n_emg):
        ch = [int(v) for v in rng.integers(-(1 << 20), 1 << 20, size=8)]
        emg.append((1000 + i * 1000, ch, _synthetic_status(i)))
    imu_mta = [(2000 + i * 2500, (1.0, 0.01 * i, -0.02 * i, 0.5)) for i in range(n_imu)]
    imu_cra = [(2100 + i * 2500, (0.7, 0.0, 0.7, 0.0)) for i in range(n_imu)]
    trig = [(3000 + i * 16000, i + 1) for i in range(n_trig)]
    return emg, imu_mta, imu_cra, trig


# Viewer notes (t_master_us, text, t_host_ns, viewer_t_s) -> /events on both writers.
_NOTES = [(40_000, "jaw click", 7_000_000_000, 1.5), (90_000, "Schmerz – links", 7_050_000_000, None)]


def _write_python(path: Path, mcu_cfg, host_bundle, cfg_hash, recs) -> None:
    emg, imu_mta, imu_cra, trig = recs
    w = Hdf5Writer(
        path=path,
        config_hash=cfg_hash,
        mcu_config=mcu_cfg,
        host_config=host_bundle,
        firmware_version="unit-test",
        session_meta={"mode": "acquisition", "subject_id": "P07", "protocol": "motion_battery_v1"},
    )
    w.open()
    for t, ch, status in emg:
        w.append(AlignedRecord("emg", t, {"channels": ch, "status": status}))
    for t, q in imu_mta:
        w.append(AlignedRecord("imu_mta", t, {"quat": q}))
    for t, q in imu_cra:
        w.append(AlignedRecord("imu_cra", t, {"quat": q}))
    for t, tid in trig:
        w.append(AlignedRecord("trigger", t, {"trigger_id": tid, "flags": (0, 0, 0, 0)}))
    w.append(
        AlignedRecord(
            "camera",
            5000,
            {
                "cam_id": "cam0",
                "sensor_timestamp_ns": 9_000_000_000,
                "trigger_id": 3,
                "frame_idx": 1,
                "video_file": "cam0.h264",
            },
        )
    )
    w.append(
        AlignedRecord(
            "camera",
            6000,
            {
                "cam_id": "cam1",
                "sensor_timestamp_ns": 9_100_000_000,
                "trigger_id": 4,
                "frame_idx": 2,
                "video_file": "cam1.h264",
            },
        )
    )
    for t, text, host, vt in _NOTES:
        w.append_event(t_master_us=t, kind="note", text=text, t_host_ns=host,
                       map_quality="pi_fit", viewer_t_s=vt)
    w.close()


def _write_native(path: Path, mcu_cfg, host_bundle, cfg_hash, recs) -> None:
    emg, imu_mta, imu_cra, trig = recs
    emg_cfg = mcu_cfg.get("emg", {})
    w = native.NativeHdf5Writer(
        path=str(path),
        config_hash=cfg_hash,
        mcu_config_json=json.dumps(mcu_cfg, separators=(",", ":")),
        host_config_json=json.dumps(host_bundle, separators=(",", ":")),
        software_version=__version__,
        schema_version=int(mcu_cfg.get("schema_version", 1)),
        firmware_version="unit-test",
        mode="acquisition",
        emg_sample_rate_sps=int(emg_cfg.get("sample_rate_sps", 1000)),
        emg_gain=int(emg_cfg.get("default_gain", 6)),
        channel_names=list(FULL_CHANNEL_NAMES),
        lsb_uv_per_channel=channel_lsb_uv(int(emg_cfg.get("default_gain", 6))),
        fusion_mta=mcu_cfg.get("imu_mta", {}).get("fusion", {}).get("primary", "game_rotation_vector"),
        fusion_cra=mcu_cfg.get("imu_cra", {}).get("fusion", {}).get("primary", "game_rotation_vector"),
        trigger_rate_hz=mcu_cfg.get("trigger", {}).get("rate_hz"),
        extra_meta={"subject_id": "P07", "protocol": "motion_battery_v1"},
    )
    w.open()
    for t, ch, status in emg:
        statp, statn = _muscle_leadoff_mask(status)
        w.append_emg(t, ch, statp, statn)
    for t, q in imu_mta:
        w.append_quat(False, t, list(q))
    for t, q in imu_cra:
        w.append_quat(True, t, list(q))
    for t, tid in trig:
        w.append_trigger(t, tid)
    w.append_camera("cam0", 5000, 9_000_000_000, 3, 1, "cam0.h264")
    w.append_camera("cam1", 6000, 9_100_000_000, 4, 2, "cam1.h264")
    for t, text, host, vt in _NOTES:
        w.append_event(t_master_us=t, kind="note", text=text, t_host_ns=host,
                       map_quality="pi_fit", viewer_t_s=vt)
    w.set_file_hashes_json("{}")
    w.close()


def test_native_hdf5_matches_python(tmp_path: Path, resolved_config):
    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    cfg_hash = applied_config_hash(resolved_config)
    recs = _make_records()

    py_path = tmp_path / "py.h5"
    c_path = tmp_path / "c.h5"
    _write_python(py_path, mcu_cfg, host_bundle, cfg_hash, recs)
    _write_native(c_path, mcu_cfg, host_bundle, cfg_hash, recs)

    with h5py.File(py_path, "r") as fp, h5py.File(c_path, "r") as fc:
        # EMG: integer datasets must match exactly; µV is a derived float32.
        assert np.array_equal(fp["emg"]["samples"][:], fc["emg"]["samples"][:])
        assert np.array_equal(fp["emg"]["t_master_us"][:], fc["emg"]["t_master_us"][:])
        # K5: samples_uv is per column (CH7/CH8 at their fixed gain), float32 math.
        gain = int(fc["emg"].attrs["gain"])
        lsb_cols = np.asarray(channel_lsb_uv(gain), dtype=np.float32)
        expect_uv = fc["emg"]["samples"][:].astype(np.float32) * lsb_cols
        assert np.array_equal(fc["emg"]["samples_uv"][:], expect_uv)
        assert np.allclose(fc["emg"].attrs["lsb_uv_per_channel"], channel_lsb_uv(gain))
        assert float(fc["emg"].attrs["lsb_uv"]) == pytest.approx(lsb_uv(gain))
        if "lsb_uv_per_channel" in fp["emg"].attrs:  # Python writer with K5
            assert np.allclose(fp["emg"]["samples_uv"][:], fc["emg"]["samples_uv"][:], rtol=1e-6)
        else:  # muscle columns agree either way
            assert np.array_equal(fp["emg"]["samples_uv"][:, :6], fc["emg"]["samples_uv"][:, :6])
        assert np.array_equal(fp["emg"]["leadoff_status"][:], fc["emg"]["leadoff_status"][:])
        # DRL (bit 7) must never leak into either writer's dataset.
        assert not np.any(fp["emg"]["leadoff_status"][:] & 0xC0)
        assert not np.any(fc["emg"]["leadoff_status"][:] & 0xC0)
        assert "leadoff_status_note" in fp["emg"].attrs
        assert "leadoff_status_note" in fc["emg"].attrs
        assert fp["emg"].attrs["sample_rate_sps"] == fc["emg"].attrs["sample_rate_sps"]
        assert fp["emg"].attrs["gain"] == fc["emg"].attrs["gain"]
        assert float(fp["emg"].attrs["lsb_uv"]) == pytest.approx(float(fc["emg"].attrs["lsb_uv"]))

        for grp in ("imu_mta", "imu_cra"):
            assert np.array_equal(fp[grp]["quat"][:], fc[grp]["quat"][:])
            assert np.array_equal(
                fp[grp]["t_master_us_quat"][:], fc[grp]["t_master_us_quat"][:]
            )

        assert np.array_equal(fp["triggers"]["trigger_id"][:], fc["triggers"]["trigger_id"][:])
        assert np.array_equal(fp["triggers"]["t_master_us"][:], fc["triggers"]["t_master_us"][:])

        for cam in ("cam0", "cam1"):
            assert np.array_equal(fp["cameras"][cam]["t_master_us"][:], fc["cameras"][cam]["t_master_us"][:])
            assert np.array_equal(
                fp["cameras"][cam]["sensor_timestamp_ns"][:],
                fc["cameras"][cam]["sensor_timestamp_ns"][:],
            )
            assert np.array_equal(fp["cameras"][cam]["trigger_id"][:], fc["cameras"][cam]["trigger_id"][:])
            assert fp["cameras"][cam].attrs.get("video_file") == fc["cameras"][cam].attrs.get("video_file")

        # /events (viewer notes): same datasets, dtypes and values.
        assert sorted(fp["events"]) == sorted(fc["events"])
        for k in fp["events"]:
            dp, dc = fp["events"][k], fc["events"][k]
            assert dp.id.get_type() == dc.id.get_type(), k
            if k == "viewer_t_s":
                assert np.array_equal(dp[()], dc[()], equal_nan=True)
            else:
                assert list(dp[()]) == list(dc[()]), k

        # Meta attrs the integrity checker depends on.
        assert fp["meta"].attrs["config_hash"] == fc["meta"].attrs["config_hash"]
        assert fp["meta"].attrs["mcu_config_json"] == fc["meta"].attrs["mcu_config_json"]
        assert fp["meta"].attrs["schema_version"] == fc["meta"].attrs["schema_version"]
        assert "file_hashes_json" in fc["meta"].attrs
        assert "session_duration_s" in fc["meta"].attrs

        # Recording date/time: both writers stamp an ISO-8601 UTC string.
        iso_re = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        assert iso_re.match(fp["meta"].attrs["recording_date"])
        assert iso_re.match(fc["meta"].attrs["recording_date"])

        # Custom session metadata (`jtsmm record --subject/--protocol/--meta`)
        # must be flattened to individual /meta attrs by both writers.
        assert fp["meta"].attrs["subject_id"] == fc["meta"].attrs["subject_id"] == "P07"
        assert fp["meta"].attrs["protocol"] == fc["meta"].attrs["protocol"] == "motion_battery_v1"


def test_native_hdf5_camera_incremental_flush(tmp_path: Path, resolved_config):
    """Repeated flush while appending camera rows must not rely on failed H5Gopen2."""
    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    cfg_hash = applied_config_hash(resolved_config)
    emg_cfg = mcu_cfg.get("emg", {})
    path = tmp_path / "cam_flush.h5"
    w = native.NativeHdf5Writer(
        path=str(path),
        config_hash=cfg_hash,
        mcu_config_json=json.dumps(mcu_cfg, separators=(",", ":")),
        host_config_json=json.dumps(host_bundle, separators=(",", ":")),
        software_version=__version__,
        schema_version=int(mcu_cfg.get("schema_version", 1)),
        firmware_version="unit-test",
        mode="acquisition",
        emg_sample_rate_sps=int(emg_cfg.get("sample_rate_sps", 1000)),
        emg_gain=int(emg_cfg.get("default_gain", 6)),
        channel_names=list(FULL_CHANNEL_NAMES),
        fusion_mta=mcu_cfg.get("imu_mta", {}).get("fusion", {}).get("primary", "game_rotation_vector"),
        fusion_cra=mcu_cfg.get("imu_cra", {}).get("fusion", {}).get("primary", "game_rotation_vector"),
        trigger_rate_hz=mcu_cfg.get("trigger", {}).get("rate_hz"),
    )
    w.open()
    for i in range(1, 6):
        w.append_camera("cam0", 1000 * i, 9_000_000_000 + i, i, i, "")
        w.append_camera("cam1", 2000 * i, 9_100_000_000 + i, i, i, "")
        w.flush()
    w.set_file_hashes_json("{}")
    w.close()

    with h5py.File(path, "r") as f:
        assert len(f["cameras"]["cam0"]["t_master_us"]) == 5
        assert len(f["cameras"]["cam1"]["t_master_us"]) == 5


def test_native_hdf5_integrity_matches_python(tmp_path: Path, resolved_config):
    """The integrity checker must reach the same verdict on both writers' output.

    This is the schema-contract gate: whatever issues (or none) the checker finds
    on the Python-written session, it must find the same on the native session.
    """
    from jtsmm.integrity.checker import check_session

    mcu_cfg, host_bundle = split_host_mcu(resolved_config)
    cfg_hash = applied_config_hash(resolved_config)
    recs = _make_records()

    py_path = tmp_path / "py.h5"
    c_path = tmp_path / "c.h5"
    _write_python(py_path, mcu_cfg, host_bundle, cfg_hash, recs)
    _write_native(c_path, mcu_cfg, host_bundle, cfg_hash, recs)

    py_report = check_session(py_path)
    c_report = check_session(c_path)
    assert sorted(c_report.issues) == sorted(py_report.issues)
    # And in particular the native path must not introduce a config_hash mismatch.
    assert not any("config_hash" in i for i in c_report.issues), c_report.issues
