# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""mock_session: realistic defaults, writer-compatible layout, injected faults."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import pytest

import jtsmm.integrity.checker as checker
from jtsmm.config.catalog import channel_lsb_uv
from jtsmm.tools.mock_session import main, write_mock_session

def _errors(path: Path) -> list[str]:
    return list(checker.check_session(path).issues)


def _tree(path: Path) -> tuple[set[str], dict[str, set[str]]]:
    """Dataset paths and attribute names per object."""
    datasets: set[str] = set()
    attrs: dict[str, set[str]] = {}
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            attrs[name] = set(obj.attrs.keys())
            if isinstance(obj, h5py.Dataset):
                datasets.add(name)
        f.visititems(visit)
    return datasets, attrs


def test_defaults_follow_the_config_payload(tmp_path: Path):
    paths = write_mock_session(tmp_path / "s.h5", duration_s=4.0, write_jts_config=False)
    with h5py.File(paths["session_h5"], "r") as f:
        mcu = json.loads(f["meta"].attrs["mcu_config_json"])
        assert f["emg"].attrs["sample_rate_sps"] == mcu["emg"]["sample_rate_sps"] == 4000
        assert f["triggers"].attrs["rate_hz"] == mcu["trigger"]["rate_hz"] == 40
        assert f["triggers/t_master_us"].shape[0] == 160
        dt = np.diff(f["emg/t_master_us"][()].astype(np.int64))
        assert np.median(dt) == 250
        # Default kinematics_accel: the accelerometer reads gravity, no linear_accel.
        acc = f["imu_mta/accel"][()]
        assert 9.6 < float(np.mean(np.linalg.norm(acc, axis=1))) < 10.0
        assert f["imu_mta/accel"].attrs["rate_hz"] == mcu["imu_mta"]["reports"]["accelerometer"]["rate_hz"]
        assert "imu_mta/linear_accel" not in f
        assert set(f["imu_mta/accel_accuracy"][()]) == {3}
        uv = f["emg/samples_uv"][()]
        counts = f["emg/samples"][()]
        lsb = f["emg"].attrs["lsb_uv_per_channel"]
        np.testing.assert_allclose(lsb, channel_lsb_uv(int(f["emg"].attrs["gain"])))
        np.testing.assert_allclose(uv, counts * lsb, rtol=1e-6, atol=1e-3)
        s_ns = f["cameras/cam0/sensor_timestamp_ns"][()]
        assert s_ns[0] < 10**15  # boot-clock ns, not a Unix epoch


def test_overrides_keep_meta_consistent(tmp_path: Path):
    paths = write_mock_session(tmp_path / "s.h5", duration_s=2.0, emg_fs=1000, trig_hz=60,
                               write_jts_config=False)
    with h5py.File(paths["session_h5"], "r") as f:
        mcu = json.loads(f["meta"].attrs["mcu_config_json"])
        assert mcu["emg"]["sample_rate_sps"] == f["emg"].attrs["sample_rate_sps"] == 1000
        assert mcu["trigger"]["rate_hz"] == f["triggers"].attrs["rate_hz"] == 60
    assert _errors(paths["session_h5"]) == []


def test_clean_mock_passes_integrity(tmp_path: Path):
    paths = write_mock_session(tmp_path / "s.h5", duration_s=20.0, write_jts_config=False)
    assert _errors(paths["session_h5"]) == []


def test_injected_faults_are_present_and_detected(tmp_path: Path):
    paths = write_mock_session(
        tmp_path / "s.h5", duration_s=20.0, write_jts_config=False,
        emg_dropouts=[(5.0, 0.8)], optical_gaps=[(12.0, 1.5)],
        drop_frame_frac=0.1, nan_pose_frac=0.05, seed=3,
    )
    with h5py.File(paths["session_h5"], "r") as f:
        t_emg = f["emg/t_master_us"][()].astype(np.int64)
        assert np.diff(t_emg).max() >= 800_000
        seq = f["emg/seq"][()].astype(np.int64)
        assert int(np.diff(seq).max()) == 3201  # 0.8 s at 4 kSPS: the sequence advances across the hole
        t_opt = f["optical/t_master_us"][()].astype(np.int64)
        assert np.diff(t_opt).max() >= 1_500_000
        assert t_opt.size < f["triggers/t_master_us"].shape[0]
        mp = f["optical/pose_mta"][()]
        lost = ~np.isfinite(mp).reshape(len(mp), -1).all(axis=1)
        assert 0 < lost.sum() < 0.15 * len(mp)
        np.testing.assert_array_equal(f["optical/quality/mta/n_tags"][()] == 0, lost)
    assert any("EMG continuity" in e for e in _errors(paths["session_h5"]))


def test_mock_contains_everything_the_writer_writes(tmp_path: Path):
    """Every dataset and attribute Hdf5Writer produces also exists in the mock."""
    from jtsmm.io.hdf5_writer import Hdf5Writer
    from jtsmm.sync.clock_align import AlignedRecord

    mock = write_mock_session(tmp_path / "mock.h5", duration_s=2.0, write_jts_config=False)
    with h5py.File(mock["session_h5"], "r") as f:
        mcu = json.loads(f["meta"].attrs["mcu_config_json"])
        host = json.loads(f["meta"].attrs["host_config_json"])

    w = Hdf5Writer(path=tmp_path / "real.h5", config_hash="x", mcu_config=mcu,
                   host_config=host, firmware_version="1.0.0")
    w.open()
    for i in range(10):
        t = i * 500
        w.append(AlignedRecord("emg", t, {"channels": [i] * 8, "status": b"\xc0\x00\x00"}))
        w.append(AlignedRecord("imu_mta", t, {"quat": (1.0, 0, 0, 0), "quat_report_id": 0x08}))
        w.append(AlignedRecord("imu_mta", t, {"accel": (0, 0, 2511), "accel_accuracy": 3}))
        w.append(AlignedRecord("imu_cra", t, {"quat": (1.0, 0, 0, 0), "quat_report_id": 0x08}))
        w.append(AlignedRecord("imu_cra", t, {"accel": (0, 0, 2511), "accel_accuracy": 3}))
        w.append(AlignedRecord("trigger", t, {"trigger_id": i + 1}))
        for cid in ("cam0", "cam1"):
            w.append(AlignedRecord("camera", t, {"cam_id": cid, "sensor_timestamp_ns": 1_000 + t,
                                                 "trigger_id": i + 1, "frame_idx": i + 1}))
        q = {b: {"n_tags": 2, "n_cameras": 2, "reprojection_rms": 0.3} for b in ("mta", "cra", "ct")}
        w.append_optical(t_master_us=t, pose_mta=np.eye(4), pose_cra=np.eye(4), pose_ct=np.eye(4),
                         quality=q)
    w.append_event(t_master_us=0, kind="phase_start", text="p", phase_kind="experiment")
    w.write_phases_table([{"name": "p", "kind": "experiment", "t_start_us": 0, "t_end_us": 10}])
    w.write_sync_report({"drops": {}})
    w.write_sync_drops({"cam0": 1, "cam1": 1})
    w.close()

    real_ds, real_attrs = _tree(tmp_path / "real.h5")
    mock_ds, mock_attrs = _tree(mock["session_h5"])
    assert real_ds - mock_ds == set()
    for name, keys in real_attrs.items():
        assert keys - mock_attrs.get(name, set()) == set(), name
    # Supervisor-level /meta keys the writer receives via session_meta.
    for key in ("mode", "emg_enabled", "firmware_version", "optical_tracking",
                "optical_pose_method", "optical_cam_ids", "experiment_plan_json"):
        assert key in mock_attrs["meta"], key


def _stub_optical(monkeypatch):
    """Load optical/tracking/experiment.py without the cv2-importing package."""
    src = Path(__file__).resolve().parents[2] / "optical" / "optical" / "tracking" / "experiment.py"
    for name in ("optical", "optical.tracking"):
        mod = types.ModuleType(name)
        mod.__path__ = []  # namespace-like stub
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("optical.tracking.experiment", src)
    exp = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "optical.tracking.experiment", exp)
    spec.loader.exec_module(exp)


def test_jts_sidecar_uses_trigger_grid(tmp_path: Path, monkeypatch):
    _stub_optical(monkeypatch)
    paths = write_mock_session(tmp_path / "s.h5", duration_s=9.0, optical_gaps=[(3.0, 1.0)])
    cfg = json.loads(Path(paths["jts_config"]).read_text())
    with h5py.File(paths["session_h5"], "r") as f:
        phases = {n.decode() if isinstance(n, bytes) else n: (int(a), int(b)) for n, a, b in
                  zip(f["phases/name"][()], f["phases/t_start_us"][()], f["phases/t_end_us"][()])}
    a, b = cfg["analysis"]["experiment"]["sub_experiments"]["open_close"]
    t0, t1 = phases["open_close"]
    # Inclusive grid interval spans the phase duration at 40 Hz.
    assert (b - a + 1) / 40.0 == pytest.approx((t1 - t0) / 1e6, abs=1 / 40)


def test_cli_parses_fault_options(tmp_path: Path, capsys):
    rc = main([str(tmp_path / "c.h5"), "--duration", "3", "--emg-dropout", "1:0.5",
               "--optical-gap", "2:0.3", "--nan-pose-frac", "0.1"])
    assert rc == 0
    with h5py.File(tmp_path / "c.h5", "r") as f:
        assert np.diff(f["emg/t_master_us"][()].astype(np.int64)).max() >= 500_000
