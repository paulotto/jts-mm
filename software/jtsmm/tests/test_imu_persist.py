# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""IMU–optical hand-eye persist helpers (§5.2.2)."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
from click.testing import CliRunner

from jtsmm.cli.imu_cmd import imu_group
from jtsmm.imu.kinematics import quat_from_rotation_matrix
from jtsmm.imu.mounting import MAX_RMS_DEG, fit_mounting_rotation
from jtsmm.imu.persist import (
    IMU_MOUNTING_JSON,
    discover_runs,
    next_run,
    slot_record_from_error,
    write_imu_mounting_recording,
)
from jtsmm.io import session_layout
from tests.test_imu_mounting import _figure_eight, _rx, _ry


def test_imu_calibration_recording_dir_default(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = session_layout.imu_calibration_recording_dir(
        None,
        experiment_name="s01",
        stamp="20260916T080000Z",
    )
    assert out == (
        tmp_path
        / "experiments"
        / "5_2_2_imu_optical_handeye"
        / "recordings"
        / "s01_20260916T080000Z"
    )


def test_write_imu_mounting_recording_copies_session(tmp_path: Path):
    src = tmp_path / "session.h5"
    with h5py.File(src, "w") as f:
        f.create_group("meta").attrs["subject_id"] = "s01"
    src_cfg = tmp_path / "optical.yaml"
    src_cfg.write_text("tracker: {}\n", encoding="utf-8")
    slots = {
        "imu_mta": {
            "slot": "imu_mta",
            "accepted": True,
            "rms_deg": 0.4,
            "n_pairs": 20,
            "n_samples": 40,
            "phase": "imu_mount_mta",
            "error": None,
            "R_mount": np.eye(3).tolist(),
            "T": np.eye(4).tolist(),
        },
        "imu_cra": slot_record_from_error("imu_cra", "nod-only"),
    }
    leaf = write_imu_mounting_recording(
        tmp_path / "run",
        slots,
        session_h5=src,
        subject_id="s01",
        run=2,
        experiment_name="imu_mounting_2",
        source_session=src,
        optical_config_path=src_cfg,
    )
    assert (leaf / IMU_MOUNTING_JSON).is_file()
    assert (leaf / "session.h5").is_file()
    assert (leaf / "optical.yaml").read_text(encoding="utf-8") == "tracker: {}\n"
    doc = json.loads((leaf / IMU_MOUNTING_JSON).read_text(encoding="utf-8"))
    assert doc["run"] == 2
    assert doc["accepted"] is False
    assert doc["slots"]["imu_mta"]["n_pairs"] == 20
    assert doc["slots"]["imu_cra"]["accepted"] is False
    meta = json.loads((leaf / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "imu_optical_handeye"
    assert meta["run"] == 2


def test_next_run_counts_all_leaves(tmp_path: Path):
    rec = tmp_path / "recordings"
    for i in range(1, 4):
        leaf = rec / f"imu_mounting_{i}"
        write_imu_mounting_recording(
            leaf,
            {"imu_mta": slot_record_from_error("imu_mta", "x")},
            run=i,
            experiment_name=f"imu_mounting_{i}",
        )
    assert next_run(rec) == 4
    assert len(discover_runs(rec)) == 3


def test_cli_fit_persists_metrics_json(tmp_path: Path, tmp_config_path: Path):
    R_mount = _ry(12.0) @ _rx(-5.0)
    n = 40
    R_opt = np.stack(_figure_eight(n))
    R_imu = np.stack([Ro @ R_mount for Ro in R_opt])
    t = np.arange(n, dtype=np.uint64) * 20_000
    poses = np.zeros((n, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ri) for Ri in R_imu])
    h5 = tmp_path / "session.h5"
    with h5py.File(h5, "w") as f:
        og = f.create_group("optical")
        og.create_dataset("t_master_us", data=t)
        og.create_dataset("pose_mta", data=poses)
        g = f.create_group("imu_mta")
        g.create_dataset("t_master_us_quat", data=t)
        g.create_dataset("quat", data=quats)
        meta = f.create_group("meta")
        meta.attrs["subject_id"] = "s09"

    rec = tmp_path / "recordings"
    runner = CliRunner()
    result = runner.invoke(
        imu_group,
        [
            "calibrate",
            "--mode",
            "fit",
            str(h5),
            "--config",
            str(tmp_config_path),
            "--imu",
            "mta",
            "--persist-imu-calibration",
            "--imu-calibration-recordings-dir",
            str(rec),
        ],
    )
    assert result.exit_code == 0, result.output
    jsons = list(rec.rglob(IMU_MOUNTING_JSON))
    assert len(jsons) == 1
    doc = json.loads(jsons[0].read_text(encoding="utf-8"))
    assert doc["run"] == 1
    assert jsons[0].parent.name.startswith("imu_mounting_1_")
    assert doc["accepted"] is True
    assert doc["slots"]["imu_mta"]["n_pairs"] >= 8
    assert doc["slots"]["imu_mta"]["rms_deg"] < MAX_RMS_DEG
    assert (jsons[0].parent / "session.h5").is_file()
    sidecar = h5.parent / IMU_MOUNTING_JSON
    assert sidecar.is_file()


def test_strict_false_logs_rejected_rms():
    R_opt = np.stack(_figure_eight(30))
    t = np.arange(30, dtype=np.uint64) * 20_000
    poses = np.zeros((30, 4, 4))
    poses[:, :3, :3] = R_opt
    poses[:, 3, 3] = 1.0
    quats = np.stack([quat_from_rotation_matrix(Ro) for Ro in R_opt])
    result = fit_mounting_rotation(
        t, poses, t, quats, slot="imu_mta", max_rms_deg=-1.0, strict=False
    )
    assert result.accepted is False
    assert result.error
    assert result.n_pairs >= 8
