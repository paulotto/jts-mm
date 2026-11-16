# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""CT tip calibration persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


# ct_pivot imports the optical preview stack, which needs OpenCV.
pytest.importorskip("cv2", reason="ct_pivot needs OpenCV (optical preview)")
pytest.importorskip("optical", reason="needs the in-repo optical package")

from jtsmm.capture.ct_pivot import write_ct_calibration_recording
from jtsmm.io import session_layout
from optical.tracking.pivot import PivotCalibrationResult


def test_ct_calibration_recording_dir_default_includes_experiment(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    out = session_layout.ct_calibration_recording_dir(
        None,
        experiment_name="s02_f_28",
        stamp="20260914T120000Z",
    )
    assert out == (
        tmp_path
        / "experiments"
        / "5_2_1_pivot_calibration_repeatability"
        / "recordings"
        / "s02_f_28_20260914T120000Z"
    )


def test_ct_calibration_recording_dir_custom_template(tmp_path: Path):
    template = str(tmp_path / "calib" / "{experiment_name}" / "recordings")
    out = session_layout.ct_calibration_recording_dir(
        template,
        experiment_name="My Exp!",
        stamp="20260914T120000Z",
    )
    assert out.name == "my_exp_20260914T120000Z"
    assert "my_exp" in out.parts
    assert out.parent.name == "recordings"


def test_ct_calibration_recording_dir_flat_base(tmp_path: Path):
    out = session_layout.ct_calibration_recording_dir(
        tmp_path / "flat",
        experiment_name="subj01",
        stamp="20260914T120000Z",
    )
    assert out == tmp_path / "flat" / "subj01_20260914T120000Z"


def test_write_ct_calibration_recording(tmp_path: Path):
    tip = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    pivot = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    result = PivotCalibrationResult(
        tip_body_mm=tip,
        pivot_world_mm=pivot,
        rms_mm=0.5,
        n_poses=3,
        residual_norm=1.0,
        tip_length_mm=float(np.linalg.norm(tip)),
        max_err_mm=0.8,
        median_err_mm=0.4,
        cond_A=10.0,
        per_pose_err_mm=(0.1, 0.2, 0.3),
    )
    poses = [np.eye(4, dtype=np.float64) for _ in range(3)]
    for i, T in enumerate(poses):
        T[:3, 3] = [i, 0.0, 0.0]
    frames = [
        {"cam0": np.zeros((8, 8), dtype=np.uint8)},
        {"cam0": np.ones((8, 8), dtype=np.uint8) * 40},
        {"cam0": np.ones((8, 8), dtype=np.uint8) * 80},
    ]
    src_bundle = tmp_path / "src_bundle.h5"
    src_bundle.write_bytes(b"calib-bytes")
    src_cfg = tmp_path / "optical.yaml"
    src_cfg.write_text("tracker: {}\n", encoding="utf-8")
    root = write_ct_calibration_recording(
        tmp_path / "run",
        result,
        poses,
        frames,
        experiment_name="s02_f_28",
        calibration_bundle_path=src_bundle,
        optical_config_path=src_cfg,
    )
    assert (root / "ct_tip_calibration.json").is_file()
    poses_doc = json.loads((root / "poses.json").read_text(encoding="utf-8"))
    assert poses_doc["n_poses"] == 3
    assert poses_doc["experiment_name"] == "s02_f_28"
    assert len(poses_doc["poses_world_body"]) == 3
    assert poses_doc["calibration_bundle"] == "calibration_bundle.h5"
    assert (root / "calibration_bundle.h5").read_bytes() == b"calib-bytes"
    assert (root / "optical.yaml").read_text(encoding="utf-8") == "tracker: {}\n"
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert meta["kind"] == "ct_tip_calibration"
    assert meta["image_size"] == [8, 8]
    assert meta["calibration_bundle"] == "calibration_bundle.h5"
    assert (root / "frames" / "00" / "cam0.jpg").is_file()
    assert (root / "frames" / "02" / "cam0.jpg").is_file()


def test_write_ct_calibration_recording_without_bundle(tmp_path: Path):
    tip = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = PivotCalibrationResult(
        tip_body_mm=tip,
        pivot_world_mm=tip,
        rms_mm=0.1,
        n_poses=1,
        residual_norm=0.1,
        tip_length_mm=float(np.linalg.norm(tip)),
        max_err_mm=0.1,
        median_err_mm=0.1,
        cond_A=1.0,
        per_pose_err_mm=(0.1,),
    )
    root = write_ct_calibration_recording(
        tmp_path / "run",
        result,
        [np.eye(4, dtype=np.float64)],
        None,
        experiment_name="x",
    )
    assert not (root / "calibration_bundle.h5").exists()
    poses_doc = json.loads((root / "poses.json").read_text(encoding="utf-8"))
    assert "calibration_bundle" not in poses_doc
    assert (root / "meta.json").is_file()
