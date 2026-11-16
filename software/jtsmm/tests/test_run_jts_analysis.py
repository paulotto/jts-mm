# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Run JTS analysis and export smoothed model-frame trajectories."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm.tools.run_jts_analysis import (
    LANDMARK_TEETH,
    find_jts_config,
    format_landmark_report,
    prepare_config,
)

_MODEL_POINTS = [
    [-0.244678, -109.416, 101.356],
    [23.6189, -106.201, 73.9031],
    [-24.0445, -106.078, 74.1795],
]
_MAND_OFFSETS = (
    np.array([5.0, 0.0, 0.0]),
    np.array([0.0, 8.0, 0.0]),
    np.array([0.0, 0.0, 6.0]),
)


def _eye_at(xyz: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = xyz
    return T


def write_jts_session(path: Path, *, n_frames: int = 50, frame_rate: float = 60.0) -> Path:
    """Optical session with non-collinear CT digitizing poses + MP motion."""
    dt_us = int(round(1e6 / frame_rate))
    t_master = np.arange(n_frames, dtype=np.uint64) * dt_us
    hp = np.tile(np.eye(4), (n_frames, 1, 1)).astype(np.float64)
    mp = np.tile(np.eye(4), (n_frames, 1, 1)).astype(np.float64)
    ct = np.full((n_frames, 4, 4), np.nan, dtype=np.float64)

    mp_origin = np.array([20.0, 0.0, 0.0])
    for i in range(n_frames):
        mp[i] = _eye_at(mp_origin)
        hp[i] = _eye_at(np.zeros(3))

    mand_windows = ((0, 2), (3, 5), (6, 8))
    for (a, b), offset in zip(mand_windows, _MAND_OFFSETS):
        pose = _eye_at(mp_origin + offset)
        ct[a : b + 1] = pose

    max_windows = ((9, 11), (12, 14), (15, 17))
    for (a, b), xyz in zip(max_windows, _MODEL_POINTS):
        pose = _eye_at(np.asarray(xyz, dtype=np.float64))
        ct[a : b + 1] = pose
    # One dropped CT frame must not poison the interval average (JTS nanmean).
    ct[1] = np.nan

    # Opening motion during the experiment window (frames 20..end).
    for i in range(20, n_frames):
        z = 4.0 * np.sin(2.0 * np.pi * (i - 20) / 12.0)
        mp[i] = _eye_at(mp_origin + np.array([0.0, 0.0, z]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["session_id"] = "jts_analysis_unit_test"
        opt = f.create_group("optical")
        opt.create_dataset("t_master_us", data=t_master)
        opt.create_dataset("pose_cra", data=hp.astype(np.float32))
        opt.create_dataset("pose_mta", data=mp.astype(np.float32))
        opt.create_dataset("pose_ct", data=ct.astype(np.float32))
    return path


def write_jts_config(path: Path, session_h5: Path, *, template_model: bool = True) -> dict:
    model_points = [list(p) for p in _MODEL_POINTS]
    cfg = {
        "data_source": {"type": "aruco_camera", "filename": str(session_h5)},
        "analysis": {
            "description": "unit test",
            "calibration": {
                "mandibular": {
                    "rigid_bodies": ["MP", "CT"],
                    "points": [
                        {"name": "mand_point_1", "frame_interval": [0, 2]},
                        {"name": "mand_point_2", "frame_interval": [3, 5]},
                        {"name": "mand_point_3", "frame_interval": [6, 8]},
                    ],
                },
                "maxillary": {
                    "rigid_bodies": ["HP", "CT"],
                    "points": [
                        {"name": "max_point_1", "frame_interval": [9, 11]},
                        {"name": "max_point_2", "frame_interval": [12, 14]},
                        {"name": "max_point_3", "frame_interval": [15, 17]},
                    ],
                },
            },
            "relative_motion": {"reference_body": "HP", "moving_body": "MP"},
            "experiment": {"frame_interval": [20, 49]},
            "coordinate_transform": {
                "enabled": True,
                "calibration_type": "maxillary",
                "model_points": model_points,
            },
            "smoothing": {"enabled": True, "window_length": 11, "poly_order": 3},
            "coordinate_origin_index": 0,
        },
        "output": {"directory": str(path.parent), "save_hdf5": True, "save_csv": True},
        "visualization": {"show_plots": False, "final_trajectory": False},
    }
    if not template_model:
        cfg["analysis"]["coordinate_transform"]["model_points"] = [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
        ]
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


def test_find_jts_config_single_sidecar(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    h5.write_bytes(b"")
    sidecar = tmp_path / "s02_f_44_jts_config.json"
    sidecar.write_text("{}", encoding="utf-8")
    assert find_jts_config(h5) == sidecar


def test_find_jts_config_explicit_and_ambiguous(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    h5.write_bytes(b"")
    a = tmp_path / "a_jts_config.json"
    b = tmp_path / "b_jts_config.json"
    a.write_text("{}", encoding="utf-8")
    b.write_text("{}", encoding="utf-8")
    assert find_jts_config(h5, a) == a.resolve()
    with pytest.raises(FileNotFoundError, match="multiple"):
        find_jts_config(h5)


def test_landmark_report_names_teeth_and_template_warning(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    cfg_path = tmp_path / "demo_jts_config.json"
    cfg = write_jts_config(cfg_path, h5)
    text = format_landmark_report(cfg)
    assert "no tooth IDs" in text
    assert "mand_point_1" in text and LANDMARK_TEETH["mand_point_1"] in text
    assert "max_point_1" in text and "upper central incisor" in text
    assert "max_point_3" in text and "upper right first molar" in text
    assert "frames [0, 2]" in text
    assert "← landmark CS origin" in text
    assert "WARNING" in text
    assert "↔ max_point_1" in text


def test_landmark_teeth_incisor_first_and_matches_optical():
    # Pilot (Oct 2026): point 1 is the midline incisor (the JTS origin at
    # coordinate_origin_index=0), point 2 the participant's left, 3 the right.
    for jaw in ("mand", "max"):
        assert "central incisor" in LANDMARK_TEETH[f"{jaw}_point_1"]
        assert " left " in LANDMARK_TEETH[f"{jaw}_point_2"]
        assert " right " in LANDMARK_TEETH[f"{jaw}_point_3"]
    experiment = pytest.importorskip("optical.tracking.experiment")
    assert LANDMARK_TEETH == experiment.LANDMARK_TEETH


def test_prepare_config_writes_sidecar_not_session(tmp_path: Path):
    h5 = tmp_path / "session.h5"
    cfg = {
        "data_source": {"type": "aruco_camera", "filename": "/old.h5"},
        "visualization": {"raw_data": True, "final_trajectory": True, "show_plots": True},
        "output": {"hdf5_append_group": "jts", "directory": "."},
    }
    out = prepare_config(cfg, session_h5=h5, out_dir=tmp_path / "out", append_session=False)
    assert out["data_source"]["filename"] == str(h5)
    assert out["visualization"]["final_trajectory"] is False
    assert out["visualization"]["raw_data"] is False
    assert out["output"]["hdf5_append_group"] is None
    appended = prepare_config(cfg, session_h5=h5, out_dir=tmp_path, append_session=True)
    assert appended["output"]["hdf5_append_group"] == "jts"


def test_cli_landmarks_only(tmp_path: Path, capsys):
    from jtsmm.tools.run_jts_analysis import main

    h5 = tmp_path / "session.h5"
    write_jts_session(h5)
    write_jts_config(tmp_path / "unit_jts_config.json", h5)
    rc = main([str(h5), "--landmarks-only"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "lower left first molar" in text
    assert "Optical lock" in text
    assert not (tmp_path / "jaw_motion.h5").exists()


def test_cli_runs_analysis_and_writes_smoothed(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("sklearn")
    pytest.importorskip("scipy")
    from jtsmm.tools.run_jts_analysis import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    write_jts_session(h5)
    write_jts_config(tmp_path / "unit_jts_config.json", h5)
    rc = main([str(h5), "--out", str(out), "--no-show"])
    assert rc == 0
    motion = out / "jaw_motion.h5"
    assert motion.is_file()
    with h5py.File(motion, "r") as f:
        assert "T_model_origin_mand_landmark_t_smooth" in f
        grp = f["T_model_origin_mand_landmark_t_smooth"]
        assert "translations" in grp or "translation" in grp or any(
            "trans" in k.lower() for k in grp.keys()
        )
    pngs = list(out.glob("*_jts_smoothed.png"))
    assert len(pngs) == 1
    csvs = list(out.glob("jaw_motion_smoothed.csv"))
    assert len(csvs) == 1


def test_interpolate_se3_fills_one_nan_frame():
    from jtsmm.tools.run_jts_analysis import interpolate_se3_gaps

    T = np.tile(np.eye(4), (5, 1, 1)).astype(np.float64)
    for i in range(5):
        T[i, :3, 3] = [i, 0.0, 0.0]
    T[2] = np.nan
    filled, n_gap = interpolate_se3_gaps(T)
    assert n_gap == 1
    np.testing.assert_allclose(filled[2, :3, 3], [2.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(filled[2, :3, :3], np.eye(3), atol=1e-9)
