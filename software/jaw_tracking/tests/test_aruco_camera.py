# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2025 Paul-Otto Müller
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""
Unit tests for the jts.aruco_camera module.

The test suite covers:
- Loading a optical tracking-session HDF5 into RigidBody objects.
- The MotionCaptureData interface (transforms + relative motion).
- Integration with the JawMotionAnalysis factory (``data_source.type == "aruco_camera"``).
- A small end-to-end calibration + relative-motion pass through the core pipeline.

Synthetic HDF5 sessions are generated in-memory following
``docs/optical/03_pose_estimation_and_output.md`` §3.1 (schema v1.0).
"""

__author__ = "Paul-Otto Müller"
__copyright__ = "Copyright 2026, Paul-Otto Müller"
__credits__ = ["Paul-Otto Müller"]
__license__ = "GNU GPLv3"
__version__ = "1.1.1"
__maintainer__ = "Paul-Otto Müller"
__status__ = "Development"
__date__ = '04.02.2026'
__url__ = "https://github.com/paulotto/jaw_tracking_system"

import h5py
import numpy as np
import pytest

from jts import core
from jts import aruco_camera as acam


BODY_NAMES = ("HP", "MP", "CT")


def _make_pose(translation, rotation=None):
    """Build a 4x4 homogeneous pose from translation (mm) and optional rotation."""
    T = np.eye(4)
    if rotation is not None:
        T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def _write_session(path, n_frames=20, frame_rate=60.0, with_nan_frame=False):
    """
    Write a minimal but schema-correct tracking session HDF5.

    Each body moves along a distinct straight line so that the relative
    transforms between bodies are easy to reason about.
    """
    offsets = {"HP": np.array([0.0, 0.0, 0.0]),
               "MP": np.array([10.0, 0.0, 0.0]),
               "CT": np.array([0.0, 20.0, 0.0])}

    poses = {}
    for body in BODY_NAMES:
        body_poses = np.zeros((n_frames, 4, 4))
        for i in range(n_frames):
            translation = offsets[body] + np.array([i * 0.5, 0.0, 0.0])
            body_poses[i] = _make_pose(translation)
        if with_nan_frame:
            body_poses[0] = np.full((4, 4), np.nan)
        poses[body] = body_poses

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "1.0"
        meta.attrs["created_utc"] = "2026-01-01T00:00:00+00:00"
        meta.attrs["rig_id"] = "test_rig"
        meta.attrs["experiment_name"] = "unit_test"
        meta.attrs["frame_rate_hz"] = float(frame_rate)
        meta.attrs["n_frames"] = n_frames
        meta.attrs["method"] = "pnp"
        meta.attrs["notes"] = ""
        meta.attrs["units_position"] = "mm"
        meta.attrs["bodies"] = np.array(BODY_NAMES, dtype="S8")

        ts = f.create_group("timestamps")
        ts.create_dataset("frame_idx", data=np.arange(n_frames, dtype=np.int64))
        ts.create_dataset("t_master_us", data=np.full(n_frames, np.nan))
        t_cam = ts.create_group("t_cam_ns")
        t_cam.create_dataset("cam0", data=np.arange(n_frames, dtype=np.float64))

        bodies_grp = f.create_group("bodies")
        quality_grp = f.create_group("quality")
        for body in BODY_NAMES:
            bp = poses[body]
            pos = bp[:, :3, 3].copy()
            rot = bp[:, :3, :3].copy()
            valid = np.array([np.all(np.isfinite(bp[i])) for i in range(n_frames)])

            bg = bodies_grp.create_group(body)
            bg.create_dataset("pose_4x4", data=bp)
            bg.create_dataset("position_mm", data=pos)
            bg.create_dataset("rotation_matrix", data=rot)
            bg.create_dataset("tracking_valid", data=valid)

            qg = quality_grp.create_group(body)
            qg.create_dataset("tags_used", data=np.full(n_frames, 2, dtype=np.int32))
            qg.create_dataset("reprojection_rms_px", data=np.full(n_frames, 0.3))
            qg.create_dataset("n_cameras", data=np.full(n_frames, 2, dtype=np.int32))

    return poses


def test_aruco_camera_data_loads(tmp_path):
    """ArucoCameraData parses meta and builds RigidBody objects for each body."""
    path = tmp_path / "session.h5"
    _write_session(path, n_frames=15, frame_rate=120.0)

    data = acam.ArucoCameraData(str(path))

    assert data.frame_rate == 120.0
    assert data.frames == 15
    assert data.rig_id == "test_rig"
    assert data.experiment_name == "unit_test"
    assert set(data.rigid_bodies) == set(BODY_NAMES)

    hp = data.rigid_bodies["HP"]
    assert hp.positions.shape == (15, 3)
    assert hp.rotations.shape == (15, 3, 3)


def test_aruco_camera_data_missing_file():
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        acam.ArucoCameraData("definitely_not_a_real_session.h5")


def test_get_rigid_body_transform(tmp_path):
    """get_rigid_body_transform returns T_world<-body with the stored translations."""
    path = tmp_path / "session.h5"
    _write_session(path, n_frames=10)

    handler = acam.ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    handler.load_data(str(path))

    T_mp = handler.get_rigid_body_transform("MP")
    assert T_mp.shape == (10, 4, 4)
    # MP starts at x=10 and advances 0.5 mm/frame.
    np.testing.assert_allclose(T_mp[0, :3, 3], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(T_mp[5, :3, 3], [12.5, 0.0, 0.0])

    with pytest.raises(ValueError):
        handler.get_rigid_body_transform("NOPE")


def test_compute_relative_transform(tmp_path):
    """Relative transform HP<-MP is the constant body offset (pure translation here)."""
    path = tmp_path / "session.h5"
    _write_session(path, n_frames=10)

    handler = acam.ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    handler.load_data(str(path))

    T_rel = handler.compute_relative_transform("HP", "MP")
    assert T_rel.shape == (10, 4, 4)
    # Both bodies share the same orientation and per-frame x-advance, so the
    # relative translation stays the constant initial offset (10 mm in x).
    for i in range(10):
        np.testing.assert_allclose(T_rel[i, :3, 3], [10.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(T_rel[i, :3, :3], np.eye(3), atol=1e-9)


def test_factory_creates_aruco_handler(tmp_path):
    """JawMotionAnalysis routes data_source.type == 'aruco_camera' to the loader."""
    path = tmp_path / "session.h5"
    _write_session(path, n_frames=8)

    config = {
        "data_source": {"type": "aruco_camera", "filename": str(path)},
        "analysis": {
            "calibration": {},
            "relative_motion": {"reference_body": "HP", "moving_body": "MP"},
            "experiment": {"frame_interval": [0, 7]},
            "smoothing": {"enabled": False},
        },
        "output": {"directory": "."},
    }

    analysis = core.JawMotionAnalysis(config)
    assert isinstance(analysis.motion_data, acam.ArucoCameraMotionCaptureData)


def test_end_to_end_calibration_and_relative_motion(tmp_path):
    """A full calibration + relative-motion pass works on a synthetic session."""
    pytest.importorskip("sklearn")  # PCA landmark frame
    path = tmp_path / "session.h5"
    _write_session(path, n_frames=30)

    config = {
        "data_source": {"type": "aruco_camera", "filename": str(path)},
        "analysis": {
            "calibration": {
                "mandibular": {
                    "rigid_bodies": ["MP", "CT"],
                    "points": [
                        {"name": "mand_point_1", "frame_interval": [0, 4]},
                        {"name": "mand_point_2", "frame_interval": [5, 9]},
                        {"name": "mand_point_3", "frame_interval": [10, 14]},
                    ],
                },
                "maxillary": {
                    "rigid_bodies": ["HP", "CT"],
                    "points": [
                        {"name": "max_point_1", "frame_interval": [0, 4]},
                        {"name": "max_point_2", "frame_interval": [5, 9]},
                        {"name": "max_point_3", "frame_interval": [10, 14]},
                    ],
                },
            },
            "relative_motion": {"reference_body": "HP", "moving_body": "MP"},
            "experiment": {"frame_interval": [15, 29]},
            "smoothing": {"enabled": False},
            "coordinate_transform": {"enabled": False, "calibration_type": "maxillary",
                                     "model_points": [[0, 0, 0], [1, 0, 0], [0, 1, 0]]},
            "coordinate_origin_index": 0,
        },
        "output": {"directory": str(tmp_path)},
    }

    analysis = core.JawMotionAnalysis(config)
    analysis.load_data()

    calib = analysis.perform_calibration()
    assert "T_MP_landmark" in calib
    assert "T_HP_landmark" in calib

    T_rel = analysis.compute_relative_motion()
    assert T_rel.shape[1:] == (4, 4)
    assert T_rel.shape[0] > 0


def test_pose_4x4_fallback(tmp_path):
    """Bodies without position_mm/rotation_matrix decompose from pose_4x4."""
    path = tmp_path / "session_pose_only.h5"
    n_frames = 6
    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["schema_version"] = "1.0"
        meta.attrs["frame_rate_hz"] = 60.0
        meta.attrs["n_frames"] = n_frames

        bodies_grp = f.create_group("bodies")
        for body in BODY_NAMES:
            poses = np.tile(np.eye(4), (n_frames, 1, 1))
            for i in range(n_frames):
                poses[i, :3, 3] = [i, 2 * i, 3 * i]
            bg = bodies_grp.create_group(body)
            bg.create_dataset("pose_4x4", data=poses)

    data = acam.ArucoCameraData(str(path))
    assert data.frames == n_frames
    np.testing.assert_allclose(data.rigid_bodies["HP"].positions[2], [2, 4, 6])


def _write_multimodal_session(path, n_frames=20, frame_rate=60.0):
    """Write a minimal multimodal session.h5 with /optical/ poses."""
    dt_us = int(round(1e6 / frame_rate))
    t_master = np.arange(n_frames, dtype=np.uint64) * dt_us

    offsets = {
        "pose_cra": np.array([0.0, 0.0, 0.0]),   # HP
        "pose_mta": np.array([10.0, 0.0, 0.0]),  # MP
        "pose_ct": np.array([0.0, 20.0, 0.0]),   # CT
    }

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["software_version"] = "test"
        meta.attrs["session_id"] = "multimodal_unit_test"

        opt = f.create_group("optical")
        opt.create_dataset("t_master_us", data=t_master)
        for ds_name, offset in offsets.items():
            poses = np.zeros((n_frames, 4, 4), dtype=np.float32)
            for i in range(n_frames):
                poses[i] = np.eye(4)
                poses[i, :3, 3] = offset + np.array([i * 0.5, 0.0, 0.0])
            opt.create_dataset(ds_name, data=poses)


def test_aruco_camera_loads_multimodal_optical(tmp_path):
    """ArucoCameraData maps /optical/pose_* onto HP/MP/CT and derives Hz."""
    path = tmp_path / "session.h5"
    _write_multimodal_session(path, n_frames=15, frame_rate=120.0)

    data = acam.ArucoCameraData(str(path))

    assert data.layout == "optical"
    assert data.frames == 15
    assert set(data.rigid_bodies) == set(BODY_NAMES)
    np.testing.assert_allclose(data.frame_rate, 120.0, rtol=1e-4)
    np.testing.assert_allclose(data.rigid_bodies["MP"].positions[0], [10.0, 0.0, 0.0])
    np.testing.assert_allclose(data.rigid_bodies["HP"].positions[5], [2.5, 0.0, 0.0])


def test_multimodal_relative_transform(tmp_path):
    """Relative HP<-MP from multimodal /optical/ matches body offset."""
    path = tmp_path / "session.h5"
    _write_multimodal_session(path, n_frames=10, frame_rate=60.0)

    handler = acam.ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    handler.load_data(str(path))

    T_rel = handler.compute_relative_transform("HP", "MP")
    assert T_rel.shape == (10, 4, 4)
    for i in range(10):
        np.testing.assert_allclose(T_rel[i, :3, 3], [10.0, 0.0, 0.0], atol=1e-5)


def test_append_jts_group_into_session(tmp_path):
    """JTS can append kinematics under /jts without wiping /optical/."""
    path = tmp_path / "session.h5"
    _write_multimodal_session(path, n_frames=8, frame_rate=60.0)

    T = np.tile(np.eye(4), (5, 1, 1))
    T[:, :3, 3] = np.arange(5)[:, None]

    from jts import helper as hlp

    hlp.store_transformations(
        [T],
        [60.0],
        path,
        group_names=["T_model_origin_mand_landmark_t"],
        mode="a",
        parent_group="jts",
        store_as_quaternion=False,
        derivative_order=0,
        unit="mm",
    )

    with h5py.File(path, "r") as f:
        assert "optical" in f
        assert "pose_mta" in f["optical"]
        assert "jts" in f
        assert "T_model_origin_mand_landmark_t" in f["jts"]
        translations = f["jts"]["T_model_origin_mand_landmark_t"]["translations"][()]
        np.testing.assert_allclose(translations[:, 0], np.arange(5))


def test_save_hdf5_results_appends_when_configured(tmp_path, monkeypatch):
    """JawMotionAnalysis._save_hdf5_results honours output.hdf5_append_group."""
    path = tmp_path / "session.h5"
    _write_multimodal_session(path, n_frames=8, frame_rate=60.0)

    config = {
        "data_source": {"type": "aruco_camera", "filename": str(path)},
        "analysis": {
            "calibration": {},
            "relative_motion": {"reference_body": "HP", "moving_body": "MP"},
            "experiment": {"frame_interval": [0, 7]},
            "smoothing": {"enabled": False},
        },
        "output": {
            "directory": str(tmp_path),
            "hdf5_append_group": "jts",
            "store_quaternions": False,
            "derivative_order": 0,
            "scale_factor": 1.0,
            "unit": "mm",
        },
    }

    analysis = core.JawMotionAnalysis(config)
    analysis.motion_data = acam.ArucoCameraMotionCaptureData({"type": "aruco_camera"})
    analysis.motion_data.load_data(str(path))
    analysis.trajectories["T_model_origin_mand_landmark_t"] = np.tile(np.eye(4), (4, 1, 1))
    analysis.active_trajectory_label = "raw"

    analysis._save_hdf5_results(tmp_path)

    with h5py.File(path, "r") as f:
        assert "optical" in f
        assert "jts" in f
        assert "T_model_origin_mand_landmark_t" in f["jts"]
    assert not (tmp_path / "jaw_motion.h5").exists()
