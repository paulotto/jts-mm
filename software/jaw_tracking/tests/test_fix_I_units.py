# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2025 Paul-Otto Müller
# SPDX-License-Identifier: CC-BY-NC-SA-4.0

"""Export units: derivatives per second in the stored unit, CSV == HDF5."""

import h5py
import numpy as np
import pytest

from jts import core
from jts import helper as hlp

V_MM_S = 12.0  # constant velocity along x
RATE = 30.0


def _constant_velocity(t_s):
    T = np.tile(np.eye(4), (len(t_s), 1, 1))
    T[:, 0, 3] = V_MM_S * t_s
    return T


def _store(path, T, frame_data=None):
    hlp.store_transformations(
        [T], [RATE], path, group_names=["g"], derivative_order=2,
        scale_factor=0.001, unit="m", frame_datasets=[frame_data],
    )


def test_derivatives_in_stored_unit_per_second(tmp_path):
    t_s = np.arange(60) / RATE
    _store(tmp_path / "a.h5", _constant_velocity(t_s))
    with h5py.File(tmp_path / "a.h5", "r") as f:
        g = f["g"]
        np.testing.assert_allclose(g["translations"][:, 0], 0.001 * V_MM_S * t_s)
        v = g["translational_derivative_order_1"][()]
        a = g["translational_derivative_order_2"][()]
        np.testing.assert_allclose(v[:, 0], V_MM_S / 1000.0, rtol=1e-9)  # 0.012 m/s
        np.testing.assert_allclose(v[:, 1:], 0.0, atol=1e-12)
        np.testing.assert_allclose(a, 0.0, atol=1e-9)
        assert g["translational_derivative_order_1"].attrs["unit"] == "m/s"
        assert g["translational_derivative_order_2"].attrs["unit"] == "m/s^2"
        assert g["rotational_derivative_order_1"].attrs["unit"] == "1/s"
        assert g.attrs["derivative_time_basis"] == "sample_rate"


def test_derivatives_use_master_time_across_gaps(tmp_path):
    """Uneven spacing (dropped frames, synthetic bridge) still gives 12 mm/s."""
    k = np.r_[np.arange(0, 20), np.arange(35, 60)]  # 15 frames missing
    t_s = k / RATE
    t_us = 1e9 + t_s * 1e6
    t_us_marked = t_us.copy()
    t_us_marked[20] = np.nan  # a synthetic row: time interpolated
    T = _constant_velocity(t_s)
    T[20, 0, 3] = 0.5 * (T[19, 0, 3] + T[21, 0, 3])
    _store(tmp_path / "b.h5", T, {"t_master_us": t_us_marked})
    with h5py.File(tmp_path / "b.h5", "r") as f:
        g = f["g"]
        assert g.attrs["derivative_time_basis"] == "t_master_us"
        np.testing.assert_allclose(g["translational_derivative_order_1"][:, 0], 0.012, rtol=1e-6)
        assert np.isnan(g["t_master_us"][20])


def test_quaternion_sign_flip_does_not_spike():
    ang = np.linspace(0.0, 0.2, 20)
    T = np.tile(np.eye(4), (20, 1, 1))
    for i, a in enumerate(ang):
        c, s = np.cos(a), np.sin(a)
        T[i, :3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    _trans, rot = hlp.compute_derivatives(T, True, 1, time_s=np.arange(20) / RATE)
    # |dq/dt| = omega / 2 for a steady rotation; no +-2 jumps.
    omega = (ang[1] - ang[0]) * RATE
    np.testing.assert_allclose(np.linalg.norm(rot[0], axis=1), omega / 2, rtol=1e-3)


def test_nan_pose_rows_export_as_nan(tmp_path):
    T = _constant_velocity(np.arange(10) / RATE)
    T[4] = np.nan
    _store(tmp_path / "c.h5", T)
    hlp.save_transformation_to_csv(T, tmp_path / "c.csv", scale_factor=0.001, unit="m")
    with h5py.File(tmp_path / "c.h5", "r") as f:
        assert np.isnan(f["g"]["rotations"][4]).all()
    back = hlp.load_csv_to_transformations(tmp_path / "c.csv")
    assert np.isnan(back[4, :3, :3]).all()
    np.testing.assert_allclose(back[5, 0, 3], 0.001 * T[5, 0, 3])


def test_csv_matches_hdf5_units(tmp_path, monkeypatch):
    config = {
        "data_source": {"type": "qualisys", "filename": "dummy.mat"},
        "analysis": {"calibration": {}, "relative_motion": {"reference_body": "A", "moving_body": "B"},
                     "experiment": {"frame_interval": [0, 9]}, "smoothing": {"enabled": False},
                     "coordinate_transform": {"enabled": False}},
        "output": {"directory": str(tmp_path), "unit": "m", "scale_factor": 0.001,
                   "save_hdf5": True, "save_csv": True, "derivative_order": 1},
    }

    class Dummy:
        frame_rate = RATE

    monkeypatch.setattr(core.JawMotionAnalysis, "_create_motion_data_handler", lambda self: Dummy())
    analysis = core.JawMotionAnalysis(config)
    T = _constant_velocity(np.arange(10) / RATE)
    T[:, 1, 3] = 250.0  # mm
    analysis.trajectories["T_model_origin_mand_landmark_t"] = T
    analysis.active_trajectory = T
    analysis.active_trajectory_label = "raw"
    analysis.save_results()

    csv_path = tmp_path / "jaw_motion_raw.csv"
    header = csv_path.read_text().splitlines()[:2]
    assert header == ["# unit=m", "# tx_m,ty_m,tz_m,qw,qx,qy,qz"]
    csv = np.loadtxt(csv_path, delimiter=",")
    with h5py.File(tmp_path / "jaw_motion.h5", "r") as f:
        g = f["T_model_origin_mand_landmark_t"]
        assert g.attrs["unit"] == "m"
        np.testing.assert_allclose(csv[:, :3], g["translations"][()])
    np.testing.assert_allclose(csv[:, 1], 0.25)
    assert pytest.approx(0.012) == float(np.median(np.diff(csv[:, 0]) * RATE))
