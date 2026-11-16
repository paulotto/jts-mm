# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Offline plot tools: IMU world accel, fusion-report filter, gaps, EMG, errors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from jtsmm.config.catalog import FULL_CHANNEL_NAMES

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

RZ90 = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))  # wxyz, +90 deg about z


# --------------------------------------------------------------------- IMU
def test_world_accel_uses_chip_orientation_only():
    from jtsmm.tools.visualize_session import world_linear_accel

    t_q = np.array([0.0, 1e6])
    q = np.array([RZ90, RZ90])
    a_world, R_wc = world_linear_accel(t_q, q, np.array([0.0, 5e5]), np.array([[1.0, 0, 0]] * 2))
    np.testing.assert_allclose(a_world, [[0.0, 1.0, 0.0]] * 2, atol=1e-12)
    assert np.isfinite(R_wc).all()


def test_mounting_is_not_applied_to_accel(tmp_path: Path):
    """With a mounting transform present the world accel must not change."""
    from jtsmm.tools import visualize_session as vs

    seen = {}
    real = vs.world_linear_accel

    def spy(t_q, q, t_lin, a):
        out = real(t_q, q, t_lin, a)
        seen["a_world"] = out[0]
        return out

    h5 = tmp_path / "s.h5"
    n = 50
    with h5py.File(h5, "w") as f:
        g = f.create_group("imu_mta")
        g.attrs["fusion_mode"] = "game_rotation_vector"
        g.create_dataset("quat", data=np.tile(np.asarray(RZ90, np.float32), (n, 1)))
        g.create_dataset("t_master_us_quat", data=np.arange(n, dtype=np.uint64) * 2500)
        g.create_dataset("quat_report_id", data=np.full(n, 0x08, np.uint8))
        g.create_dataset("linear_accel", data=np.tile(np.float32([1.0, 0, 0]), (n, 1)))
        g.create_dataset("t_master_us_linear_accel", data=np.arange(n, dtype=np.uint64) * 2500)
        T = np.eye(4)
        T[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # 90 deg about y
        f.create_dataset("calibration/imu_mta/mounting_transform", data=T)
    mp = pytest.MonkeyPatch()
    mp.setattr(vs, "world_linear_accel", spy)
    try:
        trajs = vs.load_imu_trajectories(h5, [])
    finally:
        mp.undo()
    assert [t.name for t in trajs] == ["imu_MTA_zupt"]
    np.testing.assert_allclose(seen["a_world"], [[0.0, 1.0, 0.0]] * n, atol=1e-6)


def test_quats_of_other_fusion_reports_are_ignored():
    from jtsmm.tools.visualize_session import _primary_quat_rows

    rid = np.array([0x08, 0x09] * 5, dtype=np.uint8)
    imu = SimpleNamespace(n=10, quat_report_id=rid, fusion_mode="game_rotation_vector")
    np.testing.assert_array_equal(_primary_quat_rows(imu), rid == 0x08)
    rid2 = np.array([0x09, 0x09, 0x08], dtype=np.uint8)  # unknown mode: most frequent
    imu2 = SimpleNamespace(n=3, quat_report_id=rid2, fusion_mode="unknown")
    np.testing.assert_array_equal(_primary_quat_rows(imu2), [True, True, False])


# --------------------------------------------------------------------- gaps
def test_optical_trajectory_breaks_at_dropped_frames():
    from jtsmm.tools.visualize_session import NamedTrajectory, _gapped_xyz

    k = np.r_[0:10, 14:20]  # 30 fps, frames 10..13 dropped
    T = np.tile(np.eye(4), (k.size, 1, 1))
    T[:, 0, 3] = k
    tr = NamedTrajectory("optical_MP", T, k * 33_333.0, 30.0)
    t, xyz = _gapped_xyz(tr, 0.0)
    assert t.size == k.size + 1 and np.isnan(t[10]) and np.isnan(xyz[10]).all()


def test_emg_downsample_breaks_at_sample_gap():
    from jtsmm.tools.visualize_session import _emg_downsampled

    t = np.r_[np.arange(0, 2000), np.arange(3600, 6000)] / 2000.0  # 0.8 s hole
    y = np.sin(t * 40)
    td, yd = _emg_downsampled(t, y, 2000.0, max_points=400)
    gap = np.flatnonzero(np.isnan(td))
    assert gap.size == 1 and np.isnan(yd[gap]).all()
    assert np.nanmax(td[: gap[0]]) < 1.0 <= np.nanmin(td[gap[0] + 1 :])


def _phase_session(path: Path, *, drop=(20, 21, 22)) -> None:
    n = 60
    trig = 1_000_000 + np.arange(n, dtype=np.int64) * 33_333
    keep = np.setdiff1d(np.arange(n), drop)
    poses = np.tile(np.eye(4, dtype=np.float32), (keep.size, 1, 1))
    poses[:, 0, 3] = keep
    with h5py.File(path, "w") as f:
        g = f.create_group("optical")
        g.create_dataset("t_master_us", data=trig[keep].astype(np.uint64))
        for ds in ("pose_mta", "pose_cra", "pose_ct"):
            g.create_dataset(ds, data=poses)
        tg = f.create_group("triggers")
        tg.create_dataset("t_master_us", data=trig.astype(np.uint64))
        tg.create_dataset("trigger_id", data=np.arange(1, n + 1, dtype=np.uint64))
        ph = f.create_group("phases")
        ph.create_dataset("name", data=np.array(["open_close"], dtype=h5py.string_dtype()))
        ph.create_dataset("kind", data=np.array(["experiment"], dtype=h5py.string_dtype()))
        ph.create_dataset("t_start_us", data=np.array([trig[0]], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([trig[-1] + 1], dtype=np.uint64))


def test_visualize_optical_is_on_trigger_time(tmp_path: Path):
    """30 fps with dropped frames: grid rows, trigger times, NaN in the holes."""
    from jtsmm.tools.visualize_session import load_optical_with_jts

    _phase_session(tmp_path / "s.h5")
    mp = next(t for t in load_optical_with_jts(tmp_path / "s.h5") if t.name == "optical_MP")
    trig = 1_000_000 + np.arange(60) * 33_333.0
    assert mp.T.shape[0] == 60 and mp.sample_rate_hz == pytest.approx(30.0, rel=1e-3)
    np.testing.assert_allclose(mp.t_us, trig, atol=1)
    assert np.isnan(mp.T[20:23, :3]).all()
    np.testing.assert_allclose(mp.T[23, 0, 3], 23.0)


def test_plot_mp_xyz_breaks_line_at_gap(tmp_path: Path):
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import load_trajectory, plot_xyz

    _phase_session(tmp_path / "s.h5")
    traj = load_trajectory(tmp_path / "s.h5")
    assert traj.period_s == pytest.approx(0.033333)
    plot_xyz(traj, title="x", save=None, show=True, show_phases=False)
    y = plt.gcf().axes[0].lines[0].get_ydata()
    assert np.isnan(y).sum() == 1
    plt.close("all")


def test_plot_optical_phases_breaks_line_at_gap(tmp_path: Path):
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_optical_phases import load_optical_bodies, plot_phase, slice_phase
    from jtsmm.tools.plot_mp_trajectory import load_phases

    _phase_session(tmp_path / "s.h5")
    with h5py.File(tmp_path / "s.h5", "r") as f:
        [phase] = load_phases(f)
    sliced = slice_phase(load_optical_bodies(tmp_path / "s.h5", bodies=["MP"]), phase)
    plot_phase(sliced, title="x", ylabel=("x", "y", "z"), save=None, show=True, period_s=1 / 30)
    y = plt.gcf().axes[0].lines[0].get_ydata()
    assert np.isnan(y).sum() == 1
    plt.close("all")


# --------------------------------------------------------------------- errors
def test_plot_mp_without_optical_exits_cleanly(tmp_path: Path, capsys):
    from jtsmm.tools.plot_mp_trajectory import main

    h5 = tmp_path / "emg_only.h5"
    with h5py.File(h5, "w") as f:
        f.create_group("meta")
    rc = main([str(h5), "--out", str(tmp_path / "o"), "--no-show"])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:") and "Traceback" not in err


# --------------------------------------------------------------------- EMG
def _emg_session(path: Path, fs: int = 2000) -> np.ndarray:
    t_s = np.r_[np.arange(0, 2 * fs), np.arange(int(2.8 * fs), 6 * fs)] / fs  # 0.8 s hole
    rng = np.random.default_rng(1)
    uv = np.zeros((t_s.size, 8), np.float32)
    uv[:] = 20_000.0  # electrode DC offset
    uv += (500.0 * np.sin(2 * np.pi * 50.0 * t_s))[:, None]  # mains
    uv += rng.normal(0.0, 10.0, uv.shape)  # "EMG"
    uv[:, 6] = 0.0  # bias_debug
    with h5py.File(path, "w") as f:
        g = f.create_group("emg")
        g.create_dataset("samples_uv", data=uv)
        g.create_dataset("t_master_us", data=(t_s * 1e6).astype(np.uint64))
        g.attrs["sample_rate_sps"] = fs
        g.attrs["gain"] = 6
        g.attrs["channel_names"] = np.array(FULL_CHANNEL_NAMES, dtype="S")
    return t_s


def test_visualize_emg_is_filtered_and_skips_bias_debug(tmp_path: Path):
    from jtsmm.tools.visualize_session import load_emg

    t_s = _emg_session(tmp_path / "s.h5")
    emg = load_emg(tmp_path / "s.h5")
    assert "bias_debug" not in emg.channel_names and len(emg.channel_names) == 7
    settled = t_s > 1.0
    y = emg.samples_uv[settled, 0]
    assert abs(float(np.mean(y))) < 20.0  # DC gone
    assert float(np.std(y)) < 60.0        # mains (354 uV rms) removed


def test_plot_emg_full_ylim_ignores_filter_startup():
    from jtsmm.tools.plot_emg import _settled_ylim

    t = np.linspace(0.0, 10.0, 1001)
    y = np.sin(t)
    y[:20] = 1e4  # start-up transient
    lo, hi = _settled_ylim(t, (y,), 1.0)
    assert hi < 2.0 and lo > -2.0
    assert _settled_ylim(t, (y,), 0.0) is None
