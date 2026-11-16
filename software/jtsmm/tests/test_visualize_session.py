# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Offline session visualizer: optical via JTS, IMU ZUPT xyz, all EMG channels."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.helpers import make_session_h5


def test_resolve_session_h5_file_and_dir(tmp_path: Path):
    from jtsmm.tools.visualize_session import resolve_session_h5

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.2, emg_samples=50, optical=True)
    assert resolve_session_h5(h5) == h5.resolve()
    assert resolve_session_h5(tmp_path) == h5.resolve()


def test_resolve_session_h5_missing(tmp_path: Path):
    from jtsmm.tools.visualize_session import resolve_session_h5

    with pytest.raises(FileNotFoundError):
        resolve_session_h5(tmp_path / "nope.h5")
    with pytest.raises(FileNotFoundError):
        resolve_session_h5(tmp_path)


def test_load_emg_all_channels(tmp_path: Path):
    from jtsmm.config.catalog import FULL_CHANNEL_NAMES
    from jtsmm.tools.visualize_session import load_emg

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.5, emg_samples=200, optical=True)
    emg = load_emg(h5)
    assert emg is not None
    # Muscles + DRL; CH7 bias_debug is not an electrode signal and is skipped.
    assert emg.samples_uv.shape[1] == 7
    assert emg.channel_names == [n for n in FULL_CHANNEL_NAMES if n != "bias_debug"]
    assert emg.t_us.shape[0] == emg.samples_uv.shape[0]


def test_imu_zupt_seeds_from_optical_origin(tmp_path: Path):
    from jtsmm.tools.visualize_session import (
        NamedTrajectory,
        _xyz_mm,
        load_imu_trajectories,
    )

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    # Identity optical MP at a known origin — load_imu uses that as the seed.
    seed = NamedTrajectory(
        name="optical_MP",
        T=np.tile(np.eye(4), (2, 1, 1)),
        t_us=np.array([0.0, 1e6]),
        sample_rate_hz=60.0,
        source="optical",
    )
    seed.T[:, :3, 3] = (80.0, 20.0, 450.0)
    trajs = load_imu_trajectories(h5, [seed])
    names = {t.name for t in trajs}
    assert "imu_MTA_zupt" in names
    assert "imu_CRA_zupt" in names
    mta = next(t for t in trajs if t.name == "imu_MTA_zupt")
    xyz = _xyz_mm(mta.T)
    finite = np.isfinite(xyz).all(axis=1)
    assert finite.any()
    # linear_accel is zeros in the fixture, so the estimate stays at the seed.
    np.testing.assert_allclose(xyz[finite][0], (80.0, 20.0, 450.0), atol=1e-3)


def test_derived_pose_preferred_over_zupt(tmp_path: Path):
    from jtsmm.tools.visualize_session import load_imu_trajectories

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.3, emg_samples=60, optical=True)
    n = 5
    T = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))
    T[:, 0, 3] = np.linspace(0.0, 4.0, n)
    with h5py.File(h5, "a") as f:
        g = f.create_group("derived/pose")
        g.create_dataset("pose_mta", data=T)
        g.create_dataset("t_master_us", data=np.arange(n, dtype=np.uint64) * 1000)

    trajs = load_imu_trajectories(h5, [])
    assert len(trajs) == 1
    assert trajs[0].name == "imu_MTA_derived"
    assert trajs[0].source == "derived"
    np.testing.assert_allclose(trajs[0].T[:, 0, 3], np.linspace(0.0, 4.0, n))


def test_load_optical_with_jts(tmp_path: Path):
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.visualize_session import ensure_jts_on_path, load_optical_with_jts

    try:
        ensure_jts_on_path()
    except ImportError:
        pytest.skip("jts (software/jaw_tracking) is not importable")

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.3, emg_samples=60, optical=True)
    trajs = load_optical_with_jts(h5)
    names = {t.name for t in trajs}
    assert names == {"optical_HP", "optical_MP", "optical_CT"}
    mp = next(t for t in trajs if t.name == "optical_MP")
    finite = np.isfinite(mp.T).reshape(len(mp.T), -1).all(axis=1)
    assert finite.any()
    np.testing.assert_allclose(mp.T[finite][0, :3, 3], (80.0, 20.0, 450.0), atol=1e-3)


def test_write_jts_sidecar_and_cli_png(tmp_path: Path, capsys):
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.visualize_session import ensure_jts_on_path, main

    try:
        ensure_jts_on_path()
    except ImportError:
        pytest.skip("jts (software/jaw_tracking) is not importable")

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.25, emg_samples=50, optical=True)
    rc = main(
        [
            str(h5),
            "--out",
            str(out),
            "--no-show",
            "--no-html",
            "--png",
            "--keep-sidecar",
        ]
    )
    assert rc == 0
    sidecar = out / "session_jts_trajectories.h5"
    assert sidecar.is_file()
    with h5py.File(sidecar, "r") as f:
        assert "optical_MP" in f
        assert "translations" in f["optical_MP"]
        assert "rotations" in f["optical_MP"]
        assert "imu_MTA_zupt" in f
    pngs = list(out.glob("*.png"))
    assert any("xyz_optical" in p.name for p in pngs)
    assert any("emg" in p.name for p in pngs)
    out_txt = capsys.readouterr().out
    assert "optical=3" in out_txt
    assert "emg_ch=7" in out_txt


def test_load_session_viz_emg_and_imu(tmp_path: Path):
    from jtsmm.tools.visualize_session import load_session_viz

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.3, emg_samples=60, optical=True)
    viz = load_session_viz(h5)
    assert viz.emg is not None
    assert viz.emg.samples_uv.shape[1] == 7
    assert len(viz.imu) == 2
    assert viz.path == h5.resolve()


def test_cli_writes_plotly_html(tmp_path: Path):
    pytest.importorskip("plotly")
    pytest.importorskip("scipy")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.visualize_session import ensure_jts_on_path, main

    try:
        ensure_jts_on_path()
    except ImportError:
        pytest.skip("jts (software/jaw_tracking) is not importable")

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.2, emg_samples=40, optical=True)
    with h5py.File(h5, "a") as f:
        ph = f.create_group("phases")
        ph.create_dataset("name", data=np.array(["open_close", "chewing"], dtype=h5py.string_dtype()))
        ph.create_dataset("kind", data=np.array(["experiment", "experiment"], dtype=h5py.string_dtype()))
        ph.create_dataset("t_start_us", data=np.array([0, 80_000], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([70_000, 180_000], dtype=np.uint64))
    rc = main([str(h5), "--out", str(out), "--no-show"])
    assert rc == 0
    html = out / "session_viz.html"
    assert html.is_file()
    text = html.read_text(encoding="utf-8")
    assert "optical_MP" in text
    assert "imu_MTA_zupt" in text
    assert "masseter" in text or "EMG" in text or "µV" in text


def test_minmax_downsample_preserves_peaks():
    from jtsmm.tools.visualize_session import _minmax_downsample

    t = np.linspace(0.0, 1.0, 1000)
    y = np.zeros(1000)
    y[100] = 5.0
    y[700] = -3.0
    td, yd = _minmax_downsample(t, y, max_points=40)
    assert td.size <= 40 + 2  # bin count is max_points//2, two samples per bin
    assert yd.max() == pytest.approx(5.0)
    assert yd.min() == pytest.approx(-3.0)


def test_imu_zupt_uses_accel_minus_gravity_without_linear_accel(tmp_path: Path):
    """kinematics_accel sessions: a still IMU that tilts stays put (accel used,
    gravity removed in the world frame)."""
    from jtsmm.imu.gravity_lin_accel import gravity_body
    from jtsmm.tools.visualize_session import _xyz_mm, load_imu_trajectories

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=False)
    with h5py.File(h5, "a") as f:
        for slot in ("imu_mta", "imu_cra"):
            g = f[slot]
            for name in ("linear_accel", "t_master_us_linear_accel"):
                if name in g:
                    del g[name]
            t_q = g["t_master_us_quat"][()]
            ang = np.linspace(0.0, 0.6, t_q.size)  # tilt about x while still
            q = np.column_stack([np.cos(ang / 2), np.sin(ang / 2), 0 * ang, 0 * ang])
            g["quat"][...] = q.astype(np.float32)
            acc = np.stack([gravity_body(qi) for qi in q]).astype(np.float32)
            g.create_dataset("accel", data=acc)
            g.create_dataset("t_master_us_accel", data=t_q)
    trajs = load_imu_trajectories(h5, [])
    mta = next(t for t in trajs if t.name == "imu_MTA_zupt")
    xyz = _xyz_mm(mta.T)
    assert np.isfinite(xyz).all()
    assert np.abs(xyz).max() < 1e-3
