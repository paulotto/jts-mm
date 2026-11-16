# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""MP world-frame XYZ + 3D trajectory plots."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.helpers import make_session_h5


def _add_phases_with_calib(h5: Path) -> None:
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        assert t.size >= 12
        n = t.size
        cuts = [0, n // 4, n // 2, n - 1]
        ph = f.create_group("phases")
        ph.create_dataset(
            "name",
            data=np.array(
                ["mand_point_1", "open_close", "chewing"], dtype=h5py.string_dtype()
            ),
        )
        ph.create_dataset(
            "kind",
            data=np.array(
                ["calibration_mandibular", "experiment", "experiment"],
                dtype=h5py.string_dtype(),
            ),
        )
        ph.create_dataset(
            "t_start_us",
            data=np.array([int(t[cuts[0]]), int(t[cuts[1]]), int(t[cuts[2]])], dtype=np.uint64),
        )
        ph.create_dataset(
            "t_end_us",
            data=np.array([int(t[cuts[1]]), int(t[cuts[2]]), int(t[cuts[3]])], dtype=np.uint64),
        )


def test_motion_phases_drop_calibration():
    from jtsmm.tools.plot_mp_trajectory import (
        PhaseInterval,
        is_calibration_phase,
        motion_phases,
    )

    phases = [
        PhaseInterval("mand_point_1", "calibration_mandibular", 0, 10),
        PhaseInterval("max_point_2", "calibration_maxillary", 10, 20),
        PhaseInterval("open_close", "experiment", 20, 30),
        PhaseInterval("mvc_closing", "mvc", 30, 40),
        PhaseInterval("rest", "experiment", 40, 50),
    ]
    assert [p.name for p in motion_phases(phases)] == [
        "open_close",
        "mvc_closing",
        "rest",
    ]
    assert is_calibration_phase(phases[0])
    assert not is_calibration_phase(phases[3])


def test_from_rest_phases_starts_at_rest():
    from jtsmm.tools.plot_mp_trajectory import PhaseInterval, from_rest_phases

    phases = [
        PhaseInterval("mand_point_1", "calibration_mandibular", 0, 10),
        PhaseInterval("mvc_opening", "mvc", 10, 20),
        PhaseInterval("rest", "experiment", 20, 30),
        PhaseInterval("open_close", "experiment", 30, 40),
    ]
    assert [p.name for p in from_rest_phases(phases)] == ["rest", "open_close"]
    no_rest = phases[:2] + [PhaseInterval("chewing", "experiment", 20, 30)]
    assert [p.name for p in from_rest_phases(no_rest)] == ["mvc_opening", "chewing"]


def test_from_rest_phases_drop_last():
    from jtsmm.tools.plot_mp_trajectory import PhaseInterval, from_rest_phases_drop_last

    phases = [
        PhaseInterval("mand_point_1", "calibration_mandibular", 0, 10),
        PhaseInterval("rest", "experiment", 20, 30),
        PhaseInterval("open_close", "experiment", 30, 40),
        PhaseInterval("free_motion", "experiment", 40, 50),
    ]
    assert [p.name for p in from_rest_phases_drop_last(phases)] == ["rest", "open_close"]
    only_rest = phases[:2]
    assert from_rest_phases_drop_last(only_rest) == []


def test_phase_xyz_stays_inside_interval(tmp_path: Path):
    from jtsmm.tools.plot_mp_trajectory import load_phases, load_trajectory, phase_xyz

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    traj = load_trajectory(h5, body="MP")
    with h5py.File(h5, "r") as f:
        phases = load_phases(f)
    calib = next(p for p in phases if p.name == "mand_point_1")
    motion = next(p for p in phases if p.name == "open_close")
    xyz_c = phase_xyz(traj, calib)
    xyz_m = phase_xyz(traj, motion)
    assert xyz_c.shape[0] >= 2
    assert xyz_m.shape[0] >= 2
    assert np.isfinite(xyz_m).all()


def test_plot_xyz_legend_omits_calibration(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import load_trajectory, plot_xyz

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    traj = load_trajectory(h5, body="MP")
    plot_xyz(traj, title="xyz", save=None, show=True, show_phases=True)
    ax = plt.gcf().axes[0]
    labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
    assert "mand_point_1" not in labels
    assert "open_close" in labels
    assert "chewing" in labels
    plt.close("all")


def test_combined_3d_is_single_color_without_legend(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import (
        _3D_AZIM,
        _3D_ELEV,
        _COMBINED_3D_COLOR,
        load_trajectory,
        plot_xyz_3d_combined,
    )

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    traj = load_trajectory(h5, body="MP")
    plot_xyz_3d_combined(traj, title="combined", save=None, show=True)
    fig = plt.gcf()
    ax = fig.axes[0]
    colors = {line.get_color() for line in ax.lines}
    assert colors == {_COMBINED_3D_COLOR}
    assert ax.get_legend() is None
    assert ax.elev == _3D_ELEV
    assert ax.azim == _3D_AZIM
    assert ax._axis_names[ax._vertical_axis] == "y"
    plt.close(fig)


def test_combined_3d_can_omit_last_phase(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import (
        from_rest_phases_drop_last,
        load_trajectory,
        plot_xyz_3d_combined,
    )

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    traj = load_trajectory(h5, body="MP")
    no_last = from_rest_phases_drop_last(traj.phases)
    assert [p.name for p in no_last] == ["open_close"]
    plot_xyz_3d_combined(
        traj, title="no last", save=None, show=True, phases=no_last
    )
    fig = plt.gcf()
    points = sum(len(line.get_xdata()) for line in fig.axes[0].lines)
    plot_xyz_3d_combined(traj, title="all", save=None, show=True)
    fig_all = plt.gcf()
    points_all = sum(len(line.get_xdata()) for line in fig_all.axes[0].lines)
    assert points < points_all
    plt.close("all")


def test_cli_writes_2d_and_3d_pngs(tmp_path: Path, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_mp_trajectory import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    rc = main([str(h5), "--out", str(out), "--no-show"])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == [
        "session_mp_xyz.png",
        "session_mp_xyz_3d.png",
        "session_mp_xyz_3d_no_last.png",
        "session_mp_xyz_3d_phases.png",
        "session_mp_xyz_landmarks.png",
    ]
    text = capsys.readouterr().out
    assert "open_close" in text
    assert "chewing" in text
    assert "3D motion phases: 2" not in text
    assert "plot from rest: 2" in text


def test_plot_xyz_starts_at_rest(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import load_trajectory, plot_xyz

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.8, emg_samples=80, optical=True)
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        n = t.size
        cuts = [0, n // 5, 2 * n // 5, 3 * n // 5, n - 1]
        ph = f.create_group("phases")
        ph.create_dataset(
            "name",
            data=np.array(
                ["mand_point_1", "mvc_opening", "rest", "open_close"],
                dtype=h5py.string_dtype(),
            ),
        )
        ph.create_dataset(
            "kind",
            data=np.array(
                ["calibration_mandibular", "mvc", "experiment", "experiment"],
                dtype=h5py.string_dtype(),
            ),
        )
        ph.create_dataset(
            "t_start_us",
            data=np.array([int(t[c]) for c in cuts[:-1]], dtype=np.uint64),
        )
        ph.create_dataset(
            "t_end_us",
            data=np.array([int(t[c]) for c in cuts[1:]], dtype=np.uint64),
        )

    traj = load_trajectory(h5, body="MP")
    rest = next(p for p in traj.phases if p.name == "rest")
    plot_xyz(traj, title="xyz", save=None, show=True, show_phases=True)
    ax = plt.gcf().axes[-1]
    labels = [txt.get_text() for txt in plt.gcf().axes[0].get_legend().get_texts()]
    assert "mvc_opening" not in labels
    assert "mand_point_1" not in labels
    assert "rest" in labels
    assert "open_close" in labels
    t_line = ax.lines[0].get_xdata()
    t0 = float(traj.t0_us)
    rest_s = (float(rest.t_start_us) - t0) * 1e-6
    assert float(np.nanmin(t_line)) >= rest_s - 1e-6
    plt.close("all")


def test_cli_save_stem_derives_3d_names(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_mp_trajectory import main

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    save = tmp_path / "mp_xyz.png"
    rc = main([str(h5), "--save", str(save), "--no-show"])
    assert rc == 0
    assert save.is_file()
    assert (tmp_path / "mp_xyz_3d.png").is_file()
    assert (tmp_path / "mp_xyz_3d_no_last.png").is_file()
    assert (tmp_path / "mp_xyz_3d_phases.png").is_file()
    assert (tmp_path / "mp_xyz_landmarks.png").is_file()


def test_plot_xyz_marks_cycles(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    pytest.importorskip("scipy")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import (
        cycle_boundary_times_s,
        load_trajectory,
        plot_xyz,
    )

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=8.0, emg_samples=800, optical=True, trig_hz=60)
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        n = t.size
        rest_end = max(n // 8, 8)
        poses = f["optical"]["pose_mta"][()]
        tt = (t.astype(np.float64) - float(t[0])) * 1e-6
        poses[rest_end:, 1, 3] = 20.0 + 25.0 * (
            1.0 - np.cos(2 * np.pi * 1.0 * (tt[rest_end:] - tt[rest_end]))
        )
        f["optical"]["pose_mta"][...] = poses
        ph = f.create_group("phases")
        ph.create_dataset(
            "name",
            data=np.array(["rest", "open_close"], dtype=h5py.string_dtype()),
        )
        ph.create_dataset(
            "kind",
            data=np.array(["experiment", "experiment"], dtype=h5py.string_dtype()),
        )
        ph.create_dataset("t_start_us", data=np.array([int(t[0]), int(t[rest_end])], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([int(t[rest_end]), int(t[-1])], dtype=np.uint64))

    traj = load_trajectory(h5, body="MP")
    bounds = cycle_boundary_times_s(traj)
    assert len(bounds) >= 4
    plot_xyz(traj, title="xyz", save=None, show=True, show_phases=True, show_cycles=True)
    ax = plt.gcf().axes[0]
    dotted = [ln for ln in ax.lines if ln.get_linestyle() == ":"]
    assert len(dotted) >= 4
    labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
    assert "cycles" in labels
    plt.close("all")
    plot_xyz(traj, title="xyz", save=None, show=True, show_cycles=False)
    ax = plt.gcf().axes[0]
    assert "cycles" not in [txt.get_text() for txt in (ax.get_legend().get_texts() if ax.get_legend() else [])]
    plt.close("all")


def test_mp_trajectory_is_relative_to_hp(tmp_path: Path):
    from jtsmm.tools.plot_mp_trajectory import load_trajectory, relative_origin_mm

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    with h5py.File(h5, "a") as f:
        mp = np.asarray(f["optical"]["pose_mta"][()], dtype=np.float64)
        hp = np.asarray(f["optical"]["pose_cra"][()], dtype=np.float64)
        hp[:, :3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        f["optical"]["pose_cra"][...] = hp
        f["optical"]["pose_mta"][...] = mp
        expected = relative_origin_mm(mp, hp)
    traj = load_trajectory(h5, body="MP")
    assert traj.frame == "hp"
    np.testing.assert_allclose(traj.xyz_mm, expected, atol=1e-5)


def test_phase_plots_show_movement_from_rest(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import load_trajectory, plot_xyz_3d_by_phase, rest_offset_mm

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=1.0, emg_samples=80, optical=True)
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        n = t.size
        mid = n // 2
        mp = np.asarray(f["optical"]["pose_mta"][()], dtype=np.float64)
        hp = np.asarray(f["optical"]["pose_cra"][()], dtype=np.float64)
        mp[:, :3, :3] = np.eye(3)
        hp[:, :3, :3] = np.eye(3)
        hp[:, :3, 3] = 0.0
        mp[:, :3, 3] = 0.0
        mp[:mid, 1, 3] = 10.0
        mp[mid:, 1, 3] = 40.0
        f["optical"]["pose_mta"][...] = mp
        f["optical"]["pose_cra"][...] = hp
        ph = f.create_group("phases")
        ph.create_dataset("name", data=np.array(["rest", "open_close"], dtype=h5py.string_dtype()))
        ph.create_dataset("kind", data=np.array(["experiment", "experiment"], dtype=h5py.string_dtype()))
        ph.create_dataset("t_start_us", data=np.array([int(t[0]), int(t[mid])], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([int(t[mid - 1]), int(t[-1])], dtype=np.uint64))

    traj = load_trajectory(h5, body="MP")
    offset = rest_offset_mm(traj)
    assert abs(float(offset[1]) - 10.0) < 1e-6
    plot_xyz_3d_by_phase(traj, title="phases", save=None, show=True)
    axes = [ax for ax in plt.gcf().axes if ax.get_title() == "open_close"]
    assert len(axes) == 1
    y = np.concatenate([line.get_ydata() for line in axes[0].lines])
    y = y[np.isfinite(y)]
    assert abs(float(np.median(y)) - 30.0) < 1e-4
    assert axes[0].get_ylabel() == "Y (mm from rest)"
    plt.close("all")


def test_landmarks_follow_mandible_in_hp_frame(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import load_trajectory, plot_landmarks_3d

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=1.0, emg_samples=80, optical=True)
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        n = t.size
        mid = n // 2
        ph = f.create_group("phases")
        names = ["mand_point_1", "tmj_left", "tmj_right", "rest"]
        ph.create_dataset("name", data=np.array(names, dtype=h5py.string_dtype()))
        ph.create_dataset(
            "kind",
            data=np.array(
                ["calibration_mandibular", "calibration_tmj", "calibration_tmj", "experiment"],
                dtype=h5py.string_dtype(),
            ),
        )
        cuts = [0, n // 5, 2 * n // 5, 3 * n // 5, n - 1]
        ph.create_dataset("t_start_us", data=np.array([int(t[c]) for c in cuts[:-1]], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([int(t[c]) for c in cuts[1:]], dtype=np.uint64))
        for ds in ("pose_mta", "pose_cra", "pose_ct"):
            pose = np.asarray(f["optical"][ds][()], dtype=np.float64)
            pose[:, :3, 3] = 0.0
            f["optical"][ds][...] = pose
        mp = np.asarray(f["optical"]["pose_mta"][()], dtype=np.float64)
        ct = np.asarray(f["optical"]["pose_ct"][()], dtype=np.float64)
        ct[: cuts[1], 0, 3] = 10.0
        ct[cuts[1] : cuts[2], 1, 3] = 30.0
        ct[cuts[2] : cuts[3], 1, 3] = -30.0
        mp[mid:, 0, 3] += 5.0
        f["optical"]["pose_ct"][...] = ct
        f["optical"]["pose_mta"][...] = mp

    traj = load_trajectory(h5, body="MP")
    assert set(traj.landmarks) == {"mand_point_1", "tmj_left", "tmj_right"}
    # Identity rotations: the point in HP is the CT tip of that phase, plus the later MP shift.
    incisal = traj.landmarks["mand_point_1"]
    assert abs(float(incisal[0, 0]) - 10.0) < 1e-4
    assert abs(float(incisal[mid, 0]) - 15.0) < 1e-4
    plot_landmarks_3d(traj, title="landmarks", save=None, show=True)
    ax = plt.gcf().axes[0]
    colors = {line.get_color() for line in ax.lines}
    assert "#c0392b" in colors
    assert colors <= {"#c0392b", "#196f3d", "#1e8449"}
    labels = [txt.get_text() for txt in ax.get_legend().get_texts()]
    assert labels == ["mand_point_1", "tmj_left", "tmj_right"]
    plt.close("all")


def test_model_landmark_3d_matches_jts_view(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import plot_model_landmark_3d

    xyz = np.array(
        [[0.0, -0.12, 0.09], [0.01, -0.11, 0.10], [0.0, -0.13, 0.11]],
        dtype=np.float64,
    )
    plot_model_landmark_3d(
        xyz,
        unit="m",
        title="Jaw Motion Trajectories - 3D View",
        save=None,
        show=True,
    )
    ax = plt.gcf().axes[0]
    assert ax.get_title() == "Jaw Motion Trajectories - 3D View"
    assert ax.get_xlabel() == "X [m]"
    assert ax._axis_names[ax._vertical_axis] == "z"
    assert len(ax.lines) == 1
    plt.close("all")


def test_cli_uses_smoothed_model_landmark(tmp_path: Path, monkeypatch, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools import plot_mp_trajectory as mod

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases_with_calib(h5)
    with h5py.File(h5, "r") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.float64)
    n = t.size
    xyz = np.column_stack([
        np.linspace(0.0, 0.02, n),
        np.full(n, -0.12),
        np.full(n, 0.10),
    ])
    model = mod.ModelTrajectory(
        xyz=xyz,
        unit="m",
        scale=0.001,
        t_us=t,
        hz=40.0,
        T_smooth=np.zeros((n, 4, 4)),
        points={"mand_point_1": xyz, "tmj_left": xyz + 0.01, "tmj_right": xyz - 0.01},
    )
    monkeypatch.setattr(mod, "jts_model_trajectory", lambda _path: model)
    rc = mod.main([str(h5), "--out", str(out), "--no-show", "--no-cycles"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "smoothed mandibular landmark" in text
    assert (out / "session_mp_xyz.png").is_file()
    assert (out / "session_mp_xyz_3d.png").is_file()
    assert (out / "session_mp_xyz_3d_phases.png").is_file()
    assert (out / "session_mp_xyz_landmarks.png").is_file()

    with h5py.File(h5, "r") as f:
        plotted = mod.trajectory_from_model(model, mod.load_phases(f))
    assert plotted.frame == "model"
    assert plotted.unit == "m"
    assert abs(float(np.nanmax(plotted.xyz_mm[:, 0])) - 0.02) < 1e-9


def test_model_phase_axes_are_z_up():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from jtsmm.tools.plot_mp_trajectory import (
        ModelTrajectory,
        PhaseInterval,
        plot_model_phases_3d,
    )

    t = np.arange(8, dtype=np.float64) * 1e6
    xyz = np.column_stack([np.linspace(0, 0.02, 8), np.full(8, -0.12), np.linspace(0.09, 0.11, 8)])
    model = ModelTrajectory(
        xyz=xyz, unit="m", scale=0.001, t_us=t, hz=40.0,
        T_smooth=np.zeros((8, 4, 4)), points={},
    )
    phases = [
        PhaseInterval("rest", "experiment", 0, int(3e6)),
        PhaseInterval("open_close", "experiment", int(4e6), int(7e6)),
    ]
    plot_model_phases_3d(model, phases, title="phases", save=None, show=True)
    axes = [ax for ax in plt.gcf().axes if ax.get_visible()]
    assert len(axes) == 2
    assert axes[0]._axis_names[axes[0]._vertical_axis] == "z"
    assert axes[0].get_xlabel() == "X [m]"
    plt.close("all")


def test_find_cycles_fills_long_gap():
    pytest.importorskip("scipy")
    from jtsmm.tools.plot_mp_trajectory import find_cycles

    fs = 50.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    y = np.zeros_like(t)
    for peak_t, amp in ((1.0, 1.0), (2.0, 1.0), (3.5, 0.12), (5.0, 1.0), (6.0, 1.0), (7.0, 1.0)):
        y += amp * np.exp(-0.5 * ((t - peak_t) / 0.12) ** 2)
    def peak_times(segs):
        return [float(t[a]) for a, _ in segs] + [float(t[segs[-1][1]])]

    filled = peak_times(find_cycles(y, fs, t_s=t, prominence_frac=2.0, fill_frac=0.3))
    plain = peak_times(find_cycles(y, fs, t_s=t, prominence_frac=2.0, fill_frac=50.0))
    assert any(abs(p - 3.5) < 0.15 for p in filled)
    assert not any(abs(p - 3.5) < 0.15 for p in plain)
