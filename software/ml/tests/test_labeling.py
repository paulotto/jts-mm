# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for labeling trajectory loading and figures."""

from __future__ import annotations

import h5py
import numpy as np

from ml.config.classes import MUSCLE_CHANNEL_NAMES
from ml.labeling.figures import build_label_bar, build_main_figure, parse_selection_range
from ml.labeling.phases import apply_label_to_phase, build_phase_views, list_experiment_phases
from ml.labeling.trajectory import load_labeling_grid


def _write_min_session(path, *, n_optical: int = 120, n_emg: int = 1200) -> None:
    t_opt = np.arange(n_optical, dtype=np.uint64) * 50_000  # 20 Hz-ish spacing
    pose = np.tile(np.eye(4, dtype=np.float32), (n_optical, 1, 1))
    pose[:, 0, 3] = np.linspace(0, 5, n_optical, dtype=np.float32)
    pose[:, 1, 3] = np.sin(np.linspace(0, 2 * np.pi, n_optical)).astype(np.float32)
    pose[:, 2, 3] = np.cos(np.linspace(0, 2 * np.pi, n_optical)).astype(np.float32)

    phase_lo = min(10, max(0, n_optical // 10))
    phase_hi = min(max(phase_lo + 1, n_optical - 10), n_optical - 1)

    with h5py.File(path, "w") as f:
        meta = f.create_group("meta")
        meta.attrs["session_id"] = "test_session"
        meta.attrs["subject_id"] = "SUBJ1"

        emg = f.create_group("emg")
        emg.create_dataset("samples_uv", data=np.zeros((n_emg, 8), dtype=np.float32))
        emg.create_dataset("t_master_us", data=np.arange(n_emg, dtype=np.uint64) * 1000)
        emg.attrs["sample_rate_sps"] = 1000
        emg.attrs["channel_names"] = np.array(
            list(MUSCLE_CHANNEL_NAMES) + ["bias_debug", "drl"], dtype="S"
        )

        optical = f.create_group("optical")
        optical.create_dataset("t_master_us", data=t_opt)
        optical.create_dataset("pose_mta", data=pose)
        optical.create_dataset("pose_cra", data=pose.copy())
        optical.create_dataset("pose_ct", data=pose.copy())

        phases = f.create_group("phases")
        phases.create_dataset("name", data=np.array(["open_close"], dtype="S"))
        phases.create_dataset("kind", data=np.array(["experiment"], dtype="S"))
        phases.create_dataset("t_start_us", data=np.array([t_opt[phase_lo]], dtype=np.uint64))
        phases.create_dataset("t_end_us", data=np.array([t_opt[phase_hi]], dtype=np.uint64))


def test_list_experiment_phases_order() -> None:
    from ml.io.session import PhaseInterval

    phases = [
        PhaseInterval("left_right", "experiment", 200, 300),
        PhaseInterval("open_close", "experiment", 100, 199),
        PhaseInterval("mand_point_1", "calibration_mandibular", 0, 50),
    ]
    names = [p.name for p in list_experiment_phases(phases)]
    assert names == ["open_close", "left_right"]


def test_build_phase_views_and_apply(tmp_path) -> None:
    session = tmp_path / "session.h5"
    _write_min_session(session)
    grid = load_labeling_grid(session)
    views = build_phase_views(grid)
    assert len(views) == 1
    labels = [0] * len(grid.t_s)
    view = views[0]
    updated, count = apply_label_to_phase(
        labels, view, t_start=0.0, t_end=float(view.t_s[-1]) * 0.5, class_id=1
    )
    assert count > 0
    assert any(v == 1 for v in updated)


def test_load_labeling_grid_embedded_jts(tmp_path) -> None:
    session = tmp_path / "session.h5"
    _write_min_session(session, n_optical=590)

    with h5py.File(session, "a") as f:
        jts = f.create_group("jts")
        grp = jts.create_group("T_model_origin_mand_landmark_t_smooth")
        xyz = np.stack(
            [
                np.linspace(0, 10, 590, dtype=np.float32),
                np.sin(np.linspace(0, 4 * np.pi, 590)).astype(np.float32),
                np.cos(np.linspace(0, 4 * np.pi, 590)).astype(np.float32),
            ],
            axis=1,
        )
        grp.create_dataset("translations", data=xyz)
        grp.create_dataset("translations_smooth", data=xyz)
        grp.attrs["sample_rate"] = 20.0
        grp.attrs["unit"] = "m"

    grid = load_labeling_grid(session)
    assert grid.trajectory_source == "jts"
    assert grid.jts_group == "jts/T_model_origin_mand_landmark_t_smooth"
    assert len(grid.t_master_us) == 590


def test_resolve_jts_source_prefers_embedded_when_no_sidecar(tmp_path) -> None:
    session = tmp_path / "session.h5"
    _write_min_session(session, n_optical=40)
    with h5py.File(session, "a") as f:
        jts = f.create_group("jts")
        grp = jts.create_group("T_model_origin_mand_landmark_t")
        grp.create_dataset("translations", data=np.zeros((40, 3), dtype=np.float32))
        grp.attrs["sample_rate"] = 20.0
        grp.attrs["unit"] = "m"

    from ml.labeling.trajectory import resolve_jts_source

    source = resolve_jts_source(session)
    assert source is not None
    assert source.parent_group == "jts"
    assert source.trajectory_group == "T_model_origin_mand_landmark_t"


def test_load_labeling_grid_optical_fallback(tmp_path) -> None:
    session = tmp_path / "session.h5"
    _write_min_session(session)
    grid = load_labeling_grid(session)
    assert grid.trajectory_source == "optical_fallback"
    assert len(grid.t_master_us) == 120
    assert grid.translations_smooth.shape == (120, 3)
    assert grid.t_s[0] == 0.0


def test_build_main_figure_runs() -> None:
    n = 50
    t_s = np.linspace(0, 1, n)
    xyz = np.random.randn(n, 3)
    labels = np.zeros(n, dtype=int)
    labels[10:20] = 1
    fig = build_main_figure(t_s, xyz, xyz, labels)
    assert len(fig.data) >= 3


def test_parse_selection_range_box() -> None:
    x0, x1 = parse_selection_range({"range": {"x": [0.2, 0.8]}})
    assert x0 == 0.2
    assert x1 == 0.8


def test_build_label_bar_percentages() -> None:
    labels = np.array([0, 0, 1, 1], dtype=int)
    fig = build_label_bar(labels)
    assert fig.data[0].x[1] == 50.0
