# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Tests for experiment recording path layout (--persist-frames)."""

from __future__ import annotations

from pathlib import Path

import pytest

from optical.paths import (
    SESSION_H5_NAME,
    experiment_recording_dir,
    path_slug,
    resolve_persist_recording,
)
from optical.tracking.experiment import ExperimentPlan


def _resolve_persist_outputs(**kw):
    # tracking.run imports calibration.multicam, which needs cv2.
    pytest.importorskip("cv2")
    from optical.tracking.run import _resolve_persist_outputs as fn

    return fn(**kw)


def test_path_slug_sanitizes():
    assert path_slug("Subject 01 / A") == "subject_01_a"
    assert path_slug("") == "unnamed"
    assert path_slug(None, fallback="x") == "x"


def test_experiment_recording_dir_leaf_shape(tmp_path: Path):
    leaf = experiment_recording_dir(
        tmp_path / "{experiment_folder}" / "recordings",
        experiment_folder="5_1_1_pnp_vs_triangulation",
        experiment_name="subject01_session1",
        condition="near",
        stamp="20260914T100000Z",
    )
    assert leaf.name == "subject01_session1_near_20260914T100000Z"
    assert leaf.parent.name == "recordings"
    assert leaf.parent.parent.name == "5_1_1_pnp_vs_triangulation"


def test_experiment_recording_dir_expands_home_template():
    leaf = experiment_recording_dir(
        "~/experiments/{experiment_folder}/recordings",
        experiment_folder="5_1_2_marker_size_tradeoff",
        experiment_name="s01",
        condition="mid",
        stamp="20260101T000000Z",
    )
    assert leaf == Path.home() / (
        "experiments/5_1_2_marker_size_tradeoff/recordings/"
        "s01_mid_20260101T000000Z"
    )


def test_resolve_persist_recording_from_doc():
    doc = {
        "paths": {"recordings_dir": "/tmp/exp/{experiment_folder}/recordings"},
        "experiment": {
            "name": "sub01",
            "folder": "5_1_1_pnp_vs_triangulation",
        },
    }
    leaf = resolve_persist_recording(
        doc, experiment_name=None, condition="static", stamp="20260914T120000Z"
    )
    assert leaf == Path(
        "/tmp/exp/5_1_1_pnp_vs_triangulation/recordings/"
        "sub01_static_20260914T120000Z"
    )


def test_resolve_persist_outputs_uses_yaml_condition(tmp_path: Path):
    cfg = tmp_path / "optical.yaml"
    cfg.write_text(
        "\n".join(
            [
                "paths:",
                f"  recordings_dir: {tmp_path / 'experiments'}/{{experiment_folder}}/recordings",
                "experiment:",
                "  name: subject01_session1",
                "  folder: 5_1_1_pnp_vs_triangulation",
                "  condition: static",
                "  phases: [open_close]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan = ExperimentPlan.from_dict(
        {"name": "subject01_session1", "phases": ["open_close"]}
    )
    h5, frames = _resolve_persist_outputs(
        tracker_config_path=cfg,
        plan=plan,
        persist_frames=True,
        condition=None,  # fall back to experiment.condition
        output_path=None,
        record_frames_dir=None,
    )
    assert frames is not None
    assert "static" in frames.name
    assert frames.name.startswith("subject01_session1_static_")
    assert h5 == frames / SESSION_H5_NAME


def test_resolve_persist_outputs_noop_without_flag():
    plan = ExperimentPlan.from_dict({"name": "s", "phases": ["open_close"]})
    h5, frames = _resolve_persist_outputs(
        tracker_config_path=Path("/unused"),
        plan=plan,
        persist_frames=False,
        condition=None,
        output_path=Path("/out.h5"),
        record_frames_dir=Path("/frames"),
    )
    assert h5 == Path("/out.h5")
    assert frames == Path("/frames")


def test_snapshot_persist_calibration_copies_bundle_and_config(tmp_path: Path):
    from optical.paths import (
        CALIBRATION_BUNDLE_SNAPSHOT_NAME,
        OPTICAL_CONFIG_SNAPSHOT_NAME,
        recording_calibration_bundle,
        recording_optical_config,
        snapshot_persist_calibration,
    )
    from optical.tracking.frame_store import load_session_meta, write_session_meta

    src_bundle = tmp_path / "src_bundle.h5"
    src_bundle.write_bytes(b"calib-bytes")
    src_cfg = tmp_path / "optical.yaml"
    src_cfg.write_text("tracker: {}\n", encoding="utf-8")
    leaf = tmp_path / "leaf"
    arts = snapshot_persist_calibration(
        leaf, bundle_path=src_bundle, config_path=src_cfg
    )
    assert arts["calibration_bundle"].read_bytes() == b"calib-bytes"
    assert arts["optical_config"].read_text(encoding="utf-8") == "tracker: {}\n"
    assert recording_calibration_bundle(leaf) == leaf / CALIBRATION_BUNDLE_SNAPSHOT_NAME
    assert recording_optical_config(leaf) == leaf / OPTICAL_CONFIG_SNAPSHOT_NAME

    write_session_meta(
        leaf,
        cam_ids=["cam0"],
        image_size=(728, 544),
        stream="lores_y",
        calibration_bundle=CALIBRATION_BUNDLE_SNAPSHOT_NAME,
        calibration_bundle_source=str(src_bundle),
    )
    meta = load_session_meta(leaf)
    assert meta["calibration_bundle"] == CALIBRATION_BUNDLE_SNAPSHOT_NAME
    assert meta["calibration_bundle_source"] == str(src_bundle)