# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""optical.evaluation.precision / .recordings (moved from experiments/_lib/optical_study)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from optical.evaluation import recordings
from optical.evaluation.precision import (
    mean_pose_from_samples,
    residual_stats,
    rotation_residual_deg,
    translation_residual_mm,
)


def _rz(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_residual_stats():
    assert residual_stats(np.array([1.0, 3.0])) == {"r_mean": 2.0, "sigma_r": 1.0, "n": 2}
    one = residual_stats(np.array([1.0]))
    assert one["n"] == 1 and np.isnan(one["r_mean"]) and np.isnan(one["sigma_r"])


def test_translation_residual_mm():
    ts = np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 2.0, 0], [0, -2.0, 0]])
    r = translation_residual_mm(ts)
    assert r == pytest.approx({"r_mean": 1.5, "sigma_r": 0.5, "n": 4})
    assert translation_residual_mm(np.zeros((1, 3)))["n"] == 0


def test_rotation_residual_deg_about_chordal_mean():
    Rs = np.stack([_rz(a) for a in (-0.2, 0.2, -0.4, 0.4)]) @ _rz(30.0)
    r = rotation_residual_deg(Rs)
    assert r["r_mean"] == pytest.approx(0.3, abs=1e-9)
    assert r["sigma_r"] == pytest.approx(0.1, abs=1e-9)
    # sub-milli-degree jitter stays resolvable (atan2 angle, not arccos)
    tiny = rotation_residual_deg(np.stack([_rz(1e-4), _rz(-1e-4)]))
    assert tiny["r_mean"] == pytest.approx(1e-4, rel=1e-3)
    assert rotation_residual_deg(np.eye(3)[None])["n"] == 0


def test_mean_pose_from_samples():
    ts = np.array([[0.0, 0, 0], [2.0, 4.0, 6.0]])
    out = mean_pose_from_samples(ts, np.stack([_rz(-10.0), _rz(10.0)]))
    assert out["t_mm"] == [1.0, 2.0, 3.0] and out["n"] == 2
    assert np.allclose(out["R"], np.eye(3))
    assert mean_pose_from_samples(np.zeros((0, 3)), np.zeros((0, 3, 3))) == {
        "t_mm": None, "R": None, "n": 0}


def test_parse_condition_and_latest_leaf(tmp_path):
    p = recordings.parse_condition_from_leaf
    assert p(Path("precision_test_1_near_20261001T120000Z")) == "near"
    assert p("trueness_far_lateral_d40_20261001T120000Z",
             known=["d40", "far_lateral_d40"]) == "far_lateral_d40"
    assert p("static_near_20261001T120000Z", known=["near", "static_near"]) == "static_near"
    assert p("no_stamp_here") == "no_stamp_here"
    leaves = [Path(n) for n in ("x_near_20261001T120000Z", "x_near_20261002T120000Z",
                                "x_far_20261001T120000Z", "x_other_20261003T120000Z")]
    best = recordings.latest_leaf_per_condition(leaves, ["near", "far", "mid"])
    assert best == {"near": leaves[1], "far": leaves[2]}


def test_iter_recording_leaves(tmp_path):
    for name in ("b_near_20261002T000000Z", "a_far_20261001T000000Z", "no_meta"):
        (tmp_path / name).mkdir()
    (tmp_path / "b_near_20261002T000000Z" / "meta.json").write_text("{}")
    (tmp_path / "a_far_20261001T000000Z" / "meta.json").write_text("{}")
    (tmp_path / "file.txt").write_text("x")
    assert [p.name for p in recordings.iter_recording_leaves(tmp_path)] == [
        "a_far_20261001T000000Z", "b_near_20261002T000000Z"]
    assert recordings.iter_recording_leaves(tmp_path / "missing") == []


def test_iter_session_frames_caps(monkeypatch, tmp_path):
    import optical.tracking.frame_store as fs

    monkeypatch.setattr(fs, "iter_frames", lambda d: iter([(i, {"cam0": i}) for i in range(5)]))
    assert [i for i, _ in recordings.iter_session_frames(tmp_path, max_frames=3)] == [0, 1, 2]
    assert len(list(recordings.iter_session_frames(tmp_path))) == 5


def test_write_json_is_strict(tmp_path):
    out = recordings.write_json(tmp_path / "a" / "r.json",
                                {"x": float("nan"), "a": np.array([1.5]), "i": np.int32(2)})
    assert json.loads(out.read_text()) == {"x": None, "a": [1.5], "i": 2}
