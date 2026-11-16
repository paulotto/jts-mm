# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Readers of the reference systems (biomech.io.reference) on tiny synthetic files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from biomech.io.reference import muscle_from_label, read_delsys_csv, read_qtm_mat
from biomech.validation.registration import make_pose
from biomech.validation.synthetic import write_qtm_mat, write_trigno_csv


def _poses(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    R = Rotation.from_rotvec(rng.normal(size=(n, 3))).as_matrix()
    return make_pose(R, rng.normal(size=(n, 3)) * 100)


def test_read_qtm_mat_roundtrip(tmp_path: Path):
    n = 50
    ct, mp = _poses(n, 1), _poses(n, 2)
    ct[10:15] = np.nan                                     # untracked frames
    hp = np.zeros((n, 4, 4))                               # never tracked: all zero
    markers = {"CT - 1": np.random.default_rng(3).normal(size=(n, 3))}
    path = tmp_path / "take.mat"
    write_qtm_mat(path, "take_01", 200.0, {"CT": ct, "MP": mp, "HP": hp}, start_frame=3,
                  markers=markers)
    q = read_qtm_mat(path)
    assert q.take == "take_01" and q.fs_hz == 200.0 and q.n_frames == n
    assert q.t_s[0] == pytest.approx(2 / 200.0) and np.allclose(np.diff(q.t_s), 0.005)
    assert set(q.bodies) == {"CT", "MP", "HP"}
    ok = q.bodies["CT"].valid()
    assert ok.sum() == n - 5 and not ok[10:15].any()
    np.testing.assert_allclose(q.pose("CT")[ok], ct[ok], atol=1e-9)    # column-major handled
    np.testing.assert_allclose(q.pose("MP"), mp, atol=1e-9)
    assert not q.bodies["HP"].valid().any()                # all-zero rows -> NaN
    assert np.isnan(q.bodies["CT"].residual[12])
    np.testing.assert_allclose(q.markers["CT - 1"], markers["CT - 1"])
    dets = np.linalg.det(q.pose("MP")[:, :3, :3])
    np.testing.assert_allclose(dets, 1.0, atol=1e-9)


@pytest.mark.parametrize("label,muscle", [
    ("Dig_Left", "digastric_l"), ("Mas_Right", "masseter_r"), ("Tem_Left", "temporalis_l"),
    ("TEM R", "temporalis_r"), ("Left_Masseter", "masseter_l"), ("Temporalis-Right",
                                                                   "temporalis_r"),
    ("EMG 1", None), ("Dig", None)])
def test_muscle_from_label(label, muscle):
    assert muscle_from_label(label) == muscle


def test_read_delsys_csv_layout(tmp_path: Path):
    rng = np.random.default_rng(0)
    fast, slow = 4370.3704, 2148.1481
    a = rng.normal(size=437) * 0.05                       # mV
    b = rng.normal(size=214) * 0.02
    c = rng.normal(size=437) * 0.01
    path = tmp_path / "take.csv"
    # day-specific sensor order: the reader maps by label, never by position
    write_trigno_csv(path, [("Mas_Right", "Avanti Sensor 11 (1)", 84, fast, a),
                            ("Dig_Left", "Avanti Sensor 10 (2)", 40, slow, b),
                            ("Tem_Left", "Avanti Sensor (3)", 84, fast, c)],
                     start="2000-01-02 03:04:05")
    d = read_delsys_csv(path)
    assert d.header["Application"].startswith("Trigno Discover")
    assert d.start == "2000-01-02 03:04:05" and d.length_s == pytest.approx(0.1, abs=1e-3)
    assert sorted(d.keys()) == ["digastric_l", "masseter_r", "temporalis_l"]
    t, uv = d["masseter_r"]
    assert len(t) == 437 and t[1] == pytest.approx(1 / fast, rel=1e-4)
    np.testing.assert_allclose(uv, a * 1e3, atol=1e-3)    # mV -> µV
    ch = d.channels["digastric_l"]
    assert ch.fs_hz == pytest.approx(slow) and ch.mode == "40" and len(ch.t_s) == 214
    assert ch.sensor == "Avanti Sensor 10 (2)"
    np.testing.assert_allclose(ch.uv, b * 1e3, atol=1e-3)
    only = read_delsys_csv(path, muscles=["temporalis_l"])
    assert only.keys() == ["temporalis_l"]


def test_read_delsys_csv_rejects_other_files(tmp_path: Path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Time Series"):
        read_delsys_csv(p)
