# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""One frame-interval convention end to end: recorder -> config -> JTS.

JTS ``frame_interval`` is ``[start, end]`` with both ends inclusive. The
recorder's ``optical_frame_intervals`` must produce exactly the frames inside
a phase, and every JTS consumer must read back the same count.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jtsmm.capture.phase_walker import TimedPhase, optical_frame_intervals
from jtsmm.tools.run_jts_analysis import ensure_jts_on_path, format_calib_lock_report


def _jts():
    ensure_jts_on_path()
    from jts import qualisys as qtm

    return qtm


def test_phase_rows_equal_jts_frames():
    qtm = _jts()
    t_opt = np.arange(10, dtype=np.uint64) * 1000
    # Phase [2000, 6000) contains rows 2, 3, 4, 5.
    [(_n, _k, a, b)] = optical_frame_intervals(
        [TimedPhase("p", "experiment", 2000, 6000, "exact", "exact")], t_opt
    )
    assert (a, b) == (2, 5)

    pos = np.column_stack([np.arange(10.0), np.zeros(10), np.zeros(10)])
    rot = np.tile(np.eye(3), (10, 1, 1))
    rb1 = qtm.RigidBody(name="A", positions=np.zeros((10, 3)), rotations=rot)
    rb2 = qtm.RigidBody(name="B", positions=pos, rotations=rot)

    T = qtm.T_from_rigid_body(rb2, (a, b))
    rel = qtm.QualysisData.compute_relative_transformations([rb1, rb2], [(a, b)])
    rel = rel[f"{a}-{b}"]["A-B"]
    assert T.shape[0] == rel.shape[0] == 4
    np.testing.assert_allclose(T[:, 0, 3], [2.0, 3.0, 4.0, 5.0])
    np.testing.assert_allclose(rel[:, 0, 3], [2.0, 3.0, 4.0, 5.0])


def test_calib_lock_report_counts_same_frames(tmp_path: Path):
    n = 10
    poses = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1))
    with h5py.File(tmp_path / "s.h5", "w") as f:
        g = f.create_group("optical")
        g.create_dataset("t_master_us", data=np.arange(n, dtype=np.uint64) * 1000)
        for ds in ("pose_cra", "pose_mta", "pose_ct"):
            g.create_dataset(ds, data=poses)
    cfg = {"analysis": {"calibration": {"mandibular": {
        "rigid_bodies": ["MP", "CT"],
        "points": [{"name": "mand_point_1", "frame_interval": [2, 5]}],
    }}}}
    text = format_calib_lock_report(tmp_path / "s.h5", cfg)
    assert "4/4" in text

