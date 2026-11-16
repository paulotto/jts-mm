# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Time-correct optical frames: trigger grid, phase intervals, JTS export.

Dropped camera frames are absent rows in ``/optical``; these tests use 30 Hz
triggers with holes and check that every consumer works in trigger time, not
row index.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from jtsmm.capture.phase_walker import TimedPhase, optical_frame_intervals
from jtsmm.tools.run_jts_analysis import ensure_jts_on_path
from jtsmm.tools.timegrid import (
    break_gaps,
    estimate_period_us,
    frame_grid_index,
    session_frame_period_us,
    trigger_period_us,
)

PERIOD_US = 33_333
T0_US = 50_000_000


def test_estimate_period_ignores_drops_and_refines():
    k = np.setdiff1d(np.arange(3000), [3, 4, 5, 100, 101, 2000])
    t = T0_US + k * 33_333.3333
    assert estimate_period_us(t) == pytest.approx(33_333.3333, abs=0.01)
    assert estimate_period_us([5]) is None


def test_trigger_period_uses_ids_across_lost_triggers():
    ids = np.array([1, 2, 3, 6, 7])  # triggers 4, 5 lost
    t = T0_US + (ids - 1) * PERIOD_US
    assert trigger_period_us(t, ids) == pytest.approx(PERIOD_US)
    assert trigger_period_us([], None, 30) == pytest.approx(1e6 / 30)


def test_break_gaps_inserts_nan_separator():
    t = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    y = np.arange(10.0).reshape(5, 2)
    t2, y2 = break_gaps(t, y, period=1.0)
    assert t2.shape == (6,) and np.isnan(t2[3])
    assert np.isnan(y2[3]).all()
    np.testing.assert_array_equal(y2[[0, 1, 2, 4, 5]], y)


# --------------------------------------------------------------------------- #
# Session with 30 Hz triggers and dropped frames
# --------------------------------------------------------------------------- #
N_TRIG = 120
# Short holes (<= 3 frames) plus one 0.63 s hole (80..98) inside "chewing".
LONG_GAP = tuple(range(80, 99))
DROPPED = (21, 22, 35, 60, 61, 62, 100) + LONG_GAP
MP_ORIGIN = np.array([20.0, 0.0, 0.0])
MODEL_POINTS = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]
MAND_OFFSETS = ([5.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 6.0])


def _eye_at(xyz) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


def write_30fps_session(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """HP fixed, MP moving along Z at 12 mm/s, CT digitizing six windows."""
    trig_t = T0_US + np.arange(N_TRIG, dtype=np.int64) * PERIOD_US
    keep = np.setdiff1d(np.arange(N_TRIG), DROPPED)
    n = keep.size
    hp = np.tile(np.eye(4), (n, 1, 1))
    mp = np.stack([_eye_at(MP_ORIGIN + [0.0, 0.0, 12e-6 * (trig_t[k] - T0_US)]) for k in keep])
    ct = np.full((n, 4, 4), np.nan)
    frame_row = {int(k): i for i, k in enumerate(keep)}
    windows = [(0, 2), (3, 5), (6, 8), (9, 11), (12, 14), (15, 17)]
    for w, (a, b) in enumerate(windows):
        for k in range(a, b + 1):
            i = frame_row[k]
            if w < 3:  # mandibular: CT relative to MP
                ct[i] = _eye_at(mp[i, :3, 3] + MAND_OFFSETS[w])
            else:
                ct[i] = _eye_at(MODEL_POINTS[w - 3])
    with h5py.File(path, "w") as f:
        f.create_group("meta").attrs["schema_version"] = 1
        g = f.create_group("optical")
        g.create_dataset("t_master_us", data=trig_t[keep].astype(np.uint64))
        g.create_dataset("pose_cra", data=hp.astype(np.float32))
        g.create_dataset("pose_mta", data=mp.astype(np.float32))
        g.create_dataset("pose_ct", data=ct.astype(np.float32))
        tg = f.create_group("triggers")
        tg.create_dataset("t_master_us", data=trig_t.astype(np.uint64))
        tg.create_dataset("trigger_id", data=np.arange(1, N_TRIG + 1, dtype=np.uint64))
        tg.attrs["rate_hz"] = 30
    return trig_t, keep


def test_phase_interval_spans_phase_duration(tmp_path: Path):
    trig_t, keep = write_30fps_session(tmp_path / "s.h5")
    t_opt = trig_t[keep]
    # 60-period (2.0 s) phase over the dropped block 60..62: 60 frames, not 57 rows.
    ph = TimedPhase("open_close", "experiment", int(trig_t[40]), int(trig_t[40] + 60 * PERIOD_US),
                    "exact", "exact")
    [(_n, _k, a, b)] = optical_frame_intervals([ph], t_opt)
    assert (a, b) == (40, 99)
    assert (b - a + 1) * PERIOD_US / 1e6 == pytest.approx(2.0, abs=PERIOD_US / 1e6)
    # Explicit trigger period gives the same frames.
    with h5py.File(tmp_path / "s.h5", "r") as f:
        p = session_frame_period_us(f)
    assert p == pytest.approx(PERIOD_US)
    assert optical_frame_intervals([ph], t_opt, period_us=p) == [(_n, _k, a, b)]


def test_recorder_and_jts_use_the_same_grid(tmp_path: Path):
    """phase_walker's frame numbers index the poses JTS lays out."""
    trig_t, keep = write_30fps_session(tmp_path / "s.h5")
    ensure_jts_on_path()
    from jts.aruco_camera import ArucoCameraData

    data = ArucoCameraData(str(tmp_path / "s.h5"))
    grid = frame_grid_index(trig_t[keep], estimate_period_us(trig_t[keep]))
    np.testing.assert_array_equal(grid, keep - keep[0])
    np.testing.assert_array_equal(data.timestamps["row_frame"], grid)
    assert data.frames == N_TRIG


def _config(session: Path, out_dir: Path) -> dict:
    mand = [[0, 2], [3, 5], [6, 8]]
    maxi = [[9, 11], [12, 14], [15, 17]]
    return {
        "data_source": {"type": "aruco_camera", "filename": str(session)},
        "analysis": {
            "calibration": {
                "mandibular": {"rigid_bodies": ["MP", "CT"], "points": [
                    {"name": f"mand_point_{i + 1}", "frame_interval": iv} for i, iv in enumerate(mand)]},
                "maxillary": {"rigid_bodies": ["HP", "CT"], "points": [
                    {"name": f"max_point_{i + 1}", "frame_interval": iv} for i, iv in enumerate(maxi)]},
            },
            "relative_motion": {"reference_body": "HP", "moving_body": "MP"},
            "experiment": {
                "frame_interval": [20, 119],
                "use_sub_experiments": True,
                "combine_sub_experiments": ["open_close", "chewing"],
                "sub_experiments": {"open_close": [20, 49], "chewing": [70, 119]},
                "interpolation": {"method": "cubic", "transition_frames": 10, "connect_intervals": True},
            },
            "coordinate_transform": {"enabled": True, "calibration_type": "maxillary",
                                     "model_points": MODEL_POINTS},
            "smoothing": {"enabled": True, "window_length": 11, "poly_order": 3},
            "coordinate_origin_index": 0,
        },
        "output": {"directory": str(out_dir), "save_hdf5": True, "save_csv": True,
                   "derivative_order": 2, "scale_factor": 0.001, "unit": "m"},
        "visualization": {"show_plots": False, "final_trajectory": False},
    }


def test_jts_export_carries_master_time(tmp_path: Path):
    pytest.importorskip("sklearn")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.run_jts_analysis import main

    h5 = tmp_path / "session.h5"
    trig_t, _keep = write_30fps_session(h5)
    (tmp_path / "t_jts_config.json").write_text(json.dumps(_config(h5, tmp_path / "out")))
    assert main([str(h5), "--out", str(tmp_path / "out"), "--no-show"]) == 0

    with h5py.File(tmp_path / "out" / "jaw_motion.h5", "r") as f:
        g = f["T_model_origin_mand_landmark_t"]
        t = g["t_master_us"][()]
        src = g["source_frame"][()]
        assert t.shape[0] == 30 + 10 + 50  # two phases + 10-frame synthetic bridge
        assert np.isnan(t[30:40]).all() and (src[30:40] == -1).all()
        np.testing.assert_array_equal(src[:30], np.arange(20, 50))
        np.testing.assert_array_equal(src[40:], np.arange(70, 120))
        # Real rows sit on the trigger times, including the dropped ones.
        np.testing.assert_allclose(t[:30], trig_t[20:50], atol=1)
        np.testing.assert_allclose(t[40:], trig_t[70:120], atol=1)
        assert "t_master_us" in f["T_model_origin_mand_landmark_t_smooth"]
        raw = g["translations"][()]
        dropped = np.isin(src, DROPPED)
        # Raw export is as measured: NaN wherever the frame was dropped.
        assert np.isnan(raw[dropped]).all() and np.isfinite(raw[~dropped & (src >= 0)]).all()
        np.testing.assert_array_equal(g["valid"][()], ~dropped & (src >= 0))
        sm = f["T_model_origin_mand_landmark_t_smooth"]
        smooth = sm["translations"][()]
        long_gap = np.isin(src, LONG_GAP)
        # Short holes are bridged for the smoother, the 0.63 s hole is not.
        assert np.isnan(smooth[long_gap]).all()
        assert np.isfinite(smooth[~long_gap]).all()
        np.testing.assert_array_equal(sm["gap_filled"][()], dropped & ~long_gap)
        # Velocity in m/s: MP moves at 12 mm/s (rigid registration keeps speed).
        v = sm["translational_derivative_order_1"][()]
        speed = np.linalg.norm(v[np.isfinite(v).all(axis=1) & (src >= 0)], axis=1)
        assert np.median(speed) == pytest.approx(0.012, rel=0.05)


def test_interpolate_caps_gap_length_and_keeps_input():
    from jtsmm.tools.run_jts_analysis import interpolate_se3_gaps

    T = np.tile(np.eye(4), (30, 1, 1))
    T[:, 0, 3] = np.arange(30.0)
    T[[0, 5, 6]] = np.nan        # edge hole + 2-frame hole
    T[12:22] = np.nan            # 10-frame hole
    before = T.copy()
    out, n = interpolate_se3_gaps(T, max_gap_frames=3)
    assert n == 2
    np.testing.assert_allclose(out[[5, 6], 0, 3], [5.0, 6.0])
    assert np.isnan(out[0]).all()        # no extrapolation at the edge
    assert np.isnan(out[12:22]).all()    # longer than the cap
    np.testing.assert_array_equal(np.isnan(T), np.isnan(before))  # input untouched
    _out, n_all = interpolate_se3_gaps(T)
    assert n_all == 12
