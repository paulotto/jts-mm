# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""EMG 3×1 plot: viewer filtered / raw / envelope."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.helpers import make_session_h5
from tests.test_plot_optical_phases import _add_phases


def test_filter_emg_matches_hub_native_filter():
    from jtsmm.tools.plot_emg import filter_emg, viewer_display_lpf
    from jtsmm.viewer.hub import TARGET_DISPLAY_FS
    from jtsmm.viewer.processing import CausalEmgFilter

    rng = np.random.default_rng(0)
    uv = rng.standard_normal((800, 7)).astype(np.float32) * 40.0
    fs = 2000.0
    filt, env = filter_emg(uv, fs)
    ref = CausalEmgFilter(fs, 7, lpf_hz=viewer_display_lpf(TARGET_DISPLAY_FS))
    f2, e2 = ref.process(uv)
    np.testing.assert_allclose(filt, f2, atol=1e-5)
    np.testing.assert_allclose(env, e2, atol=1e-5)
    assert np.all(env >= -1e-4)
    assert abs(viewer_display_lpf(500.0) - 225.0) < 1e-9


def test_decimate_like_viewer_minmax_and_stride():
    from jtsmm.tools.plot_emg import decimate_like_viewer

    n, c, d = 20, 2, 4
    t = np.arange(n, dtype=np.uint64)
    raw = np.arange(n * c, dtype=np.float32).reshape(n, c)
    filt = raw + 1
    env = np.abs(raw)
    tt, lo, hi, fd, ed = decimate_like_viewer(t, raw, filt, env, 2000.0, 500.0)
    assert tt.shape[0] == n // d
    np.testing.assert_array_equal(lo[0], raw[0:d].min(axis=0))
    np.testing.assert_array_equal(hi[0], raw[0:d].max(axis=0))
    np.testing.assert_array_equal(fd[1], filt[d])


def test_viewer_slice_drops_bias_debug():
    from jtsmm.config.catalog import VIEWER_EMG_PLOT_COUNT, VIEWER_EMG_PLOT_NAMES
    from jtsmm.tools.plot_emg import viewer_channel_names
    from jtsmm.viewer.sources import _slice_emg_for_viewer

    uv = np.arange(16, dtype=np.float32).reshape(2, 8)
    sl = _slice_emg_for_viewer(uv)
    assert sl.shape == (2, VIEWER_EMG_PLOT_COUNT)
    np.testing.assert_array_equal(sl[0], [0, 1, 2, 3, 4, 5, 7])
    names = viewer_channel_names(
        [
            "masseter_r",
            "temporalis_r",
            "digastric_r",
            "digastric_l",
            "masseter_l",
            "temporalis_l",
            "bias_debug",
            "drl",
        ],
        sl.shape[1],
    )
    assert names == list(VIEWER_EMG_PLOT_NAMES)


def test_cli_writes_one_png_per_channel(tmp_path: Path, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.config.catalog import VIEWER_EMG_PLOT_NAMES
    from jtsmm.tools.plot_emg import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, emg_fs=1000, optical=False)
    rc = main([str(h5), "--out", str(out), "--no-show"])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == sorted(f"session_emg_{name}.png" for name in VIEWER_EMG_PLOT_NAMES)
    text = capsys.readouterr().out
    assert "masseter_l" in text


def test_cli_channel_filter_writes_subset(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_emg import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.3, emg_samples=60, emg_fs=1000)
    rc = main(
        [str(h5), "--out", str(out), "--no-show", "--channel", "masseter_l", "drl"]
    )
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == ["session_emg_drl.png", "session_emg_masseter_l.png"]


def test_cli_phase_crop(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_emg import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=400, emg_fs=1000, optical=True)
    _add_phases(h5)
    rc = main([str(h5), "--out", str(out), "--no-show", "--phases", "chewing"])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert all(n.startswith("session_emg_chewing_") for n in pngs)
    assert any("masseter_l" in n for n in pngs)
    assert len(pngs) >= 6


def test_load_emg_drops_calibration_phases(tmp_path: Path):
    from jtsmm.tools.plot_emg import load_emg_plot

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=400, emg_fs=1000, optical=True)
    from tests.test_plot_mp_trajectory import _add_phases_with_calib

    _add_phases_with_calib(h5)
    data = load_emg_plot(h5, channels=["masseter_l"])
    assert [s.name for s in data.spans] == ["open_close", "chewing"]
    calib = load_emg_plot(h5, channels=["masseter_l"], phases=["mand_point_1"])
    assert [s.name for s in calib.spans] == ["mand_point_1"]


def _add_gapped_emg_phases(h5: Path) -> None:
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["emg"]["t_master_us"][()], dtype=np.int64)
        assert t.size >= 12
        a0, a1 = int(t[1]), int(t[t.size // 4])
        b0, b1 = int(t[(3 * t.size) // 4]), int(t[-2])
        if "phases" in f:
            del f["phases"]
        ph = f.create_group("phases")
        ph.create_dataset(
            "name", data=np.array(["mvc_opening", "mvc_closing"], dtype=h5py.string_dtype())
        )
        ph.create_dataset(
            "kind", data=np.array(["mvc", "mvc"], dtype=h5py.string_dtype())
        )
        ph.create_dataset("t_start_us", data=np.array([a0, b0], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([a1, b1], dtype=np.uint64))


def test_concat_drops_time_between_phases(tmp_path: Path):
    from jtsmm.tools.plot_emg import load_emg_plot

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.8, emg_samples=800, emg_fs=1000)
    _add_gapped_emg_phases(h5)
    with h5py.File(h5, "r") as f:
        t0 = float(f["phases"]["t_start_us"][0])
        t1 = float(f["phases"]["t_end_us"][-1])
    wall_s = (t1 - t0) * 1e-6
    data = load_emg_plot(h5, channels=["masseter_l"])
    assert data.concatenated
    assert [s.name for s in data.spans] == ["mvc_opening", "mvc_closing"]
    finite_t = data.t_s[np.isfinite(data.raw[:, 0])]
    assert finite_t.size > 10
    assert float(finite_t.max() - finite_t.min()) < wall_s * 0.9
    full = load_emg_plot(h5, channels=["masseter_l"], full=True)
    assert not full.concatenated
    assert float(full.t_s.max()) > float(data.t_s[np.isfinite(data.t_s)].max())


def test_cli_errors_without_emg(tmp_path: Path):
    from jtsmm.tools.plot_emg import main

    h5 = tmp_path / "session.h5"
    with h5py.File(h5, "w") as f:
        f.create_group("meta")
    rc = main([str(h5), "--no-show"])
    assert rc == 1
