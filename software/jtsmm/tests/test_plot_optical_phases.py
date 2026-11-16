# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical XYZ, one 3×1 matplotlib figure per experiment phase."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tests.helpers import make_session_h5


def _add_phases(h5: Path) -> None:
    with h5py.File(h5, "a") as f:
        t = np.asarray(f["optical"]["t_master_us"][()], dtype=np.int64)
        assert t.size >= 8
        a0, a1 = int(t[0]), int(t[t.size // 2])
        b0, b1 = int(t[t.size // 2]), int(t[-1])
        ph = f.create_group("phases")
        ph.create_dataset(
            "name", data=np.array(["open_close", "chewing"], dtype=h5py.string_dtype())
        )
        ph.create_dataset(
            "kind", data=np.array(["experiment", "experiment"], dtype=h5py.string_dtype())
        )
        ph.create_dataset("t_start_us", data=np.array([a0, b0], dtype=np.uint64))
        ph.create_dataset("t_end_us", data=np.array([a1, b1], dtype=np.uint64))


def test_slice_phase_stays_inside_interval(tmp_path: Path):
    from jtsmm.tools.plot_mp_trajectory import load_phases
    from jtsmm.tools.plot_optical_phases import load_optical_bodies, slice_phase

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases(h5)
    with h5py.File(h5, "r") as f:
        phases = load_phases(f)
    bodies = load_optical_bodies(h5)
    assert {b.name for b in bodies} == {"HP", "MP", "CT"}
    sliced = slice_phase(bodies, phases[0])
    assert sliced
    t0 = float(phases[0].t_start_us)
    t1 = float(phases[0].t_end_us)
    for _name, t_s, xyz in sliced:
        assert t_s.min() >= -1e-9
        assert t_s.max() <= (t1 - t0) * 1e-6 + 1e-9
        assert np.isfinite(xyz).all()


def test_cli_writes_one_png_per_phase(tmp_path: Path, capsys):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_optical_phases import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases(h5)
    rc = main([str(h5), "--out", str(out), "--no-show", "--world"])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == [
        "session_optical_chewing.png",
        "session_optical_open_close.png",
    ]
    text = capsys.readouterr().out
    assert "open_close" in text
    assert "chewing" in text


def test_cli_phase_filter(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_optical_phases import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases(h5)
    rc = main([str(h5), "--out", str(out), "--no-show", "--world", "--phases", "chewing"])
    assert rc == 0
    pngs = list(out.glob("*.png"))
    assert len(pngs) == 1
    assert pngs[0].name == "session_optical_chewing.png"


def test_relative_mp_in_hp_identity_rotation(tmp_path: Path):
    from jtsmm.tools.plot_optical_phases import load_optical_bodies, relative_mp_in_hp

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.3, emg_samples=60, optical=True)
    bodies = {b.name: b for b in load_optical_bodies(h5)}
    rel = relative_mp_in_hp(bodies["HP"], bodies["MP"])
    finite = np.isfinite(rel.xyz_mm).all(axis=1)
    assert finite.any()
    # Fixture poses are identity R; origins HP=(40,80,400), MP=(80,20,450).
    np.testing.assert_allclose(rel.xyz_mm[finite][0], (40.0, -60.0, 50.0), atol=1e-3)


def test_cli_writes_relative_pngs(tmp_path: Path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from jtsmm.tools.plot_optical_phases import main

    h5 = tmp_path / "session.h5"
    out = tmp_path / "viz"
    make_session_h5(h5, duration_s=0.4, emg_samples=80, optical=True)
    _add_phases(h5)
    rc = main([str(h5), "--out", str(out), "--no-show"])
    assert rc == 0
    pngs = sorted(p.name for p in out.glob("*.png"))
    assert pngs == [
        "session_mp_in_hp_chewing.png",
        "session_mp_in_hp_open_close.png",
    ]


def test_cli_errors_without_phases(tmp_path: Path):
    from jtsmm.tools.plot_optical_phases import main

    h5 = tmp_path / "session.h5"
    make_session_h5(h5, duration_s=0.2, emg_samples=40, optical=True)
    rc = main([str(h5), "--no-show"])
    assert rc == 1
