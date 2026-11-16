# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Synthetic takes for the W3 tests (experiment-script smoke runs and helper tests).

``make_take`` writes a full ``session.h5`` with ``jtsmm.tools.mock_session`` and
then replaces the parts the evaluation scripts look at:

* the phase table (rest + the five motion tasks, or rest + the two MVC phases);
* EMG: 3 µV noise + a 50 Hz tone everywhere, louder bursts in the tasks;
* optical: the mandible opens 0-10 mm once per second in every task (so the
  peak-to-peak detector finds cycles), with a short untracked gap in
  ``open_close``.

No subject data is involved.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

MOTION_PHASES = (
    ("rest", 1.0, 5.0),
    ("open_close", 5.0, 12.0),
    ("protrusion_retrusion", 12.0, 16.0),
    ("left_right", 16.0, 20.0),
    ("chewing", 20.0, 25.0),
    ("free_motion", 25.0, 28.0),
)
MVC_PHASES = (("rest", 1.0, 4.0), ("mvc_opening", 4.0, 8.0), ("mvc_closing", 8.0, 12.0))


def _replace(g: h5py.Group, name: str, data: np.ndarray) -> None:
    if name in g:
        del g[name]
    g.create_dataset(name, data=data)


def make_take(root: Path, name: str, *, kind: str = "motion", seed: int = 0,
              opening_mm: float = 10.0, emg_fs: int = 1000,
              burst_uv: float | None = None) -> Path:
    """Write ``root/name/session.h5``; returns the take folder.

    ``burst_uv`` is the EMG burst SD in the tasks (default 100 µV in MVC phases,
    30 µV in motion phases).
    """
    from jtsmm.tools.mock_session import write_mock_session

    phases = MOTION_PHASES if kind == "motion" else MVC_PHASES
    duration = 30.0 if kind == "motion" else 14.0
    folder = root / name
    folder.mkdir(parents=True, exist_ok=True)
    write_mock_session(folder / "session.h5", duration_s=duration, emg_fs=emg_fs,
                       write_jts_config=False, seed=seed)
    rng = np.random.default_rng(seed)
    with h5py.File(folder / "session.h5", "r+") as f:
        ph = f["phases"]
        for key in list(ph.keys()):
            del ph[key]
        str_dt = h5py.string_dtype()
        ph.create_dataset("name", data=np.asarray([p[0] for p in phases], dtype=str_dt))
        ph.create_dataset("kind", data=np.asarray(["experiment"] * len(phases), dtype=str_dt))
        ph.create_dataset("t_start_us", data=np.asarray([p[1] * 1e6 for p in phases], np.uint64))
        ph.create_dataset("t_end_us", data=np.asarray([p[2] * 1e6 for p in phases], np.uint64))
        if "phases_json" in ph.attrs:
            del ph.attrs["phases_json"]
        ph.attrs["n_phases"] = len(phases)
        if "events" in f:            # mock events name the mock phases; drop them
            del f["events"]

        emg = f["emg"]
        t = np.asarray(emg["t_master_us"], dtype=np.float64) / 1e6
        uv = rng.normal(0.0, 3.0, size=emg["samples_uv"].shape)
        uv += 20.0 * np.sin(2 * np.pi * 50.0 * t)[:, None]
        for pname, a, b in phases:
            if pname == "rest":
                continue
            m = (t >= a) & (t <= b)
            sigma = burst_uv or (100.0 if pname.startswith("mvc") else 30.0)
            env = np.abs(np.sin(np.pi * (t[m] - a)))[:, None]
            uv[m, :6] += rng.normal(0.0, sigma, size=(int(m.sum()), 6)) * env
        uv[:, 6:] = 0.0
        emg["samples_uv"][...] = uv.astype(emg["samples_uv"].dtype)

        opt = f["optical"]
        to = np.asarray(opt["t_master_us"], dtype=np.float64) / 1e6
        open_ = np.zeros_like(to)
        for pname, a, b in phases:
            if pname == "rest" or pname.startswith("mvc"):
                continue
            m = (to >= a) & (to <= b)
            # ~1 Hz with a slowly drifting rate, so cycles differ a little.
            hz = 1.0 + 0.08 * np.sin(2 * np.pi * (to[m] - a) / 3.7 + seed)
            phase = 2 * np.pi * np.cumsum(hz) * np.gradient(to[m])
            open_[m] = opening_mm * 0.5 * (1.0 - np.cos(phase - phase[0]))
        n = to.size
        cra = np.tile(np.eye(4), (n, 1, 1))
        cra[:, :3, 3] = (40.0, -65.0, 700.0)
        mta = np.tile(np.eye(4), (n, 1, 1))
        mta[:, :3, 3] = (55.0, 80.0, 645.0)
        mta[:, 1, 3] += open_
        mta[:, :3, 3] += rng.normal(0.0, 0.02, size=(n, 3))
        gap = (to >= 8.4) & (to <= 8.55)
        mta[gap] = np.nan
        _replace(opt, "pose_cra", cra.astype(np.float32))
        _replace(opt, "pose_mta", mta.astype(np.float32))
    return folder


def make_cohort(root: Path, pids: tuple[str, ...] = ("s01", "s02")) -> list[Path]:
    """Motion + MVC take per participant (MVC recorded first), named like the pilot."""
    out = []
    for i, pid in enumerate(pids):
        sex, age = ("f", 31) if i % 2 == 0 else ("m", 30 + i)
        day = 10 + i
        out.append(make_take(root, f"session_{pid}_{sex}_{age}_mvc_202503{day:02d}T090000Z_ab{i:02d}00",
                             kind="mvc", seed=10 * i + 1))
        out.append(make_take(root, f"session_{pid}_{sex}_{age}_202503{day:02d}T091000Z_cd{i:02d}00",
                             kind="motion", seed=10 * i + 2, opening_mm=10.0 + 3 * i))
    return out


def test_make_take_is_readable(tmp_path: Path) -> None:
    from biomech.io.session import SessionData

    folder = make_take(tmp_path, "session_s01_f_31_20250310T091000Z_cd0000")
    with SessionData(folder) as sd:
        assert sd.phase_names() == [p[0] for p in MOTION_PHASES]
        opt = sd.optical()
        assert opt is not None and np.isnan(opt.poses["mta"]).any()
        t, uv, names = sd.emg_phase("open_close")
        assert uv.shape[1] == 6 and uv.std() > 5.0
