# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""eval_optical_session at 30 fps with dropped frames: true lock rate, --save dirs."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from jtsmm.tools.eval_optical_session import (
    format_report,
    load_eval,
    main,
    write_synthetic_session,
)

# MP is locked on frames 0..39 of 200; drop 10 of those + 30 elsewhere.
DROP = tuple(range(30, 40)) + tuple(range(100, 130))


def test_lock_rate_against_triggers_not_rows(tmp_path: Path):
    path = write_synthetic_session(tmp_path / "s.h5", dt_us=33_333, drop=DROP)
    ev = load_eval(path)
    assert len(ev.t_us) == 160  # rows
    lo, hi = ev.session_window_us()
    assert ev.n_triggers(lo, hi) == 200
    # mand_point_1 = frames 0..39: 30 rows, 40 triggers, 30 finite MP poses.
    ph = next(p for p in ev.phases if p.name == "mand_point_1")
    assert ev.n_triggers(ph.t_start_us, ph.t_end_us) == 40
    report = format_report(ev)
    assert "/ 200 triggers" in report
    mp_line = next(line for line in report.splitlines() if line.strip().startswith("MP ") and "|" in line)
    assert "35/160" in mp_line and "35/200" in mp_line and "17.5%" in mp_line
    row = next(line for line in report.splitlines() if line.startswith("mand_point_1"))
    assert "100.0% (75.0%)" in row  # MP: all rows locked, 3/4 of the triggers


def test_lost_trigger_events_counted_from_ids(tmp_path: Path):
    path = write_synthetic_session(tmp_path / "s.h5", dt_us=33_333)
    with h5py.File(path, "a") as f:
        g = f["triggers"]
        keep = np.r_[0:50, 55:200]  # 5 trigger events missing from the table
        t, ids = g["t_master_us"][()][keep], g["trigger_id"][()][keep]
        del g["t_master_us"], g["trigger_id"]
        g.create_dataset("t_master_us", data=t)
        g.create_dataset("trigger_id", data=ids)
    ev = load_eval(path)
    assert ev.n_triggers(*ev.session_window_us()) == 200


def test_without_triggers_says_true_rate_unknown(tmp_path: Path):
    path = write_synthetic_session(tmp_path / "s.h5", dt_us=33_333, drop=DROP)
    with h5py.File(path, "a") as f:
        del f["triggers"]
    report = format_report(load_eval(path))
    assert "true rate unknown" in report


def test_save_into_new_directory(tmp_path: Path):
    path = write_synthetic_session(tmp_path / "s.h5", dt_us=33_333, drop=DROP)
    dest = tmp_path / "new" / "deeper" / "eval.png"
    assert main([str(path), "--no-plot", "--save", str(dest)]) == 0
    assert (tmp_path / "new" / "deeper" / "eval.txt").is_file()
