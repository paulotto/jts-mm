# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Optical session evaluator: lock rates, working volume, pose locations."""

from __future__ import annotations

from pathlib import Path

import pytest

from jtsmm.tools.eval_optical_session import (
    format_report,
    load_eval,
    write_synthetic_session,
)


def test_eval_optical_session_report_facts(tmp_path: Path):
    pytest.importorskip("h5py")
    path = write_synthetic_session(tmp_path / "session.h5")
    ev = load_eval(path)
    report = format_report(ev)

    assert ev.n_stereo == 200
    assert int(ev.masks["MP"].sum()) == 45
    assert int(ev.masks["HP"].sum()) == 2
    assert int(ev.masks["CT"].sum()) == 40
    assert "mand_point_1" in report
    assert "open_close" in report
    assert "Where finite poses sit" in report
    assert "Where to put effort" not in report
    assert "MP WORKING VOLUME" not in report
    assert "centroid" in report


def test_eval_optical_session_cli_no_plot(tmp_path: Path, capsys):
    pytest.importorskip("h5py")
    path = write_synthetic_session(tmp_path / "session.h5")
    from jtsmm.tools.eval_optical_session import main

    rc = main([str(path), "--no-plot"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "optical evaluation" in out
    assert "Where to put effort" not in out
    assert (tmp_path / "optical_eval.txt").is_file()


def test_eval_optical_session_strict_fails_low_lock(tmp_path: Path):
    pytest.importorskip("h5py")
    path = write_synthetic_session(tmp_path / "session.h5")
    from jtsmm.tools.eval_optical_session import main

    rc = main([str(path), "--no-plot", "--strict"])
    assert rc == 2
