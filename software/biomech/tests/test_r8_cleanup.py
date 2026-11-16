# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""J1.4: snapshots pinned by a paper set or numbers file are never deleted;
``biomech snapshot --delete`` removes only unpinned ones."""

from __future__ import annotations

from click.testing import CliRunner

import biomech.cli as cli
from biomech.results import snapshot as SN


def _study(tmp_path):
    exp = tmp_path / "experiments"
    (exp / "a").mkdir(parents=True)
    (exp / "b").mkdir()
    study = exp / "a" / "study.yaml"
    study.write_text("study: a\n")
    (exp / "a" / "paper_x.yaml").write_text("snapshot: r9a   # paper store\n"
                                            "record_snapshot: r9\nitems: []\n")
    (exp / "a" / "numbers_x.yaml").write_text("cite:\n  other:\n    snapshot: 'v4'\n")
    (exp / "b" / "paper_y.yaml").write_text("snapshot: v4\n")
    return study


def test_pinned_reads_paper_sets_numbers_files_and_the_record(tmp_path):
    pins = SN.pinned(_study(tmp_path))
    assert set(pins) == {"r9a", "r9", "v4"}
    assert sorted(pins["v4"]) == ["a/numbers_x.yaml", "b/paper_y.yaml"]


def test_cli_delete_refuses_pins_and_deletes_unpinned(tmp_path, monkeypatch):
    study = _study(tmp_path)
    out = tmp_path / "store"
    for name in ("r9", "r8"):
        d = out / "snapshots" / name
        d.mkdir(parents=True)
        (d / "snapshot.json").write_text("{}")

    class _Cfg:
        out_dir = out

    monkeypatch.setattr(cli, "_load", lambda *a, **k: _Cfg())
    run = CliRunner().invoke
    r = run(cli.main, ["snapshot", "--config", str(study), "--delete", "r9", "--yes"])
    assert r.exit_code != 0 and "pinned by a/paper_x.yaml" in r.output
    assert (out / "snapshots" / "r9").is_dir()
    r = run(cli.main, ["snapshot", "--config", str(study), "--delete", "r8"], input="n\n")
    assert r.exit_code != 0 and (out / "snapshots" / "r8").is_dir()      # not confirmed
    r = run(cli.main, ["snapshot", "--config", str(study), "--delete", "r8", "--yes"])
    assert r.exit_code == 0, r.output
    assert not (out / "snapshots" / "r8").exists()
    r = run(cli.main, ["snapshot", "--config", str(study), "--delete", "r7", "--yes"])
    assert r.exit_code != 0 and "no snapshot r7" in r.output
