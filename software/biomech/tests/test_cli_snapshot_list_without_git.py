# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""``biomech snapshot --list`` on snapshots frozen outside a git checkout (no commit)."""

from __future__ import annotations

import json

from click.testing import CliRunner


def test_list_shows_snapshots_without_a_commit(tmp_path, monkeypatch):
    from biomech import cli
    from biomech.results import snapshot as SN      # imported inside the command

    monkeypatch.setattr(SN, "available", lambda root: ["r99"])
    monkeypatch.setattr(SN, "info", lambda root, n: {
        "created_utc": "2026-11-04T00:00:00Z", "git": {"sha": None, "dirty": None},
        "note": "copy without .git"})
    cfg = tmp_path / "study.yaml"
    cfg.write_text(json.dumps({"study": "t", "paths": {"data_dir": str(tmp_path),
                                                       "out_dir": str(tmp_path / "out"),
                                                       "cohort": "cohort.yaml"}}),
                   encoding="utf-8")
    (tmp_path / "cohort.yaml").write_text("participants: []\n", encoding="utf-8")
    r = CliRunner().invoke(cli.main, ["snapshot", "--config", str(cfg), "--list"])
    assert r.exit_code == 0, r.output
    assert r.output.startswith("r99 ") and "git ?" in r.output
