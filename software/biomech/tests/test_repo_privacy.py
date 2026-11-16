# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""No subject data in git: the folders of check_repo are untracked and git-ignored."""

from __future__ import annotations

import subprocess

import pytest

from biomech.check import outputs as C
from biomech.paths import repo_root

ROOT = repo_root()
# a git checkout: an exported copy of the tracked files (the office test runs) has no .git
pytestmark = pytest.mark.skipif(ROOT is None or not (ROOT / ".git").exists(),
                                reason="not running from a git checkout")


def test_no_subject_data_is_tracked():
    r = subprocess.run(["git", "ls-files", *C.SUBJECT_DATA_DIRS], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert r.stdout.split() == []


def test_subject_data_dirs_are_git_ignored():
    r = subprocess.run(["git", "check-ignore", "--no-index",
                        *(f"{d}/x.npz" for d in C.SUBJECT_DATA_DIRS)],
                       cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert sorted(r.stdout.split()) == sorted(f"{d}/x.npz" for d in C.SUBJECT_DATA_DIRS)


def test_check_repo_reports_tracked_subject_data(tmp_path):
    """The check fails when a file under one of the folders is tracked."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    f = tmp_path / "data" / "processed" / "XY__task.npz"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    subprocess.run(["git", "add", "-f", str(f)], cwd=tmp_path, check=True)
    rep = C.Report("test")
    C.check_repo(rep, tmp_path)
    item = next(i for i in rep.items if i.name == "no subject data tracked by git")
    assert item.status == "fail" and "data/processed/XY__task.npz" in item.detail
