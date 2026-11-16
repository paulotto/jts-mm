# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The vendored JTS wins over a stale ``jts`` installed elsewhere."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from jtsmm.tools.jts_path import JTS_ROOT, ensure_jts_on_path


def _stale_jts(tmp_path: Path) -> Path:
    pkg = tmp_path / "site-packages" / "jts"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("STALE = True\n")
    return pkg.parent


def _forget_jts(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [m for m in sys.modules if m == "jts" or m.startswith("jts.")]:
        monkeypatch.delitem(sys.modules, name)


@pytest.mark.skipif(not (JTS_ROOT / "jts" / "__init__.py").is_file(), reason="no vendored JTS")
def test_vendored_copy_shadows_stale_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _forget_jts(monkeypatch)
    monkeypatch.setattr(sys, "path", [str(_stale_jts(tmp_path)), *sys.path])

    assert ensure_jts_on_path() == JTS_ROOT
    import jts

    assert not hasattr(jts, "STALE")
    assert Path(jts.__file__).resolve().parent == (JTS_ROOT / "jts").resolve()
    import jts.aruco_camera  # noqa: F401  (missing from old installs)


def test_already_imported_stale_copy_is_reported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    vendored = tmp_path / "jaw_tracking"
    (vendored / "jts").mkdir(parents=True)
    (vendored / "jts" / "__init__.py").write_text("")
    stale = types.ModuleType("jts")
    stale.__file__ = str(_stale_jts(tmp_path) / "jts" / "__init__.py")
    _forget_jts(monkeypatch)
    monkeypatch.setitem(sys.modules, "jts", stale)

    with pytest.raises(ImportError, match="different jts is already imported"):
        ensure_jts_on_path(vendored)


def test_without_vendored_tree_uses_installed_jts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _forget_jts(monkeypatch)
    site = _stale_jts(tmp_path)
    monkeypatch.setattr(sys, "path", [str(site), *sys.path])

    assert ensure_jts_on_path(tmp_path / "missing") == site.resolve()
