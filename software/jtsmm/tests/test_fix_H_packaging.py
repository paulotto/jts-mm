# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""A regular (non-editable) install must load and validate its config.

The schema/template live outside the package, so setup.py bundles copies into
jtsmm/_data at build time (_bundle_data.py). These tests build that layout in a
temp dir -- away from the checkout -- and load the config through it.
"""

from __future__ import annotations

import filecmp
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SW = Path(__file__).resolve().parents[1]
REPO = SW.parents[1]
sys.path.insert(0, str(SW))
from _bundle_data import CONFIG_ASSETS, copy_config_assets  # noqa: E402


def _assert_same_tree(a: Path, b: Path) -> None:
    cmp = filecmp.dircmp(a, b, ignore=["__pycache__"])
    assert not cmp.left_only and not cmp.right_only, (cmp.left_only, cmp.right_only)
    _, mismatch, errors = filecmp.cmpfiles(a, b, cmp.common_files, shallow=False)
    assert not mismatch and not errors, (mismatch, errors)
    for sub in cmp.common_dirs:
        _assert_same_tree(a / sub, b / sub)


def test_bundle_equals_repo_copy(tmp_path):
    dest = tmp_path / "_data"
    copy_config_assets(REPO, dest)
    for src_rel, dst_rel in CONFIG_ASSETS:
        src, dst = REPO / src_rel, dest / dst_rel
        if src.is_dir():
            _assert_same_tree(src, dst)
        else:
            assert filecmp.cmp(src, dst, shallow=False)
    # Everything the loader reads at runtime is in the bundle.
    for rel in (
        "schema/v1/jtsmm.schema.json",
        "schema/v1/config_coverage.yaml",
        "schema/v1/profiles/imu_kinematics_default.yaml",
        "etc/jtsmm.yaml",
    ):
        assert (dest / rel).is_file(), rel


def _installed_layout(tmp_path: Path) -> Path:
    site = tmp_path / "site"
    shutil.copytree(
        SW / "jtsmm",
        site / "jtsmm",
        ignore=shutil.ignore_patterns("__pycache__", "*.so", "_data"),
    )
    copy_config_assets(REPO, site / "jtsmm" / "_data")
    return site


_PROBE = r"""
import json
import jtsmm
from jtsmm.config import load, resolve_profiles
from jtsmm.config.loader import checkout_root, default_config_path, schema_dir
cfg = load()
resolve_profiles(cfg)
print(json.dumps({
    "file": jtsmm.__file__,
    "checkout": None if checkout_root() is None else str(checkout_root()),
    "schema": str(schema_dir()),
    "config": str(default_config_path()),
    "schema_version": cfg["schema_version"],
}))
"""


def test_installed_layout_loads_and_validates(tmp_path):
    site = _installed_layout(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    elsewhere = tmp_path / "cwd"
    elsewhere.mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("JTSMM_")}
    env.update(PYTHONPATH=str(site), HOME=str(home), JTSMM_DATA_DIR=str(tmp_path / "data"))
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        cwd=elsewhere,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["file"].startswith(str(site)), out
    assert out["checkout"] is None
    assert out["schema"].startswith(str(site / "jtsmm" / "_data"))
    assert out["config"] == str(site / "jtsmm" / "_data" / "etc" / "jtsmm.yaml")
    assert out["schema_version"] == 1


def _python_with_setuptools() -> str | None:
    for exe in (sys.executable, shutil.which("python3")):
        if not exe:
            continue
        ok = subprocess.run([exe, "-c", "import setuptools"], capture_output=True)
        if ok.returncode == 0:
            return exe
    return None


def test_setup_py_build_hook_bundles_assets(tmp_path):
    """The real build_py hook (not just the helper) puts the assets in the build."""
    exe = _python_with_setuptools()
    if exe is None:
        pytest.skip("no interpreter with setuptools available")
    repo = tmp_path / "repo"
    shutil.copytree(REPO / "schema" / "v1", repo / "schema" / "v1")
    sw = repo / "software" / "jtsmm"
    (sw / "etc").mkdir(parents=True)
    for name in ("setup.py", "_bundle_data.py", "pyproject.toml", "README.md", "etc/jtsmm.yaml"):
        shutil.copy2(SW / name, sw / name)
    shutil.copytree(SW / "jtsmm", sw / "jtsmm", ignore=shutil.ignore_patterns("__pycache__", "*.so"))
    build = tmp_path / "build"
    proc = subprocess.run(
        [exe, "setup.py", "-q", "build_py", "--build-lib", str(build)],
        cwd=sw,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    _assert_same_tree(REPO / "schema" / "v1", build / "jtsmm" / "_data" / "schema" / "v1")
    assert filecmp.cmp(
        SW / "etc" / "jtsmm.yaml", build / "jtsmm" / "_data" / "etc" / "jtsmm.yaml", shallow=False
    )
