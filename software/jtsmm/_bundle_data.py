# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Copy the repo-level config assets into a built ``jtsmm`` package.

The schema, profile fragments and operator template live at the repository
level (``schema/v1/``, ``software/jtsmm/etc/jtsmm.yaml``) because firmware and
docs share them. A regular (non-editable) ``pip install`` only ships what is
inside the ``jtsmm`` package, so ``setup.py`` calls :func:`copy_config_assets`
at build time to place copies under ``jtsmm/_data/``. The copies are generated,
never committed, so they cannot drift from the repo files.

Stdlib only: this runs inside the build backend, before jtsmm's dependencies
are installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# (source relative to the repo root, destination relative to jtsmm/_data)
CONFIG_ASSETS: tuple[tuple[str, str], ...] = (
    ("schema/v1", "schema/v1"),
    ("software/jtsmm/etc/jtsmm.yaml", "etc/jtsmm.yaml"),
)


def copy_config_assets(repo_root: Path, dest: Path) -> list[Path]:
    """Copy every entry of :data:`CONFIG_ASSETS` from ``repo_root`` into ``dest``.

    Returns the list of files written. Raises ``FileNotFoundError`` when a source
    is missing (e.g. building from an sdist without the repository around it).
    """
    repo_root = Path(repo_root)
    dest = Path(dest)
    written: list[Path] = []
    for src_rel, dst_rel in CONFIG_ASSETS:
        src = repo_root / src_rel
        dst = dest / dst_rel
        if src.is_dir():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            written.extend(p for p in dst.rglob("*") if p.is_file())
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            written.append(dst)
        else:
            raise FileNotFoundError(
                f"jtsmm build: config asset {src} not found -- build from a repository "
                "checkout (the schema and template live outside software/jtsmm)"
            )
    return written
