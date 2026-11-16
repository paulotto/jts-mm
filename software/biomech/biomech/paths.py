# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Where sessions are read from and where study results go.

One resolution rule for every entry point (experiment scripts, ``biomech``
CLI, viewer), so a script runs from any working directory without
``sys.path`` edits or hard-coded ``/home/pi`` paths:

* sessions: ``--data-dir`` > ``$JTSMM_DATA_DIR`` > ``<repo>/data/pilot_study/jtsmm_data``
  (when the package runs from a checkout that has it) > ``~/jtsmm_data``;
* results: ``--out-dir`` > ``$BIOMECH_RESULTS_DIR``/<study>/results > ``~/experiments/<study>/results``.
"""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "JTSMM_DATA_DIR"
RESULTS_DIR_ENV = "BIOMECH_RESULTS_DIR"
PILOT_DATA_REL = Path("data") / "pilot_study" / "jtsmm_data"


def _expand(p: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(str(p))).expanduser()


def repo_root() -> Path | None:
    """Checkout root when the package runs from one (editable install), else None.

    The checkout is recognised by ``software/biomech`` sitting two levels below
    a directory that also holds ``software/jtsmm``; a wheel installed into
    site-packages has neither, so this returns None there.
    """
    pkg_project = Path(__file__).resolve().parents[1]      # software/biomech
    root = pkg_project.parents[1] if len(pkg_project.parents) > 1 else None
    if root is None:
        return None
    if pkg_project.parent.name == "software" and (root / "software" / "jtsmm").is_dir():
        return root
    return None


def resolve_data_dir(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Session root: explicit > ``$JTSMM_DATA_DIR`` > repo pilot data > ``~/jtsmm_data``.

    The result is not required to exist; callers report an empty cohort.
    """
    if explicit not in (None, ""):
        return _expand(explicit)
    env = os.environ.get(DATA_DIR_ENV)
    if env:
        return _expand(env)
    root = repo_root()
    if root is not None and (root / PILOT_DATA_REL).is_dir():
        return root / PILOT_DATA_REL
    return Path.home() / "jtsmm_data"


def results_root() -> Path:
    """``$BIOMECH_RESULTS_DIR`` or ``~/experiments`` (one folder per study below it)."""
    env = os.environ.get(RESULTS_DIR_ENV)
    return _expand(env) if env else Path.home() / "experiments"


def results_dir(study: str, explicit: str | os.PathLike[str] | None = None) -> Path:
    """Output folder of one study: explicit ``--out-dir`` or ``<results_root>/<study>/results``."""
    if explicit not in (None, ""):
        return _expand(explicit)
    return results_root() / study / "results"
