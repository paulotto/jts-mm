# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Find (session.h5, label sidecar) pairs.

* Next to the session: ``session_*/session_labels.h5`` (``ml label`` writes it there).
* In a label directory (``--labels-dir``, e.g. from ``ml labels from-biomech``):
  ``<dir>/<session folder>_labels.h5``; without explicit sessions every sidecar in
  the directory is used and its ``session_path`` attribute locates the take.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py

from ml.config.paths import session_data_dir
from ml.io.labels import default_label_path


@dataclass(frozen=True)
class SessionRef:
    session_path: Path
    label_path: Path


def _h5(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "session.h5"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _label_for(h5: Path, labels_dir: Path | None) -> Path:
    if labels_dir is not None:
        return Path(labels_dir) / f"{h5.parent.name}_labels.h5"
    return default_label_path(h5)


def sidecars_in(labels_dir: str | Path) -> list[SessionRef]:
    """Every ``*_labels.h5`` in ``labels_dir`` with the take named in its attributes."""
    refs = []
    for lp in sorted(Path(labels_dir).glob("*_labels.h5")):
        with h5py.File(lp, "r") as f:
            sp = f["labels"].attrs.get("session_path", "")
        sp = sp.decode() if isinstance(sp, bytes) else str(sp)
        if not sp:
            raise ValueError(f"{lp} has no session_path attribute; pass the sessions explicitly")
        refs.append(SessionRef(_h5(Path(sp)), lp.resolve()))
    if not refs:
        raise FileNotFoundError(f"no *_labels.h5 in {labels_dir}")
    return refs


def resolve_session_inputs(sessions: list[Path] | None = None, *,
                           data_dir: str | Path | None = None,
                           labels_dir: str | Path | None = None) -> list[SessionRef]:
    """Explicit sessions (files or folders), else every sidecar in ``labels_dir``, else every
    labelled ``session_*`` folder under ``data_dir`` (default ``paths.session_data_dir``)."""
    ld = Path(labels_dir).expanduser().resolve() if labels_dir else None
    if sessions:
        refs = []
        for s in sessions:
            h5 = _h5(Path(s))
            lp = _label_for(h5, ld)
            if not lp.is_file():
                raise FileNotFoundError(f"missing labels for {h5}: {lp}")
            refs.append(SessionRef(h5, lp))
        return refs
    if ld is not None:
        return sidecars_in(ld)
    root = Path(data_dir).expanduser().resolve() if data_dir else session_data_dir()
    if not root.is_dir():
        raise FileNotFoundError(f"session data directory not found: {root}")
    refs = [SessionRef(d / "session.h5", default_label_path(d / "session.h5"))
            for d in sorted(root.glob("session_*")) if (d / "session.h5").is_file()]
    refs = [r for r in refs if r.label_path.is_file()]
    if not refs:
        raise FileNotFoundError(f"no labelled sessions (session_*/session_labels.h5) under {root}")
    return refs
