# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Plain-file outputs of the experiment scripts and their provenance record.

The scripts under ``experiments/`` write CSV/JSON next to their figures (not
through :class:`~biomech.results.ResultStore`). ``write_provenance`` records
what produced them with the same git state and package versions that
``biomech.pipeline.run`` writes into its run manifest (one implementation:
``git_state`` / ``package_versions`` below, also used by the pipeline).
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROVENANCE_NAME = "provenance.json"


def git_state() -> dict[str, Any]:
    """Commit and dirty flag of the checkout (None outside one)."""
    import subprocess

    from biomech.paths import repo_root

    root = repo_root()
    if root is None:
        return {"sha": None, "dirty": None}

    def git(*args: str, strip: bool = True) -> str:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True,
                             timeout=10).stdout
        return out.strip() if strip else out

    def changed(*args: str) -> list[str]:
        """Paths of ``git status --porcelain`` ("XY path"; the leading status column kept)."""
        return [ln[3:].strip('"') for ln in git("status", "--porcelain", *args,
                                                strip=False).splitlines() if ln.strip()]

    try:
        # dirty = a tracked change in what can change a result or an export: the code, the
        # study and paper configuration, the scripts (2026-11-06; local editor settings and
        # untracked scratch folders made every export "dirty"); other changed
        # paths are listed apart
        code = changed("--untracked-files=no", "--", *RESULT_PATHS)
        other = [x for x in changed() if x not in code]
        return {"sha": git("rev-parse", "HEAD") or None, "dirty": bool(code),
                "dirty_paths": code[:30], "other_changes": other[:30]}
    except (OSError, subprocess.SubprocessError):
        return {"sha": None, "dirty": None}


RESULT_PATHS = ("software", "experiments", "scripts", "schema", "firmware")


def package_versions() -> dict[str, str | None]:
    import importlib.metadata

    import biomech

    import platform

    out: dict[str, str | None] = {"python": platform.python_version(),
                                  "biomech": biomech.__version__}
    # the analysis packages a methods section names (2026-11-06: Python,
    # scikit-learn, spm1d, pingouin, statsmodels), plus the numeric stack and OpenCV
    for pkg in ("jtsmm", "jaw_tracking_system", "jtsmm-ml", "numpy", "scipy", "pandas", "h5py",
                "joblib", "scikit-learn", "spm1d", "pingouin", "statsmodels", "opencv-python",
                "opencv-contrib-python", "matplotlib", "pyarrow"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None                     # strict JSON: NaN/inf become null
    return obj


def write_json(path: str | Path, payload: Any) -> Path:
    """Indented strict JSON (NaN -> null, numpy -> Python); creates parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")
    return path


def write_csv(path: str | Path, rows: list[dict[str, Any]],
              fieldnames: list[str] | None = None) -> Path:
    """One row per dict; columns from ``fieldnames`` or the union of keys in first-seen order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            fieldnames.extend(k for k in row if k not in fieldnames)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fieldnames})
    return path


def _session_row(s: Any) -> dict[str, Any]:
    if isinstance(s, (str, Path)):
        return {"name": Path(s).name, "path": str(s)}
    if isinstance(s, dict):
        return {k: _jsonable(v) for k, v in s.items()}
    row = {"name": getattr(s, "name", str(s)), "path": str(getattr(s, "path", ""))}
    for key in ("pid", "subject_id", "take_kind", "kind", "config_hash", "software_version",
                "findings"):
        if hasattr(s, key):
            row[key] = _jsonable(getattr(s, key))
    return row


def write_provenance(out_dir: str | Path, study: str, script: str,
                     sessions: Iterable[Any] = (), *, data_dir: str | Path | None = None,
                     extra: dict[str, Any] | None = None) -> Path:
    """``<out_dir>/provenance.json``: what ran, on which takes, with which code.

    ``sessions`` may hold :class:`~biomech.io.inventory.TakeInventory`,
    :class:`~biomech.cohort.SessionInfo`, paths or dicts.
    """

    rows = [_session_row(s) for s in sessions]
    payload: dict[str, Any] = {
        "study": study,
        "script": script,
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "argv": list(sys.argv),
        "python": sys.version.split()[0],
        "data_dir": str(data_dir) if data_dir is not None else None,
        "git": git_state(),
        "versions": package_versions(),
        "n_sessions": len(rows),
        "sessions": rows,
    }
    if extra:
        payload["extra"] = extra
    return write_json(Path(out_dir) / PROVENANCE_NAME, payload)


__all__ = ["PROVENANCE_NAME", "write_csv", "write_json", "write_provenance"]
