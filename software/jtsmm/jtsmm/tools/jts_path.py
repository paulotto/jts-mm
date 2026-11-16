# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Import JTS from the vendored ``software/jaw_tracking`` copy.

The vendored copy is the one jtsmm is tested against. A stale ``jts`` installed
elsewhere (e.g. a plain copy in a conda ``site-packages``) must not shadow it:
older versions lack ``jts.aruco_camera`` and use exclusive frame-interval ends.
"""

from __future__ import annotations

import sys
from pathlib import Path

JTS_ROOT = Path(__file__).resolve().parents[3] / "jaw_tracking"


def _loaded_from(root: Path) -> bool:
    mod = sys.modules.get("jts")
    f = getattr(mod, "__file__", None)
    return f is not None and Path(f).resolve().parent.parent == root.resolve()


def ensure_jts_on_path(root: Path = JTS_ROOT) -> Path:
    """Make ``import jts`` resolve to the vendored copy; return its root.

    Without the vendored tree (jtsmm installed on its own) any installed ``jts``
    is used instead.
    """
    if not (root / "jts" / "__init__.py").is_file():
        try:
            import jts
        except ImportError as exc:
            raise ImportError(
                "the JTS package is required (vendored at software/jaw_tracking). "
                "Install it with: pip install -e software/jaw_tracking"
            ) from exc
        return Path(jts.__file__).resolve().parent.parent

    if "jts" in sys.modules:
        if _loaded_from(root):
            return root
        other = getattr(sys.modules["jts"], "__file__", "?")
        raise ImportError(
            f"a different jts is already imported ({other}); the vendored copy at "
            f"{root} is required. Remove the stale copy, then: "
            "pip install -e software/jaw_tracking"
        )

    entry = str(root)
    if entry in sys.path:
        sys.path.remove(entry)
    sys.path.insert(0, entry)
    import jts  # noqa: F401

    return root
