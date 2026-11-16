# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Reader labels for references in tables: a BibTeX key of ``docs/analysis/references.bib``
becomes "Khare et al. 2012" (tables showed raw keys). The key itself stays in the table
notes / JSON so the manuscript can swap the label for its own ``\\cite``."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

_ENTRY = re.compile(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", re.S)


def _field(body: str, name: str) -> str:
    m = re.search(rf"\b{name}\s*=\s*[{{\"](.*?)[}}\"]\s*,?\s*\n", body, re.S)
    return re.sub(r"[{}]", "", m.group(1)).strip() if m else ""


@lru_cache(maxsize=2)
def _bib(path: str | None = None) -> dict[str, tuple[str, str]]:
    """Key -> (author field, year) of every entry."""
    if path is None:
        from biomech.paths import repo_root

        root = repo_root()
        if root is None:
            return {}
        p = root / "docs" / "analysis" / "references.bib"
    else:
        p = Path(path)
    if not p.is_file():
        return {}
    text = p.read_text(encoding="utf-8")
    return {m.group(1): (_field(m.group(2), "author"), _field(m.group(2), "year"))
            for m in _ENTRY.finditer(text)}


def _last(name: str) -> str:
    name = name.strip()
    return name.split(",")[0].strip() if "," in name else name.split()[-1] if name else ""


def cite_label(key: str, path: str | None = None) -> str:
    """'Last 2020', 'Last and Other 2020' or 'Last et al. 2020'; the key when unknown."""
    authors, year = _bib(path).get(str(key), ("", ""))
    if not authors:
        return str(key)
    names = [a for a in re.split(r"\s+and\s+", authors) if a.strip()]
    if len(names) == 1:
        who = _last(names[0])
    elif len(names) == 2:
        who = f"{_last(names[0])} and {_last(names[1])}"
    else:
        who = f"{_last(names[0])} et al."
    return f"{who} {year}".strip()


def tidy(text: str) -> str:
    """ASCII stand-ins of curated literature text as the reader glyphs ("+-" -> "±",
    "um" -> "µm")."""
    t = re.sub(r"\s*\+-\s*", " ± ", str(text))
    return re.sub(r"(?<=[\d\s])um\b", "µm", t)
