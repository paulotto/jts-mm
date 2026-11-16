# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Export manifest and caption macros.

Every export folder gets

* ``manifest.json``: what was exported, from what, and the files' fingerprints::

    {"schema": 1, "meta": {"paper": ..., "target": ..., ...}, "venue": "ieee",
     "provenance": {"study", "variant", "snapshot", "newest_run", "git", "exported_utc"},
     "items": [{"label", "part", "section", "view", "params", "kind", "id", "caption",
                "caption_latex", "sources", "files": {"figures/x/fig.tex": "<sha256>", ...}}]}

* ``captions.tex`` (items with a label): ``\\figcaption{<label>}`` expands to the item's
  caption, ``\\figsources{<label>}`` to its sources; it needs ``jtsmm-pgfstyle.tex`` (written
  by the same export) for the symbols.

``check/integrity.py`` reads the manifest to verify the files and to rebuild every item from
the pinned snapshot.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def item_files(out: Path, item: dict[str, Any]) -> list[Path]:
    """The files an exported item wrote (relative paths exist under ``out``)."""
    files: list[Path] = []
    if item.get("figure"):
        fid = item["figure"]
        fdir = out / "figures" / fid
        files += sorted(p for p in fdir.rglob("*") if p.is_file()) if fdir.is_dir() else []
        files += [p for p in (out / "previews" / f"{fid}.png", out / "previews" / f"{fid}.pdf")
                  if p.is_file()]
    elif item.get("table"):
        files += [p for p in (out / "tables" / f"{item['table']}{s}"
                              for s in (".tex", ".csv", ".json")) if p.is_file()]
    cap = out / "captioned" / f"{item.get('figure') or item.get('table')}.pdf"
    if cap.is_file():
        files.append(cap)
    return files


def caption_latex(text: str) -> str:
    from biomech.viz import render_pgf

    return render_pgf._esc(text or "")


def write(out: Path, done: dict[str, Any], ctx: Any, *, venue: str,
          meta: dict[str, Any] | None = None) -> Path:
    """Write ``manifest.json`` and ``captions.tex`` for the items of one export."""
    from biomech.viz import serialize

    out = Path(out)
    prune(out, {r.get("figure") or r.get("table") for r in done.get("items", [])})
    items = []
    for r in done.get("items", []):
        files = item_files(out, r)
        items.append({
            "label": r.get("label"), "part": r.get("part"),
            "section": r.get("section") or "", "view": r["view"],
            "params": r.get("params", {}), "kind": "figure" if r.get("figure") else "table",
            "id": r.get("figure") or r.get("table"), "caption": r.get("caption", ""),
            # a written caption (paper set ``captions:``) replaces the generated one verbatim
            "caption_latex": r.get("caption_tex") or caption_latex(r.get("caption", "")),
            "caption_source": "written" if r.get("caption_tex") else "view",
            "sources": r.get("sources", []),
            "files": {str(p.relative_to(out)): sha256(p) for p in files}})
    doc = {"schema": SCHEMA, "meta": dict(meta or {}), "venue": venue,
           "provenance": serialize.provenance(ctx), "items": items}
    path = out / "manifest.json"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    labelled = [i for i in items if i["label"]]
    if labelled:
        lines = ["% Captions of the exported items (biomech export). Needs",
                 "% jtsmm-pgfstyle.tex for symbols. \\figcaption{label}, \\figsources{label}.",
                 "\\providecommand{\\figcaption}[1]{\\csname jtsmmcap@#1\\endcsname}",
                 "\\providecommand{\\figsources}[1]{\\csname jtsmmsrc@#1\\endcsname}"]
        for i in labelled:
            lines.append(f"\\expandafter\\def\\csname jtsmmcap@{i['label']}\\endcsname"
                         f"{{{i['caption_latex']}}}")
            src = caption_latex(", ".join(i["sources"]))
            lines.append(f"\\expandafter\\def\\csname jtsmmsrc@{i['label']}\\endcsname{{{src}}}")
        (out / "captions.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def prune(out: Path, keep: set[str]) -> list[str]:
    """Remove the files of items the previous export of this folder wrote (its manifest)
    that the new export no longer contains: a paper folder holds exactly the current set.
    Files the manifest never listed (hand-placed) are left alone."""
    import shutil

    if not (out / "manifest.json").is_file():
        return []
    removed = []
    for it in read(out).get("items", []):
        if it.get("id") in keep:
            continue
        for rel in it.get("files", {}):
            p = out / rel
            if p.is_file():
                p.unlink()
                removed.append(rel)
        fdir = out / "figures" / str(it.get("id"))
        if fdir.is_dir() and not any(fdir.iterdir()):
            shutil.rmtree(fdir, ignore_errors=True)
    return removed


def read(out: Path) -> dict[str, Any]:
    return json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
