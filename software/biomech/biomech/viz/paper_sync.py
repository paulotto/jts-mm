# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Copy a paper export into a manuscript repository (``biomech paper-sync``).

The export folder (``biomech export --paper ...``) holds everything a paper set produces; the
manuscript needs a small, self-contained part of it under one folder of its own (default
``results/``), named by the paper labels of ``paper_*.yaml``:

    results/
      results.tex          one \\input: helper macros, numbers, captions (pgf style if pgfplots)
      numbers.tex / .json  \\numX macros (biomech numbers)
      captions.tex         \\figcaption{label}, \\figsources{label}
      jtsmm-pgfstyle.tex   symbols and colours the captions and tables use
      figures/<label>.pdf  caption-free figure (the journal upload file)
      tables/<label>.tex   table body (booktabs / tabularx)
      source_data/<label>.csv   only with ``source_data=True``
      manifest.json        provenance of the export + sha256 of every file (check-paper reads it)
      README.md            what is here, how to include it, how to refresh it

Every copied file is checked against the sha256 the export recorded (a partial or edited
export is refused). The folder is updated in place, never deleted and recreated: a file is
written only when its content changed (via a ``.tmp`` name, then an atomic replace), and only
files listed in the previous manifest are removed (MEGA sync mistakes recreated folders for
moves; memory note 2026-11-05). A folder that exists, is not empty and has no manifest of this
command is not touched.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GENERATOR = "biomech paper-sync"
# Helper definitions of jtsmm-pgfstyle.tex that tables and captions need without pgfplots.
_HELPER_PREFIXES = ("\\def\\jtsmm", "\\ifdefined\\DeclareSIUnit", "\\definecolor")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_if_changed(dest: Path, data: bytes) -> bool:
    """Write ``data`` to ``dest`` unless it already holds it; True if written."""
    if dest.is_file() and dest.read_bytes() == data:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")       # *.tmp: ignored by MEGA
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return True


def plan(export_dir: str | Path, source_data: bool = False,
         additional: bool = False) -> tuple[dict[str, Any], dict[str, Path]]:
    """(export manifest, {destination path relative to the results folder: source file}),
    every source checked against the sha256 the export recorded."""
    ex = Path(export_dir)
    mf = ex / "manifest.json"
    if not mf.is_file():
        raise FileNotFoundError(f"{ex}: no manifest.json (not a biomech export folder)")
    man = json.loads(mf.read_text())
    files: dict[str, Path] = {}
    bad: list[str] = []

    def take(rel_src: str, rel_dest: str, recorded: str | None) -> None:
        src = Path(rel_src) if Path(rel_src).is_absolute() else ex / rel_src
        if not src.is_file():
            bad.append(f"{rel_src}: missing")
            return
        if recorded and _sha(src) != recorded:
            bad.append(f"{rel_src}: differs from the export manifest (export again)")
            return
        if rel_dest in files:
            bad.append(f"{rel_dest}: two items map to the same file")
            return
        files[rel_dest] = src

    for it in man.get("items") or []:
        label, iid, recorded = it.get("label") or it["id"], it["id"], it.get("files") or {}
        if it.get("kind") == "figure":
            rel = f"figures/{iid}/{iid}.pdf"
            take(rel, f"figures/{label}.pdf", recorded.get(rel))
            if source_data:
                rel = f"figures/{iid}/source_data.csv"
                if (ex / rel).is_file():
                    take(rel, f"source_data/{label}.csv", recorded.get(rel))
        elif it.get("kind") == "table":
            rel = f"tables/{iid}.tex"
            take(rel, f"tables/{label}.tex", recorded.get(rel))
            if source_data:
                rel = f"tables/{iid}.csv"
                if (ex / rel).is_file():
                    take(rel, f"source_data/{label}.csv", recorded.get(rel))
    for name in ("numbers.tex", "numbers.json", "captions.tex", "jtsmm-pgfstyle.tex"):
        take(name, name, None)
    for e in man.get("additional") or []:   # small data files (CSV) always (2026-11-10)
        if e.get("kind") == "data":
            take(e["file"], f"additional/{e['label']}.csv", e.get("sha256"))
    if additional:                  # interactive HTML and videos (large: opt-in, 2026-11-05)
        for e in man.get("additional") or []:
            if e.get("kind") == "interactive":
                take(e["file"], f"additional/{e['label']}.html", e.get("sha256"))
            elif e.get("kind") == "video":
                take(e["source"], f"additional/{e['label']}{Path(e['source']).suffix}",
                     e.get("sha256"))
    if bad:
        raise ValueError("export not usable:\n  " + "\n  ".join(bad))
    return man, files


def _results_tex(folder: str, style: str) -> str:
    """The one file a manuscript inputs: the helper macros (without pgfplots), the full pgf
    style when pgfplots is loaded, the numbers and the captions."""
    helpers = [ln for ln in style.splitlines() if ln.startswith(_HELPER_PREFIXES)]
    colors = [ln for ln in helpers if ln.startswith("\\definecolor")]
    rest = [ln for ln in helpers if not ln.startswith("\\definecolor")]
    return "\n".join(
        [f"% Generated by {GENERATOR} -- do not edit (README.md).",
         "% Needs siunitx (v3); tables need booktabs and tabularx.",
         "\\ifdefined\\pgfplotsset",
         f"  \\input{{{folder}/jtsmm-pgfstyle.tex}}",
         "\\else",
         *("  " + ln for ln in rest),
         "  \\ifdefined\\definecolor",
         *("    " + ln for ln in colors),
         "  \\fi",
         "\\fi",
         f"\\input{{{folder}/numbers.tex}}",
         f"\\input{{{folder}/captions.tex}}", ""])


def _versions_lines(prov: dict[str, Any]) -> list[str]:
    """The software versions that produced the numbers (the snapshot record) and that wrote
    the export, for the methods section."""
    out = []
    for key, what in (("versions_analysis", "Software of the analysis"),
                      ("versions_export", "Software of the export")):
        v = prov.get(key) or {}
        if v:
            out.append(f"- {what}: " + ", ".join(f"{k} {x}" for k, x in v.items() if x))
    return out


def _readme(man: dict[str, Any], folder: str, command: str, items: list[dict[str, Any]]) -> str:
    prov, meta = man.get("provenance") or {}, man.get("meta") or {}
    git = prov.get("git") or {}
    rows = [f"| `{i['label']}` | {i['kind']} | {i['part']} | {i['section']} | `{i['file']}` |"
            for i in items]
    add = man.get("additional") or []
    add_md = ([
        "## Additional files (data, interactive 3-D, videos)", "",
        "Data files (CSV, small) are copied by every sync to `additional/<label>.csv`. The "
        "interactive files and videos are not copied by a plain sync (tens of MB); "
        "`paper-sync --additional` (or `make sync-paper-additional`) copies them to "
        "`additional/<label>.html|.mp4`:", "",
        "| Label | Kind | What |", "|---|---|---|",
        *[f"| `{e['label']}` | {e['kind']} | "
          f"{(e.get('describe') or e.get('caption') or '').split('. ')[0].replace('|', '/')} |"
          for e in add], ""] if add else [])
    return "\n".join([
        f"# {folder}/ -- generated from the analysis results",
        "",
        f"Written by `{GENERATOR}`. **Do not edit anything here**: every file is regenerated",
        "from the analysis export and overwritten. Change numbers, figures, tables and captions",
        "in the analysis repository (views, `numbers_*.yaml`, `paper_*.yaml`), export, and sync again:",
        "",
        "```bash",
        command,
        "```",
        "",
        f"- Paper set: `{Path(str(meta.get('paper_file', ''))).name}`, target "
        f"`{meta.get('target')}`",
        f"- Results: study `{prov.get('study')}`, snapshot `{prov.get('snapshot')}`"
        + (f", variant `{prov['variant']}`" if prov.get("variant") else ""),
        f"- Code: `{str(git.get('sha', ''))[:10]}`" + (" (uncommitted changes)"
                                                       if git.get("dirty") else ""),
        f"- Exported: {prov.get('exported_utc')}",
        *_versions_lines(prov),
        "",
        "## Use in the manuscript",
        "",
        "```latex",
        f"\\input{{{folder}/results}}                    % preamble: numbers, captions, symbols",
        f"\\includegraphics{{{folder}/figures/<label>}}  % a figure, caption-free PDF",
        "\\caption{\\figcaption{<label>}}                % its caption from the export",
        f"\\input{{{folder}/tables/<label>}}             % a table body (tabularx, booktabs)",
        "\\numX, \\numXVal, \\numXCI, \\numXN           % in-text numbers (numbers.tex)",
        "```",
        "",
        "Check the manuscript against this folder (every `\\numX` and `\\figcaption{label}` "
        "defined, one reporting population):",
        "",
        "```bash",
        f"biomech check-paper <manuscript folder> --export <manuscript folder>/{folder}",
        "```",
        "",
        *add_md,
        "## Items",
        "",
        "| Label | Kind | Part | Section | File |",
        "|---|---|---|---|---|",
        *rows, ""])


def sync(export_dir: str | Path, dest: str | Path, *, source_data: bool = False,
         command: str = "", dry_run: bool = False, additional: bool = False) -> dict[str, Any]:
    """Bring ``dest`` (the manuscript's results folder) up to date with ``export_dir``.
    Returns {"written": [...], "unchanged": [...], "removed": [...]} (paths relative to
    ``dest``)."""
    dest = Path(dest)
    old_mf = dest / "manifest.json"
    old: dict[str, Any] = {}
    if old_mf.is_file():
        old = json.loads(old_mf.read_text())
    if dest.exists() and any(dest.iterdir()) and (old.get("sync") or {}).get(
            "generator") != GENERATOR:
        raise FileExistsError(f"{dest} exists and was not written by {GENERATOR}; "
                              "choose another folder or empty it")
    man, files = plan(export_dir, source_data, additional)
    folder = dest.name
    by_id = {it["id"]: it for it in man.get("items") or []}
    items = []
    for rel in sorted(files):
        if rel.startswith(("figures/", "tables/")):
            label = Path(rel).stem
            it = next(i for i in by_id.values() if (i.get("label") or i["id"]) == label)
            items.append({"label": label, "id": it["id"], "kind": it["kind"],
                          "part": it.get("part", ""), "section": it.get("section", ""),
                          "file": rel})
    contents: dict[str, bytes] = {rel: src.read_bytes() for rel, src in files.items()}
    contents["results.tex"] = _results_tex(
        folder, (Path(export_dir) / "jtsmm-pgfstyle.tex").read_text()).encode()
    contents["README.md"] = _readme(man, folder, command, items).encode()
    sync_meta = {"generator": GENERATOR, "export": str(Path(export_dir).resolve()),
                 "synced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "source_data": source_data}
    new_files = {rel: hashlib.sha256(b).hexdigest() for rel, b in sorted(contents.items())}
    if not additional:              # a plain sync keeps the additional files of an earlier one
        new_files.update({rel: h for rel, h in (old.get("files") or {}).items()
                          if rel.startswith("additional/") and (dest / rel).is_file()})
    mf = {"schema": 1, "sync": sync_meta, "provenance": man.get("provenance"),
          "meta": man.get("meta"),
          "items": [{**i, "sha256": new_files[i["file"]]} for i in items],
          "files": new_files}
    written, unchanged = [], []
    stale = sorted(set((old.get("files") or {})) - set(new_files) - {"manifest.json"})
    if dry_run:
        for rel, b in contents.items():
            (unchanged if (dest / rel).is_file() and (dest / rel).read_bytes() == b
             else written).append(rel)
        return {"written": sorted(written), "unchanged": sorted(unchanged), "removed": stale}
    for rel, b in contents.items():
        (written if _write_if_changed(dest / rel, b) else unchanged).append(rel)
    for rel in stale:                          # only files this command wrote before
        p = dest / rel
        if p.is_file():
            p.unlink()
    # the manifest last: its file list is what the next run may remove; unchanged content
    # keeps the old manifest (no rewrite for a new timestamp alone)
    prev = {k: v for k, v in old.items() if k != "sync"}
    cur = {k: v for k, v in mf.items() if k != "sync"}
    if prev != cur or (old.get("sync") or {}).get("source_data") != source_data:
        _write_if_changed(old_mf, (json.dumps(mf, indent=1, ensure_ascii=False) + "\n")
                          .encode())
        written.append("manifest.json")
    return {"written": sorted(written), "unchanged": sorted(unchanged), "removed": stale}
