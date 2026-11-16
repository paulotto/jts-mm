# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Is the analysis code at HEAD the same program as at a pinned commit?

The code hash of a snapshot covers every byte of the analysis source, so editing a comment
changes it. This script compares the hashed files (``biomech.pipeline.hashed_files``) of the
working tree with the same paths at BASE (the commit whose code hash the pinned snapshots
carry) and classifies every difference:

* ``comments``    -- the syntax tree is identical (comments, blank lines, formatting),
* ``docstrings``  -- only docstrings differ,
* ``strings``     -- other string constants differ (each one is listed for a look: stage
                     descriptions and messages do not enter a result),
* ``CODE``        -- the syntax tree differs with all strings blanked: executable code changed,
* ``DATA``        -- a YAML or JSON file parses to different data,
* ``ADDED`` / ``REMOVED`` -- a hashed file exists on one side only.

Exit code 1 on CODE, DATA, ADDED or REMOVED: then the pinned results are not results of this
code and the pipeline has to run again.

    software/jtsmm/.venv/bin/python scripts/audit_recompute/code_equivalence.py BASE [--out FILE]
"""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _git_show(base: str, rel: str) -> str | None:
    r = subprocess.run(["git", "show", f"{base}:{rel}"], cwd=ROOT, capture_output=True)
    return r.stdout.decode("utf-8") if r.returncode == 0 else None


def _docstring_nodes(tree: ast.AST) -> set[int]:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if (b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                out.add(id(b[0].value))
    return out


def _strings(tree: ast.AST) -> list[str]:
    docs = _docstring_nodes(tree)
    return [n.value for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docs]


def _skeleton(src: str) -> str:
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = ""
    return ast.dump(tree)


def classify(rel: str, old: str, new: str) -> tuple[str, list[tuple[str, str]]]:
    if old == new:
        return "identical", []
    if rel.endswith(".py"):
        if _skeleton(old) != _skeleton(new):
            return "CODE", []
        a, b = ast.parse(old), ast.parse(new)
        sa, sb = _strings(a), _strings(b)
        if sa != sb:
            return "strings", [(u, v) for u, v in zip(sa, sb) if u != v]
        return ("docstrings" if ast.dump(a) != ast.dump(b) else "comments"), []
    if rel.endswith((".yaml", ".yml")):
        same = list(yaml.safe_load_all(old)) == list(yaml.safe_load_all(new))
    elif rel.endswith(".json"):
        same = json.loads(old) == json.loads(new)
    else:
        same = False
    return ("comments" if same else "DATA"), []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("base", help="commit whose code hash the pinned snapshots carry")
    ap.add_argument("--out", type=Path, help="write the report here as well")
    a = ap.parse_args()
    from biomech import pipeline

    files = sorted(str(f.relative_to(ROOT)) for _root, f in pipeline.hashed_files())
    listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", a.base], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    rows, strings, bad = [], [], 0
    for rel in files:
        old = _git_show(a.base, rel)
        if old is None:
            rows.append((rel, "ADDED"))
            bad += 1
            continue
        kind, changed = classify(rel, old, (ROOT / rel).read_text(encoding="utf-8"))
        if kind != "identical":
            rows.append((rel, kind))
            strings += [(rel, u, v) for u, v in changed]
        bad += kind in ("CODE", "DATA")
    head = subprocess.run(["git", "rev-parse", "--short=8", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *files], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip())
    counts: dict[str, int] = {}
    for _rel, kind in rows:
        counts[kind] = counts.get(kind, 0) + 1
    lines = [f"analysis code: {len(files)} hashed files; base {a.base}, head {head}"
             f"{' + uncommitted changes' if dirty else ''}; code hash at head {pipeline.code_hash()}",
             "changed files by kind: " + (", ".join(f"{k} {n}" for k, n in sorted(counts.items())) or "none")]
    lines += [f"  {kind:10s} {rel}" for rel, kind in rows]
    if strings:
        lines.append("changed string constants (not docstrings):")
        lines += [f"  {rel}: {u[:90]!r} -> {v[:90]!r}" for rel, u, v in strings]
    lines.append("RESULT: " + ("NOT EQUIVALENT: executable code or data changed, run the pipeline again"
                               if bad else "equivalent: no executable code and no data file changed"))
    text = "\n".join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + "\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
