#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Set one scalar in jtsmm.yaml in place, keeping every comment and the layout.

Used by build_install.sh (--data-dir -> host.paths.data_dir). A yaml load/dump
round trip would strip the template's comments, which are its documentation.

    set_config_value.py FILE host.paths.data_dir /home/pi/jtsmm_data [--if-present]

Only block-style mappings are handled (that is how the template is written).
Exits 1 when the key is not found, unless --if-present.
Stdlib only: runs before/without the jtsmm venv.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_KEY_LINE = re.compile(r"^(?P<indent>[ ]*)(?P<key>[A-Za-z0-9_]+):(?P<rest>.*?)(?P<eol>\r?\n?)$")


def _split_comment(rest: str) -> str:
    """Return the trailing ``  # comment`` of a value (with its spacing), or ''."""
    m = re.search(r"\s+#.*$", rest)
    return m.group(0) if m else ""


def set_scalar(text: str, dotted: str, value: str) -> tuple[str, bool]:
    """Replace the scalar at ``dotted`` in YAML ``text``; return (text, found)."""
    want = dotted.split(".")
    stack: list[tuple[int, str]] = []
    out: list[str] = []
    found = False
    for line in text.splitlines(keepends=True):
        m = _KEY_LINE.match(line)
        stripped = line.lstrip()
        if m is None or stripped.startswith("#"):
            if stripped.startswith("- "):
                # A list item opens a new scope at its indent; never a match target.
                indent = len(line) - len(stripped)
                while stack and stack[-1][0] >= indent:
                    stack.pop()
                stack.append((indent, "[]"))
            out.append(line)
            continue
        indent = len(m.group("indent"))
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, m.group("key")))
        if not found and [k for _, k in stack] == want:
            rest = m.group("rest")
            comment = _split_comment(rest)
            line = f"{m.group('indent')}{m.group('key')}: {json.dumps(value)}{comment}{m.group('eol')}"
            found = True
        out.append(line)
    return "".join(out), found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", type=Path)
    parser.add_argument("key", help="dotted path, e.g. host.paths.data_dir")
    parser.add_argument("value")
    parser.add_argument("--if-present", action="store_true", help="no error if the key is absent")
    args = parser.parse_args(argv)
    text = args.file.read_text(encoding="utf-8")
    new, found = set_scalar(text, args.key, args.value)
    if not found:
        if args.if_present:
            return 0
        print(f"error: {args.key} not found in {args.file}", file=sys.stderr)
        return 1
    args.file.write_text(new, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
