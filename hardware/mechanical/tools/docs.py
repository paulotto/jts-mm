# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Write the generated sections of README.md from parts.yaml (make docs).

Between ``<!-- parts:begin -->`` / ``<!-- parts:end -->``: one row per variant (status, CAD
source, size, volume, date, render link); between ``<!-- usage:begin -->`` /
``<!-- usage:end -->``: the version each study used. Edit parts.yaml, never these tables.
"""

from __future__ import annotations

import re
import sys

from common import INDEX, MECH, manifest, read_json, write_text

README = MECH / "README.md"


def parts_table(man, index) -> str:
    rows = ["| Part | Variant | Status | CAD | Size (mm) | Volume (cm³) | Date | Render |",
            "|---|---|---|---|---|---|---|---|"]
    for p in man.parts.values():
        for v in p.variants.values():
            g = index.get(f"{p.id}/{v.id}", {})
            cad = "pending" if v.cad_pending else (
                f"`{p.source(v).name}` {v.fcstd_body}" if v.fcstd_body else "STL only")
            size = " × ".join(f"{x:.0f}" for x in g.get("size_mm", []))
            vol = f"{g['volume_mm3'] / 1000:.2f}" if g.get("volume_mm3") else ""
            png = man.render(p.id, v.id)
            link = f"[iso](renders/{png.name})" if png.is_file() else ""
            stl = v.stl.relative_to(MECH) if v.stl and v.stl.is_relative_to(MECH) else None
            name = f"[{v.title}]({stl})" if stl else v.title
            rows.append(f"| {p.title} | {name} (`{v.id}`) | {v.status} | {cad} | {size} | {vol} "
                        f"| {v.date} | {link} |")
    return "\n".join(rows)


def usage_table(man) -> str:
    rows = ["| Study | " + " | ".join(man.parts[p].title for p in man.parts if any(
        p in u for u in man.usage.values())) + " |"]
    cols = [p for p in man.parts if any(p in u for u in man.usage.values())]
    rows.append("|---|" + "---|" * len(cols))
    for s, u in man.usage.items():
        cells = []
        for p in cols:
            if p not in u:
                cells.append("–")
            else:
                cells.append(f"`{u[p]}`" if u[p] else "*not yet specified*")
        rows.append(f"| {s} | " + " | ".join(cells) + " |")
    if man.overrides:
        rows.append("")
        rows.append("Per-participant overrides: " + "; ".join(
            f"{s} {pid}: {', '.join(f'{k}={v}' for k, v in m.items())}"
            for s, by in man.overrides.items() for pid, m in by.items()))
    return "\n".join(rows)


def replace(text: str, key: str, body: str) -> str:
    rx = re.compile(rf"(<!-- {key}:begin -->\n).*?(<!-- {key}:end -->)", re.S)
    if not rx.search(text):
        raise SystemExit(f"README.md has no <!-- {key}:begin/end --> markers")
    return rx.sub(lambda m: m.group(1) + body + "\n" + m.group(2), text)


def main() -> int:
    man = manifest()
    index = read_json(INDEX, {})
    text = README.read_text()
    text = replace(text, "parts", parts_table(man, index))
    text = replace(text, "usage", usage_table(man))
    write_text(README, text)
    print("README.md: parts and usage tables updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
