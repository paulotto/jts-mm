#!/usr/bin/env python3
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Verify etc/jtsmm.yaml against schema + coverage (exit 1 on any problem).

Checks: the template passes the full strict validation (`jtsmm config validate`:
schema, every required manifest key, cross checks); every other schema key is
listed, live or commented out, so the template stays the complete reference.
`--sync-mcu-golden` regenerates schema/v1/golden/mcu_apply_template.json instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software" / "jtsmm"))

from jtsmm.config import build_mcu_apply_payload  # noqa: E402
from jtsmm.config.coverage import schema_leaf_paths  # noqa: E402
from jtsmm.config.loader import load_yaml, repo_root  # noqa: E402
from jtsmm.config.strict import _has_path, _load_coverage_required, validate_strict  # noqa: E402

# Schema keys the template deliberately does not list.
_NOT_IN_TEMPLATE = (
    "host.storage.",  # deprecated alias of host.paths.data_dir
    "host.emg_filter.enabled",  # deprecated, no effect
    "host.emg_filter.notch_hz",
    "host.emg_filter.highpass_hz",
    "host.emg_filter.lowpass_hz",
    "mcu.platform.custom",  # profile-override blocks (doctor only)
    "mcu.emg.custom",
    "mcu.imu_mta.custom",
    "mcu.imu_cra.custom",
    "mcu.trigger.custom",
)


def template_path() -> Path:
    return ROOT / "software" / "jtsmm" / "etc" / "jtsmm.yaml"


def golden_mcu_apply_path() -> Path:
    return repo_root() / "schema" / "v1" / "golden" / "mcu_apply_template.json"


def missing_required_paths(doc: dict) -> list[str]:
    return [p for p in _load_coverage_required() if not _has_path(doc, p)]


def unlisted_schema_keys(doc: dict, text: str) -> list[str]:
    """Schema leaves that are neither live in ``doc`` nor a ``# key:`` comment in ``text``."""
    commented = set(re.findall(r"^\s*#\s*([A-Za-z0-9_]+):", text, re.M))
    out = []
    for path in sorted(schema_leaf_paths()):
        if path.startswith(_NOT_IN_TEMPLATE) or ".reports." in path:
            continue
        if _has_path(doc, path) or path.rsplit(".", 1)[-1] in commented:
            continue
        out.append(path)
    return out


def sync_mcu_apply_golden() -> Path:
    doc = load_yaml(template_path())
    payload = build_mcu_apply_payload(doc)
    out = golden_mcu_apply_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return out


def check(template: Path) -> list[str]:
    doc = load_yaml(template)
    # Strict validation includes the manifest's required keys (missing_required_paths).
    problems = [i for i in validate_strict(doc, path=template) if i.startswith("error:")]
    problems += [
        f"schema key not listed (add it, commented out if optional): {p}"
        for p in unlisted_schema_keys(doc, template.read_text(encoding="utf-8"))
    ]
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="Accepted for compatibility; checking is the default"
    )
    parser.add_argument(
        "--sync-mcu-golden",
        action="store_true",
        help="Regenerate schema/v1/golden/mcu_apply_template.json from etc/jtsmm.yaml",
    )
    parser.add_argument("--template", type=Path, default=template_path())
    args = parser.parse_args(argv)

    if args.sync_mcu_golden:
        print(f"Wrote {sync_mcu_apply_golden()}")
        return 0

    problems = check(args.template)
    if problems:
        print(f"{args.template}: {len(problems)} problem(s):", file=sys.stderr)
        for msg in problems:
            print(f"  - {msg}", file=sys.stderr)
        return 1
    print(f"OK: {args.template} ({len(_load_coverage_required())} required paths present)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
