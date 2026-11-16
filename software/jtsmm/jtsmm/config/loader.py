# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-FileCopyrightText: 2026 David Stebich
# SPDX-License-Identifier: MIT

"""Load operator YAML and locate schema assets."""

from __future__ import annotations

import json
import os
import sys
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator

_LEGACY_CONFIG_PATH = Path("/etc/jtsmm.yaml")


def repo_root() -> Path:
    """Repository root as seen from this file.

    Only meaningful for a checkout / editable install; use :func:`checkout_root`
    when the answer must exist.
    """
    here = Path(__file__).resolve()
    # software/jtsmm/jtsmm/config/loader.py -> parents[4] = repo root
    return here.parents[4]


def checkout_root() -> Path | None:
    """The repository root when running from a checkout, else None (wheel install)."""
    root = repo_root()
    if (root / "schema" / "v1" / "jtsmm.schema.json").is_file() and (
        root / "software" / "jtsmm" / "etc" / "jtsmm.yaml"
    ).is_file():
        return root
    return None


def packaged_data_dir() -> Path:
    """``jtsmm/_data``: config assets bundled by setup.py for regular installs."""
    return Path(str(resources.files("jtsmm").joinpath("_data")))


def schema_dir() -> Path:
    """``schema/v1`` -- the checkout copy when present, else the packaged copy."""
    root = checkout_root()
    if root is not None:
        return root / "schema" / "v1"
    packaged = packaged_data_dir() / "schema" / "v1"
    if (packaged / "jtsmm.schema.json").is_file():
        return packaged
    raise FileNotFoundError(
        "jtsmm: schema/v1 not found -- neither a repository checkout nor packaged "
        f"data at {packaged}. Reinstall with scripts/install_pi.sh (editable) or "
        "`pip install software/jtsmm` from the checkout."
    )


def profiles_dir() -> Path:
    return schema_dir() / "profiles"


def template_path() -> Path:
    """The shipped operator template (``software/jtsmm/etc/jtsmm.yaml``)."""
    root = checkout_root()
    if root is not None:
        return root / "software" / "jtsmm" / "etc" / "jtsmm.yaml"
    return packaged_data_dir() / "etc" / "jtsmm.yaml"


def user_config_path() -> Path:
    """Where the operator's own config lives: ``$JTSMM_CONFIG`` or ``~/jtsmm.yaml``."""
    env = os.environ.get("JTSMM_CONFIG")
    if env:
        return Path(env).expanduser()
    return Path.home() / "jtsmm.yaml"


_TEMPLATE_WARNED = False


def default_config_path() -> Path:
    """File loaded when a command is given no ``--config``.

    ``$JTSMM_CONFIG`` wins when that file exists. A repository checkout uses
    ``software/jtsmm/etc/jtsmm.yaml`` (the file under version control). A wheel
    install uses ``~/jtsmm.yaml``, then ``/etc/jtsmm.yaml``, then the packaged
    template. That last fallback prints a one-line warning once per process:
    its session block is an example.
    """
    global _TEMPLATE_WARNED
    env = os.environ.get("JTSMM_CONFIG")
    if env:
        explicit = Path(env).expanduser()
        if explicit.is_file():
            return explicit
    if checkout_root() is not None:
        template = template_path()
        if template.is_file():
            return template
    user = Path.home() / "jtsmm.yaml"
    if user.is_file():
        return user
    if _LEGACY_CONFIG_PATH.is_file():
        return _LEGACY_CONFIG_PATH
    template = template_path()
    if template.is_file():
        if not _TEMPLATE_WARNED:
            _TEMPLATE_WARNED = True
            print(
                f"warning: no {user} -- running on the shipped template {template} "
                "(example session settings); create your own with `jtsmm config init`",
                file=sys.stderr,
            )
        return template
    return user


_LOADED_CONFIG_DIR: Path | None = None


def note_loaded_config(path: Path | str) -> None:
    """Remember the directory of the operator config just loaded (see resolve_config_path)."""
    global _LOADED_CONFIG_DIR
    _LOADED_CONFIG_DIR = Path(path).expanduser().resolve().parent


def resolve_config_path(value: str | Path) -> Path:
    """Expand ``~`` and anchor a relative path from jtsmm.yaml so it never depends on the CWD.

    Relative paths in the template are repo-relative (``software/optical/etc/...``),
    so they resolve against the checkout root when their first component exists
    there. Anything else relative resolves against the directory of the config
    file that was loaded (falling back to the checkout root, then the CWD).
    """
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    root = checkout_root()
    if root is not None and p.parts and (root / p.parts[0]).exists():
        return root / p
    base = _LOADED_CONFIG_DIR or root or Path.cwd()
    return base / p


def load_yaml(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: root must be a mapping")
    return data


def load_json_schema() -> dict[str, Any]:
    schema_path = schema_dir() / "jtsmm.schema.json"
    with schema_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def dotted_path(parts: Any) -> str:
    """``['mcu', 'emg', 'x']`` -> ``mcu.emg.x``; list indices as ``[i]``."""
    out = ""
    for part in parts:
        out += f"[{part}]" if isinstance(part, int) else (f".{part}" if out else str(part))
    return out


def validate_schema(doc: dict[str, Any]) -> list[str]:
    """Return JSON Schema validation errors, prefixed with their location (empty if valid)."""
    validator = Draft202012Validator(load_json_schema())
    out = []
    for e in validator.iter_errors(doc):
        where = dotted_path(e.absolute_path)
        out.append(f"{where}: {e.message}" if where else e.message)
    return out
