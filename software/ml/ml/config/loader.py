# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Load ``ml.yaml``.

The packaged ``software/ml/etc/ml.yaml`` is the default. Another file is used only
when passed explicitly (``--config``); ``~/ml.yaml`` is *not* picked up silently
any more (it used to override the packaged defaults without notice). Every CLI
command prints which file it used (:func:`describe_source`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import DatasetConfig, TrainConfig

_PACKAGE_ETC = Path(__file__).resolve().parents[2] / "etc"
_PACKAGE_YAML = _PACKAGE_ETC / "ml.yaml"
HOME_YAML = Path.home() / "ml.yaml"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def package_etc_dir() -> Path:
    return _PACKAGE_ETC


def default_config_path() -> Path:
    """The packaged ``software/ml/etc/ml.yaml`` (never ``~/ml.yaml``)."""
    return _PACKAGE_YAML


def package_config_path(name: str = "ml.yaml") -> Path:
    return _PACKAGE_ETC / name


def resolve_config(path: str | Path | None) -> Path:
    return Path(path).expanduser() if path is not None else default_config_path()


def describe_source(path: str | Path | None) -> str:
    """One line naming the config file in use (and an ignored ``~/ml.yaml``)."""
    p = resolve_config(path)
    msg = f"config: {p} ({'--config' if path is not None else 'packaged default'})"
    if path is None and HOME_YAML.is_file():
        msg += f"; note: {HOME_YAML} is ignored unless passed with --config"
    return msg


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping: {path}")
    return data


def load_ml_config(path: str | Path | None = None) -> dict[str, Any]:
    return load_yaml(resolve_config(path))


def label_section(doc: dict[str, Any]) -> dict[str, Any]:
    return dict(doc.get("label") or {})


def _with_output(section: dict[str, Any], doc: dict[str, Any]) -> dict[str, Any]:
    from .paths import resolve_paths_section

    resolved = resolve_paths_section(doc)
    out = dict(section.get("output") or {})
    for k in ("processed_dir", "runs_dir", "evals_dir"):
        out.setdefault(k, resolved[k])
    return {**section, "output": out}


def load_dataset_config(path: str | Path | None = None) -> DatasetConfig:
    doc = load_ml_config(path)
    sec = dict(doc.get("dataset") or {})
    sec.setdefault("seed", doc.get("seed", 42))
    return DatasetConfig.from_dict(_with_output(sec, doc))


def load_train_config(path: str | Path | None = None, **overrides: Any) -> TrainConfig:
    doc = load_ml_config(path)
    sec = dict(doc.get("train") or {})
    sec.setdefault("seed", doc.get("seed", 42))
    sec.update({k: v for k, v in overrides.items() if v is not None})
    return TrainConfig.from_dict(_with_output(sec, doc))


def config_hash(data: Any) -> str:
    """Short stable hash of a config mapping or dataclass."""
    payload = asdict(data) if is_dataclass(data) else data
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]
