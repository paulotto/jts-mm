# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Configuration: class definitions (:mod:`.classes`), typed sections (:mod:`.schema`),
YAML loading (:mod:`.loader`) and data paths (:mod:`.paths`).

Only the constants are imported eagerly: :mod:`.schema` depends on :mod:`ml.core`,
which itself needs :mod:`.classes`, so the loaders are resolved on first use.
"""

from importlib import import_module
from typing import Any

from .classes import (CLASS_COUNT, CLASS_IDS, CLASS_NAMES, MUSCLE_CHANNEL_NAMES, PHASES,
                      UNLABELLED)

_LAZY = {
    "config_hash": "loader", "default_config_path": "loader", "describe_source": "loader",
    "label_section": "loader", "load_dataset_config": "loader", "load_ml_config": "loader",
    "load_train_config": "loader", "load_yaml": "loader", "package_config_path": "loader",
    "package_etc_dir": "loader", "DatasetConfig": "schema", "TrainConfig": "schema",
    "ensure_ml_data_dirs": "paths", "evals_dir": "paths", "ml_data_dir": "paths",
    "processed_dir": "paths", "resolve_paths_section": "paths", "runs_dir": "paths",
    "session_data_dir": "paths", "sessions_dir": "paths",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CLASS_COUNT", "CLASS_IDS", "CLASS_NAMES", "MUSCLE_CHANNEL_NAMES", "PHASES",
           "UNLABELLED", *_LAZY]
