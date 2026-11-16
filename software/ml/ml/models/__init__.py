# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Classifiers behind one registry (:mod:`ml.models.registry`)."""

from .registry import REGISTRY, ModelSpec, get, names

__all__ = ["REGISTRY", "ModelSpec", "get", "names"]
