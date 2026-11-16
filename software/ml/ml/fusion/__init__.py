# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Sensor fusion: early (per-block scaling + concatenation) and late (posterior combiners)."""

from .early import BlockScaler, concat_blocks
from .late import LateFusionClassifier, late_fusion_posteriors

__all__ = ["BlockScaler", "LateFusionClassifier", "concat_blocks", "late_fusion_posteriors"]
