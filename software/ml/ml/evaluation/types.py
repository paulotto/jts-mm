# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Shared evaluation result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PredictionBundle:
    """Model outputs aligned to labeled windows."""

    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray | None = None
    t_reference_us: np.ndarray | None = None     # window end (t_end_us)
    groups: np.ndarray | None = None
    indices: np.ndarray | None = None


@dataclass
class EvaluationResult:
    """Metrics plus the predictions that produced them."""

    metrics: dict[str, Any]
    predictions: PredictionBundle
    extra: dict[str, Any] = field(default_factory=dict)
