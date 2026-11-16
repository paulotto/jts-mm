# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Labeling UI constants."""

from __future__ import annotations

from ml.config.classes import CLASS_NAMES, UNLABELLED

LABEL_COLORS: dict[int, str] = {
    0: "#888787",
    1: "#2196F3",
    2: "#FF5722",
    3: "#4CAF50",
    4: "#FF9800",
    5: "#9C27B0",
    6: "#E91E63",
    UNLABELLED: "#BDBDBD",     # unpainted / ignore: never used for training
}
# Paint buttons: the seven classes plus "ignore" (back to unlabelled).
PAINT_IDS: tuple[int, ...] = (*sorted(CLASS_NAMES), UNLABELLED)

AXIS_TITLES = (
    "X (Right / Left)  [mm]",
    "Y (Protrusion / Retrusion)  [mm]",
    "Z (Open / Close)  [mm]",
)

PHASE_COLORS = (
    "#f5cba7",
    "#a9cce3",
    "#abebc6",
    "#f5b7b1",
    "#d7bde2",
    "#f9e79f",
    "#a3e4d7",
    "#d5d8dc",
)


def display_class_name(class_id: int) -> str:
    if class_id == UNLABELLED:
        return "IGNORE"
    return CLASS_NAMES.get(class_id, str(class_id))
