# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Motion classes, channel and phase names: the single source of truth for the ML package."""

from __future__ import annotations

# Canonical order matches docs/ml/README.md §4.
CLASS_NAMES: dict[int, str] = {
    0: "REST",
    1: "OPEN",
    2: "CLOSE",
    3: "PROTRUSION",
    4: "RETRUSION",
    5: "LEFT",
    6: "RIGHT",
}

CLASS_IDS: dict[str, int] = {name: idx for idx, name in CLASS_NAMES.items()}
CLASS_COUNT = len(CLASS_NAMES)

# Label sidecars store uint8; 255 marks samples nobody labelled (ignored downstream,
# never REST). Inside the window pipeline the same samples are -1.
UNLABELLED = 255
IGNORE = -1

# The six muscle channels; this order is the feature-block order. Channels are selected by
# name (session.h5 /emg channel_names, digastrics corrected on read for files before
# 2026-11-03, an03 §1), so the order need not follow CH1..CH6 (since 2026-11-03 the recorder
# writes CH3 digastric_r, CH4 digastric_l).
MUSCLE_CHANNEL_NAMES: tuple[str, ...] = (
    "masseter_r",
    "temporalis_r",
    "digastric_l",
    "digastric_r",
    "masseter_l",
    "temporalis_l",
)
DIGASTRIC: tuple[str, ...] = ("digastric_l", "digastric_r")

# Experiment phases of the ICORR protocol. ``rest`` is unlabelled baseline (feature
# thresholds, false activations); ``free_motion`` is never trained on.
PHASES: tuple[str, ...] = (
    "rest", "open_close", "protrusion_retrusion", "left_right", "chewing", "free_motion",
)
REST_PHASE = "rest"
FREE_PHASE = "free_motion"

LABEL_SCHEMA_VERSION = "1.1"
READABLE_LABEL_SCHEMAS = ("1.0", "1.1")
