# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""The one ML core used by the ``ml`` CLI and the biomech stages.

``signals`` (causal EMG chain, masks, references, IMU signals), ``windows``
(trailing windows, gap rejection, window labels), ``features`` (vectorized
feature registry) and ``dataset`` (``WindowDataset`` and ``build_windows``).
"""
