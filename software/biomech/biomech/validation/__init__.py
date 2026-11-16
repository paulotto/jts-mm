# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""System validation against Qualisys (motion) and Delsys (EMG): the separate
``jtsmm_validation`` study (plan D0–D7). Readers live in :mod:`biomech.io.reference`;
this package holds the sync, registration, sample selection and the study's stages
(registered with ``group="validation"``, so the dataset study never runs them)."""
