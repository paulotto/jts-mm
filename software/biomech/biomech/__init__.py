# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Jaw biomechanics and neuromechanics analysis over recorded JTS-MM sessions.

Layering (docs/analysis/README.md): ``jtsmm`` signal primitives -> ``jts``
kinematics -> ``biomech`` study analysis -> ``ml`` intent decoding. Outputs are
written and read only through :class:`biomech.results.ResultStore`; analyses run
only through :func:`biomech.pipeline.run`.
"""

__version__ = "0.1.0"
