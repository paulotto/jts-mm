# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""MCU master-clock alignment for live optical capture."""

from optical.sync.frame_meta import CamFrameMeta
from optical.sync.hat_sync import HatSyncBridge

__all__ = ["CamFrameMeta", "HatSyncBridge"]
