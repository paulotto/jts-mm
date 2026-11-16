# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""Modern web viewer for EMG + IMU + camera MJPEG preview.

A modality-agnostic core (`Source` -> `Hub` -> server -> browser) that reuses the
C++ ``NativeCapture`` hot path. Supports standalone live (`jtsmm view`), replay
of recorded sessions (`jtsmm replay`), and in-recording viewing via a native
decimated tap (`jtsmm record --viewer`). Camera tiles use optical-style MJPEG
at ``/stream/{cam_id}``. See docs/software/s07.
"""

from __future__ import annotations

from jtsmm.viewer.hub import Hub
from jtsmm.viewer.sources import LiveSource, MockSource, ReplaySource, Source, TapSource

__all__ = ["Hub", "Source", "LiveSource", "TapSource", "ReplaySource", "MockSource"]
