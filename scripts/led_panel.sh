#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Walk panel GPIO + USER RGB LEDs (~4 s). Waits for "panel test OK".
set -euo pipefail
PORT="${1:-/dev/ttyACM0}"
exec 3<>"${PORT}"
stty -F "${PORT}" raw -echo min 0 time 0 2>/dev/null || true
printf 'led panel\r\n' >&3
timeout 8 sed '/panel test OK/ q' <&3
exec 3<&-
