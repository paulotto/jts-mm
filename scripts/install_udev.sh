#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Install jtsmm udev rules for stable /dev/jtsmm_hat symlink and libusb access.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/software/jtsmm/etc/udev/99-jtsmm.rules"
DST="/etc/udev/rules.d/99-jtsmm.rules"

if [[ ! -f "${SRC}" ]]; then
  echo "ERROR: missing ${SRC}" >&2
  exit 1
fi

if [[ -f "${DST}" ]] && cmp -s "${SRC}" "${DST}"; then
  echo "udev rules already installed: ${DST}"
  exit 0
fi

echo "Installing ${SRC} -> ${DST}"
if [[ "$(id -u)" -ne 0 ]]; then
  sudo cp "${SRC}" "${DST}"
  sudo chmod 644 "${DST}"
  sudo udevadm control --reload-rules
  sudo udevadm trigger
else
  cp "${SRC}" "${DST}"
  chmod 644 "${DST}"
  udevadm control --reload-rules
  udevadm trigger
fi

echo "OK: unplug/replug J_USB1 (or reboot) so /dev/jtsmm_hat appears."
