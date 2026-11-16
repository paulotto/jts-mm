#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Install JTS-MM host stack on Raspberry Pi 5 (user-home layout).
# Delegates to scripts/build_install.sh (see --help there).
#
# Editable on purpose: the rig runs from this checkout, so pulling new code
# needs no reinstall and software/optical keeps its etc/ next to the code. A
# regular install also works -- the build bundles schema/ and etc/jtsmm.yaml.
#
# --analysis (default here) also installs the offline analysis stack the
# experiments/ scripts need: jtsmm[analysis], software/jaw_tracking and
# software/biomech[plot,stats], all under a numpy==<apt version> constraint so
# apt's numpy (picamera2/simplejpeg) survives. Pass --no-analysis to skip it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TARGET_USER="${SUDO_USER:-${USER:-pi}}"
USER_HOME="$(getent passwd "${TARGET_USER}" 2>/dev/null | cut -d: -f6 || echo "/home/${TARGET_USER}")"

WIFI_EXTRA=()
if [[ -f "${USER_HOME}/.jtsmm/wifi.env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${USER_HOME}/.jtsmm/wifi.env"
  set +a
elif [[ -f "${USER_HOME}/.config/jtsmm/wifi.env" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${USER_HOME}/.config/jtsmm/wifi.env"
  set +a
fi
if [[ -n "${JTSMM_WIFI_SSID:-}" ]]; then
  WIFI_EXTRA+=(--wifi-ssid "${JTSMM_WIFI_SSID}")
fi
if [[ -n "${JTSMM_WIFI_PSK:-}" ]]; then
  WIFI_EXTRA+=(--wifi-psk "${JTSMM_WIFI_PSK}")
fi
if [[ -n "${JTSMM_WIFI_BAND:-}" ]]; then
  WIFI_EXTRA+=(--wifi-band "${JTSMM_WIFI_BAND}")
fi
if [[ -n "${JTSMM_WIFI_CHANNEL:-}" ]]; then
  WIFI_EXTRA+=(--wifi-channel "${JTSMM_WIFI_CHANNEL}")
fi

exec "${ROOT}/scripts/build_install.sh" \
  --user "${TARGET_USER}" \
  --prefix "${USER_HOME}/jtsmm" \
  --venv "${USER_HOME}/jtsmm/venv" \
  --data-dir "${USER_HOME}/jtsmm_data" \
  --config-path "${USER_HOME}/jtsmm.yaml" \
  --config \
  --editable \
  --udev \
  --pi-tune \
  --pi-network \
  --hostname jtsmm \
  --viewer \
  --cameras \
  --analysis \
  --skip-tests \
  "${WIFI_EXTRA[@]}" \
  "$@"
