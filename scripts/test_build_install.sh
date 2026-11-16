#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Integration test: run build_install.sh in an isolated temp home (like install_pi).
# Cleans up automatically on exit.
#
# Usage:
#   ./scripts/test_build_install.sh
#   JTSMM_INSTALL_TEST_USE_APT=1 ./scripts/test_build_install.sh   # run apt-get first (needs sudo)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_HOME="$(mktemp -d "${TMPDIR:-/tmp}/jtsmm-install-test.XXXXXX")"
trap 'rm -rf "${TEST_HOME}"' EXIT

echo "==> isolated test home: ${TEST_HOME}"

export HOME="${TEST_HOME}"
export USER="${USER:-$(id -un)}"
unset SUDO_USER

PREFIX="${TEST_HOME}/jtsmm"
DATA_DIR="${TEST_HOME}/jtsmm_data"
CONFIG_PATH="${TEST_HOME}/jtsmm.yaml"

apt_args=(--no-apt)
use_apt="${JTSMM_INSTALL_TEST_USE_APT:-auto}"
if [[ "${use_apt}" == "1" ]]; then
  apt_args=()
elif [[ "${use_apt}" == "auto" ]] && sudo -n true 2>/dev/null; then
  apt_args=()
fi

if [[ ${#apt_args[@]} -gt 0 ]]; then
  missing=()
  for pkg in libcap-dev libhdf5-dev libusb-1.0-0-dev; do
    if ! dpkg -s "${pkg}" >/dev/null 2>&1; then
      missing+=("${pkg}")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "ERROR: missing system packages for full install test: ${missing[*]}" >&2
    echo "Run: JTSMM_INSTALL_TEST_USE_APT=1 $0" >&2
    echo "  or: sudo apt install -y ${missing[*]}" >&2
    exit 1
  fi
fi

# Reproduce Pi installs that are launched from the repo root checkout.
cd "${ROOT}"

echo "==> build_install (cameras + native + config, cwd=${ROOT})"
"${ROOT}/scripts/build_install.sh" \
  "${apt_args[@]}" \
  --prefix "${PREFIX}" \
  --venv "${PREFIX}/venv" \
  --data-dir "${DATA_DIR}" \
  --config-path "${CONFIG_PATH}" \
  --config \
  --cameras \
  --skip-tests \
  --user "${USER}"

test -x "${PREFIX}/venv/bin/python"
test -x "${PREFIX}/venv/bin/jtsmm"
test -f "${CONFIG_PATH}"
grep -Fq "${DATA_DIR}" "${CONFIG_PATH}"

echo "==> build_install (--config-path directory)"
DIR_CONFIG_HOME="$(mktemp -d "${TMPDIR:-/tmp}/jtsmm-install-dir-config.XXXXXX")"
DIR_PREFIX="${DIR_CONFIG_HOME}/jtsmm"
"${ROOT}/scripts/build_install.sh" \
  "${apt_args[@]}" \
  --prefix "${DIR_PREFIX}" \
  --venv "${DIR_PREFIX}/venv" \
  --data-dir "${DIR_CONFIG_HOME}/data" \
  --config-path "${DIR_PREFIX}" \
  --config \
  --skip-tests \
  --user "${USER}"
test -f "${DIR_PREFIX}/jtsmm.yaml"
grep -Fq "${DIR_CONFIG_HOME}/data" "${DIR_PREFIX}/jtsmm.yaml"
rm -rf "${DIR_CONFIG_HOME}"

PY="${PREFIX}/venv/bin/python"
"${PY}" -c "import jtsmm; print('jtsmm', jtsmm.__version__)"
"${PY}" -c "import picamera2; print('picamera2', picamera2.__version__)"
"${PY}" -c "from jtsmm import native; assert native.is_available(), 'native extension missing'"

echo "OK: full install test passed (temp ${TEST_HOME})"
