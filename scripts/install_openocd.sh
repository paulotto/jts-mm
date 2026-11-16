#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Install Raspberry Pi OpenOCD (RP2350 support) to ~/pico/openocd/
set -euo pipefail

INSTALL_DIR="${HOME}/pico/openocd"
RELEASE="v2.2.0-4"

case "$(uname -m)" in
  x86_64|amd64)
    ARCH_SUFFIX="x86_64-lin"
    ;;
  aarch64|arm64)
    ARCH_SUFFIX="aarch64-lin"
    ;;
  *)
    echo "ERROR: unsupported CPU architecture: $(uname -m) (need x86_64 or aarch64)" >&2
    exit 1
    ;;
esac

TARBALL="openocd-0.12.0+dev-${ARCH_SUFFIX}.tar.gz"
URL="https://github.com/raspberrypi/pico-sdk-tools/releases/download/${RELEASE}/${TARBALL}"

install_bundled_libs() {
  local src_dir="$1"
  local dst_dir="$2"
  local lib
  shopt -s nullglob
  for lib in "${src_dir}"/lib*.so*; do
    install -m 755 "${lib}" "${dst_dir}/$(basename "${lib}")"
    echo "  bundled $(basename "${lib}")"
  done
  shopt -u nullglob
}

verify_openocd() {
  local bin="$1"
  if "${bin}" --version >/dev/null 2>&1; then
    "${bin}" --version
    return 0
  fi
  local missing
  missing="$(ldd "${bin}" 2>/dev/null | grep 'not found' || true)"
  echo "ERROR: ${bin} failed to start." >&2
  if [[ -n "${missing}" ]]; then
    echo "Missing shared libraries:" >&2
    echo "${missing}" >&2
    echo "Install system packages, then re-run this script:" >&2
    echo "  sudo apt install -y libgpiod2 libhidapi-hidraw0 libftdi1-2 libusb-1.0-0" >&2
  fi
  return 1
}

mkdir -p "${INSTALL_DIR}"
tmpdir="$(mktemp -d)"
trap 'rm -rf "${tmpdir}"' EXIT

echo "Downloading ${URL} ..."
curl -fsSL -o "${tmpdir}/${TARBALL}" "${URL}"
tar xzf "${tmpdir}/${TARBALL}" -C "${tmpdir}"

install -m 755 "${tmpdir}/openocd" "${INSTALL_DIR}/openocd"
rm -rf "${INSTALL_DIR}/scripts"
cp -a "${tmpdir}/scripts" "${INSTALL_DIR}/scripts"

echo "Installing bundled libraries (RUNPATH=\$ORIGIN):"
install_bundled_libs "${tmpdir}" "${INSTALL_DIR}"

echo "Installed to ${INSTALL_DIR} ($(uname -m))"
verify_openocd "${INSTALL_DIR}/openocd"
echo "Flash with: scripts/flash_swd.sh"
