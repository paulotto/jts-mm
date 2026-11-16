#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Flash production jtsmm.elf over SWD (Raspberry Pi Debug Probe).
# Requires Raspberry Pi OpenOCD (system apt openocd is too old for RP2350).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FW_DIR="${REPO_ROOT}/firmware/jtsmm"
BUILD_DIR="${FW_DIR}/build"
ELF="${BUILD_DIR}/jtsmm.elf"
RESCUE=0
RESET_ONLY=0
SPEEDS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Flash production firmware (firmware/jtsmm/build/jtsmm.elf) to the RP2350 on the jtsmm HAT.

Options:
  --rescue     Use rp2350-rescue.cfg (chip stuck / bad flash state)
  --reset      Warm-reset the target (reboot, no reprogram) and exit — used by
               the power-cycle config regression to exercise boot-time flash load
  --speed N    SWD clock in kHz (default: try 1000, 500, 100, 1000)
  -h, --help   Show this help

Environment:
  OPENOCD          Path to openocd binary
  OPENOCD_SCRIPTS  Path to scripts/ directory
  SWD_SPEED        Single speed in kHz (same as --speed)

Before flashing:
  1. HAT must have 5 V (Pi on + HAT stacked, or bench 5 V on header pins 2/4)
  2. USB-C on the HAT does NOT power the board — only Pi 5 V does
  3. Debug Probe USB → PC; 3-wire cable → J_SWD1 (pin1=GND, pin2=SWCLK, pin3=SWDIO)
  4. If connection fails, try flipping the JST plug 180° or update Debug Probe firmware

Build first (if missing):
  export PICO_SDK_PATH=~/pico/pico-sdk
  cd ${FW_DIR} && rm -rf build && mkdir build && cd build
  cmake .. -DPICO_PLATFORM=rp2350 && cmake --build . -j\$(nproc)

OpenOCD install (once):
  ${REPO_ROOT}/scripts/install_openocd.sh

UF2 alternative:
  Hold BOOTSEL, plug HAT USB-C, copy ${BUILD_DIR}/jtsmm.uf2 to the RP2350 drive.

See also: docs/hardware/h08_rp2350_flashing.md
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rescue) RESCUE=1; shift ;;
    --reset) RESET_ONLY=1; shift ;;
    --speed) SPEEDS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "${SWD_SPEED:-}" ]]; then
  SPEEDS=("${SWD_SPEED}")
fi
if [[ ${#SPEEDS[@]} -eq 0 ]]; then
  SPEEDS=(1000 500 100 1000)
fi

if [[ "${RESET_ONLY}" -eq 0 && ! -f "${ELF}" ]]; then
  echo "Missing ${ELF} — build first:" >&2
  echo "  export PICO_SDK_PATH=~/pico/pico-sdk" >&2
  echo "  cd ${FW_DIR} && rm -rf build && mkdir build && cd build" >&2
  echo "  cmake .. -DPICO_PLATFORM=rp2350 && cmake --build . -j\$(nproc)" >&2
  exit 1
fi

resolve_openocd() {
  if [[ -n "${OPENOCD:-}" && -x "${OPENOCD}" ]]; then
    echo "${OPENOCD}"
    return
  fi
  if [[ -x "${HOME}/pico/openocd/openocd" ]]; then
    echo "${HOME}/pico/openocd/openocd"
    return
  fi
  if command -v openocd >/dev/null 2>&1; then
    if openocd -s /usr/share/openocd/scripts -f target/rp2350.cfg -c exit 2>/dev/null; then
      echo "openocd"
      return
    fi
  fi
  echo ""
}

OPENOCD_BIN="$(resolve_openocd)"
if [[ -z "${OPENOCD_BIN}" ]]; then
  echo "No RP2350-capable OpenOCD found. Run:" >&2
  echo "  ${REPO_ROOT}/scripts/install_openocd.sh" >&2
  exit 1
fi

if [[ -d "${OPENOCD_SCRIPTS:-}" ]]; then
  SCRIPTS="${OPENOCD_SCRIPTS}"
elif [[ -d "${HOME}/pico/openocd/scripts" ]]; then
  SCRIPTS="${HOME}/pico/openocd/scripts"
elif [[ "${OPENOCD_BIN}" == "openocd" ]]; then
  SCRIPTS="/usr/share/openocd/scripts"
else
  SCRIPTS="$(dirname "${OPENOCD_BIN}")/scripts"
fi

TARGET="target/rp2350.cfg"
if [[ "${RESCUE}" -eq 1 ]]; then
  TARGET="target/rp2350-rescue.cfg"
fi

if [[ "${RESET_ONLY}" -eq 1 ]]; then
  echo "Warm-resetting target (no reprogram)..."
  "${OPENOCD_BIN}" -s "${SCRIPTS}" \
    -f interface/cmsis-dap.cfg \
    -f "${TARGET}" \
    -c "adapter speed 1000" \
    -c "init; reset run; exit"
  echo "Reset OK"
  exit 0
fi

echo "Using OpenOCD: ${OPENOCD_BIN}"
echo "Scripts:       ${SCRIPTS}"
echo "Target:        ${TARGET}"
echo "Flashing:      ${ELF}"
echo ""
echo "Check: HAT has 5 V (Pi powered + HAT stacked)? J_SWD1 pin1=GND connected?"
echo ""

flash_at_speed() {
  local speed="$1"
  echo "=== Trying adapter speed ${speed} kHz ==="
  "${OPENOCD_BIN}" -s "${SCRIPTS}" \
    -f interface/cmsis-dap.cfg \
    -f "${TARGET}" \
    -c "adapter speed ${speed}" \
    -c "program ${ELF} verify reset exit"
}

last_err=1
for speed in "${SPEEDS[@]}"; do
  if flash_at_speed "${speed}"; then
    echo "Flash OK at ${speed} kHz"
    exit 0
  fi
  last_err=$?
  echo "Failed at ${speed} kHz" >&2
done

cat >&2 <<EOF

All speeds failed (last exit ${last_err}).

Common fixes for "cannot read IDR":
  1. Power: apply 5 V via Pi 40-pin header (pins 2/4). HAT USB-C does NOT power the MCU.
  2. Ground: Debug Probe GND must reach J_SWD1 pin 1 (DGND).
  3. Cable: pin2=SWCLK, pin3=SWDIO — try rotating the JST plug 180°.
  4. Speed: export SWD_SPEED=100 and re-run, or use --speed 100
  5. Reset: hold SW_RESET1 on the HAT, run ${SCRIPT_DIR}/flash_swd.sh, release after "clock speed"
  6. Probe FW: update Debug Probe — https://github.com/raspberrypi/debugprobe/releases
  7. Rescue: ${SCRIPT_DIR}/flash_swd.sh --rescue --speed 100
  8. Skip SWD: hold BOOTSEL, plug HAT USB-C, copy ${BUILD_DIR}/jtsmm.uf2 to RP2350 drive

EOF
exit "${last_err}"
