#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Hardware-in-the-loop bench — Phase 6 build smoke + automated HW matrix.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FW_PROD="${ROOT}/firmware/jtsmm"
FW_BUILD="${FW_PROD}/build"
SW_DIR="${ROOT}/software/jtsmm"
VENV="${SW_DIR}/.venv"
HW_SCRIPT="${ROOT}/scripts/hil/run_bench_hw.sh"

CMAKE_FLAGS=(-G Ninja)
if [[ -n "${PICO_SDK_PATH:-}" ]]; then
  CMAKE_FLAGS+=(-DPICO_SDK_PATH="${PICO_SDK_PATH}")
else
  CMAKE_FLAGS+=(-DPICO_SDK_FETCH_FROM_GIT=ON -DPICO_SDK_FETCH_FROM_GIT_TAG="${PICO_SDK_FETCH_FROM_GIT_TAG:-2.1.1}")
fi

echo "=== jtsmm HIL bench (Phase 6) ==="

echo "--- Production firmware (jtsmm.elf) ---"
cmake -S "${FW_PROD}" -B "${FW_BUILD}" "${CMAKE_FLAGS[@]}"
cmake --build "${FW_BUILD}" --target jtsmm
echo "OK: ${FW_BUILD}/jtsmm.elf"

echo "--- Host package + pytest ---"
if [[ ! -d "${VENV}" ]]; then
  python3 -m venv "${VENV}"
fi
# shellcheck disable=SC1091
source "${VENV}/bin/activate"
pip install -e "${SW_DIR}[dev]" -q
pytest -q "${SW_DIR}/tests"

echo "--- Firmware host unit tests ---"
if [[ -f "${FW_PROD}/tests/CMakeLists.txt" ]]; then
  cmake -S "${FW_PROD}/tests" -B "${FW_PROD}/tests/build"
  cmake --build "${FW_PROD}/tests/build"
  ctest --test-dir "${FW_PROD}/tests/build" --output-on-failure
fi

if [[ -x "${HW_SCRIPT}" ]]; then
  echo "--- Hardware matrix (J_USB1 CDC; SWD flash unless JTSMM_SKIP_FLASH=1) ---"
  "${HW_SCRIPT}" || {
    echo "WARN: hardware matrix failed or no HAT attached"
    echo "Set JTSMM_HW_REQUIRED=1 to fail hard; JTSMM_PORT=/dev/ttyACM0 to pin port"
  }
fi

echo "=== bench complete ==="
