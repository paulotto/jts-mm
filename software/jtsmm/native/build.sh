#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Build the native capture/storage extension and drop it into jtsmm/native/.
#
# Requires: a C++17 compiler, cmake, ninja (or make), pybind11 (pip), and the
# system dev packages libusb-1.0-0-dev and libhdf5-dev.
#
# Usage: software/jtsmm/native/build.sh [--python /path/to/python]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${HERE}/build"

PYTHON="python3"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python) PYTHON="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# pybind11 ships its CMake config with the pip package; point CMake at it.
PYBIND11_DIR="$("${PYTHON}" -m pybind11 --cmakedir)"

cmake -S "${HERE}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v "${PYTHON}")" \
  -Dpybind11_DIR="${PYBIND11_DIR}"

cmake --build "${BUILD_DIR}"

echo "Built native extension into $(cd "${HERE}/../jtsmm/native" && pwd)"
