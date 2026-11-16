#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Generate compile_commands.json for clangd in each C/C++ CMake subproject.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PICO_SDK_PATH="${PICO_SDK_PATH:-$HOME/pico/pico-sdk}"

cmake_configure() {
  local dir="$1"
  shift
  echo "==> $dir"
  cmake -S "$dir" -B "$dir/build" "$@"
}

cmake_configure "$ROOT/firmware/jtsmm"
cmake_configure "$ROOT/firmware/jtsmm/tests"
cmake_configure "$ROOT/software/jtsmm/native"

echo "Done. Reload clangd: Ctrl+Shift+P -> 'clangd: Restart language server'"
