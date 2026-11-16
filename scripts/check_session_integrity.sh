#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Validate one or more recorded jtsmm session HDF5 files without having to
# write a one-off python snippet each time.
#
# Runs the same checker as `jtsmm record`'s post-session pass and the
# `jtsmm integrity check` subcommand (jtsmm.integrity.checker.check_session) and
# prints the nicely formatted summary table for each file.
#
# Usage:
#   ./scripts/check_session_integrity.sh [OPTIONS] [SESSION.h5 | SESSION_DIR ...]
#
#   With no path given, checks ./data/session.h5 if it exists.
#   A directory argument is expanded to every *.h5 file directly inside it.
#
# Options (forwarded to `jtsmm integrity check`):
#   --config PATH   Verify config_hash against this YAML in addition to the
#                    checks baked into the recording.
#   --strict        Treat warnings as failures (nonzero exit).
#   --plain         Flat warning:/error: lines instead of the summary table.
#   --write-summary Write integrity.json + integrity.txt next to the session.
#
# Exit code is nonzero if any checked file fails integrity (or has warnings
# under --strict).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

# Resolve a runnable `jtsmm` without requiring it to be installed/activated:
# prefer PATH, then the software/jtsmm dev venv, then a bare `python -m`.
jtsmm_cmd=()
if command -v jtsmm >/dev/null 2>&1; then
  jtsmm_cmd=(jtsmm)
elif [[ -x "${ROOT}/software/jtsmm/.venv/bin/jtsmm" ]]; then
  jtsmm_cmd=("${ROOT}/software/jtsmm/.venv/bin/jtsmm")
elif [[ -x "${JTSMM_VENV:-}/bin/jtsmm" ]]; then
  jtsmm_cmd=("${JTSMM_VENV}/bin/jtsmm")
else
  py="$(command -v python3 || true)"
  [[ -x "${ROOT}/software/jtsmm/.venv/bin/python" ]] && py="${ROOT}/software/jtsmm/.venv/bin/python"
  [[ -n "${py}" ]] || { echo "ERROR: no jtsmm CLI and no python3 found" >&2; exit 1; }
  jtsmm_cmd=("${py}" -m jtsmm.cli.main)
fi

forward_opts=()
paths=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --config) forward_opts+=("--config" "$2"); shift 2 ;;
    --strict|--strict-warn|--plain|--write-summary|--no-write-summary)
      forward_opts+=("$1"); shift ;;
    *) paths+=("$1"); shift ;;
  esac
done

if [[ ${#paths[@]} -eq 0 ]]; then
  default="${ROOT}/data/session.h5"
  if [[ -f "${default}" ]]; then
    paths=("${default}")
  else
    echo "ERROR: no session path given and ${default} does not exist" >&2
    usage >&2
    exit 1
  fi
fi

# Expand directory arguments to the *.h5 files directly inside them.
files=()
for p in "${paths[@]}"; do
  if [[ -d "$p" ]]; then
    shopt -s nullglob
    found=("$p"/*.h5)
    shopt -u nullglob
    if [[ ${#found[@]} -eq 0 ]]; then
      echo "WARNING: no *.h5 files in directory $p" >&2
      continue
    fi
    files+=("${found[@]}")
  else
    files+=("$p")
  fi
done

if [[ ${#files[@]} -eq 0 ]]; then
  echo "ERROR: nothing to check" >&2
  exit 1
fi

status=0
for f in "${files[@]}"; do
  if ! "${jtsmm_cmd[@]}" integrity check "${forward_opts[@]}" "$f"; then
    status=1
  fi
  echo
done

exit "${status}"
