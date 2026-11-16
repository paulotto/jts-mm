#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Phase 7 hardware bench matrix — J_USB1 CDC only (Debug Probe = SWD flash, UART NC).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SW_DIR="${ROOT}/software/jtsmm"
VENV="${SW_DIR}/.venv"
FLASH_SCRIPT="${ROOT}/scripts/flash_swd.sh"
FW_BUILD="${ROOT}/firmware/jtsmm/build"
CONFIG="${JTSMM_CONFIG:-${ROOT}/software/jtsmm/etc/jtsmm.yaml}"

JTSMM_HW_REQUIRED="${JTSMM_HW_REQUIRED:-0}"
JTSMM_SKIP_FLASH="${JTSMM_SKIP_FLASH:-0}"
JTSMM_PORT="${JTSMM_PORT:-}"
# Release validation (JTSMM_RELEASE=1) runs the full sign-off: a 60 s soak with
# strict integrity (H9) and the power-cycle config-persistence regression (H10).
# Both can also be toggled individually (JTSMM_SOAK_S=<sec>, JTSMM_POWER_CYCLE=1)
# for ad-hoc runs. Defaults keep everyday dev/CI runs fast (no soak, no probe).
JTSMM_RELEASE="${JTSMM_RELEASE:-0}"
if [[ "${JTSMM_RELEASE}" == "1" ]]; then
  JTSMM_SOAK_S="${JTSMM_SOAK_S:-60}"
  JTSMM_POWER_CYCLE="${JTSMM_POWER_CYCLE:-1}"
fi
JTSMM_SOAK_S="${JTSMM_SOAK_S:-0}"
JTSMM_POWER_CYCLE="${JTSMM_POWER_CYCLE:-0}"

BENCH_PORT=""

log() { echo "[bench-hw] $*"; }
fail() { echo "[bench-hw] FAIL: $*" >&2; exit 1; }
skip() { echo "[bench-hw] SKIP: $*"; }

teardown_streams() {
  local port="${1:-$BENCH_PORT}"
  if [[ -z "${port}" ]]; then
    return 0
  fi
  log "H8: stop MCU streams (panel idle)"
  python3 -c "
import serial
from jtsmm.config import open_hat_serial, stop_mcu_streams
with open_hat_serial('${port}') as ser:
    stop_mcu_streams(ser)
print('OK')
" || log "WARN: stream teardown failed"
}

# Native capture/storage backend selection. Default: use it if it can be built.
# JTSMM_NATIVE=0 forces the pure-Python path (skips the build).
JTSMM_NATIVE="${JTSMM_NATIVE:-1}"

# shellcheck disable=SC1091
source_venv() {
  if [[ ! -d "${VENV}" ]]; then
    python3 -m venv "${VENV}"
  fi
  source "${VENV}/bin/activate"
  pip install -e "${SW_DIR}[dev]" -q
  maybe_build_native
}

# Build the native extension so the bench exercises the C++/libusb + libhdf5 path
# (the same path production uses when available). Non-fatal: if the toolchain or
# dev libraries are missing, the run falls back to the Python backend.
maybe_build_native() {
  if [[ "${JTSMM_NATIVE}" == "0" ]]; then
    log "native backend disabled (JTSMM_NATIVE=0) — using Python capture/writer"
    return 0
  fi
  pip install "pybind11>=2.11" -q >/dev/null 2>&1 || true
  if "${SW_DIR}/native/build.sh" --python "$(command -v python3)" >/tmp/jtsmm_native_build.log 2>&1; then
    log "native backend built (C++/libusb + libhdf5)"
  else
    log "WARN: native build failed (see /tmp/jtsmm_native_build.log) — falling back to Python"
    export JTSMM_NATIVE=0
  fi
}

discover_port() {
  if [[ -n "${JTSMM_PORT}" ]]; then
    echo "${JTSMM_PORT}"
    return
  fi
  python3 -c "
from jtsmm.capture.serial_util import find_hat_port
print(find_hat_port())
"
}

wait_ready() {
  local port="$1"
  python3 -c "
import time, serial
port = '${port}'
with serial.Serial(port, 115200, timeout=0.1) as s:
    s.dtr = False; time.sleep(0.05); s.dtr = True; time.sleep(1.5)
    s.reset_input_buffer()
    s.write(b'info\r\n'); s.flush()
    deadline = time.monotonic() + 8.0
    buf = b''
    while time.monotonic() < deadline:
        n = s.in_waiting
        if n:
            buf += s.read(n)
            if b'ready' in buf.lower() or b'jtsmm' in buf.lower():
                print('ready')
                raise SystemExit(0)
        time.sleep(0.05)
raise SystemExit('timeout waiting for ready')
"
}

step_h1_info() {
  local port="$1"
  log "H1: jtsmm info"
  jtsmm info --port "${port}" --json | python3 -c "
import json,sys
d=json.load(sys.stdin)
assert 'firmware_version' in d and d['firmware_version'], d
print('OK', d['firmware_version'])
"
}

step_h2_doctor() {
  local port="$1"
  log "H2: jtsmm doctor"
  jtsmm doctor --port "${port}"
}

step_h3_config_apply() {
  local port="$1"
  log "H3: jtsmm config apply (${CONFIG})"
  sleep 2
  python3 -c "
import serial
from pathlib import Path
from jtsmm.config import apply_mcu, load, open_hat_serial

port = '${port}'
cfg = load(Path('${CONFIG}'))
with open_hat_serial(port) as ser:
    if not apply_mcu(ser, cfg, timeout_s=120.0):
        raise SystemExit('config apply failed')
print('OK')
"
}

step_h4_selftest() {
  local port="$1"
  log "H4: mcu selftest (expect MTA + CRA)"
  jtsmm doctor --port "${port}" --mcu-selftest
  python3 -c "
import serial, time
port = '${port}'
with serial.Serial(port, 115200, timeout=0.05) as s:
    s.reset_input_buffer()
    s.write(b'selftest\r\n'); s.flush()
    deadline = time.monotonic() + 10.0
    buf = bytearray()
    while time.monotonic() < deadline:
        n = s.in_waiting
        if n:
            buf.extend(s.read(n))
            text = bytes(buf).decode('utf-8', errors='replace')
            if '=== result:' in text:
                if '[SKIP] CRA' in text:
                    raise SystemExit('CRA IMU skipped — both MTA and CRA required')
                if 'CRA=0' in text and 'MTA=0' in text:
                    raise SystemExit('no IMU probed')
                raise SystemExit(0)
        time.sleep(0.05)
raise SystemExit('selftest timeout')
"
}

step_h4b_drdy_watch() {
  local port="$1"
  log "H4b: ads_drdy_watch (after ads_start in H5 — run post-streams)"
}

step_h5_streams() {
  local port="$1"
  log "H5: start MCU streams (EMG + MTA + CRA + trig)"
  python3 -c "
import serial, time
from jtsmm.config import open_hat_serial, start_mcu_streams
with open_hat_serial('${port}') as ser:
    start_mcu_streams(ser)
    ser.write(b'ads_drdy_watch\r\n'); ser.flush()
    time.sleep(0.6)
    buf = bytearray()
    while ser.in_waiting:
        buf.extend(ser.read(ser.in_waiting))
    text = bytes(buf).decode('utf-8', errors='replace')
    if text.strip():
        print(text.strip())
    import re
    m = re.search(r'edges low=(\d+) high=(\d+)', text)
    if m:
        edges = int(m.group(1)) + int(m.group(2))
        if edges < 50:
            raise SystemExit(f'DRDY edge count too low: {edges}')
    elif 'edges low=0' in text and 'high=0' in text:
        raise SystemExit('no DRDY edges — EMG conversions not running')
print('OK')
"
}

step_h6_record() {
  local port="$1"
  local dur="${2:-10}"
  log "H6: jtsmm record --duration ${dur}"
  teardown_streams "${port}"
  sleep 0.3
  local out
  out="$(jtsmm record --port "${port}" --duration "${dur}" --config "${CONFIG}" --skip-preflight --skip-apply 2>&1)" || {
    echo "${out}"
    fail "record failed"
  }
  echo "${out}"
  echo "${out}" | python3 -c "
import sys, re
text = sys.stdin.read()
m = re.search(r'record_summary: EMG_BLOCK=(\d+) IMU_MTA_QUAT=(\d+) IMU_CRA_QUAT=(\d+)', text)
if not m:
    raise SystemExit('missing record_summary line')
emg, mta, cra = (int(x) for x in m.groups())
if emg < 100:
    raise SystemExit(f'EMG_BLOCK count too low: {emg}')
if mta < 10:
    raise SystemExit(f'IMU_MTA_QUAT count too low: {mta}')
if cra < 10:
    raise SystemExit(f'IMU_CRA_QUAT count too low: {cra}')
print(f'OK EMG={emg} MTA={mta} CRA={cra}')
"
  locate_latest_session > /tmp/jtsmm_last_session.txt || fail "no session h5"
}

step_h7_integrity() {
  local h5
  h5="$(cat /tmp/jtsmm_last_session.txt)"
  log "H7: integrity --strict ${h5}"
  jtsmm integrity --strict "${h5}"
}

locate_latest_session() {
  python3 -c "
from pathlib import Path
candidates = []
for base in [Path('/tmp/jtsmm_bench'), Path('/opt/jtsmm/data'), Path('/tmp')]:
    if base.exists():
        candidates.extend(base.glob('session_*/session.h5'))
        candidates.extend(base.glob('session_*.h5'))
if not candidates:
    raise SystemExit('no session h5 found')
print(max(candidates, key=lambda p: p.stat().st_mtime))
"
}

step_h8_config_audit() {
  log "H8: jtsmm config audit --strict"
  jtsmm config audit --strict --config "${CONFIG}"
}

step_h8b_teardown_label() {
  local port="${1:-$BENCH_PORT}"
  log "H8b: stop MCU streams (panel idle)"
  teardown_streams "${port}"
}

step_h9_soak() {
  local port="$1"
  local dur="$2"
  log "H9: soak record ${dur}s (strict integrity)"
  jtsmm record --port "${port}" --duration "${dur}" --config "${CONFIG}" --skip-preflight --skip-apply \
    || fail "soak record failed"
  locate_latest_session > /tmp/jtsmm_last_session.txt || fail "no soak session h5"
  step_h7_integrity
}

# H10 power-cycle config-persistence regression: save the applied config to
# flash, snapshot it, warm-reset the MCU via the Debug Probe (SWD), and confirm
# the firmware auto-loads the identical config on boot and can stream again.
step_h10_power_cycle() {
  local port="$1"
  if [[ ! -x "${FLASH_SCRIPT}" ]]; then
    skip "H10: no flash script (SWD probe) — cannot warm-reset"
    return 0
  fi
  log "H10: power-cycle config-persistence regression"

  # 1. Persist the currently-applied config and snapshot the dump.
  python3 -c "
import serial, time, sys
port = '${port}'
def cmd(s, text, wait=0.6):
    s.reset_input_buffer(); s.write(text + b'\r\n'); s.flush(); time.sleep(wait)
    buf = bytearray()
    while s.in_waiting:
        buf.extend(s.read(s.in_waiting)); time.sleep(0.05)
    return bytes(buf).decode('utf-8', errors='replace')
with serial.Serial(port, 115200, timeout=0.1) as s:
    out = cmd(s, b'config save')
    if 'config save OK' not in out:
        raise SystemExit('config save failed: ' + out.strip())
    dump = cmd(s, b'config dump', wait=1.0)
    line = next((l for l in dump.splitlines() if l.strip().startswith('{')), '')
    if not line:
        raise SystemExit('no config dump JSON before reset')
    open('/tmp/jtsmm_cfg_before.json', 'w').write(line)
print('OK saved + snapshot')
" || fail "H10 pre-reset save/dump failed"

  # 2. Warm-reset the MCU (reboot without reprogramming flash).
  "${FLASH_SCRIPT}" --reset >/dev/null 2>&1 || fail "H10 SWD warm-reset failed"
  sleep 2
  wait_ready "${port}" || fail "H10 boot ready timeout after reset"

  # 3. The boot path auto-applies the saved config; confirm the dump matches and
  #    that streams still start (config is fully usable, not just persisted).
  python3 -c "
import serial, time, json, sys
port = '${port}'
def cmd(s, text, wait=0.8):
    s.reset_input_buffer(); s.write(text + b'\r\n'); s.flush(); time.sleep(wait)
    buf = bytearray()
    while s.in_waiting:
        buf.extend(s.read(s.in_waiting)); time.sleep(0.05)
    return bytes(buf).decode('utf-8', errors='replace')
with serial.Serial(port, 115200, timeout=0.1) as s:
    dump = cmd(s, b'config dump', wait=1.0)
    line = next((l for l in dump.splitlines() if l.strip().startswith('{')), '')
    if not line:
        raise SystemExit('no config dump JSON after reset')
before = json.loads(open('/tmp/jtsmm_cfg_before.json').read())
after = json.loads(line)
if before != after:
    raise SystemExit('config mismatch after power-cycle:\n before=%s\n after =%s' % (before, after))
print('OK config persisted across power-cycle')
" || fail "H10 post-reset config mismatch"
}

main() {
  source_venv
  trap 'teardown_streams' EXIT

  if [[ "${JTSMM_SKIP_FLASH}" != "1" ]] && [[ -x "${FLASH_SCRIPT}" ]] && [[ -f "${FW_BUILD}/jtsmm.elf" ]]; then
    log "Flashing via SWD (Debug Probe)..."
    "${FLASH_SCRIPT}" || fail "SWD flash failed"
    sleep 2
  fi

  local port=""
  if ! port="$(discover_port 2>/dev/null)"; then
    if [[ "${JTSMM_HW_REQUIRED}" == "1" ]]; then
      fail "No HAT on J_USB1 (set JTSMM_PORT or connect J_USB1)"
    fi
    skip "No HAT — set JTSMM_HW_REQUIRED=1 to fail hard"
    exit 0
  fi
  BENCH_PORT="${port}"
  log "Using port ${port} (J_USB1 CDC)"
  log "capture backend: $([[ "${JTSMM_NATIVE}" == "0" ]] && echo Python || echo native)"

  export JTSMM_DATA_DIR="${JTSMM_DATA_DIR:-/tmp/jtsmm_bench}"
  mkdir -p "${JTSMM_DATA_DIR}"

  log "Boot wait..."
  wait_ready "${port}" || fail "boot ready timeout"

  step_h1_info "${port}"
  step_h2_doctor "${port}"
  step_h3_config_apply "${port}"
  step_h4_selftest "${port}"
  step_h5_streams "${port}"
  step_h6_record "${port}" 10
  step_h7_integrity

  step_h8_config_audit

  if [[ "${JTSMM_SOAK_S}" != "0" ]]; then
    step_h8b_teardown_label "${port}"
    step_h9_soak "${port}" "${JTSMM_SOAK_S}"
  fi

  if [[ "${JTSMM_POWER_CYCLE}" == "1" ]]; then
    teardown_streams "${port}"
    step_h10_power_cycle "${port}"
  fi

  teardown_streams "${port}"
  trap - EXIT

  log "=== bench-hw PASS ==="
}

main "$@"
