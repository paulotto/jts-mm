#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Enable/disable Pi Wi-Fi for timed RF-quiet windows (persists across reboot).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -f "${SCRIPT_DIR}/pi_wifi_lib.sh" ]]; then
  echo "ERROR: missing ${SCRIPT_DIR}/pi_wifi_lib.sh — run 'git pull' in the repo." >&2
  exit 1
fi
# shellcheck source=pi_wifi_lib.sh
source "${SCRIPT_DIR}/pi_wifi_lib.sh"

STATE_FILE="/var/lib/jtsmm/wifi_disabled_until"
INDEFINITE_UNTIL=9223372036854775807

usage() {
  cat <<EOF
Usage: $(basename "$0") status|enable|disable [--duration DURATION]

  status              Show radio state and disable-until timestamp
  enable              Turn Wi-Fi on immediately; clear persisted disable
  disable             Turn Wi-Fi off (--duration optional: 30m, 2h, 1d; omit = indefinite)

Examples:
  pi_wifi disable --duration 2h
  pi_wifi enable
EOF
}

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

parse_duration() {
  local raw="$1"
  if [[ ! "${raw}" =~ ^[0-9]+[mhd]$ ]]; then
    echo "ERROR: invalid duration '${raw}' (use Xm, Xh, or Xd)" >&2
    exit 1
  fi
  local n="${raw%?}"
  local unit="${raw: -1}"
  case "${unit}" in
    m) echo $((n * 60)) ;;
    h) echo $((n * 3600)) ;;
    d) echo $((n * 86400)) ;;
  esac
}

wifi_radio_status() {
  nmcli -t -f WIFI radio 2>/dev/null || echo "unknown"
}

pi_wifi_bring_up_active() {
  local conn alt
  conn="$(tr -d '[:space:]' < "${WIFI_ACTIVE_PROFILE_FILE}" 2>/dev/null || true)"
  if [[ -z "${conn}" ]]; then
    conn="$(pi_wifi_active_conn_name)"
  fi
  if [[ -n "${conn}" ]] && nmcli connection show "${conn}" >/dev/null 2>&1; then
    run_root nmcli connection up "${conn}" 2>/dev/null && return 0
  fi
  alt="$(pi_wifi_alt_conn_name "${conn}")"
  if [[ -n "${alt}" ]] && nmcli connection show "${alt}" >/dev/null 2>&1; then
    run_root nmcli connection up "${alt}" 2>/dev/null || true
  fi
}

cmd_status() {
  local radio until human="" active
  radio="$(wifi_radio_status)"
  active="$(pi_wifi_active_conn_name)"
  echo "wifi radio: ${radio}"
  echo "active profile: ${active:-none}"
  if [[ -f "${STATE_FILE}" ]]; then
    until="$(cat "${STATE_FILE}" | awk '{print $1}')"
    if [[ "${until}" == "${INDEFINITE_UNTIL}" ]]; then
      human="indefinite (manual enable required)"
    elif [[ "${until}" -gt 0 ]]; then
      human="until epoch ${until}"
    fi
    echo "disabled_until: ${until} ${human}"
  else
    echo "disabled_until: (none)"
  fi
}

cmd_enable() {
  run_root mkdir -p /var/lib/jtsmm
  run_root rm -f "${STATE_FILE}"
  run_root nmcli radio wifi on
  pi_wifi_bring_up_active
  echo "Wi-Fi enabled"
}

cmd_disable() {
  local duration_s="${1:-0}"
  local until=0
  run_root mkdir -p /var/lib/jtsmm
  if [[ "${duration_s}" -eq 0 ]]; then
    until="${INDEFINITE_UNTIL}"
  else
  until=$(( $(date +%s) + duration_s ))
  fi
  echo "${until}" | run_root tee "${STATE_FILE}" >/dev/null
  run_root nmcli radio wifi off
  if command -v systemctl >/dev/null 2>&1; then
    run_root systemctl start jtsmm-wifi-guard.service 2>/dev/null || true
  fi
  echo "Wi-Fi disabled (until epoch ${until})"
}

# Called by jtsmm-wifi-guard.service on boot / timer.
guard_apply() {
  if [[ ! -f "${STATE_FILE}" ]]; then
    nmcli radio wifi on 2>/dev/null || true
    pi_wifi_bring_up_active
    return 0
  fi
  local until now
  until="$(awk '{print $1}' "${STATE_FILE}")"
  now="$(date +%s)"
  if [[ -z "${until}" || "${until}" -eq 0 ]]; then
    rm -f "${STATE_FILE}"
    nmcli radio wifi on 2>/dev/null || true
    return 0
  fi
  if [[ "${until}" -gt "${now}" ]]; then
    nmcli radio wifi off 2>/dev/null || true
    return 0
  fi
  rm -f "${STATE_FILE}"
  nmcli radio wifi on 2>/dev/null || true
  pi_wifi_bring_up_active
}

if [[ "${1:-}" == "--guard" ]]; then
  guard_apply
  exit 0
fi

CMD="${1:-}"
shift || true
DURATION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --duration) DURATION="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

case "${CMD}" in
  status) cmd_status ;;
  enable) cmd_enable ;;
  disable)
    secs=0
    if [[ -n "${DURATION}" ]]; then
      secs="$(parse_duration "${DURATION}")"
    fi
    cmd_disable "${secs}"
    ;;
  "") usage; exit 1 ;;
  *) echo "Unknown command: ${CMD}" >&2; usage; exit 1 ;;
esac
