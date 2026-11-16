#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Host-side smoke test for pi_network Wi-Fi profile logic (mock nmcli, no system changes).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOCK_DIR="$(mktemp -d)"
trap 'rm -rf "${MOCK_DIR}"' EXIT

CONN_DB="${MOCK_DIR}/connections.txt"
SCAN_FILE="${MOCK_DIR}/scan.txt"
MODIFY_LOG="${MOCK_DIR}/modify.log"
touch "${CONN_DB}"
touch "${MODIFY_LOG}"
printf '%s\n' 'AA\:BB\:CC\:DD\:EE\:FF:LabNet:36' > "${SCAN_FILE}"

mock_nmcli() {
  case "$*" in
    "general status")
      echo "running"
      ;;
    "-t -f DEVICE,TYPE device status")
      printf '%s\n' "wlan0:wifi"
      ;;
    "device wifi rescan ifname wlan0")
      return 0
      ;;
    "-t -f BSSID,SSID,CHAN device wifi list ifname wlan0")
      cat "${SCAN_FILE}"
      ;;
    "connection show jtsmm-wifi-LabNet-5"|"connection show jtsmm-wifi-LabNet-24"|"connection show jtsmm-wifi-SIM-5"|"connection show jtsmm-wifi-SIM-24")
      if grep -qx "${3:-}" "${CONN_DB}" 2>/dev/null; then
        echo "802-11-wireless.ssid:LabNet"
        return 0
      fi
      return 1
      ;;
    connection\ add\ *)
      if [[ "$*" == *"con-name jtsmm-wifi-LabNet-5"* ]]; then
        echo "jtsmm-wifi-LabNet-5" >> "${CONN_DB}"
      elif [[ "$*" == *"con-name jtsmm-wifi-LabNet-24"* ]]; then
        echo "jtsmm-wifi-LabNet-24" >> "${CONN_DB}"
      fi
      ;;
    connection\ modify\ *)
      printf '%s\n' "$*" >> "${MODIFY_LOG}"
      return 0
      ;;
    connection\ up\ *)
      return 0
      ;;
    "-t -f NAME connection show")
      cat "${CONN_DB}"
      ;;
    *)
      echo "mock_nmcli: unhandled: $*" >&2
      return 1
      ;;
  esac
}

export -f mock_nmcli
export SCAN_FILE CONN_DB MODIFY_LOG
cat > "${MOCK_DIR}/nmcli" <<'EOF'
#!/usr/bin/env bash
mock_nmcli "$@"
EOF
chmod +x "${MOCK_DIR}/nmcli"

export PATH="${MOCK_DIR}:${PATH}"
export JTSMM_WIFI_SSID= JTSMM_WIFI_PSK=

out="$(
  "${ROOT}/scripts/pi_network.sh" \
    --dry-run \
    --wifi-ssid 'LabNet' \
    --wifi-psk 'secret' \
    2>&1
)"
echo "${out}" | grep -q 'jtsmm-wifi-LabNet-5'
echo "${out}" | grep -q 'jtsmm-wifi-LabNet-24'

# shellcheck source=pi_wifi_lib.sh
source "${ROOT}/scripts/pi_wifi_lib.sh"
[[ "$(pi_wifi_conn_name_for_ssid 'LabNet' 5)" == "jtsmm-wifi-LabNet-5" ]]
[[ "$(pi_wifi_conn_name_for_ssid 'LabNet' 2.4)" == "jtsmm-wifi-LabNet-24" ]]
[[ "$(pi_wifi_autoconnect_priority_for_band 5)" == "100" ]]
[[ "$(pi_wifi_autoconnect_priority_for_band 2.4)" == "50" ]]
[[ "$(pi_wifi_iface)" == "wlan0" ]]
[[ "$(pi_wifi_find_bssid 'LabNet' 36)" == "AA:BB:CC:DD:EE:FF" ]]

out="$(pi_wifi_discover_ap 'LabNet' both 36)"
IFS=$'\t' read -r band chan bssid <<< "${out}"
[[ "${band}" == "a" && "${chan}" == "36" && "${bssid}" == "AA:BB:CC:DD:EE:FF" ]]

printf '%s\n' 'BB\:BB\:CC\:DD\:EE\:11:SIM:6' > "${SCAN_FILE}"
out="$(pi_wifi_discover_ap 'SIM' both 36)"
IFS=$'\t' read -r band chan bssid <<< "${out}"
[[ "${band}" == "bg" && "${chan}" == "6" && "${bssid}" == "BB:BB:CC:DD:EE:11" ]]

bands="$(pi_wifi_bands_to_provision both | tr '\n' ' ')"
[[ "${bands}" == "5 2.4 " || "${bands}" == "5 2.4" ]]

# Hex-like passphrase must not use `--` before the value (nmcli treats it as the
# psk value, then parses 0b2f43a8b… as a property name).
: > "${MODIFY_LOG}"
pi_wifi_nmcli_apply_wpa_psk 'jtsmm-wifi-test' '0b2f43a8bdeadbeef'
grep -q 'psk-flags none' "${MODIFY_LOG}"
grep -q '802-11-wireless-security.psk 0b2f43a8bdeadbeef' "${MODIFY_LOG}"
! grep -q '802-11-wireless-security.psk --' "${MODIFY_LOG}"

echo "test_pi_wifi_lib.sh: OK"
