#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Pi 5 rig networking: hostname, mDNS, firewall, Wi-Fi profiles, wifi guard.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SW="${ROOT}/software/jtsmm"

if [[ ! -f "${SCRIPT_DIR}/pi_wifi_lib.sh" ]]; then
  echo "ERROR: missing ${SCRIPT_DIR}/pi_wifi_lib.sh — run 'git pull' in the repo." >&2
  exit 1
fi
# shellcheck source=pi_wifi_lib.sh
source "${SCRIPT_DIR}/pi_wifi_lib.sh"

JTSMM_HOSTNAME="jtsmm"
JTSMM_VIEWER_PORT=8770

DO_HOSTNAME=1
DO_FIREWALL=1
DO_WIFI=1
DO_WIFI_GUARD=1
DRY_RUN=0
WIFI_SSID="${JTSMM_WIFI_SSID:-}"
WIFI_PSK="${JTSMM_WIFI_PSK:-}"
WIFI_CHANNEL="${JTSMM_WIFI_CHANNEL:-36}"
WIFI_BAND_MODE="${JTSMM_WIFI_BAND:-both}"
WIFI_CHANNEL_SET=0
WIFI_BAND_MODE_SET=0

pi_network_on_err() {
  local rc=$?
  echo "ERROR: $(basename "$0") failed at line ${BASH_LINENO[0]} (exit ${rc})." >&2
  echo "  Re-run with: bash -x $(basename "$0") ..." >&2
  exit "${rc}"
}
trap pi_network_on_err ERR

pi_network_load_wifi_credentials() {
  # sudo strips JTSMM_WIFI_* from the environment; load from the invoking user's files.
  local u="${SUDO_USER:-${USER:-}}"
  local home=""
  if [[ -n "${u}" ]]; then
    home="$(getent passwd "${u}" 2>/dev/null | cut -d: -f6 || true)"
  fi
  local f
  for f in \
    "${home:+"${home}/.jtsmm/wifi.env"}" \
    "${home:+"${home}/.config/jtsmm/wifi.env"}" \
    /etc/jtsmm/wifi.env; do
    [[ -n "${f}" && -f "${f}" ]] || continue
    # shellcheck disable=SC1090
    set +u
    set -a
    # shellcheck disable=SC1090
    source "${f}" || {
      echo "ERROR: could not read Wi-Fi credentials from ${f}" >&2
      exit 1
    }
    set +a
    set -u
    break
  done
  if [[ -z "${WIFI_SSID}" && -n "${JTSMM_WIFI_SSID:-}" ]]; then
    WIFI_SSID="${JTSMM_WIFI_SSID}"
  fi
  if [[ -z "${WIFI_PSK}" && -n "${JTSMM_WIFI_PSK:-}" ]]; then
    WIFI_PSK="${JTSMM_WIFI_PSK}"
  fi
  if [[ "${WIFI_BAND_MODE_SET}" -eq 0 && -n "${JTSMM_WIFI_BAND:-}" ]]; then
    WIFI_BAND_MODE="${JTSMM_WIFI_BAND}"
  fi
  if [[ "${WIFI_CHANNEL_SET}" -eq 0 && -n "${JTSMM_WIFI_CHANNEL:-}" ]]; then
    WIFI_CHANNEL="${JTSMM_WIFI_CHANNEL}"
  fi
}

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Configure Pi rig networking (idempotent).

  --hostname NAME     Set hostname (default: jtsmm)
  --skip-hostname     Skip hostname / /etc/hosts / avahi
  --no-firewall       Skip ufw setup
  --skip-wifi         Ethernet-only (no Wi-Fi profile)
  --wifi-ssid SSID    Wi-Fi SSID (env: JTSMM_WIFI_SSID); creates jtsmm-wifi-<ssid>-5
                      and jtsmm-wifi-<ssid>-24 profiles (5 GHz preferred)
  --wifi-psk PSK      Wi-Fi password (optional on re-run for same SSID)
  --wifi-channel N    Preferred 5 GHz channel (default: 36); 2.4 GHz uses 6 unless N<=14
  --wifi-band MODE    Profile set: both (default), 5, or 2.4
                      both: dual profiles; 5 GHz autoconnect priority 100, 2.4 GHz 50
  --skip-wifi-guard   Do not install pi_wifi systemd units
  --dry-run           Print Wi-Fi actions only (no system changes)
  -h, --help          Show this help

Wi-Fi credentials (pick one):
  export JTSMM_WIFI_SSID='...' JTSMM_WIFI_PSK='...'   # then run WITHOUT sudo
  ~/.jtsmm/wifi.env with JTSMM_WIFI_SSID / JTSMM_WIFI_PSK lines
  optional: JTSMM_WIFI_BAND=both|5|2.4  JTSMM_WIFI_CHANNEL=36
  --wifi-ssid / --wifi-psk / --wifi-band / --wifi-channel flags

Running without --wifi-ssid only works when a jtsmm-wifi-* profile already exists
or ~/.jtsmm/wifi.env defines JTSMM_WIFI_SSID.
EOF
}

pi_network_hostname() {
  [[ "${DO_HOSTNAME}" -eq 1 ]] || return 0
  echo "==> hostname -> ${JTSMM_HOSTNAME}"
  run_root hostnamectl set-hostname "${JTSMM_HOSTNAME}"
  if ! grep -q "^127.0.1.1[[:space:]]\+${JTSMM_HOSTNAME}\b" /etc/hosts 2>/dev/null; then
    if grep -q '^127.0.1.1' /etc/hosts 2>/dev/null; then
      run_root sed -i "s/^127.0.1.1.*/127.0.1.1 ${JTSMM_HOSTNAME}/" /etc/hosts
    else
      run_root bash -c "echo '127.0.1.1 ${JTSMM_HOSTNAME}' >> /etc/hosts"
    fi
  fi
  if command -v apt-get >/dev/null 2>&1; then
    run_root apt-get install -y avahi-daemon \
      || echo "WARNING: avahi-daemon install failed (offline?); continuing" >&2
    run_root systemctl enable avahi-daemon 2>/dev/null || true
    run_root systemctl start avahi-daemon 2>/dev/null || true
  fi
}

pi_network_firewall() {
  [[ "${DO_FIREWALL}" -eq 1 ]] || return 0
  if ! command -v ufw >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1; then
      run_root apt-get install -y ufw
    else
      echo "WARNING: ufw not found; skipping firewall" >&2
      return 0
    fi
  fi
  echo "==> ufw (SSH + viewer ${JTSMM_VIEWER_PORT})"
  run_root ufw default deny incoming || true
  run_root ufw default allow outgoing || true
  run_root ufw allow OpenSSH || true
  run_root ufw allow "${JTSMM_VIEWER_PORT}/tcp" comment 'jtsmm viewer' || true
  run_root ufw --force enable || true
}

pi_network_wifi_preflight() {
  [[ "${DO_WIFI}" -eq 1 ]] || return 0
  if ! command -v nmcli >/dev/null 2>&1; then
    echo "ERROR: nmcli not found — install NetworkManager (network-manager package)." >&2
    exit 1
  fi
  if ! pi_wifi_nmcli_ok; then
    echo "ERROR: NetworkManager is not running — try: sudo systemctl start NetworkManager" >&2
    exit 1
  fi
  if ! pi_wifi_normalize_band_mode "${WIFI_BAND_MODE}" >/dev/null; then
    exit 1
  fi
  local target_ssid=""
  if ! target_ssid="$(pi_wifi_resolve_target_ssid "${WIFI_SSID}")"; then
    echo "ERROR: Wi-Fi SSID required and no existing jtsmm-wifi-* profile found." >&2
    echo "  Provide credentials via one of:" >&2
    echo "    ./pi_network.sh --wifi-ssid 'YourSSID' --wifi-psk 'secret'" >&2
    echo "    ~/.jtsmm/wifi.env  (JTSMM_WIFI_SSID=... and JTSMM_WIFI_PSK=...)" >&2
    echo "    export JTSMM_WIFI_SSID=... JTSMM_WIFI_PSK=...  (run without sudo)" >&2
    echo "  Or use ./install_pi.sh from the repo root (not from scripts/)." >&2
    exit 1
  fi
  printf '%s' "${target_ssid}"
}

pi_network_wifi_discover_band() {
  local ssid="$1"
  local band="$2"
  local pref_ch="$3"
  local discover_out nm_band chan bssid band_label

  discover_out="$(pi_wifi_discover_ap "${ssid}" "${band}" "${pref_ch}" 2>/dev/null || true)"
  if [[ -n "${discover_out}" ]]; then
    IFS=$'\t' read -r nm_band chan bssid <<< "${discover_out}"
    band_label="$(pi_wifi_band_label "${band}")"
    echo "==> matched ${band_label} AP for '${ssid}' (channel ${chan}, BSSID ${bssid})" >&2
    printf '%s\t%s\t%s\t%s' "${nm_band}" "${chan}" "${bssid}" "${band}"
    return 0
  fi

  nm_band="$(pi_wifi_nm_band_for_band "${band}")"
  band_label="$(pi_wifi_band_label "${band}")"
  echo "WARNING: no visible ${band_label} AP for '${ssid}'." >&2
  echo "  Creating ${band_label} profile anyway (no BSSID lock)." >&2
  printf '%s\t\t\t%s' "${nm_band}" "${band}"
}

pi_network_wifi_upsert_profile() {
  local target_ssid="$1"
  local band="$2"
  local iface="$3"
  local pref_ch="$4"
  local discover_out nm_band matched_ch matched_bssid band_label priority conn_name

  discover_out="$(pi_network_wifi_discover_band "${target_ssid}" "${band}" "${pref_ch}")"
  IFS=$'\t' read -r nm_band matched_ch matched_bssid _band <<< "${discover_out}"
  if [[ "${nm_band}" != "a" && "${nm_band}" != "bg" ]]; then
    echo "ERROR: invalid nmcli band '${nm_band}' while configuring ${band} GHz profile" >&2
    exit 1
  fi
  band_label="$(pi_wifi_band_label "${band}")"
  conn_name="$(pi_wifi_conn_name_for_ssid "${target_ssid}" "${band}")"
  priority="$(pi_wifi_autoconnect_priority_for_band "${band}")"

  echo "==> Wi-Fi profile ${conn_name} -> ${target_ssid} (${band_label}, priority ${priority})"

  if [[ -z "${WIFI_PSK}" ]]; then
    if ! run_root nmcli -g 802-11-wireless-security.key-mgmt connection show "${conn_name}" 2>/dev/null | grep -qx 'wpa-psk'; then
      echo "ERROR: PSK required for ${conn_name} (pass --wifi-psk or set JTSMM_WIFI_PSK)" >&2
      exit 1
    fi
    if ! pi_wifi_profile_psk_is_stored "${conn_name}"; then
      echo "ERROR: PSK required for ${conn_name} (stored password missing)" >&2
      exit 1
    fi
  fi

  if nmcli connection show "${conn_name}" >/dev/null 2>&1; then
    local -a mod_args=(
      connection modify "${conn_name}"
      802-11-wireless.ssid "${target_ssid}"
      802-11-wireless.band "${nm_band}"
      802-11-wireless.powersave 2
      connection.autoconnect yes
      connection.autoconnect-priority "${priority}"
    )
    if [[ -n "${matched_bssid}" ]]; then
      mod_args+=(802-11-wireless.bssid "${matched_bssid}")
    else
      mod_args+=(802-11-wireless.bssid "")
    fi
    run_root nmcli "${mod_args[@]}"
    if [[ -n "${WIFI_PSK}" ]]; then
      run_root bash -c 'source "$3/pi_wifi_lib.sh"; pi_wifi_nmcli_apply_wpa_psk "$1" "$2"' \
        bash "${conn_name}" "${WIFI_PSK}" "${SCRIPT_DIR}"
    fi
  else
    if [[ -z "${WIFI_PSK}" ]]; then
      echo "ERROR: cannot create new profile ${conn_name} without PSK" >&2
      exit 1
    fi
    local -a add_args=(
      connection add type wifi con-name "${conn_name}" ifname "${iface}"
      ssid "${target_ssid}"
      802-11-wireless.band "${nm_band}" 802-11-wireless.powersave 2
      connection.autoconnect yes connection.autoconnect-priority "${priority}"
    )
    if [[ -n "${matched_bssid}" ]]; then
      add_args+=(802-11-wireless.bssid "${matched_bssid}")
    fi
    run_root nmcli "${add_args[@]}"
    run_root bash -c 'source "$3/pi_wifi_lib.sh"; pi_wifi_nmcli_apply_wpa_psk "$1" "$2"' \
      bash "${conn_name}" "${WIFI_PSK}" "${SCRIPT_DIR}"
  fi

  if ! run_root nmcli -g 802-11-wireless-security.key-mgmt connection show "${conn_name}" 2>/dev/null | grep -qx 'wpa-psk'; then
    echo "ERROR: ${conn_name} is not configured for WPA-PSK." >&2
    echo "  If ${target_ssid} uses WPA2-Enterprise (802.1X), this script cannot configure it." >&2
    exit 1
  fi
  if ! pi_wifi_profile_psk_is_stored "${conn_name}"; then
    psk_flags="$(run_root nmcli -g 802-11-wireless-security.psk-flags connection show "${conn_name}" 2>/dev/null || true)"
    echo "ERROR: ${conn_name} has no stored PSK (flags: ${psk_flags:-?})." >&2
    echo "  Re-run with --wifi-psk '...' (quote passwords containing \$ or !)." >&2
    exit 1
  fi

  if [[ -n "${matched_ch}" ]]; then
    echo "==> ${conn_name} ready (${band_label} channel ${matched_ch})"
  else
    echo "==> ${conn_name} ready (${band_label}, no scan match)"
  fi
}

pi_network_wifi() {
  [[ "${DO_WIFI}" -eq 1 ]] || { DO_WIFI_GUARD=0; return 0; }

  local target_ssid mode_label band iface pref_ch band_list preferred_conn need_psk conn_name
  target_ssid="$(pi_network_wifi_preflight)"
  mode_label="$(pi_wifi_normalize_band_mode "${WIFI_BAND_MODE}")"

  run_root bash -c "source '${SCRIPT_DIR}/pi_wifi_lib.sh'; pi_wifi_migrate_legacy_profile; pi_wifi_migrate_unsuffixed_profiles" || true

  need_psk=0
  while IFS= read -r band; do
    [[ -z "${band}" ]] && continue
    conn_name="$(pi_wifi_conn_name_for_ssid "${target_ssid}" "${band}")"
    if ! nmcli connection show "${conn_name}" >/dev/null 2>&1; then
      need_psk=1
      break
    fi
  done < <(pi_wifi_bands_to_provision "${mode_label}")

  if [[ "${need_psk}" -eq 1 && -z "${WIFI_PSK}" ]]; then
    echo "ERROR: PSK required for new SSID '${target_ssid}' (set JTSMM_WIFI_PSK or --wifi-psk)" >&2
    exit 1
  fi

  iface="$(pi_wifi_iface)"
  if [[ -z "${iface}" ]]; then
    echo "ERROR: no Wi-Fi interface found (nmcli device status)." >&2
    exit 1
  fi

  echo "==> Wi-Fi SSID ${target_ssid} (iface ${iface}, mode ${mode_label}, prefer 5 GHz ch ${WIFI_CHANNEL})"
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    while IFS= read -r band; do
      [[ -z "${band}" ]] && continue
      conn_name="$(pi_wifi_conn_name_for_ssid "${target_ssid}" "${band}")"
      echo "DRY-RUN: would create/update ${conn_name} (priority $(pi_wifi_autoconnect_priority_for_band "${band}"))"
    done < <(pi_wifi_bands_to_provision "${mode_label}")
    return 0
  fi

  run_root nmcli device wifi rescan ifname "${iface}" 2>/dev/null || true
  sleep 2

  while IFS= read -r band; do
    [[ -z "${band}" ]] && continue
    pref_ch="$(pi_wifi_pref_channel_for_band "${band}" "${WIFI_CHANNEL}")"
    pi_network_wifi_upsert_profile "${target_ssid}" "${band}" "${iface}" "${pref_ch}"
  done < <(pi_wifi_bands_to_provision "${mode_label}")

  run_root bash -c "source '${SCRIPT_DIR}/pi_wifi_lib.sh'; pi_wifi_set_ssid_priorities '${target_ssid}'" || true
  preferred_conn="$(pi_wifi_preferred_conn_for_ssid "${target_ssid}" "${mode_label}")"
  run_root mkdir -p "$(dirname "${WIFI_ACTIVE_PROFILE_FILE}")"
  run_root bash -c "printf '%s\n' '${preferred_conn}' > '${WIFI_ACTIVE_PROFILE_FILE}'"
  if ! run_root nmcli connection up "${preferred_conn}" ifname "${iface}"; then
    alt="$(pi_wifi_alt_conn_name "${preferred_conn}")"
    if [[ -n "${alt}" ]]; then
      run_root nmcli connection up "${alt}" ifname "${iface}" \
        || echo "WARNING: could not bring up Wi-Fi for ${target_ssid} (check PSK or AP availability)" >&2
    else
      echo "WARNING: could not bring up ${preferred_conn} (check PSK or AP availability)" >&2
    fi
  fi
  echo "==> Wi-Fi preferred profile: ${preferred_conn} (5 GHz priority over 2.4 GHz when both exist)"
}

pi_wifi_install_systemd() {
  [[ "${DO_WIFI_GUARD}" -eq 1 ]] || return 0
  local svc_src="${SW}/etc/systemd/jtsmm-wifi-guard.service"
  local tmr_src="${SW}/etc/systemd/jtsmm-wifi-guard.timer"
  local bin_src="${SCRIPT_DIR}/pi_wifi.sh"
  echo "==> install pi_wifi + jtsmm-wifi-guard systemd units"
  run_root mkdir -p /var/lib/jtsmm /usr/local/bin
  run_root install -m 0755 "${bin_src}" /usr/local/bin/pi_wifi
  run_root install -m 0644 "${svc_src}" /etc/systemd/system/jtsmm-wifi-guard.service
  run_root install -m 0644 "${tmr_src}" /etc/systemd/system/jtsmm-wifi-guard.timer
  run_root systemctl daemon-reload
  run_root systemctl enable jtsmm-wifi-guard.timer
  run_root systemctl start jtsmm-wifi-guard.timer || true
}

pi_network_summary() {
  local wifi_line="skipped — Ethernet only"
  if [[ "${DO_WIFI}" -eq 1 ]]; then
    local active profiles mode_label
    active="$(pi_wifi_active_conn_name)"
    profiles="$(pi_wifi_list_jtsmm_profiles | tr '\n' ' ')"
    mode_label="$(pi_wifi_normalize_band_mode "${WIFI_BAND_MODE}")"
    wifi_line="preferred ${active:-?}; mode ${mode_label} (5 GHz priority 100, 2.4 GHz 50)"
    if [[ -n "${profiles}" ]]; then
      wifi_line="${wifi_line}; profiles: ${profiles}"
    fi
  fi
  cat <<EOF

Rig network:
  hostname:  ${JTSMM_HOSTNAME}  (mDNS: ${JTSMM_HOSTNAME}.local)
  viewer:    http://${JTSMM_HOSTNAME}.local:${JTSMM_VIEWER_PORT}/  (ufw: allowed)
  ssh:       ssh pi@${JTSMM_HOSTNAME}.local
  wifi:      ${wifi_line}
  note:      each --wifi-ssid creates jtsmm-wifi-<ssid>-5 and -24 (unless --wifi-band limits band)
  note:      NetworkManager prefers 5 GHz when both APs are visible; disable Wi-Fi during EMG capture
Reboot recommended if hostname or Wi-Fi changed.
EOF
}

main() {
  echo "==> jtsmm Pi network setup"
  pi_network_load_wifi_credentials
  # Wi-Fi first so profile creation is not blocked by apt/hostname on an offline Pi.
  pi_network_wifi
  pi_network_hostname
  pi_network_firewall
  pi_wifi_install_systemd
  pi_network_summary
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hostname) JTSMM_HOSTNAME="$2"; shift 2 ;;
    --skip-hostname) DO_HOSTNAME=0; shift ;;
    --no-firewall) DO_FIREWALL=0; shift ;;
    --skip-wifi) DO_WIFI=0; shift ;;
    --wifi-ssid) WIFI_SSID="$2"; shift 2 ;;
    --wifi-psk) WIFI_PSK="$2"; shift 2 ;;
    --wifi-channel) WIFI_CHANNEL="$2"; WIFI_CHANNEL_SET=1; shift 2 ;;
    --wifi-band) WIFI_BAND_MODE="$2"; WIFI_BAND_MODE_SET=1; shift 2 ;;
    --skip-wifi-guard) DO_WIFI_GUARD=0; shift ;;
    --dry-run) DRY_RUN=1; DO_HOSTNAME=0; DO_FIREWALL=0; DO_WIFI_GUARD=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

main
