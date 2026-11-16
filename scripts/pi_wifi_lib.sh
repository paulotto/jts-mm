# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Shared Wi-Fi profile helpers for pi_network.sh / pi_wifi.sh (source, do not execute).
JTSMM_WIFI_CONN_PREFIX="jtsmm-wifi-"
JTSMM_WIFI_LEGACY_CONN_NAME="jtsmm-lab"
JTSMM_WIFI_PRIORITY_5=100
JTSMM_WIFI_PRIORITY_24=50
# NMSettingSecretFlags none (0x0): PSK stored in the connection keyfile (headless autoconnect).
JTSMM_WIFI_PSK_FLAGS_STORED="none"
WIFI_ACTIVE_PROFILE_FILE="/var/lib/jtsmm/wifi_active_profile"

pi_wifi_nmcli_ok() {
  command -v nmcli >/dev/null 2>&1 || return 1
  nmcli general status >/dev/null 2>&1
}

pi_wifi_iface() {
  local line dev typ
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    dev="${line%%:*}"
    typ="${line#*:}"
    if [[ "${typ}" == "wifi" && -n "${dev}" ]]; then
      printf '%s' "${dev}"
      return 0
    fi
  done < <(nmcli -t -f DEVICE,TYPE device status 2>/dev/null || true)
  printf '%s' "wlan0"
}

pi_wifi_unescape_bssid() {
  local raw="$1"
  raw="${raw//\\:/:}"
  printf '%s' "${raw}"
}

pi_wifi_normalize_band_mode() {
  local raw="${1:-auto}"
  case "${raw,,}" in
    auto|both) printf '%s' "both" ;;
    5|5ghz|5g|a) printf '%s' "5" ;;
    2.4|2.4ghz|2g|24|bg|b) printf '%s' "2.4" ;;
    *)
      echo "ERROR: unknown Wi-Fi band '${raw}' (use auto/both, 5, or 2.4)" >&2
      return 1
      ;;
  esac
}

pi_wifi_channel_band() {
  local ch="$1"
  if [[ "${ch}" -ge 32 ]]; then
    printf '%s' "5"
  elif [[ "${ch}" -ge 1 && "${ch}" -le 14 ]]; then
    printf '%s' "2.4"
  else
    printf '%s' ""
  fi
}

pi_wifi_nm_band_for_band() {
  case "$1" in
    5) printf '%s' "a" ;;
    2.4) printf '%s' "bg" ;;
    *) printf '%s' "" ;;
  esac
}

pi_wifi_band_label() {
  case "$1" in
    5) printf '%s' "5 GHz" ;;
    2.4) printf '%s' "2.4 GHz" ;;
    *) printf '%s' "?" ;;
  esac
}

# Returns nm_band<TAB>channel<TAB>bssid for the first matching visible AP, or nothing.
pi_wifi_scan_for_match() {
  local ssid="$1"
  local want_chan="${2:-}"
  local want_band="${3:-}"
  local iface="$4"
  local line chan rest found_ssid bssid band nm_band

  [[ -n "${iface}" ]] || return 1
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    chan="${line##*:}"
    rest="${line%:*}"
    found_ssid="${rest##*:}"
    bssid="$(pi_wifi_unescape_bssid "${rest%:*}")"
    [[ "${found_ssid}" == "${ssid}" ]] || continue
    band="$(pi_wifi_channel_band "${chan}")"
    [[ -n "${band}" ]] || continue
    [[ -n "${want_band}" && "${band}" != "${want_band}" ]] && continue
    [[ -n "${want_chan}" && "${chan}" != "${want_chan}" ]] && continue
    nm_band="$(pi_wifi_nm_band_for_band "${band}")"
    printf '%s\t%s\t%s' "${nm_band}" "${chan}" "${bssid}"
    return 0
  done < <(nmcli -t -f BSSID,SSID,CHAN device wifi list ifname "${iface}" 2>/dev/null || true)
  return 1
}

pi_wifi_discover_ap() {
  local ssid="$1"
  local raw_mode="${2:-both}"
  local pref_ch="${3:-36}"
  local mode result iface c

  mode="$(pi_wifi_normalize_band_mode "${raw_mode}")" || return 1
  iface="$(pi_wifi_iface)"
  [[ -n "${iface}" ]] || return 1

  if [[ "${mode}" == "both" || "${mode}" == "5" ]]; then
    if [[ "${pref_ch}" -ge 32 ]]; then
      result="$(pi_wifi_scan_for_match "${ssid}" "${pref_ch}" "5" "${iface}")" \
        && { printf '%s' "${result}"; return 0; }
    fi
    for c in 36 40 44 48 52 56 60 64 100 104 108 112 116 120 124 128 132 136 140 144 149 153 157 161 165; do
      [[ "${pref_ch}" -ge 32 && "${c}" == "${pref_ch}" ]] && continue
      result="$(pi_wifi_scan_for_match "${ssid}" "${c}" "5" "${iface}")" \
        && { printf '%s' "${result}"; return 0; }
    done
    result="$(pi_wifi_scan_for_match "${ssid}" "" "5" "${iface}")" \
      && { printf '%s' "${result}"; return 0; }
  fi

  if [[ "${mode}" == "both" || "${mode}" == "2.4" ]]; then
    if [[ "${pref_ch}" -le 14 ]]; then
      result="$(pi_wifi_scan_for_match "${ssid}" "${pref_ch}" "2.4" "${iface}")" \
        && { printf '%s' "${result}"; return 0; }
    fi
    for c in 1 6 11 2 3 4 5 7 8 9 10 12 13; do
      [[ "${pref_ch}" -le 14 && "${c}" == "${pref_ch}" ]] && continue
      result="$(pi_wifi_scan_for_match "${ssid}" "${c}" "2.4" "${iface}")" \
        && { printf '%s' "${result}"; return 0; }
    done
    result="$(pi_wifi_scan_for_match "${ssid}" "" "2.4" "${iface}")" \
      && { printf '%s' "${result}"; return 0; }
  fi

  return 1
}

pi_wifi_bands_to_provision() {
  local raw_mode="${1:-both}"
  local mode
  mode="$(pi_wifi_normalize_band_mode "${raw_mode}")" || return 1
  case "${mode}" in
    both) printf '%s\n' "5" "2.4" ;;
    5|2.4) printf '%s\n' "${mode}" ;;
  esac
}

pi_wifi_autoconnect_priority_for_band() {
  case "$1" in
    5) printf '%s' "${JTSMM_WIFI_PRIORITY_5}" ;;
    2.4) printf '%s' "${JTSMM_WIFI_PRIORITY_24}" ;;
    *) printf '%s' "0" ;;
  esac
}

pi_wifi_pref_channel_for_band() {
  local band="$1"
  local pref="${2:-36}"
  if [[ "${band}" == "5" ]]; then
    if [[ "${pref}" -ge 32 ]]; then
      printf '%s' "${pref}"
    else
      printf '%s' "36"
    fi
  else
    if [[ "${pref}" -le 14 ]]; then
      printf '%s' "${pref}"
    else
      printf '%s' "6"
    fi
  fi
}

pi_wifi_find_bssid() {
  local ssid="$1"
  local channel="$2"
  local out nm_band chan bssid
  out="$(pi_wifi_discover_ap "${ssid}" "5" "${channel}" 2>/dev/null || true)"
  [[ -n "${out}" ]] || return 0
  IFS=$'\t' read -r nm_band chan bssid <<< "${out}"
  printf '%s' "${bssid}"
}

pi_wifi_sanitize_ssid() {
  local ssid="$1"
  local safe="${ssid//[^a-zA-Z0-9._-]/_}"
  while [[ "${safe}" == *"__"* ]]; do
    safe="${safe//__/_}"
  done
  safe="${safe#_}"
  safe="${safe%_}"
  if [[ ${#safe} -gt 40 ]]; then
    safe="${safe:0:40}"
  fi
  if [[ -z "${safe}" ]]; then
    safe="unknown"
  fi
  printf '%s' "${safe}"
}

pi_wifi_conn_name_for_ssid() {
  local ssid="$1"
  local band="${2:-}"
  local base="${JTSMM_WIFI_CONN_PREFIX}$(pi_wifi_sanitize_ssid "${ssid}")"
  case "${band}" in
    5) printf '%s-5' "${base}" ;;
    2.4|24) printf '%s-24' "${base}" ;;
    "") printf '%s' "${base}" ;;
    *)
      echo "ERROR: unknown profile band '${band}'" >&2
      return 1
      ;;
  esac
}

pi_wifi_profile_has_band_suffix() {
  local conn="$1"
  [[ "${conn}" == *-5 || "${conn}" == *-24 ]]
}

pi_wifi_preferred_conn_for_ssid() {
  local ssid="$1"
  local mode="${2:-both}"
  local conn_5 conn_24
  conn_5="$(pi_wifi_conn_name_for_ssid "${ssid}" 5)"
  conn_24="$(pi_wifi_conn_name_for_ssid "${ssid}" 2.4)"
  case "$(pi_wifi_normalize_band_mode "${mode}")" in
    5)
      printf '%s' "${conn_5}"
      ;;
    2.4)
      printf '%s' "${conn_24}"
      ;;
    both)
      if pi_wifi_nmcli_ok && nmcli connection show "${conn_5}" >/dev/null 2>&1; then
        printf '%s' "${conn_5}"
      else
        printf '%s' "${conn_24}"
      fi
      ;;
  esac
}

pi_wifi_alt_conn_name() {
  local conn="$1"
  if [[ "${conn}" == *-5 ]]; then
    printf '%s-24' "${conn%-5}"
  elif [[ "${conn}" == *-24 ]]; then
    printf '%s-5' "${conn%-24}"
  fi
}

pi_wifi_profile_psk_is_stored() {
  local conn="$1"
  local flags key_mgmt
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  key_mgmt="$(nmcli -g 802-11-wireless-security.key-mgmt connection show "${conn}" 2>/dev/null || true)"
  [[ "${key_mgmt}" == "wpa-psk" ]] || return 1
  flags="$(nmcli -g 802-11-wireless-security.psk-flags connection show "${conn}" 2>/dev/null || true)"
  flags="${flags,,}"
  [[ -z "${flags}" || "${flags}" == "0" || "${flags}" == "none" || "${flags}" == *"(none)"* || "${flags}" == *"0x0"* ]] && return 0
  [[ "${flags}" == *"agent"* || "${flags}" == *"not-saved"* || "${flags}" == *"ask"* ]] && return 1
  return 0
}

pi_wifi_nmcli_apply_wpa_psk() {
  local conn="$1"
  local psk="$2"
  # Set key-mgmt + flags first, then the secret in a second call. Do not put `--`
  # between property and value: nmcli parses modify args as <prop> <value> pairs,
  # so `psk -- 0b2f4…` sets psk to "--" and treats 0b2f4… as the next property.
  nmcli connection modify "${conn}" \
    wifi-sec.key-mgmt wpa-psk \
    wifi-sec.psk-flags "${JTSMM_WIFI_PSK_FLAGS_STORED}"
  if [[ "${psk}" == -* ]]; then
    nmcli -- connection modify "${conn}" 802-11-wireless-security.psk "${psk}"
  else
    nmcli connection modify "${conn}" 802-11-wireless-security.psk "${psk}"
  fi
}

pi_wifi_profile_ssid() {
  local conn="$1"
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  nmcli -g 802-11-wireless.ssid connection show "${conn}" 2>/dev/null || true
}

pi_wifi_list_jtsmm_profiles() {
  local out
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  out="$(nmcli -t -f NAME connection show 2>/dev/null)" || return 0
  printf '%s\n' "${out}" | grep -E "^${JTSMM_WIFI_CONN_PREFIX}" || true
}

pi_wifi_active_conn_name() {
  local name=""
  if [[ -f "${WIFI_ACTIVE_PROFILE_FILE}" ]]; then
    name="$(tr -d '[:space:]' < "${WIFI_ACTIVE_PROFILE_FILE}")"
    if [[ -n "${name}" ]] && pi_wifi_nmcli_ok && nmcli connection show "${name}" >/dev/null 2>&1; then
      printf '%s' "${name}"
      return 0
    fi
    if [[ -n "${name}" ]] && pi_wifi_profile_has_band_suffix "${name}"; then
      local alt
      alt="$(pi_wifi_alt_conn_name "${name}")"
      if [[ -n "${alt}" ]] && nmcli connection show "${alt}" >/dev/null 2>&1; then
        printf '%s' "${alt}"
        return 0
      fi
    fi
  fi
  if pi_wifi_nmcli_ok && nmcli connection show "${JTSMM_WIFI_LEGACY_CONN_NAME}" >/dev/null 2>&1; then
    printf '%s' "${JTSMM_WIFI_LEGACY_CONN_NAME}"
    return 0
  fi
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  pi_wifi_list_jtsmm_profiles | head -1
}

pi_wifi_set_active_conn() {
  local conn="$1"
  mkdir -p "$(dirname "${WIFI_ACTIVE_PROFILE_FILE}")"
  printf '%s\n' "${conn}" > "${WIFI_ACTIVE_PROFILE_FILE}"
}

pi_wifi_migrate_legacy_profile() {
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  if ! nmcli connection show "${JTSMM_WIFI_LEGACY_CONN_NAME}" >/dev/null 2>&1; then
    return 0
  fi
  local ssid conn_5
  ssid="$(pi_wifi_profile_ssid "${JTSMM_WIFI_LEGACY_CONN_NAME}")"
  [[ -n "${ssid}" ]] || return 0
  conn_5="$(pi_wifi_conn_name_for_ssid "${ssid}" 5)"
  if nmcli connection show "${conn_5}" >/dev/null 2>&1; then
    nmcli connection delete "${JTSMM_WIFI_LEGACY_CONN_NAME}" 2>/dev/null || true
  else
    nmcli connection modify "${JTSMM_WIFI_LEGACY_CONN_NAME}" connection.id "${conn_5}" 2>/dev/null || true
  fi
}

pi_wifi_migrate_unsuffixed_profiles() {
  local conn ssid conn_5 conn_24
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  while IFS= read -r conn; do
    [[ -z "${conn}" ]] && continue
    pi_wifi_profile_has_band_suffix "${conn}" && continue
    [[ "${conn}" == "${JTSMM_WIFI_LEGACY_CONN_NAME}" ]] && continue
    ssid="$(pi_wifi_profile_ssid "${conn}")"
    [[ -n "${ssid}" ]] || continue
    conn_5="$(pi_wifi_conn_name_for_ssid "${ssid}" 5)"
    conn_24="$(pi_wifi_conn_name_for_ssid "${ssid}" 2.4)"
    if nmcli connection show "${conn_5}" >/dev/null 2>&1 \
      || nmcli connection show "${conn_24}" >/dev/null 2>&1; then
      nmcli connection delete "${conn}" 2>/dev/null || true
    fi
  done < <(pi_wifi_list_jtsmm_profiles)
}

pi_wifi_set_ssid_priorities() {
  local ssid="$1"
  local conn_5 conn_24 other
  if ! pi_wifi_nmcli_ok; then
    return 0
  fi
  conn_5="$(pi_wifi_conn_name_for_ssid "${ssid}" 5)"
  conn_24="$(pi_wifi_conn_name_for_ssid "${ssid}" 2.4)"
  if nmcli connection show "${conn_5}" >/dev/null 2>&1; then
    nmcli connection modify "${conn_5}" connection.autoconnect-priority "${JTSMM_WIFI_PRIORITY_5}" 2>/dev/null || true
  fi
  if nmcli connection show "${conn_24}" >/dev/null 2>&1; then
    nmcli connection modify "${conn_24}" connection.autoconnect-priority "${JTSMM_WIFI_PRIORITY_24}" 2>/dev/null || true
  fi
  while IFS= read -r other; do
    [[ -z "${other}" || "${other}" == "${conn_5}" || "${other}" == "${conn_24}" ]] && continue
    nmcli connection modify "${other}" connection.autoconnect-priority 0 2>/dev/null || true
  done < <(pi_wifi_list_jtsmm_profiles)
}

pi_wifi_resolve_target_ssid() {
  local cli_ssid="$1"
  local active stored
  if [[ -n "${cli_ssid}" ]]; then
    printf '%s' "${cli_ssid}"
    return 0
  fi
  active="$(pi_wifi_active_conn_name)"
  if [[ -n "${active}" ]]; then
    stored="$(pi_wifi_profile_ssid "${active}")"
    if [[ -n "${stored}" ]]; then
      printf '%s' "${stored}"
      return 0
    fi
  fi
  return 1
}
