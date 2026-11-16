#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Pi 5 hardware/OS tuning (groups, governor, USB current).
# Sourced by scripts/build_install.sh when --pi-tune is set.

pi_tune_apply() {
  local user_name="$1"
  local data_dir="$2"
  local prefix="$3"
  local config_dst="$4"
  local run_root="$5"  # function name: run_root or identity
  local sw_root="${6:-}"

  [[ -n "${user_name}" ]] || { echo "ERROR: --pi-tune needs --user or SUDO_USER" >&2; return 1; }
  echo "==> Pi tune for user ${user_name}"

  for grp in dialout plugdev video; do
    if getent group "${grp}" >/dev/null; then
      "${run_root}" usermod -aG "${grp}" "${user_name}" || true
    fi
  done

  "${run_root}" mkdir -p "${data_dir}" "${prefix}" "$(dirname "${config_dst}")"
  "${run_root}" chown -R "${user_name}:${user_name}" "${prefix}" "${data_dir}" "$(dirname "${config_dst}")" 2>/dev/null || true
  if [[ -f "${config_dst}" ]]; then
    "${run_root}" chown "${user_name}:${user_name}" "${config_dst}" 2>/dev/null || true
  fi

  if [[ -d /sys/devices/system/cpu/cpu0/cpufreq ]]; then
    echo "==> CPU governor -> performance (persistent via systemd)"
    local gov_svc="${sw_root}/etc/systemd/jtsmm-cpu-governor.service"
    if [[ -f "${gov_svc}" ]]; then
      "${run_root}" install -m 0644 "${gov_svc}" /etc/systemd/system/jtsmm-cpu-governor.service
      "${run_root}" systemctl daemon-reload
      "${run_root}" systemctl enable jtsmm-cpu-governor.service
      "${run_root}" systemctl start jtsmm-cpu-governor.service || true
    else
      "${run_root}" bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor >/dev/null' || true
    fi
  fi

  local boot_cfg=""
  for c in /boot/firmware/config.txt /boot/config.txt; do
    [[ -f "${c}" ]] && boot_cfg="${c}" && break
  done
  if [[ -n "${boot_cfg}" ]]; then
    if ! grep -q '^usb_max_current_enable=1' "${boot_cfg}" 2>/dev/null; then
      echo "==> append usb_max_current_enable=1 to ${boot_cfg}"
      "${run_root}" bash -c "echo 'usb_max_current_enable=1' >> '${boot_cfg}'" || true
    fi
  fi

  # SSH + disable Bluetooth (s01 §2).
  if command -v systemctl >/dev/null 2>&1; then
    "${run_root}" systemctl enable ssh 2>/dev/null || true
    "${run_root}" systemctl disable --now bluetooth 2>/dev/null || true
  fi
  if command -v rfkill >/dev/null 2>&1; then
    "${run_root}" rfkill block bluetooth 2>/dev/null || true
  fi

  echo "Pi tune done. Re-login for new groups; replug USB after --udev."
}
