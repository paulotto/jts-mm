#!/usr/bin/env bash
# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

# Build and install the JTS-MM host stack (+ optional firmware) from a source checkout.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SW="${ROOT}/software/jtsmm"
FW="${ROOT}/firmware/jtsmm"
NATIVE="${SW}/native"
OPTICAL_SW="${ROOT}/software/optical"
JTS_SW="${ROOT}/software/jaw_tracking"
BIOMECH_SW="${ROOT}/software/biomech"

PY="${PYTHON:-python3}"

DO_VENV=1
DO_NATIVE=1
DO_HOST=1
DO_FIRMWARE=0
DO_FLASH=0
DO_UDEV=0
DO_PI_TUNE=0
DO_PI_NETWORK=0
DO_CONFIG=0
DO_DEV=0
DO_CAMERAS=0
DO_VIEWER=0
DO_OPTICAL=1
DO_ANALYSIS=0
SYSTEM_SITE=0
DO_APT=1
PI_NETWORK_ARGS=()
EDITABLE=0
SKIP_TESTS=1
USER_NAME="${SUDO_USER:-${USER:-}}"

PREFIX=""
VENV=""
DATA_DIR=""
CONFIG_DST=""
PREFIX_SET=0
VENV_SET=0
DATA_DIR_SET=0
CONFIG_SET=0

home_dir_for_user() {
  local u="$1"
  local home=""
  if [[ -n "${u}" ]]; then
    home="$(getent passwd "${u}" 2>/dev/null | cut -d: -f6 || true)"
  fi
  if [[ -z "${home}" || ! -d "${home}" ]]; then
    home="${HOME:-}"
  fi
  if [[ -z "${home}" ]]; then
    echo "ERROR: cannot resolve home directory for user '${u:-<unset>}'" >&2
    exit 1
  fi
  printf '%s' "${home}"
}

apply_defaults() {
  local home
  home="$(home_dir_for_user "${USER_NAME}")"

  if [[ "${PREFIX_SET}" -eq 0 ]]; then
    PREFIX="${JTSMM_PREFIX:-${home}/jtsmm}"
  fi
  if [[ "${VENV_SET}" -eq 0 ]]; then
    VENV="${JTSMM_VENV:-${PREFIX}/venv}"
  fi
  if [[ "${DATA_DIR_SET}" -eq 0 ]]; then
    DATA_DIR="${JTSMM_DATA_DIR:-${home}/jtsmm_data}"
  fi
  if [[ "${CONFIG_SET}" -eq 0 ]]; then
    CONFIG_DST="${JTSMM_CONFIG:-${home}/jtsmm.yaml}"
  fi
  normalize_config_dst
}

normalize_config_dst() {
  [[ -n "${CONFIG_DST}" ]] || return 0
  if [[ -d "${CONFIG_DST}" ]]; then
    echo "==> --config-path is a directory; using ${CONFIG_DST%/}/jtsmm.yaml" >&2
    CONFIG_DST="${CONFIG_DST%/}/jtsmm.yaml"
  fi
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Build and install jtsmm from: ${ROOT}

Host (default on):
  --prefix DIR       Install root (default: ~/jtsmm)
  --venv PATH        Python venv (default: <prefix>/venv)
  --python PATH      Python interpreter for venv + native build (default: python3)
  --editable         pip install -e software/jtsmm (+ software/optical); the rig
                     runs from its checkout, so install_pi.sh always passes this
  --no-host          Skip pip install of the Python package
  --no-native        Skip C++/libusb/libhdf5 extension build
  --no-venv          Use current Python env only (no venv create)
  --dev              Install pytest/ruff extras and run unit tests
  --cameras          picamera2 via apt (python3-picamera2) + a venv that sees
                     system site-packages (libcamera has no pip wheel)
  --system-site-packages
                     Create/flip the venv to include system site-packages
  --no-optical       Skip pip install of software/optical (optical tracking,
                     CT pivot, IMU hand-eye need it)
  --viewer           Install fastapi/uvicorn/msgpack viewer extra
  --analysis         Offline analysis stack for the experiments/ scripts:
                     jtsmm[analysis] + software/jaw_tracking + software/biomech[plot,stats]
                     (install_pi.sh passes this; --no-analysis skips it)
  --no-analysis      Skip the analysis stack
  --config           Copy etc/jtsmm.yaml -> ${CONFIG_DST} (and set host.paths.data_dir)
  --config-path PATH Operator YAML file (default: ~/jtsmm.yaml). If a directory is
                     given, <dir>/jtsmm.yaml is used instead.
  --data-dir DIR     Session output directory (default: ~/jtsmm_data)

Firmware (optional):
  --firmware         Build firmware/jtsmm (requires PICO_SDK_PATH)
  --flash            Flash firmware via scripts/flash_swd.sh (implies --firmware)

System (optional, may need sudo):
  --udev             Install 99-jtsmm.rules via scripts/install_udev.sh
  --pi-tune          Apply recommended Pi 5 settings (groups, governor, data dir)
  --pi-network       Configure hostname, mDNS, ufw, Wi-Fi (see scripts/pi_network.sh)
  --hostname NAME    Rig hostname for --pi-network (default: jtsmm)
  --skip-hostname    Skip hostname/mDNS during --pi-network
  --no-firewall      Skip ufw during --pi-network
  --skip-wifi        Ethernet-only rig (skip Wi-Fi profile)
  --wifi-ssid SSID   Lab Wi-Fi SSID (env: JTSMM_WIFI_SSID)
  --wifi-psk PSK     Wi-Fi password (optional on re-run for same SSID)
  --wifi-channel N   Preferred Wi-Fi channel (default: 36)
  --wifi-band MODE   Wi-Fi band search: both, 5, or 2.4 (env: JTSMM_WIFI_BAND; default both)
  --user NAME        User for group membership / directory ownership (default: \$SUDO_USER or \$USER)
  --no-apt           Skip automatic apt install of build/runtime dependencies

Other:
  --run-tests        Run pytest after install (requires --dev, or pytest already installed)
  --skip-tests       Skip pytest after host install (default for production installs)
  -h, --help         Show this help

Environment:
  PICO_SDK_PATH      Required for --firmware
  JTSMM_PREFIX, JTSMM_VENV, JTSMM_DATA_DIR, JTSMM_CONFIG
  JTSMM_WIFI_SSID, JTSMM_WIFI_PSK  (or ~/.jtsmm/wifi.env; sudo strips exports unless -E)
  JTSMM_WIFI_BAND=both|5|2.4  JTSMM_WIFI_CHANNEL=N

Examples:
  # Dev laptop: editable install under ~/jtsmm
  $(basename "$0") --editable

  # Pi 5 one-shot install (apt + host + config + udev + cameras + viewer + analysis)
  $(basename "$0") --config --udev --pi-tune --pi-network --cameras --viewer --analysis --user pi

  # Full bench: host + firmware + flash
  $(basename "$0") --editable --firmware --flash --udev

When the venv sees system site-packages (--cameras, --system-site-packages, or an
existing venv that has them) every pip install runs with a constraints file
(<prefix>/pip-constraints.txt: numpy==<apt version>). The venv shares apt's numpy
(1.24 on Bookworm, 2.2 on Trixie) with picamera2/simplejpeg, and a numpy wheel of
another version pulled in by a dependency would break them. Re-use it for manual
installs: pip install -c <prefix>/pip-constraints.txt ...

By default this script runs apt-get for python3, build tools, libusb, HDF5,
and libcamera dev packages on Debian / Pi OS / Ubuntu before building.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; PREFIX_SET=1; shift 2 ;;
    --venv) VENV="$2"; VENV_SET=1; shift 2 ;;
    --python) PY="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; DATA_DIR_SET=1; shift 2 ;;
    --config-path) CONFIG_DST="$2"; CONFIG_SET=1; shift 2 ;;
    --user) USER_NAME="$2"; shift 2 ;;
    --editable) EDITABLE=1; shift ;;
    --no-host) DO_HOST=0; shift ;;
    --no-native) DO_NATIVE=0; shift ;;
    --no-venv) DO_VENV=0; shift ;;
    --dev) DO_DEV=1; SKIP_TESTS=0; shift ;;
    --cameras) DO_CAMERAS=1; SYSTEM_SITE=1; shift ;;
    --system-site-packages) SYSTEM_SITE=1; shift ;;
    --no-optical) DO_OPTICAL=0; shift ;;
    --viewer) DO_VIEWER=1; shift ;;
    --analysis) DO_ANALYSIS=1; shift ;;
    --no-analysis) DO_ANALYSIS=0; shift ;;
    --config) DO_CONFIG=1; shift ;;
    --firmware) DO_FIRMWARE=1; shift ;;
    --flash) DO_FIRMWARE=1; DO_FLASH=1; shift ;;
    --udev) DO_UDEV=1; shift ;;
    --pi-tune) DO_PI_TUNE=1; shift ;;
    --pi-network) DO_PI_NETWORK=1; shift ;;
    --hostname) PI_NETWORK_ARGS+=(--hostname "$2"); shift 2 ;;
    --skip-hostname) PI_NETWORK_ARGS+=(--skip-hostname); DO_PI_NETWORK=1; shift ;;
    --no-firewall) PI_NETWORK_ARGS+=(--no-firewall); DO_PI_NETWORK=1; shift ;;
    --skip-wifi) PI_NETWORK_ARGS+=(--skip-wifi); DO_PI_NETWORK=1; shift ;;
    --wifi-ssid) PI_NETWORK_ARGS+=(--wifi-ssid "$2"); DO_PI_NETWORK=1; shift 2 ;;
    --wifi-psk) PI_NETWORK_ARGS+=(--wifi-psk "$2"); DO_PI_NETWORK=1; shift 2 ;;
    --wifi-channel) PI_NETWORK_ARGS+=(--wifi-channel "$2"); DO_PI_NETWORK=1; shift 2 ;;
    --wifi-band) PI_NETWORK_ARGS+=(--wifi-band "$2"); DO_PI_NETWORK=1; shift 2 ;;
    --skip-wifi-guard) PI_NETWORK_ARGS+=(--skip-wifi-guard); DO_PI_NETWORK=1; shift ;;
    --no-apt) DO_APT=0; shift ;;
    --run-tests) SKIP_TESTS=0; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) apply_defaults; usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; apply_defaults; usage; exit 1 ;;
  esac
done

apply_defaults

log() { echo "==> $*"; }

run_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "ERROR: missing required command: $1" >&2
    exit 1
  fi
}

apt_hint() {
  cat >&2 <<EOF
Re-run without --no-apt to install system packages automatically, or run:
  sudo apt install -y python3 python3-venv python3-dev \\
    build-essential cmake ninja-build pkg-config \\
    libusb-1.0-0-dev libhdf5-dev libcap-dev \\
    libcamera-dev libcamera-tools
EOF
}

install_apt_deps() {
  if [[ "${DO_APT}" -ne 1 ]]; then
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "WARNING: apt-get not found; skipping system package install" >&2
    return 0
  fi

  local packages=(
    python3
    python3-venv
    python3-dev
    build-essential
    cmake
    ninja-build
    pkg-config
    libusb-1.0-0-dev
    libhdf5-dev
    libcap-dev
    libcamera-dev
    libcamera-tools
  )

  log "system packages (apt-get)"
  run_root apt-get update -qq
  if [[ "${DO_CAMERAS}" -eq 1 ]]; then
    # Pi OS ships picamera2 + the libcamera Python bindings only as apt
    # packages; the venv sees them through --system-site-packages.
    local p
    for p in python3-libcamera python3-picamera2; do
      if apt-cache show "${p}" >/dev/null 2>&1; then
        packages+=("${p}")
      else
        echo "WARNING: apt package ${p} not available here (not Pi OS?)" >&2
      fi
    done
  fi
  run_root apt-get install -y "${packages[@]}"
}

host_prereqs() {
  require_cmd "${PY}"
  if [[ "${DO_VENV}" -eq 1 ]]; then
    if ! "${PY}" -m venv --help >/dev/null 2>&1; then
      echo "ERROR: ${PY} cannot create virtual environments." >&2
      echo "Install the OS package: sudo apt install -y python3-venv" >&2
      apt_hint
      exit 1
    fi
    return
  fi
  if ! "${PY}" -c "import pip" 2>/dev/null; then
    echo "ERROR: ${PY} has no pip — pass --venv (default) or install python3-pip." >&2
    apt_hint
    exit 1
  fi
}

ensure_venv_pip() {
  if "${PY}" -c "import pip" 2>/dev/null; then
    return
  fi
  log "bootstrapping pip in venv via ensurepip"
  if ! "${PY}" -m ensurepip --upgrade >/dev/null 2>&1; then
    echo "ERROR: ${PY} has no pip after venv create." >&2
    echo "Install: sudo apt install -y python3-venv" >&2
    apt_hint
    exit 1
  fi
}

create_venv() {
  if [[ "${DO_VENV}" -eq 0 ]]; then
    return
  fi
  log "Python venv: ${VENV}"
  mkdir -p "$(dirname "${VENV}")"
  local venv_args=()
  [[ "${SYSTEM_SITE}" -eq 1 ]] && venv_args+=(--system-site-packages)
  if [[ ! -d "${VENV}" ]]; then
    "${PY}" -m venv "${venv_args[@]}" "${VENV}"
  elif [[ "${SYSTEM_SITE}" -eq 1 && -f "${VENV}/pyvenv.cfg" ]] \
      && grep -q '^include-system-site-packages = false' "${VENV}/pyvenv.cfg"; then
    log "existing venv: enabling system site-packages (apt picamera2/libcamera)"
    sed -i 's/^include-system-site-packages = false/include-system-site-packages = true/' \
      "${VENV}/pyvenv.cfg"
  fi
  # shellcheck disable=SC1091
  source "${VENV}/bin/activate"
  PY="${VENV}/bin/python"
  ensure_venv_pip
  pip install -U pip wheel
}

PIP_CONSTRAINT_ARGS=()

venv_shares_system_site() {
  [[ "${SYSTEM_SITE}" -eq 1 ]] && return 0
  [[ -f "${VENV}/pyvenv.cfg" ]] && grep -q '^include-system-site-packages = true' "${VENV}/pyvenv.cfg"
}

system_numpy_version() {
  # numpy of the interpreter the venv was created from (apt's python3-numpy), not a
  # copy pip may already have put into the venv; -s skips ~/.local.
  local base
  base="$("${PY}" -c 'import sys; print(getattr(sys, "_base_executable", "") or sys.executable)' \
    2>/dev/null)" || return 0
  "${base}" -s -c 'import numpy; print(numpy.__version__)' 2>/dev/null || true
}

write_pip_constraints() {
  # Keep apt's numpy (picamera2/simplejpeg are built against it: 1.24 on Bookworm,
  # 2.2 on Trixie) whenever the venv sees system site-packages, so no dependency
  # (scipy/pandas of --analysis, opencv-python) swaps it for another major version.
  local cons="${PREFIX}/pip-constraints.txt"
  local np=""
  venv_shares_system_site && np="$(system_numpy_version)"
  if [[ -z "${np}" ]]; then
    rm -f "${cons}"
    return 0
  fi
  mkdir -p "${PREFIX}"
  cat > "${cons}" <<CONS
# Written by scripts/build_install.sh. The venv shares apt's numpy ${np} with
# python3-picamera2 / simplejpeg, which are built against it.
numpy==${np}
CONS
  PIP_CONSTRAINT_ARGS=(-c "${cons}")
  log "pip constraints: ${cons} (numpy==${np}, apt)"
}

pip_install() {
  # pip install with the numpy constraint (write_pip_constraints), if one applies.
  pip install "${PIP_CONSTRAINT_ARGS[@]}" "$@"
}

install_host() {
  if [[ "${DO_HOST}" -ne 1 ]]; then
    return 0
  fi
  host_prereqs
  create_venv
  if [[ "${DO_VENV}" -eq 0 ]]; then
    # shellcheck disable=SC1091
    [[ -f "${VENV}/bin/activate" ]] && source "${VENV}/bin/activate" || true
  fi
  write_pip_constraints

  extras=()
  [[ "${DO_DEV}" -eq 1 ]] && extras+=(dev)
  [[ "${DO_NATIVE}" -eq 1 ]] && extras+=(native)
  [[ "${DO_CAMERAS}" -eq 1 ]] && extras+=(cameras)
  [[ "${DO_VIEWER}" -eq 1 ]] && extras+=(viewer)
  [[ "${DO_OPTICAL}" -eq 1 ]] && extras+=(optical)

  pkg="${SW}"
  if [[ ${#extras[@]} -gt 0 ]]; then
    pkg="${SW}[$(IFS=,; echo "${extras[*]}")]"
  fi

  log "pip install jtsmm ($([[ "${EDITABLE}" -eq 1 ]] && echo editable || echo regular))"
  if [[ "${EDITABLE}" -eq 1 ]]; then
    pip_install -e "${pkg}"
  else
    pip_install "${pkg}"
  fi

  if [[ "${DO_OPTICAL}" -eq 1 ]]; then
    # Not on PyPI; jtsmm imports it for optical tracking / CT pivot / IMU hand-eye.
    log "pip install optical ($([[ "${EDITABLE}" -eq 1 ]] && echo editable || echo regular))"
    if [[ "${EDITABLE}" -eq 1 ]]; then
      pip_install -e "${OPTICAL_SW}"
    else
      pip_install "${OPTICAL_SW}"
    fi
  fi

  install_analysis

  if [[ "${DO_NATIVE}" -eq 1 ]]; then
    log "native extension"
    require_cmd ninja
    "${NATIVE}/build.sh" --python "${PY}"
  fi

  if [[ "${SKIP_TESTS}" -eq 0 ]]; then
    if ! "${PY}" -c "import pytest" 2>/dev/null; then
      echo "ERROR: pytest is not installed (pass --dev to install dev extras, or --skip-tests)" >&2
      exit 1
    fi
    log "pytest"
    (cd "${SW}" && "${PY}" -m pytest tests/ -q)
  fi
}

install_analysis() {
  if [[ "${DO_ANALYSIS}" -ne 1 ]]; then
    return 0
  fi
  # experiments/ scripts (5.3.x, 5.4.x, data/) import biomech, which imports jtsmm
  # and jts (jaw_tracking); none of the three is on PyPI. biomech[stats] brings
  # spm1d/statsmodels/pingouin, biomech[plot] matplotlib.
  local mode=()
  [[ "${EDITABLE}" -eq 1 ]] && mode=(-e)
  log "pip install analysis stack ($([[ "${EDITABLE}" -eq 1 ]] && echo editable || echo regular)): jtsmm[analysis], jaw_tracking, biomech[plot,stats]"
  pip_install "${mode[@]}" "${SW}[analysis]"
  pip_install "${mode[@]}" "${JTS_SW}"
  pip_install "${mode[@]}" "${BIOMECH_SW}[plot,stats]"
  "${PY}" -c "import biomech.cohort, biomech.emg.pipeline, biomech.kinematics.cycles, jts, numpy; \
print('analysis stack ok, numpy', numpy.__version__)"
}

build_firmware() {
  if [[ "${DO_FIRMWARE}" -ne 1 ]]; then
    return 0
  fi
  if [[ -z "${PICO_SDK_PATH:-}" ]]; then
    echo "ERROR: PICO_SDK_PATH is not set (required for --firmware)" >&2
    exit 1
  fi
  require_cmd cmake
  log "firmware build"
  cmake -S "${FW}" -B "${FW}/build" -G Ninja
  cmake --build "${FW}/build" -j"$(nproc)"
  log "firmware artifacts: ${FW}/build/jtsmm.elf ${FW}/build/jtsmm.uf2"
}

flash_firmware() {
  if [[ "${DO_FLASH}" -ne 1 ]]; then
    return 0
  fi
  log "flash SWD"
  "${ROOT}/scripts/flash_swd.sh"
}

install_udev_rules() {
  if [[ "${DO_UDEV}" -ne 1 ]]; then
    return 0
  fi
  log "udev rules"
  "${ROOT}/scripts/install_udev.sh"
}

patch_yaml_data_dir() {
  # host.paths.data_dir is the key the host reads; keep the deprecated
  # host.storage.base_dir (older operator files) equal so validation stays quiet.
  # In-place text edit: a yaml dump would strip the template's comments.
  local dst="$1"
  local base="$2"
  "${PY}" "${ROOT}/scripts/set_config_value.py" "${dst}" host.paths.data_dir "${base}"
  "${PY}" "${ROOT}/scripts/set_config_value.py" "${dst}" host.storage.base_dir "${base}" --if-present
}

install_config() {
  if [[ "${DO_CONFIG}" -ne 1 ]]; then
    return 0
  fi
  normalize_config_dst
  if [[ -e "${CONFIG_DST}" && ! -f "${CONFIG_DST}" ]]; then
    echo "ERROR: --config-path must be a YAML file, not: ${CONFIG_DST}" >&2
    exit 1
  fi
  local src="${SW}/etc/jtsmm.yaml"
  log "operator config -> ${CONFIG_DST}"
  if [[ "$(id -u)" -ne 0 && ! -w "$(dirname "${CONFIG_DST}")" ]]; then
    sudo mkdir -p "$(dirname "${CONFIG_DST}")"
    sudo cp "${src}" "${CONFIG_DST}"
    sudo chown "$(id -un):" "${CONFIG_DST}"
    patch_yaml_data_dir "${CONFIG_DST}" "${DATA_DIR}"
    sudo chmod 644 "${CONFIG_DST}"
    sudo chown "${USER_NAME}:${USER_NAME}" "${CONFIG_DST}" 2>/dev/null || true
  else
    mkdir -p "$(dirname "${CONFIG_DST}")"
    cp "${src}" "${CONFIG_DST}"
    patch_yaml_data_dir "${CONFIG_DST}" "${DATA_DIR}"
    chmod 644 "${CONFIG_DST}"
  fi
}

pi_tune() {
  if [[ "${DO_PI_TUNE}" -ne 1 ]]; then
    return 0
  fi
  # shellcheck source=pi_tune.sh
  source "${ROOT}/scripts/pi_tune.sh"
  pi_tune_apply "${USER_NAME}" "${DATA_DIR}" "${PREFIX}" "${CONFIG_DST}" run_root "${SW}"
}

pi_network() {
  if [[ "${DO_PI_NETWORK}" -ne 1 ]]; then
    return 0
  fi
  log "Pi network"
  local -a env_args=()
  [[ -n "${JTSMM_WIFI_SSID:-}" ]] && env_args+=(JTSMM_WIFI_SSID="${JTSMM_WIFI_SSID}")
  [[ -n "${JTSMM_WIFI_PSK:-}" ]] && env_args+=(JTSMM_WIFI_PSK="${JTSMM_WIFI_PSK}")
  [[ -n "${JTSMM_WIFI_BAND:-}" ]] && env_args+=(JTSMM_WIFI_BAND="${JTSMM_WIFI_BAND}")
  [[ -n "${JTSMM_WIFI_CHANNEL:-}" ]] && env_args+=(JTSMM_WIFI_CHANNEL="${JTSMM_WIFI_CHANNEL}")
  if [[ ${#env_args[@]} -gt 0 ]]; then
    run_root env "${env_args[@]}" "${ROOT}/scripts/pi_network.sh" "${PI_NETWORK_ARGS[@]}"
  else
    run_root "${ROOT}/scripts/pi_network.sh" "${PI_NETWORK_ARGS[@]}"
  fi
}

write_env_hint() {
  cat <<EOF

Install complete.
  prefix:   ${PREFIX}
  venv:     ${VENV}
  data:     ${DATA_DIR}
  config:   ${CONFIG_DST} $([[ "${DO_CONFIG}" -eq 1 ]] && echo '(installed)' || echo '(template: software/jtsmm/etc/jtsmm.yaml)')

Activate and run:
  source ${VENV}/bin/activate
  export JTSMM_DATA_DIR=${DATA_DIR}
  export JTSMM_CONFIG=${CONFIG_DST}
  jtsmm doctor
  jtsmm record --duration 10

Capture tuning (optional):
  export JTSMM_SETTLE_MS=500      # USB drain after stream start (default 500)
  export JTSMM_STREAM_RESTART=1   # stop/start MCU streams after settle (default on)
EOF
  if [[ "${DO_ANALYSIS}" -eq 1 ]]; then
    cat <<EOF

Experiment scripts (any working directory; results in ~/experiments/<study>/results,
\$BIOMECH_RESULTS_DIR overrides; sessions from \$JTSMM_DATA_DIR):
  python ${ROOT}/experiments/5_4_1_emg_properties/eval_emg_properties.py --subject s01
EOF
  fi
}

main() {
  install_apt_deps
  install_host
  build_firmware
  flash_firmware
  install_udev_rules
  install_config
  pi_tune
  pi_network
  write_env_hint
}

main
