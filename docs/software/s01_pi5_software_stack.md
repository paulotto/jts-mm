<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: CC-BY-4.0
-->

# s01 — Pi 5 Software Stack

This document specifies the operating system, system services, Python environment, and the few real-time settings the Pi 5 needs to acquire EMG, IMU, and cameras concurrently without dropping data.

## 1. Operating system

- **Raspberry Pi OS Bookworm 64-bit (Lite, no desktop)** is the baseline.
  - 64-bit is required for libcamera + picamera2 + NumPy on Pi 5.
  - The Lite edition (no desktop) keeps the system clean; we run the live viewer over the network from the operator's laptop.
- Kernel: stock Pi OS kernel (currently 6.6.x in Bookworm) — **not** PREEMPT-RT. We achieve adequate determinism with `SCHED_FIFO` for hot processes; PREEMPT-RT would help but is overkill here and complicates kernel updates.
- Filesystem: NVMe via PCIe HAT, ext4 with `noatime,commit=60` mount options to reduce write amplification.
- Boot device: NVMe (root + `/home`); the microSD is only used as the bootloader landing pad if the firmware needs to be reloaded.

## 2. System services

Installed and enabled:

| Service | Purpose | Notes |
|---------|---------|-------|
| `systemd-timesyncd` | NTP for absolute wall clock | Not for fusion; just for human-readable timestamps |
| `ssh` | Remote operator access | Mandatory; the system runs headless |
| `udev` | Stable `/dev/jtsmm` symlinks for the HAT USB | See §4.1 |
| `nginx` (optional) | Serve the live viewer over HTTPS | Pi-side only; web UI lives in `software/jtsmm/viewer_web/` |

Disabled / removed:

| Service | Why removed |
|---------|------------|
| Bluetooth | RF interference risk, unused |
| `wpa_supplicant` (Wi-Fi) | Use Ethernet only during recording sessions; Wi-Fi RF noise |
| `chrony` slewing during a session | We disable NTP slewing while recording is active to avoid backwards steps on `CLOCK_MONOTONIC` mid-session |

## 3. Python environment

- **Python 3.11+** (Bookworm ships 3.11; we lock to that).
- Plain `python3 -m venv --system-site-packages` at `~/jtsmm/venv` (so apt `picamera2` is visible); packages installed with pip by `scripts/build_install.sh`.

### 3.1 Core dependencies

Authoritative list: [`software/jtsmm/pyproject.toml`](../../software/jtsmm/pyproject.toml).

| Package | Purpose | Min version |
|---------|---------|-------------|
| `numpy`, `h5py` | numerics, HDF5 I/O | 1.24 / 3.8 |
| `pyserial`, `pyusb` | USB-CDC commands, vendor bulk | 3.5 / 1.2 |
| `pyyaml`, `jsonschema`, `click` | config, validation, CLI | 6.0 / 4.0 / 8.0 |
| extra `native` | `pybind11` for the C++/libusb/libhdf5 extension | 2.11 |
| extra `cameras` / `optical` | `picamera2` (apt), `opencv-python` | 0.3 / 4.8 |
| extra `viewer` | `fastapi`, `uvicorn`, `msgpack`, `Pillow` | — |

### 3.2 Optional dependencies

| Package | Purpose |
|---------|---------|
| `ros-jazzy-*` (Ubuntu Jazzy / Pi OS does not ship ROS) | If the lab wants ROS 2 integration, ROS 2 Jazzy on Ubuntu 24.04 is the supported target; install via the Ubuntu Pi image rather than Pi OS |
| `jupyterlab` | analysis notebooks |
| `pandas` | analysis convenience |
| `matplotlib` | offline plots |

### 3.2.1 Offline analysis stack (`--analysis`)

`scripts/install_pi.sh` passes `--analysis` to `build_install.sh` (skip with
`--no-analysis`). It installs `jtsmm[analysis]`, `software/jaw_tracking` (`jts`) and
`software/biomech[plot,stats]` (scipy, pandas, matplotlib, spm1d, statsmodels,
pingouin) so the evaluation scripts under `experiments/` run on the rig. Every pip
install then uses `~/jtsmm/pip-constraints.txt` (`numpy==<apt version>`): the venv
shares apt's numpy (1.24 on Bookworm, 2.2 on Trixie) with `python3-picamera2`/`simplejpeg`,
and a dependency that pulls in another numpy would break them (seen on a Trixie
rig: a fixed `numpy<2` pulled numpy 1.26 into the venv and broke apt's scipy). Use the
same file for manual installs
(`pip install -c ~/jtsmm/pip-constraints.txt …`).

The scripts run from any directory with the venv's Python, e.g.
`python <repo checkout>/experiments/5_4_1_emg_properties/eval_emg_properties.py`.
Sessions come from `--data-dir`, else `$JTSMM_DATA_DIR`, else `~/jtsmm_data`
(`biomech.paths.resolve_data_dir`); results go to `--out-dir`, else
`$BIOMECH_RESULTS_DIR/<study>/results`, else `~/experiments/<study>/results`.

### 3.3 Native libraries

Some Python packages need apt-installed system libraries; the install script (`scripts/install_pi.sh`) handles:

```
sudo apt install -y \
  python3 python3-venv python3-dev \
  build-essential cmake ninja-build pkg-config \
  libusb-1.0-0-dev libhdf5-dev libcap-dev \
  libcamera-dev libcamera-tools
# with --cameras (default in install_pi.sh): python3-libcamera python3-picamera2
```

## 4. USB device naming and udev

### 4.1 Stable device path for the HAT

The HAT enumerates as a composite USB device (CDC + vendor bulk; Vendor ID = `0x2E8A`, Product ID = `0x0004`). The udev rule in [`software/jtsmm/etc/udev/99-jtsmm.rules`](../../software/jtsmm/etc/udev/99-jtsmm.rules):

```
SUBSYSTEM=="tty", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="0004", SYMLINK+="jtsmm_hat"
SUBSYSTEM=="usb", ATTRS{idVendor}=="2e8a", ATTRS{idProduct}=="0004", MODE="0666", GROUP="plugdev", TAG+="uaccess"
```

The second (usb-subsystem) rule gives libusb access to the vendor bulk interface without root.

This makes the HAT always reachable as `/dev/jtsmm_hat`, regardless of enumeration order.

### 4.2 Camera permissions

`libcamera` requires the user to be in the `video` group (default on Pi OS for the `pi` user). For non-default users (e.g. a dedicated `jtsmm` operator account), the install script adds them.

## 5. Pi 5 power and CPU scaling

Three independent layers affect how much power the Pi 5 can deliver to the stack (Pi + M.2 HAT+ + custom HAT + USB) and how the host CPU behaves under load.

### 5.1 PSU and USB current budget

Pi 5 negotiates **USB PD** and prefers **5 V @ 5 A** (official **27 W** USB-C PSU). That unlocks full SoC headroom and **~1.6 A** on downstream USB. If 5 V/5 A is not detected at boot, USB peripherals are limited to **600 mA** and USB boot may be disabled — a problem with NVMe HAT+, the HAT USB link, and bench SSDs even when the charger label says “60 W” (many only offer 5 V/3 A).

| Goal | Action |
|------|--------|
| Recommended | Official Raspberry Pi **27 W** (5.1 V / 5 A) PSU |
| Assume 5 A without PD | `sudo rpi-eeprom-config -e` → add `PSU_MAX_CURRENT=5000` |
| Raise USB limit in OS | `/boot/firmware/config.txt`: `usb_max_current_enable=1` (only if wiring and supply can handle it) |
| Health check | `vcgencmd get_throttled` → **`0x0`** is healthy |

See also [`op03`](../operations/op03_troubleshooting.md) §6 (throttling).

### 5.2 CPU frequency governor

Pi 5 (BCM2712, quad Cortex-A76) uses **DVFS** with coarse steps (~600 / 750 / 1000 / 1500 / up to **2.4 GHz**). Default governor on Pi OS is **`ondemand`** (scales with load). For capture sessions, `ondemand` is usually sufficient; use **`performance`** only if you need minimum scheduling jitter and accept higher idle draw.

```bash
# Current governor
cat /sys/devices/system/cpu/cpufreq/policy0/scaling_governor

# Max performance until reboot
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Minimum power until reboot
echo powersave | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

Advanced tuning (`arm_freq`, `arm_freq_min`, `force_turbo`) lives in `/boot/firmware/config.txt`; see [Raspberry Pi frequency management](https://www.raspberrypi.com/documentation/computers/raspberry-pi.html#frequency-management-and-thermal-control). `over_voltage` is largely deprecated on Pi 5.

### 5.3 Thermal and fan

With the official Active Cooler or case fan (JST-SH **FAN_PWM** header), tune curves in `/boot/firmware/config.txt`:

```ini
dtparam=fan_temp0=55000
dtparam=fan_temp0_hyst=5000
dtparam=fan_temp0_speed=120
```

(`fan_temp1` … `fan_temp3` for higher tiers.) Inspect fan state with `pinctrl get FAN_PWM`.

## 6. Real-time scheduling

Not implemented: `jtsmm record` runs as one process at default priority (no `chrt` / `taskset` pinning, no separate capture processes). The native USB capture runs in C++ threads inside that process.

### 6.1 Why no PREEMPT-RT

We considered PREEMPT-RT and concluded:

- Our timing budget is generous (>1 ms tolerable for the Pi side; the MCU handles the sub-ms work).
- PREEMPT-RT on Pi 5 is currently experimental; running it on production hardware adds support burden.
- A future v2 may add PREEMPT-RT if a real-time control loop demands it.

## 7. Project source layout

**Repository (development):** [`software/jtsmm/`](../../software/jtsmm/) — install with `pip install -e software/jtsmm` from the repo root.

**On-Pi install target** (`scripts/install_pi.sh`):

```
~/jtsmm/venv/                     # python3 -m venv --system-site-packages
~/jtsmm/pip-constraints.txt       # numpy==<apt version> (shared with picamera2)
<repo checkout>/software/jtsmm    # installed editable (+ software/optical)
<repo checkout>/software/{jaw_tracking,biomech}   # editable, --analysis (default)
~/experiments/<study>/results/    # experiments/ script outputs ($BIOMECH_RESULTS_DIR)
~/jtsmm.yaml                      # operator config (jtsmm config init)
~/jtsmm_data/                     # session HDF5 (host.paths.data_dir)
```

`jtsmm/integrity/` (post-session checks) and `jtsmm/doctor/` (pre-flight) are implemented.

Config contract: [`schema/v1/`](../../schema/v1/jtsmm.schema.json).

## 8. Configuration file

The operator-facing knob is **`~/jtsmm.yaml`** (`jtsmm config init`) — schema **v1** with `mcu.*` (sent to firmware) and `host.*` / `session.*` (Pi-only). Template: [`software/jtsmm/etc/jtsmm.yaml`](../../software/jtsmm/etc/jtsmm.yaml). Full field reference: [`s08_runtime_configuration.md`](s08_runtime_configuration.md).

Excerpt (the real file lists every key explicitly; `profile:` values are labels):

```yaml
schema_version: 1

hat:
  serial_port: /dev/jtsmm_hat
  firmware_version: ">=1.0.0,<2.0.0"

mcu:
  platform:
    profile: default              # SPI 4 MHz, I2C 400 kHz, ring buffers
  emg:
    profile: acquisition          # template: 4000 SPS, gain 6, DC lead-off on
  imu_mta:
    profile: kinematics_accel
    fusion: { primary: game_rotation_vector, rate_hz: 390 }
  imu_cra:
    profile: kinematics_accel
    fusion: { primary: game_rotation_vector, rate_hz: 390 }
  trigger:
    profile: imx296_dual_csi      # preset 30 Hz / 12 ms
    rate_hz: 40                   # template (Oct 2026 pilot), pulse.width_us 12000

host:
  cameras:
    track: B
    marker_detector: apriltag
    list:
      - { id: cam0, device: csi0, resolution: [1456, 1088],
          exposure_us: 12000, gain: 1.0, ae_enable: false, awb_enable: false,
          awb_gains: [2.3, 2.3],
          trigger: { source: mcu, expect_rate_hz: 40 } }
      - { id: cam1, device: csi1, resolution: [1456, 1088],
          exposure_us: 12000, gain: 1.0, ae_enable: false, awb_enable: false,
          awb_gains: [2.3, 2.3],
          trigger: { source: mcu, expect_rate_hz: 40 } }
  paths:
    data_dir: ~/jtsmm_data

session:
  mode: acquisition
  duration_s: null
```

**CLI:** `jtsmm record` loads YAML → strict validation → `config apply` JSON to MCU → capture (profile fragments are not merged on load; see s08). Override keys via `--config` path; `jtsmm doctor` runs validation + serial probe.

**Resolution pipeline:** YAML → [`jtsmm.config.load`](../../software/jtsmm/jtsmm/config/__init__.py) → schema + cross-validation ([`strict.py`](../../software/jtsmm/jtsmm/config/strict.py), [`validate.py`](../../software/jtsmm/jtsmm/config/validate.py)) → `config_hash` in HDF5 `/meta`.

## 9. Install procedure

Documented in detail in `scripts/install_pi.sh`; the high-level steps:

1. Flash Pi OS Bookworm 64-bit Lite to the microSD; first-boot config sets up SSH.
2. Boot, log in via SSH, clone the repo and run `scripts/install_pi.sh`, which:
   - Updates apt, installs the native dependencies in §3.3 and `python3-picamera2` / `python3-libcamera` (no pip wheels exist for libcamera).
   - Creates `~/jtsmm/venv` with `--system-site-packages` (so the apt picamera2 is visible) and installs `software/jtsmm` and `software/optical` **editable** — the rig runs from its checkout.
   - Installs the offline analysis stack (`jtsmm[analysis]`, `software/jaw_tracking`, `software/biomech[plot,stats]`, §3.2.1) under a `numpy==<apt version>` constraint; `--no-analysis` skips it.
   - Builds the native capture extension, copies the udev rules, adds the user to `dialout`/`plugdev`/`video`.
   - Writes `~/jtsmm.yaml` from the template with `host.paths.data_dir` = `~/jtsmm_data` (`--data-dir` to change).
3. Reboot. Plug in the HAT. `/dev/jtsmm_hat` should appear.
4. Run `jtsmm doctor` then `jtsmm record --duration 10` to confirm capture (use `--mock` without HAT hardware).

## 10. Update procedure

- Source code: `git pull` in the checkout (editable install); rebuild the native extension (`software/jtsmm/native/build.sh`) if it changed.
- **Firmware (build):** [`firmware/jtsmm/README.md`](../../firmware/jtsmm/README.md) — `firmware/jtsmm/build/`.
- **Firmware (flash):** `./scripts/flash_swd.sh` with a **Raspberry Pi Debug Probe** over **3-wire SWD** ([`h08`](../hardware/h08_rp2350_flashing.md) §6), or hold BOOTSEL and copy the `.uf2` to the RP2350 drive.
- OS: `sudo apt update && sudo apt upgrade` — always **before** a recording session, never during.

## 11. Health monitoring

`jtsmm doctor` runs live pre-flight checks (config, MCU serial probe, firmware version, IMX296 trigger mode, viewer port) and is the standard check before each session. `jtsmm record` prints `mcu_stats` (from `REC_STATS`) and writes `integrity.json` after each take.
