<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# jtsmm — production RP2350 firmware

Dual-core polling/DMA firmware for the jtsmm_hat: EMG @ **4 kHz** default (`acquisition`), dual BNO085 IMU, PIO camera trigger, `0x55 0xAA` framed records over a **USB vendor bulk** endpoint (binary) with **USB CDC** for text/commands.

**Host package:** [`software/jtsmm/`](../../software/jtsmm/)  
**Config contract:** [`schema/v1/`](../../schema/v1/jtsmm.schema.json)  
**Firmware spec:** [`docs/hardware/h05_microcontroller_firmware.md`](../../docs/hardware/h05_microcontroller_firmware.md)  
**Pipeline architecture (start here):** [`docs/hardware/h05_sensor_pipeline_architecture.md`](../../docs/hardware/h05_sensor_pipeline_architecture.md)  
**C/C++ layout + API comments:** [`docs/software/s09_c_cpp_api_comments.md`](../../docs/software/s09_c_cpp_api_comments.md) (`doxygen Doxyfile` → `doxygen/html/`)

## Status

| Feature | Status |
|---------|--------|
| Dual-core split: **sensor core** (`sensor_core.c`, physical core 1) sole producer; **comms core** (`comms_core.c`, physical core 0) sole USB consumer | implemented |
| PIO `trigger_gen` + `TRIGGER_EVENT` | implemented |
| Sync LED `flash on/off/once` + `LED_FLASH_EVENT` (GPIO14, timer-driven; `led_flash.pio` unused) | implemented |
| `config apply` JSON | full parse via cJSON (`config_json.c`) |
| `config dump` | active MCU config JSON snapshot |
| `config save` / `config load` (flash) | implemented; save/erase refused while streaming, trigger running or apply busy |
| DP IMU (third slot, I2C mux channel 1) | not implemented — two IMU slots (MTA, CRA); the driver selects the CRA channel of the on-HAT I2C mux |

## Version

Semantic versioning (`include/jtsmm/version.h`; `info` answers `<version>-prod`, stored per take
as `/meta` `firmware_version`; the host accepts `hat.firmware_version` `>=1.0.0,<2.0.0`).

| Version | Date | Changes |
|---|---|---|
| 1.1.0 | 2026-11-04 | BNO085 dynamic calibration as a per-sensor config switch (`mcu.imu_*.dynamic_calibration`, default on = the behaviour of 1.0.0), applied on `config apply` and after a device reset, parsed and dumped; the first build flashed with it (the build on the HAT before was from 2026-10-06). IMU watchdog: a full ring with rising drops (the host does not read the data endpoint, e.g. streams started from a serial terminal) is reported once as "ring full" and no longer triggers recoveries that reset healthy IMUs |
| 1.0.0 | 2026-07-17 | first production firmware (dual-core split, PIO trigger, `config apply` / `dump` / `save` / `load`, ADS1299 and dual BNO085 streaming); later changes up to 2026-11-04 (IMU and trigger defaults, fix A / B) kept the version string |

## Layer map

Headers live under `include/jtsmm/<layer>/`; implementations under `src/<layer>/`. TinyUSB’s `tusb_config.h` is at `include/tusb_config.h`. See [`docs/software/s09_c_cpp_api_comments.md`](../../docs/software/s09_c_cpp_api_comments.md).

| Layer | Headers | Sources | Responsibility |
|-------|---------|---------|----------------|
| app | `include/jtsmm/app/` | `src/app/` | Boot, dual-core launch, `fault` (SOS, reset reason, optional watchdog), `selftest` |
| runtime | `include/jtsmm/runtime/` | `src/runtime/` | timebase, ringbuf, host_link, config_apply, config_json, config_flash, capture_session, cmd_parser, usb_service |
| platform | `include/jtsmm/platform/` | `src/platform/` | platform_config, **spi_dma**, **i2c_dma**, dma_irq, gpio_irq, `usb/` |
| drivers | `include/jtsmm/drivers/` | `src/drivers/` | ads1299 (+ `ads1299_regs.h`), bno085, imu_bus, trigger_pio, trigger_timing, led_flash_pio, panel |
| domain | `include/jtsmm/domain/` | `src/domain/` | emg_config (+ `emg_config_check.h`), imu_config, imu_timing, trigger_config, bno085_reports, config_defaults |
| pio | — | `pio/` | trigger_gen, led_flash, ws2812 |

Apply order: `platform` → `emg` → `imu_mta` → `imu_cra` → `trigger`.

## Build

Prerequisites: ARM GCC 12+, CMake, pico-sdk with submodules ([`h05` §7.1](../../docs/hardware/h05_microcontroller_firmware.md)).

```bash
export PICO_SDK_PATH=~/pico/pico-sdk

cd firmware/jtsmm
rm -rf build && mkdir build && cd build
cmake .. -DPICO_PLATFORM=rp2350
cmake --build . -j"$(nproc)"
```

**Outputs:**

| File | Use |
|------|-----|
| `build/jtsmm.uf2` | BOOTSEL drag-and-drop |
| `build/jtsmm.elf` | SWD flash / debug |

`PICO_BOARD=pico2` is set in `CMakeLists.txt`; require **`PICO_PLATFORM=rp2350`**.

## Flash

Power the HAT (**5 V**). Flash wiring: [`docs/hardware/h08_rp2350_flashing.md`](../../docs/hardware/h08_rp2350_flashing.md).

| Method | When |
|--------|------|
| **UF2** | Hold BOOTSEL, plug USB-C, copy `jtsmm.uf2` to RP2350 volume |
| **SWD** | Debug Probe — `./scripts/flash_swd.sh` (repo root); needs the RP2350-aware OpenOCD ([`h08`](../../docs/hardware/h08_rp2350_flashing.md) §6.2) |

## Bench verification (USB serial 115200)

Connect HAT USB (`/dev/jtsmm_hat` or `/dev/ttyACM0`):

```text
info
selftest
platform_status
ads_status
trig_status
```

### Minimal streaming test

After `selftest` passes:

```text
ads_start
imu_stream mta on
imu_stream cra on
trig_start                  # or: trig_start 30 / trig_start 30 12000
```

Trigger-only (no EMG/IMU): host `jtsmm trigger [--rate 30] [--width-us 12000]` / `--stop` / `--status`, or raw CDC `trig_start` / `trig_stop` / `trig_status`.

Observe framed `0x55 0xAA` records on the host with `software/jtsmm` protocol parser or:

```bash
cd software/jtsmm && pip install -e .
jtsmm record --duration 5 --port /dev/jtsmm_hat
```

Stop (USER LED returns green, CAPTURE LED off):

```text
trig_stop
imu_stream cra off
imu_stream mta off
ads_stop
```

Or from host: `jtsmm` session teardown / `scripts/hil/run_bench_hw.sh` H8 step.

**Diagnostics:** `ads_drdy_watch` (after `ads_stop`) — expect ~2000 DRDY edges per 500 ms at the default 4 kHz. `platform_status` — check ring fill and drop counters (`tx_drops` reset on each `ads_start` when EMG was idle). `emg_guard` — EMG framing-guard state and counters. `imu_int_watch [ms]` (default 1000, max 10000; no IMU needed, not while any stream runs) — counts falling edges on the IMU INT pins GPIO16/17 through the sensor core's edge IRQ (`irq_edges`) and by polling (`poll_falls`), internal pull-ups on; tap INT to GND with a wire: `irq_edges ≥ poll_falls > 0`. `imu_status` `ts:` lines — IMU timestamp source and statistics ([`h04`](../../docs/hardware/h04_imu_subsystem.md) §6.2).

`ads_status`, `ads_regs`, `imu_status`, `ads_drdy_watch` (also while an IMU streams) and `selftest` (also while an IMU streams or the trigger runs) drive SPI0/I²C/PIO, which the **sensor core** owns while streaming. They refuse with `<cmd> REFUSED: …` unless you append `force`; running them mid-stream slips the ADS1299 RDATA byte stream and corrupts EMG frames (`frame_misalign`). Run `ads_stop` first. See [`docs/hardware/h05_microcontroller_firmware.md`](../../docs/hardware/h05_microcontroller_firmware.md) §3.3.

While a `config apply` is running, `ads_start`, `trig_start`, `imu_stream <slot> on`, `imu_savecal`, `imu_int_watch` and `selftest` answer `<cmd> BUSY` (NAK) instead of starting on half-reprogrammed hardware; stop commands are always accepted.

### Config apply (production path)

From the Pi/host with resolved YAML:

```bash
jtsmm config apply          # uses ~/jtsmm.yaml; --config PATH to override
```

Or manually (minimal JSON):

```text
config apply {"schema_version":1,"platform":{"profile":"default"},"emg":{"profile":"acquisition"},"imu_mta":{"profile":"kinematics_accel"},"imu_cra":{"profile":"kinematics_accel"},"trigger":{"profile":"imx296_dual_csi"}}
```

### Panel / debug

```text
led rgb 0 32 0          # green LED_USER
# USER2 (GPIO11): LED disco ~15 s when idle (not streaming)
trig_start              # rate/width from the applied config (40 Hz / 12 ms with etc/jtsmm.yaml;
                        #   30 Hz / 12 ms with the bare imx296_dual_csi profile or the boot preset)
# or: trig_start 40 / trig_start 40 12000  (optional overrides; polarity from config)
# host: jtsmm trigger --rate 40   (trigger-only; no EMG/IMU / no config apply)
trig_status             # running, trigger_id, stamp mode/fallback, pin_period_us, pulses vs events
trig_stamp pio|timer    # TRIGGER_EVENT timing: PIO IRQ at the pulse edge (default) or legacy timer
trig_comp on|off        # PIO cycle-overhead compensation (default on); off = pre-2026-09 pin period
pps 1                   # replies PPS_REPLY{1} on the bulk stream (host clock handshake)
```

## IMX296 trigger checklist

Before first camera capture ([`h06` §3.5](../../docs/hardware/h06_synchronization.md)):

1. Scope **`TP_TRIG`** — confirm rate and polarity match IMX296 fast trigger.
2. Pi: `echo 1 | sudo tee /sys/module/imx296/parameters/trigger_mode`
3. Match `mcu.trigger.rate_hz` and `host.cameras.list[].trigger.expect_rate_hz` in YAML.

## Tests

Host-compiled unit tests (20 ctest targets): see [`tests/README.md`](tests/README.md).

HIL script: [`scripts/hil/run_bench.sh`](../../scripts/hil/run_bench.sh) (+ `run_bench_hw.sh` for the J_USB1 hardware matrix: dual IMU, strict integrity, stream teardown).

### Hardware bench matrix (J_USB1 + SWD)

Wiring: **J_USB1** = all USB-CDC serial; **Debug Probe → J_SWD1** = flash only (probe UART unused).

```bash
export JTSMM_HW_REQUIRED=1
export JTSMM_DATA_DIR=/tmp/jtsmm_bench
./scripts/hil/run_bench_hw.sh
```

Expect on populated HAT (MTA+CRA, no electrodes): `record_summary` shows EMG_BLOCK≥1900, IMU_MTA_QUAT≥10, IMU_CRA_QUAT≥10, TRIGGER_EVENT≈400 (40 Hz template × 10 s). With the parallel-streaming architecture (strict sensor/comms core split, async SPI-DMA EMG, I2C-DMA IMU with wedge watchdog + `imu_slot_recover`, dedicated USB vendor bulk endpoint) a 10 s capture of **2 kHz EMG + dual ~390 Hz IMU** runs `integrity --strict` clean with the native host backend: ~19k EMG blocks (~1900 Hz, `missed_drdy`≈0.2%) and ~3.8k quaternions per IMU slot. At 1 kHz override, expect ~9.5k blocks with `missed_drdy=0`. Note the BNO085 game-rotation-vector tops out at its native ~398 Hz, so the template declares 390 Hz to keep the 0.5% IMU drop budget robust. H8 teardown: green USER LED, CAPTURE off.

**Note:** `selftest` must not call `imu_bus_hw_reset()` or `imu_bus_init()` after `config apply` — that clears BNO085 SHTP state.
