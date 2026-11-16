<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# app/ — boot, dual-core orchestration, fault handling, selftest

- `main.c` — entry point; runs on physical core 0 (Pico SDK boot core), initializes USB + fault handling, then hands off to the comms core.
- `sensor_core.c` — sole sensor producer. Launched on **physical core 1** via `multicore_launch_core1()`. Owns ADS/EMG SPI and IMU I2C; only writer to the EMG/IMU rings.
- `comms_core.c` — sole consumer. Runs on **physical core 0** (`comms_core_main`). Drains rings to USB vendor bulk, handles CDC commands/config, and runs 1 Hz housekeeping.
- `fault.c` — HardFault SOS on LED_FAULT, reset reason (`info` → `last reset:`), optional hardware watchdog (`-DJTSMM_WATCHDOG_MS=4000`; default 0 = off).
- `selftest.c` — the `selftest` bench command.

The filenames follow the *logical role*, not the Pico physical core number; see the mapping table in [`docs/hardware/h05_sensor_pipeline_architecture.md`](../../../../docs/hardware/h05_sensor_pipeline_architecture.md) §3.
