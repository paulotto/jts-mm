<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# Host-compiled firmware unit tests. Build:

```bash
cmake -S firmware/jtsmm/tests -B firmware/jtsmm/tests/build
cmake --build firmware/jtsmm/tests/build
ctest --test-dir firmware/jtsmm/tests/build --output-on-failure
```

Everything builds with `-DNDEBUG`: tests use `CHECK` (`check.h`, or the fix-A/B
harnesses), never `assert()`, so no test can pass because an assert compiled out.

| Test | Covers |
|------|--------|
| `test_host_link_crc` | CRC-16/CCITT-FALSE golden vectors |
| `test_host_link_tx` | TX policy against a stubbed vendor FIFO (`stubs/tusb_stub.c`): frame golden, records stay queued when the FIFO is full, EMG reserve, per-stream drop counters |
| `test_ringbuf` | capacity edges, FIFO order across wraps, peek/consume |
| `test_config_json` | golden payload parse, lossless dump → parse of every field, strict values, fixed-value keys, report lists |
| `test_config_json_keys` | every leaf of the golden payload is dumped back and either takes effect or is refused |
| `test_config_apply` | apply state machine with recording stubs (`stubs/apply_stubs.c`): a bad config leaves the running one untouched |
| `test_config_apply_drivers` | config plumbing: golden payload + one mutation, sent as `config apply` through the real CDC parser → the driver call that applies it (recording stubs `stubs/rec_drivers.c`); stream commands and capture start honour the enables; two-apply sequences |
| `test_ads1299_registers` | register level: payload → `config_json` → `emg_config_apply` → the real `ads1299.c` on a fake SPI0 with an ADS1299 register model (`ads_stubs/fake_ads1299.c`); CONFIG1-4, LOFF, CHnSET, BIAS_SENSP/N, LOFF_SENSP/N per setting, e.g. `--bias-channels 1,2` → BIAS_SENSP/N 0x03 |
| `test_trigger_timing` / `test_trigger_config` | PIO templates, start validation, K4 edge counting, defaults and profiles |
| `test_fix_A_*` / `test_fix_B_*` | ADS1299 / EMG and IMU / BNO085 packages |
| `test_imu_timing` / `test_imu_timing_off` | IMU sample timestamps: INT-edge latch state machine, SH-2 base delta / delay / rebase decoding, per-report record times, the sensor-core poll path against fakes; `_off` is built with both `IMU_TS_USE_*` switches 0 (poll time everywhere) |

`test_config_json*` and `test_config_apply*` load the golden MCU apply JSON generated from
[`software/jtsmm/etc/jtsmm.yaml`](../../../software/jtsmm/etc/jtsmm.yaml) via
`build_mcu_apply_payload()` ([`schema/v1/golden/mcu_apply_template.json`](../../../schema/v1/golden/mcu_apply_template.json)).
Regenerate it after template changes:

```bash
python scripts/sync_config_template.py --sync-mcu-golden
```

Run via `scripts/hil/run_bench.sh`.
