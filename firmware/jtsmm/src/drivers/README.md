<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# drivers/ — thin hardware access (no profile logic)

Refactored from the bring-up firmware (the SPI/I2C DMA helpers live in `platform/`):

- `ads1299.c` (+ `include/jtsmm/drivers/ads1299_regs.h`)
- `bno085.c`, `imu_bus.c`
- `trigger_pio.c`, `trigger_timing.c` (trigger template arithmetic)
- `led_flash_pio.c` (sync LED; timer-driven despite the name)
- `panel.c`
