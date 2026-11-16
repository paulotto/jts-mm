<!--
This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
SPDX-FileCopyrightText: 2026 Paul-Otto Müller
SPDX-License-Identifier: MIT
-->

# platform/ — RP2350 system config (clocks, SPI/I2C/USB/PIO, IRQ, buffers)

`platform_config.c` (applied before the domain blocks), `spi_dma.c` (ADS1299 SPI DMA), `i2c_dma.c` (BNO085 I2C/SHTP reads), `dma_irq.c` / `gpio_irq.c` (shared IRQ dispatch), `usb/usb_descriptors.c` (CDC + vendor bulk).
