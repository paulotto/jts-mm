// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub for the trigger PIO driver entry points used by trigger_config.c. */
#include "jtsmm/drivers/trigger_pio.h"

uint32_t stub_trigger_pio_clock_hz;
uint8_t stub_trigger_pio_irq_level = 0xFFu;

bool trigger_pio_init(uint32_t clock_hz) {
    stub_trigger_pio_clock_hz = clock_hz;
    return true;
}

void trigger_pio_set_irq_priority(uint8_t level) {
    stub_trigger_pio_irq_level = level;
}

/* 1 = imx296_fast, 0 = active_high, -1 = never called. */
int stub_trigger_pio_idle_fast = -1;

void trigger_pio_set_idle_polarity(bool imx296_fast) {
    stub_trigger_pio_idle_fast = imx296_fast ? 1 : 0;
}
