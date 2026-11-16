// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file timebase.c
 * @brief Master timebase wrappers around time_us_64.
 */

#include "jtsmm/runtime/timebase.h"

#include "hardware/timer.h"
#include "pico/stdlib.h"

void timebase_init(void) {
    // RP2350 64-bit microsecond timer is always available via timer_hw.
}

uint64_t t_master_us(void) {
    return time_us_64();
}
