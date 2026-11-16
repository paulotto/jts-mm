// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub (test_i2c_dma): fake clock for i2c_dma.c. */
#pragma once

#include "pico/types.h"

extern uint64_t fake_now_us;

static inline void tight_loop_contents(void) {
    fake_now_us += 10u;
}
static inline absolute_time_t get_absolute_time(void) {
    return fake_now_us;
}
static inline uint64_t to_us_since_boot(absolute_time_t t) {
    return t;
}
static inline uint32_t time_us_32(void) {
    return (uint32_t)fake_now_us;
}
