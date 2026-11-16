// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Host-test stub (fix package B): the pico/stdlib.h subset used by the IMU
 * driver/domain code. Time runs on a fake clock implemented in
 * tests/fix_B_fakes.c; sleeping advances it, so bounded waits terminate.
 */
#pragma once

#include "pico/types.h"
#include "hardware/gpio.h"
#include <stdio.h>

#define PICO_ERROR_TIMEOUT (-1)
#define PICO_ERROR_GENERIC (-2)

static inline void tight_loop_contents(void) {}
static inline void putchar_raw(int c) {
    (void)c;
}

void sleep_ms(uint32_t ms);
void sleep_us(uint64_t us);
void busy_wait_us_32(uint32_t us);
absolute_time_t get_absolute_time(void);
absolute_time_t make_timeout_time_ms(uint32_t ms);
bool time_reached(absolute_time_t t);
int64_t absolute_time_diff_us(absolute_time_t from, absolute_time_t to);
uint32_t time_us_32(void);
uint64_t time_us_64(void);
int getchar_timeout_us(uint32_t timeout_us); /* CDC input (test_config_apply_drivers) */
