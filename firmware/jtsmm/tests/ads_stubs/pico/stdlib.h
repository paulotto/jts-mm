// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub: pico/stdlib.h subset for the REAL ads1299.c. Time runs on a
 * fake clock (tests/ads_stubs/fake_ads1299.c); sleeping advances it. */
#pragma once

#include "pico/types.h"
#include "hardware/gpio.h"
#include <stdio.h>

#define PICO_ERROR_TIMEOUT (-1)

static inline void tight_loop_contents(void) {}

void sleep_ms(uint32_t ms);
void sleep_us(uint64_t us);
void busy_wait_us(uint64_t us);
void busy_wait_us_32(uint32_t us);
absolute_time_t get_absolute_time(void);
absolute_time_t make_timeout_time_ms(uint32_t ms);
absolute_time_t make_timeout_time_us(uint64_t us);
absolute_time_t delayed_by_us(absolute_time_t t, uint64_t us);
bool time_reached(absolute_time_t t);
int64_t absolute_time_diff_us(absolute_time_t from, absolute_time_t to);
uint32_t time_us_32(void);
uint64_t time_us_64(void);
