// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub (fix package B): hardware/i2c.h subset; fakes in tests/fix_B_fakes.c. */
#pragma once

#include "pico/types.h"
#include <stddef.h>

typedef struct i2c_inst i2c_inst_t;

extern i2c_inst_t *const fake_i2c0;
extern i2c_inst_t *const fake_i2c1;
#define i2c0 fake_i2c0
#define i2c1 fake_i2c1

uint i2c_init(i2c_inst_t *i2c, uint baudrate);
void i2c_deinit(i2c_inst_t *i2c);
int i2c_write_timeout_us(i2c_inst_t *i2c, uint8_t addr, const uint8_t *src, size_t len, bool nostop,
                         uint timeout_us);
int i2c_read_timeout_us(i2c_inst_t *i2c, uint8_t addr, uint8_t *dst, size_t len, bool nostop,
                        uint timeout_us);
