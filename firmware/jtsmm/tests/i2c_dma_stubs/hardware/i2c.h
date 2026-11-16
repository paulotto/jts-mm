// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub (test_i2c_dma): I2C block with the registers i2c_dma.c touches. */
#pragma once

#include "pico/types.h"
#include <stddef.h>

typedef struct {
    volatile uint32_t enable;
    volatile uint32_t tar;
    volatile uint32_t data_cmd;
    volatile uint32_t raw_intr_stat;
    volatile uint32_t clr_tx_abrt; /* reading it clears TX_ABRT on the chip */
} i2c_hw_t;

typedef struct i2c_inst {
    i2c_hw_t *hw;
} i2c_inst_t;

static inline uint i2c_get_dreq(i2c_inst_t *i2c, bool is_tx) {
    (void)i2c;
    return is_tx ? 0u : 1u;
}
