// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub (test_i2c_dma): RP2350 I2C register bits used by i2c_dma.c. */
#pragma once

#define I2C_IC_DATA_CMD_RESTART_BITS      0x00000400u
#define I2C_IC_DATA_CMD_STOP_BITS         0x00000200u
#define I2C_IC_DATA_CMD_CMD_BITS          0x00000100u
#define I2C_IC_RAW_INTR_STAT_TX_ABRT_BITS 0x00000040u
