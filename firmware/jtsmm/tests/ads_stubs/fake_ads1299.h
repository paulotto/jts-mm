// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/*
 * A small ADS1299 on a fake SPI0 (tests/ads_stubs/fake_ads1299.c): opcodes
 * RESET / SDATAC / RDATAC / START / STOP / WAKEUP and RREG / WREG are decoded
 * from the bytes the REAL driver clocks out while PIN_ADS_CS is low. Like the
 * chip it powers up (and resets) in RDATAC mode, where RREG/WREG are ignored,
 * so a missing SDATAC shows up as registers left at their reset values.
 */

#include <stdbool.h>
#include <stdint.h>

#define FAKE_ADS_NREGS 0x18

/** Register file as the chip holds it. */
extern uint8_t fake_ads_regs[FAKE_ADS_NREGS];
/** WREG writes that landed / were dropped (RDATAC mode or CS high). */
extern int fake_ads_writes;
extern int fake_ads_dropped_writes;
/** Power-on state: reset values, RDATAC mode, CS high. */
void fake_ads_power_on(void);
