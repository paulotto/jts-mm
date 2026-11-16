// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/* Host-test stub for the subset of the TinyUSB vendor API used by host_link.c.
 * Implemented in tusb_stub.c as a controllable FIFO: tests set whether the
 * endpoint is mounted and how much space is free, and can read back every byte
 * written. Defaults (mounted, effectively unlimited space) keep tests that do
 * not care about USB on the real frame-building path. */

#include <stddef.h>
#include <stdint.h>

uint32_t tud_vendor_write_available(void);
uint32_t tud_vendor_write(const void *buffer, uint32_t bufsize);
uint32_t tud_vendor_write_flush(void);
int tud_vendor_mounted(void);

/* Test controls. */
extern int stub_tud_mounted;           /* default 1 */
extern uint32_t stub_tud_available;    /* free FIFO bytes; decremented by writes */
extern uint8_t stub_tud_written[8192]; /* first bytes written since reset */
extern uint32_t stub_tud_written_len;  /* total bytes written since reset */
void stub_tud_reset(uint32_t available);
