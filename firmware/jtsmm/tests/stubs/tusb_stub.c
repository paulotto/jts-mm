// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#include "tusb.h"

#include <string.h>

int stub_tud_mounted = 1;
uint32_t stub_tud_available = 0xFFFFFFFFu;
uint8_t stub_tud_written[8192];
uint32_t stub_tud_written_len;

void stub_tud_reset(uint32_t available) {
    stub_tud_mounted = 1;
    stub_tud_available = available;
    stub_tud_written_len = 0;
}

uint32_t tud_vendor_write_available(void) {
    return stub_tud_available;
}

uint32_t tud_vendor_write(const void *buffer, uint32_t bufsize) {
    if (bufsize > stub_tud_available) {
        bufsize = stub_tud_available; /* a real FIFO takes what fits */
    }
    for (uint32_t i = 0; i < bufsize; i++) {
        uint32_t at = stub_tud_written_len + i;
        if (at < sizeof(stub_tud_written)) {
            stub_tud_written[at] = ((const uint8_t *)buffer)[i];
        }
    }
    stub_tud_written_len += bufsize;
    stub_tud_available -= bufsize;
    return bufsize;
}

uint32_t tud_vendor_write_flush(void) {
    return 0;
}

int tud_vendor_mounted(void) {
    return stub_tud_mounted;
}
