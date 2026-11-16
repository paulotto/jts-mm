// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file usb_service.c
 * @brief TinyUSB init/poll/sleep helpers.
 */

#include "jtsmm/runtime/usb_service.h"

#include "pico/stdio/driver.h"
#include "pico/stdlib.h"
#include "tusb.h"

/*
 * The firmware owns the USB stack (composite CDC + Vendor bulk), so we provide
 * our own stdio driver that routes printf()/getchar() to the CDC interface.
 * Binary records go out the vendor bulk endpoint (see host_link.c); NO binary
 * is ever written to CDC and NO text is written to the bulk endpoint.
 */

static void cdc_out_chars(const char *buf, int len) {
    if (!tud_cdc_connected()) {
        return; /* no host terminal -> drop text rather than stalling */
    }
    int done = 0;
    absolute_time_t deadline = make_timeout_time_ms(100);
    while (done < len) {
        uint32_t avail = tud_cdc_write_available();
        if (avail == 0) {
            tud_cdc_write_flush();
            tud_task();
            if (time_reached(deadline)) {
                break;
            }
            continue;
        }
        uint32_t chunk = (uint32_t)(len - done);
        if (chunk > avail) {
            chunk = avail;
        }
        done += (int)tud_cdc_write(buf + done, chunk);
    }
    tud_cdc_write_flush();
}

/** @brief Internal helper: cdc_out_flush. */
static void cdc_out_flush(void) {
    tud_cdc_write_flush();
}

/** @brief Internal helper: cdc_in_chars. */
static int cdc_in_chars(char *buf, int len) {
    if (!tud_cdc_available()) {
        return PICO_ERROR_NO_DATA;
    }
    uint32_t n = tud_cdc_read(buf, (uint32_t)len);
    return n ? (int)n : PICO_ERROR_NO_DATA;
}

static stdio_driver_t cdc_stdio_driver = {
    .out_chars = cdc_out_chars,
    .out_flush = cdc_out_flush,
    .in_chars = cdc_in_chars,
    .crlf_enabled = false,
};

void usb_service_init(void) {
    tusb_init();
    stdio_set_driver_enabled(&cdc_stdio_driver, true);
}

void usb_service_poll(void) {
    tud_task();
    tight_loop_contents();
}

void usb_service_sleep_ms(uint32_t ms) {
    absolute_time_t deadline = make_timeout_time_ms(ms);
    while (!time_reached(deadline)) {
        usb_service_poll();
        sleep_ms(1);
    }
}
