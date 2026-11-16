// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file fault.c
 * @brief Fault LED / SOS handler, hard-fault handler, optional hardware watchdog.
 */

#include "jtsmm/app/fault.h"

#include "jtsmm/drivers/panel.h"
#include "jtsmm/pins.h"
#include "jtsmm/runtime/host_link.h"

#include "hardware/gpio.h"
#include "hardware/structs/sio.h"
#include "hardware/structs/timer.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include <stdio.h>

#ifndef JTSMM_WATCHDOG_MS
#define JTSMM_WATCHDOG_MS 0
#endif

/* Timeout while a known long operation runs (config apply, flash write,
 * selftest): long enough for a BNO085 boot, short enough to catch a hang.
 * The RP2350 watchdog counts at most ~16.7 s. */
#define WATCHDOG_HOLD_MS 15000u

void fault_handler_init(void) {
    gpio_init(PIN_LED_FAULT);
    gpio_set_dir(PIN_LED_FAULT, GPIO_OUT);
    gpio_put(PIN_LED_FAULT, 0);
}

/*
 * SOS on the fault LED using nothing but SIO and the raw timer register: no
 * IRQs (a HardFault runs above all of them, so sleep_ms() would never return),
 * no flash-resident helpers beyond this file, no heap, no panel/WS2812.
 */
static void __not_in_flash_func(sos_delay_ms)(uint32_t ms) {
    uint32_t t0 = timer_hw->timerawl;
    while (timer_hw->timerawl - t0 < ms * 1000u) {
    }
}

static void __not_in_flash_func(sos_blink)(uint32_t on_ms) {
    sio_hw->gpio_set = 1u << PIN_LED_FAULT;
    sos_delay_ms(on_ms);
    sio_hw->gpio_clr = 1u << PIN_LED_FAULT;
    sos_delay_ms(200);
}

static void __not_in_flash_func(sos_cycle)(void) {
    for (int i = 0; i < 3; i++) {
        sos_blink(200);
    }
    sos_delay_ms(400);
    for (int i = 0; i < 3; i++) {
        sos_blink(600);
    }
    sos_delay_ms(400);
    for (int i = 0; i < 3; i++) {
        sos_blink(200);
    }
    sos_delay_ms(1000);
}

void fault_handler_sos(void) {
    panel_set_state(PANEL_STATE_FAULT);
    panel_fault_set(PANEL_FAULT_HARD);
    for (int cycle = 0; cycle < 3; cycle++) {
        sos_cycle();
    }
    gpio_put(PIN_LED_FAULT, 1);
}

/*
 * HardFault (either core; the vector table is shared). Blink SOS forever. If
 * the watchdog is enabled nothing kicks it any more, so the chip resets after
 * JTSMM_WATCHDOG_MS and the next boot reports "watchdog reset".
 */
void __not_in_flash_func(isr_hardfault)(void) {
    sio_hw->gpio_oe_set = 1u << PIN_LED_FAULT;
    while (true) {
        sos_cycle();
    }
}

#if JTSMM_WATCHDOG_MS > 0
static uint32_t wd_last_ticks;
static uint32_t wd_hold_depth;
static bool wd_running;
#endif

void fault_watchdog_start(void) {
#if JTSMM_WATCHDOG_MS > 0
    watchdog_enable(JTSMM_WATCHDOG_MS, true);
    wd_running = true;
    printf("watchdog: enabled, %u ms (core-1 heartbeat gated)\r\n", (unsigned)JTSMM_WATCHDOG_MS);
#endif
}

void fault_watchdog_service(uint32_t core1_ticks, bool core1_expected) {
#if JTSMM_WATCHDOG_MS > 0
    if (!wd_running) {
        return;
    }
    /* Kick only while both cores make progress: this loop (core 0) runs, and
     * core 1 has completed a poll since the last kick -- unless it is parked or
     * not launched on purpose (config apply, bench commands, failed platform). */
    if (!core1_expected || core1_ticks != wd_last_ticks) {
        wd_last_ticks = core1_ticks;
        watchdog_update();
    }
#else
    (void)core1_ticks;
    (void)core1_expected;
#endif
}

void fault_watchdog_hold(void) {
#if JTSMM_WATCHDOG_MS > 0
    if (wd_running && wd_hold_depth++ == 0u && WATCHDOG_HOLD_MS > JTSMM_WATCHDOG_MS) {
        watchdog_enable(WATCHDOG_HOLD_MS, true);
    }
#endif
}

void fault_watchdog_release(void) {
#if JTSMM_WATCHDOG_MS > 0
    if (wd_running && wd_hold_depth > 0u && --wd_hold_depth == 0u) {
        watchdog_enable(JTSMM_WATCHDOG_MS, true);
    }
#endif
}

const char *fault_reset_reason(void) {
    if (watchdog_enable_caused_reboot()) {
        return "watchdog timeout (firmware hang, or a hard fault that blinked SOS)";
    }
    if (watchdog_caused_reboot()) {
        return "watchdog/software reboot";
    }
    return "power-on / reset pin";
}

void fault_report_boot_reason(void) {
    const char *why = fault_reset_reason();
    printf("boot: last reset = %s\r\n", why);
    if (watchdog_enable_caused_reboot()) {
        char msg[96];
        snprintf(msg, sizeof(msg), "boot: last reset = %s", why);
        host_link_send_log(msg);
        panel_fault_set(PANEL_FAULT_HARD);
    }
}
