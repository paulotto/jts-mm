// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file led_flash_pio.c
 * @brief PIO Track A sync LED flasher.
 */

#include "jtsmm/drivers/led_flash_pio.h"

#include "jtsmm/pins.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"
#include "jtsmm/runtime/timebase.h"

#include "hardware/gpio.h"
#include "hardware/timer.h"
#include "pico/stdlib.h"
#include <string.h>

/*
 * Timer-driven flasher (the led_flash.pio program is not used yet).
 *
 *  - `running` means "periodic flashing requested"; `timer_active` tracks the
 *    repeating timer itself. They used to be one flag, and led_flash_pio_init()
 *    (called by every `flash on`) cleared it before led_flash_pio_start()
 *    checked it, so a second `flash on` never cancelled the first timer: two
 *    timers ran and the LED flashed twice per period. The timer is now always
 *    cancelled before a new one is added.
 *  - The LED-on edge is set in the timer callback and ended by a one-shot
 *    alarm, instead of busy-waiting pulse_width_us inside the alarm IRQ (which
 *    stalled every other timer callback, e.g. the trigger watchdog, for the
 *    whole pulse).
 */
static volatile bool running;
static bool timer_active;
static volatile uint32_t flash_id;
static uint32_t pulse_width_us = 200u;
static repeating_timer_t flash_rt;
static volatile alarm_id_t off_alarm;

/** @brief Internal helper: led_flash_push_event. */
static void led_flash_push_event(uint64_t t_us) {
    flash_id++;
    const uint32_t id = flash_id;
    jtsmm_record_t rec = {0};
    rec.type = REC_LED_FLASH_EVENT;
    rec.t_master_us = t_us;
    rec.len = 4;
    memcpy(rec.payload, &id, sizeof(id));
    ringbuf_push(RING_EVENTS, &rec);
}

/** @brief Internal helper: one-shot alarm that ends a pulse. */
static int64_t flash_off_cb(alarm_id_t id, void *user_data) {
    (void)id;
    (void)user_data;
    gpio_put(PIN_LED_FLASH, 0);
    off_alarm = 0;
    return 0; /* do not reschedule */
}

/** @brief Internal helper: cancel a pending pulse end and force the LED off. */
static void flash_off_now(void) {
    alarm_id_t pending = off_alarm;
    if (pending > 0) {
        cancel_alarm(pending);
        off_alarm = 0;
    }
    gpio_put(PIN_LED_FLASH, 0);
}

/** @brief Internal helper: flash_tick. */
static bool flash_tick(repeating_timer_t *rt) {
    (void)rt;
    if (!running) {
        return true;
    }
    uint64_t t0 = t_master_us();
    gpio_put(PIN_LED_FLASH, 1);
    alarm_id_t id = add_alarm_in_us(pulse_width_us, flash_off_cb, NULL, true);
    if (id > 0) {
        off_alarm = id;
    } else {
        gpio_put(PIN_LED_FLASH, 0); /* no alarm slot: never leave the LED on */
    }
    led_flash_push_event(t0);
    return true;
}

/** @brief Internal helper: stop the repeating timer if one is scheduled. */
static void flash_timer_cancel(void) {
    if (timer_active) {
        cancel_repeating_timer(&flash_rt);
        timer_active = false;
    }
}

bool led_flash_pio_init(uint32_t clk_hz) {
    (void)clk_hz;
    running = false;
    flash_timer_cancel();
    flash_off_now();
    flash_id = 0;
    gpio_init(PIN_LED_FLASH);
    gpio_set_dir(PIN_LED_FLASH, GPIO_OUT);
    gpio_put(PIN_LED_FLASH, 0);
    return true;
}

bool led_flash_pio_start(uint16_t rate_hz, uint32_t width_us) {
    if (rate_hz == 0) {
        return false;
    }
    uint32_t period_us = 1000000u / rate_hz;
    uint32_t width = width_us ? width_us : 200u;
    if (width >= period_us) {
        width = period_us / 2u;
    }
    /* Always re-arm: exactly one repeating timer, whatever ran before. */
    running = false;
    flash_timer_cancel();
    flash_off_now();
    pulse_width_us = width;
    running = true;
    if (!add_repeating_timer_us(-(int64_t)period_us, flash_tick, NULL, &flash_rt)) {
        running = false;
        return false;
    }
    timer_active = true;
    return true;
}

void led_flash_pio_stop(void) {
    running = false;
    flash_timer_cancel();
    flash_off_now();
}

bool led_flash_pio_once(uint32_t width_us) {
    if (width_us == 0) {
        width_us = 200u;
    }
    /* Main-loop context (a CDC command), so a short busy wait is harmless here. */
    uint64_t t0 = t_master_us();
    gpio_put(PIN_LED_FLASH, 1);
    busy_wait_us(width_us);
    gpio_put(PIN_LED_FLASH, 0);
    led_flash_push_event(t0);
    return true;
}

bool led_flash_pio_running(void) {
    return running;
}

uint32_t led_flash_pio_id(void) {
    return flash_id;
}

void led_flash_on_pio_irq(void) {
    led_flash_push_event(t_master_us());
}
