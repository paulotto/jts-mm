// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file gpio_irq.c
 * @brief Shared GPIO IRQ dispatch.
 */

#include "jtsmm/platform/gpio_irq.h"

#include "hardware/gpio.h"

#define MAX_GPIO_HANDLERS 4u

static gpio_irq_callback_t handler_list[MAX_GPIO_HANDLERS];
static size_t handler_count;
static bool gpio_callback_set;

void gpio_irq_init(void) {
    if (!gpio_callback_set) {
        gpio_set_irq_callback(gpio_irq_dispatch);
        gpio_callback_set = true;
    }
}

void gpio_irq_register(gpio_irq_callback_t cb) {
    if (cb == NULL) {
        return;
    }
    for (size_t i = 0; i < handler_count; i++) {
        if (handler_list[i] == cb) {
            return;
        }
    }
    if (handler_count < MAX_GPIO_HANDLERS) {
        handler_list[handler_count++] = cb;
    }
}

void gpio_irq_dispatch(uint gpio, uint32_t events) {
    for (size_t i = 0; i < handler_count; i++) {
        if (handler_list[i]) {
            handler_list[i](gpio, events);
        }
    }
}

void gpio_irq_enable_pin(uint gpio, uint32_t events, bool enabled) {
    gpio_irq_init();
    gpio_set_irq_enabled(gpio, events, enabled);
}
