// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub: hardware/gpio.h subset for the REAL ads1299.c (fake_ads1299.c). */
#pragma once

#include "pico/types.h"

#define GPIO_IN  false
#define GPIO_OUT true
#define GPIO_IRQ_EDGE_FALL 0x4u

enum gpio_function { GPIO_FUNC_SPI = 1, GPIO_FUNC_I2C = 3, GPIO_FUNC_SIO = 5 };
typedef void (*gpio_irq_callback_t)(uint gpio, uint32_t event_mask);

void gpio_init(uint gpio);
void gpio_set_dir(uint gpio, bool out);
void gpio_pull_up(uint gpio);
void gpio_disable_pulls(uint gpio);
void gpio_put(uint gpio, bool value);
bool gpio_get(uint gpio);
void gpio_set_function(uint gpio, enum gpio_function fn);
