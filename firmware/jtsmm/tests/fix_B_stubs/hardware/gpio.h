// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub (fix package B): hardware/gpio.h subset; fakes in tests/fix_B_fakes.c. */
#pragma once

#include "pico/types.h"

#define GPIO_IN  false
#define GPIO_OUT true

enum gpio_function { GPIO_FUNC_I2C = 3, GPIO_FUNC_SIO = 5 };

void gpio_init(uint gpio);
void gpio_set_dir(uint gpio, bool out);
void gpio_pull_up(uint gpio);
void gpio_put(uint gpio, bool value);
bool gpio_get(uint gpio);
void gpio_set_function(uint gpio, enum gpio_function fn);
