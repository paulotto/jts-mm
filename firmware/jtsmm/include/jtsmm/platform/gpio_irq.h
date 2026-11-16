// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file gpio_irq.h
 * @brief Shared GPIO IRQ registration / dispatch helpers.
 *
 * @note Per core. The SDK keeps one GPIO callback and one set of GPIO IRQ
 *       enables per core; gpio_irq_init() registers the dispatcher for the
 *       CALLING core (core 0: platform_config_apply) and gpio_irq_enable_pin()
 *       sets the calling core's enables. Nothing here serves core 1: the
 *       sensor core's IMU INT timestamp latch uses its own raw handler
 *       (imu_bus.h, imu_int_edge_core_init).
 */
#pragma once

#include "hardware/gpio.h"
#include <stdbool.h>
#include <stdint.h>

/** @brief Initialize GPIO IRQ infrastructure. */
void gpio_irq_init(void);

/**
 * @brief Register the top-level GPIO IRQ callback.
 * @param cb Pico SDK gpio IRQ callback; may be NULL to clear.
 */
void gpio_irq_register(gpio_irq_callback_t cb);

/**
 * @brief Forward a GPIO IRQ event to registered consumers.
 * @param gpio   GPIO number.
 * @param events Event mask from the SDK.
 */
void gpio_irq_dispatch(uint gpio, uint32_t events);

/**
 * @brief Enable or disable IRQ events on one pin.
 * @param gpio    GPIO number.
 * @param events  Event mask.
 * @param enabled true to enable.
 */
void gpio_irq_enable_pin(uint gpio, uint32_t events, bool enabled);
