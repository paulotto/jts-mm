// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file led_flash_pio.h
 * @brief PIO `led_flash` driver for Track A sync LED on @c PIN_LED_FLASH.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Load PIO program for the sync LED.
 * @param clk_hz PIO clock (Hz).
 * @return true on success.
 */
bool led_flash_pio_init(uint32_t clk_hz);

/**
 * @brief Start periodic LED flashes.
 * @param rate_hz        Flash rate (Hz).
 * @param pulse_width_us On-time (µs).
 * @return true on success.
 */
bool led_flash_pio_start(uint16_t rate_hz, uint32_t pulse_width_us);

/** @brief Stop periodic flashing. */
void led_flash_pio_stop(void);

/**
 * @brief Emit a single flash pulse.
 * @param pulse_width_us On-time (µs).
 * @return true on success.
 */
bool led_flash_pio_once(uint32_t pulse_width_us);

/** @brief true while periodic flashing is active. */
bool led_flash_pio_running(void);

/** @brief Monotonic flash event id. */
uint32_t led_flash_pio_id(void);

/** @brief PIO IRQ hook for flash edge events. */
void led_flash_on_pio_irq(void);
