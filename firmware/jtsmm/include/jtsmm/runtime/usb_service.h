// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file usb_service.h
 * @brief TinyUSB composite device service (CDC + vendor bulk).
 *
 * @details Initializes descriptors from usb_descriptors.c / tusb_config.h and
 *          pumps the USB stack on the comms core.
 */
#pragma once

#include <stdint.h>

/**
 * @brief Initialize TinyUSB (composite CDC + vendor bulk) and CDC stdio.
 *
 * @note Call once early on the comms core before sending records.
 */
void usb_service_init(void);

/**
 * @brief Run TinyUSB device task / endpoint progress.
 *
 * @note Call frequently from the comms-core loop.
 */
void usb_service_poll(void);

/**
 * @brief Sleep @p ms while keeping TinyUSB CDC alive.
 *
 * @param ms Duration in milliseconds.
 *
 * @details Prefer this over bare @c sleep_ms during long init so the host
 *          does not see a dead CDC pipe.
 */
void usb_service_sleep_ms(uint32_t ms);
