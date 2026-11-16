// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file timebase.h
 * @brief MCU master clock accessors (`t_master_us`).
 *
 * @details The RP2350 hardware timer is the sole master clock for EMG, IMU,
 *          and trigger stamps. See docs/hardware/h06_synchronization.md.
 */
#pragma once

#include <stdint.h>

/**
 * @brief Return microseconds since MCU boot (master timeline).
 *
 * @return 64-bit monotonic time from `time_us_64()` after @ref timebase_init.
 *
 * @note Safe to call from either core once initialized.
 */
uint64_t t_master_us(void);

/**
 * @brief Initialize the master timebase (no-op wrapper / readiness hook).
 *
 * @details Call once during boot before stamping records.
 */
void timebase_init(void);
