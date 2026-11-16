// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file comms_core.h
 * @brief Comms core: sole USB consumer on physical core 0.
 *
 * @details Runs in @c main() on the Pico SDK boot core. Drains SPSC rings to
 *          USB vendor bulk, handles CDC commands/config, and 1 Hz housekeeping
 *          (heartbeat, stats, sensor watchdog).
 */
#pragma once

#include <stdint.h>

/**
 * @brief Enter the never-returning comms-core main loop.
 *
 * @details Initializes USB, launches the sensor core, then polls forever.
 */
void comms_core_main(void);

/** @brief Number of comms-core loop iterations completed. */
uint32_t comms_poll_ticks_count(void);

/** @brief Estimated sensor-core loop rate (Hz), from housekeeping. */
uint32_t core_sensor_loop_hz(void);

/** @brief Estimated comms-core loop rate (Hz), from housekeeping. */
uint32_t core_comms_loop_hz(void);

/** @brief Emit a @c REC_STATS record with current counters. */
void comms_core_send_stats(void);
