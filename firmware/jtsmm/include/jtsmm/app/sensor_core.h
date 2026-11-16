// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file sensor_core.h
 * @brief Sensor core: sole sensor producer on physical core 1.
 *
 * @details Owns ADS/EMG SPI and IMU I2C and is the only writer to the EMG/IMU
 *          rings. Launched from the comms core via @c multicore_launch_core1.
 *          See docs/hardware/h05_sensor_pipeline_architecture.md.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/** @brief One-time init of sensor-side drivers/state (may run before launch). */
void sensor_core_init(void);

/**
 * @brief One sensor-core loop iteration (EMG/IMU poll, ring push).
 *
 * @note Runs on physical core 1 after @ref sensor_core_launch.
 */
void sensor_core_poll(void);

/** @brief Launch the sensor-core loop on physical core 1, or release it if parked. */
void sensor_core_launch(void);

/**
 * @brief Park the sensor core between two poll iterations (cooperative stop).
 *
 * @details Blocks (bounded, ~200 ms) until core 1 acknowledges and waits in a
 *          RAM-resident loop; afterwards the comms core may touch the sensor
 *          buses. @ref sensor_core_launch releases it. If core 1 does not
 *          acknowledge in time it is hard-reset instead (logged).
 * @return true if it parked (or was not running); false if it had to be reset.
 */
bool sensor_core_stop(void);

/** @brief true while core 1 is launched and not parked (expected to make progress). */
bool sensor_core_running(void);

/** @brief Number of sensor_core_poll iterations completed. */
uint32_t sensor_poll_ticks_count(void);

/** @brief Peak IMU service duration observed in one poll (µs). */
uint32_t sensor_core_imu_svc_max_us(void);

/** @brief Peak lead-off service duration observed in one poll (µs). */
uint32_t sensor_core_leadoff_max_us(void);

/** @brief Count of lead-off service invocations. */
uint32_t sensor_core_leadoff_count(void);

/** @brief Peak full @ref sensor_core_poll duration (µs). */
uint32_t sensor_core_poll_max_us(void);
