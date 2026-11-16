// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_apply.h
 * @brief Apply MCU config (JSON or defaults) onto hardware subsystems.
 *
 * @details Asynchronous only: @c config_apply_start_* queues an apply and
 *          @ref config_apply_poll advances it one step per comms-loop pass so
 *          USB stays alive. The whole config is parsed and validated into a
 *          scratch copy first; only if that succeeds are the streams stopped,
 *          the sensor core parked and the live config replaced, then applied
 *          in the order platform → emg → imu_mta → imu_cra → trigger.
 *          A config that fails validation leaves the running one untouched.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/** @brief Fingerprint of the last successfully applied config. */
uint64_t config_apply_fingerprint(void);

/** @brief Recompute and store the fingerprint from the active config. */
void config_apply_update_fingerprint(void);

/** @brief true while an async apply started by @c config_apply_start_* is running. */
bool config_apply_busy(void);

/** @brief Result of the most recently finished apply (true before the first). */
bool config_apply_last_ok(void);

/**
 * @brief Begin async apply of factory defaults.
 * @return false (nothing queued) if an apply is already running.
 */
bool config_apply_start_defaults(void);

/**
 * @brief Begin async apply of @p json (copied; the caller's buffer may be reused).
 * @param json NUL-terminated JSON (≤ @c JTSMM_CMD_LINE_MAX - 1 bytes).
 * @return false (nothing queued) if an apply is already running or @p json is NULL.
 */
bool config_apply_start_json(const char *json);

/**
 * @brief Progress an async apply one step; keep calling until not busy.
 *
 * @note Intended for the comms-core loop alongside @ref usb_service_poll.
 */
void config_apply_poll(void);
