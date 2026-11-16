// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_json.h
 * @brief Parse / dump MCU config JSON (cJSON).
 *
 * @details Host `config apply` lines and flash blobs use this schema subset
 *          of schema/v1. See docs/software/s08_runtime_configuration.md.
 */
#pragma once

#include "jtsmm/config.h"
#include <stddef.h>
#include <stdbool.h>

/** Max USB CDC line for `config apply {json}` (must match cmd_parser + config_apply). */
#define JTSMM_CMD_LINE_MAX 4096u

/**
 * @brief Parse JSON into @p cfg (does not apply to hardware).
 *
 * @details Overlays the keys present in @p json onto @p cfg; absent keys keep
 *          their current value. All five sections (platform, emg, imu_mta,
 *          imu_cra, trigger) are required and must be objects; a present key
 *          of the wrong type, a negative/fractional number, or a number that
 *          does not fit its field is an error (see @ref config_json_last_error).
 *          On failure @p cfg may be partially written: parse into a scratch copy.
 *
 * @param json NUL-terminated JSON; must not be NULL.
 * @param cfg  In/out config; must not be NULL.
 * @return true if the whole document parsed.
 */
bool config_json_parse(const char *json, jtsmm_mcu_config_t *cfg);

/** @brief Human-readable reason for the last @ref config_json_parse failure. */
const char *config_json_last_error(void);

/**
 * @brief Serialize @p cfg to JSON text.
 *
 * @param cfg     Config to dump; must not be NULL.
 * @param out     Destination buffer; must not be NULL.
 * @param out_len Capacity of @p out including NUL.
 * @return true if the full JSON fit in @p out.
 */
bool config_json_dump(const jtsmm_mcu_config_t *cfg, char *out, size_t out_len);
