// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_flash.h
 * @brief Persist / load MCU config JSON in on-chip flash.
 *
 * @details Used by `config save` / `config load`. Buffer size must fit a full
 *          apply JSON (see @c JTSMM_CMD_LINE_MAX). Save/erase stall flash for
 *          both cores (core 1 is locked out via flash_safe_execute) for up to a
 *          few hundred ms with IRQs off: only call them while nothing streams.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>

/**
 * @brief Write @p json to the config flash slot.
 * @param json NUL-terminated JSON; must not be NULL.
 * @return true if written and read back intact.
 */
bool config_flash_save(const char *json);

/**
 * @brief Read saved JSON into @p buf.
 * @param buf Destination; must not be NULL.
 * @param cap Capacity of @p buf including NUL.
 * @return true if a valid blob was loaded and NUL-terminated in @p buf.
 */
bool config_flash_load(char *buf, size_t cap);

/** @brief true if flash holds a structurally valid saved config. */
bool config_flash_has_valid(void);

/** @brief Erase the saved config slot. @return true on success. */
bool config_flash_erase(void);
