// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file bno085_reports.h
 * @brief Catalog of BNO085 SH-2 reports and USB type mapping.
 */
#pragma once

#include "jtsmm/config.h"
#include <stddef.h>
#include <stdint.h>

/**
 * @brief One catalog entry describing a BNO085 report.
 */
typedef struct {
    uint8_t report_id;         /**< SH-2 report id. */
    const char *name;          /**< Stable name used in JSON/config. */
    uint8_t usb_type;          /**< USB @c REC_* type for this report. */
    uint16_t default_rate_hz;  /**< Default enable rate (Hz). */
    uint16_t max_rate_hz;      /**< Maximum supported rate (Hz). */
    uint16_t payload_len;      /**< Expected USB payload length (bytes). */
} bno085_report_entry_t;

/**
 * @brief Return the static report catalog.
 * @param[out] count Number of entries; must not be NULL.
 * @return Pointer to the first entry (never NULL).
 */
const bno085_report_entry_t *bno085_report_catalog(size_t *count);

/**
 * @brief Find a catalog entry by SH-2 report id.
 * @param report_id SH-2 id.
 * @return Entry pointer, or NULL if unknown.
 */
const bno085_report_entry_t *bno085_report_find(uint8_t report_id);

/**
 * @brief Find a catalog entry by config name string.
 * @param name NUL-terminated name; must not be NULL.
 * @return Entry pointer, or NULL if unknown.
 */
const bno085_report_entry_t *bno085_report_find_by_name(const char *name);

/**
 * @brief Map fusion-primary enum to SH-2 report id.
 * @param primary Fusion family from config.
 * @return SH-2 report id.
 */
uint8_t bno085_fusion_report_id(imu_fusion_primary_t primary);
