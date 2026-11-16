// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file platform_config.h
 * @brief Apply platform clocks, bus init, and IRQ priorities.
 */
#pragma once

#include "jtsmm/config.h"
#include <stdbool.h>

/**
 * @brief Apply @p cfg to clocks / ring capacities / priorities.
 * @param cfg Platform config; must not be NULL.
 * @return true on success.
 */
bool platform_config_apply(const platform_config_t *cfg);

/**
 * @brief Validate @p cfg without applying.
 * @param cfg Platform config; must not be NULL.
 * @return true if values are within supported ranges.
 */
bool platform_config_validate(const platform_config_t *cfg);

/** @brief Log current platform status over CDC/USB. */
void platform_config_status(void);

/** @brief Initialize SPI/I2C buses used by ADS and IMUs. */
void platform_init_buses(void);

/** @brief Install shared IRQ handlers (GPIO/DMA) for sensor paths. */
void platform_irq_init(void);
