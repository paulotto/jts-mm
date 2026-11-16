// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file selftest.h
 * @brief Bench `selftest` command: probe ADS/IMU/trigger basics.
 */
#pragma once

#include <stdbool.h>

/**
 * @brief Run built-in self-test steps and log results over CDC/USB.
 *
 * @return true if all mandatory checks passed.
 *
 * @note Intended for interactive bench use; may briefly touch hardware.
 */
bool selftest_run(void);
