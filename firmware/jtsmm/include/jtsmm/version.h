// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file version.h
 * @brief Production firmware version (semantic versioning).
 *
 * @details `info` reports it with the `-prod` suffix and every recorded session stores that
 *          string (`/meta` `firmware_version`); the host's `hat.firmware_version` range
 *          (`jtsmm doctor`) checks it. History: README.md "Version".
 */
#pragma once

#define JTSMM_FW_VERSION "1.1.0"
