// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file cmd_parser.h
 * @brief USB CDC text command parser (`config apply`, `info`, bench cmds).
 *
 * @details Runs on the comms core; lines are delimited text over CDC.
 *          Max line length is @c JTSMM_CMD_LINE_MAX in config_json.h.
 */
#pragma once

#include <stdbool.h>

/** @brief Initialize parser state and RX buffers. */
void cmd_parser_init(void);

/**
 * @brief Poll CDC for command bytes and dispatch complete lines.
 *
 * @note Call from the comms-core loop with @ref usb_service_poll.
 */
void cmd_parser_poll(void);

/** @brief Discard any partial RX line (e.g. after abort). */
void cmd_parser_flush_rx(void);
