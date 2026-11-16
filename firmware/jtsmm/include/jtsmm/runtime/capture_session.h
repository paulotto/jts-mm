// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file capture_session.h
 * @brief Capture session start/stop/toggle for streaming modalities.
 *
 * @details Coordinates stream enable flags and panel CAP LED with EMG/IMU/trigger
 *          producers. Invoked from commands and the user button path.
 */
#pragma once

#include <stdbool.h>

/** @brief true while a capture session is active. */
bool capture_session_active(void);

/** @brief Start streaming according to active config stream_* flags. */
void capture_session_start(void);

/** @brief Stop all modality streams started by the session. */
void capture_session_stop(void);

/** @brief Toggle @ref capture_session_start / @ref capture_session_stop. */
void capture_session_toggle(void);
