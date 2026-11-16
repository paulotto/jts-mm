// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file emg_config.h
 * @brief Domain layer: apply EMG config and stream ADS frames into rings.
 */
#pragma once

#include "jtsmm/config.h"
#include <stdbool.h>

/**
 * @brief Apply @p cfg to the ADS1299 driver (registers / rates / bias).
 * @param cfg EMG config; must not be NULL.
 * @return true on success.
 */
bool emg_config_apply(const emg_config_t *cfg);

/**
 * @brief Validate @p cfg without touching hardware.
 * @param cfg EMG config; must not be NULL.
 * @return true if values are supported.
 */
bool emg_config_validate(const emg_config_t *cfg);

/**
 * @brief Expand @p cfg->profile into concrete fields when not CUSTOM.
 * @param cfg In/out config; must not be NULL.
 */
void emg_config_resolve_profile(emg_config_t *cfg);

/**
 * @brief SPI DMA completion hook: build @c REC_EMG_BLOCK and push @c RING_EMG.
 * @param t_us  Latched master time (µs).
 * @param seq   Acquisition sequence number assigned at DRDY (contract K1).
 * @param data  ADS frame bytes; must not be NULL.
 * @param len   Byte length (typically @c ADS_SPI_FRAME_BYTES).
 */
void emg_stream_on_complete(uint64_t t_us, uint32_t seq, const uint8_t *data, uint16_t len);

/**
 * @brief Start EMG streaming (DRDY → DMA → ring).
 *
 * @details Idempotent: returns true without touching the ADS or the counters if
 *          the stream is already running.
 * @return false if the ADS1299 is not configured (last init failed); callers
 *         should NAK.
 */
bool emg_stream_start(void);

/** @brief Stop EMG streaming. */
void emg_stream_stop(void);

/** @brief Count of successful EMG ring pushes (failed pushes are ringbuf_drops()). */
uint32_t emg_stream_push_ok_count(void);

/**
 * @brief Frames discarded because the ADS status word failed the framing check.
 *
 * @details Non-zero means samples were lost but no corrupt data was published.
 *          Reset per stream (like @c ads1299_frame_misalign_count), so it always
 *          describes the current recording. Surfaced in @c REC_STATS and logged
 *          once per second by the comms core.
 */
uint32_t emg_stream_misalign_drop_count(void);
