// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file host_link.h
 * @brief USB host link: framed record TX, ACK/LOG helpers, CRC stats.
 *
 * @details Owns framing onto the TinyUSB vendor bulk path used by the comms
 *          core to push @ref jtsmm_record_t frames to the Pi. CDC carries text
 *          separately. See docs/hardware/h05_microcontroller_firmware.md §3.
 */
#pragma once

#include "jtsmm/runtime/ringbuf.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Initialize TX path and statistics.
 *
 * @note Call once on the comms core after USB is up.
 */
void host_link_init(void);

/**
 * @brief Zero the TX drop counters (total and per stream) at a stream start.
 *
 * @details @ref host_link_tx_bytes is left running: it is a since-boot byte
 *          count that the host differentiates for throughput, and resetting it
 *          mid-session would read as a counter wrap. CRC errors are unchanged.
 */
void host_link_reset_tx_stats(void);

/**
 * @brief Forward queued ring records to the USB vendor FIFO (bounded work).
 *
 * @details Call from the comms-core main loop. A record stays in its ring
 *          until the FIFO has room for it.
 */
void host_link_drain(void);

/**
 * @brief Enqueue one typed record for USB transmission.
 *
 * @param type        Record type tag (`REC_*`).
 * @param t_master_us MCU master timestamp in microseconds.
 * @param payload     Record payload bytes; may be NULL if @p len is 0.
 * @param len         Payload length in bytes (≤ @c JTSMM_MAX_PAYLOAD).
 * @return true if the frame was accepted into the TX path; false if dropped
 *         (backpressure). Increments drop counters on failure.
 *
 * @note Must not be called from an ISR that cannot run the USB stack.
 */
bool host_link_send_record(uint8_t type, uint64_t t_master_us, const uint8_t *payload, uint16_t len);

/**
 * @brief Send a @c REC_CMD_ACK with success/failure flag.
 *
 * @param ok true for success acknowledgement.
 */
void host_link_send_ack(bool ok);

/**
 * @brief Send a @c REC_LOG (and typically echo on CDC).
 *
 * @param msg NUL-terminated C string; must not be NULL.
 */
void host_link_send_log(const char *msg);

/** @brief Cumulative framed bytes accepted for TX. */
uint32_t host_link_tx_bytes(void);

/**
 * @brief Frames dropped at the USB boundary since the last reset: direct sends
 *        with no FIFO space, and ring records discarded while no host is
 *        mounted. Equals the sum of the three per-stream counters below.
 *
 * @note Ring records waiting for FIFO space are not dropped here; if their ring
 *       overflows, the loss shows up in @c ringbuf_drops instead.
 */
uint32_t host_link_tx_drops(void);

/** @brief @ref host_link_tx_drops for @c REC_EMG_BLOCK frames. */
uint32_t host_link_tx_drops_emg(void);

/** @brief @ref host_link_tx_drops for IMU frames (@c REC_IMU_*). */
uint32_t host_link_tx_drops_imu(void);

/** @brief @ref host_link_tx_drops for everything else (events, logs, acks, stats). */
uint32_t host_link_tx_drops_other(void);

/** @brief Cumulative CRC errors observed on the RX/parse path (if tracked). */
uint32_t host_link_crc_errors(void);

/**
 * @brief CRC-16 used in the USB record framing.
 *
 * @param data Bytes to checksum; may be NULL if @p len is 0.
 * @param len  Number of bytes.
 * @return CRC-16 value.
 */
uint16_t host_link_crc16(const uint8_t *data, size_t len);
