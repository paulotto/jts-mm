// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file i2c_dma.h
 * @brief Per-IMU-slot I2C DMA / paced SHTP reads for BNO085.
 */
#pragma once

#include "hardware/i2c.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Which IMU I2C engine / slot is addressed.
 */
typedef enum {
    I2C_DMA_SLOT_MTA = 0, /**< Metatarsal IMU bus. */
    I2C_DMA_SLOT_CRA = 1, /**< Cranial / second IMU bus. */
} i2c_dma_slot_t;

/**
 * @brief Timeout for one I2C transaction of @p nbytes data bytes at @p baud_hz.
 *
 * @details Wire time is (nbytes + 1 address byte) x 9 bits / baud. Twice that
 *          plus 2 ms covers BNO085 clock stretching and bus gaps; never less
 *          than @p floor_us. The old fixed 8 ms DMA watchdog killed a full
 *          512-byte read even at 400 kHz (11.5 ms wire time) and was far too
 *          short at 100 kHz, which the config allows. @p baud_hz below 1 kHz
 *          (0 = unknown) counts as 100 kHz. 32-bit math (kHz, truncated =
 *          slightly longer timeout): it runs per DMA phase on the sensor core.
 *          Valid for nbytes < 400000.
 */
static inline uint32_t i2c_xfer_timeout_us(uint32_t nbytes, uint32_t baud_hz, uint32_t floor_us) {
    uint32_t khz = baud_hz / 1000u;
    if (khz == 0) {
        khz = 100u;
    }
    uint32_t wire_us = ((nbytes + 1u) * 9000u + khz - 1u) / khz;
    uint32_t t = 2u * wire_us + 2000u;
    return t < floor_us ? floor_us : t;
}

/** @brief Floor of the per-phase DMA watchdog (the previous fixed value). */
#define I2C_DMA_WATCHDOG_FLOOR_US 8000u

/** @brief Initialize I2C DMA state for both slots. */
void i2c_dma_init(void);

/**
 * @brief Progress outstanding I2C work (sensor-core poll).
 *
 * @details Order per slot: honour an abort request, advance a finished
 *          transfer, then abort on TX_ABRT (NACK) or the watchdog, then
 *          launch a kicked read.
 */
void i2c_dma_service(void);

/** @brief true while @p slot has an in-flight, kicked or abort-pending transfer. */
bool i2c_dma_busy(i2c_dma_slot_t slot);

/**
 * @brief Register SHTP packet completion callback for @p slot.
 * @param slot IMU slot.
 * @param cb   Receives (slot, t_us, pkt, len); may be NULL to clear.
 */
void i2c_dma_on_complete(i2c_dma_slot_t slot,
                         void (*cb)(i2c_dma_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len));

/**
 * @brief Register a hook run when @p slot's read moves from the 4-byte header
 *        phase to the full-packet re-read (sensor core, before that read starts).
 *
 * @details The BNO085 still holds the packet after the header-only read and may
 *          re-assert INT in between; the IMU timestamp latch drops edges from
 *          that gap here. May be NULL to clear.
 */
void i2c_dma_on_body_start(i2c_dma_slot_t slot, void (*cb)(i2c_dma_slot_t slot));

/**
 * @brief Abort any in-flight transfer on @p slot from the calling core.
 *
 * @warning Only for a core that owns the DMA engine: the sensor core itself,
 *          or the comms core while the sensor core is parked (config apply).
 *          From the comms core during streaming use @ref i2c_dma_quiesce.
 */
void i2c_dma_abort(i2c_dma_slot_t slot);

/**
 * @brief Ask the sensor core to abort @p slot's transfer at its next service.
 *
 * @details Non-blocking; @ref i2c_dma_busy stays true until it has happened.
 *          The caller must already have taken the slot off the streaming
 *          path so nothing kicks a new read.
 */
void i2c_dma_request_abort(i2c_dma_slot_t slot);

/**
 * @brief Comms-core handshake before a blocking access to @p slot's bus.
 *
 * @details Requests the abort and waits until the sensor core acknowledged
 *          it (the slot is idle). If the sensor core is not servicing at all
 *          (not launched / parked) the abort is done locally.
 * @param timeout_us Upper bound on the wait.
 * @return true if the slot is idle and the bus may be driven.
 */
bool i2c_dma_quiesce(i2c_dma_slot_t slot, uint32_t timeout_us);

/** @brief Tell @p slot's reader the bus bitrate so its watchdog scales (Hz). */
void i2c_dma_set_baud(i2c_dma_slot_t slot, uint32_t baud_hz);

/**
 * @brief Kick an SHTP read on @p slot.
 * @param slot        IMU slot.
 * @param i2c         Pico I2C instance.
 * @param addr        7-bit I2C address.
 * @param t_latch_us  Master timestamp associated with the INT/read (µs).
 */
void i2c_dma_kick_shtp_read(i2c_dma_slot_t slot, i2c_inst_t *i2c, uint8_t addr, uint64_t t_latch_us);

/**
 * @{
 * @name Wire-time instrumentation
 * @details Total elapsed (header+body) for the last completed SHTP read, the
 *          running max, and last packet length per slot. Used by `imu_status`
 *          to distinguish a slow I2C clock from BNO085 clock-stretching.
 */
uint32_t i2c_dma_last_read_us(i2c_dma_slot_t slot);
uint32_t i2c_dma_max_read_us(i2c_dma_slot_t slot);
uint16_t i2c_dma_last_len(i2c_dma_slot_t slot);
void i2c_dma_reset_stats(void);
/** @} */

/** @brief Transfers on @p slot killed by the (scaled) watchdog. */
uint32_t i2c_dma_timeout_count(i2c_dma_slot_t slot);

/**
 * @brief TX_ABRT events (NACK / arbitration lost) seen by the DMA reader on @p slot.
 *
 * @details The DW I2C block holds its TX FIFO flushed after an abort until
 *          IC_CLR_TX_ABRT is read; before this was cleared here, one NACK left
 *          every following DMA read discarding its command words until the
 *          blocking recovery path happened to clear it. Each count is one lost
 *          SHTP packet at most.
 */
uint32_t i2c_dma_tx_abort_count(i2c_dma_slot_t slot);
/** @brief Sum over both slots (REC_STATS `imu_i2c_aborts`). */
uint32_t i2c_dma_tx_abort_count_total(void);
