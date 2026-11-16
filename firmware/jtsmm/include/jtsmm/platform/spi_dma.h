// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file spi_dma.h
 * @brief SPI0 DMA reader for ADS1299 RDATA frames.
 *
 * @details Sensor core kicks a DMA read on DRDY; completion callback receives
 *          the latched @c t_master_us and @c ADS_SPI_FRAME_BYTES of data.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define ADS_SPI_FRAME_BYTES 27u /**< ADS1299 status + 8×24-bit channels. */

/** @brief Claim DMA channels / IRQ hooks for ADS SPI. */
void spi_dma_init(void);

/** @brief true while a DMA transfer is in flight. */
bool spi_dma_busy(void);

/**
 * @brief Start one ADS RDATA DMA read.
 * @param t_latch_us Master timestamp latched at DRDY (µs).
 * @param tag        Opaque value handed back to the completion callback
 *                   (the EMG acquisition seq).
 */
void spi_dma_kick_read(uint64_t t_latch_us, uint32_t tag);

/**
 * @brief Register completion callback (sensor-core context).
 * @param cb Receives (t_us, tag, data, len); may be NULL to clear.
 */
void spi_dma_on_complete(void (*cb)(uint64_t t_us, uint32_t tag, const uint8_t *data,
                                    uint16_t len));

/** @brief Successful completions that invoked the callback. */
uint32_t spi_dma_complete_count(void);

/** @brief Completions observed with no callback registered. */
uint32_t spi_dma_complete_no_cb_count(void);

/** @brief Times the DMA path was wedged / required recovery. */
uint32_t spi_dma_wedge_count(void);

/**
 * @brief Force SPI0 + DMA back to a known-idle, fully drained state.
 *
 * @details Recovery for a slipped RDATA byte stream: aborts any transfer in
 *          flight, waits (bounded) for the peripheral to go idle, drains the RX
 *          FIFO and deasserts CS. Does not touch the ADS register map, so it is
 *          safe to call from the sensor core between frames.
 */
void spi_dma_resync(void);

/** @brief Times @ref spi_dma_resync ran. */
uint32_t spi_dma_resync_count(void);

/** @brief Poll for completion when not fully IRQ-driven. */
void spi_dma_service(void);
