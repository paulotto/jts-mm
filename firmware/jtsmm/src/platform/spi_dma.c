// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file spi_dma.c
 * @brief ADS1299 SPI DMA reader.
 */

#include "jtsmm/platform/spi_dma.h"

#include "jtsmm/pins.h"

#include "hardware/dma.h"
#include "hardware/gpio.h"
#include "hardware/spi.h"
#include "pico/time.h"
#include <string.h>

/*
 * True async SPI DMA read of one ADS1299 RDATA frame (RDATA cmd + 27 data bytes,
 * full-duplex). The transfer is kicked from the sensor core and completes there
 * via spi_dma_service() polling dma_channel_is_busy() -- we deliberately do NOT
 * use the DMA NVIC IRQ (it is enabled on the comms core, and the completion
 * callback pushes the EMG ring, so servicing it on the sensor core preserves the
 * single-producer contract). This frees the CPU during the ~7 us frame clock so
 * EMG scales well beyond 1 kHz while both IMUs stream concurrently.
 */

#define ADS_SPI       spi0
#define ADS_CMD_RDATA 0x12u
#define FRAME_XFER    (ADS_SPI_FRAME_BYTES + 1u) /* RDATA cmd + 27 data bytes */

static uint tx_chan;
static uint rx_chan;
static bool dma_inited;
static uint8_t tx_buf[FRAME_XFER];
static uint8_t rx_raw[FRAME_XFER];
static uint8_t frame[ADS_SPI_FRAME_BYTES];
static volatile bool busy;
static volatile uint64_t latched_t_us;
static volatile uint32_t latched_tag;
static volatile uint64_t kick_us;
static volatile uint32_t dma_complete_count;
static volatile uint32_t dma_complete_no_cb;
static volatile uint32_t dma_wedge_count;
static volatile uint32_t dma_resync_count;
static void (*complete_cb)(uint64_t t_us, uint32_t tag, const uint8_t *data, uint16_t len);

/* A full 28-byte frame at 4 MHz is ~56 us; even at the 500 kHz config baud it is
 * ~450 us. If a transfer has not completed well past that it is wedged (e.g. a
 * bus collision), so abort and recover rather than stalling EMG for the session. */
#define SPI_DMA_WEDGE_US 2000u

/* Upper bound on the post-abort drain spin. One frame at the slow config baud is
 * ~450 us; anything past this is a dead peripheral and must not stall the sensor
 * loop (a stall costs DRDY edges, which is worse than an unrecovered frame). */
#define SPI_DMA_QUIESCE_US 1000u

/**
 * @brief Internal helper: return SPI0 + both DMA channels to a known-idle state.
 *
 * @details Aborting the DMA channels does not stop bytes already queued in the
 *          PL022 TX FIFO: those keep clocking and keep pushing bytes into RX
 *          *after* the abort. Draining RX once (as the abort path used to) can
 *          therefore leave residue that the next kick reads as the head of the
 *          frame, which is the one-byte offset that turns a clean channel into
 *          full-scale hash. So wait for BSY to clear first, then drain.
 *          Does not touch the ADS register map, so it is safe to call from the
 *          sensor core mid-stream.
 */
static void spi_dma_quiesce(void) {
    dma_channel_abort(tx_chan);
    dma_channel_abort(rx_chan);
    absolute_time_t deadline = make_timeout_time_us(SPI_DMA_QUIESCE_US);
    while (spi_is_busy(ADS_SPI) && !time_reached(deadline)) {
        /* Keep draining while it drains itself, or RX overruns and BSY lingers. */
        while (spi_is_readable(ADS_SPI)) {
            (void)spi_get_hw(ADS_SPI)->dr;
        }
        tight_loop_contents();
    }
    while (spi_is_readable(ADS_SPI)) {
        (void)spi_get_hw(ADS_SPI)->dr;
    }
    gpio_put(PIN_ADS_CS, 1);
    busy = false;
}

/** @brief Internal helper: spi_dma_abort. */
static void spi_dma_abort(void) {
    spi_dma_quiesce();
    dma_wedge_count++;
}

void spi_dma_resync(void) {
    spi_dma_quiesce();
    dma_resync_count++;
}

/** @brief Internal helper: spi_dma_try_complete. */
static void spi_dma_try_complete(void) {
    if (!busy) {
        return;
    }
    if (dma_channel_is_busy(rx_chan) || dma_channel_is_busy(tx_chan)) {
        if (time_us_64() - kick_us > SPI_DMA_WEDGE_US) {
            spi_dma_abort();
        }
        return;
    }
    gpio_put(PIN_ADS_CS, 1);
    busy = false;
    /* rx_raw[0] is clocked in during the RDATA command byte; skip it. */
    memcpy(frame, rx_raw + 1, ADS_SPI_FRAME_BYTES);
    if (complete_cb) {
        complete_cb(latched_t_us, latched_tag, frame, ADS_SPI_FRAME_BYTES);
        dma_complete_count++;
    } else {
        dma_complete_no_cb++;
    }
}

void spi_dma_init(void) {
    if (dma_inited) {
        /*
         * platform_config_apply() re-runs spi_dma_init() on every `config apply`
         * (view/record/emg check/emg impedance all trigger one, some more than
         * once per command). Re-claiming here with dma_claim_unused_channel()
         * would never release the previous pair -- after ~6-8 applies the RP2350's
         * 16 DMA channels are exhausted and the next claim panic()s, which is the
         * "board sometimes gets stuck" freeze. Channels are claimed exactly once;
         * subsequent calls only need to defensively clear any transfer left mid-
         * flight by a multicore_reset_core1() (the DMA hardware keeps running
         * across a core reset, so a stale busy=true must not block the next kick).
         */
        spi_dma_quiesce(); /* same BSY-wait + drain as the abort path */
        return;
    }
    tx_chan = dma_claim_unused_channel(true);
    rx_chan = dma_claim_unused_channel(true);
    busy = false;
    memset(tx_buf, 0, sizeof(tx_buf));
    memset(rx_raw, 0, sizeof(rx_raw));
    tx_buf[0] = ADS_CMD_RDATA;
    dma_inited = true;
}

bool spi_dma_busy(void) {
    return busy;
}

void spi_dma_on_complete(void (*cb)(uint64_t t_us, uint32_t tag, const uint8_t *data,
                                    uint16_t len)) {
    complete_cb = cb;
}

uint32_t spi_dma_complete_count(void) {
    return dma_complete_count;
}

uint32_t spi_dma_complete_no_cb_count(void) {
    return dma_complete_no_cb;
}

uint32_t spi_dma_wedge_count(void) {
    return dma_wedge_count;
}

uint32_t spi_dma_resync_count(void) {
    return dma_resync_count;
}

void spi_dma_service(void) {
    spi_dma_try_complete();
}

void spi_dma_kick_read(uint64_t t_latch_us, uint32_t tag) {
    if (busy) {
        return;
    }
    busy = true;
    latched_t_us = t_latch_us;
    latched_tag = tag;
    kick_us = time_us_64();

    /* Drop any stale RX data so the first DMA byte aligns with the frame. */
    while (spi_is_readable(ADS_SPI)) {
        (void)spi_get_hw(ADS_SPI)->dr;
    }

    gpio_put(PIN_ADS_CS, 0);

    dma_channel_config tx = dma_channel_get_default_config(tx_chan);
    channel_config_set_transfer_data_size(&tx, DMA_SIZE_8);
    channel_config_set_read_increment(&tx, true);
    channel_config_set_write_increment(&tx, false);
    channel_config_set_dreq(&tx, spi_get_dreq(ADS_SPI, true));
    dma_channel_configure(tx_chan, &tx, &spi_get_hw(ADS_SPI)->dr, tx_buf, FRAME_XFER, false);

    dma_channel_config rx = dma_channel_get_default_config(rx_chan);
    channel_config_set_transfer_data_size(&rx, DMA_SIZE_8);
    channel_config_set_read_increment(&rx, false);
    channel_config_set_write_increment(&rx, true);
    channel_config_set_dreq(&rx, spi_get_dreq(ADS_SPI, false));
    dma_channel_configure(rx_chan, &rx, rx_raw, &spi_get_hw(ADS_SPI)->dr, FRAME_XFER, false);

    /* Start RX first so it is armed before TX clocks the first byte out. */
    dma_channel_start(rx_chan);
    dma_channel_start(tx_chan);
}
