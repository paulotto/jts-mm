// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * i2c_dma.c abort handshake vs. TX_ABRT ownership.
 *
 * Reading IC_CLR_TX_ABRT clears the NACK flag for whoever is using the I2C
 * block. The sensor core's abort path must only do that for its own DMA read:
 * when no read is active the comms core may be mid blocking SDK write on the
 * same block (imu_stream off during a same-slot recovery), and clearing its
 * TX_ABRT made a NACKed write look successful. An acknowledge shows up as
 * i2c_dma_tx_abort_count() (the driver counts every one it performs).
 */
#include "jtsmm/platform/i2c_dma.h"

#include "check.h"

#include "hardware/regs/i2c.h"

#include <string.h>

uint64_t fake_now_us = 1000u;
bool fake_dma_chan_busy[8];
int fake_dma_chan_aborts[8];

static i2c_hw_t hw0;
static i2c_inst_t inst0 = {&hw0};

/* Kick a header read and let the sensor core launch it. */
static void launch_read(void) {
    i2c_dma_kick_shtp_read(I2C_DMA_SLOT_MTA, &inst0, 0x4A, fake_now_us);
    i2c_dma_service();
    CHECK(i2c_dma_busy(I2C_DMA_SLOT_MTA));
}

/* The read completes (header length 0 -> finished, no read active). */
static void complete_read(void) {
    memset(fake_dma_chan_busy, 0, sizeof(fake_dma_chan_busy));
    i2c_dma_service();
    CHECK(!i2c_dma_busy(I2C_DMA_SLOT_MTA));
}

int main(void) {
    i2c_dma_init();

    /* No DMA read active; the comms core's blocking write got NACKed and the
     * SDK has not looked at TX_ABRT yet when imu_stream off requests an abort. */
    launch_read();
    complete_read();
    uint32_t acks = i2c_dma_tx_abort_count(I2C_DMA_SLOT_MTA);
    hw0.raw_intr_stat = I2C_IC_RAW_INTR_STAT_TX_ABRT_BITS;
    i2c_dma_request_abort(I2C_DMA_SLOT_MTA);
    i2c_dma_service();
    CHECK(!i2c_dma_busy(I2C_DMA_SLOT_MTA)); /* handshake still acknowledged */
    CHECK_EQ_U(i2c_dma_tx_abort_count(I2C_DMA_SLOT_MTA), acks);
    /* Same for a local abort from the comms core with nothing in flight. */
    i2c_dma_abort(I2C_DMA_SLOT_MTA);
    CHECK_EQ_U(i2c_dma_tx_abort_count(I2C_DMA_SLOT_MTA), acks);
    hw0.raw_intr_stat = 0;

    /* A DMA read in flight that gets NACKed: the sensor core owns that
     * TX_ABRT and must acknowledge it, or the TX FIFO stays flushed. */
    launch_read();
    hw0.raw_intr_stat = I2C_IC_RAW_INTR_STAT_TX_ABRT_BITS;
    i2c_dma_service();
    CHECK(!i2c_dma_busy(I2C_DMA_SLOT_MTA));
    CHECK_EQ_U(i2c_dma_tx_abort_count(I2C_DMA_SLOT_MTA), acks + 1);
    hw0.raw_intr_stat = 0;

    /* Abort requested while a read is in flight and NACKed: acknowledged. */
    launch_read();
    hw0.raw_intr_stat = I2C_IC_RAW_INTR_STAT_TX_ABRT_BITS;
    i2c_dma_request_abort(I2C_DMA_SLOT_MTA);
    i2c_dma_service();
    CHECK(!i2c_dma_busy(I2C_DMA_SLOT_MTA));
    CHECK_EQ_U(i2c_dma_tx_abort_count(I2C_DMA_SLOT_MTA), acks + 2);
    hw0.raw_intr_stat = 0;

    CHECK_DONE("test_i2c_dma_abort");
}
