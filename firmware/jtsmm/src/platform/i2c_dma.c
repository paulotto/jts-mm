// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file i2c_dma.c
 * @brief BNO085 I2C/SHTP paced reads.
 */

#include "jtsmm/platform/i2c_dma.h"

#include "jtsmm/drivers/bno085.h"
#include "hardware/dma.h"
#include "hardware/i2c.h"
#include "hardware/regs/i2c.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"
#include <string.h>

/*
 * Non-blocking SHTP reads over I2C via DMA. The whole state machine is driven
 * from i2c_dma_service(), which MUST run only on the sensor core (physical
 * core 1): the completion callback pushes the IMU ring, so servicing it here
 * keeps the single-producer contract. We deliberately do NOT use the DMA NVIC
 * IRQ (that fires on the comms core, where platform_irq_init ran) -- completion
 * is detected by polling dma_channel_is_busy(), the same pattern spi_dma uses.
 *
 * SHTP-over-I2C quirk: each read transaction returns the current cargo from its
 * start (4-byte header + body). So we first read the 4-byte header to learn the
 * length, then re-read the FULL packet from the start into buf[0] (matching the
 * working blocking path in bno085_read_packet). MTA is on i2c0, CRA on i2c1
 * behind a PCA9546 whose channel latches, so the two slots run concurrently on
 * independent peripherals without per-read mux writes.
 *
 * Cross-core handshake: the comms core must not touch a slot's DMA channels or
 * I2C block while the sensor core may be driving them. It clears the slot's
 * streaming flag, sets abort_req (i2c_dma_request_abort / i2c_dma_quiesce) and
 * waits for i2c_dma_busy() to drop; the sensor core performs the abort at the
 * top of its next i2c_dma_service() and clears abort_req as the acknowledge.
 * Calling i2c_dma_abort() from the comms core raced the sensor core mid-launch.
 */

#define I2C_DMA_MAX_PACKET BNO085_SHTP_MAX_PACKET
#define I2C_DMA_MAX_CMDS   I2C_DMA_MAX_PACKET

/* No i2c_dma_service() tick for this long = sensor core not running. */
#define I2C_DMA_SERVICE_STALE_US 20000u

typedef enum {
    I2C_DMA_STATE_IDLE = 0,
    I2C_DMA_STATE_HEADER,
    I2C_DMA_STATE_BODY,
} i2c_dma_state_t;

typedef struct {
    volatile bool pending;
    volatile bool active;
    i2c_dma_state_t state;
    uint64_t t_latch;
    i2c_inst_t *i2c;
    uint8_t addr;
    uint8_t buf[I2C_DMA_MAX_PACKET];
    uint16_t len;
    uint8_t hdr_b1; /* byte 1 of the header pre-read (its continuation bit) */
    uint32_t cmd_words[I2C_DMA_MAX_CMDS];
    uint tx_chan;
    uint rx_chan;
    uint32_t since_us;   /* time_us_32() when the current phase started */
    uint32_t timeout_us; /* watchdog for the current phase (scaled to length and baud) */
    void (*cb)(i2c_dma_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len);
    void (*body_cb)(i2c_dma_slot_t slot);
} i2c_dma_ctx_t;

static i2c_dma_ctx_t slots[2];
static volatile bool abort_req[2];
static volatile uint32_t service_ticks;
static volatile uint32_t baud_hz[2];

/* Per-read wire-time instrumentation (see i2c_dma.h). */
static uint64_t read_t0[2];
static volatile uint32_t last_read_us[2];
static volatile uint32_t max_read_us[2];
static volatile uint16_t last_len_stat[2];
static volatile uint32_t tx_abrt_count[2];
static volatile uint32_t timeout_count[2];

uint32_t i2c_dma_tx_abort_count(i2c_dma_slot_t slot) {
    return (unsigned)slot < 2 ? tx_abrt_count[slot] : 0;
}
uint32_t i2c_dma_tx_abort_count_total(void) {
    return tx_abrt_count[0] + tx_abrt_count[1];
}
uint32_t i2c_dma_timeout_count(i2c_dma_slot_t slot) {
    return (unsigned)slot < 2 ? timeout_count[slot] : 0;
}

void i2c_dma_set_baud(i2c_dma_slot_t slot, uint32_t hz) {
    if ((unsigned)slot < 2) {
        baud_hz[slot] = hz;
    }
}

/**
 * @brief Internal helper: acknowledge a pending TX_ABRT so the TX FIFO works again.
 *
 * @details Per the DW_apb_i2c spec the TX FIFO stays flushed -- new writes to
 *          IC_DATA_CMD are silently dropped -- until IC_CLR_TX_ABRT is read.
 *          The SDK's blocking i2c_*_timeout_us() reads it on its own error
 *          path; the DMA path never did, so a single NACK (e.g. a BNO085 mid
 *          reset) wedged this slot's reads until the 3 s watchdog recovery
 *          happened to run a blocking read.
 */
static void i2c_ack_tx_abort(i2c_dma_ctx_t *ctx, i2c_dma_slot_t slot) {
    if (ctx->i2c == NULL) {
        return;
    }
    if (ctx->i2c->hw->raw_intr_stat & I2C_IC_RAW_INTR_STAT_TX_ABRT_BITS) {
        (void)ctx->i2c->hw->clr_tx_abrt;
        tx_abrt_count[slot]++;
    }
}

uint32_t i2c_dma_last_read_us(i2c_dma_slot_t slot) {
    return (unsigned)slot < 2 ? last_read_us[slot] : 0;
}
uint32_t i2c_dma_max_read_us(i2c_dma_slot_t slot) {
    return (unsigned)slot < 2 ? max_read_us[slot] : 0;
}
uint16_t i2c_dma_last_len(i2c_dma_slot_t slot) {
    return (unsigned)slot < 2 ? last_len_stat[slot] : 0;
}
void i2c_dma_reset_stats(void) {
    for (int i = 0; i < 2; i++) {
        last_read_us[i] = 0;
        max_read_us[i] = 0;
        last_len_stat[i] = 0;
    }
}

/**
 * @brief Internal helper: i2c_dma_abort_ctx.
 *
 * @details TX_ABRT is acknowledged only for a read of ours that was in flight.
 *          With no read active the comms core may own the I2C block -- e.g. a
 *          blocking SDK write during a same-slot recovery while `imu_stream
 *          off` requested this abort -- and reading IC_CLR_TX_ABRT here would
 *          clear its NACK before the SDK saw it, reporting a failed write as
 *          successful. A stale abort of our own is still cleared before the
 *          next launch (i2c_dma_start_read).
 */
static void i2c_dma_abort_ctx(i2c_dma_ctx_t *ctx, i2c_dma_slot_t slot) {
    if (ctx->active) {
        dma_channel_abort(ctx->tx_chan);
        dma_channel_abort(ctx->rx_chan);
        i2c_ack_tx_abort(ctx, slot);
    }
    ctx->pending = false;
    ctx->active = false;
    ctx->state = I2C_DMA_STATE_IDLE;
    ctx->len = 0;
}

/** @brief Internal helper: read_cmd_word. */
static uint32_t read_cmd_word(size_t idx, size_t total) {
    uint32_t cmd = I2C_IC_DATA_CMD_CMD_BITS;
    if (idx == 0) {
        cmd |= I2C_IC_DATA_CMD_RESTART_BITS;
    }
    if (idx + 1 == total) {
        cmd |= I2C_IC_DATA_CMD_STOP_BITS;
    }
    return cmd;
}

/** @brief Internal helper: i2c_prepare_target. */
static void i2c_prepare_target(i2c_inst_t *i2c, uint8_t addr) {
    i2c->hw->enable = 0;
    i2c->hw->tar = addr;
    i2c->hw->enable = 1;
}

/** @brief Internal helper: i2c_dma_finish. */
static void i2c_dma_finish(i2c_dma_ctx_t *ctx, i2c_dma_slot_t slot, bool ok) {
    ctx->active = false;
    ctx->pending = false;
    ctx->state = I2C_DMA_STATE_IDLE;
    if (ok) {
        uint32_t us = (uint32_t)(to_us_since_boot(get_absolute_time()) - read_t0[slot]);
        last_read_us[slot] = us;
        if (us > max_read_us[slot]) {
            max_read_us[slot] = us;
        }
        last_len_stat[slot] = ctx->len;
    }
    if (ok && ctx->cb) {
        ctx->cb(slot, ctx->t_latch, ctx->buf, ctx->len);
    }
}

/** @brief Internal helper: i2c_dma_start_read. */
static void i2c_dma_start_read(i2c_dma_ctx_t *ctx, i2c_dma_slot_t slot, size_t nbytes) {
    if (nbytes == 0 || nbytes > I2C_DMA_MAX_CMDS) {
        i2c_dma_finish(ctx, slot, false);
        return;
    }

    for (size_t i = 0; i < nbytes; i++) {
        ctx->cmd_words[i] = read_cmd_word(i, nbytes);
    }

    i2c_ack_tx_abort(ctx, slot);
    i2c_prepare_target(ctx->i2c, ctx->addr);

    dma_channel_config tx = dma_channel_get_default_config(ctx->tx_chan);
    channel_config_set_transfer_data_size(&tx, DMA_SIZE_32);
    channel_config_set_read_increment(&tx, true);
    channel_config_set_write_increment(&tx, false);
    channel_config_set_dreq(&tx, i2c_get_dreq(ctx->i2c, true));
    dma_channel_configure(ctx->tx_chan, &tx, &ctx->i2c->hw->data_cmd, ctx->cmd_words, nbytes, false);

    dma_channel_config rx = dma_channel_get_default_config(ctx->rx_chan);
    channel_config_set_transfer_data_size(&rx, DMA_SIZE_8);
    channel_config_set_read_increment(&rx, false);
    channel_config_set_write_increment(&rx, true);
    channel_config_set_dreq(&rx, i2c_get_dreq(ctx->i2c, false));
    dma_channel_configure(ctx->rx_chan, &rx, ctx->buf, &ctx->i2c->hw->data_cmd, nbytes, false);

    ctx->timeout_us = i2c_xfer_timeout_us((uint32_t)nbytes, baud_hz[slot], I2C_DMA_WATCHDOG_FLOOR_US);
    ctx->since_us = time_us_32();
    dma_channel_start(ctx->tx_chan);
    dma_channel_start(ctx->rx_chan);
}

/** @brief Internal helper: i2c_dma_on_rx_complete. */
static void i2c_dma_on_rx_complete(i2c_dma_slot_t slot) {
    i2c_dma_ctx_t *ctx = &slots[slot];
    dma_channel_abort(ctx->tx_chan);

    if (ctx->state == I2C_DMA_STATE_HEADER) {
        uint16_t pkt_len = (uint16_t)(ctx->buf[0] | ((uint16_t)ctx->buf[1] << 8));
        pkt_len &= 0x7FFFu;
        if (pkt_len < 4 || pkt_len > I2C_DMA_MAX_PACKET) {
            i2c_dma_finish(ctx, slot, false);
            return;
        }
        ctx->len = pkt_len;
        ctx->hdr_b1 = ctx->buf[1];
        if (pkt_len == 4) {
            i2c_dma_finish(ctx, slot, true);
            return;
        }
        /* Re-read the whole packet (header + body) from the start into buf[0]. */
        ctx->state = I2C_DMA_STATE_BODY;
        if (ctx->body_cb) {
            ctx->body_cb(slot);
        }
        i2c_dma_start_read(ctx, slot, pkt_len);
        return;
    }

    bno085_shtp_clear_reread_continuation(ctx->buf, ctx->hdr_b1);
    i2c_dma_finish(ctx, slot, true);
}

void i2c_dma_init(void) {
    static bool inited;
    if (inited) {
        return;
    }
    memset(slots, 0, sizeof(slots));
    for (int i = 0; i < 2; i++) {
        slots[i].tx_chan = dma_claim_unused_channel(true);
        slots[i].rx_chan = dma_claim_unused_channel(true);
    }
    inited = true;
}

bool i2c_dma_busy(i2c_dma_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return false;
    }
    bool req = abort_req[slot];
    __dmb();
    return req || slots[slot].pending || slots[slot].active;
}

void i2c_dma_on_complete(i2c_dma_slot_t slot,
                         void (*cb)(i2c_dma_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len)) {
    if ((unsigned)slot < 2) {
        slots[slot].cb = cb;
    }
}

void i2c_dma_on_body_start(i2c_dma_slot_t slot, void (*cb)(i2c_dma_slot_t slot)) {
    if ((unsigned)slot < 2) {
        slots[slot].body_cb = cb;
    }
}

void i2c_dma_abort(i2c_dma_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return;
    }
    i2c_dma_abort_ctx(&slots[slot], slot);
    __dmb();
    abort_req[slot] = false;
}

void i2c_dma_request_abort(i2c_dma_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return;
    }
    __dmb();
    abort_req[slot] = true;
    __dmb();
}

bool i2c_dma_quiesce(i2c_dma_slot_t slot, uint32_t timeout_us) {
    if ((unsigned)slot >= 2) {
        return false;
    }
    if (timeout_us < I2C_DMA_SERVICE_STALE_US + 1000u) {
        timeout_us = I2C_DMA_SERVICE_STALE_US + 1000u;
    }
    uint32_t t0 = time_us_32();
    uint32_t ticks0 = service_ticks;
    i2c_dma_request_abort(slot);
    while (i2c_dma_busy(slot)) {
        uint32_t elapsed = time_us_32() - t0;
        if (service_ticks == ticks0 && elapsed >= I2C_DMA_SERVICE_STALE_US) {
            /* Nobody services the engine (sensor core parked or not launched),
             * so this core owns it: abort locally. */
            i2c_dma_abort(slot);
            return true;
        }
        if (elapsed >= timeout_us) {
            return false;
        }
        tight_loop_contents();
    }
    return true;
}

void i2c_dma_kick_shtp_read(i2c_dma_slot_t slot, i2c_inst_t *i2c, uint8_t addr, uint64_t t_latch_us) {
    if ((unsigned)slot >= 2 || i2c == NULL) {
        return;
    }
    i2c_dma_ctx_t *ctx = &slots[slot];
    if (ctx->pending || ctx->active || abort_req[slot]) {
        return;
    }
    ctx->pending = true;
    ctx->t_latch = t_latch_us;
    ctx->i2c = i2c;
    ctx->addr = addr;
}

void i2c_dma_service(void) {
    service_ticks++;
    for (int i = 0; i < 2; i++) {
        i2c_dma_ctx_t *ctx = &slots[i];
        i2c_dma_slot_t slot = (i2c_dma_slot_t)i;
        __dmb();
        if (abort_req[i]) {
            /* Comms-core handshake: drop everything, then acknowledge. */
            i2c_dma_abort_ctx(ctx, slot);
            __dmb();
            abort_req[i] = false;
            continue;
        }
        if (!ctx->active) {
            continue;
        }
        /* Completion first: a transfer that finished just before a late poll
         * must not be discarded by the watchdog below (polled, no NVIC IRQ). */
        if (!dma_channel_is_busy(ctx->rx_chan)) {
            i2c_dma_on_rx_complete(slot);
            continue;
        }
        /* NACK / arbitration loss: the RX channel will never finish, so abort
         * now instead of waiting out the watchdog (acks + counts TX_ABRT). */
        if (ctx->i2c->hw->raw_intr_stat & I2C_IC_RAW_INTR_STAT_TX_ABRT_BITS) {
            i2c_dma_abort_ctx(ctx, slot);
            continue;
        }
        if ((uint32_t)(time_us_32() - ctx->since_us) > ctx->timeout_us) {
            timeout_count[i]++;
            i2c_dma_abort_ctx(ctx, slot);
        }
    }
    /* Launch any newly-kicked reads (header phase). */
    for (int i = 0; i < 2; i++) {
        i2c_dma_ctx_t *ctx = &slots[i];
        if (!ctx->pending || ctx->active || ctx->i2c == NULL || abort_req[i]) {
            continue;
        }
        ctx->pending = false;
        ctx->active = true;
        ctx->state = I2C_DMA_STATE_HEADER;
        ctx->len = 0;
        read_t0[i] = to_us_since_boot(get_absolute_time());
        i2c_dma_start_read(ctx, (i2c_dma_slot_t)i, 4u);
    }
}
