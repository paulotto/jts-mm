// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file host_link.c
 * @brief USB framed record TX and CRC helpers.
 */

#include "jtsmm/runtime/host_link.h"

#include "jtsmm/records.h"
#include "pico/stdlib.h"
#include "tusb.h"
#include <stdio.h>
#include <string.h>

static uint32_t tx_bytes;
static uint32_t tx_drops;
static uint32_t tx_drops_emg;
static uint32_t tx_drops_imu;
static uint32_t tx_drops_other;
static uint32_t crc_errors;

/*
 * Bytes of vendor TX FIFO kept free for non-EMG (rate-critical) frames. Sized to
 * hold a comfortable burst of IMU/trigger/status frames (~100+ frames) so they
 * survive a host read stall of tens of milliseconds. Must stay well below
 * CFG_TUD_VENDOR_TX_BUFSIZE so EMG still gets the bulk of the pipe.
 */
#define HOST_LINK_CRITICAL_RESERVE 8192u

#define FRAME_OVERHEAD 14u /* sync(2) + type + len + t_master_us(8) + crc(2) */

uint16_t host_link_crc16(const uint8_t *data, size_t len) {
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)data[i] << 8;
        for (int b = 0; b < 8; b++) {
            if (crc & 0x8000u) {
                crc = (uint16_t)((crc << 1) ^ 0x1021u);
            } else {
                crc <<= 1;
            }
        }
    }
    return crc;
}

void host_link_init(void) {
    tx_bytes = 0;
    crc_errors = 0;
    host_link_reset_tx_stats();
}

void host_link_reset_tx_stats(void) {
    tx_drops = 0;
    tx_drops_emg = 0;
    tx_drops_imu = 0;
    tx_drops_other = 0;
}

/** @brief Internal helper: count one dropped frame against its stream. */
static void count_drop(uint8_t type) {
    tx_drops++;
    if (type == REC_EMG_BLOCK) {
        tx_drops_emg++;
    } else if (type >= REC_IMU_MTA_QUAT && type <= REC_IMU_CRA_LIN_ACCEL) {
        tx_drops_imu++;
    } else {
        tx_drops_other++;
    }
}

/*
 * Reserve the tail of the vendor TX FIFO exclusively for the low-volume,
 * strict-budget streams (IMU quaternions, triggers, stats, acks). EMG is
 * high volume (1-2 kHz) but tolerant (>=99% rate budget with a deep ring),
 * so when the FIFO starts to back up under host read jitter EMG backs off
 * FIRST and space stays free for the ~400 Hz IMU streams. Without this the
 * indiscriminate "FIFO full -> drop whole frame" policy dropped IMU frames
 * in proportion to EMG volume, pushing IMU past its 0.5% integrity budget.
 */
static bool frame_fits(uint8_t type, uint16_t len) {
    uint32_t need = FRAME_OVERHEAD + (uint32_t)len;
    if (type == REC_EMG_BLOCK) {
        need += HOST_LINK_CRITICAL_RESERVE;
    }
    return tud_vendor_write_available() >= need;
}

/** @brief Internal helper: frame one record and commit it (space already checked). */
static void write_frame(uint8_t type, uint64_t t_master_us, const uint8_t *payload, uint16_t len) {
    uint8_t frame[12 + JTSMM_MAX_PAYLOAD + 2];
    frame[0] = JTSMM_SYNC0;
    frame[1] = JTSMM_SYNC1;
    frame[2] = type;
    frame[3] = (uint8_t)len;
    for (int i = 0; i < 8; i++) {
        frame[4 + i] = (uint8_t)(t_master_us >> (8 * i));
    }
    if (len > 0 && payload != NULL) {
        memcpy(frame + 12, payload, len);
    }
    uint16_t crc = host_link_crc16(frame + 2, (size_t)(10 + len));
    frame[12 + len] = (uint8_t)(crc & 0xFFu);
    frame[12 + len + 1] = (uint8_t)((crc >> 8) & 0xFFu);

    uint32_t total = FRAME_OVERHEAD + (uint32_t)len;
    tud_vendor_write(frame, total);
    tx_bytes += total;
}

/*
 * Whole framed records are written to the vendor bulk IN endpoint. The write is
 * strictly non-blocking: we only commit a frame if the TX FIFO can hold it in
 * full, otherwise we drop the whole frame (counted in tx_drops and its per-
 * stream counter) and return.
 *
 * Never write a partial frame (that would desync the host parser), and never
 * spin waiting for FIFO space -- if the host stalls or stops reading the bulk
 * pipe, blocking here would starve the comms core (CDC commands, config apply,
 * panel), which is exactly the wedge this replaces. The comms loop pumps
 * tud_task() every iteration, so FIFO space frees up as the host drains it.
 *
 * Ring records do not come through here: host_link_drain() checks for space
 * before it takes a record, so back-pressure stays in the ring instead.
 */
bool host_link_send_record(uint8_t type, uint64_t t_master_us, const uint8_t *payload, uint16_t len) {
    if (len > JTSMM_MAX_PAYLOAD || !tud_vendor_mounted() || !frame_fits(type, len)) {
        count_drop(type);
        return false;
    }
    write_frame(type, t_master_us, payload, len);
    return true;
}

void host_link_send_ack(bool ok) {
    uint8_t v = ok ? 1u : 0u;
    host_link_send_record(REC_CMD_ACK, 0, &v, 1);
}

void host_link_send_log(const char *msg) {
    if (msg == NULL) {
        return;
    }
    size_t n = strlen(msg);
    if (n > JTSMM_MAX_PAYLOAD) {
        n = JTSMM_MAX_PAYLOAD;
    }
    host_link_send_record(REC_LOG, 0, (const uint8_t *)msg, (uint16_t)n);
}

void host_link_drain(void) {
    /*
     * Drain the low-volume, rate-critical IMU and event rings BEFORE the high-
     * volume EMG ring, so under pressure they win FIFO space. At high EMG rates
     * (>=2 kHz) EMG would otherwise fill the FIFO and starve the ~400 Hz IMU
     * quaternion streams, pushing them past the 0.5% integrity drop budget.
     *
     * A record is only taken off its ring once the FIFO can hold it. The old
     * code popped first and then dropped the record when the FIFO was full --
     * a loss that the ring (deep, sized exactly for host stalls) would have
     * absorbed. Now a full FIFO leaves the record queued and moves on to the
     * next ring (a smaller or reserve-exempt frame may still fit); only if the
     * ring itself overflows is data lost, and that is counted by the producer
     * (ringbuf_drops). With no host mounted nobody can read the data, so rings
     * are drained and the frames counted as tx drops, as before.
     */
    static const ring_id_t order[] = {RING_IMU_MTA, RING_IMU_CRA, RING_EVENTS, RING_EMG};
    const unsigned per_ring = 8u;
    /*
     * Bound total work per call so a continuously-producing sensor core can
     * never trap the comms loop here (it must return to CDC/command handling).
     * Remaining ring contents are drained on the next comms iteration.
     */
    const unsigned max_records = 256u;
    unsigned total = 0;
    bool mounted = tud_vendor_mounted();
    bool progress;
    do {
        progress = false;
        for (size_t i = 0; i < sizeof(order) / sizeof(order[0]); i++) {
            for (unsigned n = 0; n < per_ring; n++) {
                ringbuf_view_t rec;
                if (!ringbuf_peek(order[i], &rec)) {
                    break;
                }
                if (!mounted || rec.len > JTSMM_MAX_PAYLOAD) {
                    count_drop(rec.type);
                } else if (frame_fits(rec.type, rec.len)) {
                    write_frame(rec.type, rec.t_master_us, rec.payload, rec.len);
                } else {
                    break; /* keep it queued; try the next ring */
                }
                ringbuf_consume(order[i]);
                progress = true;
                if (++total >= max_records) {
                    tud_vendor_write_flush();
                    return;
                }
            }
        }
    } while (progress);
    tud_vendor_write_flush();
}

uint32_t host_link_tx_bytes(void) {
    return tx_bytes;
}

uint32_t host_link_tx_drops(void) {
    return tx_drops;
}

uint32_t host_link_tx_drops_emg(void) {
    return tx_drops_emg;
}

uint32_t host_link_tx_drops_imu(void) {
    return tx_drops_imu;
}

uint32_t host_link_tx_drops_other(void) {
    return tx_drops_other;
}

uint32_t host_link_crc_errors(void) {
    return crc_errors;
}
