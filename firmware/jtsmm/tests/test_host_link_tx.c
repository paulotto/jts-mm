// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * host_link TX policy against a stubbed vendor FIFO (tests/stubs/tusb_stub.c):
 * full-frame golden, "no room -> record stays in the ring", EMG reserve,
 * per-stream drop counters and their reset.
 */
#include "jtsmm/records.h"
#include "jtsmm/runtime/host_link.h"
#include "jtsmm/runtime/ringbuf.h"
#include "tusb.h"

#include "check.h"

#include <string.h>

#define RESERVE 8192u /* HOST_LINK_CRITICAL_RESERVE */

static void push(ring_id_t ring, uint8_t type, uint16_t len, uint64_t t) {
    jtsmm_record_t rec = {0};
    rec.type = type;
    rec.len = len;
    rec.t_master_us = t;
    memset(rec.payload, (int)(t & 0xFFu), len);
    CHECK(ringbuf_push(ring, &rec));
}

static void test_frame_golden(void) {
    stub_tud_reset(0xFFFFFFFFu);
    const uint8_t ok = 1;
    CHECK(host_link_send_record(REC_CMD_ACK, 0x0102030405060708ull, &ok, 1));
    /* Computed independently (Python, CRC-16/CCITT-FALSE over type..payload). */
    const uint8_t golden[] = {0x55, 0xAA, 0xE0, 0x01, 0x08, 0x07, 0x06, 0x05,
                              0x04, 0x03, 0x02, 0x01, 0x01, 0x26, 0x8F};
    CHECK_EQ_U(stub_tud_written_len, sizeof(golden));
    CHECK(memcmp(stub_tud_written, golden, sizeof(golden)) == 0);
    CHECK_EQ_U(host_link_tx_bytes(), sizeof(golden));
}

/* The bug: drain popped a record and then dropped it when the FIFO was full. */
static void test_full_fifo_keeps_records(void) {
    ringbuf_init_all();
    host_link_init();
    for (uint64_t i = 0; i < 5; i++) {
        push(RING_IMU_MTA, REC_IMU_MTA_QUAT, 18, i);
    }
    /* Room for exactly three 32-byte IMU frames. */
    stub_tud_reset(3u * (14u + 18u));
    host_link_drain();
    CHECK_EQ_U(stub_tud_written_len, 3u * 32u);
    CHECK_EQ_U(ringbuf_count(RING_IMU_MTA), 2);
    CHECK_EQ_U(host_link_tx_drops(), 0);

    /* Host catches up: the two remaining records go out, in order. */
    stub_tud_reset(0xFFFFFFFFu);
    host_link_drain();
    CHECK_EQ_U(ringbuf_count(RING_IMU_MTA), 0);
    CHECK_EQ_U(stub_tud_written_len, 2u * 32u);
    CHECK_EQ_U(stub_tud_written[4], 3); /* t_master_us LSB of the 4th record */
    CHECK_EQ_U(host_link_tx_drops(), 0);
}

/* EMG backs off while the FIFO is inside the reserve; IMU still goes out. */
static void test_emg_reserve(void) {
    ringbuf_init_all();
    host_link_init();
    push(RING_EMG, REC_EMG_BLOCK, 27, 1);
    push(RING_IMU_CRA, REC_IMU_CRA_QUAT, 18, 2);
    stub_tud_reset(RESERVE + 14u + 27u - 1u);
    host_link_drain();
    CHECK_EQ_U(ringbuf_count(RING_IMU_CRA), 0);
    CHECK_EQ_U(ringbuf_count(RING_EMG), 1);
    CHECK_EQ_U(stub_tud_written_len, 32u);
    CHECK_EQ_U(host_link_tx_drops_emg(), 0);

    stub_tud_reset(RESERVE + 14u + 27u);
    host_link_drain();
    CHECK_EQ_U(ringbuf_count(RING_EMG), 0);
    CHECK_EQ_U(stub_tud_written_len, 41u);
}

/* Unmounted: nobody can read, so rings drain and drops are counted per stream. */
static void test_unmounted_and_counters(void) {
    ringbuf_init_all();
    host_link_init();
    push(RING_EMG, REC_EMG_BLOCK, 27, 1);
    push(RING_EMG, REC_EMG_BLOCK, 27, 2);
    push(RING_IMU_MTA, REC_IMU_MTA_LIN_ACCEL, 13, 3);
    push(RING_EVENTS, REC_TRIGGER_EVENT, 12, 4);
    stub_tud_reset(0xFFFFFFFFu);
    stub_tud_mounted = 0;
    host_link_drain();
    CHECK_EQ_U(stub_tud_written_len, 0);
    CHECK_EQ_U(ringbuf_count(RING_EMG) + ringbuf_count(RING_IMU_MTA) + ringbuf_count(RING_EVENTS), 0);
    CHECK_EQ_U(host_link_tx_drops_emg(), 2);
    CHECK_EQ_U(host_link_tx_drops_imu(), 1);
    CHECK_EQ_U(host_link_tx_drops_other(), 1);

    /* Direct sends (logs, acks, stats) with no FIFO space count as "other". */
    stub_tud_reset(10u);
    host_link_send_log("does not fit");
    host_link_send_ack(true);
    CHECK_EQ_U(host_link_tx_drops_other(), 3);
    CHECK_EQ_U(host_link_tx_drops(), host_link_tx_drops_emg() + host_link_tx_drops_imu() +
                                         host_link_tx_drops_other());

    /* Stream-start reset: drops to zero, byte counter keeps running. */
    stub_tud_reset(0xFFFFFFFFu);
    host_link_send_ack(true);
    uint32_t bytes = host_link_tx_bytes();
    CHECK(bytes > 0);
    host_link_reset_tx_stats();
    CHECK_EQ_U(host_link_tx_drops(), 0);
    CHECK_EQ_U(host_link_tx_drops_emg(), 0);
    CHECK_EQ_U(host_link_tx_drops_imu(), 0);
    CHECK_EQ_U(host_link_tx_drops_other(), 0);
    CHECK_EQ_U(host_link_tx_bytes(), bytes);
}

int main(void) {
    test_frame_golden();
    test_full_fifo_keeps_records();
    test_emg_reserve();
    test_unmounted_and_counters();
    CHECK_DONE("test_host_link_tx");
}
