// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#include "jtsmm/runtime/ringbuf.h"

#include "check.h"

#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static jtsmm_record_t make_rec(uint64_t n) {
    jtsmm_record_t rec = {0};
    rec.type = 0x10;
    rec.t_master_us = n;
    rec.len = 1;
    rec.payload[0] = (uint8_t)n;
    return rec;
}

static bool push_n(ring_id_t id, uint64_t n) {
    jtsmm_record_t rec = make_rec(n);
    return ringbuf_push(id, &rec);
}

static void test_basic(void) {
    ringbuf_init_all();
    bool pushed = push_n(RING_EMG, 42);
    CHECK(pushed);
    jtsmm_record_t out = {0};
    bool popped = ringbuf_pop(RING_EMG, &out);
    CHECK(popped);
    CHECK_EQ_U(out.t_master_us, 42);
    CHECK_EQ_U(ringbuf_drops(RING_EMG), 0);
    CHECK_EQ_U(ringbuf_pushes(RING_EMG), 1);
    popped = ringbuf_pop(RING_EMG, &out);
    CHECK(!popped);
}

static void test_capacity_edges(void) {
    ringbuf_init_all();
    /* Below the minimum: a 1-slot SPSC ring holds nothing, so it is raised. */
    CHECK_EQ_U(ringbuf_set_capacity(RING_EMG, 0u), RINGBUF_MIN_CAPACITY);
    CHECK_EQ_U(ringbuf_set_capacity(RING_EMG, 1u), RINGBUF_MIN_CAPACITY);
    CHECK_EQ_U(ringbuf_capacity(RING_EMG), RINGBUF_MIN_CAPACITY);
    /* Capacity 2 holds exactly one record. */
    CHECK(push_n(RING_EMG, 1));
    CHECK(!push_n(RING_EMG, 2));
    CHECK_EQ_U(ringbuf_drops(RING_EMG), 1);
    CHECK_EQ_U(ringbuf_count(RING_EMG), 1);

    /* Above the backing store: clamped, and the caller is told. */
    ringbuf_init_all();
    uint32_t imu_max = ringbuf_max_capacity(RING_IMU_MTA);
    CHECK(imu_max >= RINGBUF_MIN_CAPACITY);
    CHECK_EQ_U(ringbuf_set_capacity(RING_IMU_MTA, imu_max + 1u), imu_max);
    int imu_pushed = 0;
    for (uint32_t i = 0; i < imu_max + 10u; i++) {
        if (push_n(RING_IMU_MTA, i)) {
            imu_pushed++;
        }
    }
    CHECK_EQ_U(imu_pushed, imu_max - 1u);
    CHECK_EQ_U(ringbuf_drops(RING_IMU_MTA), 11u);

    /* The deep EMG store is honoured (2048 compact slots ~ 1.0 s at 2 kHz). */
    ringbuf_init_all();
    CHECK_EQ_U(ringbuf_max_capacity(RING_EMG), 2048u);
    CHECK_EQ_U(ringbuf_set_capacity(RING_EMG, 2048u), 2048u);
    int emg_pushed = 0;
    for (int i = 0; i < 2100; i++) {
        if (push_n(RING_EMG, (uint64_t)i)) {
            emg_pushed++;
        }
    }
    CHECK_EQ_U(emg_pushed, 2047);
    CHECK_EQ_U(ringbuf_count(RING_EMG), 2047);
    CHECK_EQ_U(ringbuf_drops(RING_EMG), 2100u - 2047u);

    /* Invalid id. */
    ringbuf_view_t v;
    CHECK_EQ_U(ringbuf_set_capacity(RING_COUNT, 8u), 0u);
    CHECK(!ringbuf_peek(RING_COUNT, &v));
    CHECK_EQ_U(ringbuf_slot_bytes(RING_COUNT), 0u);
}

/* FIFO order must hold across many index wraps with the fill level varying. */
static void test_wrap_interleave(void) {
    ringbuf_init_all();
    CHECK_EQ_U(ringbuf_set_capacity(RING_EVENTS, 5u), 5u);
    uint64_t next_push = 0;
    uint64_t next_pop = 0;
    int order_errors = 0;
    for (int round = 0; round < 200; round++) {
        int burst = 1 + (round % 4); /* 1..4 = up to capacity-1 */
        for (int i = 0; i < burst; i++) {
            if (push_n(RING_EVENTS, next_push)) {
                next_push++;
            }
        }
        int take = 1 + ((round * 7) % 4);
        for (int i = 0; i < take; i++) {
            jtsmm_record_t out;
            if (!ringbuf_pop(RING_EVENTS, &out)) {
                break;
            }
            if (out.t_master_us != next_pop || out.payload[0] != (uint8_t)next_pop) {
                order_errors++;
            }
            next_pop++;
        }
        if (ringbuf_count(RING_EVENTS) != (uint32_t)(next_push - next_pop)) {
            order_errors++;
        }
    }
    CHECK_EQ_U(order_errors, 0);
    CHECK(next_pop > 100u); /* the loop really wrapped the 5-slot ring */
    CHECK_EQ_U(ringbuf_pushes(RING_EVENTS), next_push);
}

/* peek leaves the record queued until consume; that is what host_link relies
 * on to keep a record when the USB FIFO has no room for it. */
static void test_peek_consume(void) {
    ringbuf_init_all();
    ringbuf_view_t v;
    CHECK(!ringbuf_peek(RING_IMU_CRA, &v));
    ringbuf_consume(RING_IMU_CRA); /* no-op on empty */
    CHECK_EQ_U(ringbuf_count(RING_IMU_CRA), 0);
    CHECK(push_n(RING_IMU_CRA, 7));
    CHECK(push_n(RING_IMU_CRA, 8));
    CHECK(ringbuf_peek(RING_IMU_CRA, &v) && v.t_master_us == 7);
    CHECK(ringbuf_peek(RING_IMU_CRA, &v) && v.t_master_us == 7);
    CHECK_EQ_U(ringbuf_count(RING_IMU_CRA), 2);
    ringbuf_consume(RING_IMU_CRA);
    CHECK(ringbuf_peek(RING_IMU_CRA, &v) && v.t_master_us == 8 && v.payload[0] == 8);
    ringbuf_consume(RING_IMU_CRA);
    CHECK(!ringbuf_peek(RING_IMU_CRA, &v));
}

/* A slot carries header + len payload bytes exactly (push copies only those).
 * Slots are sized per ring: the compact EMG ring takes an EMG frame (31 B) but
 * refuses anything longer than RINGBUF_EMG_PAYLOAD_MAX, counted as a drop;
 * the full-record rings take up to 240 B. Nothing overflows a slot. */
static void test_partial_copy(void) {
    ringbuf_init_all();
    CHECK_EQ_U(ringbuf_slot_bytes(RING_EMG), 48u);
    CHECK_EQ_U(ringbuf_slot_bytes(RING_IMU_MTA), 256u);
    ringbuf_view_t v;

    jtsmm_record_t big = make_rec(1);
    big.len = 240;
    memset(big.payload, 0xEE, sizeof(big.payload));
    CHECK(ringbuf_push(RING_IMU_MTA, &big));
    CHECK(ringbuf_peek(RING_IMU_MTA, &v) && v.len == 240 && v.payload[239] == 0xEE);
    ringbuf_consume(RING_IMU_MTA);

    CHECK(!ringbuf_push(RING_EMG, &big)); /* 240 B does not fit a compact slot */
    big.len = RINGBUF_EMG_PAYLOAD_MAX + 1u;
    CHECK(!ringbuf_push(RING_EMG, &big));
    CHECK_EQ_U(ringbuf_drops(RING_EMG), 2);
    CHECK_EQ_U(ringbuf_count(RING_EMG), 0);
    big.len = RINGBUF_EMG_PAYLOAD_MAX;
    CHECK(ringbuf_push(RING_EMG, &big));
    CHECK(ringbuf_peek(RING_EMG, &v) && v.len == RINGBUF_EMG_PAYLOAD_MAX &&
          v.payload[RINGBUF_EMG_PAYLOAD_MAX - 1u] == 0xEE);
    ringbuf_consume(RING_EMG);

    /* Neighbouring compact slots do not bleed into each other. */
    jtsmm_record_t rec;
    memset(&rec, 0x5A, sizeof(rec)); /* bytes past len must not matter */
    rec.type = 0x10;
    rec.t_master_us = 0x1122334455667788ull;
    rec.len = 31;
    for (uint8_t i = 0; i < 31; i++) {
        rec.payload[i] = (uint8_t)(i + 1u);
    }
    CHECK(ringbuf_push(RING_EMG, &rec));
    jtsmm_record_t next = make_rec(9);
    next.len = 31;
    memset(next.payload, 0xA5, 31);
    CHECK(ringbuf_push(RING_EMG, &next));
    jtsmm_record_t out;
    memset(&out, 0, sizeof(out));
    CHECK(ringbuf_pop(RING_EMG, &out));
    CHECK_EQ_U(out.type, 0x10);
    CHECK(out.t_master_us == 0x1122334455667788ull);
    CHECK_EQ_U(out.len, 31);
    CHECK(memcmp(out.payload, rec.payload, 31) == 0);
    CHECK(ringbuf_pop(RING_EMG, &out));
    CHECK(out.t_master_us == 9 && out.payload[0] == 0xA5 && out.payload[30] == 0xA5);

    jtsmm_record_t bad = make_rec(2);
    bad.len = 300; /* longer than any slot: refused, never truncated or overflowed */
    CHECK(!ringbuf_push(RING_IMU_MTA, &bad));
    CHECK_EQ_U(ringbuf_drops(RING_IMU_MTA), 1);
}

int main(void) {
    test_basic();
    test_partial_copy();
    test_capacity_edges();
    test_wrap_interleave();
    test_peek_consume();
    CHECK_DONE("test_ringbuf");
}
