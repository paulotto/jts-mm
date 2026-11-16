// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Fix package B: SHTP / SH-2 parser (bno085_parse_packet) and the per-report
 * sequence tracker. Pure logic, no bus.
 */

#include "fix_B_check.h"
#include "jtsmm/drivers/bno085.h"

#include <string.h>

typedef struct {
    int n;
    bno085_report_t r[16];
} collect_t;

static void collect_cb(void *ctx, const bno085_report_t *r) {
    collect_t *c = (collect_t *)ctx;
    if (c->n < 16) {
        c->r[c->n] = *r;
    }
    c->n++;
}

/* Packet builder: header + payload appended piecewise. */
typedef struct {
    uint8_t b[BNO085_SHTP_MAX_PACKET];
    uint16_t len;
} pkt_t;

static void pkt_begin(pkt_t *p, uint8_t channel, uint8_t seq) {
    memset(p, 0, sizeof(*p));
    p->b[2] = channel;
    p->b[3] = seq;
    p->len = 4;
}
static void pkt_put(pkt_t *p, const uint8_t *d, uint16_t n) {
    memcpy(p->b + p->len, d, n);
    p->len = (uint16_t)(p->len + n);
    p->b[0] = (uint8_t)(p->len & 0xFFu);
    p->b[1] = (uint8_t)((p->len >> 8) & 0x7Fu);
}
static void put_le16(uint8_t *d, int16_t v) {
    d[0] = (uint8_t)((uint16_t)v & 0xFFu);
    d[1] = (uint8_t)(((uint16_t)v >> 8) & 0xFFu);
}
/* One 10-byte motion report [id][seq][status][delay][x][y][z]. */
static void pkt_motion(pkt_t *p, uint8_t id, uint8_t seq, uint8_t status, int16_t x, int16_t y, int16_t z) {
    uint8_t r[10] = {id, seq, status, 0};
    put_le16(r + 4, x);
    put_le16(r + 6, y);
    put_le16(r + 8, z);
    pkt_put(p, r, sizeof(r));
}
static void pkt_timestamp(pkt_t *p, uint8_t id) {
    uint8_t r[5] = {id, 0x10, 0x00, 0x00, 0x00};
    pkt_put(p, r, sizeof(r));
}

static void test_multi_report_batch(void) {
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT, 7);
    pkt_timestamp(&p, BNO085_REPORT_BASE_TIMESTAMP);
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 10, 0x03, 1, 2, 3);
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 11, 0x02, 4, 5, 6);
    uint8_t grv[12] = {BNO085_REPORT_GAME_ROT_VEC, 20, 0x01, 0};
    put_le16(grv + 4, 100);
    put_le16(grv + 6, -200);
    put_le16(grv + 8, 300);
    put_le16(grv + 10, 16384);
    pkt_put(&p, grv, sizeof(grv));
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 12, 0x03, 7, 8, 9);

    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 4);
    CHECK_EQ(info.reports, 4);
    CHECK(!info.truncated);
    CHECK_EQ(info.channel, BNO085_CH_INPUT);
    CHECK_EQ(info.shtp_seq, 7);
    CHECK_EQ(c.r[0].report_id, BNO085_REPORT_ACCELEROMETER);
    CHECK_EQ(c.r[0].v[0], 1);
    CHECK_EQ(c.r[0].accuracy, 3);
    CHECK_EQ(c.r[0].seq, 10);
    CHECK(c.r[0].has_seq);
    CHECK_EQ(c.r[1].v[2], 6);
    CHECK_EQ(c.r[1].accuracy, 2);
    CHECK_EQ(c.r[2].report_id, BNO085_REPORT_GAME_ROT_VEC);
    CHECK_EQ(c.r[2].v[1], -200);
    CHECK_EQ(c.r[2].v[3], 16384);
    CHECK_EQ(c.r[3].seq, 12);
    CHECK_EQ(c.r[3].v[2], 9);
}

static void test_timestamp_rebase_mid_packet(void) {
    /* 0xFA used to be unknown: everything after it was discarded. */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT, 0);
    pkt_timestamp(&p, BNO085_REPORT_BASE_TIMESTAMP);
    pkt_motion(&p, BNO085_REPORT_GYROSCOPE, 1, 0x03, 1, 1, 1);
    pkt_timestamp(&p, BNO085_REPORT_TIMESTAMP_REBASE);
    pkt_motion(&p, BNO085_REPORT_GYROSCOPE, 2, 0x03, 2, 2, 2);
    pkt_timestamp(&p, BNO085_REPORT_BASE_TIMESTAMP);
    pkt_motion(&p, BNO085_REPORT_LINEAR_ACCEL, 5, 0x01, 256, -512, 0);

    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 3);
    CHECK(!info.truncated);
    CHECK_EQ(c.r[1].v[0], 2);
    CHECK_EQ(c.r[2].report_id, BNO085_REPORT_LINEAR_ACCEL);
    CHECK_EQ(c.r[2].v[1], -512);
}

static void test_unknown_and_short_reports(void) {
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT, 0);
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 1, 0x03, 1, 1, 1);
    uint8_t junk[3] = {0x77, 0, 0};
    pkt_put(&p, junk, sizeof(junk));
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 1);
    CHECK(info.truncated);

    /* A catalog report cut short by the packet end is not delivered. */
    pkt_begin(&p, BNO085_CH_INPUT, 0);
    uint8_t cut[6] = {BNO085_REPORT_ACCELEROMETER, 1, 3, 0, 1, 0};
    pkt_put(&p, cut, sizeof(cut));
    memset(&c, 0, sizeof(c));
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 0);
    CHECK(info.truncated);
}

static void test_gyro_rv_channel5(void) {
    /* 1 report: 14 bytes, no id byte. */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_GYRO_RV, 3);
    uint8_t r[14] = {0};
    put_le16(r + 0, 11);
    put_le16(r + 2, -22);
    put_le16(r + 4, 33);
    put_le16(r + 6, 16000);
    put_le16(r + 8, 5);
    pkt_put(&p, r, sizeof(r));
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 1);
    CHECK_EQ(c.r[0].report_id, BNO085_REPORT_GYRO_INT_ROT_VEC);
    CHECK(!c.r[0].has_seq);
    CHECK_EQ(c.r[0].accuracy, 0xFF);
    CHECK_EQ(c.r[0].v[0], 11);
    CHECK_EQ(c.r[0].v[1], -22);
    CHECK_EQ(c.r[0].v[2], 33);
    CHECK_EQ(c.r[0].v[3], 16000);
    CHECK(!info.truncated);

    /* 2 reports per packet. A first byte of 0x2A is data, not an id. */
    pkt_begin(&p, BNO085_CH_GYRO_RV, 4);
    uint8_t r2[14] = {0};
    put_le16(r2 + 0, 0x2A);
    put_le16(r2 + 6, -16384);
    pkt_put(&p, r, sizeof(r));
    pkt_put(&p, r2, sizeof(r2));
    memset(&c, 0, sizeof(c));
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 2);
    CHECK_EQ(c.r[1].v[0], 0x2A);
    CHECK_EQ(c.r[1].v[3], -16384);
    CHECK_EQ(info.shtp_seq, 4);

    /* Trailing partial record is flagged, full ones still delivered. */
    uint8_t tail[5] = {0};
    pkt_put(&p, tail, sizeof(tail));
    memset(&c, 0, sizeof(c));
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 2);
    CHECK(info.truncated);
}

static void test_continuation(void) {
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT, 9);
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 1, 0x03, 1, 1, 1);
    p.b[1] |= 0x80u; /* continuation bit */
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(!bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK(info.continuation);
    CHECK_EQ(c.n, 0);
    CHECK_EQ(info.reports, 0);
}

/* The bench packets (2026-09-25): a 21-byte game RV report with timebase, as
 * re-read after the 4-byte header pre-read -- flagged as a continuation by the
 * device. The flag is ours to undo; a pre-read that was flagged already is a
 * genuine tail and must stay rejected. */
static void test_reread_continuation_flag(void) {
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT, 3);
    pkt_timestamp(&p, BNO085_REPORT_BASE_TIMESTAMP);
    uint8_t grv[12] = {BNO085_REPORT_GAME_ROT_VEC, 20, 0x03, 0, 1, 0, 2, 0, 3, 0, 0x00, 0x40};
    pkt_put(&p, grv, sizeof(grv));
    CHECK_EQ(p.len, 21);
    uint8_t preread_b1 = p.b[1];     /* fresh packet: not flagged */
    p.b[1] |= 0x80u;                 /* what the full re-read returns */
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(!bno085_parse_packet(p.b, p.len, collect_cb, &c, &info)); /* the bench symptom */
    bno085_shtp_clear_reread_continuation(p.b, preread_b1);
    memset(&c, 0, sizeof(c));
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK(!info.continuation);
    CHECK_EQ(c.n, 1);
    CHECK_EQ(c.r[0].report_id, BNO085_REPORT_GAME_ROT_VEC);

    /* Genuine continuation: already flagged in the pre-read -> kept, rejected. */
    p.b[1] |= 0x80u;
    bno085_shtp_clear_reread_continuation(p.b, (uint8_t)(preread_b1 | 0x80u));
    memset(&c, 0, sizeof(c));
    CHECK(!bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK(info.continuation);
    CHECK_EQ(c.n, 0);
}

static void test_control_and_reset(void) {
    /* Get Feature Response (the reply to Set Feature): granted interval. */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_CONTROL, 0);
    uint8_t fc[17] = {BNO085_REPORT_GET_FEATURE_RESP, BNO085_REPORT_GAME_ROT_VEC, 0, 0, 0,
                      0xC4, 0x09, 0x00, 0x00 /* 2500 us */};
    pkt_put(&p, fc, sizeof(fc));
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK(info.feature_resp);
    CHECK_EQ(info.feature_report_id, BNO085_REPORT_GAME_ROT_VEC);
    CHECK_EQ(info.feature_interval_us, 2500);
    CHECK(!info.command_resp);
    CHECK_EQ(c.n, 1);
    CHECK_EQ(c.r[0].report_id, BNO085_REPORT_GET_FEATURE_RESP);
    CHECK_EQ(c.r[0].feature_id, BNO085_REPORT_GAME_ROT_VEC);
    CHECK_EQ(c.r[0].interval_us, 2500);

    /* Two responses in one packet: both delivered. */
    uint8_t fc2[17] = {BNO085_REPORT_GET_FEATURE_RESP, BNO085_REPORT_LINEAR_ACCEL, 0, 0, 0,
                       0xA0, 0x0F, 0x00, 0x00 /* 4000 us */};
    pkt_put(&p, fc2, sizeof(fc2));
    memset(&c, 0, sizeof(c));
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 2);
    CHECK_EQ(c.r[1].feature_id, BNO085_REPORT_LINEAR_ACCEL);
    CHECK_EQ(c.r[1].interval_us, 4000);

    pkt_begin(&p, BNO085_CH_CONTROL, 0);
    uint8_t pid[16] = {BNO085_REPORT_PRODUCT_ID_RESP};
    pkt_put(&p, pid, sizeof(pid));
    CHECK(bno085_parse_packet(p.b, p.len, NULL, NULL, &info));
    CHECK(info.product_id);

    pkt_begin(&p, BNO085_CH_EXECUTABLE, 0);
    uint8_t rc = BNO085_EXEC_RESET_COMPLETE;
    pkt_put(&p, &rc, 1);
    CHECK(bno085_parse_packet(p.b, p.len, NULL, NULL, &info));
    CHECK(info.reset_complete);

    pkt_begin(&p, BNO085_CH_COMMAND, 0);
    uint8_t adv[8] = {BNO085_CMD_ADVERTISEMENT, 0, 1, 2};
    pkt_put(&p, adv, sizeof(adv));
    CHECK(bno085_parse_packet(p.b, p.len, NULL, NULL, &info));
    CHECK(info.advertisement);
}

static void test_legacy_wrapper(void) {
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT, 0);
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 1, 0x03, 1, 1, 1);
    pkt_motion(&p, BNO085_REPORT_ACCELEROMETER, 2, 0x01, 5, 5, 5);
    bno085_parse_state_t st;
    memset(&st, 0, sizeof(st));
    CHECK(bno085_parse_shtp(p.b, p.len, &st));
    CHECK(st.got_accel);
    CHECK_EQ(st.accel_x, 5); /* last wins */
    CHECK_EQ(st.accel_accuracy, 1);
    CHECK_EQ(st.gyro_accuracy, 0xFF);
}

static void test_seq_tracker(void) {
    bno085_seq_tracker_t t;
    memset(&t, 0, sizeof(t));
    CHECK_EQ(bno085_seq_update(&t, 0x08, 10), 0); /* first sight */
    CHECK_EQ(bno085_seq_update(&t, 0x08, 11), 0);
    CHECK_EQ(bno085_seq_update(&t, 0x08, 14), 2); /* 12, 13 lost */
    CHECK_EQ(bno085_seq_update(&t, 0x08, 14), 0); /* repeat */
    CHECK_EQ(bno085_seq_update(&t, 0x04, 200), 0); /* ids independent */
    CHECK_EQ(bno085_seq_update(&t, 0x08, 15), 0);
    /* Wrap at 255 -> 0 is contiguous; 254 -> 1 lost 255 and 0. */
    CHECK_EQ(bno085_seq_update(&t, 0x01, 255), 0);
    CHECK_EQ(bno085_seq_update(&t, 0x01, 0), 0);
    CHECK_EQ(bno085_seq_update(&t, 0x01, 254), 253);
    CHECK_EQ(bno085_seq_update(&t, 0x01, 1), 2);
    /* Untracked ids never count. */
    CHECK_EQ(bno085_seq_update(&t, 0xF0, 1), 0);
    CHECK_EQ(bno085_seq_update(&t, 0xF0, 9), 0);
    /* After a reset every id starts fresh. */
    bno085_seq_reset(&t);
    CHECK_EQ(bno085_seq_update(&t, 0x08, 0), 0);
    CHECK_EQ(bno085_seq_update(&t, 0x08, 1), 0);
}

int main(void) {
    test_multi_report_batch();
    test_timestamp_rebase_mid_packet();
    test_unknown_and_short_reports();
    test_gyro_rv_channel5();
    test_continuation();
    test_reread_continuation_flag();
    test_control_and_reset();
    test_legacy_wrapper();
    test_seq_tracker();
    return fixb_report("test_fix_B_shtp");
}
