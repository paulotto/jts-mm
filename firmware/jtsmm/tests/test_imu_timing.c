// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * IMU sample timestamps: the INT-edge latch state machine and the SH-2
 * base-delta / per-report delay decoding (imu_timing.c + bno085_parse_packet).
 */

#include "fix_B_check.h"
#include "fix_B_fakes.h"

#include "jtsmm/config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/domain/imu_timing.h"
#include "jtsmm/drivers/bno085.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"

#include <string.h>

/* ---- packet builder ------------------------------------------------------ */

typedef struct {
    uint8_t b[BNO085_SHTP_MAX_PACKET];
    uint16_t len;
} pkt_t;

static void pkt_begin(pkt_t *p, uint8_t channel) {
    memset(p, 0, sizeof(*p));
    p->b[2] = channel;
    p->len = 4;
}
static void pkt_put(pkt_t *p, const uint8_t *d, uint16_t n) {
    memcpy(p->b + p->len, d, n);
    p->len = (uint16_t)(p->len + n);
    p->b[0] = (uint8_t)(p->len & 0xFFu);
    p->b[1] = (uint8_t)((p->len >> 8) & 0x7Fu);
}
/* 0xFB / 0xFA record: [id][i32 LE, 100 µs ticks]. */
static void pkt_time(pkt_t *p, uint8_t id, int32_t ticks) {
    uint32_t u = (uint32_t)ticks;
    uint8_t r[5] = {id, (uint8_t)u, (uint8_t)(u >> 8), (uint8_t)(u >> 16), (uint8_t)(u >> 24)};
    pkt_put(p, r, sizeof(r));
}
/* 10-byte motion report with a 14-bit delay and accuracy @p acc. */
static void pkt_motion_d(pkt_t *p, uint8_t id, uint8_t seq, uint16_t delay, uint8_t acc, int16_t x) {
    uint8_t status = (uint8_t)(((delay >> 8) & 0x3Fu) << 2 | (acc & 0x03u));
    uint8_t r[10] = {id, seq, status, (uint8_t)(delay & 0xFFu), (uint8_t)x, (uint8_t)((uint16_t)x >> 8)};
    pkt_put(p, r, sizeof(r));
}

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

/* ---- (a) edge latch ------------------------------------------------------ */

static void test_latch_basic(void) {
    imu_edge_latch_t l;
    memset(&l, 0, sizeof(l));
    uint64_t t = 0;

    /* Nothing latched: poll time. */
    CHECK_EQ(imu_edge_latch_take(&l, 1000, 20000, &t), IMU_TS_SRC_NO_EDGE);
    CHECK_EQ(t, 1000);

    /* An edge, then the loop kicks a read 150 µs later: the edge time wins. */
    imu_edge_latch_on_edge(&l, 5000);
    CHECK(l.pending);
    CHECK_EQ(l.edges, 1);
    CHECK_EQ(imu_edge_latch_take(&l, 5150, 20000, &t), IMU_TS_SRC_EDGE);
    CHECK_EQ(t, 5000);
    CHECK(!l.pending); /* cleared by the take, before the read runs */

    /* A second take without a new edge falls back again. */
    CHECK_EQ(imu_edge_latch_take(&l, 5200, 20000, &t), IMU_TS_SRC_NO_EDGE);
    CHECK_EQ(t, 5200);
}

static void test_latch_first_edge_kept(void) {
    imu_edge_latch_t l;
    memset(&l, 0, sizeof(l));
    uint64_t t = 0;
    imu_edge_latch_on_edge(&l, 100);
    imu_edge_latch_on_edge(&l, 180); /* noise before the take: coalesced */
    CHECK_EQ(l.edges, 2);
    CHECK_EQ(l.coalesced, 1);
    CHECK_EQ(imu_edge_latch_take(&l, 200, 20000, &t), IMU_TS_SRC_EDGE);
    CHECK_EQ(t, 100);
}

static void test_latch_reassert_during_read(void) {
    /* INT asserts at 1000; read kicked at 1040 (take). While the read runs
     * (~1 ms) the next packet re-asserts INT at 1600: it must be latched for
     * the NEXT read, not merged into this one or lost. */
    imu_edge_latch_t l;
    memset(&l, 0, sizeof(l));
    uint64_t t = 0;
    imu_edge_latch_on_edge(&l, 1000);
    CHECK_EQ(imu_edge_latch_take(&l, 1040, 20000, &t), IMU_TS_SRC_EDGE);
    CHECK_EQ(t, 1000);
    imu_edge_latch_on_edge(&l, 1600); /* during the read */
    CHECK_EQ(l.coalesced, 0);
    /* Read completes at 2050 with INT still low (the next packet waits). */
    CHECK(!imu_edge_latch_discard_if_released(&l, true));
    /* Loop comes round at 2100: stamped with the re-assertion, not 2100. */
    CHECK_EQ(imu_edge_latch_take(&l, 2100, 20000, &t), IMU_TS_SRC_EDGE);
    CHECK_EQ(t, 1600);
}

static void test_latch_released_edge_dropped(void) {
    /* An edge whose INT line is high again (glitch, or a header-phase
     * re-assertion whose packet the finished read consumed) is dropped, so
     * the next real edge is latched instead of being coalesced away. */
    imu_edge_latch_t l;
    memset(&l, 0, sizeof(l));
    uint64_t t = 0;
    imu_edge_latch_on_edge(&l, 500);
    CHECK(imu_edge_latch_discard_if_released(&l, false));
    CHECK(!l.pending);
    CHECK(!imu_edge_latch_discard_if_released(&l, false)); /* nothing left */
    imu_edge_latch_on_edge(&l, 2900);
    CHECK_EQ(l.coalesced, 0);
    CHECK_EQ(imu_edge_latch_take(&l, 3000, 20000, &t), IMU_TS_SRC_EDGE);
    CHECK_EQ(t, 2900);
}

static void test_latch_stale_and_future(void) {
    imu_edge_latch_t l;
    memset(&l, 0, sizeof(l));
    uint64_t t = 0;
    /* Stale: 25 ms old against a 20 ms bound -> poll time, consumed. */
    imu_edge_latch_on_edge(&l, 10000);
    CHECK_EQ(imu_edge_latch_take(&l, 35000, 20000, &t), IMU_TS_SRC_STALE);
    CHECK_EQ(t, 35000);
    CHECK(!l.pending);
    /* Exactly at the bound is still accepted. */
    imu_edge_latch_on_edge(&l, 40000);
    CHECK_EQ(imu_edge_latch_take(&l, 60000, 20000, &t), IMU_TS_SRC_EDGE);
    CHECK_EQ(t, 40000);
    /* Future: never used. */
    imu_edge_latch_on_edge(&l, 70001);
    CHECK_EQ(imu_edge_latch_take(&l, 70000, 20000, &t), IMU_TS_SRC_FUTURE);
    CHECK_EQ(t, 70000);
    CHECK(!l.pending);
}

static void test_latch_cleared_on_stream_start(void) {
    /* A pre-stream edge (boot/apply traffic, a bench tap) must not stamp the
     * first packet of the stream. */
    imu_edge_latch_t l;
    memset(&l, 0, sizeof(l));
    uint64_t t = 0;
    imu_edge_latch_on_edge(&l, 100);
    imu_edge_latch_clear(&l);
    CHECK(!l.pending);
    CHECK_EQ(imu_edge_latch_take(&l, 5000, 20000, &t), IMU_TS_SRC_NO_EDGE);
    CHECK_EQ(t, 5000);
    CHECK_EQ(l.edges, 1); /* the counter keeps counting (imu_status / imu_int_watch) */
}

static void test_max_age(void) {
    CHECK_EQ(imu_ts_edge_max_age_us(0), IMU_TS_EDGE_MAX_AGE_FLOOR_US);
    CHECK_EQ(imu_ts_edge_max_age_us(2500), IMU_TS_EDGE_MAX_AGE_FLOOR_US); /* 400 Hz */
    CHECK_EQ(imu_ts_edge_max_age_us(10000), 30000);                       /* 100 Hz */
    CHECK_EQ(imu_ts_edge_max_age_us(UINT32_MAX), UINT32_MAX);
}

static void test_stats_latency_window(void) {
    imu_ts_stats_t s;
    imu_ts_stats_reset(&s);
    imu_ts_stats_on_read(&s, IMU_TS_SRC_NO_EDGE, 1000, 1000);
    imu_ts_stats_on_read(&s, IMU_TS_SRC_STALE, 1000, 1000);
    CHECK_EQ(s.src[IMU_TS_SRC_NO_EDGE], 1);
    CHECK_EQ(s.src[IMU_TS_SRC_STALE], 1);
    CHECK_EQ(s.lat_n, 0); /* only edge reads have a latency */
    for (unsigned i = 0; i < IMU_TS_STATS_WINDOW; i++) {
        uint32_t lat = (i == 7) ? 900u : 20u;
        imu_ts_stats_on_read(&s, IMU_TS_SRC_EDGE, 100000u + i * 2500u, 100000u + i * 2500u - lat);
    }
    CHECK(s.lat_w_valid);
    CHECK_EQ(s.lat_max_w, 900);
    CHECK_EQ(s.lat_mean_w, (20u * (IMU_TS_STATS_WINDOW - 1u) + 900u) / IMU_TS_STATS_WINDOW);
    CHECK_EQ(s.lat_max_all, 900);
    CHECK_EQ(s.lat_n, 0);
    CHECK_EQ(s.src[IMU_TS_SRC_EDGE], IMU_TS_STATS_WINDOW);
}

/* ---- (b) SH-2 timing ----------------------------------------------------- */

static void test_delay_bit_layout(void) {
    /* delay 0x2A5C: bits 13:8 = 0x2A -> status[7:2]; bits 7:0 = 0x5C -> byte 3. */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT);
    pkt_time(&p, BNO085_REPORT_BASE_TIMESTAMP, 0x3000);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 1, 0x2A5C, 2, 7);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 2, 0x3FFF, 1, 8); /* 14-bit max */
    CHECK_EQ(p.b[4 + 5 + 2], (0x2A << 2) | 2);
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 2);
    CHECK_EQ(c.r[0].delay, 0x2A5C);
    CHECK_EQ(c.r[0].accuracy, 2); /* status bits 1:0 unaffected */
    CHECK_EQ(c.r[0].v[0], 7);
    CHECK(c.r[0].has_ts);
    CHECK_EQ(c.r[0].ts_ticks, 0x2A5C - 0x3000);
    CHECK_EQ(c.r[1].delay, 0x3FFF);
    CHECK_EQ(c.r[1].accuracy, 1);
    CHECK(info.has_base);
    CHECK_EQ(info.base_ticks, 0x3000);
    CHECK(!info.has_rebase);
}

static void test_distinct_increasing_times(void) {
    /* INT at t_int; base delta 3.0 ms; three accel samples 1.0 ms apart
     * (delay 0, 10, 20 ticks) -> t_int - 3.0, -2.0, -1.0 ms. */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT);
    pkt_time(&p, BNO085_REPORT_BASE_TIMESTAMP, 30);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 1, 0, 3, 1);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 2, 10, 3, 2);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 3, 20, 3, 3);
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 3);
    const uint64_t t_int = 10000000u;
    imu_ts_stats_t s;
    imu_ts_stats_reset(&s);
    uint64_t t0 = imu_ts_report_time(t_int, &c.r[0], &s);
    uint64_t t1 = imu_ts_report_time(t_int, &c.r[1], &s);
    uint64_t t2 = imu_ts_report_time(t_int, &c.r[2], &s);
    CHECK_EQ(t0, t_int - 3000);
    CHECK_EQ(t1, t_int - 2000);
    CHECK_EQ(t2, t_int - 1000);
    CHECK(t0 < t1 && t1 < t2 && t2 <= t_int);
    CHECK_EQ(s.clamp_late + s.clamp_early, 0);
    imu_ts_stats_on_packet(&s, &info);
    CHECK_EQ(s.base_last, 3000);
    CHECK_EQ(s.no_base_pkts, 0);
}

static void test_rebase(void) {
    /* 0xFA shifts the reference for the reports after it (CEVA sh2.c:
     * referenceDelta = -base, then += rebase). */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT);
    pkt_time(&p, BNO085_REPORT_BASE_TIMESTAMP, 50);
    pkt_motion_d(&p, BNO085_REPORT_GYROSCOPE, 1, 5, 3, 1);  /* -50 + 5 = -45 */
    pkt_time(&p, BNO085_REPORT_TIMESTAMP_REBASE, 20);
    pkt_motion_d(&p, BNO085_REPORT_GYROSCOPE, 2, 5, 3, 2);  /* -30 + 5 = -25 */
    pkt_time(&p, BNO085_REPORT_TIMESTAMP_REBASE, -10);
    pkt_motion_d(&p, BNO085_REPORT_GYROSCOPE, 3, 0, 3, 3);  /* -40 */
    pkt_time(&p, BNO085_REPORT_BASE_TIMESTAMP, 8);          /* a new base resets */
    pkt_motion_d(&p, BNO085_REPORT_GYROSCOPE, 4, 2, 3, 4);  /* -8 + 2 = -6 */
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 4);
    CHECK_EQ(c.r[0].ts_ticks, -45);
    CHECK_EQ(c.r[1].ts_ticks, -25);
    CHECK_EQ(c.r[2].ts_ticks, -40);
    CHECK_EQ(c.r[3].ts_ticks, -6);
    CHECK(info.has_rebase);
    CHECK_EQ(info.base_ticks, 50); /* first base of the packet */
    CHECK_EQ(imu_ts_report_time(1000000u, &c.r[1], NULL), 1000000u - 2500u);
}

static void test_no_base_and_gyro_rv_keep_t_int(void) {
    /* A report before any 0xFB carries no reference: it keeps t_int. */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 1, 12, 3, 1);
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 1);
    CHECK(!c.r[0].has_ts);
    CHECK(!info.has_base);
    imu_ts_stats_t s;
    imu_ts_stats_reset(&s);
    CHECK_EQ(imu_ts_report_time(777777u, &c.r[0], &s), 777777u);
    imu_ts_stats_on_packet(&s, &info);
    CHECK_EQ(s.no_base_pkts, 1);

    /* Channel 5 (gyro-integrated RV) has no timing fields at all. */
    pkt_begin(&p, BNO085_CH_GYRO_RV);
    uint8_t r[14] = {0};
    pkt_put(&p, r, sizeof(r));
    memset(&c, 0, sizeof(c));
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(c.n, 1);
    CHECK(!c.r[0].has_ts);
    CHECK_EQ(imu_ts_report_time(555u, &c.r[0], &s), 555u);
    imu_ts_stats_on_packet(&s, &info);
    CHECK_EQ(s.no_base_pkts, 1); /* not an input packet: not counted */
}

static void test_clamping(void) {
    imu_ts_stats_t s;
    imu_ts_stats_reset(&s);
    bno085_report_t r;
    memset(&r, 0, sizeof(r));
    r.has_ts = true;

    /* delay > base (sample "after" INT): clamped to t_int. */
    r.ts_ticks = 3;
    CHECK_EQ(imu_ts_report_time(50000u, &r, &s), 50000u);
    CHECK_EQ(s.clamp_late, 1);

    /* exactly 100 ms before t_int: plausible. */
    r.ts_ticks = -1000;
    CHECK_EQ(imu_ts_report_time(500000u, &r, &s), 400000u);
    CHECK_EQ(s.clamp_early, 0);

    /* > 100 ms before t_int: implausible, keeps t_int. */
    r.ts_ticks = -1001;
    CHECK_EQ(imu_ts_report_time(500000u, &r, &s), 500000u);
    CHECK_EQ(s.clamp_early, 1);

    /* would underflow the clock (right after boot): keeps t_int. */
    r.ts_ticks = -50;
    CHECK_EQ(imu_ts_report_time(1000u, &r, &s), 1000u);
    CHECK_EQ(s.clamp_early, 2);

    /* a garbage base delta far out of range (e.g. 0xFFFFFFFF = -1 tick is
     * fine, 0x7FFFFFFF is not). */
    pkt_t p;
    pkt_begin(&p, BNO085_CH_INPUT);
    pkt_time(&p, BNO085_REPORT_BASE_TIMESTAMP, 0x7FFFFFFF);
    pkt_motion_d(&p, BNO085_REPORT_ACCELEROMETER, 1, 0, 3, 1);
    collect_t c = {0};
    bno085_packet_info_t info;
    CHECK(bno085_parse_packet(p.b, p.len, collect_cb, &c, &info));
    CHECK_EQ(imu_ts_report_time(1000000000u, &c.r[0], &s), 1000000000u);
    CHECK_EQ(s.clamp_early, 3);
}

/* ---- stream path: records carry per-report sample times ------------------ */

static bool apply_slot(imu_slot_t slot, imu_sync_mode_t mode) {
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    imu_slot_config_t *sc = (slot == IMU_SLOT_CRA) ? &cfg.imu_cra : &cfg.imu_mta;
    sc->sync_mode = mode;
    sc->emg_drdy_divisor = 4;
    uint8_t wake[5] = {5, 0, BNO085_CH_WAKE, 0, 0};
    uint8_t pid[20] = {20, 0, BNO085_CH_CONTROL, 0, BNO085_REPORT_PRODUCT_ID_RESP};
    fake_i2c_queue((int)slot, wake, sizeof(wake));
    fake_i2c_queue((int)slot, pid, sizeof(pid));
    return imu_config_apply_slot(slot, sc);
}

static int drain(ring_id_t ring, jtsmm_record_t *out, int cap) {
    int n = 0;
    jtsmm_record_t rec;
    while (ringbuf_pop(ring, &rec)) {
        if (n < cap) {
            out[n] = rec;
        }
        n++;
    }
    return n;
}

static void build_kinematics_packet(pkt_t *p) {
    /* base 3.0 ms; game RV sampled 1.0 ms after the base, linear accel 2.5 ms. */
    pkt_begin(p, BNO085_CH_INPUT);
    pkt_time(p, BNO085_REPORT_BASE_TIMESTAMP, 30);
    uint8_t grv[12] = {BNO085_REPORT_GAME_ROT_VEC, 1, 0x02, 10, 0, 0, 0, 0, 0, 0, 0x00, 0x40};
    pkt_put(p, grv, sizeof(grv));
    pkt_motion_d(p, BNO085_REPORT_LINEAR_ACCEL, 1, 25, 3, 256);
}

static void test_stream_records_per_report_time(void) {
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_MTA, 64);
    fake_i2c_reset();
    CHECK(apply_slot(IMU_SLOT_MTA, IMU_SYNC_INT_DRIVEN));
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    pkt_t p;
    build_kinematics_packet(&p);
    imu_stream_on_complete(IMU_SLOT_MTA, 5000000u, p.b, p.len);
    jtsmm_record_t recs[4];
    CHECK_EQ(drain(RING_IMU_MTA, recs, 4), 2);
    CHECK_EQ(recs[0].type, REC_IMU_MTA_QUAT);
    CHECK_EQ(recs[0].t_master_us, 5000000u - 2000u);
    CHECK_EQ(recs[1].type, REC_IMU_MTA_LIN_ACCEL);
    CHECK_EQ(recs[1].t_master_us, 5000000u - 500u);
    imu_stream_stop_slot(IMU_SLOT_MTA);
}

static void test_stream_divisor_mode_keeps_kick_time(void) {
    /* emg_drdy_divisor reads are not INT-timed: SH-2 delays do not apply. */
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_MTA, 64);
    fake_i2c_reset();
    CHECK(apply_slot(IMU_SLOT_MTA, IMU_SYNC_EMG_DRDY_DIVISOR));
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    pkt_t p;
    build_kinematics_packet(&p);
    imu_stream_on_complete(IMU_SLOT_MTA, 5000000u, p.b, p.len);
    jtsmm_record_t recs[4];
    CHECK_EQ(drain(RING_IMU_MTA, recs, 4), 2);
    CHECK_EQ(recs[0].t_master_us, 5000000u);
    CHECK_EQ(recs[1].t_master_us, 5000000u);
    imu_stream_stop_slot(IMU_SLOT_MTA);
}

/* ---- sensor-core poll path: edge taken at kick time, gated, cleared ------- */

#define MTA_INT_PIN 20 /* fake_devs[0].int_pin (fix_B_fakes.c) */

static void set_int(bool asserted) {
    fake_gpio_level[MTA_INT_PIN] = !asserted;
}

static void poll_start(void) {
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_MTA, 64);
    fake_i2c_reset();
    memset(fake_dma_kicks, 0, sizeof(fake_dma_kicks));
    set_int(false);
    CHECK(apply_slot(IMU_SLOT_MTA, IMU_SYNC_INT_DRIVEN));
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
}

static void test_poll_path_edge_stamps_reads(void) {
    poll_start();
    int arms0 = fake_edge_arms[0];
    imu_stream_poll_service(); /* sensor core sees the stream: arms the edge IRQ */
    CHECK(fake_edge_armed[0]);
    CHECK_EQ(fake_edge_arms[0], arms0 + 1);
    CHECK_EQ(fake_dma_kicks[0], 0); /* INT idle: nothing to read */

    /* INT asserts; the loop comes round 120 µs later. */
    uint64_t t_edge = fake_now_us;
    imu_edge_latch_on_edge(&fake_edge[0], t_edge);
    set_int(true);
    fake_now_us += 120;
    imu_stream_poll_service();
    CHECK_EQ(fake_dma_kicks[0], 1);
    CHECK_EQ(fake_dma_kick_t[0], t_edge);
    CHECK(!fake_edge[0].pending);

    /* The read runs: INT deasserts; the BNO085 re-asserts it after the
     * header-only read (still holding the packet); the re-read starts. */
    set_int(false);
    fake_now_us += 100;
    imu_edge_latch_on_edge(&fake_edge[0], fake_now_us);
    set_int(true);
    CHECK(fake_dma_body_cb[0] != NULL);
    fake_dma_body_cb[0](I2C_DMA_SLOT_MTA); /* header-phase edge dropped */
    CHECK(!fake_edge[0].pending);
    set_int(false);
    /* The next packet asserts INT during the re-read. */
    fake_now_us += 400;
    uint64_t t_next = fake_now_us;
    imu_edge_latch_on_edge(&fake_edge[0], t_next);
    set_int(true);
    fake_now_us += 400;
    pkt_t p;
    build_kinematics_packet(&p);
    CHECK(fake_dma_cb[0] != NULL);
    fake_dma_cb[0](I2C_DMA_SLOT_MTA, fake_dma_kick_t[0], p.b, p.len); /* completion, INT low */
    jtsmm_record_t recs[4];
    CHECK_EQ(drain(RING_IMU_MTA, recs, 4), 2);
    CHECK_EQ(recs[0].t_master_us, t_edge - 2000u);
    CHECK(fake_edge[0].pending); /* kept for the next packet */
    fake_now_us += 50;
    imu_stream_poll_service();
    CHECK_EQ(fake_dma_kicks[0], 2);
    CHECK_EQ(fake_dma_kick_t[0], t_next); /* not the poll time, not the header edge */

    /* Completion with INT released: an edge latched during the read is dropped. */
    imu_edge_latch_on_edge(&fake_edge[0], fake_now_us);
    set_int(false);
    fake_dma_cb[0](I2C_DMA_SLOT_MTA, fake_dma_kick_t[0], p.b, p.len);
    CHECK(!fake_edge[0].pending);
    CHECK_EQ(drain(RING_IMU_MTA, recs, 4), 2);

    /* Stale edge (30 ms, bound 20 ms): the read gets the poll time. */
    imu_edge_latch_on_edge(&fake_edge[0], fake_now_us);
    fake_now_us += 30000;
    set_int(true);
    imu_stream_poll_service();
    CHECK_EQ(fake_dma_kicks[0], 3);
    CHECK_EQ(fake_dma_kick_t[0], fake_now_us);

    /* Glitch: edge latched but INT back high -> dropped; the next assertion
     * without an edge (IRQ missed) is stamped with the poll time. */
    set_int(false);
    imu_edge_latch_on_edge(&fake_edge[0], fake_now_us);
    fake_now_us += 10;
    imu_stream_poll_service();
    CHECK_EQ(fake_dma_kicks[0], 3);
    CHECK(!fake_edge[0].pending);
    set_int(true);
    fake_now_us += 1000;
    imu_stream_poll_service();
    CHECK_EQ(fake_dma_kicks[0], 4);
    CHECK_EQ(fake_dma_kick_t[0], fake_now_us);

    /* Stream stop: the sensor core disarms. */
    set_int(false);
    imu_stream_stop_slot(IMU_SLOT_MTA);
    imu_stream_poll_service();
    CHECK(!fake_edge_armed[0]);
}

static void test_poll_path_restart_clears_stale_edge(void) {
    poll_start();
    imu_stream_poll_service();
    CHECK(fake_edge_armed[0]);
    /* Stop and restart without the sensor core noticing the stop; an edge
     * latched in between (bench tap / apply traffic) must not be used. */
    imu_stream_stop_slot(IMU_SLOT_MTA);
    imu_edge_latch_on_edge(&fake_edge[0], fake_now_us);
    fake_now_us += 500;
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    CHECK(!fake_edge[0].pending);
    int arms0 = fake_edge_arms[0];
    set_int(true);
    fake_now_us += 100;
    imu_stream_poll_service();
    CHECK_EQ(fake_edge_arms[0], arms0 + 1); /* restart seen (stream_gen): re-armed */
    CHECK_EQ(fake_dma_kicks[0], 1);
    CHECK_EQ(fake_dma_kick_t[0], fake_now_us);
    imu_stream_stop_slot(IMU_SLOT_MTA);
    imu_stream_poll_service();
}

static void test_poll_path_divisor_mode_not_armed(void) {
    ringbuf_init_all();
    fake_i2c_reset();
    CHECK(apply_slot(IMU_SLOT_MTA, IMU_SYNC_EMG_DRDY_DIVISOR));
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    imu_stream_poll_service();
    CHECK(!fake_edge_armed[0]);
    imu_stream_stop_slot(IMU_SLOT_MTA);
    imu_stream_poll_service();
}

int main(void) {
    test_latch_basic();
    test_latch_first_edge_kept();
    test_latch_reassert_during_read();
    test_latch_released_edge_dropped();
    test_latch_stale_and_future();
    test_latch_cleared_on_stream_start();
    test_max_age();
    test_stats_latency_window();
    test_delay_bit_layout();
    test_distinct_increasing_times();
    test_rebase();
    test_no_base_and_gyro_rv_keep_t_int();
    test_clamping();
    test_stream_records_per_report_time();
    test_stream_divisor_mode_keeps_kick_time();
    test_poll_path_edge_stamps_reads();
    test_poll_path_restart_clears_stale_edge();
    test_poll_path_divisor_mode_not_armed();
    return fixb_report("test_imu_timing");
}
