// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Fix package B: IMU domain streaming path (imu_stream_on_complete) against
 * the real ring buffer -- one record per SH-2 report instance (contract K3).
 */

#include "fix_B_check.h"
#include "fix_B_fakes.h"

#include "jtsmm/config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/drivers/bno085.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"

#include <string.h>

static uint8_t pkt[BNO085_SHTP_MAX_PACKET];
static uint16_t pkt_len;

static void pkt_begin(uint8_t channel) {
    memset(pkt, 0, sizeof(pkt));
    pkt[2] = channel;
    pkt_len = 4;
}
static void pkt_put(const uint8_t *d, uint16_t n) {
    memcpy(pkt + pkt_len, d, n);
    pkt_len = (uint16_t)(pkt_len + n);
    pkt[0] = (uint8_t)(pkt_len & 0xFFu);
    pkt[1] = (uint8_t)(pkt_len >> 8);
}
static void pkt_motion(uint8_t id, uint8_t seq, int16_t x) {
    uint8_t r[10] = {id, seq, 0x03, 0, (uint8_t)x, (uint8_t)((uint16_t)x >> 8)};
    pkt_put(r, sizeof(r));
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

static void setup(void) {
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_MTA, 64);
    ringbuf_set_capacity(RING_IMU_CRA, 64);
    fake_i2c_reset();
}

/* Script a BNO085 boot on @p slot's bus: wake-channel packet, then product id. */
static void queue_boot(imu_slot_t slot) {
    uint8_t wake[5] = {5, 0, BNO085_CH_WAKE, 0, 0};
    uint8_t pid[20] = {20, 0, BNO085_CH_CONTROL, 0, BNO085_REPORT_PRODUCT_ID_RESP};
    fake_i2c_queue((int)slot, wake, sizeof(wake));
    fake_i2c_queue((int)slot, pid, sizeof(pid));
}

/* Apply the default (kinematics) config to @p slot on a scripted device. */
static bool apply_default(imu_slot_t slot) {
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    queue_boot(slot);
    return imu_config_apply_slot(slot, slot == IMU_SLOT_CRA ? &cfg.imu_cra : &cfg.imu_mta);
}

/* Bring @p slot to the streaming state with the default (kinematics) config. */
static void bring_up_slot(imu_slot_t slot) {
    CHECK(apply_default(slot));
    imu_stream_start_slot(slot);
}

static void test_one_record_per_report(void) {
    setup();
    bring_up_slot(IMU_SLOT_MTA);

    pkt_begin(BNO085_CH_INPUT);
    uint8_t ts[5] = {BNO085_REPORT_BASE_TIMESTAMP, 0, 0, 0, 0};
    pkt_put(ts, sizeof(ts));
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 1, 100);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 2, 200);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 3, 300);
    uint8_t grv[12] = {BNO085_REPORT_GAME_ROT_VEC, 9, 0x02, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x40};
    pkt_put(grv, sizeof(grv));
    pkt_put(grv, sizeof(grv));
    pkt_motion(BNO085_REPORT_LINEAR_ACCEL, 4, 512);

    imu_stream_on_complete(IMU_SLOT_MTA, 123456u, pkt, pkt_len);
    jtsmm_record_t recs[16];
    int n = drain(RING_IMU_MTA, recs, 16);
    CHECK_EQ(n, 6);
    CHECK_EQ(recs[0].type, REC_IMU_MTA_RAW);
    CHECK_EQ(recs[0].payload[0], IMU_SENSOR_ACCEL);
    int16_t x = 0;
    memcpy(&x, recs[2].payload + 1, 2);
    CHECK_EQ(x, 300);
    CHECK_EQ(recs[3].type, REC_IMU_MTA_QUAT);
    CHECK_EQ(recs[3].len, IMU_QUAT_PAYLOAD_LEN);
    CHECK_EQ(recs[3].payload[0], BNO085_REPORT_GAME_ROT_VEC);
    float w = 0.0f;
    memcpy(&w, recs[3].payload + 1, 4);
    CHECK(w > 0.99f && w < 1.01f);
    CHECK_EQ(recs[3].payload[17], 2);
    CHECK_EQ(recs[4].type, REC_IMU_MTA_QUAT);
    CHECK_EQ(recs[5].type, REC_IMU_MTA_LIN_ACCEL);
    float ax = 0.0f;
    memcpy(&ax, recs[5].payload, 4);
    CHECK(ax > 1.99f && ax < 2.01f);
    for (int i = 0; i < 6 && i < n; i++) {
        CHECK_EQ(recs[i].t_master_us, 123456u);
    }
    CHECK_EQ(drain(RING_IMU_CRA, recs, 16), 0);
}

static void test_gyro_rv_records(void) {
    setup();
    bring_up_slot(IMU_SLOT_CRA);
    pkt_begin(BNO085_CH_GYRO_RV);
    uint8_t r[14] = {0};
    r[6] = 0x00;
    r[7] = 0x40; /* real = 16384 */
    pkt_put(r, sizeof(r));
    pkt_put(r, sizeof(r));
    imu_stream_on_complete(IMU_SLOT_CRA, 1u, pkt, pkt_len);
    jtsmm_record_t recs[4];
    int n = drain(RING_IMU_CRA, recs, 4);
    CHECK_EQ(n, 2);
    CHECK_EQ(recs[0].type, REC_IMU_CRA_QUAT);
    CHECK_EQ(recs[0].len, IMU_QUAT_PAYLOAD_LEN - 1u); /* no accuracy byte */
    CHECK_EQ(recs[0].payload[0], BNO085_REPORT_GYRO_INT_ROT_VEC);
    float w = 0.0f;
    memcpy(&w, recs[1].payload + 1, 4);
    CHECK(w > 0.99f && w < 1.01f);
}

static void test_not_streaming_drops(void) {
    setup();
    bring_up_slot(IMU_SLOT_MTA);
    imu_stream_stop_slot(IMU_SLOT_MTA);
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 1, 1);
    imu_stream_on_complete(IMU_SLOT_MTA, 1u, pkt, pkt_len);
    jtsmm_record_t recs[2];
    CHECK_EQ(drain(RING_IMU_MTA, recs, 2), 0);
}

static void feed(imu_slot_t slot) {
    imu_stream_on_complete(slot, fake_now_us, pkt, pkt_len);
}

static void test_seq_gaps(void) {
    setup();
    bring_up_slot(IMU_SLOT_MTA);
    uint32_t g0 = imu_config_seq_gaps(IMU_SLOT_MTA);
    jtsmm_record_t recs[16];

    /* 1,2 | 3 contiguous; then 6 (4,5 lost); a second id is independent. */
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 1, 0);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 2, 0);
    pkt_motion(BNO085_REPORT_LINEAR_ACCEL, 100, 0);
    feed(IMU_SLOT_MTA);
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 3, 0);
    pkt_motion(BNO085_REPORT_LINEAR_ACCEL, 101, 0);
    feed(IMU_SLOT_MTA);
    CHECK_EQ(imu_config_seq_gaps(IMU_SLOT_MTA) - g0, 0);
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 6, 0);
    feed(IMU_SLOT_MTA);
    CHECK_EQ(imu_config_seq_gaps(IMU_SLOT_MTA) - g0, 2);

    /* Wrap 254, 255, 0, 1 is contiguous. */
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_GYROSCOPE, 254, 0);
    pkt_motion(BNO085_REPORT_GYROSCOPE, 255, 0);
    feed(IMU_SLOT_MTA);
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_GYROSCOPE, 0, 0);
    pkt_motion(BNO085_REPORT_GYROSCOPE, 1, 0);
    feed(IMU_SLOT_MTA);
    CHECK_EQ(imu_config_seq_gaps(IMU_SLOT_MTA) - g0, 2);
    CHECK_EQ(imu_config_seq_gaps(IMU_SLOT_CRA), 0);

    /* Channel 5 has no per-report seq: lost SHTP packets are counted. */
    uint8_t r[14] = {0};
    const uint8_t shtp_seq[] = {10, 11, 14};
    for (unsigned i = 0; i < sizeof(shtp_seq); i++) {
        pkt_begin(BNO085_CH_GYRO_RV);
        pkt[3] = shtp_seq[i];
        pkt_put(r, sizeof(r));
        pkt_put(r, sizeof(r));
        feed(IMU_SLOT_MTA);
    }
    CHECK_EQ(imu_config_seq_gaps(IMU_SLOT_MTA) - g0, 4);

    /* Stream restart forgets the old sequence state (no false gap). */
    imu_stream_stop_slot(IMU_SLOT_MTA);
    bring_up_slot(IMU_SLOT_MTA);
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 77, 0);
    feed(IMU_SLOT_MTA);
    CHECK_EQ(imu_config_seq_gaps(IMU_SLOT_MTA) - g0, 4);
    (void)drain(RING_IMU_MTA, recs, 16);
}

static void test_continuation_not_emitted(void) {
    setup();
    bring_up_slot(IMU_SLOT_CRA);
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 1, 0);
    pkt[1] |= 0x80u;
    feed(IMU_SLOT_CRA);
    jtsmm_record_t recs[2];
    CHECK_EQ(drain(RING_IMU_CRA, recs, 2), 0);
}

/* Run the cooperative recovery to completion on the fake clock. */
static void run_recovery(void) {
    for (int i = 0; i < 20000 && imu_recover_busy(); i++) {
        imu_recover_poll();
        fake_now_us += 1000u;
    }
}

static void test_spontaneous_reset(void) {
    setup();
    bring_up_slot(IMU_SLOT_CRA);
    uint32_t r0 = imu_config_reset_count(IMU_SLOT_CRA);
    jtsmm_record_t recs[8];

    /* Leftover reset indications right after stream start are ignored. */
    pkt_begin(BNO085_CH_EXECUTABLE);
    uint8_t rc = BNO085_EXEC_RESET_COMPLETE;
    pkt_put(&rc, 1);
    feed(IMU_SLOT_CRA);
    CHECK_EQ(imu_config_reset_count(IMU_SLOT_CRA), r0);
    imu_recover_poll();
    CHECK(!imu_recover_busy());

    /* A real one later: advertisement + reset-complete count once. */
    fake_now_us += 1000000u;
    pkt_begin(BNO085_CH_COMMAND);
    uint8_t adv[4] = {BNO085_CMD_ADVERTISEMENT, 0, 1, 1};
    pkt_put(adv, sizeof(adv));
    feed(IMU_SLOT_CRA);
    pkt_begin(BNO085_CH_EXECUTABLE);
    pkt_put(&rc, 1);
    feed(IMU_SLOT_CRA);
    CHECK_EQ(imu_config_reset_count(IMU_SLOT_CRA), r0 + 1);
    CHECK_EQ(imu_config_reset_count(IMU_SLOT_MTA), 0);

    /* The comms core re-enables the reports without a soft reset. */
    fake_i2c_reset();
    imu_recover_poll();
    CHECK(imu_recover_busy());
    pkt_begin(BNO085_CH_INPUT);
    pkt_motion(BNO085_REPORT_ACCELEROMETER, 1, 0);
    feed(IMU_SLOT_CRA); /* off the stream while recovering */
    CHECK_EQ(drain(RING_IMU_CRA, recs, 8), 0);
    run_recovery();
    CHECK(!imu_recover_busy());
    CHECK(fake_i2c_writes[1] >= 3); /* calibration + product id + reports */
    CHECK_EQ(fake_i2c_last_write[1][2], BNO085_CH_CONTROL);
    CHECK_EQ(fake_i2c_last_write[1][4], BNO085_REPORT_SET_FEATURE);
    CHECK(strstr(fake_last_log, "device reset") != NULL);

    /* Streaming resumed. */
    feed(IMU_SLOT_CRA);
    CHECK_EQ(drain(RING_IMU_CRA, recs, 8), 1);
    CHECK_EQ(imu_config_reset_count(IMU_SLOT_CRA), r0 + 1);
}

int main(void) {
    test_one_record_per_report();
    test_gyro_rv_records();
    test_not_streaming_drops();
    test_seq_gaps();
    test_continuation_not_emitted();
    test_spontaneous_reset();
    return fixb_report("test_fix_B_stream");
}
