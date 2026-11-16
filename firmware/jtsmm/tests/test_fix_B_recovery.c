// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Fix package B: cooperative IMU recovery (comms core) -- bus clear,
 * single-shot writes, NRST escalation across both slots, host intent at
 * FINISH, and no interleaving with config apply.
 */

#include "fix_B_check.h"
#include "fix_B_fakes.h"

#include "jtsmm/config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/drivers/bno085.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/runtime/ringbuf.h"

#include <string.h>

static void setup(void) {
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_MTA, 64);
    ringbuf_set_capacity(RING_IMU_CRA, 64);
    fake_i2c_reset();
    fake_config_apply_busy = false;
    fake_nrst_level = true;
}

static void bring_up(imu_slot_t slot) {
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    fake_queue_boot((int)slot);
    CHECK(imu_config_apply_slot(slot, slot == IMU_SLOT_CRA ? &cfg.imu_cra : &cfg.imu_mta));
    imu_stream_start_slot(slot);
}

static void run_recovery(void) {
    for (int i = 0; i < 20000 && imu_recover_busy(); i++) {
        imu_recover_poll();
        fake_now_us += 1000u;
    }
}

/* Feed one accel report and return how many records it produced. */
static int stream_records(imu_slot_t slot) {
    static uint8_t seq;
    uint8_t pkt[14] = {14, 0, BNO085_CH_INPUT, 0, BNO085_REPORT_ACCELEROMETER, seq++, 3, 0};
    imu_stream_on_complete(slot, fake_now_us, pkt, sizeof(pkt));
    int n = 0;
    jtsmm_record_t rec;
    while (ringbuf_pop(slot == IMU_SLOT_CRA ? RING_IMU_CRA : RING_IMU_MTA, &rec)) {
        n++;
    }
    return n;
}

static void test_watchdog_recovery_resumes(void) {
    setup();
    bring_up(IMU_SLOT_MTA);
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 1);
    int clears = fake_bus_clears[0];
    bno085_t *dev = imu_bus_dev(IMU_SLOT_MTA);

    fake_queue_boot(0);
    imu_slot_recover_start(IMU_SLOT_MTA);
    CHECK(imu_recover_busy());
    CHECK_EQ(fake_bus_clears[0], clears + 1);
    CHECK_EQ(dev->write_tries, 1);
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 0); /* off the stream */
    run_recovery();
    CHECK(!imu_recover_busy());
    CHECK_EQ(dev->write_tries, 0);
    CHECK(strstr(fake_last_log, "MTA pid=1") != NULL);
    CHECK(strstr(fake_last_log, "FAIL") == NULL);
    CHECK_EQ(fake_i2c_last_write[0][4], BNO085_REPORT_SET_FEATURE);
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 1);
    CHECK_EQ(fake_nrst_pulses, 0);
}

static void test_host_stop_during_recovery_sticks(void) {
    setup();
    bring_up(IMU_SLOT_CRA);
    fake_queue_boot(1);
    imu_slot_recover_start(IMU_SLOT_CRA);
    imu_stream_stop_slot(IMU_SLOT_CRA); /* host: imu_stream cra off */
    run_recovery();
    CHECK(!imu_recover_busy());
    CHECK_EQ(stream_records(IMU_SLOT_CRA), 0); /* was resumed from a stale snapshot */

    /* ...and `on` during recovery is honoured at FINISH without touching the bus. */
    bring_up(IMU_SLOT_CRA);
    fake_queue_boot(1);
    imu_slot_recover_start(IMU_SLOT_CRA);
    imu_stream_stop_slot(IMU_SLOT_CRA);
    int q = fake_dma_quiesces[1];
    imu_stream_start_slot(IMU_SLOT_CRA);
    CHECK_EQ(fake_dma_quiesces[1], q);
    run_recovery();
    CHECK_EQ(stream_records(IMU_SLOT_CRA), 1);
}

static void test_dead_slot_escalates_to_nrst_once(void) {
    setup();
    bring_up(IMU_SLOT_MTA);
    bring_up(IMU_SLOT_CRA);
    int pulses = fake_nrst_pulses;

    fake_i2c_dead[1] = true; /* CRA stops answering */
    imu_slot_recover_start(IMU_SLOT_CRA);
    for (int i = 0; i < 1000 && fake_nrst_pulses == pulses; i++) {
        imu_recover_poll();
        fake_now_us += 1000u;
    }
    /* Five single-shot soft-reset writes, not 5 x 20 retries. */
    CHECK_EQ(fake_i2c_dead_attempts[1], 5);
    CHECK_EQ(fake_nrst_pulses, pulses + 1);
    CHECK(!fake_nrst_level);
    CHECK(strstr(fake_last_log, "NRST") != NULL);
    /* NRST resets both IMUs: MTA is taken off the stream and re-enabled too. */
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 0);
    fake_queue_boot(0);
    run_recovery();
    CHECK(fake_nrst_level);
    CHECK(!imu_recover_busy());
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 1);
    CHECK_EQ(fake_i2c_last_write[0][4], BNO085_REPORT_SET_FEATURE);

    /* A second attempt within the rate limit does not pulse again. */
    imu_slot_recover_start(IMU_SLOT_CRA);
    run_recovery();
    CHECK_EQ(fake_nrst_pulses, pulses + 1);
    CHECK(strstr(fake_last_log, "CRA pid=0") != NULL);
    CHECK(strstr(fake_last_log, "FAIL") != NULL);

    /* After the interval it may. */
    fake_now_us += 61000000u;
    imu_slot_recover_start(IMU_SLOT_CRA);
    fake_queue_boot(0);
    run_recovery();
    CHECK_EQ(fake_nrst_pulses, pulses + 2);
    CHECK(fake_nrst_level);
    fake_i2c_dead[1] = false;
}

/* NRST escalation for CRA while MTA's DMA does not stop: MTA must stay on the
 * stream (it re-syncs from the reset indication) instead of going silent until
 * its 3 s watchdog because imu_slot_quiesce() had already cleared it. */
static void test_nrst_keeps_unquiesced_other_slot_streaming(void) {
    setup();
    fake_now_us += 61000000u; /* past the NRST rate limit of earlier tests */
    bring_up(IMU_SLOT_MTA);
    bring_up(IMU_SLOT_CRA);
    int pulses = fake_nrst_pulses;
    fake_i2c_dead[1] = true;
    imu_slot_recover_start(IMU_SLOT_CRA);
    fake_dma_quiesce_fail[0] = true;
    for (int i = 0; i < 1000 && fake_nrst_pulses == pulses; i++) {
        imu_recover_poll();
        fake_now_us += 1000u;
    }
    CHECK_EQ(fake_nrst_pulses, pulses + 1);
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 1);
    fake_dma_quiesce_fail[0] = false;
    fake_i2c_dead[1] = false;
    fake_queue_boot(1);
    run_recovery();
    CHECK(!imu_recover_busy());
    CHECK_EQ(stream_records(IMU_SLOT_MTA), 1);
}

static void test_apply_cancels_recovery(void) {
    setup();
    bring_up(IMU_SLOT_MTA);
    imu_slot_recover_start(IMU_SLOT_MTA);
    CHECK(imu_recover_busy());
    fake_config_apply_busy = true;
    int writes = fake_i2c_writes[0];
    imu_recover_poll();
    CHECK(!imu_recover_busy());
    CHECK_EQ(fake_i2c_writes[0], writes);
    CHECK_EQ(imu_bus_dev(IMU_SLOT_MTA)->write_tries, 0);
    fake_config_apply_busy = false;
}

int main(void) {
    test_watchdog_recovery_resumes();
    test_host_stop_during_recovery_sticks();
    test_dead_slot_escalates_to_nrst_once();
    test_apply_cancels_recovery();
    test_nrst_keeps_unquiesced_other_slot_streaming();
    return fixb_report("test_fix_B_recovery");
}
