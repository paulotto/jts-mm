// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Fix package B: IMU slot validation and apply (disabled slot, absent /
 * non-booting device, callback registration, stream refusal).
 */

#include "fix_B_check.h"
#include "fix_B_fakes.h"

#include "jtsmm/config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/drivers/bno085.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"

#include <string.h>

static imu_slot_config_t custom(uint8_t id, uint16_t rate) {
    imu_slot_config_t c;
    memset(&c, 0, sizeof(c));
    c.enabled = true;
    c.profile = IMU_PROFILE_CUSTOM;
    c.fusion_rate_hz = 390;
    c.num_reports = 1;
    c.reports[0].enabled = true;
    c.reports[0].report_id = id;
    c.reports[0].rate_hz = rate;
    return c;
}

static void test_validate(void) {
    imu_slot_config_t c = custom(0x02, 400); /* gyroscope max 400 */
    CHECK(imu_config_validate_slot(&c, 2000));
    c.reports[0].rate_hz = 401;
    CHECK(!imu_config_validate_slot(&c, 2000));
    c.reports[0].rate_hz = 0; /* used to become 400 Hz silently */
    CHECK(!imu_config_validate_slot(&c, 2000));
    c = custom(0x77, 100); /* unknown id */
    CHECK(!imu_config_validate_slot(&c, 2000));
    c = custom(0x2A, 1000); /* gyro-integrated RV max 1000 */
    CHECK(imu_config_validate_slot(&c, 2000));

    /* Fusion profile with fusion rate 0. */
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    cfg.imu_mta.fusion_rate_hz = 0;
    imu_config_resolve_profile(&cfg.imu_mta);
    CHECK(!imu_config_validate_slot(&cfg.imu_mta, 2000));
    jtsmm_config_set_defaults(&cfg);
    imu_config_resolve_profile(&cfg.imu_mta);
    CHECK(imu_config_validate_slot(&cfg.imu_mta, 2000));

    /* A disabled slot is never rejected. */
    c = custom(0x77, 0);
    c.enabled = false;
    CHECK(imu_config_validate_slot(&c, 2000));

    /* DRDY divisor: 16000 / 50 = 320 does not fit; (uint8_t)320 = 64 used to pass. */
    c = custom(0x01, 50);
    c.sync_mode = IMU_SYNC_EMG_DRDY_DIVISOR;
    c.emg_drdy_divisor = (uint8_t)(16000u / 50u);
    CHECK(!imu_config_validate_slot(&c, 16000));
    c.reports[0].rate_hz = 500;
    c.emg_drdy_divisor = 4;
    CHECK(imu_config_validate_slot(&c, 2000));
    c.emg_drdy_divisor = 5;
    CHECK(!imu_config_validate_slot(&c, 2000));
}

/* Did the write log of @p bus from index @p from on contain Set Feature(@p id, interval 0)? */
static bool saw_disable(int bus, int from, uint8_t id) {
    for (int i = from; i < fake_i2c_writes[bus] && i < FAKE_WRITE_LOG; i++) {
        const uint8_t *w = fake_i2c_write_log[bus][i];
        if (w[2] == BNO085_CH_CONTROL && w[4] == BNO085_REPORT_SET_FEATURE && w[5] == id && w[9] == 0 &&
            w[10] == 0 && w[11] == 0 && w[12] == 0) {
            return true;
        }
    }
    return false;
}

static int feed_accel(imu_slot_t slot) {
    uint8_t pkt[14] = {14, 0, BNO085_CH_INPUT, 0, BNO085_REPORT_ACCELEROMETER, 1, 3, 0};
    imu_stream_on_complete(slot, fake_now_us, pkt, sizeof(pkt));
    int n = 0;
    jtsmm_record_t rec;
    while (ringbuf_pop(slot == IMU_SLOT_CRA ? RING_IMU_CRA : RING_IMU_MTA, &rec)) {
        n++;
    }
    return n;
}

static void test_disable_slot(void) {
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_CRA, 64);
    fake_i2c_reset();
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    fake_queue_boot(1);
    CHECK(imu_config_apply_slot(IMU_SLOT_CRA, &cfg.imu_cra)); /* GRV 0x08 + accel 0x01 */
    CHECK(imu_stream_start_slot(IMU_SLOT_CRA));
    CHECK_EQ(feed_accel(IMU_SLOT_CRA), 1);

    int from = fake_i2c_writes[1];
    cfg.imu_cra.enabled = false;
    CHECK(imu_config_apply_slot(IMU_SLOT_CRA, &cfg.imu_cra));
    CHECK(saw_disable(1, from, 0x08));
    CHECK(saw_disable(1, from, 0x01));
    CHECK_EQ(feed_accel(IMU_SLOT_CRA), 0); /* apply stops the stream */
    /* `imu_stream cra on` must not force-enable it. */
    CHECK(!imu_stream_start_slot(IMU_SLOT_CRA));
    CHECK_EQ(feed_accel(IMU_SLOT_CRA), 0);

    /* Re-enabling works again. */
    cfg.imu_cra.enabled = true;
    fake_queue_boot(1);
    CHECK(imu_config_apply_slot(IMU_SLOT_CRA, &cfg.imu_cra));
    CHECK(imu_stream_start_slot(IMU_SLOT_CRA));
    CHECK_EQ(feed_accel(IMU_SLOT_CRA), 1);
    imu_stream_stop_slot(IMU_SLOT_CRA);
}

static void test_absent_and_boot_fail(void) {
    fake_i2c_reset();
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);

    /* Absent: still true (a false would keep the sensor core, i.e. EMG, down),
     * but logged, and the completion callback is registered for a later revival. */
    fake_dma_cb_set[0] = false;
    fake_probe_ok[0] = false;
    int clears = fake_bus_clears[0];
    CHECK(imu_config_apply_slot(IMU_SLOT_MTA, &cfg.imu_mta));
    CHECK(fake_dma_cb_set[0]);
    CHECK_EQ(fake_bus_clears[0], clears + 1); /* one bus clear before giving up */
    CHECK(strstr(fake_last_log, "imu_mta") != NULL);
    CHECK(strstr(fake_last_log, "NOT FOUND") != NULL);
    fake_probe_ok[0] = true;

    /* Present but never boots (nothing queued). */
    fake_dma_cb_set[0] = false;
    fake_last_log[0] = '\0';
    CHECK(imu_config_apply_slot(IMU_SLOT_MTA, &cfg.imu_mta));
    CHECK(fake_dma_cb_set[0]);
    CHECK(strstr(fake_last_log, "boot FAIL") != NULL);
    /* Stays enabled, so streaming (and with it the watchdog recovery) can start. */
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    imu_stream_stop_slot(IMU_SLOT_MTA);
}

static void queue_feature_resp(int bus, uint8_t id, uint32_t interval_us) {
    uint8_t p[21] = {21, 0, BNO085_CH_CONTROL, 0, BNO085_REPORT_GET_FEATURE_RESP, id, 0, 0, 0,
                     (uint8_t)interval_us, (uint8_t)(interval_us >> 8), (uint8_t)(interval_us >> 16),
                     (uint8_t)(interval_us >> 24)};
    fake_i2c_queue(bus, p, sizeof(p));
}

static void test_granted_intervals(void) {
    fake_i2c_reset();
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg); /* GRV @ 390 Hz + accelerometer @ 250 Hz */
    fake_queue_boot(0);
    queue_feature_resp(0, 0x08, 2500); /* sensor grants 400 Hz */
    queue_feature_resp(0, 0x01, 4000);
    fake_gpio_level[20] = false; /* MTA INT asserted while responses are pending */
    uint64_t t0 = fake_now_us;
    CHECK(imu_config_apply_slot(IMU_SLOT_MTA, &cfg.imu_mta));
    fake_gpio_level[20] = true;
    CHECK_EQ(imu_config_granted_interval_us(IMU_SLOT_MTA, 0x08), 2500);
    CHECK_EQ(imu_config_granted_interval_us(IMU_SLOT_MTA, 0x01), 4000);
    CHECK_EQ(imu_config_granted_interval_us(IMU_SLOT_MTA, 0x04), 0);
    /* Both enables returned on their response instead of sitting out 500 ms each. */
    CHECK(fake_now_us - t0 < 500000u);

    /* A later (e.g. unsolicited) response on the stream updates it. */
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    uint8_t p[21] = {21, 0, BNO085_CH_CONTROL, 0, BNO085_REPORT_GET_FEATURE_RESP, 0x01, 0, 0, 0,
                     0x6F, 0x11, 0, 0 /* 4463 us */};
    imu_stream_on_complete(IMU_SLOT_MTA, fake_now_us, p, sizeof(p));
    CHECK_EQ(imu_config_granted_interval_us(IMU_SLOT_MTA, 0x01), 4463);
    imu_stream_stop_slot(IMU_SLOT_MTA);
}

/* Boot default = kinematics_accel: the fusion quaternion plus the calibrated
 * accelerometer (gravity included) at 250 Hz, no linear acceleration. */
static void test_default_preset(void) {
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    CHECK(cfg.imu_mta.profile == IMU_PROFILE_KINEMATICS_ACCEL);
    CHECK(cfg.imu_cra.profile == IMU_PROFILE_KINEMATICS_ACCEL);
    imu_config_resolve_profile(&cfg.imu_mta);
    CHECK(cfg.imu_mta.num_reports == 2);
    CHECK(cfg.imu_mta.reports[0].report_id == 0x08);
    CHECK(cfg.imu_mta.reports[0].rate_hz == 390);
    CHECK(cfg.imu_mta.reports[1].report_id == 0x01);
    CHECK(cfg.imu_mta.reports[1].rate_hz == 250);
    CHECK(cfg.imu_mta.reports[1].usb_type == REC_IMU_MTA_RAW);
    CHECK(imu_config_validate_slot(&cfg.imu_mta, 4000));

    cfg.imu_mta.profile = IMU_PROFILE_KINEMATICS_DEFAULT;
    imu_config_resolve_profile(&cfg.imu_mta);
    CHECK(cfg.imu_mta.num_reports == 2);
    CHECK(cfg.imu_mta.reports[1].report_id == 0x04);
}

int main(void) {
    test_validate();
    test_default_preset();
    test_disable_slot();
    test_absent_and_boot_fail();
    test_granted_intervals();
    return fixb_report("test_fix_B_apply");
}
