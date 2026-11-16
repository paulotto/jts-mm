// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * IMU timestamps with both switches off (built with IMU_TS_USE_INT_EDGE=0 and
 * IMU_TS_USE_SH2_DELAY=0): the previous behaviour -- the edge IRQ is never
 * armed, a read is stamped with the poll time, and every report of a packet
 * gets that time.
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

#if IMU_TS_USE_INT_EDGE || IMU_TS_USE_SH2_DELAY
#error "test_imu_timing_off must be built with both IMU_TS_USE_* switches at 0"
#endif

#define MTA_INT_PIN 20 /* fake_devs[0].int_pin (fix_B_fakes.c) */

int main(void) {
    ringbuf_init_all();
    ringbuf_set_capacity(RING_IMU_MTA, 64);
    fake_i2c_reset();
    fake_gpio_level[MTA_INT_PIN] = true;

    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    uint8_t wake[5] = {5, 0, BNO085_CH_WAKE, 0, 0};
    uint8_t pid[20] = {20, 0, BNO085_CH_CONTROL, 0, BNO085_REPORT_PRODUCT_ID_RESP};
    fake_i2c_queue(0, wake, sizeof(wake));
    fake_i2c_queue(0, pid, sizeof(pid));
    CHECK(imu_config_apply_slot(IMU_SLOT_MTA, &cfg.imu_mta));
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));

    imu_stream_poll_service();
    CHECK(!fake_edge_armed[0]);

    /* An edge in the latch is ignored: the read gets the poll time. */
    imu_edge_latch_on_edge(&fake_edge[0], fake_now_us);
    fake_now_us += 300;
    fake_gpio_level[MTA_INT_PIN] = false;
    imu_stream_poll_service();
    CHECK_EQ(fake_dma_kicks[0], 1);
    CHECK_EQ(fake_dma_kick_t[0], fake_now_us);

    /* SH-2 base delta / delays present, yet every record gets t_us. */
    uint8_t pkt[4 + 5 + 12 + 10] = {sizeof(pkt), 0, BNO085_CH_INPUT, 0,
                                    BNO085_REPORT_BASE_TIMESTAMP, 30, 0, 0, 0,
                                    BNO085_REPORT_GAME_ROT_VEC, 1, 0x02, 10, 0, 0, 0, 0, 0, 0, 0x00, 0x40,
                                    BNO085_REPORT_LINEAR_ACCEL, 1, 0x03, 25, 0, 1, 0, 0, 0, 0};
    imu_stream_on_complete(IMU_SLOT_MTA, 4000000u, pkt, sizeof(pkt));
    jtsmm_record_t rec;
    int n = 0;
    while (ringbuf_pop(RING_IMU_MTA, &rec)) {
        CHECK_EQ(rec.t_master_us, 4000000u);
        n++;
    }
    CHECK_EQ(n, 2);
    return fixb_report("test_imu_timing_off");
}
