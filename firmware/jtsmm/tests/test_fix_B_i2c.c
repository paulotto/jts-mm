// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Fix package B: I2C timeout scaling and the blocking BNO085 transport
 * (full-packet retry, single-shot writes for the recovery path).
 */

#include "fix_B_check.h"
#include "fix_B_fakes.h"

#include "jtsmm/drivers/bno085.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/platform/i2c_dma.h"

#include <string.h>

static void test_timeout_scaling(void) {
    /* Short reads keep the old floor. */
    CHECK_EQ(i2c_xfer_timeout_us(4, 400000u, I2C_DMA_WATCHDOG_FLOOR_US), 8000);
    /* A full 512-byte read needs 11.5 ms of wire time at 400 kHz: the old
     * fixed 8 ms watchdog always killed it. */
    uint32_t t400 = i2c_xfer_timeout_us(512, 400000u, I2C_DMA_WATCHDOG_FLOOR_US);
    CHECK(t400 > 513u * 9u * 1000000u / 400000u);
    CHECK_EQ(t400, 2u * 11543u + 2000u);
    /* 100 kHz (allowed by the config) scales 4x. */
    uint32_t t100 = i2c_xfer_timeout_us(512, 100000u, I2C_DMA_WATCHDOG_FLOOR_US);
    CHECK_EQ(t100, 2u * 46170u + 2000u);
    /* Unknown baud is treated as the slowest allowed rate. */
    CHECK_EQ(i2c_xfer_timeout_us(512, 0u, I2C_DMA_WATCHDOG_FLOOR_US), t100);
    /* Monotonic in length. */
    CHECK(i2c_xfer_timeout_us(300, 100000u, 0) < i2c_xfer_timeout_us(301, 100000u, 0));
}

static void test_blocking_read_scaled_timeout(void) {
    fake_i2c_reset();
    bno085_t *dev = imu_bus_dev(IMU_SLOT_MTA);
    dev->baud_hz = 100000u;
    uint8_t p[300];
    memset(p, 0xAB, sizeof(p));
    p[0] = (uint8_t)(sizeof(p) & 0xFF);
    p[1] = (uint8_t)(sizeof(p) >> 8);
    p[2] = BNO085_CH_INPUT;
    fake_i2c_queue(0, p, sizeof(p));
    uint8_t buf[BNO085_SHTP_MAX_PACKET];
    size_t len = 0;
    CHECK(bno085_read_packet(dev, buf, sizeof(buf), &len));
    CHECK_EQ(len, sizeof(p));
    CHECK_EQ(fake_i2c_last_timeout_us, i2c_xfer_timeout_us(sizeof(p), 100000u, 10000u));
    CHECK(fake_i2c_last_timeout_us > 10000u);
}

static void test_read_retry_full_packet(void) {
    /* Header read OK, first full read fails, retry must land at buf[0]. */
    fake_i2c_reset();
    bno085_t *dev = imu_bus_dev(IMU_SLOT_MTA);
    dev->baud_hz = 400000u;
    uint8_t p[20];
    for (size_t i = 0; i < sizeof(p); i++) {
        p[i] = (uint8_t)(0x40 + i);
    }
    p[0] = sizeof(p);
    p[1] = 0;
    p[2] = BNO085_CH_CONTROL;
    fake_i2c_queue(0, p, sizeof(p));
    fake_i2c_read_fail_after = 1; /* header ok, then one failure */
    uint8_t buf[64];
    memset(buf, 0, sizeof(buf));
    size_t len = 0;
    CHECK(bno085_read_packet(dev, buf, sizeof(buf), &len));
    CHECK_EQ(len, sizeof(p));
    CHECK(memcmp(buf, p, sizeof(p)) == 0);
}

static void test_write_tries(void) {
    fake_i2c_reset();
    bno085_t *dev = imu_bus_dev(IMU_SLOT_CRA);
    uint8_t payload[2] = {BNO085_REPORT_PRODUCT_ID_REQ, 0};

    /* Recovery: one attempt, the state machine retries later. */
    dev->write_tries = 1;
    fake_i2c_fail_writes = 5;
    CHECK(!bno085_write_packet(dev, BNO085_CH_CONTROL, payload, sizeof(payload)));
    CHECK_EQ(fake_i2c_fail_writes, 4);

    /* Default (boot/apply): up to 20 attempts. */
    dev->write_tries = 0;
    CHECK(bno085_write_packet(dev, BNO085_CH_CONTROL, payload, sizeof(payload)));
    CHECK_EQ(fake_i2c_fail_writes, 0);
    CHECK_EQ(fake_i2c_writes[1], 1);
}

int main(void) {
    test_timeout_scaling();
    test_blocking_read_scaled_timeout();
    test_read_retry_full_packet();
    test_write_tries();
    return fixb_report("test_fix_B_i2c");
}
