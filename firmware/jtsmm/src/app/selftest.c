// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file selftest.c
 * @brief Interactive selftest command implementation.
 */

#include "jtsmm/app/selftest.h"

#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/config.h"
#include "jtsmm/pins.h"
#include "jtsmm/runtime/host_link.h"
#include "jtsmm/runtime/timebase.h"
#include "jtsmm/runtime/usb_service.h"

#include "hardware/gpio.h"
#include "pico/stdlib.h"
#include <stdio.h>

/** @brief Internal helper: selftest_log. */
static void selftest_log(bool pass, const char *label) {
    printf("[%s] %s\r\n", pass ? "PASS" : "FAIL", label);
    usb_service_poll();
    char msg[80];
    snprintf(msg, sizeof(msg), "selftest %s %s", pass ? "PASS" : "FAIL", label);
    host_link_send_log(msg);
    usb_service_poll();
}

/** @brief Internal helper: selftest_led_blink. */
static bool selftest_led_blink(void) {
    const uint8_t seq[][3] = {
        {32, 0, 0},
        {0, 32, 0},
        {0, 0, 32},
        {8, 8, 8},
    };
    for (size_t i = 0; i < sizeof(seq) / sizeof(seq[0]); i++) {
        panel_set_rgb(seq[i][0], seq[i][1], seq[i][2]);
        usb_service_sleep_ms(80);
    }
    panel_set_state(PANEL_STATE_IDLE);
    return true;
}

/** @brief Internal helper: selftest_ads_spi. */
static bool selftest_ads_spi(void) {
    if (!ads1299_init()) {
        return false;
    }
    uint8_t id = 0;
    if (!ads1299_read_id(&id)) {
        return false;
    }
    return id == ADS1299_CHIP_ID_EXPECT;
}

/** @brief Internal helper: selftest_imu_probe. */
static bool selftest_imu_probe(imu_slot_t slot) {
    imu_bus_prepare_slot(slot);
    return imu_bus_probe(slot);
}

/** @brief Internal helper: selftest_imu_section. */
static bool selftest_imu_section(void) {
    bool mta = selftest_imu_probe(IMU_SLOT_MTA);
    bool cra = selftest_imu_probe(IMU_SLOT_CRA);
    printf("  IMU probe MTA=%d CRA=%d\r\n", mta ? 1 : 0, cra ? 1 : 0);
    if (!mta && !cra) {
        printf("[FAIL] IMU probe (no ACK on I2C)\r\n");
        host_link_send_log("selftest FAIL IMU probe");
        return false;
    }
    if (!cra) {
        printf("[SKIP] CRA — no BNO085 (cable not connected?)\r\n");
    }
    printf("[PASS] IMU probe\r\n");
    host_link_send_log(mta ? "selftest PASS IMU MTA" : "selftest PASS IMU CRA");
    return true;
}

/** @brief Internal helper: selftest_trigger_pulse. */
static bool selftest_trigger_pulse(void) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    if (!trigger_pio_init(cfg->platform.pio_trigger_clock_hz)) {
        return false;
    }
    uint64_t before = trigger_pio_id();
    if (!trigger_pio_start(10, 1000, true)) {
        return false;
    }
    usb_service_sleep_ms(150);
    trigger_pio_stop();
    host_link_drain();
    return trigger_pio_id() > before;
}

/** @brief Internal helper: selftest_buttons. */
static bool selftest_buttons(void) {
    int u1 = gpio_get(PIN_SW_USER1);
    int u2 = gpio_get(PIN_SW_USER2);
    printf("  buttons USER1=%d USER2=%d (0=pressed)\r\n", u1, u2);
    return true;
}

bool selftest_run(void) {
    int pass = 0;
    int total = 0;

    printf("=== jtsmm selftest ===\r\n");
    usb_service_poll();
    host_link_send_log("selftest start");

    total++;
    bool ok = selftest_led_blink();
    pass += ok ? 1 : 0;
    selftest_log(ok, "LED blink");

    total++;
    ok = selftest_ads_spi();
    pass += ok ? 1 : 0;
    selftest_log(ok, "ADS1299 SPI/CHIPID");

    total++;
    ok = selftest_imu_section();
    pass += ok ? 1 : 0;

    total++;
    ok = selftest_trigger_pulse();
    pass += ok ? 1 : 0;
    selftest_log(ok, "trigger pulse");

    total++;
    ok = selftest_buttons();
    pass += ok ? 1 : 0;
    selftest_log(ok, "buttons readback");

    printf("=== result: %d/%d ===\r\n", pass, total);
    usb_service_poll();
    return pass == total;
}
