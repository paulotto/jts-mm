// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Link stand-ins for the real cmd_parser.c in test_config_apply_drivers: a CDC
 * input queue (rec_cdc_send) plus inert versions of the bench-command
 * back-ends no plumbing case uses. Recording driver stubs: rec_drivers.c. */
#include "rec_drivers.h"

#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/runtime/cmd_parser.h"

#include <stdint.h>
#include <string.h>

#define PICO_ERROR_TIMEOUT (-1)

static char cdc_rx[4200];
static size_t cdc_len, cdc_pos;

int getchar_timeout_us(uint32_t timeout_us) {
    (void)timeout_us;
    return cdc_pos < cdc_len ? (unsigned char)cdc_rx[cdc_pos++] : PICO_ERROR_TIMEOUT;
}

void rec_cdc_send(const char *line) {
    cdc_len = cdc_pos = 0;
    size_t n = strlen(line);
    if (n + 2 > sizeof(cdc_rx)) {
        return;
    }
    memcpy(cdc_rx, line, n);
    memcpy(cdc_rx + n, "\r\n", 2);
    cdc_len = n + 2;
    while (cdc_pos < cdc_len) {
        cmd_parser_poll();
    }
}

/* ---- inert bench-command back-ends ---------------------------------------- */
void ads1299_drdy_watch(void) {}
int rec_imu_int_watch_calls;
unsigned rec_imu_int_watch_ms;
void imu_int_watch(uint32_t ms) {
    rec_imu_int_watch_calls++;
    rec_imu_int_watch_ms = (unsigned)ms;
}
void ads1299_status(void) {}
void ads1299_regs(void) {}
uint32_t ads1299_frame_misalign_count(void) {
    return 0;
}
uint32_t ads1299_frame_misalign_max_streak(void) {
    return 0;
}
uint32_t ads1299_frame_resync_count(void) {
    return 0;
}
uint32_t ads1299_frame_resync_giveup_count(void) {
    return 0;
}
uint8_t ads1299_frame_misalign_last_status(void) {
    return 0;
}
bool ads1299_misalign_guard_resync(void) {
    return true;
}
void ads1299_set_misalign_guard(bool drop, bool resync) {
    (void)drop;
    (void)resync;
}
void comms_core_send_stats(void) {}
bool config_flash_erase(void) {
    return false;
}
bool config_flash_load(char *buf, size_t cap) {
    (void)buf;
    (void)cap;
    return false;
}
bool config_flash_save(const char *json) {
    (void)json;
    return false;
}
const char *fault_reset_reason(void) {
    return "test";
}
bool host_link_send_record(uint8_t type, uint64_t t_master_us, const uint8_t *payload, uint16_t len) {
    (void)type;
    (void)t_master_us;
    (void)payload;
    (void)len;
    return true;
}
bool led_flash_pio_init(uint32_t clk_hz) {
    (void)clk_hz;
    return true;
}
bool led_flash_pio_once(uint32_t pulse_width_us) {
    (void)pulse_width_us;
    return true;
}
bool led_flash_pio_start(uint16_t rate_hz, uint32_t pulse_width_us) {
    (void)rate_hz;
    (void)pulse_width_us;
    return true;
}
void led_flash_pio_stop(void) {}
bool panel_panel_test_busy(void) {
    return false;
}
bool panel_panel_test_rx_quiet(void) {
    return false;
}
void panel_panel_test_start(void) {}
void panel_set_rgb(uint8_t r, uint8_t g, uint8_t b) {
    (void)r;
    (void)g;
    (void)b;
}
void panel_set_state(panel_state_t state) {
    (void)state;
}
void platform_config_status(void) {}
bool selftest_run(void) {
    return true;
}
uint32_t spi_dma_resync_count(void) {
    return 0;
}
uint32_t spi_dma_wedge_count(void) {
    return 0;
}
void trigger_pio_print_status(void) {}
void trigger_pio_set_compensation(bool on) {
    (void)on;
}
void trigger_pio_set_stamp_mode(trigger_stamp_mode_t mode) {
    (void)mode;
}
