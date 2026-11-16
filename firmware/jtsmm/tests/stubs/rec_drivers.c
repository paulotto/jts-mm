// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Recording stubs of the driver / platform API (see rec_drivers.h). Getters
 * return what the matching setter stored, like the real drivers' shadows. */
#include "rec_drivers.h"

#include "jtsmm/config.h"
#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/drivers/bno085.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/platform/platform_config.h"
#include "jtsmm/platform/spi_dma.h"
#include "hardware/i2c.h"
#include "pico/types.h" /* fix_B_stubs: the pico/stdlib.h subset below is defined here */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static char rec_log[REC_MAX][REC_LINE];
static int rec_n;
rec_knobs_t rec_knobs = {.imu_present = {true, true}, .ads_init_ok = true};

void rec_reset(void) {
    rec_n = 0;
}
int rec_count(void) {
    return rec_n;
}
const char *rec_at(int i) {
    return (i >= 0 && i < rec_n) ? rec_log[i] : "";
}
int rec_find(const char *call, int from) {
    for (int i = from < 0 ? 0 : from; i < rec_n; i++) {
        if (strcmp(rec_log[i], call) == 0) {
            return i;
        }
    }
    return -1;
}
int rec_count_prefix(const char *prefix) {
    int n = 0;
    for (int i = 0; i < rec_n; i++) {
        n += strncmp(rec_log[i], prefix, strlen(prefix)) == 0;
    }
    return n;
}
void rec_dump(void) {
    for (int i = 0; i < rec_n; i++) {
        fprintf(stderr, "  [%d] %s\n", i, rec_log[i]);
    }
}

static void rec(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
static void rec(const char *fmt, ...) {
    if (rec_n >= REC_MAX) {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(rec_log[rec_n++], REC_LINE, fmt, ap);
    va_end(ap);
}

/* ---- ADS1299 ------------------------------------------------------------- */
static uint32_t ads_fs = 250;
static uint8_t ads_gain = 6, ads_bias_mask = 0x3F, ads_pd = 0;
static bool ads_bias, ads_probe, ads_inited, ads_streaming;

void ads1299_set_sample_rate_hz(uint32_t fs_hz) {
    rec("ads1299_set_sample_rate_hz(%u)", (unsigned)fs_hz);
    ads_fs = fs_hz;
}
uint32_t ads1299_sample_rate_hz(void) {
    return ads_fs;
}
uint32_t ads1299_stream_spi_hz(void) {
    return jtsmm_config_active()->platform.spi0_stream_hz;
}
void ads1299_set_pga_gain(uint8_t gain) {
    rec("ads1299_set_pga_gain(%u)", (unsigned)gain);
    ads_gain = gain;
}
uint8_t ads1299_pga_gain(void) {
    return ads_gain;
}
void ads1299_set_bias_drive(bool enable) {
    rec("ads1299_set_bias_drive(%d)", enable ? 1 : 0);
    ads_bias = enable;
}
bool ads1299_bias_drive(void) {
    return ads_bias;
}
void ads1299_set_bias_channels(uint8_t mask) {
    rec("ads1299_set_bias_channels(0x%02X)", (unsigned)mask);
    ads_bias_mask = mask;
}
uint8_t ads1299_bias_channels(void) {
    return ads_bias_mask;
}
void ads1299_set_pd_channels(uint8_t mask) {
    rec("ads1299_set_pd_channels(0x%02X)", (unsigned)mask);
    ads_pd = mask;
}
uint8_t ads1299_pd_channels(void) {
    return ads_pd;
}
void ads1299_set_leadoff_monitor(bool enable) {
    rec("ads1299_set_leadoff_monitor(%d)", enable ? 1 : 0);
}
void ads1299_set_leadoff_during_recording(bool during) {
    rec("ads1299_set_leadoff_during_recording(%d)", during ? 1 : 0);
}
void ads1299_set_leadoff_current_na(uint8_t current_na) {
    rec("ads1299_set_leadoff_current_na(%u)", (unsigned)current_na);
}
void ads1299_set_leadoff_ac_tone(bool ac_tone) {
    rec("ads1299_set_leadoff_ac_tone(%d)", ac_tone ? 1 : 0);
}
void ads1299_set_drl_probe(bool enable) {
    rec("ads1299_set_drl_probe(%d)", enable ? 1 : 0);
    ads_probe = enable;
}
bool ads1299_drl_probe(void) {
    return ads_probe;
}
bool ads1299_init(void) {
    rec("ads1299_init()");
    ads_inited = rec_knobs.ads_init_ok;
    return ads_inited;
}
bool ads1299_init_ok(void) {
    return ads_inited;
}
bool ads1299_test_signal(bool enable) {
    rec("ads1299_test_signal(%d)", enable ? 1 : 0);
    return true;
}
void ads1299_set_drdy_divisor_cb(void (*cb)(uint32_t count)) {
    (void)cb;
}
bool ads1299_start_stream(void) {
    rec("ads1299_start_stream()");
    ads_streaming = true;
    return true;
}
void ads1299_stop_stream(void) {
    ads_streaming = false;
}
bool ads1299_drdy_streaming(void) {
    return ads_streaming;
}
bool ads1299_stream_status_update(const uint8_t *status) {
    (void)status;
    return true;
}
bool ads1299_stream_resync_if_needed(void) {
    return false;
}
bool ads1299_misalign_guard_drop(void) {
    return true;
}
void spi_dma_on_complete(void (*cb)(uint64_t t_us, uint32_t tag, const uint8_t *data, uint16_t len)) {
    (void)cb;
}

/* ---- trigger PIO --------------------------------------------------------- */
static bool trig_running;

bool trigger_pio_init(uint32_t clock_hz) {
    rec("trigger_pio_init(%u)", (unsigned)clock_hz);
    return true;
}
void trigger_pio_set_irq_priority(uint8_t level) {
    rec("trigger_pio_set_irq_priority(%u)", (unsigned)level);
}
void trigger_pio_set_idle_polarity(bool imx296_fast) {
    rec("trigger_pio_set_idle_polarity(%d)", imx296_fast ? 1 : 0);
}
bool trigger_pio_start(uint16_t rate_hz, uint32_t pulse_width_us, bool imx296_fast) {
    rec("trigger_pio_start(%u,%u,%d)", (unsigned)rate_hz, (unsigned)pulse_width_us, imx296_fast ? 1 : 0);
    trig_running = true;
    return true;
}
void trigger_pio_stop(void) {
    trig_running = false;
}
bool trigger_pio_running(void) {
    return trig_running;
}
uint64_t trigger_pio_id(void) {
    return 0;
}

/* ---- platform / runtime -------------------------------------------------- */
bool platform_config_validate(const platform_config_t *cfg) {
    return cfg != NULL;
}
bool platform_config_apply(const platform_config_t *cfg) {
    rec("platform_config_apply(spi0=%u spi0_cfg=%u i2c=%u pio=%u)", (unsigned)cfg->spi0_stream_hz,
        (unsigned)cfg->spi0_cfg_hz, (unsigned)cfg->i2c_hz, (unsigned)cfg->pio_trigger_clock_hz);
    return true;
}
bool sensor_core_stop(void) {
    return true;
}
void sensor_core_launch(void) {}
void fault_watchdog_hold(void) {}
void fault_watchdog_release(void) {}
void host_link_send_ack(bool ok) {
    rec("host_link_send_ack(%d)", ok ? 1 : 0);
}
void host_link_send_log(const char *msg) {
    (void)msg;
}
void host_link_reset_tx_stats(void) {}
void usb_service_poll(void) {}
void panel_set_cap_active(bool on) {
    (void)on;
}
void panel_imu_mark_active(imu_slot_t slot) {
    (void)slot;
}
void panel_update_streaming_state(void) {}

/* ---- clock (pico/stdlib subset from fix_B_stubs) -------------------------- */
static uint64_t now_us = 1000000u;
void sleep_ms(uint32_t ms) {
    now_us += (uint64_t)ms * 1000u;
}
void sleep_us(uint64_t us) {
    now_us += us;
}
void busy_wait_us_32(uint32_t us) {
    now_us += us;
}
absolute_time_t get_absolute_time(void) {
    return now_us;
}
absolute_time_t make_timeout_time_ms(uint32_t ms) {
    return now_us + (uint64_t)ms * 1000u;
}
bool time_reached(absolute_time_t t) {
    return now_us >= t;
}
void usb_service_sleep_ms(uint32_t ms) {
    sleep_ms(ms);
}
int64_t absolute_time_diff_us(absolute_time_t from, absolute_time_t to) {
    return (int64_t)(to - from);
}
uint32_t time_us_32(void) {
    return (uint32_t)now_us;
}
uint64_t time_us_64(void) {
    return now_us;
}
uint64_t t_master_us(void) {
    return now_us;
}
bool gpio_get(uint gpio) {
    (void)gpio;
    return true;
}

/* ---- BNO085 / IMU bus / I2C DMA ------------------------------------------ */
struct i2c_inst {
    int idx;
};
static struct i2c_inst i2c_insts[2] = {{0}, {1}};
i2c_inst_t *const fake_i2c0 = &i2c_insts[0];
i2c_inst_t *const fake_i2c1 = &i2c_insts[1];

static bno085_t devs[2];
static const char *dev_slot(const bno085_t *dev) {
    return dev == &devs[1] ? "cra" : "mta";
}
static const char *slot_str(int slot) {
    return slot == 1 ? "cra" : "mta";
}

bno085_t *imu_bus_dev(imu_slot_t slot) {
    return (unsigned)slot < 2 ? &devs[slot] : NULL;
}
void imu_bus_prepare_slot(imu_slot_t slot) {
    (void)slot;
}
bool imu_bus_probe(imu_slot_t slot) {
    return (unsigned)slot < 2 && rec_knobs.imu_present[slot];
}
bool imu_bus_clear(imu_slot_t slot) {
    (void)slot;
    return true;
}
void imu_int_isr_install(imu_slot_t slot) {
    (void)slot;
}
void imu_int_isr_enable(imu_slot_t slot, bool on) {
    (void)slot;
    (void)on;
}
/* INT edge timestamp latch: inert (never armed). */
static imu_edge_latch_t rec_edge[2];
void imu_int_edge_arm(imu_slot_t slot, bool on) {
    (void)slot;
    (void)on;
}
bool imu_int_edge_armed(imu_slot_t slot) {
    (void)slot;
    return false;
}
void imu_int_edge_clear(imu_slot_t slot) {
    (void)slot;
}
bool imu_int_edge_core_ready(void) {
    return true;
}
imu_edge_latch_t *imu_int_edge_latch(imu_slot_t slot) {
    return &rec_edge[(unsigned)slot < 2 ? slot : 0];
}
void i2c_dma_on_body_start(i2c_dma_slot_t slot, void (*cb)(i2c_dma_slot_t slot)) {
    (void)slot;
    (void)cb;
}
bool bno085_hw_reset(void) {
    return true;
}
void bno085_bus_config(bno085_t *dev, uint baud) {
    rec("bno085_bus_config(%s,%u)", dev_slot(dev), (unsigned)baud);
    dev->baud_hz = baud;
}
bool bno085_boot_device(bno085_t *dev) {
    (void)dev;
    return true;
}
bool bno085_set_calibration(bno085_t *dev, bool accel, bool gyro, bool mag) {
    rec("bno085_set_calibration(%s,%d,%d,%d)", dev_slot(dev), accel ? 1 : 0, gyro ? 1 : 0, mag ? 1 : 0);
    return true;
}
bool bno085_enable_report(bno085_t *dev, uint8_t report_id, uint32_t interval_us) {
    rec("bno085_enable_report(%s,0x%02X,%u)", dev_slot(dev), (unsigned)report_id, (unsigned)interval_us);
    return true;
}
bool bno085_enable_report_drain(bno085_t *dev, uint8_t report_id, uint32_t interval_us,
                                uint32_t *granted_us) {
    rec("bno085_enable_report(%s,0x%02X,%u)", dev_slot(dev), (unsigned)report_id, (unsigned)interval_us);
    if (granted_us != NULL) {
        *granted_us = interval_us;
    }
    return true;
}
void i2c_dma_abort(i2c_dma_slot_t slot) {
    (void)slot;
}
void i2c_dma_on_complete(i2c_dma_slot_t slot,
                         void (*cb)(i2c_dma_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len)) {
    (void)slot;
    (void)cb;
}
void i2c_dma_set_baud(i2c_dma_slot_t slot, uint32_t baud_hz) {
    rec("i2c_dma_set_baud(%s,%u)", slot_str((int)slot), (unsigned)baud_hz);
}

/* Inert: streaming / recovery paths of imu_config.c that no case drives. */
bool bno085_int_asserted(const bno085_t *dev) {
    (void)dev;
    return false;
}
bool bno085_parse_packet(const uint8_t *pkt, uint16_t len, bno085_report_cb_t cb, void *ctx,
                         bno085_packet_info_t *info) {
    (void)pkt;
    (void)len;
    (void)cb;
    (void)ctx;
    (void)info;
    return false;
}
bool bno085_parse_shtp(const uint8_t *pkt, uint16_t len, bno085_parse_state_t *out) {
    (void)pkt;
    (void)len;
    (void)out;
    return false;
}
bool bno085_read_packet(bno085_t *dev, uint8_t *buf, size_t buf_cap, size_t *out_len) {
    (void)dev;
    (void)buf;
    (void)buf_cap;
    if (out_len != NULL) {
        *out_len = 0;
    }
    return false;
}
bool bno085_write_packet(bno085_t *dev, uint8_t channel, const uint8_t *payload, uint16_t payload_len) {
    (void)dev;
    (void)channel;
    (void)payload;
    (void)payload_len;
    return true;
}
bool bno085_request_product_id(bno085_t *dev) {
    (void)dev;
    return true;
}
bool bno085_save_dcd(bno085_t *dev) {
    (void)dev;
    return true;
}
void bno085_seq_reset(bno085_seq_tracker_t *t) {
    (void)t;
}
uint32_t bno085_seq_update(bno085_seq_tracker_t *t, uint8_t report_id, uint8_t seq) {
    (void)t;
    (void)report_id;
    (void)seq;
    return 0;
}
uint32_t imu_bus_achieved_hz(imu_slot_t slot) {
    return (unsigned)slot < 2 ? devs[slot].baud_hz : 0;
}
void imu_bus_nrst(bool high) {
    (void)high;
}
bool i2c_dma_busy(i2c_dma_slot_t slot) {
    (void)slot;
    return false;
}
void i2c_dma_request_abort(i2c_dma_slot_t slot) {
    (void)slot;
}
bool i2c_dma_quiesce(i2c_dma_slot_t slot, uint32_t timeout_us) {
    (void)slot;
    (void)timeout_us;
    return true;
}
void i2c_dma_kick_shtp_read(i2c_dma_slot_t slot, i2c_inst_t *i2c, uint8_t addr, uint64_t t_latch_us) {
    (void)slot;
    (void)i2c;
    (void)addr;
    (void)t_latch_us;
}
uint32_t i2c_dma_timeout_count(i2c_dma_slot_t slot) {
    (void)slot;
    return 0;
}
uint32_t i2c_dma_tx_abort_count(i2c_dma_slot_t slot) {
    (void)slot;
    return 0;
}
uint32_t i2c_dma_last_read_us(i2c_dma_slot_t slot) {
    (void)slot;
    return 0;
}
uint32_t i2c_dma_max_read_us(i2c_dma_slot_t slot) {
    (void)slot;
    return 0;
}
uint16_t i2c_dma_last_len(i2c_dma_slot_t slot) {
    (void)slot;
    return 0;
}
