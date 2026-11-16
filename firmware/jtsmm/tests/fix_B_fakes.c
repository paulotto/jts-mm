// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Fakes for the fix-B host tests (see fix_B_fakes.h). */

#include "fix_B_fakes.h"

#include "jtsmm/drivers/bno085.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/platform/i2c_dma.h"
#include "hardware/gpio.h"
#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <string.h>

/* ---- clock --------------------------------------------------------------- */
uint64_t fake_now_us = 1000000u;

void sleep_ms(uint32_t ms) {
    fake_now_us += (uint64_t)ms * 1000u;
}
void sleep_us(uint64_t us) {
    fake_now_us += us;
}
void busy_wait_us_32(uint32_t us) {
    fake_now_us += us;
}
absolute_time_t get_absolute_time(void) {
    return fake_now_us;
}
absolute_time_t make_timeout_time_ms(uint32_t ms) {
    return fake_now_us + (uint64_t)ms * 1000u;
}
bool time_reached(absolute_time_t t) {
    return fake_now_us >= t;
}
int64_t absolute_time_diff_us(absolute_time_t from, absolute_time_t to) {
    return (int64_t)(to - from);
}
uint32_t time_us_32(void) {
    return (uint32_t)fake_now_us;
}
uint64_t time_us_64(void) {
    return fake_now_us;
}
uint64_t t_master_us(void) {
    return fake_now_us;
}

/* ---- GPIO ---------------------------------------------------------------- */
bool fake_gpio_level[64] = {
    [0 ... 63] = true,
};
void gpio_init(uint gpio) {
    (void)gpio;
}
void gpio_set_dir(uint gpio, bool out) {
    (void)gpio;
    (void)out;
}
void gpio_pull_up(uint gpio) {
    (void)gpio;
}
void gpio_put(uint gpio, bool value) {
    if (gpio < 64) {
        fake_gpio_level[gpio] = value;
    }
}
bool gpio_get(uint gpio) {
    return gpio < 64 ? fake_gpio_level[gpio] : true;
}
void gpio_set_function(uint gpio, enum gpio_function fn) {
    (void)gpio;
    (void)fn;
}

/* ---- scripted BNO085 on I2C ---------------------------------------------- */
struct i2c_inst {
    int idx;
};
static struct i2c_inst fake_i2c_inst[2] = {{0}, {1}};
i2c_inst_t *const fake_i2c0 = &fake_i2c_inst[0];
i2c_inst_t *const fake_i2c1 = &fake_i2c_inst[1];

#define FAKE_QUEUE 16
static uint8_t queue_pkt[2][FAKE_QUEUE][BNO085_SHTP_MAX_PACKET];
static size_t queue_len[2][FAKE_QUEUE];
static int queue_head[2], queue_count[2];
/* Bytes of the head packet already delivered by a partial read (0 = none). */
static size_t queue_sent[2];

int fake_i2c_writes[2];
uint8_t fake_i2c_write_log[2][FAKE_WRITE_LOG][64];
uint8_t fake_i2c_last_write[2][64];
size_t fake_i2c_last_write_len[2];
int fake_i2c_fail_writes;
int fake_i2c_read_fail_after = -1;
bool fake_i2c_dead[2];
int fake_i2c_dead_attempts[2];
unsigned fake_i2c_last_timeout_us;

void fake_i2c_queue(int bus, const uint8_t *pkt, size_t len) {
    if (queue_count[bus] >= FAKE_QUEUE || len > BNO085_SHTP_MAX_PACKET) {
        return;
    }
    int slot = (queue_head[bus] + queue_count[bus]) % FAKE_QUEUE;
    memcpy(queue_pkt[bus][slot], pkt, len);
    queue_len[bus][slot] = len;
    queue_count[bus]++;
}

void fake_queue_boot(int bus) {
    uint8_t reset_done[5] = {5, 0, BNO085_CH_EXECUTABLE, 0, BNO085_EXEC_RESET_COMPLETE};
    uint8_t wake[5] = {5, 0, BNO085_CH_WAKE, 0, 0};
    uint8_t pid[20] = {20, 0, BNO085_CH_CONTROL, 0, BNO085_REPORT_PRODUCT_ID_RESP};
    fake_i2c_queue(bus, reset_done, sizeof(reset_done));
    fake_i2c_queue(bus, wake, sizeof(wake));
    fake_i2c_queue(bus, pid, sizeof(pid));
}

void fake_i2c_reset(void) {
    memset(queue_head, 0, sizeof(queue_head));
    memset(queue_count, 0, sizeof(queue_count));
    memset(queue_sent, 0, sizeof(queue_sent));
    memset(fake_i2c_writes, 0, sizeof(fake_i2c_writes));
    memset(fake_i2c_last_write_len, 0, sizeof(fake_i2c_last_write_len));
    fake_i2c_fail_writes = 0;
    fake_i2c_read_fail_after = -1;
    memset(fake_i2c_dead, 0, sizeof(fake_i2c_dead));
    memset(fake_i2c_dead_attempts, 0, sizeof(fake_i2c_dead_attempts));
}

uint i2c_init(i2c_inst_t *i2c, uint baudrate) {
    (void)i2c;
    return baudrate;
}
void i2c_deinit(i2c_inst_t *i2c) {
    (void)i2c;
}

int i2c_write_timeout_us(i2c_inst_t *i2c, uint8_t addr, const uint8_t *src, size_t len, bool nostop,
                         uint timeout_us) {
    (void)addr;
    (void)nostop;
    fake_i2c_last_timeout_us = timeout_us;
    fake_now_us += 50u + (uint64_t)len * 25u; /* wire time, so bounded loops end */
    if (fake_i2c_dead[i2c->idx]) {
        fake_i2c_dead_attempts[i2c->idx]++;
        return PICO_ERROR_GENERIC; /* address NACK */
    }
    if (fake_i2c_fail_writes > 0) {
        fake_i2c_fail_writes--;
        fake_now_us += timeout_us;
        return PICO_ERROR_TIMEOUT;
    }
    int bus = i2c->idx;
    size_t n = len < sizeof(fake_i2c_last_write[bus]) ? len : sizeof(fake_i2c_last_write[bus]);
    memcpy(fake_i2c_last_write[bus], src, n);
    if (fake_i2c_writes[bus] < FAKE_WRITE_LOG) {
        memcpy(fake_i2c_write_log[bus][fake_i2c_writes[bus]], src, n);
    }
    fake_i2c_writes[bus]++;
    fake_i2c_last_write_len[bus] = len;
    return (int)len;
}

/* Like the BNO085 over I2C (SHTP): a read returns a 4-byte header and then
 * cargo. A read shorter than the packet leaves the rest pending; the next read
 * delivers it behind a fresh header whose length is the rest (+4) and whose
 * continuation bit (bit 15) is set. So a 4-byte header pre-read followed by a
 * full-length read returns the whole packet -- flagged as a continuation,
 * which the drivers must undo (bench, 2026-09-25). An empty queue reads as
 * length 0. */
int i2c_read_timeout_us(i2c_inst_t *i2c, uint8_t addr, uint8_t *dst, size_t len, bool nostop,
                        uint timeout_us) {
    (void)addr;
    (void)nostop;
    fake_i2c_last_timeout_us = timeout_us;
    fake_now_us += 50u + (uint64_t)len * 25u; /* wire time, so bounded loops end */
    if (fake_i2c_dead[i2c->idx]) {
        return PICO_ERROR_GENERIC;
    }
    if (fake_i2c_read_fail_after >= 0 && fake_i2c_read_fail_after-- == 0) {
        return PICO_ERROR_TIMEOUT; /* this one read fails; cargo stays pending */
    }
    int bus = i2c->idx;
    memset(dst, 0, len);
    if (queue_count[bus] == 0) {
        return (int)len;
    }
    int h = queue_head[bus];
    const uint8_t *pkt = queue_pkt[bus][h];
    size_t total = queue_len[bus][h];
    size_t sent = queue_sent[bus];
    /* What this read carries: the header + the cargo not yet delivered. */
    uint8_t seg[BNO085_SHTP_MAX_PACKET];
    size_t seg_len;
    if (sent == 0) {
        memcpy(seg, pkt, total);
        seg_len = total;
    } else {
        seg_len = 4u + (total - sent);
        uint16_t l = (uint16_t)(seg_len | 0x8000u);
        seg[0] = (uint8_t)(l & 0xFFu);
        seg[1] = (uint8_t)(l >> 8);
        seg[2] = pkt[2];
        seg[3] = pkt[3];
        memcpy(seg + 4, pkt + sent, total - sent);
    }
    size_t n = len < seg_len ? len : seg_len;
    memcpy(dst, seg, n);
    if (len >= seg_len) {
        queue_head[bus] = (h + 1) % FAKE_QUEUE;
        queue_count[bus]--;
        queue_sent[bus] = 0;
    } else {
        /* Packet bytes delivered so far; a continuation's own header is not cargo. */
        queue_sent[bus] = sent == 0 ? (n > 4u ? n : 4u) : sent + (n > 4u ? n - 4u : 0u);
    }
    return (int)len;
}

/* ---- runtime / panel ------------------------------------------------------ */
int fake_panel_active[2];
char fake_last_log[256];

void panel_imu_mark_active(imu_slot_t slot) {
    if ((unsigned)slot < 2) {
        fake_panel_active[slot]++;
    }
}
void host_link_send_log(const char *msg) {
    strncpy(fake_last_log, msg, sizeof(fake_last_log) - 1);
}
void usb_service_poll(void) {}
void usb_service_sleep_ms(uint32_t ms) {
    sleep_ms(ms);
}

/* ---- I2C DMA (platform) --------------------------------------------------- */
bool fake_dma_cb_set[2];
fake_dma_cb_t fake_dma_cb[2];
void i2c_dma_on_complete(i2c_dma_slot_t slot,
                         void (*cb)(i2c_dma_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len)) {
    fake_dma_cb_set[slot] = (cb != NULL);
    fake_dma_cb[slot] = cb;
}
bool i2c_dma_busy(i2c_dma_slot_t slot) {
    (void)slot;
    return false;
}
int fake_dma_aborts[2];
int fake_dma_quiesces[2];
bool fake_dma_quiesce_ok = true;
bool fake_dma_quiesce_fail[2];
void i2c_dma_abort(i2c_dma_slot_t slot) {
    fake_dma_aborts[slot]++;
}
void i2c_dma_request_abort(i2c_dma_slot_t slot) {
    fake_dma_aborts[slot]++;
}
bool i2c_dma_quiesce(i2c_dma_slot_t slot, uint32_t timeout_us) {
    (void)timeout_us;
    fake_dma_quiesces[slot]++;
    return fake_dma_quiesce_ok && !fake_dma_quiesce_fail[slot];
}
void i2c_dma_set_baud(i2c_dma_slot_t slot, uint32_t baud_hz) {
    (void)slot;
    (void)baud_hz;
}
uint32_t i2c_dma_timeout_count(i2c_dma_slot_t slot) {
    (void)slot;
    return 0;
}
int fake_dma_kicks[2];
uint64_t fake_dma_kick_t[2];
void i2c_dma_kick_shtp_read(i2c_dma_slot_t slot, i2c_inst_t *i2c, uint8_t addr, uint64_t t_latch_us) {
    (void)i2c;
    (void)addr;
    fake_dma_kicks[slot]++;
    fake_dma_kick_t[slot] = t_latch_us;
}
void (*fake_dma_body_cb[2])(i2c_dma_slot_t slot);
void i2c_dma_on_body_start(i2c_dma_slot_t slot, void (*cb)(i2c_dma_slot_t slot)) {
    fake_dma_body_cb[slot] = cb;
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

/* ---- IMU bus (driver) ----------------------------------------------------- */
static bno085_t fake_devs[2];
static bool fake_devs_init;
bool fake_probe_ok[2] = {true, true};

bno085_t *imu_bus_dev(imu_slot_t slot) {
    if (!fake_devs_init) {
        bno085_init_dev(&fake_devs[0], i2c0, 0x4A, 20, 4, 5, "MTA");
        bno085_init_dev(&fake_devs[1], i2c1, 0x4A, 21, 6, 7, "CRA");
        fake_devs_init = true;
    }
    return (unsigned)slot < 2 ? &fake_devs[slot] : NULL;
}
void imu_bus_prepare_slot(imu_slot_t slot) {
    (void)slot;
}
bool imu_bus_probe(imu_slot_t slot) {
    return (unsigned)slot < 2 && fake_probe_ok[slot];
}
uint32_t imu_bus_achieved_hz(imu_slot_t slot) {
    (void)slot;
    return 400000u;
}
void imu_int_isr_install(imu_slot_t slot) {
    (void)slot;
}
int fake_bus_clears[2];
int fake_nrst_pulses;
bool fake_nrst_level = true;
bool imu_bus_clear(imu_slot_t slot) {
    fake_bus_clears[slot]++;
    return true;
}
void imu_bus_nrst(bool high) {
    if (fake_nrst_level && !high) {
        fake_nrst_pulses++;
    }
    fake_nrst_level = high;
}

/* ---- config apply (runtime) ----------------------------------------------- */
bool fake_config_apply_busy;
bool config_apply_busy(void) {
    return fake_config_apply_busy;
}
void imu_int_isr_enable(imu_slot_t slot, bool on) {
    (void)slot;
    (void)on;
}

/* ---- INT edge latch (imu_bus.c): real latch struct, fake arming ----------- */
imu_edge_latch_t fake_edge[2];
bool fake_edge_armed[2];
int fake_edge_arms[2];
void imu_int_edge_arm(imu_slot_t slot, bool on) {
    if (on) {
        fake_edge_arms[slot]++;
    }
    imu_edge_latch_clear(&fake_edge[slot]);
    fake_edge_armed[slot] = on;
}
bool imu_int_edge_armed(imu_slot_t slot) {
    return (unsigned)slot < 2 && fake_edge_armed[slot];
}
void imu_int_edge_clear(imu_slot_t slot) {
    imu_edge_latch_clear(&fake_edge[slot]);
}
bool imu_int_edge_core_ready(void) {
    return true;
}
imu_edge_latch_t *imu_int_edge_latch(imu_slot_t slot) {
    return &fake_edge[(unsigned)slot < 2 ? slot : 0];
}
