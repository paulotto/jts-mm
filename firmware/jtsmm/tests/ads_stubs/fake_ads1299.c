// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Fake SPI0 + GPIO + clock with an ADS1299 register model behind them, for the
 * REAL src/drivers/ads1299.c (see fake_ads1299.h). */
#include "fake_ads1299.h"

#include "jtsmm/pins.h"
#include "hardware/gpio.h"
#include "hardware/spi.h"
#include "pico/stdlib.h"

#include <string.h>

uint8_t fake_ads_regs[FAKE_ADS_NREGS];
int fake_ads_writes;
int fake_ads_dropped_writes;

static const uint8_t reset_values[FAKE_ADS_NREGS] = {
    [0x00] = 0x3E,                                     /* ID: ADS1299, 8 ch */
    [0x01] = 0x96, [0x02] = 0xC0, [0x03] = 0x60,       /* CONFIG1..3 */
    [0x05] = 0x61, [0x06] = 0x61, [0x07] = 0x61, [0x08] = 0x61,
    [0x09] = 0x61, [0x0A] = 0x61, [0x0B] = 0x61, [0x0C] = 0x61, /* CH1..8SET */
    [0x14] = 0x0F,                                     /* GPIO */
};

static bool rdatac;
static bool cs_low;
static bool reset_low;
static int frame_idx;
static enum { PH_NONE, PH_RREG_N, PH_WREG_N, PH_RREG, PH_WREG } phase;
static uint8_t reg_ptr;
static int reg_left;

static void chip_reset(void) {
    memcpy(fake_ads_regs, reset_values, sizeof(fake_ads_regs));
    rdatac = true; /* SBAS499C §9.4.1.2: RDATAC is the power-up/reset default */
    phase = PH_NONE;
}

void fake_ads_power_on(void) {
    chip_reset();
    cs_low = false;
    reset_low = false;
    fake_ads_writes = 0;
    fake_ads_dropped_writes = 0;
}

static bool writable(uint8_t reg) {
    return reg != 0x00 && reg != 0x12 && reg != 0x13 && reg < FAKE_ADS_NREGS;
}

/* One SPI byte: returns what DOUT shifts out while @p tx shifts in. */
static uint8_t chip_byte(uint8_t tx) {
    if (!cs_low) {
        if (phase == PH_WREG) {
            fake_ads_dropped_writes++;
        }
        return 0xFF;
    }
    uint8_t rx = 0x00;
    if (frame_idx == 0) {
        if ((tx & 0xE0u) == 0x20u) {
            phase = PH_RREG_N;
            reg_ptr = tx & 0x1Fu;
        } else if ((tx & 0xE0u) == 0x40u) {
            phase = PH_WREG_N;
            reg_ptr = tx & 0x1Fu;
        } else {
            phase = PH_NONE;
            switch (tx) {
            case 0x06: chip_reset(); break;   /* RESET */
            case 0x10: rdatac = true; break;  /* RDATAC */
            case 0x11: rdatac = false; break; /* SDATAC */
            default: break;                   /* WAKEUP/STANDBY/START/STOP/RDATA */
            }
        }
    } else if (phase == PH_RREG_N || phase == PH_WREG_N) {
        reg_left = tx + 1;
        phase = phase == PH_RREG_N ? PH_RREG : PH_WREG;
    } else if (phase == PH_RREG && reg_left > 0) {
        /* In RDATAC mode the chip ignores RREG (DOUT carries conversions). */
        rx = (!rdatac && reg_ptr < FAKE_ADS_NREGS) ? fake_ads_regs[reg_ptr] : 0x00;
        reg_ptr++;
        reg_left--;
    } else if (phase == PH_WREG && reg_left > 0) {
        if (rdatac) {
            fake_ads_dropped_writes++;
        } else if (writable(reg_ptr)) {
            fake_ads_regs[reg_ptr] = (reg_ptr == 0x03) ? (uint8_t)(tx & 0xFEu) : tx; /* BIAS_STAT r/o */
            fake_ads_writes++;
        }
        reg_ptr++;
        reg_left--;
    }
    frame_idx++;
    return rx;
}

/* ---- SPI ------------------------------------------------------------------ */
struct spi_inst {
    int idx;
};
static struct spi_inst spi_inst0 = {0};
spi_inst_t *const fake_spi0 = &spi_inst0;
static spi_hw_t spi_hw;
static bool tx_pending;
static uint baud;
#define DR_EMPTY 0xFFFFFFFFu

uint spi_init(spi_inst_t *spi, uint baudrate) {
    (void)spi;
    baud = baudrate;
    return baudrate;
}
uint spi_set_baudrate(spi_inst_t *spi, uint baudrate) {
    (void)spi;
    baud = baudrate;
    return baudrate;
}
uint spi_get_baudrate(const spi_inst_t *spi) {
    (void)spi;
    return baud;
}
void spi_set_format(spi_inst_t *spi, uint data_bits, spi_cpol_t cpol, spi_cpha_t cpha, spi_order_t order) {
    (void)spi;
    (void)data_bits;
    (void)cpol;
    (void)cpha;
    (void)order;
}
spi_hw_t *spi_get_hw(spi_inst_t *spi) {
    (void)spi;
    return &spi_hw;
}
/* The driver always checks writable before it writes DR, and readable before it
 * reads it: the byte is exchanged with the chip model in between. */
bool spi_is_writable(const spi_inst_t *spi) {
    (void)spi;
    spi_hw.dr = DR_EMPTY;
    tx_pending = true;
    return true;
}
bool spi_is_readable(const spi_inst_t *spi) {
    (void)spi;
    if (!tx_pending || spi_hw.dr == DR_EMPTY) {
        return false;
    }
    tx_pending = false;
    spi_hw.dr = chip_byte((uint8_t)spi_hw.dr);
    return true;
}

/* ---- GPIO ----------------------------------------------------------------- */
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
void gpio_disable_pulls(uint gpio) {
    (void)gpio;
}
void gpio_set_function(uint gpio, enum gpio_function fn) {
    (void)gpio;
    (void)fn;
}
void gpio_put(uint gpio, bool value) {
    if (gpio == PIN_ADS_CS) {
        if (!value && !cs_low) {
            frame_idx = 0; /* CS falling edge: new command frame */
            phase = PH_NONE;
        }
        cs_low = !value;
    } else if (gpio == PIN_ADS_RESET) {
        if (value && reset_low) {
            chip_reset(); /* RESET pin pulse */
        }
        reset_low = !value;
    }
}
bool gpio_get(uint gpio) {
    (void)gpio;
    return true; /* DRDY idle high */
}

/* ---- clock ---------------------------------------------------------------- */
static uint64_t now_us = 1000000u;
void sleep_ms(uint32_t ms) {
    now_us += (uint64_t)ms * 1000u;
}
void sleep_us(uint64_t us) {
    now_us += us;
}
void busy_wait_us(uint64_t us) {
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
absolute_time_t make_timeout_time_us(uint64_t us) {
    return now_us + us;
}
absolute_time_t delayed_by_us(absolute_time_t t, uint64_t us) {
    return t + us;
}
bool time_reached(absolute_time_t t) {
    now_us++; /* polling loops make progress */
    return now_us >= t;
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

/* ---- the rest of the firmware the driver / emg_config.c call -------------- */
void usb_service_poll(void) {}
void usb_service_sleep_ms(uint32_t ms) {
    sleep_ms(ms);
}
void gpio_irq_register(gpio_irq_callback_t cb) {
    (void)cb;
}
void gpio_irq_enable_pin(uint gpio, uint32_t events, bool enabled) {
    (void)gpio;
    (void)events;
    (void)enabled;
}
bool spi_dma_busy(void) {
    return false;
}
uint32_t spi_dma_wedge_count(void) {
    return 0;
}
void spi_dma_kick_read(uint64_t t_latch_us, uint32_t tag) {
    (void)t_latch_us;
    (void)tag;
}
void spi_dma_resync(void) {}
void spi_dma_on_complete(void (*cb)(uint64_t t_us, uint32_t tag, const uint8_t *data, uint16_t len)) {
    (void)cb;
}
void panel_set_emg_leadoff(bool any) {
    (void)any;
}
void panel_set_cap_active(bool on) {
    (void)on;
}
void imu_drdy_divisor_kick(uint32_t drdy_count) {
    (void)drdy_count;
}
