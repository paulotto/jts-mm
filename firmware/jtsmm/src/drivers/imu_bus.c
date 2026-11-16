// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_bus.c
 * @brief Dual-IMU bus wiring and INT pending flags.
 */

#include "jtsmm/drivers/imu_bus.h"

#include "jtsmm/config.h"
#include "jtsmm/pins.h"
#include "jtsmm/platform/gpio_irq.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/runtime/timebase.h"
#include "jtsmm/runtime/usb_service.h"

#include "hardware/gpio.h"
#include "hardware/i2c.h"
#include "hardware/irq.h"
#include "hardware/structs/io_bank0.h"
#include "hardware/structs/timer.h"
#include "hardware/sync.h"
#include "hardware/timer.h"
#include "pico/platform.h"
#include "pico/stdlib.h"
#include <stdio.h>

#define BNO085_EXPECTED_ADDR BNO085_I2C_ADDR_DEFAULT
#define I2C_BUS_HZ 400000u
#define I2C_PROBE_TIMEOUT_US 25000u
#define PCA9546_I2C_ADDR 0x70u
#define PCA9546_CH_CRA 0x01u

static bno085_t imu_devs[2];
static uint32_t g_achieved_hz[2];

uint32_t imu_bus_achieved_hz(imu_slot_t slot) {
    return (unsigned)slot < 2 ? g_achieved_hz[slot] : 0;
}

/** @brief Internal helper: imu_mux_select_cra. */
static bool imu_mux_select_cra(i2c_inst_t *i2c) {
    uint8_t ch = PCA9546_CH_CRA;
    return i2c_write_timeout_us(i2c, PCA9546_I2C_ADDR, &ch, 1, false, 5000) == 1;
}

/** @brief Internal helper: i2c_bus_reinit. */
static void i2c_bus_reinit(i2c_inst_t *i2c, uint sda, uint scl, uint32_t hz, imu_slot_t slot) {
    gpio_set_function(sda, GPIO_FUNC_I2C);
    gpio_set_function(scl, GPIO_FUNC_I2C);
    gpio_pull_up(sda);
    gpio_pull_up(scl);
    /* i2c_init returns the *actual* baud the divider achieves for clk_sys; keep
     * it so `imu_status` can prove the bus really runs at the configured rate. */
    g_achieved_hz[slot] = i2c_init(i2c, hz);
    i2c_dma_set_baud((slot == IMU_SLOT_CRA) ? I2C_DMA_SLOT_CRA : I2C_DMA_SLOT_MTA, g_achieved_hz[slot]);
}

bool imu_bus_init(void) {
    const platform_config_t *plat = &jtsmm_config_active()->platform;
    uint32_t hz = plat->i2c_hz ? plat->i2c_hz : I2C_BUS_HZ;

    gpio_init(PIN_IMU_NRST);
    gpio_set_dir(PIN_IMU_NRST, GPIO_OUT);
    gpio_put(PIN_IMU_NRST, 1);

    /*
     * Runs from config apply with the sensor core parked between polls
     * (sensor_core_stop(); reset only if it hung). Completion is polled, so a
     * DMA read can still be in flight: its TX channel would keep feeding read
     * commands into the freshly re-initialised I2C block. This core owns the
     * engine now, so stop both slots first.
     */
    i2c_dma_abort(I2C_DMA_SLOT_MTA);
    i2c_dma_abort(I2C_DMA_SLOT_CRA);

    i2c_bus_reinit(i2c0, PIN_IMU_MTA_SDA, PIN_IMU_MTA_SCL, hz, IMU_SLOT_MTA);
    i2c_bus_reinit(i2c1, PIN_IMU_CRA_SDA, PIN_IMU_CRA_SCL, hz, IMU_SLOT_CRA);

    bno085_init_dev(&imu_devs[IMU_SLOT_MTA], i2c0, BNO085_EXPECTED_ADDR, PIN_IMU_MTA_INT,
                    PIN_IMU_MTA_SDA, PIN_IMU_MTA_SCL, "MTA");
    bno085_init_dev(&imu_devs[IMU_SLOT_CRA], i2c1, BNO085_EXPECTED_ADDR, PIN_IMU_CRA_INT,
                    PIN_IMU_CRA_SDA, PIN_IMU_CRA_SCL, "CRA");
    imu_devs[IMU_SLOT_MTA].baud_hz = g_achieved_hz[IMU_SLOT_MTA];
    imu_devs[IMU_SLOT_CRA].baud_hz = g_achieved_hz[IMU_SLOT_CRA];
    return true;
}

bno085_t *imu_bus_dev(imu_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return NULL;
    }
    return &imu_devs[slot];
}

void imu_bus_get_dev(imu_slot_t slot, bno085_t *out) {
    if (out && (unsigned)slot < 2) {
        *out = imu_devs[slot];
    }
}

void imu_bus_prepare_slot(imu_slot_t slot) {
    if (slot == IMU_SLOT_CRA) {
        (void)imu_mux_select_cra(imu_devs[slot].i2c);
    }
}

bool imu_bus_hw_reset(void) {
    return bno085_hw_reset();
}

void imu_bus_nrst(bool high) {
    gpio_put(PIN_IMU_NRST, high);
}

/* Open-drain emulation on SIO: drive low = output 0, release = input + pull-up.
 * Lines are never driven high, so a device holding them low is not fought. */
static void od_low(uint pin) {
    gpio_put(pin, 0);
    gpio_set_dir(pin, GPIO_OUT);
}
static void od_release(uint pin) {
    gpio_set_dir(pin, GPIO_IN);
}

bool imu_bus_clear(imu_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return false;
    }
    bno085_t *dev = &imu_devs[slot];
    uint sda = dev->sda_pin;
    uint scl = dev->scl_pin;
    const platform_config_t *plat = &jtsmm_config_active()->platform;
    uint32_t hz = plat->i2c_hz ? plat->i2c_hz : I2C_BUS_HZ;

    /* Bus recovery, with STOP:
     * hand the pins to SIO, and if a slave holds SDA low (stuck mid-byte after
     * an aborted transfer) clock SCL up to 9 times until it lets go, then
     * issue a STOP so every slave (BNO085 and the CRA PCA9546) is idle. */
    i2c_deinit(dev->i2c);
    gpio_set_function(sda, GPIO_FUNC_SIO);
    gpio_set_function(scl, GPIO_FUNC_SIO);
    od_release(sda);
    od_release(scl);
    gpio_pull_up(sda);
    gpio_pull_up(scl);
    busy_wait_us_32(10);
    if (gpio_get(scl)) {
        for (int i = 0; i < 9 && !gpio_get(sda); i++) {
            od_low(scl);
            busy_wait_us_32(5);
            od_release(scl);
            busy_wait_us_32(5);
        }
        /* STOP: SDA low while SCL low, SCL high, then SDA high. */
        od_low(scl);
        busy_wait_us_32(5);
        od_low(sda);
        busy_wait_us_32(5);
        od_release(scl);
        busy_wait_us_32(5);
        od_release(sda);
        busy_wait_us_32(5);
    }
    bool idle = gpio_get(sda) && gpio_get(scl);
    i2c_bus_reinit(dev->i2c, sda, scl, hz, slot);
    dev->baud_hz = g_achieved_hz[slot];
    return idle;
}

bool imu_bus_probe(imu_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return false;
    }
    imu_bus_prepare_slot(slot);
    bno085_t *dev = &imu_devs[slot];
    uint8_t byte = 0;
    int w = i2c_write_timeout_us(dev->i2c, dev->addr, &byte, 1, false, I2C_PROBE_TIMEOUT_US);
    int r = -2;
    if (gpio_get(dev->int_pin) == 0) {
        r = i2c_read_timeout_us(dev->i2c, dev->addr, &byte, 1, false, I2C_PROBE_TIMEOUT_US);
    }
    if (w == 1 || r == 1) {
        return true;
    }
    usb_service_poll();
    w = i2c_write_timeout_us(dev->i2c, dev->addr, &byte, 1, false, I2C_PROBE_TIMEOUT_US);
    return w == 1;
}

static volatile bool imu_pending[2];

/** @brief Internal helper: imu_int_handler. */
static void imu_int_handler(uint gpio, uint32_t events) {
    (void)events;
    if (gpio == PIN_IMU_MTA_INT) {
        imu_pending[IMU_SLOT_MTA] = true;
    } else if (gpio == PIN_IMU_CRA_INT) {
        imu_pending[IMU_SLOT_CRA] = true;
    }
}

void imu_int_isr_install(imu_slot_t slot) {
    uint pin = (slot == IMU_SLOT_CRA) ? PIN_IMU_CRA_INT : PIN_IMU_MTA_INT;
    gpio_init(pin);
    gpio_set_dir(pin, GPIO_IN);
    gpio_pull_up(pin);
    static bool registered;
    if (!registered) {
        gpio_irq_register(imu_int_handler);
        registered = true;
    }
    gpio_irq_enable_pin(PIN_IMU_MTA_INT, GPIO_IRQ_EDGE_FALL, false);
    gpio_irq_enable_pin(PIN_IMU_CRA_INT, GPIO_IRQ_EDGE_FALL, false);
}

bool imu_int_pending(imu_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return false;
    }
    return imu_pending[slot];
}

void imu_int_clear_pending(imu_slot_t slot) {
    if ((unsigned)slot < 2) {
        imu_pending[slot] = false;
    }
}

void imu_int_set_pending(imu_slot_t slot) {
    if ((unsigned)slot < 2) {
        imu_pending[slot] = true;
    }
}

void imu_int_isr_enable(imu_slot_t slot, bool on) {
    uint pin = (slot == IMU_SLOT_CRA) ? PIN_IMU_CRA_INT : PIN_IMU_MTA_INT;
    gpio_irq_enable_pin(pin, GPIO_IRQ_EDGE_FALL, on);
}

/* ---- INT edge timestamp latch (core 1), see imu_bus.h ---------------------- */

static imu_edge_latch_t imu_edge[2];
static volatile bool imu_edge_ready;
static bool imu_edge_handler_added; /* RAM survives a core-1 reset: add once */
static volatile bool imu_edge_armed_flag[2];

static inline __attribute__((always_inline)) uint imu_int_pin(int slot) {
    return slot == IMU_SLOT_CRA ? PIN_IMU_CRA_INT : PIN_IMU_MTA_INT;
}

/** @brief proc1 INTE register word and falling-edge bit for @p pin. */
static inline io_rw_32 *imu_int_proc1_inte(uint pin, uint32_t *bit) {
    *bit = (uint32_t)GPIO_IRQ_EDGE_FALL << (4u * (pin & 7u));
    return &io_bank0_hw->proc1_irq_ctrl.inte[pin >> 3];
}

/* time_us_64() read inline, so the RAM-resident ISR does not call into flash
 * (the SDK's timer_time_us_64 is a flash function). Same hi/lo/hi sequence. */
static inline __attribute__((always_inline)) uint64_t imu_isr_time_us(void) {
    timer_hw_t *t = PICO_DEFAULT_TIMER_INSTANCE();
    uint32_t hi = t->timerawh;
    uint32_t lo;
    for (;;) {
        lo = t->timerawl;
        uint32_t hi2 = t->timerawh;
        if (hi2 == hi) {
            break;
        }
        hi = hi2;
    }
    return ((uint64_t)hi << 32) | lo;
}

/** @brief Raw IO_IRQ_BANK0 handler for GPIO16/17 (core 1): timestamp, latch, acknowledge. */
static void __not_in_flash_func(imu_int_edge_isr)(void) {
    if (get_core_num() != 1u) {
        return; /* shared vector table: core 0's IO_IRQ_BANK0 is not ours */
    }
    uint64_t t = imu_isr_time_us();
    for (int s = 0; s < 2; s++) {
        uint pin = imu_int_pin(s);
        uint32_t ev = (io_bank0_hw->proc1_irq_ctrl.ints[pin >> 3] >> (4u * (pin & 7u))) & 0xFu;
        if (ev == 0u) {
            continue;
        }
        gpio_acknowledge_irq(pin, ev);
        if (ev & GPIO_IRQ_EDGE_FALL) {
            imu_edge_latch_on_edge(&imu_edge[s], t);
        }
    }
}

void imu_int_edge_core_init(void) {
    if (get_core_num() != 1u) {
        return;
    }
    /* After a core-1 reset the proc1 enables are as the old core left them. */
    for (int s = 0; s < 2; s++) {
        uint32_t bit;
        io_rw_32 *en = imu_int_proc1_inte(imu_int_pin(s), &bit);
        hw_clear_bits(en, bit);
        imu_edge_armed_flag[s] = false;
        imu_edge_latch_clear(&imu_edge[s]);
    }
    if (!imu_edge_handler_added) {
        /* The SDK asserts on a second raw handler for the same pins/core. */
        gpio_add_raw_irq_handler_masked((1u << PIN_IMU_MTA_INT) | (1u << PIN_IMU_CRA_INT),
                                        imu_int_edge_isr);
        imu_edge_handler_added = true;
    }
    irq_set_enabled(IO_IRQ_BANK0, true); /* core 1's NVIC only */
    imu_edge_ready = true;
}

bool imu_int_edge_core_ready(void) {
    return imu_edge_ready;
}

void imu_int_edge_arm(imu_slot_t slot, bool on) {
    if ((unsigned)slot >= 2) {
        return;
    }
    uint pin = imu_int_pin(slot);
    uint32_t bit;
    io_rw_32 *en = imu_int_proc1_inte(pin, &bit);
    uint32_t irq = save_and_disable_interrupts();
    if (on) {
        gpio_acknowledge_irq(pin, GPIO_IRQ_EDGE_FALL); /* stale pre-arm edge */
        imu_edge_latch_clear(&imu_edge[slot]);
        hw_set_bits(en, bit);
    } else {
        hw_clear_bits(en, bit);
        imu_edge_latch_clear(&imu_edge[slot]);
    }
    imu_edge_armed_flag[slot] = on;
    restore_interrupts(irq);
}

bool imu_int_edge_armed(imu_slot_t slot) {
    return (unsigned)slot < 2 && imu_edge_armed_flag[slot];
}

void imu_int_edge_clear(imu_slot_t slot) {
    if ((unsigned)slot < 2) {
        imu_edge_latch_clear(&imu_edge[slot]);
    }
}

imu_edge_latch_t *imu_int_edge_latch(imu_slot_t slot) {
    return &imu_edge[(unsigned)slot < 2 ? slot : 0];
}

void imu_int_watch(uint32_t ms) {
    static const char *const names[2] = {"mta", "cra"};
    uint32_t e0[2];
    uint32_t polled[2] = {0, 0};
    bool last[2];
    bool touched[2] = {false, false};
    for (int s = 0; s < 2; s++) {
        uint pin = imu_int_pin(s);
        if (gpio_get_function(pin) != GPIO_FUNC_SIO) {
            gpio_init(pin); /* never configured (no IMU apply yet): SIO input */
            gpio_set_dir(pin, GPIO_IN);
        }
        gpio_pull_up(pin); /* no INT pull-up on the HAT; production setting too */
    }
    busy_wait_us_32(100); /* let the ~50 kOhm pull-up charge the open pin */
    for (int s = 0; s < 2; s++) {
        uint pin = imu_int_pin(s);
        last[s] = gpio_get(pin);
        e0[s] = imu_edge[s].edges;
        if (!imu_edge_armed_flag[s]) { /* a streaming slot (forced) is left as is */
            uint32_t bit;
            io_rw_32 *en = imu_int_proc1_inte(pin, &bit);
            gpio_acknowledge_irq(pin, GPIO_IRQ_EDGE_FALL);
            hw_set_bits(en, bit); /* atomic alias: safe while core 1 runs */
            touched[s] = true;
        }
    }
    absolute_time_t end = make_timeout_time_ms(ms);
    uint32_t next_usb = time_us_32();
    while (!time_reached(end)) {
        for (int s = 0; s < 2; s++) {
            bool now = gpio_get(imu_int_pin(s));
            if (last[s] && !now) {
                polled[s]++;
            }
            last[s] = now;
        }
        if ((int32_t)(time_us_32() - next_usb) >= 0) {
            usb_service_poll(); /* keep CDC alive over a multi-second watch */
            next_usb += 1000u;
        }
    }
    for (int s = 0; s < 2; s++) {
        if (touched[s]) {
            uint32_t bit;
            io_rw_32 *en = imu_int_proc1_inte(imu_int_pin(s), &bit);
            hw_clear_bits(en, bit);
            imu_edge_latch_clear(&imu_edge[s]); /* bench edges must not stamp a stream */
        }
    }
    printf("imu_int_watch %lu ms (internal pull-ups on; tap INT to GND):\r\n", (unsigned long)ms);
    for (int s = 0; s < 2; s++) {
        printf("  %s GPIO%u: irq_edges=%lu poll_falls=%lu level=%d\r\n", names[s], imu_int_pin(s),
               (unsigned long)(imu_edge[s].edges - e0[s]), (unsigned long)polled[s],
               gpio_get(imu_int_pin(s)) ? 1 : 0);
    }
    if (imu_edge_ready) {
        printf("  core1 INT IRQ handler ready (irq_edges >= poll_falls expected: a wire tap bounces;"
               " irq_edges=0 with poll_falls>0 = IRQ path broken)\r\n");
    } else {
        printf("  core1 INT IRQ handler NOT installed (sensor core not started): irq_edges stay 0\r\n");
    }
}
