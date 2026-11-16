// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file platform_config.c
 * @brief Platform clock/bus/IRQ apply.
 */

#include "jtsmm/platform/platform_config.h"

#include "jtsmm/app/sensor_core.h"
#include "jtsmm/domain/emg_config.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/config.h"
#include "jtsmm/pins.h"
#include "jtsmm/platform/dma_irq.h"
#include "jtsmm/platform/gpio_irq.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/platform/spi_dma.h"
#include "jtsmm/runtime/ringbuf.h"
#include "jtsmm/runtime/timebase.h"

#include "hardware/clocks.h"
#include "hardware/gpio.h"
#include "hardware/spi.h"
#include "pico/stdlib.h"
#include <stdio.h>

static platform_config_t active_platform;

bool platform_config_validate(const platform_config_t *cfg) {
    if (cfg == NULL) {
        return false;
    }
    if (cfg->spi0_stream_hz < 500000u || cfg->spi0_stream_hz > 4000000u) {
        /* The ADS1299 driver caps the stream clock at 4 MHz; refuse more here
         * so the applied config says what the bus actually runs at. */
        printf("config apply: platform.spi0.hz=%lu unsupported (500000..4000000)\r\n",
               (unsigned long)cfg->spi0_stream_hz);
        return false;
    }
    if (cfg->i2c_hz < 100000u || cfg->i2c_hz > 400000u) {
        printf("config apply: platform.i2c.hz=%lu unsupported (100000..400000)\r\n",
               (unsigned long)cfg->i2c_hz);
        return false;
    }
    if (cfg->pio_trigger_clock_hz != 1000000u) {
        /* trigger_timing assumes 1 PIO tick = 1 us; config_json refuses this
         * too, this catches any other path into the config. */
        printf("config apply: pio trigger clock %lu Hz unsupported (must be 1000000)\r\n",
               (unsigned long)cfg->pio_trigger_clock_hz);
        return false;
    }
    if (cfg->emg_ring_slots < RINGBUF_MIN_CAPACITY || cfg->imu_ring_slots < RINGBUF_MIN_CAPACITY ||
        cfg->event_ring_slots < RINGBUF_MIN_CAPACITY) {
        printf("config apply: ring slots emg=%u imu=%u event=%u; each ring needs >= %u\r\n",
               cfg->emg_ring_slots, cfg->imu_ring_slots, cfg->event_ring_slots,
               (unsigned)RINGBUF_MIN_CAPACITY);
        return false;
    }
    /* No total-slot cap: slots differ in size per ring, and each ring is clamped
     * to its own backing store (and reported) by ring_apply(). */
    return true;
}

/** @brief Internal helper: size one ring and say so if it was clamped. */
static void ring_apply(ring_id_t id, const char *name, uint32_t requested) {
    uint32_t used = ringbuf_set_capacity(id, requested);
    if (used != requested) {
        printf("platform: %s ring %lu slots requested, %lu used (backing store)\r\n", name,
               (unsigned long)requested, (unsigned long)used);
    }
}

void platform_init_buses(void) {
    spi_init(spi0, 500000u);
    spi_set_format(spi0, 8, SPI_CPOL_0, SPI_CPHA_1, SPI_MSB_FIRST);
    gpio_set_function(PIN_ADS_MISO, GPIO_FUNC_SPI);
    gpio_set_function(PIN_ADS_SCLK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_ADS_MOSI, GPIO_FUNC_SPI);

    (void)imu_bus_init();
}

void platform_irq_init(void) {
    gpio_irq_init();
    dma_irq_init();
}

bool platform_config_apply(const platform_config_t *cfg) {
    if (!platform_config_validate(cfg)) {
        return false;
    }
    active_platform = *cfg;

    ringbuf_init_all();
    ring_apply(RING_EMG, "emg", cfg->emg_ring_slots);
    ring_apply(RING_IMU_MTA, "imu_mta", cfg->imu_ring_slots);
    ring_apply(RING_IMU_CRA, "imu_cra", cfg->imu_ring_slots);
    ring_apply(RING_EVENTS, "event", cfg->event_ring_slots);

    timebase_init();
    platform_irq_init();
    platform_init_buses();
    spi_dma_init();
    i2c_dma_init();
    return true;
}

void platform_config_status(void) {
    printf("platform_status:\r\n");
    printf("  cpu_hz=%lu spi0=%lu i2c=%lu\r\n", (unsigned long)active_platform.cpu_hz,
           (unsigned long)active_platform.spi0_stream_hz, (unsigned long)active_platform.i2c_hz);
    printf("  rings emg=%lu/%lu imu_mta=%lu/%lu imu_cra=%lu/%lu events=%lu/%lu\r\n",
           (unsigned long)ringbuf_count(RING_EMG), (unsigned long)ringbuf_capacity(RING_EMG),
           (unsigned long)ringbuf_count(RING_IMU_MTA), (unsigned long)ringbuf_capacity(RING_IMU_MTA),
           (unsigned long)ringbuf_count(RING_IMU_CRA), (unsigned long)ringbuf_capacity(RING_IMU_CRA),
           (unsigned long)ringbuf_count(RING_EVENTS), (unsigned long)ringbuf_capacity(RING_EVENTS));
    printf("  drops emg=%lu imu_mta=%lu imu_cra=%lu events=%lu\r\n",
           (unsigned long)ringbuf_drops(RING_EMG), (unsigned long)ringbuf_drops(RING_IMU_MTA),
           (unsigned long)ringbuf_drops(RING_IMU_CRA), (unsigned long)ringbuf_drops(RING_EVENTS));
    printf("  emg_push_ok=%lu spi_dma_done=%lu spi_dma_no_cb=%lu sensor_poll_ticks=%lu\r\n",
           (unsigned long)emg_stream_push_ok_count(),
           (unsigned long)spi_dma_complete_count(),
           (unsigned long)spi_dma_complete_no_cb_count(),
           (unsigned long)sensor_poll_ticks_count());
}
