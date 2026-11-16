// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file sensor_core.c
 * @brief Sensor core loop on physical core 1 (EMG/IMU producers).
 */

#include "jtsmm/app/sensor_core.h"

#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/platform/spi_dma.h"
#include "jtsmm/runtime/host_link.h"
#include "hardware/gpio.h"
#include "hardware/sync.h"
#include "pico/flash.h"
#include "pico/multicore.h"
#include "pico/stdlib.h"
#include <stdio.h>

/*
 * sensor_core.c runs on PHYSICAL core 1 (launched via multicore_launch_core1) and
 * is the SOLE sensor producer: it owns all ADS/EMG SPI access and all IMU I2C
 * access, and is the only writer to the EMG and IMU rings. The comms core
 * (comms_core.c, physical core 0) never touches these buses or pushes these rings
 * -> strict SPSC (see Parallel Sensor Streaming plan).
 */

static volatile uint32_t sensor_poll_ticks;
/*
 * Investigation instrumentation (lock-free time_us_32, so it does not perturb
 * the dual-core timing the way the spinlock'd 64-bit clock would):
 *  - imu_svc_max_us:     worst-case blocking IMU I2C section per poll.
 *  - leadoff_max_us:     worst-case leadoff SDATAC/reg-read/RDATAC sequence,
 *                        during which EMG conversion is paused (== an EMG gap).
 *  - leadoff_count:      number of leadoff polls performed (expect ~1/s).
 *  - poll_max_us:        worst-case whole-poll duration (any other stall source).
 * A negative difference is skipped: a warm reset over SWD restarts the timer
 * under a running core 1, and the wrapped ~4.29e9 us pinned these maxima until
 * the next boot (bench, 2026-09-25).
 */
static volatile uint32_t imu_svc_max_us;
static volatile uint32_t leadoff_max_us;
static volatile uint32_t leadoff_count;
static volatile uint32_t poll_max_us;

void sensor_core_init(void) {
    panel_init();
}

uint32_t sensor_poll_ticks_count(void) {
    return sensor_poll_ticks;
}

uint32_t sensor_core_imu_svc_max_us(void) {
    return imu_svc_max_us;
}

uint32_t sensor_core_leadoff_max_us(void) {
    return leadoff_max_us;
}

uint32_t sensor_core_leadoff_count(void) {
    return leadoff_count;
}

uint32_t sensor_core_poll_max_us(void) {
    return poll_max_us;
}

void sensor_core_poll(void) {
    sensor_poll_ticks++;
    uint32_t poll_t0 = time_us_32();

    /* EMG has top priority: drain SPI DMA then service any pending DRDY edges. */
    for (unsigned i = 0; i < 12u; i++) {
        spi_dma_service();
        ads1299_drdy_service();
    }

    /*
     * IMU I2C is fully async: imu_stream_poll_service() only *kicks* a DMA SHTP
     * read when the IMU INT is asserted and no transfer is in flight, and
     * i2c_dma_service() polls DMA completion (no blocking i2c_*_timeout_us on this
     * hot path). So EMG SPI and IMU I2C genuinely overlap; the blocking BNO085
     * calls live only in boot/config/recovery, off this loop.
     */
    uint32_t imu_t0 = time_us_32();
    imu_stream_poll_service();
    i2c_dma_service();
    uint32_t imu_dt = time_us_32() - imu_t0;
    if ((int32_t)imu_dt >= 0 && imu_dt > imu_svc_max_us) {
        imu_svc_max_us = imu_dt;
    }

    /*
     * Lead-off status shares the ADS SPI0 bus with the async EMG DMA. Draining
     * any in-flight EMG transfer first (and only polling when the engine is idle)
     * prevents the leadoff register reads from colliding with a DMA mid-frame,
     * which would corrupt SCLK framing and wedge the RX channel.
     */
    static absolute_time_t next_leadoff;
    if (absolute_time_diff_us(get_absolute_time(), next_leadoff) <= 0) {
        spi_dma_service();
        if (!spi_dma_busy()) {
            uint32_t lo_t0 = time_us_32();
            ads1299_leadoff_poll();
            uint32_t lo_dt = time_us_32() - lo_t0;
            if ((int32_t)lo_dt >= 0 && lo_dt > leadoff_max_us) {
                leadoff_max_us = lo_dt;
            }
            leadoff_count++;
            next_leadoff = make_timeout_time_ms(1000);
        }
    }

    uint32_t poll_dt = time_us_32() - poll_t0;
    if ((int32_t)poll_dt >= 0 && poll_dt > poll_max_us) {
        poll_max_us = poll_dt;
    }
}

/*
 * Cooperative stop ("park").
 *
 * sensor_core_stop() used to call multicore_reset_core1() on a core that could
 * be anywhere: mid SPI/I2C register sequence, mid DMA re-arm, or holding a
 * hardware spinlock / SDK lock, which a reset leaves taken forever. Now the
 * comms core raises park_req; core 1 notices at the top of its loop -- between
 * whole sensor_core_poll() iterations, so no bus transaction is half done --
 * acknowledges, and waits in a RAM-resident loop until released. Only if it
 * does not acknowledge within SENSOR_CORE_PARK_TIMEOUT_US (a genuinely hung
 * core) is it reset, as before, and that is logged.
 *
 * While parked core 1 executes from RAM with IRQs enabled, so the
 * flash_safe_execute() lockout (config_flash.c) and a flash erase are safe too.
 */
#define SENSOR_CORE_PARK_TIMEOUT_US 200000u

static volatile bool park_req;
static volatile bool parked;
static bool sensor_core_launched;

/** @brief Internal helper: park loop, RAM-resident (no flash access while parked). */
static void __not_in_flash_func(sensor_core_park)(void) {
    parked = true;
    __dmb();
    while (park_req) {
        __wfe();
    }
    parked = false;
    __dmb();
}

/** @brief Internal helper: sensor_core_entry. */
static void sensor_core_entry(void) {
    /* Let config_flash's flash_safe_execute() pause this core around a flash
     * erase/program (multicore lockout victim; the handler runs from RAM). */
    (void)flash_safe_execute_core_init();
    /* IMU INT edge timestamps: the raw GPIO IRQ handler and IO_IRQ_BANK0 are
     * per core, so they are set up here, on the core that consumes them. */
    imu_int_edge_core_init();
    while (true) {
        if (park_req) {
            sensor_core_park();
        }
        sensor_core_poll();
        tight_loop_contents();
    }
}

void sensor_core_launch(void) {
    if (sensor_core_launched) {
        /* Parked (or already running): release it. */
        park_req = false;
        __dmb();
        __sev();
        return;
    }
    park_req = false;
    parked = false;
    sensor_core_launched = true;
    multicore_launch_core1(sensor_core_entry);
    /* config_flash picks its locking strategy from the victim flag; do not
     * return while core 1 runs from flash without being lockable. */
    uint32_t t0 = time_us_32();
    while (!multicore_lockout_victim_is_initialized(1) && time_us_32() - t0 < 10000u) {
        tight_loop_contents();
    }
}

bool sensor_core_stop(void) {
    if (!sensor_core_launched) {
        return true;
    }
    park_req = true;
    __dmb();
    uint32_t t0 = time_us_32();
    while (!parked) {
        if (time_us_32() - t0 > SENSOR_CORE_PARK_TIMEOUT_US) {
            /* Fallback: core 1 is stuck inside one poll. Reset it (the old
             * behaviour); sensor_core_launch() then starts it from scratch. */
            multicore_reset_core1();
            sensor_core_launched = false;
            park_req = false;
            parked = false;
            printf("sensor core: no park ack in %lu ms, core 1 hard-reset\r\n",
                   (unsigned long)(SENSOR_CORE_PARK_TIMEOUT_US / 1000u));
            host_link_send_log("sensor core did not park; core 1 hard-reset (possible stuck bus)");
            return false;
        }
        tight_loop_contents();
    }
    return true;
}

bool sensor_core_running(void) {
    return sensor_core_launched && !park_req;
}
