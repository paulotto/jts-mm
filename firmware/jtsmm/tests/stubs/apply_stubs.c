// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Recording stubs of the hardware-facing API used by config_apply.c. Only the
 * public headers are relied on, so driver/domain internals can change freely. */
#include "apply_stubs.h"

#include "jtsmm/config.h"
#include "jtsmm/domain/emg_config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/domain/trigger_config.h"
#include "jtsmm/drivers/trigger_timing.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/platform/platform_config.h"

#include <string.h>

apply_stubs_t apply_stubs;

void apply_stubs_reset(void) {
    memset(&apply_stubs, 0, sizeof(apply_stubs));
    apply_stubs.platform_apply_ok = true;
    apply_stubs.emg_apply_ok = true;
}

/* sensor core / watchdog / USB / host link / panel */
bool sensor_core_stop(void) {
    apply_stubs.sensor_stop++;
    return true;
}
void sensor_core_launch(void) {
    apply_stubs.sensor_launch++;
}
void fault_watchdog_hold(void) {
    apply_stubs.wd_hold++;
}
void fault_watchdog_release(void) {
    apply_stubs.wd_release++;
}
void usb_service_poll(void) {}
void host_link_send_ack(bool ok) {
    if (ok) {
        apply_stubs.acks_ok++;
    } else {
        apply_stubs.acks_fail++;
    }
}
void host_link_send_log(const char *msg) {
    (void)msg;
    apply_stubs.logs++;
}
void panel_update_streaming_state(void) {}

/* platform */
bool platform_config_validate(const platform_config_t *cfg) {
    return cfg != NULL && cfg->spi0_stream_hz >= 500000u && cfg->spi0_stream_hz <= 4000000u;
}
bool platform_config_apply(const platform_config_t *cfg) {
    (void)cfg;
    apply_stubs.platform_apply++;
    return apply_stubs.platform_apply_ok;
}
void i2c_dma_abort(i2c_dma_slot_t slot) {
    (void)slot;
    apply_stubs.i2c_abort++;
}

/* EMG domain */
void emg_config_resolve_profile(emg_config_t *cfg) {
    (void)cfg;
}
bool emg_config_apply(const emg_config_t *cfg) {
    (void)cfg;
    apply_stubs.emg_apply++;
    return apply_stubs.emg_apply_ok;
}
void emg_stream_stop(void) {
    apply_stubs.emg_stream_stop++;
}

/* IMU domain */
void imu_config_resolve_profile(imu_slot_config_t *cfg) {
    cfg->num_reports = 1; /* any non-empty expansion */
}
bool imu_config_validate_slot(const imu_slot_config_t *cfg, uint16_t emg_fs_hz) {
    (void)cfg;
    (void)emg_fs_hz;
    return true;
}
bool imu_config_apply_slot(imu_slot_t slot, const imu_slot_config_t *cfg) {
    (void)cfg;
    if (slot == IMU_SLOT_CRA) {
        apply_stubs.imu_apply_cra++;
    } else {
        apply_stubs.imu_apply_mta++;
    }
    return true;
}
void imu_stream_stop_slot(imu_slot_t slot) {
    (void)slot;
    apply_stubs.imu_stream_stop++;
}

/* trigger */
void trigger_config_resolve_profile(trigger_config_t *cfg) {
    cfg->rate_hz = 30;
    cfg->pulse_width_us = 12000;
}
bool trigger_config_apply(const trigger_config_t *cfg) {
    (void)cfg;
    apply_stubs.trigger_apply++;
    return true;
}
bool trigger_pio_running(void) {
    return apply_stubs.trigger_running;
}
uint64_t trigger_pio_id(void) {
    return 42;
}
void trigger_pio_stop(void) {
    apply_stubs.trigger_stop++;
    apply_stubs.trigger_running = false;
}
