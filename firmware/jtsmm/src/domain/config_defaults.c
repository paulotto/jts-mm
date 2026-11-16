// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_defaults.c
 * @brief Factory defaults for jtsmm_mcu_config_t.
 */

#include "jtsmm/config.h"

static jtsmm_mcu_config_t g_cfg;

void jtsmm_config_set_defaults(jtsmm_mcu_config_t *cfg) {
    *cfg = (jtsmm_mcu_config_t){0};
    cfg->schema_version = 1;
    cfg->platform = (platform_config_t){
        .cpu_hz = 150000000u,
        .xosc_hz = 12000000u,
        .core_layout = 0,
        .spi0_stream_hz = 4000000u,
        .spi0_cfg_hz = 500000u,
        .i2c_hz = 400000u,
        .pio_trigger_clock_hz = 1000000u,
        .emg_ring_slots = 2048u, /* ~1.0 s at 2 kHz (compact EMG slots, ringbuf.c) */
        .imu_ring_slots = 64u,
        .event_ring_slots = 32u,
        .irq_emg_priority = 0,
        .irq_imu_priority = 1,
        .irq_trigger_priority = 2,
    };
    cfg->emg = (emg_config_t){
        .enabled = true,
        .profile = EMG_PROFILE_ACQUISITION,
        .sample_rate_sps = 4000u, /* default since 2026-09-25 (flat to 500 Hz) */
        .default_gain = 6u,
        .bias = {.enabled = true, .channel_mask = 0x3Fu},
        .pd_channel_mask = 0x00u,
        .leadoff = {.enabled = true, .during_recording = true, .ac_tone = false, .current_na = 6u},
        .stream_channels = 6u,
    };
    cfg->imu_mta = (imu_slot_config_t){
        .enabled = true,
        /*
         * Boot default matches the operator template (etc/jtsmm.yaml):
         * kinematics_accel = fusion quaternion + gravity-included accelerometer.
         * imu_config_resolve_profile() expands this to the same report set and
         * rates that the host IMU_PROFILE_REPORTS["kinematics_accel"] produces
         * (fusion @ fusion_rate_hz, accelerometer @ 250), so a slot behaves
         * identically whether configured by the host or left at boot. 250 Hz is
         * the slot linear_accel used in kinematics_default (200 quantizes badly
         * to ~120 Hz next to the fusion report). Gyro is omitted because fusion +
         * gyro + a 3-axis vector throttles all three (~331/331/79).
         * A re-apply now disables reports the new config drops (imu_config.c),
         * so switching presets no longer leaves stale reports streaming.
         */
        .profile = IMU_PROFILE_KINEMATICS_ACCEL,
        .fusion_primary = IMU_FUSION_GAME_ROTATION_VECTOR,
        /*
         * The BNO085 game-rotation-vector output tops out at its native ~398 Hz
         * (report-interval quantization) regardless of a higher request, so a
         * declared 400 Hz leaves the 0.5% integrity budget with no headroom for
         * even one transient dropped report. Declare 390 Hz -- at/just below the
         * native rate -- so the sensor reliably over-delivers and the strict
         * drop-rate gate passes robustly with margin.
         */
        .fusion_rate_hz = 390u,
        .sync_mode = IMU_SYNC_INT_DRIVEN,
        /* On, as in every recording so far (config.h); the host may switch it off. */
        .dynamic_calibration = {.accel = true, .gyro = true, .mag = true},
    };
    cfg->imu_cra = cfg->imu_mta;
    cfg->imu_cra.enabled = true;
    cfg->trigger = (trigger_config_t){
        .enabled = true,
        .profile = TRIGGER_PROFILE_IMX296_DUAL_CSI,
        /* The rig's operating point (etc/jtsmm.yaml, h06 §3.4): 30 fps, 12 ms
         * IMX296 exposure. Used by a bare trig_start after `config init`. */
        .rate_hz = 30u,
        .pulse_width_us = 12000u,
        .polarity = 0,
        .outputs = {.csi1 = true, .csi2 = true, .scope = true},
    };
    cfg->stream_emg = false;
    cfg->stream_imu_mta = false;
    cfg->stream_imu_cra = false;
}

const jtsmm_mcu_config_t *jtsmm_config_active(void) {
    return &g_cfg;
}

jtsmm_mcu_config_t *jtsmm_config_mutable(void) {
    return &g_cfg;
}
