// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/* Field-by-field comparison of two MCU configs, shared by the config tests. */

#include "jtsmm/config.h"

#include <stdbool.h>
#include <stdio.h>

/* Every field of jtsmm_mcu_config_t, one by one (no memcmp: padding). */
static bool diffs_quiet; /* suppress the per-field report */

static int count_diffs(const jtsmm_mcu_config_t *a, const jtsmm_mcu_config_t *b) {
    int d = 0;
#define F(x)                                                                     \
    do {                                                                         \
        if (a->x != b->x) {                                                      \
            if (!diffs_quiet) {                                                  \
                fprintf(stderr, "  field differs: %s (%lld vs %lld)\n", #x,      \
                        (long long)a->x, (long long)b->x);                       \
            }                                                                    \
            d++;                                                                 \
        }                                                                        \
    } while (0)
    F(schema_version);
    F(platform.cpu_hz);
    F(platform.xosc_hz);
    F(platform.core_layout);
    F(platform.spi0_stream_hz);
    F(platform.spi0_cfg_hz);
    F(platform.i2c_hz);
    F(platform.pio_trigger_clock_hz);
    F(platform.emg_ring_slots);
    F(platform.imu_ring_slots);
    F(platform.event_ring_slots);
    F(platform.irq_emg_priority);
    F(platform.irq_imu_priority);
    F(platform.irq_trigger_priority);
    F(emg.enabled);
    F(emg.profile);
    F(emg.sample_rate_sps);
    F(emg.default_gain);
    F(emg.bias.enabled);
    F(emg.bias.channel_mask);
    F(emg.bias.ch7_readback);
    F(emg.pd_channel_mask);
    F(emg.leadoff.enabled);
    F(emg.leadoff.during_recording);
    F(emg.leadoff.ac_tone);
    F(emg.leadoff.current_na);
    F(emg.leadoff.probe_drl);
    F(emg.test_signal.enabled);
    F(emg.test_signal.amplitude_mv);
    for (int i = 0; i < 8; i++) {
        F(emg.channels[i].enabled);
        F(emg.channels[i].gain);
        F(emg.channels[i].power_down);
        F(emg.channels[i].input_mux);
    }
    F(emg.stream_channels);
    const imu_slot_config_t *sa[2] = {&a->imu_mta, &a->imu_cra};
    const imu_slot_config_t *sb[2] = {&b->imu_mta, &b->imu_cra};
    for (int s = 0; s < 2; s++) {
        const imu_slot_config_t *x = sa[s];
        const imu_slot_config_t *y = sb[s];
        int before = d;
        d += x->enabled != y->enabled;
        d += x->profile != y->profile;
        d += x->fusion_primary != y->fusion_primary;
        d += x->fusion_rate_hz != y->fusion_rate_hz;
        d += x->sync_mode != y->sync_mode;
        d += x->emg_drdy_divisor != y->emg_drdy_divisor;
        d += x->dynamic_calibration.accel != y->dynamic_calibration.accel;
        d += x->dynamic_calibration.gyro != y->dynamic_calibration.gyro;
        d += x->dynamic_calibration.mag != y->dynamic_calibration.mag;
        d += x->num_reports != y->num_reports;
        for (int i = 0; i < IMU_MAX_REPORTS && i < x->num_reports; i++) {
            d += x->reports[i].enabled != y->reports[i].enabled;
            d += x->reports[i].report_id != y->reports[i].report_id;
            d += x->reports[i].rate_hz != y->reports[i].rate_hz;
            d += x->reports[i].usb_type != y->reports[i].usb_type;
            d += x->reports[i].payload_len != y->reports[i].payload_len;
        }
        if (d != before && !diffs_quiet) {
            fprintf(stderr, "  imu slot %d differs\n", s);
        }
    }
    F(trigger.enabled);
    F(trigger.profile);
    F(trigger.rate_hz);
    F(trigger.pulse_width_us);
    F(trigger.polarity);
    F(trigger.outputs.csi1);
    F(trigger.outputs.csi2);
    F(trigger.outputs.scope);
    F(trigger.outputs.sync_led);
    F(trigger.start_on);
    F(stream_emg);
    F(stream_imu_mta);
    F(stream_imu_cra);
#undef F
    return d;
}
