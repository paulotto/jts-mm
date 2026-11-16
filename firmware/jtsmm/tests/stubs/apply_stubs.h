// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/* Recording stubs for everything config_apply.c calls outside the config /
 * JSON layer (tests/stubs/apply_stubs.c). Counters are reset by apply_stubs_reset(). */

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    int sensor_stop;
    int sensor_launch;
    int emg_stream_stop;
    int imu_stream_stop;
    int trigger_stop;
    int i2c_abort;
    int platform_apply;
    int emg_apply;
    int imu_apply_mta;
    int imu_apply_cra;
    int trigger_apply;
    int acks_ok;
    int acks_fail;
    int logs;
    int wd_hold;
    int wd_release;
    /* knobs */
    bool trigger_running;
    bool platform_apply_ok;
    bool emg_apply_ok;
} apply_stubs_t;

extern apply_stubs_t apply_stubs;

void apply_stubs_reset(void);
