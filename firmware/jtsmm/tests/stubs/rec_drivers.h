// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/*
 * Recording stubs of the driver / platform API below the config domain
 * (tests/stubs/rec_drivers.c): every hardware-facing setter appends one line
 * "name(args)" to a call log, so a test can assert that a config value reached
 * the driver call that applies it. Link against the REAL config_apply.c,
 * config_json.c, emg_config.c, imu_config.c, trigger_config.c and
 * capture_session.c. IMU slots print as mta/cra, masks as 0x%02X.
 */

#include <stdbool.h>
#include <stddef.h>

#define REC_MAX 512
#define REC_LINE 96

/** Clear the call log (knobs keep their values). */
void rec_reset(void);
/** Number of logged calls. */
int rec_count(void);
/** The i-th logged call. */
const char *rec_at(int i);
/** Index of the first call equal to @p call at or after @p from; -1 if none. */
int rec_find(const char *call, int from);
/** Number of calls starting with @p prefix. */
int rec_count_prefix(const char *prefix);
/** Print the log to stderr (on a failure). */
void rec_dump(void);
/** Type @p line (plus CR LF) into the real cmd_parser (tests/stubs/rec_cmd_stubs.c). */
void rec_cdc_send(const char *line);

/** imu_int_watch() calls and the duration of the last one (rec_cmd_stubs.c). */
extern int rec_imu_int_watch_calls;
extern unsigned rec_imu_int_watch_ms;

/** Knobs: device presence / results the stubs report. */
typedef struct {
    bool imu_present[2]; /* imu_bus_probe() */
    bool ads_init_ok;    /* ads1299_init() */
} rec_knobs_t;
extern rec_knobs_t rec_knobs;
