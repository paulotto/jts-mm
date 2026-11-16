// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * config_apply state machine against recording stubs (tests/stubs/apply_stubs.c):
 * a config that fails validation must leave the running config, streams and
 * sensor core untouched; a good one quiesces, applies and relaunches.
 */
#include "jtsmm/runtime/config_apply.h"
#include "jtsmm/runtime/config_json.h"

#include "apply_stubs.h"
#include "check.h"
#include "config_compare.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GOLDEN_MCU_APPLY_JSON
#define GOLDEN_MCU_APPLY_JSON "../../../schema/v1/golden/mcu_apply_template.json"
#endif

static char *read_file(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (fh == NULL) {
        return NULL;
    }
    fseek(fh, 0, SEEK_END);
    long size = ftell(fh);
    fseek(fh, 0, SEEK_SET);
    char *buf = (char *)malloc((size_t)(size > 0 ? size : 0) + 1u);
    size_t got = (buf && size > 0) ? fread(buf, 1u, (size_t)size, fh) : 0;
    fclose(fh);
    if (buf == NULL || got != (size_t)size) {
        free(buf);
        return NULL;
    }
    buf[size] = '\0';
    return buf;
}

/* Compact the golden (multi-line) JSON into one line, as the host sends it. */
static void one_line(char *s) {
    char *w = s;
    for (char *r = s; *r; r++) {
        if (*r != '\n' && *r != '\r') {
            *w++ = *r;
        }
    }
    *w = '\0';
}

static void run_to_idle(void) {
    for (int i = 0; i < 100 && config_apply_busy(); i++) {
        config_apply_poll();
    }
    CHECK(!config_apply_busy());
}

static void test_good_apply(const char *golden) {
    apply_stubs_reset();
    jtsmm_config_set_defaults(jtsmm_config_mutable());
    jtsmm_config_mutable()->stream_emg = true;
    apply_stubs.trigger_running = true;
    CHECK(config_apply_start_json(golden));
    CHECK(config_apply_busy());
    CHECK(!config_apply_start_json(golden)); /* busy: second start refused */
    CHECK(!config_apply_start_defaults());
    run_to_idle();
    CHECK(config_apply_last_ok());
    CHECK_EQ_U(apply_stubs.acks_ok, 1);
    CHECK_EQ_U(apply_stubs.acks_fail, 0);
    /* quiesce: every stream stopped, trigger stop logged, core 1 parked, DMA aborted */
    CHECK_EQ_U(apply_stubs.trigger_stop, 1);
    CHECK_EQ_U(apply_stubs.emg_stream_stop, 1);
    CHECK_EQ_U(apply_stubs.imu_stream_stop, 2);
    CHECK_EQ_U(apply_stubs.sensor_stop, 1);
    CHECK_EQ_U(apply_stubs.i2c_abort, 2);
    CHECK(apply_stubs.logs >= 1);
    CHECK(!jtsmm_config_active()->stream_emg);
    /* applied in order, core 1 relaunched, watchdog hold balanced */
    CHECK_EQ_U(apply_stubs.platform_apply, 1);
    CHECK_EQ_U(apply_stubs.emg_apply, 1);
    CHECK_EQ_U(apply_stubs.imu_apply_mta, 1);
    CHECK_EQ_U(apply_stubs.imu_apply_cra, 1);
    CHECK_EQ_U(apply_stubs.trigger_apply, 1);
    CHECK_EQ_U(apply_stubs.sensor_launch, 1);
    CHECK_EQ_U(apply_stubs.wd_hold, apply_stubs.wd_release);
    CHECK_EQ_U(jtsmm_config_active()->trigger.rate_hz, 40); /* golden = 40 Hz template */
    CHECK_EQ_U(jtsmm_config_active()->trigger.pulse_width_us, 12000);
}

/* The finding: a bad config used to overwrite the live config with defaults and
 * stop core 1 before the JSON was even parsed. */
static void test_bad_apply_keeps_running_config(const char *golden) {
    const char *bad[] = {
        "{\"platform\":",                                        /* malformed */
        "{\"platform\":{},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},\"trigger\":null}", /* typed */
        "{\"platform\":{},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},"
        "\"trigger\":{\"rate_hz\":100,\"pulse\":{\"width_us\":12000}}}", /* width >= period */
        "{\"platform\":{\"spi0\":{\"hz\":8000000}},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},"
        "\"trigger\":{}}",                                       /* platform validation */
    };
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        /* A running, non-default config with streams on. */
        apply_stubs_reset();
        jtsmm_config_set_defaults(jtsmm_config_mutable());
        CHECK(config_json_parse(golden, jtsmm_config_mutable()));
        jtsmm_config_mutable()->trigger.rate_hz = 45;
        jtsmm_config_mutable()->stream_emg = true;
        jtsmm_config_mutable()->stream_imu_mta = true;
        apply_stubs.trigger_running = true;
        jtsmm_mcu_config_t before = *jtsmm_config_active();

        CHECK(config_apply_start_json(bad[i]));
        run_to_idle();
        if (count_diffs(&before, jtsmm_config_active()) != 0) {
            fprintf(stderr, "  bad case %zu changed the live config\n", i);
            check_failures++;
        }
        CHECK(!config_apply_last_ok());
        CHECK_EQ_U(apply_stubs.acks_fail, 1);
        CHECK_EQ_U(apply_stubs.sensor_stop, 0);
        CHECK_EQ_U(apply_stubs.trigger_stop, 0);
        CHECK_EQ_U(apply_stubs.emg_stream_stop, 0);
        CHECK_EQ_U(apply_stubs.imu_stream_stop, 0);
        CHECK_EQ_U(apply_stubs.platform_apply, 0);
        CHECK_EQ_U(apply_stubs.wd_hold, 0);
        CHECK(apply_stubs.trigger_running);
    }
}

static void test_hardware_failures(const char *golden) {
    /* Platform failed: buses not up, so core 1 must stay stopped. */
    apply_stubs_reset();
    apply_stubs.platform_apply_ok = false;
    CHECK(config_apply_start_json(golden));
    run_to_idle();
    CHECK(!config_apply_last_ok());
    CHECK_EQ_U(apply_stubs.sensor_launch, 0);
    CHECK_EQ_U(apply_stubs.acks_fail, 1);
    CHECK_EQ_U(apply_stubs.wd_hold, apply_stubs.wd_release);

    /* A later sensor step failed: platform is up, keep what works running. */
    apply_stubs_reset();
    apply_stubs.emg_apply_ok = false;
    CHECK(config_apply_start_json(golden));
    run_to_idle();
    CHECK(!config_apply_last_ok());
    CHECK_EQ_U(apply_stubs.sensor_launch, 1);
    CHECK_EQ_U(apply_stubs.trigger_apply, 1); /* later steps still run */
}

static void test_enables(void) {
    /* emg.enabled=false skips the ADS1299 apply; a disabled CRA slot is still
     * applied so its reports are turned off. */
    apply_stubs_reset();
    CHECK(config_apply_start_json("{\"platform\":{},\"emg\":{\"enabled\":false},\"imu_mta\":{},"
                                  "\"imu_cra\":{\"enabled\":false},\"trigger\":{}}"));
    run_to_idle();
    CHECK(config_apply_last_ok());
    CHECK_EQ_U(apply_stubs.emg_apply, 0);
    CHECK_EQ_U(apply_stubs.imu_apply_cra, 1);
    CHECK(!jtsmm_config_active()->emg.enabled);

    /* config init (defaults) goes through the same path. */
    apply_stubs_reset();
    CHECK(config_apply_start_defaults());
    run_to_idle();
    CHECK(config_apply_last_ok());
    CHECK_EQ_U(apply_stubs.sensor_stop, 1);
    CHECK_EQ_U(apply_stubs.emg_apply, 1);
    CHECK_EQ_U(jtsmm_config_active()->trigger.rate_hz, 30);
}

int main(int argc, char **argv) {
    char *golden = read_file((argc > 1) ? argv[1] : GOLDEN_MCU_APPLY_JSON);
    if (golden == NULL) {
        fprintf(stderr, "cannot read golden JSON\n");
        return 1;
    }
    one_line(golden);
    test_good_apply(golden);
    test_bad_apply_keeps_running_config(golden);
    test_hardware_failures(golden);
    test_enables();
    free(golden);
    CHECK_DONE("test_config_apply");
}
