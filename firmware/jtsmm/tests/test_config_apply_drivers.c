// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Config plumbing, firmware half: every `config apply` value reaches the driver
 * call that applies it.
 *
 * The REAL cmd_parser / config_apply / config_json / emg_config / imu_config /
 * trigger_config / capture_session run against recording stubs of the driver
 * API (tests/stubs/rec_drivers.c). Each case mutates the golden host payload
 * (schema/v1/golden/mcu_apply_template.json), sends it as `config apply <json>`
 * over the CDC parser, and requires the listed driver calls -- which the golden
 * payload alone must not produce, so the mutation is what caused them. Trigger
 * rate/width/polarity only reach the PIO when the trigger starts, so those
 * cases also start a capture (capture_session_start, the panel button path);
 * the CDC stream commands (`ads_start`, `trig_start [rate [width]]`,
 * `imu_stream <slot> on`) are checked against the applied config separately.
 * The register level below this is test_ads1299_registers.c.
 */
#include "jtsmm/config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/runtime/capture_session.h"
#include "jtsmm/runtime/config_apply.h"
#include "jtsmm/runtime/config_json.h"

#include "cJSON.h"
#include "check.h"
#include "rec_drivers.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef GOLDEN_MCU_APPLY_JSON
#define GOLDEN_MCU_APPLY_JSON "../../../schema/v1/golden/mcu_apply_template.json"
#endif

static char *golden;

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

/* One mutation of the golden payload: dotted path = JSON text (NULL = delete). */
typedef struct {
    const char *path;
    const char *json;
} mut_t;

#define MAX_MUTS 6
#define MAX_CALLS 4

/* Build the one-line apply JSON: golden + mutations. Caller frees. */
static char *payload(const mut_t *muts) {
    cJSON *root = cJSON_Parse(golden);
    CHECK(root != NULL);
    for (int m = 0; m < MAX_MUTS && muts != NULL && muts[m].path != NULL; m++) {
        char buf[128];
        snprintf(buf, sizeof(buf), "%s", muts[m].path);
        cJSON *parent = root;
        char *key = buf;
        for (char *dot = strchr(key, '.'); dot != NULL; dot = strchr(key, '.')) {
            *dot = '\0';
            cJSON *child = cJSON_GetObjectItemCaseSensitive(parent, key);
            if (child == NULL) {
                child = cJSON_AddObjectToObject(parent, key);
            }
            parent = child;
            key = dot + 1;
        }
        cJSON_DeleteItemFromObjectCaseSensitive(parent, key);
        if (muts[m].json != NULL) {
            cJSON *value = cJSON_Parse(muts[m].json);
            CHECK(value != NULL);
            cJSON_AddItemToObject(parent, key, value);
        }
    }
    char *out = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    return out;
}

/* The host's path: `config apply <json>` typed into the real CDC parser. */
static bool apply(const mut_t *muts) {
    char *json = payload(muts);
    static char line[JTSMM_CMD_LINE_MAX];
    snprintf(line, sizeof(line), "config apply %s", json);
    free(json);
    CHECK(strlen(line) + 1 < sizeof(line));
    rec_cdc_send(line);
    CHECK(config_apply_busy());
    for (int i = 0; i < 100 && config_apply_busy(); i++) {
        config_apply_poll();
    }
    CHECK(!config_apply_busy());
    return config_apply_last_ok();
}

/* Golden applied once more so every case starts from the same device state
 * (imu_config remembers which reports the previous apply enabled). */
static void settle_on_golden(void) {
    capture_session_stop();
    CHECK(apply(NULL));
    rec_reset();
}

static int golden_log_has(const char *call) {
    static char golden_log[REC_MAX][REC_LINE];
    static int golden_n = -1;
    if (golden_n < 0) {
        settle_on_golden();
        CHECK(apply(NULL));
        capture_session_start();
        golden_n = rec_count();
        for (int i = 0; i < golden_n; i++) {
            snprintf(golden_log[i], REC_LINE, "%s", rec_at(i));
        }
        capture_session_stop();
    }
    for (int i = 0; i < golden_n; i++) {
        if (strcmp(golden_log[i], call) == 0) {
            return 1;
        }
    }
    return 0;
}

typedef struct {
    const char *name;
    mut_t muts[MAX_MUTS];
    const char *expect[MAX_CALLS]; /* driver calls that must appear */
    const char *absent[MAX_CALLS]; /* call prefixes that must not appear */
    bool start;                    /* also start a capture (trigger/EMG/IMU start) */
    bool rejected;                 /* single-value key: apply must FAIL, no driver touched */
} case_t;

static const case_t CASES[] = {
    /* EMG -> ads1299_set_* (emg_config_apply) */
    {"emg.bias.enabled=false", {{"emg.bias.enabled", "false"}}, {"ads1299_set_bias_drive(0)"}},
    {"emg.bias.channels=0x03 (--bias-channels 1,2)", {{"emg.bias.channels", "3"}},
     {"ads1299_set_bias_channels(0x03)"}},
    {"emg.pd_channels=0x37 + bias 0x08 (--active-channels 4)",
     {{"emg.pd_channels", "55"}, {"emg.bias.channels", "8"}},
     {"ads1299_set_pd_channels(0x37)", "ads1299_set_bias_channels(0x08)"}},
    {"powered-down channels never drive the DRL", {{"emg.pd_channels", "55"}},
     {"ads1299_set_pd_channels(0x37)", "ads1299_set_bias_channels(0x08)"}},
    {"DRL probe: CH1-6 down, bias off, probe_drl",
     {{"emg.pd_channels", "63"}, {"emg.bias", "{\"enabled\":false}"}, {"emg.leadoff.probe_drl", "true"}},
     {"ads1299_set_pd_channels(0x3F)", "ads1299_set_bias_drive(0)", "ads1299_set_drl_probe(1)",
      "ads1299_set_bias_channels(0x00)"}},
    {"emg.default_gain=24", {{"emg.default_gain", "24"}}, {"ads1299_set_pga_gain(24)"}},
    {"emg.sample_rate_sps=1000", {{"emg.sample_rate_sps", "1000"}}, {"ads1299_set_sample_rate_hz(1000)"}},
    {"emg.sample_rate_sps=250", {{"emg.sample_rate_sps", "250"}}, {"ads1299_set_sample_rate_hz(250)"}},
    {"emg.leadoff.enabled=false",
     {{"emg.leadoff.enabled", "false"}, {"emg.leadoff.during_recording", "false"}},
     {"ads1299_set_leadoff_monitor(0)"}},
    {"emg.leadoff.during_recording=false", {{"emg.leadoff.during_recording", "false"}},
     {"ads1299_set_leadoff_during_recording(0)"}},
    {"emg.leadoff.current_na=24", {{"emg.leadoff.current_na", "24"}},
     {"ads1299_set_leadoff_current_na(24)"}},
    {"emg.leadoff.ac_tone=true",
     {{"emg.leadoff.ac_tone", "true"}, {"emg.leadoff.during_recording", "false"}},
     {"ads1299_set_leadoff_ac_tone(1)"}},
    {"emg.leadoff.probe_drl=true", {{"emg.leadoff.probe_drl", "true"}}, {"ads1299_set_drl_probe(1)"}},
    {"emg.test_signal.enabled=true", {{"emg.test_signal.enabled", "true"}}, {"ads1299_test_signal(1)"}},
    {"emg.profile=noise_characterization applies (2000 SPS, no lead-off current while streaming)",
     {{"emg", "{\"profile\":\"noise_characterization\"}"}},
     {"ads1299_set_leadoff_during_recording(0)"}, {"ads1299_set_sample_rate_hz(16000)"}},
    {"emg.enabled=false: ADS1299 left alone, no EMG start", {{"emg.enabled", "false"}},
     {NULL}, {"ads1299_init(", "ads1299_set_", "ads1299_start_stream("}, true},
    /* trigger -> trigger_pio_* (config apply sets clock/IRQ; rate/width/polarity at start) */
    {"trigger.rate_hz=60", {{"trigger.rate_hz", "60"}}, {"trigger_pio_start(60,12000,1)"}, {NULL}, true},
    {"trigger.pulse.width_us=5000", {{"trigger.pulse.width_us", "5000"}}, {"trigger_pio_start(40,5000,1)"},
     {NULL}, true},
    {"trigger.pulse.polarity=active_high (idles LOW from the apply on)",
     {{"trigger.pulse.polarity", "\"active_high\""}},
     {"trigger_pio_set_idle_polarity(0)", "trigger_pio_start(40,12000,0)"}, {NULL}, true},
    {"trigger.enabled=false: no trigger start", {{"trigger.enabled", "false"}}, {NULL},
     {"trigger_pio_start("}, true},
    {"platform.pio.trigger.clock_hz=2000000 is rejected (PIO program assumes 1 MHz)",
     {{"platform.pio.trigger.clock_hz", "2000000"}}, {NULL}, {"trigger_pio_", "ads1299_", "bno085_"},
     false, true},
    /* platform -> platform_config_apply / IMU bus */
    {"platform.spi0.hz=2000000", {{"platform.spi0.hz", "2000000"}},
     {"platform_config_apply(spi0=2000000 spi0_cfg=500000 i2c=400000 pio=1000000)"}},
    {"platform.spi0.cfg_hz=1000000", {{"platform.spi0.cfg_hz", "1000000"}},
     {"platform_config_apply(spi0=4000000 spi0_cfg=1000000 i2c=400000 pio=1000000)"}},
    {"platform.i2c.hz=100000", {{"platform.i2c.hz", "100000"}},
     {"platform_config_apply(spi0=4000000 spi0_cfg=500000 i2c=100000 pio=1000000)",
      "bno085_bus_config(mta,100000)", "bno085_bus_config(cra,100000)"}},
    /* IMU -> bno085 Set Feature (interval = 1e6 / rate) */
    {"imu_mta + gyroscope @100 Hz",
     {{"imu_mta.reports.gyroscope", "{\"enabled\":true,\"rate_hz\":100}"}},
     {"bno085_enable_report(mta,0x02,10000)"}, {"bno085_enable_report(cra,0x02,"}},
    {"imu_mta game_rotation_vector @200 Hz", {{"imu_mta.reports.game_rotation_vector.rate_hz", "200"}},
     {"bno085_enable_report(mta,0x08,5000)"}, {"bno085_enable_report(mta,0x08,2564)"}},
    {"imu_cra report set = linear_acceleration only",
     {{"imu_cra.reports", "{\"linear_acceleration\":{\"enabled\":true,\"rate_hz\":125}}"}},
     {"bno085_enable_report(cra,0x04,8000)", "bno085_enable_report(cra,0x08,0)",
      "bno085_enable_report(cra,0x01,0)"},
     {"bno085_enable_report(cra,0x08,2564)"}},
    /* IMU -> SH-2 ME Calibration flags (default all on, in the golden payload too) */
    {"imu_mta.dynamic_calibration all off",
     {{"imu_mta.dynamic_calibration", "{\"accel\":false,\"gyro\":false,\"mag\":false}"}},
     {"bno085_set_calibration(mta,0,0,0)"}, {"bno085_set_calibration(mta,1"}},
    {"imu_cra.dynamic_calibration.gyro=false", {{"imu_cra.dynamic_calibration.gyro", "false"}},
     {"bno085_set_calibration(cra,1,0,1)"}, {"bno085_set_calibration(cra,1,1,"}},
    {"imu_cra.enabled=false: reports turned off", {{"imu_cra.enabled", "false"}},
     {"bno085_enable_report(cra,0x08,0)", "bno085_enable_report(cra,0x01,0)"},
     {"bno085_enable_report(cra,0x08,2564)", "bno085_bus_config(cra,"}, true},
};

static bool log_has_prefix_from(const char *prefix, int from) {
    for (int i = from; i < rec_count(); i++) {
        if (strncmp(rec_at(i), prefix, strlen(prefix)) == 0) {
            return true;
        }
    }
    return false;
}

static void run_case(const case_t *c) {
    settle_on_golden();
    bool ok = apply(c->muts);
    bool failed = false;
    if (c->rejected ? (ok || rec_find("host_link_send_ack(0)", 0) < 0)
                    : (!ok || rec_find("host_link_send_ack(1)", 0) < 0)) {
        fprintf(stderr, "%s: config apply %s\n", c->name, c->rejected ? "was not rejected" : "failed");
        failed = true;
    }
    if (c->start) {
        capture_session_start();
    }
    for (int k = 0; k < MAX_CALLS && c->expect[k] != NULL; k++) {
        if (rec_find(c->expect[k], 0) < 0) {
            fprintf(stderr, "%s: missing driver call %s\n", c->name, c->expect[k]);
            failed = true;
        } else if (golden_log_has(c->expect[k])) {
            fprintf(stderr, "%s: %s also happens for the golden payload\n", c->name, c->expect[k]);
            failed = true;
        }
    }
    for (int k = 0; k < MAX_CALLS && c->absent[k] != NULL; k++) {
        if (log_has_prefix_from(c->absent[k], 0)) {
            fprintf(stderr, "%s: unexpected driver call %s...\n", c->name, c->absent[k]);
            failed = true;
        }
    }
    if (failed) {
        rec_dump();
        check_failures++;
    }
    capture_session_stop();
}

/* Disabled parts refuse to start (capture_session_start, like ads_start /
 * trig_start / `imu_stream on`). */
static void test_disabled_parts_do_not_start(void) {
    static const mut_t emg_off[] = {{"emg.enabled", "false"}, {NULL, NULL}};
    static const mut_t cra_off[] = {{"imu_cra.enabled", "false"}, {NULL, NULL}};
    static const mut_t trig_off[] = {{"trigger.enabled", "false"}, {NULL, NULL}};

    settle_on_golden();
    CHECK(apply(emg_off));
    capture_session_start();
    CHECK(!jtsmm_config_active()->stream_emg);
    CHECK(jtsmm_config_active()->stream_imu_mta);
    capture_session_stop();

    settle_on_golden();
    CHECK(apply(cra_off));
    CHECK(!imu_stream_start_slot(IMU_SLOT_CRA));
    CHECK(imu_stream_start_slot(IMU_SLOT_MTA));
    imu_stream_stop_slot(IMU_SLOT_MTA);
    capture_session_start();
    CHECK(!jtsmm_config_active()->stream_imu_cra);
    CHECK(jtsmm_config_active()->stream_emg);
    capture_session_stop();

    settle_on_golden();
    CHECK(apply(trig_off));
    capture_session_start();
    CHECK(rec_count_prefix("trigger_pio_start(") == 0);
    CHECK(jtsmm_config_active()->stream_emg);
    capture_session_stop();
}

/* The host's stream commands over CDC honour the applied config. */
static void test_cdc_stream_commands(void) {
    static const mut_t emg_off[] = {{"emg.enabled", "false"}, {NULL, NULL}};
    static const mut_t trig_off[] = {{"trigger.enabled", "false"}, {NULL, NULL}};
    static const mut_t cra_off[] = {{"imu_cra.enabled", "false"}, {NULL, NULL}};
    static const mut_t trig[] = {{"trigger.rate_hz", "60"}, {"trigger.pulse.width_us", "4000"},
                                 {"trigger.pulse.polarity", "\"active_high\""}, {NULL, NULL}};

    settle_on_golden();
    rec_cdc_send("ads_start");
    CHECK(rec_find("ads1299_start_stream()", 0) >= 0);
    CHECK(rec_find("host_link_send_ack(1)", 0) >= 0);
    rec_cdc_send("ads_stop");

    settle_on_golden();
    CHECK(apply(emg_off));
    rec_reset();
    rec_cdc_send("ads_start");
    CHECK(rec_find("host_link_send_ack(0)", 0) >= 0);
    CHECK(rec_count_prefix("ads1299_start_stream(") == 0);
    CHECK(!jtsmm_config_active()->stream_emg);

    settle_on_golden();
    CHECK(apply(trig));
    rec_reset();
    rec_cdc_send("trig_start");
    CHECK(rec_find("trigger_pio_start(60,4000,0)", 0) >= 0);
    rec_cdc_send("trig_stop");
    rec_reset();
    rec_cdc_send("trig_start 50"); /* a rate override keeps the configured width */
    CHECK(rec_find("trigger_pio_start(50,4000,0)", 0) >= 0);
    rec_cdc_send("trig_stop");
    rec_reset();
    rec_cdc_send("trig_start 100 2000");
    CHECK(rec_find("trigger_pio_start(100,2000,0)", 0) >= 0);
    rec_cdc_send("trig_stop");
    rec_reset();
    rec_cdc_send("trig_start 250"); /* 4000 us is not below the 4000 us period: refused */
    CHECK(rec_count_prefix("trigger_pio_start(") == 0);
    CHECK(rec_find("host_link_send_ack(0)", 0) >= 0);

    settle_on_golden();
    CHECK(apply(trig_off));
    rec_reset();
    rec_cdc_send("trig_start");
    rec_cdc_send("trig_start 60");
    CHECK(rec_count_prefix("trigger_pio_start(") == 0);
    CHECK_EQ_U(rec_count_prefix("host_link_send_ack(0)"), 2);

    settle_on_golden();
    CHECK(apply(cra_off));
    rec_reset();
    rec_cdc_send("imu_stream cra on");
    CHECK(rec_find("host_link_send_ack(0)", 0) >= 0);
    CHECK(!jtsmm_config_active()->stream_imu_cra);
    rec_reset();
    rec_cdc_send("imu_stream mta on");
    CHECK(rec_find("host_link_send_ack(1)", 0) >= 0);
    CHECK(jtsmm_config_active()->stream_imu_mta);
    rec_cdc_send("imu_stream mta off");
}

/* Run a rejected apply with stdout captured; true if the CDC text has @p want. */
static bool rejected_apply_prints(const mut_t *muts, const char *want) {
    static char out[8192];
    fflush(stdout);
    FILE *tmp = tmpfile();
    CHECK(tmp != NULL);
    if (tmp == NULL) {
        return false;
    }
    int saved = dup(fileno(stdout));
    dup2(fileno(tmp), fileno(stdout));
    bool ok = apply(muts);
    fflush(stdout);
    dup2(saved, fileno(stdout));
    close(saved);
    rewind(tmp);
    size_t n = fread(out, 1u, sizeof(out) - 1u, tmp);
    out[n] = '\0';
    fclose(tmp);
    CHECK(!ok);
    if (strstr(out, want) == NULL) {
        fprintf(stderr, "expected '%s' in:\n%s\n", want, out);
        return false;
    }
    return true;
}

/* A rejected EMG block names the actual reason, not always the sample rate. */
static void test_emg_rejection_reason(void) {
    static const mut_t bad_gain[] = {{"emg.default_gain", "7"}, {NULL, NULL}};
    static const mut_t bad_current[] = {{"emg.leadoff.current_na", "12"}, {NULL, NULL}};
    static const mut_t bad_fs[] = {{"emg.sample_rate_sps", "777"}, {NULL, NULL}};
    settle_on_golden();
    CHECK(rejected_apply_prints(bad_gain, "config apply FAIL (emg: default_gain must be"));
    CHECK(rejected_apply_prints(bad_current, "config apply FAIL (emg: leadoff.current_na must be"));
    CHECK(rejected_apply_prints(bad_fs, "config apply FAIL (emg: sample_rate_sps must be"));
}

/* Factory defaults (`config init`, boot without a host apply) keep DC lead-off
 * on while streaming, like the host template; otherwise the rig had no lead-off
 * indication at all until a host apply. */
static void test_factory_defaults_enable_leadoff(void) {
    settle_on_golden();
    rec_cdc_send("config init");
    CHECK(config_apply_busy());
    for (int i = 0; i < 100 && config_apply_busy(); i++) {
        config_apply_poll();
    }
    CHECK(config_apply_last_ok());
    CHECK(jtsmm_config_active()->emg.leadoff.enabled);
    CHECK(jtsmm_config_active()->emg.leadoff.during_recording);
    CHECK(!jtsmm_config_active()->emg.leadoff.ac_tone);
    CHECK(rec_find("ads1299_set_leadoff_monitor(1)", 0) >= 0);
    CHECK(rec_find("ads1299_set_leadoff_during_recording(1)", 0) >= 0);
}

/* Stream / bus commands typed while an apply is mid-flight are refused with
 * `<cmd> BUSY` + NAK instead of starting something a later apply step undoes
 * (CFG_EMG's ads1299_init halts the stream, CFG_TRIGGER re-inits the PIO). */
static void test_stream_commands_refused_while_applying(void) {
    static const char *const cmds[] = {"ads_start", "trig_start", "trig_start 60", "imu_stream mta on",
                                       "imu_stream cra on", "imu_savecal", "selftest"};
    const unsigned n_cmds = (unsigned)(sizeof(cmds) / sizeof(cmds[0]));
    settle_on_golden();
    char *json = payload(NULL);
    static char line[JTSMM_CMD_LINE_MAX];
    snprintf(line, sizeof(line), "config apply %s", json);
    free(json);
    rec_cdc_send(line);
    for (int step = 0; step < 3; step++) { /* before VALIDATE, then mid-apply */
        CHECK(config_apply_busy());
        rec_reset();
        for (unsigned i = 0; i < n_cmds; i++) {
            rec_cdc_send(cmds[i]);
        }
        CHECK_EQ_U(rec_count_prefix("host_link_send_ack(0)"), n_cmds);
        CHECK_EQ_U(rec_count_prefix("host_link_send_ack(1)"), 0);
        CHECK_EQ_U(rec_count_prefix("ads1299_start_stream("), 0);
        CHECK_EQ_U(rec_count_prefix("trigger_pio_start("), 0);
        CHECK(!jtsmm_config_active()->stream_emg);
        CHECK(!jtsmm_config_active()->stream_imu_mta);
        CHECK(!jtsmm_config_active()->stream_imu_cra);
        /* Stop commands stay allowed (the host's K6 path sends them). */
        rec_reset();
        rec_cdc_send("imu_stream mta off");
        rec_cdc_send("ads_stop");
        rec_cdc_send("trig_stop");
        CHECK_EQ_U(rec_count_prefix("host_link_send_ack(1)"), 3);
        config_apply_poll();
    }
    for (int i = 0; i < 100 && config_apply_busy(); i++) {
        config_apply_poll();
    }
    CHECK(!config_apply_busy());
    CHECK(config_apply_last_ok());
    /* Once the apply is done the same commands work again. */
    rec_reset();
    rec_cdc_send("ads_start");
    rec_cdc_send("trig_start");
    rec_cdc_send("imu_stream mta on");
    CHECK_EQ_U(rec_count_prefix("host_link_send_ack(1)"), 3);
    CHECK(rec_find("ads1299_start_stream()", 0) >= 0);
    CHECK(rec_count_prefix("trigger_pio_start(") == 1);
    CHECK(jtsmm_config_active()->stream_imu_mta);
    rec_cdc_send("imu_stream mta off");
    rec_cdc_send("trig_stop");
    rec_cdc_send("ads_stop");
}

/* imu_int_watch: duration argument, refused while an IMU slot streams (its INT
 * is live and the busy-wait starves the ring drain) or an apply runs. */
static void test_imu_int_watch_command(void) {
    settle_on_golden();
    rec_imu_int_watch_calls = 0;
    rec_cdc_send("imu_int_watch");
    CHECK_EQ_U(rec_imu_int_watch_calls, 1);
    CHECK_EQ_U(rec_imu_int_watch_ms, 1000);
    rec_cdc_send("imu_int_watch 250");
    CHECK_EQ_U(rec_imu_int_watch_ms, 250);
    rec_cdc_send("imu_int_watch 99999");
    CHECK_EQ_U(rec_imu_int_watch_ms, 10000);
    CHECK_EQ_U(rec_imu_int_watch_calls, 3);

    rec_cdc_send("imu_stream mta on");
    CHECK(jtsmm_config_active()->stream_imu_mta);
    rec_reset();
    rec_cdc_send("imu_int_watch");
    CHECK_EQ_U(rec_imu_int_watch_calls, 3);
    CHECK(rec_find("host_link_send_ack(0)", 0) >= 0);
    rec_cdc_send("imu_stream mta off");

    char *json = payload(NULL);
    static char line[JTSMM_CMD_LINE_MAX];
    snprintf(line, sizeof(line), "config apply %s", json);
    free(json);
    rec_cdc_send(line);
    CHECK(config_apply_busy());
    rec_reset();
    rec_cdc_send("imu_int_watch");
    CHECK_EQ_U(rec_imu_int_watch_calls, 3);
    CHECK(rec_find("host_link_send_ack(0)", 0) >= 0);
    for (int i = 0; i < 100 && config_apply_busy(); i++) {
        config_apply_poll();
    }
    CHECK(!config_apply_busy());
}

/* Two applies in a row: the second must undo what only the first asked for. */
static void test_two_apply_sequences(void) {
    /* EMG check DRL probe, then the recording config restored. */
    static const mut_t probe[] = {
        {"emg.pd_channels", "63"}, {"emg.bias", "{\"enabled\":false}"}, {"emg.leadoff.probe_drl", "true"},
        {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(probe));
    rec_reset();
    CHECK(apply(NULL));
    CHECK(rec_find("ads1299_set_pd_channels(0x00)", 0) >= 0);
    CHECK(rec_find("ads1299_set_bias_drive(1)", 0) >= 0);
    CHECK(rec_find("ads1299_set_bias_channels(0x3F)", 0) >= 0);
    CHECK(rec_find("ads1299_set_drl_probe(0)", 0) >= 0);

    /* --impedance-style AC tone + 24 nA, then back to DC 6 nA. */
    static const mut_t tone[] = {
        {"emg.leadoff.ac_tone", "true"}, {"emg.leadoff.during_recording", "false"},
        {"emg.leadoff.current_na", "24"}, {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(tone));
    rec_reset();
    CHECK(apply(NULL));
    CHECK(rec_find("ads1299_set_leadoff_ac_tone(0)", 0) >= 0);
    CHECK(rec_find("ads1299_set_leadoff_current_na(6)", 0) >= 0);
    CHECK(rec_find("ads1299_set_leadoff_during_recording(1)", 0) >= 0);

    /* An extra IMU report is turned off again by the next apply. */
    static const mut_t gyro[] = {{"imu_mta.reports.gyroscope", "{\"enabled\":true,\"rate_hz\":100}"},
                                 {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(gyro));
    rec_reset();
    CHECK(apply(NULL));
    CHECK(rec_find("bno085_enable_report(mta,0x02,0)", 0) >= 0);
    CHECK(rec_count_prefix("bno085_enable_report(mta,0x02,10000)") == 0);

    /* A disabled slot comes back with its reports and can stream again. */
    static const mut_t cra_off[] = {{"imu_cra.enabled", "false"}, {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(cra_off));
    CHECK(!imu_stream_start_slot(IMU_SLOT_CRA));
    rec_reset();
    CHECK(apply(NULL));
    CHECK(rec_find("bno085_enable_report(cra,0x08,2564)", 0) >= 0);
    CHECK(rec_find("bno085_enable_report(cra,0x01,4000)", 0) >= 0);
    CHECK(imu_stream_start_slot(IMU_SLOT_CRA));
    imu_stream_stop_slot(IMU_SLOT_CRA);

    /* emg.enabled=false, then true again: the ADS1299 is configured again. */
    static const mut_t emg_off[] = {{"emg.enabled", "false"}, {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(emg_off));
    rec_reset();
    CHECK(apply(NULL));
    CHECK(rec_find("ads1299_init()", 0) >= 0);
    capture_session_start();
    CHECK(jtsmm_config_active()->stream_emg);
    CHECK(rec_find("ads1299_start_stream()", 0) >= 0);
    capture_session_stop();

    /* Trigger rate changed and changed back. */
    static const mut_t trig60[] = {{"trigger.rate_hz", "60"}, {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(trig60));
    capture_session_start();
    capture_session_stop();
    rec_reset();
    CHECK(apply(NULL));
    capture_session_start();
    CHECK(rec_find("trigger_pio_start(40,12000,1)", 0) >= 0); /* golden = 40 Hz */
    capture_session_stop();

    /* A rejected apply touches no driver and keeps the running config. */
    static const mut_t bad_fs[] = {{"emg.default_gain", "24"}, {NULL, NULL}};
    static const mut_t bad[] = {{"emg.sample_rate_sps", "777"}, {NULL, NULL}};
    settle_on_golden();
    CHECK(apply(bad_fs));
    rec_reset();
    CHECK(!apply(bad));
    CHECK(rec_find("host_link_send_ack(0)", 0) >= 0);
    CHECK(rec_count_prefix("ads1299_") == 0);
    CHECK(rec_count_prefix("bno085_") == 0);
    CHECK(rec_count_prefix("platform_config_apply") == 0);
    CHECK_EQ_U(jtsmm_config_active()->emg.default_gain, 24);
}

int main(void) {
    golden = read_file(GOLDEN_MCU_APPLY_JSON);
    CHECK(golden != NULL);
    if (golden == NULL) {
        CHECK_DONE("test_config_apply_drivers");
    }
    jtsmm_config_set_defaults(jtsmm_config_mutable());
    for (size_t i = 0; i < sizeof(CASES) / sizeof(CASES[0]); i++) {
        run_case(&CASES[i]);
    }
    test_disabled_parts_do_not_start();
    test_cdc_stream_commands();
    test_stream_commands_refused_while_applying();
    test_factory_defaults_enable_leadoff();
    test_emg_rejection_reason();
    test_two_apply_sequences();
    test_imu_int_watch_command();
    free(golden);
    CHECK_DONE("test_config_apply_drivers");
}
