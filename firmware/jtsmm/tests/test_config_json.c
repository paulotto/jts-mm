// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#include "jtsmm/records.h"
#include "jtsmm/runtime/config_json.h"
#include "jtsmm/runtime/ringbuf.h"

#include "check.h"
#include "config_compare.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GOLDEN_MCU_APPLY_JSON
#define GOLDEN_MCU_APPLY_JSON "../../../schema/v1/golden/mcu_apply_template.json"
#endif

#define FLASH_JSON_MAX (4096u - 16u) /* config_flash.c CFG_MAX_JSON */

static char *read_file(const char *path) {
    FILE *fh = fopen(path, "rb");
    if (fh == NULL) {
        return NULL;
    }
    if (fseek(fh, 0, SEEK_END) != 0) {
        fclose(fh);
        return NULL;
    }
    long size = ftell(fh);
    if (size < 0 || fseek(fh, 0, SEEK_SET) != 0) {
        fclose(fh);
        return NULL;
    }
    char *buf = (char *)malloc((size_t)size + 1u);
    if (buf == NULL) {
        fclose(fh);
        return NULL;
    }
    size_t got = fread(buf, 1u, (size_t)size, fh);
    fclose(fh);
    if (got != (size_t)size) {
        free(buf);
        return NULL;
    }
    buf[size] = '\0';
    return buf;
}

static int imu_report_rate(const imu_slot_config_t *slot, uint8_t report_id) {
    for (uint8_t i = 0; i < slot->num_reports; i++) {
        if (slot->reports[i].report_id == report_id) {
            return (int)slot->reports[i].rate_hz;
        }
    }
    return -1;
}

/* Replace the first occurrence of @p from in @p src (must exist) into @p dst. */
static bool patch(char *dst, size_t cap, const char *src, const char *from, const char *to) {
    const char *at = strstr(src, from);
    if (at == NULL) {
        fprintf(stderr, "patch: '%s' not found\n", from);
        return false;
    }
    size_t head = (size_t)(at - src);
    int n = snprintf(dst, cap, "%.*s%s%s", (int)head, src, to, at + strlen(from));
    return n > 0 && (size_t)n < cap;
}

static void expect_production_template(const jtsmm_mcu_config_t *cfg) {
    CHECK_EQ_U(cfg->schema_version, 1u);
    CHECK_EQ_U(cfg->platform.cpu_hz, 150000000u);
    CHECK_EQ_U(cfg->platform.spi0_stream_hz, 4000000u);
    CHECK_EQ_U(cfg->platform.i2c_hz, 400000u);
    CHECK_EQ_U(cfg->platform.pio_trigger_clock_hz, 1000000u);
    /* Byte budgets / each ring's slot size (ringbuf_slot_bytes): compact 48 B
     * EMG slots, 256 B IMU/event slots. All map to whole slots within the
     * backing stores (EMG 2048, IMU 64, events 64), so nothing is clamped. */
    CHECK_EQ_U(ringbuf_slot_bytes(RING_EMG), 48u);
    CHECK_EQ_U(ringbuf_slot_bytes(RING_IMU_MTA), 256u);
    CHECK_EQ_U(cfg->platform.emg_ring_slots, 2048u);  /* 98304 / 48 */
    CHECK_EQ_U(cfg->platform.imu_ring_slots, 64u);    /* 16384 / 256 */
    CHECK_EQ_U(cfg->platform.event_ring_slots, 32u);  /* 8192 / 256 */
    CHECK_EQ_U(cfg->platform.irq_emg_priority, 0u);
    CHECK_EQ_U(cfg->platform.irq_imu_priority, 1u);
    CHECK_EQ_U(cfg->platform.irq_trigger_priority, 2u);

    CHECK_EQ_U(cfg->emg.sample_rate_sps, 4000u);
    CHECK_EQ_U(cfg->emg.default_gain, 6u);
    CHECK_EQ_U(cfg->emg.stream_channels, 6u);
    CHECK(cfg->emg.bias.enabled);
    CHECK(cfg->emg.leadoff.enabled);
    CHECK(cfg->emg.leadoff.during_recording);
    CHECK(!cfg->emg.leadoff.ac_tone);
    CHECK_EQ_U(cfg->emg.leadoff.current_na, 6u);
    CHECK(cfg->emg.profile == EMG_PROFILE_CUSTOM);

    CHECK(cfg->imu_mta.enabled);
    CHECK_EQ_U(cfg->imu_mta.fusion_rate_hz, 390u);
    CHECK(cfg->imu_mta.sync_mode == IMU_SYNC_INT_DRIVEN);
    CHECK_EQ_U(cfg->imu_mta.num_reports, 2u);
    CHECK(imu_report_rate(&cfg->imu_mta, 0x08) == 390);
    CHECK(imu_report_rate(&cfg->imu_mta, 0x01) == 250); /* accelerometer, kinematics_accel */
    CHECK(cfg->imu_cra.enabled);
    CHECK_EQ_U(cfg->imu_cra.num_reports, 2u);

    /* Tracks mcu.trigger in software/jtsmm/etc/jtsmm.yaml via the golden file
     * (regenerate with scripts/sync_config_template.py --sync-mcu-golden).
     * 40 Hz / 12 ms exposure (Oct 2026 pilot); the firmware boot default
     * stays 30 Hz (config_defaults.c). */
    CHECK_EQ_U(cfg->trigger.rate_hz, 40u);
    CHECK_EQ_U(cfg->trigger.pulse_width_us, 12000u);
    CHECK_EQ_U(cfg->trigger.polarity, 0u);
    CHECK(cfg->trigger.outputs.csi1);
    CHECK(cfg->trigger.outputs.csi2);
    CHECK(cfg->trigger.outputs.scope);
    CHECK(!cfg->trigger.outputs.sync_led);
}

static imu_report_cfg_t rep(uint8_t id, uint16_t rate, uint8_t type, uint16_t len) {
    return (imu_report_cfg_t){.enabled = true, .report_id = id, .rate_hz = rate, .usb_type = type,
                              .payload_len = len};
}

/* A config with every JSON-backed field away from its default. */
static void make_non_default(jtsmm_mcu_config_t *c) {
    jtsmm_config_set_defaults(c);
    c->schema_version = 1;
    /* cpu_hz / xosc_hz / irq.emg / irq.imu / outputs / ch7_readback /
     * amplitude_mv only accept their one true value (see test_fixed_values). */
    c->platform.spi0_stream_hz = 6000000u;
    c->platform.spi0_cfg_hz = 250000u;
    c->platform.i2c_hz = 100000u;
    c->platform.emg_ring_slots = 128;
    c->platform.imu_ring_slots = 40;
    c->platform.event_ring_slots = 20;
    c->platform.irq_trigger_priority = 1;

    c->emg.enabled = false;
    c->emg.profile = EMG_PROFILE_CUSTOM;
    c->emg.sample_rate_sps = 1000;
    c->emg.default_gain = 24;
    c->emg.stream_channels = 8;
    c->emg.pd_channel_mask = 0x21;
    c->emg.bias.enabled = false;
    c->emg.bias.channel_mask = 0x0F;
    c->emg.leadoff.enabled = false;
    c->emg.leadoff.during_recording = true;
    c->emg.leadoff.ac_tone = true;
    c->emg.leadoff.probe_drl = true;
    c->emg.leadoff.current_na = 24;
    c->emg.test_signal.enabled = true;

    c->imu_mta.enabled = false;
    c->imu_mta.profile = IMU_PROFILE_CUSTOM;
    c->imu_mta.fusion_primary = IMU_FUSION_ROTATION_VECTOR;
    c->imu_mta.fusion_rate_hz = 200;
    c->imu_mta.sync_mode = IMU_SYNC_EMG_DRDY_DIVISOR;
    c->imu_mta.emg_drdy_divisor = 5;
    c->imu_mta.dynamic_calibration.accel = false;
    c->imu_mta.dynamic_calibration.mag = false;
    c->imu_mta.num_reports = 2;
    c->imu_mta.reports[0] = rep(0x05, 200, REC_IMU_MTA_QUAT, IMU_QUAT_PAYLOAD_LEN);
    c->imu_mta.reports[1] = rep(0x02, 123, REC_IMU_MTA_RAW, IMU_RAW_PAYLOAD_LEN);
    c->imu_cra = c->imu_mta;
    c->imu_cra.enabled = true;
    c->imu_cra.fusion_primary = IMU_FUSION_GYRO_INTEGRATED_RV;
    c->imu_cra.sync_mode = IMU_SYNC_INT_DRIVEN;
    c->imu_cra.emg_drdy_divisor = 0;
    c->imu_cra.num_reports = 1;
    c->imu_cra.reports[0] = rep(0x2A, 500, REC_IMU_MTA_QUAT, IMU_QUAT_PAYLOAD_LEN);

    c->trigger.enabled = false;
    c->trigger.profile = TRIGGER_PROFILE_CUSTOM;
    c->trigger.rate_hz = 45;
    c->trigger.pulse_width_us = 7000;
    c->trigger.polarity = 1;
}

static void test_golden(const char *golden_path) {
    char *json = read_file(golden_path);
    if (json == NULL) {
        fprintf(stderr, "cannot read golden JSON: %s\n", golden_path);
        CHECK(json != NULL);
        return;
    }
    jtsmm_mcu_config_t cfg;
    jtsmm_config_set_defaults(&cfg);
    CHECK(config_json_parse(json, &cfg));
    expect_production_template(&cfg);

    char dumped[JTSMM_CMD_LINE_MAX];
    CHECK(config_json_dump(&cfg, dumped, sizeof(dumped)));
    /* The dump reports what the rings really hold: whole slots of each ring. */
    CHECK(strstr(dumped, "\"buffers\":{\"emg_ring\":98304,\"imu_ring\":16384,\"event_ring\":8192}") != NULL);
    jtsmm_mcu_config_t round;
    jtsmm_config_set_defaults(&round);
    CHECK(config_json_parse(dumped, &round));
    CHECK_EQ_U(count_diffs(&cfg, &round), 0);
    free(json);
}

static void test_roundtrip_all_fields(void) {
    jtsmm_mcu_config_t a;
    make_non_default(&a);
    jtsmm_mcu_config_t def;
    jtsmm_config_set_defaults(&def);
    /* Guard the guard: the fixture really differs from the defaults. */
    diffs_quiet = true;
    CHECK(count_diffs(&a, &def) > 30);
    diffs_quiet = false;

    char dumped[JTSMM_CMD_LINE_MAX];
    CHECK(config_json_dump(&a, dumped, sizeof(dumped)));
    jtsmm_mcu_config_t b;
    jtsmm_config_set_defaults(&b);
    bool ok = config_json_parse(dumped, &b);
    CHECK(ok);
    if (!ok) {
        fprintf(stderr, "  parse error: %s\n  json: %s\n", config_json_last_error(), dumped);
    }
    CHECK_EQ_U(count_diffs(&a, &b), 0);

    /* A named profile comes back as CUSTOM with identical concrete values
     * (the dump always carries the resolved numbers), which applies the same. */
    a.emg.profile = EMG_PROFILE_BENCH;
    a.trigger.profile = TRIGGER_PROFILE_BENCH_SCOPE;
    CHECK(config_json_dump(&a, dumped, sizeof(dumped)));
    jtsmm_config_set_defaults(&b);
    CHECK(config_json_parse(dumped, &b));
    CHECK(b.emg.profile == EMG_PROFILE_CUSTOM);
    CHECK(b.trigger.profile == TRIGGER_PROFILE_CUSTOM);
    b.emg.profile = a.emg.profile;
    b.trigger.profile = a.trigger.profile;
    CHECK_EQ_U(count_diffs(&a, &b), 0);
}

/* The dump must fail, not silently truncate, when the buffer is too small. */
static void test_dump_truncation(void) {
    jtsmm_mcu_config_t a;
    make_non_default(&a);
    char full[JTSMM_CMD_LINE_MAX];
    CHECK(config_json_dump(&a, full, sizeof(full)));
    size_t len = strlen(full);
    char small[JTSMM_CMD_LINE_MAX];
    CHECK(!config_json_dump(&a, small, len));     /* no room for the NUL */
    CHECK(!config_json_dump(&a, small, len - 7u));
    CHECK(config_json_dump(&a, small, len + 1u)); /* exactly enough */
    CHECK(strcmp(small, full) == 0);

    /* Worst case (8 reports per slot, wide numbers) still fits a command line
     * and the flash record, so `config save` -> `config load` always works. */
    for (int s = 0; s < 2; s++) {
        imu_slot_config_t *slot = s ? &a.imu_cra : &a.imu_mta;
        static const uint8_t ids[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x08, 0x09, 0x2A};
        slot->num_reports = IMU_MAX_REPORTS;
        for (int i = 0; i < IMU_MAX_REPORTS; i++) {
            slot->reports[i] = rep(ids[i], 65535, 0, 0);
        }
        slot->fusion_rate_hz = 65535;
        slot->emg_drdy_divisor = 255;
    }
    a.platform.spi0_stream_hz = 4294967295u;
    a.platform.spi0_cfg_hz = 4294967295u;
    a.platform.i2c_hz = 4294967295u;
    CHECK(config_json_dump(&a, full, sizeof(full)));
    CHECK(strlen(full) < FLASH_JSON_MAX);
    CHECK(strlen(full) < JTSMM_CMD_LINE_MAX - 1u);
}

static bool parse_patched(const char *base, const char *from, const char *to, jtsmm_mcu_config_t *out) {
    static char buf[JTSMM_CMD_LINE_MAX];
    if (!patch(buf, sizeof(buf), base, from, to)) {
        check_failures++;
        return false;
    }
    jtsmm_config_set_defaults(out);
    return config_json_parse(buf, out);
}

static void test_strict_values(void) {
    jtsmm_mcu_config_t d;
    jtsmm_config_set_defaults(&d);
    d.trigger.rate_hz = 30; /* the patterns below spell out 30 */
    static char base[JTSMM_CMD_LINE_MAX];
    CHECK(config_json_dump(&d, base, sizeof(base)));
    jtsmm_mcu_config_t c;

    /* Baseline: the unpatched dump parses. */
    CHECK(parse_patched(base, "\"schema_version\":1", "\"schema_version\":1", &c));

    /* Out of range for the field it lands in: 65566 used to wrap to 30. */
    const char *rate = "\"rate_hz\":30,\"pulse\"";
    CHECK(!parse_patched(base, rate, "\"rate_hz\":65566,\"pulse\"", &c));
    CHECK(strstr(config_json_last_error(), "trigger.rate_hz") != NULL);
    CHECK(!parse_patched(base, rate, "\"rate_hz\":-1,\"pulse\"", &c));
    CHECK(!parse_patched(base, rate, "\"rate_hz\":30.5,\"pulse\"", &c));
    CHECK(!parse_patched(base, rate, "\"rate_hz\":\"30\",\"pulse\"", &c));
    CHECK(parse_patched(base, rate, "\"rate_hz\":65535,\"pulse\"", &c));
    CHECK_EQ_U(c.trigger.rate_hz, 65535u); /* parses; trigger validation rejects it */
    CHECK(!parse_patched(base, "\"default_gain\":6", "\"default_gain\":256", &c));
    CHECK(!parse_patched(base, "\"sample_rate_sps\":4000", "\"sample_rate_sps\":70000", &c));
    CHECK(!parse_patched(base, "\"emg\":0", "\"emg\":-3", &c));

    /* Booleans: true/false or exactly 0/1. */
    const char *en = "\"trigger\":{\"enabled\":true";
    CHECK(parse_patched(base, en, "\"trigger\":{\"enabled\":0", &c));
    CHECK(!c.trigger.enabled);
    CHECK(parse_patched(base, en, "\"trigger\":{\"enabled\":1", &c));
    CHECK(c.trigger.enabled);
    CHECK(!parse_patched(base, en, "\"trigger\":{\"enabled\":2", &c));
    CHECK(!parse_patched(base, en, "\"trigger\":{\"enabled\":\"true\"", &c));
    CHECK(!parse_patched(base, en, "\"trigger\":{\"enabled\":null", &c));

    /* Sections must be objects; all five are required. */
    const char *trig_start = strstr(base, "\"trigger\":{\"enabled\"");
    CHECK(trig_start != NULL);
    if (trig_start != NULL) {
        static char buf[JTSMM_CMD_LINE_MAX];
        snprintf(buf, sizeof(buf), "%.*s\"trigger\":null}", (int)(trig_start - base), base);
        jtsmm_config_set_defaults(&c);
        CHECK(!config_json_parse(buf, &c));
        CHECK(strstr(config_json_last_error(), "trigger") != NULL);
        snprintf(buf, sizeof(buf), "%.*s\"trigger\":[]}", (int)(trig_start - base), base);
        CHECK(!config_json_parse(buf, &c));
        snprintf(buf, sizeof(buf), "%.*s\"x\":{}}", (int)(trig_start - base), base);
        CHECK(!config_json_parse(buf, &c));
        CHECK(strstr(config_json_last_error(), "missing") != NULL);
    }
    CHECK(!parse_patched(base, "\"leadoff\":{", "\"leadoff\":7,\"x\":{", &c));
    CHECK(!parse_patched(base, "\"decimation_factor\":0}", "\"decimation_factor\":0},\"reports\":[]",
                         &c));
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse("[1,2]", &c));
    CHECK(!config_json_parse("{\"platform\":", &c));
    CHECK(strstr(config_json_last_error(), "malformed") != NULL);
}

static void test_missing_keys_and_polarity(void) {
    /* Empty sections: everything keeps its default. */
    const char *minimal = "{\"platform\":{},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},\"trigger\":{}}";
    jtsmm_mcu_config_t c, d;
    jtsmm_config_set_defaults(&c);
    jtsmm_config_set_defaults(&d);
    CHECK(config_json_parse(minimal, &c));
    CHECK_EQ_U(count_diffs(&c, &d), 0);

    /* A pulse object without polarity keeps the current polarity (it used to
     * force imx296_fast). */
    const char *no_pol = "{\"platform\":{},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},"
                         "\"trigger\":{\"pulse\":{\"width_us\":5000}}}";
    jtsmm_config_set_defaults(&c);
    c.trigger.polarity = 1;
    CHECK(config_json_parse(no_pol, &c));
    CHECK_EQ_U(c.trigger.polarity, 1u);
    CHECK_EQ_U(c.trigger.pulse_width_us, 5000u);

    const char *hi = "{\"platform\":{},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},"
                     "\"trigger\":{\"pulse\":{\"polarity\":\"active_high\"}}}";
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(hi, &c));
    CHECK_EQ_U(c.trigger.polarity, 1u);
    const char *bad = "{\"platform\":{},\"emg\":{},\"imu_mta\":{},\"imu_cra\":{},"
                      "\"trigger\":{\"pulse\":{\"polarity\":\"active_low\"}}}";
    CHECK(!config_json_parse(bad, &c));
    CHECK(strstr(config_json_last_error(), "polarity") != NULL);

    /* leadoff without during_recording keeps it (it used to force false). */
    const char *lo = "{\"platform\":{},\"emg\":{\"leadoff\":{\"enabled\":true}},\"imu_mta\":{},"
                     "\"imu_cra\":{},\"trigger\":{}}";
    jtsmm_config_set_defaults(&c);
    c.emg.leadoff.during_recording = true;
    CHECK(config_json_parse(lo, &c));
    CHECK(c.emg.leadoff.during_recording);
}

/* BNO085 dynamic calibration: on for every sensor unless the host turns it off, per
 * sensor; a missing section or key keeps the default (all on, every recording so far). */
static void test_dynamic_calibration(void) {
    jtsmm_mcu_config_t c;
    jtsmm_config_set_defaults(&c);
    CHECK(c.imu_mta.dynamic_calibration.accel && c.imu_mta.dynamic_calibration.gyro &&
          c.imu_mta.dynamic_calibration.mag);
    CHECK(c.imu_cra.dynamic_calibration.accel && c.imu_cra.dynamic_calibration.gyro &&
          c.imu_cra.dynamic_calibration.mag);
    const char *off = "{\"platform\":{},\"emg\":{},\"imu_mta\":{\"dynamic_calibration\":"
                      "{\"accel\":false,\"gyro\":false,\"mag\":false}},"
                      "\"imu_cra\":{\"dynamic_calibration\":{\"gyro\":false}},\"trigger\":{}}";
    CHECK(config_json_parse(off, &c));
    CHECK(!c.imu_mta.dynamic_calibration.accel && !c.imu_mta.dynamic_calibration.gyro &&
          !c.imu_mta.dynamic_calibration.mag);
    CHECK(c.imu_cra.dynamic_calibration.accel && !c.imu_cra.dynamic_calibration.gyro &&
          c.imu_cra.dynamic_calibration.mag);
    char dumped[JTSMM_CMD_LINE_MAX];
    CHECK(config_json_dump(&c, dumped, sizeof(dumped)));
    CHECK(strstr(dumped, "\"dynamic_calibration\":{\"accel\":false,\"gyro\":false,\"mag\":false}") != NULL);
    CHECK(strstr(dumped, "\"dynamic_calibration\":{\"accel\":true,\"gyro\":false,\"mag\":true}") != NULL);
    jtsmm_config_set_defaults(&c);
    const char *bad = "{\"platform\":{},\"emg\":{},\"imu_mta\":{\"dynamic_calibration\":"
                      "{\"gyro\":\"off\"}},\"imu_cra\":{},\"trigger\":{}}";
    CHECK(!config_json_parse(bad, &c));
    CHECK(strstr(config_json_last_error(), "gyro") != NULL);
}

#define WRAP(platform, emg, mta, trig)                                           \
    "{\"platform\":{" platform "},\"emg\":{" emg "},\"imu_mta\":{" mta               \
    "},\"imu_cra\":{},\"trigger\":{" trig "}}"

/* Keys that describe fixed facts accept only the true value; anything else is
 * refused loudly instead of being parsed and ignored. */
static void test_fixed_values(void) {
    jtsmm_mcu_config_t c;
    struct {
        const char *json;
        bool ok;
        const char *err;
    } cases[] = {
        {WRAP("\"system\":{\"cpu_mhz\":150,\"xosc_hz\":12000000}", "", "", ""), true, NULL},
        {WRAP("\"system\":{\"cpu_mhz\":133}", "", "", ""), false, "cpu_mhz"},
        {WRAP("\"system\":{\"xosc_hz\":16000000}", "", "", ""), false, "xosc_hz"},
        {WRAP("\"profile\":\"default\"", "", "", ""), true, NULL},
        {WRAP("\"profile\":\"high_throughput\"", "", "", ""), false, "profile"},
        {WRAP("\"irq\":{\"emg\":0,\"imu\":1,\"trigger\":3}", "", "", ""), true, NULL},
        {WRAP("\"irq\":{\"emg\":2}", "", "", ""), false, "irq.emg"},
        {WRAP("\"irq\":{\"imu\":0}", "", "", ""), false, "irq.imu"},
        {WRAP("\"irq\":{\"trigger\":4}", "", "", ""), false, "irq.trigger"},
        {WRAP("", "\"stream_channels\":8", "", ""), true, NULL},
        {WRAP("", "\"stream_channels\":7", "", ""), false, "stream_channels"},
        {WRAP("", "\"bias\":{\"ch7_readback\":true}", "", ""), false, "ch7_readback"},
        {WRAP("", "\"test_signal\":{\"enabled\":true,\"amplitude_mv\":0}", "", ""), true, NULL},
        {WRAP("", "\"test_signal\":{\"amplitude_mv\":2}", "", ""), false, "amplitude_mv"},
        {WRAP("", "", "", "\"outputs\":{\"csi1\":true,\"csi2\":true,\"scope\":true,\"sync_led\":false}"),
         true, NULL},
        {WRAP("", "", "", "\"outputs\":{\"csi2\":false}"), false, "outputs"},
        {WRAP("", "", "", "\"outputs\":{\"sync_led\":true}"), false, "sync_led"},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        jtsmm_config_set_defaults(&c);
        bool ok = config_json_parse(cases[i].json, &c);
        if (ok != cases[i].ok ||
            (!ok && strstr(config_json_last_error(), cases[i].err) == NULL)) {
            fprintf(stderr, "  fixed-value case %zu: ok=%d err='%s'\n", i, ok,
                    config_json_last_error());
            check_failures++;
        }
    }
}

/* An explicit report list is taken literally: nothing enabled is an error (it
 * used to re-expand to the profile's GRV + lin accel), and so are reports this
 * firmware does not implement or more than IMU_MAX_REPORTS. */
static void test_imu_report_lists(void) {
    jtsmm_mcu_config_t c;
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse(
        WRAP("", "", "\"reports\":{\"game_rotation_vector\":{\"enabled\":false}}", ""), &c));
    CHECK(strstr(config_json_last_error(), "none enabled") != NULL);
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse(WRAP("", "", "\"reports\":{}", ""), &c));
    /* ...but a disabled slot may have an empty list. */
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(WRAP("", "", "\"enabled\":false,\"reports\":{}", ""), &c));
    CHECK(!c.imu_mta.enabled);
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse(
        WRAP("", "", "\"reports\":{\"step_counter\":{\"rate_hz\":10}}", ""), &c));
    CHECK(strstr(config_json_last_error(), "not implemented") != NULL);
    /* Unknown but disabled: ignored. */
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(WRAP("", "",
                                 "\"reports\":{\"step_counter\":{\"enabled\":false},"
                                 "\"gyroscope\":{\"rate_hz\":100}}",
                                 ""),
                            &c));
    CHECK_EQ_U(c.imu_mta.num_reports, 1);
    CHECK(c.imu_mta.profile == IMU_PROFILE_CUSTOM);
    /* Nine enabled reports (the table only has 8 distinct ones, so repeat). */
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse(
        WRAP("", "",
             "\"reports\":{\"accelerometer\":{},\"gyroscope\":{},\"magnetometer\":{},"
             "\"linear_acceleration\":{},\"rotation_vector\":{},\"game_rotation_vector\":{},"
             "\"geomagnetic_rotation_vector\":{},\"gyro_integrated_rotation_vector\":{},"
             "\"accelerometer\":{}}",
             ""),
        &c));
    CHECK(strstr(config_json_last_error(), "more than") != NULL);

    /* Profile names: kinematics_accel is a firmware preset; a name without one
     * (datasheet_motion is host-only) used to become kinematics_default silently
     * and now needs its reports spelled out. */
    jtsmm_config_set_defaults(&c);
    c.imu_mta.profile = IMU_PROFILE_KINEMATICS_DEFAULT;
    CHECK(config_json_parse(WRAP("", "", "\"profile\":\"kinematics_accel\"", ""), &c));
    CHECK(c.imu_mta.profile == IMU_PROFILE_KINEMATICS_ACCEL);
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse(WRAP("", "", "\"profile\":\"datasheet_motion\"", ""), &c));
    CHECK(strstr(config_json_last_error(), "no firmware preset") != NULL);
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(WRAP("", "",
                                 "\"profile\":\"datasheet_motion\","
                                 "\"reports\":{\"accelerometer\":{\"rate_hz\":400}}",
                                 ""),
                            &c));
    CHECK(c.imu_mta.profile == IMU_PROFILE_CUSTOM);
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(WRAP("", "", "\"enabled\":false,\"profile\":\"nope\"", ""), &c));

    /* EMG-DRDY divisor must fit its uint8_t (16000/50 = 320 used to wrap to 64). */
    jtsmm_config_set_defaults(&c);
    CHECK(!config_json_parse(WRAP("", "\"sample_rate_sps\":16000",
                                  "\"sync\":{\"mode\":\"emg_drdy_divisor\"},"
                                  "\"reports\":{\"gyroscope\":{\"rate_hz\":50}}",
                                  ""),
                             &c));
    CHECK(strstr(config_json_last_error(), "divisor") != NULL);
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(WRAP("", "\"sample_rate_sps\":2000",
                                 "\"sync\":{\"mode\":\"emg_drdy_divisor\"},"
                                 "\"reports\":{\"gyroscope\":{\"rate_hz\":250}}",
                                 ""),
                            &c));
    CHECK_EQ_U(c.imu_mta.emg_drdy_divisor, 8u);
}

/* Host/schema send platform.pio.trigger.clock_hz; only 1 MHz is supported. */
static void test_pio_clock_key(void) {
    jtsmm_mcu_config_t c;
    const char *ok = "{\"platform\":{\"pio\":{\"trigger\":{\"clock_hz\":1000000}}},\"emg\":{},"
                     "\"imu_mta\":{},\"imu_cra\":{},\"trigger\":{}}";
    jtsmm_config_set_defaults(&c);
    CHECK(config_json_parse(ok, &c));
    CHECK_EQ_U(c.platform.pio_trigger_clock_hz, 1000000u);
    const char *two = "{\"platform\":{\"pio\":{\"trigger\":{\"clock_hz\":2000000}}},\"emg\":{},"
                      "\"imu_mta\":{},\"imu_cra\":{},\"trigger\":{}}";
    CHECK(!config_json_parse(two, &c));
    CHECK(strstr(config_json_last_error(), "clock_hz") != NULL);
    const char *legacy = "{\"platform\":{\"pio\":{\"trigger_clock_hz\":125000000}},\"emg\":{},"
                         "\"imu_mta\":{},\"imu_cra\":{},\"trigger\":{}}";
    CHECK(!config_json_parse(legacy, &c));
    const char *wrong_type = "{\"platform\":{\"pio\":{\"trigger\":5}},\"emg\":{},"
                             "\"imu_mta\":{},\"imu_cra\":{},\"trigger\":{}}";
    CHECK(!config_json_parse(wrong_type, &c));
}

int main(int argc, char **argv) {
    test_golden((argc > 1) ? argv[1] : GOLDEN_MCU_APPLY_JSON);
    test_roundtrip_all_fields();
    test_dump_truncation();
    test_strict_values();
    test_missing_keys_and_polarity();
    test_pio_clock_key();
    test_fixed_values();
    test_imu_report_lists();
    test_dynamic_calibration();
    CHECK_DONE("test_config_json");
}
