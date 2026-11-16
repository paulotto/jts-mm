// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Config plumbing, register level: the EMG part of a `config apply` payload ends
 * up in the right ADS1299 registers.
 *
 * The REAL config_json.c parses the golden host payload plus one mutation, the
 * REAL emg_config_apply() drives the REAL src/drivers/ads1299.c, and the driver
 * talks to a fake SPI0 with an ADS1299 register model behind it
 * (tests/ads_stubs/fake_ads1299.c: RESET/SDATAC/RDATAC/RREG/WREG framed by the
 * ADS CS pin). The expected register values are written out literally from the
 * datasheet (SBAS499C) and the board contract K8, not taken from ads1299_regs.h,
 * so a wrong helper there fails here too. E.g. `jtsmm record --bias-channels
 * 1,2` sends emg.bias.channels = 0x03 (tests/test_cli_plumbing.py) and must
 * give BIAS_SENSP = BIAS_SENSN = 0x03.
 */
#include "jtsmm/config.h"
#include "jtsmm/domain/emg_config.h"
#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/runtime/config_json.h"

#include "cJSON.h"
#include "check.h"
#include "fake_ads1299.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GOLDEN_MCU_APPLY_JSON
#define GOLDEN_MCU_APPLY_JSON "../../../schema/v1/golden/mcu_apply_template.json"
#endif

enum {
    CONFIG1 = 0x01, CONFIG2 = 0x02, CONFIG3 = 0x03, LOFF = 0x04, CH1SET = 0x05,
    BIAS_SENSP = 0x0D, BIAS_SENSN = 0x0E, LOFF_SENSP = 0x0F, LOFF_SENSN = 0x10, CONFIG4 = 0x17,
};

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

typedef struct {
    const char *path; /* under "emg." */
    const char *json;
} mut_t;

/* Golden payload + mutations -> parsed config -> emg_config_apply(). */
static bool apply_emg(const mut_t *muts, int n) {
    cJSON *root = cJSON_Parse(golden);
    cJSON *emg = cJSON_GetObjectItemCaseSensitive(root, "emg");
    CHECK(emg != NULL);
    for (int m = 0; m < n; m++) {
        char buf[64];
        snprintf(buf, sizeof(buf), "%s", muts[m].path);
        cJSON *parent = emg;
        char *key = buf;
        char *dot = strchr(key, '.');
        if (dot != NULL) {
            *dot = '\0';
            parent = cJSON_GetObjectItemCaseSensitive(emg, key);
            if (parent == NULL) {
                parent = cJSON_AddObjectToObject(emg, key);
            }
            key = dot + 1;
        }
        cJSON_DeleteItemFromObjectCaseSensitive(parent, key);
        cJSON_AddItemToObject(parent, key, cJSON_Parse(muts[m].json));
    }
    char *json = cJSON_PrintUnformatted(root);
    cJSON_Delete(root);
    jtsmm_mcu_config_t *cfg = jtsmm_config_mutable();
    jtsmm_config_set_defaults(cfg);
    bool parsed = config_json_parse(json, cfg);
    free(json);
    CHECK(parsed);
    int dropped_before = fake_ads_dropped_writes;
    bool ok = parsed && emg_config_apply(&cfg->emg);
    CHECK_EQ_U(fake_ads_dropped_writes, dropped_before); /* never a WREG in RDATAC mode */
    return ok;
}

typedef struct {
    uint8_t reg;
    uint8_t want;
} reg_want_t;

static int expect_regs(const char *name, const reg_want_t *want, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        uint8_t got = fake_ads_regs[want[i].reg];
        if (got != want[i].want) {
            fprintf(stderr, "%s: reg 0x%02X = 0x%02X, want 0x%02X\n", name, (unsigned)want[i].reg,
                    (unsigned)got, (unsigned)want[i].want);
            bad++;
        }
    }
    check_failures += bad;
    return bad;
}

/* CH1SET..CH6SET all equal to @p v. */
static void expect_muscle_chset(const char *name, uint8_t v) {
    reg_want_t w[6];
    for (int i = 0; i < 6; i++) {
        w[i] = (reg_want_t){(uint8_t)(CH1SET + i), v};
    }
    expect_regs(name, w, 6);
}

/* The golden payload = the operator template's recording config. */
static const reg_want_t GOLDEN_IMAGE[] = {
    {CONFIG1, 0x92},    /* DR=010: 4000 SPS (default) */
    {CONFIG2, 0xC0},    /* test signal off */
    {CONFIG3, 0xEC},    /* K8: PD_REFBUF | BIASREF_INT | PD_BIAS (never 0xE4) */
    {CONFIG4, 0x02},    /* lead-off comparators on */
    {LOFF, 0x00},       /* 6 nA DC */
    {CH1SET + 6, 0x81}, /* CH7: no patient pin, powered down */
    {CH1SET + 7, 0x67}, /* CH8: gain 24, MUX=111 bias drive to IN8N */
    {BIAS_SENSP, 0x3F},
    {BIAS_SENSN, 0x3F},
    {LOFF_SENSP, 0x3F},
    {LOFF_SENSN, 0xBF}, /* + IN8N (DRL electrode) */
};
#define N(a) ((int)(sizeof(a) / sizeof((a)[0])))

static void test_golden(void) {
    CHECK(apply_emg(NULL, 0));
    CHECK(ads1299_init_ok());
    expect_regs("golden", GOLDEN_IMAGE, N(GOLDEN_IMAGE));
    expect_muscle_chset("golden", 0x30); /* gain 6, normal input, SRB2 off */
}

static void test_bias(void) {
    static const mut_t ch12[] = {{"bias.channels", "3"}};
    CHECK(apply_emg(ch12, 1));
    static const reg_want_t w1[] = {{BIAS_SENSP, 0x03}, {BIAS_SENSN, 0x03}, {CONFIG3, 0xEC}};
    expect_regs("--bias-channels 1,2", w1, N(w1));

    static const mut_t off[] = {{"bias.enabled", "false"}};
    CHECK(apply_emg(off, 1));
    static const reg_want_t w2[] = {{CONFIG3, 0xE0}, {BIAS_SENSP, 0x00}, {BIAS_SENSN, 0x00},
                                    {CH1SET + 7, 0x81}, {LOFF_SENSP, 0x3F}, {LOFF_SENSN, 0x3F}};
    expect_regs("--no-bias-drive", w2, N(w2));
}

static void test_active_channels(void) {
    /* --active-channels 4: pd_channels 0x37, bias derived 0x08 */
    static const mut_t m[] = {{"pd_channels", "55"}, {"bias.channels", "8"}};
    CHECK(apply_emg(m, 2));
    static const reg_want_t w[] = {
        {CH1SET + 0, 0x81}, {CH1SET + 1, 0x81}, {CH1SET + 2, 0x81}, {CH1SET + 3, 0x30},
        {CH1SET + 4, 0x81}, {CH1SET + 5, 0x81}, {BIAS_SENSP, 0x08}, {BIAS_SENSN, 0x08},
        {LOFF_SENSP, 0x08}, {LOFF_SENSN, 0x88}};
    expect_regs("--active-channels 4", w, N(w));

    /* A powered-down channel never feeds the DRL, even if the bias mask names it. */
    static const mut_t pd_only[] = {{"pd_channels", "55"}};
    CHECK(apply_emg(pd_only, 1));
    static const reg_want_t w2[] = {{BIAS_SENSP, 0x08}, {BIAS_SENSN, 0x08}};
    expect_regs("pd masks bias", w2, N(w2));

    /* ... and the driver enforces it by itself (ads1299 API directly). */
    CHECK(apply_emg(NULL, 0));
    ads1299_set_pd_channels(0x37);
    ads1299_set_bias_channels(0x3F);
    CHECK(ads1299_init());
    expect_regs("driver masks bias", w2, N(w2));
}

static void test_gain_and_rate(void) {
    static const struct {
        const char *gain;
        uint8_t chset;
    } gains[] = {{"1", 0x00}, {"2", 0x10}, {"4", 0x20}, {"8", 0x40}, {"12", 0x50}, {"24", 0x60}};
    for (int i = 0; i < N(gains); i++) {
        mut_t m = {"default_gain", gains[i].gain};
        CHECK(apply_emg(&m, 1));
        char name[32];
        snprintf(name, sizeof(name), "--gain %s", gains[i].gain);
        expect_muscle_chset(name, gains[i].chset);
        CHECK_EQ_U(fake_ads_regs[CH1SET + 7], 0x67); /* CH8 stays at its fixed gain 24 */
    }
    static const struct {
        const char *fs;
        uint8_t config1;
    } rates[] = {{"250", 0x96}, {"500", 0x95}, {"1000", 0x94}, {"2000", 0x93}, {"4000", 0x92}};
    for (int i = 0; i < N(rates); i++) {
        mut_t m = {"sample_rate_sps", rates[i].fs};
        CHECK(apply_emg(&m, 1));
        CHECK_EQ_U(fake_ads_regs[CONFIG1], rates[i].config1);
    }
}

static void test_leadoff(void) {
    static const mut_t off_rec[] = {{"leadoff.during_recording", "false"}};
    CHECK(apply_emg(off_rec, 1));
    static const reg_want_t w0[] = {{CONFIG4, 0x00}, {LOFF, 0x00}, {LOFF_SENSP, 0x00}, {LOFF_SENSN, 0x00}};
    expect_regs("--no-leadoff", w0, N(w0));

    static const mut_t off[] = {{"leadoff.enabled", "false"}, {"leadoff.during_recording", "false"}};
    CHECK(apply_emg(off, 2));
    expect_regs("leadoff.enabled=false", w0, N(w0));

    static const mut_t i24[] = {{"leadoff.current_na", "24"}};
    CHECK(apply_emg(i24, 1));
    static const reg_want_t w1[] = {{LOFF, 0x04}, {CONFIG4, 0x02}};
    expect_regs("current_na=24", w1, N(w1));

    /* emg impedance: AC 31.2 Hz tone (FLEAD_OFF=10), 6 and 24 nA */
    static const mut_t ac[] = {{"leadoff.ac_tone", "true"}};
    CHECK(apply_emg(ac, 1));
    CHECK_EQ_U(fake_ads_regs[LOFF], 0x02);
    static const mut_t ac24[] = {{"leadoff.ac_tone", "true"}, {"leadoff.current_na", "24"}};
    CHECK(apply_emg(ac24, 2));
    CHECK_EQ_U(fake_ads_regs[LOFF], 0x06);
}

/* The EMG check's DRL probe: CH1-6 down, bias off, IN8N a plain input with the
 * strongest excitation (ILEAD_OFF=11) and only its comparator sensed. */
static const mut_t PROBE[] = {
    {"pd_channels", "63"}, {"bias", "{\"enabled\":false}"}, {"leadoff.probe_drl", "true"}};

static void test_drl_probe(void) {
    CHECK(apply_emg(PROBE, N(PROBE)));
    expect_muscle_chset("DRL probe", 0x81);
    static const reg_want_t w[] = {
        {CH1SET + 6, 0x81}, {CH1SET + 7, 0x60}, {CONFIG3, 0xE0}, {BIAS_SENSP, 0x00}, {BIAS_SENSN, 0x00},
        {LOFF_SENSP, 0x00}, {LOFF_SENSN, 0x80}, {LOFF, 0x0C}, {CONFIG4, 0x02}};
    expect_regs("DRL probe", w, N(w));
}

static void test_test_signal(void) {
    static const mut_t ts[] = {{"test_signal.enabled", "true"}};
    CHECK(apply_emg(ts, 1));
    CHECK_EQ_U(fake_ads_regs[CONFIG2], 0xD0); /* INT_CAL */
    /* MUX=101 at each channel's own gain, so the file's scaling holds:
     * muscle gain 6 (0x35), CH7/CH8 gain 24 (0x65). Was a fixed gain 1 (0x05). */
    for (int ch = 0; ch < 6; ch++) {
        CHECK_EQ_U(fake_ads_regs[CH1SET + ch], 0x35);
    }
    CHECK_EQ_U(fake_ads_regs[CH1SET + 6], 0x65);
    CHECK_EQ_U(fake_ads_regs[CH1SET + 7], 0x65);

    static const mut_t ts12[] = {{"test_signal.enabled", "true"}, {"default_gain", "12"}};
    CHECK(apply_emg(ts12, 2));
    for (int ch = 0; ch < 6; ch++) {
        CHECK_EQ_U(fake_ads_regs[CH1SET + ch], 0x55); /* gain 12 */
    }
}

/* Two applies in a row: nothing of the first survives the second. */
static void test_sequences(void) {
    CHECK(apply_emg(PROBE, N(PROBE)));
    CHECK(apply_emg(NULL, 0));
    expect_regs("probe -> restore", GOLDEN_IMAGE, N(GOLDEN_IMAGE));
    expect_muscle_chset("probe -> restore", 0x30);

    static const mut_t ts[] = {{"test_signal.enabled", "true"}};
    CHECK(apply_emg(ts, 1));
    CHECK(apply_emg(NULL, 0));
    expect_regs("test signal -> off", GOLDEN_IMAGE, N(GOLDEN_IMAGE));
    expect_muscle_chset("test signal -> off", 0x30);

    static const mut_t m[] = {{"pd_channels", "55"}, {"bias.channels", "8"}, {"default_gain", "24"}};
    CHECK(apply_emg(m, N(m)));
    CHECK(apply_emg(NULL, 0));
    expect_regs("active 4 -> all", GOLDEN_IMAGE, N(GOLDEN_IMAGE));
    expect_muscle_chset("active 4 -> all", 0x30);
}

int main(void) {
    golden = read_file(GOLDEN_MCU_APPLY_JSON);
    CHECK(golden != NULL);
    if (golden == NULL) {
        CHECK_DONE("test_ads1299_registers");
    }
    fake_ads_power_on();
    test_golden();
    test_bias();
    test_active_channels();
    test_gain_and_rate();
    test_leadoff();
    test_drl_probe();
    test_test_signal();
    test_sequences();
    CHECK(fake_ads_writes > 0);
    free(golden);
    CHECK_DONE("test_ads1299_registers");
}
