// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Per-key plumbing check over every leaf of the golden `config apply` payload
 * (schema/v1/golden/mcu_apply_template.json, generated from etc/jtsmm.yaml):
 *
 *   1. dump(parse(golden)) writes the key back (config save -> load keeps it);
 *   2. changing the key's value either changes the parsed config or is refused
 *      with an error -- a key the firmware accepts but ignores fails the test.
 *
 * Keys that are knowingly inert sit in INERT_KEYS with the reason. A new key in
 * the golden file therefore fails here until the firmware honours it (or it is
 * listed with a reason), which is what keeps host and firmware from drifting.
 */
#include "jtsmm/runtime/config_json.h"

#include "cJSON.h"
#include "check.h"
#include "config_compare.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef GOLDEN_MCU_APPLY_JSON
#define GOLDEN_MCU_APPLY_JSON "../../../schema/v1/golden/mcu_apply_template.json"
#endif

/* Leaves whose value cannot change the parsed config, and why. */
static const struct {
    const char *path;
    bool dumped; /* written back by config_json_dump */
    const char *why;
} INERT_KEYS[] = {
    {"platform.profile", false, "only default|custom accepted; no stored field"},
    /* The payload always carries explicit values (sample_rate_sps, rate_hz,
     * reports), which force the block to CUSTOM, so the profile name cannot
     * change anything. (imu fusion.* is stored and dumped, but like the profile
     * only feeds profile expansion; see the host-side config audit #25.) */
    {"emg.profile", true, "explicit values always follow (profile -> custom)"},
    {"imu_mta.profile", true, "explicit reports always follow"},
    {"imu_cra.profile", true, "explicit reports always follow"},
    {"trigger.profile", true, "explicit rate/width always follow"},
};

static const char *inert_reason(const char *path, bool *dumped) {
    for (size_t i = 0; i < sizeof(INERT_KEYS) / sizeof(INERT_KEYS[0]); i++) {
        if (strcmp(INERT_KEYS[i].path, path) == 0) {
            *dumped = INERT_KEYS[i].dumped;
            return INERT_KEYS[i].why;
        }
    }
    *dumped = true;
    return NULL;
}

/* Alternative values for string leaves (anything else gets "zz_unknown"). */
static const char *alt_string(const char *path, const char *cur) {
    if (strstr(path, "polarity")) {
        return strcmp(cur, "imx296_fast") == 0 ? "active_high" : "imx296_fast";
    }
    if (strstr(path, "fusion.primary")) {
        return strcmp(cur, "rotation_vector") == 0 ? "game_rotation_vector" : "rotation_vector";
    }
    if (strstr(path, "sync.mode")) {
        return strcmp(cur, "int_driven") == 0 ? "emg_drdy_divisor" : "int_driven";
    }
    return "zz_unknown";
}

static cJSON *find_path(cJSON *root, const char *path) {
    char buf[160];
    snprintf(buf, sizeof(buf), "%s", path);
    cJSON *node = root;
    for (char *tok = strtok(buf, "."); tok != NULL && node != NULL; tok = strtok(NULL, ".")) {
        node = cJSON_GetObjectItemCaseSensitive(node, tok);
    }
    return node;
}

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

static cJSON *golden_root;
static cJSON *dump_root;
static jtsmm_mcu_config_t baseline;
static int leaves;

static void check_leaf(const char *path, const cJSON *leaf) {
    leaves++;
    bool dumped_expected;
    const char *inert = inert_reason(path, &dumped_expected);

    /* 1. written back by the dump */
    bool in_dump = find_path(dump_root, path) != NULL;
    if (in_dump != dumped_expected) {
        fprintf(stderr, "  %s: %s the dump\n", path, in_dump ? "unexpectedly in" : "missing from");
        check_failures++;
    }

    /* 2. a different value takes effect or is refused */
    cJSON *copy = cJSON_Duplicate(golden_root, 1);
    cJSON *target = find_path(copy, path);
    if (cJSON_IsBool(target)) {
        cJSON_SetBoolValue(target, !cJSON_IsTrue(target));
    } else if (cJSON_IsNumber(target)) {
        /* 2n+1: always a different integer, and big enough to move a ring
         * size by at least one slot (sizes are bytes, quantised to slots). */
        cJSON_SetNumberValue(target, target->valuedouble * 2.0 + 1.0);
    } else if (cJSON_IsString(target)) {
        cJSON_SetValuestring(target, alt_string(path, target->valuestring));
    }
    char *text = cJSON_PrintUnformatted(copy);
    jtsmm_mcu_config_t c;
    jtsmm_config_set_defaults(&c);
    bool ok = config_json_parse(text, &c);
    diffs_quiet = true;
    bool changed = ok && count_diffs(&baseline, &c) > 0;
    diffs_quiet = false;
    if (inert == NULL && ok && !changed) {
        fprintf(stderr, "  %s: accepted a new value but the config did not change (ignored key)\n",
                path);
        check_failures++;
    }
    if (inert != NULL && changed) {
        fprintf(stderr, "  %s: listed as inert (%s) but now takes effect; drop it from INERT_KEYS\n",
                path, inert);
        check_failures++;
    }
    cJSON_free(text);
    cJSON_Delete(copy);
    (void)leaf;
}

static void walk(const cJSON *node, const char *prefix) {
    for (const cJSON *child = node->child; child != NULL; child = child->next) {
        char path[160];
        snprintf(path, sizeof(path), "%s%s%s", prefix, prefix[0] ? "." : "", child->string);
        if (cJSON_IsObject(child)) {
            walk(child, path);
        } else {
            check_leaf(path, child);
        }
    }
}

int main(int argc, char **argv) {
    const char *golden_path = (argc > 1) ? argv[1] : GOLDEN_MCU_APPLY_JSON;
    char *json = read_file(golden_path);
    if (json == NULL) {
        fprintf(stderr, "cannot read golden JSON: %s\n", golden_path);
        return 1;
    }
    golden_root = cJSON_Parse(json);
    CHECK(golden_root != NULL);
    jtsmm_config_set_defaults(&baseline);
    CHECK(config_json_parse(json, &baseline));
    static char dumped[JTSMM_CMD_LINE_MAX];
    CHECK(config_json_dump(&baseline, dumped, sizeof(dumped)));
    dump_root = cJSON_Parse(dumped);
    CHECK(dump_root != NULL);
    if (golden_root != NULL && dump_root != NULL) {
        walk(golden_root, "");
    }
    CHECK(leaves > 30); /* the walk really covered the payload */
    printf("checked %d golden leaves\n", leaves);
    cJSON_Delete(golden_root);
    cJSON_Delete(dump_root);
    free(json);
    CHECK_DONE("test_config_json_keys");
}
