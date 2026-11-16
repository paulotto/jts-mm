// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Host tests for EMG config validation (include/jtsmm/domain/emg_config_check.h)
 * and the REC_EMG_BLOCK payload layout (contract K1). CHECK survives NDEBUG.
 */
#include "jtsmm/domain/emg_config_check.h"
#include "jtsmm/domain/emg_block.h"
#include "jtsmm/records.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static emg_config_t good(void) {
    emg_config_t c;
    memset(&c, 0, sizeof(c));
    c.enabled = true;
    c.profile = EMG_PROFILE_CUSTOM;
    c.sample_rate_sps = 2000u;
    c.default_gain = 6u;
    c.bias.enabled = true;
    c.bias.channel_mask = 0x3Fu;
    c.leadoff.enabled = true;
    c.leadoff.current_na = 6u;
    c.stream_channels = 6u;
    return c;
}

static void test_validation(void) {
    emg_config_t c = good();
    CHECK(emg_config_check(&c) == NULL);
    CHECK(emg_config_check(NULL) != NULL);

    static const uint16_t rates_ok[] = {250, 500, 1000, 2000, 4000};
    for (unsigned i = 0; i < 5; i++) {
        c = good();
        c.sample_rate_sps = rates_ok[i];
        CHECK(emg_config_check(&c) == NULL);
    }
    static const uint16_t rates_bad[] = {0, 100, 3000, 8000, 16000};
    for (unsigned i = 0; i < 5; i++) {
        c = good();
        c.sample_rate_sps = rates_bad[i];
        CHECK(emg_config_check(&c) != NULL);
    }

    for (unsigned g = 0; g < 256u; g++) {
        c = good();
        c.default_gain = (uint8_t)g;
        bool ok = g == 1 || g == 2 || g == 4 || g == 6 || g == 8 || g == 12 || g == 24;
        CHECK((emg_config_check(&c) == NULL) == ok);
    }

    c = good();
    c.leadoff.current_na = 24u;
    CHECK(emg_config_check(&c) == NULL);
    c.leadoff.current_na = 12u; /* used to be silently run as 6 nA */
    CHECK(emg_config_check(&c) != NULL);
    c.leadoff.current_na = 0u;
    CHECK(emg_config_check(&c) != NULL);
    c.leadoff.enabled = false; /* current irrelevant when lead-off is off */
    CHECK(emg_config_check(&c) == NULL);

    c = good();
    c.pd_channel_mask = 0x3Fu; /* every bias channel powered down */
    CHECK(emg_config_check(&c) != NULL);
    c.bias.enabled = false;
    CHECK(emg_config_check(&c) == NULL);
    c = good();
    c.bias.channel_mask = 0x01u;
    c.pd_channel_mask = 0x3Eu;
    CHECK(emg_config_check(&c) == NULL);
}

static void test_block_payload(void) {
    CHECK(EMG_BLOCK_PAYLOAD_LEN == 31u);
    CHECK(EMG_BLOCK_ADS_BYTES == 27u);
    CHECK(EMG_BLOCK_PAYLOAD_LEN <= JTSMM_MAX_PAYLOAD);

    uint8_t frame[EMG_BLOCK_ADS_BYTES];
    for (unsigned i = 0; i < sizeof(frame); i++) {
        frame[i] = (uint8_t)(0xC0u + i);
    }
    uint8_t out[EMG_BLOCK_PAYLOAD_LEN + 1];
    memset(out, 0xAA, sizeof(out));
    CHECK(emg_block_pack(out, frame, 0x12345678u) == EMG_BLOCK_PAYLOAD_LEN);
    CHECK(memcmp(out, frame, sizeof(frame)) == 0);
    CHECK(out[27] == 0x78u && out[28] == 0x56u && out[29] == 0x34u && out[30] == 0x12u);
    CHECK(out[31] == 0xAAu); /* no overrun */

    emg_block_pack(out, frame, 0xFFFFFFFFu);
    CHECK(out[27] == 0xFFu && out[30] == 0xFFu);
    emg_block_pack(out, frame, 0u);
    CHECK(out[27] == 0u && out[28] == 0u && out[29] == 0u && out[30] == 0u);
}

int main(void) {
    test_validation();
    test_block_payload();
    if (failures) {
        printf("test_fix_A_emg: %d FAILED\n", failures);
        return 1;
    }
    printf("test_fix_A_emg OK\n");
    return 0;
}
