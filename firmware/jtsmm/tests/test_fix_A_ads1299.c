// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Host tests for the pure ADS1299 register/stream arithmetic
 * (include/jtsmm/drivers/ads1299_regs.h). Uses CHECK, not assert, so the
 * checks survive -DNDEBUG.
 */
#include "jtsmm/drivers/ads1299_regs.h"

#include <stdio.h>

static int failures;

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                      \
        }                                                                    \
    } while (0)

static void test_config3(void) {
    /* SBAS499C §10.1.2.2: "WREG CONFIG3 b'x1xx 1100 // Turn on BIAS amplifier,
     * set internal BIASREF voltage". */
    CHECK(ads_config3_value(true) == 0xECu);
    CHECK(ads_config3_value(false) == 0xE0u);
    CHECK(ads_config3_value(true) & ADS_CONFIG3_PD_BIAS);
    CHECK(ads_config3_value(true) & ADS_CONFIG3_BIASREF_INT);
    /* 0xE4 (external BIASREF = AGND on this board) must never be produced. */
    CHECK(ads_config3_value(true) != 0xE4u && ads_config3_value(false) != 0xE4u);
    /* BIAS_STAT (bit 0) is read-only: a set status bit must not fail readback. */
    CHECK(ads_reg_readback_ok(ADS_REG_CONFIG3, 0xECu, 0xEDu));
    CHECK(ads_reg_readback_ok(ADS_REG_CONFIG3, 0xE0u, 0xE1u));
    CHECK(!ads_reg_readback_ok(ADS_REG_CONFIG3, 0xECu, 0xE8u)); /* PD_BIAS lost */
    CHECK(!ads_reg_readback_ok(ADS_REG_CONFIG1, 0x93u, 0x92u));
    CHECK(ads_reg_readback_ok(ADS_REG_LOFF_SENSP, 0x3Fu, 0x3Fu));
}

static void test_config4_and_leadoff_masks(void) {
    /* SBAS499C §10.1.2.1: "WREG CONFIG4 0x02 // Turn on dc lead-off comparators". */
    CHECK(ads_config4_value(true) == 0x02u);
    CHECK(ads_config4_value(false) == 0x00u);
    CHECK(ADS_REG_LOFF_STATP == 0x12u && ADS_REG_LOFF_STATN == 0x13u);

    uint8_t mp = 0;
    uint8_t mn = 0;
    /* IN8N excitation for the DRL (0x80 in SENSN) is never a published bit. */
    ads_leadoff_status_masks(0x3Fu, 0xBFu, &mp, &mn);
    CHECK(mp == 0x3Fu && mn == 0x3Fu);
    ads_leadoff_status_masks(0xFFu, 0xFFu, &mp, &mn);
    CHECK(mp == 0x3Fu && mn == 0x3Fu);
    /* Powered-down CH2 (not in SENS) must not light the LED. */
    ads_leadoff_status_masks(0x3Du, 0x3Du, &mp, &mn);
    CHECK(mp == 0x3Du && mn == 0x3Du);

    CHECK(!ads_leadoff_any(0xC0u, 0x80u, 0x3Fu, 0xBFu)); /* CH7/CH8 only */
    CHECK(!ads_leadoff_any(0x02u, 0x00u, 0x3Du, 0x3Du)); /* CH2 not sensed */
    CHECK(ads_leadoff_any(0x01u, 0x00u, 0x3Fu, 0x3Fu));
    CHECK(ads_leadoff_any(0x00u, 0x20u, 0x3Fu, 0x3Fu));
    CHECK(!ads_leadoff_any(0xFFu, 0xFFu, 0x00u, 0x00u)); /* lead-off disabled */

    /* Status word 1100 | STATP | STATN | GPIO. */
    uint8_t status[3] = {0xC1u, 0x23u, 0x40u};
    uint8_t sp = 0;
    uint8_t sn = 0;
    ads_status_word_leadoff(status, &sp, &sn);
    CHECK(sp == 0x12u && sn == 0x34u);
    CHECK(ads_status_word_framed(status));
    status[0] = 0x61u;
    CHECK(!ads_status_word_framed(status));
}

static void test_id(void) {
    CHECK(ads_id_valid(0x3Eu));  /* the value real ADS1299 parts report */
    CHECK(ads_id_valid(0x1Eu));  /* other REV_ID */
    CHECK(ads_id_valid(0xFEu));
    CHECK(!ads_id_valid(0xFFu)); /* MISO floating high */
    CHECK(!ads_id_valid(0x00u)); /* MISO stuck low */
    CHECK(!ads_id_valid(0xF8u)); /* previously accepted "shifted" IDs */
    CHECK(!ads_id_valid(0xF9u));
    CHECK(!ads_id_valid(0xFAu));
    CHECK(!ads_id_valid(0xFBu));
    CHECK(!ads_id_valid(0x3Du)); /* ADS1299-6 */
    CHECK(!ads_id_valid(0x3Cu)); /* ADS1299-4 */
    CHECK(!ads_id_valid(0x2Eu)); /* bit 4 must read 1 */
}

static void test_config2_test_signal(void) {
    /* Reserved bits 7:5 stay 110b, bit 3 stays 0, only INT_CAL toggles. */
    CHECK(ads_config2_test_signal(0xC0u, true) == 0xD0u);
    CHECK(ads_config2_test_signal(0xD0u, false) == 0xC0u);
    CHECK(ads_config2_test_signal(0xC5u, true) == 0xD5u);  /* CAL_AMP/FREQ kept */
    CHECK(ads_config2_test_signal(0xD5u, false) == 0xC5u);
    for (unsigned v = 0; v < 256u; v++) {
        uint8_t on = ads_config2_test_signal((uint8_t)v, true);
        uint8_t off = ads_config2_test_signal((uint8_t)v, false);
        CHECK((on & 0xE8u) == 0xC0u && (on & 0x10u));
        CHECK((off & 0xF8u) == 0xC0u);
    }
}

static void test_gain_current_rate(void) {
    static const uint8_t ok[] = {1, 2, 4, 6, 8, 12, 24};
    static const uint8_t code[] = {0x00, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60};
    for (unsigned i = 0; i < sizeof(ok); i++) {
        CHECK(ads_gain_supported(ok[i]));
        CHECK(ads_chset_for_gain(ok[i]) == code[i]);
    }
    unsigned n_ok = 0;
    for (unsigned g = 0; g < 256u; g++) {
        n_ok += ads_gain_supported((uint8_t)g) ? 1u : 0u;
    }
    CHECK(n_ok == 7u);
    CHECK(!ads_gain_supported(0u) && !ads_gain_supported(3u) && !ads_gain_supported(16u));

    CHECK(ads_leadoff_current_supported(6u) && ads_leadoff_current_supported(24u));
    CHECK(!ads_leadoff_current_supported(0u) && !ads_leadoff_current_supported(12u));
    CHECK(!ads_leadoff_current_supported(60u));

    CHECK(ads_sample_rate_supported(250u) && ads_sample_rate_supported(2000u));
    CHECK(ads_sample_rate_supported(4000u));
    CHECK(!ads_sample_rate_supported(16000u) && !ads_sample_rate_supported(8000u));
    CHECK(ads_config1_for_fs(2000u) == 0x93u && ads_config1_for_fs(250u) == 0x96u);
    CHECK(ads_config1_for_fs(4000u) == 0x92u); /* DR=010, SBAS499C Table 13 */
}

static void test_channels_and_spi(void) {
    CHECK(ads_ch8set(true, false) == 0x67u);  /* DRL: MUX=111 */
    CHECK(ads_ch8set(false, true) == 0x60u);  /* probe: plain input */
    CHECK(ads_ch8set(true, true) == 0x60u);   /* probe wins */
    CHECK(ads_ch8set(false, false) == 0x81u); /* nothing on CH8: powered down */
    CHECK(ADS_CHSET_OFF == 0x81u);

    CHECK(ads_clamp_stream_hz(8000000u) == 4000000u);
    CHECK(ads_clamp_stream_hz(4000000u) == 4000000u);
    CHECK(ads_clamp_stream_hz(2000000u) == 2000000u);
}

static void test_drdy_tracker(void) {
    const uint32_t T = 500u; /* 2 kSPS */
    ads_drdy_track_t s = {0};

    CHECK(ads_drdy_track(&s, 10000u, T) == 0u); /* first read of a stream */
    CHECK(ads_drdy_track(&s, 10503u, T) == 0u); /* prompt */
    CHECK(ads_drdy_track(&s, 11002u, T) == 0u);

    /* Late by 0.9 T but before the next conversion: nothing lost (round()
     * would have reported a phantom loss here). */
    CHECK(ads_drdy_track(&s, 11002u + T + 450u, T) == 0u);
    /* ...and the phase was not dragged along by the late read. */
    CHECK(ads_drdy_track(&s, 11002u + 2u * T + 3u, T) == 0u);

    /* Stall across two conversions, ending in the first or second half of the
     * third period: exactly two lost either way. */
    uint64_t base = 11002u + 2u * T + 3u;
    CHECK(ads_drdy_track(&s, base + 3u * T + 20u, T) == 2u);
    base += 3u * T + 20u;
    CHECK(ads_drdy_track(&s, base + 3u * T + 400u, T) == 2u);
    base += 3u * T; /* ref stays on the predicted edge after a late read */
    CHECK(ads_drdy_track(&s, base + T + 5u, T) == 0u);

    /* The conversion that falls inside a late read's DMA frame is hidden (the
     * frame's SCLKs pull DRDY back high); it must still be counted. Read at
     * +450 us, 56 us frame at 4 MHz -> the edge at +500 us lands inside it. */
    ads_drdy_track_t h = {0};
    CHECK(ads_drdy_track(&h, 1000u, T) == 0u);
    CHECK(ads_drdy_track(&h, 1000u + T + 450u, T) == 0u); /* very late read */
    CHECK(ads_drdy_track(&h, 1000u + 3u * T + 4u, T) == 1u);

    /* ADS oscillator 2 % slow / fast: prompt reads never report a loss. */
    ads_drdy_track_t d = {0};
    uint64_t t = 0;
    uint32_t lost = 0;
    CHECK(ads_drdy_track(&d, 0u, T) == 0u);
    for (int i = 0; i < 5000; i++) {
        t += 510u;
        lost += ads_drdy_track(&d, t + (uint64_t)(i % 7), T);
    }
    CHECK(lost == 0u);
    ads_drdy_track_t f = {0};
    t = 0;
    lost = 0;
    CHECK(ads_drdy_track(&f, 0u, T) == 0u);
    for (int i = 0; i < 5000; i++) {
        t += 490u;
        lost += ads_drdy_track(&f, t + (uint64_t)(i % 7), T);
    }
    CHECK(lost == 0u);

    /* Long stall: 100 periods. */
    ads_drdy_track_t l = {0};
    CHECK(ads_drdy_track(&l, 0u, T) == 0u);
    CHECK(ads_drdy_track(&l, 100u * T + 7u, T) == 99u);
}

int main(void) {
    test_config3();
    test_config4_and_leadoff_masks();
    test_id();
    test_config2_test_signal();
    test_gain_current_rate();
    test_channels_and_spi();
    test_drdy_tracker();
    if (failures) {
        printf("test_fix_A_ads1299: %d FAILED\n", failures);
        return 1;
    }
    printf("test_fix_A_ads1299 OK\n");
    return 0;
}
