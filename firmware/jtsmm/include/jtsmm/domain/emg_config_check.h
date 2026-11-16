// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file emg_config_check.h
 * @brief Pure validation of an EMG config block (no hardware access).
 *
 * @details Used by emg_config_validate(); header-only so the host tests can
 *          exercise the exact rules. Unsupported values are rejected, never
 *          remapped: the host derives lsb_uv and lead-off thresholds from what
 *          it requested, so a silent substitution scales the whole recording.
 */
#pragma once

#include "jtsmm/config.h"
#include "jtsmm/drivers/ads1299_regs.h"

#include <stddef.h>

/**
 * @brief Why @p cfg cannot be applied, or NULL if it can.
 * @param cfg EMG config (may be NULL).
 * @return Static reason string, or NULL when valid.
 */
static inline const char *emg_config_check(const emg_config_t *cfg) {
    if (cfg == NULL) {
        return "no config";
    }
    if (cfg->sample_rate_sps == 16000u) {
        /* One RDATA frame per DRDY: 56 us of SPI at 4 MHz in a 62.5 us period,
         * plus DRDY polling latency, loses most frames, and 16k records/s alone
         * saturate the USB full-speed link. Not usable for any measurement. */
        return "sample_rate_sps 16000 not supported by this firmware (max 4000)";
    }
    if (!ads_sample_rate_supported(cfg->sample_rate_sps)) {
        return "sample_rate_sps must be 250, 500, 1000, 2000 or 4000";
    }
    if (!ads_gain_supported(cfg->default_gain)) {
        return "default_gain must be 1, 2, 4, 6, 8, 12 or 24";
    }
    if (cfg->leadoff.enabled && !ads_leadoff_current_supported(cfg->leadoff.current_na)) {
        return "leadoff.current_na must be 6 or 24";
    }
    if (cfg->bias.enabled &&
        (cfg->bias.channel_mask & (uint8_t)~cfg->pd_channel_mask & ADS_MUSCLE_MASK) == 0u) {
        return "bias enabled but no powered channel in bias.channels";
    }
    return NULL;
}
