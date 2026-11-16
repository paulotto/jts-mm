// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file trigger_config.c
 * @brief Trigger domain apply and TRIGGER_EVENT emit.
 */

#include "jtsmm/domain/trigger_config.h"

#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/drivers/trigger_timing.h"
#include "jtsmm/config.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"
#include <string.h>

void trigger_config_resolve_profile(trigger_config_t *cfg) {
    if (cfg->profile == TRIGGER_PROFILE_CUSTOM) {
        return;
    }
    switch (cfg->profile) {
    case TRIGGER_PROFILE_IMX296_HIGH_FPS:
        cfg->rate_hz = 100;
        cfg->pulse_width_us = 2000;
        cfg->outputs.csi1 = cfg->outputs.csi2 = cfg->outputs.scope = true;
        break;
    case TRIGGER_PROFILE_BENCH_SCOPE:
        cfg->rate_hz = 10;
        cfg->pulse_width_us = 1000;
        cfg->outputs.scope = true;
        break;
    case TRIGGER_PROFILE_TRACK_A_LED:
        /* h06 §3.4: webcams + sync LED at 30 Hz. The pulse on PIN_TRIG only
         * lights the TRIG LED here (no XTRIG camera), so keep it short. */
        cfg->rate_hz = 30;
        cfg->pulse_width_us = 200;
        break;
    case TRIGGER_PROFILE_IMX296_DUAL_CSI:
    default:
        /* 30 fps / 12 ms exposure: the rig's operating point (was 60 Hz / 4 ms). */
        cfg->rate_hz = 30;
        cfg->pulse_width_us = 12000;
        cfg->outputs.csi1 = cfg->outputs.csi2 = cfg->outputs.scope = true;
        break;
    }
}

bool trigger_config_validate(const trigger_config_t *cfg) {
    return cfg != NULL && trigger_timing_check(cfg->rate_hz, cfg->pulse_width_us) == NULL;
}

bool trigger_config_apply(const trigger_config_t *cfg) {
    if (!trigger_config_validate(cfg)) {
        return false;
    }
    const platform_config_t *plat = &jtsmm_config_active()->platform;
    /* Idle at this polarity's inactive level from now on, not only after the
     * first trig_start. */
    trigger_pio_set_idle_polarity(cfg->polarity == 0);
    if (!trigger_pio_init(plat->pio_trigger_clock_hz)) {
        return false;
    }
    trigger_pio_set_irq_priority(plat->irq_trigger_priority);
    return true;
}

void trigger_on_pio_event(uint64_t t_us, uint64_t trigger_id, uint8_t stamp_src, bool fallback) {
    jtsmm_record_t rec = {0};
    rec.type = REC_TRIGGER_EVENT;
    rec.t_master_us = t_us;
    rec.len = TRIGGER_EVENT_PAYLOAD_LEN;
    /* flags[0] = stamp source, flags[1] = watchdog fallback engaged. Old firmware
     * sent all-zero flags (TRIG_STAMP_UNKNOWN), so a host can tell the three
     * apart and `jtsmm integrity` can refuse to trust a timer-stamped session
     * that expected edge stamps. flags[2..3] reserved (0). */
    rec.payload[0] = stamp_src;
    rec.payload[1] = fallback ? 1u : 0u;
    memcpy(rec.payload + 4, &trigger_id, 8);
    ringbuf_push(RING_EVENTS, &rec);
}
