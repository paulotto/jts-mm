// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file trigger_timing.c
 * @brief Pure trigger period/template arithmetic (see trigger_timing.h).
 */

#include "jtsmm/drivers/trigger_timing.h"

#include <stddef.h>

const char *trigger_timing_check(uint32_t rate_hz, uint32_t width_us) {
    if (rate_hz == 0 || rate_hz > TRIGGER_MAX_RATE_HZ) {
        return "rate_hz must be 1..200";
    }
    if (width_us == 0) {
        return "pulse width must be > 0 us";
    }
    if (width_us >= 1000000u / rate_hz) {
        return "pulse width must be shorter than the period (1e6/rate_hz us)";
    }
    return NULL;
}

bool trigger_timing_idle_high(bool imx296_fast) {
    /* Inactive level: HIGH when exposure is the LOW phase, else LOW. */
    return imx296_fast;
}

uint32_t trigger_timing_edge_track(trigger_edge_track_t *s, uint64_t t_us, uint32_t period_us) {
    if (!s->valid || period_us == 0u) {
        s->ref_us = t_us;
        s->valid = true;
        return 1u;
    }
    uint64_t tol = period_us / 16u;
    uint64_t k = 1u;
    if (t_us + tol >= s->ref_us + period_us) {
        k = (t_us + tol - s->ref_us) / period_us;
    }
    uint64_t pred = s->ref_us + k * period_us;
    s->ref_us = (t_us < pred + tol) ? t_us : pred;
    return k > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)k;
}

bool trigger_timing_compute(uint16_t rate_hz, uint32_t pulse_width_us, bool imx296_fast,
                            bool compensate, trigger_timing_t *out) {
    if (out == NULL || rate_hz == 0) {
        return false;
    }
    uint32_t period_us = 1000000u / rate_hz;
    /* 0 keeps its documented "half period" meaning for internal callers; a
     * width that does not fit the period is an error now -- it used to be
     * halved silently, i.e. the camera exposure changed without a word. */
    if (pulse_width_us == 0) {
        pulse_width_us = period_us / 2u;
    }
    if (pulse_width_us >= period_us) {
        return false;
    }

    /* IMX296 fast trigger: exposure ≈ low time + ~14 µs (h06 §3.1), so the
     * requested width is the LOW phase. Other polarity: width is the HIGH phase. */
    uint32_t low_us;
    uint32_t high_us;
    if (imx296_fast) {
        low_us = pulse_width_us;
        high_us = period_us - pulse_width_us;
    } else {
        high_us = pulse_width_us;
        low_us = period_us - pulse_width_us;
    }

    /*
     * The PIO program spends TRIG_PIO_*_OVERHEAD_CYCLES per phase on the reload
     * and IRQ instructions, on top of the `jmp y--` countdown (ticks+1 cycles).
     * Uncompensated, a 60 Hz request produced 16671 µs periods against a
     * 16666 µs software timer: 5 µs/frame of drift, a full frame every ~56 s.
     * Compensation keeps the *pin* period equal to period_us; the templates
     * themselves are what a scope on TP_TRIG must agree with.
     */
    uint32_t low_ticks = low_us;
    uint32_t high_ticks = high_us;
    if (compensate) {
        low_ticks = (low_us > TRIG_PIO_LOW_OVERHEAD_CYCLES) ? low_us - TRIG_PIO_LOW_OVERHEAD_CYCLES : 1u;
        high_ticks = (high_us > TRIG_PIO_HIGH_OVERHEAD_CYCLES) ? high_us - TRIG_PIO_HIGH_OVERHEAD_CYCLES : 1u;
    }
    /* `jmp y--` with y == 0 still runs once; a template of 0 would not shorten
     * the phase any further and reads confusingly, so keep >= 1. */
    if (low_ticks == 0) {
        low_ticks = 1u;
    }
    if (high_ticks == 0) {
        high_ticks = 1u;
    }

    out->period_us = period_us;
    out->low_us = low_us;
    out->high_us = high_us;
    out->low_ticks = low_ticks;
    out->high_ticks = high_ticks;
    out->pin_period_us = (low_ticks + TRIG_PIO_LOW_OVERHEAD_CYCLES) +
                         (high_ticks + TRIG_PIO_HIGH_OVERHEAD_CYCLES);
    return true;
}
