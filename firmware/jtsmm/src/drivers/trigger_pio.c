// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file trigger_pio.c
 * @brief PIO camera trigger generator + edge-stamped TRIGGER_EVENTs.
 */

#include "jtsmm/drivers/trigger_pio.h"

#include "jtsmm/domain/trigger_config.h"
#include "jtsmm/drivers/trigger_timing.h"
#include "jtsmm/pins.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/timebase.h"
#include "trigger_gen.pio.h"

#include "hardware/gpio.h"
#include "hardware/irq.h"
#include "hardware/pio.h"
#include "hardware/sync.h"
#include "hardware/timer.h"
#include "pico/stdlib.h"
#include <stdio.h>

/*
 * Timing model.
 *
 * The PIO produces the pulse train; a software repeating_timer runs at the same
 * period. Previously the timer alone stamped TRIGGER_EVENT, on the assumption
 * that the two stay in lockstep because they share a crystal. They do not: the
 * PIO loop costs 5 extra cycles per period (trigger_gen.pio), so at 60 Hz the
 * timer ran 5 us/frame fast against the pin and the recorded camera times
 * drifted a full frame every ~56 s. Nothing downstream could see it, because a
 * camera row's t_master_us *is* the trigger's t_master_us.
 *
 * Now:
 *   1. Templates are compensated so the pin period is exactly period_us
 *      (trigger_timing.c). `trig_comp off` restores the legacy templates.
 *   2. The PIO raises IRQ flag 0 on the instruction that drives the exposure-
 *      start edge; trigger_pio_irq_handler() latches t_master_us() there. The
 *      stamp is now a measurement of the edge, not a prediction of it.
 *   3. The timer is demoted to a watchdog. If the PIO IRQ goes silent for two
 *      periods it takes over stamping for the rest of the run (latched), back-
 *      fills any ids it missed, and comms_core logs the fact once per second.
 *      Worst case is therefore exactly the old behaviour plus a warning.
 *      `trig_stamp timer` selects the old behaviour outright for an A/B.
 *   4. Lost edges (contract K4). One IRQ flag cannot count: if the IRQ is not
 *      serviced within a period, the next edge re-sets the same flag and two
 *      pulses produce one IRQ. The ISR therefore derives the number of edges
 *      since the last counted one from the pin schedule
 *      (trigger_timing_edge_track) and advances trigger_id by that many, so
 *      the host sees an id gap -- a missing trigger -- instead of every later
 *      frame silently pairing with the wrong trigger. The PIO IRQ also runs at
 *      a higher NVIC priority than everything else on this core (irq.trigger).
 *
 * All of this runs on the comms core (trig_start arrives over CDC or from the
 * panel button, both comms-core paths), so the PIO IRQ and the timer callback
 * are the only producers of REC_TRIGGER_EVENT and ringbuf_push() masks IRQs for
 * RING_EVENTS to keep them from interleaving with the panel button push.
 */

#define TRIG_PIO_IRQ_FLAG 0u
/* Periods the PIO IRQ may be silent before the timer takes over. */
#define TRIG_FALLBACK_SILENT_PERIODS 2u

static PIO trig_pio;
static uint trig_sm;
static uint trig_offset;
static volatile bool running;
static bool pio_claimed;
static bool pio_irq_installed;
static bool timer_active;
static volatile uint64_t trigger_id;
static uint32_t clock_hz = 1000000u;
static repeating_timer_t tick_rt;

static trigger_stamp_mode_t stamp_mode = TRIG_STAMP_MODE_PIO_IRQ;
static trigger_stamp_mode_t active_mode = TRIG_STAMP_MODE_PIO_IRQ;
static bool comp_enabled = true;
static bool active_fast;
/* Polarity whose inactive level the pin idles at: set from the applied config
 * (trigger_config_apply) and by every start. Default = factory polarity 0. */
static bool idle_fast = true;
static uint32_t active_period_us;
static uint32_t active_pin_period_us;

static volatile uint64_t last_pio_event_us;
static trigger_edge_track_t edge_track; /* ISR-only after trig_start resets it */
static volatile uint32_t pio_pulse_count;
static volatile uint32_t timer_event_count;
static volatile uint32_t pio_irq_spurious;
static volatile bool fallback_active;
static volatile uint32_t ticks_since_start;
static volatile uint32_t lost_edges;
static uint8_t irq_level; /* platform.irq.trigger: 0 = highest */

/** @brief Internal helper: tick_timer_cancel. */
static void tick_timer_cancel(void) {
    if (timer_active) {
        cancel_repeating_timer(&tick_rt);
        timer_active = false;
    }
}

/** @brief Internal helper: emit one TRIGGER_EVENT for the edge @p advance ids on. */
static void emit_event_n(uint64_t t_us, uint32_t advance, uint8_t stamp_src, bool fallback) {
    trigger_id += advance;
    trigger_on_pio_event(t_us, trigger_id, stamp_src, fallback);
}

/** @brief Internal helper: emit one TRIGGER_EVENT. */
static void emit_event(uint64_t t_us, uint8_t stamp_src, bool fallback) {
    emit_event_n(t_us, 1u, stamp_src, fallback);
}

/** @brief PIO IRQ: fires on the exposure-start edge of every pulse. */
static void trigger_pio_irq_handler(void) {
    if (!pio_interrupt_get(trig_pio, TRIG_PIO_IRQ_FLAG)) {
        pio_irq_spurious++;
        return;
    }
    pio_interrupt_clear(trig_pio, TRIG_PIO_IRQ_FLAG);
    if (!running) {
        return;
    }
    /* Edges since the last counted one, against the pin schedule; the tracker's
     * ref is this IRQ's time when it was prompt, else the scheduled edge time
     * (a late IRQ's own timestamp would be up to a period after the edge). */
    uint32_t k = trigger_timing_edge_track(&edge_track, t_master_us(), active_pin_period_us);
    uint64_t t = edge_track.ref_us;
    if (k > 1u) {
        lost_edges += k - 1u;
    }
    last_pio_event_us = t;
    pio_pulse_count++;
    if (active_mode != TRIG_STAMP_MODE_PIO_IRQ || fallback_active) {
        /* Counted for diagnostics (trig_status shows pulses vs events), but the
         * timer owns the event stream in this mode. */
        return;
    }
    emit_event_n(t, k, TRIG_STAMP_PIO_IRQ, false);
}

/** @brief Internal helper: tick_cb (software timer; stamper or watchdog). */
static bool tick_cb(repeating_timer_t *rt) {
    (void)rt;
    if (!running) {
        return true;
    }
    if (active_mode == TRIG_STAMP_MODE_TIMER) {
        timer_event_count++;
        emit_event(t_master_us(), TRIG_STAMP_TIMER, false);
        return true;
    }

    /* PIO-IRQ mode: watchdog. Give the first pulse a couple of periods. */
    uint32_t n = ++ticks_since_start;
    if (n < TRIG_FALLBACK_SILENT_PERIODS + 1u) {
        return true;
    }
    /* The PIO IRQ outranks this timer callback: mask it while deciding and
     * back-filling so the two cannot interleave trigger_id updates. */
    uint32_t irq = save_and_disable_interrupts();
    uint64_t now = t_master_us();
    if (!fallback_active) {
        uint64_t last = last_pio_event_us;
        bool silent = (last == 0) ||
                      ((now - last) > (uint64_t)TRIG_FALLBACK_SILENT_PERIODS * active_period_us);
        if (!silent) {
            restore_interrupts(irq);
            return true;
        }
        /*
         * Take over. Back-fill the ids the PIO should have stamped by now so the
         * host's sequence-consume association still sees pulse k as id k; their
         * times are reconstructed at one period spacing, which is what the
         * legacy timer would have produced anyway.
         */
        fallback_active = true;
        uint64_t expected = (uint64_t)n - 1u; /* pulses before this tick */
        while (trigger_id < expected) {
            uint64_t k_back = expected - trigger_id; /* 1 = most recent missed */
            uint64_t t_back = now - k_back * (uint64_t)active_period_us;
            timer_event_count++;
            emit_event(t_back, TRIG_STAMP_TIMER, true);
        }
    }
    timer_event_count++;
    emit_event(now, TRIG_STAMP_TIMER, true);
    restore_interrupts(irq);
    return true;
}

/**
 * @brief Internal helper: stop the SM and hold the pin at its idle level.
 *
 * @details The idle level is the INACTIVE level of the polarity
 *          (trigger_timing_idle_high): HIGH for imx296_fast, whose exposure is
 *          the LOW phase (h06 §3.1; buffer non-inverting, divider only
 *          level-shifts, the camera pulls XTR up itself). The line used to idle
 *          LOW for both polarities, i.e. at the camera's *exposing* level, and a
 *          trig_stop during the HIGH phase added a stray falling edge that
 *          started an exposure no rising edge ended. Stopped during the LOW
 *          (exposure) phase, parking HIGH now ends that exposure cleanly: one
 *          short frame whose TRIGGER_EVENT already exists.
 *
 *          The SIO output value and direction are loaded before the pad is
 *          switched from PIO to SIO, so the handover makes no edge (gpio_init()
 *          here used to float the pin and then drive it LOW).
 *
 * @note PIN_TRIG also drives the TRIG LED (active high): for imx296_fast it is
 *       lit while idle and blinks dark with the exposures.
 */
static void trigger_pin_park_idle(void) {
    if (pio_claimed) {
        pio_sm_set_enabled(trig_pio, trig_sm, false);
    }
    gpio_put(PIN_TRIG, trigger_timing_idle_high(idle_fast));
    gpio_set_dir(PIN_TRIG, GPIO_OUT);
    gpio_set_function(PIN_TRIG, GPIO_FUNC_SIO);
}

void trigger_pio_set_idle_polarity(bool imx296_fast) {
    idle_fast = imx296_fast;
    if (!running) {
        trigger_pin_park_idle();
    }
}

bool trigger_pio_init(uint32_t clk_hz) {
    clock_hz = clk_hz ? clk_hz : 1000000u;
    /* Cancel any live tick timer so re-init never orphans a firing timer. */
    tick_timer_cancel();
    running = false;
    trigger_id = 0;

    trigger_pin_park_idle();

    /* Claim the PIO SM once; re-init reuses it instead of leaking SMs. */
    if (!pio_claimed) {
        bool ok = pio_claim_free_sm_and_add_program_for_gpio_range(
            &trigger_gen_program, &trig_pio, &trig_sm, &trig_offset, PIN_TRIG, 1, true);
        if (!ok) {
            return false;
        }
        pio_claimed = true;
    }
    trigger_gen_program_init(trig_pio, trig_sm, trig_offset, PIN_TRIG, (float)clock_hz,
                             trigger_timing_idle_high(idle_fast));

    /*
     * Route PIO IRQ flag 0 of this block to its IRQ0 line and take that line on
     * this (comms) core. Installed once: irq_set_exclusive_handler() asserts if
     * a handler is already present, and nothing else in the firmware claims a
     * PIO IRQ (ws2812 is FIFO-only, led_flash is timer-driven).
     */
    if (!pio_irq_installed) {
        int irq_num = pio_get_irq_num(trig_pio, 0);
        pio_interrupt_clear(trig_pio, TRIG_PIO_IRQ_FLAG);
        pio_set_irq0_source_enabled(
            trig_pio, (pio_interrupt_source_t)(pis_interrupt0 + TRIG_PIO_IRQ_FLAG), true);
        irq_set_exclusive_handler((uint)irq_num, trigger_pio_irq_handler);
        pio_irq_installed = true;
        trigger_pio_set_irq_priority(irq_level);
        irq_set_enabled((uint)irq_num, true);
    }

    /* Program loaded; keep the pin at its idle level until trig_start. */
    trigger_pin_park_idle();
    return true;
}

void trigger_pio_set_irq_priority(uint8_t level) {
    irq_level = level > 3u ? 3u : level;
    if (pio_irq_installed) {
        /* 0x00..0x60: always above the SDK default (0x80) that USB, DMA, GPIO
         * and the timer alarms run at, so edge stamping preempts all of them. */
        irq_set_priority((uint)pio_get_irq_num(trig_pio, 0), (uint8_t)(irq_level << 5));
    }
}

bool trigger_pio_start(uint16_t rate_hz, uint32_t pulse_width_us, bool imx296_fast) {
    trigger_timing_t tm;
    if (!pio_claimed || !trigger_timing_compute(rate_hz, pulse_width_us, imx296_fast,
                                                comp_enabled, &tm)) {
        return false;
    }

    /* Stop everything before touching shared state. */
    tick_timer_cancel();
    running = false;
    __dmb();
    pio_sm_set_enabled(trig_pio, trig_sm, false);

    /*
     * Re-init the SM so the program restarts at its `pull` instructions. Without
     * this a stop/start resumed inside the wrap loop and never consumed the two
     * new FIFO words, i.e. a changed rate/width (or compensation setting) was
     * silently ignored until the next `config apply`.
     */
    /* A polarity change re-parks at the new idle level first (one edge while
     * stopped), so the handover to the SM below is edge-free. */
    if (imx296_fast != idle_fast) {
        idle_fast = imx296_fast;
        trigger_pin_park_idle();
    }
    trigger_gen_program_init(trig_pio, trig_sm, trig_offset, PIN_TRIG, (float)clock_hz,
                             trigger_timing_idle_high(idle_fast));
    pio_sm_clear_fifos(trig_pio, trig_sm);
    pio_interrupt_clear(trig_pio, TRIG_PIO_IRQ_FLAG);

    active_mode = stamp_mode;
    active_fast = imx296_fast;
    active_period_us = tm.period_us;
    active_pin_period_us = tm.pin_period_us;
    trigger_id = 0;
    last_pio_event_us = 0;
    edge_track = (trigger_edge_track_t){0};
    pio_pulse_count = 0;
    timer_event_count = 0;
    pio_irq_spurious = 0;
    fallback_active = false;
    ticks_since_start = 0;
    lost_edges = 0;

    pio_sm_put_blocking(trig_pio, trig_sm, tm.low_ticks);
    pio_sm_put_blocking(trig_pio, trig_sm, tm.high_ticks);

    running = true;
    __dmb();
    pio_sm_set_enabled(trig_pio, trig_sm, true);

    /* Exactly one tick timer at all times: stamper (TIMER mode) or watchdog.
     * Without it there is no fallback stamping (and no events at all in TIMER
     * mode), so failing to get an alarm slot fails the start. */
    if (!add_repeating_timer_us(-(int64_t)tm.period_us, tick_cb, NULL, &tick_rt)) {
        trigger_pio_stop();
        return false;
    }
    timer_active = true;
    return true;
}

void trigger_pio_stop(void) {
    running = false;
    __dmb();
    tick_timer_cancel();
    trigger_pin_park_idle();
    active_period_us = 0;
    active_pin_period_us = 0;
}

bool trigger_pio_running(void) {
    return running;
}

uint64_t trigger_pio_id(void) {
    return trigger_id;
}

void trigger_pio_set_stamp_mode(trigger_stamp_mode_t mode) {
    stamp_mode = (mode == TRIG_STAMP_MODE_TIMER) ? TRIG_STAMP_MODE_TIMER : TRIG_STAMP_MODE_PIO_IRQ;
}

trigger_stamp_mode_t trigger_pio_stamp_mode(void) {
    return stamp_mode;
}

trigger_stamp_mode_t trigger_pio_active_stamp_mode(void) {
    return active_mode;
}

void trigger_pio_set_compensation(bool on) {
    comp_enabled = on;
}

bool trigger_pio_compensation(void) {
    return comp_enabled;
}

uint32_t trigger_pio_pulse_count(void) {
    return pio_pulse_count;
}

uint32_t trigger_pio_timer_event_count(void) {
    return timer_event_count;
}

bool trigger_pio_fallback_active(void) {
    return fallback_active;
}

uint32_t trigger_pio_spurious_irq_count(void) {
    return pio_irq_spurious;
}

uint32_t trigger_pio_lost_edge_count(void) {
    return lost_edges;
}

uint32_t trigger_pio_active_period_us(void) {
    return active_period_us;
}

uint32_t trigger_pio_active_pin_period_us(void) {
    return active_pin_period_us;
}

bool trigger_pio_active_imx296_fast(void) {
    return active_fast;
}

/** @brief Internal helper: mode_str. */
static const char *mode_str(trigger_stamp_mode_t m) {
    return (m == TRIG_STAMP_MODE_TIMER) ? "timer" : "pio_irq";
}

void trigger_pio_print_status(void) {
    printf("trig running=%d id=%llu\r\n", running ? 1 : 0, (unsigned long long)trigger_id);
    printf("  stamp: configured=%s active=%s fallback=%s\r\n", mode_str(stamp_mode),
           mode_str(active_mode), fallback_active ? "YES (timer took over)" : "no");
    printf("  comp=%s period_us=%lu pin_period_us=%lu edge=%s\r\n", comp_enabled ? "on" : "off",
           (unsigned long)active_period_us, (unsigned long)active_pin_period_us,
           active_fast ? "exposure_start(low)" : "exposure_end(low)");
    printf("  pio_pulses=%lu timer_events=%lu spurious_irq=%lu lost_edges=%lu irq_level=%u\r\n",
           (unsigned long)pio_pulse_count, (unsigned long)timer_event_count,
           (unsigned long)pio_irq_spurious, (unsigned long)lost_edges, (unsigned)irq_level);
    if (running && active_mode == TRIG_STAMP_MODE_PIO_IRQ && !fallback_active &&
        pio_pulse_count == 0 && ticks_since_start > TRIG_FALLBACK_SILENT_PERIODS) {
        printf("  WARNING: no PIO pulses seen yet\r\n");
    }
}

void trigger_pio_isr_install(void) {
    /* The PIO IRQ is installed in trigger_pio_init(); nothing to do here. */
}
