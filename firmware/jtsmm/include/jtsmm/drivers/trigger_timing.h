// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file trigger_timing.h
 * @brief Pure trigger period/template arithmetic (host-testable, no Pico deps).
 *
 * @details Kept separate from trigger_pio.c so the cycle-overhead compensation
 *          can be unit-tested on the host (tests/test_trigger_timing.c). The
 *          overhead constants MUST match pio/trigger_gen.pio.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/** Highest supported trigger rate (Hz); also the schema maximum. */
#define TRIGGER_MAX_RATE_HZ 200u

/** Cycles per phase spent outside the `jmp y--` countdown (see trigger_gen.pio). */
#define TRIG_PIO_HIGH_OVERHEAD_CYCLES 2u
#define TRIG_PIO_LOW_OVERHEAD_CYCLES  3u

/**
 * @brief Resolved pulse timing for one @c trig_start.
 */
typedef struct {
    uint32_t period_us;   /**< 1e6 / rate_hz (integer). */
    uint32_t low_us;      /**< Requested low-phase duration on the pin. */
    uint32_t high_us;     /**< Requested high-phase duration on the pin. */
    uint32_t low_ticks;   /**< Value pushed to the PIO for the low phase. */
    uint32_t high_ticks;  /**< Value pushed to the PIO for the high phase. */
    uint32_t pin_period_us; /**< Period actually produced on the pin with these ticks. */
} trigger_timing_t;

/**
 * @brief Validate a requested rate / pulse width (config apply and `trig_start`).
 *
 * @param rate_hz  Requested rate; taken as 32 bit so an out-of-range command
 *                 argument cannot wrap into range through a uint16 cast.
 * @param width_us Requested pulse width (the exposure for imx296_fast).
 * @return NULL if usable, else a short reason (rate outside 1..200 Hz, width 0,
 *         or width >= period -- which used to be halved silently).
 */
const char *trigger_timing_check(uint32_t rate_hz, uint32_t width_us);

/**
 * @brief Level the trigger line rests at between runs: the INACTIVE level.
 *
 * @details imx296_fast: the IMX296 integrates while XTRIG is LOW (h06 §3.1;
 *          the 74LVC2G241 fan-out is non-inverting and the divider only
 *          level-shifts), so the line idles HIGH -- as the camera's own pull-up
 *          holds XTR when unplugged. Other polarity (exposure = HIGH phase):
 *          idles LOW. Idling at the exposing level kept the sensor integrating
 *          between runs, and a trig_stop mid-HIGH-phase added a stray falling
 *          edge that started an exposure no rising edge ended.
 * @param imx296_fast Configured/active polarity (pulse.polarity == 0).
 * @return true = HIGH.
 */
bool trigger_timing_idle_high(bool imx296_fast);

/**
 * @brief Edge-schedule tracker for the PIO trigger IRQ (contract K4).
 *
 * @details @c ref_us estimates when the edge of the last serviced IRQ actually
 *          happened. It is re-anchored to the IRQ time when that IRQ was prompt
 *          and otherwise advanced on the pin schedule, so one late IRQ does not
 *          shift the reference for the next one (same scheme as
 *          ads_drdy_track() for the ADS1299 DRDY line).
 */
typedef struct {
    uint64_t ref_us; /**< Estimated time of the last counted edge. */
    bool valid;      /**< false until the first IRQ of a run. */
} trigger_edge_track_t;

/**
 * @brief Pulse edges represented by one PIO IRQ (contract K4).
 *
 * @details The PIO raises one IRQ flag per exposure-start edge; if the comms
 *          core cannot service it within a period, further edges set the same
 *          flag and merge into one IRQ. Returns
 *          k = max(1, floor((t - ref + T/16) / T)), the edges since the last
 *          counted one, measured against the edge *schedule* rather than the
 *          previous (possibly late) stamp: counting from the previous stamp
 *          turned a single IRQ served >= 3/4 T late into two edges and shifted
 *          every later trigger_id by one. The T/16 tolerance absorbs IRQ latency
 *          and oscillator drift; an IRQ before the predicted edge is always the
 *          next edge (a flag cannot be raised before its edge). Afterwards
 *          @c ref_us = t if t < ref + k*T + T/16 (prompt), else ref + k*T.
 * @param s         Tracker state (zero-init at trig_start).
 * @param t_us      Time the IRQ was serviced.
 * @param period_us Pin period (µs); 0 → always 1 and ref = t.
 * @return Edges since the last counted edge (>= 1); @c s->ref_us is then the
 *         best estimate of this edge's time (use it as the stamp).
 */
uint32_t trigger_timing_edge_track(trigger_edge_track_t *s, uint64_t t_us, uint32_t period_us);

/**
 * @brief Compute phase durations and PIO templates.
 *
 * @param rate_hz         Pulse rate; 0 → false.
 * @param pulse_width_us  Requested pulse width (0 → period/2; >= period → false).
 * @param imx296_fast     true → exposure on the LOW phase (width = low time).
 * @param compensate      true → subtract the PIO instruction overhead so the pin
 *                        period equals @c period_us exactly; false → legacy
 *                        templates (pin period = period_us + 5 cycles).
 * @param out             Filled on success; must not be NULL.
 * @return false on invalid input.
 */
bool trigger_timing_compute(uint16_t rate_hz, uint32_t pulse_width_us, bool imx296_fast,
                            bool compensate, trigger_timing_t *out);
