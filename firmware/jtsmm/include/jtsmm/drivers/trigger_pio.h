// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file trigger_pio.h
 * @brief PIO `trigger_gen` driver for camera XTRIG pulses + trigger_id.
 *
 * @details Emits pulses on @c PIN_TRIG. By default @c TRIGGER_EVENT is stamped
 *          from a PIO IRQ raised on the exposure-start edge of every pulse
 *          (@ref TRIG_STAMP_MODE_PIO_IRQ); a software @c repeating_timer at the
 *          same period runs as a watchdog and takes over stamping — loudly — if
 *          the PIO IRQ ever goes silent. @ref TRIG_STAMP_MODE_TIMER restores the
 *          legacy timer-only stamping for A/B comparison on the bench.
 *          See docs/hardware/h06_synchronization.md §3.1.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/** @brief Where TRIGGER_EVENT timestamps come from. */
typedef enum {
    TRIG_STAMP_MODE_PIO_IRQ = 0, /**< Latched in the PIO IRQ at the pulse edge (default). */
    TRIG_STAMP_MODE_TIMER = 1,   /**< Legacy: free-running software timer at period_us. */
} trigger_stamp_mode_t;

/**
 * @brief Load PIO program, install the PIO IRQ, and idle @c PIN_TRIG at the
 *        inactive level of the idle polarity (see trigger_pio_set_idle_polarity).
 * @param clock_hz PIO clock (Hz); typically 1e6 so ticks == microseconds.
 * @return true if a PIO SM was claimed/initialized.
 */
bool trigger_pio_init(uint32_t clock_hz);

/**
 * @brief Polarity whose inactive level @c PIN_TRIG idles at (HIGH for
 *        imx296_fast, LOW for active_high; trigger_timing_idle_high()).
 *
 * @details Set from the applied config before any trig_start; every start also
 *          adopts its own polarity. Re-parks the pin at once when stopped.
 *          Default before the first call: imx296_fast (factory polarity).
 */
void trigger_pio_set_idle_polarity(bool imx296_fast);

/**
 * @brief Start periodic trigger pulses (resets trigger_id to 0 for this run).
 * @param rate_hz         Pulse rate (Hz); must be > 0.
 * @param pulse_width_us  Pulse width (µs); for @p imx296_fast this is low-time.
 * @param imx296_fast     true → IMX296 fast-trigger polarity (exposure on low).
 * @return false if the timing is invalid (see @c trigger_timing_check), no alarm
 *         slot is free for the watchdog timer, or the PIO is not initialised.
 */
bool trigger_pio_start(uint16_t rate_hz, uint32_t pulse_width_us, bool imx296_fast);

/** @brief Stop pulses and drive the pin to its idle (inactive) level. */
void trigger_pio_stop(void);

/**
 * @brief NVIC priority of the trigger PIO IRQ (platform.irq.trigger).
 * @param level 0 (highest) .. 3; mapped to 0x00..0x60, i.e. always above the
 *              SDK default priority every other IRQ in this firmware uses.
 */
void trigger_pio_set_irq_priority(uint8_t level);

/** @brief true while the generator is running. */
bool trigger_pio_running(void);

/** @brief Current monotonic trigger_id (last emitted / in progress). */
uint64_t trigger_pio_id(void);

/**
 * @{
 * @name Stamp-mode and compensation controls (CDC `trig_stamp` / `trig_comp`)
 * @details Both take effect at the next @ref trigger_pio_start and persist until
 *          power-cycle. Defaults: PIO-IRQ stamping, compensation on.
 */
void trigger_pio_set_stamp_mode(trigger_stamp_mode_t mode);
trigger_stamp_mode_t trigger_pio_stamp_mode(void);
/** @brief Mode latched at the current/last start (what the events actually used). */
trigger_stamp_mode_t trigger_pio_active_stamp_mode(void);
void trigger_pio_set_compensation(bool on);
bool trigger_pio_compensation(void);
/** @} */

/**
 * @{
 * @name Diagnostics (per run; reset at start)
 */
/** @brief Pulses seen by the PIO IRQ (counted in every mode). */
uint32_t trigger_pio_pulse_count(void);
/** @brief Events stamped by the software timer (mode TIMER, or fallback). */
uint32_t trigger_pio_timer_event_count(void);
/** @brief true once the timer watchdog has taken over stamping this run. */
bool trigger_pio_fallback_active(void);
/** @brief PIO IRQ entries with no flag set (should stay 0). */
uint32_t trigger_pio_spurious_irq_count(void);
/**
 * @brief Pulses whose IRQ merged into a later one (IRQ latency > one period).
 * @details Each advanced trigger_id by the number of merged edges (contract K4),
 *          so the host sees these as id gaps.
 */
uint32_t trigger_pio_lost_edge_count(void);
/** @brief Period the templates were computed for, µs (0 when stopped). */
uint32_t trigger_pio_active_period_us(void);
/** @brief Period the pin actually produces with the active templates, µs. */
uint32_t trigger_pio_active_pin_period_us(void);
/** @brief true if the active run uses IMX296 fast polarity (IRQ edge = exposure start). */
bool trigger_pio_active_imx296_fast(void);
/** @} */

/** @brief Print a one-screen status block on CDC (`trig_status`). */
void trigger_pio_print_status(void);

/**
 * @brief Historical hook kept for API compatibility; the PIO IRQ is installed
 *        by @ref trigger_pio_init.
 */
void trigger_pio_isr_install(void);
