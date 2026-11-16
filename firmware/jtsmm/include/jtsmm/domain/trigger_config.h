// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file trigger_config.h
 * @brief Domain layer: apply trigger config and emit @c REC_TRIGGER_EVENT.
 */
#pragma once

#include "jtsmm/config.h"
#include <stdbool.h>

/**
 * @brief Apply @p cfg to PIO trigger / LED outputs.
 * @param cfg Trigger config; must not be NULL.
 * @return true on success.
 */
bool trigger_config_apply(const trigger_config_t *cfg);

/**
 * @brief Validate @p cfg without touching hardware.
 * @param cfg Trigger config; must not be NULL.
 * @return true if rate is 1..200 Hz and 0 < pulse width < period
 *         (see @c trigger_timing_check for the reason when false).
 */
bool trigger_config_validate(const trigger_config_t *cfg);

/**
 * @brief Expand @p cfg->profile into concrete fields when not CUSTOM.
 * @param cfg In/out config; must not be NULL.
 */
void trigger_config_resolve_profile(trigger_config_t *cfg);

/**
 * @brief PIO/timer callback: push a @c REC_TRIGGER_EVENT into @c RING_EVENTS.
 * @param t_us       Master timestamp (µs).
 * @param trigger_id Monotonic trigger index from the trigger driver.
 * @param stamp_src  @c TRIG_STAMP_PIO_IRQ or @c TRIG_STAMP_TIMER (records.h);
 *                   written to payload[0] so the host can tell how the event
 *                   was timed.
 * @param fallback   true if the timer took over from a silent PIO IRQ
 *                   (payload[1] = 1).
 */
void trigger_on_pio_event(uint64_t t_us, uint64_t trigger_id, uint8_t stamp_src, bool fallback);
