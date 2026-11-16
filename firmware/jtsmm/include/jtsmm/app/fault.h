// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file fault.h
 * @brief Fault LED / SOS indicator, hard-fault handler, optional hardware watchdog.
 *
 * @details The hardware watchdog is compile-time selectable and OFF by default
 *          (it has not been validated on the PCB yet). Enable it with
 *          `cmake -DJTSMM_WATCHDOG_MS=4000 ...`: the comms loop then kicks it
 *          only while core 1 also makes progress, long operations (config
 *          apply, flash write, selftest) stretch it to 15 s, and a hang or a
 *          hard fault (SOS on LED_FAULT) resets the chip; the next boot logs
 *          "last reset = watchdog timeout". The HardFault SOS handler is always
 *          installed.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/** @brief Initialize fault GPIO / state. */
void fault_handler_init(void);

/**
 * @brief Blink three SOS cycles on the fault LED, then leave it on.
 *
 * @details For unrecoverable, but not hard-fault, conditions (the HardFault
 *          handler blinks SOS forever by itself).
 */
void fault_handler_sos(void);

/** @brief Start the watchdog (no-op unless built with JTSMM_WATCHDOG_MS > 0). */
void fault_watchdog_start(void);

/**
 * @brief Comms-loop kick, gated on core-1 progress.
 * @param core1_ticks    Sensor-core poll counter.
 * @param core1_expected true if core 1 should be running (not parked/stopped).
 */
void fault_watchdog_service(uint32_t core1_ticks, bool core1_expected);

/** @brief Stretch the timeout around a known long operation (nestable). */
void fault_watchdog_hold(void);

/** @brief End a @ref fault_watchdog_hold. */
void fault_watchdog_release(void);

/** @brief Human-readable reason for the last reset. */
const char *fault_reset_reason(void);

/** @brief Print (CDC) and, after a watchdog reset, log (REC_LOG) the reset reason. */
void fault_report_boot_reason(void);
