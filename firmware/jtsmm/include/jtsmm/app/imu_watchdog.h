// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_watchdog.h
 * @brief IMU stall watchdog decision (pure, unit-tested; comms_core logs and acts).
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define IMU_WATCHDOG_STALL_SECONDS 3u
/* Seconds between recovery attempts while a slot stays stalled, and how many
 * attempts before we stop poking the bus (the log keeps saying it is dead). */
#define IMU_WATCHDOG_RETRY_SECONDS 10u
#define IMU_WATCHDOG_MAX_ATTEMPTS  6u

typedef enum {
    IMU_WD_NONE = 0, /**< nothing to do this second */
    IMU_WD_RECOVER,  /**< start recovery attempt @c attempts (1-based) */
    IMU_WD_GIVE_UP,  /**< out of attempts: log that the slot stays dead */
    IMU_WD_HOST_FULL, /**< ring full, the host is not reading: log once, no recovery */
} imu_wd_action_t;

/** Per-slot watchdog state, zero-initialised. */
typedef struct {
    uint32_t last_pushes;
    uint32_t last_drops;
    uint16_t stall_s;
    uint8_t attempts;
    bool host_full; /**< HOST_FULL reported for the current episode */
} imu_wd_state_t;

/**
 * @brief One second of a slot's stall watchdog.
 *
 * @details Trips at STALL_SECONDS without a new ring push, retries every
 *          RETRY_SECONDS up to MAX_ATTEMPTS, then keeps reporting the give-up.
 *          The attempt budget resets whenever the slot pushes again AND
 *          whenever its stream is off: it used to survive a stopped stream,
 *          so after one take with a silent IMU every later stream gave up
 *          after 3 s without a single recovery attempt (bench, 2026-09-28).
 *          No push but new drops means the sensor still delivers and the ring is
 *          full because the host does not read the data endpoint (e.g. streams
 *          started from a serial terminal): that is reported once per episode as
 *          HOST_FULL and never triggers a recovery, which used to reset healthy
 *          IMUs (bench, 2026-11-04).
 * @param streaming Host wants the stream on (cfg->stream_imu_*).
 * @param pushes    ringbuf_pushes() of the slot's ring.
 * @param drops     ringbuf_drops() of the slot's ring (failed pushes, ring full).
 */
static inline imu_wd_action_t imu_watchdog_tick(imu_wd_state_t *s, bool streaming,
                                                uint32_t pushes, uint32_t drops) {
    if (!streaming || pushes != s->last_pushes) {
        s->stall_s = 0;
        s->attempts = 0;
        s->last_pushes = pushes;
        s->last_drops = drops;
        s->host_full = false;
        return IMU_WD_NONE;
    }
    if (drops != s->last_drops) {
        /* Producer alive, consumer missing: not a stalled sensor. */
        s->last_drops = drops;
        s->stall_s = 0;
        s->attempts = 0;
        if (!s->host_full) {
            s->host_full = true;
            return IMU_WD_HOST_FULL;
        }
        return IMU_WD_NONE;
    }
    if (s->stall_s < 0xFFFFu) {
        s->stall_s++;
    }
    bool due = (s->stall_s == IMU_WATCHDOG_STALL_SECONDS) ||
               (s->stall_s > IMU_WATCHDOG_STALL_SECONDS &&
                ((s->stall_s - IMU_WATCHDOG_STALL_SECONDS) % IMU_WATCHDOG_RETRY_SECONDS) == 0u);
    if (!due) {
        return IMU_WD_NONE;
    }
    if (s->attempts >= IMU_WATCHDOG_MAX_ATTEMPTS) {
        return IMU_WD_GIVE_UP;
    }
    s->attempts++;
    return IMU_WD_RECOVER;
}
