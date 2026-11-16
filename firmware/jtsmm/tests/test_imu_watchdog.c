// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* IMU stall watchdog decisions (include/jtsmm/app/imu_watchdog.h). */
#include "jtsmm/app/imu_watchdog.h"

#include "check.h"

/* Run `secs` seconds of a stalled, streaming slot; count the actions. */
static void run(imu_wd_state_t *s, unsigned secs, int *recover, int *give_up) {
    for (unsigned i = 0; i < secs; i++) {
        imu_wd_action_t a = imu_watchdog_tick(s, true, 0u, 0u);
        if (a == IMU_WD_RECOVER) {
            (*recover)++;
        } else if (a == IMU_WD_GIVE_UP) {
            (*give_up)++;
        }
    }
}

/* Trip at 3 s, retry every 10 s, 6 attempts, then give up every 10 s. */
static void test_schedule(void) {
    imu_wd_state_t s = {0};
    int rec = 0, up = 0;
    run(&s, 2, &rec, &up);
    CHECK_EQ_U(rec, 0);
    run(&s, 1, &rec, &up); /* 3 s */
    CHECK_EQ_U(rec, 1);
    run(&s, 50, &rec, &up); /* 53 s: attempts at 13, 23, 33, 43, 53 */
    CHECK_EQ_U(rec, 6);
    CHECK_EQ_U(up, 0);
    run(&s, 30, &rec, &up); /* 63, 73, 83 */
    CHECK_EQ_U(rec, 6);
    CHECK_EQ_U(up, 3);
}

/* Bench 2026-09-28: a stopped stream must restore the attempt budget. */
static void test_stopped_stream_resets_attempts(void) {
    imu_wd_state_t s = {0};
    int rec = 0, up = 0;
    run(&s, 70, &rec, &up); /* first take: all 6 attempts used */
    CHECK_EQ_U(rec, 6);
    CHECK(imu_watchdog_tick(&s, false, 0u, 0u) == IMU_WD_NONE); /* stream stopped */
    rec = up = 0;
    run(&s, 3, &rec, &up); /* next take, still no IMU */
    CHECK_EQ_U(rec, 1);    /* a fresh recovery attempt, not an immediate give-up */
    CHECK_EQ_U(up, 0);
}

/* Data flowing resets the stall timer and the budget. */
static void test_pushes_reset(void) {
    imu_wd_state_t s = {0};
    int rec = 0, up = 0;
    run(&s, 13, &rec, &up);
    CHECK_EQ_U(rec, 2);
    CHECK(imu_watchdog_tick(&s, true, 5u, 0u) == IMU_WD_NONE); /* the slot pushed */
    CHECK_EQ_U(s.attempts, 0);
    CHECK_EQ_U(s.stall_s, 0);
    CHECK(imu_watchdog_tick(&s, true, 5u, 0u) == IMU_WD_NONE); /* 1 s stalled again */
    CHECK_EQ_U(s.stall_s, 1);
}

/* Bench 2026-11-04: streams started from a serial terminal, nobody reads the data
 * endpoint, the ring fills; pushes stop but drops rise. Report once, never recover. */
static void test_full_ring_is_not_a_stall(void) {
    imu_wd_state_t s = {0};
    uint32_t drops = 0;
    int full = 0, rec = 0;
    for (unsigned i = 0; i < 60; i++) {
        drops += 400u; /* ~400 Hz of reports that no longer fit */
        imu_wd_action_t a = imu_watchdog_tick(&s, true, 0u, drops);
        full += (a == IMU_WD_HOST_FULL);
        rec += (a == IMU_WD_RECOVER || a == IMU_WD_GIVE_UP);
    }
    CHECK_EQ_U(full, 1);
    CHECK_EQ_U(rec, 0);
    CHECK_EQ_U(s.stall_s, 0);
    /* the host reads again: pushes resume and the episode ends */
    CHECK(imu_watchdog_tick(&s, true, 10u, drops) == IMU_WD_NONE);
    CHECK(!s.host_full);
    /* a real stall afterwards (no pushes, no drops) still recovers at 3 s */
    int r2 = 0, up = 0;
    for (unsigned i = 0; i < 3; i++) {
        imu_wd_action_t a = imu_watchdog_tick(&s, true, 10u, drops);
        r2 += (a == IMU_WD_RECOVER);
        up += (a == IMU_WD_GIVE_UP);
    }
    CHECK_EQ_U(r2, 1);
    CHECK_EQ_U(up, 0);
    /* a second full episode is reported again */
    CHECK(imu_watchdog_tick(&s, true, 11u, drops) == IMU_WD_NONE);
    CHECK(imu_watchdog_tick(&s, true, 11u, drops + 400u) == IMU_WD_HOST_FULL);
}

int main(void) {
    test_full_ring_is_not_a_stall();
    test_schedule();
    test_stopped_stream_resets_attempts();
    test_pushes_reset();
    CHECK_DONE("test_imu_watchdog");
}
