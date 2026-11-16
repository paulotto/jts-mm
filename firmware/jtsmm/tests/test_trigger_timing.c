// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#include "jtsmm/drivers/trigger_timing.h"

#include "check.h"

#include <stdio.h>

/* K4: edges represented by one PIO IRQ, counted against the edge schedule. */
static void test_edge_track(void) {
    const uint32_t P = 33333;
    const uint64_t E0 = 1000000u; /* first edge */
    trigger_edge_track_t s = {0};

    /* First IRQ of a run: one edge, anchored where it was served. */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 3, P), 1);
    CHECK(s.ref_us == E0 + 3);
    /* Prompt IRQs with normal latency jitter either way. */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + P + 40, P), 1);
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 2 * P + 1, P), 1);
    CHECK(s.ref_us == E0 + 2 * P + 1); /* prompt: re-anchored on the IRQ */

    /* Single IRQ served 0.9 T late: still one edge (counting from the previous
     * stamp said 2 -> trigger_id permanently +1). The stamp is the scheduled
     * edge, not the late service time. The next prompt IRQ is one edge again. */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 3 * P + (9 * P) / 10, P), 1);
    CHECK(s.ref_us == E0 + 3 * P + 1);
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 4 * P + 5, P), 1);
    CHECK(s.ref_us == E0 + 4 * P + 5);

    /* Half-period late, then the next one prompt. */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 5 * P + P / 2, P), 1);
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 6 * P + 2, P), 1);

    /* True lost edge: the edge at 7P merged into the IRQ at 8P (2 T gap, prompt). */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 8 * P + 4, P), 2);
    CHECK(s.ref_us == E0 + 8 * P + 4);
    /* Merged AND late: edges 9P, 10P merged into an IRQ served at 11P - 0.2T
     * (i.e. edge 10P served 0.8 T late) -> 2 edges, stamp on the schedule. */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 11 * P - P / 5, P), 2);
    CHECK(s.ref_us == E0 + 10 * P + 4);
    /* Four edges merged. */
    CHECK_EQ_U(trigger_timing_edge_track(&s, E0 + 14 * P + 10, P), 4);

    /* Drift +-0.1 % between the pin and the tracker's period, 1000 edges with
     * IRQ latency up to 0.8 T every 7th edge: never a phantom edge. */
    for (int sign = -1; sign <= 1; sign += 2) {
        trigger_edge_track_t d = {0};
        double true_p = P * (1.0 + sign * 0.001);
        uint64_t total = 0;
        for (int i = 0; i < 1000; i++) {
            uint64_t edge = E0 + (uint64_t)(i * true_p);
            uint64_t lat = (i % 7 == 3) ? (uint64_t)(0.8 * P) : (uint64_t)(i % 5);
            total += trigger_timing_edge_track(&d, edge + lat, P);
        }
        CHECK_EQ_U(total, 1000);
    }

    /* Period 0 (not started): always one edge, anchored on the IRQ. */
    trigger_edge_track_t z = {0};
    CHECK_EQ_U(trigger_timing_edge_track(&z, 1000, 0), 1);
    CHECK_EQ_U(trigger_timing_edge_track(&z, 900000, 0), 1);
}

int main(void) {
    trigger_timing_t tm;

    /* 60 Hz IMX296 fast, 4 ms exposure, compensated: the pin period must be
     * exactly the integer period the software timer also runs at. */
    CHECK(trigger_timing_compute(60, 4000, true, true, &tm));
    CHECK(tm.period_us == 16666);
    CHECK(tm.low_us == 4000);
    CHECK(tm.high_us == 12666);
    CHECK(tm.low_ticks == 4000 - TRIG_PIO_LOW_OVERHEAD_CYCLES);
    CHECK(tm.high_ticks == 12666 - TRIG_PIO_HIGH_OVERHEAD_CYCLES);
    CHECK(tm.pin_period_us == tm.period_us);

    /* 30 Hz / 12 ms: the rig's operating point and the firmware default. */
    CHECK(trigger_timing_compute(30, 12000, true, true, &tm));
    CHECK_EQ_U(tm.period_us, 33333);
    CHECK_EQ_U(tm.low_us, 12000);
    CHECK_EQ_U(tm.high_us, 21333);
    CHECK_EQ_U(tm.low_ticks, 12000 - TRIG_PIO_LOW_OVERHEAD_CYCLES);
    CHECK_EQ_U(tm.high_ticks, 21333 - TRIG_PIO_HIGH_OVERHEAD_CYCLES);
    CHECK_EQ_U(tm.pin_period_us, tm.period_us);

    /* Legacy (uncompensated) templates: pin period is 5 cycles long. This is the
     * drift the compensation exists to remove; pin it so nobody "simplifies" the
     * .pio without updating the overhead constants. */
    CHECK(trigger_timing_compute(60, 4000, true, false, &tm));
    CHECK(tm.low_ticks == 4000);
    CHECK(tm.high_ticks == 12666);
    CHECK(tm.pin_period_us == 16666 + TRIG_PIO_LOW_OVERHEAD_CYCLES + TRIG_PIO_HIGH_OVERHEAD_CYCLES);

    /* Non-fast polarity: width is the HIGH phase. */
    CHECK(trigger_timing_compute(100, 2000, false, true, &tm));
    CHECK(tm.period_us == 10000);
    CHECK(tm.high_us == 2000);
    CHECK(tm.low_us == 8000);
    CHECK(tm.pin_period_us == 10000);

    /* Width 0 keeps meaning "half period" for internal callers; a width that
     * does not fit the period is refused (it used to be halved silently). */
    CHECK(trigger_timing_compute(10, 0, true, true, &tm));
    CHECK(tm.low_us == 50000 && tm.high_us == 50000);
    CHECK(!trigger_timing_compute(10, 500000, true, true, &tm));
    CHECK(!trigger_timing_compute(10, 100000, true, true, &tm)); /* == period */
    CHECK(trigger_timing_compute(10, 99999, true, true, &tm));

    /* Degenerate tiny phases never underflow the template. */
    CHECK(trigger_timing_compute(200, 1, true, true, &tm));
    CHECK(tm.low_ticks >= 1 && tm.high_ticks >= 1);

    /* Invalid input. */
    CHECK(!trigger_timing_compute(0, 4000, true, true, &tm));
    CHECK(!trigger_timing_compute(60, 4000, true, true, NULL));

    /* Request validation (config apply and trig_start). Takes 32-bit values so
     * `trig_start 65566` cannot wrap to 30 Hz through a uint16 cast. */
    CHECK(trigger_timing_check(30, 12000) == NULL);
    CHECK(trigger_timing_check(200, 4999) == NULL);
    CHECK(trigger_timing_check(0, 1000) != NULL);
    CHECK(trigger_timing_check(201, 1000) != NULL);
    CHECK(trigger_timing_check(65566, 1000) != NULL);
    CHECK(trigger_timing_check(30, 0) != NULL);
    CHECK(trigger_timing_check(30, 33333) != NULL); /* == period */
    CHECK(trigger_timing_check(100, 12000) != NULL); /* 30 Hz width at 100 Hz */

    test_edge_track();

    /* Idle = inactive level: HIGH for imx296_fast (exposure while LOW), LOW for
     * active_high (exposure while HIGH). */
    CHECK(trigger_timing_idle_high(true));
    CHECK(!trigger_timing_idle_high(false));

    CHECK_DONE("test_trigger_timing");
}
