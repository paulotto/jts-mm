// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_timing.h
 * @brief IMU sample timestamps: INT-edge latch and SH-2 per-report delay (pure logic).
 *
 * @details An IMU record's `t_master_us` is built in two steps:
 *   1. t_int -- when the BNO085 asserted INT. A falling-edge IRQ on the sensor
 *      core (core 1) latches time_us_64() into an @ref imu_edge_latch_t; the
 *      loop takes it when it kicks the read. Without a usable edge it falls
 *      back to the poll time (the old behaviour) and counts why.
 *   2. per report: t_sample = t_int - base_delta*100us (+ rebase*100us)
 *      + delay*100us, from the SH-2 Base Timestamp Reference (0xFB), Timestamp
 *      Rebase (0xFA) and each input report's 14-bit delay field -- the same
 *      arithmetic as CEVA's sh2.c. Clamped to never be later than t_int.
 *
 * Everything here is free of hardware access so the host tests can drive it;
 * the IRQ plumbing lives in imu_bus.c and the call sites in imu_config.c.
 */
#pragma once

#include "jtsmm/drivers/bno085.h"
#include <stdbool.h>
#include <stdint.h>

/** Use the INT falling-edge time as t_int (0 = poll time, the previous behaviour). */
#ifndef IMU_TS_USE_INT_EDGE
#define IMU_TS_USE_INT_EDGE 1
#endif
/** Apply the SH-2 base-delta / per-report delay (0 = every report gets t_int). */
#ifndef IMU_TS_USE_SH2_DELAY
#define IMU_TS_USE_SH2_DELAY 1
#endif

/** An edge older than max(IMU_TS_EDGE_AGE_PERIODS report periods, this) is stale. */
#define IMU_TS_EDGE_MAX_AGE_FLOOR_US 20000u
#define IMU_TS_EDGE_AGE_PERIODS 3u
/** A report more than this before t_int is implausible: it keeps t_int. */
#define IMU_TS_MAX_BACKDATE_US 100000
/** Packets per statistics window (edge->kick latency, base delta). */
#define IMU_TS_STATS_WINDOW 256u

#if defined(__GNUC__)
#define IMU_TS_INLINE static inline __attribute__((always_inline))
#else
#define IMU_TS_INLINE static inline
#endif

/**
 * @brief One slot's INT edge latch.
 *
 * @details Written by the INT IRQ and consumed by the sensor-core loop on the
 *          same core (core 1); the consumer masks IRQs around take/discard.
 *          Only the first edge after a take is kept: the INT line stays low
 *          until the read starts, so a later edge before the take is noise.
 */
typedef struct {
    volatile uint64_t edge_us;   /**< time_us_64() of the latched edge. */
    volatile bool pending;       /**< edge_us is valid and not consumed yet. */
    volatile uint32_t edges;     /**< every falling edge seen by the IRQ. */
    volatile uint32_t coalesced; /**< edges ignored because one was pending. */
} imu_edge_latch_t;

/** Why a read got the timestamp it got. */
typedef enum {
    IMU_TS_SRC_EDGE = 0, /**< latched INT edge. */
    IMU_TS_SRC_NO_EDGE,  /**< poll time: nothing latched (INT already low at arm, IRQ lost...). */
    IMU_TS_SRC_STALE,    /**< poll time: latched edge older than the bound. */
    IMU_TS_SRC_FUTURE,   /**< poll time: latched edge later than now (never expected). */
} imu_ts_src_t;

/** @brief IRQ side: latch @p t_us unless an edge is already pending. RAM-safe (inline). */
IMU_TS_INLINE void imu_edge_latch_on_edge(imu_edge_latch_t *l, uint64_t t_us) {
    l->edges++;
    if (l->pending) {
        l->coalesced++;
        return;
    }
    l->edge_us = t_us;
    l->pending = true;
}

/** @brief Forget a pending edge (stream start/stop/recovery, arm/disarm). */
void imu_edge_latch_clear(imu_edge_latch_t *l);

/**
 * @brief Drop a pending edge whose INT line has been released again.
 *
 * @details Call only while no read is in flight for the slot (or right at its
 *          completion). A real data-ready edge keeps INT low until the host
 *          reads, so pending + INT high means a glitch, or an edge whose packet
 *          the finished read already consumed; keeping it would stamp the next
 *          packet too early.
 * @return true if an edge was dropped.
 */
bool imu_edge_latch_discard_if_released(imu_edge_latch_t *l, bool int_asserted);

/**
 * @brief Loop side, right before kicking a read: take the edge (clears pending).
 *
 * @param now_us     Poll time (time_us_64()) -- the fallback timestamp.
 * @param max_age_us Edges older than this are stale.
 * @param t_out      Out: the timestamp to use for this read.
 * @return Source of @p t_out.
 */
imu_ts_src_t imu_edge_latch_take(imu_edge_latch_t *l, uint64_t now_us, uint32_t max_age_us,
                                 uint64_t *t_out);

/** @brief Stale-edge bound for a stream whose fastest report period is @p period_us (0 = unknown). */
uint32_t imu_ts_edge_max_age_us(uint32_t period_us);

/**
 * @brief Per-slot timestamp statistics (written by the sensor core, read by imu_status).
 */
typedef struct {
    uint32_t src[4];    /**< reads per @ref imu_ts_src_t. */
    uint32_t spurious;  /**< edges dropped by @ref imu_edge_latch_discard_if_released. */
    uint32_t hdr_drops; /**< edges dropped between the header read and the full re-read. */
    /* edge -> read-kick latency (µs), edge-stamped reads only */
    uint32_t lat_n, lat_sum, lat_max;    /**< current window */
    uint32_t lat_mean_w, lat_max_w;      /**< last full window */
    bool lat_w_valid;
    uint32_t lat_max_all;
    /* SH-2 base delta (µs, positive = sample before INT), packets carrying 0xFB */
    uint32_t base_n;
    int64_t base_sum;
    int32_t base_min, base_max;            /**< current window */
    int32_t base_mean_w, base_min_w, base_max_w; /**< last full window */
    bool base_w_valid;
    int32_t base_last;
    uint32_t no_base_pkts;  /**< input packets without 0xFB (reports keep t_int). */
    uint32_t clamp_late;    /**< report computed later than t_int -> t_int. */
    uint32_t clamp_early;   /**< report > IMU_TS_MAX_BACKDATE_US before t_int -> t_int. */
} imu_ts_stats_t;

/** @brief Zero @p s (stream start). */
void imu_ts_stats_reset(imu_ts_stats_t *s);

/** @brief Account one read: its source and, for edge reads, kick latency now - t_edge. */
void imu_ts_stats_on_read(imu_ts_stats_t *s, imu_ts_src_t src, uint64_t now_us, uint64_t t_used);

/** @brief Account one parsed input packet's base-delta record (@p info from bno085_parse_packet). */
void imu_ts_stats_on_packet(imu_ts_stats_t *s, const bno085_packet_info_t *info);

/**
 * @brief Sample time of one report.
 *
 * @details t_int + (-base + rebase + delay) * 100 µs when the report carries SH-2
 *          timing (@c r->has_ts); t_int otherwise (channel-5 gyro RV, a report
 *          before any 0xFB, control responses). Results later than t_int are
 *          clamped to t_int and results more than IMU_TS_MAX_BACKDATE_US
 *          earlier keep t_int; both are counted in @p s (may be NULL).
 */
uint64_t imu_ts_report_time(uint64_t t_int_us, const bno085_report_t *r, imu_ts_stats_t *s);
