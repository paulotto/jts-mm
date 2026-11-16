// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_timing.c
 * @brief IMU sample timestamps: INT-edge latch and SH-2 per-report delay (pure logic).
 */

#include "jtsmm/domain/imu_timing.h"

#include <string.h>

void imu_edge_latch_clear(imu_edge_latch_t *l) {
    l->pending = false;
}

bool imu_edge_latch_discard_if_released(imu_edge_latch_t *l, bool int_asserted) {
    if (!l->pending || int_asserted) {
        return false;
    }
    l->pending = false;
    return true;
}

imu_ts_src_t imu_edge_latch_take(imu_edge_latch_t *l, uint64_t now_us, uint32_t max_age_us,
                                 uint64_t *t_out) {
    *t_out = now_us;
    if (!l->pending) {
        return IMU_TS_SRC_NO_EDGE;
    }
    uint64_t edge = l->edge_us;
    /* Consumed here, before the read is kicked: an INT re-assertion while the
     * read runs is then latched afresh for the next packet. */
    l->pending = false;
    if (edge > now_us) {
        return IMU_TS_SRC_FUTURE;
    }
    if (now_us - edge > (uint64_t)max_age_us) {
        return IMU_TS_SRC_STALE;
    }
    *t_out = edge;
    return IMU_TS_SRC_EDGE;
}

uint32_t imu_ts_edge_max_age_us(uint32_t period_us) {
    uint64_t age = (uint64_t)period_us * IMU_TS_EDGE_AGE_PERIODS;
    if (age < IMU_TS_EDGE_MAX_AGE_FLOOR_US) {
        age = IMU_TS_EDGE_MAX_AGE_FLOOR_US;
    }
    return age > UINT32_MAX ? UINT32_MAX : (uint32_t)age;
}

void imu_ts_stats_reset(imu_ts_stats_t *s) {
    memset(s, 0, sizeof(*s));
}

void imu_ts_stats_on_read(imu_ts_stats_t *s, imu_ts_src_t src, uint64_t now_us, uint64_t t_used) {
    if ((unsigned)src < 4u) {
        s->src[src]++;
    }
    if (src != IMU_TS_SRC_EDGE || t_used > now_us) {
        return;
    }
    uint64_t d = now_us - t_used;
    uint32_t lat = d > UINT32_MAX ? UINT32_MAX : (uint32_t)d;
    s->lat_sum += lat; /* <= IMU_TS_STATS_WINDOW * max age: no overflow */
    if (lat > s->lat_max) {
        s->lat_max = lat;
    }
    if (lat > s->lat_max_all) {
        s->lat_max_all = lat;
    }
    if (++s->lat_n >= IMU_TS_STATS_WINDOW) {
        s->lat_mean_w = s->lat_sum / s->lat_n;
        s->lat_max_w = s->lat_max;
        s->lat_w_valid = true;
        s->lat_n = 0;
        s->lat_sum = 0;
        s->lat_max = 0;
    }
}

void imu_ts_stats_on_packet(imu_ts_stats_t *s, const bno085_packet_info_t *info) {
    if (info->channel != BNO085_CH_INPUT && info->channel != BNO085_CH_WAKE) {
        return;
    }
    if (info->reports == 0) {
        return;
    }
    if (!info->has_base) {
        s->no_base_pkts++;
        return;
    }
    int64_t us = (int64_t)info->base_ticks * 100;
    int32_t b = us > INT32_MAX ? INT32_MAX : (us < INT32_MIN ? INT32_MIN : (int32_t)us);
    s->base_last = b;
    if (s->base_n == 0 || b < s->base_min) {
        s->base_min = b;
    }
    if (s->base_n == 0 || b > s->base_max) {
        s->base_max = b;
    }
    s->base_sum += b;
    if (++s->base_n >= IMU_TS_STATS_WINDOW) {
        s->base_mean_w = (int32_t)(s->base_sum / (int64_t)s->base_n);
        s->base_min_w = s->base_min;
        s->base_max_w = s->base_max;
        s->base_w_valid = true;
        s->base_n = 0;
        s->base_sum = 0;
    }
}

uint64_t imu_ts_report_time(uint64_t t_int_us, const bno085_report_t *r, imu_ts_stats_t *s) {
    if (r == NULL || !r->has_ts) {
        return t_int_us;
    }
    int64_t off = (int64_t)r->ts_ticks * 100;
    if (off > 0) {
        /* A sample cannot postdate the INT that announced it: rounding or an
         * odd rebase. Never stamp later than t_int. */
        if (s) {
            s->clamp_late++;
        }
        return t_int_us;
    }
    if (off < -(int64_t)IMU_TS_MAX_BACKDATE_US || (uint64_t)(-off) > t_int_us) {
        /* Implausible (a corrupt delay field, a misparsed record): keep the
         * old behaviour for this report rather than invent a time. */
        if (s) {
            s->clamp_early++;
        }
        return t_int_us;
    }
    return t_int_us - (uint64_t)(-off);
}
