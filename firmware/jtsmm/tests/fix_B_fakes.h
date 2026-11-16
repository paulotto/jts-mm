// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * Fakes for the fix-B host tests: a fake clock, GPIO, a scripted BNO085 on
 * the I2C bus, and link-time stand-ins for the platform/runtime functions the
 * IMU driver/domain code calls. See fix_B_fakes.c.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "jtsmm/domain/imu_timing.h"
#include "jtsmm/platform/i2c_dma.h"

/** Fake microsecond clock (advanced by sleep_ms/sleep_us/busy_wait and by tests). */
extern uint64_t fake_now_us;

/** Level of each GPIO as seen by gpio_get() (default high = INT idle). */
extern bool fake_gpio_level[64];

/** Queue one SHTP packet the fake device returns on the next reads of @p bus (0/1). */
void fake_i2c_queue(int bus, const uint8_t *pkt, size_t len);
/** Queue what a booting BNO085 sends: reset-complete, a wake-channel packet, product id. */
void fake_queue_boot(int bus);
/** Drop all queued packets and write logs. */
void fake_i2c_reset(void);
/** Number of i2c writes seen on @p bus, a log of the first ones, and the last one's bytes. */
#define FAKE_WRITE_LOG 64
extern int fake_i2c_writes[2];
extern uint8_t fake_i2c_write_log[2][FAKE_WRITE_LOG][64];
extern uint8_t fake_i2c_last_write[2][64];
extern size_t fake_i2c_last_write_len[2];
/** Next @p n writes on any bus fail (NACK). */
extern int fake_i2c_fail_writes;
/** After this many successful reads, one read fails (-1 = never). */
extern int fake_i2c_read_fail_after;
/** Bus @p i has no responding device (every transaction NACKs); attempts counted. */
extern bool fake_i2c_dead[2];
extern int fake_i2c_dead_attempts[2];

/** imu_bus_clear() calls per slot; NRST low pulses and current level. */
extern int fake_bus_clears[2];
extern int fake_nrst_pulses;
extern bool fake_nrst_level;
/** config_apply_busy() result. */
extern bool fake_config_apply_busy;
/** Timeout argument of the last i2c read/write call. */
extern unsigned fake_i2c_last_timeout_us;

/** A completion callback is registered for the DMA slot. */
extern bool fake_dma_cb_set[2];
/** The completion callback itself (drive it as i2c_dma_service() would). */
typedef void (*fake_dma_cb_t)(i2c_dma_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len);
extern fake_dma_cb_t fake_dma_cb[2];
/** I2C DMA handshake calls seen per slot; quiesce result to return. */
extern int fake_dma_aborts[2];
extern int fake_dma_quiesces[2];
extern bool fake_dma_quiesce_ok;
/** Per-slot override: quiesce of this slot fails. */
extern bool fake_dma_quiesce_fail[2];
/** imu_bus_probe() result per slot. */
extern bool fake_probe_ok[2];
/** i2c_dma_kick_shtp_read() calls per slot and the t_latch of the last one. */
extern int fake_dma_kicks[2];
extern uint64_t fake_dma_kick_t[2];
/** Hook registered with i2c_dma_on_body_start() per slot. */
extern void (*fake_dma_body_cb[2])(i2c_dma_slot_t slot);
/** INT edge latch per slot (the real struct), arm state and arm calls. */
extern imu_edge_latch_t fake_edge[2];
extern bool fake_edge_armed[2];
extern int fake_edge_arms[2];

/** Count of panel_imu_mark_active() calls per slot. */
extern int fake_panel_active[2];
/** Last host log line. */
extern char fake_last_log[256];
