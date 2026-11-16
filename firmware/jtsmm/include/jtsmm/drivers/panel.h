// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file panel.h
 * @brief Front-panel LEDs, user RGB, buttons, and fault bitfield.
 */
#pragma once

#include "jtsmm/drivers/imu_bus.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Coarse panel operating state for status LEDs.
 */
typedef enum {
    PANEL_STATE_IDLE = 0,     /**< Idle / not streaming. */
    PANEL_STATE_ARMED,        /**< Armed / ready. */
    PANEL_STATE_STREAMING,    /**< Capture session active. */
    PANEL_STATE_FAULT,        /**< Fault indicated. */
} panel_state_t;

#define PANEL_FAULT_WATCHDOG_EMG  (1u << 0) /**< EMG watchdog trip. */
#define PANEL_FAULT_HARD          (1u << 1) /**< Hard / SOS-class fault. */
#define PANEL_FAULT_IMU_MTA       (1u << 2) /**< MTA IMU fault. */
#define PANEL_FAULT_IMU_CRA       (1u << 3) /**< CRA IMU fault. */

/** @brief Initialize panel GPIOs / WS2812. */
void panel_init(void);

/** @brief Set coarse panel state (drives status LEDs). */
void panel_set_state(panel_state_t state);

/** @brief Toggle heartbeat LED (call at ~1 Hz). */
void panel_heartbeat_tick(void);

/** @brief Drive capture-active LED. */
void panel_set_cap_active(bool on);

/** @brief Pulse IMU activity LED for @p slot. */
void panel_imu_mark_active(imu_slot_t slot);

/** @brief Drive EMG lead-off indicator when @p any is true. */
void panel_set_emg_leadoff(bool any);

/** @brief Poll user buttons; may toggle capture / emit button events. */
void panel_poll_buttons(void);

/** @brief Set user RGB (0–255 per channel). */
void panel_set_rgb(uint8_t r, uint8_t g, uint8_t b);

/**
 * @brief Set user RGB with extra brightness shift.
 * @param extra_brightness_shift Left-shift applied to components.
 */
void panel_set_rgb_ex(uint8_t r, uint8_t g, uint8_t b, uint8_t extra_brightness_shift);

/** @brief Refresh WS2812 from last RGB set. */
void panel_refresh_user_rgb(void);

/** @brief Recompute streaming-related LEDs from capture/stream flags. */
void panel_update_streaming_state(void);

/** @brief Periodic status LED update. */
void panel_status_tick(void);

/** @brief OR fault bits into the fault latch. */
void panel_fault_set(uint32_t bits);

/** @brief Clear fault bits from the fault latch. */
void panel_fault_clear(uint32_t bits);

/** @brief Advance disco / attract-mode animation if active. */
void panel_disco_tick(void);

/** @brief true while disco mode is active. */
bool panel_disco_active(void);

/** @brief Start interactive panel LED test sequence. */
void panel_panel_test_start(void);

/** @brief Poll panel test state machine; @return true when finished. */
bool panel_panel_test_poll(void);

/** @brief true while panel test is running. */
bool panel_panel_test_busy(void);

/** @brief true if RX should stay quiet during panel test. */
bool panel_panel_test_rx_quiet(void);
