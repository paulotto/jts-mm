// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file pins.h
 * @brief Canonical RP2350A GPIO map for jtsmm_hat.
 *
 * @details Must match docs/hardware/h05_microcontroller_firmware.md §1.1.
 *          Values are SDK GPIO numbers (not package pin numbers).
 */
#pragma once

/** @{ @name ADS1299 — SPI0 */
#define PIN_ADS_MISO   0
#define PIN_ADS_SCLK   2
#define PIN_ADS_MOSI   3
#define PIN_ADS_CS     5
#define PIN_ADS_DRDY   6  /**< Data-ready (active low). */
#define PIN_ADS_RESET  7
#define PIN_ADS_PWDN   8
#define PIN_ADS_START  9
/** @} */

/** @{ @name Panel buttons (active low, internal pull-up) */
#define PIN_SW_USER1  10
#define PIN_SW_USER2  11
/** @} */

#define PIN_LED_USER   4  /**< User RGB WS2812B-2020 (GRB order). */
#define PIN_TRIG       13 /**< Camera XTRIG out, idles at the inactive level of pulse.polarity (HIGH for imx296_fast); the TRIG LED on the same pin (active high) is lit while idle for imx296_fast. */
#define PIN_LED_FLASH  14 /**< Track A sync LED (optional / DNP on default GS HAT). */

/** @{ @name Status LEDs (0603, active high unless noted) */
#define PIN_LED_HB     15 /**< Heartbeat. */
#define PIN_LED_FAULT  25
#define PIN_LED_CAP    24 /**< Capture / streaming active. */
#define PIN_LED_MTA    26 /**< MTA IMU activity. */
#define PIN_LED_CRA    27 /**< CRA IMU activity. */
#define PIN_LED_DP     28 /**< DP IMU (unpopulated on v1). */
#define PIN_LED_EMG_LO 29 /**< EMG lead-off / fault hint. */
/** @} */

/** @{ @name IMU I2C + interrupt / reset */
#define PIN_IMU_CRA_SDA 18
#define PIN_IMU_CRA_SCL 19
#define PIN_IMU_MTA_SDA 20
#define PIN_IMU_MTA_SCL 21
#define PIN_IMU_NRST    22 /**< Shared IMU reset (active low). */
#define PIN_IMU_MTA_INT 16 /**< MTA BNO085 INT. */
#define PIN_IMU_CRA_INT 17 /**< CRA BNO085 INT. */
/** @} */

#define ADS1299_CHIP_ID 0x3E /**< Expected ADS1299 device ID register value. */
