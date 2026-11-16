// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file records.h
 * @brief USB binary record type tags and fixed payload sizes.
 *
 * @details Wire format uses sync bytes @c JTSMM_SYNC0/@c JTSMM_SYNC1 then a
 *          framed record (type, @c t_master_us, payload, CRC). See
 *          docs/hardware/h05_microcontroller_firmware.md §3 and
 *          docs/software/s05_imu_processing.md §1 for IMU layouts.
 */
#pragma once

#include <stdint.h>

#define JTSMM_SYNC0           0x55u /**< First framing sync byte. */
#define JTSMM_SYNC1           0xAAu /**< Second framing sync byte. */

#define REC_EMG_BLOCK         0x10u /**< ADS1299 multi-channel sample block. */
#define REC_IMU_MTA_QUAT      0x11u /**< MTA slot fusion quaternion. */
#define REC_IMU_MTA_RAW       0x12u /**< MTA slot tagged raw accel/gyro/mag. */
#define REC_IMU_MTA_LIN_ACCEL 0x13u /**< MTA slot linear acceleration. */
#define REC_IMU_CRA_QUAT      0x14u /**< CRA slot fusion quaternion. */
#define REC_IMU_CRA_RAW       0x15u /**< CRA slot tagged raw accel/gyro/mag. */
#define REC_IMU_CRA_LIN_ACCEL 0x16u /**< CRA slot linear acceleration. */
#define REC_TRIGGER_EVENT     0x20u /**< Camera XTRIG / trigger_id event. */
#define REC_LED_FLASH_EVENT   0x21u /**< Track A sync-LED flash event. */
#define REC_BUTTON_EVENT      0x22u /**< Panel button edge event. */
#define REC_HEARTBEAT         0x30u /**< Periodic liveness record. */
#define REC_PPS_REPLY         0x40u /**< Reply to host PPS handshake. */
#define REC_ADS_STATUS        0x50u /**< ADS1299 status / lead-off word. */
#define REC_STATS             0x51u /**< Link / ring statistics payload. */
#define REC_LOG               0x70u /**< Human-readable log line (also on CDC). */
#define REC_CMD_ACK           0xE0u /**< Command acknowledgement. */

/**
 * @{
 * @name REC_EMG_BLOCK payload (contract K1)
 * @details `[ADS status 3 B][8 x 24-bit channels 24 B][seq u32 LE]`. `seq` is
 *          the acquisition sequence number assigned when the conversion is
 *          read: +1 per conversion, and conversions the firmware knows it lost
 *          advance it too, so a host-side seq gap is the total loss after the
 *          ADC. Wraps at 2^32. Hosts accept 27 (older firmware, no seq) and 31.
 */
#define EMG_BLOCK_ADS_BYTES   27u /**< ADS1299 status + 8 x 24-bit channel bytes. */
#define EMG_BLOCK_SEQ_OFFSET  27u /**< Offset of the u32 LE seq. */
#define EMG_BLOCK_PAYLOAD_LEN 31u /**< Bytes in @c REC_EMG_BLOCK payload. */
/** @} */
#define TRIGGER_EVENT_PAYLOAD_LEN 12u /**< Bytes in @c REC_TRIGGER_EVENT payload. */

/**
 * @{
 * @name TRIGGER_EVENT flags[0]: how t_master_us was obtained
 * @details Mirrored by TRIGGER_STAMP_* in software/jtsmm/jtsmm/protocol/records.py.
 */
#define TRIG_STAMP_UNKNOWN 0u /**< Pre-2026-09 firmware: software timer, untagged. */
#define TRIG_STAMP_PIO_IRQ 1u /**< Latched in the PIO IRQ on the exposure-start edge. */
#define TRIG_STAMP_TIMER   2u /**< Software repeating_timer (legacy mode or fallback). */
/** @} */
#define BUTTON_EVENT_PAYLOAD_LEN 5u   /**< Bytes in @c REC_BUTTON_EVENT payload. */

/**
 * @brief Tagged IMU wire records (docs/software/s05_imu_processing.md §1).
 *
 * @details Each SHTP report instance is one self-describing record so accel,
 *          gyro, mag, and fusion quaternions can run at independent rates:
 *          - QUAT:      `[report_id:u8][w,x,y,z:f32x4][acc:u8]` → 17/18 B
 *          - RAW:       `[sensor_id:u8][x,y,z:int16x3][acc:u8]` → 7/8 B
 *          - LIN_ACCEL: `[x,y,z:f32x3][acc:u8]` → 12/13 B
 *
 *          Optional trailing @c acc is SH-2 accuracy (0=unreliable..3=high).
 *          Hosts accept with or without it; legacy hosts used 16 B QUAT / 18 B RAW.
 */
#define IMU_QUAT_PAYLOAD_LEN      18u /**< Max quaternion payload including accuracy. */
#define IMU_RAW_PAYLOAD_LEN       8u  /**< Max tagged raw payload including accuracy. */
#define IMU_LIN_ACCEL_PAYLOAD_LEN 13u /**< Max linear-accel payload including accuracy. */

/** RAW record sensor tags (match BNO085 SH-2 input report ids). */
#define IMU_SENSOR_ACCEL 0x01u /**< Accelerometer. */
#define IMU_SENSOR_GYRO  0x02u /**< Gyroscope. */
#define IMU_SENSOR_MAG   0x03u /**< Magnetometer. */

#define JTSMM_MAX_PAYLOAD 240u /**< Maximum record payload length (bytes). */
