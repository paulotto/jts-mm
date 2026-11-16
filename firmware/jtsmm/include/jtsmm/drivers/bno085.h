// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file bno085.h
 * @brief BNO085 SH-2 / SHTP I2C driver (single device instance).
 *
 * @details Low-level transport and feature enable used by @c imu_bus / domain
 *          IMU streaming. See Hillcrest SH-2 reference for report ids.
 */
#pragma once

#include "hardware/i2c.h"
#include <stdbool.h>
#include <stdint.h>

#define BNO085_I2C_ADDR_DEFAULT 0x4A /**< Default 7-bit I2C address. */
#define BNO085_SHTP_MAX_PACKET 512u  /**< Max SHTP packet buffer (bytes). */

/** @{ @name SHTP channels */
#define BNO085_CH_COMMAND    0u
#define BNO085_CH_EXECUTABLE 1u
#define BNO085_CH_CONTROL    2u
#define BNO085_CH_INPUT      3u
#define BNO085_CH_WAKE       4u
#define BNO085_CH_GYRO_RV    5u /**< Gyro-integrated rotation vector (SH-2 §6.5.44). */
/** @} */

/** @{ @name Control / product-id report ids */
#define BNO085_REPORT_PRODUCT_ID_REQ   0xF9u
#define BNO085_REPORT_PRODUCT_ID_RESP  0xF8u
#define BNO085_REPORT_SET_FEATURE      0xFDu
#define BNO085_REPORT_COMMAND_REQ      0xF2u
#define BNO085_REPORT_COMMAND_RESP     0xF1u
#define BNO085_REPORT_GET_FEATURE_RESP 0xFCu
/** @} */

/** @{ @name SH-2 command ids in Command Request (0xF2) — SH-2 §6.4 */
#define BNO085_CMD_SAVE_DCD     0x06u /**< Persist Dynamic Calibration Data to flash. */
#define BNO085_CMD_ME_CALIBRATE 0x07u /**< Configure motion-engine dynamic calibration. */
/** @} */

/** @{ @name Input sensor / fusion report ids */
#define BNO085_REPORT_ACCELEROMETER   0x01u
#define BNO085_REPORT_GYROSCOPE       0x02u
#define BNO085_REPORT_MAGNETOMETER    0x03u
#define BNO085_REPORT_LINEAR_ACCEL    0x04u
#define BNO085_REPORT_ROTATION_VECTOR 0x05u
#define BNO085_REPORT_GAME_ROT_VEC    0x08u
#define BNO085_REPORT_GEOMAG_ROT_VEC  0x09u
#define BNO085_REPORT_GYRO_INT_ROT_VEC 0x2Au /**< Channel-5 only; no seq/status/delay. */
#define BNO085_REPORT_BASE_TIMESTAMP  0xFBu
#define BNO085_REPORT_TIMESTAMP_REBASE 0xFAu
/** @} */

#define BNO085_SHTP_CONTINUATION 0x8000u /**< Header length bit 15: continuation fragment. */

/**
 * @brief Undo the continuation flag our own header pre-read causes.
 *
 * @details Both read paths read the 4-byte SHTP header first, then re-read the
 *          whole packet. Over I2C the BNO085 counts the 4-byte pre-read as a
 *          partial read, so the re-read comes back with bit 15 set although it
 *          holds the complete packet from its start. The parser drops flagged
 *          packets as fragments, which discarded every report and every Get
 *          Feature Response on real hardware (bench, 2026-09-25: parse=0,
 *          cont_pkts ~ reads, granted=none). A flag already present in the
 *          pre-read marks a genuine tail of an earlier transfer and is kept.
 * @param buf        The re-read packet (header at buf[0..3]).
 * @param preread_b1 Byte 1 of the pre-read header.
 */
static inline void bno085_shtp_clear_reread_continuation(uint8_t *buf, uint8_t preread_b1) {
    if ((preread_b1 & 0x80u) == 0u) {
        buf[1] &= 0x7Fu;
    }
}
#define BNO085_GYRO_RV_REPORT_LEN 14u    /**< Channel-5 report: [i,j,k,real Q14][wx,wy,wz Q10]. */
#define BNO085_EXEC_RESET_COMPLETE 0x01u /**< Channel-1 payload[0]: device finished a reset. */
#define BNO085_CMD_ADVERTISEMENT 0x00u   /**< Channel-0 payload[0]: SHTP advertisement. */

/**
 * @brief Per-device I2C / pin / sequence state.
 */
typedef struct {
    i2c_inst_t *i2c;     /**< Pico I2C instance. */
    uint8_t addr;        /**< 7-bit address. */
    uint8_t int_pin;     /**< INT GPIO. */
    uint8_t sda_pin;     /**< SDA GPIO. */
    uint8_t scl_pin;     /**< SCL GPIO. */
    uint8_t seq[6];      /**< Per-channel SHTP sequence counters. */
    uint8_t cmd_seq;     /**< Sequence counter for Command Request (0xF2). */
    uint8_t write_tries; /**< Attempts per blocking write; 0 = default (20). Recovery uses 1. */
    uint32_t baud_hz;    /**< Achieved bus bitrate; scales blocking timeouts (0 = assume 100 kHz). */
    const char *label;   /**< Debug label (e.g. "MTA"). */
} bno085_t;

/**
 * @brief Parsed fields extracted from one or more SHTP input reports.
 */
typedef struct {
    bool product_id;       /**< Saw product-id response. */
    bool set_feature_ack;  /**< Saw set-feature acknowledgement. */
    bool got_accel;        /**< Accel sample present. */
    bool got_gyro;         /**< Gyro sample present. */
    bool got_mag;          /**< Mag sample present. */
    bool got_rotation;     /**< Fusion quaternion present. */
    bool got_linear_accel; /**< Linear accel present. */
    uint16_t sw_part;      /**< Software part number. */
    uint8_t sw_maj;        /**< Software major. */
    uint8_t sw_min;        /**< Software minor. */
    uint8_t quat_report_id; /**< Report id (0x05/0x08/0x09/0x2A) for quat_* fields. */
    int16_t quat_i, quat_j, quat_k, quat_real; /**< Q-point quaternion components. */
    int16_t accel_x, accel_y, accel_z;
    int16_t gyro_x, gyro_y, gyro_z;
    int16_t mag_x, mag_y, mag_z;
    float lin_ax, lin_ay, lin_az;
    uint8_t lin_accel_accuracy;
    /**
     * SH-2 per-report accuracy (status bits 1:0): 0=unreliable … 3=high.
     * 0xFF = not present in this packet.
     */
    uint8_t quat_accuracy, accel_accuracy, gyro_accuracy, mag_accuracy;
    char error[64]; /**< Last parse/transport error string. */
} bno085_parse_state_t;

/**
 * @brief One decoded SH-2 report instance (see @ref bno085_parse_packet).
 *
 * @details Sensor reports carry their Q-point data words in @c v: x,y,z for
 *          motion vectors, i,j,k,real for quaternions. A Get Feature Response
 *          (0xFC) is delivered too, with @c feature_id / @c interval_us set.
 */
typedef struct {
    uint8_t report_id;    /**< SH-2 report id (0x2A for channel-5 reports, 0xFC for feature responses). */
    uint8_t seq;          /**< SH-2 per-report sequence number (byte 1); valid if @c has_seq. */
    bool has_seq;         /**< false for channel-5 gyro-integrated RV (no per-report header). */
    uint8_t accuracy;     /**< Status bits 1:0 (0=unreliable..3=high); 0xFF = not carried. */
    int16_t v[4];         /**< Data words; v[3] only for quaternions. */
    uint8_t feature_id;   /**< 0xFC only: the report the response describes. */
    uint32_t interval_us; /**< 0xFC only: report interval the sensor actually granted. */
    uint16_t delay;       /**< Input reports: 14-bit delay field (status[7:2]:byte 3), 100 µs ticks. */
    bool has_ts;          /**< A Base Timestamp Reference (0xFB) preceded this report in its packet. */
    int32_t ts_ticks;     /**< Valid if @c has_ts: sample time minus INT assertion, 100 µs
                               ticks = -base_delta + rebase + delay (CEVA sh2.c). */
} bno085_report_t;

/** @brief Per-report callback of @ref bno085_parse_packet. */
typedef void (*bno085_report_cb_t)(void *ctx, const bno085_report_t *report);

/**
 * @brief Packet-level facts found by @ref bno085_parse_packet.
 */
typedef struct {
    uint8_t channel;          /**< SHTP channel (header byte 2). */
    uint8_t shtp_seq;         /**< SHTP per-channel sequence (header byte 3). */
    uint16_t reports;         /**< Report instances delivered to the callback (0xFC included). */
    bool continuation;        /**< Continuation fragment: not parsed as a fresh packet. */
    bool truncated;           /**< Trailing bytes skipped (unknown report id or short report). */
    bool reset_complete;      /**< Channel 1 "reset complete": the device rebooted. */
    bool advertisement;       /**< Channel 0 advertisement: sent after every device reset. */
    bool product_id;          /**< Product-id response (0xF8). */
    bool command_resp;        /**< Command response (0xF1). */
    bool feature_resp;        /**< Get Feature Response (0xFC); last one in the two fields below. */
    uint8_t feature_report_id;
    uint32_t feature_interval_us;
    bool has_base;            /**< Input packet carried a Base Timestamp Reference (0xFB). */
    int32_t base_ticks;       /**< Its base delta (first 0xFB), 100 µs ticks: INT assertion
                                   minus base timestamp (>= 0 in practice). */
    bool has_rebase;          /**< Input packet carried a Timestamp Rebase (0xFA). */
} bno085_packet_info_t;

/**
 * @brief Tracks SH-2 per-report sequence numbers to count lost reports.
 *
 * @details Ids >= @ref BNO085_SEQ_TRACK_IDS are ignored. A loss of a multiple
 *          of 256 consecutive reports of one id aliases to 0 (8-bit counter).
 */
#define BNO085_SEQ_TRACK_IDS 64u
typedef struct {
    uint8_t last[BNO085_SEQ_TRACK_IDS];
    uint64_t valid; /**< Bit n set once id n has been seen. */
} bno085_seq_tracker_t;

/** @brief Forget all ids (e.g. after a device reset restarts every counter). */
void bno085_seq_reset(bno085_seq_tracker_t *t);

/**
 * @brief Record @p seq for @p report_id.
 * @return Reports missing between the previous and this one (0 when
 *         contiguous, on first sight, on a repeat, or for untracked ids).
 */
uint32_t bno085_seq_update(bno085_seq_tracker_t *t, uint8_t report_id, uint8_t seq);

/**
 * @brief Parse one SHTP packet, calling @p cb once per report instance.
 *
 * @details Handles every report of a batched input packet (channels 3/4;
 *          0xFB/0xFA timestamp records are folded into each following report's
 *          @c ts_ticks, SH-2 §7.2), the header-less 14-byte
 *          gyro-integrated RV reports of channel 5, control responses
 *          (0xF8/0xF1/0xFC) on channel 2, and the reset indications on
 *          channels 0/1. Pure function: no I/O.
 * @param pkt  Full SHTP packet (4-byte header first).
 * @param len  Bytes valid in @p pkt.
 * @param cb   Per-report callback; may be NULL.
 * @param ctx  Passed to @p cb.
 * @param info Out: packet-level facts; must not be NULL.
 * @return false for a malformed or continuation packet (nothing delivered).
 */
bool bno085_parse_packet(const uint8_t *pkt, uint16_t len, bno085_report_cb_t cb, void *ctx,
                         bno085_packet_info_t *info);

/**
 * @brief Fill @p dev with bus/pin identity (does not talk on the wire).
 */
void bno085_init_dev(bno085_t *dev, i2c_inst_t *i2c, uint8_t addr, uint8_t int_pin, uint sda_pin,
                     uint scl_pin, const char *label);

/** @brief Pulse shared hardware reset. @return true on success. */
bool bno085_hw_reset(void);

/**
 * @brief Configure I2C baud for @p dev's bus.
 * @param dev  Device; must not be NULL.
 * @param baud Bitrate (Hz).
 */
void bno085_bus_config(bno085_t *dev, uint baud);

/**
 * @brief Blocking SHTP read into @p buf.
 * @return true if a packet was read into @p out_len.
 */
bool bno085_read_packet(bno085_t *dev, uint8_t *buf, size_t buf_cap, size_t *out_len);

/**
 * @brief Non-blocking SHTP read attempt.
 * @return true if a packet was available and copied.
 */
bool bno085_try_read_packet(bno085_t *dev, uint8_t *buf, size_t buf_cap, size_t *out_len);

/**
 * @brief Write one SHTP packet on @p channel.
 * @return true on success.
 */
bool bno085_write_packet(bno085_t *dev, uint8_t channel, const uint8_t *payload, uint16_t payload_len);

/** @brief Request product-id report. @return true if request queued/sent. */
bool bno085_request_product_id(bno085_t *dev);

/** @brief Soft-reset via SHTP executable channel. @return true on success. */
bool bno085_shtp_soft_reset(bno085_t *dev);

/** @brief Full boot sequence (reset + product id + ready). @return true on success. */
bool bno085_boot_device(bno085_t *dev);

/**
 * @brief Enable a report and wait (<= 500 ms) for its Get Feature Response.
 * @param interval_us Requested report period in microseconds.
 * @param granted_us  Out (may be NULL): interval the sensor granted, from the
 *                    0xFC response; 0 if none arrived in time.
 * @return false only if the Set Feature write failed.
 */
bool bno085_enable_report_drain(bno085_t *dev, uint8_t report_id, uint32_t interval_us,
                                uint32_t *granted_us);

/**
 * @brief Enable a sensor/fusion report at @p interval_us.
 */
bool bno085_enable_report(bno085_t *dev, uint8_t report_id, uint32_t interval_us);

/**
 * @brief Configure motion-engine dynamic (auto) calibration.
 * @param accel/gyro/mag true to enable that sensor's ME calibration.
 */
bool bno085_set_calibration(bno085_t *dev, bool accel, bool gyro, bool mag);

/**
 * @brief Persist Dynamic Calibration Data to BNO085 flash (Save DCD).
 *
 * @note Must be issued while the bus is idle (not streaming).
 */
bool bno085_save_dcd(bno085_t *dev);

/**
 * @brief Parse an SHTP packet into @p out (last report of each kind wins).
 *
 * @details Convenience wrapper over @ref bno085_parse_packet for the blocking
 *          boot/recovery paths; the streaming path uses the per-report form.
 * @return true if parsing produced usable fields (may still set @c out->error).
 */
bool bno085_parse_shtp(const uint8_t *pkt, uint16_t len, bno085_parse_state_t *out);

/** @brief true if INT GPIO is asserted for @p dev. */
bool bno085_int_asserted(const bno085_t *dev);
