// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file record.hpp
 * @brief USB record type IDs and the decoded @ref jtsmm::Record container.
 *
 * @details Kept free of pybind11 so the parser and HDF5 writer can share it.
 *          Values mirror `software/jtsmm/jtsmm/protocol/types.py::RecordType`.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace jtsmm {

/**
 * @brief Wire record type tags (USB framed payload).
 */
enum RecordType : uint8_t {
    REC_EMG_BLOCK = 0x10,         /**< ADS1299 sample block. */
    REC_IMU_MTA_QUAT = 0x11,      /**< MTA fusion quaternion. */
    REC_IMU_MTA_RAW = 0x12,       /**< MTA tagged raw motion. */
    REC_IMU_MTA_LIN_ACCEL = 0x13, /**< MTA linear acceleration. */
    REC_IMU_CRA_QUAT = 0x14,      /**< CRA fusion quaternion. */
    REC_IMU_CRA_RAW = 0x15,       /**< CRA tagged raw motion. */
    REC_IMU_CRA_LIN_ACCEL = 0x16, /**< CRA linear acceleration. */
    REC_IMU_DP_QUAT = 0x17,       /**< DP IMU quat (unpopulated on v1). */
    REC_IMU_DP_RAW = 0x18,
    REC_IMU_DP_LIN_ACCEL = 0x19,
    REC_TRIGGER_EVENT = 0x20,     /**< Camera trigger event. */
    REC_LED_FLASH_EVENT = 0x21,   /**< Sync LED flash event. */
    REC_BUTTON_EVENT = 0x22,      /**< Panel button edge event. */
    REC_HEARTBEAT = 0x30,
    REC_PPS_REPLY = 0x40,
    REC_ADS_STATUS = 0x50,
    REC_STATS = 0x51,
    REC_LOG = 0x70,
    REC_CMD_ACK = 0xE0,
};

/**
 * @brief One validated frame: header stripped, CRC verified, payload copied out.
 */
struct Record {
    uint8_t record_type = 0;           /**< @ref RecordType value. */
    uint64_t t_master_us = 0;          /**< MCU master timestamp (µs). */
    std::vector<uint8_t> payload;      /**< Payload bytes. */
};

}  // namespace jtsmm
