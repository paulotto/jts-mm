// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file crc16.hpp
 * @brief CRC16-CCITT (poly 0x1021, init 0xFFFF) for USB frame validation.
 *
 * @details Mirrors firmware `host_link` and Python `protocol.crc.crc16_ccitt`
 *          so frames validate identically on both paths.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace jtsmm {

/**
 * @brief Compute CRC16-CCITT over @p data.
 *
 * @param data Bytes to checksum; may be nullptr if @p len is 0.
 * @param len  Number of bytes.
 * @param init Initial CRC (default 0xFFFF).
 * @return CRC16 value.
 */
inline uint16_t crc16_ccitt(const uint8_t *data, size_t len, uint16_t init = 0xFFFFu) {
    uint16_t crc = init;
    for (size_t i = 0; i < len; ++i) {
        crc ^= static_cast<uint16_t>(data[i]) << 8;
        for (int b = 0; b < 8; ++b) {
            if (crc & 0x8000u) {
                crc = static_cast<uint16_t>((crc << 1) ^ 0x1021u);
            } else {
                crc = static_cast<uint16_t>(crc << 1);
            }
        }
    }
    return crc;
}

}  // namespace jtsmm
