// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file record_decode.hpp
 * @brief Payload decode helpers for the native hot path (no Python objects).
 *
 * @details Shared by the pybind module and HDF5 writer. Unit conversions mirror
 *          `jtsmm/imu/units.py` and ADS1299 RDATA byte order in firmware.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include "jtsmm_native/record.hpp"

namespace jtsmm {

/**
 * @brief Sign-extend a 24-bit two's-complement value packed in 32 bits.
 * @param raw Lower 24 bits of an ADS channel word.
 */
inline int32_t sign_extend24(uint32_t raw) {
    if (raw & 0x800000u) {
        return static_cast<int32_t>(raw) - (1 << 24);
    }
    return static_cast<int32_t>(raw);
}

/**
 * @brief Read a big-endian 24-bit ADS1299 channel count.
 * @param o Pointer to 3 bytes; must not be null.
 */
inline uint32_t read_emg_count24_be(const uint8_t *o) {
    return (static_cast<uint32_t>(o[0]) << 16) | (static_cast<uint32_t>(o[1]) << 8) |
           static_cast<uint32_t>(o[2]);
}

/** @brief Read little-endian float32. */
inline float read_f32(const uint8_t *p) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
}

/** @brief Read little-endian int16. */
inline int16_t read_i16(const uint8_t *p) {
    return static_cast<int16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

/** @brief Read little-endian uint64. */
inline uint64_t read_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

/**
 * @brief SH-2 fusion report id → datasheet name (mirrors records.FUSION_REPORT_IDS).
 * @return nullptr for an unknown id.
 */
inline const char *fusion_report_name(uint8_t report_id) {
    switch (report_id) {
        case 0x05: return "rotation_vector";
        case 0x08: return "game_rotation_vector";
        case 0x09: return "geomagnetic_rotation_vector";
        case 0x2A: return "gyro_integrated_rotation_vector";
        default: return nullptr;
    }
}

/** @brief BNO085 accel Q8 count → m/s². */
inline float accel_count_to_si(int16_t c) { return static_cast<float>(c) / 256.0f; }
/** @brief BNO085 gyro Q9 count → rad/s. */
inline float gyro_count_to_si(int16_t c) { return static_cast<float>(c) / 512.0f; }
/** @brief BNO085 mag Q4 count → µT. */
inline float mag_count_to_si(int16_t c) { return static_cast<float>(c) / 16.0f; }

/**
 * @brief Decode @p rec and append the corresponding row(s) to @p writer.
 * @return true when the record was written to the session file.
 */
bool append_record_to_session(class Hdf5SessionWriter &writer, const Record &rec);

}  // namespace jtsmm
