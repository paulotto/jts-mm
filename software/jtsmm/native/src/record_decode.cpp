// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file record_decode.cpp
 * @brief Record → HDF5 append dispatch.
 */

#include "jtsmm_native/record_decode.hpp"

#include "jtsmm_native/hdf5_session.hpp"

#include <cstring>

namespace jtsmm {

namespace {

/** SH-2 calibration status (bits 1:0) at byte @p idx, or "absent" for older firmware. */
uint8_t accuracy_at(const uint8_t *p, size_t n, size_t idx) {
    return n > idx ? static_cast<uint8_t>(p[idx] & 0x03u) : Hdf5SessionWriter::kAccuracyAbsent;
}

}  // namespace

bool append_record_to_session(Hdf5SessionWriter &writer, const Record &rec) {
    const uint8_t *p = rec.payload.data();
    const size_t n = rec.payload.size();
    switch (rec.record_type) {
        case REC_EMG_BLOCK: {
            // 27 B, or 31 B with a trailing u32 acquisition seq (K1).
            if (n != 27 && n != 31) {
                return false;
            }
            int32_t ch[8];
            for (int i = 0; i < 8; ++i) {
                const uint8_t *o = p + 3 + i * 3;
                ch[i] = sign_extend24(read_emg_count24_be(o));
            }
            // ADS1299 status word (first 3 payload bytes), muscle CH1-6 only
            // (& 0x3F drops bit6=CH7/spare and bit7=CH8/DRL -- see
            // hdf5_session.cpp's leadoff_status_note for why DRL is excluded).
            const uint8_t statp = static_cast<uint8_t>(((p[0] & 0x0Fu) << 4) | (p[1] >> 4)) & 0x3Fu;
            const uint8_t statn = static_cast<uint8_t>(((p[1] & 0x0Fu) << 4) | (p[2] >> 4)) & 0x3Fu;
            const bool has_seq = (n == 31);
            const uint32_t seq = has_seq ? static_cast<uint32_t>(p[27]) |
                                               (static_cast<uint32_t>(p[28]) << 8) |
                                               (static_cast<uint32_t>(p[29]) << 16) |
                                               (static_cast<uint32_t>(p[30]) << 24)
                                         : 0u;
            writer.append_emg(rec.t_master_us, ch, statp, statn, has_seq, seq);
            return true;
        }
        case REC_IMU_MTA_QUAT:
        case REC_IMU_CRA_QUAT: {
            const bool cranium = (rec.record_type == REC_IMU_CRA_QUAT);
            // Tagged (17/18 B): [report_id][w,x,y,z][acc?]. Legacy (16 B): untagged.
            if (n >= 17) {
                const uint8_t report_id = p[0];
                const float q[4] = {read_f32(p + 1), read_f32(p + 5), read_f32(p + 9),
                                    read_f32(p + 13)};
                writer.append_quat(cranium, rec.t_master_us, q, report_id, accuracy_at(p, n, 17));
                return true;
            }
            if (n >= 16) {
                const float q[4] = {read_f32(p), read_f32(p + 4), read_f32(p + 8),
                                    read_f32(p + 12)};
                writer.append_quat(cranium, rec.t_master_us, q, 0);
                return true;
            }
            return false;
        }
        case REC_IMU_MTA_RAW:
        case REC_IMU_CRA_RAW: {
            const bool cranium = (rec.record_type == REC_IMU_CRA_RAW);
            // Tagged (7/8 B): [sensor_id][x,y,z][acc?]. Legacy (18 B): accel+gyro+mag.
            if (n >= 7 && n < 18) {
                const uint8_t sensor_id = p[0];
                const int16_t x = read_i16(p + 1), y = read_i16(p + 3), z = read_i16(p + 5);
                float xyz[3];
                if (sensor_id == 0x01) {
                    xyz[0] = accel_count_to_si(x); xyz[1] = accel_count_to_si(y);
                    xyz[2] = accel_count_to_si(z);
                } else if (sensor_id == 0x02) {
                    xyz[0] = gyro_count_to_si(x); xyz[1] = gyro_count_to_si(y);
                    xyz[2] = gyro_count_to_si(z);
                } else if (sensor_id == 0x03) {
                    xyz[0] = mag_count_to_si(x); xyz[1] = mag_count_to_si(y);
                    xyz[2] = mag_count_to_si(z);
                } else {
                    return false;
                }
                writer.append_imu_motion(cranium, sensor_id, rec.t_master_us, xyz,
                                         accuracy_at(p, n, 7));
                return true;
            }
            if (n >= 18) {
                const float accel[3] = {accel_count_to_si(read_i16(p)),
                                        accel_count_to_si(read_i16(p + 2)),
                                        accel_count_to_si(read_i16(p + 4))};
                const float gyro[3] = {gyro_count_to_si(read_i16(p + 6)),
                                       gyro_count_to_si(read_i16(p + 8)),
                                       gyro_count_to_si(read_i16(p + 10))};
                const float mag[3] = {mag_count_to_si(read_i16(p + 12)),
                                      mag_count_to_si(read_i16(p + 14)),
                                      mag_count_to_si(read_i16(p + 16))};
                writer.append_imu_motion(cranium, 0x01, rec.t_master_us, accel);
                writer.append_imu_motion(cranium, 0x02, rec.t_master_us, gyro);
                writer.append_imu_motion(cranium, 0x03, rec.t_master_us, mag);
                return true;
            }
            return false;
        }
        case REC_IMU_MTA_LIN_ACCEL:
        case REC_IMU_CRA_LIN_ACCEL: {
            if (n < 12) {
                return false;
            }
            const bool cranium = (rec.record_type == REC_IMU_CRA_LIN_ACCEL);
            const float xyz[3] = {read_f32(p), read_f32(p + 4), read_f32(p + 8)};
            writer.append_lin_accel(cranium, rec.t_master_us, xyz, accuracy_at(p, n, 12));
            return true;
        }
        case REC_TRIGGER_EVENT: {
            if (n < 12) {
                return false;
            }
            writer.append_trigger(rec.t_master_us, read_u64(p + 4));
            return true;
        }
        default:
            return false;
    }
}

}  // namespace jtsmm
