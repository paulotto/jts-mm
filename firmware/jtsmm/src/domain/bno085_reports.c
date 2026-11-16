// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file bno085_reports.c
 * @brief BNO085 report catalog tables.
 */

#include "jtsmm/domain/bno085_reports.h"

#include "jtsmm/config.h"
#include "jtsmm/records.h"
#include <string.h>

/*
 * The seven BNO085 datasheet motion/rotation outputs (Adafruit_4754_BNO085.pdf
 * §"Motion Vectors" / "Rotation Vectors"). payload_len is the tagged USB record
 * length: RAW = 1 (sensor_id) + 3*int16 = 7 B; QUAT = 1 (report_id) + 4*f32 =
 * 17 B; LIN_ACCEL = 3*f32 = 12 B (records.h). max_rate_hz reflects the sensor
 * ceiling; fusion tops near ~398 Hz so we cap declared fusion at 400.
 */
static const bno085_report_entry_t catalog[] = {
    {0x01, "accelerometer", REC_IMU_MTA_RAW, 100, 500, IMU_RAW_PAYLOAD_LEN},
    {0x02, "gyroscope", REC_IMU_MTA_RAW, 100, 400, IMU_RAW_PAYLOAD_LEN},
    {0x03, "magnetometer", REC_IMU_MTA_RAW, 100, 100, IMU_RAW_PAYLOAD_LEN},
    {0x04, "linear_acceleration", REC_IMU_MTA_LIN_ACCEL, 100, 400, IMU_LIN_ACCEL_PAYLOAD_LEN},
    {0x05, "rotation_vector", REC_IMU_MTA_QUAT, 100, 400, IMU_QUAT_PAYLOAD_LEN},
    {0x08, "game_rotation_vector", REC_IMU_MTA_QUAT, 100, 400, IMU_QUAT_PAYLOAD_LEN},
    {0x09, "geomagnetic_rotation_vector", REC_IMU_MTA_QUAT, 100, 400, IMU_QUAT_PAYLOAD_LEN},
    {0x2A, "gyro_integrated_rotation_vector", REC_IMU_MTA_QUAT, 200, 1000, IMU_QUAT_PAYLOAD_LEN},
};

const bno085_report_entry_t *bno085_report_catalog(size_t *count) {
    if (count) {
        *count = sizeof(catalog) / sizeof(catalog[0]);
    }
    return catalog;
}

const bno085_report_entry_t *bno085_report_find(uint8_t report_id) {
    for (size_t i = 0; i < sizeof(catalog) / sizeof(catalog[0]); i++) {
        if (catalog[i].report_id == report_id) {
            return &catalog[i];
        }
    }
    return NULL;
}

const bno085_report_entry_t *bno085_report_find_by_name(const char *name) {
    if (name == NULL) {
        return NULL;
    }
    for (size_t i = 0; i < sizeof(catalog) / sizeof(catalog[0]); i++) {
        if (catalog[i].name && strcmp(catalog[i].name, name) == 0) {
            return &catalog[i];
        }
    }
    return NULL;
}

uint8_t bno085_fusion_report_id(imu_fusion_primary_t primary) {
    switch (primary) {
    case IMU_FUSION_ROTATION_VECTOR:
        return 0x05;
    case IMU_FUSION_GEOMAGNETIC_ROTATION_VECTOR:
        return 0x09;
    case IMU_FUSION_GYRO_INTEGRATED_RV:
        return 0x2A;
    default:
        return 0x08;
    }
}
