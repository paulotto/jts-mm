// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config.h
 * @brief MCU configuration types mirroring schema/v1.
 *
 * @details Active config is held in RAM and applied via config_apply /
 *          domain helpers. Host YAML/JSON maps into these structs through
 *          config_json. See docs/software/s08_runtime_configuration.md.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Platform clock / ring / IRQ profile selector.
 */
typedef enum {
    PLATFORM_PROFILE_DEFAULT = 0,      /**< Nominal dual-core production clocks. */
    PLATFORM_PROFILE_HIGH_THROUGHPUT,  /**< Higher SPI/USB-oriented settings. */
    PLATFORM_PROFILE_DEBUG,            /**< Debug-friendly clocks / priorities. */
    PLATFORM_PROFILE_CUSTOM,           /**< Explicit field values from JSON. */
} platform_profile_t;

/**
 * @brief EMG acquisition profile selector.
 */
typedef enum {
    EMG_PROFILE_ACQUISITION = 0,         /**< Default science capture (~2 kHz). */
    EMG_PROFILE_HIGH_RATE,               /**< Higher sample-rate profile. */
    EMG_PROFILE_BENCH,                   /**< Bench / bring-up defaults. */
    EMG_PROFILE_NOISE_CHARACTERIZATION,  /**< High-rate noise lab profile. */
    EMG_PROFILE_CUSTOM,                  /**< Explicit field values from JSON. */
} emg_profile_t;

/**
 * @brief Per-IMU-slot report profile selector.
 */
typedef enum {
    IMU_PROFILE_FUSED_ONLY = 0,       /**< Fusion quaternion only. */
    IMU_PROFILE_RAW_ONLY,             /**< Raw accel/gyro/mag only. */
    IMU_PROFILE_KINEMATICS_DEFAULT,   /**< Fusion quaternion + linear acceleration. */
    IMU_PROFILE_ALL,                  /**< Enable all configured reports. */
    IMU_PROFILE_CUSTOM,               /**< Explicit report table from JSON. */
    IMU_PROFILE_KINEMATICS_ACCEL,     /**< Fusion quaternion + gravity-included accel (default). */
} imu_profile_t;

/**
 * @brief How IMU samples are paced relative to EMG.
 */
typedef enum {
    IMU_SYNC_INT_DRIVEN = 0,     /**< Use BNO085 INT / poll service. */
    IMU_SYNC_EMG_DRDY_DIVISOR,   /**< Pace from EMG DRDY divisor. */
} imu_sync_mode_t;

/**
 * @brief Primary BNO085 fusion report family.
 */
typedef enum {
    IMU_FUSION_GAME_ROTATION_VECTOR = 0,     /**< Game RV (no mag). */
    IMU_FUSION_ROTATION_VECTOR,              /**< Absolute RV. */
    IMU_FUSION_GEOMAGNETIC_ROTATION_VECTOR,  /**< Geomagnetic RV. */
    IMU_FUSION_GYRO_INTEGRATED_RV,           /**< Gyro-integrated RV. */
} imu_fusion_primary_t;

/**
 * @brief Camera trigger / LED profile selector.
 */
typedef enum {
    TRIGGER_PROFILE_IMX296_DUAL_CSI = 0, /**< Default Track B 2× CSI @ 30 Hz, 12 ms. */
    TRIGGER_PROFILE_IMX296_HIGH_FPS,     /**< Higher FPS Track B. */
    TRIGGER_PROFILE_BENCH_SCOPE,         /**< Slow scope-friendly pulses. */
    TRIGGER_PROFILE_TRACK_A_LED,         /**< Track A webcams @ 30 Hz (sync LED via `flash on`). */
    TRIGGER_PROFILE_CUSTOM,              /**< Explicit rate/width/outputs. */
} trigger_profile_t;

/**
 * @brief Platform clocks, ring sizes, and IRQ priorities.
 */
typedef struct {
    uint32_t cpu_hz;               /**< CPU clock (Hz). */
    uint32_t xosc_hz;              /**< Crystal frequency (Hz). */
    uint8_t core_layout;           /**< 0 = dual-core, 1 = single-core. */
    uint32_t spi0_stream_hz;       /**< ADS SPI bitrate while streaming. */
    uint32_t spi0_cfg_hz;          /**< ADS SPI bitrate during register config. */
    uint32_t i2c_hz;               /**< IMU I2C bitrate. */
    uint32_t pio_trigger_clock_hz; /**< PIO clock for trigger_gen (typically 1e6). */
    uint16_t emg_ring_slots;       /**< @c RING_EMG capacity. */
    uint16_t imu_ring_slots;       /**< Per-IMU ring capacity. */
    uint16_t event_ring_slots;     /**< @c RING_EVENTS capacity. */
    uint8_t irq_emg_priority;      /**< NVIC priority for EMG path. */
    uint8_t irq_imu_priority;      /**< NVIC priority for IMU path. */
    uint8_t irq_trigger_priority;  /**< NVIC priority for trigger path. */
} platform_config_t;

/**
 * @brief Per-channel EMG settings (ADS1299 CH1–CH8).
 */
typedef struct {
    bool enabled;      /**< Channel participates in stream when true. */
    uint8_t gain;      /**< PGA gain code (device-specific). */
    bool power_down;   /**< Power-down this channel. */
    uint8_t input_mux; /**< Input mux / CHnSET bits. */
} emg_channel_cfg_t;

/**
 * @brief Full EMG (ADS1299) configuration block.
 */
typedef struct {
    bool enabled;              /**< EMG subsystem enabled. */
    emg_profile_t profile;     /**< Profile selector. */
    uint16_t sample_rate_sps;  /**< Sample rate (SPS). */
    uint8_t default_gain;      /**< Default PGA for muscle channels. */
    struct {
        bool enabled;          /**< Bias / DRL drive enabled. */
        uint8_t channel_mask;  /**< BIAS_SENSP/N mask (bit0=CH1 …). */
        bool ch7_readback;     /**< Expose bias channel readback. */
    } bias;
    uint8_t pd_channel_mask; /**< CH1–CH6 power-down mask (bit0=CH1); 0 = all active. */
    struct {
        bool enabled;           /**< Lead-off detection enabled. */
        bool during_recording;  /**< Keep lead-off active while streaming. */
        bool ac_tone;           /**< AC lead-off tone. */
        uint8_t current_na;     /**< Lead-off current (nA), device-coded. */
        /**
         * Forces CH8 (DRL/IN8N) out of BIAS_DRN into a plain electrode input
         * for the isolated DRL probe in `emg check` (host must also set
         * bias.enabled=false); see ads1299.c ads_write_spare_channels().
         */
        bool probe_drl;
    } leadoff;
    struct {
        bool enabled;         /**< Internal test signal. */
        uint8_t amplitude_mv; /**< Amplitude (mV), device-coded. */
    } test_signal;
    emg_channel_cfg_t channels[8]; /**< Per-channel overrides. */
    uint8_t stream_channels;       /**< 6 or 8 channels streamed. */
} emg_config_t;

/**
 * @brief One enabled BNO085 report in a slot's report table.
 */
typedef struct {
    bool enabled;       /**< Report active. */
    uint8_t report_id;  /**< SH-2 report id. */
    uint16_t rate_hz;   /**< Requested rate (Hz). */
    uint8_t usb_type;   /**< USB @c REC_* type for this report. */
    uint16_t payload_len; /**< Expected payload length (bytes). */
} imu_report_cfg_t;

#define IMU_MAX_REPORTS 8 /**< Max reports per IMU slot. */

/**
 * @brief Configuration for one IMU slot (MTA or CRA).
 */
typedef struct {
    bool enabled;                      /**< Slot enabled. */
    imu_profile_t profile;             /**< Profile selector. */
    imu_fusion_primary_t fusion_primary; /**< Primary fusion family. */
    uint16_t fusion_rate_hz;           /**< Fusion rate (Hz). */
    imu_sync_mode_t sync_mode;         /**< Sync / pacing mode. */
    uint8_t emg_drdy_divisor;          /**< DRDY divisor when sync_mode uses it. */
    /**
     * BNO085 motion-engine dynamic calibration per sensor (SH-2 Configure ME Calibration),
     * issued on every apply and after a device reset. Default all on: every recording up
     * to 2026-11-02 ran with it, and additions to that dataset keep it; off = the saved
     * calibration (imu_savecal, DCD) stays fixed during a take (cv02 §7.1).
     */
    struct {
        bool accel; /**< Accelerometer dynamic calibration. */
        bool gyro;  /**< Gyroscope dynamic calibration (in-motion; still-ZRO is automatic). */
        bool mag;   /**< Magnetometer dynamic calibration (unused by game RV). */
    } dynamic_calibration;
    uint8_t num_reports;               /**< Valid entries in @c reports[]. */
    imu_report_cfg_t reports[IMU_MAX_REPORTS]; /**< Explicit report table. */
} imu_slot_config_t;

/**
 * @brief Camera trigger / sync-LED configuration.
 */
typedef struct {
    bool enabled;              /**< Trigger subsystem enabled. */
    trigger_profile_t profile; /**< Profile selector. */
    uint16_t rate_hz;          /**< Pulse rate (Hz). */
    uint32_t pulse_width_us;   /**< Pulse width (µs); IMX296 fast = low time. */
    uint8_t polarity;          /**< 0 = imx296_fast, 1 = active_high. */
    struct {
        bool csi1;     /**< CSI1 XTRIG fan-out (hard-wired on; only true accepted). */
        bool csi2;     /**< CSI2 XTRIG fan-out (hard-wired on; only true accepted). */
        bool scope;    /**< Scope / test-pad tap (hard-wired on; only true accepted). */
        bool sync_led; /**< Never driven by the trigger (only false accepted). */
    } outputs;
    uint8_t start_on; /**< Autostart policy (implementation-defined). */
} trigger_config_t;

/**
 * @brief Complete MCU configuration snapshot.
 */
typedef struct {
    uint8_t schema_version;      /**< Config schema version. */
    platform_config_t platform;  /**< Platform clocks / rings. */
    emg_config_t emg;            /**< EMG block. */
    imu_slot_config_t imu_mta;   /**< Metatarsal IMU. */
    imu_slot_config_t imu_cra;   /**< Cranial / second IMU. */
    trigger_config_t trigger;    /**< Trigger / LED. */
    bool stream_emg;             /**< Emit EMG records when capturing. */
    bool stream_imu_mta;         /**< Emit MTA IMU records when capturing. */
    bool stream_imu_cra;         /**< Emit CRA IMU records when capturing. */
} jtsmm_mcu_config_t;

/**
 * @brief Fill @p cfg with factory defaults for all blocks.
 *
 * @param cfg Output config; must not be NULL.
 */
void jtsmm_config_set_defaults(jtsmm_mcu_config_t *cfg);

/**
 * @brief Pointer to the active (applied) MCU config.
 *
 * @return Non-NULL pointer to the live config snapshot.
 *
 * @note Treat as read-only outside config_apply / domain apply paths.
 */
const jtsmm_mcu_config_t *jtsmm_config_active(void);

/**
 * @brief Mutable pointer to the working MCU config.
 *
 * @return Non-NULL pointer for in-place edits before apply/save.
 */
jtsmm_mcu_config_t *jtsmm_config_mutable(void);
