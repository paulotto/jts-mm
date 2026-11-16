// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_config.h
 * @brief Domain layer: apply IMU slot config, stream SHTP into rings, recovery.
 */
#pragma once

#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/config.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Apply @p cfg to one IMU slot (features, rates, sync).
 *
 * @details A disabled @p cfg is stored too and turns off the reports a
 *          previous apply enabled. A device that is absent or fails to boot
 *          is reported on CDC and in a host log line, but still returns true:
 *          a false return fails the whole config apply, which then does not
 *          launch the sensor core (no EMG either). Callers should call this
 *          for both slots, enabled or not.
 * @param slot IMU slot (MTA/CRA).
 * @param cfg  Slot config; must not be NULL.
 * @return true unless the arguments are invalid.
 */
bool imu_config_apply_slot(imu_slot_t slot, const imu_slot_config_t *cfg);

/**
 * @brief Validate @p cfg against @p emg_fs_hz (for DRDY-divisor sync).
 * @details Rejects (with a CDC reason) enabled reports with rate 0, rates
 *          above the catalog maximum or unknown ids, fusion rate 0 for a
 *          fusion profile, and a DRDY divisor that does not equal
 *          emg_fs / rate exactly (full width, no uint8 truncation).
 * @param cfg       Slot config; must not be NULL.
 * @param emg_fs_hz Active EMG sample rate (SPS).
 * @return true if values are supported (always for a disabled slot).
 */
bool imu_config_validate_slot(const imu_slot_config_t *cfg, uint16_t emg_fs_hz);

/**
 * @brief Expand @p cfg->profile into concrete report table when not CUSTOM.
 * @param cfg In/out slot config; must not be NULL.
 */
void imu_config_resolve_profile(imu_slot_config_t *cfg);

/**
 * @brief I2C completion hook: parse SHTP and push IMU USB records.
 *
 * @details One record per report; each record's t_master_us is its sample
 *          time, @p t_us corrected by the SH-2 base delta / report delay
 *          (imu_timing.h; not in emg_drdy_divisor mode).
 * @param slot IMU slot.
 * @param t_us INT assertion time of the read (edge latch, else poll time), µs.
 * @param pkt  SHTP packet; must not be NULL.
 * @param len  Packet length.
 */
void imu_stream_on_complete(imu_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len);

/**
 * @brief Start streaming for @p slot.
 * @return false if the slot is disabled in the applied config (refused, with a
 *         CDC message) or its bus could not be taken over; callers should then
 *         not report the stream as on.
 */
bool imu_stream_start_slot(imu_slot_t slot);

/** @brief Stop streaming for @p slot. */
void imu_stream_stop_slot(imu_slot_t slot);

/**
 * @brief EMG-DRDY divisor kick when @c IMU_SYNC_EMG_DRDY_DIVISOR is active.
 * @param drdy_count Cumulative DRDY edges.
 */
void imu_drdy_divisor_kick(uint32_t drdy_count);

/** @brief Poll INT-pending slots and service I2C (sensor core). */
void imu_stream_poll_service(void);

/** @brief Log IMU status over CDC/USB. */
void imu_config_status(void);

/**
 * @brief Persist each enabled IMU's dynamic calibration (Save DCD) to flash.
 *
 * @note Requires the slot to be idle (not streaming).
 */
void imu_config_save_calibration(void);

/**
 * @brief Cooperative IMU recovery for a stalled slot.
 *
 * @details @ref imu_slot_recover_start arms recovery; @ref imu_recover_poll
 *          (every comms-loop iteration) advances one bounded I2C step so USB
 *          drain continues between steps.
 */
void imu_slot_recover_start(imu_slot_t slot);

/** @brief Progress cooperative recovery one step (comms core). */
void imu_recover_poll(void);

/** @brief true while cooperative recovery is in progress. */
bool imu_recover_busy(void);

/**
 * @brief SH-2 reports lost on @p slot since boot (REC_STATS, contract K2).
 *
 * @details From the per-report sequence number of every input report (gaps
 *          between consecutive reports of one id); for channel-5 gyro RV,
 *          lost SHTP packets. Covers loss between the BNO085 and the ring
 *          (I2C aborts, dropped packets), not ring/USB drops. ~0 when healthy.
 * @param slot 0 = MTA, 1 = CRA.
 */
uint32_t imu_config_seq_gaps(uint8_t slot);

/**
 * @brief Spontaneous BNO085 resets seen on @p slot while streaming (K2).
 *
 * @details Each one is followed by an automatic report re-enable from the
 *          comms core (imu_recover_poll).
 * @param slot 0 = MTA, 1 = CRA.
 */
uint32_t imu_config_reset_count(uint8_t slot);

/**
 * @brief Report interval the BNO085 granted for @p report_id on @p slot (µs).
 *
 * @details From the Get Feature Response (0xFC) to Set Feature at apply, kept
 *          current from the stream. The sensor quantizes requested rates, so
 *          this -- not the requested rate -- is what a rate check should
 *          expect. Also printed by `imu_status`.
 * @return 0 if unknown (no response seen, report not configured).
 */
uint32_t imu_config_granted_interval_us(uint8_t slot, uint8_t report_id);
