// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file ads1299.h
 * @brief ADS1299 SPI driver: config, DRDY, streaming, lead-off helpers.
 *
 * @details Used by the EMG domain layer on the sensor core. Streaming uses
 *          SPI DMA (@ref spi_dma_kick_read) kicked from DRDY.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

#define ADS1299_CHIP_ID_EXPECT 0x3E /**< Expected ID register value. */

/** @brief Configure SPI pins / mode for the ADS1299. @return true on success. */
bool ads1299_bus_init(void);

/** @brief Program registers from current setter state; leave ready to stream. @return true on success. */
bool ads1299_init(void);

/**
 * @brief Enter continuous conversion / streaming mode.
 *
 * @details No-op returning true if already streaming (no START, no SPI retime,
 *          no counter reset). Refuses (false) unless the last @ref ads1299_init
 *          succeeded.
 * @return true if streaming.
 */
bool ads1299_start_stream(void);

/**
 * @brief true if the last @ref ads1299_init completed and verified the chip.
 *
 * @details Latched: cleared when an init starts, set only when it succeeds.
 *          Command paths use it to NAK a stream start after a failed apply.
 */
bool ads1299_init_ok(void);

/** @brief Stop continuous conversion. */
void ads1299_stop_stream(void);

/** @brief Live check: CONFIG3 read back from the chip matches the configured bias mode. */
bool ads1299_config_ok(void);

/** @brief Last init stage index (debug; implementation-defined). */
int ads1299_init_stage(void);

/**
 * @brief Read device ID into @p id_out.
 * @param[out] id_out Destination; must not be NULL.
 * @return true if the SPI read succeeded.
 */
bool ads1299_read_id(uint8_t *id_out);

/** @brief Raw ID register read (0 on failure). */
uint8_t ads1299_read_id_raw(void);

/** @brief Log ADS status registers over CDC/USB. */
void ads1299_status(void);

/** @brief Bench helper: watch DRDY edges and log. */
void ads1299_drdy_watch(void);

/** @brief Dump key registers over CDC/USB. */
void ads1299_regs(void);

/**
 * @brief Enable or disable the internal test signal.
 * @param enable true to enable.
 * @return true on success.
 */
bool ads1299_test_signal(bool enable);

/** @brief Configured sample rate (Hz). */
uint32_t ads1299_sample_rate_hz(void);

/**
 * @brief SPI0 rate used while streaming: platform spi0_stream_hz capped at
 *        ADS_SPI_STREAM_MAX_HZ (4 MHz, tSDECODE for the gapless DMA frame).
 */
uint32_t ads1299_stream_spi_hz(void);

/**
 * @brief Set sample rate before @ref ads1299_init (250/500/1000/2000 Hz; else 250).
 * @param fs_hz Desired SPS.
 */
void ads1299_set_sample_rate_hz(uint32_t fs_hz);

/**
 * @brief Set PGA gain for muscle channels before init (1/2/4/6/8/12/24).
 * @param gain Gain code.
 */
void ads1299_set_pga_gain(uint8_t gain);

/** @brief Current muscle-channel PGA gain. */
uint8_t ads1299_pga_gain(void);

/**
 * @brief Enable/disable bias (DRL) drive before init.
 * @param enable true to enable.
 */
void ads1299_set_bias_drive(bool enable);

/** @brief true when bias/DRL drive is configured. */
bool ads1299_bias_drive(void);

/**
 * @brief Set BIAS_SENSP/N mask (bit0=CH1 … bit5=CH6) before init.
 * @param mask_ch1_to_ch6 Channel mask.
 */
void ads1299_set_bias_channels(uint8_t mask_ch1_to_ch6);

/** @brief Current bias channel mask (lower six bits). */
uint8_t ads1299_bias_channels(void);

/**
 * @brief Power-down mask over CH1–CH6 (bit0=CH1) before init.
 *
 * @details Powered-down muscle channels are set to the ADS "off" input so
 *          unused/floating inputs do not add noise.
 */
void ads1299_set_pd_channels(uint8_t mask_ch1_to_ch6);

/** @brief Channels currently powered down (lower six bits). */
uint8_t ads1299_pd_channels(void);

/**
 * @brief Force CH8SET (DRL/IN8N) out of BIAS_DRN into a plain electrode input.
 *
 * @details So lead-off sees the real patient electrode instead of the bias
 *          amplifier output; only meaningful with bias drive off. Used by the
 *          isolated DRL probe in `emg check`.
 */
void ads1299_set_drl_probe(bool enable);

/** @brief true when DRL probe mode is active. */
bool ads1299_drl_probe(void);

/** @brief Install the DRDY GPIO IRQ handler (left masked: DRDY is polled, see ads1299_drdy_service()). */
void ads1299_drdy_isr_install(void);

/**
 * @brief Enable or mask the DRDY IRQ.
 * @param on true to enable.
 */
void ads1299_drdy_isr_enable(bool on);

/**
 * @brief Conversions lost this stream (never read), estimated from DRDY timing.
 *
 * @details The same count advances the acquisition seq carried in every
 *          REC_EMG_BLOCK, so host seq gaps include these (contract K1).
 */
uint32_t ads1299_missed_drdy_count(void);

/** @brief Maximum observed gap between DRDY edges (µs). */
uint32_t ads1299_max_drdy_gap_us(void);

/** @brief Count of frame misalignment detections. */
uint32_t ads1299_frame_misalign_count(void);

/**
 * @brief Decode lead-off / framing from a streamed status word.
 *
 * @details Called from the EMG DMA-complete callback on the sensor core.
 * @param status Three status bytes from the ADS frame; must not be NULL.
 * @return true if the status prefix was the fixed 0b1100, i.e. the 24 sample
 *         bytes behind it are real data. false means the RDATA byte stream
 *         slipped and the frame must not be published: a one-byte offset scales
 *         samples by ~256 and rotates them between channels, so it reads as
 *         full-scale noise rather than as a dropout.
 */
bool ads1299_stream_status_update(const uint8_t *status);

/**
 * @brief Recover RDATA framing once misalignment has persisted.
 *
 * @details No-op until @c ADS_RESYNC_STREAK_TRIP consecutive frames have failed
 *          @ref ads1299_stream_status_update, so single-frame glitches cost
 *          nothing. Sensor-core safe: never touches USB and is time-bounded.
 * @return true if a resync was performed.
 */
bool ads1299_stream_resync_if_needed(void);

/**
 * @brief Longest run of consecutive misaligned frames seen this stream.
 *
 * @details Cleared only by a correctly framed frame, not by a resync, so this
 *          measures how long the stream was actually unusable rather than how
 *          often @ref ads1299_stream_resync_if_needed intervened.
 */
uint32_t ads1299_frame_misalign_max_streak(void);

/** @brief Framing resyncs performed this stream. */
uint32_t ads1299_frame_resync_count(void);

/** @brief Times resync was skipped because the per-stream retry cap was hit. */
uint32_t ads1299_frame_resync_giveup_count(void);

/** @brief First status byte of the most recent misaligned frame (diagnostic). */
uint8_t ads1299_frame_misalign_last_status(void);

/**
 * @brief true while the DRDY-driven EMG stream is active.
 *
 * @details Commands that drive SPI0 from the comms core must consult this: the
 *          sensor core owns the bus while streaming and there is no mutex.
 */
bool ads1299_drdy_streaming(void);

/**
 * @brief Configure the EMG framing guard (both default on).
 *
 * @param drop   Discard misaligned frames instead of publishing them.
 * @param resync Attempt SPI/DMA + SDATAC recovery on a sustained slip.
 */
void ads1299_set_misalign_guard(bool drop, bool resync);

/** @brief true if misaligned frames are being dropped. */
bool ads1299_misalign_guard_drop(void);

/** @brief true if automatic framing resync is enabled. */
bool ads1299_misalign_guard_resync(void);

/** @brief Cumulative DRDY edges observed. */
uint32_t ads1299_drdy_edge_count(void);

/**
 * @brief Register callback invoked with DRDY edge count (divisor sync).
 * @param cb Callback; may be NULL to clear.
 */
void ads1299_set_drdy_divisor_cb(void (*cb)(uint32_t count));

/** @brief Poll the DRDY level and kick the SPI DMA read (sensor core; the DRDY IRQ is unused). */
void ads1299_drdy_service(void);

/** @brief Enable hardware lead-off monitor. */
void ads1299_set_leadoff_monitor(bool enable);

/** @brief Keep lead-off active during recording when true. */
void ads1299_set_leadoff_during_recording(bool during);

/** @brief Set lead-off current (nA, device-coded). */
void ads1299_set_leadoff_current_na(uint8_t current_na);

/** @brief Enable AC lead-off tone when true. */
void ads1299_set_leadoff_ac_tone(bool ac_tone);

/** @brief true if lead-off hardware path is currently active. */
bool ads1299_leadoff_hw_active(void);

/** @brief Poll lead-off status / emit ADS status records as needed. */
void ads1299_leadoff_poll(void);
