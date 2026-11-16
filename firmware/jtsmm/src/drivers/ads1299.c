// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file ads1299.c
 * @brief ADS1299 register/stream/lead-off driver.
 */

#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/drivers/ads1299_regs.h"

#include "jtsmm/drivers/panel.h"
#include "jtsmm/config.h"
#include "jtsmm/pins.h"
#include "jtsmm/platform/gpio_irq.h"
#include "jtsmm/platform/spi_dma.h"
#include "jtsmm/runtime/timebase.h"
#include "jtsmm/runtime/usb_service.h"

#include "hardware/gpio.h"
#include "hardware/irq.h"
#include "hardware/spi.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

#define ADS_SPI spi0
static uint32_t ads_spi_cfg_hz = 500000u;
static uint32_t ads_spi_run_hz = 4000000u;

#define ADS_CMD_WAKEUP  0x02
#define ADS_CMD_STANDBY 0x04
#define ADS_CMD_RESET   0x06
#define ADS_CMD_START   0x08
#define ADS_CMD_STOP    0x0A
#define ADS_CMD_RDATAC  0x10
#define ADS_CMD_SDATAC  0x11
#define ADS_CMD_RDATA   0x12

static uint32_t stream_fs_hz = 250u;
static uint8_t muscle_pga_gain = 6u;
static uint8_t muscle_pd_mask = 0u;
static bool bias_drive_enabled = false;
static uint8_t bias_channel_mask = 0x3Fu;
static bool leadoff_monitor_enabled = false;
static bool leadoff_during_recording = false;
static bool leadoff_ac_tone = false;
static uint8_t leadoff_current_na = 6u;
/* Forces CH8SET back to a plain electrode input (MUX=000) and keeps the
 * LOFF_SENSN[7] excitation on even though `bias_drive_enabled` is false --
 * see ads_leadoff_channel_masks() and ads_ch8set(). Needed
 * because during normal DRL operation (bias on, MUX8=111 "BIAS_DRN") the
 * ADS1299's own bias amplifier actively drives IN8N through a low-impedance
 * closed loop, so the lead-off comparator/ADC just see the amplifier's own
 * output and never reflect whether the external mastoid electrode is really
 * connected. Only used for the short, isolated DRL probe in `jtsmm emg
 * check` (bias must be disabled by the caller for this to take effect). */
static bool drl_probe_enabled = false;
static bool leadoff_hw_enabled = false;
/* LOFF_SENSP/N as last written. Comparator status is only published through
 * ads_leadoff_any(), which keeps the sensed CH1..CH6 bits and drops CH7/CH8
 * (SBAS499C §9.6.1.12: LOFF_STATP is only valid where LOFF_SENSP is set). */
static volatile uint8_t loff_sensp_active;
static volatile uint8_t loff_sensn_active;

void ads1299_set_sample_rate_hz(uint32_t fs_hz) {
    /* emg_config_validate() rejects anything else before it gets here. */
    stream_fs_hz = ads_sample_rate_supported(fs_hz) ? fs_hz : 250u;
}

void ads1299_set_pga_gain(uint8_t gain) {
    if (ads_gain_supported(gain)) {
        muscle_pga_gain = gain;
    }
}

uint8_t ads1299_pga_gain(void) {
    return muscle_pga_gain;
}

void ads1299_set_bias_drive(bool enable) {
    bias_drive_enabled = enable;
}

bool ads1299_bias_drive(void) {
    return bias_drive_enabled;
}

void ads1299_set_bias_channels(uint8_t mask_ch1_to_ch6) {
    bias_channel_mask = (uint8_t)(mask_ch1_to_ch6 & 0x3Fu);
    /* Do not default 0 -> 0x3F: a zero mask must not silently pull powered-down
     * or unspecified channels into the DRL average (see emg_config_apply). */
}

uint8_t ads1299_bias_channels(void) {
    return bias_channel_mask;
}

void ads1299_set_pd_channels(uint8_t mask_ch1_to_ch6) {
    muscle_pd_mask = (uint8_t)(mask_ch1_to_ch6 & 0x3Fu);
}

uint8_t ads1299_pd_channels(void) {
    return muscle_pd_mask;
}

void ads1299_set_drl_probe(bool enable) {
    drl_probe_enabled = enable;
}

bool ads1299_drl_probe(void) {
    return drl_probe_enabled;
}

#define ADS_CONFIG2_DEFAULT 0xC0u
/*
 * CONFIG3 (SBAS499C Table 15), see ads1299_regs.h:
 *   ref only  0xE0 = PD_REFBUF
 *   bias on   0xEC = PD_REFBUF | BIASREF_INT | PD_BIAS
 * The bias amplifier needs PD_BIAS=1 to be powered at all (the old 0xE8 left it
 * off). BIASREF_INT=1 makes its reference the internal (AVDD+AVSS)/2 = 2.5 V
 * mid-supply; the BIASREF pin (tied to AGND on this board) is then unused. The
 * amplifier "uses either the analog midsupply voltage, or the voltage present at
 * the BIASREF pin" (SBAS499C §10.1.3), so BIASREF_INT=0 (0xE4) would drive the
 * body to 0 V -- the negative rail of the unipolar 0..5 V supply.
 */
#define ADS_CONFIG3_INTERNAL_REF ADS_CONFIG3_REF_ONLY
#define ADS_CH1SET_DEFAULT  0x61u

/* CONFIG3 has two valid "healthy" values depending on whether DRL bias drive
 * is on (ADS_CONFIG3_BIAS_ON) or off (ADS_CONFIG3_INTERNAL_REF, ref-buffer-only).
 * Debug printouts must compare against whichever one is actually configured,
 * not hardcode the ref-only value -- otherwise a normal bias-on session always
 * prints a spurious "ref buffer off" warning. */
static uint8_t ads_expected_config3(void) {
    return ads_config3_value(bias_drive_enabled);
}

/** @brief Internal helper: CONFIG3 readback matches, ignoring read-only BIAS_STAT. */
static bool ads_config3_matches(uint8_t got) {
    return ads_reg_readback_ok(ADS_REG_CONFIG3, ads_expected_config3(), got);
}

static bool bus_ready;
static bool streaming;
static int last_init_stage;
/* Latched true only at the end of a fully successful ads1299_init(); cleared at
 * its start. bus_ready only says the SPI pins are set up, not that the chip took
 * its configuration, so streaming is gated on this instead. */
static volatile bool init_ok;
static volatile bool drdy_streaming;
static volatile uint32_t drdy_count;
static volatile uint32_t missed_drdy;
static volatile bool drdy_armed = true;
/* ads1299_drdy_service() state, sensor core only; see the comment there. */
static volatile bool drdy_pending;
static volatile uint64_t drdy_pending_t;
static bool read_inflight;
static uint32_t inflight_wedge_mark;
static ads_drdy_track_t drdy_track;
static uint32_t drdy_period_us = 500u;
/* Acquisition sequence number of the next conversion (contract K1). */
static uint32_t acq_seq_next;
static void (*drdy_divisor_cb)(uint32_t count);
/* Scheduling instrumentation: largest observed interval between two serviced
 * DRDY edges. At 2 kHz the nominal interval is 500 µs; a value materially above
 * that means the sensor loop stalled long enough to drop an EMG sample. */
static volatile uint64_t last_drdy_service_us;
static volatile uint32_t max_drdy_gap_us;
/* Leadoff status decoded from the ADS status word embedded in every EMG frame,
 * plus a framing guard: the status word's top nibble is a fixed 0b1100, so any
 * frame whose prefix differs indicates the RDATA byte stream slipped. */
static volatile bool stream_leadoff_active;
static volatile uint32_t frame_misalign_count;

/*
 * Framing guard state.
 *
 * A slipped RDATA byte stream is not a small error: shifting the 24-bit words by
 * one byte multiplies sample magnitudes by ~256 and rotates data between
 * channels, so the trace becomes full-scale hash that is no longer a function of
 * any single electrode. That is indistinguishable from analog noise on screen,
 * which is why it has to be detected and reported rather than inferred.
 *
 * Two independent knobs, both defaulting to on, both switchable at runtime via
 * the `emg_guard` CDC command so a slip can be observed with the guard off and
 * the fix confirmed with it on, without reflashing.
 */
#define ADS_RESYNC_STREAK_TRIP    4u  /* consecutive bad frames before resync */
#define ADS_RESYNC_MAX_PER_STREAM 64u /* stop retrying; keep counting + logging */

/* Two separate runs, because they answer different questions.
 * frame_misalign_streak is the resync TRIP counter and is cleared whenever a
 * resync fires, so it never exceeds ADS_RESYNC_STREAK_TRIP once the guard is
 * on. frame_misalign_run is the diagnostic one: it is cleared only by a good
 * frame, so the reported worst run reflects how long the stream was actually
 * unusable rather than how often we intervened. */
static volatile uint32_t frame_misalign_streak;
static volatile uint32_t frame_misalign_run;
static volatile uint32_t frame_misalign_max_streak;
static volatile uint32_t frame_resync_count;
static volatile uint32_t frame_resync_giveups;
static volatile uint8_t frame_misalign_last_status0;
static volatile bool guard_drop_misaligned = true;
static volatile bool guard_resync_enabled = true;

/** @brief Internal helper: ads1299_id_valid (pattern check in ads_id_valid()). */
static bool ads1299_id_valid(uint8_t raw, uint8_t *chip_id_out) {
    bool ok = ads_id_valid(raw);
    *chip_id_out = ok ? ADS1299_CHIP_ID : raw;
    return ok;
}

/*
 * SPI0 hand-off between the cores while the EMG stream is NOT running.
 *
 * The idle lead-off poll runs on the sensor core (core 1) once a second; the
 * comms core (core 0) touches the ADS for `config apply`/selftest inits, bench
 * register dumps and stream start. There is no mutex (a core-1 reset in the
 * middle of a poll would leave one held forever), so this is a two-flag
 * handshake: core 0 announces a claim and waits (bounded) for a poll already in
 * flight to finish; core 1 announces itself and backs off if core 0 has
 * claimed. A poll is bounded by ADS_NOPOLL_TIMEOUT_US per transaction, far
 * below ADS_CLAIM_WAIT_US. While streaming, core 1 owns the bus and the command
 * layer refuses bus-touching commands (cmd_bus_busy()).
 */
#define ADS_CLAIM_WAIT_US 2000u
static volatile uint32_t core0_bus_claims;
static volatile bool core1_bus_active;

/** @brief Internal helper: core 0 takes SPI0 from the idle poll (nestable). */
static void ads_bus_claim(void) {
    core0_bus_claims = core0_bus_claims + 1u;
    __dmb();
    absolute_time_t deadline = make_timeout_time_us(ADS_CLAIM_WAIT_US);
    while (core1_bus_active) {
        if (time_reached(deadline)) {
            /* Core 1 was reset mid-poll (sensor_core_stop): the flag is stale. */
            core1_bus_active = false;
            break;
        }
        tight_loop_contents();
    }
}

/** @brief Internal helper: core 0 returns SPI0. */
static void ads_bus_release(void) {
    __dmb();
    if (core0_bus_claims > 0u) {
        core0_bus_claims = core0_bus_claims - 1u;
    }
}

/** @brief Internal helper: core 1 enters the bus unless core 0 holds it. */
static bool ads_bus_core1_enter(void) {
    core1_bus_active = true;
    __dmb();
    if (core0_bus_claims != 0u) {
        core1_bus_active = false;
        return false;
    }
    return true;
}

/** @brief Internal helper: core 1 leaves the bus. */
static void ads_bus_core1_exit(void) {
    __dmb();
    core1_bus_active = false;
}

/**
 * @brief Internal helper: ads_cs.
 *
 * @details Every transaction starts with an empty RX FIFO. The byte helpers
 *          read exactly one RX byte per TX byte, so a stale byte left behind by
 *          an aborted transfer or a bus collision would otherwise shift every
 *          later register read by one byte for good.
 */
static void ads_cs(bool active) {
    if (active) {
        while (spi_is_readable(ADS_SPI)) {
            (void)spi_get_hw(ADS_SPI)->dr;
        }
    }
    gpio_put(PIN_ADS_CS, active ? 0 : 1);
}

/** @brief Internal helper: ads_spi_cfg_speed. */
static void ads_spi_cfg_speed(void) {
    spi_set_baudrate(ADS_SPI, ads_spi_cfg_hz);
}

/** @brief Internal helper: ads_spi_run_speed. */
static void ads_spi_run_speed(void) {
    spi_set_baudrate(ADS_SPI, ads_spi_run_hz);
}

// SBAS499 §9.5.3.1: >= 4 tCLK (~2 us) between command bytes at fast SCLK
/** @brief Internal helper: ads_spi_write_byte. */
static bool ads_spi_write_byte(uint8_t value) {
    const uint32_t timeout_us = 100000u;
    absolute_time_t deadline = delayed_by_us(get_absolute_time(), timeout_us);
    while (!spi_is_writable(ADS_SPI)) {
        if (time_reached(deadline)) {
            return false;
        }
        usb_service_poll();
    }
    spi_get_hw(ADS_SPI)->dr = value;
    while (!spi_is_readable(ADS_SPI)) {
        if (time_reached(deadline)) {
            return false;
        }
        usb_service_poll();
    }
    (void)spi_get_hw(ADS_SPI)->dr;
    sleep_us(4);
    return true;
}

/** @brief Internal helper: ads_spi_read_byte. */
static bool ads_spi_read_byte(uint8_t *value) {
    const uint32_t timeout_us = 100000u;
    absolute_time_t deadline = delayed_by_us(get_absolute_time(), timeout_us);
    while (!spi_is_writable(ADS_SPI)) {
        if (time_reached(deadline)) {
            return false;
        }
        usb_service_poll();
    }
    spi_get_hw(ADS_SPI)->dr = 0x00u;
    while (!spi_is_readable(ADS_SPI)) {
        if (time_reached(deadline)) {
            return false;
        }
        usb_service_poll();
    }
    *value = (uint8_t)spi_get_hw(ADS_SPI)->dr;
    sleep_us(4);
    return true;
}

/** @brief Internal helper: ads_send_cmd. */
static bool ads_send_cmd(uint8_t cmd) {
    ads_cs(true);
    bool ok = ads_spi_write_byte(cmd);
    ads_cs(false);
    sleep_us(16);
    return ok;
}

/*
 * Poll-free, tightly bounded SPI helpers for the SENSOR core (core 1).
 *
 * ads_spi_write_byte() calls usb_service_poll() (tud_task()) while it waits and
 * sleep_us() afterwards. tud_task() is only legal on the comms core, which owns
 * TinyUSB, and sleep_us() goes through the alarm pool, whose spin locks core 0
 * also takes. The framing-guard resync and the idle lead-off poll both run on
 * the sensor core, so they use these variants: busy_wait_us() only, and they
 * give up in microseconds rather than stalling the DRDY loop for 100 ms.
 */
#define ADS_NOPOLL_TIMEOUT_US 200u
/* SBAS499C §9.5.3.1: 4 tCLK (1.96 us at 2.048 MHz) between the end of one
 * byte and the next. Timing table: tSCCS = 4 tCLK from the last SCLK to CS
 * high, tCSH = 2 tCLK CS high. */
#define ADS_TSDECODE_US 2u
#define ADS_TSCCS_US    2u
#define ADS_TCSH_US     2u

/** @brief Internal helper: one full-duplex byte, no USB, no alarm pool. */
static bool ads_spi_xfer_nopoll(uint8_t tx, uint8_t *rx, absolute_time_t deadline) {
    while (!spi_is_writable(ADS_SPI)) {
        if (time_reached(deadline)) {
            return false;
        }
        tight_loop_contents();
    }
    spi_get_hw(ADS_SPI)->dr = tx;
    while (!spi_is_readable(ADS_SPI)) {
        if (time_reached(deadline)) {
            return false;
        }
        tight_loop_contents();
    }
    uint8_t v = (uint8_t)spi_get_hw(ADS_SPI)->dr;
    if (rx) {
        *rx = v;
    }
    return true;
}

/** @brief Internal helper: ads_send_cmd_nopoll. */
static bool ads_send_cmd_nopoll(uint8_t cmd) {
    absolute_time_t deadline = make_timeout_time_us(ADS_NOPOLL_TIMEOUT_US);
    ads_cs(true);
    bool ok = ads_spi_xfer_nopoll(cmd, NULL, deadline);
    /* RX full means the 8th SCLK has happened; hold CS for tSCCS after it. */
    busy_wait_us(ADS_TSCCS_US);
    ads_cs(false);
    busy_wait_us(16); /* SBAS499 §9.5.3.1 inter-command spacing */
    return ok;
}

/** @brief Internal helper: RREG of one register, sensor-core safe. */
static bool ads_read_reg_nopoll(uint8_t reg, uint8_t *value) {
    absolute_time_t deadline = make_timeout_time_us(ADS_NOPOLL_TIMEOUT_US);
    uint8_t v = 0;
    ads_cs(true);
    bool ok = ads_spi_xfer_nopoll((uint8_t)(0x20u | (reg & 0x1Fu)), NULL, deadline);
    busy_wait_us(ADS_TSDECODE_US);
    ok = ok && ads_spi_xfer_nopoll(0x00u, NULL, deadline);
    busy_wait_us(ADS_TSDECODE_US);
    ok = ok && ads_spi_xfer_nopoll(0x00u, &v, deadline);
    busy_wait_us(ADS_TSCCS_US);
    ads_cs(false);
    busy_wait_us(ADS_TCSH_US);
    if (ok) {
        *value = v;
    }
    return ok;
}

/** @brief Internal helper: ads_sdatac_only. */
static void ads_sdatac_only(void) {
    ads_send_cmd(ADS_CMD_SDATAC);
    sleep_us(200);
}

// Full halt — only for programming / ads_test reconfig
/** @brief Internal helper: ads_halt_conversions. */
static void ads_halt_conversions(void) {
    gpio_put(PIN_ADS_START, 0);
    ads_send_cmd(ADS_CMD_STOP);
    streaming = false;
    usb_service_sleep_ms(1);
    for (int i = 0; i < 4; i++) {
        ads_send_cmd(ADS_CMD_SDATAC);
        sleep_us(200);
    }
    usb_service_sleep_ms(2);
}

/** @brief Internal helper: ads_prep_reg_read (core 0; pair with ads_done_reg_read). */
static void ads_prep_reg_read(void) {
    ads_bus_claim();
    ads_spi_cfg_speed();
    ads_sdatac_only();
}

/** @brief Internal helper: ads_done_reg_read. */
static void ads_done_reg_read(void) {
    if (streaming) {
        ads_spi_run_speed();
    }
    ads_bus_release();
}

/** @brief Internal helper: ads_write_reg_raw. */
static void ads_write_reg_raw(uint8_t reg, uint8_t value) {
    ads_cs(true);
    ads_spi_write_byte((uint8_t)(0x40 | (reg & 0x1F)));
    ads_spi_write_byte(0x00);
    ads_spi_write_byte(value);
    ads_cs(false);
    sleep_us(16);
}

/** @brief Internal helper: ads_loff_reg_for_current. */
static uint8_t ads_loff_reg_for_current(uint8_t current_na) {
    /* LOFF (addr 0x04), SBAS499C Table 16: [7:5]=COMP_TH (000=95%/5%, unused
     * here); [4]=reserved(0); [3:2]=ILEAD_OFF (00=6 nA, 01=24 nA, 10=6 uA,
     * 11=24 uA -- the uA options are for BIAS-electrode checks, not skin |Z|,
     * and would swing the PGA hard into the rails here); [1:0]=FLEAD_OFF
     * (00=DC, 01=AC 7.8 Hz, 10=AC 31.2 Hz, 11=AC fDR/4).
     *
     * We use the fixed 31.2 Hz "one-time/periodic" tone (bits=10), not fDR/4:
     * it is derived from the ADS1299 master clock, not the configured output
     * data rate, so its frequency is exactly known on the host without having
     * to re-derive it per sample-rate profile, and 31.2 Hz resolves cleanly in
     * a short capture at every supported SPS (250-2000). It is only enabled for
     * the dedicated `emg check`/`emg impedance` sessions (not during a live
     * recording), so sitting inside the EMG band is not a contamination risk. */
    uint8_t ilead = (current_na >= 24u) ? 1u : 0u;
    if (drl_probe_enabled) {
        /* Isolated DRL/bias-electrode probe (IN8N via CH8, bias drive off,
         * MUX forced to normal input -- see drl_probe_enabled's declaration).
         * IN8P is permanently unwired on this board (h03 §5), so CH8's
         * *differential* reading (and the comparator's absolute threshold)
         * cannot be trusted to rail cleanly either way -- host-side detection
         * instead uses noise/RMS on the streamed CH8 column (an open,
         * high-impedance IN8N picks up far more ambient interference than a
         * low-impedance skin contact damps out; see drl_contact_off() in
         * capture.py). Use the strongest excitation current available (24
         * uA, ILEAD_OFF=11 -- the range the datasheet documents specifically
         * for bias-electrode checks like this one, and self-limited to
         * ~23 uA by the on-board 220 kOhm anyway, h03 §5.1) to pin IN8N to a
         * firm, repeatable DC bias point rather than an arbitrary floating
         * level. */
        ilead = 3u;
    }
    if (leadoff_ac_tone) {
        return (uint8_t)((ilead << 2) | 0x02u);
    }
    return (uint8_t)(ilead << 2);
}

/** @brief Internal helper: ads_leadoff_channel_masks. */
static void ads_leadoff_channel_masks(uint8_t *sensp_out, uint8_t *sensn_out) {
    uint8_t muscle = (uint8_t)((~muscle_pd_mask) & ADS_MUSCLE_MASK);
    *sensp_out = muscle;
    *sensn_out = muscle;
    if (bias_drive_enabled || drl_probe_enabled) {
        /* CH8 IN8N carries the mastoid DRL electrode (MUX8=111 in normal
         * acquisition; forced to MUX=000 by drl_probe_enabled instead, see
         * ads_ch8set()). Either way the DC lead-off excitation
         * on IN8N is the only way to test that pin. */
        *sensn_out = (uint8_t)(muscle | 0x80u);
    }
}

/** @brief Internal helper: ads_read_reg_raw. */
static uint8_t ads_read_reg_raw(uint8_t reg) {
    uint8_t value = 0;

    ads_cs(true);
    ads_spi_write_byte((uint8_t)(0x20 | (reg & 0x1F)));
    ads_spi_write_byte(0x00);
    ads_spi_read_byte(&value);
    ads_cs(false);
    sleep_us(16);
    return value;
}

/** @brief Internal helper: ads_effective_bias_mask. */
static uint8_t ads_effective_bias_mask(void) {
    return (uint8_t)(bias_channel_mask & (uint8_t)(~muscle_pd_mask) & 0x3Fu);
}

/** @brief Internal helper: ads_write_reg_check. */
static bool ads_write_reg_check(uint8_t reg, uint8_t value) {
    ads_halt_conversions();
    ads_write_reg_raw(reg, value);
    uint8_t got = ads_read_reg_raw(reg);
    return ads_reg_readback_ok(reg, value, got);
}

/*
 * Power-up wait before the first reset pulse. SBAS499C §11.1: reset only after
 * tPOR = 2^18 tCLK (128 ms at 2.048 MHz) *and* VCAP1 > 1.1 V; the electrical
 * characteristics give 150 ms typical "from power-up to DRDY low". The device
 * leaves power-down when PWDN goes high, so count from there.
 */
#define ADS_TPOR_US 150000
static absolute_time_t ads_pwr_up_at;

/** @brief Internal helper: ads_hw_reset. */
static void ads_hw_reset(void) {
    int64_t since_up = absolute_time_diff_us(ads_pwr_up_at, get_absolute_time());
    if (since_up < ADS_TPOR_US) {
        usb_service_sleep_ms((uint32_t)((ADS_TPOR_US - since_up + 999) / 1000));
    }
    gpio_put(PIN_ADS_RESET, 0);
    usb_service_sleep_ms(2);
    gpio_put(PIN_ADS_RESET, 1);
    usb_service_sleep_ms(10);
}

/** @brief Internal helper: ads_gpio_init. */
static void ads_gpio_init(void) {
    gpio_init(PIN_ADS_CS);
    gpio_set_dir(PIN_ADS_CS, GPIO_OUT);
    gpio_put(PIN_ADS_CS, 1);

    gpio_init(PIN_ADS_RESET);
    gpio_set_dir(PIN_ADS_RESET, GPIO_OUT);
    gpio_put(PIN_ADS_RESET, 1);

    gpio_init(PIN_ADS_PWDN);
    gpio_set_dir(PIN_ADS_PWDN, GPIO_OUT);
    gpio_put(PIN_ADS_PWDN, 1);
    ads_pwr_up_at = get_absolute_time();

    gpio_init(PIN_ADS_START);
    gpio_set_dir(PIN_ADS_START, GPIO_OUT);
    gpio_put(PIN_ADS_START, 0);

    gpio_init(PIN_ADS_DRDY);
    gpio_set_dir(PIN_ADS_DRDY, GPIO_IN);
    gpio_disable_pulls(PIN_ADS_DRDY);
}

/** @brief Internal helper: ads1299_start_conversions. */
static void ads1299_start_conversions(void) {
    ads_send_cmd(ADS_CMD_START);
    gpio_put(PIN_ADS_START, 1);
    streaming = true;
    usb_service_sleep_ms(4);
}

/** @brief Internal helper: ads1299_stop_conversions. */
static void ads1299_stop_conversions(void) {
    ads_send_cmd(ADS_CMD_STOP);
    gpio_put(PIN_ADS_START, 0);
    streaming = false;
    sleep_us(16);
}

/**
 * @brief Internal helper: take the SPI rates from the active platform config.
 *
 * @details Runs on every bus init (hence every ads1299_init / config apply), not
 *          only the first, so a changed spi0_*_hz takes effect without a reboot.
 *          The stream rate is capped at ADS_SPI_STREAM_MAX_HZ: the DMA frame has
 *          no gap between the RDATA opcode and the first data byte, so one byte
 *          time must cover tSDECODE (SBAS499C §9.5.3.1).
 */
static void ads_refresh_spi_rates(void) {
    const platform_config_t *plat = &jtsmm_config_active()->platform;
    if (plat->spi0_cfg_hz) {
        ads_spi_cfg_hz = plat->spi0_cfg_hz;
    }
    if (plat->spi0_stream_hz) {
        ads_spi_run_hz = plat->spi0_stream_hz;
    }
    ads_spi_run_hz = ads_clamp_stream_hz(ads_spi_run_hz);
}

uint32_t ads1299_stream_spi_hz(void) {
    ads_refresh_spi_rates();
    return ads_spi_run_hz;
}

bool ads1299_bus_init(void) {
    ads_refresh_spi_rates();
    if (bus_ready) {
        return true;
    }

    ads_gpio_init();

    spi_init(ADS_SPI, ads_spi_cfg_hz);
    spi_set_format(ADS_SPI, 8, SPI_CPOL_0, SPI_CPHA_1, SPI_MSB_FIRST);
    gpio_set_function(PIN_ADS_MISO, GPIO_FUNC_SPI);
    gpio_set_function(PIN_ADS_SCLK, GPIO_FUNC_SPI);
    gpio_set_function(PIN_ADS_MOSI, GPIO_FUNC_SPI);

    bus_ready = true;
    streaming = false;
    last_init_stage = 0;
    return true;
}

/** @brief Internal helper: ads1299_wakeup_sequence. */
static void ads1299_wakeup_sequence(void) {
    ads_hw_reset();
    ads_send_cmd(ADS_CMD_RESET);
    usb_service_sleep_ms(2);
    ads_halt_conversions();
    ads_send_cmd(ADS_CMD_WAKEUP);
    usb_service_sleep_ms(1);
}

/** One register of the configuration image. */
typedef struct {
    uint8_t reg;
    uint8_t value;
} ads_reg_value_t;

#define ADS_IMAGE_MAX 18u

/**
 * @brief Internal helper: complete register image for the current setter state.
 *
 * @details Single source of truth for what program_config() writes and then
 *          verifies. CONFIG3 comes last so the bias amplifier is only powered
 *          once BIAS_SENSP/N select its inputs.
 * @param[out] img  At least ADS_IMAGE_MAX entries.
 * @param[out] leadoff_on  Whether the lead-off comparators are enabled.
 * @return Number of entries, or 0 if bias drive is on with no usable channel.
 */
static size_t ads_config_image(ads_reg_value_t *img, bool *leadoff_on) {
    size_t n = 0;
    uint8_t bias_mask = 0;
    if (bias_drive_enabled) {
        bias_mask = ads_effective_bias_mask();
        if (bias_mask == 0) {
            return 0;
        }
    }
    uint8_t sensp = 0;
    uint8_t sensn = 0;
    if (leadoff_monitor_enabled && leadoff_during_recording) {
        ads_leadoff_channel_masks(&sensp, &sensn);
    }
    bool loff = (sensp | sensn) != 0;

    img[n++] = (ads_reg_value_t){ADS_REG_CONFIG1, ads_config1_for_fs(stream_fs_hz)};
    img[n++] = (ads_reg_value_t){ADS_REG_CONFIG2, ADS_CONFIG2_DEFAULT};
    uint8_t chset = ads_chset_for_gain(muscle_pga_gain);
    for (uint8_t i = 0; i < 6; i++) {
        bool pd = (muscle_pd_mask & (uint8_t)(1u << i)) != 0;
        img[n++] = (ads_reg_value_t){(uint8_t)(ADS_REG_CH1SET + i), pd ? ADS_CHSET_OFF : chset};
    }
    /* CH7 has no patient pin: power it down instead of amplifying an open,
     * floating input pair. CH8 per mode, see ads_ch8set(). */
    img[n++] = (ads_reg_value_t){(uint8_t)(ADS_REG_CH1SET + 6), ADS_CHSET_OFF};
    img[n++] = (ads_reg_value_t){(uint8_t)(ADS_REG_CH1SET + 7),
                                 ads_ch8set(bias_drive_enabled, drl_probe_enabled)};
    img[n++] = (ads_reg_value_t){ADS_REG_BIAS_SENSP, bias_mask};
    img[n++] = (ads_reg_value_t){ADS_REG_BIAS_SENSN, bias_mask};
    img[n++] = (ads_reg_value_t){ADS_REG_LOFF, loff ? ads_loff_reg_for_current(leadoff_current_na) : 0u};
    img[n++] = (ads_reg_value_t){ADS_REG_LOFF_SENSP, sensp};
    img[n++] = (ads_reg_value_t){ADS_REG_LOFF_SENSN, sensn};
    /* PD_LOFF_COMP=1 means comparators ENABLED (SBAS499C Table 28, §10.1.2.1). */
    img[n++] = (ads_reg_value_t){ADS_REG_CONFIG4, ads_config4_value(loff)};
    img[n++] = (ads_reg_value_t){ADS_REG_CONFIG3, ads_config3_value(bias_drive_enabled)};
    *leadoff_on = loff;
    return n;
}

/**
 * @brief Internal helper: write the configuration image and verify every register.
 *
 * @return false (with a log line naming the register) on the first readback
 *         mismatch; read-only bits are masked (ads_reg_readback_mask()).
 */
static bool ads_apply_config_image(void) {
    ads_reg_value_t img[ADS_IMAGE_MAX];
    bool loff = false;
    size_t n = ads_config_image(img, &loff);
    if (n == 0) {
        printf("ads init: bias drive enabled but no powered channel in bias mask\r\n");
        last_init_stage = 6;
        return false;
    }
    for (size_t i = 0; i < n; i++) {
        ads_write_reg_raw(img[i].reg, img[i].value);
    }
    loff_sensp_active = 0;
    loff_sensn_active = 0;
    leadoff_hw_enabled = false;
    for (size_t i = 0; i < n; i++) {
        uint8_t got = ads_read_reg_raw(img[i].reg);
        if (!ads_reg_readback_ok(img[i].reg, img[i].value, got)) {
            printf("ads init FAIL: reg 0x%02X wrote 0x%02X read 0x%02X -- chip not configured\r\n",
                   (unsigned)img[i].reg, (unsigned)img[i].value, (unsigned)got);
            last_init_stage = 5;
            return false;
        }
        if (img[i].reg == ADS_REG_LOFF_SENSP) {
            loff_sensp_active = img[i].value;
        } else if (img[i].reg == ADS_REG_LOFF_SENSN) {
            loff_sensn_active = img[i].value;
        }
    }
    leadoff_hw_enabled = loff;
    return true;
}

/** @brief Internal helper: ads1299_program_config. */
static bool ads1299_program_config(void) {
    ads_halt_conversions();
    ads_spi_cfg_speed();

    // SBAS499 Fig. 67: enable internal ref before other config
    if (!ads_write_reg_check(ADS_REG_CONFIG3, ADS_CONFIG3_INTERNAL_REF)) {
        printf("ads init FAIL: CONFIG3 (internal reference) did not read back\r\n");
        last_init_stage = 3;
        return false;
    }
    usb_service_sleep_ms(150);

    ads_sdatac_only();
    return ads_apply_config_image();
}

/** @brief Internal helper: ads1299_init body (caller holds the bus claim). */
static bool ads1299_init_claimed(void) {
    last_init_stage = 0;
    init_ok = false;

    if (!ads1299_bus_init()) {
        last_init_stage = -1;
        return false;
    }

    usb_service_sleep_ms(50);
    ads1299_wakeup_sequence();

    uint8_t id = 0;
    if (!ads1299_read_id(&id)) {
        last_init_stage = 1;
        return false;
    }

    if (!ads1299_program_config()) {
        return false;
    }

    ads_spi_run_speed();
    ads_sdatac_only();
    last_init_stage = 0;
    init_ok = true;
    return true;
}

bool ads1299_init(void) {
    ads_bus_claim();
    bool ok = ads1299_init_claimed();
    ads_bus_release();
    return ok;
}

int ads1299_init_stage(void) {
    return last_init_stage;
}

bool ads1299_init_ok(void) {
    return init_ok;
}

bool ads1299_config_ok(void) {
    ads_prep_reg_read();
    uint8_t cfg3 = ads_read_reg_raw(ADS_REG_CONFIG3);
    ads_done_reg_read();
    return ads_config3_matches(cfg3);
}

uint8_t ads1299_read_id_raw(void) {
    if (!ads1299_bus_init()) {
        return 0;
    }
    ads_prep_reg_read();
    (void)ads_read_reg_raw(ADS_REG_ID);
    uint8_t id = ads_read_reg_raw(ADS_REG_ID);
    ads_done_reg_read();
    return id;
}

bool ads1299_read_id(uint8_t *id_out) {
    uint8_t raw = ads1299_read_id_raw();
    return ads1299_id_valid(raw, id_out);
}

void ads1299_regs(void) {
    ads_prep_reg_read();

    printf("ads_regs:\r\n");
    printf("  ID=0x%02X CONFIG1=0x%02X CONFIG2=0x%02X CONFIG3=0x%02X (want 0x%02X)\r\n",
           ads_read_reg_raw(ADS_REG_ID),
           ads_read_reg_raw(ADS_REG_CONFIG1),
           ads_read_reg_raw(ADS_REG_CONFIG2),
           ads_read_reg_raw(ADS_REG_CONFIG3),
           ads_expected_config3());
    printf("  CH1SET=0x%02X (def 0x%02X, want 0x%02X after init)\r\n",
           ads_read_reg_raw(ADS_REG_CH1SET),
           ADS_CH1SET_DEFAULT,
           ads_chset_for_gain(muscle_pga_gain));
    printf("  CH7SET=0x%02X CH8SET=0x%02X (want 0x%02X / 0x%02X: CH7 off; CH8 DRL 0x%02X, "
           "probe 0x%02X, else off)\r\n",
           ads_read_reg_raw(ADS_REG_CH1SET + 6),
           ads_read_reg_raw(ADS_REG_CH1SET + 7),
           ADS_CHSET_OFF, ads_ch8set(bias_drive_enabled, drl_probe_enabled),
           ADS_CHSET_DRL, ADS_CHSET_NORMAL24);
    printf("  BIAS_SENSP=0x%02X BIAS_SENSN=0x%02X (want 0x%02X when bias on)\r\n",
           ads_read_reg_raw(ADS_REG_BIAS_SENSP),
           ads_read_reg_raw(ADS_REG_BIAS_SENSN),
           ads_effective_bias_mask());
    printf("  CONFIG3 want 0x%02X (bias) or 0x%02X (ref only)\r\n",
           ADS_CONFIG3_BIAS_ON,
           ADS_CONFIG3_INTERNAL_REF);

    ads_done_reg_read();
}

void ads1299_debug(void) {
    ads1299_bus_init();
    ads_prep_reg_read();

    uint8_t raw_a = ads_read_reg_raw(ADS_REG_ID);
    uint8_t raw_b = ads_read_reg_raw(ADS_REG_ID);
    uint8_t cfg1 = ads_read_reg_raw(ADS_REG_CONFIG1);
    uint8_t cfg2 = ads_read_reg_raw(ADS_REG_CONFIG2);
    uint8_t cfg3 = ads_read_reg_raw(ADS_REG_CONFIG3);
    uint8_t ch1 = ads_read_reg_raw(ADS_REG_CH1SET);
    uint8_t chip_id = 0;
    bool ok = ads1299_id_valid(raw_b, &chip_id);

    ads_done_reg_read();

    printf("ads_debug:\r\n");
    printf("  ID reads: 0x%02X 0x%02X -> chip 0x%02X %s\r\n",
           raw_a, raw_b, chip_id, ok ? "OK" : "FAIL");
    printf("  CONFIG1=0x%02X CONFIG2=0x%02X CONFIG3=0x%02X (want 0x%02X, bias_drive=%d)\r\n",
           cfg1, cfg2, cfg3, ads_expected_config3(), bias_drive_enabled);
    printf("  CH1SET=0x%02X (def 0x%02X)\r\n", ch1, ADS_CH1SET_DEFAULT);
    if (!ads_config3_matches(cfg3)) {
        printf("  ** WREG failing or ref buffer won't start — measure TP_VREFP1 vs TP_AGND1\r\n");
    }
    printf("  streaming=%d DRDY=%d START=%d PWDN=%d RESET=%d\r\n",
           streaming, gpio_get(PIN_ADS_DRDY), gpio_get(PIN_ADS_START),
           gpio_get(PIN_ADS_PWDN), gpio_get(PIN_ADS_RESET));
    printf("  SPI mode 1 @ %u Hz\r\n", (unsigned)spi_get_baudrate(ADS_SPI));
}

void ads1299_status(void) {
    ads_prep_reg_read();

    uint8_t cfg1 = ads_read_reg_raw(ADS_REG_CONFIG1);
    uint8_t cfg2 = ads_read_reg_raw(ADS_REG_CONFIG2);
    uint8_t cfg3 = ads_read_reg_raw(ADS_REG_CONFIG3);
    uint8_t cfg4 = ads_read_reg_raw(ADS_REG_CONFIG4);
    uint8_t ch1 = ads_read_reg_raw(ADS_REG_CH1SET);
    uint8_t ch8 = ads_read_reg_raw((uint8_t)(ADS_REG_CH1SET + 7));
    uint8_t loff = ads_read_reg_raw(ADS_REG_LOFF);
    uint8_t loff_sensp = ads_read_reg_raw(ADS_REG_LOFF_SENSP);
    uint8_t loff_sensn = ads_read_reg_raw(ADS_REG_LOFF_SENSN);
    uint8_t loff_statp = ads_read_reg_raw(ADS_REG_LOFF_STATP);
    uint8_t loff_statn = ads_read_reg_raw(ADS_REG_LOFF_STATN);

    ads_done_reg_read();

    printf("ads_status:\r\n");
    printf("  CONFIG1=0x%02X CONFIG2=0x%02X CONFIG3=0x%02X (want 0x%02X, bias_drive=%d)\r\n",
           cfg1, cfg2, cfg3, ads_expected_config3(), bias_drive_enabled);
    printf("  CH1SET=0x%02X (gain=%u)\r\n", ch1, muscle_pga_gain);
    printf("  CONFIG4=0x%02X (want 0x%02X) CH8SET=0x%02X LOFF=0x%02X SENSP=0x%02X SENSN=0x%02X "
           "STATP=0x%02X STATN=0x%02X drl_probe=%d\r\n",
           cfg4, ads_config4_value(leadoff_hw_enabled), ch8, loff, loff_sensp, loff_sensn,
           loff_statp, loff_statn, drl_probe_enabled);
    printf("  leadoff published (CH1-CH6 sensed only)=%d\r\n",
           leadoff_hw_enabled &&
               ads_leadoff_any(loff_statp, loff_statn, loff_sensp_active, loff_sensn_active));
    if (!ads_config3_matches(cfg3)) {
        printf("  WARN: CONFIG3 mismatch (ref buffer / bias amp) — measure TP_VREFP1 (~4.5V vs TP_AGND1)\r\n");
    }
    printf("  streaming=%d drdy_edges=%lu missed=%lu DRDY=%d START=%d\r\n",
           drdy_streaming, (unsigned long)drdy_count, (unsigned long)missed_drdy,
           gpio_get(PIN_ADS_DRDY), gpio_get(PIN_ADS_START));
}

void ads1299_drdy_watch(void) {
    int lows = 0;
    int highs = 0;
    int last = gpio_get(PIN_ADS_DRDY);
    absolute_time_t end = make_timeout_time_ms(500);

    while (!time_reached(end)) {
        int now = gpio_get(PIN_ADS_DRDY);
        if (now != last) {
            if (now == 0) {
                lows++;
            } else {
                highs++;
            }
            last = now;
        }
        tight_loop_contents();
    }
    printf("ads_drdy_watch 500ms: edges low=%d high=%d final=%d (expect ~125 edges @ 250 SPS)\r\n",
           lows, highs, gpio_get(PIN_ADS_DRDY));
}

bool ads1299_test_signal(bool enable) {
    if (!bus_ready) {
        ads1299_init();
    }
    ads_bus_claim();

    ads1299_stop_conversions();
    ads_spi_cfg_speed();
    ads_sdatac_only();

    /* Only INT_CAL changes; reserved CONFIG2[7:5] stay 110b (SBAS499C Table 14). */
    uint8_t cfg2 = ads_config2_test_signal(ads_read_reg_raw(ADS_REG_CONFIG2), enable);
    ads_write_reg_raw(ADS_REG_CONFIG2, cfg2);

    bool ok = true;
    if (enable) {
        for (uint8_t i = 0; i < 8; i++) {
            uint8_t normal = (i < 6) ? ads_chset_for_gain(muscle_pga_gain) : ADS_CHSET_NORMAL24;
            ads_write_reg_raw((uint8_t)(ADS_REG_CH1SET + i), ads_chset_test(normal));
        }
    } else {
        /* Back to the configured image (includes CONFIG2 = default, test off). */
        ok = ads_apply_config_image();
        if (!ok) {
            init_ok = false; /* registers no longer match the configuration */
        }
    }

    ads_spi_run_speed();
    ads_sdatac_only();
    ads1299_start_conversions();
    ads_bus_release();
    return ok;
}

/** @brief Internal helper: sign_extend24. */
static int32_t sign_extend24(uint32_t v) {
    if (v & 0x800000u) {
        v |= 0xFF000000u;
    }
    return (int32_t)v;
}

/** @brief Internal helper: ads1299_wait_drdy. */
static bool ads1299_wait_drdy(uint32_t timeout_us) {
    absolute_time_t deadline = make_timeout_time_ms((timeout_us + 999) / 1000);
    while (gpio_get(PIN_ADS_DRDY)) {
        if (time_reached(deadline)) {
            return false;
        }
        usb_service_poll();
        tight_loop_contents();
    }
    return true;
}

/** @brief Internal helper: ads_read_frame_rdata. */
static bool ads_read_frame_rdata(uint8_t status[3], int32_t ch[8]) {
    ads_cs(true);
    ads_spi_write_byte(ADS_CMD_RDATA);

    uint8_t tx[27];
    uint8_t rx[27];
    memset(tx, 0, sizeof(tx));
    for (size_t i = 0; i < sizeof(rx); i++) {
        ads_spi_read_byte(&rx[i]);
    }
    ads_cs(false);

    status[0] = rx[0];
    status[1] = rx[1];
    status[2] = rx[2];

    for (int i = 0; i < 8; i++) {
        uint32_t raw = ((uint32_t)rx[3 + i * 3] << 16) |
                       ((uint32_t)rx[4 + i * 3] << 8) |
                       (uint32_t)rx[5 + i * 3];
        ch[i] = sign_extend24(raw);
    }
    return true;
}

bool ads1299_read_sample(uint8_t status[3], int32_t ch[8]) {
    if (!bus_ready || !streaming) {
        return false;
    }
    if (!ads1299_wait_drdy(500000)) {
        return false;
    }
    return ads_read_frame_rdata(status, ch);
}

bool ads1299_drdy_ready(void) {
    return !gpio_get(PIN_ADS_DRDY);
}

uint32_t ads1299_sample_rate_hz(void) {
    return stream_fs_hz;
}

// --- Production ISR/DMA streaming path ---

/** @brief Internal helper: ads_drdy_gpio_handler. */
static void ads_drdy_gpio_handler(uint gpio, uint32_t events) {
    (void)gpio;
    (void)events;
    /* DRDY polled on the sensor core via ads1299_drdy_service(). */
}

/*
 * Level-polled, async-DMA-aware DRDY service (sensor core).
 *
 * DRDY behaviour (SBAS499C §9.4.4.1/2): it falls when a conversion is ready and
 * "returns high on the first SCLK falling edge" of a read. If nobody reads, it
 * only pulses high for 4 tCLK (~2 us) before the next conversion -- far too
 * short for a poll loop to see reliably.
 *
 * - armed: a DRDY low seen while armed is a new conversion. We re-arm when DRDY
 *   is seen high *or* when our own read completes: its SCLKs forced DRDY high,
 *   so any later low must be a new conversion even if the high phase was never
 *   observed. (The old code re-armed only on an observed high; a conversion
 *   landing between the last SCLK and the next poll then stalled the stream
 *   until a poll happened to hit one of the 2 us pulses.)
 * - pending: a new conversion seen while the DMA is still busy is read as soon
 *   as it completes (RDATA returns the latest conversion) instead of being
 *   dropped and counted as missed.
 * - lost conversions (stalls, a conversion hidden inside a late read's frame)
 *   are estimated from the detection time by ads_drdy_track() and advance both
 *   missed_drdy and the acquisition seq, so the host sees them as a seq gap
 *   (contract K1).
 */
void ads1299_drdy_service(void) {
    __dmb();
    if (!drdy_streaming) {
        return;
    }

    bool busy = spi_dma_busy();
    if (read_inflight && !busy) {
        read_inflight = false;
        /* A wedge abort may have cut the frame before any SCLK: then DRDY was
         * never forced high and re-arming could read the same conversion twice. */
        if (!drdy_pending && spi_dma_wedge_count() == inflight_wedge_mark) {
            drdy_armed = true;
        }
    }

    if (gpio_get(PIN_ADS_DRDY) != 0) {
        drdy_armed = true;
    } else if (drdy_armed) {
        drdy_armed = false;
        drdy_pending = true;
        drdy_pending_t = t_master_us();
    }
    if (!drdy_pending || busy) {
        return;
    }

    drdy_pending = false;
    uint64_t t = drdy_pending_t;
    if (last_drdy_service_us != 0) {
        uint64_t gap = t - last_drdy_service_us;
        if (gap > max_drdy_gap_us) {
            max_drdy_gap_us = (uint32_t)gap;
        }
    }
    last_drdy_service_us = t;
    uint32_t lost = ads_drdy_track(&drdy_track, t, drdy_period_us);
    missed_drdy += lost;
    uint32_t seq = acq_seq_next + lost; /* wraps at 2^32 by design */
    acq_seq_next = seq + 1u;
    drdy_count++;
    if (drdy_divisor_cb) {
        drdy_divisor_cb(drdy_count);
    }
    inflight_wedge_mark = spi_dma_wedge_count();
    read_inflight = true;
    spi_dma_kick_read(t, seq);
}

void ads1299_drdy_isr_install(void) {
    static bool registered;
    if (!registered) {
        gpio_irq_register(ads_drdy_gpio_handler);
        registered = true;
    }
    gpio_irq_enable_pin(PIN_ADS_DRDY, GPIO_IRQ_EDGE_FALL, false);
}

void ads1299_drdy_isr_enable(bool on) {
    gpio_irq_enable_pin(PIN_ADS_DRDY, GPIO_IRQ_EDGE_FALL, on);
}

bool ads1299_start_stream(void) {
    if (drdy_streaming) {
        /* Already streaming: a second START would restart (resynchronise) the
         * conversions, spi_set_baudrate() would retime SPI0 under the sensor
         * core's in-flight DMA, and the counter reset below would erase this
         * stream's missed/misaligned counts. Nothing to do. */
        return true;
    }
    if (!bus_ready || !init_ok) {
        /* Never stream from a chip whose last configuration failed: registers
         * may be at reset defaults (gain 24, bias off), and every microvolt the
         * host derives from the configured gain would be wrong. */
        return false;
    }
    ads_bus_claim();
    ads1299_drdy_isr_install();
    ads1299_start_conversions();
    /* Force the fast streaming SPI baud NOW. Register programming and idle
     * leadoff polls leave the bus at the slow config baud (~500 kHz), where a
     * 28-byte RDATA frame takes ~448 us -- perilously close to the 500 us DRDY
     * period, so DRDY edges land while the DMA is still busy (missed_drdy).
     * Previously only the first streaming leadoff poll bumped this to run speed;
     * setting it here makes the reader healthy from the first sample and lets the
     * disruptive leadoff poll be gated during recording without regressing. */
    ads_spi_run_speed();
    drdy_count = 0;
    missed_drdy = 0;
    drdy_pending = false;
    read_inflight = false;
    drdy_track = (ads_drdy_track_t){0};
    drdy_period_us = 1000000u / (stream_fs_hz ? stream_fs_hz : 250u);
    acq_seq_next = 0;
    last_drdy_service_us = 0;
    max_drdy_gap_us = 0;
    frame_misalign_count = 0;
    frame_misalign_streak = 0;
    frame_misalign_run = 0;
    frame_misalign_max_streak = 0;
    frame_resync_count = 0;
    frame_resync_giveups = 0;
    frame_misalign_last_status0 = 0;
    stream_leadoff_active = false;
    drdy_armed = true;
    drdy_streaming = true;
    __dmb();
    ads_bus_release(); /* core 1 owns SPI0 from here (DMA reads, no idle poll) */
    ads1299_drdy_isr_enable(false);
    return true;
}

void ads1299_stop_stream(void) {
    drdy_streaming = false;
    __dmb();
    ads1299_drdy_isr_enable(false);
}

uint32_t ads1299_drdy_edge_count(void) {
    return drdy_count;
}

uint32_t ads1299_missed_drdy_count(void) {
    return missed_drdy;
}

uint32_t ads1299_max_drdy_gap_us(void) {
    return max_drdy_gap_us;
}

void ads1299_set_drdy_divisor_cb(void (*cb)(uint32_t count)) {
    drdy_divisor_cb = cb;
}

void ads1299_set_leadoff_monitor(bool enable) {
    leadoff_monitor_enabled = enable;
    if (!enable) {
        panel_set_emg_leadoff(false);
    }
}

void ads1299_set_leadoff_during_recording(bool during) {
    leadoff_during_recording = during;
}

void ads1299_set_leadoff_current_na(uint8_t current_na) {
    leadoff_current_na = current_na;
}

void ads1299_set_leadoff_ac_tone(bool ac_tone) {
    leadoff_ac_tone = ac_tone;
}

bool ads1299_leadoff_hw_active(void) {
    return leadoff_hw_enabled;
}

void ads1299_leadoff_poll(void) {
    if (!bus_ready || !leadoff_monitor_enabled) {
        return;
    }
    if (drdy_streaming) {
        /* While streaming, leadoff is decoded from the status word of every EMG
         * frame (ads1299_stream_status_update) when the LOFF comparators are
         * actually enabled. When during_recording=false ads_config_image()
         * programs LOFF/LOFF_SENSP/N = 0 and the comparators off at config
         * apply (ads1299_init), so the status bits mean nothing -- do not
         * publish them (leadoff_hw_enabled is false then). */
        panel_set_emg_leadoff(leadoff_hw_enabled && stream_leadoff_active);
        return;
    }
    if (!leadoff_hw_enabled) {
        /* Comparators are off, the status registers mean nothing: no bus traffic. */
        panel_set_emg_leadoff(false);
        return;
    }
    /* Sensor core: only the *_nopoll helpers (no tud_task(), no sleep_us()). */
    if (!ads_bus_core1_enter()) {
        return; /* core 0 is using SPI0; try again next second */
    }
    ads_spi_cfg_speed();
    uint8_t statp = 0;
    uint8_t statn = 0;
    bool ok = ads_send_cmd_nopoll(ADS_CMD_SDATAC) &&
              ads_read_reg_nopoll(ADS_REG_LOFF_STATP, &statp) &&
              ads_read_reg_nopoll(ADS_REG_LOFF_STATN, &statn);
    if (streaming) {
        ads_spi_run_speed();
    }
    ads_bus_core1_exit();
    if (!ok) {
        return; /* bus did not answer in time: keep the last LED state */
    }
    panel_set_emg_leadoff(ads_leadoff_any(statp, statn, loff_sensp_active, loff_sensn_active));
}

bool ads1299_stream_status_update(const uint8_t *status) {
    /* ADS1299 24-bit status word: 1100 + LOFF_STATP[8] + LOFF_STATN[8] + GPIO[4].
     * A prefix other than 0b1100 means the RDATA byte stream slipped framing. */
    if ((status[0] & 0xF0u) != 0xC0u) {
        frame_misalign_count++;
        frame_misalign_last_status0 = status[0];
        frame_misalign_streak = frame_misalign_streak + 1u;
        uint32_t run = frame_misalign_run + 1u;
        frame_misalign_run = run;
        if (run > frame_misalign_max_streak) {
            frame_misalign_max_streak = run;
        }
        /* The sample bytes behind this status word are garbage; the caller must
         * not treat them as data. Leave stream_leadoff_active untouched -- the
         * LOFF bits in a slipped frame are meaningless either way. */
        return false;
    }
    frame_misalign_streak = 0;
    frame_misalign_run = 0;
    if (!leadoff_hw_enabled) {
        stream_leadoff_active = false;
        return true;
    }
    uint8_t statp = 0;
    uint8_t statn = 0;
    ads_status_word_leadoff(status, &statp, &statn);
    stream_leadoff_active = ads_leadoff_any(statp, statn, loff_sensp_active, loff_sensn_active);
    return true;
}

bool ads1299_stream_resync_if_needed(void) {
    if (frame_misalign_streak < ADS_RESYNC_STREAK_TRIP) {
        return false;
    }
    if (!guard_resync_enabled) {
        return false;
    }
    if (frame_resync_count >= ADS_RESYNC_MAX_PER_STREAM) {
        /* Persistent slip is a hardware/config problem, not a transient one.
         * Stop poking the bus (each resync costs a DRDY edge) but keep counting
         * so `stats` still shows the stream is unusable. */
        frame_resync_giveups++;
        frame_misalign_streak = 0;
        return false;
    }
    frame_resync_count++;
    /* Flush the SPI/DMA side first: the residue is in our FIFOs, not the ADS. */
    spi_dma_resync();
    /* Then re-assert SDATAC so the ADS serial state machine is back in the mode
     * the DMA reader assumes (one RDATA per CS-framed transaction). Best effort:
     * if the bus does not respond we still cleared our own side. */
    (void)ads_send_cmd_nopoll(ADS_CMD_SDATAC);
    /* Deliberately do NOT touch drdy_armed: ads1299_drdy_service() re-arms when
     * it sees the completed read (whose SCLKs, like this SDATAC's, forced DRDY
     * high), and forcing it here could kick a second read of the conversion we
     * just finished. A conversion lost to the resync shows up as a seq gap. */
    frame_misalign_streak = 0;
    return true;
}

uint32_t ads1299_frame_misalign_max_streak(void) {
    return frame_misalign_max_streak;
}

uint32_t ads1299_frame_resync_count(void) {
    return frame_resync_count;
}

uint32_t ads1299_frame_resync_giveup_count(void) {
    return frame_resync_giveups;
}

uint8_t ads1299_frame_misalign_last_status(void) {
    return frame_misalign_last_status0;
}

bool ads1299_drdy_streaming(void) {
    return drdy_streaming;
}

void ads1299_set_misalign_guard(bool drop, bool resync) {
    guard_drop_misaligned = drop;
    guard_resync_enabled = resync;
}

bool ads1299_misalign_guard_drop(void) {
    return guard_drop_misaligned;
}

bool ads1299_misalign_guard_resync(void) {
    return guard_resync_enabled;
}

uint32_t ads1299_frame_misalign_count(void) {
    return frame_misalign_count;
}

