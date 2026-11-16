// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file ads1299_regs.h
 * @brief ADS1299 register map and the pure register/stream arithmetic.
 *
 * @details Header-only and hardware-free so the host unit tests
 *          (firmware/jtsmm/tests/test_fix_A_ads1299.c) exercise exactly the
 *          values the driver writes. Bit layouts cite TI SBAS499C (ADS1299
 *          datasheet, Jan 2017) Table 11 and §9.6.1.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/** @{ @name Register addresses (SBAS499C Table 11) */
#define ADS_REG_ID         0x00u
#define ADS_REG_CONFIG1    0x01u
#define ADS_REG_CONFIG2    0x02u
#define ADS_REG_CONFIG3    0x03u
#define ADS_REG_LOFF       0x04u
#define ADS_REG_CH1SET     0x05u /**< CHnSET = ADS_REG_CH1SET + (n - 1). */
#define ADS_REG_BIAS_SENSP 0x0Du
#define ADS_REG_BIAS_SENSN 0x0Eu
#define ADS_REG_LOFF_SENSP 0x0Fu
#define ADS_REG_LOFF_SENSN 0x10u
#define ADS_REG_LOFF_STATP 0x12u /**< Read-only comparator status, P side. */
#define ADS_REG_LOFF_STATN 0x13u /**< Read-only comparator status, N side. */
#define ADS_REG_CONFIG4    0x17u
/** @} */

/** @{ @name CONFIG3 (SBAS499C Table 15) */
#define ADS_CONFIG3_PD_REFBUF   0x80u /**< 1 = internal reference buffer ON. */
#define ADS_CONFIG3_RESERVED    0x60u /**< Bits 6:5, always write 11b. */
#define ADS_CONFIG3_BIASREF_INT 0x08u /**< 1 = bias reference (AVDD+AVSS)/2 internal. */
#define ADS_CONFIG3_PD_BIAS     0x04u /**< 1 = bias amplifier ON. */
#define ADS_CONFIG3_BIAS_STAT   0x01u /**< Read-only; must be masked on readback. */
/** Reference buffer only, bias amplifier off. */
#define ADS_CONFIG3_REF_ONLY (ADS_CONFIG3_PD_REFBUF | ADS_CONFIG3_RESERVED)
/**
 * Bias on = 0xEC: ref buffer + internal mid-supply BIASREF + bias amp powered.
 * BIASREF_INT must be 1 on this board: the BIASREF pin is tied to AGND, so the
 * external option (0xE4) would drive the body to 0 V, i.e. onto the negative
 * rail of the unipolar 0..5 V supply instead of to mid-supply.
 */
#define ADS_CONFIG3_BIAS_ON \
    (ADS_CONFIG3_REF_ONLY | ADS_CONFIG3_BIASREF_INT | ADS_CONFIG3_PD_BIAS)
/** @} */

/** CONFIG4 bit 1, PD_LOFF_COMP: 1 = comparators ENABLED (SBAS499C Table 28). */
#define ADS_CONFIG4_PD_LOFF_COMP 0x02u

/** @{ @name CHnSET encodings (SBAS499C Table 17) */
#define ADS_CHSET_OFF      0x81u /**< PD=1, MUX=001 (input shorted): powered down. */
#define ADS_CHSET_NORMAL24 0x60u /**< PD=0, gain 24, MUX=000 normal electrode input. */
#define ADS_CHSET_DRL      0x67u /**< PD=0, gain 24, MUX=111: BIASIN routed to IN8N. */
#define ADS_CHSET_MUX_TEST 0x05u /**< MUX=101: internal test signal (gain bits separate). */
/** @} */

/** Muscle channels CH1..CH6; CH7 has no patient pin, CH8 is the DRL pin (IN8N). */
#define ADS_MUSCLE_MASK 0x3Fu

/**
 * Highest SPI0 rate for the DMA RDATA frame. The DMA clocks the RDATA opcode and
 * the 27 data bytes back to back, so the only thing separating the opcode from
 * the next byte is one byte time. SBAS499C §9.5.3.1: tSDECODE = 4 tCLK = 1.96 us
 * at 2.048 MHz, which a 4 MHz byte (2.0 us) just meets; anything faster needs
 * an explicit gap the DMA cannot insert.
 */
#define ADS_SPI_STREAM_MAX_HZ 4000000u

/** @brief CONFIG3 value for the requested bias state. */
static inline uint8_t ads_config3_value(bool bias_on) {
    return (uint8_t)(bias_on ? ADS_CONFIG3_BIAS_ON : ADS_CONFIG3_REF_ONLY);
}

/** @brief CONFIG4 value: comparators on only when lead-off sensing is wanted. */
static inline uint8_t ads_config4_value(bool leadoff_on) {
    return (uint8_t)(leadoff_on ? ADS_CONFIG4_PD_LOFF_COMP : 0x00u);
}

/**
 * @brief Bits of @p reg that read back what was written.
 *
 * @details CONFIG3 bit 0 (BIAS_STAT) is read-only status; ID and LOFF_STATP/N
 *          are entirely read-only and cannot be verified.
 */
static inline uint8_t ads_reg_readback_mask(uint8_t reg) {
    switch (reg) {
    case ADS_REG_CONFIG3:
        return (uint8_t)~ADS_CONFIG3_BIAS_STAT;
    case ADS_REG_ID:
    case ADS_REG_LOFF_STATP:
    case ADS_REG_LOFF_STATN:
        return 0x00u;
    default:
        return 0xFFu;
    }
}

/** @brief true if @p got matches @p wrote on every writable bit of @p reg. */
static inline bool ads_reg_readback_ok(uint8_t reg, uint8_t wrote, uint8_t got) {
    uint8_t m = ads_reg_readback_mask(reg);
    return (uint8_t)(wrote & m) == (uint8_t)(got & m);
}

/**
 * @brief ID register check (SBAS499C Table 12).
 *
 * @details Bit 4 always reads 1, DEV_ID[3:2] = 11 (ADS1299-x), NU_CH[1:0] = 10
 *          (8 channels; this board needs CH8 for the DRL). REV_ID[7:5] may change
 *          with silicon revision and is ignored. A bus that floats high (0xFF) or
 *          low (0x00) fails.
 */
static inline bool ads_id_valid(uint8_t raw) {
    return (raw & 0x1Fu) == 0x1Eu;
}

/** @brief PGA gains the ADS1299 supports (SBAS499C Table 17). */
static inline bool ads_gain_supported(uint8_t gain) {
    switch (gain) {
    case 1u:
    case 2u:
    case 4u:
    case 6u:
    case 8u:
    case 12u:
    case 24u:
        return true;
    default:
        return false;
    }
}

/** @brief CHnSET for a powered muscle channel at @p gain (normal input, SRB2 off). */
static inline uint8_t ads_chset_for_gain(uint8_t gain) {
    uint8_t pga;
    switch (gain) {
    case 1u: pga = 0x0u; break;
    case 2u: pga = 0x1u; break;
    case 4u: pga = 0x2u; break;
    case 8u: pga = 0x4u; break;
    case 12u: pga = 0x5u; break;
    case 24u: pga = 0x6u; break;
    case 6u:
    default: pga = 0x3u; break;
    }
    return (uint8_t)(pga << 4);
}

/**
 * @brief CHnSET routing the internal test signal at the channel's own gain.
 *
 * @details @p chset_normal is the channel's normal CHnSET (its gain bits are
 *          kept). The session file scales every channel by its configured gain
 *          (muscle gain, 24 for CH7/CH8); the old fixed gain 1 made a
 *          test-signal take read 6x too small (bench, 2026-09-25). The signal,
 *          +-VREF/2.4 mV = +-1.875 mV, stays far below full scale at gain 24.
 */
static inline uint8_t ads_chset_test(uint8_t chset_normal) {
    return (uint8_t)((chset_normal & 0x70u) | ADS_CHSET_MUX_TEST);
}

/**
 * @brief Lead-off excitation currents usable for skin contact (LOFF ILEAD_OFF 00/01).
 *
 * @details 6 uA / 24 uA exist but rail the PGA on a normal electrode pair, and
 *          the config field is nA in a uint8_t anyway.
 */
static inline bool ads_leadoff_current_supported(uint8_t current_na) {
    return current_na == 6u || current_na == 24u;
}

/** @brief Sample rates this firmware can stream (250..2000 SPS). */
static inline bool ads_sample_rate_supported(uint32_t fs_hz) {
    return fs_hz == 250u || fs_hz == 500u || fs_hz == 1000u || fs_hz == 2000u || fs_hz == 4000u;
}

/** @brief CONFIG1 for @p fs_hz (SBAS499C Table 13); unsupported -> 250 SPS. */
static inline uint8_t ads_config1_for_fs(uint32_t fs_hz) {
    switch (fs_hz) {
    case 500u: return 0x95u;
    case 1000u: return 0x94u;
    case 2000u: return 0x93u;
    case 4000u: return 0x92u;
    default: return 0x96u;
    }
}

/**
 * @brief CONFIG2 with the internal test signal switched on/off.
 *
 * @details Bits 7:5 are reserved and must be written 110b, bit 3 must be 0
 *          (SBAS499C Table 14); only INT_CAL (bit 4) changes, CAL_AMP/CAL_FREQ
 *          are kept. The previous masks (0x87 / ~0x70) cleared reserved bit 6.
 */
static inline uint8_t ads_config2_test_signal(uint8_t cfg2, bool enable) {
    return (uint8_t)(0xC0u | (cfg2 & 0x07u) | (enable ? 0x10u : 0x00u));
}

/**
 * @brief CH8SET for the current mode.
 *
 * @details DRL probe: plain electrode input so IN8N is read passively.
 *          Bias on: MUX=111 routes the bias drive to IN8N (the DRL pin).
 *          Otherwise IN8N/IN8P are an open pin pair -> power the channel down.
 */
static inline uint8_t ads_ch8set(bool bias_on, bool drl_probe) {
    if (drl_probe) {
        return ADS_CHSET_NORMAL24;
    }
    return bias_on ? ADS_CHSET_DRL : ADS_CHSET_OFF;
}

/**
 * @brief Comparator bits that may be published for the given LOFF_SENS masks.
 *
 * @details SBAS499C §9.6.1.12/13: LOFF_STATP/N bits are only valid where the
 *          matching LOFF_SENSP/N bit is set. CH7 (no patient pin) and CH8 (IN8P
 *          unwired; IN8N bias-driven or probed via RMS on the host) are never
 *          meaningful, so only CH1..CH6 survive (contract K8).
 */
static inline void ads_leadoff_status_masks(uint8_t sensp, uint8_t sensn, uint8_t *mask_p,
                                            uint8_t *mask_n) {
    *mask_p = (uint8_t)(sensp & ADS_MUSCLE_MASK);
    *mask_n = (uint8_t)(sensn & ADS_MUSCLE_MASK);
}

/** @brief true if any published (masked) comparator reports an open lead. */
static inline bool ads_leadoff_any(uint8_t statp, uint8_t statn, uint8_t sensp, uint8_t sensn) {
    uint8_t mp;
    uint8_t mn;
    ads_leadoff_status_masks(sensp, sensn, &mp, &mn);
    return ((statp & mp) | (statn & mn)) != 0u;
}

/**
 * @brief Split the 24-bit stream status word into LOFF_STATP / LOFF_STATN.
 *
 * @details Layout 1100 + LOFF_STATP[8] + LOFF_STATN[8] + GPIO[4] (SBAS499C §9.4.4.2).
 */
static inline void ads_status_word_leadoff(const uint8_t status[3], uint8_t *statp,
                                           uint8_t *statn) {
    *statp = (uint8_t)(((status[0] & 0x0Fu) << 4) | ((status[1] & 0xF0u) >> 4));
    *statn = (uint8_t)(((status[1] & 0x0Fu) << 4) | ((status[2] & 0xF0u) >> 4));
}

/** @brief true if the fixed 1100b prefix is present (RDATA framing intact). */
static inline bool ads_status_word_framed(const uint8_t status[3]) {
    return (status[0] & 0xF0u) == 0xC0u;
}

/** @brief Clamp the requested stream SPI rate to what tSDECODE allows. */
static inline uint32_t ads_clamp_stream_hz(uint32_t hz) {
    return hz > ADS_SPI_STREAM_MAX_HZ ? ADS_SPI_STREAM_MAX_HZ : hz;
}

/**
 * @brief Conversion tracker for the level-polled DRDY line.
 *
 * @details @c ref_us estimates when the conversion that was last read actually
 *          completed. It is re-anchored to the detection time whenever that
 *          detection was prompt, and otherwise advanced by whole periods, so a
 *          single late read does not shift the phase for the next one.
 */
typedef struct {
    uint64_t ref_us; /**< Estimated DRDY falling edge of the last read conversion. */
    bool valid;      /**< false until the first read of a stream. */
} ads_drdy_track_t;

/**
 * @brief Conversions skipped since the last read, from the DRDY detection time.
 *
 * @details The ADS converts every @p period_us and a read always returns the
 *          newest conversion, so k = floor((t - ref) / T) conversions have
 *          completed since the last one read and k - 1 were lost. floor, not
 *          round: after a stall the detection lag behind the newest edge is
 *          uniform over [0, T), and round() would count a phantom loss for
 *          every stall that ends in the second half of a period. A small
 *          tolerance (T/16) absorbs the anchor's own polling latency and
 *          ADS-oscillator drift; a detection before the predicted edge is always
 *          the next conversion (k = 1), since a read cannot precede DRDY.
 *
 * @param s         Tracker state (zero-init for a new stream).
 * @param t_us      Time DRDY was seen low for this read.
 * @param period_us Conversion period (1e6 / fs).
 * @return Conversions missed before this one (0 when none).
 */
static inline uint32_t ads_drdy_track(ads_drdy_track_t *s, uint64_t t_us, uint32_t period_us) {
    if (!s->valid || period_us == 0u) {
        s->ref_us = t_us;
        s->valid = true;
        return 0u;
    }
    uint64_t tol = period_us / 16u;
    uint64_t k = 1u;
    if (t_us + tol >= s->ref_us + period_us) {
        k = (t_us + tol - s->ref_us) / period_us;
    }
    uint64_t pred = s->ref_us + k * period_us;
    s->ref_us = (t_us < pred + tol) ? t_us : pred;
    uint64_t missed = k - 1u;
    return missed > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)missed;
}
