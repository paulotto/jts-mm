// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file bno085.c
 * @brief BNO085 SHTP/SH-2 driver.
 */

#include "jtsmm/drivers/bno085.h"

#include "jtsmm/pins.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/runtime/usb_service.h"
#include "hardware/i2c.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

/* Floor of every blocking transaction timeout; scaled up with length / baud
 * (i2c_xfer_timeout_us) so a long packet at 100 kHz is not cut off. */
#define BNO085_I2C_TIMEOUT_US 10000u
#define BNO085_I2C_RETRIES 20u

/** @brief Internal helper: int_asserted. */
static bool int_asserted(uint pin) {
    return gpio_get(pin) == 0;
}

/** @brief Internal helper: blocking timeout for @p nbytes on @p dev's bus. */
static uint bno085_timeout_us(const bno085_t *dev, size_t nbytes) {
    return i2c_xfer_timeout_us((uint32_t)nbytes, dev->baud_hz, BNO085_I2C_TIMEOUT_US);
}

void bno085_bus_config(bno085_t *dev, uint baud) {
    gpio_set_function(dev->sda_pin, GPIO_FUNC_I2C);
    gpio_set_function(dev->scl_pin, GPIO_FUNC_I2C);
    gpio_pull_up(dev->sda_pin);
    gpio_pull_up(dev->scl_pin);
    dev->baud_hz = i2c_init(dev->i2c, baud);
}

void bno085_init_dev(bno085_t *dev, i2c_inst_t *i2c, uint8_t addr, uint8_t int_pin, uint sda_pin,
                     uint scl_pin, const char *label) {
    memset(dev, 0, sizeof(*dev));
    dev->i2c = i2c;
    dev->addr = addr;
    dev->int_pin = int_pin;
    dev->sda_pin = (uint8_t)sda_pin;
    dev->scl_pin = (uint8_t)scl_pin;
    dev->label = label;
    gpio_init(int_pin);
    gpio_set_dir(int_pin, GPIO_IN);
    gpio_pull_up(int_pin);
}

bool bno085_hw_reset(void) {
    gpio_init(PIN_IMU_NRST);
    gpio_set_dir(PIN_IMU_NRST, GPIO_OUT);
    gpio_put(PIN_IMU_NRST, 1);
    usb_service_sleep_ms(10);
    gpio_put(PIN_IMU_NRST, 0);
    usb_service_sleep_ms(10);
    gpio_put(PIN_IMU_NRST, 1);
    usb_service_sleep_ms(300);
    return true;
}

/**
 * @brief Internal helper: i2c_write_retry.
 *
 * @details @c dev->write_tries attempts (default 20). Each failed attempt can
 *          block for a full timeout, so the default costs ~200 ms against a
 *          silent device -- fine at boot, but the cooperative recovery runs on
 *          the comms core between ring drains and sets 1, retrying from its
 *          state machine instead (otherwise the EMG ring overflows).
 */
static int i2c_write_retry(bno085_t *dev, const uint8_t *buf, size_t len, bool nostop) {
    uint32_t tries = dev->write_tries ? dev->write_tries : BNO085_I2C_RETRIES;
    for (uint32_t attempt = 0; attempt < tries; attempt++) {
        int ret = i2c_write_timeout_us(dev->i2c, dev->addr, buf, len, nostop, bno085_timeout_us(dev, len));
        if (ret == (int)len) {
            return ret;
        }
        if (attempt + 1u < tries) {
            usb_service_poll();
        }
    }
    return PICO_ERROR_TIMEOUT;
}

bool bno085_write_packet(bno085_t *dev, uint8_t channel, const uint8_t *payload, uint16_t payload_len) {
    if (channel >= sizeof(dev->seq) || payload_len + 4u > BNO085_SHTP_MAX_PACKET) {
        return false;
    }
    uint8_t buf[BNO085_SHTP_MAX_PACKET];
    uint16_t total = (uint16_t)(payload_len + 4u);
    buf[0] = (uint8_t)(total & 0xFFu);
    buf[1] = (uint8_t)((total >> 8) & 0x7Fu);
    buf[2] = channel;
    buf[3] = dev->seq[channel]++;
    memcpy(buf + 4, payload, payload_len);
    int ret = i2c_write_retry(dev, buf, total, false);
    return ret == (int)total;
}

bool bno085_read_packet(bno085_t *dev, uint8_t *buf, size_t buf_cap, size_t *out_len) {
    if (buf_cap < 4) {
        return false;
    }
    uint8_t header[4];
    int ret = i2c_read_timeout_us(dev->i2c, dev->addr, header, 4, false, bno085_timeout_us(dev, 4));
    if (ret != 4) {
        return false;
    }
    uint16_t pkt_len = (uint16_t)(header[0] | ((uint16_t)header[1] << 8));
    pkt_len &= 0x7FFFu;
    if (pkt_len == 0) {
        *out_len = 0;
        return true;
    }
    if (pkt_len < 4 || pkt_len > BNO085_SHTP_MAX_PACKET || pkt_len > buf_cap) {
        return false;
    }
    /*
     * Every I2C read returns the cargo from its 4-byte header on, so a retry
     * must re-read the FULL packet into buf[0]. The old fallback read into
     * buf+4, leaving the header duplicated in the body (misaligned packet).
     * The retry may already deliver the next cargo, so trust the header that
     * came with the bytes, and fail if it is longer than what was read.
     */
    for (int attempt = 0; attempt < 2; attempt++) {
        ret = i2c_read_timeout_us(dev->i2c, dev->addr, buf, pkt_len, false, bno085_timeout_us(dev, pkt_len));
        if (ret != (int)pkt_len) {
            continue;
        }
        bno085_shtp_clear_reread_continuation(buf, header[1]);
        uint16_t got = (uint16_t)((buf[0] | ((uint16_t)buf[1] << 8)) & 0x7FFFu);
        if (got == 0) {
            *out_len = 0;
            return true;
        }
        if (got < 4 || got > pkt_len) {
            return false;
        }
        *out_len = got;
        return true;
    }
    return false;
}

bool bno085_try_read_packet(bno085_t *dev, uint8_t *buf, size_t buf_cap, size_t *out_len) {
    if (!bno085_int_asserted(dev)) {
        return false;
    }
    return bno085_read_packet(dev, buf, buf_cap, out_len);
}

bool bno085_request_product_id(bno085_t *dev) {
    uint8_t req[] = {BNO085_REPORT_PRODUCT_ID_REQ, 0x00u};
    return bno085_write_packet(dev, BNO085_CH_CONTROL, req, sizeof(req));
}

/** @brief Internal helper: bno085_sync_after_reset. */
static bool bno085_sync_after_reset(bno085_t *dev, uint32_t timeout_ms) {
    uint8_t buf[BNO085_SHTP_MAX_PACKET];
    absolute_time_t end = make_timeout_time_ms(timeout_ms);
    uint32_t packets = 0;
    while (!time_reached(end)) {
        size_t len = 0;
        if (!bno085_read_packet(dev, buf, sizeof(buf), &len)) {
            if (packets > 0 && !int_asserted(dev->int_pin)) {
                return true;
            }
            sleep_ms(10);
            continue;
        }
        if (len >= 4) {
            packets++;
            if (buf[2] == BNO085_CH_WAKE) {
                return true;
            }
        }
    }
    return packets > 0;
}

bool bno085_shtp_soft_reset(bno085_t *dev) {
    uint8_t payload[] = {1};
    memset(dev->seq, 0, sizeof(dev->seq));
    for (int attempt = 0; attempt < 5; attempt++) {
        if (bno085_write_packet(dev, BNO085_CH_EXECUTABLE, payload, sizeof(payload))) {
            usb_service_sleep_ms(300);
            return true;
        }
        usb_service_sleep_ms(30);
    }
    return false;
}

/** @brief Internal helper: bno085_wait_product_id. */
static bool bno085_wait_product_id(bno085_t *dev, uint32_t timeout_ms) {
    uint8_t buf[BNO085_SHTP_MAX_PACKET];
    absolute_time_t end = make_timeout_time_ms(timeout_ms);
    while (!time_reached(end)) {
        size_t len = 0;
        if (!bno085_read_packet(dev, buf, sizeof(buf), &len)) {
            sleep_ms(10);
            continue;
        }
        bno085_parse_state_t parsed;
        memset(&parsed, 0, sizeof(parsed));
        if (bno085_parse_shtp(buf, (uint16_t)len, &parsed) && parsed.product_id) {
            return true;
        }
        if (len >= 6 && buf[2] == BNO085_CH_CONTROL && buf[4] == BNO085_REPORT_PRODUCT_ID_RESP) {
            return true;
        }
    }
    (void)dev;
    return false;
}

bool bno085_boot_device(bno085_t *dev) {
    memset(dev->seq, 0, sizeof(dev->seq));
    if (!bno085_sync_after_reset(dev, 800)) {
        if (!bno085_shtp_soft_reset(dev) || !bno085_sync_after_reset(dev, 3000)) {
            return false;
        }
    }
    if (!bno085_request_product_id(dev)) {
        return false;
    }
    return bno085_wait_product_id(dev, 2000);
}

/** @brief Context of feature_resp_cb(): the report whose 0xFC we wait for. */
typedef struct {
    uint8_t report_id;
    bool found;
    uint32_t interval_us;
} feature_wait_t;

/** @brief Internal helper: pick the Get Feature Response for one report. */
static void feature_resp_cb(void *ctx, const bno085_report_t *r) {
    feature_wait_t *w = (feature_wait_t *)ctx;
    if (r->report_id == BNO085_REPORT_GET_FEATURE_RESP && r->feature_id == w->report_id) {
        w->found = true;
        w->interval_us = r->interval_us;
    }
}

bool bno085_enable_report_drain(bno085_t *dev, uint8_t report_id, uint32_t interval_us,
                                uint32_t *granted_us) {
    if (granted_us != NULL) {
        *granted_us = 0;
    }
    if (!bno085_enable_report(dev, report_id, interval_us)) {
        return false;
    }
    /*
     * Set Feature is answered with a Get Feature Response (0xFC) carrying the
     * interval the sensor actually granted. The old loop waited for 0xF1
     * (Command Response), which Set Feature never sends, so every enable sat
     * out the full 500 ms and the granted rate was thrown away.
     */
    feature_wait_t w = {.report_id = report_id, .found = false, .interval_us = 0};
    absolute_time_t end = make_timeout_time_ms(500);
    while (!time_reached(end)) {
        if (!int_asserted(dev->int_pin)) {
            sleep_ms(2);
            continue;
        }
        uint8_t buf[BNO085_SHTP_MAX_PACKET];
        size_t len = 0;
        if (!bno085_read_packet(dev, buf, sizeof(buf), &len)) {
            sleep_ms(2);
            continue;
        }
        bno085_packet_info_t info;
        (void)bno085_parse_packet(buf, (uint16_t)len, feature_resp_cb, &w, &info);
        if (w.found) {
            if (granted_us != NULL) {
                *granted_us = w.interval_us;
            }
            return true;
        }
    }
    return true; /* written; the response just did not show up (granted stays 0) */
}

bool bno085_enable_report(bno085_t *dev, uint8_t report_id, uint32_t interval_us) {
    uint8_t cmd[17];
    memset(cmd, 0, sizeof(cmd));
    cmd[0] = BNO085_REPORT_SET_FEATURE;
    cmd[1] = report_id;
    cmd[5] = (uint8_t)(interval_us & 0xFFu);
    cmd[6] = (uint8_t)((interval_us >> 8) & 0xFFu);
    cmd[7] = (uint8_t)((interval_us >> 16) & 0xFFu);
    cmd[8] = (uint8_t)((interval_us >> 24) & 0xFFu);
    return bno085_write_packet(dev, BNO085_CH_CONTROL, cmd, sizeof(cmd));
}

// Command Request (0xF2) payload: [id][cmd_seq][command][P0..P8] = 12 bytes.
/** @brief Internal helper: bno085_command_request. */
static bool bno085_command_request(bno085_t *dev, uint8_t command, const uint8_t params[9]) {
    uint8_t cmd[12];
    memset(cmd, 0, sizeof(cmd));
    cmd[0] = BNO085_REPORT_COMMAND_REQ;
    cmd[1] = dev->cmd_seq++;
    cmd[2] = command;
    if (params != NULL) {
        memcpy(cmd + 3, params, 9);
    }
    return bno085_write_packet(dev, BNO085_CH_CONTROL, cmd, sizeof(cmd));
}

bool bno085_set_calibration(bno085_t *dev, bool accel, bool gyro, bool mag) {
    // ME Calibration params: P0=accel, P1=gyro, P2=mag, P3=0, P4=planar accel.
    uint8_t p[9] = {0};
    p[0] = accel ? 1u : 0u;
    p[1] = gyro ? 1u : 0u;
    p[2] = mag ? 1u : 0u;
    return bno085_command_request(dev, BNO085_CMD_ME_CALIBRATE, p);
}

bool bno085_save_dcd(bno085_t *dev) {
    return bno085_command_request(dev, BNO085_CMD_SAVE_DCD, NULL);
}

bool bno085_int_asserted(const bno085_t *dev) {
    return int_asserted(dev->int_pin);
}

/**
 * @brief Internal helper: length of one input-channel record, 0 if unknown.
 *
 * @details Sizes per SH-2 §6.5 / CEVA sh2.c. Every record of a batched packet
 *          must be sized correctly or all records after it are lost, so an
 *          unknown id ends the packet (flagged as truncated).
 */
static size_t bno085_input_report_size(uint8_t report_id) {
    switch (report_id) {
    case BNO085_REPORT_BASE_TIMESTAMP:
    case BNO085_REPORT_TIMESTAMP_REBASE:
        return 5u;
    case BNO085_REPORT_ACCELEROMETER:
    case BNO085_REPORT_GYROSCOPE:
    case BNO085_REPORT_MAGNETOMETER:
    case BNO085_REPORT_LINEAR_ACCEL:
    case 0x06u: /* gravity */
        return 10u;
    case BNO085_REPORT_GAME_ROT_VEC:
    case 0x29u: /* AR/VR-stabilized game RV */
        return 12u;
    case BNO085_REPORT_ROTATION_VECTOR:
    case BNO085_REPORT_GEOMAG_ROT_VEC:
    case 0x28u: /* AR/VR-stabilized RV */
        /* These carry a 2-byte accuracy estimate after the quaternion (SH-2
         * §6.5.18 / §6.5.20); only the *game* RV omits it. */
        return 14u;
    case 0x07u: /* gyroscope uncalibrated */
    case 0x0Fu: /* magnetic field uncalibrated */
    case 0x14u: /* raw accelerometer */
    case 0x15u: /* raw gyroscope */
    case 0x16u: /* raw magnetometer */
        return 16u;
    default:
        return 0u;
    }
}

/** @brief Internal helper: little-endian int16 at @p p. */
static inline int16_t rd_i16(const uint8_t *p) {
    return (int16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

/** @brief Internal helper: little-endian uint32 at @p p. */
static inline uint32_t rd_u32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

/**
 * @brief Internal helper: channel 3/4 payload -- a batch of sized records.
 *
 * @details The old parser folded every record into one struct, so a packet
 *          holding two accel samples (normal under batching) kept only the
 *          last; it also only skipped a 0xFB at offset 0 and stopped at 0xFA,
 *          dropping the rest of the packet. Now each record is delivered.
 */
static void parse_input_payload(const uint8_t *p, size_t plen, bno085_report_cb_t cb, void *ctx,
                                bno085_packet_info_t *info) {
    size_t off = 0;
    /*
     * SH-2 timing (reference manual §7.2, CEVA sh2.c sensorhubInputHandler):
     * 0xFB [base delta i32 LE] sets the packet's base timestamp to INT
     * assertion - base delta; 0xFA [rebase i32 LE] shifts it for the records
     * that follow; each report's delay (status[7:2] = delay[13:8], byte 3 =
     * delay[7:0]) is its sample time after the base. All in 100 µs ticks.
     */
    bool have_ref = false;
    int32_t ref_ticks = 0;
    while (off < plen) {
        uint8_t id = p[off];
        size_t rlen = bno085_input_report_size(id);
        if (rlen == 0 || off + rlen > plen) {
            info->truncated = true;
            return;
        }
        if (id == BNO085_REPORT_BASE_TIMESTAMP) {
            int32_t base = (int32_t)rd_u32(p + off + 1);
            ref_ticks = (int32_t)(0 - (uint32_t)base); /* wraps like CEVA's -timebase */
            have_ref = true;
            if (!info->has_base) {
                info->has_base = true;
                info->base_ticks = base;
            }
        } else if (id == BNO085_REPORT_TIMESTAMP_REBASE) {
            ref_ticks = (int32_t)((uint32_t)ref_ticks + rd_u32(p + off + 1));
            info->has_rebase = true;
        } else {
            // Common prefix: [id][seq][status][delay][data...]; status bits 1:0 = accuracy.
            const uint8_t *b = p + off;
            bno085_report_t r;
            r.report_id = id;
            r.seq = b[1];
            r.has_seq = true;
            r.accuracy = (uint8_t)(b[2] & 0x03u);
            r.v[0] = rd_i16(b + 4);
            r.v[1] = rd_i16(b + 6);
            r.v[2] = rd_i16(b + 8);
            r.v[3] = (rlen >= 12u) ? rd_i16(b + 10) : 0;
            r.feature_id = 0;
            r.interval_us = 0;
            r.delay = (uint16_t)(((uint16_t)(b[2] & 0xFCu) << 6) | b[3]);
            r.has_ts = have_ref;
            r.ts_ticks = have_ref ? (int32_t)((uint32_t)ref_ticks + r.delay) : 0;
            info->reports++;
            if (cb) {
                cb(ctx, &r);
            }
        }
        off += rlen;
    }
}

/**
 * @brief Internal helper: channel 5 payload -- gyro-integrated RV.
 *
 * @details No report id / seq / status / delay: back-to-back 14-byte records
 *          [i,j,k,real Q14][wx,wy,wz Q10] from byte 0, possibly several per
 *          packet (CEVA sh2.c sensorhubInputGyroRvHandler). The old parser
 *          expected a leading 0x2A id byte, so it never matched a real packet.
 *          Angular velocity has no wire record and is not forwarded.
 */
static void parse_gyro_rv_payload(const uint8_t *p, size_t plen, bno085_report_cb_t cb, void *ctx,
                                  bno085_packet_info_t *info) {
    size_t off = 0;
    for (; off + BNO085_GYRO_RV_REPORT_LEN <= plen; off += BNO085_GYRO_RV_REPORT_LEN) {
        bno085_report_t r;
        r.report_id = BNO085_REPORT_GYRO_INT_ROT_VEC;
        r.seq = 0;
        r.has_seq = false;
        r.accuracy = 0xFFu;
        r.v[0] = rd_i16(p + off);
        r.v[1] = rd_i16(p + off + 2);
        r.v[2] = rd_i16(p + off + 4);
        r.v[3] = rd_i16(p + off + 6);
        r.feature_id = 0;
        r.interval_us = 0;
        r.delay = 0;
        r.has_ts = false; /* no timing fields: the report keeps t_int (as in sh2.c) */
        r.ts_ticks = 0;
        info->reports++;
        if (cb) {
            cb(ctx, &r);
        }
    }
    if (off != plen) {
        info->truncated = true;
    }
}

/** @brief Internal helper: channel 2 payload -- control responses. */
static void parse_control_payload(const uint8_t *p, size_t plen, bno085_report_cb_t cb, void *ctx,
                                  bno085_packet_info_t *info) {
    size_t off = 0;
    while (off < plen) {
        uint8_t id = p[off];
        size_t rlen;
        if (id == BNO085_REPORT_PRODUCT_ID_RESP) {
            info->product_id = true;
            rlen = 16u;
        } else if (id == BNO085_REPORT_COMMAND_RESP) {
            info->command_resp = true;
            rlen = 16u;
        } else if (id == BNO085_REPORT_GET_FEATURE_RESP) {
            // [0xFC][report id][flags][sensitivity u16][interval u32][batch u32][specific u32]
            rlen = 17u;
            if (off + 9u > plen) {
                return;
            }
            info->feature_resp = true;
            info->feature_report_id = p[off + 1];
            info->feature_interval_us = rd_u32(p + off + 5);
            bno085_report_t r;
            memset(&r, 0, sizeof(r));
            r.report_id = BNO085_REPORT_GET_FEATURE_RESP;
            r.accuracy = 0xFFu;
            r.feature_id = info->feature_report_id;
            r.interval_us = info->feature_interval_us;
            info->reports++;
            if (cb) {
                cb(ctx, &r);
            }
        } else {
            return; /* other control responses (FRS etc.) are not used here */
        }
        off += rlen;
    }
}

void bno085_seq_reset(bno085_seq_tracker_t *t) {
    t->valid = 0;
}

uint32_t bno085_seq_update(bno085_seq_tracker_t *t, uint8_t report_id, uint8_t seq) {
    if (report_id >= BNO085_SEQ_TRACK_IDS) {
        return 0;
    }
    uint64_t bit = (uint64_t)1u << report_id;
    uint8_t last = t->last[report_id];
    t->last[report_id] = seq;
    if ((t->valid & bit) == 0) {
        t->valid |= bit;
        return 0;
    }
    uint8_t diff = (uint8_t)(seq - last);
    return diff == 0u ? 0u : (uint32_t)(diff - 1u);
}

bool bno085_parse_packet(const uint8_t *pkt, uint16_t len, bno085_report_cb_t cb, void *ctx,
                         bno085_packet_info_t *info) {
    memset(info, 0, sizeof(*info));
    if (pkt == NULL || len < 4) {
        return false;
    }
    uint16_t hdr_len = (uint16_t)(pkt[0] | ((uint16_t)pkt[1] << 8));
    info->channel = pkt[2];
    info->shtp_seq = pkt[3];
    if (hdr_len & BNO085_SHTP_CONTINUATION) {
        /* Tail of a transfer larger than one read: its first bytes are not a
         * report boundary, so parsing it as a fresh packet would emit garbage. */
        info->continuation = true;
        return false;
    }
    const uint8_t *payload = pkt + 4;
    size_t payload_len = len - 4u;
    if (payload_len == 0) {
        return true;
    }
    switch (info->channel) {
    case BNO085_CH_COMMAND:
        info->advertisement = (payload[0] == BNO085_CMD_ADVERTISEMENT);
        break;
    case BNO085_CH_EXECUTABLE:
        info->reset_complete = (payload[0] == BNO085_EXEC_RESET_COMPLETE);
        break;
    case BNO085_CH_CONTROL:
        parse_control_payload(payload, payload_len, cb, ctx, info);
        break;
    case BNO085_CH_INPUT:
    case BNO085_CH_WAKE:
        parse_input_payload(payload, payload_len, cb, ctx, info);
        break;
    case BNO085_CH_GYRO_RV:
        parse_gyro_rv_payload(payload, payload_len, cb, ctx, info);
        break;
    default:
        break;
    }
    return true;
}

/** @brief Internal helper: legacy last-wins fill of bno085_parse_state_t. */
static void parse_state_cb(void *ctx, const bno085_report_t *r) {
    bno085_parse_state_t *out = (bno085_parse_state_t *)ctx;
    switch (r->report_id) {
    case BNO085_REPORT_ROTATION_VECTOR:
    case BNO085_REPORT_GAME_ROT_VEC:
    case BNO085_REPORT_GEOMAG_ROT_VEC:
    case BNO085_REPORT_GYRO_INT_ROT_VEC:
        out->got_rotation = true;
        out->quat_report_id = r->report_id;
        out->quat_accuracy = r->accuracy;
        out->quat_i = r->v[0];
        out->quat_j = r->v[1];
        out->quat_k = r->v[2];
        out->quat_real = r->v[3];
        break;
    case BNO085_REPORT_ACCELEROMETER:
        out->got_accel = true;
        out->accel_accuracy = r->accuracy;
        out->accel_x = r->v[0];
        out->accel_y = r->v[1];
        out->accel_z = r->v[2];
        break;
    case BNO085_REPORT_GYROSCOPE:
        out->got_gyro = true;
        out->gyro_accuracy = r->accuracy;
        out->gyro_x = r->v[0];
        out->gyro_y = r->v[1];
        out->gyro_z = r->v[2];
        break;
    case BNO085_REPORT_MAGNETOMETER:
        out->got_mag = true;
        out->mag_accuracy = r->accuracy;
        out->mag_x = r->v[0];
        out->mag_y = r->v[1];
        out->mag_z = r->v[2];
        break;
    case BNO085_REPORT_LINEAR_ACCEL:
        // Linear acceleration is reported as Q8 int16 in m/s^2 (SH-2 6.5.10).
        out->got_linear_accel = true;
        out->lin_accel_accuracy = r->accuracy;
        out->lin_ax = (float)r->v[0] / 256.0f;
        out->lin_ay = (float)r->v[1] / 256.0f;
        out->lin_az = (float)r->v[2] / 256.0f;
        break;
    default:
        break;
    }
}

bool bno085_parse_shtp(const uint8_t *pkt, uint16_t len, bno085_parse_state_t *out) {
    if (pkt == NULL || out == NULL || len < 5) {
        return false;
    }
    out->quat_accuracy = out->accel_accuracy = 0xFFu;
    out->gyro_accuracy = out->mag_accuracy = out->lin_accel_accuracy = 0xFFu;
    bno085_packet_info_t info;
    (void)bno085_parse_packet(pkt, len, parse_state_cb, out, &info);
    out->product_id = out->product_id || info.product_id;
    out->set_feature_ack = out->set_feature_ack || info.command_resp || info.feature_resp;
    return out->got_accel || out->got_gyro || out->got_mag || out->got_rotation ||
           out->got_linear_accel || out->product_id || out->set_feature_ack;
}
