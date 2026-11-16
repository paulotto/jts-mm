// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_config.c
 * @brief IMU domain apply/stream/recovery.
 */

#include "jtsmm/domain/imu_config.h"

#include "jtsmm/domain/bno085_reports.h"
#include "jtsmm/domain/imu_timing.h"
#include "jtsmm/drivers/bno085.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/config.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"
#include "jtsmm/runtime/timebase.h"
#include "jtsmm/runtime/config_apply.h"
#include "jtsmm/runtime/usb_service.h"
#include "jtsmm/runtime/host_link.h"
#include "hardware/gpio.h"
#include "hardware/sync.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

static imu_slot_config_t active_cfg[2];
static volatile bool streaming_slot[2];
static bool stream_wanted[2]; /* host intent (imu_stream on/off); comms core only */

/* Outcome of the last apply / recovery per slot, for imu_status. */
typedef enum {
    IMU_SLOT_STATE_UNCONFIGURED = 0,
    IMU_SLOT_STATE_DISABLED,
    IMU_SLOT_STATE_ABSENT,
    IMU_SLOT_STATE_BOOT_FAIL,
    IMU_SLOT_STATE_OK,
} imu_slot_state_t;
static imu_slot_state_t slot_state[2];
static const char *const slot_state_names[] = {"unconfigured", "disabled", "absent", "boot_fail", "ok"};

static void imu_recover_cancel(imu_slot_t slot);

// Report ids enabled by the previous apply per slot, so a re-apply can turn the
// old set off before enabling the new one (the BNO085 keeps streaming a report
// until it is explicitly disabled — otherwise stale reports leak onto the bus).
static uint8_t prev_report_ids[2][IMU_MAX_REPORTS];
static uint8_t prev_report_count[2];

// Latest SH-2 accuracy / calibration status per slot, captured live from the
// stream (status byte bits 1:0 -> 0..3). Index: [slot][kind], kind 0=quat,
// 1=accel, 2=gyro, 3=mag. 0xFF = not observed yet.
static volatile uint8_t slot_accuracy[2][4] = {
    {0xFFu, 0xFFu, 0xFFu, 0xFFu},
    {0xFFu, 0xFFu, 0xFFu, 0xFFu},
};

/*
 * Stream health, written only by the sensor core in the completion path and
 * read by the comms core (REC_STATS, imu_status):
 *  - seq_gaps: reports lost between the BNO085 and the ring, from the SH-2
 *    per-report sequence number (byte 1 of every input report); channel-5
 *    gyro RV has none, so its SHTP packet sequence is used (lost packets).
 *  - reset_count: spontaneous device resets seen while streaming (channel-1
 *    reset-complete or channel-0 advertisement). reset_pending asks the comms
 *    core to re-enable the reports, which a reset silently turned off.
 */
static volatile uint32_t seq_gaps[2];
static volatile uint32_t reset_count[2];
static volatile uint32_t cont_packets[2];
static volatile uint32_t trunc_packets[2];
static volatile bool reset_pending[2];
static volatile uint32_t stream_start_us32[2];
static bno085_seq_tracker_t seq_trk[2]; /* sensor core while streaming; comms core otherwise */

/* Report interval the BNO085 granted (Get Feature Response 0xFC) per
 * active_cfg report index; 0 = no response seen. Filled at apply and kept
 * current from the stream (re-enables after a recovery, and the sensor's own
 * unsolicited 0xFC when another report changes its schedule). */
static volatile uint32_t granted_us[2][IMU_MAX_REPORTS];

/* Timestamp statistics per slot (imu_timing.h). Written by the sensor core
 * only (reset when it sees a stream (re)start), read by imu_status. */
static imu_ts_stats_t ts_stats[2];
/* Bumped by the comms core every time it (re)starts a slot's stream, so the
 * sensor core notices a restart even when it never saw streaming_slot drop. */
static volatile uint32_t stream_gen[2];
/* Sensor-core view of the stream (imu_ts_track_stream). */
static bool c1_active[2];
static uint32_t c1_gen[2];
static uint32_t edge_max_age_us[2];

/* Reset indications this soon after stream start are leftovers of the boot or
 * recovery that preceded it, not a new reset. */
#define IMU_RESET_GRACE_US 250000u

/** @brief Internal helper: accuracy_label. */
static const char *accuracy_label(uint8_t acc) {
    switch (acc) {
    case 0u: return "unreliable";
    case 1u: return "low";
    case 2u: return "med";
    case 3u: return "high";
    default: return "n/a";
    }
}

/** @brief Internal helper: slot_usb_base. */
static uint8_t slot_usb_base(imu_slot_t slot) {
    return (slot == IMU_SLOT_CRA) ? REC_IMU_CRA_QUAT : REC_IMU_MTA_QUAT;
}

/** @brief Internal helper: slot_ring. */
static ring_id_t slot_ring(imu_slot_t slot) {
    return (slot == IMU_SLOT_CRA) ? RING_IMU_CRA : RING_IMU_MTA;
}

/** @brief Internal helper: slot_dma. */
static i2c_dma_slot_t slot_dma(imu_slot_t slot) {
    return (slot == IMU_SLOT_CRA) ? I2C_DMA_SLOT_CRA : I2C_DMA_SLOT_MTA;
}

/* Bound on the comms-core wait for the sensor core to release a slot's bus. */
#define IMU_QUIESCE_TIMEOUT_US 50000u

/**
 * @brief Internal helper: take @p slot off the streaming path (comms core).
 *
 * @details Clears streaming first so the sensor core stops kicking reads, then
 *          hands the in-flight DMA abort to the sensor core and waits for its
 *          acknowledge (i2c_dma_quiesce). Only after a true return may this
 *          core drive the slot's I2C bus with blocking calls.
 */
static bool imu_slot_quiesce(imu_slot_t slot) {
    streaming_slot[slot] = false;
    __dmb();
    imu_int_edge_clear(slot);
    return i2c_dma_quiesce(slot_dma(slot), IMU_QUIESCE_TIMEOUT_US);
}

void imu_config_resolve_profile(imu_slot_config_t *cfg) {
    if (cfg->profile == IMU_PROFILE_CUSTOM) {
        return;
    }
    cfg->num_reports = 0;
    uint8_t fusion_id = bno085_fusion_report_id(cfg->fusion_primary);
    const bno085_report_entry_t *fusion = bno085_report_find(fusion_id);
    if (cfg->profile == IMU_PROFILE_FUSED_ONLY && fusion) {
        cfg->reports[0] = (imu_report_cfg_t){
            .enabled = true,
            .report_id = fusion_id,
            .rate_hz = cfg->fusion_rate_hz,
            .usb_type = fusion->usb_type,
            .payload_len = fusion->payload_len,
        };
        cfg->num_reports = 1;
        return;
    }
    if (cfg->profile == IMU_PROFILE_KINEMATICS_ACCEL) {
        // Fixed rates matching host IMU_PROFILE_REPORTS["kinematics_accel"]
        // (catalog.py): fusion @ fusion_rate_hz + calibrated accelerometer (gravity
        // included) @ 250, the slot linear_accel holds in kinematics_default. The
        // host removes gravity offline with the optical orientation.
        const bno085_report_entry_t *acc = bno085_report_find(0x01);
        if (fusion) {
            cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                .enabled = true, .report_id = fusion_id, .rate_hz = cfg->fusion_rate_hz,
                .usb_type = fusion->usb_type, .payload_len = fusion->payload_len};
        }
        if (acc) {
            cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                .enabled = true, .report_id = 0x01, .rate_hz = 250u,
                .usb_type = acc->usb_type, .payload_len = acc->payload_len};
        }
        return;
    }
    if (cfg->profile == IMU_PROFILE_KINEMATICS_DEFAULT) {
        // Fixed rates matching host IMU_PROFILE_REPORTS["kinematics_default"]
        // (catalog.py): fusion @ fusion_rate_hz + linear_accel @ 250. linear_accel
        // is requested at 250 (not 200) because 200 Hz aligns badly with the
        // fusion schedule and collapses to ~120 Hz, while 250 quantizes cleanly to
        // ~220 Hz. Gyro is intentionally omitted -- fusion + gyro + lin throttles
        // all three (~331/331/79).
        const bno085_report_entry_t *lin = bno085_report_find(0x04);
        if (fusion) {
            cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                .enabled = true, .report_id = fusion_id, .rate_hz = cfg->fusion_rate_hz,
                .usb_type = fusion->usb_type, .payload_len = fusion->payload_len};
        }
        if (lin) {
            cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                .enabled = true, .report_id = 0x04, .rate_hz = 250u,
                .usb_type = lin->usb_type, .payload_len = lin->payload_len};
        }
        return;
    }
    if (cfg->profile == IMU_PROFILE_RAW_ONLY) {
        const uint8_t raw_ids[] = {0x01, 0x02};
        for (size_t i = 0; i < sizeof(raw_ids) / sizeof(raw_ids[0]); i++) {
            const bno085_report_entry_t *e = bno085_report_find(raw_ids[i]);
            if (e) {
                cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                    .enabled = true, .report_id = e->report_id, .rate_hz = e->default_rate_hz,
                    .usb_type = e->usb_type, .payload_len = e->payload_len};
            }
        }
        return;
    }
    if (cfg->profile == IMU_PROFILE_ALL) {
        // datasheet_motion + one fusion quaternion; capped at IMU_MAX_REPORTS.
        const uint8_t all_ids[] = {0x01, 0x02, 0x03, 0x04};
        for (size_t i = 0;
             i < sizeof(all_ids) / sizeof(all_ids[0]) && cfg->num_reports < IMU_MAX_REPORTS; i++) {
            const bno085_report_entry_t *e = bno085_report_find(all_ids[i]);
            if (e) {
                cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                    .enabled = true, .report_id = e->report_id, .rate_hz = e->default_rate_hz,
                    .usb_type = e->usb_type, .payload_len = e->payload_len};
            }
        }
        if (fusion && cfg->num_reports < IMU_MAX_REPORTS) {
            cfg->reports[cfg->num_reports++] = (imu_report_cfg_t){
                .enabled = true, .report_id = fusion_id, .rate_hz = cfg->fusion_rate_hz,
                .usb_type = fusion->usb_type, .payload_len = fusion->payload_len};
        }
        return;
    }
}

/** @brief Internal helper: slot name for logs. */
static const char *slot_name(imu_slot_t slot) {
    return (slot == IMU_SLOT_CRA) ? "cra" : "mta";
}

bool imu_config_validate_slot(const imu_slot_config_t *cfg, uint16_t emg_fs_hz) {
    if (cfg == NULL) {
        return false;
    }
    if (!cfg->enabled) {
        return true; /* nothing is sent to a disabled slot */
    }
    /* Rate 0 used to be enabled silently at 400 Hz, and rates above the
     * sensor ceiling were requested as-is; reject both so the host hears
     * about it (config apply FAIL) instead of getting a different stream. */
    if (cfg->profile != IMU_PROFILE_CUSTOM && cfg->profile != IMU_PROFILE_RAW_ONLY &&
        cfg->fusion_rate_hz == 0) {
        printf("imu config: fusion rate_hz 0 invalid\r\n");
        return false;
    }
    for (uint8_t i = 0; i < cfg->num_reports && i < IMU_MAX_REPORTS; i++) {
        const imu_report_cfg_t *r = &cfg->reports[i];
        if (!r->enabled) {
            continue;
        }
        const bno085_report_entry_t *e = bno085_report_find(r->report_id);
        if (e == NULL) {
            printf("imu config: report 0x%02X unknown\r\n", r->report_id);
            return false;
        }
        if (r->rate_hz == 0 || r->rate_hz > e->max_rate_hz) {
            printf("imu config: %s rate_hz %u invalid (1..%u)\r\n", e->name, (unsigned)r->rate_hz,
                   (unsigned)e->max_rate_hz);
            return false;
        }
    }
    if (cfg->sync_mode == IMU_SYNC_EMG_DRDY_DIVISOR) {
        if (cfg->emg_drdy_divisor == 0) {
            return false;
        }
        for (uint8_t i = 0; i < cfg->num_reports && i < IMU_MAX_REPORTS; i++) {
            uint16_t imu_hz = cfg->reports[i].rate_hz;
            if (imu_hz == 0 || (emg_fs_hz % imu_hz) != 0) {
                return false;
            }
            /* Full-width quotient: (uint8_t) truncation let e.g. 16000/50 = 320
             * alias to a divisor of 64 and pass. */
            uint32_t div = (uint32_t)emg_fs_hz / imu_hz;
            if (div > 255u || div != cfg->emg_drdy_divisor) {
                printf("imu config: emg_drdy_divisor %u != %lu (fs %u / %u Hz)\r\n",
                       (unsigned)cfg->emg_drdy_divisor, (unsigned long)div, (unsigned)emg_fs_hz,
                       (unsigned)imu_hz);
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Internal helper (sensor core): drop @p slot's latched edge if INT is released.
 * @return true if an edge was dropped.
 */
static bool imu_edge_drop_if_released(imu_slot_t slot) {
    bno085_t *dev = imu_bus_dev(slot);
    if (dev == NULL || !imu_int_edge_armed(slot)) {
        return false;
    }
    uint32_t irq = save_and_disable_interrupts();
    bool dropped = imu_edge_latch_discard_if_released(imu_int_edge_latch(slot), bno085_int_asserted(dev));
    restore_interrupts(irq);
    return dropped;
}

/**
 * @brief Internal helper (sensor core): the header-only read is done, the full re-read starts.
 *
 * @details The BNO085 still holds the packet after the 4-byte header read and
 *          may re-assert INT until the re-read starts. That edge belongs to the
 *          packet being read, not to the next one: forget it, so the first
 *          edge from here on (the next packet) is the one kept.
 */
static void imu_dma_body_start(i2c_dma_slot_t dma_slot) {
    imu_slot_t slot = (dma_slot == I2C_DMA_SLOT_CRA) ? IMU_SLOT_CRA : IMU_SLOT_MTA;
    if (!imu_int_edge_armed(slot)) {
        return;
    }
    imu_edge_latch_t *l = imu_int_edge_latch(slot);
    uint32_t irq = save_and_disable_interrupts();
    bool had = l->pending;
    imu_edge_latch_clear(l);
    restore_interrupts(irq);
    if (had) {
        ts_stats[slot].hdr_drops++;
    }
}

/** @brief Internal helper: imu_dma_complete. */
static void imu_dma_complete(i2c_dma_slot_t dma_slot, uint64_t t_us, const uint8_t *pkt, uint16_t len) {
    imu_slot_t slot = (dma_slot == I2C_DMA_SLOT_CRA) ? IMU_SLOT_CRA : IMU_SLOT_MTA;
    /* An edge latched during this read whose INT is high again was transient
     * (or its packet came with this read): it must not stamp the next one. */
    if (imu_edge_drop_if_released(slot)) {
        ts_stats[slot].spurious++;
    }
    imu_stream_on_complete(slot, t_us, pkt, len);
}

/** @brief Internal helper: report an apply-time device problem on CDC and to the host log. */
static void apply_report(imu_slot_t slot, imu_slot_state_t state, const char *what) {
    slot_state[slot] = state;
    char msg[112];
    snprintf(msg, sizeof(msg),
             "imu_%s: %s -- slot stays configured; the stream watchdog retries once streaming",
             slot_name(slot), what);
    printf("%s\r\n", msg);
    host_link_send_log(msg);
}

/** @brief Internal helper: turn off every report a previous apply enabled on @p slot. */
static void disable_prev_reports(imu_slot_t slot) {
    bno085_t *dev = imu_bus_dev(slot);
    if (prev_report_count[slot] == 0 || dev == NULL) {
        return;
    }
    imu_bus_prepare_slot(slot);
    if (imu_bus_probe(slot)) {
        for (uint8_t i = 0; i < prev_report_count[slot]; i++) {
            (void)bno085_enable_report(dev, prev_report_ids[slot][i], 0u); // interval 0 disables
        }
    }
    prev_report_count[slot] = 0;
}

bool imu_config_apply_slot(imu_slot_t slot, const imu_slot_config_t *cfg) {
    if ((unsigned)slot >= 2 || cfg == NULL) {
        return false;
    }
    /*
     * Config apply parks the sensor core, so this core owns the slot's DMA:
     * drop a read it may have left mid-flight, and any recovery in progress.
     * The apply also clears the stream flags (stream_imu_* = false), so the
     * slot is left stopped until the host starts it again.
     */
    imu_recover_cancel(slot);
    stream_wanted[slot] = false;
    streaming_slot[slot] = false;
    reset_pending[slot] = false;
    __dmb();
    i2c_dma_abort(slot_dma(slot));
    /* Registered first and unconditionally: an early return below (device
     * absent / boot failed) used to leave the callback unset, so a slot the
     * watchdog later revived dropped every packet it read. */
    i2c_dma_on_complete(slot_dma(slot), imu_dma_complete);
    i2c_dma_on_body_start(slot_dma(slot), imu_dma_body_start);
    imu_int_isr_install(slot);
    imu_int_edge_clear(slot);
    for (uint8_t i = 0; i < IMU_MAX_REPORTS; i++) {
        granted_us[slot][i] = 0;
    }

    active_cfg[slot] = *cfg;
    imu_config_resolve_profile(&active_cfg[slot]);
    active_cfg[slot].enabled = cfg->enabled;
    if (!cfg->enabled) {
        /* Store the disabled config (the old one used to survive, and
         * `imu_stream on` then streamed it) and turn off what the previous
         * apply enabled on the device. */
        disable_prev_reports(slot);
        slot_state[slot] = IMU_SLOT_STATE_DISABLED;
        printf("imu_%s: disabled\r\n", slot_name(slot));
        return true;
    }

    static bool hw_reset_done;
    if (!hw_reset_done) {
        (void)bno085_hw_reset();
        hw_reset_done = true;
    }
    imu_bus_prepare_slot(slot);
    if (!imu_bus_probe(slot)) {
        /* A slave left holding SDA low (e.g. by a reset mid-transfer) makes
         * every probe fail: clear the bus once and look again. */
        (void)imu_bus_clear(slot);
        imu_bus_prepare_slot(slot);
        if (!imu_bus_probe(slot)) {
            /* Not an apply failure on purpose: config apply would then skip
             * launching the sensor core and take EMG down with a missing IMU. */
            apply_report(slot, IMU_SLOT_STATE_ABSENT, "NOT FOUND on I2C (probe failed)");
            return true;
        }
    }

    bno085_t *dev = imu_bus_dev(slot);
    if (dev == NULL) {
        return false;
    }

    bno085_bus_config(dev, jtsmm_config_active()->platform.i2c_hz);
    i2c_dma_set_baud(slot_dma(slot), dev->baud_hz);
    if (!bno085_boot_device(dev)) {
        apply_report(slot, IMU_SLOT_STATE_BOOT_FAIL, "boot FAIL (no SHTP/product id)");
        return true;
    }
    slot_state[slot] = IMU_SLOT_STATE_OK;

    // BNO085 motion-engine dynamic calibration per sensor as configured (default all
    // on, config.h). Safe here: apply runs while the slot is idle (not yet streaming).
    (void)bno085_set_calibration(dev, cfg->dynamic_calibration.accel, cfg->dynamic_calibration.gyro,
                                 cfg->dynamic_calibration.mag);
    slot_accuracy[slot][0] = slot_accuracy[slot][1] = 0xFFu;
    slot_accuracy[slot][2] = slot_accuracy[slot][3] = 0xFFu;

    // Turn off any report that a prior apply left enabled but the new config no
    // longer wants, so switching presets never leaves stale reports streaming.
    for (uint8_t i = 0; i < prev_report_count[slot]; i++) {
        uint8_t old_id = prev_report_ids[slot][i];
        bool still_wanted = false;
        for (uint8_t j = 0; j < active_cfg[slot].num_reports; j++) {
            if (active_cfg[slot].reports[j].enabled &&
                active_cfg[slot].reports[j].report_id == old_id) {
                still_wanted = true;
                break;
            }
        }
        if (!still_wanted) {
            (void)bno085_enable_report(dev, old_id, 0u);  // interval 0 disables (SH-2 set-feature)
        }
    }

    prev_report_count[slot] = 0;
    for (uint8_t i = 0; i < active_cfg[slot].num_reports; i++) {
        const imu_report_cfg_t *r = &active_cfg[slot].reports[i];
        if (!r->enabled || r->rate_hz == 0) {
            continue; /* rate 0 is rejected by validation; never guess a rate */
        }
        uint32_t granted = 0;
        if (!bno085_enable_report_drain(dev, r->report_id, 1000000u / r->rate_hz, &granted)) {
            printf("imu_%s: report 0x%02X enable FAIL\r\n", slot_name(slot), r->report_id);
        } else if (granted == 0) {
            printf("imu_%s: report 0x%02X no Get Feature Response\r\n", slot_name(slot), r->report_id);
        }
        granted_us[slot][i] = granted;
        if (prev_report_count[slot] < IMU_MAX_REPORTS) {
            prev_report_ids[slot][prev_report_count[slot]++] = r->report_id;
        }
    }
    return true;
}

/** @brief Internal helper: pack_quat (SH-2 i,j,k,real Q14 -> w,x,y,z float). */
static void pack_quat(uint8_t *out, const int16_t v[4]) {
    float scale = 1.0f / 16384.0f;
    float w = v[3] * scale;
    float x = v[0] * scale;
    float y = v[1] * scale;
    float z = v[2] * scale;
    memcpy(out, &w, 4);
    memcpy(out + 4, &x, 4);
    memcpy(out + 8, &y, 4);
    memcpy(out + 12, &z, 4);
}

// Emit one tagged RAW motion-vector record ([sensor_id][x,y,z int16][accuracy]).
/** @brief Internal helper: push_raw. */
static void push_raw(imu_slot_t slot, uint64_t t_us, uint8_t sensor_id, const int16_t *xyz,
                     uint8_t accuracy) {
    jtsmm_record_t rec = {0};
    rec.type = (uint8_t)(slot_usb_base(slot) + 1u);
    rec.t_master_us = t_us;
    rec.len = IMU_RAW_PAYLOAD_LEN;
    rec.payload[0] = sensor_id;
    memcpy(rec.payload + 1, xyz, 6);
    rec.payload[7] = accuracy;
    ringbuf_push(slot_ring(slot), &rec);
}

/** @brief Per-packet context handed to emit_report(). */
typedef struct {
    imu_slot_t slot;
    uint64_t t_int; /* INT assertion (edge) or poll time of the read, see imu_timing.h */
    bool sh2;       /* apply the SH-2 per-report delay */
    uint32_t gaps;  /* SH-2 sequence gaps found in this packet */
} imu_emit_ctx_t;

/**
 * @brief Internal helper: one ring record per SH-2 report instance (K3).
 *
 * @details Called from the parse loop, so a batched packet with several
 *          reports of one type yields one record each instead of keeping only
 *          the last. Each record carries its own sample time: t_int corrected
 *          by the packet's SH-2 base delta and the report's delay field.
 */
static void emit_report(void *vctx, const bno085_report_t *r) {
    imu_emit_ctx_t *e = (imu_emit_ctx_t *)vctx;
    imu_slot_t slot = e->slot;
    if (r->has_seq) {
        e->gaps += bno085_seq_update(&seq_trk[slot], r->report_id, r->seq);
    }
    const uint64_t t_us = e->sh2 ? imu_ts_report_time(e->t_int, r, &ts_stats[slot]) : e->t_int;
    switch (r->report_id) {
    case BNO085_REPORT_ROTATION_VECTOR:
    case BNO085_REPORT_GAME_ROT_VEC:
    case BNO085_REPORT_GEOMAG_ROT_VEC:
    case BNO085_REPORT_GYRO_INT_ROT_VEC: {
        slot_accuracy[slot][0] = r->accuracy;
        jtsmm_record_t rec = {0};
        rec.type = slot_usb_base(slot);
        rec.t_master_us = t_us;
        rec.payload[0] = r->report_id;
        pack_quat(rec.payload + 1, r->v);
        if (r->accuracy == 0xFFu) {
            /* Gyro-integrated RV carries no status byte: send the 17 B form the
             * host already accepts rather than a bogus accuracy. */
            rec.len = IMU_QUAT_PAYLOAD_LEN - 1u;
        } else {
            rec.len = IMU_QUAT_PAYLOAD_LEN;
            rec.payload[17] = r->accuracy;
        }
        ringbuf_push(slot_ring(slot), &rec);
        break;
    }
    case BNO085_REPORT_ACCELEROMETER:
        slot_accuracy[slot][1] = r->accuracy;
        push_raw(slot, t_us, IMU_SENSOR_ACCEL, r->v, r->accuracy);
        break;
    case BNO085_REPORT_GYROSCOPE:
        slot_accuracy[slot][2] = r->accuracy;
        push_raw(slot, t_us, IMU_SENSOR_GYRO, r->v, r->accuracy);
        break;
    case BNO085_REPORT_MAGNETOMETER:
        slot_accuracy[slot][3] = r->accuracy;
        push_raw(slot, t_us, IMU_SENSOR_MAG, r->v, r->accuracy);
        break;
    case BNO085_REPORT_LINEAR_ACCEL: {
        // Linear acceleration is reported as Q8 int16 in m/s^2 (SH-2 6.5.10).
        float lin[3] = {(float)r->v[0] / 256.0f, (float)r->v[1] / 256.0f, (float)r->v[2] / 256.0f};
        jtsmm_record_t rec = {0};
        rec.type = (uint8_t)(slot_usb_base(slot) + 2u);
        rec.t_master_us = t_us;
        rec.len = IMU_LIN_ACCEL_PAYLOAD_LEN;
        memcpy(rec.payload, lin, 12);
        rec.payload[12] = r->accuracy;
        ringbuf_push(slot_ring(slot), &rec);
        break;
    }
    case BNO085_REPORT_GET_FEATURE_RESP:
        for (uint8_t i = 0; i < active_cfg[slot].num_reports && i < IMU_MAX_REPORTS; i++) {
            if (active_cfg[slot].reports[i].report_id == r->feature_id) {
                granted_us[slot][i] = r->interval_us;
            }
        }
        break;
    default:
        break; /* other control responses and reports without a wire record */
    }
}

void imu_stream_on_complete(imu_slot_t slot, uint64_t t_us, const uint8_t *pkt, uint16_t len) {
    if ((unsigned)slot >= 2 || pkt == NULL) {
        return;
    }
    __dmb();
    if (!streaming_slot[slot]) {
        return;
    }
    /* emg_drdy_divisor reads are stamped with the DRDY-paced kick time, not an
     * INT assertion, so the SH-2 delays (relative to INT) do not apply. */
    imu_emit_ctx_t ctx = {
        .slot = slot,
        .t_int = t_us,
        .sh2 = IMU_TS_USE_SH2_DELAY && active_cfg[slot].sync_mode != IMU_SYNC_EMG_DRDY_DIVISOR,
        .gaps = 0,
    };
    bno085_packet_info_t info;
    if (!bno085_parse_packet(pkt, len, emit_report, &ctx, &info)) {
        if (info.continuation) {
            cont_packets[slot]++;
        }
        return;
    }
    imu_ts_stats_on_packet(&ts_stats[slot], &info);
    if (info.truncated) {
        trunc_packets[slot]++;
    }
    if (info.channel == BNO085_CH_GYRO_RV) {
        /* No per-report sequence on channel 5: count lost packets instead. */
        ctx.gaps += bno085_seq_update(&seq_trk[slot], BNO085_REPORT_GYRO_INT_ROT_VEC, info.shtp_seq);
    }
    if (ctx.gaps) {
        seq_gaps[slot] += ctx.gaps;
    }
    if (info.reset_complete || info.advertisement) {
        /* The device rebooted (brown-out, ESD, cable glitch): every report is
         * now off and the stream would stay silent until the 3 s watchdog.
         * Count it and have the comms core re-enable the reports. */
        bno085_seq_reset(&seq_trk[slot]);
        if ((uint32_t)(time_us_32() - stream_start_us32[slot]) >= IMU_RESET_GRACE_US) {
            if (!reset_pending[slot]) {
                reset_count[slot]++;
            }
            reset_pending[slot] = true;
        }
    }
    if (info.reports > 0) {
        panel_imu_mark_active(slot);
    }
}

uint32_t imu_config_seq_gaps(uint8_t slot) {
    return slot < 2 ? seq_gaps[slot] : 0;
}

uint32_t imu_config_reset_count(uint8_t slot) {
    return slot < 2 ? reset_count[slot] : 0;
}

uint32_t imu_config_granted_interval_us(uint8_t slot, uint8_t report_id) {
    if (slot >= 2) {
        return 0;
    }
    for (uint8_t i = 0; i < active_cfg[slot].num_reports && i < IMU_MAX_REPORTS; i++) {
        if (active_cfg[slot].reports[i].enabled && active_cfg[slot].reports[i].report_id == report_id) {
            return granted_us[slot][i];
        }
    }
    return 0;
}

void imu_drdy_divisor_kick(uint32_t drdy_count) {
    /*
     * Paced variant of imu_poll_slot(): kick one async SHTP read every
     * emg_drdy_divisor DRDY edges. This runs inside the EMG DRDY service on
     * the sensor core, so it must be as cheap as the INT-driven path -- the
     * previous version did a blocking 5 ms I2C read here, stalling DRDY, and
     * did so without checking for an in-flight DMA read on the same bus.
     */
    for (int s = 0; s < 2; s++) {
        const imu_slot_config_t *cfg = &active_cfg[s];
        if (cfg->sync_mode != IMU_SYNC_EMG_DRDY_DIVISOR || cfg->emg_drdy_divisor == 0) {
            continue;
        }
        if ((drdy_count % cfg->emg_drdy_divisor) != 0) {
            continue;
        }
        __dmb();
        if (!streaming_slot[s] || !cfg->enabled) {
            continue;
        }
        bno085_t *dev = imu_bus_dev((imu_slot_t)s);
        if (dev == NULL || !bno085_int_asserted(dev)) {
            continue;
        }
        if (i2c_dma_busy(slot_dma((imu_slot_t)s))) {
            continue;
        }
        i2c_dma_kick_shtp_read(slot_dma((imu_slot_t)s), dev->i2c, dev->addr, t_master_us());
    }
}

/** @brief Internal helper: shortest period among @p slot's enabled reports (µs, 0 = none). */
static uint32_t fastest_period_us(imu_slot_t slot) {
    uint32_t best = 0;
    for (uint8_t i = 0; i < active_cfg[slot].num_reports && i < IMU_MAX_REPORTS; i++) {
        const imu_report_cfg_t *r = &active_cfg[slot].reports[i];
        if (!r->enabled || r->rate_hz == 0) {
            continue;
        }
        uint32_t p = granted_us[slot][i] ? granted_us[slot][i] : 1000000u / r->rate_hz;
        if (best == 0 || p < best) {
            best = p;
        }
    }
    return best;
}

/**
 * @brief Internal helper (sensor core): notice stream start / restart / stop.
 *
 * @details Per-stream timestamp state is reset here, on the core that writes
 *          it, instead of from the comms core while a completion may run.
 */
static void imu_ts_track_stream(imu_slot_t slot, bool active) {
    /* INT-edge stamping for INT-driven reads only; emg_drdy_divisor keeps the
     * DRDY-paced kick time (the edge IRQ stays off). */
    bool want_edge = IMU_TS_USE_INT_EDGE && active &&
                     active_cfg[slot].sync_mode != IMU_SYNC_EMG_DRDY_DIVISOR;
    uint32_t gen = stream_gen[slot];
    if (active && (!c1_active[slot] || c1_gen[slot] != gen)) {
        c1_active[slot] = true;
        c1_gen[slot] = gen;
        imu_ts_stats_reset(&ts_stats[slot]);
        edge_max_age_us[slot] = imu_ts_edge_max_age_us(fastest_period_us(slot));
        if (want_edge) {
            imu_int_edge_arm(slot, true); /* (re)arm: acks a stale edge, clears the latch */
        }
    } else if (!active && c1_active[slot]) {
        c1_active[slot] = false;
    }
    if (want_edge != imu_int_edge_armed(slot)) {
        imu_int_edge_arm(slot, want_edge);
    }
}

/** @brief Internal helper: imu_poll_slot. */
static void imu_poll_slot(imu_slot_t slot) {
    __dmb();
    bool active = streaming_slot[slot] && active_cfg[slot].enabled;
    imu_ts_track_stream(slot, active);
    if (!active) {
        return;
    }
    if (active_cfg[slot].sync_mode == IMU_SYNC_EMG_DRDY_DIVISOR) {
        return; /* paced from the DRDY service instead (imu_drdy_divisor_kick) */
    }
    bno085_t *dev = imu_bus_dev(slot);
    if (dev == NULL) {
        return;
    }
    /*
     * Non-blocking: while INT is asserted and no DMA is in flight for this slot,
     * kick an async SHTP read. Completion (i2c_dma_service on this same sensor
     * core) fires imu_dma_complete -> imu_stream_on_complete -> ring push. This
     * lets EMG SPI and IMU I2C overlap instead of the old blocking read that
     * stalled the whole sensor loop. The CRA PCA9546 channel is selected once at
     * stream start and latches, so no per-read mux write is needed here.
     */
    if (i2c_dma_busy(slot_dma(slot))) {
        return;
    }
    uint64_t t_latch = 0;
    if (imu_int_edge_armed(slot)) {
        /*
         * INT-edge timestamp (imu_timing.h). With this core's IRQs masked, so
         * the edge IRQ cannot slip in between: drop an edge whose INT is high
         * again, read the level, and take the edge (clearing it) before the
         * read is kicked -- an INT re-assertion during the read is then latched
         * for the next packet. No usable edge: poll time, counted.
         */
        imu_edge_latch_t *l = imu_int_edge_latch(slot);
        uint32_t irq = save_and_disable_interrupts();
        bool asserted = bno085_int_asserted(dev);
        bool dropped = imu_edge_latch_discard_if_released(l, asserted);
        imu_ts_src_t src = IMU_TS_SRC_NO_EDGE;
        uint64_t now = 0;
        if (asserted) {
            now = t_master_us();
            src = imu_edge_latch_take(l, now, edge_max_age_us[slot], &t_latch);
        }
        restore_interrupts(irq);
        if (dropped) {
            ts_stats[slot].spurious++;
        }
        if (!asserted) {
            return;
        }
        imu_ts_stats_on_read(&ts_stats[slot], src, now, t_latch);
    } else {
        if (!bno085_int_asserted(dev)) {
            return;
        }
        t_latch = t_master_us();
    }
    i2c_dma_kick_shtp_read(slot_dma(slot), dev->i2c, dev->addr, t_latch);
}

void imu_stream_poll_service(void) {
    for (int s = 0; s < 2; s++) {
        imu_poll_slot((imu_slot_t)s);
    }
}

/*
 * Cooperative IMU recovery state machine (runs on the comms core).
 *
 * The old imu_slot_recover() blocked the comms core for up to several seconds
 * (soft-reset settle + SHTP re-sync + product-id wait), during which
 * host_link_drain() did not run, so the EMG ring (~128 ms at 2 kHz) overflowed
 * into emg_drops and USB went silent for the whole reset. This machine performs
 * at most ONE bounded I2C transaction (or a timed wait) per imu_recover_poll()
 * call, so the comms loop keeps draining rings to USB between steps. Writes are
 * single-shot here (dev->write_tries = 1); the machine does the retrying, so
 * one step never blocks for the ~200 ms of the default 20-try write.
 *
 * Escalation per attempt: I2C bus clear -> SHTP soft reset -> as a last resort
 * a pulse on the NRST line shared by both IMUs (rate limited; the other slot is
 * re-synced and re-enabled too). A spontaneous device reset seen by the stream
 * enters at SYNC. Whether streaming resumes at FINISH follows stream_wanted[]
 * (host intent), not a snapshot, so an `imu_stream off` during recovery sticks.
 *
 * streaming_slot[] stays false for the entire sequence, so the sensor core
 * leaves this slot's I2C bus alone (single-owner contract preserved). The two
 * slots recover independently on separate peripherals (MTA=i2c0, CRA=i2c1).
 */
typedef enum {
    IMU_REC_IDLE = 0,
    IMU_REC_RESET_WRITE,
    IMU_REC_NRST_LOW,
    IMU_REC_RESET_SETTLE,
    IMU_REC_SYNC,
    IMU_REC_REQ_PID,
    IMU_REC_WAIT_PID,
    IMU_REC_ENABLE,
    IMU_REC_FINISH,
} imu_rec_state_t;

typedef struct {
    imu_rec_state_t state;
    bool after_reset; /* entered because the device reset itself (no soft reset needed) */
    bool pid_ok;
    bool nrst_done;      /* NRST already pulsed during this attempt */
    uint8_t reset_attempts;
    uint8_t enable_idx;  /* ENABLE step: 0 = ME calibration, n = report n-1 */
    uint8_t write_fails; /* failed single-shot writes in ENABLE */
    uint32_t packets;
    absolute_time_t deadline;
} imu_rec_ctx_t;

static imu_rec_ctx_t rec_ctx[2];

static bool nrst_used;
static absolute_time_t nrst_last;

#define IMU_REC_SETTLE_MS     300u
#define IMU_REC_SYNC_MS       3000u
#define IMU_REC_PID_MS        2000u
#define IMU_REC_RESET_TRIES   5u
#define IMU_REC_WRITE_FAILS   5u
#define IMU_REC_RETRY_MS      20u
#define IMU_REC_NRST_PULSE_MS 10u
/* NRST resets BOTH IMUs, costing the healthy slot about a second of data, and
 * an unplugged slot would otherwise pulse it on every watchdog retry: allow it
 * at most once per interval. Set IMU_REC_NRST_ENABLE to 0 to never pulse it. */
#define IMU_REC_NRST_ENABLE          1
#define IMU_REC_NRST_MIN_INTERVAL_MS 60000u

bool imu_recover_busy(void) {
    return rec_ctx[0].state != IMU_REC_IDLE || rec_ctx[1].state != IMU_REC_IDLE;
}

/** @brief Internal helper: stop a slot's recovery without touching the bus. */
static void imu_recover_cancel(imu_slot_t slot) {
    if (rec_ctx[slot].state == IMU_REC_NRST_LOW) {
        imu_bus_nrst(true);
    }
    rec_ctx[slot].state = IMU_REC_IDLE;
    bno085_t *dev = imu_bus_dev(slot);
    if (dev != NULL) {
        dev->write_tries = 0;
    }
}

/**
 * @brief Internal helper: arm recovery for @p slot.
 * @param after_reset true when the stream saw the device reset itself: it is
 *        already rebooting, so skip bus clear and soft reset and go straight
 *        to re-sync / product id / re-enable.
 */
static void imu_recover_begin(imu_slot_t slot, bool after_reset) {
    if ((unsigned)slot >= 2 || !active_cfg[slot].enabled) {
        return;
    }
    imu_rec_ctx_t *rc = &rec_ctx[slot];
    bno085_t *dev = imu_bus_dev(slot);
    if (rc->state != IMU_REC_IDLE || dev == NULL) {
        return; /* already recovering this slot */
    }
    /*
     * Take the slot off the streaming path so the sensor core stops kicking DMA
     * on this bus, and have it kill any in-flight transfer before we drive the
     * bus from here. (This used to call i2c_dma_abort() from the comms core
     * *before* clearing streaming, racing the sensor core, which owns the DMA.)
     * The CRA mux channel latches, so selecting it once now is enough for the
     * whole sequence (a BNO085 soft reset does not disturb the PCA9546).
     */
    if (!imu_slot_quiesce(slot)) {
        host_link_send_log(slot == IMU_SLOT_CRA ? "imu recover: CRA DMA did not stop; retry later"
                                                : "imu recover: MTA DMA did not stop; retry later");
        reset_pending[slot] = false;
        streaming_slot[slot] = stream_wanted[slot];
        __dmb();
        return;
    }
    /* Cleared only now that the sensor core is off this slot, so the second
     * indication of the same reset (advertisement + reset-complete) that may
     * still have been parsed in between does not arm another cycle. */
    reset_pending[slot] = false;
    dev->write_tries = 1;
    if (!after_reset && !imu_bus_clear(slot)) {
        host_link_send_log(slot == IMU_SLOT_CRA ? "imu recover: CRA I2C lines stuck low after bus clear"
                                                : "imu recover: MTA I2C lines stuck low after bus clear");
    }
    imu_bus_prepare_slot(slot);
    rc->after_reset = after_reset;
    rc->pid_ok = false;
    rc->nrst_done = false;
    rc->reset_attempts = 0;
    rc->enable_idx = 0;
    rc->write_fails = 0;
    if (after_reset) {
        rc->packets = 1; /* the device already spoke; SYNC may finish once INT idles */
        rc->deadline = make_timeout_time_ms(IMU_REC_SYNC_MS);
        rc->state = IMU_REC_SYNC;
    } else {
        rc->packets = 0;
        rc->deadline = get_absolute_time();
        rc->state = IMU_REC_RESET_WRITE;
    }
}

void imu_slot_recover_start(imu_slot_t slot) {
    imu_recover_begin(slot, false);
}

/** @brief Internal helper: may the shared NRST line be pulsed now? */
static bool imu_nrst_allowed(void) {
#if IMU_REC_NRST_ENABLE
    return !nrst_used ||
           absolute_time_diff_us(nrst_last, get_absolute_time()) >= (int64_t)IMU_REC_NRST_MIN_INTERVAL_MS * 1000;
#else
    return false;
#endif
}

/**
 * @brief Internal helper: last resort -- pulse the shared NRST for @p slot.
 *
 * @details The line resets both IMUs, so the other slot (if configured) is
 *          taken off the stream and sent through settle/sync/re-enable too.
 *          If it cannot be quiesced, its own stream sees the reset indication
 *          and re-enables itself.
 */
static void imu_recover_pulse_nrst(imu_slot_t slot) {
    imu_slot_t other = (slot == IMU_SLOT_CRA) ? IMU_SLOT_MTA : IMU_SLOT_CRA;
    imu_rec_ctx_t *ro = &rec_ctx[other];
    bno085_t *odev = imu_bus_dev(other);
    bool take_other = active_cfg[other].enabled && odev != NULL;
    if (take_other && ro->state == IMU_REC_IDLE && !imu_slot_quiesce(other)) {
        /* imu_slot_quiesce() already took the slot off the stream; put it back
         * (as imu_recover_begin does) so it re-syncs on the reset indication
         * instead of staying silent until its 3 s watchdog. */
        streaming_slot[other] = stream_wanted[other];
        __dmb();
        take_other = false;
    }
    if (take_other) {
        reset_pending[other] = false;
        odev->write_tries = 1;
        memset(odev->seq, 0, sizeof(odev->seq));
        ro->after_reset = true;
        ro->pid_ok = false;
        ro->nrst_done = true;
        ro->enable_idx = 0;
        ro->write_fails = 0;
        ro->packets = 0;
        ro->deadline = make_timeout_time_ms(IMU_REC_NRST_PULSE_MS + IMU_REC_SETTLE_MS);
        ro->state = IMU_REC_RESET_SETTLE;
    }
    host_link_send_log(slot == IMU_SLOT_CRA ? "imu recover: CRA not responding, pulsing shared NRST (resets both IMUs)"
                                            : "imu recover: MTA not responding, pulsing shared NRST (resets both IMUs)");
    nrst_used = true;
    nrst_last = get_absolute_time();
    imu_bus_nrst(false);
    imu_rec_ctx_t *rc = &rec_ctx[slot];
    rc->nrst_done = true;
    rc->deadline = make_timeout_time_ms(IMU_REC_NRST_PULSE_MS);
    rc->state = IMU_REC_NRST_LOW;
}

/** @brief Internal helper: give up this attempt, or escalate to NRST once. */
static void imu_recover_fail_or_nrst(imu_slot_t slot) {
    imu_rec_ctx_t *rc = &rec_ctx[slot];
    if (!rc->nrst_done && imu_nrst_allowed()) {
        imu_recover_pulse_nrst(slot);
    } else {
        rc->state = IMU_REC_FINISH; /* the comms watchdog retries later */
    }
}

/** @brief Internal helper: one single-shot write of the ENABLE sequence. */
static void imu_recover_enable_step(imu_slot_t slot, bno085_t *dev) {
    imu_rec_ctx_t *rc = &rec_ctx[slot];
    if (!time_reached(rc->deadline)) {
        return; /* retry backoff */
    }
    bool ok;
    if (rc->enable_idx == 0) {
        /* After a device reset the motion-engine calibration config that apply
         * set is not guaranteed to survive, so re-issue it with the reports. */
        imu_bus_prepare_slot(slot);
        const imu_slot_config_t *c = &active_cfg[slot];
        ok = bno085_set_calibration(dev, c->dynamic_calibration.accel, c->dynamic_calibration.gyro,
                                    c->dynamic_calibration.mag);
    } else {
        uint8_t i = (uint8_t)(rc->enable_idx - 1u);
        if (i >= active_cfg[slot].num_reports) {
            rc->state = IMU_REC_FINISH;
            return;
        }
        const imu_report_cfg_t *r = &active_cfg[slot].reports[i];
        if (!r->enabled || r->rate_hz == 0) {
            rc->enable_idx++;
            return;
        }
        ok = bno085_enable_report(dev, r->report_id, 1000000u / r->rate_hz);
    }
    if (ok) {
        rc->enable_idx++;
    } else if (++rc->write_fails >= IMU_REC_WRITE_FAILS) {
        rc->state = IMU_REC_FINISH;
    } else {
        rc->deadline = make_timeout_time_ms(IMU_REC_RETRY_MS);
    }
}

/** @brief Internal helper: imu_recover_step. */
static void imu_recover_step(imu_slot_t slot) {
    imu_rec_ctx_t *rc = &rec_ctx[slot];
    if (rc->state == IMU_REC_IDLE) {
        return;
    }
    bno085_t *dev = imu_bus_dev(slot);
    if (dev == NULL) {
        rc->state = IMU_REC_IDLE;
        return;
    }

    switch (rc->state) {
    case IMU_REC_RESET_WRITE: {
        if (!time_reached(rc->deadline)) {
            return; /* inter-attempt backoff */
        }
        uint8_t payload = 1u; /* SHTP executable: reset request */
        memset(dev->seq, 0, sizeof(dev->seq));
        if (bno085_write_packet(dev, BNO085_CH_EXECUTABLE, &payload, 1)) {
            rc->deadline = make_timeout_time_ms(IMU_REC_SETTLE_MS);
            rc->state = IMU_REC_RESET_SETTLE;
        } else if (++rc->reset_attempts >= IMU_REC_RESET_TRIES) {
            imu_recover_fail_or_nrst(slot); /* device does not even ACK */
        } else {
            rc->deadline = make_timeout_time_ms(30);
        }
        return;
    }
    case IMU_REC_NRST_LOW:
        if (time_reached(rc->deadline)) {
            imu_bus_nrst(true);
            memset(dev->seq, 0, sizeof(dev->seq));
            rc->deadline = make_timeout_time_ms(IMU_REC_SETTLE_MS);
            rc->state = IMU_REC_RESET_SETTLE;
        }
        return;
    case IMU_REC_RESET_SETTLE:
        if (time_reached(rc->deadline)) {
            imu_bus_prepare_slot(slot); /* re-select the CRA mux channel in case NRST reached it */
            rc->packets = 0;
            rc->deadline = make_timeout_time_ms(IMU_REC_SYNC_MS);
            rc->state = IMU_REC_SYNC;
        }
        return;
    case IMU_REC_SYNC: {
        uint8_t buf[BNO085_SHTP_MAX_PACKET];
        size_t len = 0;
        if (bno085_read_packet(dev, buf, sizeof(buf), &len) && len >= 4) {
            rc->packets++;
            if (buf[2] == BNO085_CH_WAKE) {
                rc->state = IMU_REC_REQ_PID;
                return;
            }
        } else if (rc->packets > 0 && !bno085_int_asserted(dev)) {
            /* Spoke, and has nothing more queued (a read with no data may NACK
             * or return an empty header). */
            rc->state = IMU_REC_REQ_PID;
            return;
        }
        if (time_reached(rc->deadline)) {
            /* Proceed if the device spoke at all; otherwise escalate / abandon. */
            if (rc->packets > 0) {
                rc->state = IMU_REC_REQ_PID;
            } else {
                imu_recover_fail_or_nrst(slot);
            }
        }
        return;
    }
    case IMU_REC_REQ_PID:
        (void)bno085_request_product_id(dev);
        rc->deadline = make_timeout_time_ms(IMU_REC_PID_MS);
        rc->state = IMU_REC_WAIT_PID;
        return;
    case IMU_REC_WAIT_PID: {
        uint8_t buf[BNO085_SHTP_MAX_PACKET];
        size_t len = 0;
        if (bno085_read_packet(dev, buf, sizeof(buf), &len) && len > 0) {
            bno085_packet_info_t info;
            (void)bno085_parse_packet(buf, (uint16_t)len, NULL, NULL, &info);
            if (info.product_id) {
                rc->pid_ok = true;
                rc->deadline = get_absolute_time();
                rc->state = IMU_REC_ENABLE;
                return;
            }
        }
        if (time_reached(rc->deadline)) {
            /* Mirror the old boot path: re-enable reports even without a clean PID
             * ack -- it frequently revives the stream, and the log records pid=0. */
            rc->deadline = get_absolute_time();
            rc->state = IMU_REC_ENABLE;
        }
        return;
    }
    case IMU_REC_ENABLE:
        imu_recover_enable_step(slot, dev);
        return;
    case IMU_REC_FINISH: {
        dev->write_tries = 0;
        if (rc->pid_ok) {
            slot_state[slot] = IMU_SLOT_STATE_OK;
        }
        if (stream_wanted[slot]) {
            bno085_seq_reset(&seq_trk[slot]);
            stream_start_us32[slot] = time_us_32();
            stream_gen[slot]++;
            imu_int_edge_clear(slot); /* no edge from the recovery traffic */
            streaming_slot[slot] = true;
            __dmb();
        }
        char msg[64];
        snprintf(msg, sizeof(msg), "imu recover: %s pid=%d%s%s", (slot == IMU_SLOT_CRA) ? "CRA" : "MTA",
                 rc->pid_ok ? 1 : 0, rc->after_reset ? " (device reset)" : "",
                 (rc->enable_idx == 0 || rc->write_fails >= IMU_REC_WRITE_FAILS) ? " FAIL" : "");
        host_link_send_log(msg);
        rc->state = IMU_REC_IDLE;
        return;
    }
    default:
        rc->state = IMU_REC_IDLE;
        return;
    }
}

void imu_recover_poll(void) {
    /* Config apply re-initialises the buses and devices from this same loop:
     * never interleave recovery steps with it, and drop any recovery that was
     * running when it began (the apply replaces its state). */
    static bool was_applying;
    if (config_apply_busy()) {
        if (!was_applying) {
            imu_recover_cancel(IMU_SLOT_MTA);
            imu_recover_cancel(IMU_SLOT_CRA);
        }
        was_applying = true;
        return;
    }
    was_applying = false;
    for (int s = 0; s < 2; s++) {
        __dmb();
        if (reset_pending[s] && rec_ctx[s].state == IMU_REC_IDLE) {
            /* The sensor core saw the device reset itself mid-stream. */
            if (streaming_slot[s]) {
                imu_recover_begin((imu_slot_t)s, true);
            } else {
                reset_pending[s] = false;
            }
        }
        imu_recover_step((imu_slot_t)s);
    }
}

bool imu_stream_start_slot(imu_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return false;
    }
    if (!active_cfg[slot].enabled) {
        /* This used to probe the bus and force-enable the slot, streaming
         * whatever config an earlier apply had left behind. */
        stream_wanted[slot] = false;
        printf("imu_%s: stream REFUSED: slot disabled in config (enable it and re-apply)\r\n",
               slot_name(slot));
        return false;
    }
    stream_wanted[slot] = true;
    if (rec_ctx[slot].state != IMU_REC_IDLE) {
        return true; /* recovery owns the bus; its FINISH starts the stream */
    }
    /* A repeated `imu_stream on` used to re-select the mux with a blocking
     * write while the sensor core's DMA was reading the same bus. */
    if (!imu_slot_quiesce(slot)) {
        printf("imu_%s: stream start FAIL (DMA did not stop)\r\n", slot_name(slot));
        return false;
    }
    /*
     * Select the CRA mux channel once now; it latches, so subsequent DMA reads
     * on i2c1 stay routed to CRA without per-read mux writes (MTA on i2c0 is
     * unaffected). Must run before streaming starts kicking DMA reads.
     */
    imu_bus_prepare_slot(slot);
    bno085_seq_reset(&seq_trk[slot]);
    stream_start_us32[slot] = time_us_32();
    stream_gen[slot]++;
    imu_int_edge_clear(slot); /* no pre-stream edge may stamp the first packet */
    streaming_slot[slot] = true;
    __dmb();
    return true;
}

void imu_stream_stop_slot(imu_slot_t slot) {
    if ((unsigned)slot < 2) {
        stream_wanted[slot] = false;
        streaming_slot[slot] = false;
        __dmb();
        imu_int_edge_clear(slot);
        /* Let the sensor core drop an in-flight read; any later blocking
         * access on this bus waits for that via imu_slot_quiesce(). */
        i2c_dma_request_abort(slot_dma(slot));
    }
}

/**
 * @brief Internal helper: imu_status lines for @p slot's timestamp source (imu_timing.h).
 *
 * @details Counters are written by the sensor core; a torn read only skews
 *          one line of diagnostics. Windows are the last IMU_TS_STATS_WINDOW
 *          reads / packets (the running window until the first one fills).
 */
static void imu_config_print_ts(imu_slot_t s) {
    const imu_ts_stats_t *ts = &ts_stats[s];
    const imu_edge_latch_t *l = imu_int_edge_latch(s);
    const char *src;
    if (imu_int_edge_armed(s)) {
        src = "int_edge";
    } else if (active_cfg[s].sync_mode == IMU_SYNC_EMG_DRDY_DIVISOR) {
        src = "drdy_kick";
    } else if (!IMU_TS_USE_INT_EDGE) {
        src = "poll(IMU_TS_USE_INT_EDGE=0)";
    } else {
        src = "not_armed";
    }
    printf("  ts: t_int=%s core1_irq=%d sh2_delay=%d int_edges=%lu coalesced=%lu\r\n", src,
           imu_int_edge_core_ready() ? 1 : 0, IMU_TS_USE_SH2_DELAY ? 1 : 0, (unsigned long)l->edges,
           (unsigned long)l->coalesced);
    printf("  ts: reads edge=%lu poll_fallback no_edge=%lu stale=%lu future=%lu dropped released=%lu hdr=%lu\r\n",
           (unsigned long)ts->src[IMU_TS_SRC_EDGE], (unsigned long)ts->src[IMU_TS_SRC_NO_EDGE],
           (unsigned long)ts->src[IMU_TS_SRC_STALE], (unsigned long)ts->src[IMU_TS_SRC_FUTURE],
           (unsigned long)ts->spurious, (unsigned long)ts->hdr_drops);
    uint32_t lat_mean = ts->lat_w_valid ? ts->lat_mean_w : (ts->lat_n ? ts->lat_sum / ts->lat_n : 0);
    uint32_t lat_max = ts->lat_w_valid ? ts->lat_max_w : ts->lat_max;
    printf("  ts: edge->kick us mean=%lu max=%lu (%s) max_all=%lu stale_bound=%luus\r\n",
           (unsigned long)lat_mean, (unsigned long)lat_max, ts->lat_w_valid ? "last 256 reads" : "partial",
           (unsigned long)ts->lat_max_all, (unsigned long)edge_max_age_us[s]);
    int32_t b_mean = ts->base_w_valid ? ts->base_mean_w
                                      : (ts->base_n ? (int32_t)(ts->base_sum / (int64_t)ts->base_n) : 0);
    int32_t b_min = ts->base_w_valid ? ts->base_min_w : ts->base_min;
    int32_t b_max = ts->base_w_valid ? ts->base_max_w : ts->base_max;
    printf("  ts: base_delta us last=%ld mean=%ld min=%ld max=%ld (%s) no_base_pkts=%lu clamp late=%lu early=%lu\r\n",
           (long)ts->base_last, (long)b_mean, (long)b_min, (long)b_max,
           ts->base_w_valid ? "last 256 pkts" : "partial", (unsigned long)ts->no_base_pkts,
           (unsigned long)ts->clamp_late, (unsigned long)ts->clamp_early);
}

void imu_config_status(void) {
    const char *labels[2] = {"mta", "cra"};
    for (int s = 0; s < 2; s++) {
        bno085_t *dev = imu_bus_dev((imu_slot_t)s);
        printf("imu_%s: enabled=%d state=%s streaming=%d wanted=%d recovering=%d int=%d reports=%u\r\n",
               labels[s], active_cfg[s].enabled ? 1 : 0, slot_state_names[slot_state[s]],
               streaming_slot[s] ? 1 : 0, stream_wanted[s] ? 1 : 0,
               rec_ctx[s].state != IMU_REC_IDLE ? 1 : 0, (dev && bno085_int_asserted(dev)) ? 1 : 0,
               active_cfg[s].num_reports);
        printf("  seq_gaps=%lu resets=%lu cont_pkts=%lu trunc_pkts=%lu i2c_timeouts=%lu\r\n",
               (unsigned long)seq_gaps[s], (unsigned long)reset_count[s],
               (unsigned long)cont_packets[s], (unsigned long)trunc_packets[s],
               (unsigned long)i2c_dma_timeout_count((i2c_dma_slot_t)s));
        printf("  i2c_tx_aborts=%lu (NACK/arb-lost seen by the DMA reader; each costs a packet)\r\n",
               (unsigned long)i2c_dma_tx_abort_count((i2c_dma_slot_t)s));
        printf("  i2c_hz=%lu (cfg=%lu) read_us last=%lu max=%lu len=%u\r\n",
               (unsigned long)imu_bus_achieved_hz((imu_slot_t)s),
               (unsigned long)jtsmm_config_active()->platform.i2c_hz,
               (unsigned long)i2c_dma_last_read_us((i2c_dma_slot_t)s),
               (unsigned long)i2c_dma_max_read_us((i2c_dma_slot_t)s),
               (unsigned)i2c_dma_last_len((i2c_dma_slot_t)s));
        printf("  cal quat=%s accel=%s gyro=%s mag=%s (0=unreliable..3=high)\r\n",
               accuracy_label(slot_accuracy[s][0]), accuracy_label(slot_accuracy[s][1]),
               accuracy_label(slot_accuracy[s][2]), accuracy_label(slot_accuracy[s][3]));
        imu_config_print_ts((imu_slot_t)s);
        for (uint8_t i = 0; i < active_cfg[s].num_reports && i < IMU_MAX_REPORTS; i++) {
            const imu_report_cfg_t *r = &active_cfg[s].reports[i];
            if (!r->enabled) {
                continue;
            }
            uint32_t g = granted_us[s][i];
            if (g == 0) {
                printf("  report 0x%02X req=%uHz granted=none\r\n", r->report_id, (unsigned)r->rate_hz);
            } else {
                uint32_t dhz = (uint32_t)((10000000ull + g / 2u) / g); /* 0.1 Hz units */
                printf("  report 0x%02X req=%uHz granted=%luus (%lu.%luHz)\r\n", r->report_id,
                       (unsigned)r->rate_hz, (unsigned long)g, (unsigned long)(dhz / 10u),
                       (unsigned long)(dhz % 10u));
            }
        }
        if (!active_cfg[s].enabled || dev == NULL) {
            continue;
        }
        /* A live read would collide with the sensor core's DMA on this bus. */
        __dmb();
        if (streaming_slot[s] || rec_ctx[s].state != IMU_REC_IDLE) {
            printf("  read skipped (slot busy: streaming or recovering)\r\n");
            continue;
        }
        if (!i2c_dma_quiesce((i2c_dma_slot_t)s, IMU_QUIESCE_TIMEOUT_US)) {
            printf("  read skipped (DMA did not stop)\r\n");
            continue;
        }
        imu_bus_prepare_slot((imu_slot_t)s);
        uint8_t pkt[BNO085_SHTP_MAX_PACKET];
        size_t len = 0;
        bool ok = bno085_read_packet(dev, pkt, sizeof(pkt), &len);
        bno085_parse_state_t parsed;
        memset(&parsed, 0, sizeof(parsed));
        bool parsed_ok = ok && len > 0 && bno085_parse_shtp(pkt, (uint16_t)len, &parsed);
        printf("  read ok=%d len=%u parse=%d rot=%d\r\n", ok ? 1 : 0, (unsigned)len,
               parsed_ok ? 1 : 0, parsed.got_rotation ? 1 : 0);
    }
}

void imu_config_save_calibration(void) {
    const char *labels[2] = {"mta", "cra"};
    for (int s = 0; s < 2; s++) {
        bno085_t *dev = imu_bus_dev((imu_slot_t)s);
        if (!active_cfg[s].enabled || dev == NULL) {
            continue;
        }
        // The sensor core owns the bus while streaming; issuing a control-channel
        // write from here would collide with its DMA reads. Require an idle slot.
        __dmb();
        if (streaming_slot[s] || rec_ctx[s].state != IMU_REC_IDLE) {
            printf("imu_%s: savecal skipped (stop streaming first)\r\n", labels[s]);
            continue;
        }
        // Right after stream teardown the sensor core may still be finishing a
        // DMA read on this bus (the old fixed 2 ms settle did not guarantee
        // otherwise): wait for its acknowledge, then write with one retry.
        if (!i2c_dma_quiesce((i2c_dma_slot_t)s, IMU_QUIESCE_TIMEOUT_US)) {
            printf("imu_%s: save DCD FAIL (DMA did not stop)\r\n", labels[s]);
            continue;
        }
        imu_bus_prepare_slot((imu_slot_t)s);
        bool ok = bno085_save_dcd(dev);
        if (!ok) {
            imu_bus_prepare_slot((imu_slot_t)s);
            sleep_ms(5);
            ok = bno085_save_dcd(dev);
        }
        printf("imu_%s: save DCD %s\r\n", labels[s], ok ? "sent" : "FAIL");
    }
}
