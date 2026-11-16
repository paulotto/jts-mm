// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file comms_core.c
 * @brief Comms core loop on physical core 0 (USB consumer, commands).
 */

#include "jtsmm/app/comms_core.h"
#include "jtsmm/app/imu_watchdog.h"

#include "jtsmm/app/fault.h"
#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/domain/emg_config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/app/sensor_core.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/platform/spi_dma.h"
#include "jtsmm/config.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/cmd_parser.h"
#include "jtsmm/runtime/config_apply.h"
#include "jtsmm/runtime/config_flash.h"
#include "jtsmm/runtime/config_json.h"
#include "jtsmm/runtime/host_link.h"
#include "jtsmm/runtime/ringbuf.h"
#include "jtsmm/runtime/timebase.h"
#include "jtsmm/version.h"
#include "jtsmm/runtime/usb_service.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <string.h>

static volatile uint32_t comms_poll_ticks;
static volatile uint32_t sensor_loop_hz;
static volatile uint32_t comms_loop_hz;

uint32_t comms_poll_ticks_count(void) {
    return comms_poll_ticks;
}

uint32_t core_sensor_loop_hz(void) {
    return sensor_loop_hz;
}

uint32_t core_comms_loop_hz(void) {
    return comms_loop_hz;
}

/** @brief Internal helper: put_u32. */
static void put_u32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFFu);
    p[1] = (uint8_t)((v >> 8) & 0xFFu);
    p[2] = (uint8_t)((v >> 16) & 0xFFu);
    p[3] = (uint8_t)((v >> 24) & 0xFFu);
}

/*
 * Per-slot IMU sequence-gap / reset counters for REC_STATS. The IMU domain
 * (imu_config.c) provides the real ones; these weak fallbacks report 0 so the
 * record layout stays fixed when it does not (yet).
 */
uint32_t imu_config_seq_gaps(uint8_t slot);
uint32_t imu_config_reset_count(uint8_t slot);

__attribute__((weak)) uint32_t imu_config_seq_gaps(uint8_t slot) {
    (void)slot;
    return 0;
}

__attribute__((weak)) uint32_t imu_config_reset_count(uint8_t slot) {
    (void)slot;
    return 0;
}

static uint32_t stats_seq;

/*
 * REC_STATS is a flat array of little-endian u32 counters; the host maps them by
 * index (STATS_FIELDS in software/jtsmm/jtsmm/protocol/records.py and
 * kStatsFields in software/jtsmm/native/src/module.cpp) and decodes by count,
 * so an older firmware's shorter record still parses. Append only, and add the
 * matching names to BOTH host tables in the same change.
 */
void comms_core_send_stats(void) {
    uint8_t payload[144];
    put_u32(payload + 0, ads1299_drdy_edge_count());
    put_u32(payload + 4, ads1299_missed_drdy_count());
    put_u32(payload + 8, ringbuf_pushes(RING_EMG));
    put_u32(payload + 12, ringbuf_drops(RING_EMG));
    put_u32(payload + 16, ringbuf_pushes(RING_IMU_MTA));
    put_u32(payload + 20, ringbuf_drops(RING_IMU_MTA));
    put_u32(payload + 24, ringbuf_pushes(RING_IMU_CRA));
    put_u32(payload + 28, ringbuf_drops(RING_IMU_CRA));
    put_u32(payload + 32, ringbuf_pushes(RING_EVENTS));
    put_u32(payload + 36, ringbuf_drops(RING_EVENTS));
    put_u32(payload + 40, host_link_tx_bytes());
    put_u32(payload + 44, core_sensor_loop_hz());
    put_u32(payload + 48, core_comms_loop_hz());
    put_u32(payload + 52, host_link_tx_drops());
    put_u32(payload + 56, ads1299_max_drdy_gap_us());
    put_u32(payload + 60, sensor_core_imu_svc_max_us());
    put_u32(payload + 64, sensor_core_leadoff_max_us());
    put_u32(payload + 68, sensor_core_leadoff_count());
    put_u32(payload + 72, sensor_core_poll_max_us());
    put_u32(payload + 76, ads1299_frame_misalign_count());
    put_u32(payload + 80, emg_stream_misalign_drop_count());
    put_u32(payload + 84, ads1299_frame_misalign_max_streak());
    put_u32(payload + 88, ads1299_frame_resync_count());
    put_u32(payload + 92, spi_dma_wedge_count());
    put_u32(payload + 96, trigger_pio_pulse_count());
    put_u32(payload + 100, trigger_pio_timer_event_count());
    put_u32(payload + 104, i2c_dma_tx_abort_count_total());
    /* Per-slot too: the total cannot say which bus is faulty, and a bad port
     * (rather than a bad IMU) is exactly what one slot stuck at zero pushes
     * with thousands of aborts looks like. MTA = total - CRA. */
    put_u32(payload + 108, i2c_dma_tx_abort_count(I2C_DMA_SLOT_CRA));
    /* [28..35] Frames lost at the USB boundary per stream; IMU sequence gaps
     * and sensor resets per slot; and a record sequence number (1 = first
     * after boot) so the host can tell a lost STATS record from a quiet one. */
    put_u32(payload + 112, host_link_tx_drops_emg());
    put_u32(payload + 116, host_link_tx_drops_imu());
    put_u32(payload + 120, host_link_tx_drops_other());
    put_u32(payload + 124, imu_config_seq_gaps((uint8_t)IMU_SLOT_MTA));
    put_u32(payload + 128, imu_config_seq_gaps((uint8_t)IMU_SLOT_CRA));
    put_u32(payload + 132, imu_config_reset_count((uint8_t)IMU_SLOT_MTA));
    put_u32(payload + 136, imu_config_reset_count((uint8_t)IMU_SLOT_CRA));
    put_u32(payload + 140, ++stats_seq);
    host_link_send_record(REC_STATS, t_master_us(), payload, sizeof(payload));
}

/** @brief Internal helper: comms_core_send_heartbeat. */
static void comms_core_send_heartbeat(void) {
    uint8_t payload[16] = {0};
    uint32_t tx = host_link_tx_bytes();
    uint32_t missed = ads1299_missed_drdy_count();
    uint64_t fp = config_apply_fingerprint();
    memcpy(payload, &tx, sizeof(tx));
    memcpy(payload + 4, &missed, sizeof(missed));
    memcpy(payload + 8, &fp, sizeof(fp));
    host_link_send_record(REC_HEARTBEAT, t_master_us(), payload, sizeof(payload));
}

/*
 * Sensor-health watchdog (called once per second). missed_drdy is the ADS1299
 * driver's per-stream estimate of conversions that were never read (from DRDY
 * timing; the same count advances the EMG seq, contract K1) -- i.e. genuine EMG
 * sample loss. A healthy stream sits near 0/s even at 2 kHz. It restarts at 0
 * with every stream, so a drop in the counter re-baselines instead of reading
 * as a huge delta. If the per-second miss rate stays above the threshold
 * for several consecutive seconds the pipeline is degraded (host not draining,
 * bus wedged), so we latch the panel FAULT LED and emit one REC_LOG. When the
 * rate recovers we clear the fault and log recovery. This is advisory only; the
 * stream keeps running and EMG continues to absorb back-pressure first.
 */
#define WATCHDOG_MISSED_PER_S_TRIP 20u
#define WATCHDOG_TRIP_SECONDS      3u

/** @brief Internal helper: comms_core_watchdog_tick. */
static void comms_core_watchdog_tick(void) {
    static uint32_t last_missed;
    static uint32_t over_seconds;
    static bool tripped;

    uint32_t missed = ads1299_missed_drdy_count();
    if (missed < last_missed) {
        last_missed = 0; /* new stream */
    }
    uint32_t delta = missed - last_missed;
    last_missed = missed;

    if (delta > WATCHDOG_MISSED_PER_S_TRIP) {
        if (over_seconds < WATCHDOG_TRIP_SECONDS) {
            over_seconds++;
        }
        if (over_seconds >= WATCHDOG_TRIP_SECONDS && !tripped) {
            tripped = true;
            panel_fault_set(PANEL_FAULT_WATCHDOG_EMG);
            panel_set_state(PANEL_STATE_FAULT);
            host_link_send_log("watchdog: sustained missed_drdy (EMG loss)");
        }
    } else {
        over_seconds = 0;
        if (tripped) {
            tripped = false;
            panel_fault_clear(PANEL_FAULT_WATCHDOG_EMG);
            host_link_send_log("watchdog: missed_drdy recovered");
            panel_set_state(PANEL_STATE_STREAMING);
            panel_update_streaming_state();
        }
    }
}

/*
 * EMG framing reporter (called once per second).
 *
 * The sensor core detects a slipped RDATA byte stream per frame but cannot say
 * so: printf and host_link belong to this core. It only bumps counters, and we
 * turn a change in those counters into exactly one CDC line plus one REC_LOG per
 * second. Rate-limiting is the point -- at 2 kHz a sustained slip would
 * otherwise emit 2000 log records per second and drown the data stream it is
 * trying to describe.
 *
 * This is a reporter, not a watchdog: it never latches a panel fault and never
 * alters the stream, so it cannot make a marginal session worse.
 */
static void comms_core_emg_guard_tick(void) {
    static uint32_t last_misalign;
    static uint32_t last_resyncs;
    static bool was_bad;

    uint32_t misalign = ads1299_frame_misalign_count();
    uint32_t resyncs = ads1299_frame_resync_count();
    /*
     * ads1299_start_stream() zeroes these per stream, so starting a second
     * recording in the same power cycle makes them run backwards. The unsigned
     * subtraction below would wrap that into a ~4.29e9 "bad frames in the last
     * second" line on a perfectly clean stream -- precisely the kind of false
     * alarm this reporter exists to rule out. Re-baseline to the new stream
     * instead, and drop the stale was_bad so the restart cannot emit a phantom
     * "recovered" either: each stream's health is reported on its own terms.
     */
    if (misalign < last_misalign || resyncs < last_resyncs) {
        last_misalign = 0;
        last_resyncs = 0;
        was_bad = false;
    }
    uint32_t d_misalign = misalign - last_misalign;
    uint32_t d_resyncs = resyncs - last_resyncs;
    last_misalign = misalign;
    last_resyncs = resyncs;

    char msg[JTSMM_MAX_PAYLOAD];
    if (d_misalign > 0) {
        was_bad = true;
        snprintf(msg, sizeof(msg),
                 "EMG FRAMING SLIP: %lu bad frames in the last second (total %lu, "
                 "dropped %lu, resyncs %lu/+%lu, worst streak %lu, status0=0x%02X). "
                 "This is corrupted SPI framing, not electrode noise.",
                 (unsigned long)d_misalign, (unsigned long)misalign,
                 (unsigned long)emg_stream_misalign_drop_count(), (unsigned long)resyncs,
                 (unsigned long)d_resyncs,
                 (unsigned long)ads1299_frame_misalign_max_streak(),
                 (unsigned)ads1299_frame_misalign_last_status());
        host_link_send_log(msg);
        printf("%s\r\n", msg);
        if (ads1299_frame_resync_giveup_count() > 0) {
            host_link_send_log("EMG FRAMING SLIP: resync retry cap reached — "
                               "check SPI wiring/clock and stop bench commands");
        }
    } else if (was_bad) {
        was_bad = false;
        snprintf(msg, sizeof(msg),
                 "EMG framing recovered: no bad frames this second (total %lu bad, "
                 "%lu dropped, %lu resyncs)",
                 (unsigned long)misalign, (unsigned long)emg_stream_misalign_drop_count(),
                 (unsigned long)resyncs);
        host_link_send_log(msg);
        printf("%s\r\n", msg);
    }
}

/*
 * Trigger stamping reporter (once per second).
 *
 * TRIGGER_EVENT is normally stamped in the PIO IRQ at the pulse edge. If that
 * IRQ goes silent the software timer takes over (trigger_pio.c) so the session
 * keeps its trigger stream, but the stamps are then predictions, not
 * measurements, and the host must know. One line when it engages, then a
 * reminder every 10 s while the run continues.
 */
static void comms_core_trigger_stamp_tick(void) {
    static bool was_fallback;
    static uint8_t remind;

    bool fb = trigger_pio_fallback_active();
    if (fb && !was_fallback) {
        was_fallback = true;
        remind = 0;
        host_link_send_log("TRIGGER STAMP FALLBACK: PIO IRQ silent, timer is stamping "
                           "TRIGGER_EVENT (check `trig_status`; run `trig_stamp timer` "
                           "to make this explicit, or reflash if unexpected)");
        printf("TRIGGER STAMP FALLBACK: PIO IRQ silent; timer took over stamping "
               "(pio_pulses=%lu timer_events=%lu)\r\n",
               (unsigned long)trigger_pio_pulse_count(),
               (unsigned long)trigger_pio_timer_event_count());
    } else if (fb && ++remind >= 10u) {
        remind = 0;
        printf("TRIGGER STAMP FALLBACK still active (timer_events=%lu)\r\n",
               (unsigned long)trigger_pio_timer_event_count());
    } else if (!fb) {
        was_fallback = false;
    }

    /* Edges merged into a late PIO IRQ (K4): the ids already skip them, but
     * say so once per second so the session log explains the gap. */
    static uint32_t last_lost;
    uint32_t lost = trigger_pio_lost_edge_count();
    if (lost < last_lost) {
        last_lost = 0; /* reset by a new trig_start */
    }
    if (lost != last_lost) {
        char msg[JTSMM_MAX_PAYLOAD];
        snprintf(msg, sizeof(msg),
                 "TRIGGER EDGES LOST: %lu pulse(s) merged into late PIO IRQs (total %lu this run); "
                 "trigger_id skips them",
                 (unsigned long)(lost - last_lost), (unsigned long)lost);
        host_link_send_log(msg);
        printf("%s\r\n", msg);
        last_lost = lost;
    }
}

/** @brief Internal helper: log and act on one slot's watchdog decision (imu_watchdog.h). */
static void imu_watchdog_slot(const char *name, imu_slot_t slot, bool streaming, uint32_t pushes,
                              uint32_t drops, imu_wd_state_t *st) {
    imu_wd_action_t act = imu_watchdog_tick(st, streaming, pushes, drops);
    if (act == IMU_WD_NONE) {
        return;
    }
    char msg[JTSMM_MAX_PAYLOAD];
    if (act == IMU_WD_HOST_FULL) {
        snprintf(msg, sizeof(msg),
                 "IMU WATCHDOG: %s ring full -- the host is not reading the data endpoint "
                 "(sensor alive, no recovery); stream through jtsmm record/view",
                 name);
        host_link_send_log(msg);
        printf("%s\r\n", msg);
        return;
    }
    if (act == IMU_WD_GIVE_UP) {
        snprintf(msg, sizeof(msg),
                 "IMU WATCHDOG: %s still stalled after %u recovery attempts (%us) -- "
                 "giving up; check cable/mux/power, then imu_stream %s off/on",
                 name, (unsigned)st->attempts, (unsigned)st->stall_s, name);
        host_link_send_log(msg);
        printf("%s\r\n", msg);
        return;
    }
    snprintf(msg, sizeof(msg), "IMU WATCHDOG: %s stalled %us, recovery attempt %u/%u", name,
             (unsigned)st->stall_s, (unsigned)st->attempts, (unsigned)IMU_WATCHDOG_MAX_ATTEMPTS);
    host_link_send_log(msg);
    printf("%s\r\n", msg);
    imu_slot_recover_start(slot);
}

/** @brief Internal helper: comms_core_imu_watchdog_tick. */
static void comms_core_imu_watchdog_tick(void) {
    static imu_wd_state_t mta;
    static imu_wd_state_t cra;

    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    imu_watchdog_slot("mta", IMU_SLOT_MTA, cfg->stream_imu_mta, ringbuf_pushes(RING_IMU_MTA),
                      ringbuf_drops(RING_IMU_MTA), &mta);
    imu_watchdog_slot("cra", IMU_SLOT_CRA, cfg->stream_imu_cra, ringbuf_pushes(RING_IMU_CRA),
                      ringbuf_drops(RING_IMU_CRA), &cra);
}

/** @brief Internal helper: run a queued boot-time apply to completion. */
static void comms_core_run_apply(void) {
    while (config_apply_busy()) {
        config_apply_poll();
        usb_service_poll();
    }
}

void comms_core_main(void) {
    absolute_time_t boot_deadline = make_timeout_time_ms(1500);
    while (!time_reached(boot_deadline)) {
        usb_service_poll();
        sleep_ms(1);
    }
    host_link_init();
    cmd_parser_init();
    printf("\r\njtsmm production firmware %s\r\n", JTSMM_FW_VERSION);
    printf("Type 'info' or 'selftest'; 'config apply {...}'\r\n");
    printf("ready (run config apply or selftest)\r\n");

    /*
     * Bring the platform fully up (buses, DMA, sensors) on this core BEFORE the
     * sensor core is launched, so the sensor core never runs while the platform
     * is being initialized. This is the fix for the sensor-core boot-time freeze.
     * config_apply_poll() launches the sensor core in its CFG_FINISH step.
     *
     * A saved config that no longer parses/validates (or fails to apply) must not
     * leave the board without buses and sensor core: fall back to defaults.
     */
    bool saved_started = false;
    if (config_flash_has_valid()) {
        static char boot_json[JTSMM_CMD_LINE_MAX];
        if (config_flash_load(boot_json, sizeof(boot_json))) {
            printf("config load: saved config found, applying\r\n");
            saved_started = config_apply_start_json(boot_json);
        }
    }
    if (!saved_started) {
        (void)config_apply_start_defaults();
    }
    comms_core_run_apply();
    if (saved_started && !config_apply_last_ok()) {
        printf("config load: saved config FAILED, falling back to defaults "
               "(fix it and 'config save', or 'config erase')\r\n");
        host_link_send_log("boot: saved config failed to apply; running factory defaults");
        (void)config_apply_start_defaults();
        comms_core_run_apply();
    }
    fault_report_boot_reason();
    fault_watchdog_start();

    absolute_time_t next_hb = make_timeout_time_ms(1000);
    absolute_time_t next_panel = make_timeout_time_ms(500);
    absolute_time_t next_status = make_timeout_time_ms(100);
    absolute_time_t next_rate = make_timeout_time_ms(1000);
    uint32_t last_sensor_ticks = sensor_poll_ticks_count();
    uint32_t last_comms_ticks = comms_poll_ticks;

    while (true) {
        comms_poll_ticks++;
        fault_watchdog_service(sensor_poll_ticks_count(), sensor_core_running());
        /*
         * Comms core (physical core 0) is a pure consumer: USB service, ring
         * drain, command/config handling, panel LEDs and event producers.
         * It must NOT touch the ADS/IMU buses or push EMG/IMU rings -- those
         * are owned exclusively by the sensor core (sensor_core.c). See plan.
         */
        usb_service_poll();
        host_link_drain();
        host_link_drain();
        /*
         * Advance any in-progress IMU recovery by one bounded step. It sits
         * between the two rounds of ring draining so a multi-second reset/re-sync
         * no longer starves host_link_drain() and overflows the EMG ring.
         */
        imu_recover_poll();
        cmd_parser_poll();
        config_apply_poll();
        panel_poll_buttons();
        panel_disco_tick();
        panel_panel_test_poll();
        if (time_reached(next_panel)) {
            panel_heartbeat_tick();
            next_panel = make_timeout_time_ms(500);
        }
        if (time_reached(next_status)) {
            panel_status_tick();
            next_status = make_timeout_time_ms(100);
        }
        if (time_reached(next_rate)) {
            uint32_t s = sensor_poll_ticks_count();
            uint32_t c = comms_poll_ticks;
            sensor_loop_hz = s - last_sensor_ticks;
            comms_loop_hz = c - last_comms_ticks;
            last_sensor_ticks = s;
            last_comms_ticks = c;
            next_rate = make_timeout_time_ms(1000);
        }
        if (time_reached(next_hb)) {
            comms_core_send_heartbeat();
            comms_core_send_stats();
            comms_core_watchdog_tick();
            comms_core_emg_guard_tick();
            comms_core_trigger_stamp_tick();
            comms_core_imu_watchdog_tick();
            next_hb = make_timeout_time_ms(1000);
        }
        tight_loop_contents();
    }
}
