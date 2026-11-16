// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_apply.c
 * @brief Apply JSON/defaults onto hardware subsystems.
 */

#include "jtsmm/runtime/config_apply.h"

#include "jtsmm/app/fault.h"
#include "jtsmm/app/sensor_core.h"
#include "jtsmm/domain/emg_config.h"
#include "jtsmm/domain/emg_config_check.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/domain/trigger_config.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/drivers/trigger_timing.h"
#include "jtsmm/config.h"
#include "jtsmm/platform/i2c_dma.h"
#include "jtsmm/platform/platform_config.h"
#include "jtsmm/runtime/config_json.h"
#include "jtsmm/runtime/host_link.h"
#include "jtsmm/runtime/usb_service.h"
#include <stdio.h>
#include <string.h>

/*
 * Apply sequence (one step per config_apply_poll() so USB keeps running):
 *
 *   CFG_VALIDATE  parse + resolve + validate everything into a scratch config.
 *                 On any error: reply FAIL with the reason and stop here --
 *                 nothing has been touched, the old config and any running
 *                 streams carry on (at boot, comms_core falls back to defaults).
 *                 On success: quiesce (stop trigger/IMU/EMG streams, park the
 *                 sensor core, abort IMU I2C DMA) and copy scratch -> live.
 *   CFG_PLATFORM .. CFG_TRIGGER   apply the live config to the hardware.
 *   CFG_FINISH    ack + reply, and relaunch the sensor core whenever the
 *                 platform (buses/DMA) came up -- even if a later sensor step
 *                 failed, so whatever did come up keeps working.
 *
 * Previously the live config was overwritten and the sensor core stopped before
 * the JSON had even been parsed, so a typo in `config apply` left the rig with
 * defaults-plus-half-a-config and no sensor core.
 */
typedef enum {
    CFG_IDLE = 0,
    CFG_VALIDATE,
    CFG_PLATFORM,
    CFG_EMG,
    CFG_IMU_MTA,
    CFG_IMU_CRA,
    CFG_TRIGGER,
    CFG_FINISH,
} cfg_step_t;

static cfg_step_t cfg_step = CFG_IDLE;
static bool cfg_ok = true;
static bool cfg_last_ok = true;
static bool cfg_was_init = false;
static bool cfg_platform_up = false;
static const char *cfg_fail_step;
static char cfg_json_buf[JTSMM_CMD_LINE_MAX];
static jtsmm_mcu_config_t cfg_scratch;

/** @brief Internal helper: config_apply_yield. */
static void config_apply_yield(void) {
    for (int i = 0; i < 50; i++) {
        usb_service_poll();
    }
}

static uint64_t g_config_fingerprint;

/** @brief Internal helper: fnv1a64_update. */
static uint64_t fnv1a64_update(uint64_t h, uint64_t v) {
    h ^= v;
    return h * 0x100000001b3ULL;
}

/** @brief Internal helper: fnv1a64_bytes. */
static uint64_t fnv1a64_bytes(uint64_t h, const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        h = fnv1a64_update(h, p[i]);
    }
    return h;
}

void config_apply_update_fingerprint(void) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    uint64_t h = 0xcbf29ce484222325ULL;
    h = fnv1a64_update(h, cfg->schema_version);
    h = fnv1a64_update(h, cfg->platform.spi0_stream_hz);
    h = fnv1a64_update(h, cfg->platform.i2c_hz);
    h = fnv1a64_update(h, cfg->emg.sample_rate_sps);
    h = fnv1a64_update(h, cfg->emg.default_gain);
    h = fnv1a64_update(h, cfg->emg.profile);
    h = fnv1a64_update(h, cfg->emg.bias.channel_mask);
    h = fnv1a64_update(h, cfg->trigger.rate_hz);
    h = fnv1a64_update(h, cfg->trigger.pulse_width_us);
    h = fnv1a64_update(h, cfg->imu_mta.fusion_rate_hz);
    h = fnv1a64_update(h, cfg->imu_mta.sync_mode);
    h = fnv1a64_update(h, cfg->imu_mta.num_reports);
    h = fnv1a64_bytes(h, cfg->imu_mta.reports, sizeof(cfg->imu_mta.reports[0]) * cfg->imu_mta.num_reports);
    h = fnv1a64_update(h, cfg->imu_cra.fusion_rate_hz);
    h = fnv1a64_update(h, cfg->imu_cra.sync_mode);
    h = fnv1a64_update(h, cfg->imu_cra.num_reports);
    h = fnv1a64_bytes(h, cfg->imu_cra.reports, sizeof(cfg->imu_cra.reports[0]) * cfg->imu_cra.num_reports);
    g_config_fingerprint = h;
}

uint64_t config_apply_fingerprint(void) {
    return g_config_fingerprint;
}

/** @brief Internal helper: resolve_missing_profiles. */
static void resolve_missing_profiles(jtsmm_mcu_config_t *cfg) {
    if (cfg->emg.profile != EMG_PROFILE_CUSTOM) {
        emg_config_resolve_profile(&cfg->emg);
    }
    if (cfg->imu_mta.num_reports == 0) {
        imu_config_resolve_profile(&cfg->imu_mta);
    }
    if (cfg->imu_cra.num_reports == 0) {
        imu_config_resolve_profile(&cfg->imu_cra);
    }
    if (cfg->trigger.profile != TRIGGER_PROFILE_CUSTOM) {
        trigger_config_resolve_profile(&cfg->trigger);
    }
}

/**
 * @brief Internal helper: expand profiles, then re-overlay explicit JSON values.
 *
 * @details Order matters here. resolve_missing_profiles() writes concrete values
 *          for every field its profile owns -- for `acquisition` that is
 *          sample_rate_sps, default_gain, bias.enabled, leadoff.enabled and
 *          leadoff.during_recording. Running it *after* the parse therefore
 *          overwrites whatever the host asked for: a request for gain 24 came
 *          back as gain 6 while the host still derived lsb_uv from 24, scaling
 *          the entire recording by 4x, and `leadoff.during_recording: true` (the
 *          documented default in schema/v1) was forced back to false while the
 *          host kept computing lead-off rails from it.
 *
 *          That is masked today only by a side effect: an explicit
 *          `sample_rate_sps` flips the profile to CUSTOM, which short-circuits
 *          resolution -- so correctness depends on the host happening to send
 *          one particular key. Re-parsing after resolution makes the precedence
 *          explicit and independent of that: the profile supplies defaults, an
 *          explicit host value always wins.
 *
 * @param cfg  Config to resolve in place.
 * @param json Apply JSON to re-overlay; NULL/empty for the defaults path.
 */
static bool resolve_and_overlay(jtsmm_mcu_config_t *cfg, const char *json) {
    resolve_missing_profiles(cfg);
    if (json == NULL || json[0] == '\0') {
        return true;
    }
    emg_config_t before = cfg->emg;
    if (!config_json_parse(json, cfg)) {
        return false;
    }
    if (before.default_gain != cfg->emg.default_gain ||
        before.sample_rate_sps != cfg->emg.sample_rate_sps ||
        before.bias.enabled != cfg->emg.bias.enabled ||
        before.leadoff.enabled != cfg->emg.leadoff.enabled ||
        before.leadoff.during_recording != cfg->emg.leadoff.during_recording) {
        printf("config apply: host overrides profile defaults: fs %u->%u gain %u->%u "
               "bias %d->%d leadoff %d->%d during_rec %d->%d\r\n",
               before.sample_rate_sps, cfg->emg.sample_rate_sps,
               (unsigned)before.default_gain, (unsigned)cfg->emg.default_gain,
               before.bias.enabled ? 1 : 0, cfg->emg.bias.enabled ? 1 : 0,
               before.leadoff.enabled ? 1 : 0, cfg->emg.leadoff.enabled ? 1 : 0,
               before.leadoff.during_recording ? 1 : 0,
               cfg->emg.leadoff.during_recording ? 1 : 0);
    }
    return true;
}

/**
 * @brief Internal helper: build and fully validate @p out without touching hardware.
 * @return NULL on success, else a short reason.
 */
static const char *prepare_scratch(jtsmm_mcu_config_t *out, const char *json) {
    static char reason[128];
    jtsmm_config_set_defaults(out);
    if (json != NULL && json[0] != '\0') {
        if (!config_json_parse(json, out)) {
            snprintf(reason, sizeof(reason), "parse: %s", config_json_last_error());
            return reason;
        }
    }
    if (!resolve_and_overlay(out, json)) {
        snprintf(reason, sizeof(reason), "re-parse: %s", config_json_last_error());
        return reason;
    }
    if (!platform_config_validate(&out->platform)) {
        return "platform: unsupported value";
    }
    /* The real reason (gain, lead-off current, bias mask, rate): this used to
     * report every EMG rejection as "unsupported sample_rate_sps". */
    const char *emg_why = emg_config_check(&out->emg);
    if (emg_why != NULL) {
        snprintf(reason, sizeof(reason), "emg: %s", emg_why);
        return reason;
    }
    if (out->imu_mta.enabled && !imu_config_validate_slot(&out->imu_mta, out->emg.sample_rate_sps)) {
        return "imu_mta: invalid sync/report rates";
    }
    if (out->imu_cra.enabled && !imu_config_validate_slot(&out->imu_cra, out->emg.sample_rate_sps)) {
        return "imu_cra: invalid sync/report rates";
    }
    const char *trig = trigger_timing_check(out->trigger.rate_hz, out->trigger.pulse_width_us);
    if (trig != NULL) {
        snprintf(reason, sizeof(reason), "trigger: %s", trig);
        return reason;
    }
    return NULL;
}

/**
 * @brief Internal helper: stop everything that uses the sensor buses / trigger.
 *
 * @details Streams first (the same calls ads_stop / imu_stream off / trig_stop
 *          make), then park the sensor core so nothing is mid-transfer, then
 *          abort any IMU I2C DMA it left in flight. A trigger stopped here is
 *          logged: a host that expected it to keep running must be able to see
 *          why the TRIGGER_EVENTs ended.
 */
static void quiesce(void) {
    jtsmm_mcu_config_t *live = jtsmm_config_mutable();
    char msg[96];
    bool trig = trigger_pio_running();
    if (trig || live->stream_emg || live->stream_imu_mta || live->stream_imu_cra) {
        snprintf(msg, sizeof(msg), "config apply: stopping%s%s%s%s", trig ? " trigger" : "",
                 live->stream_emg ? " emg" : "", live->stream_imu_mta ? " imu_mta" : "",
                 live->stream_imu_cra ? " imu_cra" : "");
        if (trig) {
            size_t n = strlen(msg);
            snprintf(msg + n, sizeof(msg) - n, " (last trigger_id %llu)",
                     (unsigned long long)trigger_pio_id());
        }
        host_link_send_log(msg);
        printf("%s\r\n", msg);
    }
    trigger_pio_stop();
    imu_stream_stop_slot(IMU_SLOT_CRA);
    imu_stream_stop_slot(IMU_SLOT_MTA);
    emg_stream_stop();
    live->stream_emg = false;
    live->stream_imu_mta = false;
    live->stream_imu_cra = false;
    (void)sensor_core_stop();
    i2c_dma_abort(I2C_DMA_SLOT_MTA);
    i2c_dma_abort(I2C_DMA_SLOT_CRA);
    panel_update_streaming_state();
}

/** @brief Internal helper: mark a hardware step failed (first failure wins). */
static void step_result(bool ok, const char *step) {
    if (!ok && cfg_ok) {
        cfg_fail_step = step;
    }
    cfg_ok = cfg_ok && ok;
}

bool config_apply_busy(void) {
    return cfg_step != CFG_IDLE;
}

bool config_apply_last_ok(void) {
    return cfg_last_ok;
}

bool config_apply_start_defaults(void) {
    if (cfg_step != CFG_IDLE) {
        return false;
    }
    cfg_was_init = true;
    cfg_json_buf[0] = '\0';
    cfg_step = CFG_VALIDATE;
    printf("config init started\r\n");
    return true;
}

bool config_apply_start_json(const char *json) {
    if (cfg_step != CFG_IDLE || json == NULL) {
        return false;
    }
    strncpy(cfg_json_buf, json, sizeof(cfg_json_buf) - 1);
    cfg_json_buf[sizeof(cfg_json_buf) - 1] = '\0';
    cfg_was_init = false;
    cfg_step = CFG_VALIDATE;
    printf("config apply started\r\n");
    return true;
}

/** @brief Internal helper: final reply line. */
static void print_result(bool ok, const char *reason) {
    const char *what = cfg_was_init ? "config init" : "config apply";
    if (ok) {
        printf("%s OK\r\n", what);
    } else {
        printf("%s FAIL (%s)\r\n", what, reason ? reason : "unknown");
    }
}

void config_apply_poll(void) {
    if (cfg_step == CFG_IDLE) {
        return;
    }

    jtsmm_mcu_config_t *cfg = jtsmm_config_mutable();

    switch (cfg_step) {
    case CFG_VALIDATE: {
        const char *reason = prepare_scratch(&cfg_scratch, cfg_json_buf);
        if (reason != NULL) {
            /* Nothing touched: old config, streams and sensor core carry on. */
            host_link_send_ack(false);
            print_result(false, reason);
            cfg_last_ok = false;
            cfg_json_buf[0] = '\0';
            cfg_step = CFG_IDLE;
            break;
        }
        /* Hardware steps can block for seconds (BNO085 boot): stretch the
         * watchdog until CFG_FINISH instead of letting it fire mid-apply. */
        fault_watchdog_hold();
        quiesce();
        *cfg = cfg_scratch;
        cfg_ok = true;
        cfg_platform_up = false;
        cfg_fail_step = NULL;
        cfg_step = CFG_PLATFORM;
        break;
    }

    case CFG_PLATFORM:
        cfg_platform_up = platform_config_apply(&cfg->platform);
        step_result(cfg_platform_up, "platform apply");
        cfg_step = CFG_EMG;
        break;

    case CFG_EMG:
        if (cfg->emg.enabled) {
            step_result(emg_config_apply(&cfg->emg), "emg apply");
        } else {
            /* emg.enabled=false is honoured, not ignored: the ADS1299 is left
             * unconfigured and ads_start / the panel capture refuse EMG. */
            printf("config apply: emg.enabled=false, ADS1299 not configured\r\n");
        }
        cfg_step = CFG_IMU_MTA;
        break;

    case CFG_IMU_MTA:
        step_result(imu_config_apply_slot(IMU_SLOT_MTA, &cfg->imu_mta), "imu_mta apply");
        cfg_step = CFG_IMU_CRA;
        break;

    case CFG_IMU_CRA:
        /* Also for a disabled slot: the IMU domain then turns its reports off
         * and records the slot as disabled (streams on it are refused). */
        step_result(imu_config_apply_slot(IMU_SLOT_CRA, &cfg->imu_cra), "imu_cra apply");
        cfg_step = CFG_TRIGGER;
        break;

    case CFG_TRIGGER:
        step_result(trigger_config_apply(&cfg->trigger), "trigger apply");
        if (cfg_ok) {
            config_apply_update_fingerprint();
        }
        cfg_step = CFG_FINISH;
        break;

    case CFG_FINISH:
        fault_watchdog_release();
        host_link_send_ack(cfg_ok);
        print_result(cfg_ok, cfg_fail_step);
        cfg_last_ok = cfg_ok;
        cfg_json_buf[0] = '\0';
        cfg_step = CFG_IDLE;
        /* Buses/DMA are up: (re)launch the sensor producer core, also after a
         * later sensor step failed, so whatever did come up keeps streaming. */
        if (cfg_platform_up) {
            sensor_core_launch();
        } else {
            printf("config apply: platform not up, sensor core left stopped\r\n");
        }
        break;

    default:
        cfg_step = CFG_IDLE;
        break;
    }

    config_apply_yield();
}
