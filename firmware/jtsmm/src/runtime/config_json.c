// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_json.c
 * @brief cJSON parse/dump of MCU config.
 */

#include "jtsmm/runtime/config_json.h"

#include "jtsmm/domain/bno085_reports.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"
#include "cJSON.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* Ring byte budgets convert with each ring's own slot size (ringbuf_slot_bytes:
 * 48 B compact EMG slots, 256 B IMU/event slots), and `config dump` multiplies
 * back with the same size, so the golden budgets map to whole slots and round
 * trip exactly. */

/*
 * Parse policy
 * ------------
 * - A key that is absent keeps the value already in the config (factory default
 *   or profile expansion). Earlier some absent keys were forced instead: a
 *   `pulse` object without `polarity` silently became imx296_fast, and a
 *   `leadoff` object without `during_recording` forced it false.
 * - A key that is present but unusable is an error, never a silent default:
 *   numbers must be non-negative integers that fit the field they land in
 *   (`rate_hz: 65566` used to wrap to 30 in a uint16), booleans are true/false
 *   or the numbers 0/1, and every section must be a JSON object
 *   (`"trigger": null` used to parse as "all defaults").
 * - Enum strings the firmware does not know map to the default as before: the
 *   schema allows a few names (e.g. imu profile `datasheet_motion`) that the
 *   firmware simply does not distinguish.
 * - A key the firmware parses must take effect or be refused. Keys that
 *   describe fixed hardware/firmware facts (CPU/crystal clock, the non-trigger
 *   IRQ priorities, the hard-wired trigger fan-out, CH7 readback, test-signal
 *   amplitude) accept only the value that is actually true and are otherwise
 *   refused with a reason, instead of being parsed and silently ignored.
 * The first error is kept for config_json_last_error().
 */
static char parse_err[96];

/** @brief Internal helper: record the first parse error. */
static void set_err(const char *fmt, ...) {
    if (parse_err[0] != '\0') {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(parse_err, sizeof(parse_err), fmt, ap);
    va_end(ap);
}

const char *config_json_last_error(void) {
    return parse_err[0] != '\0' ? parse_err : "no error";
}

/**
 * @brief Internal helper: optional sub-object.
 * @return false (error set) if @p key is present but not an object; else true
 *         with @p *out = the object or NULL when absent.
 */
static bool json_section(cJSON *obj, const char *key, const char *where, cJSON **out) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    *out = NULL;
    if (item == NULL) {
        return true;
    }
    if (!cJSON_IsObject(item)) {
        set_err("%s.%s: expected an object", where, key);
        return false;
    }
    *out = item;
    return true;
}

/**
 * @brief Internal helper: optional boolean (true/false or the numbers 0/1).
 * @return false (error set) if present with any other value; @p *out is only
 *         written when the key is present.
 */
static bool json_bool(cJSON *obj, const char *key, const char *where, bool *out) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    if (item == NULL) {
        return true;
    }
    if (cJSON_IsBool(item)) {
        *out = cJSON_IsTrue(item);
        return true;
    }
    if (cJSON_IsNumber(item) && (item->valuedouble == 0.0 || item->valuedouble == 1.0)) {
        *out = item->valuedouble == 1.0;
        return true;
    }
    set_err("%s.%s: expected true/false", where, key);
    return false;
}

/**
 * @brief Internal helper: optional unsigned integer in [0, @p max].
 * @return false (error set) if present but not a number, negative, fractional
 *         or above @p max; @p *present tells whether @p *out was written.
 */
static bool json_uint(cJSON *obj, const char *key, uint32_t max, const char *where, uint32_t *out,
                      bool *present) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    *present = false;
    if (item == NULL) {
        return true;
    }
    double v = cJSON_IsNumber(item) ? item->valuedouble : -1.0;
    if (!cJSON_IsNumber(item) || v < 0.0 || v > (double)max || v != (double)(uint32_t)v) {
        set_err("%s.%s: expected an integer 0..%lu", where, key, (unsigned long)max);
        return false;
    }
    *out = (uint32_t)v;
    *present = true;
    return true;
}

/** @brief Internal helper: optional string; false (error set) if not a string. */
static bool json_str(cJSON *obj, const char *key, const char *where, const char **out) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(obj, key);
    *out = NULL;
    if (item == NULL) {
        return true;
    }
    if (!cJSON_IsString(item) || item->valuestring == NULL) {
        set_err("%s.%s: expected a string", where, key);
        return false;
    }
    *out = item->valuestring;
    return true;
}

/** @brief Internal helper: refuse a parsed value the firmware cannot honour. */
static bool fixed_value(bool ok, const char *where, const char *key, const char *why) {
    if (!ok) {
        set_err("%s.%s: %s", where, key, why);
    }
    return ok;
}

/* Field helpers: bail out of the enclosing parse_* on error, keep on absence. */
#define GET_BOOL(obj, key, dst)                                                  \
    do {                                                                         \
        if (!json_bool((obj), (key), where, &(dst))) {                           \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define GET_UINT(obj, key, max, dst, type)                                       \
    do {                                                                         \
        uint32_t v_;                                                             \
        bool p_;                                                                 \
        if (!json_uint((obj), (key), (max), where, &v_, &p_)) {                  \
            return false;                                                        \
        }                                                                        \
        if (p_) {                                                                \
            (dst) = (type)v_;                                                    \
        }                                                                        \
    } while (0)

#define GET_SECTION(obj, key, dst)                                               \
    do {                                                                         \
        if (!json_section((obj), (key), where, &(dst))) {                        \
            return false;                                                        \
        }                                                                        \
    } while (0)

#define GET_STR(obj, key, dst)                                                   \
    do {                                                                         \
        if (!json_str((obj), (key), where, &(dst))) {                            \
            return false;                                                        \
        }                                                                        \
    } while (0)

/** @brief Internal helper: parse_emg_profile. */
static emg_profile_t parse_emg_profile(const char *s) {
    if (s == NULL) {
        return EMG_PROFILE_ACQUISITION;
    }
    if (strcmp(s, "high_rate") == 0) {
        return EMG_PROFILE_HIGH_RATE;
    }
    if (strcmp(s, "bench") == 0) {
        return EMG_PROFILE_BENCH;
    }
    if (strcmp(s, "noise_characterization") == 0) {
        return EMG_PROFILE_NOISE_CHARACTERIZATION;
    }
    if (strcmp(s, "custom") == 0) {
        return EMG_PROFILE_CUSTOM;
    }
    return EMG_PROFILE_ACQUISITION;
}

/**
 * @brief Internal helper: parse_imu_profile.
 * @return false for a name this firmware has no preset for (e.g. the host-only
 *         datasheet_motion); the caller accepts that only with an explicit
 *         report list, which replaces the preset anyway.
 */
static bool parse_imu_profile(const char *s, imu_profile_t *out) {
    static const struct {
        const char *name;
        imu_profile_t profile;
    } kNames[] = {
        {"kinematics_accel", IMU_PROFILE_KINEMATICS_ACCEL},
        {"kinematics_default", IMU_PROFILE_KINEMATICS_DEFAULT},
        {"fused_only", IMU_PROFILE_FUSED_ONLY},
        {"raw_only", IMU_PROFILE_RAW_ONLY},
        {"all", IMU_PROFILE_ALL},
        {"custom", IMU_PROFILE_CUSTOM},
    };
    for (size_t i = 0; i < sizeof(kNames) / sizeof(kNames[0]); i++) {
        if (strcmp(s, kNames[i].name) == 0) {
            *out = kNames[i].profile;
            return true;
        }
    }
    return false;
}

/** @brief Internal helper: parse_fusion_primary. */
static imu_fusion_primary_t parse_fusion_primary(const char *s) {
    if (s == NULL) {
        return IMU_FUSION_GAME_ROTATION_VECTOR;
    }
    if (strcmp(s, "rotation_vector") == 0) {
        return IMU_FUSION_ROTATION_VECTOR;
    }
    if (strcmp(s, "geomagnetic_rotation_vector") == 0) {
        return IMU_FUSION_GEOMAGNETIC_ROTATION_VECTOR;
    }
    if (strcmp(s, "gyro_integrated_rotation_vector") == 0) {
        return IMU_FUSION_GYRO_INTEGRATED_RV;
    }
    return IMU_FUSION_GAME_ROTATION_VECTOR;
}

/** @brief Internal helper: parse_trigger_profile. */
static trigger_profile_t parse_trigger_profile(const char *s) {
    if (s == NULL) {
        return TRIGGER_PROFILE_IMX296_DUAL_CSI;
    }
    if (strcmp(s, "imx296_high_fps") == 0) {
        return TRIGGER_PROFILE_IMX296_HIGH_FPS;
    }
    if (strcmp(s, "bench_scope") == 0) {
        return TRIGGER_PROFILE_BENCH_SCOPE;
    }
    if (strcmp(s, "track_a_led") == 0) {
        return TRIGGER_PROFILE_TRACK_A_LED;
    }
    if (strcmp(s, "custom") == 0) {
        return TRIGGER_PROFILE_CUSTOM;
    }
    return TRIGGER_PROFILE_IMX296_DUAL_CSI;
}

/**
 * @brief Internal helper: a ring's byte budget -> slots of that ring.
 *
 * @details Slot sizes differ per ring (ringbuf_slot_bytes: compact EMG slots,
 *          full-record IMU/event slots), and `config dump` multiplies back, so
 *          the round trip is exact. The upper bound only guards a bogus value;
 *          ringbuf_set_capacity() clamps to each backing store and
 *          platform_config reports that.
 */
static uint32_t ring_bytes_to_slots(ring_id_t id, uint32_t bytes) {
    uint32_t slot = ringbuf_slot_bytes(id);
    if (bytes < slot) {
        return 16u;
    }
    uint32_t slots = bytes / slot;
    if (slots < 8u) {
        slots = 8u;
    }
    if (slots > 4096u) {
        slots = 4096u;
    }
    return slots;
}

/** @brief Internal helper: parse_platform. */
static bool parse_platform(cJSON *obj, platform_config_t *out) {
    /* Each sub-block re-declares `where` so errors name the full key path. */
    const char *where = "platform";
    const char *prof;
    GET_STR(obj, "profile", prof);
    if (prof && strcmp(prof, "default") != 0 && strcmp(prof, "custom") != 0) {
        set_err("platform.profile: '%.20s' not implemented (default|custom)", prof);
        return false;
    }
    cJSON *sec;
    GET_SECTION(obj, "system", sec);
    if (sec) {
        const char *where = "platform.system";
        uint32_t mhz;
        bool present;
        if (!json_uint(sec, "cpu_mhz", 4294u, where, &mhz, &present)) {
            return false;
        }
        if (present) {
            out->cpu_hz = mhz * 1000000u;
        }
        GET_UINT(sec, "xosc_hz", UINT32_MAX, out->xosc_hz, uint32_t);
        /* The firmware never reclocks: pico-sdk runs clk_sys at 150 MHz off the
         * 12 MHz crystal, and every timing constant assumes exactly that. */
        if (!fixed_value(out->cpu_hz == 150000000u, where, "cpu_mhz",
                         "only 150 is supported (no reclocking)") ||
            !fixed_value(out->xosc_hz == 12000000u, where, "xosc_hz",
                         "the board crystal is 12000000")) {
            return false;
        }
    }
    GET_SECTION(obj, "spi0", sec);
    if (sec) {
        const char *where = "platform.spi0";
        GET_UINT(sec, "hz", UINT32_MAX, out->spi0_stream_hz, uint32_t);
        GET_UINT(sec, "cfg_hz", UINT32_MAX, out->spi0_cfg_hz, uint32_t);
    }
    GET_SECTION(obj, "i2c", sec);
    if (sec) {
        const char *where = "platform.i2c";
        GET_UINT(sec, "hz", UINT32_MAX, out->i2c_hz, uint32_t);
    }
    GET_SECTION(obj, "pio", sec);
    if (sec) {
        const char *where = "platform.pio";
        /* Schema key is platform.pio.trigger.clock_hz. The flat
         * pio.trigger_clock_hz is what this parser used to (only) read; keep
         * accepting it for hand-written configs. */
        cJSON *trig;
        GET_SECTION(sec, "trigger", trig);
        if (trig) {
            const char *where = "platform.pio.trigger";
            GET_UINT(trig, "clock_hz", UINT32_MAX, out->pio_trigger_clock_hz, uint32_t);
        }
        GET_UINT(sec, "trigger_clock_hz", UINT32_MAX, out->pio_trigger_clock_hz, uint32_t);
        /* trigger_timing.c counts one PIO tick as one microsecond (templates and
         * the ISR's period arithmetic). Rescaling silently would hide a config
         * mistake, so anything else is refused. */
        if (out->pio_trigger_clock_hz != 1000000u) {
            set_err("platform.pio.trigger.clock_hz=%lu: only 1000000 is supported",
                    (unsigned long)out->pio_trigger_clock_hz);
            return false;
        }
    }
    GET_SECTION(obj, "buffers", sec);
    if (sec) {
        const char *where = "platform.buffers";
        uint32_t bytes = 0;
        bool present;
        if (!json_uint(sec, "emg_ring", UINT32_MAX, where, &bytes, &present)) {
            return false;
        }
        if (present) {
            out->emg_ring_slots = (uint16_t)ring_bytes_to_slots(RING_EMG, bytes);
        }
        if (!json_uint(sec, "imu_ring", UINT32_MAX, where, &bytes, &present)) {
            return false;
        }
        if (present) {
            out->imu_ring_slots = (uint16_t)ring_bytes_to_slots(RING_IMU_MTA, bytes);
        }
        if (!json_uint(sec, "event_ring", UINT32_MAX, where, &bytes, &present)) {
            return false;
        }
        if (present) {
            out->event_ring_slots = (uint16_t)ring_bytes_to_slots(RING_EVENTS, bytes);
        }
    }
    GET_SECTION(obj, "irq", sec);
    if (sec) {
        const char *where = "platform.irq";
        GET_UINT(sec, "emg", 255u, out->irq_emg_priority, uint8_t);
        GET_UINT(sec, "imu", 255u, out->irq_imu_priority, uint8_t);
        GET_UINT(sec, "trigger", 3u, out->irq_trigger_priority, uint8_t);
        /* EMG and IMU are serviced by polling on core 1, so these two have no
         * knob to turn; only the trigger priority is applied (trigger_pio.c). */
        if (!fixed_value(out->irq_emg_priority == 0u, where, "emg",
                         "fixed at 0 (EMG is polled on core 1)") ||
            !fixed_value(out->irq_imu_priority == 1u, where, "imu",
                         "fixed at 1 (IMU is polled on core 1)")) {
            return false;
        }
    }
    return true;
}

/** @brief Internal helper: parse_emg. */
static bool parse_emg(cJSON *obj, emg_config_t *out) {
    const char *where = "emg";
    const char *prof;
    GET_BOOL(obj, "enabled", out->enabled);
    GET_STR(obj, "profile", prof);
    if (prof) {
        out->profile = parse_emg_profile(prof);
    }
    uint32_t fs;
    bool fs_present;
    if (!json_uint(obj, "sample_rate_sps", UINT16_MAX, where, &fs, &fs_present)) {
        return false;
    }
    if (fs_present) {
        out->sample_rate_sps = (uint16_t)fs;
        out->profile = EMG_PROFILE_CUSTOM;
    }
    GET_UINT(obj, "default_gain", 255u, out->default_gain, uint8_t);
    GET_UINT(obj, "stream_channels", 255u, out->stream_channels, uint8_t);
    /* Every EMG_BLOCK carries all 8 channels; 6 vs 8 is the host's selection. */
    if (!fixed_value(out->stream_channels == 6u || out->stream_channels == 8u, where,
                     "stream_channels", "must be 6 or 8")) {
        return false;
    }
    uint32_t pd = out->pd_channel_mask;
    GET_UINT(obj, "pd_channels", 255u, pd, uint32_t);
    out->pd_channel_mask = (uint8_t)(pd & 0x3Fu);

    cJSON *sec;
    GET_SECTION(obj, "bias", sec);
    if (sec) {
        GET_BOOL(sec, "enabled", out->bias.enabled);
        GET_UINT(sec, "channels", 255u, out->bias.channel_mask, uint8_t);
        GET_BOOL(sec, "ch7_readback", out->bias.ch7_readback);
        if (!fixed_value(!out->bias.ch7_readback, where, "bias.ch7_readback",
                         "not implemented (CH7 always streams as wired)")) {
            return false;
        }
    }
    GET_SECTION(obj, "leadoff", sec);
    if (sec) {
        GET_BOOL(sec, "enabled", out->leadoff.enabled);
        GET_BOOL(sec, "during_recording", out->leadoff.during_recording);
        GET_BOOL(sec, "ac_tone", out->leadoff.ac_tone);
        GET_BOOL(sec, "probe_drl", out->leadoff.probe_drl);
        GET_UINT(sec, "current_na", 255u, out->leadoff.current_na, uint8_t);
    }
    GET_SECTION(obj, "test_signal", sec);
    if (sec) {
        GET_BOOL(sec, "enabled", out->test_signal.enabled);
        GET_UINT(sec, "amplitude_mv", 255u, out->test_signal.amplitude_mv, uint8_t);
        if (!fixed_value(out->test_signal.amplitude_mv == 0u, where, "test_signal.amplitude_mv",
                         "not adjustable; 0 = the ADS1299 default test signal")) {
            return false;
        }
    }
    return true;
}

/** @brief Internal helper: parse_imu_reports. */
static bool parse_imu_reports(cJSON *reports, const char *where, imu_slot_config_t *out) {
    out->num_reports = 0;
    for (cJSON *item = reports->child; item != NULL; item = item->next) {
        if (!cJSON_IsObject(item) || item->string == NULL) {
            set_err("%s.reports: each report must be an object", where);
            return false;
        }
        bool enabled = true;
        if (!json_bool(item, "enabled", where, &enabled)) {
            return false;
        }
        if (!enabled) {
            continue;
        }
        const bno085_report_entry_t *spec = bno085_report_find_by_name(item->string);
        if (spec == NULL) {
            set_err("%s.reports.%.32s: not implemented by this firmware", where, item->string);
            return false;
        }
        if (out->num_reports >= IMU_MAX_REPORTS) {
            set_err("%s.reports: more than %d enabled", where, IMU_MAX_REPORTS);
            return false;
        }
        uint32_t rate = spec->default_rate_hz;
        bool present;
        if (!json_uint(item, "rate_hz", UINT16_MAX, where, &rate, &present)) {
            return false;
        }
        out->reports[out->num_reports++] = (imu_report_cfg_t){
            .enabled = true,
            .report_id = spec->report_id,
            .rate_hz = (uint16_t)rate,
            .usb_type = spec->usb_type,
            .payload_len = (uint16_t)spec->payload_len,
        };
    }
    return true;
}

/** @brief Internal helper: parse_imu_slot. */
static bool parse_imu_slot(cJSON *obj, const char *where, imu_slot_config_t *out, uint16_t emg_fs_hz) {
    const char *str;
    GET_BOOL(obj, "enabled", out->enabled);
    GET_STR(obj, "profile", str);
    /* An unknown name used to fall back to kinematics_default silently. */
    const char *unknown_profile = NULL;
    if (str && !parse_imu_profile(str, &out->profile)) {
        unknown_profile = str;
    }
    cJSON *sec;
    GET_SECTION(obj, "fusion", sec);
    if (sec) {
        GET_STR(sec, "primary", str);
        if (str) {
            out->fusion_primary = parse_fusion_primary(str);
        }
        GET_UINT(sec, "rate_hz", UINT16_MAX, out->fusion_rate_hz, uint16_t);
    }
    GET_SECTION(obj, "sync", sec);
    if (sec) {
        GET_STR(sec, "mode", str);
        if (str) {
            out->sync_mode = strcmp(str, "emg_drdy_divisor") == 0 ? IMU_SYNC_EMG_DRDY_DIVISOR
                                                                   : IMU_SYNC_INT_DRIVEN;
        }
        GET_UINT(sec, "decimation_factor", 255u, out->emg_drdy_divisor, uint8_t);
    }
    GET_SECTION(obj, "dynamic_calibration", sec);
    if (sec) {
        GET_BOOL(sec, "accel", out->dynamic_calibration.accel);
        GET_BOOL(sec, "gyro", out->dynamic_calibration.gyro);
        GET_BOOL(sec, "mag", out->dynamic_calibration.mag);
    }
    GET_SECTION(obj, "reports", sec);
    if (sec) {
        if (!parse_imu_reports(sec, where, out)) {
            return false;
        }
        if (out->num_reports > 0) {
            out->profile = IMU_PROFILE_CUSTOM;
            unknown_profile = NULL;
        } else if (out->enabled) {
            /* An explicit list with nothing enabled used to fall through to
             * the profile expansion, silently re-enabling GRV + lin accel. */
            set_err("%s.reports: none enabled (set %s.enabled=false to turn the slot off)",
                    where, where);
            return false;
        }
    }
    if (unknown_profile != NULL && out->enabled) {
        set_err("%s.profile: '%s' has no firmware preset; send its reports explicitly", where,
                unknown_profile);
        return false;
    }
    if (out->sync_mode == IMU_SYNC_EMG_DRDY_DIVISOR && out->emg_drdy_divisor == 0 &&
        out->num_reports > 0 && emg_fs_hz > 0) {
        uint16_t imu_hz = out->reports[0].rate_hz;
        if (imu_hz > 0 && (emg_fs_hz % imu_hz) == 0) {
            uint32_t div = emg_fs_hz / imu_hz;
            if (div > 255u) {
                /* Used to be truncated to uint8_t: a wrong IMU rate, silently. */
                set_err("%s.sync: EMG/IMU divisor %lu exceeds 255", where, (unsigned long)div);
                return false;
            }
            out->emg_drdy_divisor = (uint8_t)div;
        }
    }
    return true;
}

/** @brief Internal helper: parse_trigger. */
static bool parse_trigger(cJSON *obj, trigger_config_t *out) {
    const char *where = "trigger";
    const char *str;
    GET_BOOL(obj, "enabled", out->enabled);
    GET_STR(obj, "profile", str);
    if (str) {
        out->profile = parse_trigger_profile(str);
    }
    uint32_t rate;
    bool rate_present;
    if (!json_uint(obj, "rate_hz", UINT16_MAX, where, &rate, &rate_present)) {
        return false;
    }
    if (rate_present) {
        out->rate_hz = (uint16_t)rate;
        out->profile = TRIGGER_PROFILE_CUSTOM;
    }
    cJSON *sec;
    GET_SECTION(obj, "pulse", sec);
    if (sec) {
        GET_UINT(sec, "width_us", UINT32_MAX, out->pulse_width_us, uint32_t);
        GET_STR(sec, "polarity", str);
        if (str) {
            if (strcmp(str, "imx296_fast") == 0) {
                out->polarity = 0;
            } else if (strcmp(str, "active_high") == 0) {
                out->polarity = 1;
            } else {
                set_err("trigger.pulse.polarity: '%.24s' is not imx296_fast|active_high", str);
                return false;
            }
        }
    }
    GET_SECTION(obj, "outputs", sec);
    if (sec) {
        GET_BOOL(sec, "csi1", out->outputs.csi1);
        GET_BOOL(sec, "csi2", out->outputs.csi2);
        GET_BOOL(sec, "scope", out->outputs.scope);
        GET_BOOL(sec, "sync_led", out->outputs.sync_led);
        /* PIN_TRIG feeds both 74LVC2G241 buffers with their enables tied on
         * (h06 §3.2): CSI1, CSI2 and the scope tap are always driven, and the
         * sync LED is a separate pin driven by `flash on`, never the trigger. */
        if (!fixed_value(out->outputs.csi1 && out->outputs.csi2 && out->outputs.scope, where,
                         "outputs", "csi1/csi2/scope are hard-wired on") ||
            !fixed_value(!out->outputs.sync_led, where, "outputs.sync_led",
                         "the trigger never drives the sync LED (use `flash on`)")) {
            return false;
        }
    }
    return true;
}

/** @brief Internal helper: a required top-level section that must be an object. */
static cJSON *required_section(cJSON *root, const char *key) {
    cJSON *item = cJSON_GetObjectItemCaseSensitive(root, key);
    if (item == NULL) {
        set_err("missing section '%s'", key);
        return NULL;
    }
    if (!cJSON_IsObject(item)) {
        set_err("section '%s' must be an object", key);
        return NULL;
    }
    return item;
}

bool config_json_parse(const char *json, jtsmm_mcu_config_t *cfg) {
    parse_err[0] = '\0';
    if (json == NULL || cfg == NULL) {
        set_err("no input");
        return false;
    }
    cJSON *root = cJSON_Parse(json);
    if (root == NULL) {
        set_err("malformed JSON");
        return false;
    }
    bool ok = cJSON_IsObject(root);
    if (!ok) {
        set_err("top level must be an object");
    }
    if (ok) {
        const char *where = "config";
        uint32_t schema;
        bool present;
        ok = json_uint(root, "schema_version", 255u, where, &schema, &present);
        if (ok && present) {
            cfg->schema_version = (uint8_t)schema;
        }
    }
    cJSON *platform = ok ? required_section(root, "platform") : NULL;
    cJSON *emg = platform ? required_section(root, "emg") : NULL;
    cJSON *imu_mta = emg ? required_section(root, "imu_mta") : NULL;
    cJSON *imu_cra = imu_mta ? required_section(root, "imu_cra") : NULL;
    cJSON *trigger = imu_cra ? required_section(root, "trigger") : NULL;
    ok = trigger != NULL && parse_platform(platform, &cfg->platform) && parse_emg(emg, &cfg->emg);
    uint16_t emg_fs = cfg->emg.sample_rate_sps;
    ok = ok && parse_imu_slot(imu_mta, "imu_mta", &cfg->imu_mta, emg_fs) &&
         parse_imu_slot(imu_cra, "imu_cra", &cfg->imu_cra, emg_fs) &&
         parse_trigger(trigger, &cfg->trigger);

    cJSON_Delete(root);
    return ok;
}

/** @brief Internal helper: append to the dump; fails on error or truncation. */
static bool append_str(char *out, size_t out_len, size_t *pos, const char *fmt, ...) {
    if (*pos >= out_len) {
        return false;
    }
    size_t room = out_len - *pos;
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(out + *pos, room, fmt, ap);
    va_end(ap);
    /* n >= room means vsnprintf truncated; the old check only caught n < 0, so
     * a too-small buffer "succeeded" with cut-off JSON that config save would
     * then have written to flash. */
    if (n < 0 || (size_t)n >= room) {
        return false;
    }
    *pos += (size_t)n;
    return true;
}

/*
 * The dumped JSON must round-trip through config_json_parse (used by
 * `config save` -> flash -> boot `config load`): every field the parser reads is
 * written back, in the exact schema the parser expects (ring sizes in BYTES,
 * IMU reports as a name-keyed object, the real profile strings). Fields that
 * were missing here (polarity, emg/imu/trigger enables, stream_channels,
 * ac_tone, probe_drl, ch7_readback, test_signal, sync_led, PIO/CPU clocks)
 * silently came back as defaults after a power-cycle. tests/test_config_json.c
 * compares every field of a non-default config after dump -> parse.
 */
static const char *emg_profile_str(emg_profile_t p) {
    switch (p) {
    case EMG_PROFILE_HIGH_RATE:
        return "high_rate";
    case EMG_PROFILE_BENCH:
        return "bench";
    case EMG_PROFILE_NOISE_CHARACTERIZATION:
        return "noise_characterization";
    case EMG_PROFILE_CUSTOM:
        return "custom";
    default:
        return "acquisition";
    }
}

/** @brief Internal helper: imu_profile_str. */
static const char *imu_profile_str(imu_profile_t p) {
    switch (p) {
    case IMU_PROFILE_FUSED_ONLY:
        return "fused_only";
    case IMU_PROFILE_RAW_ONLY:
        return "raw_only";
    case IMU_PROFILE_ALL:
        return "all";
    case IMU_PROFILE_CUSTOM:
        return "custom";
    case IMU_PROFILE_KINEMATICS_ACCEL:
        return "kinematics_accel";
    default:
        return "kinematics_default";
    }
}

/** @brief Internal helper: fusion_primary_str. */
static const char *fusion_primary_str(imu_fusion_primary_t p) {
    switch (p) {
    case IMU_FUSION_ROTATION_VECTOR:
        return "rotation_vector";
    case IMU_FUSION_GEOMAGNETIC_ROTATION_VECTOR:
        return "geomagnetic_rotation_vector";
    case IMU_FUSION_GYRO_INTEGRATED_RV:
        return "gyro_integrated_rotation_vector";
    default:
        return "game_rotation_vector";
    }
}

/** @brief Internal helper: trigger_profile_str. */
static const char *trigger_profile_str(trigger_profile_t p) {
    switch (p) {
    case TRIGGER_PROFILE_IMX296_HIGH_FPS:
        return "imx296_high_fps";
    case TRIGGER_PROFILE_BENCH_SCOPE:
        return "bench_scope";
    case TRIGGER_PROFILE_TRACK_A_LED:
        return "track_a_led";
    case TRIGGER_PROFILE_CUSTOM:
        return "custom";
    default:
        return "imx296_dual_csi";
    }
}

/** @brief Internal helper: JSON boolean literal. */
static const char *jb(bool v) {
    return v ? "true" : "false";
}

/** @brief Internal helper: dump_imu_slot. */
static bool dump_imu_slot(char *out, size_t out_len, size_t *pos, const imu_slot_config_t *s) {
    if (!append_str(out, out_len, pos,
                    "{\"enabled\":%s,\"profile\":\"%s\",\"fusion\":{\"primary\":\"%s\",\"rate_hz\":%u},"
                    "\"sync\":{\"mode\":\"%s\",\"decimation_factor\":%u},"
                    "\"dynamic_calibration\":{\"accel\":%s,\"gyro\":%s,\"mag\":%s}",
                    jb(s->enabled), imu_profile_str(s->profile), fusion_primary_str(s->fusion_primary),
                    s->fusion_rate_hz,
                    s->sync_mode == IMU_SYNC_EMG_DRDY_DIVISOR ? "emg_drdy_divisor" : "int_driven",
                    s->emg_drdy_divisor, jb(s->dynamic_calibration.accel),
                    jb(s->dynamic_calibration.gyro), jb(s->dynamic_calibration.mag))) {
        return false;
    }
    if (s->num_reports == 0) {
        /* Unresolved (disabled) slot: no list, so a reload resolves the profile. */
        return append_str(out, out_len, pos, "}");
    }
    if (!append_str(out, out_len, pos, ",\"reports\":{")) {
        return false;
    }
    for (uint8_t i = 0; i < s->num_reports; i++) {
        const bno085_report_entry_t *spec = bno085_report_find(s->reports[i].report_id);
        const char *name = spec && spec->name ? spec->name : "unknown";
        if (!append_str(out, out_len, pos, "%s\"%s\":{\"enabled\":true,\"rate_hz\":%u}", i ? "," : "", name,
                        s->reports[i].rate_hz)) {
            return false;
        }
    }
    return append_str(out, out_len, pos, "}}");
}

bool config_json_dump(const jtsmm_mcu_config_t *cfg, char *out, size_t out_len) {
    if (cfg == NULL || out == NULL || out_len < 32) {
        return false;
    }
    const platform_config_t *p = &cfg->platform;
    const emg_config_t *e = &cfg->emg;
    const trigger_config_t *t = &cfg->trigger;
    size_t pos = 0;
    bool ok =
        append_str(out, out_len, &pos,
                   "{\"schema_version\":%u,\"platform\":{"
                   "\"system\":{\"cpu_mhz\":%lu,\"xosc_hz\":%lu},"
                   "\"spi0\":{\"hz\":%lu,\"cfg_hz\":%lu},\"i2c\":{\"hz\":%lu},"
                   "\"pio\":{\"trigger\":{\"clock_hz\":%lu}},"
                   "\"buffers\":{\"emg_ring\":%lu,\"imu_ring\":%lu,\"event_ring\":%lu},"
                   "\"irq\":{\"emg\":%u,\"imu\":%u,\"trigger\":%u}},",
                   cfg->schema_version, (unsigned long)(p->cpu_hz / 1000000u),
                   (unsigned long)p->xosc_hz, (unsigned long)p->spi0_stream_hz,
                   (unsigned long)p->spi0_cfg_hz, (unsigned long)p->i2c_hz,
                   (unsigned long)p->pio_trigger_clock_hz,
                   (unsigned long)((uint32_t)p->emg_ring_slots * ringbuf_slot_bytes(RING_EMG)),
                   (unsigned long)((uint32_t)p->imu_ring_slots * ringbuf_slot_bytes(RING_IMU_MTA)),
                   (unsigned long)((uint32_t)p->event_ring_slots * ringbuf_slot_bytes(RING_EVENTS)),
                   p->irq_emg_priority, p->irq_imu_priority, p->irq_trigger_priority) &&
        append_str(out, out_len, &pos,
                   "\"emg\":{\"enabled\":%s,\"profile\":\"%s\",\"sample_rate_sps\":%u,"
                   "\"default_gain\":%u,\"stream_channels\":%u,\"pd_channels\":%u,"
                   "\"bias\":{\"enabled\":%s,\"channels\":%u,\"ch7_readback\":%s},"
                   "\"leadoff\":{\"enabled\":%s,\"during_recording\":%s,\"ac_tone\":%s,"
                   "\"probe_drl\":%s,\"current_na\":%u},"
                   "\"test_signal\":{\"enabled\":%s,\"amplitude_mv\":%u}},\"imu_mta\":",
                   jb(e->enabled), emg_profile_str(e->profile), e->sample_rate_sps, e->default_gain,
                   e->stream_channels, e->pd_channel_mask, jb(e->bias.enabled), e->bias.channel_mask,
                   jb(e->bias.ch7_readback), jb(e->leadoff.enabled), jb(e->leadoff.during_recording),
                   jb(e->leadoff.ac_tone), jb(e->leadoff.probe_drl), e->leadoff.current_na,
                   jb(e->test_signal.enabled), e->test_signal.amplitude_mv) &&
        dump_imu_slot(out, out_len, &pos, &cfg->imu_mta) &&
        append_str(out, out_len, &pos, ",\"imu_cra\":") &&
        dump_imu_slot(out, out_len, &pos, &cfg->imu_cra) &&
        append_str(out, out_len, &pos,
                   ",\"trigger\":{\"enabled\":%s,\"profile\":\"%s\",\"rate_hz\":%u,"
                   "\"pulse\":{\"width_us\":%lu,\"polarity\":\"%s\"},"
                   "\"outputs\":{\"csi1\":%s,\"csi2\":%s,\"scope\":%s,\"sync_led\":%s}}}",
                   jb(t->enabled), trigger_profile_str(t->profile), t->rate_hz,
                   (unsigned long)t->pulse_width_us, t->polarity == 1 ? "active_high" : "imx296_fast",
                   jb(t->outputs.csi1), jb(t->outputs.csi2), jb(t->outputs.scope),
                   jb(t->outputs.sync_led));
    return ok;
}
