// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file cmd_parser.c
 * @brief CDC text command line parser.
 */

#include "jtsmm/runtime/cmd_parser.h"

#include "jtsmm/domain/emg_config.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/domain/trigger_config.h"
#include "jtsmm/app/comms_core.h"
#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/drivers/imu_bus.h"
#include "jtsmm/drivers/led_flash_pio.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/drivers/trigger_timing.h"
#include "jtsmm/app/fault.h"
#include "jtsmm/app/selftest.h"
#include "jtsmm/config.h"
#include "jtsmm/records.h"
#include "jtsmm/platform/platform_config.h"
#include "jtsmm/platform/spi_dma.h"
#include "jtsmm/runtime/config_apply.h"
#include "jtsmm/runtime/config_flash.h"
#include "jtsmm/runtime/config_json.h"
#include "jtsmm/runtime/host_link.h"
#include "jtsmm/runtime/timebase.h"
#include "jtsmm/version.h"
#include "pico/stdlib.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FW_VERSION JTSMM_FW_VERSION "-prod"

static char line_buf[JTSMM_CMD_LINE_MAX];
static size_t line_len;
static bool line_overflow;
/* config dump / config save output; static because the core-0 stack is small. */
static char dump_buf[JTSMM_CMD_LINE_MAX];

/** @brief Internal helper: cmd_line_is_echo_artifact. */
static bool cmd_line_is_echo_artifact(const char *line) {
    if (line[0] == '\0') {
        return true;
    }
    if (strncmp(line, "unknown:", 8) == 0) {
        return true;
    }
    if (strncmp(line, "panel test", 10) == 0) {
        return true;
    }
    if (line[0] == ' ') {
        return true;
    }
    return false;
}

/** @brief Internal helper: text after a command word, or "" when there is none. */
static const char *cmd_arg(const char *line, size_t name_len) {
    const char *arg = line + name_len;
    while (*arg == ' ' || *arg == '\t') {
        arg++;
    }
    return arg;
}

/* What a bench command collides with (cmd_bus_busy). */
#define BUS_EMG  (1u << 0) /* SPI0 / ADS1299 while the EMG stream runs */
#define BUS_IMU  (1u << 1) /* IMU I2C while a slot streams or recovers */
#define BUS_TRIG (1u << 2) /* the trigger PIO while pulses run */

/**
 * @brief Internal helper: refuse bus-touching bench commands while a stream owns the bus.
 *
 * @details While the EMG stream runs, SPI0 (ADS1299) belongs to the sensor
 *          core, and while an IMU slot streams its I2C bus does; there is no
 *          mutex between the cores. A register read issued from *this* core
 *          retimes SPI0 down to the slow config baud and toggles CS while the
 *          sensor core has a DMA transfer in flight, which slips the RDATA byte
 *          stream -- an EMG trace full of hash that looks like analog noise.
 *          selftest additionally probes both IMUs over I2C and re-initialises
 *          the trigger PIO, and ads_drdy_watch busy-waits 500 ms on this core
 *          (starving the ring drain). Collisions with the *idle* lead-off poll
 *          are handled inside the ADS1299 driver (SPI0 claim handshake), and
 *          imu_status skips live reads on busy slots itself.
 *
 *          Default is to refuse. `<cmd> force` preserves the old behaviour for
 *          deliberate bench work, but announces the risk on both CDC and the
 *          record stream so a forced command is never mistaken for a clean one.
 *
 * @param conflicts BUS_* bits this command collides with.
 * @return true if the caller must abort the command.
 */
static bool cmd_bus_busy(const char *name, const char *arg, uint32_t conflicts) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    const char *what = NULL;
    const char *stop = NULL;
    if ((conflicts & BUS_EMG) && ads1299_drdy_streaming()) {
        what = "EMG stream active; SPI0 belongs to the sensor core";
        stop = "ads_stop";
    } else if ((conflicts & BUS_IMU) &&
               (cfg->stream_imu_mta || cfg->stream_imu_cra || imu_recover_busy())) {
        what = "IMU stream/recovery active; I2C belongs to the sensor core";
        stop = "imu_stream <slot> off";
    } else if ((conflicts & BUS_TRIG) && trigger_pio_running()) {
        what = "trigger running; the command re-initialises the trigger PIO";
        stop = "trig_stop";
    }
    if (what == NULL) {
        return false;
    }
    if (arg != NULL && strstr(arg, "force") != NULL) {
        printf("%s: FORCED (%s) - expect corrupted data\r\n", name, what);
        host_link_send_log("bench command forced during a stream (data may corrupt)");
        return false;
    }
    printf("%s REFUSED: %s.\r\n", name, what);
    printf("  run '%s' first, or '%s force' to override on the bench.\r\n", stop, name);
    host_link_send_ack(false);
    return true;
}

/**
 * @brief Internal helper: why a flash save/erase must not run now (NULL = ok).
 *
 * @details A sector erase stalls flash for both cores with IRQs off for up to a
 *          few hundred ms: the trigger PIO IRQ, USB and the sensor core would all
 *          miss their deadlines mid-recording. Refuse instead of degrading data.
 */
static const char *cmd_flash_blocked(void) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    if (config_apply_busy()) {
        return "config apply running";
    }
    if (ads1299_drdy_streaming() || cfg->stream_emg) {
        return "EMG streaming; ads_stop first";
    }
    if (cfg->stream_imu_mta || cfg->stream_imu_cra) {
        return "IMU streaming; imu_stream off first";
    }
    if (trigger_pio_running()) {
        return "trigger running; trig_stop first";
    }
    return NULL;
}

/**
 * @brief Internal helper: refuse a stream/bus command while a config apply runs.
 *
 * @details The apply runs one step per comms loop with the sensor core parked
 *          and the hardware half reprogrammed. A stream started in between is
 *          undone by a later step without any notice: ads_start after VALIDATE
 *          is halted by CFG_EMG's ads1299_init (drdy_streaming stays true, no
 *          data), trig_start is killed by CFG_TRIGGER's trigger_pio_init, and
 *          `imu_stream on` is cleared again. imu_savecal / selftest would talk
 *          to buses the apply is re-initialising. capture_session.c refuses the
 *          panel path the same way. Stop commands stay allowed (the apply stops
 *          everything anyway, and the host's K6 retry sends them).
 *
 * @return true if the caller must abort the command (`<name> BUSY` + NAK sent).
 */
static bool cmd_apply_busy(const char *name) {
    if (!config_apply_busy()) {
        return false;
    }
    printf("%s BUSY\r\n", name);
    host_link_send_ack(false);
    return true;
}

/** @brief Internal helper: print EMG framing guard state + counters. */
static void cmd_emg_guard_print(void) {
    printf("emg_guard: drop=%s resync=%s streaming=%d\r\n",
           ads1299_misalign_guard_drop() ? "on" : "off",
           ads1299_misalign_guard_resync() ? "on" : "off",
           ads1299_drdy_streaming() ? 1 : 0);
    printf("  misaligned=%lu dropped=%lu worst_streak=%lu last_status0=0x%02X\r\n",
           (unsigned long)ads1299_frame_misalign_count(),
           (unsigned long)emg_stream_misalign_drop_count(),
           (unsigned long)ads1299_frame_misalign_max_streak(),
           (unsigned)ads1299_frame_misalign_last_status());
    printf("  resyncs=%lu giveups=%lu spi_wedges=%lu spi_resyncs=%lu\r\n",
           (unsigned long)ads1299_frame_resync_count(),
           (unsigned long)ads1299_frame_resync_giveup_count(),
           (unsigned long)spi_dma_wedge_count(), (unsigned long)spi_dma_resync_count());
}

/** @brief Internal helper: dispatch_line. */
static void dispatch_line(char *line) {
    while (*line == ' ' || *line == '\t') {
        line++;
    }
    if (cmd_line_is_echo_artifact(line)) {
        return;
    }

    if (strcmp(line, "info") == 0) {
        printf("jtsmm %s (RP2350 production)\r\n", FW_VERSION);
        printf("  last reset: %s\r\n", fault_reset_reason());
    } else if (strcmp(line, "config init") == 0) {
        /* Same async path as config apply (streams stopped, sensor core parked,
         * USB serviced between steps); replies "config init OK|FAIL" when done. */
        if (!config_apply_start_defaults()) {
            printf("config apply BUSY\r\n");
        }
    } else if (strncmp(line, "config apply", 12) == 0) {
        const char *json = line + 12;
        while (*json == ' ') {
            json++;
        }
        /* The host matches this exact "config apply BUSY" reply (contract K6). */
        if (config_apply_busy()) {
            printf("config apply BUSY\r\n");
        } else if (json[0] == '\0') {
            printf("config apply FAIL (no JSON; use 'config init' for defaults)\r\n");
            host_link_send_ack(false);
        } else {
            (void)config_apply_start_json(json);
        }
    } else if (strcmp(line, "config dump") == 0) {
        if (config_json_dump(jtsmm_config_active(), dump_buf, sizeof(dump_buf))) {
            printf("%s\r\n", dump_buf);
        } else {
            printf("config dump FAIL\r\n");
        }
    } else if (strcmp(line, "config save") == 0) {
        const char *why = cmd_flash_blocked();
        bool ok = false;
        if (why == NULL) {
            if (!config_json_dump(jtsmm_config_active(), dump_buf, sizeof(dump_buf))) {
                why = "dump did not fit";
            } else if (!config_flash_save(dump_buf)) {
                why = "flash write/verify failed";
            } else {
                ok = true;
            }
        }
        host_link_send_ack(ok);
        if (ok) {
            printf("config save OK\r\n");
        } else {
            printf("config save FAIL (%s)\r\n", why);
        }
    } else if (strcmp(line, "config load") == 0) {
        static char load_buf[JTSMM_CMD_LINE_MAX];
        if (config_apply_busy()) {
            printf("config apply BUSY\r\n");
        } else if (!config_flash_load(load_buf, sizeof(load_buf))) {
            printf("config load FAIL (no valid saved config)\r\n");
            host_link_send_ack(false);
        } else if (config_apply_start_json(load_buf)) {
            printf("config load started\r\n");
        } else {
            printf("config load FAIL (apply did not start)\r\n");
            host_link_send_ack(false);
        }
    } else if (strcmp(line, "config erase") == 0) {
        const char *why = cmd_flash_blocked();
        bool ok = why == NULL && config_flash_erase();
        host_link_send_ack(ok);
        if (ok) {
            printf("config erase OK\r\n");
        } else {
            printf("config erase FAIL (%s)\r\n", why ? why : "flash erase failed");
        }
    } else if (strcmp(line, "stats") == 0) {
        comms_core_send_stats();
    } else if (strcmp(line, "platform_status") == 0) {
        platform_config_status();
    } else if (strncmp(line, "ads_status", 10) == 0 && (line[10] == '\0' || line[10] == ' ')) {
        if (cmd_bus_busy("ads_status", cmd_arg(line, 10), BUS_EMG)) {
            return;
        }
        ads1299_status();
    } else if (strncmp(line, "ads_regs", 8) == 0 && (line[8] == '\0' || line[8] == ' ')) {
        /* Channel + bias registers (CH7/CH8SET, BIAS_SENSP/N) for checking that
         * --bias-drive / --bias-channels / --active-channels reached the chip. */
        if (cmd_bus_busy("ads_regs", cmd_arg(line, 8), BUS_EMG)) {
            return;
        }
        ads1299_regs();
    } else if (strcmp(line, "ads_start") == 0 && cmd_apply_busy("ads_start")) {
        return;
    } else if (strcmp(line, "ads_start") == 0 && !jtsmm_config_active()->emg.enabled) {
        printf("ads_start FAIL (emg.enabled is false in the applied config)\r\n");
        host_link_send_ack(false);
    } else if (strcmp(line, "ads_start") == 0) {
        /* Per-session drop counters, as documented: only a fresh EMG start
         * resets them, not a repeated ads_start mid-stream. */
        bool fresh = !ads1299_drdy_streaming();
        if (!ads1299_init_ok() || !emg_stream_start()) {
            /* The ADS1299 init failed (or never ran): a stream "started" now
             * would only produce garbage or nothing. NAK instead. */
            printf("ads_start FAIL (ADS1299 not configured; see config apply / selftest)\r\n");
            host_link_send_ack(false);
            return;
        }
        if (fresh) {
            host_link_reset_tx_stats();
        }
        jtsmm_config_mutable()->stream_emg = true;
        panel_set_state(PANEL_STATE_STREAMING);
        host_link_send_ack(true);
    } else if (strncmp(line, "ads_drdy_watch", 14) == 0 && (line[14] == '\0' || line[14] == ' ')) {
        /* Also a 500 ms busy loop on this core: even without the bus conflict it
         * starves host_link_drain() long enough to overflow the EMG ring. */
        if (cmd_bus_busy("ads_drdy_watch", cmd_arg(line, 14), BUS_EMG | BUS_IMU)) {
            return;
        }
        ads1299_drdy_watch();
    } else if (strncmp(line, "emg_guard", 9) == 0 && (line[9] == '\0' || line[9] == ' ')) {
        /* Runtime A/B switch for the framing guard so a slip can be observed
         * with the guard off and the fix confirmed with it on, without a
         * reflash. Both knobs default to on at boot. */
        const char *arg = cmd_arg(line, 9);
        if (strncmp(arg, "drop ", 5) == 0) {
            ads1299_set_misalign_guard(strstr(arg + 5, "on") != NULL,
                                       ads1299_misalign_guard_resync());
        } else if (strncmp(arg, "resync ", 7) == 0) {
            ads1299_set_misalign_guard(ads1299_misalign_guard_drop(),
                                       strstr(arg + 7, "on") != NULL);
        } else if (arg[0] != '\0') {
            printf("usage: emg_guard [drop on|off] [resync on|off]\r\n");
        }
        cmd_emg_guard_print();
    } else if (strcmp(line, "ads_stop") == 0) {
        emg_stream_stop();
        jtsmm_config_mutable()->stream_emg = false;
        panel_update_streaming_state();
        host_link_send_ack(true);
    } else if (strncmp(line, "imu_stream", 10) == 0) {
        const char *arg = line + 10;
        while (*arg == ' ') {
            arg++;
        }
        bool on = (strstr(arg, "on") != NULL);
        bool cra = (strstr(arg, "cra") != NULL);
        if (on && cmd_apply_busy("imu_stream")) {
            return;
        }
        imu_slot_t slot = cra ? IMU_SLOT_CRA : IMU_SLOT_MTA;
        bool *flag = cra ? &jtsmm_config_mutable()->stream_imu_cra
                         : &jtsmm_config_mutable()->stream_imu_mta;
        bool ok = true;
        if (on) {
            /* false = slot disabled in the applied config or bus not taken
             * over (imu_config prints why): do not report the stream as on. */
            ok = imu_stream_start_slot(slot);
            *flag = ok;
        } else {
            imu_stream_stop_slot(slot);
            *flag = false;
        }
        panel_update_streaming_state();
        host_link_send_ack(ok);
        if (!ok) {
            printf("imu_stream %s on FAIL\r\n", cra ? "cra" : "mta");
        }
    } else if (strncmp(line, "trig_start", 10) == 0 && (line[10] == '\0' || line[10] == ' ')) {
        /* Optional: trig_start [rate_hz] [pulse_width_us]. Bare form uses active config. */
        if (cmd_apply_busy("trig_start")) {
            return;
        }
        const trigger_config_t *t = &jtsmm_config_active()->trigger;
        if (!t->enabled) {
            printf("trig_start FAIL (trigger.enabled is false in the applied config)\r\n");
            host_link_send_ack(false);
            return;
        }
        unsigned rate = t->rate_hz;
        unsigned width = t->pulse_width_us;
        if (line[10] == ' ') {
            int n = sscanf(line + 10, "%u %u", &rate, &width);
            if (n < 1) {
                printf("trig_start FAIL (usage: trig_start [rate_hz] [pulse_width_us])\r\n");
                host_link_send_ack(false);
                return;
            }
            if (n == 1) {
                width = t->pulse_width_us;
            }
        }
        /* Validate the full 32-bit values: the old (uint16_t) cast let 65566
         * through as 30 Hz, and a width >= period was halved silently. */
        const char *why = trigger_timing_check(rate, width);
        if (why != NULL) {
            printf("trig_start FAIL (%s; rate=%u width=%u)\r\n", why, rate, width);
            host_link_send_ack(false);
            return;
        }
        bool ok = trigger_pio_start((uint16_t)rate, width, t->polarity == 0);
        panel_update_streaming_state();
        host_link_send_ack(ok);
        if (!ok) {
            printf("trig_start FAIL\r\n");
        }
    } else if (strncmp(line, "pps", 3) == 0 && (line[3] == '\0' || line[3] == ' ')) {
        /*
         * Clock handshake: the host sends `pps <id>` and notes its own clock
         * before and after; we reply with REC_PPS_REPLY carrying <id> and the
         * t_master_us at which the command was parsed. The host fits
         * pi_clock -> t_master from a handful of these (sync/clock_align.py),
         * which is what lets it cross-check camera SensorTimestamps against
         * the trigger it paired them with. No CDC echo: the reply is a record.
         */
        const char *arg = cmd_arg(line, 3);
        unsigned long long id = 0;
        if (arg[0] == '\0' || sscanf(arg, "%llu", &id) != 1) {
            printf("usage: pps <id>\r\n");
            host_link_send_ack(false);
            return;
        }
        uint64_t t = t_master_us();
        uint8_t payload[8];
        for (int i = 0; i < 8; i++) {
            payload[i] = (uint8_t)((id >> (8 * i)) & 0xFFu);
        }
        host_link_send_record(REC_PPS_REPLY, t, payload, sizeof(payload));
    } else if (strcmp(line, "trig_stop") == 0) {
        trigger_pio_stop();
        panel_update_streaming_state();
        host_link_send_ack(true);
    } else if (strcmp(line, "trig_status") == 0) {
        trigger_pio_print_status();
    } else if (strncmp(line, "trig_stamp", 10) == 0 && (line[10] == '\0' || line[10] == ' ')) {
        /* Runtime A/B for TRIGGER_EVENT timing: `pio` (default) stamps in the
         * PIO IRQ at the pulse edge; `timer` is the legacy software timer.
         * Takes effect at the next trig_start; back to `pio` on power-cycle. */
        const char *arg = cmd_arg(line, 10);
        if (strncmp(arg, "pio", 3) == 0) {
            trigger_pio_set_stamp_mode(TRIG_STAMP_MODE_PIO_IRQ);
        } else if (strncmp(arg, "timer", 5) == 0) {
            trigger_pio_set_stamp_mode(TRIG_STAMP_MODE_TIMER);
        } else if (arg[0] != '\0') {
            printf("usage: trig_stamp [pio|timer]\r\n");
        }
        if (arg[0] != '\0' && trigger_pio_running()) {
            printf("note: applies at the next trig_start (currently running)\r\n");
        }
        trigger_pio_print_status();
    } else if (strncmp(line, "trig_comp", 9) == 0 && (line[9] == '\0' || line[9] == ' ')) {
        /* PIO cycle-overhead compensation (trigger_timing.c). `off` reproduces
         * the legacy pin period (period_us + 5) for a scope comparison. */
        const char *arg = cmd_arg(line, 9);
        if (strncmp(arg, "on", 2) == 0) {
            trigger_pio_set_compensation(true);
        } else if (strncmp(arg, "off", 3) == 0) {
            trigger_pio_set_compensation(false);
        } else if (arg[0] != '\0') {
            printf("usage: trig_comp [on|off]\r\n");
        }
        if (arg[0] != '\0' && trigger_pio_running()) {
            printf("note: applies at the next trig_start (currently running)\r\n");
        }
        trigger_pio_print_status();
    } else if (strcmp(line, "flash off") == 0) {
        led_flash_pio_stop();
        host_link_send_ack(true);
    } else if (strcmp(line, "flash once") == 0) {
        bool ok = led_flash_pio_once(200);
        host_link_send_ack(ok);
    } else if (strncmp(line, "flash on", 8) == 0) {
        unsigned hz = 1;
        unsigned width = 200;
        sscanf(line + 8, "%u %u", &hz, &width);
        if (hz == 0) {
            hz = 1;
        }
        if (!led_flash_pio_init(jtsmm_config_active()->platform.pio_trigger_clock_hz)) {
            printf("flash on FAIL (pio init)\r\n");
        } else {
            bool ok = led_flash_pio_start((uint16_t)hz, width);
            host_link_send_ack(ok);
            printf("flash on %s\r\n", ok ? "OK" : "FAIL");
        }
    } else if (strncmp(line, "imu_status", 10) == 0 && (line[10] == '\0' || line[10] == ' ')) {
        /* Reads an SHTP packet per idle slot (imu_config skips streaming or
         * recovering slots itself); the EMG refusal is kept as documented. */
        if (cmd_bus_busy("imu_status", cmd_arg(line, 10), BUS_EMG)) {
            return;
        }
        imu_config_status();
    } else if (strncmp(line, "imu_int_watch", 13) == 0 && (line[13] == '\0' || line[13] == ' ')) {
        /* Bench check of the INT edge IRQ without an IMU (tap INT to GND).
         * Busy-waits on this core like ads_drdy_watch, and would count the
         * streaming slot's own INT edges: refused under any stream. */
        const char *arg = cmd_arg(line, 13);
        if (cmd_apply_busy("imu_int_watch") ||
            cmd_bus_busy("imu_int_watch", arg, BUS_EMG | BUS_IMU)) {
            return;
        }
        unsigned long ms = strtoul(arg, NULL, 10);
        if (ms == 0) {
            ms = 1000;
        } else if (ms > 10000) {
            ms = 10000;
        }
        fault_watchdog_hold();
        imu_int_watch((uint32_t)ms);
        fault_watchdog_release();
    } else if (strcmp(line, "imu_savecal") == 0) {
        if (cmd_apply_busy("imu_savecal")) {
            return;
        }
        fault_watchdog_hold();
        imu_config_save_calibration();
        fault_watchdog_release();
        host_link_send_ack(true);
    } else if (strncmp(line, "selftest", 8) == 0 && (line[8] == '\0' || line[8] == ' ')) {
        /* selftest_run() calls ads1299_init(), i.e. it reprograms every ADS
         * register from this core. Doing that under an active stream does not
         * just slip one frame, it reconfigures the front end mid-recording. */
        if (cmd_apply_busy("selftest") ||
            cmd_bus_busy("selftest", cmd_arg(line, 8), BUS_EMG | BUS_IMU | BUS_TRIG)) {
            return;
        }
        fault_watchdog_hold();
        bool ok = selftest_run();
        fault_watchdog_release();
        host_link_send_ack(ok);
    } else if (strncmp(line, "led rgb", 7) == 0) {
        unsigned r = 0, g = 0, b = 0;
        sscanf(line + 7, "%u %u %u", &r, &g, &b);
        panel_set_rgb((uint8_t)r, (uint8_t)g, (uint8_t)b);
    } else if (strcmp(line, "led panel") == 0) {
        cmd_parser_flush_rx();
        if (panel_panel_test_busy()) {
            printf("panel test BUSY\r\n");
        } else {
            panel_panel_test_start();
            printf("panel test started (watch panel LEDs, ~4 s)\r\n");
            cmd_parser_flush_rx();
        }
    } else {
        printf("unknown: %s\r\n", line);
    }
}

void cmd_parser_init(void) {
    line_len = 0;
    line_overflow = false;
}

void cmd_parser_flush_rx(void) {
    int c;
    while ((c = getchar_timeout_us(0)) != PICO_ERROR_TIMEOUT) {
        (void)c;
    }
    line_len = 0;
    line_overflow = false;
    line_buf[0] = '\0';
}

void cmd_parser_poll(void) {
    if (panel_panel_test_rx_quiet()) {
        cmd_parser_flush_rx();
        return;
    }
    int c = getchar_timeout_us(0);
    if (c == PICO_ERROR_TIMEOUT) {
        return;
    }
    if (c == '\r' || c == '\n') {
        if (line_overflow) {
            /* The only command that long is a config apply; refuse it whole. */
            printf("config apply FAIL (line too long)\r\n");
            host_link_send_ack(false);
        } else {
            line_buf[line_len] = '\0';
            dispatch_line(line_buf);
        }
        line_len = 0;
        line_overflow = false;
    } else if (line_overflow) {
        /* Discard to the end of the line. Resetting line_len here used to turn
         * the tail of a > 4095-byte line into a second "command". */
    } else if (c >= 0 && line_len + 1 < sizeof(line_buf)) {
        line_buf[line_len++] = (char)c;
    } else if (c >= 0) {
        line_overflow = true;
    }
}
