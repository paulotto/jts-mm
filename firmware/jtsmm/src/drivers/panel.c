// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file panel.c
 * @brief Panel LEDs, RGB, buttons, faults.
 */

#include "jtsmm/drivers/panel.h"

#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/config.h"
#include "jtsmm/pins.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/capture_session.h"
#include "jtsmm/runtime/ringbuf.h"
#include "jtsmm/runtime/host_link.h"
#include "jtsmm/runtime/timebase.h"

#include "hardware/gpio.h"
#include "hardware/pio.h"
#include "pico/stdlib.h"
#include "ws2812.pio.h"
#include <stdio.h>

static PIO ws2812_pio;
static uint ws2812_sm;
static uint ws2812_offset;
static bool ws2812_ready;
static panel_state_t state = PANEL_STATE_IDLE;
static bool disco_active;
static absolute_time_t disco_end;
static absolute_time_t disco_next_step;
static uint8_t disco_step;

/*
 * Cross-core rule: only the comms core (core 0) touches panel GPIOs, the
 * WS2812 and fault_mask. The sensor core reports activity through single-word
 * stores (panel_imu_mark_active, panel_set_emg_leadoff), which panel_status_tick
 * consumes. The old code cleared fault bits (a non-atomic read-modify-write that
 * could lose a bit set by core 0) and pushed a WS2812 word from core 1 for every
 * IMU packet.
 */
static volatile uint32_t fault_mask;
static bool emg_leadoff_lit;
static volatile bool emg_leadoff_req;
static volatile uint32_t imu_last_active_us[2]; /* time_us_32(); 0 = never */
static bool ws2812_last_valid;
static uint32_t ws2812_last_word;

static struct {
    bool active;
    uint8_t step;
    absolute_time_t next;
    absolute_time_t rx_quiet_until;
} panel_test;

#define PANEL_TEST_GPIO_STEPS 6u
#define PANEL_TEST_RGB_STEPS  4u
#define PANEL_TEST_STEPS      (PANEL_TEST_GPIO_STEPS + PANEL_TEST_RGB_STEPS)

#define FAULT_WATCHDOG_EMG  (1u << 0)
#define FAULT_HARD          (1u << 1)
#define FAULT_IMU_MTA       (1u << 2)
#define FAULT_IMU_CRA       (1u << 3)

#define IMU_LED_TIMEOUT_US  2000000ull

/* WS2812 on +5V_IN — scale all USER RGB requests (~25% brightness). */
#define USER_LED_BRIGHTNESS_SHIFT 2u

static const uint8_t panel_led_pins[] = {
    PIN_LED_HB, PIN_LED_FAULT, PIN_LED_CAP, PIN_LED_MTA,
    PIN_LED_CRA, PIN_LED_DP, PIN_LED_EMG_LO,
};

/* Never PIN_TRIG: it is the camera XTRIG line, so "disco" would fire both
 * cameras (and emit no TRIGGER_EVENT for those frames). */
static const uint8_t disco_panel_pins[] = {
    PIN_LED_HB, PIN_LED_FAULT, PIN_LED_CAP, PIN_LED_MTA,
    PIN_LED_CRA, PIN_LED_DP, PIN_LED_EMG_LO,
};

/** @brief Internal helper: clamp_u8. */
static uint8_t clamp_u8(int v) {
    if (v < 0) {
        return 0;
    }
    if (v > 255) {
        return 255;
    }
    return (uint8_t)v;
}

/** @brief Internal helper: panel_apply_fault_led. */
static void panel_apply_fault_led(void) {
    gpio_put(PIN_LED_FAULT, fault_mask != 0);
}

void panel_fault_set(uint32_t bits) {
    fault_mask |= bits;
    panel_apply_fault_led();
    panel_refresh_user_rgb();
}

void panel_fault_clear(uint32_t bits) {
    fault_mask &= ~bits;
    panel_apply_fault_led();
    panel_refresh_user_rgb();
}

/** @brief Internal helper: panel_leds_all. */
static void panel_leds_all(bool on) {
    for (size_t i = 0; i < sizeof(disco_panel_pins); i++) {
        gpio_put(disco_panel_pins[i], on);
    }
}

/** @brief Internal helper: disco_rgb. */
static void disco_rgb(uint8_t step, uint8_t *r, uint8_t *g, uint8_t *b) {
    uint8_t wheel = (uint8_t)(step % 6);
    switch (wheel) {
    case 0: *r = 64; *g = 0;  *b = 0;  break;
    case 1: *r = 64; *g = 32; *b = 0;  break;
    case 2: *r = 0;  *g = 64; *b = 0;  break;
    case 3: *r = 0;  *g = 64; *b = 64; break;
    case 4: *r = 0;  *g = 0;  *b = 64; break;
    default: *r = 64; *g = 0;  *b = 64; break;
    }
}

/** @brief Internal helper: panel_disco_start. */
static void panel_disco_start(void) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    if (cfg->stream_emg || cfg->stream_imu_mta || cfg->stream_imu_cra || trigger_pio_running()) {
        return;
    }
    disco_active = true;
    disco_step = 0;
    disco_end = make_timeout_time_ms(15000);
    disco_next_step = get_absolute_time();
}

bool panel_disco_active(void) {
    return disco_active;
}

void panel_disco_tick(void) {
    if (!disco_active) {
        return;
    }
    if (time_reached(disco_end)) {
        disco_active = false;
        panel_leds_all(false);
        panel_set_state(PANEL_STATE_IDLE);
        return;
    }
    if (!time_reached(disco_next_step)) {
        return;
    }
    disco_next_step = delayed_by_ms(get_absolute_time(), 100);
    const size_t n = sizeof(disco_panel_pins) / sizeof(disco_panel_pins[0]);
    panel_leds_all(false);
    if ((disco_step & 7u) == 0u) {
        panel_leds_all(true);
    } else {
        gpio_put(disco_panel_pins[disco_step % n], 1);
        gpio_put(disco_panel_pins[(disco_step + n / 2u) % n], 1);
    }
    uint8_t r, g, b;
    disco_rgb(disco_step, &r, &g, &b);
    panel_set_rgb(r, g, b);
    disco_step++;
}

/** @brief Internal helper: panel_leds_init. */
static void panel_leds_init(void) {
    /* PIN_TRIG is the camera XTRIG line (+ TRIG LED): hold it at the idle
     * (inactive) level of the configured polarity from boot on, not LOW. */
    trigger_pio_set_idle_polarity(jtsmm_config_active()->trigger.polarity == 0);
    for (size_t i = 0; i < sizeof(panel_led_pins); i++) {
        gpio_init(panel_led_pins[i]);
        gpio_set_dir(panel_led_pins[i], GPIO_OUT);
        gpio_put(panel_led_pins[i], 0);
    }
}

/** @brief Internal helper: ws2812_init. */
static void ws2812_init(void) {
    ws2812_ready = pio_claim_free_sm_and_add_program_for_gpio_range(
        &ws2812_program, &ws2812_pio, &ws2812_sm, &ws2812_offset, PIN_LED_USER, 1, true);
    if (ws2812_ready) {
        ws2812_program_init(ws2812_pio, ws2812_sm, ws2812_offset, PIN_LED_USER, 800000, false);
    }
}

void panel_init(void) {
    panel_leds_init();
    ws2812_init();
    gpio_init(PIN_SW_USER1);
    gpio_set_dir(PIN_SW_USER1, GPIO_IN);
    gpio_pull_up(PIN_SW_USER1);
    gpio_init(PIN_SW_USER2);
    gpio_set_dir(PIN_SW_USER2, GPIO_IN);
    gpio_pull_up(PIN_SW_USER2);
    panel_set_state(PANEL_STATE_IDLE);
}

void panel_set_rgb(uint8_t r, uint8_t g, uint8_t b) {
    panel_set_rgb_ex(r, g, b, 0);
}

void panel_set_rgb_ex(uint8_t r, uint8_t g, uint8_t b, uint8_t extra_brightness_shift) {
    if (!ws2812_ready) {
        return;
    }
    uint8_t shift = (uint8_t)(USER_LED_BRIGHTNESS_SHIFT + extra_brightness_shift);
    r = (uint8_t)(r >> shift);
    g = (uint8_t)(g >> shift);
    b = (uint8_t)(b >> shift);
    uint32_t word = (uint32_t)g << 24 | (uint32_t)r << 16 | (uint32_t)b << 8;
    /* The status tick refreshes every 100 ms; only push an actual change. */
    if (ws2812_last_valid && word == ws2812_last_word) {
        return;
    }
    ws2812_last_word = word;
    ws2812_last_valid = true;
    pio_sm_put_blocking(ws2812_pio, ws2812_sm, word);
}

void panel_refresh_user_rgb(void) {
    if (disco_active) {
        return;
    }
    if (fault_mask != 0) {
        panel_set_rgb(255, 0, 0);
        return;
    }

    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    bool emg = cfg->stream_emg;
    bool mta = cfg->stream_imu_mta;
    bool cra = cfg->stream_imu_cra;
    bool trig = trigger_pio_running();

    if (emg || mta || cra || trig) {
        int r = 0;
        int g = 0;
        int b = 0;
        uint8_t modes = 0;
        if (emg) {
            g += 96;
            b += 64;
            modes++;
        }
        if (mta) {
            r += 32;
            b += 160;
            modes++;
        }
        if (cra) {
            r += 32;
            b += 160;
            modes++;
        }
        if (trig) {
            r += 128;
            g += 64;
            modes++;
        }
        if (emg_leadoff_lit) {
            r += 160;
            g += 96;
        }
        uint8_t extra_shift = (modes >= 3u) ? 2u : (modes >= 2u) ? 1u : 0u;
        panel_set_rgb_ex(clamp_u8(r), clamp_u8(g), clamp_u8(b), extra_shift);
        return;
    }

    switch (state) {
    case PANEL_STATE_ARMED:
        panel_set_rgb(255, 128, 0);
        break;
    case PANEL_STATE_STREAMING:
        panel_set_rgb(0, 48, 255);
        break;
    case PANEL_STATE_FAULT:
        panel_set_rgb(255, 0, 0);
        break;
    case PANEL_STATE_IDLE:
    default:
        panel_set_rgb(0, 64, 0);
        break;
    }
}

void panel_set_state(panel_state_t s) {
    if (disco_active) {
        return;
    }
    state = s;
    panel_refresh_user_rgb();
}

void panel_heartbeat_tick(void) {
    if (disco_active) {
        return;
    }
    static bool hb;
    hb = !hb;
    gpio_put(PIN_LED_HB, hb);
}

void panel_set_cap_active(bool on) {
    gpio_put(PIN_LED_CAP, on);
}

void panel_imu_mark_active(imu_slot_t slot) {
    if ((unsigned)slot >= 2) {
        return;
    }
    /* Sensor core, every IMU packet: one aligned 32-bit store, nothing else.
     * panel_status_tick() turns it into LEDs / fault bits on the comms core. */
    uint32_t now = time_us_32();
    imu_last_active_us[slot] = now ? now : 1u;
}

void panel_set_emg_leadoff(bool any) {
    /* May run on either core; applied by panel_status_tick() on core 0. */
    emg_leadoff_req = any;
}

/** @brief Internal helper: imu_stream_healthy. */
static bool imu_stream_healthy(imu_slot_t slot, bool streaming) {
    uint32_t last = imu_last_active_us[slot];
    if (!streaming || last == 0) {
        return false;
    }
    return (time_us_32() - last) < (uint32_t)IMU_LED_TIMEOUT_US;
}

void panel_status_tick(void) {
    if (disco_active) {
        return;
    }

    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    bool mta_ok = imu_stream_healthy(IMU_SLOT_MTA, cfg->stream_imu_mta);
    bool cra_ok = imu_stream_healthy(IMU_SLOT_CRA, cfg->stream_imu_cra);

    gpio_put(PIN_LED_MTA, mta_ok);
    gpio_put(PIN_LED_CRA, cra_ok);
    gpio_put(PIN_LED_DP, false);

    bool lo = emg_leadoff_req;
    if (lo != emg_leadoff_lit) {
        emg_leadoff_lit = lo;
        gpio_put(PIN_LED_EMG_LO, lo);
    }

    /* Fault bits change only here and on explicit panel_fault_* calls, all on
     * this core; set when a streaming slot goes quiet, clear once it is back. */
    uint32_t imu_fault = 0;
    if (cfg->stream_imu_mta && !mta_ok) {
        imu_fault |= FAULT_IMU_MTA;
    }
    if (cfg->stream_imu_cra && !cra_ok) {
        imu_fault |= FAULT_IMU_CRA;
    }
    uint32_t mask = (fault_mask & ~(FAULT_IMU_MTA | FAULT_IMU_CRA)) | imu_fault;
    if (mask != fault_mask) {
        fault_mask = mask;
        panel_apply_fault_led();
    }

    panel_refresh_user_rgb(); /* WS2812 is only written when the colour changes */
}

void panel_update_streaming_state(void) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    bool any = cfg->stream_emg || cfg->stream_imu_mta || cfg->stream_imu_cra || trigger_pio_running();
    if (any) {
        if (state != PANEL_STATE_FAULT && fault_mask == 0) {
            panel_set_state(PANEL_STATE_STREAMING);
        }
    } else {
        if (state == PANEL_STATE_STREAMING) {
            panel_set_state(PANEL_STATE_IDLE);
        }
        gpio_put(PIN_LED_MTA, 0);
        gpio_put(PIN_LED_CRA, 0);
        gpio_put(PIN_LED_DP, 0);
        panel_set_cap_active(false);
        imu_last_active_us[IMU_SLOT_MTA] = 0;
        imu_last_active_us[IMU_SLOT_CRA] = 0;
        panel_refresh_user_rgb();
    }
    panel_status_tick();
}

/** @brief Internal helper: push_button_event. */
static void push_button_event(uint8_t button_id) {
    jtsmm_record_t rec = {0};
    rec.type = REC_BUTTON_EVENT;
    rec.t_master_us = t_master_us();
    rec.len = BUTTON_EVENT_PAYLOAD_LEN;
    rec.payload[0] = button_id;
    rec.payload[1] = 1;
    ringbuf_push(RING_EVENTS, &rec);
}

void panel_poll_buttons(void) {
    static bool u1_rel = true;
    static bool u2_rel = true;

    if (!gpio_get(PIN_SW_USER1)) {
        if (u1_rel) {
            sleep_ms(20);
            if (!gpio_get(PIN_SW_USER1)) {
                push_button_event(1);
                capture_session_toggle();
                /* A host recording cannot see this button; if it stops the
                 * streams mid-take the only trace was data ending early. */
                host_link_send_log(capture_session_active()
                                       ? "PANEL: USER1 pressed -> capture STARTED from the panel"
                                       : "PANEL: USER1 pressed -> capture STOPPED from the panel");
                printf("PANEL: USER1 -> capture %s\r\n",
                       capture_session_active() ? "started" : "STOPPED");
                u1_rel = false;
            }
        }
    } else {
        u1_rel = true;
    }

    if (!gpio_get(PIN_SW_USER2)) {
        if (u2_rel) {
            sleep_ms(20);
            if (!gpio_get(PIN_SW_USER2)) {
                panel_disco_start();
                u2_rel = false;
            }
        }
    } else {
        u2_rel = true;
    }
}

void panel_panel_test_start(void) {
    if (panel_test.active) {
        return;
    }
    panel_leds_all(false);
    panel_test.active = true;
    panel_test.step = 0;
    panel_test.next = get_absolute_time();
    panel_test.rx_quiet_until = make_timeout_time_ms(0);
}

bool panel_panel_test_busy(void) {
    return panel_test.active;
}

bool panel_panel_test_rx_quiet(void) {
    return panel_test.active || time_reached(panel_test.rx_quiet_until) == false;
}

bool panel_panel_test_poll(void) {
    if (!panel_test.active) {
        return false;
    }
    if (!time_reached(panel_test.next)) {
        return true;
    }

    static const struct {
        uint8_t pin;
        const char *name;
    } leds[] = {
        {PIN_LED_CAP, "CAP"},
        {PIN_LED_FAULT, "FAULT"},
        {PIN_LED_MTA, "MTA"},
        {PIN_LED_CRA, "CRA"},
        {PIN_LED_DP, "DP"},
        {PIN_LED_EMG_LO, "EMG_LO"},
    };
    static const uint8_t rgb_seq[][3] = {
        {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255},
    };

    panel_leds_all(false);
    if (panel_test.step < PANEL_TEST_GPIO_STEPS) {
        gpio_put(leds[panel_test.step].pin, 1);
    } else {
        uint8_t rgb_i = (uint8_t)(panel_test.step - PANEL_TEST_GPIO_STEPS);
        panel_set_rgb(rgb_seq[rgb_i][0], rgb_seq[rgb_i][1], rgb_seq[rgb_i][2]);
    }

    panel_test.step++;
    if (panel_test.step >= PANEL_TEST_STEPS) {
        panel_test.active = false;
        panel_leds_all(false);
        panel_set_state(PANEL_STATE_IDLE);
        panel_status_tick();
        panel_test.rx_quiet_until = make_timeout_time_ms(800);
        printf("panel test OK\r\n");
        host_link_send_ack(true);
        return false;
    }

    panel_test.next = delayed_by_ms(get_absolute_time(), 400);
    return true;
}
