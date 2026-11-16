// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Trigger defaults / profiles: every profile must resolve to a config that
 * passes validation, and the defaults match the rig (30 Hz, 12 ms). */
#include "jtsmm/config.h"
#include "jtsmm/domain/trigger_config.h"

#include "check.h"

extern uint32_t stub_trigger_pio_clock_hz;
extern uint8_t stub_trigger_pio_irq_level;
extern int stub_trigger_pio_idle_fast;

int main(void) {
    jtsmm_mcu_config_t c;
    jtsmm_config_set_defaults(&c);
    CHECK_EQ_U(c.trigger.rate_hz, 30u);
    CHECK_EQ_U(c.trigger.pulse_width_us, 12000u);
    CHECK(c.trigger.profile == TRIGGER_PROFILE_IMX296_DUAL_CSI);
    CHECK(trigger_config_validate(&c.trigger));

    const trigger_profile_t profiles[] = {TRIGGER_PROFILE_IMX296_DUAL_CSI, TRIGGER_PROFILE_IMX296_HIGH_FPS,
                                          TRIGGER_PROFILE_BENCH_SCOPE, TRIGGER_PROFILE_TRACK_A_LED};
    for (size_t i = 0; i < sizeof(profiles) / sizeof(profiles[0]); i++) {
        trigger_config_t t = c.trigger;
        t.profile = profiles[i];
        t.rate_hz = 0;
        t.pulse_width_us = 0;
        trigger_config_resolve_profile(&t);
        CHECK(trigger_config_validate(&t));
        CHECK(t.outputs.csi1 && t.outputs.csi2 && t.outputs.scope && !t.outputs.sync_led);
        if (profiles[i] == TRIGGER_PROFILE_IMX296_DUAL_CSI) {
            CHECK_EQ_U(t.rate_hz, 30u);
            CHECK_EQ_U(t.pulse_width_us, 12000u);
        }
        if (profiles[i] == TRIGGER_PROFILE_TRACK_A_LED) {
            CHECK_EQ_U(t.rate_hz, 30u); /* h06 §3.4 */
        }
    }

    /* Validation: width must fit the period (no silent halving downstream). */
    trigger_config_t t = c.trigger;
    t.rate_hz = 100;
    t.pulse_width_us = 12000;
    CHECK(!trigger_config_validate(&t));
    t.rate_hz = 201;
    t.pulse_width_us = 100;
    CHECK(!trigger_config_validate(&t));

    /* Apply hands the PIO clock and the IRQ priority to the driver. */
    jtsmm_config_set_defaults(jtsmm_config_mutable());
    jtsmm_config_mutable()->platform.irq_trigger_priority = 1;
    CHECK(trigger_config_apply(&jtsmm_config_active()->trigger));
    CHECK_EQ_U(stub_trigger_pio_clock_hz, 1000000u);
    CHECK_EQ_U(stub_trigger_pio_irq_level, 1u);
    /* ...and the polarity whose inactive level the line idles at (before any
     * trig_start): imx296_fast (0) -> idle HIGH, active_high (1) -> idle LOW. */
    CHECK_EQ_U(stub_trigger_pio_idle_fast, 1);
    jtsmm_config_mutable()->trigger.polarity = 1;
    CHECK(trigger_config_apply(&jtsmm_config_active()->trigger));
    CHECK_EQ_U(stub_trigger_pio_idle_fast, 0);

    CHECK_DONE("test_trigger_config");
}
