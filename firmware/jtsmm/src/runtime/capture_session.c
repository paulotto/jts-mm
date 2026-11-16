// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file capture_session.c
 * @brief Capture session start/stop coordination.
 */

#include "jtsmm/runtime/capture_session.h"

#include "jtsmm/domain/emg_config.h"
#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/domain/imu_config.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/drivers/trigger_pio.h"
#include "jtsmm/config.h"
#include "jtsmm/runtime/config_apply.h"
#include "jtsmm/runtime/host_link.h"

bool capture_session_active(void) {
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    return cfg->stream_emg || cfg->stream_imu_mta || cfg->stream_imu_cra || trigger_pio_running();
}

void capture_session_start(void) {
    if (config_apply_busy() || capture_session_active()) {
        return;
    }

    /* Honour the applied enables, like ads_start / trig_start do. */
    const jtsmm_mcu_config_t *cfg = jtsmm_config_active();
    host_link_reset_tx_stats();
    /* Stream flags follow what actually started (a disabled/absent part NAKs). */
    jtsmm_config_mutable()->stream_emg = cfg->emg.enabled && ads1299_init_ok() && emg_stream_start();
    jtsmm_config_mutable()->stream_imu_mta = imu_stream_start_slot(IMU_SLOT_MTA);
    jtsmm_config_mutable()->stream_imu_cra = imu_stream_start_slot(IMU_SLOT_CRA);

    const trigger_config_t *t = &cfg->trigger;
    if (t->enabled && !trigger_pio_start(t->rate_hz, t->pulse_width_us, t->polarity == 0)) {
        host_link_send_log("PANEL: capture started but trig_start FAILED");
    }
    panel_update_streaming_state();
}

void capture_session_stop(void) {
    if (config_apply_busy() || !capture_session_active()) {
        return;
    }

    trigger_pio_stop();
    imu_stream_stop_slot(IMU_SLOT_CRA);
    jtsmm_config_mutable()->stream_imu_cra = false;
    imu_stream_stop_slot(IMU_SLOT_MTA);
    jtsmm_config_mutable()->stream_imu_mta = false;
    emg_stream_stop();
    jtsmm_config_mutable()->stream_emg = false;
    panel_update_streaming_state();
}

void capture_session_toggle(void) {
    if (capture_session_active()) {
        capture_session_stop();
    } else {
        capture_session_start();
    }
}
