// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file emg_config.c
 * @brief EMG domain apply/stream into RING_EMG.
 */

#include "jtsmm/domain/emg_config.h"

#include "jtsmm/domain/emg_block.h"
#include "jtsmm/domain/emg_config_check.h"

#include "jtsmm/domain/imu_config.h"
#include "jtsmm/drivers/ads1299.h"
#include "jtsmm/drivers/panel.h"
#include "jtsmm/config.h"
#include "jtsmm/platform/spi_dma.h"
#include "jtsmm/records.h"
#include "jtsmm/runtime/ringbuf.h"
#include <stdio.h>
#include <string.h>

static volatile uint32_t emg_push_ok;
static volatile uint32_t emg_misalign_drops;

uint32_t emg_stream_push_ok_count(void) {
    return emg_push_ok;
}

uint32_t emg_stream_misalign_drop_count(void) {
    return emg_misalign_drops;
}

void emg_config_resolve_profile(emg_config_t *cfg) {
    if (cfg->profile == EMG_PROFILE_CUSTOM) {
        return;
    }
    switch (cfg->profile) {
    case EMG_PROFILE_HIGH_RATE:
        cfg->sample_rate_sps = 4000;
        break;
    case EMG_PROFILE_BENCH:
        cfg->sample_rate_sps = 250;
        cfg->leadoff.enabled = true;
        break;
    case EMG_PROFILE_NOISE_CHARACTERIZATION:
        /* Acquisition settings with no lead-off current while streaming, as the
         * host catalog resolves it (EMG_PROFILE_OVERRIDES). It used to ask for
         * 16 kSPS, which emg_config_check() rejects, so the profile could
         * never be applied. */
    case EMG_PROFILE_ACQUISITION:
    default:
        cfg->sample_rate_sps = 4000;
        cfg->default_gain = 6;
        cfg->bias.enabled = true;
        cfg->leadoff.enabled = true;
        /* DC lead-off stays on while streaming, as in the host template
         * (schema/v1/profiles/emg_acquisition.yaml, h03 §6): a constant current
         * only offsets connected inputs (removed by the high-pass) and rails
         * open ones, so it is the live contact indication. Off, there is no
         * lead-off indication at all -- the idle poll needs it too. */
        cfg->leadoff.during_recording = true;
        if (cfg->profile == EMG_PROFILE_NOISE_CHARACTERIZATION) {
            cfg->leadoff.during_recording = false;
        }
        break;
    }
}

bool emg_config_validate(const emg_config_t *cfg) {
    const char *why = emg_config_check(cfg);
    if (why != NULL) {
        printf("emg config REJECTED: %s\r\n", why);
        return false;
    }
    return true;
}

bool emg_config_apply(const emg_config_t *cfg) {
    if (!emg_config_validate(cfg)) {
        return false;
    }
    ads1299_set_sample_rate_hz(cfg->sample_rate_sps);
    ads1299_set_pga_gain(cfg->default_gain);
    ads1299_set_bias_drive(cfg->bias.enabled);
    ads1299_set_pd_channels(cfg->pd_channel_mask);
    /* Non-empty with bias on: checked by emg_config_check(). */
    ads1299_set_bias_channels((uint8_t)(cfg->bias.channel_mask &
                                        (uint8_t)(~cfg->pd_channel_mask) & 0x3Fu));
    ads1299_set_leadoff_monitor(cfg->leadoff.enabled);
    ads1299_set_leadoff_during_recording(cfg->leadoff.during_recording);
    ads1299_set_leadoff_current_na(cfg->leadoff.current_na);
    ads1299_set_leadoff_ac_tone(cfg->leadoff.ac_tone);
    ads1299_set_drl_probe(cfg->leadoff.probe_drl);
    spi_dma_on_complete(emg_stream_on_complete);
    ads1299_set_drdy_divisor_cb(imu_drdy_divisor_kick);
    if (!ads1299_init()) {
        return false;
    }
    if (cfg->test_signal.enabled) {
        (void)ads1299_test_signal(true);
    }
    /*
     * Report what the hardware is ACTUALLY set to, read back from the driver
     * rather than echoed from cfg. The host derives lsb_uv -- and therefore
     * every microvolt written to the HDF5 -- from the gain it requested, so a
     * silent firmware/host disagreement scales the whole recording. Profile
     * expansion can still rewrite these fields, so print the end state where
     * the operator (and `jtsmm doctor`) can see it.
     */
    printf("emg applied: fs=%luHz gain=%u bias=%d bias_mask=0x%02X pd_mask=0x%02X "
           "leadoff=%d during_rec=%d ac_tone=%d drl_probe=%d test_signal=%d\r\n",
           (unsigned long)ads1299_sample_rate_hz(), (unsigned)ads1299_pga_gain(),
           ads1299_bias_drive() ? 1 : 0, (unsigned)ads1299_bias_channels(),
           (unsigned)ads1299_pd_channels(), cfg->leadoff.enabled ? 1 : 0,
           cfg->leadoff.during_recording ? 1 : 0, cfg->leadoff.ac_tone ? 1 : 0,
           ads1299_drl_probe() ? 1 : 0, cfg->test_signal.enabled ? 1 : 0);
    {
        uint32_t req_hz = jtsmm_config_active()->platform.spi0_stream_hz;
        uint32_t spi_hz = ads1299_stream_spi_hz();
        if (req_hz != spi_hz) {
            printf("emg WARNING: platform spi0_stream_hz=%lu exceeds the ADS1299 tSDECODE limit "
                   "for the DMA frame (SBAS499C 9.5.3.1); streaming at %lu Hz\r\n",
                   (unsigned long)req_hz, (unsigned long)spi_hz);
        }
        /* One 28-byte RDATA frame must clock out well inside a DRDY period or
         * DRDY edges land while the DMA is still busy (missed frames). */
        uint32_t fs = ads1299_sample_rate_hz();
        if (spi_hz > 0 && fs > 0) {
            uint32_t frame_us = (28u * 8u * 1000000u) / spi_hz;
            uint32_t period_us = 1000000u / fs;
            if (frame_us * 3u > period_us * 2u) {
                printf("emg WARNING: fs=%luHz leaves %luus per sample but one SPI frame takes "
                       "%luus at %luHz -- expect missed frames; lower fs or raise "
                       "spi0_stream_hz (max 4 MHz)\r\n",
                       (unsigned long)fs, (unsigned long)period_us, (unsigned long)frame_us,
                       (unsigned long)spi_hz);
            }
        }
    }
    if (cfg->default_gain != ads1299_pga_gain()) {
        printf("emg WARNING: requested gain %u was rejected, running at %u — host "
               "lsb_uv will be wrong by %ux\r\n",
               (unsigned)cfg->default_gain, (unsigned)ads1299_pga_gain(),
               (unsigned)(cfg->default_gain ? cfg->default_gain : 1u));
    }
    return true;
}

void emg_stream_on_complete(uint64_t t_us, uint32_t seq, const uint8_t *data, uint16_t len) {
    if (data == NULL || len < EMG_BLOCK_ADS_BYTES) {
        return;
    }
    /* Decode leadoff + validate framing from the ADS status word (first 3 bytes)
     * so leadoff monitoring no longer needs a disruptive SDATAC/RDATAC poll. */
    if (!ads1299_stream_status_update(data)) {
        /*
         * The byte stream slipped, so these 24 bytes are not this sample's
         * channels. Publishing them anyway -- with a valid timestamp and a valid
         * CRC -- is the worst possible outcome: the host, the HDF5 file and the
         * integrity checker all accept them as real microvolts, and on screen it
         * is a burst of high-amplitude hash that no electrode touch can change.
         * A dropped sample is recoverable (it shows up as a seq gap the
         * integrity checker reports); fabricated data is not.
         *
         * No logging from here: this is the sensor core, and host_link/printf
         * are the comms core's. comms_core_emg_guard_tick() reports the counter.
         */
        emg_misalign_drops++;
        (void)ads1299_stream_resync_if_needed();
        if (ads1299_misalign_guard_drop()) {
            return;
        }
    }
    /* No zero-fill of the 240-byte payload per frame: ringbuf_push copies only
     * the header and rec.len bytes, all written by emg_block_pack(). */
    jtsmm_record_t rec;
    rec.type = REC_EMG_BLOCK;
    rec.t_master_us = t_us;
    rec.len = emg_block_pack(rec.payload, data, seq);
    /* A full ring is counted by ringbuf_drops(); only real pushes count here. */
    if (ringbuf_push(RING_EMG, &rec)) {
        emg_push_ok++;
    }
    panel_set_cap_active(true);
}

bool emg_stream_start(void) {
    if (ads1299_drdy_streaming()) {
        /* Already streaming (e.g. `ads_start` twice, or capture_session_start
         * after a manual start): must not reset counters or touch the ADS. */
        return true;
    }
    if (!ads1299_init_ok()) {
        return false;
    }
    /* Per-stream, to match ads1299_start_stream()'s reset of frame_misalign.
     * The two counters describe the same event and are printed side by side by
     * the guard tick, `emg_guard`, `jtsmm integrity` and the stats line; leaving
     * this one cumulative meant a clean recording could report the previous
     * recording's drops ("1 misaligned frames, 500 dropped"). */
    emg_misalign_drops = 0;
    spi_dma_on_complete(emg_stream_on_complete);
    if (!ads1299_start_stream()) {
        return false;
    }
    panel_set_cap_active(true);
    return true;
}

void emg_stream_stop(void) {
    ads1299_stop_stream();
    panel_set_cap_active(false);
}
