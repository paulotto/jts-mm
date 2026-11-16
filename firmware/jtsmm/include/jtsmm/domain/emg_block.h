// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file emg_block.h
 * @brief Pure packing of the REC_EMG_BLOCK payload (contract K1).
 */
#pragma once

#include "jtsmm/records.h"

#include <stdint.h>
#include <string.h>

/**
 * @brief Pack one ADS frame plus its acquisition seq into @p out.
 * @param out       At least EMG_BLOCK_PAYLOAD_LEN bytes.
 * @param ads_frame EMG_BLOCK_ADS_BYTES bytes (status + 8 channels) as read.
 * @param seq       Acquisition sequence number, written little-endian.
 * @return EMG_BLOCK_PAYLOAD_LEN.
 */
static inline uint16_t emg_block_pack(uint8_t *out, const uint8_t *ads_frame, uint32_t seq) {
    memcpy(out, ads_frame, EMG_BLOCK_ADS_BYTES);
    out[EMG_BLOCK_SEQ_OFFSET + 0u] = (uint8_t)(seq & 0xFFu);
    out[EMG_BLOCK_SEQ_OFFSET + 1u] = (uint8_t)((seq >> 8) & 0xFFu);
    out[EMG_BLOCK_SEQ_OFFSET + 2u] = (uint8_t)((seq >> 16) & 0xFFu);
    out[EMG_BLOCK_SEQ_OFFSET + 3u] = (uint8_t)((seq >> 24) & 0xFFu);
    return (uint16_t)EMG_BLOCK_PAYLOAD_LEN;
}
