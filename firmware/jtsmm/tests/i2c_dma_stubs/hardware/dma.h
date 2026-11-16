// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub (test_i2c_dma): DMA channels whose busy state the test sets. */
#pragma once

#include "pico/types.h"

enum dma_channel_transfer_size { DMA_SIZE_8 = 0, DMA_SIZE_16 = 1, DMA_SIZE_32 = 2 };

typedef struct {
    uint32_t ctrl;
} dma_channel_config;

extern bool fake_dma_chan_busy[8];
extern int fake_dma_chan_aborts[8];

static inline int dma_claim_unused_channel(bool required) {
    static int next;
    (void)required;
    return next++;
}
static inline void dma_channel_abort(uint chan) {
    fake_dma_chan_aborts[chan]++;
    fake_dma_chan_busy[chan] = false;
}
static inline bool dma_channel_is_busy(uint chan) {
    return fake_dma_chan_busy[chan];
}
static inline dma_channel_config dma_channel_get_default_config(uint chan) {
    (void)chan;
    return (dma_channel_config){0};
}
static inline void channel_config_set_transfer_data_size(dma_channel_config *c, enum dma_channel_transfer_size s) {
    (void)c;
    (void)s;
}
static inline void channel_config_set_read_increment(dma_channel_config *c, bool incr) {
    (void)c;
    (void)incr;
}
static inline void channel_config_set_write_increment(dma_channel_config *c, bool incr) {
    (void)c;
    (void)incr;
}
static inline void channel_config_set_dreq(dma_channel_config *c, uint dreq) {
    (void)c;
    (void)dreq;
}
static inline void dma_channel_configure(uint chan, const dma_channel_config *c, volatile void *write_addr,
                                         const volatile void *read_addr, uint count, bool trigger) {
    (void)chan;
    (void)c;
    (void)write_addr;
    (void)read_addr;
    (void)count;
    (void)trigger;
}
static inline void dma_channel_start(uint chan) {
    fake_dma_chan_busy[chan] = true;
}
