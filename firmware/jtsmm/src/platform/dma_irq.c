// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file dma_irq.c
 * @brief Shared DMA IRQ dispatch.
 */

#include "jtsmm/platform/dma_irq.h"

#include "hardware/irq.h"

#define MAX_DMA_HANDLERS 4u

static irq_handler_t handler_list[MAX_DMA_HANDLERS];
static size_t handler_count;
static bool dma_irq_installed;

void dma_irq_init(void) {
    if (!dma_irq_installed) {
        irq_add_shared_handler(DMA_IRQ_0, dma_irq_dispatch, PICO_SHARED_IRQ_HANDLER_DEFAULT_ORDER_PRIORITY);
        irq_set_enabled(DMA_IRQ_0, true);
        dma_irq_installed = true;
    }
}

void dma_irq_register(irq_handler_t handler) {
    if (handler == NULL) {
        return;
    }
    for (size_t i = 0; i < handler_count; i++) {
        if (handler_list[i] == handler) {
            return;
        }
    }
    if (handler_count < MAX_DMA_HANDLERS) {
        handler_list[handler_count++] = handler;
    }
}

void dma_irq_dispatch(void) {
    for (size_t i = 0; i < handler_count; i++) {
        if (handler_list[i]) {
            handler_list[i]();
        }
    }
}
