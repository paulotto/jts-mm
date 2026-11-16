// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file dma_irq.h
 * @brief Shared DMA IRQ dispatch for SPI/I2C DMA completions.
 */
#pragma once

#include "pico/stdlib.h"

/** @brief Install the shared DMA IRQ handler. */
void dma_irq_init(void);

/**
 * @brief Register a secondary handler invoked from the shared IRQ.
 * @param handler Function pointer; may be NULL to clear.
 */
void dma_irq_register(irq_handler_t handler);

/** @brief Dispatch registered DMA completion work (from IRQ context). */
void dma_irq_dispatch(void);
