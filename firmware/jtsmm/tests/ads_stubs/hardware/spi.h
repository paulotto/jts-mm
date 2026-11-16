// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/* Host-test stub: hardware/spi.h subset for the REAL ads1299.c, backed by the
 * ADS1299 register model in tests/ads_stubs/fake_ads1299.c. */
#pragma once

#include "pico/types.h"

/* spi_get_hw(spi)->dr: the driver writes a TX byte, polls spi_is_readable(),
 * then reads the RX byte. The fake completes the byte inside spi_is_readable(). */
typedef struct {
    volatile uint32_t dr;
} spi_hw_t;

typedef struct spi_inst spi_inst_t;
extern spi_inst_t *const fake_spi0;
#define spi0 fake_spi0

typedef enum { SPI_CPOL_0 = 0, SPI_CPOL_1 = 1 } spi_cpol_t;
typedef enum { SPI_CPHA_0 = 0, SPI_CPHA_1 = 1 } spi_cpha_t;
typedef enum { SPI_LSB_FIRST = 0, SPI_MSB_FIRST = 1 } spi_order_t;

uint spi_init(spi_inst_t *spi, uint baudrate);
uint spi_set_baudrate(spi_inst_t *spi, uint baudrate);
uint spi_get_baudrate(const spi_inst_t *spi);
void spi_set_format(spi_inst_t *spi, uint data_bits, spi_cpol_t cpol, spi_cpha_t cpha, spi_order_t order);
spi_hw_t *spi_get_hw(spi_inst_t *spi);
bool spi_is_writable(const spi_inst_t *spi);
bool spi_is_readable(const spi_inst_t *spi);
