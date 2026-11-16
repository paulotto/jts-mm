// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file imu_bus.h
 * @brief Dual BNO085 I2C bus helpers, slot handles, and INT pending flags.
 */
#pragma once

#include "jtsmm/drivers/bno085.h"
#include "jtsmm/domain/imu_timing.h"
#include "hardware/i2c.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Logical IMU mounting slots on the HAT.
 */
typedef enum {
    IMU_SLOT_MTA = 0, /**< Metatarsal / primary motion IMU. */
    IMU_SLOT_CRA = 1, /**< Cranial / second IMU. */
} imu_slot_t;

/** @brief Initialize both IMU I2C buses and device structs. @return true on success. */
bool imu_bus_init(void);

/**
 * @brief Achieved I2C bitrate for @p slot after configuration (Hz).
 */
uint32_t imu_bus_achieved_hz(imu_slot_t slot);

/**
 * @brief Pointer to the live @ref bno085_t for @p slot.
 * @return Non-NULL device pointer after @ref imu_bus_init.
 */
bno085_t *imu_bus_dev(imu_slot_t slot);

/**
 * @brief Copy device descriptor for @p slot into @p out.
 * @param slot IMU slot.
 * @param out  Destination; must not be NULL.
 */
void imu_bus_get_dev(imu_slot_t slot, bno085_t *out);

/** @brief Probe I2C presence for @p slot. @return true if responsive. */
bool imu_bus_probe(imu_slot_t slot);

/** @brief Hardware reset both IMUs via shared @c PIN_IMU_NRST. @return true on success. */
bool imu_bus_hw_reset(void);

/**
 * @brief Drive the shared IMU reset line (false = in reset). Non-blocking;
 *        the caller times the pulse. Resets BOTH IMUs.
 */
void imu_bus_nrst(bool high);

/**
 * @brief I2C bus clear for @p slot: release a slave stuck holding SDA low
 *        (up to 9 SCL pulses), issue a STOP, re-init the I2C block.
 *
 * @details Bit-bangs the pins for < 1 ms. The slot's DMA must be idle
 *          (quiesced / sensor core parked) -- the I2C block is re-initialised.
 * @return true if SDA and SCL both read high (bus idle) afterwards.
 */
bool imu_bus_clear(imu_slot_t slot);

/** @brief Prepare GPIOs / state for @p slot before feature enable. */
void imu_bus_prepare_slot(imu_slot_t slot);

/**
 * @brief Configure @p slot's INT pin (input, internal pull-up) and register the
 *        legacy pending-flag handler with the core-0 GPIO dispatcher (gpio_irq.c).
 *
 * @details The pending flags below are not used by streaming; the timestamp
 *          latch (imu_int_edge_*) is a separate core-1 raw handler.
 */
void imu_int_isr_install(imu_slot_t slot);

/**
 * @brief Enable or mask the legacy INT IRQ for @p slot on the CALLING core.
 * @param slot IMU slot.
 * @param on   true to enable.
 */
void imu_int_isr_enable(imu_slot_t slot, bool on);

/**
 * @{
 * @name INT edge timestamp latch (sensor core)
 *
 * The BNO085 INT falling edge is latched by a raw GPIO IRQ handler that runs
 * on core 1 only. IO_IRQ_BANK0 has separate enable/status registers per core
 * (proc0_irq_ctrl / proc1_irq_ctrl) and a separate NVIC per core, while the
 * handler chain lives in the vector table both cores share. gpio_irq.c's
 * dispatcher is core 0's SDK callback (config apply runs on core 0), so a pin
 * enabled through it interrupts core 0 -- the wrong core for data the sensor
 * core consumes. Instead imu_int_edge_core_init() adds a raw handler for
 * GPIO16/17 from core 1 (gpio_add_raw_irq_handler_masked, which also masks
 * those pins out of core 1's SDK callback dispatch) and enables IO_IRQ_BANK0
 * in core 1's NVIC only. The handler checks the core number and reads
 * proc1_irq_ctrl, so it is a no-op should core 0 ever take IO_IRQ_BANK0 for
 * its own pins.
 */

/** @brief Install the raw INT handler and enable IO_IRQ_BANK0 on this core. Call on core 1. */
void imu_int_edge_core_init(void);

/** @brief true once imu_int_edge_core_init() ran on core 1. */
bool imu_int_edge_core_ready(void);

/**
 * @brief Arm / disarm @p slot's falling-edge IRQ for streaming (core 1 only).
 * @details Acknowledges a stale hardware edge and clears the latch either way.
 */
void imu_int_edge_arm(imu_slot_t slot, bool on);

/** @brief true while @p slot is armed by @ref imu_int_edge_arm. */
bool imu_int_edge_armed(imu_slot_t slot);

/** @brief Forget @p slot's pending edge (any core). */
void imu_int_edge_clear(imu_slot_t slot);

/** @brief @p slot's latch (slot 0's for an invalid slot; never NULL). */
imu_edge_latch_t *imu_int_edge_latch(imu_slot_t slot);

/**
 * @brief Bench check of the INT IRQ path, no IMU needed (comms core, blocking).
 *
 * @details Enables the falling-edge IRQ of both INT pins on core 1 (the
 *          streaming handler, so the production path is what gets tested) for
 *          @p ms with the internal pull-up on (the HAT has no INT pull-up), and
 *          prints per pin: IRQ-counted edges, falling edges seen by polling the
 *          level, and the level at the end. A slot armed for streaming is left
 *          as it is; the caller refuses the command while a slot streams or an
 *          apply runs.
 */
void imu_int_watch(uint32_t ms);
/** @} */

/** @brief true if software pending flag is set for @p slot. */
bool imu_int_pending(imu_slot_t slot);

/** @brief Clear software pending flag for @p slot. */
void imu_int_clear_pending(imu_slot_t slot);

/** @brief Set software pending flag for @p slot (ISR or poll). */
void imu_int_set_pending(imu_slot_t slot);
