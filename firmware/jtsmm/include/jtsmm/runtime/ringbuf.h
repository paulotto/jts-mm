// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file ringbuf.h
 * @brief SPSC record rings between sensor core and comms core.
 *
 * @details Sensor core (physical core 1) pushes; comms core (physical core 0)
 *          pops and forwards via @ref host_link_send_record. Capacities come
 *          from platform config. See docs/hardware/h05_sensor_pipeline_architecture.md.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

/**
 * @brief One framed record waiting in a ring slot.
 */
typedef struct {
    uint8_t type;           /**< Record type tag (`REC_*` from records.h). */
    uint64_t t_master_us;   /**< Master timestamp (µs). */
    uint16_t len;           /**< Valid payload bytes (≤ 240). */
    uint8_t payload[240];   /**< Payload bytes. */
} jtsmm_record_t;

/**
 * @brief Logical ring identifiers.
 */
typedef enum {
    RING_EMG = 0,   /**< EMG sample blocks. */
    RING_IMU_MTA,   /**< MTA IMU reports. */
    RING_IMU_CRA,   /**< CRA IMU reports. */
    RING_EVENTS,    /**< Trigger / LED / button / misc events. */
    RING_COUNT,     /**< Number of rings (not a usable id). */
} ring_id_t;

/**
 * @brief Initialize all rings to default empty state.
 */
void ringbuf_init_all(void);

/** Smallest usable capacity: an SPSC ring of N slots holds N-1 records. */
#define RINGBUF_MIN_CAPACITY 2u

/** Largest payload the compact EMG ring holds (an EMG frame is 31 B). Longer
 *  records pushed onto @ref RING_EMG are refused and counted as drops. */
#define RINGBUF_EMG_PAYLOAD_MAX 32u

/**
 * @brief Set slot capacity for one ring.
 *
 * @param id  Ring to resize.
 * @param cap Requested number of slots (see @ref ringbuf_slot_bytes).
 * @return Effective capacity: @p cap clamped to
 *         [@ref RINGBUF_MIN_CAPACITY, @ref ringbuf_max_capacity]; 0 for a bad id.
 *
 * @note Call during config apply before streaming; not hot-path safe mid-stream.
 */
uint32_t ringbuf_set_capacity(ring_id_t id, uint32_t cap);

/** @brief Backing-store slot count for @p id (upper bound for the capacity). */
uint32_t ringbuf_max_capacity(ring_id_t id);

/** @brief Current configured capacity of @p id (holds capacity-1 records). */
uint32_t ringbuf_capacity(ring_id_t id);

/**
 * @brief Push a record onto a ring (producer / sensor core).
 *
 * @param id  Destination ring.
 * @param rec Record to copy; must not be NULL.
 * @return true on success; false if full, or @p rec is longer than this ring's
 *         slots (both counted via @ref ringbuf_drops).
 */
bool ringbuf_push(ring_id_t id, const jtsmm_record_t *rec);

/**
 * @brief Pop one record from a ring (consumer / comms core).
 *
 * @param id  Source ring.
 * @param rec Output buffer; must not be NULL.
 * @return true if a record was copied; false if empty.
 */
bool ringbuf_pop(ring_id_t id, jtsmm_record_t *rec);

/**
 * @brief A queued record as seen in its ring slot (consumer side).
 *
 * @details Slots are sized per ring (the EMG ring holds only
 *          @ref RINGBUF_EMG_PAYLOAD_MAX payload bytes), so the consumer gets a
 *          view instead of a @ref jtsmm_record_t pointer.
 */
typedef struct {
    uint8_t type;            /**< Record type tag. */
    uint64_t t_master_us;    /**< Master timestamp (µs). */
    uint16_t len;            /**< Valid payload bytes. */
    const uint8_t *payload;  /**< Points into the slot; valid until @ref ringbuf_consume. */
} ringbuf_view_t;

/**
 * @brief Oldest queued record, left in the ring (consumer / comms core).
 *
 * @details Lets the consumer decide whether it can forward the record before
 *          taking it: pair with @ref ringbuf_consume once it is handled. The
 *          view stays valid until then (the producer never writes that slot).
 * @return true and fills @p out, or false if the ring is empty or @p id is bad.
 */
bool ringbuf_peek(ring_id_t id, ringbuf_view_t *out);

/** @brief Release the record returned by @ref ringbuf_peek (no-op if empty). */
void ringbuf_consume(ring_id_t id);

/**
 * @brief Bytes of RAM one slot of @p id takes; config byte budgets divide by it.
 */
uint32_t ringbuf_slot_bytes(ring_id_t id);

/**
 * @brief Cumulative push failures (ring full) for @p id.
 */
uint32_t ringbuf_drops(ring_id_t id);

/**
 * @brief Cumulative successful pushes for @p id.
 */
uint32_t ringbuf_pushes(ring_id_t id);

/**
 * @brief Current number of queued records in @p id.
 */
uint32_t ringbuf_count(ring_id_t id);
