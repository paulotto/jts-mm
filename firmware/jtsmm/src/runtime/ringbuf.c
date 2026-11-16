// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file ringbuf.c
 * @brief SPSC record ring implementation.
 */

#include "jtsmm/runtime/ringbuf.h"

#include "hardware/sync.h"
#include <stddef.h>
#include <string.h>

/*
 * Each ring has its own backing store of fixed-size slots, sized per ring:
 * a slot is the header below plus that ring's largest payload. EMG frames are
 * 31 B, so the EMG ring uses compact 48 B slots -- a full 264 B record per
 * 2 kHz frame held only 248 slots (124 ms) in 64 KB, and a viewer start that
 * opened USB 1.5 s after the stream lost ~1000 frames. Now 2048 EMG slots
 * (~1.0 s at 2 kHz) in 96 KB; IMU and event records keep full 240 B payloads.
 * Total ~130 KB of the RP2350's 520 KB SRAM.
 */
typedef struct {
    uint64_t t_master_us;
    uint16_t len;
    uint8_t type;
    uint8_t payload[];
} ring_slot_t;

#define SLOT_BYTES(payload_max) \
    ((((uint32_t)offsetof(ring_slot_t, payload) + (payload_max)) + 7u) & ~7u)

#define FULL_PAYLOAD_MAX  ((uint16_t)sizeof(((jtsmm_record_t *)0)->payload))
#define EMG_SLOT_BYTES    SLOT_BYTES(RINGBUF_EMG_PAYLOAD_MAX)
#define FULL_SLOT_BYTES   SLOT_BYTES(FULL_PAYLOAD_MAX)

#define EMG_RING_SLOTS   2048u
#define IMU_RING_SLOTS     64u
#define EVENT_RING_SLOTS   64u

typedef struct {
    uint8_t *store;
    uint32_t max_slots;
    uint16_t slot_bytes;
    uint16_t payload_max;
    volatile uint32_t head;
    volatile uint32_t tail;
    volatile uint32_t drops;
    volatile uint32_t pushes;
    uint32_t capacity;
} spsc_ring_t;

static uint8_t emg_store[EMG_RING_SLOTS * EMG_SLOT_BYTES] __attribute__((aligned(8)));
static uint8_t imu_mta_store[IMU_RING_SLOTS * FULL_SLOT_BYTES] __attribute__((aligned(8)));
static uint8_t imu_cra_store[IMU_RING_SLOTS * FULL_SLOT_BYTES] __attribute__((aligned(8)));
static uint8_t event_store[EVENT_RING_SLOTS * FULL_SLOT_BYTES] __attribute__((aligned(8)));

#define RING(store_, slots_, payload_max_)                                          \
    {.store = (store_), .max_slots = (slots_), .slot_bytes = SLOT_BYTES(payload_max_), \
     .payload_max = (payload_max_)}

static spsc_ring_t rings[RING_COUNT] = {
    [RING_EMG] = RING(emg_store, EMG_RING_SLOTS, RINGBUF_EMG_PAYLOAD_MAX),
    [RING_IMU_MTA] = RING(imu_mta_store, IMU_RING_SLOTS, FULL_PAYLOAD_MAX),
    [RING_IMU_CRA] = RING(imu_cra_store, IMU_RING_SLOTS, FULL_PAYLOAD_MAX),
    [RING_EVENTS] = RING(event_store, EVENT_RING_SLOTS, FULL_PAYLOAD_MAX),
};

static ring_slot_t *slot_at(const spsc_ring_t *r, uint32_t i) {
    return (ring_slot_t *)(void *)(r->store + (size_t)i * r->slot_bytes);
}

void ringbuf_init_all(void) {
    for (int i = 0; i < RING_COUNT; i++) {
        rings[i].head = 0;
        rings[i].tail = 0;
        rings[i].drops = 0;
        rings[i].pushes = 0;
        rings[i].capacity = rings[i].max_slots;
    }
}

uint32_t ringbuf_set_capacity(ring_id_t id, uint32_t cap) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    /*
     * An SPSC ring of capacity N holds N-1 records, so N=1 silently dropped
     * every push. 0 used to mean "max" and anything above the backing store was
     * clamped without a word; the caller now gets the effective value back and
     * platform_config reports any difference.
     */
    if (cap < RINGBUF_MIN_CAPACITY) {
        cap = RINGBUF_MIN_CAPACITY;
    }
    if (cap > rings[id].max_slots) {
        cap = rings[id].max_slots;
    }
    rings[id].capacity = cap;
    return cap;
}

uint32_t ringbuf_max_capacity(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    return rings[id].max_slots;
}

uint32_t ringbuf_capacity(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    return rings[id].capacity;
}

/*
 * Ordering (the two cores share no lock):
 *   producer: read tail -> dmb (acquire) -> write slot -> dmb (release) -> head
 *   consumer: read head -> dmb (acquire) -> read slot  -> dmb (release) -> tail
 * The acquire barriers matter as much as the release ones: without them the
 * compiler may hoist the (non-volatile) slot access above the volatile index
 * load and read a slot the producer has not finished writing (or overwrite one
 * the consumer is still copying).
 */

/** @brief Internal helper: push_slot. */
static bool push_slot(spsc_ring_t *r, const jtsmm_record_t *rec) {
    /* A record larger than this ring's slots cannot be queued (only EMG frames
     * go to the compact EMG ring); count it like a full ring. */
    if (rec->len > r->payload_max) {
        r->drops++;
        return false;
    }
    uint32_t head = r->head;
    uint32_t next = (head + 1u) % r->capacity;
    if (next == r->tail) {
        r->drops++;
        return false;
    }
    __dmb();
    /* Header + the valid payload bytes only. */
    ring_slot_t *dst = slot_at(r, head);
    dst->type = rec->type;
    dst->t_master_us = rec->t_master_us;
    dst->len = rec->len;
    memcpy(dst->payload, rec->payload, rec->len);
    __dmb();
    r->head = next;
    r->pushes++;
    return true;
}

/** @brief Internal helper: peek_slot (consumer side). */
static const ring_slot_t *peek_slot(spsc_ring_t *r) {
    uint32_t tail = r->tail;
    if (tail == r->head) {
        return NULL;
    }
    __dmb();
    return slot_at(r, tail);
}

/** @brief Internal helper: consume_slot (consumer side). */
static void consume_slot(spsc_ring_t *r) {
    uint32_t tail = r->tail;
    if (tail == r->head) {
        return;
    }
    __dmb();
    r->tail = (tail + 1u) % r->capacity;
}

bool ringbuf_push(ring_id_t id, const jtsmm_record_t *rec) {
    if ((unsigned)id >= RING_COUNT) {
        return false;
    }
    if (id == RING_EVENTS) {
        /*
         * RING_EVENTS is the one ring with several producers on one core: the
         * trigger PIO IRQ / timer callbacks (IRQ context) and the panel button
         * (comms main loop). An IRQ landing between a main-loop slot write and
         * its head advance would overwrite that slot. Masking IRQs around the
         * slot copy serialises them; the sensor rings stay lock-free.
         */
        uint32_t irq = save_and_disable_interrupts();
        bool ok = push_slot(&rings[id], rec);
        restore_interrupts(irq);
        return ok;
    }
    return push_slot(&rings[id], rec);
}

bool ringbuf_pop(ring_id_t id, jtsmm_record_t *rec) {
    if ((unsigned)id >= RING_COUNT || rec == NULL) {
        return false;
    }
    const ring_slot_t *slot = peek_slot(&rings[id]);
    if (slot == NULL) {
        return false;
    }
    rec->type = slot->type;
    rec->t_master_us = slot->t_master_us;
    rec->len = slot->len;
    memcpy(rec->payload, slot->payload, slot->len);
    consume_slot(&rings[id]);
    return true;
}

bool ringbuf_peek(ring_id_t id, ringbuf_view_t *out) {
    if ((unsigned)id >= RING_COUNT || out == NULL) {
        return false;
    }
    const ring_slot_t *slot = peek_slot(&rings[id]);
    if (slot == NULL) {
        return false;
    }
    out->type = slot->type;
    out->t_master_us = slot->t_master_us;
    out->len = slot->len;
    out->payload = slot->payload;
    return true;
}

uint32_t ringbuf_slot_bytes(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    return rings[id].slot_bytes;
}

void ringbuf_consume(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return;
    }
    consume_slot(&rings[id]);
}

uint32_t ringbuf_drops(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    return rings[id].drops;
}

uint32_t ringbuf_pushes(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    return rings[id].pushes;
}

uint32_t ringbuf_count(ring_id_t id) {
    if ((unsigned)id >= RING_COUNT) {
        return 0;
    }
    spsc_ring_t *r = &rings[id];
    uint32_t head = r->head;
    uint32_t tail = r->tail;
    if (head >= tail) {
        return head - tail;
    }
    return r->capacity - tail + head;
}
