// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file config_flash.c
 * @brief Flash persistence for MCU config JSON.
 */

#include "jtsmm/runtime/config_flash.h"

#include "jtsmm/app/fault.h"
#include "hardware/flash.h"
#include "hardware/sync.h"
#include "pico/flash.h"
#include "pico/multicore.h"
#include <string.h>

#define CFG_MAGIC 0x4d53544aU /* JTSM little-endian */
#define CFG_VERSION 1u
#define CFG_MAX_JSON (FLASH_SECTOR_SIZE - 16u)

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t reserved;
    uint16_t length;
    uint32_t crc32;
} cfg_header_t;

/** @brief Internal helper: cfg_flash_offset. */
static uint32_t cfg_flash_offset(void) {
    return (uint32_t)(PICO_FLASH_SIZE_BYTES - FLASH_SECTOR_SIZE);
}

/** @brief Internal helper: crc32_ieee. */
static uint32_t crc32_ieee(const uint8_t *data, size_t len) {
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (int b = 0; b < 8; b++) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xedb88320u & mask);
        }
    }
    return ~crc;
}

/** @brief Internal helper: cfg_sector_ptr. */
static const uint8_t *cfg_sector_ptr(void) {
    return (const uint8_t *)(XIP_BASE + cfg_flash_offset());
}

/** @brief Internal helper: parse_header. */
static bool parse_header(cfg_header_t *hdr, const uint8_t *sector) {
    memcpy(hdr, sector, sizeof(*hdr));
    if (hdr->magic != CFG_MAGIC) {
        return false;
    }
    if (hdr->version != CFG_VERSION) {
        return false;
    }
    if (hdr->length == 0 || hdr->length > CFG_MAX_JSON) {
        return false;
    }
    uint32_t crc = crc32_ieee(sector + sizeof(cfg_header_t), hdr->length);
    return crc == hdr->crc32;
}

bool config_flash_has_valid(void) {
    cfg_header_t hdr;
    return parse_header(&hdr, cfg_sector_ptr());
}

bool config_flash_load(char *buf, size_t cap) {
    if (buf == NULL || cap == 0) {
        return false;
    }
    const uint8_t *sector = cfg_sector_ptr();
    cfg_header_t hdr;
    if (!parse_header(&hdr, sector)) {
        return false;
    }
    if ((size_t)hdr.length + 1u > cap) {
        return false;
    }
    memcpy(buf, sector + sizeof(cfg_header_t), hdr.length);
    buf[hdr.length] = '\0';
    return true;
}

/*
 * Flash writes stall XIP for both cores. The old code only disabled core-0 IRQs
 * while core 1 kept executing (sensor loop, idle lead-off poll) from flash --
 * undefined behaviour during an erase. Now:
 *  - core 1 running: flash_safe_execute() uses the SDK multicore lockout; core 1
 *    registered itself as victim in sensor_core_entry() and waits in a RAM
 *    handler with IRQs off while this core disables its own IRQs and writes.
 *  - core 1 not launched (or reset): it sits in the bootrom, which does not
 *    touch flash, so disabling core-0 IRQs is enough.
 * The caller (cmd_parser) refuses save/erase while streams run, so the lockout
 * never pauses a live acquisition.
 */
typedef struct {
    uint32_t offset;
    const uint8_t *data; /* NULL = erase only */
} flash_op_t;

/** @brief Internal helper: the flash operation itself (runs with both cores quiet). */
static void flash_op_fn(void *param) {
    const flash_op_t *op = (const flash_op_t *)param;
    flash_range_erase(op->offset, FLASH_SECTOR_SIZE);
    if (op->data != NULL) {
        flash_range_program(op->offset, op->data, FLASH_SECTOR_SIZE);
    }
}

/** @brief Internal helper: run @p op with both cores kept off the flash. */
static bool flash_op_run(flash_op_t *op) {
    bool ok = true;
    fault_watchdog_hold(); /* a sector erase can take a few hundred ms */
    if (multicore_lockout_victim_is_initialized(1)) {
        ok = flash_safe_execute(flash_op_fn, op, 100) == PICO_OK;
    } else {
        uint32_t ints = save_and_disable_interrupts();
        flash_op_fn(op);
        restore_interrupts(ints);
    }
    fault_watchdog_release();
    return ok;
}

bool config_flash_erase(void) {
    flash_op_t op = {.offset = cfg_flash_offset(), .data = NULL};
    return flash_op_run(&op) && !config_flash_has_valid();
}

bool config_flash_save(const char *json) {
    if (json == NULL) {
        return false;
    }
    size_t len = strlen(json);
    if (len == 0 || len > CFG_MAX_JSON) {
        return false;
    }

    static uint8_t sector_buf[FLASH_SECTOR_SIZE];
    memset(sector_buf, 0xff, sizeof(sector_buf));

    cfg_header_t hdr = {
        .magic = CFG_MAGIC,
        .version = CFG_VERSION,
        .reserved = 0,
        .length = (uint16_t)len,
        .crc32 = 0,
    };
    memcpy(sector_buf + sizeof(hdr), json, len);
    hdr.crc32 = crc32_ieee(sector_buf + sizeof(hdr), len);
    memcpy(sector_buf, &hdr, sizeof(hdr));

    flash_op_t op = {.offset = cfg_flash_offset(), .data = sector_buf};
    if (!flash_op_run(&op)) {
        return false;
    }
    /* Read back through XIP: header/CRC valid and the payload is ours. */
    return config_flash_has_valid() &&
           memcmp(cfg_sector_ptr() + sizeof(cfg_header_t), json, len) == 0;
}
