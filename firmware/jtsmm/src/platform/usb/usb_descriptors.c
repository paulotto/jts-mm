// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/*
 * jtsmm composite USB descriptors: CDC (text + command line) + Vendor (bulk IN
 * for framed binary records). Replaces pico_stdio_usb's descriptors.
 */
/**
 * @file usb_descriptors.c
 * @brief TinyUSB composite descriptors (CDC + vendor bulk).
 */

#include "tusb.h"
#include "pico/unique_id.h"
#include <string.h>

/* VID 0x2E8A (Raspberry Pi) + a PID in the 0x0xxx range so the existing
 * "2e8a idProduct [01]* MODE=0666" udev rule grants libusb access without root. */
#define JTSMM_USB_VID 0x2E8Au
#define JTSMM_USB_PID 0x0004u

//--------------------------------------------------------------------+
// Device Descriptor
//--------------------------------------------------------------------+
static const tusb_desc_device_t desc_device = {
    .bLength = sizeof(tusb_desc_device_t),
    .bDescriptorType = TUSB_DESC_DEVICE,
    .bcdUSB = 0x0200,

    /* IAD is required so the host groups the two CDC interfaces together. */
    .bDeviceClass = TUSB_CLASS_MISC,
    .bDeviceSubClass = MISC_SUBCLASS_COMMON,
    .bDeviceProtocol = MISC_PROTOCOL_IAD,
    .bMaxPacketSize0 = CFG_TUD_ENDPOINT0_SIZE,

    .idVendor = JTSMM_USB_VID,
    .idProduct = JTSMM_USB_PID,
    .bcdDevice = 0x0100,

    .iManufacturer = 0x01,
    .iProduct = 0x02,
    .iSerialNumber = 0x03,

    .bNumConfigurations = 0x01,
};

uint8_t const *tud_descriptor_device_cb(void) {
    return (uint8_t const *)&desc_device;
}

//--------------------------------------------------------------------+
// Configuration Descriptor
//--------------------------------------------------------------------+
enum {
    ITF_NUM_CDC = 0,
    ITF_NUM_CDC_DATA,
    ITF_NUM_VENDOR,
    ITF_NUM_TOTAL,
};

#define CONFIG_TOTAL_LEN (TUD_CONFIG_DESC_LEN + TUD_CDC_DESC_LEN + TUD_VENDOR_DESC_LEN)

#define EPNUM_CDC_NOTIF  0x81
#define EPNUM_CDC_OUT    0x02
#define EPNUM_CDC_IN     0x82
#define EPNUM_VENDOR_OUT 0x03
#define EPNUM_VENDOR_IN  0x83

static uint8_t const desc_configuration[] = {
    TUD_CONFIG_DESCRIPTOR(1, ITF_NUM_TOTAL, 0, CONFIG_TOTAL_LEN, 0x00, 100),

    // CDC: itf, string, notify EP + size, data OUT/IN EP, data EP size
    TUD_CDC_DESCRIPTOR(ITF_NUM_CDC, 4, EPNUM_CDC_NOTIF, 8, EPNUM_CDC_OUT, EPNUM_CDC_IN, 64),

    // Vendor: itf, string, bulk OUT/IN EP, EP size
    TUD_VENDOR_DESCRIPTOR(ITF_NUM_VENDOR, 5, EPNUM_VENDOR_OUT, EPNUM_VENDOR_IN, 64),
};

uint8_t const *tud_descriptor_configuration_cb(uint8_t index) {
    (void)index;
    return desc_configuration;
}

//--------------------------------------------------------------------+
// String Descriptors
//--------------------------------------------------------------------+
enum {
    STRID_LANGID = 0,
    STRID_MANUFACTURER,
    STRID_PRODUCT,
    STRID_SERIAL,
    STRID_CDC,
    STRID_VENDOR,
};

static char const *string_desc_arr[] = {
    (const char[]){0x09, 0x04}, // 0: English (0x0409)
    "jtsmm",                    // 1: Manufacturer
    "jtsmm HAT",                // 2: Product
    NULL,                       // 3: Serial (filled from unique board id)
    "jtsmm CDC",                // 4: CDC interface
    "jtsmm bulk",               // 5: Vendor interface
};

static uint16_t _desc_str[32 + 1];

/** @brief Internal helper: board_serial_hex. */
static size_t board_serial_hex(uint16_t *dst, size_t max_chars) {
    pico_unique_board_id_t id;
    pico_get_unique_board_id(&id);
    static const char hex[] = "0123456789ABCDEF";
    size_t n = 0;
    for (size_t i = 0; i < PICO_UNIQUE_BOARD_ID_SIZE_BYTES && n + 2 <= max_chars; i++) {
        dst[n++] = (uint16_t)hex[(id.id[i] >> 4) & 0xF];
        dst[n++] = (uint16_t)hex[id.id[i] & 0xF];
    }
    return n;
}

uint16_t const *tud_descriptor_string_cb(uint8_t index, uint16_t langid) {
    (void)langid;
    size_t chr_count;

    switch (index) {
    case STRID_LANGID:
        memcpy(&_desc_str[1], string_desc_arr[0], 2);
        chr_count = 1;
        break;
    case STRID_SERIAL:
        chr_count = board_serial_hex(_desc_str + 1, 32);
        break;
    default:
        if (index >= sizeof(string_desc_arr) / sizeof(string_desc_arr[0])) {
            return NULL;
        }
        const char *str = string_desc_arr[index];
        chr_count = strlen(str);
        size_t const max_count = sizeof(_desc_str) / sizeof(_desc_str[0]) - 1;
        if (chr_count > max_count) {
            chr_count = max_count;
        }
        for (size_t i = 0; i < chr_count; i++) {
            _desc_str[1 + i] = str[i];
        }
        break;
    }

    _desc_str[0] = (uint16_t)((TUSB_DESC_STRING << 8) | (2 * chr_count + 2));
    return _desc_str;
}
