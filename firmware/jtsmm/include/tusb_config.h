// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file tusb_config.h
 * @brief TinyUSB board config: CDC text + vendor bulk binary.
 *
 * @details Replaces pico_stdio_usb so firmware owns descriptors and can add
 *          the vendor bulk endpoint. See src/platform/usb/usb_descriptors.c.
 */
#ifndef _JTSMM_TUSB_CONFIG_H
#define _JTSMM_TUSB_CONFIG_H

/*
 * TinyUSB configuration for the jtsmm composite device:
 *   - CDC (interface 0/1): text banner, status, and command line (stdio).
 *   - Vendor (interface 2): dedicated bulk IN pipe for framed binary records.
 *
 * This replaces pico_stdio_usb so the firmware owns its descriptors and can add
 * the vendor bulk endpoint. See src/platform/usb/usb_descriptors.c.
 */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CFG_TUSB_MCU
#error CFG_TUSB_MCU must be defined (set by the pico-sdk tinyusb integration)
#endif

#ifndef CFG_TUSB_OS
#define CFG_TUSB_OS OPT_OS_NONE
#endif

#ifndef CFG_TUSB_DEBUG
#define CFG_TUSB_DEBUG 0
#endif

#define CFG_TUD_ENABLED 1

#ifndef CFG_TUD_MAX_SPEED
#define CFG_TUD_MAX_SPEED OPT_MODE_FULL_SPEED
#endif

/* RP2350 has a single Full-Speed USB controller on root port 0, device mode. */
#ifndef CFG_TUSB_RHPORT0_MODE
#define CFG_TUSB_RHPORT0_MODE (OPT_MODE_DEVICE | OPT_MODE_FULL_SPEED)
#endif

#ifndef CFG_TUSB_MEM_SECTION
#define CFG_TUSB_MEM_SECTION
#endif

#ifndef CFG_TUSB_MEM_ALIGN
#define CFG_TUSB_MEM_ALIGN __attribute__((aligned(4)))
#endif

#ifndef CFG_TUD_ENDPOINT0_SIZE
#define CFG_TUD_ENDPOINT0_SIZE 64
#endif

#define CFG_TUD_CDC    1
#define CFG_TUD_MSC    0
#define CFG_TUD_HID    0
#define CFG_TUD_MIDI   0
#define CFG_TUD_VENDOR 1

/* CDC only carries low-rate text; modest FIFOs are fine. */
#define CFG_TUD_CDC_RX_BUFSIZE 256
#define CFG_TUD_CDC_TX_BUFSIZE 256
#define CFG_TUD_CDC_EP_BUFSIZE 64

/*
 * Vendor bulk IN carries the full binary record stream. A large TX FIFO lets
 * host_link batch many records per USB frame and absorb host scheduling jitter.
 * At ~130 KB/s (2 kHz EMG + dual IMU) this depth buys ~0.25 s of slack, enough
 * to ride out host read gaps (pyusb re-arm latency, GIL, parse backpressure)
 * without dropping frames -- the dominant cause of residual IMU quaternion loss.
 *
 * NOTE: TinyUSB stores the FIFO depth in a uint16_t, so this MUST stay <= 32768
 * (65536 wraps to 0 and silently disables the pipe).
 */
#define CFG_TUD_VENDOR_RX_BUFSIZE 64
#define CFG_TUD_VENDOR_TX_BUFSIZE 32768

/*
 * Software transfer buffer per bulk transfer (NOT wMaxPacketSize, which stays
 * 64 for full speed in usb_descriptors.c). TinyUSB moves up to this many bytes
 * from the TX FIFO per transfer and the RP2350 DCD sends them as back-to-back
 * 64-byte packets; the next transfer is only armed from tud_task(). With the
 * default 64 every single packet needed a comms-loop pass, so a slow loop pass
 * (config apply step, IMU recovery) directly throttled the pipe. 512 B lets one
 * pass move 8 packets. Costs 2 x 448 B of RAM (IN and OUT buffers).
 */
#define CFG_TUD_VENDOR_EPSIZE 512

#ifdef __cplusplus
}
#endif

#endif /* _JTSMM_TUSB_CONFIG_H */
