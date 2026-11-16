# This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
# SPDX-FileCopyrightText: 2026 Paul-Otto Müller
# SPDX-License-Identifier: MIT

"""libusb/pyusb reader for the jtsmm vendor bulk IN endpoint.

The firmware presents a composite USB device: a CDC ACM interface for text and
commands (handled by ``serial_util``/pyserial) and a dedicated vendor interface
whose bulk IN endpoint carries the framed binary record stream. Splitting the two
means the binary stream is never contaminated by text (no parser resyncs) and can
be batched into full USB packets.

Access requires libusb. On Linux the existing udev rule
``ATTRS{idVendor}=="2e8a", ... MODE="0666"`` (or ``plugdev`` membership) grants a
non-root user access to the vendor interface.

Test / diagnostic only (see ``capture_mcu``); recordings use the native libusb
reader in ``software/jtsmm/native``. A read that times out after receiving part
of the request returns those bytes (pyusb >= 1.2 libusb1 backend), so no data is
lost on the timeout path.
"""

from __future__ import annotations

import time

import usb.core
import usb.util

JTSMM_VID = 0x2E8A
JTSMM_PIDS = (0x0004,)

VENDOR_INTERFACE = 2
VENDOR_EP_IN = 0x83

# Large read requests let one libusb transfer drain a big burst from the MCU
# vendor FIFO in a single round-trip, minimizing the window in which no IN
# request is armed (during which the endpoint NAKs and the FIFO backs up).
_READ_CHUNK = 16384


class BulkReaderError(RuntimeError):
    pass


class BulkReader:
    """Reads raw bytes from the vendor bulk IN endpoint."""

    def __init__(self, *, vid: int = JTSMM_VID, pids: tuple[int, ...] = JTSMM_PIDS) -> None:
        self._vid = vid
        self._pids = pids
        self._dev: usb.core.Device | None = None
        self._ep_in = None
        self._claimed = False

    def open(self) -> None:
        dev = None
        for pid in self._pids:
            dev = usb.core.find(idVendor=self._vid, idProduct=pid)
            if dev is not None:
                break
        if dev is None:
            raise BulkReaderError(
                f"No jtsmm USB device (VID=0x{self._vid:04X}, "
                f"PID in {[hex(p) for p in self._pids]}). Is it plugged into J_USB1?"
            )

        # The kernel does not bind a driver to the vendor interface, but detach
        # defensively in case a prior run left one attached.
        try:
            if dev.is_kernel_driver_active(VENDOR_INTERFACE):
                dev.detach_kernel_driver(VENDOR_INTERFACE)
        except (usb.core.USBError, NotImplementedError):
            pass

        cfg = dev.get_active_configuration()
        intf = cfg[(VENDOR_INTERFACE, 0)]
        ep_in = usb.util.find_descriptor(
            intf,
            custom_match=lambda e: usb.util.endpoint_direction(e.bEndpointAddress)
            == usb.util.ENDPOINT_IN,
        )
        if ep_in is None:
            raise BulkReaderError("Vendor bulk IN endpoint not found")

        try:
            usb.util.claim_interface(dev, VENDOR_INTERFACE)
            self._claimed = True
        except usb.core.USBError as exc:
            raise BulkReaderError(f"Could not claim vendor interface: {exc}") from exc

        self._dev = dev
        self._ep_in = ep_in

    def drain(self, duration_ms: int = 300) -> int:
        """Discard buffered/stale records so a fresh capture starts clean.

        The MCU keeps producing while no host reads the bulk pipe (e.g. between
        `start_mcu_streams` and the capture attaching), so its TX FIFO can hold
        records with old timestamps. Reading them into a session would inflate
        the time span and wreck rate/continuity checks. We discard everything
        available for a bounded window, then let the real reader take over.
        """
        if self._ep_in is None:
            raise BulkReaderError("BulkReader not open")
        discarded = 0
        deadline = time.monotonic() + duration_ms / 1000.0
        while time.monotonic() < deadline:
            data = self.read(timeout_ms=20)
            if not data:
                break
            discarded += len(data)
        return discarded

    def read(self, timeout_ms: int = 50) -> bytes:
        """Return available bytes, or b'' on timeout."""
        if self._ep_in is None:
            raise BulkReaderError("BulkReader not open")
        try:
            data = self._ep_in.read(_READ_CHUNK, timeout=timeout_ms)
        except usb.core.USBError as exc:
            # ETIMEDOUT (no data within the window) is expected; propagate others.
            if exc.errno in (110, None) or "timeout" in str(exc).lower():
                return b""
            raise
        return bytes(data)

    def close(self) -> None:
        if self._dev is not None:
            try:
                if self._claimed:
                    usb.util.release_interface(self._dev, VENDOR_INTERFACE)
            except usb.core.USBError:
                pass
            try:
                usb.util.dispose_resources(self._dev)
            except usb.core.USBError:
                pass
        self._dev = None
        self._ep_in = None
        self._claimed = False
