// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file bulk_reader.hpp
 * @brief Async libusb bulk-IN reader for the jtsmm vendor interface.
 *
 * @details Multiple transfers stay in flight so an IN request is always armed;
 *          that keeps the MCU vendor FIFO from backing up (`tx_drops`) at high
 *          EMG rates. Mirrors `jtsmm/capture/usb_bulk.py::BulkReader` without the GIL.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

struct libusb_context;
struct libusb_device_handle;
struct libusb_transfer;

namespace jtsmm {

/**
 * @brief Exception thrown for USB open/claim/transfer setup failures.
 */
class BulkReaderError : public std::runtime_error {
public:
    explicit BulkReaderError(const std::string &msg) : std::runtime_error(msg) {}
};

/**
 * @brief Per-status transfer counters and link state (see @ref BulkReader::link_stats).
 */
struct BulkLinkStats {
    uint64_t bytes_rx = 0;       /**< Bytes handed to read() (incl. partial transfers). */
    uint64_t errors = 0;         /**< LIBUSB_TRANSFER_ERROR / TIMED_OUT completions. */
    uint64_t stalls = 0;         /**< LIBUSB_TRANSFER_STALL completions (fatal). */
    uint64_t overflows = 0;      /**< LIBUSB_TRANSFER_OVERFLOW completions. */
    uint64_t no_device = 0;      /**< LIBUSB_TRANSFER_NO_DEVICE completions. */
    uint64_t partial_bytes = 0;  /**< Bytes kept from transfers that ended in an error. */
    uint64_t lost_bytes = 0;     /**< Bytes received but discarded (cancelled mid-run). */
    uint64_t resubmit_failures = 0; /**< libusb_submit_transfer failed in the callback. */
    bool device_lost = false;    /**< No transfer left armed while running: link is dead. */
};

/**
 * @brief Multi-transfer libusb bulk-IN reader with a dedicated event thread.
 */
class BulkReader {
public:
    static constexpr uint16_t kVid = 0x2E8A;          /**< Raspberry Pi Foundation VID. */
    static constexpr uint16_t kPid = 0x0004;          /**< jtsmm HAT PID. */
    static constexpr int kVendorInterface = 2;       /**< Vendor interface number. */
    static constexpr uint8_t kEndpointIn = 0x83;     /**< Bulk IN endpoint address. */
    /**
     * 48 in-flight transfers × 16 KiB ≈ 768 KiB of kernel-side buffering so the
     * MCU vendor FIFO keeps draining under brief host CPU starvation.
     */
    static constexpr int kNumTransfers = 48;
    static constexpr int kTransferSize = 16384;

    BulkReader() = default;
    ~BulkReader();

    BulkReader(const BulkReader &) = delete;
    BulkReader &operator=(const BulkReader &) = delete;

    /** @brief Open the device, claim the vendor interface, arm transfers. */
    void open();

    /**
     * @brief Discard buffered/stale bytes for a bounded window.
     * @param duration_ms Drain window (ms).
     * @return Bytes discarded.
     */
    size_t drain(int duration_ms = 300);

    /**
     * @brief Drain until quiet for @p idle_ms or @p max_ms elapses.
     * @return Bytes discarded.
     */
    size_t drain_until_quiet(int idle_ms = 200, int max_ms = 2000);

    /**
     * @brief Pop one available chunk, or empty after @p timeout_ms.
     * @param timeout_ms Wait timeout (ms).
     * @return Chunk bytes (possibly empty).
     */
    std::vector<uint8_t> read(int timeout_ms = 50);

    /** @brief Stop the event thread and release the device. */
    void close();

    /** @brief Cumulative bytes received. */
    uint64_t bytes_rx() const { return bytes_rx_.load(); }

    /** @brief Snapshot of transfer counters and the device-lost flag. */
    BulkLinkStats link_stats() const;

    /** @brief True once no transfer is armed any more while running (unplug, stall, error storm). */
    bool device_lost() const { return device_lost_.load(); }

    /**
     * @brief libusb completion callback entry (not public API).
     * @param xfer Completed transfer.
     */
    void on_transfer(libusb_transfer *xfer);

    /** Consecutive failed completions after which a transfer is not re-armed (no RT busy loop). */
    static constexpr int kMaxConsecutiveErrors = 2 * kNumTransfers;
    /** Upper bound for reaping cancelled transfers at close. */
    static constexpr int kCloseDrainMs = 2000;

private:
    void event_loop();
    void push_chunk(const uint8_t *data, int len);
    void retire_transfer();
    void release();

    libusb_context *ctx_ = nullptr;
    libusb_device_handle *handle_ = nullptr;
    std::vector<libusb_transfer *> transfers_;
    bool claimed_ = false;
    bool detached_ = false;

    std::thread event_thread_;
    std::atomic<bool> running_{false};
    std::atomic<int> inflight_{0};
    std::atomic<uint64_t> bytes_rx_{0};
    std::atomic<uint64_t> errors_{0};
    std::atomic<uint64_t> stalls_{0};
    std::atomic<uint64_t> overflows_{0};
    std::atomic<uint64_t> no_device_{0};
    std::atomic<uint64_t> partial_bytes_{0};
    std::atomic<uint64_t> lost_bytes_{0};
    std::atomic<uint64_t> resubmit_failures_{0};
    std::atomic<int> consecutive_errors_{0};
    std::atomic<bool> device_lost_{false};
    std::atomic<bool> reaped_{true};  // every submitted transfer came back

    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::vector<uint8_t>> chunks_;
};

}  // namespace jtsmm
