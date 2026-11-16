// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file bulk_reader.cpp
 * @brief libusb bulk-IN reader implementation.
 */

#include "jtsmm_native/bulk_reader.hpp"

#include "jtsmm_native/rt_priority.hpp"

#include <libusb-1.0/libusb.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

namespace jtsmm {

namespace {
void LIBUSB_CALL transfer_cb(libusb_transfer *xfer) {
    auto *self = static_cast<BulkReader *>(xfer->user_data);
    self->on_transfer(xfer);
}
}  // namespace

BulkReader::~BulkReader() { close(); }

void BulkReader::open() {
    if (ctx_ != nullptr) {
        return;
    }
    if (libusb_init(&ctx_) != 0) {
        ctx_ = nullptr;
        throw BulkReaderError("libusb_init failed");
    }

    handle_ = libusb_open_device_with_vid_pid(ctx_, kVid, kPid);
    if (handle_ == nullptr) {
        libusb_exit(ctx_);
        ctx_ = nullptr;
        throw BulkReaderError(
            "No jtsmm USB device (VID=0x2E8A, PID=0x0004). Is it plugged into J_USB1?");
    }

    // The kernel does not bind a driver to the vendor interface, but detach
    // defensively in case a prior run left one attached.
    if (libusb_kernel_driver_active(handle_, kVendorInterface) == 1) {
        if (libusb_detach_kernel_driver(handle_, kVendorInterface) == 0) {
            detached_ = true;
        }
    }

    if (libusb_claim_interface(handle_, kVendorInterface) != 0) {
        libusb_close(handle_);
        handle_ = nullptr;
        libusb_exit(ctx_);
        ctx_ = nullptr;
        throw BulkReaderError("Could not claim vendor interface (check udev/libusb permissions)");
    }
    claimed_ = true;

    running_.store(true);
    device_lost_.store(false);
    consecutive_errors_.store(0);
    reaped_.store(false);
    // Event thread first: if a submit fails part-way, close() can then cancel and
    // reap the transfers already in flight. Before, open() threw with transfers
    // armed and nobody handling events, and close() freed them under libusb.
    event_thread_ = std::thread(&BulkReader::event_loop, this);
    try {
        for (int i = 0; i < kNumTransfers; ++i) {
            libusb_transfer *xfer = libusb_alloc_transfer(0);
            if (xfer == nullptr) {
                throw BulkReaderError("libusb_alloc_transfer failed");
            }
            auto *buffer = new uint8_t[kTransferSize];
            libusb_fill_bulk_transfer(xfer, handle_, kEndpointIn, buffer, kTransferSize,
                                      transfer_cb, this, 0 /* no timeout; cancelled on close */);
            transfers_.push_back(xfer);
            inflight_.fetch_add(1);  // before submit: the callback may run immediately
            if (libusb_submit_transfer(xfer) != 0) {
                inflight_.fetch_sub(1);
                throw BulkReaderError("libusb_submit_transfer failed");
            }
        }
    } catch (...) {
        close();
        throw;
    }
}

void BulkReader::on_transfer(libusb_transfer *xfer) {
    const int len = xfer->actual_length;
    bool rearm = running_.load();
    bool failed = true;
    switch (xfer->status) {
        case LIBUSB_TRANSFER_COMPLETED:
            failed = false;
            consecutive_errors_.store(0);
            break;
        case LIBUSB_TRANSFER_CANCELLED:
            // Only close() cancels; bytes of a transfer cancelled mid-run are lost.
            if (len > 0 && running_.load()) {
                lost_bytes_.fetch_add(static_cast<uint64_t>(len));
            }
            rearm = false;
            failed = false;
            break;
        case LIBUSB_TRANSFER_NO_DEVICE:
            no_device_.fetch_add(1);
            rearm = false;
            break;
        case LIBUSB_TRANSFER_STALL:
            // A halted endpoint fails every resubmit at once: re-arming it spun the
            // RT event thread forever. Treat it as fatal (device_lost once all
            // transfers are retired); clearing the halt needs a re-open.
            stalls_.fetch_add(1);
            rearm = false;
            break;
        case LIBUSB_TRANSFER_OVERFLOW:
            overflows_.fetch_add(1);
            break;
        default:  // LIBUSB_TRANSFER_ERROR, LIBUSB_TRANSFER_TIMED_OUT
            errors_.fetch_add(1);
            break;
    }
    if (len > 0 && xfer->status != LIBUSB_TRANSFER_CANCELLED) {
        // Bytes that arrived before an error are real stream data; the frame
        // parser's CRC rejects anything torn. They used to be dropped uncounted.
        if (failed) {
            partial_bytes_.fetch_add(static_cast<uint64_t>(len));
        }
        push_chunk(xfer->buffer, len);
    }
    if (failed && consecutive_errors_.fetch_add(1) + 1 > kMaxConsecutiveErrors) {
        rearm = false;  // error storm: stop re-arming instead of busy-looping
    }

    if (rearm) {
        if (libusb_submit_transfer(xfer) == 0) {
            return;
        }
        resubmit_failures_.fetch_add(1);
    }
    retire_transfer();
}

void BulkReader::retire_transfer() {
    if (inflight_.fetch_sub(1) == 1 && running_.load()) {
        // Nothing armed any more while we are supposed to be reading: unplug,
        // stall or an error storm. Flag it so the recorder can stop loudly
        // instead of writing a silently truncated session.
        if (!device_lost_.exchange(true)) {
            std::fprintf(stderr,
                         "jtsmm native: USB link lost (no bulk transfer armed; "
                         "errors=%llu stalls=%llu no_device=%llu)\n",
                         static_cast<unsigned long long>(errors_.load()),
                         static_cast<unsigned long long>(stalls_.load()),
                         static_cast<unsigned long long>(no_device_.load()));
        }
        cv_.notify_all();
    }
}

void BulkReader::push_chunk(const uint8_t *data, int len) {
    bytes_rx_.fetch_add(static_cast<uint64_t>(len));
    {
        std::lock_guard<std::mutex> lock(mu_);
        chunks_.emplace_back(data, data + len);
    }
    cv_.notify_one();
}

BulkLinkStats BulkReader::link_stats() const {
    BulkLinkStats s;
    s.bytes_rx = bytes_rx_.load();
    s.errors = errors_.load();
    s.stalls = stalls_.load();
    s.overflows = overflows_.load();
    s.no_device = no_device_.load();
    s.partial_bytes = partial_bytes_.load();
    s.lost_bytes = lost_bytes_.load();
    s.resubmit_failures = resubmit_failures_.load();
    s.device_lost = device_lost_.load();
    return s;
}

void BulkReader::event_loop() {
    try_realtime_priority(20);  // Highest of the capture threads: keep the endpoint armed.
    while (running_.load()) {
        timeval tv{0, 100000};  // 100 ms wakeups so shutdown is responsive.
        libusb_handle_events_timeout_completed(ctx_, &tv, nullptr);
    }
    // Reap every transfer before close() frees it. Re-cancel on each pass: a
    // callback that saw running_ just before close() may have re-armed its
    // transfer after the first cancel. Bounded, so a wedged device cannot hang
    // the recorder's shutdown forever.
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(kCloseDrainMs);
    while (inflight_.load() > 0 && std::chrono::steady_clock::now() < deadline) {
        for (libusb_transfer *xfer : transfers_) {
            libusb_cancel_transfer(xfer);  // NOT_FOUND for idle ones is fine
        }
        timeval tv{0, 50000};
        libusb_handle_events_timeout_completed(ctx_, &tv, nullptr);
    }
    reaped_.store(inflight_.load() <= 0);
}

size_t BulkReader::drain(int duration_ms) {
    size_t discarded = 0;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        auto chunk = read(20);
        if (chunk.empty()) {
            break;
        }
        discarded += chunk.size();
    }
    return discarded;
}

size_t BulkReader::drain_until_quiet(int idle_ms, int max_ms) {
    size_t discarded = 0;
    const auto start = std::chrono::steady_clock::now();
    const auto max_deadline = start + std::chrono::milliseconds(max_ms);
    auto quiet_since = start;
    while (std::chrono::steady_clock::now() < max_deadline) {
        auto chunk = read(20);
        if (!chunk.empty()) {
            discarded += chunk.size();
            quiet_since = std::chrono::steady_clock::now();
            continue;
        }
        if (std::chrono::steady_clock::now() - quiet_since >=
            std::chrono::milliseconds(idle_ms)) {
            break;
        }
    }
    return discarded;
}

std::vector<uint8_t> BulkReader::read(int timeout_ms) {
    std::unique_lock<std::mutex> lock(mu_);
    if (chunks_.empty()) {
        cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] {
            return !chunks_.empty() || !running_.load() || device_lost_.load();
        });
    }
    if (chunks_.empty()) {
        return {};
    }
    std::vector<uint8_t> chunk = std::move(chunks_.front());
    chunks_.pop_front();
    return chunk;
}

void BulkReader::close() {
    if (ctx_ == nullptr) {
        return;
    }
    running_.store(false);
    cv_.notify_all();

    for (libusb_transfer *xfer : transfers_) {
        libusb_cancel_transfer(xfer);
    }
    if (event_thread_.joinable()) {
        event_thread_.join();
    }
    if (!reaped_.load()) {
        // A transfer never came back within kCloseDrainMs. Freeing it (or the
        // context) while libusb may still complete it is a use-after-free, so
        // leak the lot instead; the process is shutting the capture down anyway.
        std::fprintf(stderr,
                     "jtsmm native: %d USB transfer(s) did not return on close; "
                     "leaking the libusb context instead of freeing live transfers\n",
                     inflight_.load());
        transfers_.clear();
        handle_ = nullptr;
        ctx_ = nullptr;
        claimed_ = detached_ = false;
        return;
    }
    release();
}

void BulkReader::release() {
    for (libusb_transfer *xfer : transfers_) {
        delete[] xfer->buffer;
        libusb_free_transfer(xfer);
    }
    transfers_.clear();

    if (handle_ != nullptr) {
        if (claimed_) {
            libusb_release_interface(handle_, kVendorInterface);
            claimed_ = false;
        }
        if (detached_) {
            libusb_attach_kernel_driver(handle_, kVendorInterface);
            detached_ = false;
        }
        libusb_close(handle_);
        handle_ = nullptr;
    }
    libusb_exit(ctx_);
    ctx_ = nullptr;
    {
        std::lock_guard<std::mutex> lock(mu_);
        chunks_.clear();
    }
}

}  // namespace jtsmm
