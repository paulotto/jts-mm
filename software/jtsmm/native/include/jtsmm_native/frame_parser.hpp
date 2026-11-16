// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file frame_parser.hpp
 * @brief Incremental 0x55 0xAA frame parser with CRC16 resync.
 *
 * @details Direct port of `software/jtsmm/jtsmm/protocol/frame.py::FrameParser`
 *          so native and Python paths produce byte-identical records.
 */
#pragma once

#include <cstdint>
#include <deque>
#include <vector>

#include "jtsmm_native/crc16.hpp"
#include "jtsmm_native/record.hpp"

namespace jtsmm {

/**
 * @brief Streaming parser for the MCU vendor-bulk framed record format.
 */
class FrameParser {
public:
    static constexpr size_t HEADER_SIZE = 4;   /**< sync(2) + type + len */
    static constexpr size_t T_MASTER_SIZE = 8; /**< little-endian t_master_us */
    static constexpr size_t CRC_SIZE = 2;
    static constexpr size_t MAX_PAYLOAD = 240;

    /**
     * @brief Append raw bytes and return every complete, CRC-valid frame.
     * @param data Pointer to bytes; must not be null if @p len > 0.
     * @param len  Number of bytes.
     * @return Newly completed records (may be empty).
     */
    std::vector<Record> feed(const uint8_t *data, size_t len) {
        buf_.insert(buf_.end(), data, data + len);
        std::vector<Record> out;

        size_t consume = 0;
        const size_t n = buf_.size();
        while (true) {
            if (n - consume < HEADER_SIZE) {
                break;
            }
            if (buf_[consume] != 0x55 || buf_[consume + 1] != 0xAA) {
                ++consume;
                ++resyncs_;
                continue;
            }
            if (n - consume < HEADER_SIZE + T_MASTER_SIZE) {
                break;
            }
            const uint8_t rec_type = buf_[consume + 2];
            const uint8_t payload_len = buf_[consume + 3];
            if (payload_len > MAX_PAYLOAD) {
                ++consume;
                ++resyncs_;
                continue;
            }
            const size_t frame_len = HEADER_SIZE + T_MASTER_SIZE + payload_len + CRC_SIZE;
            if (n - consume < frame_len) {
                break;
            }

            // CRC covers type, len, t_master, payload (everything after sync,
            // excluding trailing CRC) — matches frame.py body slice.
            const uint8_t *body = &buf_[consume + 2];
            const size_t body_len = 1 + 1 + T_MASTER_SIZE + payload_len;
            const uint16_t crc_expected =
                static_cast<uint16_t>(buf_[consume + frame_len - 2]) |
                static_cast<uint16_t>(buf_[consume + frame_len - 1]) << 8;
            if (crc16_ccitt(body, body_len) != crc_expected) {
                ++crc_errors_;
                ++consume;
                ++resyncs_;
                continue;
            }

            Record rec;
            rec.record_type = rec_type;
            uint64_t t = 0;
            for (size_t i = 0; i < T_MASTER_SIZE; ++i) {
                t |= static_cast<uint64_t>(buf_[consume + HEADER_SIZE + i]) << (8 * i);
            }
            rec.t_master_us = t;
            const uint8_t *pl = &buf_[consume + HEADER_SIZE + T_MASTER_SIZE];
            rec.payload.assign(pl, pl + payload_len);
            out.push_back(std::move(rec));
            ++frames_ok_;
            consume += frame_len;
        }

        if (consume) {
            buf_.erase(buf_.begin(), buf_.begin() + static_cast<long>(consume));
        }
        return out;
    }

    /** @brief Cumulative CRC mismatches. */
    uint64_t crc_errors() const { return crc_errors_; }
    /** @brief Cumulative successfully parsed frames. */
    uint64_t frames_ok() const { return frames_ok_; }
    /** @brief Cumulative resync / skip events. */
    uint64_t resyncs() const { return resyncs_; }

private:
    std::vector<uint8_t> buf_;
    uint64_t crc_errors_ = 0;
    uint64_t frames_ok_ = 0;
    uint64_t resyncs_ = 0;
};

}  // namespace jtsmm
