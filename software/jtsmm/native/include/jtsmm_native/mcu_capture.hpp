// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file mcu_capture.hpp
 * @brief Threaded MCU USB capture with optional HDF5 pump and viewer tap.
 *
 * @details Mirrors `jtsmm/capture/capture_mcu.py`: a read thread keeps USB bulk
 *          transfers armed while a parse thread runs @ref FrameParser so parsing
 *          never gates the wire.
 */
#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "jtsmm_native/bulk_reader.hpp"
#include "jtsmm_native/frame_parser.hpp"
#include "jtsmm_native/hdf5_session.hpp"
#include "jtsmm_native/record.hpp"

namespace jtsmm {

/** @brief Aggregate USB/parser counters for diagnostics. */
struct CaptureStats {
    uint64_t bytes_rx = 0;    /**< Bytes from @ref BulkReader. */
    uint64_t records_rx = 0;  /**< Parsed records enqueued. */
    uint64_t crc_errors = 0;  /**< Frame CRC failures. */
    uint64_t resyncs = 0;     /**< Parser resync events. */
    /** Raw USB chunks discarded because the parse thread fell > queue_max behind.
     *  Each one desyncs the frame parser until the next 0x55AA (see resyncs). */
    uint64_t raw_chunks_dropped = 0;
    /** Parsed records discarded because the consumer fell > queue_max behind. */
    uint64_t records_dropped = 0;
    /** USB transfer counters (@ref BulkLinkStats); kept after stop(). */
    BulkLinkStats link;
};

/**
 * @brief Snapshot handed to the viewer at ~30 Hz.
 *
 * @details EMG is decimated to per-channel min/max buckets (s07 §1.4); IMU
 *          batches samples since the last poll. Raw counts are kept so Python
 *          applies the same `lsb_uv` scaling as the writer.
 */
struct ViewerFrame {
    static constexpr int kChannels = 8;
    static constexpr int kMotionStreams = 4;
    static constexpr size_t kMaxImuBatch = 512;
    size_t n_buckets = 0;
    std::vector<int32_t> emg_min;  // n_buckets * kChannels, row-major
    std::vector<int32_t> emg_max;  // n_buckets * kChannels, row-major
    // Per-bucket arithmetic mean (raw counts). The viewer's filtered/envelope
    // traces must be computed from this, not from (min+max)/2: the midpoint of
    // a min/max band is a nonlinear decimator with no anti-aliasing, so EMG
    // content above the display Nyquist folded into the "filtered" view as hash.
    std::vector<float> emg_mean;   // n_buckets * kChannels, row-major
    std::vector<uint64_t> emg_t;   // n_buckets (bucket start t_master_us)
    uint64_t emg_seen = 0;         // total EMG records observed since enable

    bool have_mta = false;
    std::array<float, 4> mta_quat{{1, 0, 0, 0}};
    uint8_t mta_quat_report_id = 0;
    uint64_t mta_t = 0;
    bool have_cra = false;
    std::array<float, 4> cra_quat{{1, 0, 0, 0}};
    uint8_t cra_quat_report_id = 0;
    uint64_t cra_t = 0;

    // Batched IMU samples since the last poll (empty when none).
    std::vector<uint64_t> batch_mta_quat_t;
    std::vector<float> batch_mta_quat;  // n * 4, w-first
    std::vector<uint8_t> batch_mta_quat_rid;
    std::vector<uint8_t> batch_mta_quat_acc;  // 255 = absent
    std::vector<uint64_t> batch_cra_quat_t;
    std::vector<float> batch_cra_quat;
    std::vector<uint8_t> batch_cra_quat_rid;
    std::vector<uint8_t> batch_cra_quat_acc;

    std::vector<uint64_t> batch_motion_t[2][kMotionStreams];
    std::vector<float> batch_motion_v[2][kMotionStreams];  // n * 3 per stream
    std::vector<uint8_t> batch_motion_acc[2][kMotionStreams];  // 255 = absent

    uint64_t trig_seen = 0;

    // Latest per-slot motion vectors in SI ([slot 0=mta/1=cra][0=accel,1=gyro,
    // 2=mag,3=linear_accel]); accel/lin m/s^2, gyro rad/s, mag uT.
    bool have_motion[2][4] = {};
    std::array<float, 3> motion[2][4] = {};

    // BNO085 calibration/accuracy (0..3) per stream; 255 = unknown.
    static constexpr int kCalQuat = 0;
    static constexpr int kCalAccel = 1;
    static constexpr int kCalGyro = 2;
    static constexpr int kCalMag = 3;
    static constexpr int kCalLin = 4;
    static constexpr int kCalStreams = 5;
    std::array<uint8_t, kCalStreams> mta_cal{{255, 255, 255, 255, 255}};
    std::array<uint8_t, kCalStreams> cra_cal{{255, 255, 255, 255, 255}};
    std::array<bool, kCalStreams> have_mta_cal{{}};
    std::array<bool, kCalStreams> have_cra_cal{{}};

    bool have_leadoff = false;
    std::array<uint8_t, 3> status{{0, 0, 0}};

    bool have_stats = false;
    std::vector<uint8_t> stats_payload;
};

/**
 * @brief Decimating accumulator updated in O(1) per record inside @ref McuCapture::pump_to_writer.
 */
struct ViewerTap {
    static constexpr int kChannels = ViewerFrame::kChannels;
    static constexpr int kMotionStreams = ViewerFrame::kMotionStreams;
    static constexpr size_t kMaxImuBatch = ViewerFrame::kMaxImuBatch;
    std::atomic<bool> enabled{false};
    int bucket_samples = 4;          // EMG records folded into one min/max bucket
    size_t max_buckets = 4096;       // cap between polls; fold overflow into last
    std::mutex mu;

    std::vector<int32_t> emg_min;    // completed buckets, row-major (n * kChannels)
    std::vector<int32_t> emg_max;
    std::vector<float> emg_mean;
    std::vector<uint64_t> emg_t;
    uint64_t emg_seen = 0;

    int cur_count = 0;
    std::array<int32_t, kChannels> cur_min{};
    std::array<int32_t, kChannels> cur_max{};
    std::array<int64_t, kChannels> cur_sum{};
    uint64_t cur_t = 0;

    bool have_mta = false;
    std::array<float, 4> mta_quat{{1, 0, 0, 0}};
    uint8_t mta_quat_report_id = 0;
    uint64_t mta_t = 0;
    bool have_cra = false;
    std::array<float, 4> cra_quat{{1, 0, 0, 0}};
    uint8_t cra_quat_report_id = 0;
    uint64_t cra_t = 0;

    std::vector<uint64_t> batch_mta_quat_t;
    std::vector<float> batch_mta_quat;
    std::vector<uint8_t> batch_mta_quat_rid;
    std::vector<uint8_t> batch_mta_quat_acc;
    std::vector<uint64_t> batch_cra_quat_t;
    std::vector<float> batch_cra_quat;
    std::vector<uint8_t> batch_cra_quat_rid;
    std::vector<uint8_t> batch_cra_quat_acc;

    std::vector<uint64_t> batch_motion_t[2][kMotionStreams];
    std::vector<float> batch_motion_v[2][kMotionStreams];
    std::vector<uint8_t> batch_motion_acc[2][kMotionStreams];

    uint64_t trig_seen = 0;

    bool have_motion[2][4] = {};
    std::array<float, 3> motion[2][4] = {};

    std::array<uint8_t, ViewerFrame::kCalStreams> mta_cal{{255, 255, 255, 255, 255}};
    std::array<uint8_t, ViewerFrame::kCalStreams> cra_cal{{255, 255, 255, 255, 255}};
    std::array<bool, ViewerFrame::kCalStreams> have_mta_cal{{}};
    std::array<bool, ViewerFrame::kCalStreams> have_cra_cal{{}};

    bool have_leadoff = false;
    std::array<uint8_t, 3> status{{0, 0, 0}};

    bool have_stats = false;
    std::vector<uint8_t> stats_payload;
};

/**
 * @brief Dual-thread MCU capture: USB read + parse → record queue / HDF5 pump.
 */
class McuCapture {
public:
    /**
     * @param queue_max Max parsed records retained before oldest are dropped.
     */
    explicit McuCapture(size_t queue_max = 65536) : queue_max_(queue_max) {}
    ~McuCapture();

    /** @brief Open USB and start read/parse threads. */
    void start();
    /** @brief Stop threads and close USB. */
    void stop();

    /**
     * @brief Pop one parsed record, waiting up to @p timeout_ms.
     * @return Record or nullopt on timeout.
     */
    std::optional<Record> next_record(int timeout_ms);

    /** @brief Snapshot of capture/parser counters (still valid after stop()). */
    CaptureStats stats() const;

    /**
     * @brief Seconds since the parser last produced a record (since start() if
     *        none yet); 0 when not started. For a "no data" watchdog.
     */
    double last_record_age_s() const;

    /** @brief Test hook: start only the parse thread (no USB) for @ref inject_raw. */
    void start_parser_only();

    /** @brief Test hook: feed raw wire bytes to the parse thread as if read from USB. */
    void inject_raw(const uint8_t *data, size_t len);

    /**
     * @brief Result of draining the parsed-record queue into HDF5.
     */
    struct PumpResult {
        uint64_t emg = 0;  /**< EMG rows appended. */
        uint64_t mta = 0;  /**< MTA IMU rows appended. */
        uint64_t cra = 0;  /**< CRA IMU rows appended. */
        uint64_t trig = 0; /**< Trigger rows appended. */
        std::optional<Record> last_stats; /**< Latest STATS record, if any. */
        /** @brief Trigger id/time for host ClockAligner. */
        struct TriggerNote {
            uint64_t t_master_us = 0;
            uint64_t trigger_id = 0;
            uint8_t stamp_source = 0; /**< TRIGGER_EVENT flags[0]: 0 legacy, 1 PIO IRQ, 2 timer. */
            bool stamp_fallback = false; /**< flags[1]: timer watchdog took over. */
        };
        std::vector<TriggerNote> triggers;
        /** @brief REC_PPS_REPLY seen this pump (clock handshake); not written to HDF5. */
        struct PpsNote {
            uint64_t t_master_us = 0;
            uint64_t pps_id = 0;
        };
        std::vector<PpsNote> pps;
        /** @brief REC_LOG text lines from the firmware (recovery, watchdog, apply
         *  warnings); the caller prints them and stores them in /events. */
        struct LogNote {
            uint64_t t_master_us = 0;
            std::string text;
        };
        std::vector<LogNote> logs;
        /** Records written to the file this pump, per wire record type. */
        std::array<uint64_t, 256> by_type{};
    };

    /**
     * @brief Drain the parsed-record queue into @p writer (no Python).
     * @param max_records Cap on records processed this call.
     */
    PumpResult pump_to_writer(Hdf5SessionWriter &writer, size_t max_records = 8192);

    /** @brief Drop queued parsed records (e.g. after settle). */
    void discard_queued_records();

    /**
     * @brief Enable/disable the in-recording viewer tap.
     * @param bucket_samples EMG records folded into one min/max bucket.
     */
    void set_viewer_tap_enabled(bool enabled, int bucket_samples = 4);

    /** @brief Pop a decimated @ref ViewerFrame (clears completed buckets). */
    ViewerFrame get_viewer_frame();

private:
    void read_loop();
    void parse_loop();
    void tap_update(const Record &rec);

    ViewerTap tap_;

    size_t queue_max_;
    // Guards reader_ against stats() racing stop()'s reset (was a use-after-free).
    mutable std::mutex reader_mu_;
    std::unique_ptr<BulkReader> reader_;
    BulkLinkStats final_link_;  // snapshot taken at stop(): bytes_rx read 0 afterwards
    FrameParser parser_;        // parse thread only; counters published below
    std::atomic<uint64_t> crc_errors_{0};
    std::atomic<uint64_t> resyncs_{0};
    std::atomic<int64_t> start_ns_{0};
    std::atomic<int64_t> last_record_ns_{0};
    std::thread read_thread_;
    std::thread parse_thread_;
    std::atomic<bool> stop_{false};

    std::mutex raw_mu_;
    std::condition_variable raw_cv_;
    std::deque<std::vector<uint8_t>> raw_chunks_;

    std::mutex rec_mu_;
    std::condition_variable rec_cv_;
    std::deque<Record> records_;
    std::atomic<uint64_t> records_rx_{0};
    std::atomic<uint64_t> raw_chunks_dropped_{0};
    std::atomic<uint64_t> records_dropped_{0};
};

}  // namespace jtsmm
