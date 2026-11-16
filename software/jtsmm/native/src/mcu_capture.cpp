// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file mcu_capture.cpp
 * @brief Threaded MCU capture / pump / viewer tap.
 */

#include "jtsmm_native/mcu_capture.hpp"

#include "jtsmm_native/hdf5_session.hpp"
#include "jtsmm_native/record_decode.hpp"
#include "jtsmm_native/rt_priority.hpp"

namespace jtsmm {

namespace {

constexpr size_t kImuBatchCap = ViewerTap::kMaxImuBatch;

int64_t steady_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void trim_u64(std::vector<uint64_t> &v, size_t cap) {
    if (v.size() > cap) {
        v.erase(v.begin(), v.end() - static_cast<std::ptrdiff_t>(cap));
    }
}

void trim_f32(std::vector<float> &v, size_t cap, size_t stride) {
    const size_t max = cap * stride;
    if (v.size() > max) {
        v.erase(v.begin(), v.end() - static_cast<std::ptrdiff_t>(max));
    }
}

void trim_u8(std::vector<uint8_t> &v, size_t cap) {
    if (v.size() > cap) {
        v.erase(v.begin(), v.end() - static_cast<std::ptrdiff_t>(cap));
    }
}

void append_quat_batch(ViewerTap &tap, int slot, uint64_t t_us, const std::array<float, 4> &q,
                       uint8_t report_id, uint8_t acc, bool have_acc) {
    auto &tq = slot == 0 ? tap.batch_mta_quat_t : tap.batch_cra_quat_t;
    auto &vq = slot == 0 ? tap.batch_mta_quat : tap.batch_cra_quat;
    auto &rid = slot == 0 ? tap.batch_mta_quat_rid : tap.batch_cra_quat_rid;
    auto &aq = slot == 0 ? tap.batch_mta_quat_acc : tap.batch_cra_quat_acc;
    tq.push_back(t_us);
    for (float c : q) {
        vq.push_back(c);
    }
    rid.push_back(report_id);
    aq.push_back(have_acc ? acc : 255u);
    trim_u64(tq, kImuBatchCap);
    trim_f32(vq, kImuBatchCap, 4);
    trim_u8(rid, kImuBatchCap);
    trim_u8(aq, kImuBatchCap);
}

void append_motion_batch(ViewerTap &tap, int slot, int idx, uint64_t t_us,
                         const std::array<float, 3> &v, uint8_t acc, bool have_acc) {
    auto &tq = tap.batch_motion_t[slot][idx];
    auto &vq = tap.batch_motion_v[slot][idx];
    auto &aq = tap.batch_motion_acc[slot][idx];
    tq.push_back(t_us);
    for (float c : v) {
        vq.push_back(c);
    }
    aq.push_back(have_acc ? acc : 255u);
    trim_u64(tq, kImuBatchCap);
    trim_f32(vq, kImuBatchCap, 3);
    trim_u8(aq, kImuBatchCap);
}

}  // namespace

McuCapture::~McuCapture() { stop(); }

void McuCapture::start() {
    if (read_thread_.joinable() || parse_thread_.joinable()) {
        return;
    }
    auto reader = std::make_unique<BulkReader>();
    reader->open();
    reader->drain_until_quiet(150, 1500);
    {
        std::lock_guard<std::mutex> lock(reader_mu_);
        reader_ = std::move(reader);
        final_link_ = BulkLinkStats{};
    }
    stop_.store(false);
    start_ns_.store(steady_ns());
    last_record_ns_.store(0);
    read_thread_ = std::thread(&McuCapture::read_loop, this);
    parse_thread_ = std::thread(&McuCapture::parse_loop, this);
}

void McuCapture::start_parser_only() {
    if (read_thread_.joinable() || parse_thread_.joinable()) {
        return;
    }
    stop_.store(false);
    start_ns_.store(steady_ns());
    last_record_ns_.store(0);
    parse_thread_ = std::thread(&McuCapture::parse_loop, this);
}

void McuCapture::inject_raw(const uint8_t *data, size_t len) {
    {
        std::lock_guard<std::mutex> lock(raw_mu_);
        raw_chunks_.emplace_back(data, data + len);
    }
    raw_cv_.notify_one();
}

void McuCapture::stop() {
    stop_.store(true);
    raw_cv_.notify_all();
    rec_cv_.notify_all();
    if (read_thread_.joinable()) {
        read_thread_.join();
    }
    if (parse_thread_.joinable()) {
        parse_thread_.join();
    }
    {
        // Keep the final USB counters: the supervisor reads link_stats after
        // stop(), and bytes_rx used to come back as 0 once reader_ was reset.
        std::lock_guard<std::mutex> lock(reader_mu_);
        if (reader_) {
            reader_->close();
            final_link_ = reader_->link_stats();
            reader_.reset();
        }
    }
    {
        std::lock_guard<std::mutex> lock(raw_mu_);
        raw_chunks_.clear();
    }
    {
        std::lock_guard<std::mutex> lock(rec_mu_);
        records_.clear();
    }
}

void McuCapture::read_loop() {
    try_realtime_priority(15);  // Below the USB event thread, above the parser.
    while (!stop_.load()) {
        std::vector<uint8_t> chunk = reader_->read(50);
        if (chunk.empty()) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(raw_mu_);
            if (raw_chunks_.size() >= queue_max_) {
                // Silent loss used to be the only option; now it is at least counted
                // (CaptureStats / sync report / integrity) so a stalled host shows up.
                raw_chunks_.pop_front();
                raw_chunks_dropped_.fetch_add(1);
            }
            raw_chunks_.push_back(std::move(chunk));
        }
        raw_cv_.notify_one();
    }
}

void McuCapture::parse_loop() {
    try_realtime_priority(10);  // Lowest capture-thread priority; still above default work.
    while (true) {
        std::vector<uint8_t> chunk;
        {
            std::unique_lock<std::mutex> lock(raw_mu_);
            if (raw_chunks_.empty()) {
                if (stop_.load()) {
                    break;  // checked under raw_mu_ (was an unlocked deque read)
                }
                raw_cv_.wait_for(lock, std::chrono::milliseconds(50),
                                 [this] { return !raw_chunks_.empty() || stop_.load(); });
            }
            if (raw_chunks_.empty()) {
                continue;
            }
            chunk = std::move(raw_chunks_.front());
            raw_chunks_.pop_front();
        }
        std::vector<Record> recs = parser_.feed(chunk.data(), chunk.size());
        // FrameParser is single-threaded; publish its counters for stats().
        crc_errors_.store(parser_.crc_errors());
        resyncs_.store(parser_.resyncs());
        if (recs.empty()) {
            continue;
        }
        last_record_ns_.store(steady_ns());
        {
            std::lock_guard<std::mutex> lock(rec_mu_);
            for (auto &rec : recs) {
                if (records_.size() >= queue_max_) {
                    records_.pop_front();
                    records_dropped_.fetch_add(1);
                }
                records_.push_back(std::move(rec));
                records_rx_.fetch_add(1);
            }
        }
        rec_cv_.notify_all();
    }
}

std::optional<Record> McuCapture::next_record(int timeout_ms) {
    std::unique_lock<std::mutex> lock(rec_mu_);
    if (records_.empty() && timeout_ms > 0) {
        rec_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms),
                         [this] { return !records_.empty() || stop_.load(); });
    }
    if (records_.empty()) {
        return std::nullopt;
    }
    Record rec = std::move(records_.front());
    records_.pop_front();
    return rec;
}

CaptureStats McuCapture::stats() const {
    CaptureStats s;
    s.records_rx = records_rx_.load();
    {
        std::lock_guard<std::mutex> lock(reader_mu_);
        s.link = reader_ ? reader_->link_stats() : final_link_;
    }
    s.bytes_rx = s.link.bytes_rx;
    s.crc_errors = crc_errors_.load();
    s.resyncs = resyncs_.load();
    s.raw_chunks_dropped = raw_chunks_dropped_.load();
    s.records_dropped = records_dropped_.load();
    return s;
}

double McuCapture::last_record_age_s() const {
    const int64_t start = start_ns_.load();
    if (start == 0) {
        return 0.0;
    }
    const int64_t last = last_record_ns_.load();
    return static_cast<double>(steady_ns() - (last != 0 ? last : start)) * 1e-9;
}

void McuCapture::discard_queued_records() {
    std::lock_guard<std::mutex> lock(rec_mu_);
    records_.clear();
}

McuCapture::PumpResult McuCapture::pump_to_writer(Hdf5SessionWriter &writer, size_t max_records) {
    PumpResult out;
    std::deque<Record> batch;
    {
        std::lock_guard<std::mutex> lock(rec_mu_);
        while (!records_.empty() && batch.size() < max_records) {
            batch.push_back(std::move(records_.front()));
            records_.pop_front();
        }
    }
    const bool tap_on = tap_.enabled.load(std::memory_order_relaxed);
    for (Record &rec : batch) {
        if (tap_on) {
            tap_update(rec);
        }
        if (rec.record_type == REC_STATS) {
            out.last_stats = std::move(rec);
            continue;
        }
        if (rec.record_type == REC_TRIGGER_EVENT && rec.payload.size() >= 12) {
            PumpResult::TriggerNote note;
            note.t_master_us = rec.t_master_us;
            note.trigger_id = read_u64(rec.payload.data() + 4);
            note.stamp_source = rec.payload[0];
            note.stamp_fallback = rec.payload[1] != 0;
            out.triggers.push_back(note);
        }
        if (rec.record_type == REC_PPS_REPLY && rec.payload.size() >= 8) {
            PumpResult::PpsNote note;
            note.t_master_us = rec.t_master_us;
            note.pps_id = read_u64(rec.payload.data());
            out.pps.push_back(note);
            continue;
        }
        if (rec.record_type == REC_LOG) {
            PumpResult::LogNote note;
            note.t_master_us = rec.t_master_us;
            note.text.assign(reinterpret_cast<const char *>(rec.payload.data()), rec.payload.size());
            out.logs.push_back(std::move(note));
            continue;
        }
        if (append_record_to_session(writer, rec)) {
            out.by_type[rec.record_type]++;
            // Legacy per-slot totals (quat + raw + linear accel lumped together);
            // use by_type for per-record-type counts.
            switch (rec.record_type) {
                case REC_EMG_BLOCK:
                    out.emg++;
                    break;
                case REC_IMU_MTA_QUAT:
                case REC_IMU_MTA_RAW:
                case REC_IMU_MTA_LIN_ACCEL:
                    out.mta++;
                    break;
                case REC_IMU_CRA_QUAT:
                case REC_IMU_CRA_RAW:
                case REC_IMU_CRA_LIN_ACCEL:
                    out.cra++;
                    break;
                case REC_TRIGGER_EVENT:
                    out.trig++;
                    break;
                default:
                    break;
            }
        }
    }
    return out;
}

void McuCapture::set_viewer_tap_enabled(bool enabled, int bucket_samples) {
    std::lock_guard<std::mutex> lock(tap_.mu);
    tap_.bucket_samples = bucket_samples > 0 ? bucket_samples : 1;
    // Reset accumulators on any (re)configuration so the first frame is clean.
    tap_.emg_min.clear();
    tap_.emg_max.clear();
    tap_.emg_mean.clear();
    tap_.emg_t.clear();
    tap_.emg_seen = 0;
    tap_.cur_count = 0;
    tap_.cur_sum.fill(0);
    tap_.have_mta = false;
    tap_.have_cra = false;
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < 4; ++i) {
            tap_.have_motion[s][i] = false;
        }
    }
    tap_.have_leadoff = false;
    tap_.have_stats = false;
    tap_.stats_payload.clear();
    tap_.enabled.store(enabled, std::memory_order_relaxed);
}

void McuCapture::tap_update(const Record &rec) {
    switch (rec.record_type) {
        case REC_EMG_BLOCK: {
            if (rec.payload.size() < 27) {
                return;
            }
            const uint8_t *p = rec.payload.data();
            std::lock_guard<std::mutex> lock(tap_.mu);
            tap_.status = {p[0], p[1], p[2]};
            tap_.have_leadoff = true;
            for (int ch = 0; ch < ViewerTap::kChannels; ++ch) {
                const uint8_t *o = p + 3 + ch * 3;
                int32_t v = sign_extend24(read_emg_count24_be(o));
                if (tap_.cur_count == 0) {
                    tap_.cur_min[ch] = v;
                    tap_.cur_max[ch] = v;
                    tap_.cur_sum[ch] = v;
                } else {
                    if (v < tap_.cur_min[ch]) tap_.cur_min[ch] = v;
                    if (v > tap_.cur_max[ch]) tap_.cur_max[ch] = v;
                    tap_.cur_sum[ch] += v;
                }
            }
            if (tap_.cur_count == 0) {
                tap_.cur_t = rec.t_master_us;
            }
            tap_.cur_count++;
            tap_.emg_seen++;
            if (tap_.cur_count >= tap_.bucket_samples) {
                const size_t nb = tap_.emg_t.size();
                if (nb >= tap_.max_buckets && nb > 0) {
                    // Consumer is lagging: fold this bucket into the last one so we
                    // keep the peak envelope without growing unboundedly.
                    const size_t base = (nb - 1) * ViewerTap::kChannels;
                    const float inv = 1.0f / static_cast<float>(tap_.cur_count);
                    for (int ch = 0; ch < ViewerTap::kChannels; ++ch) {
                        if (tap_.cur_min[ch] < tap_.emg_min[base + ch])
                            tap_.emg_min[base + ch] = tap_.cur_min[ch];
                        if (tap_.cur_max[ch] > tap_.emg_max[base + ch])
                            tap_.emg_max[base + ch] = tap_.cur_max[ch];
                        // Fold: average of the two bucket means (equal weights; the
                        // consumer is lagging, exactness does not matter here).
                        tap_.emg_mean[base + ch] =
                            0.5f * (tap_.emg_mean[base + ch] +
                                    static_cast<float>(tap_.cur_sum[ch]) * inv);
                    }
                } else {
                    const float inv = 1.0f / static_cast<float>(tap_.cur_count);
                    for (int ch = 0; ch < ViewerTap::kChannels; ++ch) {
                        tap_.emg_min.push_back(tap_.cur_min[ch]);
                        tap_.emg_max.push_back(tap_.cur_max[ch]);
                        tap_.emg_mean.push_back(static_cast<float>(tap_.cur_sum[ch]) * inv);
                    }
                    tap_.emg_t.push_back(tap_.cur_t);
                }
                tap_.cur_count = 0;
            }
            break;
        }
        case REC_IMU_MTA_QUAT:
        case REC_IMU_CRA_QUAT: {
            const int slot = (rec.record_type == REC_IMU_CRA_QUAT) ? 1 : 0;
            const uint8_t *p = rec.payload.data();
            const size_t n = rec.payload.size();
            std::array<float, 4> q;
            uint8_t report_id = 0;
            if (n >= 17) {
                report_id = p[0];
                q = {read_f32(p + 1), read_f32(p + 5), read_f32(p + 9), read_f32(p + 13)};
            } else if (n >= 16) {
                q = {read_f32(p), read_f32(p + 4), read_f32(p + 8), read_f32(p + 12)};
            } else {
                return;
            }
            std::lock_guard<std::mutex> lock(tap_.mu);
            if (slot == 0) {
                tap_.mta_quat = q;
                tap_.mta_quat_report_id = report_id;
                tap_.mta_t = rec.t_master_us;
                tap_.have_mta = true;
                if (n >= 18) {
                    tap_.mta_cal[ViewerFrame::kCalQuat] = static_cast<uint8_t>(p[17] & 3u);
                    tap_.have_mta_cal[ViewerFrame::kCalQuat] = true;
                }
            } else {
                tap_.cra_quat = q;
                tap_.cra_quat_report_id = report_id;
                tap_.cra_t = rec.t_master_us;
                tap_.have_cra = true;
                if (n >= 18) {
                    tap_.cra_cal[ViewerFrame::kCalQuat] = static_cast<uint8_t>(p[17] & 3u);
                    tap_.have_cra_cal[ViewerFrame::kCalQuat] = true;
                }
            }
            const uint8_t acc = (n >= 18) ? static_cast<uint8_t>(p[17] & 3u) : 255u;
            append_quat_batch(tap_, slot, rec.t_master_us, q, report_id, acc, n >= 18);
            break;
        }
        case REC_IMU_MTA_RAW:
        case REC_IMU_CRA_RAW: {
            const int slot = (rec.record_type == REC_IMU_CRA_RAW) ? 1 : 0;
            const uint8_t *p = rec.payload.data();
            const size_t n = rec.payload.size();
            std::lock_guard<std::mutex> lock(tap_.mu);
            auto store = [&](int idx, float x, float y, float z, uint8_t acc, bool have_acc) {
                tap_.motion[slot][idx] = {x, y, z};
                tap_.have_motion[slot][idx] = true;
                append_motion_batch(tap_, slot, idx, rec.t_master_us, tap_.motion[slot][idx], acc,
                                    have_acc);
            };
            if (n >= 7 && n < 18) {
                const uint8_t sid = p[0];
                const int16_t x = read_i16(p + 1), y = read_i16(p + 3), z = read_i16(p + 5);
                const uint8_t acc = (n >= 8) ? static_cast<uint8_t>(p[7] & 3u) : 255u;
                const bool have_acc = n >= 8;
                if (sid == 0x01) {
                    store(0, accel_count_to_si(x), accel_count_to_si(y), accel_count_to_si(z), acc,
                          have_acc);
                    if (n >= 8) {
                        if (slot == 0) {
                            tap_.mta_cal[ViewerFrame::kCalAccel] = acc;
                            tap_.have_mta_cal[ViewerFrame::kCalAccel] = true;
                        } else {
                            tap_.cra_cal[ViewerFrame::kCalAccel] = acc;
                            tap_.have_cra_cal[ViewerFrame::kCalAccel] = true;
                        }
                    }
                } else if (sid == 0x02) {
                    store(1, gyro_count_to_si(x), gyro_count_to_si(y), gyro_count_to_si(z), acc,
                          have_acc);
                    if (n >= 8) {
                        if (slot == 0) {
                            tap_.mta_cal[ViewerFrame::kCalGyro] = acc;
                            tap_.have_mta_cal[ViewerFrame::kCalGyro] = true;
                        } else {
                            tap_.cra_cal[ViewerFrame::kCalGyro] = acc;
                            tap_.have_cra_cal[ViewerFrame::kCalGyro] = true;
                        }
                    }
                } else if (sid == 0x03) {
                    store(2, mag_count_to_si(x), mag_count_to_si(y), mag_count_to_si(z), acc,
                          have_acc);
                    if (n >= 8) {
                        if (slot == 0) {
                            tap_.mta_cal[ViewerFrame::kCalMag] = acc;
                            tap_.have_mta_cal[ViewerFrame::kCalMag] = true;
                        } else {
                            tap_.cra_cal[ViewerFrame::kCalMag] = acc;
                            tap_.have_cra_cal[ViewerFrame::kCalMag] = true;
                        }
                    }
                }
            } else if (n >= 18) {
                store(0, accel_count_to_si(read_i16(p)), accel_count_to_si(read_i16(p + 2)),
                      accel_count_to_si(read_i16(p + 4)), 255u, false);
                store(1, gyro_count_to_si(read_i16(p + 6)), gyro_count_to_si(read_i16(p + 8)),
                      gyro_count_to_si(read_i16(p + 10)), 255u, false);
                store(2, mag_count_to_si(read_i16(p + 12)), mag_count_to_si(read_i16(p + 14)),
                      mag_count_to_si(read_i16(p + 16)), 255u, false);
            }
            break;
        }
        case REC_IMU_MTA_LIN_ACCEL:
        case REC_IMU_CRA_LIN_ACCEL: {
            if (rec.payload.size() < 12) return;
            const int slot = (rec.record_type == REC_IMU_CRA_LIN_ACCEL) ? 1 : 0;
            const uint8_t *p = rec.payload.data();
            const size_t n = rec.payload.size();
            std::lock_guard<std::mutex> lock(tap_.mu);
            tap_.motion[slot][3] = {read_f32(p), read_f32(p + 4), read_f32(p + 8)};
            tap_.have_motion[slot][3] = true;
            const uint8_t acc = (n >= 13) ? static_cast<uint8_t>(p[12] & 3u) : 255u;
            append_motion_batch(tap_, slot, 3, rec.t_master_us, tap_.motion[slot][3], acc, n >= 13);
            if (n >= 13) {
                const uint8_t acc = static_cast<uint8_t>(p[12] & 3u);
                if (slot == 0) {
                    tap_.mta_cal[ViewerFrame::kCalLin] = acc;
                    tap_.have_mta_cal[ViewerFrame::kCalLin] = true;
                } else {
                    tap_.cra_cal[ViewerFrame::kCalLin] = acc;
                    tap_.have_cra_cal[ViewerFrame::kCalLin] = true;
                }
            }
            break;
        }
        case REC_TRIGGER_EVENT: {
            std::lock_guard<std::mutex> lock(tap_.mu);
            tap_.trig_seen++;
            break;
        }
        case REC_STATS: {
            std::lock_guard<std::mutex> lock(tap_.mu);
            tap_.stats_payload = rec.payload;
            tap_.have_stats = true;
            break;
        }
        default:
            break;
    }
}

ViewerFrame McuCapture::get_viewer_frame() {
    ViewerFrame f;
    std::lock_guard<std::mutex> lock(tap_.mu);
    // Flush any partial bucket so the tail sample is not lost between polls.
    if (tap_.cur_count > 0) {
        const float inv = 1.0f / static_cast<float>(tap_.cur_count);
        for (int ch = 0; ch < ViewerTap::kChannels; ++ch) {
            tap_.emg_min.push_back(tap_.cur_min[ch]);
            tap_.emg_max.push_back(tap_.cur_max[ch]);
            tap_.emg_mean.push_back(static_cast<float>(tap_.cur_sum[ch]) * inv);
        }
        tap_.emg_t.push_back(tap_.cur_t);
        tap_.cur_count = 0;
    }
    f.n_buckets = tap_.emg_t.size();
    f.emg_min = std::move(tap_.emg_min);
    f.emg_max = std::move(tap_.emg_max);
    f.emg_mean = std::move(tap_.emg_mean);
    f.emg_t = std::move(tap_.emg_t);
    f.emg_seen = tap_.emg_seen;
    tap_.emg_min.clear();
    tap_.emg_max.clear();
    tap_.emg_mean.clear();
    tap_.emg_t.clear();

    f.have_mta = tap_.have_mta;
    f.mta_quat = tap_.mta_quat;
    f.mta_quat_report_id = tap_.mta_quat_report_id;
    f.mta_t = tap_.mta_t;
    f.have_cra = tap_.have_cra;
    f.cra_quat = tap_.cra_quat;
    f.cra_quat_report_id = tap_.cra_quat_report_id;
    f.cra_t = tap_.cra_t;
    f.batch_mta_quat_t = std::move(tap_.batch_mta_quat_t);
    f.batch_mta_quat = std::move(tap_.batch_mta_quat);
    f.batch_mta_quat_rid = std::move(tap_.batch_mta_quat_rid);
    f.batch_mta_quat_acc = std::move(tap_.batch_mta_quat_acc);
    f.batch_cra_quat_t = std::move(tap_.batch_cra_quat_t);
    f.batch_cra_quat = std::move(tap_.batch_cra_quat);
    f.batch_cra_quat_rid = std::move(tap_.batch_cra_quat_rid);
    f.batch_cra_quat_acc = std::move(tap_.batch_cra_quat_acc);
    tap_.batch_mta_quat_t.clear();
    tap_.batch_mta_quat.clear();
    tap_.batch_mta_quat_rid.clear();
    tap_.batch_mta_quat_acc.clear();
    tap_.batch_cra_quat_t.clear();
    tap_.batch_cra_quat.clear();
    tap_.batch_cra_quat_rid.clear();
    tap_.batch_cra_quat_acc.clear();
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < ViewerFrame::kMotionStreams; ++i) {
            f.batch_motion_t[s][i] = std::move(tap_.batch_motion_t[s][i]);
            f.batch_motion_v[s][i] = std::move(tap_.batch_motion_v[s][i]);
            f.batch_motion_acc[s][i] = std::move(tap_.batch_motion_acc[s][i]);
            tap_.batch_motion_t[s][i].clear();
            tap_.batch_motion_v[s][i].clear();
            tap_.batch_motion_acc[s][i].clear();
        }
    }
    f.trig_seen = tap_.trig_seen;
    tap_.trig_seen = 0;
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < 4; ++i) {
            f.have_motion[s][i] = tap_.have_motion[s][i];
            f.motion[s][i] = tap_.motion[s][i];
        }
    }
    f.mta_cal = tap_.mta_cal;
    f.cra_cal = tap_.cra_cal;
    f.have_mta_cal = tap_.have_mta_cal;
    f.have_cra_cal = tap_.have_cra_cal;
    f.have_leadoff = tap_.have_leadoff;
    f.status = tap_.status;
    f.have_stats = tap_.have_stats;
    f.stats_payload = tap_.stats_payload;
    return f;
}

}  // namespace jtsmm
