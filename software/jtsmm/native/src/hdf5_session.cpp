// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file hdf5_session.cpp
 * @brief libhdf5 session writer implementation.
 */

#include "jtsmm_native/hdf5_session.hpp"

#include "jtsmm_native/record_decode.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <exception>
#include <stdexcept>

namespace jtsmm {

namespace {

constexpr hsize_t kChunkRows = 1024;
// Keep in sync with ACCURACY_NOTE in jtsmm/io/hdf5_writer.py.
constexpr const char *kAccuracyNote =
    "BNO085 calibration status per sample: 0 unreliable, 1 low, 2 medium, 3 high; "
    "255 = not sent (older firmware)";

void check(int64_t status, const char *what) {
    if (status < 0) {
        throw std::runtime_error(std::string("HDF5 error: ") + what);
    }
}

/** Closes an HDF5 handle on scope exit so a throw cannot leak it (a leaked
 *  group/dataspace would keep the file open after H5Fclose). */
class H5Id {
public:
    H5Id(hid_t id, herr_t (*closer)(hid_t)) : id_(id), closer_(closer) {}
    ~H5Id() {
        if (id_ >= 0) {
            closer_(id_);
        }
    }
    H5Id(const H5Id &) = delete;
    H5Id &operator=(const H5Id &) = delete;
    operator hid_t() const { return id_; }
    hid_t get() const { return id_; }

private:
    hid_t id_;
    herr_t (*closer_)(hid_t);
};

double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

/** Create-or-replace attribute @p name. H5Acreate2 fails on an existing name, which
 *  used to lose e.g. a second sync report or an extra_meta key colliding with a
 *  built-in one without any error. */
void put_attr(hid_t loc, const char *name, hid_t type, hid_t space, const void *buf) {
    const htri_t exists = H5Aexists(loc, name);
    check(exists, name);
    if (exists > 0) {
        check(H5Adelete(loc, name), name);
    }
    H5Id attr(H5Acreate2(loc, name, type, space, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    check(attr.get(), name);
    check(H5Awrite(attr, type, buf), name);
}

void write_str_attr(hid_t loc, const char *name, const std::string &value) {
    // Variable-length UTF-8 to match how h5py stores Python str attrs, so both
    // writers read back identically as str (not fixed-length bytes).
    H5Id stype(H5Tcopy(H5T_C_S1), H5Tclose);
    check(stype.get(), name);
    check(H5Tset_size(stype, H5T_VARIABLE), name);
    check(H5Tset_cset(stype, H5T_CSET_UTF8), name);
    H5Id space(H5Screate(H5S_SCALAR), H5Sclose);
    check(space.get(), name);
    const char *ptr = value.c_str();
    put_attr(loc, name, stype, space, &ptr);
}

void write_scalar_attr(hid_t loc, const char *name, hid_t type, const void *value) {
    H5Id space(H5Screate(H5S_SCALAR), H5Sclose);
    check(space.get(), name);
    put_attr(loc, name, type, space, value);
}

void write_int_attr(hid_t loc, const char *name, long long value) {
    write_scalar_attr(loc, name, H5T_NATIVE_LLONG, &value);
}

void write_double_attr(hid_t loc, const char *name, double value) {
    write_scalar_attr(loc, name, H5T_NATIVE_DOUBLE, &value);
}

void write_doublevec_attr(hid_t loc, const char *name, const std::vector<double> &values) {
    hsize_t dim = values.size();
    H5Id space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
    check(space.get(), name);
    put_attr(loc, name, H5T_NATIVE_DOUBLE, space, values.data());
}

void write_strvec_attr(hid_t loc, const char *name, const std::vector<std::string> &values) {
    if (values.empty()) {
        return;
    }
    size_t maxlen = 1;
    for (const auto &s : values) {
        maxlen = std::max(maxlen, s.size() + 1);
    }
    H5Id stype(H5Tcopy(H5T_C_S1), H5Tclose);
    check(stype.get(), name);
    check(H5Tset_size(stype, maxlen), name);
    check(H5Tset_strpad(stype, H5T_STR_NULLPAD), name);
    std::vector<char> buf(values.size() * maxlen, 0);
    for (size_t i = 0; i < values.size(); ++i) {
        std::copy(values[i].begin(), values[i].end(), buf.begin() + i * maxlen);
    }
    hsize_t dim = values.size();
    H5Id space(H5Screate_simple(1, &dim, nullptr), H5Sclose);
    check(space.get(), name);
    put_attr(loc, name, stype, space, buf.data());
}

void make_group(hid_t file, const char *name) {
    H5Id g(H5Gcreate2(file, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
    check(g.get(), name);
}

hid_t open_group(hid_t parent, const char *name) {
    hid_t g = H5Gopen2(parent, name, H5P_DEFAULT);
    check(g, name);
    return g;
}

hid_t open_or_create_group(hid_t parent, const char *name) {
    const htri_t exists = H5Lexists(parent, name, H5P_DEFAULT);
    check(exists, name);
    if (exists > 0) {
        return open_group(parent, name);
    }
    hid_t g = H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    check(g, name);
    return g;
}

// Extendable, chunked dataset with initial 0 rows. ncols==1 -> rank-1 dataset.
hid_t make_ext_dataset(hid_t loc, const char *name, hid_t storage_type, hsize_t ncols) {
    const int rank = (ncols <= 1) ? 1 : 2;
    hsize_t dims[2] = {0, ncols};
    hsize_t maxdims[2] = {H5S_UNLIMITED, ncols};
    hsize_t chunk[2] = {kChunkRows, ncols};
    H5Id space(H5Screate_simple(rank, dims, maxdims), H5Sclose);
    check(space.get(), name);
    H5Id dcpl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose);
    check(dcpl.get(), name);
    check(H5Pset_chunk(dcpl, rank, chunk), name);
    hid_t ds = H5Dcreate2(loc, name, storage_type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
    check(ds, name);
    return ds;
}

void append_rows(hid_t ds, const void *data, hsize_t nrows, hsize_t ncols, hid_t memtype) {
    const int rank = (ncols <= 1) ? 1 : 2;
    hsize_t cur[2] = {0, 0};
    {
        H5Id fspace(H5Dget_space(ds), H5Sclose);
        check(fspace.get(), "get_space");
        check(H5Sget_simple_extent_dims(fspace, cur, nullptr), "get_dims");
    }

    hsize_t newdims[2] = {cur[0] + nrows, ncols};
    check(H5Dset_extent(ds, newdims), "set_extent");

    bool ok = false;
    {
        H5Id fspace(H5Dget_space(ds), H5Sclose);
        hsize_t start[2] = {cur[0], 0};
        hsize_t count[2] = {nrows, ncols};
        H5Id mspace(H5Screate_simple(rank, count, nullptr), H5Sclose);
        ok = fspace.get() >= 0 && mspace.get() >= 0 &&
             H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr) >= 0 &&
             H5Dwrite(ds, memtype, mspace, fspace, H5P_DEFAULT, data) >= 0;
    }
    if (!ok) {
        // Undo the extent so a retry does not leave fill-value rows behind.
        hsize_t olddims[2] = {cur[0], ncols};
        H5Dset_extent(ds, olddims);
        throw std::runtime_error("HDF5 error: write");
    }
}

// Variable-length UTF-8 string type: what h5py.string_dtype() stores, so the
// /events string datasets read back as str from both writers.
hid_t vlen_utf8_type() {
    hid_t t = H5Tcopy(H5T_C_S1);
    check(t, "H5Tcopy");
    if (H5Tset_size(t, H5T_VARIABLE) < 0 || H5Tset_cset(t, H5T_CSET_UTF8) < 0) {
        H5Tclose(t);
        throw std::runtime_error("HDF5 error: vlen string type");
    }
    return t;
}

// Overwrite element @p row of a rank-1 dataset.
void write_cell(hid_t ds, hsize_t row, hid_t memtype, const void *buf) {
    H5Id fspace(H5Dget_space(ds), H5Sclose);
    check(fspace.get(), "get_space");
    hsize_t start[1] = {row};
    hsize_t count[1] = {1};
    check(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, start, nullptr, count, nullptr), "select");
    H5Id mspace(H5Screate_simple(1, count, nullptr), H5Sclose);
    check(mspace.get(), "mspace");
    check(H5Dwrite(ds, memtype, mspace, fspace, H5P_DEFAULT, buf), "write cell");
}

std::vector<const char *> c_strs(const std::vector<std::string> &v) {
    std::vector<const char *> out;
    out.reserve(v.size());
    for (const auto &s : v) {
        out.push_back(s.c_str());
    }
    return out;
}

double lsb_uv(int gain) {
    const int g = gain ? gain : 1;  // mirrors catalog.lsb_uv
    return (4.5 / (static_cast<double>(g) * static_cast<double>(1 << 23))) * 1e6;
}

}  // namespace

Hdf5SessionWriter::Hdf5SessionWriter(Hdf5SessionConfig cfg) : cfg_(std::move(cfg)) {
    lsb_uv_ = lsb_uv(cfg_.emg_gain);
    if (cfg_.lsb_uv_per_channel.empty()) {
        cfg_.lsb_uv_per_channel.assign(8, lsb_uv_);
    }
    if (cfg_.lsb_uv_per_channel.size() != 8) {
        throw std::invalid_argument("lsb_uv_per_channel must have 8 entries");
    }
    for (double v : cfg_.lsb_uv_per_channel) {
        lsb_col_f_.push_back(static_cast<float>(v));
    }
}

Hdf5SessionWriter::~Hdf5SessionWriter() noexcept {
    // A destructor that throws (disk full at the final flush) is std::terminate.
    try {
        close();
    } catch (...) {
    }
}

void Hdf5SessionWriter::open() {
    if (file_ >= 0) {
        throw std::logic_error("Hdf5SessionWriter already open");
    }
    {
        H5Id fapl(H5Pcreate(H5P_FILE_ACCESS), H5Pclose);
        check(fapl.get(), "H5Pcreate");
        check(H5Pset_libver_bounds(fapl, H5F_LIBVER_LATEST, H5F_LIBVER_LATEST), "libver");
        // H5Fclose then closes anything still open instead of silently keeping the
        // file alive (and unflushed) behind a leaked handle.
        check(H5Pset_fclose_degree(fapl, H5F_CLOSE_STRONG), "fclose_degree");
        file_ = H5Fcreate(cfg_.path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        check(file_, "H5Fcreate");
    }
    opened_at_ = static_cast<double>(std::time(nullptr));

    {
        H5Id meta(H5Gcreate2(file_, "meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose);
        check(meta.get(), "meta");
        write_str_attr(meta, "software_version", cfg_.software_version);
        write_int_attr(meta, "schema_version", cfg_.schema_version);
        write_str_attr(meta, "jts_schema_version", "1.1");
        write_double_attr(meta, "created_unix_utc", opened_at_);
        write_str_attr(meta, "config_hash", cfg_.config_hash);
        write_str_attr(meta, "mcu_config_json", cfg_.mcu_config_json);
        write_str_attr(meta, "host_config_json", cfg_.host_config_json);
        if (!cfg_.firmware_version.empty()) {
            write_str_attr(meta, "firmware_version", cfg_.firmware_version);
        }
        write_str_attr(meta, "mode", cfg_.mode);
        {
            // Human-readable recording date/time alongside `created_unix_utc`.
            const std::time_t t = static_cast<std::time_t>(opened_at_);
            std::tm tmv{};
            gmtime_r(&t, &tmv);
            char buf[32];
            std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
            write_str_attr(meta, "recording_date", std::string(buf));
        }
        for (const auto &[key, value] : cfg_.extra_meta) {
            write_str_attr(meta, key.c_str(), value);
        }
    }

    make_group(file_, "emg");
    make_group(file_, "imu_mta");
    make_group(file_, "imu_cra");
    make_group(file_, "cameras");
    make_group(file_, "triggers");
    make_group(file_, "sync");
    make_group(file_, "events");

    // Get a readable skeleton on disk right away, so a crash in the first
    // interval still leaves a valid (empty) session file.
    check(H5Fflush(file_, H5F_SCOPE_LOCAL), "H5Fflush");
    last_disk_flush_ = steady_s();
}

void Hdf5SessionWriter::append_emg(uint64_t t_master_us, const int32_t channels[8],
                                   uint8_t leadoff_statp, uint8_t leadoff_statn, bool has_seq,
                                   uint32_t seq) {
    // The first EMG row decides whether this session gets /emg/seq (the firmware
    // format does not change mid-take); later rows keep it row-aligned.
    if (emg_has_seq_ < 0) {
        emg_has_seq_ = has_seq ? 1 : 0;
    }
    if (emg_has_seq_ == 1) {
        emg_seq_.push_back(has_seq ? seq : 0);
    }
    emg_samples_.insert(emg_samples_.end(), channels, channels + 8);
    emg_t_.push_back(t_master_us);
    emg_loff_.push_back(static_cast<uint8_t>(leadoff_statp & 0x3Fu));
    emg_loff_.push_back(static_cast<uint8_t>(leadoff_statn & 0x3Fu));
}

void Hdf5SessionWriter::append_quat(bool cranium, uint64_t t_master_us, const float quat[4],
                                    uint8_t report_id, uint8_t accuracy) {
    const int slot = cranium ? 1 : 0;
    imu_q_[slot].insert(imu_q_[slot].end(), quat, quat + 4);
    imu_t_[slot].push_back(t_master_us);
    imu_qid_[slot].push_back(report_id);
    imu_qacc_[slot].push_back(accuracy);
}

void Hdf5SessionWriter::append_imu_motion(bool cranium, uint8_t sensor_id, uint64_t t_master_us,
                                          const float xyz[3], uint8_t accuracy) {
    const int slot = cranium ? 1 : 0;
    int idx;
    switch (sensor_id) {
        case 0x01: idx = 0; break;  // accel
        case 0x02: idx = 1; break;  // gyro
        case 0x03: idx = 2; break;  // mag
        default: return;
    }
    MotionBuf &b = imu_motion_[slot][idx];
    b.data.insert(b.data.end(), xyz, xyz + 3);
    b.t.push_back(t_master_us);
    b.acc.push_back(accuracy);
}

void Hdf5SessionWriter::append_lin_accel(bool cranium, uint64_t t_master_us, const float xyz[3],
                                         uint8_t accuracy) {
    const int slot = cranium ? 1 : 0;
    MotionBuf &b = imu_motion_[slot][3];  // linear_accel
    b.data.insert(b.data.end(), xyz, xyz + 3);
    b.t.push_back(t_master_us);
    b.acc.push_back(accuracy);
}

void Hdf5SessionWriter::append_trigger(uint64_t t_master_us, uint64_t trigger_id) {
    trig_id_.push_back(trigger_id);
    trig_t_.push_back(t_master_us);
}

void Hdf5SessionWriter::append_camera(const std::string &cam_id, uint64_t t_master_us,
                                      uint64_t sensor_timestamp_ns, uint64_t trigger_id,
                                      uint64_t frame_idx, const std::string &video_file) {
    CamBuffers &slot = cam_[cam_id];
    slot.t.push_back(t_master_us);
    slot.sensor_ts.push_back(sensor_timestamp_ns);
    slot.trig_id.push_back(trigger_id);
    slot.frame_idx.push_back(frame_idx);
    if (!video_file.empty() && slot.video_file.empty()) {
        slot.video_file = video_file;
    }
}

uint64_t Hdf5SessionWriter::append_event(uint64_t t_master_us, const std::string &kind,
                                         const std::string &text, const std::string &phase_kind,
                                         uint64_t t_host_ns, const std::string &map_quality,
                                         double viewer_t_s) {
    ev_t_.push_back(t_master_us);
    ev_kind_.push_back(kind);
    ev_text_.push_back(text);
    ev_phase_kind_.push_back(phase_kind);
    ev_host_.push_back(t_host_ns);
    ev_quality_.push_back(map_quality);
    ev_view_t_.push_back(viewer_t_s);
    return ev_flushed_ + ev_t_.size() - 1;
}

void Hdf5SessionWriter::set_event_time(uint64_t row, uint64_t t_master_us,
                                       const std::string &map_quality) {
    if (row >= ev_flushed_ + ev_t_.size()) {
        throw std::out_of_range("no /events row " + std::to_string(row));
    }
    if (row >= ev_flushed_) {
        const size_t i = static_cast<size_t>(row - ev_flushed_);
        ev_t_[i] = t_master_us;
        ev_quality_[i] = map_quality;
    }
    // Rows already in the datasets: flushed ones, plus those a partial
    // (failed) flush landed before the retry.
    if (row < ev_flushed_ + ev_t_col_.done) {
        write_cell(ev_t_col_.ds, row, H5T_NATIVE_UINT64, &t_master_us);
    }
    if (row < ev_flushed_ + ev_quality_col_.done) {
        H5Id st(vlen_utf8_type(), H5Tclose);
        const char *ptr = map_quality.c_str();
        write_cell(ev_quality_col_.ds, row, st, &ptr);
    }
}

// Append the buffered rows [c.done, n_total) to c.ds. After a partial group flush
// (e.g. `samples` written, `samples_uv` failed) the retry resumes each dataset
// where it stopped instead of appending the whole buffer again.
void Hdf5SessionWriter::append_col(Col &c, const void *base, size_t row_bytes, size_t n_total,
                                   hsize_t ncols, hid_t memtype) {
    if (c.done >= n_total) {
        return;
    }
    if (fail_countdown_ > 0 && --fail_countdown_ == 0) {
        throw std::runtime_error("HDF5 error: injected append failure");
    }
    append_rows(c.ds, static_cast<const uint8_t *>(base) + c.done * row_bytes, n_total - c.done,
                ncols, memtype);
    c.done = n_total;
}

void Hdf5SessionWriter::flush_emg() {
    if (emg_t_.empty()) {
        return;
    }
    const size_t n = emg_t_.size();
    H5Id g(open_group(file_, "emg"), H5Gclose);
    if (emg_samples_col_.ds < 0) emg_samples_col_.ds = make_ext_dataset(g, "samples", H5T_STD_I32LE, 8);
    if (emg_uv_col_.ds < 0) emg_uv_col_.ds = make_ext_dataset(g, "samples_uv", H5T_IEEE_F32LE, 8);
    if (emg_t_col_.ds < 0) emg_t_col_.ds = make_ext_dataset(g, "t_master_us", H5T_STD_U64LE, 1);
    if (emg_loff_col_.ds < 0) emg_loff_col_.ds = make_ext_dataset(g, "leadoff_status", H5T_STD_U8LE, 2);
    if (emg_has_seq_ == 1 && emg_seq_col_.ds < 0) {
        emg_seq_col_.ds = make_ext_dataset(g, "seq", H5T_STD_U32LE, 1);
    }
    if (!emg_attrs_done_) {
        write_int_attr(g, "sample_rate_sps", cfg_.emg_sample_rate_sps);
        write_int_attr(g, "gain", cfg_.emg_gain);
        // Scalar = muscle-channel LSB (kept for old readers); per-column values
        // cover CH7/CH8, which run at a fixed gain.
        write_double_attr(g, "lsb_uv", lsb_uv_);
        write_doublevec_attr(g, "lsb_uv_per_channel", cfg_.lsb_uv_per_channel);
        write_strvec_attr(g, "channel_names", cfg_.channel_names);
        write_str_attr(
            g, "leadoff_status_note",
            "columns=[statp_mask,statn_mask] from the ADS1299 status word; "
            "bit i = CH(i+1), muscle CH1-6 only (bits 6-7 always 0). "
            "1 = lead-off comparator tripped (that input/leg reads as "
            "disconnected). All-zero for the whole session unless "
            "mcu.emg.leadoff.during_recording was true (firmware clears the "
            "LOFF registers otherwise, so the bits are stale, not 'connected'). "
            "DRL (CH8/IN8N) is intentionally excluded: its comparator shares a "
            "current-return path with the six muscle channels' own source+sink "
            "pairs, which masks a genuine DRL disconnect while they are live -- "
            "use `jtsmm emg check` (isolates DRL by muting CH1-6 first) for "
            "DRL contact instead.");
        emg_attrs_done_ = true;
    }
    append_col(emg_samples_col_, emg_samples_.data(), 8 * sizeof(int32_t), n, 8, H5T_NATIVE_INT32);
    if (emg_uv_col_.done < n) {
        // samples_uv[:, c] = float32(sample) * float32(lsb[c]) in float32 math;
        // the Python writer (io/hdf5_writer.py) computes the same float32
        // product, so both files hold bit-identical samples_uv.
        std::vector<float> uv(emg_samples_.size());
        for (size_t i = 0; i < emg_samples_.size(); ++i) {
            uv[i] = static_cast<float>(emg_samples_[i]) * lsb_col_f_[i % 8];
        }
        append_col(emg_uv_col_, uv.data(), 8 * sizeof(float), n, 8, H5T_NATIVE_FLOAT);
    }
    append_col(emg_t_col_, emg_t_.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
    append_col(emg_loff_col_, emg_loff_.data(), 2, n, 2, H5T_NATIVE_UINT8);
    if (emg_seq_col_.ds >= 0) {
        append_col(emg_seq_col_, emg_seq_.data(), sizeof(uint32_t), n, 1, H5T_NATIVE_UINT32);
    }
    emg_samples_.clear();
    emg_t_.clear();
    emg_loff_.clear();
    emg_seq_.clear();
    emg_samples_col_.done = emg_uv_col_.done = emg_t_col_.done = emg_loff_col_.done = 0;
    emg_seq_col_.done = 0;
}

void Hdf5SessionWriter::flush_imu(int slot) {
    if (imu_t_[slot].empty()) {
        return;
    }
    const size_t n = imu_t_[slot].size();
    const char *gname = (slot == 1) ? "imu_cra" : "imu_mta";
    H5Id g(open_group(file_, gname), H5Gclose);
    if (imu_q_col_[slot].ds < 0) imu_q_col_[slot].ds = make_ext_dataset(g, "quat", H5T_IEEE_F32LE, 4);
    if (imu_t_col_[slot].ds < 0) imu_t_col_[slot].ds = make_ext_dataset(g, "t_master_us_quat", H5T_STD_U64LE, 1);
    if (imu_qid_col_[slot].ds < 0) imu_qid_col_[slot].ds = make_ext_dataset(g, "quat_report_id", H5T_STD_U8LE, 1);
    if (imu_qacc_col_[slot].ds < 0) {
        imu_qacc_col_[slot].ds = make_ext_dataset(g, "quat_accuracy", H5T_STD_U8LE, 1);
        write_str_attr(imu_qacc_col_[slot].ds, "note", kAccuracyNote);
    }
    if (!imu_attrs_done_[slot]) {
        const std::string &mode = (slot == 1) ? cfg_.fusion_cra : cfg_.fusion_mta;
        write_str_attr(g, "fusion_mode", mode);
        // Like hdf5_writer.py: name the fusion actually on the wire when the first
        // batch carries exactly one non-zero report id.
        int only = -1;
        bool single = true;
        for (uint8_t rid : imu_qid_[slot]) {
            if (rid == 0) continue;
            if (only < 0) {
                only = rid;
            } else if (only != rid) {
                single = false;
                break;
            }
        }
        if (single && only > 0) {
            const char *name = fusion_report_name(static_cast<uint8_t>(only));
            write_str_attr(g, "quat_fusion", name ? std::string(name) : mode);
        }
        imu_attrs_done_[slot] = true;
    }
    append_col(imu_q_col_[slot], imu_q_[slot].data(), 4 * sizeof(float), n, 4, H5T_NATIVE_FLOAT);
    append_col(imu_t_col_[slot], imu_t_[slot].data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
    append_col(imu_qid_col_[slot], imu_qid_[slot].data(), 1, n, 1, H5T_NATIVE_UINT8);
    append_col(imu_qacc_col_[slot], imu_qacc_[slot].data(), 1, n, 1, H5T_NATIVE_UINT8);
    imu_q_[slot].clear();
    imu_t_[slot].clear();
    imu_qid_[slot].clear();
    imu_qacc_[slot].clear();
    imu_q_col_[slot].done = imu_t_col_[slot].done = imu_qid_col_[slot].done = 0;
    imu_qacc_col_[slot].done = 0;
}

void Hdf5SessionWriter::flush_motion(int slot) {
    static const char *kNames[4] = {"accel", "gyro", "mag", "linear_accel"};
    static const char *kUnits[4] = {"m/s^2", "rad/s", "uT", "m/s^2"};
    static const char *kTNames[4] = {"t_master_us_accel", "t_master_us_gyro", "t_master_us_mag",
                                     "t_master_us_linear_accel"};
    static const char *kANames[4] = {"accel_accuracy", "gyro_accuracy", "mag_accuracy",
                                     "linear_accel_accuracy"};
    const char *gname = (slot == 1) ? "imu_cra" : "imu_mta";
    for (int i = 0; i < 4; ++i) {
        MotionBuf &b = imu_motion_[slot][i];
        if (b.t.empty()) {
            continue;
        }
        H5Id g(open_group(file_, gname), H5Gclose);
        const size_t n = b.t.size();
        if (b.ds.ds < 0) b.ds.ds = make_ext_dataset(g, kNames[i], H5T_IEEE_F32LE, 3);
        if (b.t_ds.ds < 0) b.t_ds.ds = make_ext_dataset(g, kTNames[i], H5T_STD_U64LE, 1);
        if (b.acc_ds.ds < 0) {
            b.acc_ds.ds = make_ext_dataset(g, kANames[i], H5T_STD_U8LE, 1);
            write_str_attr(b.acc_ds.ds, "note", kAccuracyNote);
        }
        if (!b.attrs_done) {
            write_str_attr(b.ds.ds, "units", kUnits[i]);
            write_str_attr(b.ds.ds, "device", "BNO085");
            if (cfg_.imu_rate_hz[slot][i] > 0) {
                write_int_attr(b.ds.ds, "rate_hz", cfg_.imu_rate_hz[slot][i]);
            }
            b.attrs_done = true;
        }
        append_col(b.ds, b.data.data(), 3 * sizeof(float), n, 3, H5T_NATIVE_FLOAT);
        append_col(b.t_ds, b.t.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
        append_col(b.acc_ds, b.acc.data(), 1, n, 1, H5T_NATIVE_UINT8);
        b.data.clear();
        b.t.clear();
        b.acc.clear();
        b.ds.done = b.t_ds.done = b.acc_ds.done = 0;
    }
}

void Hdf5SessionWriter::flush_trig() {
    if (trig_t_.empty()) {
        return;
    }
    const size_t n = trig_t_.size();
    H5Id g(open_group(file_, "triggers"), H5Gclose);
    if (trig_id_col_.ds < 0) trig_id_col_.ds = make_ext_dataset(g, "trigger_id", H5T_STD_U64LE, 1);
    if (trig_t_col_.ds < 0) trig_t_col_.ds = make_ext_dataset(g, "t_master_us", H5T_STD_U64LE, 1);
    if (!trig_attrs_done_) {
        if (cfg_.have_trigger_rate) {
            // Same type as the YAML value, as h5py stores it in the Python writer.
            if (cfg_.trigger_rate_is_int) {
                write_int_attr(g, "rate_hz", static_cast<long long>(cfg_.trigger_rate_hz));
            } else {
                write_double_attr(g, "rate_hz", cfg_.trigger_rate_hz);
            }
        }
        write_str_attr(g, "source", "mcu_pio");
        trig_attrs_done_ = true;
    }
    append_col(trig_id_col_, trig_id_.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
    append_col(trig_t_col_, trig_t_.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
    trig_id_.clear();
    trig_t_.clear();
    trig_id_col_.done = trig_t_col_.done = 0;
}

void Hdf5SessionWriter::flush_cameras() {
    H5Id cam_root(open_group(file_, "cameras"), H5Gclose);
    for (auto &entry : cam_) {
        const std::string &cid = entry.first;
        CamBuffers &slot = entry.second;
        if (slot.t.empty()) {
            continue;
        }
        const size_t n = slot.t.size();
        H5Id grp(open_or_create_group(cam_root, cid.c_str()), H5Gclose);
        if (slot.t_ds.ds < 0) slot.t_ds.ds = make_ext_dataset(grp, "t_master_us", H5T_STD_U64LE, 1);
        if (slot.sensor_ds.ds < 0) slot.sensor_ds.ds = make_ext_dataset(grp, "sensor_timestamp_ns", H5T_STD_U64LE, 1);
        if (slot.trig_ds.ds < 0) slot.trig_ds.ds = make_ext_dataset(grp, "trigger_id", H5T_STD_U64LE, 1);
        if (!slot.attrs_done) {
            if (!slot.video_file.empty()) {
                write_str_attr(grp, "video_file", slot.video_file);
            }
            slot.attrs_done = true;
        }
        append_col(slot.t_ds, slot.t.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
        append_col(slot.sensor_ds, slot.sensor_ts.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
        append_col(slot.trig_ds, slot.trig_id.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
        // frame_idx appears once any frame index is non-zero; after that it is
        // appended with every batch so it stays row-aligned from then on.
        const bool has_frames =
            std::any_of(slot.frame_idx.begin(), slot.frame_idx.end(), [](uint64_t f) { return f != 0; });
        if (slot.frame_ds.ds < 0 && has_frames) {
            slot.frame_ds.ds = make_ext_dataset(grp, "frame_idx", H5T_STD_U64LE, 1);
        }
        if (slot.frame_ds.ds >= 0) {
            append_col(slot.frame_ds, slot.frame_idx.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
        }
        slot.t.clear();
        slot.sensor_ts.clear();
        slot.trig_id.clear();
        slot.frame_idx.clear();
        slot.t_ds.done = slot.sensor_ds.done = slot.trig_ds.done = slot.frame_ds.done = 0;
    }
}

void Hdf5SessionWriter::flush_events() {
    if (ev_t_.empty()) {
        return;
    }
    const size_t n = ev_t_.size();
    H5Id g(open_group(file_, "events"), H5Gclose);
    H5Id st(vlen_utf8_type(), H5Tclose);
    // Same names/dtypes as hdf5_writer.py::_flush_events.
    if (ev_t_col_.ds < 0) ev_t_col_.ds = make_ext_dataset(g, "t_master_us", H5T_STD_U64LE, 1);
    if (ev_kind_col_.ds < 0) ev_kind_col_.ds = make_ext_dataset(g, "kind", st, 1);
    if (ev_text_col_.ds < 0) ev_text_col_.ds = make_ext_dataset(g, "text", st, 1);
    if (ev_phase_kind_col_.ds < 0) ev_phase_kind_col_.ds = make_ext_dataset(g, "phase_kind", st, 1);
    if (ev_host_col_.ds < 0) ev_host_col_.ds = make_ext_dataset(g, "t_host_ns", H5T_STD_U64LE, 1);
    if (ev_quality_col_.ds < 0) ev_quality_col_.ds = make_ext_dataset(g, "map_quality", st, 1);
    if (ev_view_t_col_.ds < 0) ev_view_t_col_.ds = make_ext_dataset(g, "viewer_t_s", H5T_IEEE_F64LE, 1);
    const size_t pb = sizeof(const char *);
    append_col(ev_t_col_, ev_t_.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
    append_col(ev_kind_col_, c_strs(ev_kind_).data(), pb, n, 1, st);
    append_col(ev_text_col_, c_strs(ev_text_).data(), pb, n, 1, st);
    append_col(ev_phase_kind_col_, c_strs(ev_phase_kind_).data(), pb, n, 1, st);
    append_col(ev_host_col_, ev_host_.data(), sizeof(uint64_t), n, 1, H5T_NATIVE_UINT64);
    append_col(ev_quality_col_, c_strs(ev_quality_).data(), pb, n, 1, st);
    append_col(ev_view_t_col_, ev_view_t_.data(), sizeof(double), n, 1, H5T_NATIVE_DOUBLE);
    ev_flushed_ += n;
    ev_t_.clear();
    ev_kind_.clear();
    ev_text_.clear();
    ev_phase_kind_.clear();
    ev_host_.clear();
    ev_quality_.clear();
    ev_view_t_.clear();
    ev_t_col_.done = ev_kind_col_.done = ev_text_col_.done = ev_phase_kind_col_.done = 0;
    ev_host_col_.done = ev_quality_col_.done = ev_view_t_col_.done = 0;
}

// Flush every stream even if one fails (a failing EMG append must not also cost
// the IMU/trigger rows), then rethrow the first error.
void Hdf5SessionWriter::flush_buffers() {
    std::exception_ptr err;
    auto guarded = [&err](auto &&fn) {
        try {
            fn();
        } catch (...) {
            if (!err) err = std::current_exception();
        }
    };
    guarded([this] { flush_emg(); });
    guarded([this] { flush_imu(0); });
    guarded([this] { flush_imu(1); });
    guarded([this] { flush_motion(0); });
    guarded([this] { flush_motion(1); });
    guarded([this] { flush_trig(); });
    guarded([this] { flush_cameras(); });
    guarded([this] { flush_events(); });
    if (err) {
        std::rethrow_exception(err);
    }
}

void Hdf5SessionWriter::flush(bool force_disk) {
    if (file_ < 0) {
        return;
    }
    flush_buffers();
    // Rows written by H5Dwrite (and the metadata describing them) can sit in the
    // library caches indefinitely; without H5Fflush a SIGKILL/OOM/SSH drop left
    // a file that not even `h5clear` could open. Rate-limited: it is not free.
    const double now = steady_s();
    if (force_disk || now - last_disk_flush_ >= cfg_.disk_flush_interval_s) {
        check(H5Fflush(file_, H5F_SCOPE_LOCAL), "H5Fflush");
        last_disk_flush_ = now;
    }
}

void Hdf5SessionWriter::set_sync_report(const std::string &report_json) {
    if (file_ < 0) {
        return;
    }
    H5Id g(open_group(file_, "sync"), H5Gclose);
    write_str_attr(g, "report_json", report_json);
}

void Hdf5SessionWriter::write_sync_drops_json(const std::string &) {
    // Drop counts are captured in the sync report_json; the per-key /sync/drops
    // datasets in the Python writer are advisory and not consumed by integrity.
}

void Hdf5SessionWriter::set_file_hashes_json(const std::string &json) {
    file_hashes_json_ = json.empty() ? "{}" : json;
}

void Hdf5SessionWriter::close_datasets() noexcept {
    auto close_col = [](Col &c) {
        if (c.ds >= 0) H5Dclose(c.ds);
        c.ds = -1;
        c.done = 0;
    };
    close_col(emg_samples_col_);
    close_col(emg_uv_col_);
    close_col(emg_t_col_);
    close_col(emg_loff_col_);
    close_col(emg_seq_col_);
    for (int s = 0; s < 2; ++s) {
        close_col(imu_q_col_[s]);
        close_col(imu_t_col_[s]);
        close_col(imu_qid_col_[s]);
        for (int i = 0; i < 4; ++i) {
            close_col(imu_motion_[s][i].ds);
            close_col(imu_motion_[s][i].t_ds);
        }
    }
    close_col(trig_id_col_);
    close_col(trig_t_col_);
    close_col(ev_t_col_);
    close_col(ev_kind_col_);
    close_col(ev_text_col_);
    close_col(ev_phase_kind_col_);
    close_col(ev_host_col_);
    close_col(ev_quality_col_);
    close_col(ev_view_t_col_);
    for (auto &entry : cam_) {
        close_col(entry.second.t_ds);
        close_col(entry.second.sensor_ds);
        close_col(entry.second.trig_ds);
        close_col(entry.second.frame_ds);
    }
}

void Hdf5SessionWriter::close() {
    if (file_ < 0) {
        return;
    }
    std::exception_ptr err;
    try {
        flush_buffers();
    } catch (...) {
        err = std::current_exception();
    }
    try {
        H5Id meta(open_group(file_, "meta"), H5Gclose);
        double dur = static_cast<double>(std::time(nullptr)) - opened_at_;
        if (dur < 0.0) {
            dur = 0.0;
        }
        write_double_attr(meta, "session_duration_s", dur);
        write_str_attr(meta, "file_hashes_json", file_hashes_json_);
    } catch (...) {
        if (!err) err = std::current_exception();
    }

    // Always release the file, even after an error above: a writer stuck "open"
    // would retry the failing flush in its destructor and keep the fd forever.
    close_datasets();
    const hid_t f = file_;
    file_ = -1;
    if (H5Fclose(f) < 0 && !err) {
        err = std::make_exception_ptr(std::runtime_error("HDF5 error: H5Fclose"));
    }
    cam_.clear();
    emg_samples_.clear();
    emg_t_.clear();
    emg_loff_.clear();
    emg_seq_.clear();
    for (int s = 0; s < 2; ++s) {
        imu_q_[s].clear();
        imu_t_[s].clear();
        imu_qid_[s].clear();
        for (int i = 0; i < 4; ++i) {
            imu_motion_[s][i].data.clear();
            imu_motion_[s][i].t.clear();
        }
    }
    trig_id_.clear();
    trig_t_.clear();
    ev_t_.clear();
    ev_kind_.clear();
    ev_text_.clear();
    ev_phase_kind_.clear();
    ev_host_.clear();
    ev_quality_.clear();
    ev_view_t_.clear();
    if (err) {
        std::rethrow_exception(err);
    }
}

}  // namespace jtsmm
