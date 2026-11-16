// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file hdf5_session.hpp
 * @brief Native HDF5 session writer (libhdf5 C API).
 *
 * @details Produces the same layout as `jtsmm/io/hdf5_writer.py::Hdf5Writer`
 *          (s02 §8): `/meta`, `/emg`, `/imu_*`, `/triggers`, `/sync`, `/cameras`,
 *          `/events`.
 *          Rows buffer in RAM and flush to chunked extendable datasets.
 *
 *          Failure model: every HDF5 call is checked and throws
 *          `std::runtime_error`. A failed flush keeps the unwritten rows and
 *          remembers, per dataset, how many buffered rows already landed, so a
 *          retry never duplicates rows (datasets of one group stay equal length).
 *          `close()` always releases the file, then rethrows the first error; the
 *          destructor never throws.
 */

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <hdf5.h>

namespace jtsmm {

/**
 * @brief Construction-time parameters for @ref Hdf5SessionWriter.
 */
struct Hdf5SessionConfig {
    std::string path;                      /**< Output `.h5` path. */
    std::string software_version = "0.1.0";
    int schema_version = 1;
    std::string config_hash;
    std::string mcu_config_json = "{}";
    std::string host_config_json = "{}";
    std::string firmware_version;  /**< Empty → attr omitted. */
    std::string mode = "acquisition";

    int emg_sample_rate_sps = 1000;
    int emg_gain = 6;
    std::vector<std::string> channel_names;
    /**
     * Per-column µV/LSB (8 values, from `jtsmm.config.catalog.channel_lsb_uv`:
     * CH7/CH8 run at a fixed gain). Empty → every column uses the muscle LSB.
     */
    std::vector<double> lsb_uv_per_channel;

    std::string fusion_mta = "game_rotation_vector";
    std::string fusion_cra = "game_rotation_vector";
    bool have_trigger_rate = false;
    double trigger_rate_hz = 0.0;
    bool trigger_rate_is_int = true;  /**< Write `rate_hz` as int64 (else float64). */

    /** Configured IMU report rates [slot 0=mta,1=cra][accel,gyro,mag,linear_accel]; 0 → no attr. */
    int imu_rate_hz[2][4] = {};

    /**
     * Minimum wall time between `H5Fflush` calls from @ref Hdf5SessionWriter::flush.
     * Without H5Fflush a killed recorder leaves an unreadable file (metadata only
     * in RAM); with it, `h5clear -s` recovers everything up to the last flush.
     */
    double disk_flush_interval_s = 1.0;

    /**
     * Extra `/meta` string attrs (`jtsmm record --meta key=value`).
     * Mirrors Python `session_meta` → per-key attr flattening.
     */
    std::map<std::string, std::string> extra_meta;
};

/**
 * @brief Incremental HDF5 session writer used by native capture.
 */
class Hdf5SessionWriter {
public:
    explicit Hdf5SessionWriter(Hdf5SessionConfig cfg);
    /** @brief Closes the file; swallows errors (call close() to see them). */
    ~Hdf5SessionWriter() noexcept;

    Hdf5SessionWriter(const Hdf5SessionWriter &) = delete;
    Hdf5SessionWriter &operator=(const Hdf5SessionWriter &) = delete;

    /** @brief Create/truncate the file and write `/meta` attrs. */
    void open();

    /**
     * @brief Append one EMG sample row.
     *
     * @param t_master_us MCU timestamp (µs).
     * @param channels    Eight int32 channel counts.
     * @param leadoff_statp LOFF_STATP bits for CH1–6 only (pre-masked 0x3F).
     * @param leadoff_statn LOFF_STATN bits for CH1–6 only (pre-masked 0x3F).
     * @param has_seq     Record carried the firmware acquisition seq (31-byte EMG).
     * @param seq         That seq; `/emg/seq` exists iff the first EMG row had one.
     *
     * @note DRL (CH8) is intentionally excluded from lead-off masks.
     */
    void append_emg(uint64_t t_master_us, const int32_t channels[8], uint8_t leadoff_statp = 0,
                    uint8_t leadoff_statn = 0, bool has_seq = false, uint32_t seq = 0);

    /**
     * @brief Append a fusion quaternion.
     * @param cranium true → CRA group; false → MTA.
     */
    void append_quat(bool cranium, uint64_t t_master_us, const float quat[4], uint8_t report_id = 0,
                     uint8_t accuracy = kAccuracyAbsent);

    /**
     * @brief Append SI motion vector (accel/gyro/mag).
     * @param sensor_id 0x01/0x02/0x03 (accel/gyro/mag); @p xyz already SI.
     */
    void append_imu_motion(bool cranium, uint8_t sensor_id, uint64_t t_master_us,
                           const float xyz[3], uint8_t accuracy = kAccuracyAbsent);

    /** @brief Append linear acceleration (m/s²). */
    void append_lin_accel(bool cranium, uint64_t t_master_us, const float xyz[3],
                          uint8_t accuracy = kAccuracyAbsent);

    /** BNO085 calibration status 0..3 per sample; this value = record carried none. */
    static constexpr uint8_t kAccuracyAbsent = 255;

    /** @brief Append a trigger event row. */
    void append_trigger(uint64_t t_master_us, uint64_t trigger_id);

    /**
     * @brief Append camera metadata row under `/cameras/<cam_id>/`.
     */
    void append_camera(const std::string &cam_id, uint64_t t_master_us, uint64_t sensor_timestamp_ns,
                       uint64_t trigger_id, uint64_t frame_idx, const std::string &video_file = "");

    /**
     * @brief Append a session event row under `/events` (viewer note, ...).
     *
     * Same datasets, dtypes and row semantics as `Hdf5Writer.append_event`:
     * `t_master_us`/`t_host_ns` uint64, `kind`/`text`/`phase_kind`/`map_quality`
     * variable-length UTF-8 strings, `viewer_t_s` float64 (NaN = none).
     * Buffered like every other stream and written by @ref flush.
     * @return The row index of this event in `/events`.
     */
    uint64_t append_event(uint64_t t_master_us, const std::string &kind, const std::string &text,
                          const std::string &phase_kind = "", uint64_t t_host_ns = 0,
                          const std::string &map_quality = "",
                          double viewer_t_s = std::numeric_limits<double>::quiet_NaN());
    /**
     * @brief Re-stamp event @p row, whether still buffered or already on disk
     *        (close-time re-map of host stamps with the final clock fit).
     * @throws std::out_of_range for an unknown row; std::runtime_error on HDF5 failure.
     */
    void set_event_time(uint64_t row, uint64_t t_master_us, const std::string &map_quality);
    /**
     * @brief Write buffered rows to the datasets; `H5Fflush` at most every
     *        `disk_flush_interval_s` (or now when @p force_disk).
     * @throws std::runtime_error on HDF5 failure (rows stay buffered for retry).
     */
    void flush(bool force_disk = false);

    /** @brief Store sync report JSON under `/sync` (replaces a previous one). */
    void set_sync_report(const std::string &report_json);

    /** @brief Store drop counters JSON under `/sync`. */
    void write_sync_drops_json(const std::string &drops_json);

    /** @brief Store file-hash JSON under `/meta`. */
    void set_file_hashes_json(const std::string &json);

    /**
     * @brief Flush and close the file. The file is released even when the final
     *        flush fails; the first error is rethrown afterwards.
     */
    void close();

    /** @brief Test hook: make the @p n-th dataset append from now on throw (0 = off). */
    void debug_fail_nth_append(int n) { fail_countdown_ = n; }

private:
    /** One extendable dataset plus how many buffered rows it already holds. */
    struct Col {
        hid_t ds = -1;
        size_t done = 0;
    };

    Hdf5SessionConfig cfg_;
    hid_t file_ = -1;
    double opened_at_ = 0.0;
    double last_disk_flush_ = 0.0;
    double lsb_uv_ = 0.0;
    std::vector<float> lsb_col_f_;  // per-column LSB as float32 (8)
    int fail_countdown_ = 0;

    // EMG buffers.
    std::vector<int32_t> emg_samples_;  // row-major (n,8)
    std::vector<uint64_t> emg_t_;
    // row-major (n,2): [statp_mask, statn_mask], muscle CH1-6 only (bits0-5).
    std::vector<uint8_t> emg_loff_;
    std::vector<uint32_t> emg_seq_;
    int emg_has_seq_ = -1;  // -1 undecided, 0 no /emg/seq, 1 /emg/seq
    Col emg_samples_col_, emg_uv_col_, emg_t_col_, emg_loff_col_, emg_seq_col_;
    bool emg_attrs_done_ = false;

    // IMU buffers (mta = index 0, cra = index 1).
    std::vector<float> imu_q_[2];  // row-major (n,4)
    std::vector<uint64_t> imu_t_[2];
    std::vector<uint8_t> imu_qid_[2];  // per-quat fusion report id (tagged wire)
    std::vector<uint8_t> imu_qacc_[2]; // per-quat calibration status (0..3, 255 = none)
    Col imu_q_col_[2], imu_t_col_[2], imu_qid_col_[2], imu_qacc_col_[2];
    bool imu_attrs_done_[2] = {false, false};

    // Per-slot motion vectors: 0=accel, 1=gyro, 2=mag, 3=linear_accel. Each row is
    // an SI (x,y,z) float triple with its own timestamp vector, mirroring
    // hdf5_writer.py exactly (dataset names, units/device/rate_hz attrs).
    struct MotionBuf {
        std::vector<float> data;  // row-major (n,3)
        std::vector<uint64_t> t;
        std::vector<uint8_t> acc;  // calibration status per row (0..3, 255 = none)
        Col ds, t_ds, acc_ds;
        bool attrs_done = false;
    };
    MotionBuf imu_motion_[2][4];

    // Trigger buffers.
    std::vector<uint64_t> trig_id_;
    std::vector<uint64_t> trig_t_;
    Col trig_id_col_, trig_t_col_;
    bool trig_attrs_done_ = false;

    struct CamBuffers {
        std::vector<uint64_t> t;
        std::vector<uint64_t> sensor_ts;
        std::vector<uint64_t> trig_id;
        std::vector<uint64_t> frame_idx;
        Col t_ds, sensor_ds, trig_ds, frame_ds;
        bool attrs_done = false;
        std::string video_file;
    };
    std::unordered_map<std::string, CamBuffers> cam_;
    // Event buffers (/events); ev_flushed_ = rows already in the datasets.
    std::vector<uint64_t> ev_t_, ev_host_;
    std::vector<std::string> ev_kind_, ev_text_, ev_phase_kind_, ev_quality_;
    std::vector<double> ev_view_t_;
    Col ev_t_col_, ev_kind_col_, ev_text_col_, ev_phase_kind_col_, ev_host_col_, ev_quality_col_,
        ev_view_t_col_;
    uint64_t ev_flushed_ = 0;

    std::string file_hashes_json_ = "{}";

    void append_col(Col &c, const void *base, size_t row_bytes, size_t n_total, hsize_t ncols,
                    hid_t memtype);
    void flush_buffers();
    void flush_emg();
    void flush_imu(int slot);
    void flush_motion(int slot);
    void flush_trig();
    void flush_cameras();
    void flush_events();
    void close_datasets() noexcept;
};

}  // namespace jtsmm
