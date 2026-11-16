// This file is part of JTS-MM (https://github.com/paulotto/jts-mm).
// SPDX-FileCopyrightText: 2026 Paul-Otto Müller
// SPDX-License-Identifier: MIT

/**
 * @file module.cpp
 * @brief pybind11 module exports for jtsmm.native.
 */

// pybind11 bindings for the native capture path.
//
// Exposes a FrameParser and an McuCapture whose records carry the same fields as
// the Python jtsmm.protocol.McuRecord (record_type, t_master_us, payload, decoded,
// type_name), so the native path is a drop-in for the Python one and the two can
// be compared byte-for-byte in tests.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>

#include "jtsmm_native/frame_parser.hpp"
#include "jtsmm_native/mcu_capture.hpp"
#include "jtsmm_native/record.hpp"
#include "jtsmm_native/record_decode.hpp"

#ifdef JTSMM_HAVE_HDF5
#include "jtsmm_native/hdf5_session.hpp"
#endif

namespace py = pybind11;
using namespace jtsmm;

namespace {

// Keep in lockstep with STATS_FIELDS in jtsmm/protocol/records.py and with
// comms_core_send_stats() in the firmware. Append only.
const std::array<const char *, 36> kStatsFields = {
    "drdy_count",  "missed_drdy",   "emg_push",       "emg_drops",
    "imu_mta_push", "imu_mta_drops", "imu_cra_push",  "imu_cra_drops",
    "events_push", "events_drops",  "usb_tx_bytes",   "sensor_loop_hz",
    "comms_loop_hz", "tx_drops",    "max_drdy_gap_us", "imu_svc_max_us",
    "leadoff_max_us", "leadoff_count", "poll_max_us",  "frame_misalign",
    "emg_misalign_drops", "misalign_max_streak", "emg_resyncs", "spi_wedges",
    "trig_pio_events", "trig_timer_events", "imu_i2c_aborts",
    "imu_cra_i2c_aborts",
    // Appended 2026-09; older firmware sends only the 28 above (decode is by count).
    "tx_drops_emg", "tx_drops_imu", "tx_drops_other", "imu_mta_seq_gaps",
    "imu_cra_seq_gaps", "imu_mta_resets", "imu_cra_resets", "stats_seq",
};

int32_t sign_extend24(uint32_t raw) {
    if (raw & 0x800000u) {
        return static_cast<int32_t>(raw) - (1 << 24);
    }
    return static_cast<int32_t>(raw);
}

float read_f32(const uint8_t *p) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
}

int16_t read_i16(const uint8_t *p) {
    return static_cast<int16_t>(static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8));
}

uint32_t read_u32(const uint8_t *p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_u64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    return v;
}

py::bytes payload_bytes(const Record &rec) {
    return py::bytes(reinterpret_cast<const char *>(rec.payload.data()), rec.payload.size());
}

// Mirror of jtsmm.protocol.records.decode_record. On malformed payloads we fall
// back to {"raw": payload}, exactly as the Python parser does.
py::dict decode(const Record &rec) {
    py::dict d;
    const uint8_t *p = rec.payload.data();
    const size_t n = rec.payload.size();
    try {
        switch (rec.record_type) {
            case REC_EMG_BLOCK: {
                // 27 B, or 31 B with a trailing u32 acquisition seq (records.py).
                if (n != 27 && n != 31) throw std::runtime_error("emg len");
                py::list ch;
                for (int i = 0; i < 8; ++i) {
                    const uint8_t *o = p + 3 + i * 3;
                    uint32_t raw = read_emg_count24_be(o);
                    ch.append(sign_extend24(raw));
                }
                d["status"] = py::bytes(reinterpret_cast<const char *>(p), 3);
                d["channels"] = ch;
                if (n == 31) {
                    d["seq"] = read_u32(p + 27);
                }
                return d;
            }
            case REC_IMU_MTA_QUAT:
            case REC_IMU_CRA_QUAT:
            case REC_IMU_DP_QUAT: {
                // Tagged (17 B): [report_id][w,x,y,z]. Legacy (16 B): untagged.
                if (n >= 17) {
                    uint8_t report_id = p[0];
                    d["quat"] = py::make_tuple(read_f32(p + 1), read_f32(p + 5), read_f32(p + 9),
                                               read_f32(p + 13));
                    d["quat_report_id"] = report_id;
                    const char *f = fusion_report_name(report_id);
                    d["fusion"] = f ? py::object(py::str(f)) : py::object(py::none());
                    if (n > 17) {
                        d["quat_accuracy"] = static_cast<int>(p[17] & 3u);
                    }
                } else if (n >= 16) {
                    d["quat"] = py::make_tuple(read_f32(p), read_f32(p + 4), read_f32(p + 8),
                                               read_f32(p + 12));
                    d["quat_report_id"] = py::none();
                    d["fusion"] = py::none();
                } else {
                    throw std::runtime_error("quat len");
                }
                return d;
            }
            case REC_IMU_MTA_RAW:
            case REC_IMU_CRA_RAW:
            case REC_IMU_DP_RAW: {
                // Tagged (7 B): [sensor_id][x,y,z]. Legacy (18 B): accel+gyro+mag.
                if (n >= 7 && n < 18) {
                    uint8_t sensor_id = p[0];
                    py::tuple xyz = py::make_tuple(read_i16(p + 1), read_i16(p + 3), read_i16(p + 5));
                    d["sensor_id"] = sensor_id;
                    const char *key = nullptr;
                    if (sensor_id == 0x01) {
                        key = "accel";
                    } else if (sensor_id == 0x02) {
                        key = "gyro";
                    } else if (sensor_id == 0x03) {
                        key = "mag";
                    }
                    if (key != nullptr) {
                        d[key] = xyz;
                        if (n > 7) {
                            d[py::str(std::string(key) + "_accuracy")] =
                                static_cast<int>(p[7] & 3u);
                        }
                    } else {
                        d["raw"] = xyz;
                    }
                } else if (n >= 18) {
                    d["accel"] = py::make_tuple(read_i16(p), read_i16(p + 2), read_i16(p + 4));
                    d["gyro"] = py::make_tuple(read_i16(p + 6), read_i16(p + 8), read_i16(p + 10));
                    d["mag"] = py::make_tuple(read_i16(p + 12), read_i16(p + 14), read_i16(p + 16));
                } else {
                    throw std::runtime_error("raw len");
                }
                return d;
            }
            case REC_IMU_MTA_LIN_ACCEL:
            case REC_IMU_CRA_LIN_ACCEL:
            case REC_IMU_DP_LIN_ACCEL: {
                if (n < 12) throw std::runtime_error("lin len");
                d["linear_accel"] = py::make_tuple(read_f32(p), read_f32(p + 4), read_f32(p + 8));
                if (n > 12) {
                    d["linear_accel_accuracy"] = static_cast<int>(p[12] & 3u);
                }
                return d;
            }
            case REC_TRIGGER_EVENT: {
                if (n < 12) throw std::runtime_error("trig len");
                d["flags"] = py::make_tuple(p[0], p[1], p[2], p[3]);
                d["trigger_id"] = read_u64(p + 4);
                // Mirrors records.TRIGGER_STAMP_NAMES.
                switch (p[0]) {
                    case 0: d["stamp_source"] = "unknown"; break;
                    case 1: d["stamp_source"] = "pio_irq"; break;
                    case 2: d["stamp_source"] = "timer"; break;
                    default: d["stamp_source"] = py::str("0x{:02X}").format(p[0]); break;
                }
                d["stamp_fallback"] = (p[1] != 0);
                return d;
            }
            case REC_LED_FLASH_EVENT: {
                if (n < 4) throw std::runtime_error("led len");
                d["flash_id"] = read_u32(p);
                return d;
            }
            case REC_BUTTON_EVENT: {
                // [button_id:u8][pressed:u8][reserved:3] (firmware panel.c).
                if (n < 2) throw std::runtime_error("button len");
                d["button_id"] = p[0];
                d["pressed"] = (p[1] != 0);
                return d;
            }
            case REC_PPS_REPLY: {
                if (n < 8) throw std::runtime_error("pps len");
                d["pps_id"] = read_u64(p);
                return d;
            }
            case REC_STATS: {
                size_t count = n / 4;
                for (size_t i = 0; i < kStatsFields.size() && i < count; ++i) {
                    d[kStatsFields[i]] = read_u32(p + i * 4);
                }
                return d;
            }
            case REC_LOG: {
                // errors="replace" like records.py: a torn/garbled log line must
                // still decode as text, not fall back to {"raw": ...}.
                PyObject *txt = PyUnicode_DecodeUTF8(reinterpret_cast<const char *>(p),
                                                     static_cast<Py_ssize_t>(n), "replace");
                if (txt == nullptr) throw py::error_already_set();
                d["text"] = py::reinterpret_steal<py::str>(txt);
                return d;
            }
            case REC_CMD_ACK: {
                d["ok"] = (n > 0) ? (p[0] != 0) : false;
                return d;
            }
            default:
                d["raw"] = payload_bytes(rec);
                return d;
        }
    } catch (const std::exception &) {
        py::dict fallback;
        fallback["raw"] = payload_bytes(rec);
        return fallback;
    }
}

const char *type_name(uint8_t t) {
    switch (t) {
        case REC_EMG_BLOCK: return "EMG_BLOCK";
        case REC_IMU_MTA_QUAT: return "IMU_MTA_QUAT";
        case REC_IMU_MTA_RAW: return "IMU_MTA_RAW";
        case REC_IMU_MTA_LIN_ACCEL: return "IMU_MTA_LIN_ACCEL";
        case REC_IMU_CRA_QUAT: return "IMU_CRA_QUAT";
        case REC_IMU_CRA_RAW: return "IMU_CRA_RAW";
        case REC_IMU_CRA_LIN_ACCEL: return "IMU_CRA_LIN_ACCEL";
        case REC_IMU_DP_QUAT: return "IMU_DP_QUAT";
        case REC_IMU_DP_RAW: return "IMU_DP_RAW";
        case REC_IMU_DP_LIN_ACCEL: return "IMU_DP_LIN_ACCEL";
        case REC_TRIGGER_EVENT: return "TRIGGER_EVENT";
        case REC_LED_FLASH_EVENT: return "LED_FLASH_EVENT";
        case REC_BUTTON_EVENT: return "BUTTON_EVENT";
        case REC_HEARTBEAT: return "HEARTBEAT";
        case REC_PPS_REPLY: return "PPS_REPLY";
        case REC_ADS_STATUS: return "ADS_STATUS";
        case REC_STATS: return "STATS";
        case REC_LOG: return "LOG";
        case REC_CMD_ACK: return "CMD_ACK";
        default: return nullptr;
    }
}

// Build a Python object matching jtsmm.protocol.McuRecord's attribute surface.
py::object make_record(const Record &rec) {
    py::dict obj;
    obj["record_type"] = rec.record_type;
    obj["t_master_us"] = rec.t_master_us;
    obj["payload"] = payload_bytes(rec);
    obj["decoded"] = decode(rec);
    const char *name = type_name(rec.record_type);
    obj["type_name"] = name ? py::str(name)
                            : py::str(py::str("0x{:02X}").format(rec.record_type));
    // SimpleNamespace gives attribute access (rec.record_type, rec.decoded, ...).
    py::object ns = py::module_::import("types").attr("SimpleNamespace");
    return ns(**obj);
}

// Decode a raw STATS payload into the same dict shape as decode() produces.
py::object decode_stats_payload(const std::vector<uint8_t> &payload) {
    Record rec;
    rec.record_type = REC_STATS;
    rec.payload = payload;
    return decode(rec);
}

py::dict viewer_cal_dict(const std::array<uint8_t, ViewerFrame::kCalStreams> &cal,
                         const std::array<bool, ViewerFrame::kCalStreams> &have) {
    static const char *keys[ViewerFrame::kCalStreams] = {
        "quat", "accel", "gyro", "mag", "linear_accel"};
    py::dict out;
    for (int i = 0; i < ViewerFrame::kCalStreams; ++i) {
        if (have[i]) {
            out[keys[i]] = cal[i];
        }
    }
    return out;
}

// Build a Python view of the native ViewerFrame (NumPy arrays + scalars).
py::object make_viewer_frame(const ViewerFrame &f) {
    py::dict obj;
    const auto n = static_cast<py::ssize_t>(f.n_buckets);
    const py::ssize_t ch = ViewerFrame::kChannels;

    py::array_t<int32_t> emg_min({n, ch});
    py::array_t<int32_t> emg_max({n, ch});
    py::array_t<float> emg_mean({n, ch});
    py::array_t<uint64_t> emg_t(static_cast<py::ssize_t>(f.n_buckets));
    if (f.n_buckets > 0) {
        std::memcpy(emg_min.mutable_data(), f.emg_min.data(),
                    f.emg_min.size() * sizeof(int32_t));
        std::memcpy(emg_max.mutable_data(), f.emg_max.data(),
                    f.emg_max.size() * sizeof(int32_t));
        std::memcpy(emg_mean.mutable_data(), f.emg_mean.data(),
                    f.emg_mean.size() * sizeof(float));
        std::memcpy(emg_t.mutable_data(), f.emg_t.data(), f.emg_t.size() * sizeof(uint64_t));
    }
    obj["emg_min"] = emg_min;
    obj["emg_max"] = emg_max;
    obj["emg_mean"] = emg_mean;
    obj["emg_t"] = emg_t;
    obj["emg_seen"] = f.emg_seen;

    if (f.have_mta) {
        obj["mta_quat"] =
            py::make_tuple(f.mta_quat[0], f.mta_quat[1], f.mta_quat[2], f.mta_quat[3]);
        obj["mta_quat_report_id"] = f.mta_quat_report_id;
    } else {
        obj["mta_quat"] = py::none();
        obj["mta_quat_report_id"] = py::none();
    }
    obj["mta_t"] = f.mta_t;
    if (f.have_cra) {
        obj["cra_quat"] =
            py::make_tuple(f.cra_quat[0], f.cra_quat[1], f.cra_quat[2], f.cra_quat[3]);
        obj["cra_quat_report_id"] = f.cra_quat_report_id;
    } else {
        obj["cra_quat"] = py::none();
        obj["cra_quat_report_id"] = py::none();
    }
    obj["cra_t"] = f.cra_t;
    obj["mta_cal"] = viewer_cal_dict(f.mta_cal, f.have_mta_cal);
    obj["cra_cal"] = viewer_cal_dict(f.cra_cal, f.have_cra_cal);
    obj["trig_seen"] = f.trig_seen;

    auto export_quat_batch = [&](const char *times_key, const char *vals_key,
                                 const std::vector<uint64_t> &t,
                                 const std::vector<float> &q) {
        const auto n = static_cast<py::ssize_t>(t.size());
        if (n <= 0) {
            return;
        }
        py::array_t<uint64_t> tt(n);
        py::array_t<float> qq({n, static_cast<py::ssize_t>(4)});
        std::memcpy(tt.mutable_data(), t.data(), t.size() * sizeof(uint64_t));
        std::memcpy(qq.mutable_data(), q.data(), q.size() * sizeof(float));
        obj[times_key] = tt;
        obj[vals_key] = qq;
    };
    export_quat_batch("mta_quat_times", "mta_quat_vals", f.batch_mta_quat_t, f.batch_mta_quat);
    export_quat_batch("cra_quat_times", "cra_quat_vals", f.batch_cra_quat_t, f.batch_cra_quat);

    static const char *kMotionBatchKeys[2][4] = {
        {"mta_accel", "mta_gyro", "mta_mag", "mta_lin"},
        {"cra_accel", "cra_gyro", "cra_mag", "cra_lin"},
    };
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < 4; ++i) {
            const auto n = static_cast<py::ssize_t>(f.batch_motion_t[s][i].size());
            if (n > 0) {
                py::array_t<uint64_t> tt(n);
                py::array_t<float> vv({n, static_cast<py::ssize_t>(3)});
                std::memcpy(tt.mutable_data(), f.batch_motion_t[s][i].data(),
                            f.batch_motion_t[s][i].size() * sizeof(uint64_t));
                std::memcpy(vv.mutable_data(), f.batch_motion_v[s][i].data(),
                            f.batch_motion_v[s][i].size() * sizeof(float));
                const std::string t_key = std::string(kMotionBatchKeys[s][i]) + "_t";
                const std::string v_key = std::string(kMotionBatchKeys[s][i]) + "_v";
                obj[t_key.c_str()] = tt;
                obj[v_key.c_str()] = vv;
            }
        }
    }

    // Latest per-slot motion vectors (SI). Keys mta_accel/mta_gyro/mta_mag/
    // mta_lin and cra_* ; None when not yet seen.
    static const char *kMotionKeys[2][4] = {
        {"mta_accel", "mta_gyro", "mta_mag", "mta_lin"},
        {"cra_accel", "cra_gyro", "cra_mag", "cra_lin"},
    };
    for (int s = 0; s < 2; ++s) {
        for (int i = 0; i < 4; ++i) {
            if (f.have_motion[s][i]) {
                obj[kMotionKeys[s][i]] =
                    py::make_tuple(f.motion[s][i][0], f.motion[s][i][1], f.motion[s][i][2]);
            } else {
                obj[kMotionKeys[s][i]] = py::none();
            }
        }
    }

    if (f.have_leadoff) {
        obj["status"] = py::bytes(reinterpret_cast<const char *>(f.status.data()), 3);
    } else {
        obj["status"] = py::none();
    }
    if (f.have_stats) {
        obj["stats"] = decode_stats_payload(f.stats_payload);
    } else {
        obj["stats"] = py::none();
    }

    py::object ns = py::module_::import("types").attr("SimpleNamespace");
    return ns(**obj);
}

}  // namespace

PYBIND11_MODULE(_jtsmm_native, m) {
    m.doc() = "Native (C++/libusb) capture + frame parse for jtsmm.";
#ifdef JTSMM_HAVE_HDF5
    // After an H5Fclose that failed on I/O (disk full) libhdf5 1.10 keeps a
    // half-closed file id and segfaults in its atexit H5_term_library. Writers
    // close their files explicitly (and flush to disk periodically), so skip the
    // library's atexit teardown. Must run before any other HDF5 call.
    H5dont_atexit();
#endif

    py::class_<FrameParser>(m, "FrameParser")
        .def(py::init<>())
        .def(
            "feed",
            [](FrameParser &self, py::buffer data) {
                py::buffer_info info = data.request();
                auto recs = self.feed(static_cast<const uint8_t *>(info.ptr),
                                      static_cast<size_t>(info.size * info.itemsize));
                py::list out;
                for (const auto &r : recs) {
                    out.append(make_record(r));
                }
                return out;
            },
            py::arg("data"))
        .def_property_readonly("crc_errors", &FrameParser::crc_errors)
        .def_property_readonly("frames_ok", &FrameParser::frames_ok)
        .def_property_readonly("resyncs", &FrameParser::resyncs);

    py::class_<McuCapture>(m, "NativeMcuCapture")
        .def(py::init<size_t>(), py::arg("queue_max") = 65536)
        .def("start", &McuCapture::start, py::call_guard<py::gil_scoped_release>())
        .def("stop", &McuCapture::stop, py::call_guard<py::gil_scoped_release>())
        .def(
            "next_record",
            [](McuCapture &self, double timeout_s) -> py::object {
                std::optional<Record> rec;
                {
                    py::gil_scoped_release release;
                    rec = self.next_record(static_cast<int>(timeout_s * 1000.0));
                }
                if (!rec) {
                    return py::none();
                }
                return make_record(*rec);
            },
            py::arg("timeout_s") = 0.1)
        .def_property_readonly("bytes_rx", [](const McuCapture &c) { return c.stats().bytes_rx; })
        .def_property_readonly("records_rx",
                               [](const McuCapture &c) { return c.stats().records_rx; })
        .def_property_readonly("crc_errors",
                               [](const McuCapture &c) { return c.stats().crc_errors; })
        .def_property_readonly("resyncs", [](const McuCapture &c) { return c.stats().resyncs; })
        .def_property_readonly("raw_chunks_dropped",
                               [](const McuCapture &c) { return c.stats().raw_chunks_dropped; })
        .def_property_readonly("records_dropped",
                               [](const McuCapture &c) { return c.stats().records_dropped; })
        .def(
            "link_counters",
            [](const McuCapture &c) {
                // USB transfer outcome counters + liveness (valid after stop()).
                const CaptureStats s = c.stats();
                py::dict d;
                d["bytes_rx"] = s.link.bytes_rx;
                d["usb_errors"] = s.link.errors;
                d["usb_stalls"] = s.link.stalls;
                d["usb_overflows"] = s.link.overflows;
                d["usb_no_device"] = s.link.no_device;
                d["usb_partial_bytes"] = s.link.partial_bytes;
                d["usb_lost_bytes"] = s.link.lost_bytes;
                d["usb_resubmit_failures"] = s.link.resubmit_failures;
                d["device_lost"] = s.link.device_lost;
                d["last_record_age_s"] = c.last_record_age_s();
                return d;
            })
        .def("_start_parser_only", &McuCapture::start_parser_only,
             py::call_guard<py::gil_scoped_release>())
        .def(
            "inject_raw",
            [](McuCapture &self, py::buffer data) {
                // Test hook: bytes go through the same parse thread as USB data.
                py::buffer_info info = data.request();
                self.inject_raw(static_cast<const uint8_t *>(info.ptr),
                                static_cast<size_t>(info.size * info.itemsize));
            },
            py::arg("data"))
        .def(
            "discard_queued_records",
            [](McuCapture &self) {
                py::gil_scoped_release release;
                self.discard_queued_records();
            })
        .def(
            "pump_to_writer",
            [](McuCapture &self, Hdf5SessionWriter &writer, size_t max_records) {
                McuCapture::PumpResult result;
                {
                    py::gil_scoped_release release;
                    result = self.pump_to_writer(writer, max_records);
                }
                py::dict out;
                out["emg"] = result.emg;
                out["mta"] = result.mta;
                out["cra"] = result.cra;
                out["trig"] = result.trig;
                py::dict by_type;
                for (size_t t = 0; t < result.by_type.size(); ++t) {
                    if (result.by_type[t] != 0) {
                        by_type[py::int_(t)] = result.by_type[t];
                    }
                }
                out["by_type"] = by_type;
                if (result.last_stats) {
                    out["last_stats"] = make_record(*result.last_stats);
                } else {
                    out["last_stats"] = py::none();
                }
                py::list triggers;
                for (const auto &note : result.triggers) {
                    py::dict tr;
                    tr["t_master_us"] = note.t_master_us;
                    tr["trigger_id"] = note.trigger_id;
                    tr["stamp_source"] = note.stamp_source;
                    tr["stamp_fallback"] = note.stamp_fallback;
                    triggers.append(tr);
                }
                out["triggers"] = triggers;
                py::list pps;
                for (const auto &note : result.pps) {
                    py::dict pr;
                    pr["t_master_us"] = note.t_master_us;
                    pr["pps_id"] = note.pps_id;
                    pps.append(pr);
                }
                out["pps"] = pps;
                py::list logs;
                for (const auto &note : result.logs) {
                    py::dict lg;
                    lg["t_master_us"] = note.t_master_us;
                    // errors="replace" like make_record: a torn line still decodes.
                    PyObject *txt = PyUnicode_DecodeUTF8(note.text.data(),
                                                         static_cast<Py_ssize_t>(note.text.size()),
                                                         "replace");
                    if (txt == nullptr) throw py::error_already_set();
                    lg["text"] = py::reinterpret_steal<py::str>(txt);
                    logs.append(lg);
                }
                out["logs"] = logs;
                return out;
            },
            py::arg("writer"), py::arg("max_records") = 8192)
        .def("set_viewer_tap_enabled", &McuCapture::set_viewer_tap_enabled, py::arg("enabled"),
             py::arg("bucket_samples") = 4)
        .def("get_viewer_frame", [](McuCapture &self) {
            ViewerFrame f;
            {
                py::gil_scoped_release release;
                f = self.get_viewer_frame();
            }
            return make_viewer_frame(f);
        });

#ifdef JTSMM_HAVE_HDF5
    py::class_<Hdf5SessionWriter>(m, "NativeHdf5Writer")
        .def(py::init([](const std::string &path, const std::string &config_hash,
                         const std::string &mcu_config_json, const std::string &host_config_json,
                         const std::string &software_version, int schema_version,
                         const std::string &firmware_version, const std::string &mode,
                         int emg_sample_rate_sps, int emg_gain,
                         std::vector<std::string> channel_names, const std::string &fusion_mta,
                         const std::string &fusion_cra, py::object trigger_rate_hz,
                         std::map<std::string, std::string> extra_meta,
                         double disk_flush_interval_s, std::vector<double> lsb_uv_per_channel,
                         std::map<std::string, int> imu_rates_mta,
                         std::map<std::string, int> imu_rates_cra) {
                 Hdf5SessionConfig c;
                 c.path = path;
                 c.config_hash = config_hash;
                 c.mcu_config_json = mcu_config_json;
                 c.host_config_json = host_config_json;
                 c.software_version = software_version;
                 c.schema_version = schema_version;
                 c.firmware_version = firmware_version;
                 c.mode = mode;
                 c.emg_sample_rate_sps = emg_sample_rate_sps;
                 c.emg_gain = emg_gain;
                 c.channel_names = std::move(channel_names);
                 c.fusion_mta = fusion_mta;
                 c.fusion_cra = fusion_cra;
                 if (!trigger_rate_hz.is_none()) {
                     // YAML may give 30 or 29.97; cast<int>() threw on a float and
                     // killed the recorder before the file was even created.
                     c.have_trigger_rate = true;
                     c.trigger_rate_is_int = py::isinstance<py::int_>(trigger_rate_hz) &&
                                             !py::isinstance<py::bool_>(trigger_rate_hz);
                     c.trigger_rate_hz = trigger_rate_hz.cast<double>();
                 }
                 c.extra_meta = std::move(extra_meta);
                 c.disk_flush_interval_s = disk_flush_interval_s;
                 c.lsb_uv_per_channel = std::move(lsb_uv_per_channel);
                 // Report names as in mcu_config imu_*.reports (hdf5_writer.py rates).
                 static const char *kReports[4] = {"accelerometer", "gyroscope", "magnetometer",
                                                   "linear_acceleration"};
                 const std::map<std::string, int> *rates[2] = {&imu_rates_mta, &imu_rates_cra};
                 for (int s = 0; s < 2; ++s) {
                     for (int i = 0; i < 4; ++i) {
                         auto it = rates[s]->find(kReports[i]);
                         c.imu_rate_hz[s][i] = (it != rates[s]->end()) ? it->second : 0;
                     }
                 }
                 return std::make_unique<Hdf5SessionWriter>(std::move(c));
             }),
             py::arg("path"), py::arg("config_hash"), py::arg("mcu_config_json") = "{}",
             py::arg("host_config_json") = "{}", py::arg("software_version") = "0.1.0",
             py::arg("schema_version") = 1, py::arg("firmware_version") = "",
             py::arg("mode") = "acquisition", py::arg("emg_sample_rate_sps") = 1000,
             py::arg("emg_gain") = 6, py::arg("channel_names") = std::vector<std::string>{},
             py::arg("fusion_mta") = "game_rotation_vector",
             py::arg("fusion_cra") = "game_rotation_vector",
             py::arg("trigger_rate_hz") = py::none(),
             py::arg("extra_meta") = std::map<std::string, std::string>{},
             py::arg("disk_flush_interval_s") = 1.0,
             py::arg("lsb_uv_per_channel") = std::vector<double>{},
             py::arg("imu_rates_mta") = std::map<std::string, int>{},
             py::arg("imu_rates_cra") = std::map<std::string, int>{})
        .def("open", &Hdf5SessionWriter::open, py::call_guard<py::gil_scoped_release>())
        .def(
            "append_emg",
            [](Hdf5SessionWriter &self, uint64_t t, const std::vector<int32_t> &channels,
               uint8_t leadoff_statp, uint8_t leadoff_statn, std::optional<uint32_t> seq) {
                if (channels.size() != 8) {
                    throw std::runtime_error("append_emg expects 8 channels");
                }
                self.append_emg(t, channels.data(), leadoff_statp, leadoff_statn, seq.has_value(),
                                seq.value_or(0));
            },
            py::arg("t_master_us"), py::arg("channels"), py::arg("leadoff_statp") = 0,
            py::arg("leadoff_statn") = 0, py::arg("seq") = py::none())
        .def(
            "append_quat",
            [](Hdf5SessionWriter &self, bool cranium, uint64_t t,
               const std::vector<float> &quat, uint8_t report_id, uint8_t accuracy) {
                if (quat.size() != 4) {
                    throw std::runtime_error("append_quat expects 4 components");
                }
                self.append_quat(cranium, t, quat.data(), report_id, accuracy);
            },
            py::arg("cranium"), py::arg("t_master_us"), py::arg("quat"),
            py::arg("report_id") = 0, py::arg("accuracy") = Hdf5SessionWriter::kAccuracyAbsent)
        .def(
            "append_imu_motion",
            [](Hdf5SessionWriter &self, bool cranium, uint8_t sensor_id, uint64_t t,
               const std::vector<float> &xyz, uint8_t accuracy) {
                if (xyz.size() != 3) {
                    throw std::runtime_error("append_imu_motion expects 3 components");
                }
                self.append_imu_motion(cranium, sensor_id, t, xyz.data(), accuracy);
            },
            py::arg("cranium"), py::arg("sensor_id"), py::arg("t_master_us"), py::arg("xyz"),
            py::arg("accuracy") = Hdf5SessionWriter::kAccuracyAbsent)
        .def(
            "append_lin_accel",
            [](Hdf5SessionWriter &self, bool cranium, uint64_t t,
               const std::vector<float> &xyz, uint8_t accuracy) {
                if (xyz.size() != 3) {
                    throw std::runtime_error("append_lin_accel expects 3 components");
                }
                self.append_lin_accel(cranium, t, xyz.data(), accuracy);
            },
            py::arg("cranium"), py::arg("t_master_us"), py::arg("xyz"),
            py::arg("accuracy") = Hdf5SessionWriter::kAccuracyAbsent)
        .def("append_trigger", &Hdf5SessionWriter::append_trigger, py::arg("t_master_us"),
             py::arg("trigger_id"))
        .def(
            "append_camera",
            [](Hdf5SessionWriter &self, const std::string &cam_id, uint64_t t_master_us,
               uint64_t sensor_timestamp_ns, uint64_t trigger_id, uint64_t frame_idx,
               const std::string &video_file) {
                self.append_camera(cam_id, t_master_us, sensor_timestamp_ns, trigger_id, frame_idx,
                                 video_file);
            },
            py::arg("cam_id"), py::arg("t_master_us"), py::arg("sensor_timestamp_ns"),
            py::arg("trigger_id"), py::arg("frame_idx") = 0, py::arg("video_file") = "")
        .def(
            "append_event",
            [](Hdf5SessionWriter &self, uint64_t t_master_us, const std::string &kind,
               const std::string &text, const std::string &phase_kind, uint64_t t_host_ns,
               const std::string &map_quality, std::optional<double> viewer_t_s) {
                return self.append_event(
                    t_master_us, kind, text, phase_kind, t_host_ns, map_quality,
                    viewer_t_s ? *viewer_t_s : std::numeric_limits<double>::quiet_NaN());
            },
            py::arg("t_master_us"), py::arg("kind"), py::arg("text"), py::arg("phase_kind") = "",
            py::arg("t_host_ns") = 0, py::arg("map_quality") = "",
            py::arg("viewer_t_s") = py::none(),
            "Append an /events row (same layout as Hdf5Writer.append_event); returns its row.")
        .def("set_event_time", &Hdf5SessionWriter::set_event_time, py::arg("row"),
             py::arg("t_master_us"), py::arg("map_quality"),
             "Re-stamp an /events row (buffered or on disk); IndexError for an unknown row.")
        .def("flush", &Hdf5SessionWriter::flush, py::arg("force_disk") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("set_sync_report", &Hdf5SessionWriter::set_sync_report, py::arg("report_json"))
        .def("write_sync_drops_json", &Hdf5SessionWriter::write_sync_drops_json, py::arg("drops_json"))
        .def("set_file_hashes_json", &Hdf5SessionWriter::set_file_hashes_json, py::arg("json"))
        .def("close", &Hdf5SessionWriter::close, py::call_guard<py::gil_scoped_release>())
        .def("_debug_fail_nth_append", &Hdf5SessionWriter::debug_fail_nth_append, py::arg("n"));
#endif
}
